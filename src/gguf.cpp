#include "gguf.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

// ─────────────────────────────────────────────
//  Lettura dell'header
// ─────────────────────────────────────────────

bool gguf_read_header(std::ifstream& f, GGUFHeader& out) {
    // Leggiamo i 4 campi dell'header in sequenza
    // L'ordine è fisso e definito dalla spec GGUF
    f.read(reinterpret_cast<char*>(&out.magic),     sizeof(out.magic));
    f.read(reinterpret_cast<char*>(&out.version),   sizeof(out.version));
    f.read(reinterpret_cast<char*>(&out.n_tensors), sizeof(out.n_tensors));
    f.read(reinterpret_cast<char*>(&out.n_kv),      sizeof(out.n_kv));

    // Verifica che il file sia effettivamente un GGUF
    if (out.magic != GGUF_MAGIC) {
        std::cerr << "[ERRORE] Magic non valido: 0x"
                  << std::hex << out.magic << "\n";
        return false;
    }

    // Verifica che la versione sia supportata
    if (out.version != GGUF_VERSION) {
        std::cerr << "[AVVISO] Versione inattesa: "
                  << out.version << " (attesa: " << GGUF_VERSION << ")\n";
        // Non blocchiamo — potrebbe funzionare comunque
    }

    return f.good();
}

// ─────────────────────────────────────────────
//  Lettura di una stringa GGUF
//
//  Nel formato GGUF le stringhe NON sono
//  null-terminate. La struttura è:
//    [uint64: lunghezza in byte][byte × lunghezza]
//  Questo permette stringhe con \0 interno
//  e lunghezze arbitrarie senza scansione lineare
// ─────────────────────────────────────────────

std::string gguf_read_string(std::ifstream& f) {
    // Prima leggiamo quanti byte occupa la stringa
    uint64_t len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof(len));

    // Poi leggiamo esattamente quei byte
    std::string s(len, '\0');
    f.read(s.data(), static_cast<std::streamsize>(len));
    return s;
}

// ─────────────────────────────────────────────
//  Lettura di un valore tipizzato
//
//  GGUF è un formato binario strettamente tipizzato:
//  ogni valore è preceduto dal suo tipo (vedi GGUFValueType).
//  Qui facciamo il dispatch sul tipo e leggiamo
//  esattamente i byte necessari.
//
//  NOTA: i tipi ARRAY sono ricorsivi — un array
//  contiene un tipo elemento + un contatore +
//  N valori di quel tipo. Per ora li leggiamo
//  ma li saltiamo (li gestiremo nella prossima fase)
// ─────────────────────────────────────────────

GGUFValue gguf_read_value(std::ifstream& f, GGUFValueType type) {
    switch (type) {
        case GGUFValueType::UINT8:   { uint8_t  v; f.read((char*)&v, 1); return v; }
        case GGUFValueType::INT8:    { int8_t   v; f.read((char*)&v, 1); return v; }
        case GGUFValueType::UINT16:  { uint16_t v; f.read((char*)&v, 2); return v; }
        case GGUFValueType::INT16:   { int16_t  v; f.read((char*)&v, 2); return v; }
        case GGUFValueType::UINT32:  { uint32_t v; f.read((char*)&v, 4); return v; }
        case GGUFValueType::INT32:   { int32_t  v; f.read((char*)&v, 4); return v; }
        case GGUFValueType::FLOAT32: { float    v; f.read((char*)&v, 4); return v; }
        case GGUFValueType::BOOL:    { uint8_t  v; f.read((char*)&v, 1); return (bool)v; }
        case GGUFValueType::STRING:  { return gguf_read_string(f); }
        case GGUFValueType::UINT64:  { uint64_t v; f.read((char*)&v, 8); return v; }
        case GGUFValueType::INT64:   { int64_t  v; f.read((char*)&v, 8); return v; }
        case GGUFValueType::FLOAT64: { double   v; f.read((char*)&v, 8); return v; }

        case GGUFValueType::ARRAY: {
            // Un array GGUF è strutturato così:
            //   [uint32: tipo degli elementi]
            //   [uint64: numero di elementi]
            //   [elemento × n]
            // Leggiamo tipo e contatore...
            uint32_t elem_type_raw = 0;
            uint64_t arr_len       = 0;
            f.read((char*)&elem_type_raw, 4);
            f.read((char*)&arr_len,       8);

            auto elem_type = static_cast<GGUFValueType>(elem_type_raw);

            // ...poi consumiamo tutti gli elementi dal file
            // (per ora li scartiamo, li useremo nelle fasi successive)
            for (uint64_t i = 0; i < arr_len; i++) {
                gguf_read_value(f, elem_type);
            }

            // Ritorniamo una stringa descrittiva come placeholder
            return std::string("[array:" + std::to_string(arr_len) + "]");
        }

        default:
            // Tipo sconosciuto — non dovrebbe mai succedere con file validi
            std::cerr << "[ERRORE] Tipo valore sconosciuto: "
                      << static_cast<uint32_t>(type) << "\n";
            return std::string("[tipo sconosciuto]");
    }
}

// ─────────────────────────────────────────────
//  Lettura di tutti i metadata KV
//
//  La sezione metadata segue immediatamente l'header.
//  Per ogni coppia chiave-valore il formato è:
//    [stringa: chiave]
//    [uint32:  tipo del valore]
//    [valore:  dipende dal tipo]
//
//  Ripetiamo per n_kv volte (letto nell'header)
// ─────────────────────────────────────────────

bool gguf_read_metadata(std::ifstream& f, GGUFContext& ctx) {
    ctx.metadata.reserve(ctx.header.n_kv);

    for (uint64_t i = 0; i < ctx.header.n_kv; i++) {
        GGUFKV kv;

        // 1) Leggi la chiave (è sempre una stringa GGUF)
        kv.key = gguf_read_string(f);

        // 2) Leggi il tipo del valore (uint32)
        uint32_t type_raw = 0;
        f.read(reinterpret_cast<char*>(&type_raw), sizeof(type_raw));
        kv.type = static_cast<GGUFValueType>(type_raw);

        // 3) Leggi il valore vero e proprio
        kv.value = gguf_read_value(f, kv.type);

        ctx.metadata.push_back(std::move(kv));

        // Controlla che il file non sia corrotto mentre leggiamo
        if (!f.good()) {
            std::cerr << "[ERRORE] File corrotto al KV #" << i
                      << " (chiave: " << kv.key << ")\n";
            return false;
        }
    }
    return true;
}

// ─────────────────────────────────────────────
//  Stampa del contesto — solo per debug/didattica
//
//  std::visit è il modo idiomatico C++17 per
//  estrarre il valore da un std::variant senza
//  sapere a compile-time quale tipo contiene
// ─────────────────────────────────────────────

void gguf_print_context(const GGUFContext& ctx) {
    std::cout << "\n═══════════════════════════════════════\n";
    std::cout << "  EIE-LLM — GGUF Context Dump\n";
    std::cout << "═══════════════════════════════════════\n";
    std::cout << "  Versione : " << ctx.header.version   << "\n";
    std::cout << "  Tensori  : " << ctx.header.n_tensors << "\n";
    std::cout << "  Metadata : " << ctx.header.n_kv      << "\n";
    std::cout << "───────────────────────────────────────\n";

    for (const auto& kv : ctx.metadata) {
        // Stampa la chiave con padding fisso per allineamento
        std::cout << "  " << std::left << std::setw(42) << kv.key << " = ";

        // std::visit applica questa lambda al tipo concreto
        // contenuto nel variant — il compilatore genera
        // il dispatch automaticamente per tutti i tipi
        std::visit([](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, bool>)
                std::cout << (v ? "true" : "false");
            else if constexpr (std::is_same_v<T, uint8_t> ||
                               std::is_same_v<T, int8_t>)
                std::cout << static_cast<int>(v); // evita stampa come char
            else
                std::cout << v;
        }, kv.value);

        std::cout << "\n";
    }
    std::cout << "═══════════════════════════════════════\n\n";
}