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

    // Stampa la tabella dei tensori
    std::cout << "\n";
    std::cout << "  TENSORI\n";
    std::cout << "───────────────────────────────────────\n";
    std::cout << std::left
              << std::setw(45) << "  Nome"
              << std::setw(6)  << "Tipo"
              << std::setw(28) << "  Shape"
              << "Bytes\n";
    std::cout << "───────────────────────────────────────\n";

    for (const auto& ti : ctx.tensors) {
        // Costruisce la stringa shape es: [768 x 768]
        std::string shape_str = "[";
        for (uint32_t d = 0; d < ti.n_dims; d++) {
            if (d > 0) shape_str += " x ";
            shape_str += std::to_string(ti.shape[d]);
        }
        shape_str += "]";

        std::cout << "  " << std::left
                  << std::setw(43) << ti.name
                  << std::setw(6)  << ggml_type_name(ti.type)
                  << std::setw(28) << shape_str
                  << gguf_tensor_size_bytes(ti) << " B\n";
    }

    std::cout << "═══════════════════════════════════════\n\n";
}

// ─────────────────────────────────────────────
//  Nome leggibile del tipo tensore
//  Utile nei messaggi di debug e nel dump
// ─────────────────────────────────────────────
const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return "F32";
        case GGMLType::F16:  return "F16";
        case GGMLType::Q4_0: return "Q4_0";
        case GGMLType::Q4_1: return "Q4_1";
        case GGMLType::Q8_0: return "Q8_0";
        case GGMLType::Q8_1: return "Q8_1";
        case GGMLType::I8:   return "I8";
        case GGMLType::I16:  return "I16";
        case GGMLType::I32:  return "I32";
        default:             return "???";
    }
}

// ─────────────────────────────────────────────
//  Numero totale di elementi del tensore
//
//  Un tensore [768, 12, 1, 1] ha
//  768 × 12 × 1 × 1 = 9216 elementi.
//  Moltiplichiamo solo le dimensioni attive
//  (le restanti valgono 1 per convenzione)
// ─────────────────────────────────────────────
uint64_t gguf_tensor_n_elements(const GGUFTensorInfo& ti) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < ti.n_dims; i++)
        n *= ti.shape[i];
    return n;
}

// ─────────────────────────────────────────────
//  Dimensione in byte del tensore
//
//  Per i tipi quantizzati il calcolo NON è
//  semplicemente n_elementi × sizeof(tipo).
//  I pesi sono raggruppati in "block" da 32
//  elementi con un fattore di scala condiviso.
//
//  Q8_0: ogni block = 32 elementi × 1 byte
//        + 1 float16 (scale) = 34 byte
//  F32:  ogni elemento = 4 byte (semplice)
//  F16:  ogni elemento = 2 byte (semplice)
// ─────────────────────────────────────────────
uint64_t gguf_tensor_size_bytes(const GGUFTensorInfo& ti) {
    uint64_t n = gguf_tensor_n_elements(ti);
    switch (ti.type) {
        case GGMLType::F32:  return n * 4;
        case GGMLType::F16:  return n * 2;
        case GGMLType::Q8_0: return (n / 32) * 34;  // 32 valori + 2 byte scale
        case GGMLType::Q4_0: return (n / 32) * 18;  // 32 valori a 4bit + 2 byte scale
        case GGMLType::Q4_1: return (n / 32) * 20;  // come Q4_0 + min value
        default:             return n * 4;           // fallback a F32
    }
}

// ─────────────────────────────────────────────
//  Lettura delle info di tutti i tensori
//
//  Segue immediatamente la sezione metadata.
//  Per ogni tensore il formato GGUF è:
//    [stringa:  nome]
//    [uint32:   numero di dimensioni]
//    [uint64 × n_dims: shape]
//    [uint32:   tipo GGMLType]
//    [uint64:   offset dall'inizio della data section]
//
//  NOTA: l'offset è relativo all'inizio della
//  "data section" del file, non all'inizio del file.
//  La data section inizia dopo header + metadata +
//  tensor info, allineata a 32 byte.
// ─────────────────────────────────────────────
bool gguf_read_tensor_info(std::ifstream& f, GGUFContext& ctx) {
    ctx.tensors.reserve(ctx.header.n_tensors);

    for (uint64_t i = 0; i < ctx.header.n_tensors; i++) {
        GGUFTensorInfo ti;

        // Inizializza shape a 1 (dimensioni non usate = 1 per convenzione)
        for (auto& s : ti.shape) s = 1;

        // 1) Nome del tensore
        ti.name = gguf_read_string(f);

        // 2) Numero di dimensioni
        f.read(reinterpret_cast<char*>(&ti.n_dims), sizeof(ti.n_dims));

        // 3) Shape: leggi solo le n_dims dimensioni presenti nel file
        for (uint32_t d = 0; d < ti.n_dims; d++)
            f.read(reinterpret_cast<char*>(&ti.shape[d]), sizeof(ti.shape[d]));

        // 4) Tipo dato del tensore
        uint32_t type_raw = 0;
        f.read(reinterpret_cast<char*>(&type_raw), sizeof(type_raw));
        ti.type = static_cast<GGMLType>(type_raw);

        // 5) Offset nella data section
        f.read(reinterpret_cast<char*>(&ti.offset), sizeof(ti.offset));

        ctx.tensors.push_back(std::move(ti));

        if (!f.good()) {
            std::cerr << "[ERRORE] File corrotto al tensore #" << i
                      << " (nome: " << ti.name << ")\n";
            return false;
        }
    }
    return true;
}