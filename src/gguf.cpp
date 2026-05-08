#include "gguf.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>

// ─────────────────────────────────────────────
//  Lettura header
// ─────────────────────────────────────────────
bool gguf_read_header(std::ifstream& f, GGUFHeader& out) {
    f.read(reinterpret_cast<char*>(&out.magic),     sizeof(out.magic));
    f.read(reinterpret_cast<char*>(&out.version),   sizeof(out.version));
    f.read(reinterpret_cast<char*>(&out.n_tensors), sizeof(out.n_tensors));
    f.read(reinterpret_cast<char*>(&out.n_kv),      sizeof(out.n_kv));

    if (out.magic != GGUF_MAGIC) {
        std::cerr << "[ERRORE] Magic non valido: 0x"
                  << std::hex << out.magic << "\n";
        return false;
    }
    if (out.version != GGUF_VERSION) {
        std::cerr << "[AVVISO] Versione inattesa: " << out.version << "\n";
    }
    return f.good();
}

// ─────────────────────────────────────────────
//  Lettura stringa GGUF
//  Formato: [uint64 lunghezza][bytes]
//  NON null-terminate come le stringhe C
// ─────────────────────────────────────────────
std::string gguf_read_string(std::ifstream& f) {
    uint64_t len = 0;
    f.read(reinterpret_cast<char*>(&len), sizeof(len));
    std::string s(len, '\0');
    f.read(s.data(), static_cast<std::streamsize>(len));
    return s;
}

// ─────────────────────────────────────────────
//  Lettura valore tipizzato
//
//  Ogni valore è preceduto dal suo tipo.
//  Wrappamo tutto in GGUFValue.data che è
//  il variant interno alla struct.
// ─────────────────────────────────────────────
GGUFValue gguf_read_value(std::ifstream& f, GGUFValueType type) {
    GGUFValue v;
    switch (type) {
        case GGUFValueType::UINT8:   { uint8_t  x; f.read((char*)&x,1); v.data=x; break; }
        case GGUFValueType::INT8:    { int8_t   x; f.read((char*)&x,1); v.data=x; break; }
        case GGUFValueType::UINT16:  { uint16_t x; f.read((char*)&x,2); v.data=x; break; }
        case GGUFValueType::INT16:   { int16_t  x; f.read((char*)&x,2); v.data=x; break; }
        case GGUFValueType::UINT32:  { uint32_t x; f.read((char*)&x,4); v.data=x; break; }
        case GGUFValueType::INT32:   { int32_t  x; f.read((char*)&x,4); v.data=x; break; }
        case GGUFValueType::FLOAT32: { float    x; f.read((char*)&x,4); v.data=x; break; }
        case GGUFValueType::BOOL:    { uint8_t  x; f.read((char*)&x,1); v.data=(bool)x; break; }
        case GGUFValueType::STRING:  { v.data = gguf_read_string(f); break; }
        case GGUFValueType::UINT64:  { uint64_t x; f.read((char*)&x,8); v.data=x; break; }
        case GGUFValueType::INT64:   { int64_t  x; f.read((char*)&x,8); v.data=x; break; }
        case GGUFValueType::FLOAT64: { double   x; f.read((char*)&x,8); v.data=x; break; }

        case GGUFValueType::ARRAY: {
            // Struttura array GGUF:
            //   [uint32: tipo elementi]
            //   [uint64: numero elementi]
            //   [elemento × n]
            uint32_t elem_type_raw = 0;
            uint64_t arr_len       = 0;
            f.read((char*)&elem_type_raw, 4);
            f.read((char*)&arr_len,       8);

            GGUFArray arr;
            arr.elem_type = static_cast<GGUFValueType>(elem_type_raw);
            arr.values.reserve(arr_len);

            for (uint64_t i = 0; i < arr_len; i++)
                arr.values.push_back(gguf_read_value(f, arr.elem_type));

            v.data = std::move(arr);
            break;
        }
        default:
            std::cerr << "[ERRORE] Tipo sconosciuto: "
                      << static_cast<uint32_t>(type) << "\n";
            v.data = std::string("[tipo sconosciuto]");
    }
    return v;
}

// ─────────────────────────────────────────────
//  Lettura metadata KV
//  Formato per ogni coppia:
//    [stringa: chiave]
//    [uint32:  tipo valore]
//    [valore:  dipende dal tipo]
// ─────────────────────────────────────────────
bool gguf_read_metadata(std::ifstream& f, GGUFContext& ctx) {
    ctx.metadata.reserve(ctx.header.n_kv);
    for (uint64_t i = 0; i < ctx.header.n_kv; i++) {
        GGUFKV kv;
        kv.key  = gguf_read_string(f);
        uint32_t type_raw = 0;
        f.read(reinterpret_cast<char*>(&type_raw), sizeof(type_raw));
        kv.type  = static_cast<GGUFValueType>(type_raw);
        kv.value = gguf_read_value(f, kv.type);

        ctx.metadata.push_back(std::move(kv));
        if (!f.good()) {
            std::cerr << "[ERRORE] File corrotto al KV #" << i << "\n";
            return false;
        }
    }
    return true;
}

// ─────────────────────────────────────────────
//  Nome leggibile del tipo tensore
// ─────────────────────────────────────────────
const char* ggml_type_name(GGMLType type) {
    switch (type) {
        case GGMLType::F32:  return "F32";
        case GGMLType::F16:  return "F16";
        case GGMLType::Q4_0: return "Q4_0";
        case GGMLType::Q4_1: return "Q4_1";
        case GGMLType::Q8_0: return "Q8_0";
        case GGMLType::Q8_1: return "Q8_1";
        case GGMLType::Q2_K: return "Q2_K";
        case GGMLType::Q3_K: return "Q3_K";
        case GGMLType::Q4_K: return "Q4_K";
        case GGMLType::Q5_K: return "Q5_K";
        case GGMLType::Q6_K: return "Q6_K";
        case GGMLType::Q8_K: return "Q8_K";
        case GGMLType::I8:   return "I8";
        case GGMLType::I16:  return "I16";
        case GGMLType::I32:  return "I32";
        default:             return "???";
    }
}

// ─────────────────────────────────────────────
//  Numero totale di elementi del tensore
//  Prodotto di tutte le dimensioni attive
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
//  Per i tipi quantizzati i pesi sono raggruppati
//  in block da 32 elementi con scale condiviso:
//  Q8_0: 32 byte dati + 2 byte scale = 34 byte/block
//  Q4_0: 16 byte dati + 2 byte scale = 18 byte/block
// ─────────────────────────────────────────────
uint64_t gguf_tensor_size_bytes(const GGUFTensorInfo& ti) {
    uint64_t n = gguf_tensor_n_elements(ti);
    switch (ti.type) {
        case GGMLType::F32:  return n * 4;
        case GGMLType::F16:  return n * 2;
        case GGMLType::Q8_0: return (n / 32)  * 34;
        case GGMLType::Q4_0: return (n / 32)  * 18;
        case GGMLType::Q4_1: return (n / 32)  * 20;
        // K-quants: super-block da 256 elementi
        case GGMLType::Q4_K: return (n / 256) * 144;
        case GGMLType::Q6_K: return (n / 256) * 210;
        case GGMLType::Q2_K: return (n / 256) * 84;
        case GGMLType::Q3_K: return (n / 256) * 110;
        case GGMLType::Q5_K: return (n / 256) * 176;
        default:             return n * 4;
    }
}
// ─────────────────────────────────────────────
//  Lettura info tensori
//  Formato per ogni tensore:
//    [stringa:  nome]
//    [uint32:   n_dims]
//    [uint64 × n_dims: shape]
//    [uint32:   tipo GGMLType]
//    [uint64:   offset nella data section]
// ─────────────────────────────────────────────
bool gguf_read_tensor_info(std::ifstream& f, GGUFContext& ctx) {
    ctx.tensors.reserve(ctx.header.n_tensors);
    for (uint64_t i = 0; i < ctx.header.n_tensors; i++) {
        GGUFTensorInfo ti;
        for (auto& s : ti.shape) s = 1;

        ti.name = gguf_read_string(f);
        f.read(reinterpret_cast<char*>(&ti.n_dims), sizeof(ti.n_dims));
        for (uint32_t d = 0; d < ti.n_dims; d++)
            f.read(reinterpret_cast<char*>(&ti.shape[d]), sizeof(ti.shape[d]));
        uint32_t type_raw = 0;
        f.read(reinterpret_cast<char*>(&type_raw), sizeof(type_raw));
        ti.type = static_cast<GGMLType>(type_raw);
        f.read(reinterpret_cast<char*>(&ti.offset), sizeof(ti.offset));

        ctx.tensors.push_back(std::move(ti));
        if (!f.good()) {
            std::cerr << "[ERRORE] File corrotto al tensore #" << i << "\n";
            return false;
        }
    }
    return true;
}

// ─────────────────────────────────────────────
//  Calcolo offset data section
//  La data section inizia al primo multiplo
//  di 32 byte dopo la fine della tensor info
// ─────────────────────────────────────────────
uint64_t gguf_calc_data_offset(std::ifstream& f) {
    uint64_t pos = static_cast<uint64_t>(f.tellg());
    return (pos + 31) / 32 * 32;
}

// ─────────────────────────────────────────────
//  Caricamento pesi in RAM
//  Per ogni tensore: seek all'offset assoluto
//  e copia i byte grezzi nel vettore data
// ─────────────────────────────────────────────
bool gguf_load_tensors(std::ifstream& f, GGUFContext& ctx) {
    ctx.data_offset = gguf_calc_data_offset(f);
    ctx.weights.reserve(ctx.tensors.size());

    for (const auto& info : ctx.tensors) {
        GGUFTensor tensor;
        tensor.info = info;

        uint64_t abs_offset = ctx.data_offset + info.offset;
        f.seekg(static_cast<std::streamoff>(abs_offset));
        if (!f.good()) {
            std::cerr << "[ERRORE] Seek fallito: " << info.name << "\n";
            return false;
        }

        uint64_t size = gguf_tensor_size_bytes(info);
        tensor.data.resize(size);
        f.read(reinterpret_cast<char*>(tensor.data.data()),
               static_cast<std::streamsize>(size));
        if (!f.good()) {
            std::cerr << "[ERRORE] Lettura fallita: " << info.name << "\n";
            return false;
        }

        ctx.weights.push_back(std::move(tensor));
    }
    return true;
}

// ─────────────────────────────────────────────
//  Ricerca tensore per nome — O(n) lineare
//  Sufficiente per 148 tensori di GPT-2
// ─────────────────────────────────────────────
const GGUFTensor* gguf_find_tensor(const GGUFContext& ctx,
                                   const std::string& name) {
    for (const auto& t : ctx.weights)
        if (t.name() == name) return &t;
    return nullptr;
}

// ─────────────────────────────────────────────
//  Stampa contesto — solo per debug/didattica
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
        std::cout << "  " << std::left << std::setw(42) << kv.key << " = ";
        std::visit([](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, bool>)
                std::cout << (v ? "true" : "false");
            else if constexpr (std::is_same_v<T, uint8_t> ||
                               std::is_same_v<T, int8_t>)
                std::cout << static_cast<int>(v);
            else if constexpr (std::is_same_v<T, GGUFArray>)
                std::cout << "[array:" << v.values.size() << "]";
            else
                std::cout << v;
        }, kv.value.data);  // ← nota: .data sul variant interno
        std::cout << "\n";
    }

    std::cout << "───────────────────────────────────────\n";
    std::cout << std::left
              << std::setw(45) << "  Nome"
              << std::setw(6)  << "Tipo"
              << "Shape\n";
    std::cout << "───────────────────────────────────────\n";

    for (const auto& ti : ctx.tensors) {
        std::string shape_str = "[";
        for (uint32_t d = 0; d < ti.n_dims; d++) {
            if (d > 0) shape_str += " x ";
            shape_str += std::to_string(ti.shape[d]);
        }
        shape_str += "]";
        std::cout << "  " << std::left
                  << std::setw(43) << ti.name
                  << std::setw(6)  << ggml_type_name(ti.type)
                  << shape_str << "\n";
    }
    std::cout << "═══════════════════════════════════════\n\n";
}

// ─────────────────────────────────────────────
//  Riepilogo utilizzo RAM
// ─────────────────────────────────────────────
void gguf_print_memory_usage(const GGUFContext& ctx) {
    uint64_t total = 0;
    for (const auto& t : ctx.weights) total += t.data.size();

    std::cout << "\n═══════════════════════════════════════\n";
    std::cout << "  EIE-LLM — Memoria pesi\n";
    std::cout << "═══════════════════════════════════════\n";
    std::cout << "  Tensori : " << ctx.weights.size() << "\n";
    std::cout << "  RAM     : " << std::fixed << std::setprecision(1)
              << total / (1024.0 * 1024.0) << " MB\n";
    std::cout << "═══════════════════════════════════════\n\n";
}