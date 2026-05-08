#pragma once
#include <string>
#include <vector>
#include <variant>
#include <cstdint>
#include <fstream>

// ─────────────────────────────────────────────
//  Costanti del formato GGUF
// ─────────────────────────────────────────────

// Magic number: i primi 4 byte di ogni file GGUF valido
// In ASCII corrisponde alla stringa "GGUF"
static constexpr uint32_t GGUF_MAGIC   = 0x46554747;

// Versione minima supportata del formato GGUF
static constexpr uint32_t GGUF_VERSION = 3;

// ─────────────────────────────────────────────
//  Tipi dei valori nei metadata
//  Ogni chiave del metadata ha un tipo associato
//  definito da questo enum (dal formato GGUF spec)
// ─────────────────────────────────────────────
enum class GGUFValueType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// ─────────────────────────────────────────────
//  GGUFValue e GGUFArray sono mutuamente
//  ricorsivi: un valore può essere un array,
//  un array contiene valori.
//
//  In C++ non si possono dichiarare due tipi
//  che si riferiscono a vicenda con using/variant
//  direttamente. La soluzione è:
//  1) Wrappare il variant in una struct (GGUFValue)
//  2) Fare forward declaration di GGUFValue
//  3) Definire GGUFArray che usa GGUFValue
//  4) Definire GGUFValue completo dopo
// ─────────────────────────────────────────────

// Forward declaration — il compilatore sa che
// GGUFValue esiste anche se non è ancora definito
struct GGUFValue;

// ─────────────────────────────────────────────
//  Array di valori GGUF
//  Usato per vocabolario, merge rules, scores.
//  Tutti gli elementi hanno lo stesso tipo base.
// ─────────────────────────────────────────────
struct GGUFArray {
    GGUFValueType          elem_type;
    std::vector<GGUFValue> values;   // usa GGUFValue via forward decl
};

// ─────────────────────────────────────────────
//  Valore GGUF — definizione completa
//
//  Wrappa un std::variant con tutti i tipi
//  possibili incluso GGUFArray (ricorsivo).
//  Usiamo una struct invece di un using diretto
//  perché il variant non può contenere se stesso
//  nemmeno indirettamente con un alias.
// ─────────────────────────────────────────────
struct GGUFValue {
    std::variant<
        uint8_t, int8_t,
        uint16_t, int16_t,
        uint32_t, int32_t,
        float,
        bool,
        std::string,
        uint64_t, int64_t,
        double,
        GGUFArray
    > data;

    // Helper type-safe per estrarre il valore
    // Uso: auto* v = kv.value.get_if<uint32_t>()
    template<typename T>
    T* get_if() { return std::get_if<T>(&data); }

    template<typename T>
    const T* get_if() const { return std::get_if<T>(&data); }
};

// ─────────────────────────────────────────────
//  Una singola coppia chiave-valore del metadata
//  Esempio: "general.architecture" = "gpt2"
// ─────────────────────────────────────────────
struct GGUFKV {
    std::string   key;
    GGUFValueType type;
    GGUFValue     value;
};

// ─────────────────────────────────────────────
//  Header del file GGUF
// ─────────────────────────────────────────────
struct GGUFHeader {
    uint32_t magic;      // deve essere GGUF_MAGIC
    uint32_t version;    // versione formato (3)
    uint64_t n_tensors;  // numero di tensori
    uint64_t n_kv;       // numero di coppie KV nei metadata
};

// ─────────────────────────────────────────────
//  Tipi di dato dei tensori
//  Definiscono come i pesi sono memorizzati:
//  F32 = float pieno, F16 = mezza precisione,
//  Q4/Q8 = quantizzati (meno bit = meno RAM)
// ─────────────────────────────────────────────
enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    I8      = 24,
    I16     = 25,
    I32     = 26,
};

// Numero massimo di dimensioni di un tensore GGUF
static constexpr uint32_t GGML_MAX_DIMS = 4;

// ─────────────────────────────────────────────
//  Informazioni su un singolo tensore
//
//  Contiene solo i metadati — i dati grezzi
//  vengono caricati in GGUFTensor separatamente
// ─────────────────────────────────────────────
struct GGUFTensorInfo {
    std::string name;                   // es: "blk.0.attn_q.weight"
    uint32_t    n_dims;                 // numero di dimensioni (1-4)
    uint64_t    shape[GGML_MAX_DIMS];   // es: [768, 768, 1, 1]
    GGMLType    type;                   // tipo dato (F32, Q8_0, ecc.)
    uint64_t    offset;                 // posizione nella data section
};

// ─────────────────────────────────────────────
//  Tensore caricato in RAM
//
//  Unisce i metadati ai dati grezzi copiati
//  dal file. I byte sono nel formato originale
//  (quantizzati o float) senza conversioni.
// ─────────────────────────────────────────────
struct GGUFTensor {
    GGUFTensorInfo       info;
    std::vector<uint8_t> data;  // dati grezzi in RAM

    // Accesso rapido alle info più usate
    const std::string& name()   const { return info.name; }
    GGMLType           type()   const { return info.type; }
    uint64_t           offset() const { return info.offset; }
};

// ─────────────────────────────────────────────
//  Contesto GGUF — tutto ciò che leggiamo
//  dal file, cresce ad ogni fase del progetto
// ─────────────────────────────────────────────
struct GGUFContext {
    GGUFHeader                  header;
    std::vector<GGUFKV>         metadata;
    std::vector<GGUFTensorInfo> tensors;      // indice tensori
    std::vector<GGUFTensor>     weights;      // dati in RAM
    uint64_t                    data_offset;  // inizio data section
};

// ─────────────────────────────────────────────
//  Funzioni pubbliche del modulo GGUF
// ─────────────────────────────────────────────

// Legge e valida l'header del file GGUF
// Ritorna false se il file non è valido
bool gguf_read_header(std::ifstream& f, GGUFHeader& out);

// Legge una stringa GGUF dal file
// Nel formato GGUF le stringhe sono: [uint64 lunghezza][bytes]
// NON sono null-terminate come in C
std::string gguf_read_string(std::ifstream& f);

// Legge un singolo valore dal file dato il suo tipo
GGUFValue gguf_read_value(std::ifstream& f, GGUFValueType type);

// Legge tutti i metadata KV dal file
// Deve essere chiamata DOPO gguf_read_header
bool gguf_read_metadata(std::ifstream& f, GGUFContext& ctx);

// Stampa tutto il contesto in modo leggibile — utile per debug
void gguf_print_context(const GGUFContext& ctx);

// Legge le info di tutti i tensori dal file
// Deve essere chiamata DOPO gguf_read_metadata
bool gguf_read_tensor_info(std::ifstream& f, GGUFContext& ctx);

// Converte GGMLType in stringa leggibile — utile per debug
const char* ggml_type_name(GGMLType type);

// Calcola il numero totale di elementi di un tensore
// moltiplicando tutte le sue dimensioni
uint64_t gguf_tensor_n_elements(const GGUFTensorInfo& ti);

// Calcola la dimensione in byte di un tensore
// tenendo conto della quantizzazione
uint64_t gguf_tensor_size_bytes(const GGUFTensorInfo& ti);

// Calcola e ritorna la posizione della data section nel file
// (dopo header + metadata + tensor info, allineata a 32 byte)
uint64_t gguf_calc_data_offset(std::ifstream& f);

// Carica tutti i pesi in RAM
// Deve essere chiamata dopo gguf_read_tensor_info
bool gguf_load_tensors(std::ifstream& f, GGUFContext& ctx);

// Cerca un tensore per nome — ritorna nullptr se non trovato
const GGUFTensor* gguf_find_tensor(const GGUFContext& ctx,
                                   const std::string& name);
// Stampa un riepilogo della RAM occupata dai pesi
void gguf_print_memory_usage(const GGUFContext& ctx);