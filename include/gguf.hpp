#pragma once
#include <string>
#include <vector>
#include <variant>
#include <cstdint>

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
//  Valore di un metadata: può essere uno
//  qualsiasi dei tipi supportati da GGUF.
//  Usiamo std::variant per rappresentarli tutti
//  in modo type-safe senza union grezze.
// ─────────────────────────────────────────────
using GGUFValue = std::variant<
    uint8_t, int8_t,
    uint16_t, int16_t,
    uint32_t, int32_t,
    float,
    bool,
    std::string,
    uint64_t, int64_t,
    double
>;

// ─────────────────────────────────────────────
//  Una singola coppia chiave-valore del metadata
//  Esempio: "general.architecture" = "gpt2"
// ─────────────────────────────────────────────
struct GGUFKV {
    std::string  key;
    GGUFValueType type;
    GGUFValue    value;
};

// ─────────────────────────────────────────────
//  Tipi di dato dei tensori GGUF
//  Definiscono come i pesi sono memorizzati:
//  F32 = float pieno, F16 = mezza precisione,
//  Q4/Q8 = quantizzati (meno bit = meno RAM)
// ─────────────────────────────────────────────
enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    Q8_0    = 8,   // quello usato dal nostro GPT-2
    Q8_1    = 9,
    I8      = 24,
    I16     = 25,
    I32     = 26,
};

// ─────────────────────────────────────────────
//  Numero massimo di dimensioni di un tensore
//  In GGUF un tensore può avere al massimo
//  4 dimensioni (es: [vocab, embd, 1, 1])
// ─────────────────────────────────────────────
static constexpr uint32_t GGML_MAX_DIMS = 4;

// ─────────────────────────────────────────────
//  Informazioni su un singolo tensore
//
//  ATTENZIONE: qui salviamo solo i METADATI
//  del tensore, non i dati veri e propri.
//  I pesi reali stanno nel file a partire
//  da 'offset' — li caricheremo nella Fase 2
// ─────────────────────────────────────────────
struct GGUFTensorInfo {
    std::string name;                      // es: "blk.0.attn_q.weight"
    uint32_t    n_dims;                    // numero di dimensioni (1-4)
    uint64_t    shape[GGML_MAX_DIMS];      // dimensioni es: [768, 768, 1, 1]
    GGMLType    type;                      // tipo dato (F32, Q8_0, ecc.)
    uint64_t    offset;                    // posizione dei dati nel file
};

// ─────────────────────────────────────────────
//  Header del file GGUF
//  Contiene le informazioni base lette
//  dai primi byte del file
// ─────────────────────────────────────────────
struct GGUFHeader {
    uint32_t magic;       // Deve essere GGUF_MAGIC
    uint32_t version;     // Versione del formato (3)
    uint64_t n_tensors;   // Quanti tensori contiene il file
    uint64_t n_kv;        // Quante coppie chiave-valore nei metadata
};

// ─────────────────────────────────────────────
//  Contesto GGUF: tutto ciò che leggiamo dal file
//  Questa struttura cresce ad ogni fase:
//  ora contiene header + metadata,
//  nelle fasi successive aggiungeremo i tensori
// ─────────────────────────────────────────────
struct GGUFContext {
    GGUFHeader                 header;
    std::vector<GGUFKV>        metadata;
    std::vector<GGUFTensorInfo> tensors;
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