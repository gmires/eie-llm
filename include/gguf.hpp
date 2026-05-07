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
    GGUFHeader       header;
    std::vector<GGUFKV> metadata;  // tutti i KV letti dal file
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