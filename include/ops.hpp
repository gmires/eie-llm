#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include "gguf.hpp"

// ─────────────────────────────────────────────
//  Tensore quantizzato — dequantizzazione lazy
//
//  Mantiene i pesi nel formato originale GGUF
//  senza espanderli in float32 al caricamento.
//
//  Convenzione shape (stessa di GGUF/ggml):
//    n_cols = shape[0] = in_dim  (dimensione più veloce)
//    n_rows = shape[1] = out_dim
//
//  Layout byte per riga (row-major):
//    F32  : n_cols × 4 byte
//    F16  : n_cols × 2 byte
//    Q4_K : (n_cols / 256) × 144 byte
//    Q6_K : (n_cols / 256) × 210 byte
//    Q8_0 : (n_cols / 32)  × 34  byte
// ─────────────────────────────────────────────
struct QuantTensor {
    GGMLType             type   = GGMLType::F32;
    uint64_t             n_rows = 0;
    uint64_t             n_cols = 0;
    std::vector<uint8_t> data;

    bool empty() const { return data.empty(); }
};

// ─────────────────────────────────────────────
//  Operazioni primitive sui tensori
//
//  Tutte le operazioni lavorano su float32
//  in memoria contigua (row-major).
//  Un tensore 2D [righe × colonne] è memorizzato
//  come: [riga0col0, riga0col1, ..., riga1col0, ...]
//
//  Convenzione dimensioni GPT-2:
//    n_embd  = 768   (dimensione embedding)
//    n_head  = 12    (numero attention heads)
//    n_ctx   = 1024  (context length massimo)
//    n_vocab = 50257 (dimensione vocabolario)
//    n_layer = 12    (numero di layer)
//    d_head  = 64    (n_embd / n_head)
// ─────────────────────────────────────────────

// ─────────────────────────────────────────────
//  Dequantizzazione Q8_0 → float32
//
//  Il formato Q8_0 comprime i pesi in questo modo:
//  - I valori float32 originali vengono divisi in
//    blocchi da 32 elementi
//  - Per ogni blocco si calcola il valore massimo
//    assoluto (scale = max_abs / 127.0)
//  - Ogni valore viene quantizzato a int8:
//    q = round(v / scale)  → range [-127, 127]
//
//  Struttura di ogni blocco (34 byte totali):
//    [float16: scale][int8 × 32: valori quantizzati]
//
//  Dequantizzazione inversa:
//    v = q * scale
//
//  Parametri:
//    src     : buffer raw Q8_0 dal file GGUF
//    dst     : buffer float32 di output (già allocato)
//    n_elem  : numero totale di elementi
// ─────────────────────────────────────────────
void dequantize_q8_0(const uint8_t* src, float* dst, uint64_t n_elem);

// ─────────────────────────────────────────────
//  Dequantizzazione Q4_K
//
//  Q4_K è un formato "K-quant" introdotto da
//  llama.cpp. Ogni super-block contiene 256
//  elementi organizzati in 8 sub-block da 32.
//
//  Struttura di ogni super-block (144 byte):
//    [2 byte:  scale_min float16]
//    [2 byte:  scale_max float16]
//    [12 byte: scales per sub-block (6 bit × 8 × 2)]
//    [128 byte: valori 4bit × 256]
//
//  Per ogni sub-block b:
//    scale = scales[b]
//    min   = mins[b]
//    val   = (nibble * scale) - min
// ─────────────────────────────────────────────
void dequantize_q4_k(const uint8_t* src, float* dst, uint64_t n_elem);

// ─────────────────────────────────────────────
//  Dequantizzazione Q6_K
//
//  Q6_K usa 6 bit per elemento organizzati in
//  super-block da 256 elementi.
//
//  Struttura di ogni super-block (210 byte):
//    [128 byte: bit bassi (4 bit × 256)]
//    [64 byte:  bit alti  (2 bit × 256)]
//    [16 byte:  scales per sub-block (int8 × 16)]
//    [2 byte:   super scale float16]
//
//  Per ogni elemento:
//    q6 = (4bit_low) | (2bit_high << 4)  → range [0,63]
//    val = (q6 - 32) * scale * super_scale
// ─────────────────────────────────────────────
void dequantize_q6_k(const uint8_t* src, float* dst, uint64_t n_elem);

// ─────────────────────────────────────────────
//  Conversione float16 → float32
//
//  GPT-2 usa float16 (half precision) per alcuni
//  tensori. IEEE 754 float16:
//    bit 15    : segno
//    bit 14-10 : esponente (bias 15)
//    bit 9-0   : mantissa
//
//  Convertiamo manualmente senza librerie esterne.
// ─────────────────────────────────────────────
float fp16_to_fp32(uint16_t h);

// ─────────────────────────────────────────────
//  Ottieni un tensore dequantizzato come float32
//
//  Funzione di convenienza che:
//  1) Legge il tipo del tensore
//  2) Applica la dequantizzazione corretta
//  3) Ritorna un vettore float32 pronto all'uso
//
//  Supporta: F32 (copia diretta), Q8_0
// ─────────────────────────────────────────────
std::vector<float> tensor_to_float(const GGUFTensor& t);

// ─────────────────────────────────────────────
//  Matrix multiplication: C = A × B
//
//  A: matrice [m × k]  (row-major)
//  B: matrice [k × n]  (row-major)
//  C: matrice [m × n]  (row-major, output)
//
//  Implementazione naive O(m×k×n) — didattica
//  e corretta, non ottimizzata per performance.
//  In un engine reale si userebbe BLAS/AVX.
// ─────────────────────────────────────────────
void matmul(const float* A, const float* B, float* C, int m, int k, int n);

// ─────────────────────────────────────────────
//  Matrix-vector multiplication: y = A × x
//
//  Caso speciale molto comune nel forward pass:
//  moltiplichiamo una matrice di pesi per un
//  singolo vettore (un token alla volta).
//
//  A: matrice [out_dim × in_dim]  (row-major)
//  x: vettore [in_dim]
//  y: vettore [out_dim]           (output)
// ─────────────────────────────────────────────
void matvec(const float* A, const float* x, float* y, int out_dim, int in_dim);

// ─────────────────────────────────────────────
//  Addizione vettore elemento per elemento
//  out[i] = a[i] + b[i]
// ─────────────────────────────────────────────
void vec_add(const float* a, const float* b, float* out, int n);

// ─────────────────────────────────────────────
//  Copia vettore
//  dst[i] = src[i]
// ─────────────────────────────────────────────
void vec_copy(const float* src, float* dst, int n);

// ─────────────────────────────────────────────
//  Softmax in-place su un vettore
//
//  Formula numericamente stabile:
//    1) sottrai il massimo (evita overflow)
//    2) calcola exp()
//    3) dividi per la somma (normalizza)
//
//  softmax(x)[i] = exp(x[i] - max) / Σexp(x[j] - max)
// ─────────────────────────────────────────────
void softmax(float* x, int n);

// ─────────────────────────────────────────────
//  GELU activation function
//
//  Usata da GPT-2 nel feed-forward network.
//  Formula approssimata (usata da GPT-2):
//    GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715x³)))
//
//  Modifica il vettore in-place.
// ─────────────────────────────────────────────
void gelu(float* x, int n);

// ─────────────────────────────────────────────
//  RMSNorm
//
//  Normalizza x usando solo la root mean square,
//  senza sottrarre la media (più semplice e veloce
//  di LayerNorm). Solo gamma (w), nessun beta.
//
//  RMS(x)   = √( (1/n) Σ xᵢ² )
//  out[i]   = w[i] * x[i] / RMS(x)
// ─────────────────────────────────────────────
void rms_norm(const float* x, const float* w, float* out, int n, float eps);

// ─────────────────────────────────────────────
//  RoPE — Rotary Position Embedding
//
//  Applica la rotazione in-place al vettore x
//  (Q oppure K) alla posizione pos.
//  Ogni coppia (x[2i], x[2i+1]) viene ruotata
//  di θᵢ = pos / (freq_base ^ (2i / rope_dim))
//
//  x        : vettore da ruotare [n_elem]
//  pos      : posizione del token
//  n_heads  : numero di head
//  d_head   : dimensione per head
//  rope_dim : dimensioni su cui applicare RoPE
//  freq_base: base della frequenza (default 10000)
// ─────────────────────────────────────────────
void rope(float* x, int pos, int n_heads, int d_head, int rope_dim, float freq_base);

// ─────────────────────────────────────────────
//  SiLU activation
//
//  SiLU(x) = x * σ(x) = x / (1 + e^(-x))
//  Usata nel gate di SwiGLU.
//  Modifica il vettore in-place.
// ─────────────────────────────────────────────
void silu(float* x, int n);

// ─────────────────────────────────────────────
//  Kernel matvec quantizzati: y = A × x
//
//  Ogni kernel dequantizza una riga di A alla
//  volta e accumula il prodotto scalare con x,
//  senza mai materializzare la matrice in float32.
//
//  Parametri:
//    A       : pesi raw nel formato specifico
//    x       : vettore input  [in_dim]
//    y       : vettore output [out_dim]
//    out_dim : numero di righe di A
//    in_dim  : numero di colonne di A (= len di x)
// ─────────────────────────────────────────────
void matvec_q4k (const uint8_t*   A, const float* x, float* y, int out_dim, int in_dim);
void matvec_q6k (const uint8_t*   A, const float* x, float* y, int out_dim, int in_dim);
void matvec_q8_0(const uint8_t*   A, const float* x, float* y, int out_dim, int in_dim);
void matvec_f16 (const uint16_t*  A, const float* x, float* y, int out_dim, int in_dim);

// Dispatch automatico sul tipo di QuantTensor
// — usa A.n_rows come out_dim, A.n_cols come in_dim
void matvec_quant(const QuantTensor& A, const float* x, float* y);

// Prodotto matrice-vettore BATCH per pesi quantizzati.
//
//  y_batch[N × out_dim] = A @ x_batch^T
//
//  x_batch è una matrice [N × in_dim] in row-major:
//    token t occupa gli elementi [t*in_dim .. (t+1)*in_dim-1]
//
//  y_batch è una matrice [N × out_dim] in row-major.
//
//  La dequantizzazione avviene riga per riga di A
//  (come matvec_quant) ma ogni riga dequantizzata
//  viene riutilizzata per tutti i N token — questo
//  è il cuore dell'ottimizzazione del prefill batch.
void matvec_quant_batch(const QuantTensor& A, const float* x_batch, float* y_batch, int N);

// Dequantizza la riga row_idx di A in out[0..n_cols-1]
// Usata per l'embedding lookup (accesso singola riga)
void dequant_row(const QuantTensor& A, int row_idx, float* out);