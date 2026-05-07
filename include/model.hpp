#pragma once
#include <vector>
#include <string>
#include "gguf.hpp"
#include "ops.hpp"

// ─────────────────────────────────────────────
//  Iperparametri del modello
//
//  Tutti i valori vengono letti dai metadata
//  GGUF — non sono hardcoded. Questo permette
//  di supportare qualsiasi variante GPT-2
//  (small, medium, large, xl) con lo stesso
//  codice cambiando solo il file .gguf
// ─────────────────────────────────────────────
struct ModelConfig {
    int n_vocab;   // dimensione vocabolario (50257)
    int n_ctx;     // context length massimo  (1024)
    int n_embd;    // dimensione embedding    (768)
    int n_head;    // numero attention heads  (12)
    int n_layer;   // numero di layer         (12)
    int n_ff;      // dimensione feed-forward (3072 = 4 × n_embd)
    int d_head;    // dimensione per head     (64  = n_embd / n_head)
};

// ─────────────────────────────────────────────
//  Pesi di un singolo layer transformer
//
//  Ogni layer ha 8 tensori di pesi:
//  - 2 per la layer norm prima dell'attention
//  - 4 per le proiezioni Q/K/V e output
//  - 2 per la layer norm prima del FFN
//  - 4 per il feed-forward network (fc1, fc2
//    con bias rispettivi)
//
//  I tensori sono già dequantizzati in float32
//  pronti per il calcolo.
// ─────────────────────────────────────────────
struct LayerWeights {
    // LayerNorm 1 — prima della self-attention
    std::vector<float> ln1_w;   // gamma [n_embd]
    std::vector<float> ln1_b;   // beta  [n_embd]

    // Self-attention — proiezioni Q, K, V combinate
    // In GPT-2 le proiezioni Q/K/V sono una sola
    // matrice [3*n_embd × n_embd] poi splittata
    std::vector<float> attn_qkv_w;  // [3*n_embd × n_embd]
    std::vector<float> attn_qkv_b;  // [3*n_embd]

    // Self-attention — proiezione output
    std::vector<float> attn_out_w;  // [n_embd × n_embd]
    std::vector<float> attn_out_b;  // [n_embd]

    // LayerNorm 2 — prima del FFN
    std::vector<float> ln2_w;   // gamma [n_embd]
    std::vector<float> ln2_b;   // beta  [n_embd]

    // Feed-Forward Network
    std::vector<float> ffn_fc1_w;  // [n_ff × n_embd]
    std::vector<float> ffn_fc1_b;  // [n_ff]
    std::vector<float> ffn_fc2_w;  // [n_embd × n_ff]
    std::vector<float> ffn_fc2_b;  // [n_embd]
};

// ─────────────────────────────────────────────
//  Pesi globali del modello
//
//  Separati dai layer weights perché esistono
//  una sola volta (non per ogni layer)
// ─────────────────────────────────────────────
struct ModelWeights {
    // Token embedding: ogni token ID → vettore float
    // Matrice [n_vocab × n_embd]
    std::vector<float> token_embd;

    // Positional embedding: ogni posizione → vettore float
    // Matrice [n_ctx × n_embd]
    std::vector<float> pos_embd;

    // LayerNorm finale (dopo tutti i layer)
    std::vector<float> ln_f_w;  // [n_embd]
    std::vector<float> ln_f_b;  // [n_embd]

    // Pesi dei layer (uno per ogni transformer block)
    std::vector<LayerWeights> layers;
};

// ─────────────────────────────────────────────
//  KV Cache — memoria dell'attention
//
//  Durante la generazione autoregressiva
//  ricalcolare K e V per tutti i token
//  precedenti ad ogni step sarebbe O(n²).
//  La KV cache salva K e V già calcolati
//  e li riusa negli step successivi → O(n).
//
//  Struttura per ogni layer:
//    k_cache[pos][head][d_head]
//    v_cache[pos][head][d_head]
//  Memorizzata flat come:
//    [n_ctx × n_head × d_head]
// ─────────────────────────────────────────────
struct KVCache {
    // k_cache[layer][pos * n_head * d_head + head * d_head + d]
    std::vector<std::vector<float>> k;  // un vettore per layer
    std::vector<std::vector<float>> v;  // un vettore per layer
    int n_cached = 0;  // quanti token sono già in cache
};

// ─────────────────────────────────────────────
//  Modello completo
// ─────────────────────────────────────────────
struct Model {
    ModelConfig  config;
    ModelWeights weights;
    KVCache      kv_cache;
};

// ─────────────────────────────────────────────
//  Funzioni pubbliche del modulo model
// ─────────────────────────────────────────────

// Legge gli iperparametri dai metadata GGUF
bool model_load_config(ModelConfig& cfg, const GGUFContext& ctx);

// Carica tutti i pesi dal GGUFContext
// dequantizzando in float32
bool model_load_weights(Model& model, const GGUFContext& ctx);

// Inizializza la KV cache (alloca la memoria)
void model_init_kvcache(Model& model);

// Stampa la configurazione del modello
void model_print_config(const ModelConfig& cfg);

// ─────────────────────────────────────────────
//  Operazioni del forward pass
// ─────────────────────────────────────────────

// Layer Normalization
//
// Normalizza un vettore x di dimensione n:
//   1) calcola media e varianza di x
//   2) normalizza: x̂ = (x - mean) / sqrt(var + eps)
//   3) scala e trasla: out = gamma * x̂ + beta
//
// gamma (w) e beta (b) sono i pesi appresi.
// eps = 1e-5 evita divisione per zero.
void layer_norm(const float* x, const float* w, const float* b, float* out, int n);

// Embedding lookup
//
// Dato un token ID, estrae il suo vettore embedding
// dalla matrice dei token embeddings e somma
// il positional embedding per la posizione pos.
//
// out[i] = token_embd[token_id * n_embd + i]
//        + pos_embd[pos * n_embd + i]
void embedding_lookup(const float* token_embd, const float* pos_embd, int token_id, int pos, float* out, int n_embd);

// Self-attention per un singolo token
//
// x       : input [n_embd] — già layer-normalizzato
// out     : output [n_embd]
// lw      : pesi del layer corrente
// cache   : KV cache da aggiornare
// cfg     : configurazione del modello
// layer   : indice del layer (per la KV cache)
// pos     : posizione del token corrente
void self_attention(const float* x, float* out, const LayerWeights& lw, KVCache& cache, const ModelConfig& cfg, int layer, int pos);