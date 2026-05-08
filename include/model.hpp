#pragma once
#include <vector>
#include <string>
#include "gguf.hpp"
#include "ops.hpp"
#include "bench.hpp"

// ─────────────────────────────────────────────
//  Tipo di architettura
//  Determina quali blocchi usare nel forward pass
// ─────────────────────────────────────────────
enum class ArchType {
    GPT2,   // LayerNorm, pos embedding assoluto, QKV unito, GELU
    LLAMA,  // RMSNorm, RoPE, Q/K/V separati, GQA, SwiGLU
};

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
    ArchType arch; // tipo di architettura
    int n_vocab;   // dimensione vocabolario (50257)
    int n_ctx;     // context length massimo  (1024)
    int n_embd;    // dimensione embedding    (768)
    int n_head;    // numero attention heads  (12)
    int n_head_kv        = 0;
    int n_layer;   // numero di layer         (12)
    int n_ff;      // dimensione feed-forward (3072 = 4 × n_embd)
    int d_head;    // dimensione per head     (64  = n_embd / n_head)
    int rope_dim         = 0;   // dimensioni RoPE (LLaMA)
    float norm_eps       = 1e-5f;  // epsilon per RMSNorm/LayerNorm
    float rope_freq_base = 10000.0f;    
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
    // ── Norm prima dell'attention ─────────────
    std::vector<float> ln1_w;   // GPT2: LayerNorm gamma | LLaMA: RMSNorm
    std::vector<float> ln1_b;   // GPT2: LayerNorm beta  | LLaMA: non usato

    // ── Self-attention ────────────────────────
    // GPT2: unica matrice QKV combinata
    std::vector<float> attn_qkv_w;
    std::vector<float> attn_qkv_b;

    // LLaMA: Q, K, V separati (K e V più piccoli con GQA)
    std::vector<float> attn_q_w;
    std::vector<float> attn_k_w;
    std::vector<float> attn_v_w;

    // Output projection (comune a entrambi)
    std::vector<float> attn_out_w;
    std::vector<float> attn_out_b;  // GPT2 ha il bias, LLaMA no

    // ── Norm prima del FFN ────────────────────
    std::vector<float> ln2_w;
    std::vector<float> ln2_b;   // LLaMA: non usato

    // ── FFN ───────────────────────────────────
    // GPT2: fc1 (up) + fc2 (down) con GELU
    std::vector<float> ffn_fc1_w;
    std::vector<float> ffn_fc1_b;
    std::vector<float> ffn_fc2_w;
    std::vector<float> ffn_fc2_b;

    // LLaMA: gate + up + down con SwiGLU (no bias)
    std::vector<float> ffn_gate_w;  // W1 — passa per SiLU
    std::vector<float> ffn_up_w;    // W3 — moltiplicato col gate
    std::vector<float> ffn_down_w;  // W2 — proiezione finale
};

// ─────────────────────────────────────────────
//  Pesi globali del modello
//
//  Separati dai layer weights perché esistono
//  una sola volta (non per ogni layer)
// ─────────────────────────────────────────────
struct ModelWeights {
    // Token embedding (comune)
    std::vector<float> token_embd;

    // GPT2: positional embedding assoluto
    std::vector<float> pos_embd;

    // LayerNorm/RMSNorm finale
    std::vector<float> ln_f_w;
    std::vector<float> ln_f_b;   // LLaMA: non usato

    // LLaMA: lm_head separato (non usa weight tying)
    std::vector<float> output_w;

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
//  Buffer temporanei per l'inferenza
//
//  Ogni forward step alloca e dealloca decine di
//  vettori temporanei (Q, K, V, gate, up, scores…).
//  Pre-allocarli una sola volta elimina centinaia
//  di malloc/free per token generato.
//
//  Dimensioni al worst case (LLaMA):
//    x, residual, ln_out, attn_out, ffn_out, ln_final → n_embd (2048)
//    Q                                                 → n_embd (2048)
//    K, V                                              → kv_dim (256)
//    gate, up                                          → n_ff   (5632)
//    scores                                            → n_ctx  (2048)
// ─────────────────────────────────────────────
struct InferBuffers {
    // ── Buffer del forward pass ───────────────
    std::vector<float> x;        // stato corrente del token [n_embd]
    std::vector<float> residual; // salvataggio residual connection [n_embd]
    std::vector<float> ln_out;   // output della norm (prima di attention o FFN) [n_embd]
    std::vector<float> attn_out; // output FINALE dell'attention (dopo proiezione) [n_embd]
    std::vector<float> ffn_out;  // output del FFN [n_embd]
    std::vector<float> ln_final; // output della norm finale [n_embd]

    // ── Buffer interni dell'attention ─────────
    // ATTENZIONE: attn_acc e attn_out sono buffer DISTINTI.
    // attn_acc = accumulatore della weighted sum di V (dentro self_attention)
    // attn_out = output della proiezione finale (scritto da matvec su attn_acc)
    // Devono essere separati perché il matvec finale legge da attn_acc
    // e scrive in attn_out: se coincidessero si avrebbe aliasing.
    std::vector<float> attn_acc; // accumulatore weighted sum V [n_embd]
    std::vector<float> Q;        // query vettori [n_embd]
    std::vector<float> K;        // key vettori [kv_dim]
    std::vector<float> V;        // value vettori [kv_dim]
    std::vector<float> scores;   // attention scores [n_ctx]

    // ── Buffer interni del FFN ────────────────
    std::vector<float> gate;     // gate SwiGLU dopo SiLU [n_ff]
    std::vector<float> up;       // up projection SwiGLU [n_ff]
};

// ─────────────────────────────────────────────
//  Modello completo
// ─────────────────────────────────────────────
struct Model {
    ModelConfig  config;
    ModelWeights weights;
    KVCache      kv_cache;
    InferBuffers bufs;     // buffer pre-allocati per l'inferenza
    BenchAccum   bench;
};

// ─────────────────────────────────────────────
//  Funzioni pubbliche del modulo model
// ─────────────────────────────────────────────

// Legge gli iperparametri dai metadata GGUF
bool model_load_config(ModelConfig& cfg, const GGUFContext& ctx);

// Carica tutti i pesi dal GGUFContext
// dequantizzando in float32
bool model_load_weights(Model& model, const GGUFContext& ctx);

// Inizializza la KV cache e pre-alloca i buffer di inferenza.
// Deve essere chiamata dopo model_load_config.
// Chiamarla di nuovo resetta la KV cache (nuova conversazione).
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
// Self-attention GPT-2.
// Usa bufs.scores come buffer temporaneo per gli attention scores.
void self_attention(const float* x, float* out, const LayerWeights& lw, KVCache& cache, const ModelConfig& cfg, int layer, int pos, InferBuffers& bufs);

// Feed-Forward Network GPT-2.
// Usa bufs.gate come buffer intermedio (proiezione up).
void feed_forward(const float* x, float* out, const LayerWeights& lw, const ModelConfig& cfg, InferBuffers& bufs);

// Self-attention LLaMA con GQA e RoPE.
// Usa bufs.Q, bufs.K, bufs.V, bufs.scores come buffer temporanei.
void self_attention_llama(const float* x, float* out, const LayerWeights& lw, KVCache& cache, const ModelConfig& cfg, int layer, int pos, InferBuffers& bufs);

// Feed-forward LLaMA con SwiGLU.
// Usa bufs.gate e bufs.up come buffer temporanei.
// out = ffn_down_w · ( SiLU(ffn_gate_w·x) ⊙ (ffn_up_w·x) )
void feed_forward_llama(const float* x, float* out, const LayerWeights& lw, const ModelConfig& cfg, InferBuffers& bufs);

// Forward pass completo per un singolo token
//
// Esegue tutti i 12 layer del transformer
// e ritorna i logits sul vocabolario.
// Aggiorna la KV cache internamente.
//
// token_id : ID del token corrente
// pos      : posizione nella sequenza
// logits   : output [n_vocab] — probabilità grezze
void forward(Model& model, int token_id, int pos, std::vector<float>& logits, bool bench_mode = false);

// ─────────────────────────────────────────────
//  Parametri di sampling raggruppati
//
//  Raccoglie tutti i parametri in una struct
//  per passarli comodamente alle funzioni
//  di generazione senza liste di argomenti lunghe
// ─────────────────────────────────────────────
struct SamplingParams {
    float temperature  = 1.0f;   // temperatura del sampling
    float top_p        = 0.9f;   // nucleus sampling
    int   top_k        = 40;     // mantieni solo i k token più probabili
    float rep_penalty  = 1.1f;   // repetition penalty
    bool  greedy       = false;  // ignora tutto e fa argmax
};

// Sampling greedy — sceglie il token con
// probabilità massima (argmax sui logits)
int sample_argmax(const std::vector<float>& logits);

// Sampling con temperatura
//
// temperature > 1 → distribuzione più uniforme (più creativo)
// temperature < 1 → distribuzione più concentrata (più deterministico)
// temperature = 1 → sampling dalla distribuzione originale
int sample_temperature(std::vector<float> logits, float temperature);

// ─────────────────────────────────────────────
//  Top-p (nucleus) sampling
//
//  Campiona solo dal sottoinsieme di token
//  la cui probabilità cumulativa raggiunge p.
//  Questo elimina i token improbabili mantenendo
//  la varietà dell'output.
//
//  Algoritmo:
//  1) Ordina i token per probabilità decrescente
//  2) Prendi i primi k token tali che la loro
//     somma cumulativa >= p (nucleus)
//  3) Rinormalizza le probabilità nel nucleus
//  4) Campiona dal nucleus
//
//  p = 1.0 → equivale a sampling normale
//  p = 0.9 → top 90% della distribuzione
//  p = 0.1 → molto conservativo
// ─────────────────────────────────────────────
int sample_topp(std::vector<float> logits, float p, float temperature);

// ─────────────────────────────────────────────
//  Repetition penalty
//
//  Penalizza i token già presenti nel contesto
//  per ridurre le ripetizioni.
//
//  Formula (come in llama.cpp):
//    se logit > 0: logit /= penalty
//    se logit < 0: logit *= penalty
//
//  Questo abbassa i logit positivi e abbassa
//  ulteriormente quelli negativi — in entrambi
//  i casi il token diventa meno probabile.
//
//  penalty = 1.0 → nessun effetto
//  penalty = 1.1 → leggera penalità
//  penalty = 1.3 → penalità moderata (consigliato)
//  penalty = 1.5 → penalità forte
//
//  context_ids : token già generati (prompt + output)
// ─────────────────────────────────────────────
void apply_repetition_penalty(std::vector<float>& logits, const std::vector<int>& context_ids, float penalty);

// ─────────────────────────────────────────────
//  Top-k sampling
//
//  Mantiene solo i k token con logit più alto
//  e azzera tutti gli altri prima del softmax.
//  Riduce il rischio di campionare token
//  completamente improbabili.
//
//  k = 1   → equivale a greedy
//  k = 40  → valore tipico usato da GPT-2
//  k = 0   → disabilitato (tutti i token)
// ─────────────────────────────────────────────
int sample_topk_topp(std::vector<float> logits, int   top_k, float top_p, float temperature);

