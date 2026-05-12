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
//  Tipo di RoPE (Rotary Position Embedding)
//
//  NORM  — LLaMA standard: ruota coppie consecutive
//          (x[0],x[1]), (x[2],x[3])...
//  NEOX  — Qwen2, Phi, Gemma: ruota coppie scambiate
//          (x[0],x[half]), (x[1],x[half+1])...
// ─────────────────────────────────────────────
enum class RopeType {
    NORM,
    NEOX
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
    RopeType rope_type   = RopeType::NORM;  // tipo di RoPE (NORM/NEOX)
};

// ─────────────────────────────────────────────
//  Pesi di un singolo layer transformer
//
//  Le matrici di peso grandi (attn, FFN) sono
//  mantenute nel formato GGUF originale come
//  QuantTensor — la dequantizzazione avviene
//  riga per riga durante matvec_quant().
//
//  Norme e bias restano float32: sono piccoli
//  (n_embd float = 8 KB) e generalmente già F32
//  nel file GGUF.
// ─────────────────────────────────────────────
struct LayerWeights {
    // ── Norm prima dell'attention ─────────────
    std::vector<float> ln1_w;   // GPT2: LayerNorm gamma | LLaMA: RMSNorm
    std::vector<float> ln1_b;   // GPT2: LayerNorm beta  | LLaMA: non usato

    // ── Self-attention ────────────────────────
    // GPT2: unica matrice QKV combinata
    QuantTensor        attn_qkv_w;
    std::vector<float> attn_qkv_b;

    // LLaMA: Q, K, V separati (K e V più piccoli con GQA)
    QuantTensor attn_q_w;
    QuantTensor attn_k_w;
    QuantTensor attn_v_w;
    // Qwen2 ha bias su Q, K, V (LLaMA standard no)
    std::vector<float> attn_q_b;
    std::vector<float> attn_k_b;
    std::vector<float> attn_v_b;

    // Output projection (comune a entrambi)
    QuantTensor        attn_out_w;
    std::vector<float> attn_out_b;  // GPT2 ha il bias, LLaMA no

    // ── Norm prima del FFN ────────────────────
    std::vector<float> ln2_w;
    std::vector<float> ln2_b;   // LLaMA: non usato

    // ── FFN ───────────────────────────────────
    // GPT2: fc1 (up) + fc2 (down) con GELU
    QuantTensor        ffn_fc1_w;
    std::vector<float> ffn_fc1_b;
    QuantTensor        ffn_fc2_w;
    std::vector<float> ffn_fc2_b;

    // LLaMA: gate + up + down con SwiGLU (no bias)
    QuantTensor ffn_gate_w;  // W1 — passa per SiLU
    QuantTensor ffn_up_w;    // W3 — moltiplicato col gate
    QuantTensor ffn_down_w;  // W2 — proiezione finale
};

// ─────────────────────────────────────────────
//  Pesi globali del modello
//
//  Separati dai layer weights perché esistono
//  una sola volta (non per ogni layer)
// ─────────────────────────────────────────────
struct ModelWeights {
    // Token embedding — grande (n_vocab × n_embd), rimane quantizzato
    // Usato sia come embedding lookup (dequant_row) sia come lm_head (matvec_quant)
    QuantTensor token_embd;

    // GPT2: positional embedding assoluto — piccolo (n_ctx × n_embd), float32
    std::vector<float> pos_embd;

    // LayerNorm/RMSNorm finale — piccolo, float32
    std::vector<float> ln_f_w;
    std::vector<float> ln_f_b;   // LLaMA: non usato

    // LLaMA: lm_head separato — grande (n_vocab × n_embd), rimane quantizzato
    QuantTensor output_w;

    // Qwen2: bias opzionale sul lm_head (output.bias)
    std::vector<float> output_b;

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
//  Attention Snapshot — salva i pesi attention
//
//  Struttura per esportare la matrice di attention
//  dopo softmax per ogni layer, head, query e key.
//  Usata dall'endpoint /v1/inspect/attention per
//  "guardare dentro" il modello.
//
//  Dimensione: n_layer × n_head × seq_len × seq_len
//  Per GPT-2 small con prompt di 10 token:
//    12 × 12 × 10 × 10 = 14,400 float ≈ 57 KB
//  Per TinyLlama con prompt di 20 token:
//    22 × 32 × 20 × 20 = 281,600 float ≈ 1.1 MB
//  La memoria cresce come O(seq_len²) quindi si
//  limita ai prompt corti (≤ 100 token).
// ─────────────────────────────────────────────
struct AttentionSnapshot {
    int n_layers = 0;   // numero di layer
    int n_heads  = 0;   // numero di attention heads
    int seq_len  = 0;   // lunghezza della sequenza

    // Dati flat: index = ((layer * n_heads + head) * seq_len + q_pos) * seq_len + k_pos
    std::vector<float> data;

    // Inizializza le dimensioni e alloca la memoria
    void init(int nl, int nh, int sl) {
        n_layers = nl;
        n_heads  = nh;
        seq_len  = sl;
        data.assign(static_cast<size_t>(nl) * nh * sl * sl, 0.0f);
    }

    // Accesso alla cella [layer][head][q][k]
    float& at(int layer, int head, int q_pos, int k_pos) {
        return data[static_cast<size_t>(((layer * n_heads + head) * seq_len + q_pos) * seq_len + k_pos)];
    }

    const float& at(int layer, int head, int q_pos, int k_pos) const {
        return data[static_cast<size_t>(((layer * n_heads + head) * seq_len + q_pos) * seq_len + k_pos)];
    }
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
// Se snap != nullptr, salva gli attention scores (post-softmax)
// nello snapshot per visualizzazione.
void self_attention(const float* x, float* out, const LayerWeights& lw, KVCache& cache, const ModelConfig& cfg, int layer, int pos, InferBuffers& bufs, AttentionSnapshot* snap = nullptr);

// Feed-Forward Network GPT-2.
// Usa bufs.gate come buffer intermedio (proiezione up).
void feed_forward(const float* x, float* out, const LayerWeights& lw, const ModelConfig& cfg, InferBuffers& bufs);

// Self-attention LLaMA con GQA e RoPE.
// Usa bufs.Q, bufs.K, bufs.V, bufs.scores come buffer temporanei.
// Se snap != nullptr, salva gli attention scores (post-softmax)
// nello snapshot per visualizzazione.
void self_attention_llama(const float* x, float* out, const LayerWeights& lw, KVCache& cache, const ModelConfig& cfg, int layer, int pos, InferBuffers& bufs, AttentionSnapshot* snap = nullptr);

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
// snap     : se non nullptr, salva gli attention scores
void forward(Model& model, int token_id, int pos, std::vector<float>& logits, bool bench_mode = false, AttentionSnapshot* snap = nullptr);

// Forward pass BATCH per il prefill.
//
// Processa tutti i token del prompt in un UNICO
// passaggio attraverso i layer, sfruttando il
// riutilizzo dei pesi in cache (matvec_quant_batch).
//
// Dopo il prefill:
//   - kv_cache.n_cached = token_ids.size()
//   - bufs.x contiene l'output dell'ULTIMO token
//   - logits contiene i logits dell'ULTIMO token
//     (quelli necessari per il primo sampling)
//
// La generazione autoregressiva continua con
// forward() token per token come al solito.
void forward_prefill(Model& model, const std::vector<int>& token_ids, std::vector<float>& logits);

// Forward pass BATCH per verifica (Speculative Decoding).
//
// Come forward_prefill ma partendo da base_pos e
// restituendo i logits per OGNI token del batch.
// Gli attention attendono a TUTTO il contesto precedente.
//
// Usato per verificare K token draft in un solo
// passaggio: confronta argmax(logits_i) con draft_{i+1}.
void forward_verify(Model& model, const std::vector<int>& token_ids,
                    int base_pos, std::vector<std::vector<float>>& all_logits);

// Esporta gli attention scores per un prompt.
//
// Esegue il forward pass sequenziale su tutti i token
// del prompt e salva gli attention scores (post-softmax)
// per ogni layer, head, query e key.
//
// Restituisce un JSON con:
//   - tokens: array di stringhe (i token del prompt)
//   - layers: array di layer, ognuno con array di head,
//     ognuno con una matrice [q][k] di pesi attention.
//
// max_len: lunghezza massima del prompt (per limitare
//          la memoria O(seq_len²)). Default 100.
std::string inspect_attention(Model& model, const Tokenizer& tok,
                               const std::string& prompt, int max_len = 100);

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

