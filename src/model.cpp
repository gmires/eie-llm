#include "model.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

// ─────────────────────────────────────────────
//  Helper: leggi uint32 dai metadata
// ─────────────────────────────────────────────
static uint32_t get_u32(const GGUFContext& ctx,
                        const std::string& key,
                        uint32_t def = 0) {
    for (const auto& kv : ctx.metadata)
        if (kv.key == key)
            if (auto* v = kv.value.get_if<uint32_t>())
                return *v;
    return def;
}

// ─────────────────────────────────────────────
//  Helper: carica un tensore per nome
//  e dequantizza in float32.
//  Termina il programma se non trovato —
//  un tensore mancante è un errore fatale.
// ─────────────────────────────────────────────
static std::vector<float> load_tensor(const GGUFContext& ctx,
                                      const std::string& name) {
    const GGUFTensor* t = gguf_find_tensor(ctx, name);
    if (!t) {
        std::cerr << "[ERRORE] Tensore non trovato: " << name << "\n";
        return {};
    }
    return tensor_to_float(*t);
}

// ─────────────────────────────────────────────
//  Carica la configurazione dai metadata GGUF
//
//  GPT-2 small ha questi valori:
//    n_vocab  = 50257
//    n_ctx    = 1024
//    n_embd   = 768
//    n_head   = 12
//    n_layer  = 12
//    n_ff     = 3072  (4 × n_embd, non sempre
//                      presente nei metadata —
//                      la calcoliamo noi)
// ─────────────────────────────────────────────
bool model_load_config(ModelConfig& cfg, const GGUFContext& ctx) {
    cfg.n_vocab = get_u32(ctx, "tokenizer.ggml.vocab_size", 50257);
    cfg.n_ctx   = get_u32(ctx, "gpt2.context_length",       1024);
    cfg.n_embd  = get_u32(ctx, "gpt2.embedding_length",     768);
    cfg.n_head  = get_u32(ctx, "gpt2.attention.head_count", 12);
    cfg.n_layer = get_u32(ctx, "gpt2.block_count",          12);

    // n_ff non è sempre nei metadata — usiamo il valore
    // standard GPT-2: 4 × n_embd
    cfg.n_ff    = get_u32(ctx, "gpt2.feed_forward_length",
                          cfg.n_embd * 4);

    // d_head si calcola sempre da n_embd / n_head
    cfg.d_head  = cfg.n_embd / cfg.n_head;

    if (cfg.n_embd == 0 || cfg.n_head == 0 || cfg.n_layer == 0) {
        std::cerr << "[ERRORE] Configurazione non valida\n";
        return false;
    }
    return true;
}

// ─────────────────────────────────────────────
//  Carica tutti i pesi del modello
//
//  I nomi dei tensori in GPT-2 GGUF seguono
//  questa convenzione:
//
//  Globali:
//    token_embd.weight       → token embeddings
//    position_embd.weight    → positional embeddings
//    output_norm.weight/bias → layer norm finale
//
//  Per layer i (0-based):
//    blk.i.attn_norm.weight/bias  → ln1
//    blk.i.attn_qkv.weight/bias   → Q/K/V proiez.
//    blk.i.attn_output.weight/bias → out proiez.
//    blk.i.ffn_norm.weight/bias   → ln2
//    blk.i.ffn_up.weight/bias     → fc1
//    blk.i.ffn_down.weight/bias   → fc2
// ─────────────────────────────────────────────
bool model_load_weights(Model& model, const GGUFContext& ctx) {
    const ModelConfig& cfg = model.config;
    ModelWeights& w = model.weights;

    std::cout << "  Caricamento pesi globali...\n";

    // ── Pesi globali ──────────────────────────
    w.token_embd = load_tensor(ctx, "token_embd.weight");
    if (w.token_embd.empty()) return false;

    w.pos_embd = load_tensor(ctx, "position_embd.weight");
    if (w.pos_embd.empty()) return false;

    w.ln_f_w = load_tensor(ctx, "output_norm.weight");
    w.ln_f_b = load_tensor(ctx, "output_norm.bias");
    if (w.ln_f_w.empty() || w.ln_f_b.empty()) return false;

    // ── Pesi per layer ────────────────────────
    std::cout << "  Caricamento " << cfg.n_layer << " layer...\n";
    w.layers.resize(cfg.n_layer);

    for (int i = 0; i < cfg.n_layer; i++) {
        std::string p = "blk." + std::to_string(i) + ".";
        LayerWeights& lw = w.layers[i];

        // Layer norm 1
        lw.ln1_w = load_tensor(ctx, p + "attn_norm.weight");
        lw.ln1_b = load_tensor(ctx, p + "attn_norm.bias");

        // Attention Q/K/V
        lw.attn_qkv_w = load_tensor(ctx, p + "attn_qkv.weight");
        lw.attn_qkv_b = load_tensor(ctx, p + "attn_qkv.bias");

        // Attention output
        lw.attn_out_w = load_tensor(ctx, p + "attn_output.weight");
        lw.attn_out_b = load_tensor(ctx, p + "attn_output.bias");

        // Layer norm 2
        lw.ln2_w = load_tensor(ctx, p + "ffn_norm.weight");
        lw.ln2_b = load_tensor(ctx, p + "ffn_norm.bias");

        // FFN
        lw.ffn_fc1_w = load_tensor(ctx, p + "ffn_up.weight");
        lw.ffn_fc1_b = load_tensor(ctx, p + "ffn_up.bias");
        lw.ffn_fc2_w = load_tensor(ctx, p + "ffn_down.weight");
        lw.ffn_fc2_b = load_tensor(ctx, p + "ffn_down.bias");

        // Verifica che i tensori critici siano stati caricati
        if (lw.attn_qkv_w.empty() || lw.ffn_fc1_w.empty()) {
            std::cerr << "[ERRORE] Layer " << i
                      << ": tensori mancanti\n";
            return false;
        }
    }

    return true;
}

// ─────────────────────────────────────────────
//  Inizializza la KV Cache
//
//  Alloca la memoria per tutti i layer.
//  Ogni layer ha due buffer (K e V) di dimensione:
//    n_ctx × n_head × d_head = n_ctx × n_embd
//  Inizializzati a zero.
// ─────────────────────────────────────────────
void model_init_kvcache(Model& model) {
    const ModelConfig& cfg = model.config;
    int cache_size = cfg.n_ctx * cfg.n_embd;

    model.kv_cache.k.assign(cfg.n_layer,
                            std::vector<float>(cache_size, 0.0f));
    model.kv_cache.v.assign(cfg.n_layer,
                            std::vector<float>(cache_size, 0.0f));
    model.kv_cache.n_cached = 0;
}

// ─────────────────────────────────────────────
//  Stampa configurazione
// ─────────────────────────────────────────────
void model_print_config(const ModelConfig& cfg) {
    std::cout << "\n═══════════════════════════════════════\n";
    std::cout << "  EIE-LLM — Model Config\n";
    std::cout << "═══════════════════════════════════════\n";
    std::cout << "  n_vocab  : " << cfg.n_vocab  << "\n";
    std::cout << "  n_ctx    : " << cfg.n_ctx    << "\n";
    std::cout << "  n_embd   : " << cfg.n_embd   << "\n";
    std::cout << "  n_head   : " << cfg.n_head   << "\n";
    std::cout << "  n_layer  : " << cfg.n_layer  << "\n";
    std::cout << "  n_ff     : " << cfg.n_ff     << "\n";
    std::cout << "  d_head   : " << cfg.d_head   << "\n";
    std::cout << "═══════════════════════════════════════\n\n";
}

// ─────────────────────────────────────────────
//  Layer Normalization
//
//  Algoritmo:
//  1) Calcola la media: μ = (1/n) Σ x[i]
//  2) Calcola la varianza: σ² = (1/n) Σ (x[i]-μ)²
//  3) Normalizza: x̂[i] = (x[i] - μ) / √(σ² + ε)
//  4) Scala e trasla: out[i] = w[i] * x̂[i] + b[i]
//
//  w (gamma) e b (beta) sono parametri appresi
//  che permettono alla rete di "de-normalizzare"
//  se necessario.
//  ε = 1e-5 evita divisione per zero quando σ²≈0
// ─────────────────────────────────────────────
void layer_norm(const float* x, const float* w, const float* b,
                float* out, int n) {
    static constexpr float EPS = 1e-5f;

    // Calcola la media
    float mean = 0.0f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;

    // Calcola la varianza
    float var = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= n;

    // Normalizza, scala e trasla
    float inv_std = 1.0f / sqrtf(var + EPS);
    for (int i = 0; i < n; i++)
        out[i] = w[i] * ((x[i] - mean) * inv_std) + b[i];
}

// ─────────────────────────────────────────────
//  Embedding lookup
//
//  L'embedding è una semplice operazione di
//  indicizzazione: ogni token ID seleziona
//  una riga dalla matrice degli embedding.
//
//  GPT-2 usa embeddings assoluti di posizione
//  (a differenza di RoPE usato da LLaMA):
//  il positional embedding per la posizione pos
//  viene semplicemente sommato al token embedding.
//
//  out = token_embd[token_id] + pos_embd[pos]
// ─────────────────────────────────────────────
void embedding_lookup(const float* token_embd,
                      const float* pos_embd,
                      int token_id, int pos,
                      float* out, int n_embd) {
    // Puntatore alla riga del token embedding
    const float* te = token_embd + token_id * n_embd;
    // Puntatore alla riga del positional embedding
    const float* pe = pos_embd   + pos       * n_embd;

    // Somma elemento per elemento
    for (int i = 0; i < n_embd; i++)
        out[i] = te[i] + pe[i];
}