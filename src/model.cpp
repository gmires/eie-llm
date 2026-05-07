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

// ─────────────────────────────────────────────
//  Self-Attention con KV Cache
//
//  Implementa la scaled dot-product attention
//  per un singolo token alla posizione pos.
//
//  Formula:
//    Attention(Q,K,V) = softmax(Q·Kᵀ / √d_head) · V
//
//  La KV cache evita di ricalcolare K e V
//  per i token già processati:
//  - al passo pos=0 processiamo solo pos 0
//  - al passo pos=1 Q viene da pos 1,
//    ma K e V vengono da pos 0 e 1
//  - al passo pos=n Q viene da pos n,
//    ma K e V vengono da pos 0..n
// ─────────────────────────────────────────────
void self_attention(const float* x, float* out, const LayerWeights& lw, KVCache& cache, const ModelConfig& cfg, int layer, int pos) {

    const int n_embd  = cfg.n_embd;
    const int n_head  = cfg.n_head;
    const int d_head  = cfg.d_head;

    // ── Step 1: calcola Q, K, V ───────────────
    //
    // La matrice attn_qkv_w ha shape [3*n_embd × n_embd]
    // e produce Q, K, V concatenati in un unico vettore.
    // Lo splittiamo in 3 parti uguali da n_embd ciascuna.
    std::vector<float> qkv(3 * n_embd);
    matvec(lw.attn_qkv_w.data(), x, qkv.data(), 3 * n_embd, n_embd);

    // Aggiungi i bias
    vec_add(qkv.data(), lw.attn_qkv_b.data(), qkv.data(), 3 * n_embd);

    // Puntatori alle 3 sezioni del vettore qkv
    const float* Q = qkv.data();
    const float* K = qkv.data() + n_embd;
    const float* V = qkv.data() + 2 * n_embd;

    // ── Step 2: salva K e V nella cache ───────
    //
    // Ogni posizione occupa n_embd float nella cache.
    // Layout: cache[pos * n_embd + i]
    float* k_pos = cache.k[layer].data() + pos * n_embd;
    float* v_pos = cache.v[layer].data() + pos * n_embd;
    vec_copy(K, k_pos, n_embd);
    vec_copy(V, v_pos, n_embd);

    // ── Step 3: attention per ogni head ───────
    //
    // Ogni head lavora su una "fetta" di Q, K, V
    // di dimensione d_head = n_embd / n_head.
    // Head h lavora su indici [h*d_head, (h+1)*d_head)
    std::vector<float> attn_out(n_embd, 0.0f);

    // scores[t] conterrà il peso di attenzione
    // del token corrente verso il token t
    std::vector<float> scores(pos + 1);

    for (int h = 0; h < n_head; h++) {
        // Offset della fetta di questo head
        int h_off = h * d_head;

        // Q per questo head
        const float* Qh = Q + h_off;

        // ── Calcola scores = Q·Kᵀ / √d_head ──
        //
        // Per ogni posizione t già in cache
        // calcoliamo il dot product di Qh con Kh[t].
        // Dividiamo per √d_head per stabilità
        // (evita che i dot product crescano troppo
        //  con d_head grande, saturando il softmax)
        float scale = 1.0f / sqrtf((float)d_head);

        for (int t = 0; t <= pos; t++) {
            // K per il token t, head h dalla cache
            const float* Kh_t = cache.k[layer].data()
                                + t * n_embd + h_off;

            // Dot product Q·K
            float dot = 0.0f;
            for (int d = 0; d < d_head; d++)
                dot += Qh[d] * Kh_t[d];

            scores[t] = dot * scale;
        }

        // ── Softmax sugli scores ───────────────
        //
        // Trasforma i punteggi grezzi in pesi
        // di attenzione che sommano a 1.
        // La maschera causale è implicita:
        // iteriamo solo su t <= pos quindi non
        // vediamo mai token futuri.
        softmax(scores.data(), pos + 1);

        // ── Weighted sum dei V ─────────────────
        //
        // L'output di questo head è la somma
        // pesata dei vettori V di tutti i token,
        // dove i pesi sono gli attention scores.
        float* out_h = attn_out.data() + h_off;

        for (int t = 0; t <= pos; t++) {
            const float* Vh_t = cache.v[layer].data()
                                + t * n_embd + h_off;
            for (int d = 0; d < d_head; d++)
                out_h[d] += scores[t] * Vh_t[d];
        }
    }

    // ── Step 4: proiezione output ──────────────
    //
    // Riproietta il risultato dell'attention
    // nello spazio n_embd con una matrice lineare.
    // attn_out_w: [n_embd × n_embd]
    matvec(lw.attn_out_w.data(), attn_out.data(), out, n_embd, n_embd);
    vec_add(out, lw.attn_out_b.data(), out, n_embd);
}