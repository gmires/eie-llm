#include "model.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <unordered_set>

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
//  Helper: carica un tensore e dequantizza in float32.
//  Usato per tensori piccoli (norme, bias) che
//  vengono sempre acceduti come float32.
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
//  Helper: carica un tensore mantenendo il
//  formato quantizzato originale (QuantTensor).
//  Usato per le matrici di peso grandi —
//  la dequantizzazione avviene riga per riga
//  durante matvec_quant() al momento del calcolo.
// ─────────────────────────────────────────────
static QuantTensor load_quant_tensor(const GGUFContext& ctx,
                                     const std::string& name) {
    const GGUFTensor* t = gguf_find_tensor(ctx, name);
    if (!t) {
        std::cerr << "[ERRORE] Tensore non trovato: " << name << "\n";
        return {};
    }
    QuantTensor qt;
    qt.type   = t->info.type;
    qt.n_cols = t->info.shape[0];                                      // in_dim
    qt.n_rows = (t->info.n_dims >= 2) ? t->info.shape[1] : 1;         // out_dim
    qt.data   = t->data;
    return qt;
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
    // ── Rileva architettura ───────────────────
    // Il campo "general.architecture" dice tutto
    std::string arch_str;
    for (const auto& kv : ctx.metadata)
        if (kv.key == "general.architecture")
            if (auto* s = kv.value.get_if<std::string>())
                arch_str = *s;

    if (arch_str == "llama") {
        cfg.arch = ArchType::LLAMA;

        cfg.n_vocab  = get_u32(ctx, "tokenizer.ggml.vocab_size", 32000);
        cfg.n_ctx    = get_u32(ctx, "llama.context_length",      2048);
        cfg.n_embd   = get_u32(ctx, "llama.embedding_length",    2048);
        cfg.n_head   = get_u32(ctx, "llama.attention.head_count", 32);
        cfg.n_head_kv= get_u32(ctx, "llama.attention.head_count_kv", cfg.n_head);
        cfg.n_layer  = get_u32(ctx, "llama.block_count",         22);
        cfg.n_ff     = get_u32(ctx, "llama.feed_forward_length", 5632);
        cfg.rope_dim = get_u32(ctx, "llama.rope.dimension_count", 0);

        // RMSNorm epsilon
        for (const auto& kv : ctx.metadata)
            if (kv.key == "llama.attention.layer_norm_rms_epsilon")
                if (auto* v = kv.value.get_if<float>())
                    cfg.norm_eps = *v;

        // RoPE freq base
        for (const auto& kv : ctx.metadata)
            if (kv.key == "llama.rope.freq_base")
                if (auto* v = kv.value.get_if<float>())
                    cfg.rope_freq_base = *v;

    } else {
        // Default: GPT-2
        cfg.arch = ArchType::GPT2;

        cfg.n_vocab   = get_u32(ctx, "tokenizer.ggml.vocab_size", 50257);
        cfg.n_ctx     = get_u32(ctx, "gpt2.context_length",       1024);
        cfg.n_embd    = get_u32(ctx, "gpt2.embedding_length",     768);
        cfg.n_head    = get_u32(ctx, "gpt2.attention.head_count", 12);
        cfg.n_head_kv = cfg.n_head;  // GPT2 non ha GQA
        cfg.n_layer   = get_u32(ctx, "gpt2.block_count",          12);
        cfg.n_ff      = get_u32(ctx, "gpt2.feed_forward_length",  cfg.n_embd * 4);
        cfg.norm_eps  = 1e-5f;
    }

    cfg.d_head = cfg.n_embd / cfg.n_head;

    // rope_dim fallback: se non specificato nei metadata usa d_head
    if (cfg.arch == ArchType::LLAMA && cfg.rope_dim == 0)
        cfg.rope_dim = cfg.d_head;

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

    std::cout << "  Architettura : "
              << (cfg.arch == ArchType::LLAMA ? "LLaMA" : "GPT-2") << "\n";
    std::cout << "  Caricamento pesi globali...\n";

    // ── Pesi globali comuni ───────────────────
    w.token_embd = load_quant_tensor(ctx, "token_embd.weight");
    if (w.token_embd.empty()) return false;

    w.ln_f_w = load_tensor(ctx, "output_norm.weight");
    if (w.ln_f_w.empty()) return false;

    if (cfg.arch == ArchType::GPT2) {
        // GPT2: positional embedding + lm_head = token_embd (weight tying)
        w.pos_embd = load_tensor(ctx, "position_embd.weight");
        if (w.pos_embd.empty()) return false;

        w.ln_f_b = load_tensor(ctx, "output_norm.bias");
        if (w.ln_f_b.empty()) return false;

    } else {
        // LLaMA: lm_head separato, no positional embedding
        w.output_w = load_quant_tensor(ctx, "output.weight");
        if (w.output_w.empty()) return false;
        // ln_f_b non esiste in LLaMA — lascia vuoto
    }

    // ── Pesi per layer ────────────────────────
    std::cout << "  Caricamento " << cfg.n_layer << " layer...\n";
    w.layers.resize(cfg.n_layer);

    for (int i = 0; i < cfg.n_layer; i++) {
        std::string p = "blk." + std::to_string(i) + ".";
        LayerWeights& lw = w.layers[i];

        if (cfg.arch == ArchType::GPT2) {
            lw.ln1_w      = load_tensor      (ctx, p + "attn_norm.weight");
            lw.ln1_b      = load_tensor      (ctx, p + "attn_norm.bias");
            lw.attn_qkv_w = load_quant_tensor(ctx, p + "attn_qkv.weight");
            lw.attn_qkv_b = load_tensor      (ctx, p + "attn_qkv.bias");
            lw.attn_out_w = load_quant_tensor(ctx, p + "attn_output.weight");
            lw.attn_out_b = load_tensor      (ctx, p + "attn_output.bias");
            lw.ln2_w      = load_tensor      (ctx, p + "ffn_norm.weight");
            lw.ln2_b      = load_tensor      (ctx, p + "ffn_norm.bias");
            lw.ffn_fc1_w  = load_quant_tensor(ctx, p + "ffn_up.weight");
            lw.ffn_fc1_b  = load_tensor      (ctx, p + "ffn_up.bias");
            lw.ffn_fc2_w  = load_quant_tensor(ctx, p + "ffn_down.weight");
            lw.ffn_fc2_b  = load_tensor      (ctx, p + "ffn_down.bias");

            if (lw.attn_qkv_w.empty() || lw.ffn_fc1_w.empty()) {
                std::cerr << "[ERRORE] GPT2 layer " << i << " incompleto\n";
                return false;
            }

        } else {
            // LLaMA: no bias, Q/K/V separati
            lw.ln1_w      = load_tensor      (ctx, p + "attn_norm.weight");
            lw.attn_q_w   = load_quant_tensor(ctx, p + "attn_q.weight");
            lw.attn_k_w   = load_quant_tensor(ctx, p + "attn_k.weight");
            lw.attn_v_w   = load_quant_tensor(ctx, p + "attn_v.weight");
            lw.attn_out_w = load_quant_tensor(ctx, p + "attn_output.weight");
            lw.ln2_w      = load_tensor      (ctx, p + "ffn_norm.weight");
            lw.ffn_gate_w = load_quant_tensor(ctx, p + "ffn_gate.weight");
            lw.ffn_up_w   = load_quant_tensor(ctx, p + "ffn_up.weight");
            lw.ffn_down_w = load_quant_tensor(ctx, p + "ffn_down.weight");

            if (lw.attn_q_w.empty() || lw.ffn_gate_w.empty()) {
                std::cerr << "[ERRORE] LLaMA layer " << i << " incompleto\n";
                return false;
            }
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
// ─────────────────────────────────────────────
//  Inizializza KV cache e buffer di inferenza
//
//  La KV cache viene azzerata ad ogni reset
//  (nuova conversazione). I buffer di inferenza
//  vengono allocati una sola volta e riusati
//  ad ogni forward step — eliminano centinaia
//  di malloc/free per token generato.
//
//  Se chiamata più volte, la KV cache viene
//  resettata ma i buffer NON vengono riallocati
//  (hanno già la dimensione giusta).
// ─────────────────────────────────────────────
void model_init_kvcache(Model& model) {
    const ModelConfig& cfg = model.config;

    // ── KV Cache ──────────────────────────────
    // LLaMA GQA: la cache usa n_head_kv, non n_head
    int kv_dim     = cfg.n_head_kv * cfg.d_head;
    int cache_size = cfg.n_ctx * kv_dim;

    model.kv_cache.k.assign(cfg.n_layer, std::vector<float>(cache_size, 0.0f));
    model.kv_cache.v.assign(cfg.n_layer, std::vector<float>(cache_size, 0.0f));
    model.kv_cache.n_cached = 0;

    // ── Buffer di inferenza ───────────────────
    // Alloca (o ridimensiona) i buffer una sola volta.
    // resize() non riallocat se la size non cambia.
    InferBuffers& b = model.bufs;
    b.x       .resize(cfg.n_embd);
    b.residual.resize(cfg.n_embd);
    b.ln_out  .resize(cfg.n_embd);
    b.attn_out.resize(cfg.n_embd);  // output FINALE attention (dopo matvec proj)
    b.attn_acc.resize(cfg.n_embd);  // accumulatore INTERNO weighted sum V
    b.ffn_out .resize(cfg.n_embd);
    b.ln_final.resize(cfg.n_embd);
    b.Q       .resize(cfg.n_embd);  // n_head × d_head
    b.K       .resize(kv_dim);      // n_head_kv × d_head
    b.V       .resize(kv_dim);
    b.scores  .resize(cfg.n_ctx);   // worst case: contesto pieno
    b.gate    .resize(cfg.n_ff);    // FFN SwiGLU
    b.up      .resize(cfg.n_ff);
}
// ─────────────────────────────────────────────
//  Stampa configurazione
// ─────────────────────────────────────────────
void model_print_config(const ModelConfig& cfg) {
    std::cout << "\n═══════════════════════════════════════\n";
    std::cout << "  EIE-LLM — Model Config\n";
    std::cout << "═══════════════════════════════════════\n";
    std::cout << "  arch     : "
              << (cfg.arch == ArchType::LLAMA ? "LLaMA" : "GPT-2") << "\n";
    std::cout << "  n_vocab  : " << cfg.n_vocab       << "\n";
    std::cout << "  n_ctx    : " << cfg.n_ctx          << "\n";
    std::cout << "  n_embd   : " << cfg.n_embd         << "\n";
    std::cout << "  n_head   : " << cfg.n_head         << "\n";
    std::cout << "  n_head_kv: " << cfg.n_head_kv      << "\n";
    std::cout << "  n_layer  : " << cfg.n_layer        << "\n";
    std::cout << "  n_ff     : " << cfg.n_ff           << "\n";
    std::cout << "  d_head   : " << cfg.d_head         << "\n";
    if (cfg.arch == ArchType::LLAMA) {
        std::cout << "  rope_dim : " << cfg.rope_dim       << "\n";
        std::cout << "  rope_base: " << cfg.rope_freq_base << "\n";
        std::cout << "  norm_eps : " << cfg.norm_eps       << "\n";
    }
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
void layer_norm(const float* x, const float* w, const float* b, float* out, int n) {
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
void embedding_lookup(const float* token_embd, const float* pos_embd, int token_id, int pos, float* out, int n_embd) {
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
void self_attention(const float* x, float* out, const LayerWeights& lw, KVCache& cache, const ModelConfig& cfg, int layer, int pos, InferBuffers& bufs) {

    const int n_embd  = cfg.n_embd;
    const int n_head  = cfg.n_head;
    const int d_head  = cfg.d_head;

    // ── Step 1: calcola Q, K, V ───────────────
    //
    // La matrice attn_qkv_w ha shape [3*n_embd × n_embd]
    // e produce Q, K, V concatenati in un unico vettore.
    // Usiamo bufs.Q come buffer temporaneo per [Q|K|V].
    // bufs.Q ha dimensione n_embd — troppo piccolo per
    // Q+K+V in GPT-2 (3*n_embd), quindi allochiamo qkv
    // come vettore locale. Solo per GPT-2, che ha n_embd
    // piccolo (768), l'overhead è trascurabile.
    std::vector<float> qkv(3 * n_embd);
    matvec_quant(lw.attn_qkv_w, x, qkv.data());
    vec_add(qkv.data(), lw.attn_qkv_b.data(), qkv.data(), 3 * n_embd);

    const float* Q = qkv.data();
    const float* K = qkv.data() + n_embd;
    const float* V = qkv.data() + 2 * n_embd;

    // ── Step 2: salva K e V nella cache ───────
    float* k_pos = cache.k[layer].data() + pos * n_embd;
    float* v_pos = cache.v[layer].data() + pos * n_embd;
    vec_copy(K, k_pos, n_embd);
    vec_copy(V, v_pos, n_embd);

    // ── Step 3: attention per ogni head ───────
    // bufs.attn_acc = accumulatore interno della weighted sum di V.
    // DEVE essere distinto da `out` (= bufs.attn_out nel chiamante)
    // perché il matvec finale legge da attn_acc e scrive in out —
    // se coincidessero ci sarebbe aliasing e il risultato sarebbe corrotto.
    std::fill(bufs.attn_acc.begin(), bufs.attn_acc.end(), 0.0f);

    float scale = 1.0f / sqrtf((float)d_head);

    for (int h = 0; h < n_head; h++) {
        int h_off = h * d_head;
        const float* Qh = Q + h_off;

        for (int t = 0; t <= pos; t++) {
            const float* Kh_t = cache.k[layer].data() + t * n_embd + h_off;
            float dot = 0.0f;
            for (int d = 0; d < d_head; d++)
                dot += Qh[d] * Kh_t[d];
            bufs.scores[t] = dot * scale;
        }

        softmax(bufs.scores.data(), pos + 1);

        float* acc_h = bufs.attn_acc.data() + h_off;
        for (int t = 0; t <= pos; t++) {
            const float* Vh_t = cache.v[layer].data() + t * n_embd + h_off;
            for (int d = 0; d < d_head; d++)
                acc_h[d] += bufs.scores[t] * Vh_t[d];
        }
    }

    // ── Step 4: proiezione output ──────────────
    // Legge da attn_acc, scrive in out (bufs.attn_out) — no aliasing.
    matvec_quant(lw.attn_out_w, bufs.attn_acc.data(), out);
    vec_add(out, lw.attn_out_b.data(), out, n_embd);
}

// ─────────────────────────────────────────────
//  Feed-Forward Network (FFN)
//
//  GPT-2 usa un FFN a 2 strati con GELU:
//
//  1) Proiezione up: n_embd → n_ff (=3072)
//     h = x · fc1_w^T + fc1_b
//
//  2) Attivazione non lineare
//     h = GELU(h)
//
//  3) Proiezione down: n_ff → n_embd
//     out = h · fc2_w^T + fc2_b
//
//  La dimensione intermedia n_ff = 4 × n_embd
//  è una scelta architetturale di GPT-2.
//  Espandere e poi ricomprimere permette alla
//  rete di fare computazioni più complesse.
// ─────────────────────────────────────────────
void feed_forward(const float* x, float* out,
                  const LayerWeights& lw,
                  const ModelConfig& cfg,
                  InferBuffers& bufs) {
    const int n_embd = cfg.n_embd;
    const int n_ff   = cfg.n_ff;

    // Riusa bufs.gate come buffer intermedio per la proiezione up.
    // Step 1: proiezione up x → gate [n_embd → n_ff]
    matvec_quant(lw.ffn_fc1_w, x, bufs.gate.data());
    vec_add(bufs.gate.data(), lw.ffn_fc1_b.data(), bufs.gate.data(), n_ff);

    // Step 2: attivazione GELU in-place
    gelu(bufs.gate.data(), n_ff);

    // Step 3: proiezione down → out [n_ff → n_embd]
    matvec_quant(lw.ffn_fc2_w, bufs.gate.data(), out);
    vec_add(out, lw.ffn_fc2_b.data(), out, n_embd);
}

// ─────────────────────────────────────────────
//  Self-Attention LLaMA con GQA e RoPE
//
//  Differenze rispetto a GPT-2:
//
//  1) Q, K, V sono proiettati separatamente
//     Q: [n_embd × n_embd]      → n_head    vettori d_head
//     K: [n_embd × n_head_kv*d] → n_head_kv vettori d_head
//     V: [n_embd × n_head_kv*d] → n_head_kv vettori d_head
//
//  2) RoPE applicato a Q e K prima della cache
//     Ruota le coppie di dimensioni in base alla posizione
//
//  3) GQA: ogni gruppo di (n_head/n_head_kv) head Q
//     condivide lo stesso head K e V
//     kv_head = q_head / (n_head / n_head_kv)
// ─────────────────────────────────────────────
void self_attention_llama(const float* x, float* out, const LayerWeights& lw, KVCache& cache, const ModelConfig& cfg, int layer, int pos, InferBuffers& bufs) {

    const int n_head    = cfg.n_head;
    const int n_head_kv = cfg.n_head_kv;
    const int d_head    = cfg.d_head;
    const int rope_dim  = cfg.rope_dim;
    const int kv_dim    = n_head_kv * d_head;

    // ── Step 1: proiezione Q, K, V separata ──
    // Riusa i buffer pre-allocati invece di allocare vettori locali.
    matvec_quant(lw.attn_q_w, x, bufs.Q.data());
    matvec_quant(lw.attn_k_w, x, bufs.K.data());
    matvec_quant(lw.attn_v_w, x, bufs.V.data());

    // ── Step 2: applica RoPE a Q e K ─────────
    //
    // RoPE ruota ogni coppia di dimensioni in base
    // alla posizione. Applicato prima di salvare
    // in cache in modo che la cache contenga già
    // i vettori ruotati — non bisogna ricalcolare
    // la rotazione per i token già in cache.
    rope(bufs.Q.data(), pos, n_head,    d_head, rope_dim, cfg.rope_freq_base);
    rope(bufs.K.data(), pos, n_head_kv, d_head, rope_dim, cfg.rope_freq_base);

    // ── Step 3: salva K e V nella cache ──────
    float* k_pos = cache.k[layer].data() + pos * kv_dim;
    float* v_pos = cache.v[layer].data() + pos * kv_dim;
    vec_copy(bufs.K.data(), k_pos, kv_dim);
    vec_copy(bufs.V.data(), v_pos, kv_dim);

    // ── Step 4: attention con GQA ─────────────
    //
    // GQA: n_head Q head sono divisi in n_head_kv gruppi.
    // Ogni gruppo condivide un head K e V.
    // Head Q h → kv_head = h / (n_head / n_head_kv)
    int gqa_ratio = n_head / n_head_kv;

    // Usa attn_acc come accumulatore (separato da out = bufs.attn_out).
    std::fill(bufs.attn_acc.begin(), bufs.attn_acc.end(), 0.0f);
    float scale = 1.0f / sqrtf((float)d_head);

    for (int h = 0; h < n_head; h++) {
        int kv_h   = h / gqa_ratio;
        int h_off  = h    * d_head;
        int kv_off = kv_h * d_head;

        const float* Qh = bufs.Q.data() + h_off;

        for (int t = 0; t <= pos; t++) {
            const float* Kh_t = cache.k[layer].data() + t * kv_dim + kv_off;
            float dot = 0.0f;
            for (int d = 0; d < d_head; d++)
                dot += Qh[d] * Kh_t[d];
            bufs.scores[t] = dot * scale;
        }

        softmax(bufs.scores.data(), pos + 1);

        float* acc_h = bufs.attn_acc.data() + h_off;
        for (int t = 0; t <= pos; t++) {
            const float* Vh_t = cache.v[layer].data() + t * kv_dim + kv_off;
            for (int d = 0; d < d_head; d++)
                acc_h[d] += bufs.scores[t] * Vh_t[d];
        }
    }

    // ── Step 5: proiezione output ─────────────
    // Legge da attn_acc, scrive in out (bufs.attn_out) — no aliasing.
    matvec_quant(lw.attn_out_w, bufs.attn_acc.data(), out);
}

// ─────────────────────────────────────────────
//  Feed-Forward LLaMA con SwiGLU
//
//  SwiGLU usa 3 matrici invece di 2:
//
//    gate = SiLU( x · ffn_gate_w^T )   [n_embd → n_ff]
//    up   =       x · ffn_up_w^T        [n_embd → n_ff]
//    h    = gate ⊙ up                   [n_ff] (Hadamard)
//    out  = h · ffn_down_w^T            [n_ff → n_embd]
//
//  Il gate moltiplicativo rende la rete più
//  espressiva — ogni dimensione di up viene
//  scalata da un gate che dipende dall'input.
//  SiLU è più smooth di ReLU e migliora il
//  flusso del gradiente durante il training.
// ─────────────────────────────────────────────
void feed_forward_llama(const float* x, float* out, const LayerWeights& lw, const ModelConfig& cfg, InferBuffers& bufs) {
    const int n_ff = cfg.n_ff;

    // Riusa bufs.gate e bufs.up — nessuna allocazione heap.
    // Proiezione gate e up parallele
    matvec_quant(lw.ffn_gate_w, x, bufs.gate.data());
    matvec_quant(lw.ffn_up_w,   x, bufs.up.data());

    // SiLU sul gate, poi Hadamard: gate ⊙ up
    silu(bufs.gate.data(), n_ff);
    for (int i = 0; i < n_ff; i++)
        bufs.gate[i] *= bufs.up[i];

    // Proiezione down: n_ff → n_embd
    matvec_quant(lw.ffn_down_w, bufs.gate.data(), out);
}

// ─────────────────────────────────────────────
//  Forward pass completo
//
//  Per ogni layer il flusso è:
//
//  x_in
//   │
//   ├─ residual = x_in                  (salva per dopo)
//   │
//   ├─ x = LayerNorm1(x_in)
//   ├─ x = SelfAttention(x)
//   ├─ x = x + residual                 (residual connection 1)
//   │
//   ├─ residual = x                     (salva per dopo)
//   ├─ x = LayerNorm2(x)
//   ├─ x = FFN(x)
//   └─ x = x + residual                 (residual connection 2)
//
//  Le residual connections sono fondamentali:
//  permettono al gradiente di fluire direttamente
//  attraverso i layer durante il training,
//  rendendo possibile addestrare reti molto profonde.
//  Durante l'inferenza stabilizzano il segnale.
//
//  Dopo tutti i layer:
//    x = LayerNorm_finale(x)
//    logits = x · token_embd^T   (lm_head)
//
//  Nota: GPT-2 usa weight tying — la matrice
//  lm_head è la STESSA di token_embd trasposta.
//  Questo riduce i parametri e migliora la qualità.
// ─────────────────────────────────────────────
void forward(Model& model, int token_id, int pos, std::vector<float>& logits, bool bench_mode) {

    const ModelConfig& cfg = model.config;
    const ModelWeights& w  = model.weights;
    const int n_embd        = cfg.n_embd;
    const bool is_llama     = (cfg.arch == ArchType::LLAMA);

    // Tutti i buffer temporanei sono pre-allocati in model.bufs —
    // nessuna allocazione heap durante il forward pass.
    InferBuffers& b = model.bufs;

    // ── Step 1: embedding lookup ──────────────
    auto t0 = now();

    // Dequantizza solo la riga corrispondente al token corrente
    dequant_row(w.token_embd, token_id, b.x.data());
    if (!is_llama) {
        // GPT2: somma il positional embedding (già float32)
        const float* pe = w.pos_embd.data() + pos * n_embd;
        vec_add(b.x.data(), pe, b.x.data(), n_embd);
    }

    auto t1 = now();

    // ── Step 2: loop sui layer ─────────────────
    double attn_ms_step = 0.0;
    double ffn_ms_step  = 0.0;

    for (int l = 0; l < cfg.n_layer; l++) {
        const LayerWeights& lw = w.layers[l];

        // ── Salva residual ────────────────────
        vec_copy(b.x.data(), b.residual.data(), n_embd);

        // ── Norm 1 ────────────────────────────
        auto ta0 = now();
        if (is_llama)
            rms_norm(b.x.data(), lw.ln1_w.data(),
                     b.ln_out.data(), n_embd, cfg.norm_eps);
        else
            layer_norm(b.x.data(), lw.ln1_w.data(),
                       lw.ln1_b.data(), b.ln_out.data(), n_embd);

        // ── Self-Attention ────────────────────
        if (is_llama)
            self_attention_llama(b.ln_out.data(), b.attn_out.data(),
                                 lw, model.kv_cache, cfg, l, pos, b);
        else
            self_attention(b.ln_out.data(), b.attn_out.data(),
                           lw, model.kv_cache, cfg, l, pos, b);

        auto ta1 = now();

        // ── Residual connection 1 ─────────────
        vec_add(b.attn_out.data(), b.residual.data(), b.x.data(), n_embd);

        // ── Salva residual ────────────────────
        vec_copy(b.x.data(), b.residual.data(), n_embd);

        // ── Norm 2 ────────────────────────────
        auto tf0 = now();
        if (is_llama)
            rms_norm(b.x.data(), lw.ln2_w.data(),
                     b.ln_out.data(), n_embd, cfg.norm_eps);
        else
            layer_norm(b.x.data(), lw.ln2_w.data(),
                       lw.ln2_b.data(), b.ln_out.data(), n_embd);

        // ── FFN ───────────────────────────────
        if (is_llama)
            feed_forward_llama(b.ln_out.data(), b.ffn_out.data(), lw, cfg, b);
        else
            feed_forward(b.ln_out.data(), b.ffn_out.data(), lw, cfg, b);

        auto tf1 = now();

        // ── Residual connection 2 ─────────────
        vec_add(b.ffn_out.data(), b.residual.data(), b.x.data(), n_embd);

        attn_ms_step += ms_between(ta0, ta1);
        ffn_ms_step  += ms_between(tf0, tf1);
    }

    // ── Step 3: norm finale ───────────────────
    if (is_llama)
        rms_norm(b.x.data(), w.ln_f_w.data(),
                 b.ln_final.data(), n_embd, cfg.norm_eps);
    else
        layer_norm(b.x.data(), w.ln_f_w.data(),
                   w.ln_f_b.data(), b.ln_final.data(), n_embd);

    // ── Step 4: lm_head → logits ──────────────
    logits.resize(cfg.n_vocab);
    // LLaMA: lm_head separato | GPT2: weight tying con token_embd
    const QuantTensor& lm_head = is_llama ? w.output_w : w.token_embd;
    matvec_quant(lm_head, b.ln_final.data(), logits.data());

    // ── Accumula tempi benchmark ───────────────
    if (bench_mode) {
        model.bench.embed_ms  += ms_between(t0, t1);
        model.bench.attn_ms   += attn_ms_step;
        model.bench.ffn_ms    += ffn_ms_step;
        model.bench.n_steps++;
    }
}

// ─────────────────────────────────────────────
//  Sampling greedy (argmax)
//
//  Sceglie sempre il token con logit massimo.
//  Deterministico ma tende a produrre testo
//  ripetitivo e poco creativo.
// ─────────────────────────────────────────────
int sample_argmax(const std::vector<float>& logits) {
    return static_cast<int>(
        std::max_element(logits.begin(), logits.end())
        - logits.begin()
    );
}

// ─────────────────────────────────────────────
//  Sampling con temperatura
//
//  1) Dividi i logits per la temperatura
//     temperature > 1 → appiattisce la distribuzione
//     temperature < 1 → la concentra sul massimo
//  2) Applica softmax per ottenere probabilità
//  3) Campiona dalla distribuzione risultante
//
//  Usiamo il metodo della CDF inversa:
//  generiamo un numero random r in [0,1]
//  e troviamo il primo token t tale che
//  la somma cumulativa delle probabilità >= r
// ─────────────────────────────────────────────
int sample_temperature(std::vector<float> logits, float temperature) {
    // Dividi per la temperatura
    for (auto& l : logits)
        l /= temperature;

    // Softmax → probabilità
    softmax(logits.data(), static_cast<int>(logits.size()));

    // Campiona dalla distribuzione
    float r = static_cast<float>(rand()) / RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < (int)logits.size(); i++) {
        cumsum += logits[i];
        if (cumsum >= r) return i;
    }
    // Fallback — non dovrebbe mai succedere
    return static_cast<int>(logits.size()) - 1;
}

// ─────────────────────────────────────────────
//  Repetition penalty
//
//  Scorriamo tutti i token già visti nel contesto
//  e modifichiamo il loro logit.
//  Usiamo un set per evitare di penalizzare
//  lo stesso token più volte anche se appare
//  più volte nel contesto.
// ─────────────────────────────────────────────
void apply_repetition_penalty(std::vector<float>& logits,
                               const std::vector<int>& context_ids,
                               float penalty) {
    if (penalty == 1.0f) return;  // nessun effetto

    // Insieme dei token unici già visti
    // (penalizziamo ogni token al massimo una volta)
    std::unordered_set<int> seen;

    for (int id : context_ids) {
        // Salta token fuori range o già penalizzati
        if (id < 0 || id >= (int)logits.size()) continue;
        if (seen.count(id)) continue;
        seen.insert(id);

        // Applica la penalità
        if (logits[id] > 0.0f)
            logits[id] /= penalty;
        else
            logits[id] *= penalty;
    }
}

// ─────────────────────────────────────────────
//  Top-p (nucleus) sampling
//
//  Passi dettagliati:
//  1) Dividi per temperatura
//  2) Softmax → probabilità
//  3) Ordina per probabilità decrescente
//     mantenendo gli indici originali
//  4) Trova il nucleus: accumula probabilità
//     finché non raggiungiamo p
//  5) Campiona dal nucleus con CDF inversa
// ─────────────────────────────────────────────
int sample_topp(std::vector<float> logits, float p, float temperature) {
    const int n = static_cast<int>(logits.size());

    // Step 1: applica temperatura
    if (temperature > 0.0f)
        for (auto& l : logits) l /= temperature;

    // Step 2: softmax → probabilità
    softmax(logits.data(), n);

    // Step 3: crea lista (prob, indice) e ordina
    // per probabilità decrescente
    std::vector<std::pair<float, int>> sorted(n);
    for (int i = 0; i < n; i++)
        sorted[i] = {logits[i], i};

    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                  return a.first > b.first;  // decrescente
              });

    // Step 4: trova il nucleus
    // Accumula token finché la somma cumulativa >= p
    float cumsum = 0.0f;
    int   nucleus_size = 0;
    for (int i = 0; i < n; i++) {
        cumsum += sorted[i].first;
        nucleus_size++;
        if (cumsum >= p) break;
    }

    // Step 5: rinormalizza le probabilità nel nucleus
    // e campiona con CDF inversa
    float nucleus_sum = 0.0f;
    for (int i = 0; i < nucleus_size; i++)
        nucleus_sum += sorted[i].first;

    float r = static_cast<float>(rand()) / RAND_MAX * nucleus_sum;
    float cumulative = 0.0f;

    for (int i = 0; i < nucleus_size; i++) {
        cumulative += sorted[i].first;
        if (cumulative >= r)
            return sorted[i].second;
    }

    // Fallback — non dovrebbe mai succedere
    return sorted[0].second;
}

// ─────────────────────────────────────────────
//  Top-k + Top-p sampling combinati
//
//  Algoritmo:
//  1) Applica temperatura
//  2) Ordina per logit decrescente
//  3) Top-k: tieni solo i primi k token
//  4) Softmax sul sottoinsieme rimasto
//  5) Top-p: taglia ulteriormente al nucleus p
//  6) Rinormalizza e campiona
//
//  Applicarli in sequenza (prima k poi p)
//  è la pratica standard — top-k dà un tetto
//  duro al numero di candidati, top-p
//  affina ulteriormente la distribuzione.
// ─────────────────────────────────────────────
int sample_topk_topp(std::vector<float> logits, int   top_k, float top_p, float temperature) {
    const int n = static_cast<int>(logits.size());

    // Step 1: applica temperatura
    if (temperature > 0.0f)
        for (auto& l : logits) l /= temperature;

    // Step 2: crea lista (logit, indice) e ordina
    std::vector<std::pair<float, int>> sorted(n);
    for (int i = 0; i < n; i++)
        sorted[i] = {logits[i], i};
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) {
                  return a.first > b.first;
              });

    // Step 3: top-k — tronca la lista a k elementi
    // top_k = 0 significa "disabilitato"
    int k = (top_k > 0 && top_k < n) ? top_k : n;
    sorted.resize(k);

    // Step 4: softmax sul sottoinsieme
    // Lavoriamo su un vettore temporaneo
    std::vector<float> probs(k);
    for (int i = 0; i < k; i++)
        probs[i] = sorted[i].first;
    softmax(probs.data(), k);

    // Step 5: top-p — trova il nucleus
    float cumsum = 0.0f;
    int nucleus = k;
    for (int i = 0; i < k; i++) {
        cumsum += probs[i];
        if (cumsum >= top_p) {
            nucleus = i + 1;
            break;
        }
    }

    // Step 6: rinormalizza il nucleus e campiona
    float nucleus_sum = 0.0f;
    for (int i = 0; i < nucleus; i++)
        nucleus_sum += probs[i];

    float r = static_cast<float>(rand()) / RAND_MAX * nucleus_sum;
    float cumulative = 0.0f;
    for (int i = 0; i < nucleus; i++) {
        cumulative += probs[i];
        if (cumulative >= r)
            return sorted[i].second;
    }

    return sorted[0].second;
}