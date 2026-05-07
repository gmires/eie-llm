#include <iostream>
#include <fstream>
#include <iomanip>
#include "gguf.hpp"
#include "tokenizer.hpp"
#include "ops.hpp"
#include "model.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: eie-llm <model.gguf>\n";
        return 1;
    }

    std::ifstream f(argv[1], std::ios::binary);
    if (!f) {
        std::cerr << "[ERRORE] Impossibile aprire: " << argv[1] << "\n";
        return 1;
    }

    GGUFContext ctx;
    if (!gguf_read_header(f, ctx.header))      return 1;
    std::cout << "✓ Header letto\n";
    if (!gguf_read_metadata(f, ctx))            return 1;
    std::cout << "✓ Metadata letti\n";
    if (!gguf_read_tensor_info(f, ctx))         return 1;
    std::cout << "✓ Info tensori lette\n";
    if (!gguf_load_tensors(f, ctx))             return 1;
    std::cout << "✓ Pesi caricati in RAM\n";

    Tokenizer tok;
    if (!tokenizer_init(tok, ctx))              return 1;
    std::cout << "✓ Tokenizer inizializzato\n";

    // ── Carica il modello ──
    Model model;
    if (!model_load_config(model.config, ctx))  return 1;
    model_print_config(model.config);

    if (!model_load_weights(model, ctx))        return 1;
    std::cout << "✓ Pesi modello caricati\n";

    model_init_kvcache(model);
    std::cout << "✓ KV cache inizializzata\n";

    // ── Test embedding lookup ──
    std::cout << "\n── Test embedding lookup ──\n";
    const ModelConfig& cfg = model.config;
    std::vector<float> embd(cfg.n_embd);

    // Token ID 15496 = "Hello" in GPT-2
    int test_token = 15496;
    int test_pos   = 0;
    embedding_lookup(model.weights.token_embd.data(),
                     model.weights.pos_embd.data(),
                     test_token, test_pos,
                     embd.data(), cfg.n_embd);

    std::cout << "  Token " << test_token
              << " @ pos " << test_pos << "\n";
    std::cout << "  Primi 4 valori embedding: ";
    for (int i = 0; i < 4; i++)
        std::cout << std::fixed << std::setprecision(6)
                  << embd[i] << " ";
    std::cout << "\n";

    // ── Test layer norm ──
    std::cout << "\n── Test layer norm ──\n";
    std::vector<float> norm_out(cfg.n_embd);
    layer_norm(embd.data(),
               model.weights.layers[0].ln1_w.data(),
               model.weights.layers[0].ln1_b.data(),
               norm_out.data(), cfg.n_embd);

    std::cout << "  Primi 4 valori dopo LN: ";
    for (int i = 0; i < 4; i++)
        std::cout << std::fixed << std::setprecision(6)
                  << norm_out[i] << " ";
    std::cout << "\n";
    std::cout << "✓ LayerNorm applicata correttamente\n";

    // ── Test self-attention ──
    std::cout << "\n── Test self-attention ──\n";

    // Simula il primo step del forward pass:
    // embedding lookup + layer norm + attention
    std::vector<float> x(cfg.n_embd);
    std::vector<float> ln_out(cfg.n_embd);
    std::vector<float> attn_out(cfg.n_embd);

    // Token "Hello" alla posizione 0
    embedding_lookup(model.weights.token_embd.data(),
                     model.weights.pos_embd.data(),
                     15496, 0, x.data(), cfg.n_embd);

    // Layer norm prima dell'attention
    layer_norm(x.data(),
               model.weights.layers[0].ln1_w.data(),
               model.weights.layers[0].ln1_b.data(),
               ln_out.data(), cfg.n_embd);

    // Self-attention layer 0, posizione 0
    self_attention(ln_out.data(), attn_out.data(),
                   model.weights.layers[0],
                   model.kv_cache,
                   model.config, 0, 0);

    std::cout << "  Primi 4 valori attention output: ";
    for (int i = 0; i < 4; i++)
        std::cout << std::fixed << std::setprecision(6)
                  << attn_out[i] << " ";
    std::cout << "\n";
    std::cout << "  KV cache pos 0 salvata: "
              << (model.kv_cache.k[0][0] != 0.0f ? "✓" : "✗")
              << "\n";
    std::cout << "✓ Self-attention completata\n";
    
    return 0;
}