#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include "gguf.hpp"
#include "tokenizer.hpp"
#include "ops.hpp"
#include "model.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: eie-llm <model.gguf> [prompt] [n_tokens]\n";
        return 1;
    }

    // Argomenti opzionali
    std::string prompt   = argc >= 3 ? argv[2] : "Hello, I am a language model";
    int         n_tokens = argc >= 4 ? std::atoi(argv[3]) : 20;

    std::ifstream f(argv[1], std::ios::binary);
    if (!f) {
        std::cerr << "[ERRORE] Impossibile aprire: " << argv[1] << "\n";
        return 1;
    }

    // ── Caricamento ───────────────────────────
    GGUFContext ctx;
    if (!gguf_read_header(f, ctx.header))      return 1;
    if (!gguf_read_metadata(f, ctx))            return 1;
    if (!gguf_read_tensor_info(f, ctx))         return 1;
    if (!gguf_load_tensors(f, ctx))             return 1;
    std::cout << "✓ Modello caricato\n";

    Tokenizer tok;
    if (!tokenizer_init(tok, ctx))              return 1;
    std::cout << "✓ Tokenizer pronto\n";

    Model model;
    if (!model_load_config(model.config, ctx))  return 1;
    if (!model_load_weights(model, ctx))        return 1;
    model_init_kvcache(model);
    std::cout << "✓ Modello pronto\n\n";

    // ── Tokenizza il prompt ───────────────────
    auto input_ids = tokenizer_encode(tok, prompt);
    std::cout << "Prompt : \"" << prompt << "\"\n";
    std::cout << "Tokens : " << input_ids.size() << "\n\n";

    // ── Generazione ───────────────────────────
    //
    // Il loop di generazione funziona in 2 fasi:
    //
    // FASE 1 — Prefill (processa il prompt)
    //   Passiamo tutti i token del prompt al modello
    //   uno per uno per riempire la KV cache.
    //   Solo i logits dell'ultimo token ci interessano.
    //
    // FASE 2 — Generazione (token per token)
    //   Ad ogni step:
    //   1) Forward pass con il token corrente
    //   2) Campiona il token successivo dai logits
    //   3) Decodifica e stampa il token
    //   4) Usa il token campionato come input
    //      del prossimo step

    srand(static_cast<unsigned>(time(nullptr)));

    std::vector<float> logits;
    std::cout << "Output : \"" << prompt;
    std::cout.flush();

    // Fase 1: prefill del prompt
    int pos = 0;
    for (int id : input_ids) {
        forward(model, id, pos, logits);
        pos++;
    }
    // Aggiorna la cache
    model.kv_cache.n_cached = pos;

    // Fase 2: generazione
    int next_token = sample_argmax(logits);

    for (int i = 0; i < n_tokens; i++) {
        // Decodifica e stampa il token appena generato
        std::string token_str = tokenizer_decode(tok, {next_token});
        std::cout << token_str;
        std::cout.flush();

        // Stop se generiamo il token di fine testo
        // In GPT-2 il token EOS ha ID 50256
        if (next_token == 50256) break;

        // Forward pass con il token appena generato
        forward(model, next_token, pos, logits);
        pos++;

        // Campiona il prossimo token
        // Puoi cambiare con sample_temperature(logits, 0.8f)
        // per output più vari
        next_token = sample_argmax(logits);
    }

    std::cout << "\"\n\n";
    std::cout << "✓ Generazione completata (" << n_tokens << " token)\n";

    return 0;
}