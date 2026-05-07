#include <iostream>
#include <fstream>
#include "gguf.hpp"
#include "tokenizer.hpp"
#include "model.hpp"
#include "shell.hpp"

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

    // ── Caricamento ───────────────────────────
    std::cerr << "Caricamento modello...\n";

    GGUFContext ctx;
    if (!gguf_read_header(f, ctx.header))      return 1;
    if (!gguf_read_metadata(f, ctx))            return 1;
    if (!gguf_read_tensor_info(f, ctx))         return 1;
    if (!gguf_load_tensors(f, ctx))             return 1;

    Tokenizer tok;
    if (!tokenizer_init(tok, ctx))              return 1;

    Model model;
    if (!model_load_config(model.config, ctx))  return 1;
    if (!model_load_weights(model, ctx))        return 1;
    model_init_kvcache(model);

    std::cerr << "✓ Pronto!\n";

    // ── Avvia la shell ────────────────────────
    shell_run(model, tok);

    return 0;
}