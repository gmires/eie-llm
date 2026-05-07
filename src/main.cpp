#include <iostream>
#include <fstream>
#include <iomanip>
#include "gguf.hpp"
#include "tokenizer.hpp"

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
    std::cout << "✓ Metadata letti (" << ctx.metadata.size() << " KV)\n";

    if (!gguf_read_tensor_info(f, ctx))         return 1;
    std::cout << "✓ Info tensori lette (" << ctx.tensors.size() << ")\n";

    if (!gguf_load_tensors(f, ctx))             return 1;
    std::cout << "✓ Pesi caricati in RAM\n";
    gguf_print_memory_usage(ctx);

    // ── Fase 5: tokenizer ──
    Tokenizer tok;
    if (!tokenizer_init(tok, ctx))              return 1;
    std::cout << "✓ Tokenizer inizializzato\n";
    tokenizer_print_info(tok);

    // ── Test encode/decode ──
    std::string test = "Hello, world!";
    std::cout << "Test: \"" << test << "\"\n";

    auto ids = tokenizer_encode(tok, test);
    std::cout << "  ID : [";
    for (size_t i = 0; i < ids.size(); i++) {
        if (i > 0) std::cout << ", ";
        std::cout << ids[i];
    }
    std::cout << "]\n";

    std::string decoded = tokenizer_decode(tok, ids);
    std::cout << "  Out: \"" << decoded << "\"\n";

    if (decoded == test)
        std::cout << "✓ Round-trip corretto!\n";
    else
        std::cout << "⚠ Round-trip differisce\n";

    return 0;
}