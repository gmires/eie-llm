#include <iostream>
#include <fstream>
#include "gguf.hpp"

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

    // ── Fase 1: leggi e valida l'header ──
    GGUFContext ctx;
    if (!gguf_read_header(f, ctx.header)) return 1;
    std::cout << "✓ Header letto correttamente\n";

    // ── Fase 2: leggi tutti i metadata KV ──
    if (!gguf_read_metadata(f, ctx)) return 1;
    std::cout << "✓ Metadata letti correttamente\n";

    // ── Fase 3: leggi le info dei tensori ──
    if (!gguf_read_tensor_info(f, ctx)) return 1;
    std::cout << "✓ Info tensori lette correttamente\n";
    
    // ── Stampa tutto per verifica ──
    gguf_print_context(ctx);

    return 0;
}