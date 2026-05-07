#include <iostream>
#include <fstream>
#include <iomanip>
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

    GGUFContext ctx;

    // ── Fase 1: header ──
    if (!gguf_read_header(f, ctx.header)) return 1;
    std::cout << "✓ Header letto\n";

    // ── Fase 2: metadata KV ──
    if (!gguf_read_metadata(f, ctx)) return 1;
    std::cout << "✓ Metadata letti (" << ctx.metadata.size() << " KV)\n";

    // ── Fase 3: info tensori ──
    if (!gguf_read_tensor_info(f, ctx)) return 1;
    std::cout << "✓ Info tensori lette (" << ctx.tensors.size() << " tensori)\n";

    // ── Fase 4: carica i pesi in RAM ──
    if (!gguf_load_tensors(f, ctx)) return 1;
    std::cout << "✓ Pesi caricati in RAM\n";

    // ── Stampa riepilogo memoria ──
    gguf_print_memory_usage(ctx);

    // ── Test: cerca e verifica un tensore specifico ──
    const char* test_name = "token_embd.weight";
    const GGUFTensor* embd = gguf_find_tensor(ctx, test_name);
    if (embd) {
        std::cout << "✓ Tensore trovato: " << embd->name() << "\n";
        std::cout << "  Tipo   : " << ggml_type_name(embd->type()) << "\n";
        std::cout << "  Bytes  : " << embd->data.size() << "\n";
        std::cout << "  Primi 8 byte (hex): ";
        for (int i = 0; i < 8; i++)
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                      << (int)embd->data[i] << " ";
        std::cout << std::dec << "\n";
    } else {
        std::cerr << "[ERRORE] Tensore non trovato: " << test_name << "\n";
    }

    return 0;
}