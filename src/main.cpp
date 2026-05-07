#include <iostream>
#include <fstream>
#include <iomanip>
#include "gguf.hpp"
#include "tokenizer.hpp"
#include "ops.hpp"

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
    gguf_print_memory_usage(ctx);

    Tokenizer tok;
    if (!tokenizer_init(tok, ctx))              return 1;
    std::cout << "✓ Tokenizer inizializzato\n";

    // ── Test ops ──────────────────────────────

    // Test 1: dequantizzazione del primo tensore Q8_0
    std::cout << "\n── Test operazioni ──\n";
    const GGUFTensor* embd = gguf_find_tensor(ctx, "token_embd.weight");
    if (embd) {
        auto floats = tensor_to_float(*embd);
        std::cout << "✓ Dequantizzazione token_embd.weight\n";
        std::cout << "  Elementi : " << floats.size() << "\n";
        std::cout << "  Primi 4 valori: ";
        for (int i = 0; i < 4; i++)
            std::cout << std::fixed << std::setprecision(6)
                      << floats[i] << " ";
        std::cout << "\n";
    }

    // Test 2: softmax su un vettore semplice
    float logits[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    softmax(logits, 4);
    std::cout << "✓ Softmax [1,2,3,4] → [";
    for (int i = 0; i < 4; i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(4) << logits[i];
    }
    std::cout << "] (somma deve essere 1.0)\n";

    // Verifica che la somma sia 1
    float sum = logits[0]+logits[1]+logits[2]+logits[3];
    std::cout << "  Somma: " << std::fixed << std::setprecision(6)
              << sum << (fabsf(sum - 1.0f) < 1e-6f ? " ✓" : " ✗") << "\n";

    // Test 3: GELU su valori noti
    float gelu_test[3] = {-1.0f, 0.0f, 1.0f};
    gelu(gelu_test, 3);
    std::cout << "✓ GELU [-1, 0, 1] → [";
    for (int i = 0; i < 3; i++) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(6) << gelu_test[i];
    }
    std::cout << "]\n";
    std::cout << "  (atteso: ~[-0.158655, 0.0, 0.841345])\n";

    return 0;
}