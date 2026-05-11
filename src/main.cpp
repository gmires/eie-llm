#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include "gguf.hpp"
#include "tokenizer.hpp"
#include "model.hpp"
#include "shell.hpp"
#include "server.hpp"
#include "bench.hpp"
#include "cpuinfo.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso:\n"
                  << "  eie-llm <model.gguf>                   → shell\n"
                  << "  eie-llm <model.gguf> --server [porta]  → server\n"
                  << "  eie-llm <model.gguf> --bench [n_tok]   → benchmark\n";
        return 1;
    }

    // ── Caricamento con timing ─────────────────
    auto t_load_start = now();

    std::ifstream f(argv[1], std::ios::binary);
    if (!f) {
        std::cerr << "[ERRORE] Impossibile aprire: " << argv[1] << "\n";
        return 1;
    }

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

    auto t_load_end = now();
    double load_ms = ms_between(t_load_start, t_load_end);

    std::cerr << "✓ Pronto! (caricamento: "
              << std::fixed << std::setprecision(0)
              << load_ms << " ms)\n";

    // ── Info build e ottimizzazioni ───────────
    std::cerr << "  Build type: "
#ifdef NDEBUG
              << "Release (-O3)"
#else
              << "Debug (-O0)"
#endif
              << "\n";

#ifdef _OPENMP
    std::cerr << "  OpenMP:     attivo (multicore)\n";
#else
    std::cerr << "  OpenMP:     non disponibile\n";
#endif

    CPUFeatures feat = cpu_features();
    std::cerr << "  AVX2:       " << (feat.avx2 ? "sì" : "no") << "\n";
    std::cerr << "  FMA:        " << (feat.fma  ? "sì" : "no") << "\n";
    if (feat.avx2 && !avx2_enabled()) {
        std::cerr << "  AVX2 disabilitato via EIE_NO_AVX2=1\n";
    }

    // ── Modalità ──────────────────────────────
    std::string mode = argc >= 3 ? argv[2] : "";

    if (mode == "--server") {
        int port = (argc >= 4) ? std::atoi(argv[3]) : 8080;
        server_run(model, tok, port);

    } else if (mode == "--bench") {
        int n_tokens = (argc >= 4) ? std::atoi(argv[3]) : 50;

        std::string prompt = "The history of artificial intelligence "
                             "began in the early days of computing";

        std::cerr << "Benchmark in corso...\n";
        std::cerr << "  Prompt  : \"" << prompt << "\"\n";
        std::cerr << "  Genera  : " << n_tokens << " token\n\n";

        BenchResult r = bench_run(model, tok, prompt, n_tokens);
        r.load_ms = load_ms;
        bench_print(r);

    } else {
        shell_run(model, tok);
    }

    return 0;
}