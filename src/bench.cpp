#include "bench.hpp"
#include <iostream>
#include <iomanip>

#include "model.hpp"
#include "tokenizer.hpp"
#include "ops.hpp"

BenchResult bench_run(Model& model, const Tokenizer& tok, const std::string& prompt, int n_generate) {
    BenchResult r;

    // Azzera gli accumulatori
    model.bench = BenchAccum{};

    // ── Tokenizza il prompt ───────────────────
    auto input_ids = tokenizer_encode(tok, prompt);
    r.prefill_tokens  = static_cast<int>(input_ids.size());
    r.generate_tokens = n_generate;

    std::vector<float> logits;

    // ── Benchmark prefill ─────────────────────
    model_init_kvcache(model);

    auto tp0 = now();
    int pos = 0;
    for (int id : input_ids) {
        // prefill senza bench_mode — non ci interessa
        // il breakdown del prefill, solo il tempo totale
        forward(model, id, pos, logits, false);
        pos++;
    }
    auto tp1 = now();
    r.prefill_ms = ms_between(tp0, tp1);
    model.kv_cache.n_cached = pos;

    // ── Benchmark generazione ─────────────────
    // Ora attiviamo bench_mode per raccogliere
    // il breakdown operazione per operazione
    model.bench = BenchAccum{};  // azzera di nuovo

    int next_token = sample_argmax(logits);  // greedy per riproducibilità

    auto tg0 = now();
    for (int i = 0; i < n_generate; i++) {
        if (next_token == 50256) {
            r.generate_tokens = i;
            break;
        }

        forward(model, next_token, pos, logits, true);
        pos++;

        // Sampling — greedy per il benchmark
        // (vogliamo misurare il forward, non il sampling)
        auto tsa0 = now();
        next_token = sample_argmax(logits);
        auto tsa1 = now();

        model.bench.sample_ms += ms_between(tsa0, tsa1);
    }
    auto tg1 = now();
    r.generate_ms = ms_between(tg0, tg1);

    // ── Calcola medie per token ───────────────
    if (model.bench.n_steps > 0) {
        double n = model.bench.n_steps;
        r.avg_embed_ms  = model.bench.embed_ms  / n;
        r.avg_attn_ms   = model.bench.attn_ms   / n;
        r.avg_ffn_ms    = model.bench.ffn_ms    / n;
        r.avg_sample_ms = model.bench.sample_ms / n;
    }

    return r;
}

// ─────────────────────────────────────────────
//  Stampa risultati benchmark
//
//  Calcola tokens/sec dividendo il numero di
//  token per il tempo in secondi.
//  Separiamo prefill e generazione perché hanno
//  caratteristiche molto diverse:
//  - prefill: elabora N token in parallelo
//    (anche se nel nostro engine è sequenziale)
//  - generazione: un token alla volta,
//    limitata dalla latenza della KV cache
// ─────────────────────────────────────────────
void bench_print(const BenchResult& r) {
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════╗\n";
    std::cout << "║   EIE-LLM — Benchmark Results         ║\n";
    std::cout << "╠═══════════════════════════════════════╣\n";

    // ── Caricamento ───────────────────────────
    std::cout << "║  CARICAMENTO                          ║\n";
    std::cout << "║  Tempo totale : "
              << std::fixed << std::setprecision(1)
              << std::setw(8) << r.load_ms << " ms"
              << "                 ║\n";

    std::cout << "╠═══════════════════════════════════════╣\n";

    // ── Prefill ───────────────────────────────
    double prefill_tps = r.prefill_tokens /
                         (r.prefill_ms / 1000.0);
    std::cout << "║  PREFILL (elaborazione prompt)        ║\n";
    std::cout << "║  Token      : "
              << std::setw(8) << r.prefill_tokens
              << "                      ║\n";
    std::cout << "║  Tempo      : "
              << std::setw(8) << std::fixed << std::setprecision(1)
              << r.prefill_ms << " ms"
              << "                 ║\n";
    std::cout << "║  Velocità   : "
              << std::setw(8) << std::fixed << std::setprecision(1)
              << prefill_tps << " tok/s"
              << "              ║\n";
    std::cout << "║  Per token  : "
              << std::setw(8) << std::fixed << std::setprecision(2)
              << (r.prefill_ms / r.prefill_tokens) << " ms/tok"
              << "             ║\n";

    std::cout << "╠═══════════════════════════════════════╣\n";

    // ── Generazione ───────────────────────────
    if (r.generate_tokens > 0) {
        double gen_tps = r.generate_tokens /
                         (r.generate_ms / 1000.0);
        std::cout << "║  GENERAZIONE                          ║\n";
        std::cout << "║  Token      : "
                  << std::setw(8) << r.generate_tokens
                  << "                      ║\n";
        std::cout << "║  Tempo      : "
                  << std::setw(8) << std::fixed << std::setprecision(1)
                  << r.generate_ms << " ms"
                  << "                 ║\n";
        std::cout << "║  Velocità   : "
                  << std::setw(8) << std::fixed << std::setprecision(1)
                  << gen_tps << " tok/s"
                  << "              ║\n";
        std::cout << "║  Per token  : "
                  << std::setw(8) << std::fixed << std::setprecision(2)
                  << (r.generate_ms / r.generate_tokens) << " ms/tok"
                  << "             ║\n";

        std::cout << "╠═══════════════════════════════════════╣\n";

        // ── Breakdown per operazione ──────────
        std::cout << "║  BREAKDOWN (media per token)          ║\n";
        std::cout << "║  Embedding  : "
                  << std::setw(8) << std::fixed << std::setprecision(3)
                  << r.avg_embed_ms << " ms"
                  << "                 ║\n";
        std::cout << "║  Attention  : "
                  << std::setw(8) << std::fixed << std::setprecision(3)
                  << r.avg_attn_ms << " ms"
                  << "                 ║\n";
        std::cout << "║  FFN        : "
                  << std::setw(8) << std::fixed << std::setprecision(3)
                  << r.avg_ffn_ms << " ms"
                  << "                 ║\n";
        std::cout << "║  Sampling   : "
                  << std::setw(8) << std::fixed << std::setprecision(3)
                  << r.avg_sample_ms << " ms"
                  << "                 ║\n";

        // Mostra la percentuale di ogni fase
        double total_op = r.avg_embed_ms + r.avg_attn_ms
                        + r.avg_ffn_ms   + r.avg_sample_ms;
        if (total_op > 0) {
            std::cout << "╠═══════════════════════════════════════╣\n";
            std::cout << "║  DISTRIBUZIONE TEMPO                  ║\n";

            auto print_bar = [&](const std::string& label,
                                  double val) {
                double pct = val / total_op * 100.0;
                int    bar = static_cast<int>(pct / 5.0); // 1 char = 5%
                std::cout << "║  " << std::left << std::setw(10) << label
                          << " ";
                for (int i = 0; i < 20; i++)
                    std::cout << (i < bar ? "█" : "░");
                std::cout << " "
                          << std::right << std::setw(4)
                          << std::fixed << std::setprecision(0)
                          << pct << "%  ║\n";
            };

            print_bar("Embedding",  r.avg_embed_ms);
            print_bar("Attention",  r.avg_attn_ms);
            print_bar("FFN",        r.avg_ffn_ms);
            print_bar("Sampling",   r.avg_sample_ms);
        }
    }

    std::cout << "╚═══════════════════════════════════════╝\n\n";
}