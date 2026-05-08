#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <cstdint>

// ─────────────────────────────────────────────
//  Timing utilities
//
//  Usiamo std::chrono::high_resolution_clock
//  per misurazioni ad alta precisione.
//  Tutto è in millisecondi (double) per leggibilità.
// ─────────────────────────────────────────────

// Tipo per i timestamp
using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

// Ritorna il timestamp corrente
inline TimePoint now() {
    return std::chrono::high_resolution_clock::now();
}

// Calcola i millisecondi tra due timestamp
inline double ms_between(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(
        end - start).count();
}

// ─────────────────────────────────────────────
//  Accumulatori per il benchmark
//
//  Vengono aggiornati ad ogni forward pass
//  quando il benchmark è attivo.
//  Azzerati prima di ogni sessione di misura.
// ─────────────────────────────────────────────
struct BenchAccum {
    double embed_ms  = 0.0;
    double attn_ms   = 0.0;
    double ffn_ms    = 0.0;
    double sample_ms = 0.0;
    int    n_steps   = 0;    // quanti forward pass contribuiscono
};


// ─────────────────────────────────────────────
//  Risultati del benchmark
//
//  Raccoglie tutte le misure in una struct
//  per stamparle in modo organizzato alla fine
// ─────────────────────────────────────────────
struct BenchResult {
    // Caricamento
    double load_ms       = 0.0;  // tempo caricamento modello

    // Prefill (elaborazione prompt)
    double prefill_ms    = 0.0;  // tempo totale prefill
    int    prefill_tokens = 0;   // token del prompt

    // Generazione
    double generate_ms   = 0.0;  // tempo totale generazione
    int    generate_tokens = 0;  // token generati

    // Breakdown per operazione (media per token)
    double avg_embed_ms  = 0.0;  // embedding lookup
    double avg_attn_ms   = 0.0;  // self-attention (tutti i layer)
    double avg_ffn_ms    = 0.0;  // feed-forward (tutti i layer)
    double avg_sample_ms = 0.0;  // sampling
};

struct Model;
struct Tokenizer;

// Esegue il benchmark completo:
// misura caricamento, prefill e generazione
// Ritorna i risultati in BenchResult
BenchResult bench_run(Model& model, const Tokenizer& tok, const std::string& prompt, int n_generate);

// Stampa i risultati in modo leggibile
void bench_print(const BenchResult& r);

