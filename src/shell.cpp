#include "shell.hpp"
#include "model.hpp"
#include "tokenizer.hpp"
#include "ops.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>

// ─────────────────────────────────────────────
//  Configurazione della shell
//  Tutti i parametri di generazione
//  sono raggruppati qui per comodità
// ─────────────────────────────────────────────
struct ShellConfig {
    int   max_tokens  = 200;    // token massimi per risposta
    float temperature = 1.0f;   // 1.0 = sampling normale
    bool  greedy      = false;  // true = argmax, false = temperatura
};

// ─────────────────────────────────────────────
//  Stampa i comandi disponibili
// ─────────────────────────────────────────────
static void print_help() {
    std::cout << "\n  Comandi disponibili:\n"
              << "  :help              mostra questo messaggio\n"
              << "  :tokens <n>        imposta max token (default 200)\n"
              << "  :temp <f>          imposta temperatura (default 1.0)\n"
              << "  :greedy            attiva sampling greedy\n"
              << "  :sample            attiva sampling con temperatura\n"
              << "  :reset             azzera la KV cache\n"
              << "  :quit              esci\n"
              << "  qualsiasi testo    genera completamento\n\n";
}

// ─────────────────────────────────────────────
//  Genera e stampa i token in streaming
//
//  A differenza del main, qui stampiamo ogni
//  token appena viene generato (flush immediato)
//  per dare la sensazione di generazione live.
//  Azzeriamo anche la KV cache prima di ogni
//  nuova generazione.
// ─────────────────────────────────────────────
static void generate(Model& model,
                     const Tokenizer& tok,
                     const std::string& prompt,
                     const ShellConfig& cfg) {
    // Azzera la KV cache per ogni nuova generazione
    model_init_kvcache(model);

    // Tokenizza il prompt
    auto input_ids = tokenizer_encode(tok, prompt);
    if (input_ids.empty()) {
        std::cout << "[AVVISO] Prompt vuoto\n";
        return;
    }

    std::vector<float> logits;
    std::cout << "\n";

    // ── Fase 1: prefill ───────────────────────
    // Processa tutti i token del prompt
    // riempiendo la KV cache senza stampare nulla
    int pos = 0;
    for (int id : input_ids) {
        forward(model, id, pos, logits);
        pos++;
    }
    model.kv_cache.n_cached = pos;

    // ── Fase 2: generazione streaming ─────────
    // Stampa il prompt originale
    std::cout << prompt;
    std::cout.flush();

    // Campiona il primo token dopo il prompt
    int next_token = cfg.greedy
        ? sample_argmax(logits)
        : sample_temperature(logits, cfg.temperature);

    int generated = 0;
    while (generated < cfg.max_tokens) {
        // Token di fine testo GPT-2 = 50256
        if (next_token == 50256) break;

        // Decodifica e stampa immediatamente
        std::cout << tokenizer_decode(tok, {next_token});
        std::cout.flush();

        // Forward pass per il prossimo token
        forward(model, next_token, pos, logits);
        pos++;
        generated++;

        // Campiona il prossimo token
        next_token = cfg.greedy
            ? sample_argmax(logits)
            : sample_temperature(logits, cfg.temperature);
    }

    std::cout << "\n\n";
}

// ─────────────────────────────────────────────
//  Loop principale della shell
//
//  Legge comandi e prompt da stdin in loop.
//  I comandi iniziano con ':' e modificano
//  la configurazione senza generare testo.
//  Tutto il resto viene trattato come prompt.
// ─────────────────────────────────────────────
void shell_run(Model& model, const Tokenizer& tok) {
    ShellConfig cfg;
    srand(static_cast<unsigned>(time(nullptr)));

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════╗\n";
    std::cout << "║   EIE-LLM — Educational Infer Engine  ║\n";
    std::cout << "║   GPT-2 small  •  CPU only  •  C++17  ║\n";
    std::cout << "╚═══════════════════════════════════════╝\n";
    print_help();

    std::string line;
    while (true) {
        // Prompt della shell
        std::cout << "eie> ";
        std::cout.flush();

        if (!std::getline(std::cin, line)) break;  // EOF (Ctrl+D)
        if (line.empty()) continue;

        // ── Gestione comandi ──────────────────
        if (line[0] == ':') {

            if (line == ":quit" || line == ":q") {
                std::cout << "Arrivederci!\n";
                break;
            }
            else if (line == ":help" || line == ":h") {
                print_help();
            }
            else if (line == ":reset") {
                model_init_kvcache(model);
                std::cout << "  KV cache azzerata\n\n";
            }
            else if (line == ":greedy") {
                cfg.greedy = true;
                std::cout << "  Modalità: greedy (argmax)\n\n";
            }
            else if (line == ":sample") {
                cfg.greedy = false;
                std::cout << "  Modalità: sampling (temperatura "
                          << cfg.temperature << ")\n\n";
            }
            else if (line.substr(0, 7) == ":tokens") {
                try {
                    cfg.max_tokens = std::stoi(line.substr(8));
                    std::cout << "  Max token: "
                              << cfg.max_tokens << "\n\n";
                } catch (...) {
                    std::cout << "  Uso: :tokens <numero>\n\n";
                }
            }
            else if (line.substr(0, 5) == ":temp") {
                try {
                    cfg.temperature = std::stof(line.substr(6));
                    cfg.greedy      = false;
                    std::cout << "  Temperatura: "
                              << cfg.temperature << "\n\n";
                } catch (...) {
                    std::cout << "  Uso: :temp <float>\n\n";
                }
            }
            else {
                std::cout << "  Comando sconosciuto. "
                          << "Usa :help per la lista.\n\n";
            }
        }
        else {
            // ── Generazione testo ─────────────
            generate(model, tok, line, cfg);
        }
    }
}