#include "shell.hpp"
#include "model.hpp"
#include "tokenizer.hpp"
#include "ops.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>

// ─────────────────────────────────────────────
//  Stampa i comandi disponibili
// ─────────────────────────────────────────────
static void print_help() {
    std::cout
        << "\n  Comandi disponibili:\n"
        << "  :help                mostra questo messaggio\n"
        << "  :tokens <n>          max token da generare (default 200)\n"
        << "  :temp <f>            temperatura, es. 0.8  (default 1.0)\n"
        << "  :topp <f>            nucleus sampling p    (default 0.9)\n"
        << "  :topk <n>            top-k sampling, 0=disabilitato (default 40)\n"        
        << "  :penalty <f>         repetition penalty    (default 1.1)\n"
        << "  :greedy              attiva sampling greedy (argmax)\n"
        << "  :sample              attiva top-p sampling\n"
        << "  :params              mostra parametri correnti\n"
        << "  :reset               azzera la KV cache\n"
        << "  :quit                esci\n"
        << "  qualsiasi testo      genera completamento\n\n";
}

// ─────────────────────────────────────────────
//  Stampa i parametri correnti
// ─────────────────────────────────────────────
static void print_params(const SamplingParams& p, int max_tokens) {
    std::cout << "\n  Parametri correnti:\n"
              << "  modalità   : " << (p.greedy ? "greedy" : "top-p") << "\n"
              << "  temperature: " << p.temperature  << "\n"
              << "  top_p      : " << p.top_p        << "\n"
              << "  top_k      : " << p.top_k        << "\n"
              << "  rep_penalty: " << p.rep_penalty  << "\n"
              << "  max_tokens : " << max_tokens     << "\n\n";
}

// ─────────────────────────────────────────────
//  Generazione con streaming
//
//  Mantiene un vettore context_ids che cresce
//  ad ogni token generato — serve per la
//  repetition penalty che ha bisogno di sapere
//  quali token sono già stati prodotti.
// ─────────────────────────────────────────────
static void generate(Model& model,
                     const Tokenizer& tok,
                     const std::string& prompt,
                     const SamplingParams& params,
                     int max_tokens) {
    // Reset KV cache per ogni nuova generazione
    model_init_kvcache(model);

    // Tokenizza il prompt
    auto input_ids = tokenizer_encode(tok, prompt);
    if (input_ids.empty()) {
        std::cout << "[AVVISO] Prompt vuoto\n";
        return;
    }

    // context_ids tiene traccia di tutti i token
    // visti (prompt + generati) per la rep penalty
    std::vector<int> context_ids = input_ids;

    std::vector<float> logits;
    std::cout << "\n" << prompt;
    std::cout.flush();

    // ── Fase 1: prefill ───────────────────────
    int pos = 0;
    for (int id : input_ids) {
        forward(model, id, pos, logits);
        pos++;
    }
    model.kv_cache.n_cached = pos;

    // ── Fase 2: generazione streaming ─────────
    int generated = 0;
    while (generated < max_tokens) {

        // Applica repetition penalty prima del sampling
        apply_repetition_penalty(logits, context_ids,
                                 params.rep_penalty);

        // Campiona il prossimo token
        int next_token;
        if (params.greedy) {
            next_token = sample_argmax(logits);
        } else {
            next_token = params.greedy
                ? sample_argmax(logits)
                : sample_topk_topp(logits, params.top_k, params.top_p, params.temperature);
        }

        // Token EOS di GPT-2
        if (next_token == 50256) break;

        // Stampa e aggiorna il contesto
        std::cout << tokenizer_decode(tok, {next_token});
        std::cout.flush();
        context_ids.push_back(next_token);

        // Forward pass per il prossimo token
        forward(model, next_token, pos, logits);
        pos++;
        generated++;
    }

    std::cout << "\n\n";
}

// ─────────────────────────────────────────────
//  Loop principale della shell
// ─────────────────────────────────────────────
void shell_run(Model& model, const Tokenizer& tok) {
    SamplingParams params;
    int max_tokens = 200;
    srand(static_cast<unsigned>(time(nullptr)));

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════╗\n";
    std::cout << "║   EIE-LLM — Educational Infer Engine  ║\n";
    std::cout << "║   GPT-2 small  •  CPU only  •  C++17  ║\n";
    std::cout << "╚═══════════════════════════════════════╝\n";
    print_help();

    std::string line;
    while (true) {
        std::cout << "eie> ";
        std::cout.flush();

        if (!std::getline(std::cin, line)) break;
        if (line.empty()) continue;

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
                params.greedy = true;
                std::cout << "  Modalità: greedy\n\n";
            }
            else if (line == ":sample") {
                params.greedy = false;
                std::cout << "  Modalità: top-p sampling\n\n";
            }
            else if (line == ":params") {
                print_params(params, max_tokens);
            }
            else if (line.substr(0, 7) == ":tokens") {
                try {
                    max_tokens = std::stoi(line.substr(8));
                    std::cout << "  Max token: " << max_tokens << "\n\n";
                } catch (...) {
                    std::cout << "  Uso: :tokens <intero>\n\n";
                }
            }
            else if (line.substr(0, 5) == ":temp") {
                try {
                    params.temperature = std::stof(line.substr(6));
                    params.greedy = false;
                    std::cout << "  Temperatura: "
                              << params.temperature << "\n\n";
                } catch (...) {
                    std::cout << "  Uso: :temp <float>\n\n";
                }
            }
            else if (line.substr(0, 5) == ":topp") {
                try {
                    params.top_p = std::stof(line.substr(6));
                    params.greedy = false;
                    std::cout << "  Top-p: " << params.top_p << "\n\n";
                } catch (...) {
                    std::cout << "  Uso: :topp <float 0.0-1.0>\n\n";
                }
            }
            else if (line.substr(0, 5) == ":topk") {
                try {
                    params.top_k = std::stoi(line.substr(6));
                    params.greedy = false;
                    std::cout << "  Top-k: " << params.top_k
                              << (params.top_k == 0 ?
                                  " (disabilitato)\n\n" : "\n\n");
                } catch (...) {
                    std::cout << "  Uso: :topk <intero >= 0>\n\n";
                }
            }
            else if (line.substr(0, 8) == ":penalty") {
                try {
                    params.rep_penalty = std::stof(line.substr(9));
                    std::cout << "  Repetition penalty: "
                              << params.rep_penalty << "\n\n";
                } catch (...) {
                    std::cout << "  Uso: :penalty <float >= 1.0>\n\n";
                }
            }
            else {
                std::cout << "  Comando sconosciuto. "
                          << "Usa :help per la lista.\n\n";
            }
        } else {
            generate(model, tok, line, params, max_tokens);
        }
    }
}