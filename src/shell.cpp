#include "shell.hpp"
#include "model.hpp"
#include "tokenizer.hpp"
#include "ops.hpp"
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include "linenoise.hpp"

// ─────────────────────────────────────────────
//  File dove salviamo la history tra sessioni
//  Viene caricato all'avvio e salvato all'uscita
// ─────────────────────────────────────────────
static const char* HISTORY_FILE = ".eie_history";

// ─────────────────────────────────────────────
//  Sanitizza l'output del modello
//
//  GPT-2 BPE byte-level mappa i 256 byte a
//  caratteri Unicode specifici. In particolare
//  i byte 0x00-0x20 e 0x7F vengono mappati al
//  blocco Unicode U+0100-U+017F (Latin Extended)
//  invece di usare i caratteri di controllo diretti.
//
//  Esempio:
//    byte 0x0A (\n) → U+010A (Ċ) nel vocabolario GPT-2
//    byte 0x20 ( )  → U+0120 (Ġ) → già gestito dal decoder
//
//  Strategia:
//  - ASCII stampabile [0x20-0x7E] → passa
//  - Newline 0x0A → passa (utile per la formattazione)
//  - Caratteri Latin Extended U+0100-U+017F →
//    potrebbero essere byte rimappati da GPT-2,
//    li convertiamo al byte originale se stampabile
//    altrimenti li scartiamo
//  - Resto del Unicode → passa (emoji, CJK, ecc.
//    potrebbero essere intenzionali)
//  - Byte di controllo → scarta
// ─────────────────────────────────────────────
static std::string sanitize_output(const std::string& s) {
    std::string out;
    out.reserve(s.size());

    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);

        // ── ASCII stampabile → passa direttamente ──
        if (c >= 0x20 && c <= 0x7E) {
            out += s[i++];
            continue;
        }

        // ── Newline → passa ──
        if (c == 0x0A) {
            out += s[i++];
            continue;
        }

        // ── Sequenza UTF-8 a 2 byte ──────────────
        // Formato: [110xxxxx][10xxxxxx]
        // Copriamo U+0080 .. U+07FF
        if (c >= 0xC0 && c <= 0xDF && i + 1 < s.size()) {
            unsigned char c2 = static_cast<unsigned char>(s[i+1]);

            if ((c2 & 0xC0) == 0x80) {
                // Ricostruisci il codepoint Unicode
                uint32_t cp = ((c & 0x1F) << 6) | (c2 & 0x3F);

                // U+0100–U+017F = Latin Extended-A
                // GPT-2 usa questo blocco per rimappare
                // i byte 0x00-0x7F non stampabili.
                // Recuperiamo il byte originale:
                //   byte_originale = codepoint - 0x100 + 0x00
                // ma solo se il risultato è stampabile
                if (cp >= 0x0100 && cp <= 0x017F) {
                    uint8_t original_byte = static_cast<uint8_t>(cp - 0x100);
                    // Tieni solo se stampabile o newline
                    if (original_byte >= 0x20 && original_byte <= 0x7E)
                        out += static_cast<char>(original_byte);
                    else if (original_byte == 0x0A)
                        out += '\n';
                    // altrimenti scarta silenziosamente
                } else {
                    // Altro carattere a 2 byte — passa così com'è
                    out += s[i];
                    out += s[i+1];
                }
                i += 2;
                continue;
            }
        }

        // ── Sequenze UTF-8 a 3-4 byte → passa ────
        // (es. emoji, CJK — probabilmente intenzionali)
        if (c >= 0xE0 && c <= 0xEF && i + 2 < s.size()) {
            out += s[i]; out += s[i+1]; out += s[i+2];
            i += 3;
            continue;
        }
        if (c >= 0xF0 && c <= 0xF7 && i + 3 < s.size()) {
            out += s[i]; out += s[i+1]; out += s[i+2]; out += s[i+3];
            i += 4;
            continue;
        }

        // ── Tutto il resto (byte di controllo, ecc.) → scarta ──
        i++;
    }

    return out;
}
// ─────────────────────────────────────────────
//  Stampa i comandi disponibili
// ─────────────────────────────────────────────
static void print_help() {
    std::cout
        << "\n  Comandi disponibili:\n"
        << "  :help                mostra questo messaggio\n"
        << "  :tokens <n>          max token da generare (default 200)\n"
        << "  :temp <f>            temperatura, es. 0.8  (default 1.0)\n"
        << "  :topk <n>            top-k sampling        (default 40)\n"
        << "  :topp <f>            nucleus sampling p    (default 0.9)\n"
        << "  :penalty <f>         repetition penalty    (default 1.1)\n"
        << "  :greedy              attiva sampling greedy (argmax)\n"
        << "  :sample              attiva top-k + top-p sampling\n"
        << "  :params              mostra parametri correnti\n"
        << "  :reset               azzera la KV cache\n"
        << "  :quit                esci\n"
        << "  qualsiasi testo      genera completamento\n"
        << "\n  Scorciatoie tastiera:\n"
        << "  ↑ ↓                  naviga la history\n"
        << "  Ctrl+R               cerca nella history\n"
        << "  Ctrl+A / Ctrl+E      inizio / fine riga\n"
        << "  Ctrl+C               annulla riga corrente\n"
        << "  Ctrl+D               esci\n\n";
}

// ─────────────────────────────────────────────
//  Stampa i parametri correnti
// ─────────────────────────────────────────────
static void print_params(const SamplingParams& p, int max_tokens) {
    std::cout << "\n  Parametri correnti:\n"
              << "  modalità   : " << (p.greedy ? "greedy" : "top-k+top-p") << "\n"
              << "  temperature: " << p.temperature  << "\n"
              << "  top_k      : " << p.top_k        << "\n"
              << "  top_p      : " << p.top_p        << "\n"
              << "  rep_penalty: " << p.rep_penalty  << "\n"
              << "  max_tokens : " << max_tokens     << "\n\n";
}

// ─────────────────────────────────────────────
//  Generazione con streaming e sanitizzazione
// ─────────────────────────────────────────────
static void generate(Model& model,
                     const Tokenizer& tok,
                     const std::string& prompt,
                     const SamplingParams& params,
                     int max_tokens) {
    model_init_kvcache(model);

    auto input_ids = tokenizer_encode(tok, prompt);
    if (input_ids.empty()) {
        std::cout << "[AVVISO] Prompt vuoto\n";
        return;
    }

    std::vector<int> context_ids = input_ids;
    std::vector<float> logits;

    std::cout << "\n" << prompt;
    std::cout.flush();

    // Prefill
    int pos = 0;
    for (int id : input_ids) {
        forward(model, id, pos, logits);
        pos++;
    }
    model.kv_cache.n_cached = pos;

    // Generazione
    int generated = 0;
    while (generated < max_tokens) {
        apply_repetition_penalty(logits, context_ids,
                                 params.rep_penalty);

        int next_token;
        if (params.greedy)
            next_token = sample_argmax(logits);
        else
            next_token = sample_topk_topp(logits,
                                          params.top_k,
                                          params.top_p,
                                          params.temperature);

        if (next_token == 50256) break;

        // Sanitizza prima di stampare
        std::string raw = tokenizer_decode(tok, {next_token});
        std::cout << sanitize_output(raw);
        std::cout.flush();

        context_ids.push_back(next_token);
        forward(model, next_token, pos, logits);
        pos++;
        generated++;
    }

    std::cout << "\n\n";
}

// ─────────────────────────────────────────────
//  Loop principale della shell
//
//  Usiamo linenoise invece di std::getline:
//  - history con ↑↓
//  - Ctrl+R per ricerca
//  - editing inline completo
//  - history persistente su file tra sessioni
// ─────────────────────────────────────────────
void shell_run(Model& model, const Tokenizer& tok) {
    SamplingParams params;
    int max_tokens = 200;
    srand(static_cast<unsigned>(time(nullptr)));

    // Configura linenoise
    linenoise::SetMultiLine(false);
    linenoise::SetHistoryMaxLen(100);
    linenoise::LoadHistory(HISTORY_FILE);  // carica history precedente

    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════╗\n";
    std::cout << "║   EIE-LLM — Educational Infer Engine  ║\n";
    std::cout << "║   GPT-2 small  •  CPU only  •  C++17  ║\n";
    std::cout << "╚═══════════════════════════════════════╝\n";
    print_help();

    while (true) {
        std::string line;

        // linenoise gestisce il prompt, la history e l'editing
        // Ritorna true se l'utente ha premuto Ctrl+D (EOF)
        bool quit = linenoise::Readline("eie> ", line);
        if (quit) {
            std::cout << "Arrivederci!\n";
            break;
        }

        if (line.empty()) continue;

        // Aggiungi alla history (evita duplicati consecutivi)
        linenoise::AddHistory(line.c_str());

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
                std::cout << "  Modalità: top-k + top-p sampling\n\n";
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
            else if (line.substr(0, 5) == ":topk") {
                try {
                    params.top_k = std::stoi(line.substr(6));
                    params.greedy = false;
                    std::cout << "  Top-k: " << params.top_k << "\n\n";
                } catch (...) {
                    std::cout << "  Uso: :topk <intero>\n\n";
                }
            }
            else if (line.substr(0, 5) == ":topp") {
                try {
                    params.top_p = std::stof(line.substr(6));
                    params.greedy = false;
                    std::cout << "  Top-p: " << params.top_p << "\n\n";
                } catch (...) {
                    std::cout << "  Uso: :topp <float>\n\n";
                }
            }
            else if (line.substr(0, 8) == ":penalty") {
                try {
                    params.rep_penalty = std::stof(line.substr(9));
                    std::cout << "  Repetition penalty: "
                              << params.rep_penalty << "\n\n";
                } catch (...) {
                    std::cout << "  Uso: :penalty <float>\n\n";
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

    // Salva la history su file prima di uscire
    linenoise::SaveHistory(HISTORY_FILE);
}