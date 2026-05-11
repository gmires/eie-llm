#include "shell.hpp"
#include "model.hpp"
#include "tokenizer.hpp"
#include "ops.hpp"
#include "prefix_cache.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <ctime>
#include "linenoise.hpp"

// ─────────────────────────────────────────────
//  Prefix Cache per la shell interattiva.
//
//  Anche se la shell gestisce un solo utente,
//  la cache è utile quando lo stesso prompt
//  viene ripetuto (es. testing con prompt fissi).
//  Dimostra lo stesso meccanismo del server.
// ─────────────────────────────────────────────
static PrefixCache shell_prefix_cache;

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
static void print_help(bool chat_mode, bool speculative_mode, int draft_k) {
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
        << "  :speculative         attiva/disattiva speculative decoding\n"
        << "  :draft <n>           token draft da generare (default 4)\n"
        << "  :cacheinfo           mostra statistiche prefix cache\n"
        << "  :cacheclear          svuota la prefix cache\n"
        << "  :inspect <testo>     esporta attention scores in JSON\n"
        << "  :chat                modalità chat (applica template)\n"
        << "  :raw                 modalità raw  (prompt diretto)\n"
        << "  :params              mostra parametri correnti\n"
        << "  :reset               azzera la KV cache\n"
        << "  :quit                esci\n"
        << "  qualsiasi testo      genera completamento\n"
        << "\n  Modalità corrente: " << (chat_mode ? "chat" : "raw")
        << " | Speculative: " << (speculative_mode ? "on (draft=" + std::to_string(draft_k) + ")" : "off") << "\n"
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
static void print_params(const SamplingParams& p, int max_tokens,
                         bool speculative_mode, int draft_k) {
    std::cout << "\n  Parametri correnti:\n"
              << "  modalità    : " << (p.greedy ? "greedy" : "top-k+top-p") << "\n"
              << "  temperature : " << p.temperature  << "\n"
              << "  top_k       : " << p.top_k        << "\n"
              << "  top_p       : " << p.top_p        << "\n"
              << "  rep_penalty : " << p.rep_penalty  << "\n"
              << "  max_tokens  : " << max_tokens     << "\n"
              << "  speculative : " << (speculative_mode ? "on" : "off") << "\n"
              << "  draft_k     : " << draft_k        << "\n\n";
}

// ─────────────────────────────────────────────
//  Generazione con streaming e sanitizzazione
//
//  print_prompt controlla se stampare il testo del prompt
//  prima della generazione. In modalità chat viene disabilitato
//  perché il prompt include i tag del template (<|user|> ecc.)
//  che non devono apparire sullo schermo.
// ─────────────────────────────────────────────
static void generate(Model& model,
                     const Tokenizer& tok,
                     const std::string& prompt,
                     const SamplingParams& params,
                     int max_tokens,
                     bool print_prompt = true,
                     PrefixCache* pcache = nullptr) {

    auto input_ids = tokenizer_encode(tok, prompt);
    if (input_ids.empty()) {
        std::cout << "[AVVISO] Prompt vuoto\n";
        return;
    }

    std::vector<int> context_ids = input_ids;
    std::vector<float> logits;
    int pos = 0;

    if (print_prompt) {
        std::cout << "\n" << prompt;
        std::cout.flush();
    }

    // Prefill — con supporto a prefix cache
    if (!input_ids.empty()) {
        bool cache_used = false;
        if (pcache) {
            int cached_tokens = 0;
            if (pcache->lookup(prompt, model, cached_tokens)) {
                if (cached_tokens == (int)input_ids.size()) {
                    // Cache hit esatto: riusa KV cache
                    std::cout << " [cache hit]";
                    std::cout.flush();
                    forward(model, input_ids.back(), cached_tokens - 1, logits);
                    pos = cached_tokens;
                    cache_used = true;
                }
            }
        }

        if (!cache_used) {
            model_init_kvcache(model);
            forward_prefill(model, input_ids, logits);
            pos = static_cast<int>(input_ids.size());
            if (pcache) {
                pcache->store(prompt, model, pos);
            }
        }
    } else {
        model_init_kvcache(model);
        logits.resize(model.config.n_vocab);
        pos = 0;
    }

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

        if (next_token == tok.eos_id) break;

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
//  Generazione con Speculative Decoding
//
//  Algoritmo draft & verify:
//  1. Genera K token draft in modalità greedy
//     (senza stamparli — potrebbero essere rifiutati)
//  2. Verifica tutti i draft in un unico forward
//     pass batch (forward_verify)
//  3. Confronta ogni draft con il campionamento del
//     target model dai logits della verifica
//  4. Accetta i token finché coincidono; al primo
//     rifiuto usa il token campionato e rigenera
//
//  Il guadagno di velocità deriva dal fatto che
//  la verifica batch è più efficiente di K forward
//  pass sequenziali grazie al riutilizzo dei pesi
//  in cache (matvec_quant_batch).
// ─────────────────────────────────────────────
static void generate_speculative(Model& model,
                                  const Tokenizer& tok,
                                  const std::string& prompt,
                                  const SamplingParams& params,
                                  int max_tokens,
                                  int draft_k = 4,
                                  bool print_prompt = true) {
    model_init_kvcache(model);

    auto input_ids = tokenizer_encode(tok, prompt);
    if (input_ids.empty()) {
        std::cout << "[AVVISO] Prompt vuoto\n";
        return;
    }

    std::vector<int> context_ids = input_ids;
    std::vector<float> logits;

    if (print_prompt) {
        std::cout << "\n" << prompt;
        std::cout.flush();
    }

    // Prefill
    if (!input_ids.empty()) {
        forward_prefill(model, input_ids, logits);
    } else {
        logits.resize(model.config.n_vocab);
    }
    int pos = static_cast<int>(input_ids.size());

    // Campiona il primo token
    apply_repetition_penalty(logits, context_ids, params.rep_penalty);
    int next_token = params.greedy
        ? sample_argmax(logits)
        : sample_topk_topp(logits, params.top_k, params.top_p, params.temperature);

    int generated = 0;
    while (generated < max_tokens && next_token != tok.eos_id) {

        // =====================================================================
        //  FASE 1 — DRAFT: genera K token "provvisori" in modalità greedy
        //
        //  Usiamo il modello stesso come "draft model" (semplificazione
        //  didattica; in produzione si usa un modello più piccolo).
        //  I token draft NON vengono stampati: potrebbero essere rifiutati.
        //  Durante questa fase, la KV cache viene sporca: dopo il draft
        //  conterrà i token draft, che dobbiamo annullare prima della verifica.
        // =====================================================================
        size_t ctx_before = context_ids.size();
        std::vector<float> current_logits = logits;

        std::vector<int> draft_tokens;
        int draft_token = next_token;   // primo token da verificare
        int draft_pos = pos;            // posizione corrente nella sequenza

        for (int i = 0; i < draft_k && draft_token != tok.eos_id; i++) {
            draft_tokens.push_back(draft_token);
            context_ids.push_back(draft_token);

            // Forward sequenziale per ottenere i logits del prossimo token
            forward(model, draft_token, draft_pos, logits, false);
            draft_pos++;

            // Greedy: scegli sempre il token con probabilità massima
            // (veloce, niente sampling stochastico)
            apply_repetition_penalty(logits, context_ids, params.rep_penalty);
            draft_token = sample_argmax(logits);
        }

        // =====================================================================
        //  FASE 2 — ROLLBACK: annulla le modifiche del draft
        //
        //  I draft token non sono ancora "ufficiali". Dobbiamo ripristinare
        //  lo stato come se il draft non fosse mai successo:
        //  - Tronca la KV cache alla posizione prima del draft (pos)
        //  - Ripristina il contesto (rimuove i draft token)
        //
        //  Nota: i dati oltre pos rimangono nell'array ma non vengono
        //  letti perché n_cached indica dove finiscono i dati validi.
        // =====================================================================
        model.kv_cache.n_cached = pos;
        context_ids.resize(ctx_before);

        // =====================================================================
        //  FASE 3 — VERIFY: verifica batch con forward_verify()
        //
        //  Questa è la chiave del guadagno di velocità: invece di fare
        //  K forward pass sequenziali, ne facciamo UNO solo in batch.
        //  Il modello processa tutti i draft token insieme, riutilizzando
        //  i pesi in cache (matvec_quant_batch) → molto più veloce.
        //
        //  forward_verify restituisce all_logits[i] = logits dopo aver
        //  processato il token i del batch. Questo ci permette di
        //  verificare ogni draft in modo indipendente.
        // =====================================================================
        std::vector<std::vector<float>> all_logits;
        if (!draft_tokens.empty()) {
            forward_verify(model, draft_tokens, pos, all_logits);
        }

        // =====================================================================
        //  FASE 4 — VERIFICA SEQUENZIALE: confronta draft vs target
        //
        //  Anche se abbiamo fatto un forward batch, la verifica deve essere
        //  sequenziale: per verificare draft[i] usiamo i logits predetti
        //  DOPO aver processato draft[i-1]. Questo perché il modello deve
        //  vedere i token precedenti per predire il prossimo.
        //
        //  Esempio con draft = [d1, d2, d3]:
        //    - Verifica d1: sample(logits_iniziali) == d1 ?
        //    - Verifica d2: sample(all_logits[0]) == d2 ?
        //    - Verifica d3: sample(all_logits[1]) == d3 ?
        //
        //  Se tutti passano: abbiamo generato K token con UN solo forward!
        //  Se d2 fallisce: accettiamo d1, usiamo il token verificato al posto
        //  di d2, e rigeneriamo da lì.
        // =====================================================================
        int accepted = 0;
        std::vector<float> verify_logits = current_logits;
        bool rejected = false;

        for (int i = 0; i < (int)draft_tokens.size(); i++) {
            // Campiona il token che IL MODELLO VORREBBE generare
            // (non il draft) per confrontarlo col draft
            apply_repetition_penalty(verify_logits, context_ids, params.rep_penalty);
            int verified_token = params.greedy
                ? sample_argmax(verify_logits)
                : sample_topk_topp(verify_logits, params.top_k, params.top_p, params.temperature);

            if (verified_token == draft_tokens[i]) {
                // ── DRAFT ACCETTATO ──────────────────────────────────
                // Il modello concorda col draft: questo token è valido!
                accepted++;
                context_ids.push_back(draft_tokens[i]);
                std::cout << sanitize_output(tokenizer_decode(tok, {draft_tokens[i]}));
                std::cout.flush();
                generated++;

                // Passa ai logits successivi per verificare il prossimo draft
                if (i < (int)all_logits.size()) {
                    verify_logits = all_logits[i];
                }
                if (generated >= max_tokens) break;
            } else {
                // ── DRAFT RIFIUTATO ──────────────────────────────────
                // Il modello NON concorda: il draft era sbagliato.
                // Usiamo il token "corretto" (quello campionato) e
                // tronchiamo la KV cache ai soli token accettati.
                context_ids.push_back(verified_token);
                std::cout << sanitize_output(tokenizer_decode(tok, {verified_token}));
                std::cout.flush();
                generated++;

                // Tronca la cache: rimuove i draft rifiutati
                model.kv_cache.n_cached = pos + accepted;

                // Forward dal token verificato per ottenere i logits
                // della prossima iterazione
                pos = pos + accepted;
                forward(model, verified_token, pos, logits, false);
                pos++;

                next_token = params.greedy
                    ? sample_argmax(logits)
                    : sample_topk_topp(logits, params.top_k, params.top_p, params.temperature);
                rejected = true;
                break;
            }
        }

        // =====================================================================
        //  FASE 5 — TUTTI I DRAFT ACCETTATI
        //
        //  Se arriviamo qui senza rejection, significa che tutti i K draft
        //  erano corretti. Abbiamo generato K token con un solo forward!
        //  Ora dobbiamo campionare il prossimo token dai logits dell'ultimo
        //  draft per continuare il loop.
        // =====================================================================
        if (!rejected) {
            if (accepted == (int)draft_tokens.size() && accepted > 0) {
                // Tronca la cache al numero effettivo di token accettati
                model.kv_cache.n_cached = pos + accepted;
                pos = pos + accepted;

                // Campiona il prossimo token dai logits dell'ultimo draft
                verify_logits = all_logits.back();
                apply_repetition_penalty(verify_logits, context_ids, params.rep_penalty);
                next_token = params.greedy
                    ? sample_argmax(verify_logits)
                    : sample_topk_topp(verify_logits, params.top_k, params.top_p, params.temperature);
            } else if (accepted == 0) {
                // Nessun draft (draft_token era EOS all'inizio)
                next_token = tok.eos_id;
            }
        }
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
    int draft_k = 4;
    bool speculative_mode = false;
    srand(static_cast<unsigned>(time(nullptr)));

    // Modalità chat attiva automaticamente se il modello ha un template.
    bool chat_mode = (model.config.arch == ArchType::LLAMA &&
                      tok.has_chat_template);

    // Configura linenoise
    linenoise::SetMultiLine(false);
    linenoise::SetHistoryMaxLen(100);
    linenoise::LoadHistory(HISTORY_FILE);

    std::string model_name = model.config.arch == ArchType::LLAMA
        ? "TinyLlama 1.1B" : "GPT-2 small";

    std::cout << "╔═══════════════════════════════════════╗\n";
    std::cout << "║   EIE-LLM — Educational Infer Engine  ║\n";
    std::cout << "║   " << std::left << std::setw(35)
              << (model_name + "  •  CPU only  •  C++17")
              << "║\n";
    std::cout << "╚═══════════════════════════════════════╝\n";
    print_help(chat_mode, speculative_mode, draft_k);

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
                print_help(chat_mode, speculative_mode, draft_k);
            }
            else if (line == ":reset") {
                model_init_kvcache(model);
                std::cout << "  KV cache azzerata\n\n";
            }
            else if (line == ":greedy") {
                params.greedy = true;
                std::cout << "  Sampling: greedy\n\n";
            }
            else if (line == ":sample") {
                params.greedy = false;
                std::cout << "  Sampling: top-k + top-p\n\n";
            }
            else if (line == ":chat") {
                // Abilita modalità chat: il prompt viene avvolto nel
                // template prima della tokenizzazione.
                if (tok.has_chat_template) {
                    chat_mode = true;
                    std::cout << "  Modalità: chat (template attivo)\n\n";
                } else {
                    std::cout << "  Il modello non ha un chat template.\n\n";
                }
            }
            else if (line == ":raw") {
                // Disabilita modalità chat: il prompt viene passato
                // al modello invariato (utile per completamento libero).
                chat_mode = false;
                std::cout << "  Modalità: raw (prompt diretto)\n\n";
            }
            else if (line == ":params") {
                print_params(params, max_tokens, speculative_mode, draft_k);
            }
            else if (line == ":speculative") {
                speculative_mode = !speculative_mode;
                std::cout << "  Speculative decoding: "
                          << (speculative_mode ? "on" : "off")
                          << (speculative_mode ? " (draft=" + std::to_string(draft_k) + ")" : "")
                          << "\n\n";
            }
            else if (line.substr(0, 7) == ":draft ") {
                try {
                    draft_k = std::stoi(line.substr(7));
                    draft_k = std::max(1, std::min(draft_k, 16));
                    std::cout << "  Draft tokens: " << draft_k << "\n\n";
                } catch (...) {
                    std::cout << "  Uso: :draft <intero>\n\n";
                }
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
            else if (line == ":cacheinfo") {
                shell_prefix_cache.print_stats();
            }
            else if (line == ":cacheclear") {
                shell_prefix_cache.clear();
                std::cout << "  Prefix cache svuotata\n\n";
            }
            else if (line.substr(0, 9) == ":inspect ") {
                std::string inspect_prompt = line.substr(9);
                if (inspect_prompt.empty()) {
                    std::cout << "  Uso: :inspect <testo>\n\n";
                } else {
                    std::string json = inspect_attention(model, tok, inspect_prompt, 20);
                    std::cout << "\n" << json << "\n\n";
                }
            }
            else {
                std::cout << "  Comando sconosciuto. "
                          << "Usa :help per la lista.\n\n";
            }
        } else {
            if (chat_mode) {
                std::string formatted = apply_chat_template(tok, line);
                std::cout << "\nAssistente: ";
                std::cout.flush();
                if (speculative_mode)
                    generate_speculative(model, tok, formatted, params, max_tokens,
                                         draft_k, /*print_prompt=*/false);
                else
                    generate(model, tok, formatted, params, max_tokens,
                             /*print_prompt=*/false, &shell_prefix_cache);
            } else {
                if (speculative_mode)
                    generate_speculative(model, tok, line, params, max_tokens,
                                         draft_k, /*print_prompt=*/true);
                else
                    generate(model, tok, line, params, max_tokens,
                             /*print_prompt=*/true, &shell_prefix_cache);
            }
        }
    }

    // Salva la history su file prima di uscire
    linenoise::SaveHistory(HISTORY_FILE);
}
