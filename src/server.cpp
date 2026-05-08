#include "server.hpp"
#include "model.hpp"
#include "tokenizer.hpp"
#include "ops.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <mutex>
#include "httplib.h"

// ─────────────────────────────────────────────
//  Mutex per serializzare le richieste
//
//  Il modello e la KV cache NON sono thread-safe.
//  Un mutex garantisce che una sola richiesta
//  alla volta usi il modello.
//  In un engine di produzione si userebbe un
//  pool di modelli o una request queue.
// ─────────────────────────────────────────────
static std::mutex model_mutex;

// ─────────────────────────────────────────────
//  Parser JSON minimale
//
//  Non usiamo una libreria JSON per mantenere
//  zero dipendenze aggiuntive. Estraiamo i
//  campi che ci servono con semplice ricerca
//  di stringhe — funziona per il nostro formato
//  di input che è semplice e ben definito.
//
//  In produzione si userebbe nlohmann/json
//  o simili.
// ─────────────────────────────────────────────

// Estrae il valore stringa di una chiave JSON
// Esempio: {"prompt":"hello"} → "hello"
static std::string json_get_string(const std::string& json,
                                   const std::string& key) {
    // Cerca "key":"
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    // Trova il ':' dopo la chiave
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos++;

    // Salta spazi
    while (pos < json.size() && json[pos] == ' ') pos++;

    // Leggi il valore stringa tra virgolette
    if (json[pos] != '"') return "";
    pos++;

    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        // Gestisci escape sequences (\n, \t, \", \\)
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            switch (json[pos]) {
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                default:   result += json[pos]; break;
            }
        } else {
            result += json[pos];
        }
        pos++;
    }
    return result;
}

// Estrae il valore numerico (float) di una chiave JSON
// Esempio: {"temperature":0.8} → 0.8
static float json_get_float(const std::string& json,
                             const std::string& key,
                             float default_val) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos = json.find(':', pos);
    if (pos == std::string::npos) return default_val;
    pos++;

    while (pos < json.size() && json[pos] == ' ') pos++;

    // Leggi il numero fino a virgola, } o spazio
    std::string num;
    while (pos < json.size() &&
           json[pos] != ',' &&
           json[pos] != '}' &&
           json[pos] != ' ') {
        num += json[pos++];
    }

    try { return std::stof(num); }
    catch (...) { return default_val; }
}

// Estrae il valore intero di una chiave JSON
static int json_get_int(const std::string& json,
                        const std::string& key,
                        int default_val) {
    return static_cast<int>(
        json_get_float(json, key, static_cast<float>(default_val))
    );
}

// Estrae un valore booleano da una chiave JSON.
// Riconosce "true" e "false" (senza virgolette), es: {"chat":true}
static bool json_get_bool(const std::string& json,
                          const std::string& key,
                          bool default_val) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return default_val;

    pos = json.find(':', pos);
    if (pos == std::string::npos) return default_val;
    pos++;

    while (pos < json.size() && json[pos] == ' ') pos++;

    if (json.compare(pos, 4, "true")  == 0) return true;
    if (json.compare(pos, 5, "false") == 0) return false;
    return default_val;
}

// Estrae il contenuto dell'ultimo messaggio con role "user"
// dall'array "messages" in formato OpenAI chat.
//
// Formato atteso (minimo):
//   {"messages":[{"role":"user","content":"Ciao"}]}
//
// Cerca l'ultima occorrenza di "role":"user" e legge il "content"
// successivo. Robusto abbastanza per i casi tipici; non è un parser
// JSON completo — usare una libreria reale in produzione.
static std::string json_get_last_user_message(const std::string& json) {
    std::string result;

    size_t search_from = 0;
    while (true) {
        // Cerca la prossima occorrenza di "role"
        size_t role_pos = json.find("\"role\"", search_from);
        if (role_pos == std::string::npos) break;

        // Verifica che il valore sia "user"
        size_t colon = json.find(':', role_pos);
        if (colon == std::string::npos) break;
        size_t val_start = json.find('"', colon + 1);
        if (val_start == std::string::npos) break;
        val_start++;
        size_t val_end = json.find('"', val_start);
        if (val_end == std::string::npos) break;

        std::string role = json.substr(val_start, val_end - val_start);

        if (role == "user") {
            // Leggi il "content" successivo (nella stessa entry)
            size_t content_pos = json.find("\"content\"", val_end);
            // Ma non oltre il prossimo "role" o la fine dell'oggetto
            size_t next_role = json.find("\"role\"", val_end + 1);
            if (content_pos != std::string::npos &&
                (next_role == std::string::npos || content_pos < next_role)) {
                result = json_get_string(json.substr(content_pos - 1), "content");
            }
        }

        search_from = val_end + 1;
    }
    return result;
}

// ─────────────────────────────────────────────
//  Escape caratteri speciali per JSON
//
//  Il testo generato può contenere caratteri
//  che romperebbero il JSON se inseriti raw:
//  virgolette, backslash, newline, ecc.
// ─────────────────────────────────────────────
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;      break;
        }
    }
    return out;
}

// ─────────────────────────────────────────────
//  Generazione testo per il server
//
//  Versione della funzione generate() della shell
//  adattata per il server:
//  - ritorna il testo invece di stamparlo
//  - usa il mutex per serializzare l'accesso
//  - resetta la KV cache ad ogni chiamata
// ─────────────────────────────────────────────
static std::string generate_text(Model& model,
                                  const Tokenizer& tok,
                                  const std::string& prompt,
                                  int max_tokens,
                                  const SamplingParams& params) {
    std::lock_guard<std::mutex> lock(model_mutex);
    model_init_kvcache(model);

    auto input_ids = tokenizer_encode(tok, prompt);
    if (input_ids.empty()) return "";

    std::vector<int> context_ids = input_ids;
    std::vector<float> logits;

    int pos = 0;
    for (int id : input_ids) {
        forward(model, id, pos, logits);
        pos++;
    }
    model.kv_cache.n_cached = pos;

    std::string output;

    int next_token;
    // Campiona il primo token
    apply_repetition_penalty(logits, context_ids, params.rep_penalty);
    next_token = params.greedy
            ? sample_argmax(logits)
            : sample_topk_topp(logits, params.top_k, params.top_p, params.temperature);

    for (int i = 0; i < max_tokens; i++) {
        if (next_token == tok.eos_id) break;

        output += tokenizer_decode(tok, {next_token});
        context_ids.push_back(next_token);

        forward(model, next_token, pos, logits);
        pos++;

        apply_repetition_penalty(logits, context_ids, params.rep_penalty);
        next_token = params.greedy
            ? sample_argmax(logits)
            : sample_topk_topp(logits, params.top_k, params.top_p, params.temperature);
    }
    return output;
}

// ─────────────────────────────────────────────
//  Avvio del server
//
//  Registriamo due handler:
//
//  GET /health
//    Risponde sempre con status ok.
//    Utile per verificare che il server sia
//    raggiungibile prima di inviare richieste.
//
//  POST /v1/completions
//    Riceve un JSON con prompt e parametri,
//    genera il testo e risponde in formato
//    compatibile con l'API OpenAI.
// ─────────────────────────────────────────────
void server_run(Model& model, const Tokenizer& tok, int port) {
    httplib::Server svr;

    // ── GET /health ───────────────────────────
    svr.Get("/health", [](const httplib::Request&,
                           httplib::Response& res) {
        res.set_content(
            "{\"status\":\"ok\",\"model\":\"gpt2\"}",
            "application/json"
        );
    });

    // ── Logica comune per gestire una richiesta ──
    // Estratta in lambda per riutilizzarla in entrambi gli endpoint.
    // Riceve il prompt già formattato (con o senza chat template)
    // e restituisce il JSON di risposta.
    auto handle_request = [&model, &tok](
            const std::string& prompt,
            int max_tokens,
            const SamplingParams& params,
            const std::string& endpoint_label) -> std::string {

        std::cout << "[REQ:" << endpoint_label << "] prompt=\""
                  << prompt.substr(0, 40)
                  << (prompt.size() > 40 ? "..." : "")
                  << "\" max_tokens=" << max_tokens
                  << " temp=" << params.temperature << "\n";

        int prompt_tokens = static_cast<int>(
            tokenizer_encode(tok, prompt).size());

        std::string generated = generate_text(
            model, tok, prompt, max_tokens, params);

        int completion_tokens = static_cast<int>(
            tokenizer_encode(tok, generated).size());

        std::cout << "[RES] generati " << completion_tokens << " token\n";

        std::ostringstream j;
        j << "{"
          << "\"object\":\"text_completion\","
          << "\"choices\":[{"
          <<   "\"text\":\"" << json_escape(generated) << "\","
          <<   "\"index\":0,"
          <<   "\"finish_reason\":\"stop\""
          << "}],"
          << "\"usage\":{"
          <<   "\"prompt_tokens\":"     << prompt_tokens     << ","
          <<   "\"completion_tokens\":" << completion_tokens << ","
          <<   "\"total_tokens\":"      << (prompt_tokens + completion_tokens)
          << "}"
          << "}";
        return j.str();
    };

    // ── POST /v1/completions ──────────────────
    //
    //  Endpoint per il completamento di testo grezzo.
    //  Campo opzionale "chat": true — se presente e il modello
    //  ha un chat template, avvolge il prompt nel template
    //  prima della generazione.
    //
    //  Esempio con chat template:
    //    {"prompt":"Qual è la capitale della Francia?","chat":true}
    //  Esempio grezzo (default):
    //    {"prompt":"The capital of France is"}
    svr.Post("/v1/completions",
        [&model, &tok, &handle_request](const httplib::Request& req,
                                        httplib::Response& res) {

        std::string prompt     = json_get_string(req.body, "prompt");
        int         max_tokens = json_get_int   (req.body, "max_tokens", 50);
        max_tokens = std::min(max_tokens, 500);

        SamplingParams params;
        params.temperature = json_get_float(req.body, "temperature",        1.0f);
        params.top_p       = json_get_float(req.body, "top_p",              0.9f);
        params.top_k       = json_get_int  (req.body, "top_k",              40);
        params.rep_penalty = json_get_float(req.body, "repetition_penalty", 1.1f);
        params.greedy      = (params.temperature <= 0.0f);

        if (prompt.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"prompt mancante o vuoto\"}",
                            "application/json");
            return;
        }

        // Se "chat":true e il modello ha un template, applica la formattazione.
        bool use_chat = json_get_bool(req.body, "chat", false);
        if (use_chat && tok.has_chat_template)
            prompt = apply_chat_template(tok, prompt);

        res.set_content(
            handle_request(prompt, max_tokens, params, "completions"),
            "application/json");
    });

    // ── POST /v1/chat/completions ─────────────
    //
    //  Endpoint compatibile con l'API OpenAI chat.
    //  Accetta un array "messages" con messaggi role/content.
    //  Estrae l'ultimo messaggio con role "user" e lo avvolge
    //  nel chat template del modello.
    //
    //  Esempio di richiesta:
    //    {
    //      "messages": [
    //        {"role": "system", "content": "Sei un assistente utile."},
    //        {"role": "user",   "content": "Qual è la capitale dell'Italia?"}
    //      ],
    //      "max_tokens": 100
    //    }
    //
    //  Nota: il parser JSON è minimalista — gestisce l'uso tipico
    //  ma non è un parser completo. Usare una libreria in produzione.
    svr.Post("/v1/chat/completions",
        [&model, &tok, &handle_request](const httplib::Request& req,
                                        httplib::Response& res) {

        std::string user_msg = json_get_last_user_message(req.body);
        int max_tokens = json_get_int(req.body, "max_tokens", 100);
        max_tokens = std::min(max_tokens, 500);

        SamplingParams params;
        params.temperature = json_get_float(req.body, "temperature",        1.0f);
        params.top_p       = json_get_float(req.body, "top_p",              0.9f);
        params.top_k       = json_get_int  (req.body, "top_k",              40);
        params.rep_penalty = json_get_float(req.body, "repetition_penalty", 1.1f);
        params.greedy      = (params.temperature <= 0.0f);

        if (user_msg.empty()) {
            res.status = 400;
            res.set_content(
                "{\"error\":\"nessun messaggio utente trovato in messages\"}",
                "application/json");
            return;
        }

        // Applica il chat template se disponibile, altrimenti usa il testo grezzo.
        std::string prompt = tok.has_chat_template
            ? apply_chat_template(tok, user_msg)
            : user_msg;

        res.set_content(
            handle_request(prompt, max_tokens, params, "chat/completions"),
            "application/json");
    });
    
    // ── Avvio ─────────────────────────────────
    std::cout << "\n╔═══════════════════════════════════════╗\n";
    std::cout << "║   EIE-LLM Server                      ║\n";
    std::cout << "║   http://localhost:" << port
              << "               ║\n";
    std::cout << "╠═══════════════════════════════════════╣\n";
    std::cout << "║   GET  /health                        ║\n";
    std::cout << "║   POST /v1/completions                ║\n";
    std::cout << "║   POST /v1/chat/completions           ║\n";
    std::cout << "╠═══════════════════════════════════════╣\n";
    std::cout << "║   Chat template: "
              << std::left << std::setw(21)
              << (tok.has_chat_template ? "attivo" : "non disponibile")
              << "║\n";
    std::cout << "╚═══════════════════════════════════════╝\n\n";
    std::cout << "Premi Ctrl+C per fermare il server\n\n";

    svr.listen("0.0.0.0", port);
}