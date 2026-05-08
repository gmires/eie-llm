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

    // ── POST /v1/completions ──────────────────
svr.Post("/v1/completions",
        [&model, &tok](const httplib::Request& req,
                       httplib::Response& res) {

        std::string prompt     = json_get_string(req.body, "prompt");
        int         max_tokens = json_get_int   (req.body, "max_tokens",  50);
        max_tokens = std::min(max_tokens, 500);

        SamplingParams params;
        params.temperature = json_get_float(req.body, "temperature",        1.0f);
        params.top_p       = json_get_float(req.body, "top_p",              0.9f);
        params.top_k       = json_get_int  (req.body, "top_k",              40);        
        params.rep_penalty = json_get_float(req.body, "repetition_penalty", 1.1f);
        params.greedy      = (params.temperature <= 0.0f);

        if (prompt.empty()) {
            res.status = 400;
            res.set_content(
                "{\"error\":\"prompt mancante o vuoto\"}",
                "application/json");
            return;
        }

        std::cout << "[REQ] prompt=\""
                  << prompt.substr(0, 40)
                  << (prompt.size() > 40 ? "..." : "")
                  << "\" max_tokens=" << max_tokens
                  << " temp=" << params.temperature
                  << " top_p=" << params.top_p
                  << " top_k=" << params.top_k
                  << " penalty=" << params.rep_penalty << "\n";

        int prompt_tokens = static_cast<int>(
            tokenizer_encode(tok, prompt).size());

        std::string generated = generate_text(
            model, tok, prompt, max_tokens, params);

        int completion_tokens = static_cast<int>(
            tokenizer_encode(tok, generated).size());

        std::cout << "[RES] generati "
                  << completion_tokens << " token\n";

        std::ostringstream json;
        json << "{"
             << "\"object\":\"text_completion\","
             << "\"model\":\"gpt2\","
             << "\"choices\":[{"
             <<   "\"text\":\"" << json_escape(generated) << "\","
             <<   "\"index\":0,"
             <<   "\"finish_reason\":\"stop\""
             << "}],"
             << "\"usage\":{"
             <<   "\"prompt_tokens\":"     << prompt_tokens     << ","
             <<   "\"completion_tokens\":" << completion_tokens << ","
             <<   "\"total_tokens\":"
             <<   (prompt_tokens + completion_tokens)
             << "}"
             << "}";

        res.set_content(json.str(), "application/json");
    });
    
    // ── Avvio ─────────────────────────────────
    std::cout << "\n╔═══════════════════════════════════════╗\n";
    std::cout << "║   EIE-LLM Server                      ║\n";
    std::cout << "║   http://localhost:" << port
              << "               ║\n";
    std::cout << "╠═══════════════════════════════════════╣\n";
    std::cout << "║   GET  /health                        ║\n";
    std::cout << "║   POST /v1/completions                ║\n";
    std::cout << "╚═══════════════════════════════════════╝\n\n";
    std::cout << "Premi Ctrl+C per fermare il server\n\n";

    svr.listen("0.0.0.0", port);
}