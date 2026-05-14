#include "server.hpp"
#include "model.hpp"
#include "tokenizer.hpp"
#include "ops.hpp"
#include "prefix_cache.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <mutex>
#include <functional>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <thread>
#include <chrono>
#include <iomanip>
#include "httplib.h"

// ═════════════════════════════════════════════════════════════════════════════
//  FASE 22 — CONTINUOUS BATCHING (versione semplificata)
// ═════════════════════════════════════════════════════════════════════════════
//
//  In questa fase separiamo COMPLETAMENTE il thread HTTP dal thread di
//  inferenza. Il thread HTTP accetta richieste, le mette in coda e
//  risponde solo quando il lavoro è finito. Il thread di inferenza
//  preleva richieste dalla coda e le esegue sequenzialmente.
//
//  ARCHITETTURA:
//    ┌─────────────┐     push()      ┌─────────────────┐
//    │ Thread HTTP │ ──────────────→ │   RequestQueue  │
//    │  (httplib)  │                 │  (thread-safe)  │
//    └─────────────┘                 └─────────────────┘
//           ↑                              │ pop()
//           │                              ↓
//    attesa su cv                   ┌─────────────────┐
//    (req->cv.wait)                 │ Thread Inferenza│
//                                   │ (inference_thread│
//                                   └─────────────────┘
//
//  VANTAGGI:
//  1. Il thread HTTP non si blocca durante la generazione → può accettare
//     nuove connessioni mentre il modello sta generando token.
//  2. Time-To-First-Byte (TTFB) migliorato per il throughput del server.
//  3. Struttura che si avvicina ai veri engine di produzione (vLLM, TGI).
//
//  LIMITAZIONE (didattica):
//  Avendo un solo modello con una sola KV cache, processiamo le richieste
//  UNA ALLA VOLTA. In un engine reale, il thread di inferenza farebbe
//  UN unico forward pass batch con il token corrente di TUTTE le richieste
//  attive, massimizzando il riutilizzo dei pesi in GPU/CPU cache.
// ═════════════════════════════════════════════════════════════════════════════


// ─────────────────────────────────────────────
//  Stato di una richiesta nel server
//
//  Ogni richiesta HTTP viene convertita in un
//  oggetto ServerRequest che viene messo in coda
//  e processato dal thread di inferenza.
//  Questo disaccoppia la gestione HTTP (veloce,
//  asincrona) dalla computazione del modello
//  (lenta, sequenziale).
//
//  Il thread HTTP aspetta su 'cv' finché
//  'done' non diventa true.
// ─────────────────────────────────────────────
struct ServerRequest {
    // ── Input dal client ──────────────────────
    std::string prompt;          // testo del prompt
    int max_tokens;              // quanti token generare al massimo
    SamplingParams params;       // temperatura, top_p, top_k, ecc.
    std::string endpoint_label;  // "completions" o "chat/completions"

    // ── Stato della generazione (scritto dal thread di inferenza) ──
    std::vector<int> input_ids;      // token del prompt
    std::vector<int> context_ids;    // prompt + token generati (per repetition penalty)
    std::vector<float> logits;       // logits dell'ultimo forward
    std::string output_text;         // testo accumulato
    int pos = 0;                     // posizione corrente nella sequenza
    int generated = 0;               // contatore token generati
    int prompt_tokens = 0;           // numero di token del prompt
    int completion_tokens = 0;       // numero di token generati

    // ── Flags di completamento ────────────────
    bool done = false;               // true quando la generazione è finita
    bool error = false;              // true se c'è stato un errore
    std::string error_msg;           // messaggio di errore (se error=true)

    // ── Sincronizzazione thread ───────────────
    // Il thread HTTP acquisisce 'mtx', controlla 'done',
// chiama cv.wait(). Il thread di inferenza, a fine lavoro,
    // acquisisce 'mtx', setta 'done=true', chiama cv.notify_one().
    std::mutex mtx;
    std::condition_variable cv;
};

// ─────────────────────────────────────────────
//  Coda di richieste — thread-safe
//
//  Il thread HTTP inserisce richieste con push().
//  Il thread di inferenza le preleva con pop().
//  Se la coda è vuota, pop() blocca il thread
//  finché non arriva una nuova richiesta o
//  finché non viene segnalato lo shutdown.
// ─────────────────────────────────────────────
class RequestQueue {
public:
    void push(std::shared_ptr<ServerRequest> req) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push(req);
        cv.notify_one();  // sveglia il thread di inferenza
    }

    std::shared_ptr<ServerRequest> pop() {
        std::unique_lock<std::mutex> lock(mtx);
        // Attendi finché la coda non è vuota O finché non arriva shutdown
        cv.wait(lock, [this] { return !queue.empty() || shutdown; });
        if (shutdown && queue.empty()) return nullptr;
        auto req = queue.front();
        queue.pop();
        return req;
    }

    bool empty() {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.empty();
    }

    void signal_shutdown() {
        std::lock_guard<std::mutex> lock(mtx);
        shutdown = true;
        cv.notify_all();  // sveglia tutti i thread in attesa
    }

private:
    std::queue<std::shared_ptr<ServerRequest>> queue;
    std::mutex mtx;
    std::condition_variable cv;
    bool shutdown = false;
};

// ─────────────────────────────────────────────
//  Prefix Cache globale per il server
//
//  Condivisa tra tutte le richieste. Ogni richiesta
//  che arriva con un prompt già visto può riusare
//  la KV cache senza ricalcolarla.
//
//  Questo è particolarmente utile in un server
//  dove molte richieste condividono lo stesso
//  system prompt o prefisso.
// ─────────────────────────────────────────────
static PrefixCache prefix_cache;


// ═════════════════════════════════════════════════════════════════════════════
//  PARSER JSON MINIMALE
// ═════════════════════════════════════════════════════════════════════════════
//
//  Non usiamo una libreria JSON per mantenere zero dipendenze aggiuntive.
//  Estraiamo i campi che ci servono con semplice ricerca di stringhe —
//  funziona per il nostro formato di input che è semplice e ben definito.
//
//  In produzione si userebbe nlohmann/json o simili.
// ═════════════════════════════════════════════════════════════════════════════

// Estrae il valore stringa di una chiave JSON
// Esempio: {"prompt":"hello"} → "hello"
static std::string json_get_string(const std::string& json,
                                   const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

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

// Estrae il valore intero di una chiave JSON (delega a json_get_float)
static int json_get_int(const std::string& json,
                        const std::string& key,
                        int default_val) {
    return static_cast<int>(
        json_get_float(json, key, static_cast<float>(default_val))
    );
}

// Estrae un valore booleano da una chiave JSON.
// Riconosce "true" e "false" (senza virgolette).
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
// Cerca l'ultima occorrenza di "role":"user" e legge il "content"
// successivo. Robusto abbastanza per i casi tipici; non è un parser
// JSON completo — usare una libreria reale in produzione.
static std::string json_get_last_user_message(const std::string& json) {
    std::string result;
    size_t search_from = 0;

    while (true) {
        size_t role_pos = json.find("\"role\"", search_from);
        if (role_pos == std::string::npos) break;

        size_t colon = json.find(':', role_pos);
        if (colon == std::string::npos) break;
        size_t val_start = json.find('"', colon + 1);
        if (val_start == std::string::npos) break;
        val_start++;
        size_t val_end = json.find('"', val_start);
        if (val_end == std::string::npos) break;

        std::string role = json.substr(val_start, val_end - val_start);

        if (role == "user") {
            size_t content_pos = json.find("\"content\"", val_end);
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

// Estrae tutti i messaggi dall'array "messages" in formato OpenAI chat.
// Restituisce un vettore di coppie {role, content} nell'ordine originale.
static std::vector<std::pair<std::string, std::string>>
json_get_chat_messages(const std::string& json) {
    std::vector<std::pair<std::string, std::string>> result;

    // Cerca l'inizio dell'array "messages"
    size_t messages_pos = json.find("\"messages\"");
    if (messages_pos == std::string::npos) return result;

    size_t arr_start = json.find('[', messages_pos);
    if (arr_start == std::string::npos) return result;

    size_t arr_end = json.find(']', arr_start);
    if (arr_end == std::string::npos) return result;

    std::string arr = json.substr(arr_start + 1, arr_end - arr_start - 1);

    // Per ogni oggetto nell'array, estrai role e content
    size_t obj_start = 0;
    while ((obj_start = arr.find('{', obj_start)) != std::string::npos) {
        size_t obj_end = arr.find('}', obj_start);
        if (obj_end == std::string::npos) break;

        std::string obj = arr.substr(obj_start + 1, obj_end - obj_start - 1);

        std::string role;
        std::string content;

        // Estrai role
        size_t rpos = obj.find("\"role\"");
        if (rpos != std::string::npos) {
            size_t rc = obj.find(':', rpos);
            if (rc != std::string::npos) {
                rc++;
                while (rc < obj.size() && (obj[rc] == ' ' || obj[rc] == '"')) rc++;
                size_t re = rc;
                while (re < obj.size() && obj[re] != '"') re++;
                role = obj.substr(rc, re - rc);
            }
        }

        // Estrai content
        size_t cpos = obj.find("\"content\"");
        if (cpos != std::string::npos) {
            size_t cc = obj.find(':', cpos);
            if (cc != std::string::npos) {
                cc++;
                while (cc < obj.size() && (obj[cc] == ' ' || obj[cc] == '"')) cc++;
                // Leggi content gestendo escape
                while (cc < obj.size() && obj[cc] != '"') {
                    if (obj[cc] == '\\' && cc + 1 < obj.size()) {
                        cc++;
                        switch (obj[cc]) {
                            case 'n': content += '\n'; break;
                            case 't': content += '\t'; break;
                            case '"': content += '"'; break;
                            case '\\': content += '\\'; break;
                            default: content += obj[cc]; break;
                        }
                    } else {
                        content += obj[cc];
                    }
                    cc++;
                }
            }
        }

        if (!role.empty()) {
            result.push_back({role, content});
        }

        obj_start = obj_end + 1;
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


// ═════════════════════════════════════════════════════════════════════════════
//  HELPERS PER SERVER-SENT EVENTS (SSE)
// ═════════════════════════════════════════════════════════════════════════════
//
//  Formato OpenAI-compatibile per lo streaming:
//    data: {"choices":[{"delta":{"content":"..."}}]}
//
//  Ogni evento termina con \n\n (richiesto dal protocollo SSE).
// ═════════════════════════════════════════════════════════════════════════════

static std::string sse_text_chunk(const std::string& text) {
    return "data: {\"choices\":[{\"text\":\"" + json_escape(text) +
           "\",\"index\":0,\"finish_reason\":null}]}\n\n";
}

static std::string sse_chat_chunk(const std::string& text) {
    return "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" +
           json_escape(text) + "\"},\"finish_reason\":null}]}\n\n";
}

static std::string sse_text_done() {
    return "data: {\"choices\":[{\"text\":\"\",\"index\":0,"
           "\"finish_reason\":\"stop\"}]}\n\n"
           "data: [DONE]\n\n";
}

static std::string sse_chat_done() {
    return "data: {\"choices\":[{\"index\":0,\"delta\":{},"
           "\"finish_reason\":\"stop\"}]}\n\n"
           "data: [DONE]\n\n";
}


// ═════════════════════════════════════════════════════════════════════════════
//  FUNZIONI DI GENERAZIONE
// ═════════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────
//  Generazione per una richiesta in coda
//
//  Questa funzione è CHIAMATA SOLO dal thread
//  di inferenza. Riceve una ServerRequest già
//  popolata con prompt, max_tokens e params.
//
//  Esegue il prefill (con prefix cache), il
//  sampling e la generazione autoregressiva,
//  scrivendo i risultati direttamente nella
//  struct della richiesta.
//
//  Al termine, setta req.done = true e notifica
//  il thread HTTP che è in attesa su req.cv.
//
//  NOTA: NON c'è alcun mutex sul modello perché
//  questa funzione viene eseguita ESCLUSIVAMENTE
//  dal thread di inferenza. Solo un'istanza di
//  questa funzione è attiva in ogni istante.
// ─────────────────────────────────────────────
static void generate_for_request(Model& model,
                                 const Tokenizer& tok,
                                 ServerRequest& req) {
    std::cout << "[GEN] inizio generazione per endpoint=" << req.endpoint_label
              << " prompt=\"" << req.prompt.substr(0, 40)
              << (req.prompt.size() > 40 ? "..." : "") << "\"\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    // ── Fase 1: Tokenizzazione del prompt ─────
    req.input_ids = tokenizer_encode(tok, req.prompt);
    req.context_ids = req.input_ids;
    req.logits.resize(model.config.n_vocab);
    req.pos = 0;
    req.prompt_tokens = static_cast<int>(req.input_ids.size());
    req.generated = 0;
    req.output_text.clear();

    // ── Fase 2: Prefill con Prefix Cache ──────
    // La prefix cache cerca se questo prompt è già stato
    // visto. Se sì, copia la KV cache salvata nel modello
    // e salta il prefill (risparmio enorme in O(n²)).
    if (!req.input_ids.empty()) {
        int cached_tokens = 0;
        bool cache_hit = prefix_cache.lookup(req.prompt, model, cached_tokens);

        if (cache_hit && cached_tokens == (int)req.input_ids.size()) {
            // Cache hit ESATTO: il prompt è identico a uno già visto.
            // La KV cache è già popolata per tutti i token.
            // Ricalcoliamo solo i logits dell'ultimo token.
            forward(model, req.input_ids.back(), cached_tokens - 1, req.logits);
            req.pos = cached_tokens;
        } else {
            // Cache miss (o hit parziale che non gestiamo):
            // facciamo il prefill completo da zero.
            model_init_kvcache(model);
            forward_prefill(model, req.input_ids, req.logits);
            req.pos = static_cast<int>(req.input_ids.size());

            // Salva nella cache per le prossime richieste
            prefix_cache.store(req.prompt, model, req.pos);
        }
    } else {
        // Prompt vuoto: inizializza cache e logits da zero
        model_init_kvcache(model);
        req.pos = 0;
    }

    auto t_prefill = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t0).count();

    // ── Fase 3: Sampling del primo token ──────
    int next_token;
    apply_repetition_penalty(req.logits, req.context_ids, req.params.rep_penalty);
    next_token = req.params.greedy
            ? sample_argmax(req.logits)
            : sample_topk_topp(req.logits, req.params.top_k,
                               req.params.top_p, req.params.temperature);

    // ── Fase 4: Loop di generazione ───────────
    for (int i = 0; i < req.max_tokens; i++) {
        if (next_token == tok.eos_id) {
            break;  // fine sequenza
        }

        std::string token_str = sanitize_output(tokenizer_decode(tok, {next_token}));
        req.output_text += token_str;
        req.context_ids.push_back(next_token);
        req.generated++;

        // Forward pass per il prossimo token
        forward(model, next_token, req.pos, req.logits);
        req.pos++;

        // Sampling per il token successivo
        apply_repetition_penalty(req.logits, req.context_ids, req.params.rep_penalty);
        next_token = req.params.greedy
            ? sample_argmax(req.logits)
            : sample_topk_topp(req.logits, req.params.top_k,
                               req.params.top_p, req.params.temperature);
    }

    // Conta i token di completamento per la risposta JSON
    req.completion_tokens = static_cast<int>(
        tokenizer_encode(tok, req.output_text).size());

    auto t_end = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t_end - t_prefill).count();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();

    std::cout << "[GEN] completamento finito, " << req.completion_tokens
              << " token generati | prefill=" << std::fixed << std::setprecision(1)
              << prefill_ms << "ms gen=" << gen_ms << "ms total=" << total_ms << "ms"
              << " (" << (req.generated / (gen_ms / 1000.0)) << " tok/s)" << std::endl;

    // ── Fase 5: Segnala completamento ─────────
    // Il thread HTTP è in attesa su req.cv: lo svegliamo.
    {
        std::lock_guard<std::mutex> lock(req.mtx);
        req.done = true;
    }
    req.cv.notify_one();
}

// ─────────────────────────────────────────────
//  Generazione in streaming (sincrona)
//
//  A differenza di generate_for_request(), questa
//  funzione è pensata per lo STREAMING.
//  Viene chiamata DIRETTAMENTE dal thread HTTP
//  (dentro il content provider di httplib) e NON
//  passa per la coda del thread di inferenza.
//
//  Il callback riceve ogni token man mano che
//  viene generato, permettendo di inviarlo al
//  client via SSE immediatamente.
//
//  Ritorna false se il callback ha segnalato
//  che il client si è disconnesso.
//
//  NOTA DIDATTICA: in un vero engine di produzione,
//  anche lo streaming passerebbe per un scheduler
//  che gestisce il modello condiviso. Qui, per
//  semplicità, lo streaming accede direttamente
//  al modello. In uno scenario reale con molte
//  richieste concorrenti, servirebbe un mutex
//  o un sistema di scheduling più sofisticato.
// ─────────────────────────────────────────────
static bool generate_streaming_for_request(
        Model& model,
        const Tokenizer& tok,
        const std::string& prompt,
        int max_tokens,
        const SamplingParams& params,
        std::function<bool(const std::string& token,
                           bool is_first,
                           bool is_last)> callback) {

    std::cout << "[GEN_STREAM] prompt=\"" << prompt.substr(0, 40)
              << (prompt.size() > 40 ? "..." : "")
              << "\" max_tokens=" << max_tokens << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();

    auto input_ids = tokenizer_encode(tok, prompt);
    std::vector<int> context_ids = input_ids;
    std::vector<float> logits;
    int pos = 0;

    std::cout << "[GEN_STREAM] prompt tokenizzato, n=" << input_ids.size() << "\n";

    // Prefill con prefix cache
    if (!input_ids.empty()) {
        int cached_tokens = 0;
        bool cache_hit = prefix_cache.lookup(prompt, model, cached_tokens);

        if (cache_hit && cached_tokens == (int)input_ids.size()) {
            forward(model, input_ids.back(), cached_tokens - 1, logits);
            pos = cached_tokens;
        } else {
            model_init_kvcache(model);
            forward_prefill(model, input_ids, logits);
            pos = static_cast<int>(input_ids.size());
            prefix_cache.store(prompt, model, pos);
        }
    } else {
        model_init_kvcache(model);
        logits.resize(model.config.n_vocab);
        pos = 0;
    }

    auto t_prefill = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill - t0).count();

    // Primo token
    int next_token;
    apply_repetition_penalty(logits, context_ids, params.rep_penalty);
    next_token = params.greedy
            ? sample_argmax(logits)
            : sample_topk_topp(logits, params.top_k, params.top_p, params.temperature);

    // Loop di generazione con callback
    bool client_ok = true;
    int generated = 0;
    for (int i = 0; i < max_tokens && client_ok; i++) {
        if (next_token == tok.eos_id) {
            std::cout << "[GEN_STREAM] EOS raggiunto dopo " << generated << " token\n";
            client_ok = callback("", false, true);
            break;
        }

        std::string token_str = sanitize_output(tokenizer_decode(tok, {next_token}));
        context_ids.push_back(next_token);
        generated++;

        client_ok = callback(token_str, i == 0, false);
        if (!client_ok) {
            std::cout << "[GEN_STREAM] client_ok=false al token " << generated << "\n";
            break;
        }

        forward(model, next_token, pos, logits);
        pos++;

        apply_repetition_penalty(logits, context_ids, params.rep_penalty);
        next_token = params.greedy
            ? sample_argmax(logits)
            : sample_topk_topp(logits, params.top_k, params.top_p, params.temperature);
    }

    // Se usciti per max_tokens (non EOS), segnala fine
    if (client_ok && next_token != tok.eos_id) {
        std::cout << "[GEN_STREAM] max_tokens raggiunto, generated=" << generated << "\n";
        callback("", false, true);
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    double gen_ms = std::chrono::duration<double, std::milli>(t_end - t_prefill).count();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();

    std::cout << "[GEN_STREAM] terminato, generated=" << generated
              << " | prefill=" << std::fixed << std::setprecision(1)
              << prefill_ms << "ms gen=" << gen_ms << "ms total=" << total_ms << "ms"
              << " (" << (generated / (gen_ms / 1000.0)) << " tok/s)" << std::endl;
    return client_ok;
}

// ─────────────────────────────────────────────
//  Costruzione della risposta JSON
//
//  Formato compatibile con l'API OpenAI:
//  {
//    "object": "text_completion",
//    "choices": [{"text": "...", "index": 0, "finish_reason": "stop"}],
//    "usage": {"prompt_tokens": N, "completion_tokens": M, "total_tokens": N+M}
//  }
// ─────────────────────────────────────────────
static std::string build_response_json(const ServerRequest& req) {
    std::ostringstream j;
    j << "{"
      << "\"object\":\"text_completion\","
      << "\"choices\":[{"
      <<   "\"text\":\"" << json_escape(req.output_text) << "\","
      <<   "\"index\":0,"
      <<   "\"finish_reason\":\"stop\""
      << "}],"
      << "\"usage\":{"
      <<   "\"prompt_tokens\":"     << req.prompt_tokens     << ","
      <<   "\"completion_tokens\":" << req.completion_tokens << ","
      <<   "\"total_tokens\":"      << (req.prompt_tokens + req.completion_tokens)
      << "}"
      << "}";
    return j.str();
}

// ─────────────────────────────────────────────
//  Thread di inferenza (background)
//
//  Questo thread gira per tutta la vita del server.
//  Il suo unico compito è:
//    1. Prelevare una richiesta dalla coda (bloccante)
//    2. Chiamare generate_for_request()
//    3. Tornare al punto 1
//
//  Quando queue.pop() ritorna nullptr, significa
//  che il server sta chiudendo (shutdown) e il
//  thread termina.
//
//  GESTIONE ERRORI: se generate_for_request() lancia
//  un'eccezione (improbabile ma possibile), catturiamo
//  l'errore, segnaliamo req.error e notifichiamo comunque
//  il thread HTTP in attesa, così il client non resta
//  bloccato all'infinito.
// ─────────────────────────────────────────────
static void inference_thread(Model& model,
                             const Tokenizer& tok,
                             RequestQueue& queue) {
    std::cout << "[INFERENCE] Thread di inferenza avviato\n";

    while (true) {
        auto req = queue.pop();
        if (!req) {
            std::cout << "[INFERENCE] Shutdown ricevuto, thread terminato\n";
            break;
        }

        std::cout << "[INFERENCE] Processo richiesta "
                  << req->endpoint_label << "\n";

        try {
            generate_for_request(model, tok, *req);
        } catch (const std::exception& e) {
            std::cerr << "[INFERENCE] ERRORE: " << e.what() << "\n";
            std::lock_guard<std::mutex> lock(req->mtx);
            req->error = true;
            req->error_msg = e.what();
            req->done = true;
            req->cv.notify_one();
        } catch (...) {
            std::cerr << "[INFERENCE] ERRORE sconosciuto\n";
            std::lock_guard<std::mutex> lock(req->mtx);
            req->error = true;
            req->error_msg = "errore sconosciuto durante l'inferenza";
            req->done = true;
            req->cv.notify_one();
        }
    }
}


// ═════════════════════════════════════════════════════════════════════════════
//  SERVER RUN — punto di ingresso principale
// ═════════════════════════════════════════════════════════════════════════════
//
//  1. Crea la RequestQueue
//  2. Avvia il thread di inferenza in background
//  3. Registra gli endpoint HTTP
//  4. Entra nel loop di ascolto di httplib
//
//  Gli endpoint non-streaming delegano al thread di inferenza
//  tramite la coda. Gli endpoint streaming e inspect lavorano
//  sincronamente nel thread HTTP.
// ═════════════════════════════════════════════════════════════════════════════
void server_run(Model& model, const Tokenizer& tok, int port) {
    httplib::Server svr;
    RequestQueue queue;

    // ── Avvia il thread di inferenza ──────────
    std::thread infer_worker(inference_thread,
                             std::ref(model),
                             std::ref(tok),
                             std::ref(queue));

    // ── CORS globale ──────────────────────────
    // Permette alla Web UI (o a qualsiasi client)
    // di chiamare le API anche se servita da un
    // dominio diverso (es. file:// o altro host).
    svr.set_post_routing_handler(
        [](const httplib::Request&, httplib::Response& res) {
            res.set_header("Access-Control-Allow-Origin", "*");
            res.set_header("Access-Control-Allow-Methods",
                           "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers",
                           "Content-Type");
        });

    // ── Preflight OPTIONS ─────────────────────
    // I browser inviano OPTIONS prima di POST cross-origin.
    svr.Options(".*",
        [](const httplib::Request&, httplib::Response& res) {
            res.status = 204;
        });

    // ── GET /health ───────────────────────────
    // Endpoint di health check — restituisce stato, nome modello
    // e parametri di sampling consigliati per la Web UI.
    svr.Get("/health", [&model, &tok](const httplib::Request&,
                                       httplib::Response& res) {
        std::string arch;
        std::string model_name;
        int rec_temp_idx = 2;   // indice nella tabella parametri
        if (model.config.arch == ArchType::GPT2) {
            arch = "gpt2";
            model_name = "GPT-2 small";
            rec_temp_idx = 0;
        } else {
            // Rileva il nome del modello dal chat template o default
            if      (tok.chat_template.find("<|im_start|>") != std::string::npos &&
                     tok.chat_template.find("enable_thinking") != std::string::npos)
                { arch = "qwen3"; model_name = "Qwen3-1.7B"; rec_temp_idx = 4; }
            else if (tok.chat_template.find("<|im_start|>") != std::string::npos)
                { arch = "qwen2"; model_name = "Qwen2.5-1.5B"; rec_temp_idx = 3; }
            else if (tok.chat_template.find("<|start_header_id|>") != std::string::npos)
                { arch = "llama3"; model_name = "Llama-3.2-3B"; rec_temp_idx = 2; }
            else
                { arch = "llama"; model_name = "TinyLlama 1.1B"; rec_temp_idx = 1; }
        }

        // Parametri consigliati per ogni modello
        // [temp, top_k, top_p, rep_penalty, enable_thinking]
        static const float rec_params[5][5] = {
            {1.0f, 40, 0.9f, 1.0f, 0.0f},  // GPT-2
            {0.7f, 50, 0.95f, 1.1f, 0.0f}, // TinyLlama
            {0.6f, 40, 0.9f, 1.1f, 0.0f},  // Llama-3.2
            {0.7f, 40, 0.8f, 1.05f, 0.0f}, // Qwen2.5
            {0.6f, 20, 0.95f, 1.5f, 1.0f}, // Qwen3 (thinking)
        };
        const float* p = rec_params[rec_temp_idx];

        std::ostringstream j;
        j << "{"
          << "\"status\":\"ok\","
          << "\"model\":\"" << model_name << "\","
          << "\"arch\":\"" << arch << "\","
          << "\"recommended\":{"
          << "\"temperature\":" << p[0] << ","
          << "\"top_k\":"       << (int)p[1] << ","
          << "\"top_p\":"       << p[2] << ","
          << "\"repetition_penalty\":" << p[3] << ","
          << "\"enable_thinking\":"    << (p[4] > 0.5f ? "true" : "false")
          << "}"
          << "}";
        res.set_content(j.str(), "application/json");
    });

    // ── POST /v1/completions ──────────────────
    //
    //  Endpoint per il completamento di testo grezzo.
    //  Se "stream": true, usa chunked transfer encoding
    //  per emettere token via SSE man mano.
    //
    //  Se "stream": false, crea una ServerRequest,
    //  la mette in coda, e attende che il thread di
    //  inferenza la processi e la segnali come done.
    svr.Post("/v1/completions",
        [&model, &tok, &queue](const httplib::Request& req,
                               httplib::Response& res) {

        std::cout << "[SERVER] POST /v1/completions arrivata"
                  << " (body=" << req.body.size() << " bytes)\n";

        std::string prompt     = json_get_string(req.body, "prompt");
        int         max_tokens = json_get_int   (req.body, "max_tokens", 200);
        max_tokens = std::min(max_tokens, 800);

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

        bool stream = json_get_bool(req.body, "stream", false);
        std::cout << "[SERVER] /v1/completions stream=" << (stream ? "true" : "false") << "\n";

        if (!stream) {
            // ═════════════════════════════════════
            //  RISPOSTA NON-STREAMING (via coda)
            // ═════════════════════════════════════
            // Creiamo una richiesta condivisa, la mettiamo
            // in coda, e aspettiamo che il thread di
            // inferenza la completi.
            auto sreq = std::make_shared<ServerRequest>();
            sreq->prompt = prompt;
            sreq->max_tokens = max_tokens;
            sreq->params = params;
            sreq->endpoint_label = "completions";

            std::cout << "[REQ:completions] prompt=\""
                      << prompt.substr(0, 40)
                      << (prompt.size() > 40 ? "..." : "")
                      << "\" max_tokens=" << max_tokens
                      << " temp=" << params.temperature << "\n";

            queue.push(sreq);

            // Attendere che il thread di inferenza finisca
            std::unique_lock<std::mutex> lock(sreq->mtx);
            sreq->cv.wait(lock, [&sreq] { return sreq->done; });

            if (sreq->error) {
                res.status = 500;
                res.set_content(
                    "{\"error\":\"" + json_escape(sreq->error_msg) + "\"}",
                    "application/json");
                return;
            }

            std::cout << "[RES] generati " << sreq->completion_tokens
                      << " token\n";
            res.set_content(build_response_json(*sreq), "application/json");
            return;
        }

        // ═════════════════════════════════════════
        //  RISPOSTA STREAMING (sincrona)
        // ═════════════════════════════════════════
        // Lo streaming NON passa per la coda: il thread HTTP
        // genera i token direttamente e li invia al client
        // via chunked transfer encoding. Questo mantiene la
        // Time-To-First-Token (TTFT) bassissima.
        res.set_chunked_content_provider("text/event-stream",
            [prompt, max_tokens, params, &model, &tok]
            (size_t offset, httplib::DataSink &sink) -> bool {
                if (offset > 0) {
                    sink.done();  // chiudi il chunked stream
                    return true;
                }

                std::cout << "[STREAM] /v1/completions content_provider iniziato\n";
                generate_streaming_for_request(
                    model, tok, prompt, max_tokens, params,
                    [&sink](const std::string& token,
                            bool /*is_first*/, bool is_last) -> bool {
                        std::string event = is_last
                            ? sse_text_done()
                            : sse_chat_chunk(token);
                        bool ok = sink.write(event.data(), event.size());
                        if (!ok) {
                            std::cout << "[STREAM] client disconnesso (sink.write=false)\n";
                        }
                        return ok;
                    });

                std::cout << "[STREAM] /v1/completions content_provider terminato\n";
                sink.done();  // invia il chunk finale 0\r\n\r\n
                return true;
            });
    });

    // ── POST /v1/chat/completions ─────────────
    //
    //  Endpoint compatibile con l'API OpenAI chat.
    //  Accetta un array "messages" con role/content,
    //  estrae TUTTA la conversazione, applica il chat
    //  template multi-turn se disponibile, e genera la risposta.
    svr.Post("/v1/chat/completions",
        [&model, &tok, &queue](const httplib::Request& req,
                               httplib::Response& res) {

        std::cout << "[SERVER] POST /v1/chat/completions arrivata"
                  << " (body=" << req.body.size() << " bytes)\n";

        auto messages = json_get_chat_messages(req.body);
        int max_tokens = json_get_int(req.body, "max_tokens", 200);
        max_tokens = std::min(max_tokens, 800);

        SamplingParams params;
        params.temperature = json_get_float(req.body, "temperature",        1.0f);
        params.top_p       = json_get_float(req.body, "top_p",              0.9f);
        params.top_k       = json_get_int  (req.body, "top_k",              40);
        params.rep_penalty = json_get_float(req.body, "repetition_penalty", 1.1f);
        params.greedy      = (params.temperature <= 0.0f);

        if (messages.empty()) {
            res.status = 400;
            res.set_content(
                "{\"error\":\"nessun messaggio trovato in messages\"}",
                "application/json");
            return;
        }

        // Thinking mode: di default true (il modello decide se ragionare),
        // false = risposta diretta senza ragionamento (Qwen3).
        bool enable_thinking = json_get_bool(req.body, "enable_thinking", true);

        std::string prompt = tok.has_chat_template
            ? apply_chat_template_conversation(tok, messages, enable_thinking)
            : json_get_last_user_message(req.body);

        bool stream = json_get_bool(req.body, "stream", false);
        std::cout << "[SERVER] /v1/chat/completions stream=" << (stream ? "true" : "false") << "\n";

        if (!stream) {
            // ═════════════════════════════════════
            //  RISPOSTA NON-STREAMING (via coda)
            // ═════════════════════════════════════
            auto sreq = std::make_shared<ServerRequest>();
            sreq->prompt = prompt;
            sreq->max_tokens = max_tokens;
            sreq->params = params;
            sreq->endpoint_label = "chat/completions";

            std::cout << "[REQ:chat/completions] prompt=\""
                      << prompt.substr(0, 40)
                      << (prompt.size() > 40 ? "..." : "")
                      << "\" max_tokens=" << max_tokens
                      << " temp=" << params.temperature << "\n";

            queue.push(sreq);

            std::unique_lock<std::mutex> lock(sreq->mtx);
            sreq->cv.wait(lock, [&sreq] { return sreq->done; });

            if (sreq->error) {
                res.status = 500;
                res.set_content(
                    "{\"error\":\"" + json_escape(sreq->error_msg) + "\"}",
                    "application/json");
                return;
            }

            std::cout << "[RES] generati " << sreq->completion_tokens
                      << " token\n";
            res.set_content(build_response_json(*sreq), "application/json");
            return;
        }

        // ═════════════════════════════════════════
        //  RISPOSTA STREAMING (sincrona)
        // ═════════════════════════════════════════
        res.set_chunked_content_provider("text/event-stream",
            [prompt, max_tokens, params, &model, &tok]
            (size_t offset, httplib::DataSink &sink) -> bool {
                if (offset > 0) {
                    sink.done();
                    return true;
                }

                std::cout << "[STREAM] /v1/chat/completions content_provider iniziato\n";
                generate_streaming_for_request(
                    model, tok, prompt, max_tokens, params,
                    [&sink](const std::string& token,
                            bool /*is_first*/, bool is_last) -> bool {
                        std::string event = is_last
                            ? sse_chat_done()
                            : sse_chat_chunk(token);
                        bool ok = sink.write(event.data(), event.size());
                        if (!ok) {
                            std::cout << "[STREAM] client disconnesso (sink.write=false)\n";
                        }
                        return ok;
                    });

                std::cout << "[STREAM] /v1/chat/completions content_provider terminato\n";
                sink.done();
                return true;
            });
    });

    // ── POST /v1/inspect/attention ────────────
    //
    //  Endpoint per esportare gli attention scores
    //  di un prompt. Esegue il forward pass e salva
    //  i pesi attention per ogni layer/head.
    //
    //  Questo endpoint è SINCRONO: non passa per la
    //  coda ma viene eseguito direttamente dal thread
    //  HTTP. Questo è accettabile perché:
    //    1. È un endpoint di diagnostica/debug
    //    2. Non è usato in produzione ad alto carico
    //    3. Mantenere la semplicità didattica
    //
    //  NOTA: in uno scenario reale con molte richieste
    //  in coda, bisognerebbe o serializzare anche questo
    //  endpoint o usare un modello dedicato all'inspect.
    svr.Post("/v1/inspect/attention",
        [&model, &tok](const httplib::Request& req,
                       httplib::Response& res) {

        std::string prompt = json_get_string(req.body, "prompt");
        int max_len = json_get_int(req.body, "max_len", 100);
        max_len = std::min(max_len, 100);

        if (prompt.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"prompt mancante\"}",
                            "application/json");
            return;
        }

        std::cout << "[INSPECT] attention per: \""
                  << prompt.substr(0, 40)
                  << (prompt.size() > 40 ? "..." : "")
                  << "\"\n";

        std::string json = inspect_attention(model, tok, prompt, max_len);
        res.set_content(json, "application/json");
    });

    // ── Monta directory webui per la UI statica ──
    svr.set_mount_point("/", "./webui");

    // ── Banner di avvio ───────────────────────
    std::cout << "\n╔═══════════════════════════════════════╗\n";
    std::cout << "║   EIE-LLM Server                      ║\n";
    std::cout << "║   http://localhost:" << port
              << "               ║\n";
    std::cout << "╠═══════════════════════════════════════╣\n";
    std::cout << "║   GET  /health                        ║\n";
    std::cout << "║   POST /v1/completions                ║\n";
    std::cout << "║   POST /v1/completions (stream)       ║\n";
    std::cout << "║   POST /v1/chat/completions           ║\n";
    std::cout << "║   POST /v1/chat/completions (stream)  ║\n";
    std::cout << "║   POST /v1/inspect/attention          ║\n";
    std::cout << "╠═══════════════════════════════════════╣\n";
    std::cout << "║   Web UI: http://localhost:" << port
              << "/            ║\n";
    std::cout << "╠═══════════════════════════════════════╣\n";
    std::cout << "║   Chat template: "
              << std::left << std::setw(21)
              << (tok.has_chat_template ? "attivo" : "non disponibile")
              << "║\n";
    std::cout << "║   Prefix cache: attivo                ║\n";
    std::cout << "║   Continuous batching: attivo         ║\n";
    std::cout << "╚═══════════════════════════════════════╝\n\n";
    std::cout << "Premi Ctrl+C per fermare il server\n\n";

    // ── Avvio server (bloccante) ──────────────
    svr.listen("0.0.0.0", port);

    // ── Shutdown pulito ───────────────────────
    // Quando listen() ritorna (es. Ctrl+C), segnaliamo
    // lo shutdown alla coda così il thread di inferenza
    // può terminare gracefulmente.
    queue.signal_shutdown();
    if (infer_worker.joinable()) {
        infer_worker.join();
    }
}
