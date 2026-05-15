#include "tokenizer.hpp"
#include <iostream>
#include <algorithm>
#include <climits>
#include <iomanip>
#include <array>
#include <cstdio>

// ─────────────────────────────────────────────
//  Helper: cerca array nei metadata
// ─────────────────────────────────────────────
static const GGUFArray* find_array(const GGUFContext& ctx,
                                   const std::string& key) {
    for (const auto& kv : ctx.metadata)
        if (kv.key == key)
            return kv.value.get_if<GGUFArray>();
    return nullptr;
}

static uint32_t find_uint32(const GGUFContext& ctx,
                             const std::string& key,
                             uint32_t def = 0) {
    for (const auto& kv : ctx.metadata)
        if (kv.key == key)
            if (auto* v = kv.value.get_if<uint32_t>())
                return *v;
    return def;
}

// ─────────────────────────────────────────────
//  Inizializzazione tokenizer
//
//  Rileva il tipo (GPT2 o SentencePiece) dal
//  campo "tokenizer.ggml.model" nei metadata,
//  poi carica vocabolario e merge rules.
// ─────────────────────────────────────────────
bool tokenizer_init(Tokenizer& tok, const GGUFContext& ctx) {

    // ── Rileva tipo ───────────────────────────
    std::string tok_model;
    for (const auto& kv : ctx.metadata)
        if (kv.key == "tokenizer.ggml.model")
            if (auto* s = kv.value.get_if<std::string>())
                tok_model = *s;

    tok.type = (tok_model == "llama" || tok_model == "sentencepiece")
        ? TokenizerType::SENTENCEPIECE
        : TokenizerType::GPT2;

    // ── Token speciali ────────────────────────
    tok.bos_id = find_uint32(ctx, "tokenizer.ggml.bos_token_id", 1);
    tok.eos_id = find_uint32(ctx, "tokenizer.ggml.eos_token_id", 2);
    tok.unk_id = find_uint32(ctx, "tokenizer.ggml.unknown_token_id", 0);

    // ── Vocabolario ───────────────────────────
    const GGUFArray* tokens_arr = find_array(ctx, "tokenizer.ggml.tokens");
    if (!tokens_arr) {
        std::cerr << "[ERRORE] tokenizer.ggml.tokens non trovato\n";
        return false;
    }

    tok.vocab_size = static_cast<int>(tokens_arr->values.size());
    tok.id_to_token.reserve(tok.vocab_size);
    tok.token_to_id.reserve(tok.vocab_size);

    for (int id = 0; id < tok.vocab_size; id++) {
        const auto* s = tokens_arr->values[id].get_if<std::string>();
        if (!s) {
            std::cerr << "[ERRORE] Token #" << id << " non stringa\n";
            return false;
        }
        tok.id_to_token.push_back(*s);
        tok.token_to_id[*s] = id;
    }

    // ── Scores (SentencePiece) ────────────────
    const GGUFArray* scores_arr = find_array(ctx, "tokenizer.ggml.scores");
    if (scores_arr) {
        tok.scores.reserve(tok.vocab_size);
        for (const auto& v : scores_arr->values)
            if (auto* f = v.get_if<float>())
                tok.scores.push_back(*f);
    }

    // ── Token type (SentencePiece) ────────────
    const GGUFArray* types_arr = find_array(ctx, "tokenizer.ggml.token_type");
    if (types_arr) {
        tok.token_type.reserve(tok.vocab_size);
        for (const auto& v : types_arr->values)
            if (auto* i = v.get_if<int32_t>())
                tok.token_type.push_back(*i);
    }

    // ── Token speciali (controllo) per modelli come Llama-3 ──
    // Questi token NON devono essere normalizzati (SentencePiece aggiungerebbe
    // ▁ all'inizio o sostituirebbe gli spazi). Vengono estratti e passati
    // attraverso come token singoli durante l'encode.
    for (int id = 0; id < tok.vocab_size; id++) {
        if (id < (int)tok.token_type.size() && tok.token_type[id] == 3) {
            tok.special_tokens.push_back({tok.id_to_token[id], id});
        }
    }
    // Ordina per lunghezza decrescente: matching greedy preferisce
    // token lunghi (es. <|start_header_id|> prima di <|eot_id|>).
    std::sort(tok.special_tokens.begin(), tok.special_tokens.end(),
              [](const auto& a, const auto& b) {
                  return a.first.size() > b.first.size();
              });

    // ── Merge rules (usate solo per GPT-2 BPE) ───
    const GGUFArray* merges_arr = find_array(ctx, "tokenizer.ggml.merges");
    if (merges_arr) {
        tok.merges.reserve(merges_arr->values.size());
        for (int rank = 0; rank < (int)merges_arr->values.size(); rank++) {
            const auto* s = merges_arr->values[rank].get_if<std::string>();
            if (!s) continue;
            size_t space = s->find(' ');
            if (space == std::string::npos) continue;
            BPEMerge merge;
            merge.first  = s->substr(0, space);
            merge.second = s->substr(space + 1);
            merge.rank   = rank;
            tok.merge_rank[merge.first + " " + merge.second] = rank;
            tok.merges.push_back(std::move(merge));
        }
    }

    // ── Lunghezza massima token ───────────────
    // Calcolata una volta sola per limitare la finestra di ricerca
    // nell'algoritmo Viterbi, evitando substr inutilmente lunghe.
    tok.max_token_len = 1;
    for (const auto& s : tok.id_to_token)
        if ((int)s.size() > tok.max_token_len)
            tok.max_token_len = (int)s.size();

    // ── Chat template ─────────────────────────
    for (const auto& kv : ctx.metadata)
        if (kv.key == "tokenizer.chat_template")
            if (auto* s = kv.value.get_if<std::string>())
                tok.chat_template = *s;

    tok.has_chat_template = !tok.chat_template.empty();

    // Stringa EOS usata nel template (es. "</s>" per LLaMA)
    if (tok.eos_id >= 0 && tok.eos_id < tok.vocab_size)
        tok.eos_token_str = tok.id_to_token[tok.eos_id];

    return true;
}

// ─────────────────────────────────────────────
//  Split UTF-8 in caratteri
// ─────────────────────────────────────────────
static std::vector<std::string> split_utf8(const std::string& s) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < s.size(); ) {
        unsigned char c = s[i];
        int len = 1;
        if      (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

// ─────────────────────────────────────────────
//  Applica BPE merge su un vettore di token
// ─────────────────────────────────────────────
static void apply_bpe(std::vector<std::string>& tokens,
                      const Tokenizer& tok) {
    while (tokens.size() > 1) {
        int    best_rank = INT_MAX;
        size_t best_idx  = 0;
        bool   found     = false;
        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            auto it = tok.merge_rank.find(tokens[i] + " " + tokens[i+1]);
            if (it != tok.merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx  = i;
                found     = true;
            }
        }
        if (!found) break;
        tokens[best_idx] += tokens[best_idx + 1];
        tokens.erase(tokens.begin() + best_idx + 1);
    }
}

// ─────────────────────────────────────────────
//  Encode GPT-2
// ─────────────────────────────────────────────
static std::vector<int> encode_gpt2(const Tokenizer& tok,
                                    const std::string& text) {
    static const std::string SPACE_SYM = "\xc4\xa0"; // Ġ

    // Pre-tokenizzazione
    std::vector<std::string> words;
    std::string current;
    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];
        if (c == ' ' && !current.empty()) {
            words.push_back(current);
            current = SPACE_SYM;
        } else if (ispunct((unsigned char)c) && c != '\'') {
            if (!current.empty()) { words.push_back(current); current.clear(); }
            words.push_back(std::string(1, c));
        } else {
            current += c;
        }
    }
    if (!current.empty()) words.push_back(current);

    std::vector<int> result;
    for (const auto& word : words) {
        auto tokens = split_utf8(word);
        apply_bpe(tokens, tok);
        for (const auto& t : tokens) {
            auto it = tok.token_to_id.find(t);
            result.push_back(it != tok.token_to_id.end()
                             ? it->second : tok.unk_id);
        }
    }
    return result;
}

// ─────────────────────────────────────────────
//  Encode SentencePiece — algoritmo Viterbi unigram
//
//  SentencePiece LLaMA non usa BPE tradizionale:
//  usa un modello unigram dove ogni token ha un
//  punteggio (log-probabilità). La tokenizzazione
//  ottimale è quella che massimizza la somma dei
//  punteggi — trovata con programmazione dinamica.
//
//  Normalizzazione del testo:
//    "Hello world" → "▁Hello▁world"
//    (▁ = U+2581, UTF-8: 0xE2 0x96 0x81)
//    Ogni spazio diventa ▁; ▁ viene preposto all'inizio.
//
//  Algoritmo Viterbi (forward pass):
//    best_score[i] = massima somma log-prob per coprire
//                    i byte normalizzati da 0 a i.
//    Per ogni posizione i, si provano tutti i token del
//    vocabolario che iniziano in i e durano al più
//    max_token_len byte. Si aggiorna best_score[i+len]
//    se il nuovo punteggio è migliore.
//
//  Fallback su byte-token <0xXX>:
//    Se nessun token del vocabolario copre un byte,
//    si usa il token speciale "<0xXX>" (sempre presente
//    nel vocabolario SentencePiece). Questo garantisce
//    che l'algoritmo raggiunga sempre la fine del testo.
//
//  Backtrace (backward pass):
//    Partendo dalla fine, si risale la catena di best_tok
//    e best_prev per ricostruire la sequenza di token.
// ─────────────────────────────────────────────
// ─────────────────────────────────────────────
//  Viterbi su testo già normalizzato
// ─────────────────────────────────────────────
static std::vector<int> viterbi_sentencepiece(const Tokenizer& tok,
                                                const std::string& norm) {
    int n = (int)norm.size();
    if (n == 0) return {};

    // Precalcola mappa byte → token di fallback
    std::array<int, 256> byte_tok;
    byte_tok.fill(tok.unk_id);
    for (int b = 0; b < 256; b++) {
        char hex[8];
        std::snprintf(hex, sizeof(hex), "<0x%02X>", (unsigned char)b);
        auto it = tok.token_to_id.find(std::string(hex));
        if (it != tok.token_to_id.end())
            byte_tok[b] = it->second;
    }

    std::vector<float> best_score(n + 1, -1e30f);
    std::vector<int>   best_tok  (n + 1, -1);
    std::vector<int>   best_prev (n + 1, -1);

    best_score[0] = 0.0f;

    for (int i = 0; i < n; i++) {
        if (best_score[i] < -1e29f) continue;

        int end = std::min(n, i + tok.max_token_len);
        for (int j = i + 1; j <= end; j++) {
            auto it = tok.token_to_id.find(norm.substr(i, j - i));
            if (it == tok.token_to_id.end()) continue;

            int   tid   = it->second;
            float score = (!tok.scores.empty() && tid < (int)tok.scores.size())
                          ? tok.scores[tid] : 0.0f;
            float ns    = best_score[i] + score;

            if (ns > best_score[j]) {
                best_score[j] = ns;
                best_tok  [j] = tid;
                best_prev [j] = i;
            }
        }

        if (i + 1 <= n && best_score[i + 1] < -1e29f) {
            unsigned char b = (unsigned char)norm[i];
            int tid  = byte_tok[b];
            float ns = best_score[i] - 10.0f;
            best_score[i + 1] = ns;
            best_tok  [i + 1] = tid;
            best_prev [i + 1] = i;
        }
    }

    std::vector<int> result;
    int pos = n;
    while (pos > 0 && best_tok[pos] != -1) {
        result.push_back(best_tok[pos]);
        pos = best_prev[pos];
    }
    std::reverse(result.begin(), result.end());
    return result;
}

// ─────────────────────────────────────────────
//  Encode SentencePiece con supporto token speciali
//
//  Spezza il testo in segmenti separati dai token speciali
//  (token_type == 3, es. <|start_header_id|> di Llama-3).
//  I token speciali NON vengono normalizzati (SentencePiece
//  aggiungerebbe ▁ all'inizio), ma passano attraverso
//  come token singoli. Il testo tra i token speciali viene
//  normalizzato e tokenizzato con Viterbi come al solito.
// ─────────────────────────────────────────────
static std::vector<int> encode_sentencepiece(const Tokenizer& tok,
                                              const std::string& text) {
    static const std::string SP = "\xe2\x96\x81";  // ▁ in UTF-8

    if (text.empty()) return {};

    std::string norm;
    norm.reserve(text.size() + SP.size());
    norm = SP;
    for (char c : text) {
        if (c == ' ') norm += SP;
        else          norm += c;
    }
    return viterbi_sentencepiece(tok, norm);
}

// ─────────────────────────────────────────────
//  Encode pubblico
// ─────────────────────────────────────────────
std::vector<int> tokenizer_encode(const Tokenizer& tok,
                                   const std::string& text,
                                   bool add_bos) {
    if (text.empty()) return {};

    std::vector<int> ids;

    // SentencePiece aggiunge sempre BOS all'inizio
    if (add_bos && tok.type == TokenizerType::SENTENCEPIECE)
        ids.push_back(tok.bos_id);

    // Se non ci sono token speciali, usa l'encoder nativo direttamente
    if (tok.special_tokens.empty()) {
        auto encoded = (tok.type == TokenizerType::SENTENCEPIECE)
            ? encode_sentencepiece(tok, text)
            : encode_gpt2(tok, text);
        ids.insert(ids.end(), encoded.begin(), encoded.end());
        return ids;
    }

    // ── Token speciali splitting (GPT-2 e SentencePiece) ──
    // Spezza il testo in segmenti: testo normale | token speciale.
    // I token speciali (control) passano attraverso senza essere
    // processati dall'encoder (che li spezzerebbe in caratteri).
    struct Segment { bool is_special; std::string text; int special_id; };
    std::vector<Segment> segments;

    size_t pos = 0;
    while (pos < text.size()) {
        bool found = false;
        for (const auto& [tok_str, tok_id] : tok.special_tokens) {
            if (!tok_str.empty() && text.compare(pos, tok_str.size(), tok_str) == 0) {
                segments.push_back({true, tok_str, tok_id});
                pos += tok_str.size();
                found = true;
                break;
            }
        }
        if (found) continue;

        size_t next_special = std::string::npos;
        for (const auto& [tok_str, tok_id] : tok.special_tokens) {
            size_t found = text.find(tok_str, pos);
            if (found != std::string::npos && found < next_special)
                next_special = found;
        }

        if (next_special == std::string::npos) {
            segments.push_back({false, text.substr(pos), -1});
            break;
        } else {
            segments.push_back({false, text.substr(pos, next_special - pos), -1});
            pos = next_special;
        }
    }

    for (const auto& seg : segments) {
        if (seg.is_special) {
            ids.push_back(seg.special_id);
        } else {
            auto encoded = (tok.type == TokenizerType::SENTENCEPIECE)
                ? encode_sentencepiece(tok, seg.text)
                : encode_gpt2(tok, seg.text);
            ids.insert(ids.end(), encoded.begin(), encoded.end());
        }
    }
    return ids;
}

// ─────────────────────────────────────────────
//  Decode
//
//  GPT-2: sostituisce Ġ con spazio
//  SentencePiece: sostituisce ▁ con spazio,
//                 gestisce token byte speciali
// ─────────────────────────────────────────────
std::string tokenizer_decode(const Tokenizer& tok,
                             const std::vector<int>& ids) {
    std::string result;

    for (int id : ids) {
        if (id < 0 || id >= tok.vocab_size) continue;

        // Salta token speciali di controllo
        if (!tok.token_type.empty() &&
            id < (int)tok.token_type.size()) {
            int tt = tok.token_type[id];
            if (tt == 3) continue;  // token di controllo (BOS, EOS, ecc.)
        }

        std::string token = tok.id_to_token[id];

        if (tok.type == TokenizerType::SENTENCEPIECE) {
            // Sostituisci ▁ con spazio
            static const std::string SP  = "\xe2\x96\x81";
            static const std::string SPC = " ";
            size_t pos = 0;
            while ((pos = token.find(SP, pos)) != std::string::npos) {
                token.replace(pos, SP.size(), SPC);
                pos += 1;
            }

            // Gestisci token byte: formato "<0xXX>"
            // SentencePiece usa questi per byte non UTF-8
            if (token.size() == 6 &&
                token[0] == '<' && token[1] == '0' &&
                token[2] == 'x' && token[5] == '>') {
                try {
                    uint8_t byte = static_cast<uint8_t>(
                        std::stoi(token.substr(3, 2), nullptr, 16));
                    if (byte >= 0x20 && byte <= 0x7E)
                        token = std::string(1, (char)byte);
                    else
                        token = "";  // scarta byte non stampabili
                } catch (...) {}
            }
        } else {
            // GPT-2: sostituisci Ġ con spazio
            static const std::string G  = "\xc4\xa0";
            static const std::string SP = " ";
            size_t pos = 0;
            while ((pos = token.find(G, pos)) != std::string::npos) {
                token.replace(pos, G.size(), SP);
                pos += 1;
            }
        }

        result += token;
    }
    return result;
}

// ─────────────────────────────────────────────
//  apply_chat_template
//
//  Formatta un messaggio utente secondo il chat template
//  letto dai metadata GGUF, senza interpretare Jinja2.
//
//  Strategia: rileva il "sapore" del template cercando
//  tag caratteristici, poi applica la formattazione fissa
//  corrispondente. Supporta due formati comuni:
//
//  ● TinyLlama / LLaMA-chat  (tag <|user|>):
//      [<|system|>\n{system}</s>\n]
//      <|user|>\n{user}</s>
//      <|assistant|>\n
//
//  ● ChatML  (tag <|im_start|>, usato da Mistral/Qwen):
//      [<|im_start|>system\n{system}<|im_end|>\n]
//      <|im_start|>user\n{user}<|im_end|>
//      <|im_start|>assistant\n
//
//  Il tag <|assistant|> (o equivalente) finale segnala al
//  modello che deve iniziare a generare la risposta.
//  L'EOS token viene letto da tok.eos_token_str (es. "</s>").
//
//  Se il template non è riconosciuto, il testo viene restituito
//  invariato — meglio un prompt grezzo che un formato sbagliato.
// ─────────────────────────────────────────────
std::string apply_chat_template(const Tokenizer& tok,
                                const std::string& user_msg,
                                const std::string& system_msg) {
    const std::string& eos = tok.eos_token_str;

    // Rilevamento del formato dal template grezzo
    bool is_llama_chat = tok.chat_template.find("<|user|>")     != std::string::npos;
    bool is_chatml     = tok.chat_template.find("<|im_start|>") != std::string::npos;
    bool is_llama3     = tok.chat_template.find("<|start_header_id|>") != std::string::npos;

    std::string result;

    if (is_llama3) {
        // ── Formato LLaMA-3 / Llama-3.2 ───────────
        // Il tokenizer SentencePiece aggiunge automaticamente BOS (128000)
        // all'inizio della sequenza, quindi non serve <|begin_of_text|>.
        if (!system_msg.empty()) {
            result += "<|start_header_id|>system<|end_header_id|>\n\n"
                   +  system_msg + "<|eot_id|>";
        }
        result += "<|start_header_id|>user<|end_header_id|>\n\n"
               +  user_msg + "<|eot_id|>"
               +  "<|start_header_id|>assistant<|end_header_id|>\n\n";

    } else if (is_llama_chat) {
        // ── Formato TinyLlama / LLaMA-chat ───────
        if (!system_msg.empty())
            result += "<|system|>\n" + system_msg + eos + "\n";
        result += "<|user|>\n" + user_msg + eos + "\n<|assistant|>\n";

    } else if (is_chatml) {
        // ── Formato ChatML ────────────────────────
        if (!system_msg.empty())
            result += "<|im_start|>system\n" + system_msg + "<|im_end|>\n";
        result += "<|im_start|>user\n" + user_msg + "<|im_end|>\n"
               +  "<|im_start|>assistant\n";

    } else {
        // ── Fallback: nessun formato riconosciuto ─
        result = user_msg;
    }

    return result;
}

// ─────────────────────────────────────────────
//  apply_chat_template_conversation
//
//  Come apply_chat_template ma per conversazioni multi-turn.
//  Itera su tutti i messaggi e li formatta secondo il template.
//  Aggiunge il tag assistant finale per invitare il modello a rispondere.
//
//  enable_thinking controlla il thinking mode di Qwen3:
//    true  — il modello decide liberamente se usare <think>...
//    false — il template inserisce <think>\n\n</think>\n\n prima del
//            tag assistant, segnalando al modello di non ragionare.
// ─────────────────────────────────────────────
std::string apply_chat_template_conversation(
    const Tokenizer& tok,
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool enable_thinking) {

    const std::string& eos = tok.eos_token_str;

    bool is_llama_chat = tok.chat_template.find("<|user|>")     != std::string::npos;
    bool is_chatml     = tok.chat_template.find("<|im_start|>") != std::string::npos;
    bool is_llama3     = tok.chat_template.find("<|start_header_id|>") != std::string::npos;

    std::string result;

    if (is_llama3) {
        // Formato LLaMA-3 / Llama-3.2
        for (const auto& [role, content] : messages) {
            if (role == "system") {
                result += "<|start_header_id|>system<|end_header_id|>\n\n"
                       +  content + "<|eot_id|>";
            } else if (role == "user") {
                result += "<|start_header_id|>user<|end_header_id|>\n\n"
                       +  content + "<|eot_id|>";
            } else if (role == "assistant") {
                result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
                       +  content + "<|eot_id|>";
            }
        }
        result += "<|start_header_id|>assistant<|end_header_id|>\n\n";

    } else if (is_llama_chat) {
        // Formato TinyLlama / LLaMA-chat
        for (const auto& [role, content] : messages) {
            if (role == "system") {
                result += "<|system|>\n" + content + eos + "\n";
            } else if (role == "user") {
                result += "<|user|>\n" + content + eos + "\n";
            } else if (role == "assistant") {
                result += "<|assistant|>\n" + content + eos + "\n";
            }
        }
        result += "<|assistant|>\n";

    } else if (is_chatml) {
        // Formato ChatML (Mistral/Qwen)
        // Supporta anche il thinking mode di Qwen3.
        for (const auto& [role, content] : messages) {
            result += "<|im_start|>" + role + "\n" + content + "<|im_end|>\n";
        }
        result += "<|im_start|>assistant\n";

        // Thinking mode OFF: inserisce il tag vuoto per disabilitare
        // il ragionamento (Qwen3 interpreta <think>\n\n</think>\n\n
        // come segnale per saltare il reasoning).
        if (!enable_thinking) {
            result += "<think>\n\n</think>\n\n";
        }

    } else {
        // Fallback: concatenazione grezza
        for (const auto& [role, content] : messages) {
            result += role + ": " + content + "\n";
        }
        result += "assistant: ";
    }

    return result;
}

// ─────────────────────────────────────────────
//  parse_think_tags
//
//  Separa il contenuto <think>...</think> dal resto
//  del testo generato dal modello.
//
//  Il formato tipico di Qwen3 in thinking mode è:
//    <think>
//    Il ragionamento...
//    </think>
//
//    La risposta finale.
//
//  Se non ci sono tag think, la coppia restituita
//  ha first vuoto e second = testo originale.
// ─────────────────────────────────────────────
std::pair<std::string, std::string> parse_think_tags(const std::string& text) {
    std::string thinking;
    std::string reply;

    size_t think_start = text.find("<think>");
    if (think_start == std::string::npos) {
        // Nessun tag think: tutto è reply
        return {"", text};
    }

    size_t think_content_start = think_start + 7; // 7 = len("<think>")
    size_t think_end = text.find("</think>", think_content_start);

    if (think_end == std::string::npos) {
        // Tag di apertura ma non di chiusura: tutto dopo <think> è thinking
        thinking = text.substr(think_content_start);
        return {thinking, ""};
    }

    // Estrae il contenuto thinking (tra i tag) e toglie whitespace
    thinking = text.substr(think_content_start, think_end - think_content_start);
    // Se il contenuto è solo whitespace, lo consideriamo vuoto
    bool only_ws = true;
    for (char ch : thinking)
        if (ch != ' ' && ch != '\n' && ch != '\t' && ch != '\r') { only_ws = false; break; }
    if (only_ws) thinking.clear();

    // Il resto dopo </think> è la risposta
    size_t reply_start = think_end + 8; // 8 = len("</think>")
    while (reply_start < text.size() && (text[reply_start] == '\n' || text[reply_start] == ' '))
        reply_start++;
    reply = (reply_start < text.size()) ? text.substr(reply_start) : "";

    return {thinking, reply};
}

// ─────────────────────────────────────────────
//  sanitize_output
//
//  Converte i caratteri Latin Extended-A (U+0100-U+017F)
//  del BPE byte-level GPT-2/Qwen ai byte originali.
//  Senza questa conversione, l'output mostrerebbe caratteri
//  come Ġ (al posto dello spazio), Ċ (al posto del newline)
//  e sequenze âĢĵ (al posto di trattini/punteggiatura).
//
//  Funzionamento:
//  - ASCII stampabile (0x20-0x7E): passa direttamente
//  - Newline (0x0A): passa come \n
//  - Latin Extended-A (U+0100-U+017F): riconverte al byte
//    originale sottraendo 0x100
//  - Ogni altra sequenza UTF-8: passa intatta
// ─────────────────────────────────────────────
std::string sanitize_output(const std::string& s) {
    // Converte i caratteri Latin Extended-A (U+0100-U+017F) del BPE
    // byte-level GPT-2/Qwen ai byte originali.
    //
    // Regole:
    //   - ASCII stampabile (0x20-0x7E) + newline/tab: passa
    //   - 2-byte UTF-8 in U+0100-U+017F: byte mapping → byte originale
    //   - Ogni altro byte (inclusi UTF-8 multi-byte, continuation bytes):
    //     passa intatto — meglio un carattere raw che uno perso
    std::string out;
    out.reserve(s.size());

    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);

        // ASCII stampabile + newline + tab: passa direttamente
        if (c >= 0x20 && c <= 0x7E) { out += s[i++]; continue; }
        if (c == 0x0A || c == 0x09) { out += s[i++]; continue; }

        // Possibile 2-byte UTF-8: verifica se è Latin Extended-A (byte mapping)
        if (c >= 0xC0 && c <= 0xDF && i + 1 < s.size()) {
            unsigned char c2 = static_cast<unsigned char>(s[i+1]);
            if ((c2 & 0xC0) == 0x80) {
                uint32_t cp = ((c & 0x1F) << 6) | (c2 & 0x3F);
                // U+0100-U+017F = byte mapping GPT-2: converti al byte originale
                if (cp >= 0x100 && cp <= 0x17F) {
                    uint8_t b = static_cast<uint8_t>(cp - 0x100);
                    if (b >= 0x20 && b <= 0x7E) out += (char)b;
                    else if (b == 0x0A) out += '\n';
                    i += 2; continue;
                }
                // Altri caratteri 2-byte validi (es. è, à): passa intatti
                out += s[i]; out += s[i+1];
                i += 2; continue;
            }
            // Continuazione non valida: passa il primo byte,
            // il secondo verrà gestito al prossimo ciclo
            out += s[i++]; continue;
        }

        // Tutto il resto (UTF-8 multi-byte, continuation bytes solitari,
        // byte di controllo): passa intatto — meglio visibile che perso
        // Questo gestisce il caso in cui un carattere UTF-8 multi-byte
        // è stato splittato in più token dal BPE e arriva in pezzi.
        out += s[i++];
    }
    return out;
}

// ─────────────────────────────────────────────
//  Print info
// ─────────────────────────────────────────────
void tokenizer_print_info(const Tokenizer& tok) {
    std::cout << "\n═══════════════════════════════════════\n";
    std::cout << "  EIE-LLM — Tokenizer Info\n";
    std::cout << "═══════════════════════════════════════\n";
    std::cout << "  Tipo        : "
              << (tok.type == TokenizerType::SENTENCEPIECE
                  ? "SentencePiece (Viterbi unigram)" : "GPT-2 BPE") << "\n";
    std::cout << "  Vocab size  : " << tok.vocab_size      << "\n";
    std::cout << "  Merge rules : " << tok.merges.size()   << "\n";
    std::cout << "  Scores      : " << tok.scores.size()   << "\n";
    std::cout << "  Max tok len : " << tok.max_token_len   << " byte\n";
    std::cout << "  BOS id      : " << tok.bos_id          << "\n";
    std::cout << "  EOS id      : " << tok.eos_id
              << " (\"" << tok.eos_token_str << "\")\n";
    std::cout << "  Chat template: "
              << (tok.has_chat_template ? "sì" : "no") << "\n";
    std::cout << "  Primi 10 token:\n";
    for (int i = 0; i < std::min(10, tok.vocab_size); i++) {
        std::cout << "    [" << std::setw(5) << i << "] = \"";
        for (unsigned char c : tok.id_to_token[i])
            if (c >= 32 && c < 127) std::cout << c;
            else std::cout << "\\x" << std::hex
                           << std::setw(2) << std::setfill('0')
                           << (int)c << std::dec;
        std::cout << "\"\n";
    }
    std::cout << "═══════════════════════════════════════\n\n";
}