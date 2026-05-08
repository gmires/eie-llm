#include "tokenizer.hpp"
#include <iostream>
#include <algorithm>
#include <climits>
#include <iomanip>

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

    // ── Merge rules ───────────────────────────
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
//  Encode SentencePiece
//
//  SentencePiece usa ▁ (U+2581, UTF-8: 0xE2 0x96 0x81)
//  per rappresentare lo spazio che precede una parola.
//
//  Pre-tokenizzazione:
//    "Hello world" → ["▁Hello", "▁world"]
//    La prima parola NON ha ▁ se il testo inizia senza spazio.
//
//  BPE identico a GPT-2 ma sul vocabolario SentencePiece.
// ─────────────────────────────────────────────
static std::vector<int> encode_sentencepiece(const Tokenizer& tok,
                                              const std::string& text) {
    // ▁ in UTF-8
    static const std::string SP = "\xe2\x96\x81";

    if (text.empty()) return {};

    // Dividi in parole aggiungendo ▁ dove c'è uno spazio
    std::vector<std::string> words;
    std::string current;
    bool first = true;

    for (size_t i = 0; i < text.size(); ) {
        unsigned char c = text[i];

        if (c == ' ') {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            // Il prossimo token avrà ▁ come prefisso
            current = SP;
            i++;
        } else {
            // Prima parola senza spazio iniziale
            if (first && current.empty())
                current = SP;  // SentencePiece aggiunge sempre ▁
            first = false;

            // Copia il carattere UTF-8
            int len = 1;
            if      (c >= 0xF0) len = 4;
            else if (c >= 0xE0) len = 3;
            else if (c >= 0xC0) len = 2;
            current += text.substr(i, len);
            i += len;
        }
    }
    if (!current.empty()) words.push_back(current);

    // BPE su ogni parola
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

    auto encoded = (tok.type == TokenizerType::SENTENCEPIECE)
        ? encode_sentencepiece(tok, text)
        : encode_gpt2(tok, text);

    ids.insert(ids.end(), encoded.begin(), encoded.end());
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
//  Print info
// ─────────────────────────────────────────────
void tokenizer_print_info(const Tokenizer& tok) {
    std::cout << "\n═══════════════════════════════════════\n";
    std::cout << "  EIE-LLM — Tokenizer Info\n";
    std::cout << "═══════════════════════════════════════\n";
    std::cout << "  Tipo        : "
              << (tok.type == TokenizerType::SENTENCEPIECE
                  ? "SentencePiece" : "GPT-2 BPE") << "\n";
    std::cout << "  Vocab size  : " << tok.vocab_size    << "\n";
    std::cout << "  Merge rules : " << tok.merges.size() << "\n";
    std::cout << "  BOS id      : " << tok.bos_id        << "\n";
    std::cout << "  EOS id      : " << tok.eos_id        << "\n";
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