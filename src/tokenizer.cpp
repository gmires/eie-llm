#include "tokenizer.hpp"
#include <iostream>
#include <algorithm>
#include <climits>
#include <iomanip>

// ─────────────────────────────────────────────
//  Helper: cerca un array nei metadata GGUF
//  data la chiave. Ritorna nullptr se non trovato
//  o se il valore non è un GGUFArray.
// ─────────────────────────────────────────────
static const GGUFArray* find_array(const GGUFContext& ctx,
                                   const std::string& key) {
    for (const auto& kv : ctx.metadata)
        if (kv.key == key)
            return kv.value.get_if<GGUFArray>();
    return nullptr;
}

// ─────────────────────────────────────────────
//  Inizializzazione tokenizer
//
//  I dati del tokenizer GPT-2 sono nei metadata
//  GGUF con queste chiavi:
//
//  "tokenizer.ggml.tokens"
//      array di stringhe — il vocabolario completo
//      50257 token, uno per indice
//
//  "tokenizer.ggml.merges"
//      array di stringhe — le regole BPE
//      formato: "token1 token2" (separati da spazio)
//      l'indice nell'array è il rank della regola
// ─────────────────────────────────────────────
bool tokenizer_init(Tokenizer& tok, const GGUFContext& ctx) {

    // ── Carica il vocabolario ─────────────────
    const GGUFArray* tokens_arr = find_array(ctx, "tokenizer.ggml.tokens");
    if (!tokens_arr) {
        std::cerr << "[ERRORE] tokenizer.ggml.tokens non trovato\n";
        return false;
    }

    tok.vocab_size = static_cast<int>(tokens_arr->values.size());
    tok.id_to_token.reserve(tok.vocab_size);
    tok.token_to_id.reserve(tok.vocab_size);

    for (int id = 0; id < tok.vocab_size; id++) {
        // Ogni elemento è una stringa — usiamo get_if
        // sul GGUFValue per estrarlo in modo type-safe
        const auto* s = tokens_arr->values[id].get_if<std::string>();
        if (!s) {
            std::cerr << "[ERRORE] Token #" << id << " non è una stringa\n";
            return false;
        }
        tok.id_to_token.push_back(*s);
        tok.token_to_id[*s] = id;
    }

    // ── Carica le regole di merge BPE ─────────
    const GGUFArray* merges_arr = find_array(ctx, "tokenizer.ggml.merges");
    if (!merges_arr) {
        std::cerr << "[ERRORE] tokenizer.ggml.merges non trovato\n";
        return false;
    }

    tok.merges.reserve(merges_arr->values.size());

    for (int rank = 0; rank < (int)merges_arr->values.size(); rank++) {
        const auto* s = merges_arr->values[rank].get_if<std::string>();
        if (!s) continue;

        // Ogni regola è "primo secondo" — dividi sullo spazio
        size_t space = s->find(' ');
        if (space == std::string::npos) continue;

        BPEMerge merge;
        merge.first  = s->substr(0, space);
        merge.second = s->substr(space + 1);
        merge.rank   = rank;

        // Chiave lookup: "primo secondo"
        // Ci permette di trovare il rank in O(1)
        // durante l'algoritmo di merge
        tok.merge_rank[merge.first + " " + merge.second] = rank;
        tok.merges.push_back(std::move(merge));
    }

    return true;
}

// ─────────────────────────────────────────────
//  Pre-tokenizzazione
//
//  GPT-2 usa un pre-tokenizer che:
//  1) Non divide mai dentro le parole
//  2) Tratta la punteggiatura come token separati
//  3) Rappresenta lo spazio che precede una parola
//     con il carattere Ġ (U+0120, UTF-8: 0xC4 0xA0)
//
//  Esempi:
//    "Hello world"  → ["Hello", "Ġworld"]
//    "Hello, world" → ["Hello", ",", "Ġworld"]
//
//  Questa versione è semplificata ma corretta
//  per la maggior parte dei casi comuni.
// ─────────────────────────────────────────────
static std::vector<std::string> pretokenize(const std::string& text) {
    // Ġ in UTF-8 = 0xC4 0xA0
    // GPT-2 lo usa per marcare "spazio + parola"
    static const std::string SPACE_SYMBOL = "\xc4\xa0";

    std::vector<std::string> words;
    std::string current;

    for (size_t i = 0; i < text.size(); i++) {
        char c = text[i];

        if (c == ' ') {
            // Salva la parola corrente se non vuota
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            // Il prossimo token inizierà con Ġ
            current = SPACE_SYMBOL;
        } else if (ispunct((unsigned char)c) && c != '\'') {
            // La punteggiatura (tranne apostrofo) è
            // un token separato in GPT-2
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            words.push_back(std::string(1, c));
        } else {
            current += c;
        }
    }

    if (!current.empty())
        words.push_back(current);

    return words;
}

// ─────────────────────────────────────────────
//  Divide una parola in caratteri UTF-8
//
//  UTF-8 usa 1-4 byte per carattere.
//  Dobbiamo dividere per carattere (non per byte)
//  altrimenti spezziamo sequenze multibyte come Ġ.
//
//  Schema byte iniziale UTF-8:
//    0xxxxxxx → 1 byte  (ASCII)
//    110xxxxx → 2 byte
//    1110xxxx → 3 byte
//    11110xxx → 4 byte
// ─────────────────────────────────────────────
static std::vector<std::string> split_into_chars(const std::string& word) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < word.size(); ) {
        unsigned char c = static_cast<unsigned char>(word[i]);
        int len = 1;
        if      (c >= 0xF0) len = 4;
        else if (c >= 0xE0) len = 3;
        else if (c >= 0xC0) len = 2;
        chars.push_back(word.substr(i, len));
        i += len;
    }
    return chars;
}

// ─────────────────────────────────────────────
//  Algoritmo BPE su una singola parola
//
//  Dato un vettore di token (inizialmente i
//  singoli caratteri della parola), applica
//  ripetutamente la merge con rank più basso
//  finché non ce ne sono più di applicabili.
//
//  Complessità: O(n² × m) nel caso peggiore
//  dove n = token attuali, m = regole di merge.
//  Con il lookup O(1) via merge_rank diventa
//  O(n²) per parola — accettabile per GPT-2.
// ─────────────────────────────────────────────
static void apply_bpe(std::vector<std::string>& tokens,
                      const Tokenizer& tok) {
    while (tokens.size() > 1) {
        // Cerca la coppia adiacente con rank minimo
        int    best_rank = INT_MAX;
        size_t best_idx  = 0;
        bool   found     = false;

        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            // Costruiamo la chiave "tok1 tok2" per il lookup
            std::string key = tokens[i] + " " + tokens[i + 1];
            auto it = tok.merge_rank.find(key);
            if (it != tok.merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx  = i;
                found     = true;
            }
        }

        // Nessuna merge applicabile → la parola è tokenizzata
        if (!found) break;

        // Unisci la coppia trovata in un unico token
        tokens[best_idx] = tokens[best_idx] + tokens[best_idx + 1];
        tokens.erase(tokens.begin() + best_idx + 1);
    }
}

// ─────────────────────────────────────────────
//  Encode: testo → sequenza di ID
//
//  Pipeline completa:
//  testo → pretokenize → per ogni parola:
//    → split in caratteri UTF-8
//    → apply_bpe
//    → lookup ID nel vocabolario
//  → sequenza finale di ID
// ─────────────────────────────────────────────
std::vector<int> tokenizer_encode(const Tokenizer& tok,
                                  const std::string& text) {
    if (text.empty()) return {};

    std::vector<int> result;

    // Fase 1: dividi il testo in parole
    std::vector<std::string> words = pretokenize(text);

    for (const auto& word : words) {
        // Fase 2: parti dai singoli caratteri UTF-8
        std::vector<std::string> tokens = split_into_chars(word);

        // Fase 3: applica le regole BPE
        apply_bpe(tokens, tok);

        // Fase 4: converti ogni token nel suo ID
        for (const auto& t : tokens) {
            auto it = tok.token_to_id.find(t);
            if (it != tok.token_to_id.end()) {
                result.push_back(it->second);
            } else {
                // Token non nel vocabolario — non dovrebbe
                // mai succedere con BPE byte-level completo
                std::cerr << "[AVVISO] Token sconosciuto: \"" << t << "\"\n";
                result.push_back(0);
            }
        }
    }

    return result;
}

// ─────────────────────────────────────────────
//  Decode: sequenza di ID → testo
//
//  Molto più semplice dell'encode:
//  1) Ogni ID → stringa token via id_to_token
//  2) Concatena tutto
//  3) Sostituisci Ġ (0xC4 0xA0) con spazio reale
//
//  Nota: il primo token di una frase spesso
//  NON ha Ġ iniziale (non è preceduto da spazio)
//  quindi la decodifica è naturalmente corretta.
// ─────────────────────────────────────────────
std::string tokenizer_decode(const Tokenizer& tok,
                             const std::vector<int>& ids) {
    std::string result;
    static const std::string SPACE_SYMBOL = "\xc4\xa0";

    for (int id : ids) {
        if (id < 0 || id >= tok.vocab_size) continue;
        std::string token = tok.id_to_token[id];

        // Sostituisci tutte le occorrenze di Ġ con spazio
        size_t pos = 0;
        while ((pos = token.find(SPACE_SYMBOL, pos)) != std::string::npos) {
            token.replace(pos, SPACE_SYMBOL.size(), " ");
            pos += 1;
        }
        result += token;
    }
    return result;
}

// ─────────────────────────────────────────────
//  Stampa info tokenizer — solo per debug
// ─────────────────────────────────────────────
void tokenizer_print_info(const Tokenizer& tok) {
    std::cout << "\n═══════════════════════════════════════\n";
    std::cout << "  EIE-LLM — Tokenizer Info\n";
    std::cout << "═══════════════════════════════════════\n";
    std::cout << "  Vocab size  : " << tok.vocab_size    << "\n";
    std::cout << "  Merge rules : " << tok.merges.size() << "\n";
    std::cout << "  Primi 10 token del vocabolario:\n";

    for (int i = 0; i < std::min(10, tok.vocab_size); i++) {
        std::cout << "    [" << std::setw(5) << i << "] = \"";
        // Stampa in modo sicuro — salta byte non stampabili
        for (unsigned char c : tok.id_to_token[i])
            if (c >= 32 && c < 127) std::cout << c;
            else std::cout << "\\x" << std::hex
                           << std::setw(2) << std::setfill('0')
                           << (int)c << std::dec;
        std::cout << "\"\n";
    }
    std::cout << "═══════════════════════════════════════\n\n";
}