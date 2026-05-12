#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "gguf.hpp"

// ─────────────────────────────────────────────
//  Tipo di tokenizer
// ─────────────────────────────────────────────
enum class TokenizerType {
    GPT2,         // BPE byte-level, spazio = Ġ (U+0120)
    SENTENCEPIECE // BPE con SentencePiece, spazio = ▁ (U+2581)
};

// ─────────────────────────────────────────────
//  Regola di merge BPE
// ─────────────────────────────────────────────
struct BPEMerge {
    std::string first;
    std::string second;
    int         rank;
};

// ─────────────────────────────────────────────
//  Tokenizer
// ─────────────────────────────────────────────
struct Tokenizer {
    TokenizerType type = TokenizerType::GPT2;

    // Vocabolario bidirezionale
    std::vector<std::string>             id_to_token;
    std::unordered_map<std::string, int> token_to_id;

    // Regole BPE (usate solo per GPT-2)
    std::vector<BPEMerge>                merges;
    std::unordered_map<std::string, int> merge_rank;

    // Scores (SentencePiece unigram) — log-prob di ogni token.
    // Usati dall'algoritmo Viterbi per trovare la segmentazione
    // a massima verosimiglianza.
    std::vector<float> scores;

    // Tipi token (SentencePiece)
    // 1=normale, 2=unknown, 3=controllo, 6=byte
    std::vector<int> token_type;

    // Token speciali
    int bos_id     = 1;   // Begin of Sequence
    int eos_id     = 2;   // End of Sequence
    int unk_id     = 0;   // Unknown

    int vocab_size = 0;

    // Lunghezza massima di un token in byte — calcolata a init time
    // per limitare la finestra di ricerca del Viterbi.
    int max_token_len = 64;

    // ── Chat template ─────────────────────────
    // Template Jinja2 grezzo letto dal campo GGUF "tokenizer.chat_template".
    // Non viene interpretato: serve solo per rilevare il formato.
    std::string chat_template;

    // Stringa EOS, es. "</s>" per LLaMA. Inserita nel template tra
    // il messaggio utente e il tag <|assistant|>.
    std::string eos_token_str;

    // true se il GGUF contiene un chat template riconoscibile.
    bool has_chat_template = false;

    // Token speciali (controllo) per modelli come Llama-3.
    // Popolati durante l'init: tutti i token con token_type == 3,
    // ordinati per lunghezza decrescente per il matching greedy.
    std::vector<std::pair<std::string, int>> special_tokens;
};

// Inizializza il tokenizer dai metadata GGUF
bool tokenizer_init(Tokenizer& tok, const GGUFContext& ctx);

// Encode: testo → ID
std::vector<int> tokenizer_encode(const Tokenizer& tok,
                                  const std::string& text,
                                  bool add_bos = true);

// Decode: ID → testo
std::string tokenizer_decode(const Tokenizer& tok,
                             const std::vector<int>& ids);

// Applica il chat template al messaggio utente.
//
// Legge il pattern dal campo chat_template del tokenizer e
// costruisce il prompt completo pronto per la tokenizzazione.
// Supporta TinyLlama (<|user|>) e ChatML (<|im_start|>).
// Se system_msg è vuoto viene omesso.
//
// Esempio TinyLlama:
//   user_msg = "Ciao"
//   → "<|user|>\nCiao</s>\n<|assistant|>\n"
std::string apply_chat_template(const Tokenizer& tok,
                                const std::string& user_msg,
                                const std::string& system_msg = "");

// Applica il chat template a una conversazione multi-turn.
//
// Ogni messaggio è una coppia {role, content} dove role può essere
// "system", "user" o "assistant". Costruisce il prompt completo
// con tutta la history della conversazione.
//
// Esempio ChatML con 2 turni:
//   [{"user","Ciao"},{"assistant","Ciao!"},{"user","Come stai?"}]
//   → "<|im_start|>user\nCiao<|im_end|>\n<|im_start|>assistant\nCiao!<|im_end|>\n"
//      "<|im_start|>user\nCome stai?<|im_end|>\n<|im_start|>assistant\n"
std::string apply_chat_template_conversation(
    const Tokenizer& tok,
    const std::vector<std::pair<std::string, std::string>>& messages);

// Info debug
void tokenizer_print_info(const Tokenizer& tok);