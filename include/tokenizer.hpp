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

    // Regole BPE
    std::vector<BPEMerge>                merges;
    std::unordered_map<std::string, int> merge_rank;

    // Scores (SentencePiece) — log-prob di ogni token
    std::vector<float> scores;

    // Tipi token (SentencePiece)
    // 1=normale, 2=unknown, 3=controllo, 6=byte
    std::vector<int> token_type;

    // Token speciali
    int bos_id     = 1;   // Begin of Sequence
    int eos_id     = 2;   // End of Sequence
    int unk_id     = 0;   // Unknown

    int vocab_size = 0;
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

// Info debug
void tokenizer_print_info(const Tokenizer& tok);