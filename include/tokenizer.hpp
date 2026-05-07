#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "gguf.hpp"

// ─────────────────────────────────────────────
//  Tokenizer BPE per GPT-2
//
//  BPE (Byte Pair Encoding) funziona in 3 fasi:
//
//  1) PRE-TOKENIZZAZIONE
//     Il testo viene diviso in "parole" usando
//     uno schema basato su spazi e punteggiatura.
//     GPT-2 usa il carattere speciale Ġ (U+0120)
//     per rappresentare uno spazio che precede
//     una parola: " hello" → "Ġhello"
//
//  2) ENCODE BPE
//     Ogni parola parte come sequenza di singoli
//     byte/caratteri. Poi si applicano le regole
//     di merge in ordine di priorità (rank):
//     rank basso = alta priorità = applicato prima.
//     Es: "h","e","l","l","o"
//       → merge "h"+"e" = "he"   (rank 100)
//       → merge "l"+"l" = "ll"   (rank 200)
//       → merge "he"+"ll" = "hell" (rank 500)
//       → merge "hell"+"o" = "hello" (rank 800)
//
//  3) LOOKUP ID
//     Ogni token risultante viene cercato nel
//     vocabolario e convertito nel suo ID numerico
// ─────────────────────────────────────────────

// ─────────────────────────────────────────────
//  Regola di merge BPE
//  Dice: "se vedi first seguito da second,
//         uniscili in un unico token"
//  Il rank determina la priorità:
//  rank più basso = applicato prima
// ─────────────────────────────────────────────
struct BPEMerge {
    std::string first;   // primo token della coppia
    std::string second;  // secondo token della coppia
    int         rank;    // priorità (0 = massima)
};

// ─────────────────────────────────────────────
//  Struttura principale del Tokenizer
// ─────────────────────────────────────────────
struct Tokenizer {
    // Vocabolario bidirezionale
    // id_to_token: dato un ID ritorna la stringa
    // token_to_id: data una stringa ritorna l'ID
    std::vector<std::string>             id_to_token;
    std::unordered_map<std::string, int> token_to_id;

    // Regole BPE ordinate per rank
    std::vector<BPEMerge> merges;

    // Lookup O(1) per trovare il rank di una coppia
    // Chiave: "primo secondo" (con spazio separatore)
    // Valore: rank della regola
    std::unordered_map<std::string, int> merge_rank;

    // Dimensione del vocabolario (50257 per GPT-2)
    int vocab_size = 0;
};

// ─────────────────────────────────────────────
//  Funzioni pubbliche del tokenizer
// ─────────────────────────────────────────────

// Inizializza il tokenizer leggendo vocabolario
// e merge rules dai metadata del file GGUF.
// Deve essere chiamata dopo gguf_read_metadata.
bool tokenizer_init(Tokenizer& tok, const GGUFContext& ctx);

// Codifica una stringa in sequenza di token ID.
// "Hello world" → [15496, 995]
std::vector<int> tokenizer_encode(const Tokenizer& tok,
                                  const std::string& text);

// Decodifica una sequenza di ID in stringa.
// [15496, 995] → "Hello world"
std::string tokenizer_decode(const Tokenizer& tok,
                             const std::vector<int>& ids);

// Stampa info e statistiche — solo per debug
void tokenizer_print_info(const Tokenizer& tok);