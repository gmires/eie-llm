#pragma once
#include "model.hpp"
#include "tokenizer.hpp"

// ─────────────────────────────────────────────
//  Shell interattiva per EIE-LLM
//
//  Fornisce un'interfaccia a riga di comando
//  per interagire col modello. Supporta:
//  - generazione di testo da prompt
//  - configurazione temperatura e max token
//  - reset della KV cache
//  - sampling greedy o probabilistico
// ─────────────────────────────────────────────

// Avvia il loop interattivo della shell.
// Blocca finché l'utente non digita :quit
// o invia EOF (Ctrl+D).
void shell_run(Model& model, const Tokenizer& tok);