#pragma once
#include "model.hpp"
#include "tokenizer.hpp"

// ─────────────────────────────────────────────
//  Server HTTP per EIE-LLM
//
//  Espone un'API REST compatibile con il formato
//  OpenAI /v1/completions — questo permette di
//  usare EIE-LLM con qualsiasi client che parla
//  il protocollo OpenAI (curl, Python openai, ecc)
//
//  Endpoint implementati:
//    GET  /health
//         → {"status":"ok","model":"gpt2"}
//
//    POST /v1/completions
//         → genera testo dal prompt
//
//  Formato richiesta /v1/completions:
//    {
//      "prompt": "Hello world",
//      "max_tokens": 50,
//      "temperature": 0.8
//    }
//
//  Formato risposta:
//    {
//      "choices": [{"text": "...testo generato..."}],
//      "model": "gpt2",
//      "usage": {"prompt_tokens": 2, "completion_tokens": 50}
//    }
// ─────────────────────────────────────────────

// Avvia il server HTTP sulla porta specificata.
// Blocca finché il server non viene fermato
// con Ctrl+C.
void server_run(Model& model, const Tokenizer& tok, int port = 8080);