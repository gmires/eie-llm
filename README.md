# EIE-LLM — Educational Inference Engine

Motore di inferenza LLM scritto in **C++17 da zero**, con scopo didattico.
Costruito passo passo: dal parsing del file GGUF fino a un server HTTP
compatibile con l'API OpenAI. Zero dipendenze pesanti, solo stdlib + httplib.

---

## Roadmap

- [x] Fase 1 — Parser GGUF: header + metadata KV
- [x] Fase 2 — Lettura info tensori
- [x] Fase 3 — Tokenizer BPE
- [x] Fase 4 — Forward pass GPT-2
- [x] Fase 5 — Shell interattiva
- [x] Fase 6 — Server HTTP minimale
- [x] Fase 7 — Sampling avanzato (top-k, top-p, repetition penalty)

---

## Cosa abbiamo costruito

| Modulo | File | Cosa fa |
|--------|------|---------|
| Parser GGUF | `gguf.hpp/cpp` | Legge header, metadata KV, info tensori dal file binario |
| Tensor ops | `ops.hpp/cpp` | Dequantizzazione Q8\_0/F16, matmul, matvec, softmax, GELU |
| Tokenizer | `tokenizer.hpp/cpp` | BPE encode/decode con vocabolario e merge rules da GGUF |
| Modello | `model.hpp/cpp` | Config, pesi, KV cache, LayerNorm, Self-Attention, FFN |
| Shell | `shell.hpp/cpp` | REPL interattiva con comandi e streaming token per token |
| Server | `server.hpp/cpp` | HTTP server con endpoint `/v1/completions` |

---

## Requisiti

- CMake >= 3.16
- GCC >= 13 oppure Clang >= 15
- C++17
- `wget` oppure `curl` (per lo script di setup)

---

## Setup

Lo script scarica il modello GPT-2 (~176MB) e la libreria httplib:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

---

## Build

```bash
# Debug (default)
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Release (più veloce)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

---

## Uso

### Shell interattiva

```bash
./build/eie-llm models/gpt2.Q8_0.gguf
```

Comandi disponibili nella shell:

| Comando | Descrizione |
|---------|-------------|
| `:help` | Mostra la lista dei comandi |
| `:tokens <n>` | Max token da generare (default 200) |
| `:temp <f>` | Temperatura, es. `0.8` (default 1.0) |
| `:topk <n>` | Top-k sampling, `0` = disabilitato (default 40) |
| `:topp <f>` | Nucleus sampling p, es. `0.9` (default 0.9) |
| `:penalty <f>` | Repetition penalty, es. `1.3` (default 1.1) |
| `:greedy` | Attiva sampling greedy (argmax) |
| `:sample` | Attiva top-k + top-p sampling |
| `:params` | Mostra parametri correnti |
| `:reset` | Azzera la KV cache |
| `:quit` | Esci |
| `qualsiasi testo` | Genera il completamento |

**Configurazione consigliata per output bilanciato:**

```
eie> :topk 40
eie> :topp 0.9
eie> :temp 0.8
eie> :penalty 1.2
```

### Server HTTP

```bash
# Porta default 8080
./build/eie-llm models/gpt2.Q8_0.gguf --server

# Porta custom
./build/eie-llm models/gpt2.Q8_0.gguf --server 9090
```

---

## API

### `GET /health`

```bash
curl http://localhost:8080/health
```

```json
{"status":"ok","model":"gpt2"}
```

### `POST /v1/completions`

Formato compatibile con l'API OpenAI.

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The meaning of life is",
    "max_tokens": 50,
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.2
  }'
```

**Parametri della richiesta:**

| Campo | Tipo | Default | Descrizione |
|-------|------|---------|-------------|
| `prompt` | string | — | Testo di input (obbligatorio) |
| `max_tokens` | int | 50 | Token massimi da generare (max 500) |
| `temperature` | float | 1.0 | Temperatura del sampling |
| `top_k` | int | 40 | Mantieni i k token più probabili (0 = disabilitato) |
| `top_p` | float | 0.9 | Nucleus sampling — taglia al p% cumulativo |
| `repetition_penalty` | float | 1.1 | Penalizza token già generati (1.0 = disabilitato) |

**Risposta:**

```json
{
  "object": "text_completion",
  "model": "gpt2",
  "choices": [
    {
      "text": "...testo generato...",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 50,
    "total_tokens": 55
  }
}
```

---

## Sampling — come funziona

### Greedy (argmax)
Sceglie sempre il token con probabilità massima. Deterministico ma tende
a produrre testo ripetitivo.

### Top-k
Mantiene solo i `k` token con logit più alto, azzera tutti gli altri.
Riduce il rischio di campionare token improbabili.

```
k = 40, logits ordinati:
[0.15, 0.12, 0.09, ..., 0.001, 0.0001, ...]
 ──────────────────────────────────────
 tieni solo i primi 40 → campiona da questi
```

### Top-p (nucleus sampling)
Mantiene il sottoinsieme minimo di token la cui probabilità cumulativa
raggiunge `p`. Si adatta dinamicamente alla distribuzione.

```
p = 0.9, probs ordinate:
[0.40, 0.30, 0.15, 0.08, 0.04, 0.02, ...]
 ─────────────────────────
 somma = 0.93 >= 0.9 → taglia qui (4 token)
```

### Top-k + Top-p combinati (default)
Prima applica top-k (tetto duro), poi top-p (affina la distribuzione).
È la configurazione usata da GPT-2 originale.

### Repetition penalty
Penalizza i token già apparsi nel contesto dividendo o moltiplicando
il loro logit per il fattore `penalty`.

```
penalty = 1.3
token già visto con logit +2.0 → +2.0 / 1.3 = +1.54
token già visto con logit -1.0 → -1.0 * 1.3 = -1.30
```

---

## Architettura interna

### Forward pass GPT-2

```
token ID
    │
    ▼
[embedding lookup]
    token_embd[token_id] + pos_embd[pos]
    │
    ▼  × 12 layer
┌─────────────────────────────────────────┐
│                                         │
│  x ──► LayerNorm1 ──► Self-Attention    │
│  │                         │            │
│  └─────────── + ◄──────────┘            │  residual connection
│               │                         │
│  x ──► LayerNorm2 ──► FFN               │
│  │                     │                │
│  └─────────── + ◄──────┘                │  residual connection
│                                         │
└─────────────────────────────────────────┘
    │
    ▼
[LayerNorm finale]
    │
    ▼
[lm_head = token_embd^T]    logits [50257]
    │
    ▼
sampling → token successivo
```

### Self-Attention con KV Cache

```
x [n_embd=768]
    │
    ▼
QKV projection [3×768]
    │
    ├── Q [768] ──────────────────────────────────┐
    ├── K [768] → salva in kv_cache.k[layer][pos] │
    └── V [768] → salva in kv_cache.v[layer][pos] │
                                                   │
    per ogni head h (12 heads × 64 dim):           │
        scores[t] = Q_h · K_h[t] / √64  t∈[0,pos] │
        scores    = softmax(scores)                 │
        out_h     = Σ scores[t] * V_h[t] ◄─────────┘
    │
    ▼
concatena heads → output projection [768]
```

### Quantizzazione Q8\_0

```
Blocco originale (32 float32):
    [f0, f1, f2, ..., f31]

Quantizzazione:
    scale = max(|fi|) / 127.0   → float16 (2 byte)
    qi    = round(fi / scale)   → int8    (1 byte × 32)

Blocco Q8_0 (34 byte totali):
    [scale_f16 | q0 q1 q2 ... q31]

Dequantizzazione:
    fi = qi * scale
```

---

## Struttura del progetto

```
eie-llm/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── gguf.hpp          # Strutture e funzioni GGUF
│   ├── ops.hpp           # Operazioni primitive sui tensori
│   ├── tokenizer.hpp     # Tokenizer BPE
│   ├── model.hpp         # Strutture modello e forward pass
│   ├── shell.hpp         # Shell interattiva
│   └── server.hpp        # Server HTTP
├── src/
│   ├── main.cpp          # Entry point
│   ├── gguf.cpp
│   ├── ops.cpp
│   ├── tokenizer.cpp
│   ├── model.cpp
│   ├── shell.cpp
│   └── server.cpp
├── models/               # Modelli GGUF (non inclusi nel repo)
│   └── .gitkeep
├── scripts/
│   └── setup.sh          # Download modello + dipendenze
└── third_party/          # httplib.h (non incluso nel repo)
```

---

## Modello

GPT-2 small — 124M parametri, architettura transformer decoder-only.

| Parametro | Valore |
|-----------|--------|
| `n_vocab` | 50257 |
| `n_ctx` | 1024 |
| `n_embd` | 768 |
| `n_head` | 12 |
| `n_layer` | 12 |
| `n_ff` | 3072 |
| `d_head` | 64 |
| Formato file | GGUF Q8\_0 |
| Dimensione | ~176 MB |

---

## Prossimi passi possibili

| Obiettivo | Cosa aggiungere |
|-----------|----------------|
| **Performance** | AVX2/NEON per matmul, thread pool per i layer |
| **Modelli più grandi** | Supporto LLaMA (RoPE, SwiGLU, RMSNorm) |
| **Memoria** | Quantizzazione KV cache, sliding window attention |
| **Streaming HTTP** | Server-Sent Events per output token per token |
| **Benchmark** | Misura tokens/sec, latenza prefill vs generazione |

---

## Licenza

Progetto didattico — MIT License.