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
| `:tokens <n>` | Imposta il numero massimo di token generati |
| `:temp <f>` | Imposta la temperatura (es. `0.8`) |
| `:greedy` | Attiva sampling greedy (argmax) |
| `:sample` | Attiva sampling con temperatura |
| `:reset` | Azzera la KV cache |
| `:quit` | Esci |
| `qualsiasi testo` | Genera il completamento |

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

Verifica che il server sia attivo.

```bash
curl http://localhost:8080/health
```

```json
{"status":"ok","model":"gpt2"}
```

### `POST /v1/completions`

Genera testo a partire da un prompt.
Formato compatibile con l'API OpenAI.

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The meaning of life is",
    "max_tokens": 50,
    "temperature": 0.8
  }'
```

**Parametri della richiesta:**

| Campo | Tipo | Default | Descrizione |
|-------|------|---------|-------------|
| `prompt` | string | — | Testo di input (obbligatorio) |
| `max_tokens` | int | 50 | Token massimi da generare (max 500) |
| `temperature` | float | 1.0 | Temperatura del sampling (0 = greedy) |

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

**Esempio con Python:**

```python
import urllib.request, json

data = json.dumps({
    "prompt": "Hello, I am",
    "max_tokens": 30,
    "temperature": 0.8
}).encode()

req = urllib.request.Request(
    "http://localhost:8080/v1/completions",
    data=data,
    headers={"Content-Type": "application/json"}
)

response = json.loads(urllib.request.urlopen(req).read())
print(response["choices"][0]["text"])
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
        scores[t] = Q_h · K_h[t] / √64   t∈[0,pos]│
        scores    = softmax(scores)                 │
        out_h     = Σ scores[t] * V_h[t]  ◄────────┘
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

Blocco Q8_0 (34 byte):
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
| **Qualità output** | Top-p sampling, repetition penalty |
| **Memoria** | Quantizzazione KV cache, sliding window attention |
| **Streaming HTTP** | Server-Sent Events per output token per token |
| **Benchmark** | Misura tokens/sec, latenza prefill vs generazione |

---

## Licenza

Progetto didattico — MIT License.