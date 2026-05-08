# EIE-LLM — Educational Inference Engine

Motore di inferenza LLM scritto in **C++17 da zero**, con scopo didattico.
Costruito passo passo: dal parsing del file GGUF fino a un server HTTP
compatibile con l'API OpenAI. Zero dipendenze pesanti, solo stdlib + httplib.

---

## Stato del progetto

| Architettura | Stato |
|---|---|
| **GPT-2** (Q8\_0) | Funzionante |
| **LLaMA / TinyLLaMA** (Q4\_K\_M) | **Work in progress — output non corretto** |

> **Nota LLaMA**: il codice per l'architettura LLaMA è presente (RoPE, RMSNorm,
> SwiGLU, GQA, dequantizzazione Q4\_K e Q6\_K) ma il modello non produce ancora
> testo coerente. Le dequantizzazioni sono state corrette e verificate contro
> ggml-quants.c, ma rimane almeno un bug nel forward pass o nel tokenizer
> SentencePiece da individuare.

---

## Roadmap

- [x] Fase 1 — Parser GGUF: header + metadata KV
- [x] Fase 2 — Lettura info tensori
- [x] Fase 3 — Tokenizer BPE (GPT-2 byte-level)
- [x] Fase 4 — Forward pass GPT-2 (LayerNorm, GELU, MHA)
- [x] Fase 5 — Shell interattiva con linenoise
- [x] Fase 6 — Server HTTP minimale
- [x] Fase 7 — Sampling avanzato (top-k, top-p, repetition penalty)
- [x] Fase 8 — Dequantizzazione Q4\_K e Q6\_K
- [x] Fase 9 — Architettura LLaMA (RoPE, RMSNorm, SwiGLU, GQA)
- [ ] Fase 10 — **Debug forward pass LLaMA** *(in corso)*

---

## Cosa abbiamo costruito

| Modulo | File | Cosa fa |
|--------|------|---------|
| Parser GGUF | `gguf.hpp/cpp` | Legge header, metadata KV, info tensori, dati raw dal file binario |
| Tensor ops | `ops.hpp/cpp` | Dequantizzazione F16/Q8\_0/Q4\_K/Q6\_K, fp16→fp32, matmul, matvec, softmax, RMSNorm, RoPE, GELU, SiLU |
| Tokenizer | `tokenizer.hpp/cpp` | BPE encode/decode per GPT-2 (byte-level) e LLaMA (SentencePiece) |
| Modello | `model.hpp/cpp` | Config, pesi, KV cache, LayerNorm/RMSNorm, Self-Attention (MHA e GQA), FFN (GELU e SwiGLU), forward pass |
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

Lo script scarica il modello GPT-2 (~176 MB) e la libreria httplib:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

Per TinyLLaMA (~638 MB), scaricare manualmente:

```bash
wget -P models/ https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf \
     -O models/tinyllama.Q4_K_M.gguf
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

> **Nota build**: se si modificano file sorgente e i cambiamenti non vengono
> rilevati, usare `cmake --build build --clean-first` per forzare la
> ricompilazione completa.

---

## Uso

### Shell interattiva

```bash
# GPT-2 (funzionante)
./build/eie-llm models/gpt2.Q8_0.gguf

# TinyLLaMA (WIP — output non corretto)
./build/eie-llm models/tinyllama.Q4_K_M.gguf
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
│  x ──► LayerNorm2 ──► FFN (GELU)        │
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

### Forward pass LLaMA (implementato, WIP)

```
token ID
    │
    ▼
[embedding lookup]          solo token embedding (no positional)
    │
    ▼  × 22 layer (TinyLLaMA)
┌─────────────────────────────────────────┐
│                                         │
│  x ──► RMSNorm1 ──► Self-Attention      │
│  │         RoPE su Q e K                │
│  │         GQA: 32 head Q / 4 head KV   │
│  │                         │            │
│  └─────────── + ◄──────────┘            │  residual connection
│               │                         │
│  x ──► RMSNorm2 ──► FFN (SwiGLU)        │
│  │         gate = SiLU(W1·x)            │
│  │         out  = (gate ⊙ W3·x) · W2    │
│  └─────────── + ◄──────────┘            │  residual connection
│                                         │
└─────────────────────────────────────────┘
    │
    ▼
[RMSNorm finale]
    │
    ▼
[output.weight]             lm_head separato (no weight tying)
    │
    ▼
logits [32000] → sampling → token successivo
```

### Self-Attention con KV Cache (GPT-2)

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

### Quantizzazione — formati supportati

**Q8\_0** (usato da GPT-2):
```
Blocco: 2 byte scale (float16) + 32 byte int8
Dequantizzazione: val = int8 * scale
```

**Q4\_K** (usato da TinyLLaMA):
```
Super-block: 256 elementi, 144 byte
  2 byte d (scale globale scales), 2 byte dmin (scale globale minimi)
  12 byte: 8 scale + 8 minimi compressi a 6 bit ciascuno
  128 byte qs: nibble 4 bit, due sub-block per ogni 32 byte
    low nibble  → sub-block pari,  elemento l
    high nibble → sub-block dispari, elemento l
Dequantizzazione: val = nibble * (sv * d) - (mv * dmin)
```

**Q6\_K** (usato da TinyLLaMA per output.weight e attn_v):
```
Super-block: 256 elementi, 210 byte
  128 byte ql: 4 bit bassi di ogni elemento
   64 byte qh: 2 bit alti per 4 elementi/byte (stride 32)
   16 byte scales: int8 × 16 scale
    2 byte d: super-scale float16
Dequantizzazione: q6 = (low4 | high2<<4) - 32
                  val = d * scale[k] * q6
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
│   ├── tokenizer.hpp     # Tokenizer BPE (GPT-2 e SentencePiece)
│   ├── model.hpp         # Strutture modello e forward pass
│   ├── shell.hpp         # Shell interattiva
│   └── server.hpp        # Server HTTP
├── src/
│   ├── main.cpp          # Entry point
│   ├── gguf.cpp          # Parser GGUF
│   ├── ops.cpp           # Dequantizzazione e operazioni vettoriali
│   ├── tokenizer.cpp     # BPE encode/decode
│   ├── model.cpp         # Forward pass GPT-2 e LLaMA
│   ├── shell.cpp         # Shell interattiva
│   └── server.cpp        # Server HTTP
├── models/               # Modelli GGUF (non inclusi nel repo)
│   └── .gitkeep
├── scripts/
│   └── setup.sh          # Download modello + dipendenze
├── tools/
│   ├── inspect_gguf.py   # Ispezione file GGUF
│   ├── debug_q4k.py      # Debug dequantizzazione Q4_K
│   └── verify_q4k.py     # Verifica Q4_K contro riferimento ggml
└── third_party/          # httplib.h (non incluso nel repo)
```

---

## Modelli

### GPT-2 small (funzionante)

124M parametri, architettura transformer decoder-only.

| Parametro | Valore |
|-----------|--------|
| `n_vocab` | 50257 |
| `n_ctx` | 1024 |
| `n_embd` | 768 |
| `n_head` | 12 |
| `n_layer` | 12 |
| `n_ff` | 3072 |
| `d_head` | 64 |
| Quantizzazione | Q8\_0 |
| Dimensione file | ~176 MB |

### TinyLLaMA 1.1B Chat (WIP)

1.1B parametri, architettura LLaMA con GQA.

| Parametro | Valore |
|-----------|--------|
| `n_vocab` | 32000 |
| `n_ctx` | 2048 |
| `n_embd` | 2048 |
| `n_head` | 32 |
| `n_head_kv` | 4 (GQA) |
| `n_layer` | 22 |
| `n_ff` | 5632 |
| `d_head` | 64 |
| `rope_dim` | 64 |
| Quantizzazione | Q4\_K\_M |
| Dimensione file | ~638 MB |

---

## Prossimi passi

| Obiettivo | Priorità |
|-----------|----------|
| **Debug LLaMA** — individuare il bug nel forward pass | Alta |
| **Tokenizer SentencePiece** — verificare encoding LLaMA | Alta |
| **Performance** — AVX2/NEON per matmul, thread pool | Media |
| **Streaming HTTP** — Server-Sent Events per output token per token | Bassa |
| **Memoria** — Quantizzazione KV cache, sliding window attention | Bassa |

---

## Licenza

Progetto didattico — MIT License.
