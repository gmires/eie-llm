# EIE-LLM — Educational Inference Engine

Motore di inferenza per modelli linguistici (LLM) scritto in **C++17 da zero**, con scopo **didattico**.

L'obiettivo non è la velocità o la completezza: è capire dall'interno come funziona un LLM, costruendo pezzo per pezzo tutto il necessario — dalla lettura del file binario fino alla generazione di testo nella shell.

---

## Cosa fa questo progetto

Prende un file `.gguf` (il formato usato da llama.cpp per distribuire modelli) e lo esegue interamente in C++, senza dipendere da framework come PyTorch o ONNX.

Puoi:
- **Chattare** con TinyLlama dalla riga di comando
- **Completare testo** con GPT-2
- **Interrogarlo via HTTP** con un'API compatibile OpenAI
- **Misurare le prestazioni** con la modalità benchmark

---

## Modelli supportati

| Modello | Stato | File |
|---------|-------|------|
| **GPT-2 small** (124M, Q8\_0) | Funzionante | `models/gpt2.Q8_0.gguf` |
| **TinyLlama 1.1B Chat** (Q4\_K\_M) | Funzionante | `models/tinyllama.Q4_K_M.gguf` |

---

## Requisiti

- CMake ≥ 3.16
- GCC ≥ 13 oppure Clang ≥ 15 (serve C++17)
- `wget` o `curl` per il download dei modelli

---

## Setup e build

```bash
# 1. Scarica GPT-2 (~176 MB) e la libreria httplib
chmod +x scripts/setup.sh && ./scripts/setup.sh

# 2. Scarica TinyLlama (~638 MB)
wget -P models/ \
  https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf \
  -O models/tinyllama.Q4_K_M.gguf

# 3. Compila in Release (necessario per prestazioni accettabili)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

> **Nota:** compilare in **Release** invece di Debug dà un guadagno immediato di 2-4x grazie alle ottimizzazioni del compilatore (SIMD automatico, inlining, eliminazione dei controlli di debug).

---

## Utilizzo

### Shell interattiva

```bash
./build/eie-llm models/tinyllama.Q4_K_M.gguf
./build/eie-llm models/gpt2.Q8_0.gguf
```

TinyLlama si avvia automaticamente in **modalità chat** perché ha un template nel file GGUF. GPT-2 parte in modalità raw (completamento libero).

**Comandi disponibili:**

| Comando | Descrizione |
|---------|-------------|
| `:chat` | Modalità chat — avvolge l'input nel template `<\|user\|>` |
| `:raw` | Modalità raw — prompt passato direttamente al modello |
| `:tokens <n>` | Max token da generare (default: 200) |
| `:temp <f>` | Temperatura del sampling, es. `0.8` (default: 1.0) |
| `:topk <n>` | Top-k sampling — mantieni solo i k token più probabili (default: 40) |
| `:topp <f>` | Nucleus sampling — taglia al p% cumulativo (default: 0.9) |
| `:penalty <f>` | Repetition penalty — scoraggia le ripetizioni (default: 1.1) |
| `:greedy` | Sampling deterministico (sceglie sempre il token più probabile) |
| `:sample` | Sampling stocastico — top-k + top-p (default) |
| `:params` | Mostra i parametri correnti |
| `:reset` | Svuota la KV cache (ricomincia da capo) |
| `:quit` | Esci |

**Esempio:**
```
eie> Qual è la capitale della Francia?
Assistente: La capitale della Francia è Parigi.
```

### Benchmark

```bash
./build/eie-llm models/tinyllama.Q4_K_M.gguf --bench 50
```

Genera 50 token e misura token/sec, latenza del prefill e tempo di caricamento.

### Server HTTP

```bash
# Porta default 8080
./build/eie-llm models/tinyllama.Q4_K_M.gguf --server

# Porta custom
./build/eie-llm models/tinyllama.Q4_K_M.gguf --server 9090
```

---

## API HTTP

### `GET /health`

Controlla che il server sia attivo.

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

### `POST /v1/completions`

Completamento di testo. Con `"chat": true` applica il template del modello.

```bash
# Modalità raw (GPT-2, completamento libero)
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":20}'

# Modalità chat (TinyLlama)
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Qual è la capitale della Francia?","chat":true,"max_tokens":50}'
```

**Parametri:**

| Campo | Tipo | Default | Descrizione |
|-------|------|---------|-------------|
| `prompt` | stringa | — | Testo di input (obbligatorio) |
| `chat` | booleano | `false` | Se `true`, applica il chat template del modello |
| `max_tokens` | intero | 50 | Token massimi (limite: 500) |
| `temperature` | float | 1.0 | Temperatura del sampling |
| `top_k` | intero | 40 | Top-k (0 = disabilitato) |
| `top_p` | float | 0.9 | Nucleus sampling |
| `repetition_penalty` | float | 1.1 | Penalità ripetizioni (1.0 = nessuna) |

### `POST /v1/chat/completions`

Endpoint compatibile con l'API OpenAI. Accetta un array di messaggi ed estrae automaticamente l'ultimo messaggio `user`.

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Qual è la capitale della Francia?"}
    ],
    "max_tokens": 100
  }'
```

**Risposta (entrambi gli endpoint):**

```json
{
  "object": "text_completion",
  "choices": [{"text": "La capitale della Francia è Parigi.", "index": 0, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21}
}
```

---

## Struttura del codice

```
eie-llm/
├── src/
│   ├── main.cpp        — punto di ingresso, sceglie shell/server/bench
│   ├── gguf.cpp        — legge il file .gguf (header, metadata, tensori)
│   ├── ops.cpp         — operazioni matematiche (matmul, softmax, RoPE…)
│   ├── tokenizer.cpp   — converte testo ↔ ID numerici
│   ├── model.cpp       — forward pass (calcola i logits da un token)
│   ├── shell.cpp       — shell interattiva
│   └── server.cpp      — server HTTP
├── include/            — header corrispondenti
├── models/             — file .gguf (non inclusi nel repo)
├── scripts/setup.sh    — download modelli e dipendenze
└── tools/              — script Python per debug e ispezione GGUF
```

---

## Come funziona internamente

### 1. Il file GGUF

I modelli sono distribuiti nel formato **GGUF** (un file binario strutturato). Contiene:
- **Metadata**: architettura, dimensioni, tipo di tokenizer, chat template…
- **Tensori**: i pesi del modello, eventualmente compressi (quantizzati)

`gguf.cpp` legge tutto questo e lo carica in RAM.

### 2. Quantizzazione — perché i modelli sono così piccoli

Un modello con 1.1 miliardi di parametri in float32 occuperebbe ~4.4 GB. TinyLlama in Q4\_K\_M pesa ~638 MB perché ogni peso è compresso a ~4 bit invece di 32.

`ops.cpp` decomprime i pesi al volo prima di usarli. I formati supportati:

| Formato | Bit/peso | Usato da |
|---------|----------|----------|
| F32 | 32 | embedding di GPT-2 |
| F16 | 16 | alcuni layer |
| Q8\_0 | 8 | GPT-2 |
| Q4\_K | 4 | TinyLlama (pesi principali) |
| Q6\_K | 6 | TinyLlama (output e attention V) |

> **Nota:** nella versione attuale tutti i pesi vengono dequantizzati in float32 **al caricamento**, occupando circa 4.2 GB in RAM. Vedi la sezione [Prestazioni e limiti](#prestazioni-e-limiti) per i dettagli.

### 3. Il Tokenizer

Il modello non capisce le parole, solo numeri. Il tokenizer converte testo → lista di ID interi.

**GPT-2** usa un algoritmo BPE (Byte Pair Encoding): le parole vengono spezzate in pezzi frequenti, es. "playing" → ["play", "ing"].

**TinyLlama** usa SentencePiece con algoritmo **Viterbi unigram**: invece di applicare regole di merge fisse, trova la segmentazione che massimizza la somma delle log-probabilità di ogni token. Ogni token ha uno "score" (log-prob) nei metadata GGUF; il Viterbi trova il percorso ottimale con programmazione dinamica.

Entrambi aggiungono un simbolo speciale per gli spazi: GPT-2 usa `Ġ` (U+0120), TinyLlama usa `▁` (U+2581).

### 4. Il Forward Pass — come il modello produce il testo

Dato un token (un numero), il modello produce una distribuzione di probabilità sul prossimo token. Questo avviene in più "layer" sovrapposti, ognuno composto da:

**Self-Attention** — ogni token "guarda" i token precedenti e decide cosa è rilevante.

**Feed-Forward Network (FFN)** — elabora l'informazione raccolta dall'attention.

I due modelli differiscono nei dettagli:

| Componente | GPT-2 | TinyLlama |
|---|---|---|
| Normalizzazione | LayerNorm | RMSNorm |
| Positional encoding | Embedding assoluto | RoPE (rotazionale) |
| Attention | Multi-Head (MHA) | Grouped Query (GQA, 32Q/4KV) |
| Attivazione FFN | GELU | SiLU + gate (SwiGLU) |

#### La KV Cache

Senza ottimizzazioni, per generare il token in posizione N bisognerebbe ricalcolare l'attention su tutti gli N token precedenti — costo O(N²). La **KV cache** salva le chiavi (K) e i valori (V) già calcolati, così ogni nuovo token costa O(1) invece di O(N).

#### Buffer di inferenza pre-allocati

Ogni forward step necessita di vettori temporanei: Q, K, V, gate, up, scores… Allocarli e deallocarli ad ogni token significherebbe centinaia di `malloc`/`free` per token generato. La struct `InferBuffers` in `model.hpp` li pre-alloca una sola volta all'avvio e li riusa ad ogni step.

Un dettaglio importante: l'accumulatore interno della attention (`attn_acc`, che raccoglie la weighted sum dei V) e il buffer di output finale (`attn_out`, scritto dal `matvec` di proiezione) **devono essere buffer distinti**. Se coincidessero, il `matvec` scriverebbe sull'input mentre lo legge ancora, corrompendo il risultato (aliasing).

### 5. Il Chat Template

TinyLlama è un modello addestrato per le conversazioni. Si aspetta un formato preciso:

```
<|user|>
{messaggio dell'utente}</s>
<|assistant|>
```

Il file GGUF include il template in formato Jinja2. `apply_chat_template` in `tokenizer.cpp` rileva il formato dal template e lo applica senza dover implementare un interprete Jinja2 completo. Supporta anche il formato ChatML (`<|im_start|>`) usato da Mistral e Qwen.

### 6. Il Sampling — come si sceglie il prossimo token

Il modello produce un vettore di ~32000 logit (uno per token). Per scegliere il prossimo token:

**Greedy**: prende il massimo. Deterministico, tende a ripetersi.

**Top-k**: mantiene solo i k token più probabili, campiona da questi. Evita token improbabili senza essere deterministico.

**Top-p (nucleus)**: mantiene il sottoinsieme minimo di token la cui probabilità cumulativa supera p. Adattivo: quando c'è un token molto probabile, il nucleus è piccolo; quando la distribuzione è uniforme, il nucleus è più ampio.

**Repetition penalty**: penalizza i token già apparsi nel contesto, dividendo/moltiplicando il loro logit per un fattore > 1.

---

## Prestazioni e limiti

Questa è un'implementazione **didattica**: correttezza e leggibilità del codice vengono prima delle prestazioni. Detto questo, è utile capire dove va il tempo e la memoria.

### Memoria — il problema della dequantizzazione anticipata

Il file GGUF di TinyLlama occupa 638 MB su disco (Q4\_K\_M, ~4.5 bit/peso). Nella versione attuale tutti i pesi vengono convertiti in float32 al caricamento:

| Tensore | float32 |
|---|---|
| token\_embd + output.weight | ~512 MB |
| 22 × (attn Q/K/V/out) | ~440 MB |
| 22 × (ffn gate + up + down) | ~2.9 GB |
| **Totale** | **~4.2 GB** |

La soluzione corretta è **mantenere i pesi nel formato quantizzato originale** e dequantizzare per super-block durante il forward pass. In questo modo si occuperebbero solo ~638 MB e il forward sarebbe anche più veloce (meno dati da leggere dalla RAM, il bandwidth è spesso il vero collo di bottiglia).

### Velocità — il collo di bottiglia è `matvec`

Per ogni token generato, TinyLlama esegue circa **970 milioni di multiply-accumulate** solo nelle proiezioni lineari (Q/K/V, FFN):

| Operazione | MAC per token |
|---|---|
| attn Q + K + V + out (×22 layer) | ~200M |
| ffn gate + up + down (×22 layer) | ~760M |
| **Totale** | **~970M** |

Con il `matvec` scalare attuale (~1 GFLOPS) → circa **1 secondo/token**.

**Miglioramenti possibili, dal più semplice al più complesso:**

| Intervento | Come | Guadagno atteso |
|---|---|---|
| Build Release + `-march=native` | `cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-march=native"` | 2-4x (SIMD automatico) |
| OpenMP su `matvec` | `#pragma omp parallel for` sul loop esterno + `-fopenmp` | Nx (N = core) |
| AVX2 esplicito | Intrinsics `_mm256_*` nel loop interno di `matvec` | 6-8x |
| Pesi quantizzati in RAM | Riscrivere `matvec` per lavorare su byte Q4\_K/Q6\_K | −3.6 GB RAM + 20-40% velocità |

---

## Roadmap

- [x] Fase 1 — Parser GGUF: header + metadata KV
- [x] Fase 2 — Lettura info tensori e pesi
- [x] Fase 3 — Tokenizer BPE (GPT-2 byte-level)
- [x] Fase 4 — Forward pass GPT-2 (LayerNorm, GELU, MHA)
- [x] Fase 5 — Shell interattiva con linenoise
- [x] Fase 6 — Server HTTP minimale
- [x] Fase 7 — Sampling avanzato (top-k, top-p, repetition penalty)
- [x] Fase 8 — Dequantizzazione Q4\_K e Q6\_K
- [x] Fase 9 — Architettura LLaMA (RoPE, RMSNorm, SwiGLU, GQA)
- [x] Fase 10 — Fix dequantizzazione Q6\_K (bug indici scale)
- [x] Fase 11 — Tokenizer SentencePiece Viterbi unigram
- [x] Fase 12 — Chat template (shell + server HTTP)
- [x] Fase 13 — Buffer di inferenza pre-allocati (eliminazione malloc nel hot path)
- [ ] Fase 14 — Pesi quantizzati in RAM (−3.6 GB, matvec su Q4\_K/Q6\_K)
- [ ] Fase 15 — Ottimizzazioni matmul (AVX2/NEON, OpenMP)
- [ ] Fase 16 — Streaming HTTP (Server-Sent Events, token per token)

---

## Licenza

Progetto didattico — MIT License.
