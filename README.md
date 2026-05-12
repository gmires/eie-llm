# EIE-LLM — Educational Inference Engine

Motore di inferenza per modelli linguistici (LLM) scritto in **C++17 da zero**, con scopo **didattico**.

L'obiettivo non è la velocità o la completezza: è **capire dall'interno** come funziona un LLM, costruendo pezzo per pezzo tutto il necessario — dalla lettura del file binario fino alla generazione di testo nella shell.

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
| **GPT-2 small** (124M, Q8_0) | Funzionante | `models/gpt2.Q8_0.gguf` |
| **TinyLlama 1.1B Chat** (Q4_K_M) | Funzionante | `models/tinyllama.Q4_K_M.gguf` |
| **Llama-3.2-3B-Instruct** (Q4_K_M) | Funzionante | `models/llama-3.2-3b.Q4_K_M.gguf` |
| **Qwen2.5-1.5B-Instruct** (Q4_K_M) | Funzionante | `models/qwen2.5-1.5b.Q4_K_M.gguf` |

---

## Requisiti

- CMake ≥ 3.16
- GCC ≥ 13 oppure Clang ≥ 15 (serve C++17)
- `wget` o `curl` per il download dei modelli

---

## Setup e build — guida passo passo

Questa sezione ti guida dall'installazione zero alla prima risposta del modello. Ogni passo spiega *cosa* stai facendo e *perché*.

### 1. Prerequisiti

Assicurati di avere installato:

| Strumento | Versione minima | A cosa serve |
|-----------|----------------|--------------|
| `cmake` | 3.16 | Genera i file di build |
| `g++` o `clang++` | 13 / 15 | Compilatore C++17 |
| `wget` o `curl` | — | Download modelli e librerie |
| `xxd` | — | Verifica integrità file GGUF |

Su Ubuntu/Debian:
```bash
sudo apt update && sudo apt install cmake g++ wget xxd
```

### 2. Scarica modelli e librerie

Lo script `setup.sh` è il punto di ingresso raccomandato. Se lo lanci **senza argomenti**, ti guida con un menu interattivo:

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

Ti verrà chiesto:
1. **Quale modello scaricare** — puoi scegliere tra GPT-2 (piccolo, inglese), TinyLlama (chat, multilingue) o Llama-3.2-3B (più potente, ottimo italiano). Se è la prima volta, scegli **Llama-3.2-3B** per la migliore esperienza.
2. **Se aggiornare le librerie third_party** — `httplib.h` e `linenoise.hpp` sono già nel repository, ma puoi aggiornarle all'ultima versione.

Se preferisci la modalità automatica (es. in uno script CI):
```bash
./scripts/setup.sh --llama32   # solo Llama-3.2
./scripts/setup.sh --all       # tutti i modelli
./scripts/setup.sh --gpt2      # solo GPT-2 (per test rapidi)
```

### 3. Compila

**Questo passo è fondamentale.** Il progetto è ottimizzato per CPU e la differenza tra Debug e Release è enorme:

```bash
# ✅ Corretto: Release con ottimizzazioni massime
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

`-DCMAKE_BUILD_TYPE=Release` abilita `-O3` (ottimizzazione aggressiva), SIMD automatico, inlining, e rimuove i controlli di debug. Senza questo flag, il modello è **10-50x più lento**.

All'avvio, il programma stampa le ottimizzazioni attive:
```
  Build type: Release (-O3)
  OpenMP:     attivo (multicore)
  AVX2:       sì
  FMA:        sì
```

Se vedi `Debug (-O0)`, ricompila in Release.

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
| `stream` | booleano | `false` | Se `true`, invia token via SSE in tempo reale |

**Streaming (`stream: true`):**

```bash
curl -N -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"The capital of France is","max_tokens":20,"stream":true}'
```

L'opzione `-N` (no-buffer) di `curl` è essenziale per vedere i token man mano che arrivano. La risposta usa il formato SSE:

```
data: {"choices":[{"text":" the","index":0,"finish_reason":null}]}

data: {"choices":[{"text":" city","index":0,"finish_reason":null}]}

data: {"choices":[{"text":"","index":0,"finish_reason":"stop"}]}

data: [DONE]
```

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

**Streaming (`stream: true`):**

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Ciao!"}],
    "max_tokens": 50,
    "stream": true
  }'
```

```
data: {"choices":[{"index":0,"delta":{"content":"Ciao"},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Risposta non-streaming (entrambi gli endpoint):**

```json
{
  "object": "text_completion",
  "choices": [{"text": "La capitale della Francia è Parigi.", "index": 0, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21}
}
```

---

## Client Python

Oltre a `curl`, puoi usare il client Python ufficiale in `client/`.

### Installazione

```bash
cd client
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Esempi

```bash
# Health check
python client/client.py --mode health

# Completamento
python client/client.py --mode complete --prompt "The sky is"

# Chat con streaming
python client/client.py --mode chat --message "Ciao!" --stream

# Attention heatmap (tutti i layer)
python client/client.py --mode inspect --prompt "The cat sat" --plot

# Attention singolo layer/head
python client/client.py --mode inspect --prompt "Hello" --plot --layer 0 --head 0

# Salva heatmap su file
python client/client.py --mode inspect --prompt "Hello" --plot --output heatmap.png
```

Il client supporta tutti i parametri di sampling (`--temperature`, `--top-k`, `--top-p`, `--greedy`) e lo streaming SSE (`--stream`).

Vedi `client/README.md` per la documentazione completa.

---

## Web UI (OpenWebUI-style)

Oltre al client Python, EIE-LLM include un'interfaccia web grafica completa in `webui/`. Si avvia automaticamente insieme al server — basta aprire il browser all'indirizzo del server.

### Avvio

```bash
./build/eie-llm models/tinyllama.Q4_K_M.gguf --server 8080
# Apri http://localhost:8080 nel browser
```

### Chat

L'interfaccia principale è divisa in tre aree:

- **Sidebar sinistra** — cronologia delle conversazioni. Ogni chat ha un titolo tratto dal primo messaggio utente. Puoi crearne di nuove (+), eliminarle (×), importare ed esportare l'intero archivio in JSON.
- **Area centrale** — i messaggi vengono renderizzati in Markdown con syntax highlighting per i blocchi di codice. I token dell'assistente arrivano in streaming via SSE e vengono aggiunti alla bubble in tempo reale.
- **Barra in basso** — textarea per l'input (supporta invio per mandare, Shift+Invio per newline). Il pulsante verde invia, il pulsante rosso ferma la generazione.

### Impostazioni (⚙️)

Cliccando l'ingranaggio si apre un pannello laterale con tutti i parametri di sampling:

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| **Temperatura** | 1.0 | Quanto "creativa" è la risposta. 0 = deterministico (greedy). |
| **Max tokens** | 100 | Numero massimo di token da generare (limite server: 500). |
| **Top-k** | 40 | Campiona solo tra i k token più probabili. 0 = disabilitato. |
| **Top-p** | 0.9 | Nucleus sampling: taglia la coda della distribuzione. |
| **Repetition penalty** | 1.1 | Scoraggia le ripetizioni (>1.0 attivo). |
| **Streaming** | ON | Se disattivato, la risposta arriva tutta insieme (bloccante). |
| **Greedy** | OFF | Sempre il token più probabile, nessuna casualità. |
| **Modalità chat** | ON | Usa `/v1/chat/completions` con il chat template del modello. |

Le impostazioni vengono salvate automaticamente in `localStorage`.

### Attention Heatmap (🔥)

Il pannello attention permette di "guardare dentro" il modello e vedere come i token si "guardano" tra loro. È uno strumento didattico potente per capire il meccanismo della self-attention.

#### Come funziona

1. Inserisci un testo breve (max 100 token) nella textarea e premi **Analizza**.
2. Il server esegue il forward pass e salva i pesi attention per ogni layer e head.
3. La heatmap viene disegnata su un canvas HTML5.

#### Come leggere la matrice

La heatmap è una matrice quadrata dove ogni cella `(q, k)` rappresenta il peso attention del token query `q` sul token key `k`:

- **Righe (Y, Query)** — il token che sta "chiedendo" attenzione (quello corrente).
- **Colonne (X, Key)** — i token precedenti a cui può rivolgersi.
- **Colore** — gradiente da **blu** (peso ~0.0) a **rosso** (peso ~1.0).
- **Valori numerici** — se le celle sono abbastanza grandi, il numero viene disegnato al centro (es. `0.35`).
- **Tooltip** — passando il mouse su una cella compare un popup con: token query, token key, e peso esatto a 4 decimali.

#### Proprietà fondamentali

1. **Ogni riga somma a 1.0** — dopo la softmax, l'attenzione è una distribuzione di probabilità.
2. **Triangolare inferiore** — la causal mask impedisce a un token di guardare al futuro. La parte sopra la diagonale è sempre 0.
3. **La diagonale ha spesso pesi alti** — ogni token si guarda almeno a se stesso.

#### Layer e Head

Un transformer ha molteplici layer (strati) e ogni layer ha molteplici head (teste d'attenzione). Ogni head specializza su pattern diversi:

- **Head locali** — attenzione concentrata sui token vicini (pattern sintattici come aggettivo-sostantivo).
- **Head globali** — attenzione a token lontani (relazioni semantiche come pronome-sostantivo).
- **Head di posizione** — attenzione basata sulla distanza, indipendentemente dal contenuto.

Puoi visualizzare:
- Un **layer e head specifici** — per studiare il comportamento di una singola testa.
- La **media su tutti i layer/head** — per una visione d'insieme stabile e meno rumorosa.

#### Esempio didattico

Con il testo "The cat sat on the mat":
- Il token "sat" (riga) avrà probabilmente un peso alto sulla colonna "cat" (il soggetto del verbo).
- Il token "mat" potrebbe guardare a "the" immediatamente precedente (pattern sintattico).
- Il token "The" (primo) ha unico peso 1.0 su se stesso.

Questo rende concreto il concetto astratto di "self-attention": il modello impara autonomamente quali relazioni tra token sono utili per predire il prossimo.

### Architettura tecnica

- **Zero build step** — HTML/CSS/JS vanilla, nessun bundler.
- **Zero dipendenze locali** — marked.js e highlight.js da CDN.
- **SPA** — logica tutta in `app.js` (~750 righe).
- **File statici** — serviti da `httplib::Server::set_mount_point("/", "./webui")`.
- **Streaming SSE** — `fetch()` con `ReadableStream` e parsing manuale degli eventi `data: ...`.

Vedi `webui/README.md` per i dettagli.

---

## Struttura del codice

```
eie-llm/
├── src/
│   ├── main.cpp        — punto di ingresso, sceglie shell/server/bench
│   ├── gguf.cpp        — legge il file .gguf (header, metadata, tensori)
│   ├── ops.cpp         — operazioni matematiche (matmul, softmax, RoPE…)
│   ├── ops_avx2.cpp    — kernel matvec ottimizzati AVX2+FMA (opzionale)
│   ├── cpuinfo.cpp     — rilevamento runtime delle capability SIMD
│   ├── tokenizer.cpp   — converte testo ↔ ID numerici
│   ├── model.cpp       — forward pass (calcola i logits da un token)
│   ├── shell.cpp       — shell interattiva
│   └── server.cpp      — server HTTP
├── include/            — header corrispondenti
├── models/             — file .gguf (non inclusi nel repo)
├── scripts/setup.sh    — download modelli e dipendenze
├── client/             — client Python CLI
├── webui/              — interfaccia web (HTML/CSS/JS)
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

Un modello con 1.1 miliardi di parametri in float32 occuperebbe ~4.4 GB. TinyLlama in Q4_K_M pesa ~638 MB perché ogni peso è compresso a ~4 bit invece di 32.

`ops.cpp` decomprime i pesi al volo, **riga per riga durante il forward pass**, senza mai materializzare l'intera matrice in float32. I formati supportati:

| Formato | Bit/peso | Usato da |
|---------|----------|----------|
| F32 | 32 | norme, bias, positional embedding |
| F16 | 16 | alcuni layer |
| Q8_0 | 8 | GPT-2 |
| Q4_K | 4 | TinyLlama (pesi principali) |
| Q6_K | 6 | TinyLlama (output e attention V) |

I pesi sono rappresentati in memoria dal tipo `QuantTensor` (raw bytes + tipo + shape). Il dispatch avviene in `matvec_quant()`: per F32 fa un cast diretto, per F16/Q8_0/Q4_K/Q6_K dequantizza un super-block alla volta accumulando il prodotto scalare.

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

## Ottimizzazioni SIMD — una lezione pratica

### Cosa sono AVX2 e FMA?

Le CPU moderne hanno **registri SIMD** (Single Instruction, Multiple Data): invece di operare su un singolo numero alla volta, possono operare su un **vettore di numeri** con una sola istruzione.

- **AVX2** (Advanced Vector Extensions 2) introduce registri a **256 bit**. Un registro `__m256` può contenere 8 float a 32 bit (o 4 double a 64 bit).
- **FMA** (Fused Multiply-Add) permette di calcolare `a * b + c` in un'unica istruzione, con precisione maggiore rispetto a fare prima `a * b` e poi `+ c` separatamente.
- **F16C** converte 8 numeri float16 → float32 (o viceversa) in un ciclo.

Perché è importante? Nel transformer, il 90% del tempo è speso nel prodotto matrice-vettore (`matvec`). Con il codice scalare:
```cpp
for (int j = 0; j < in_dim; j++)
    sum += row[j] * x[j];  // 1 multiply-add per ciclo
```

Con AVX2+FMA:
```cpp
__m256 a_vec = _mm256_loadu_ps(row + j);
__m256 x_vec = _mm256_loadu_ps(x + j);
sum_vec = _mm256_fmadd_ps(a_vec, x_vec, sum_vec);  // 8 multiply-add per ciclo
```

Teoricamente, questo è **8x più veloce**.

### Perché non `-march=native` su tutto il binario?

Se compilassimo l'intero progetto con `-march=native`, il compilatore inserirebbe istruzioni AVX2 ovunque: in `main()`, in `gguf.cpp`, persino nei costruttori di `std::vector`. Questo renderebbe il binario **incompatibile con CPU senza AVX2** (crash all'avvio con `SIGILL`).

La soluzione usata qui è il **dispatcher runtime** — un pattern classico nei motori di calcolo:

1. **`cpuinfo.cpp`** interroga la CPU via `CPUID` per sapere se AVX2, FMA e F16C sono presenti. Verifica anche che l'OS abbia abilitato lo stato YMM (registro `XCR0` via `XGETBV`), altrimenti le istruzioni AVX causerebbero un crash anche se la CPU le supporta.

2. **`ops_avx2.cpp`** è compilato **solo** con `-mavx2 -mfma` (tramite `set_source_files_properties` in CMake). Tutte le funzioni qui dentro usano intrinsics esplicite.

3. **`ops.cpp`** contiene le versioni scalari di riferimento. All'inizio di ogni kernel c'è un dispatcher:

```cpp
static CPUFeatures f = cpu_features();
if (f.avx2 && f.fma && avx2_enabled()) {
    matvec_xxx_avx2(A, x, y, out_dim, in_dim);
    return;
}
// ... fallback scalare
```

Il `static` garantisce che il rilevamento avvenga una sola volta, alla prima chiamata. Se la CPU non ha AVX2, il programma continua tranquillamente con i loop scalari.

**Disabilitare AVX2 manualmente** — se si sospetta un problema numerico o si vuole confrontare le prestazioni, è possibile forzare il fallback scalare:

```bash
EIE_NO_AVX2=1 ./build/eie-llm models/tinyllama.Q4_K_M.gguf
```

### Kernel AVX2 implementati

| Formato | Tecnica SIMD | Note |
|---|---|---|
| **F32** | `_mm256_loadu_ps` + `_mm256_fmadd_ps` | 8 float per ciclo, riduzione orizzontale finale |
| **F16** | `_mm256_cvtph_ps` (F16C) + FMADD | Converte 8 half → float in un'istruzione |
| **Q8_0** | `_mm256_cvtepi8_epi32` + broadcast scale | Scompatta 32 int8 in 4× __m256 float |
| **Q4_K** | Dequant + FMADD su buffer allineato | Dequantizza sub-block 32 elem, poi 4 FMADD |
| **Q6_K** | Dequant + FMADD su buffer allineato | Stessa strategia di Q4_K, buffer 64 float |

### La strategia "dequantizza-then-dot"

Per i formati quantizzati (Q4_K, Q6_K), fare masking e shift bit-a-bit in-SIMD è complessissimo. I pesi sono compressi in nibble (4 bit), con scale e minimi condivisi tra gruppi. Invece di implementare la dequantizzazione puramente in assembly SIMD, usiamo una strategia ibrida:

1. **Dequantizzazione scalare**: estraiamo i nibble, applichiamo scale e minimi, ottenendo 32 (o 64) float in un buffer temporaneo sullo stack.
2. **Dot product SIMD**: usiamo gli stessi FMADD del caso F32 sul buffer temporaneo.

Questo elimina il loop scalare interno di 32 iterazioni, che era il vero collo di bottiglia, mantenendo il codice leggibile e corretto.

### Un bug reale: cosa succede quando manca un FMADD

Durante lo sviluppo del kernel Q6_K, un errore di copia-incolla ha causato la mancata elaborazione di 64 elementi per super-block (il secondo gruppo di 32 elementi nella prima metà). Il risultato? Output completamente incoerente, caratteri strani, risposte senza senso.

Questo accade perché i pesi Q6_K sono usati per `attn_v` (i valori dell'attention) e `output.weight` (la proiezione finale). Un errore del 25% su questi pesi si propaga in modo catastrofico attraverso i 22 layer del modello.

**Morale**: le ottimizzazioni SIMD sono potenti ma fragili. Per questo il progetto include:
- Test unitari che confrontano ogni kernel AVX2 con la versione scalare
- La variabile `EIE_NO_AVX2` per fallback immediato
- Commenti dettagliati che spiegano ogni passaggio SIMD

### Prefill batch — processare il prompt tutto insieme

Durante la generazione autoregressiva, ogni nuovo token dipende da tutti i precedenti: **deve essere sequenziale**. Ma durante il **prefill** (processamento del prompt iniziale), tutti i token sono già noti.

Prima, il prefill faceva un loop sequenziale:
```cpp
for (int id : prompt_tokens) {
    forward(model, id, pos, logits);  // UN token alla volta
    pos++;
}
```

Questo ha due problemi:
1. **Overhead**: N chiamate a `forward`, ognuna con setup/teardown di layer.
2. **Cache dei pesi**: ogni matvec ricarica la stessa riga di pesi dequantizzata dalla memoria.

La soluzione è `forward_prefill()`: un singolo passaggio batch che processa tutti i token del prompt insieme. La chiave è `matvec_quant_batch()`:

```cpp
// Dequantizza la riga i una sola volta,
// accumula il dot product per TUTTI i token
for (int i = 0; i < out_dim; i++) {
    dequant_row(A, i, row.data());
    for (int j = 0; j < in_dim; j++) {
        float w = row[j];
        for (int t = 0; t < N; t++) {
            y_batch[t * out_dim + i] += w * x_batch[t * in_dim + j];
        }
    }
}
```

Il peso `w` viene caricato **una sola volta** e riutilizzato per tutti i N token — un enorme risparmio di banda memoria quando N è grande.

**Perché non sempre?** Per prompt corti (≤ 32 token), il loop sequenziale con AVX2 è ancora più veloce del batch scalare. `forward_prefill` sceglie automaticamente la strategia migliore in base alla lunghezza del prompt.

---

## Prestazioni e limiti

Questa è un'implementazione **didattica**: correttezza e leggibilità del codice vengono prima delle prestazioni. Detto questo, è utile capire dove va il tempo e la memoria.

### Memoria — dequantizzazione lazy

I pesi sono mantenuti nel formato quantizzato originale (`QuantTensor`) e dequantizzati riga per riga durante ogni `matvec_quant`. Non viene mai allocata in RAM la versione float32 dell'intera matrice.

| | Disco (quantizzato) | RAM pesi (prima) | RAM pesi (ora) |
|---|---|---|---|
| token_embed + output.weight | ~110 MB | ~512 MB | ~110 MB |
| 22 × (attn Q/K/V/out) | ~120 MB | ~440 MB | ~120 MB |
| 22 × (ffn gate + up + down) | ~408 MB | ~2.9 GB | ~408 MB |
| **Totale pesi** | **~638 MB** | **~4.2 GB** | **~638 MB** |

In pratica il processo occupa ~700–800 MB totali (pesi + KV cache + buffer attivazioni), contro i ~4.5 GB precedenti.

Un ulteriore vantaggio è la **cache efficiency**: una riga di `ffn_gate_w` in Q4_K è 1152 byte (entra in L2), contro 8 KB in float32. Ciò riduce la pressione sul memory bandwidth, che era il vero collo di bottiglia.

### Velocità — il collo di bottiglia è `matvec`

Per ogni token generato, TinyLlama esegue circa **970 milioni di multiply-accumulate** solo nelle proiezioni lineari (Q/K/V, FFN):

| Operazione | MAC per token |
|---|---|
| attn Q + K + V + out (×22 layer) | ~200M |
| ffn gate + up + down (×22 layer) | ~760M |
| **Totale** | **~970M** |

Con il `matvec` scalare (~1 GFLOPS) → circa **1 secondo/token**.
Con il kernel AVX2 attivo, il throughput sale significativamente grazie all'elaborazione SIMD a 256 bit.

**Stato delle ottimizzazioni:**

| Intervento | Stato | Guadagno |
|---|---|---|
| Build Release (`-O3`) | ✅ Attivo | 2-4× (inlining, SIMD automatico) |
| AVX2/FMA esplicito | ✅ Attivo (tutti i formati) | 4-8× sul matvec |
| Prefill batch | ✅ Attivo (N > 32) | riduce overhead per prompt lunghi |
| OpenMP su matvec e ops | ✅ Attivo (soglia automatica) | ~3× su multicore |

### OpenMP — parallelizzazione multicore

Le CPU moderne hanno **più core fisici** (tipicamente 4-16). OpenMP permette di distribuire il lavoro su tutti i core con un semplice pragma:

```cpp
#pragma omp parallel for
for (int i = 0; i < out_dim; i++) {
    // calcolo della riga i — indipendente dalle altre
}
```

**Attenzione all'overhead**: creare e sincronizzare thread ha un costo fisso (~10-50 µs). Se il loop è troppo piccolo, l'overhead supera il guadagno. Per questo usiamo una **soglia dinamica**:

```cpp
#pragma omp parallel for if(out_dim > 512)
```

OpenMP si attiva solo quando il lavoro è sufficiente da distribuire. Questo è particolarmente importante per la generazione autoregressiva, dove ogni token fa molti matvec di dimensioni medie (768-3072 righe).

**Risultati misurati** (CPU 8-core, Q4_K_M):

*TinyLlama 1.1B:*

| Fase | Senza OpenMP | Con OpenMP | Speedup |
|---|---|---|---|---|
| Prefill (13 token) | 2.0 tok/s | 8.2 tok/s | **4.1×** |
| Generazione (per token) | 426 ms | 175 ms | **2.4×** |

*Llama-3.2-3B:*

| Fase | Senza OpenMP | Con OpenMP | Speedup |
|---|---|---|---|---|
| Prefill (12 token) | ~0.4 tok/s | 1.7 tok/s | **4.3×** |
| Generazione (per token) | ~2000 ms | 514 ms | **3.9×** |

### Perché non parallelizzare il loop `n_head`?

La self-attention ha un loop esterno su `n_head` (32 iterazioni in TinyLlama). Sembra perfetto per OpenMP, ma c'è un problema: dentro ogni head ci sono chiamate a `matvec_quant`, che è GIÀ parallelizzata con OpenMP. Parallelizzare entrambi creerebbe **nested parallelism** — 32 head × 8 thread = 256 thread, con overhead e contesa della cache che degraderebbero le prestazioni.

Per questo parallelizziamo solo il loop interno più costoso (il matvec) e lasciamo le head sequenziali.

---

## Roadmap

- [x] Fase 1 — Parser GGUF: header + metadata KV
- [x] Fase 2 — Lettura info tensori e pesi
- [x] Fase 3 — Tokenizer BPE (GPT-2 byte-level)
- [x] Fase 4 — Forward pass GPT-2 (LayerNorm, GELU, MHA)
- [x] Fase 5 — Shell interattiva con linenoise
- [x] Fase 6 — Server HTTP minimale
- [x] Fase 7 — Sampling avanzato (top-k, top-p, repetition penalty)
- [x] Fase 8 — Dequantizzazione Q4_K e Q6_K
- [x] Fase 9 — Architettura LLaMA (RoPE, RMSNorm, SwiGLU, GQA)
- [x] Fase 10 — Fix dequantizzazione Q6_K (bug indici scale)
- [x] Fase 11 — Tokenizer SentencePiece Viterbi unigram
- [x] Fase 12 — Chat template (shell + server HTTP)
- [x] Fase 13 — Buffer di inferenza pre-allocati (eliminazione malloc nel hot path)
- [x] Fase 14 — Dequantizzazione lazy: `QuantTensor` + kernel `matvec_quant` (−3.5 GB RAM, F32/F16/Q8_0/Q4_K/Q6_K)
- [x] Fase 15 — Ottimizzazioni matvec AVX2+FMA con dispatcher runtime (tutti i formati quantizzati)
- [x] Fase 16 — Prefill batch per prompt lunghi (matvec_quant_batch + forward_prefill)
- [x] Fase 17 — OpenMP su matvec e ops vettoriali con soglia automatica
- [x] Fase 18 — Streaming HTTP (Server-Sent Events, token per token)
- [x] Fase 19 — Speculative Decoding: draft & verify per accelerare la generazione
- [x] Fase 20 — KV Cache Prefix Sharing: riuso della cache per prefissi comuni
- [x] Fase 21 — Attention Heatmap Export: "guardare dentro" il modello
- [x] Fase 22 — Continuous Batching: throughput del server su richieste multiple

---

## Prossimi modelli — Task da integrare

Obiettivo: aggiungere supporto per modelli più performanti in italiano, verificando il funzionamento end-to-end uno per uno.

### Task A — Llama-3.2-3B-Instruct *(percorso più sicuro)*

**Stato:** ✅ Completato

**Architettura:** LLaMA (identica a TinyLlama, più grande) — RMSNorm, RoPE, GQA, SwiGLU.  
**Peso:** ~2.0 GB (Q4_K_M)  
**Italiano:** Buono (vocabolario multilingue migliorato)  
**Lavoro stimato:** ~1 giorno  
**Difficoltà:** 🟢 Bassa

**Cosa è stato fatto:**
1. ✅ Scaricato GGUF Q4_K_M da HuggingFace (`bartowski/Llama-3.2-3B-Instruct-GGUF`)
2. ✅ Verificato che i nomi tensori nel GGUF corrispondano al loader esistente (`model_load_weights`)
3. ✅ Aggiornato `rope_freq_base` nei metadata (Llama-3.2 usa 500000 vs 10000 di TinyLlama)
4. ✅ Testato tokenizer BPE (funziona, ma ha vocabolario esteso a 128256 token)
5. ✅ Verificato chat template nei metadati GGUF
6. ✅ Test end-to-end: shell, server, streaming, attention heatmap
7. ✅ Aggiornato `README.md` con i benchmark e la compatibilità

**Bug critici risolti durante l'integrazione:**
- **`n_vocab` errato nei metadata:** Il GGUF di Llama-3.2 non contiene `tokenizer.ggml.vocab_size`, quindi il default di `32000` era completamente errato (vocab reale: `128256`). Questo causava `logits.resize(32000)` ma `matvec_q6k` scriveva `128256` elementi → **massiccio heap buffer overflow** e segfault. Fix: `model_load_config()` ora verifica la vera dimensione da `token_embd.weight.shape[1]`.
- **OpenMP data race in `matvec_quant_batch`:** Il vettore `row` era dichiarato fuori dal `parallel for` → tutti i thread lo condividevano durante `dequant_row()`. Fix: spostato `row` dentro una regione `omp parallel` privata per thread.
- **Tokenizer BPE senza supporto token speciali:** Llama-3.2 usa un tokenizer BPE (classificato come GPT-2) con ~256 token speciali di controllo (`<|start_header_id|>`, `<|eot_id|>`, ecc.). Il nostro encoder li spezzava in caratteri individuali perché non li riconosceva. Fix: `tokenizer_encode()` ora spezza il testo in segmenti, passando i token speciali come singoli ID e tokenizzando solo il testo normale.
- **Chat template LLaMA-3 non riconosciuto:** Il template usa tag `<|start_header_id|>` / `<|end_header_id|>` / `<|eot_id|>` invece dei tag TinyLlama `<|user|>` / `<|assistant|>`. Il rilevamento del template falliva e il prompt veniva passato grezzo al modello. Fix: aggiunto rilevamento e formattazione del formato LLaMA-3 in `apply_chat_template()`.

**Benchmark (CPU 8-core, Q4_K_M):**

| Modello | Prefill | Generazione | Memoria modello |
|---|---|---|---|
| GPT-2 small | — | — | ~120 MB |
| TinyLlama 1.1B | 8.2 tok/s | 5.7 tok/s | ~700 MB |
| **Llama-3.2-3B** | **1.7 tok/s** | **1.9 tok/s** | **~1.9 GB** |

**Perché prima:** architettura identica a LLaMA già supportata, quasi zero modifiche al C++. Conferma che il motore è robusto per tutta la famiglia LLaMA.

---

### Task B — Qwen2.5-1.5B-Instruct *(miglior compromesso qualità/peso)*

**Stato:** ✅ Completato

**Architettura:** Qwen2 (architetturalmente identica a LLaMA: RMSNorm, RoPE, SwiGLU, GQA). Le uniche differenze rispetto a LLaMA standard sono: nomi metadata `qwen2.*`, bias opzionali su Q/K/V e sull'output, e **RoPE tipo NEOX** (coppie scambiate invece di consecutive).  
**Peso:** ~941 MB (Q4_K_M) — la metà di Llama-3.2!  
**Italiano:** Ottimo (dataset multilingue curato)  
**Lavoro stimato:** ~2-3 giorni  
**Difficoltà:** 🟡 Media

**Cosa è stato fatto:**
1. ✅ Scaricato GGUF Q4_K_M da HuggingFace (`bartowski/Qwen2.5-1.5B-Instruct-GGUF`)
2. ✅ Aggiunto supporto metadati `qwen2.*` in `model_load_config()` (stesso path LLaMA con prefisso diverso)
3. ✅ Aggiunte struct `RopeType` (`NORM` / `NEOX`) e campo `rope_type` in `ModelConfig`
4. ✅ Implementato `rope_neox()` in `src/ops.cpp` — ruota coppie scambiate `(i, i+half_dim)` invece di `(2i, 2i+1)`
5. ✅ Aggiunti bias opzionali `attn_q_b`, `attn_k_b`, `attn_v_b` in `LayerWeights` e caricamento con `required=false`
6. ✅ Aggiunto bias opzionale `output_b` in `ModelWeights` e applicazione in `forward()` / `forward_prefill()` / `forward_verify()`
7. ✅ Abilitato caricamento opzionale `output.weight` (tie embedding) — Llama-3.2 e Qwen2 condividono input/output embeddings
8. ✅ Verificato tokenizer Qwen: BPE con token speciali `<|im_start|>`, `<|im_end|>` già gestiti dallo splitting dei token speciali
9. ✅ Verificato chat template Qwen (formato ChatML) già supportato da `apply_chat_template()`
10. ✅ Test end-to-end: shell, server, streaming SSE, attention heatmap

**Bug critici risolti durante l'integrazione:**
- **`model_load_weights` cancellato accidentalmente:** Un commit precedente aveva rimosso l'intera funzione `model_load_weights()` dal file `src/model.cpp`, rompendo la build. Ricostruita la funzione mantenendo tutte le modifiche Qwen2 (bias opzionali, `output_b`, ecc.).
- **RoPE NEOX mancante:** Qwen2 usa `LLAMA_ROPE_TYPE_NEOX` (coppie scambiate), non `NORM` come standard LLaMA. Senza questo fix, il forward su sequenze multi-token produceva attention score esplosi (es. 105.79 invece di ~1-2) e output completamente casuale (caratteri cinesi, parole senza senso). Fix: `self_attention_llama()` e `self_attention_llama_prefill()` ora dispatchano su `rope()` o `rope_neox()` in base a `cfg.rope_type`.
- **Bias Q/K/V disabilitati per debug:** Durante l'investigazione del problema RoPE, i bias erano stati commentati. Riabilitati — confermato che non causano il problema e sono necessari per la correttezza numerica di Qwen2.

**Benchmark (CPU 8-core, Q4_K_M):**

| Modello | Prefill | Generazione | Memoria modello |
|---|---|---|---|
| GPT-2 small | — | — | ~120 MB |
| TinyLlama 1.1B | 8.2 tok/s | 5.7 tok/s | ~700 MB |
| Llama-3.2-3B | 1.7 tok/s | 1.9 tok/s | ~1.9 GB |
| **Qwen2.5-1.5B** | **3.6 tok/s** | **3.0 tok/s** | **~941 MB** |

**Perché secondo:** pesa la metà di Llama-3.2, è più veloce (~3 tok/s vs ~1.9 tok/s), e la qualità in italiano è ottima. È il miglior compromesso qualità/peso per l'hardware CPU-only. L'architettura Qwen2 è identica a LLaMA — le uniche differenze sono nomi metadata, bias opzionali e RoPE NEOX.

---

### Task D — Qwen3-1.7B-Instruct *(il futuro)*

**Stato:** ⬜ Da fare

**Architettura:** Qwen3 (simile a Qwen2 ma con **thinking mode** `/think` e `/no_think`)  
**Peso:** ~1.1 GB (Q4_K_M)  
**Italiano:** Eccellente (119 lingue)  
**Lavoro stimato:** ~3-4 giorni  
**Difficoltà:** 🟡 Media-Alta

**Cosa fare:**
1. Scaricare GGUF Q4_K_M da HuggingFace (`Qwen/Qwen3-1.7B-GGUF`)
2. Verificare compatibilità tokenizer e tie embedding (come Task B)
3. Aggiungere supporto per **thinking mode** nella shell e nella Web UI
   - Parsing dei tag `<think>...</think>` nella risposta
   - Toggle nell'UI per mostrare/nascondere il ragionamento
4. Verificare nuovi formati di chat template con `/think` e `/no_think`
5. Test end-to-end con thinking attivo e disattivato
6. Benchmark comparativo: Qwen3 vs Qwen2.5 vs Llama-3.2
7. Aggiornare `README.md` e `webui/README.md`

**Perché terzo:** richiede tutto il lavoro di Task B più la gestione del thinking mode. È il modello più interessante ma conviene avere la base Qwen2.5 stabile prima.

---

### Ordine di esecuzione consigliato

```
Task A (Llama-3.2) → Task B (Qwen2.5) → Task D (Qwen3)
```

Ogni task include:
- Download e verifica del modello
- Eventuali modifiche al codice C++
- Test shell interattiva
- Test server HTTP + Web UI
- Test streaming SSE
- Test attention heatmap
- Benchmark e aggiornamento documentazione

---

## Integrazioni future

### KV Cache persistente su file

**Problema:** In conversazioni lunghe (es. 500+ token), il prefill deve rielaborare l'intera history ad ogni nuovo messaggio. Questo è O(n²) e diventa molto lento — con Qwen2.5, 500 token di history richiedono ~15-20 secondi di prefill.

**Soluzione proposta:** Salvare la KV cache su file alla fine di ogni turno di conversazione, e ricaricarla all'inizio del turno successivo. La KV cache per un modello 1.5B con context 8192 occupa:
- K cache: `n_layer × n_ctx × n_head_kv × d_head × 4 byte` ≈ 28 × 8192 × 2 × 128 × 4 ≈ **234 MB**
- V cache: uguale ≈ **234 MB**
- Totale: ~**470 MB** per conversazione — gestibile su SSD moderni.

**Implementazione:**
1. Aggiungere `model_save_kvcache(const Model&, const std::string& path)` — serializza K e V in formato binario raw (float32).
2. Aggiungere `model_load_kvcache(Model&, const std::string& path)` — deserializza e verifica dimensioni (n_layer, n_ctx, kv_dim).
3. Aggiungere un campo `conversation_id` nelle richieste HTTP della Web UI, e un meccanismo di mappaggio `conversation_id → file_cache` nel server.
4. La Web UI salva `conversation_id` in `localStorage` insieme alla cronologia chat; il server salva/ripristina la KV cache associata.

**Vantaggi:**
- Prefill di conversazioni lunghe passa da O(n²) a O(m) dove m è la lunghezza del nuovo messaggio.
- UX della Web UI molto più fluida su conversazioni lunghe.
- Implementazione relativamente semplice: la KV cache è già flat e contigua in memoria.

**Limitazioni:**
- Ogni conversazione occupa ~470 MB su disco.
- Non è adatto a deployment multi-utente con migliaia di conversazioni attive (servirebbe un eviction policy LRU).

**Priorità:** Media — miglioramento significativo dell'UX ma non bloccante per l'uso didattico.

---

## Piano di sviluppo — Fasi 18-22

### Fase 18 — Streaming SSE (Server-Sent Events)
**Obiettivo**: il server invia i token man mano che vengono generati.

**Perché**: l'utente percepisce la latenza come *Time-To-First-Token* (TTFT): il tempo dal click alla prima parola, non il tempo totale. Streaming = UX migliore.

**Implementazione** (`src/server.cpp`):
- Refactor della generazione in `generate_text_with_callback()`: accetta un `std::function` che viene chiamato per ogni token. Questo permette di riutilizzare la stessa logica sia per risposte bloccanti che streaming.
- `generate_text()` è diventato un semplice wrapper che accumula i token in una stringa.
- Per lo streaming, gli endpoint leggono `"stream": true` dal JSON e usano `res.set_chunked_content_provider("text/event-stream", ...)` di httplib.h.
- Il *content provider* esegue la generazione interamente nel suo callback: per ogni token chiama `sink.write()` con il formato SSE OpenAI-compatibile:
  ```
  data: {"choices":[{"delta":{"content":"..."}}]}\n\n
  ```
- L'evento di fine stream invia `finish_reason: "stop"` seguito da `data: [DONE]\n\n`.
- Se il client si disconnette, `sink.write()` ritorna `false` e il callback interrompe la generazione, rilasciando il lock sul modello.

**Formato SSE per chat completions (OpenAI-compatibile)**:
```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Ciao!"}],"stream":true}'
```

**Cosa si impara**:
- La generazione autoregressiva è intrinsecamente sequenziale: non possiamo "pre-calcolare" i token, ma possiamo renderli visibili immediatamente.
- `httplib.h` supporta nativamente chunked transfer encoding: `set_chunked_content_provider()` gestisce automaticamente il framing dei chunk, basta scrivere i dati nel `DataSink`.
- Il pattern "callback per token" è elegante: la stessa logica di generazione alimenta sia l'API classica che quella streaming, senza duplicazione di codice.
- Attenzione ai *capture* nelle lambda: `model` e `tok` vivono nello scope di `server_run` (sicuro per reference), mentre `prompt` e `params` devono essere catturati per valore perché il content provider viene eseguito dopo che il handler HTTP è ritornato.

---

### Fase 19 — Speculative Decoding
**Obiettivo**: accelerare la generazione con draft & verify.

**Perché**: è una delle ottimizzazioni più eleganti degli engine moderni (llama.cpp, vLLM, TGI). La verifica batch di N token è molto più veloce della generazione sequenziale di N token.

**Implementazione** (`src/shell.cpp`, `src/model.cpp`):

Il speculative decoding richiede due componenti chiave:

1. **`forward_verify()` in `model.cpp`**: come `forward_prefill()` ma con due differenze fondamentali:
   - Accetta un parametro `base_pos` per indicare da quale posizione nella KV cache iniziare. Questo permette di verificare token draft che si aggiungono a token già generati.
   - Gli attention prefill (`self_attention_prefill` e `self_attention_llama_prefill`) sono stati modificati per supportare `base_pos`: scrivono K,V alle posizioni `base_pos + t` e calcolano attention scores su TUTTO il contesto precedente (token 0..`base_pos+pos`), non solo sui token del batch.
   - Restituisce i logits per **ogni** token del batch in `all_logits`, non solo per l'ultimo. `all_logits[i]` contiene i logits predetti dal modello dopo aver processato il token `i`.

2. **`generate_speculative()` in `shell.cpp`**: implementa l'algoritmo draft & verify:
   ```
   while (non finito):
       // DRAFT: genera K token greedy (senza stamparli)
       draft = [d1, d2, ..., dK]
       
       // ROLLBACK: annulla le modifiche alla KV cache e al contesto
       // (i draft non sono ancora "ufficiali")
       
       // VERIFY: forward_verify(draft, base_pos) → all_logits
       
       // VERIFICA sequenziale:
       // - d1 è corretto? sample(logits_correnti) == d1 ?
       // - d2 è corretto? sample(all_logits[0]) == d2 ?
       // - d3 è corretto? sample(all_logits[1]) == d3 ?
       // - ...
       
       // Se di verifica fallisce al token i:
       //   - Accetta d1..d_{i-1}
       //   - Usa il token campionato dai logits_i come di
       //   - Tronca la KV cache e rigenera da lì
       
       // Se tutti K passano:
       //   - Accetta tutti i draft
       //   - Campiona il prossimo token da all_logits.back()
   ```

**Comandi shell**:
```
:speculative    attiva/disattiva speculative decoding
:draft <n>      imposta il numero di token draft (default 4)
```

**Cosa si impara**:
- La verifica batch è corretta perché i logits `all_logits[i]` dipendono solo dai token fino alla posizione `i`, non dai token successivi nel draft. Quindi anche se processiamo tutti i draft insieme, i logits per il token `i` sono validi per verificare `draft[i+1]`.
- Il rollback della KV cache è semplice: basta reimpostare `n_cached` alla posizione precedente. I dati "extra" rimangono nell'array ma non vengono letti.
- Con lo stesso modello come draft, il guadagno di velocità è limitato (il draft non è più veloce del target). In sistemi reali, il draft è un modello più piccolo o con layer saltati. Il pattern rimane identico.

---

### Fase 20 — KV Cache Prefix Sharing
**Obiettivo**: condividere la KV cache del prefisso tra richieste multiple.

**Perché**: nei server reali, il 50-80% dei token nei prompt è un system prompt identico per tutti. Ricalcolare la KV cache ogni volta è uno spreco enorme. Con la prefix cache, la seconda richiesta con lo stesso prompt salta completamente il prefill.

**Implementazione** (`include/prefix_cache.hpp`, `src/prefix_cache.cpp`, `src/server.cpp`, `src/shell.cpp`):

- **`PrefixCache`**: classe thread-safe con mutex che mappa `hash(prompt) → KVCache`. Ogni entry contiene:
  - `kv_cache`: copia completa di K e V per tutti i layer
  - `n_tokens`: quanti token sono memorizzati
  - `last_used`: timestamp per LRU eviction

- **Lookup (`server.cpp` e `shell.cpp`)**: prima del prefill, `prefix_cache.lookup(prompt, model, n_tokens)`:
  - Se il prompt è in cache e `n_tokens == input_ids.size()` (hit esatto): copia la KV cache nel modello, esegue un `forward()` sull'ultimo token per ricalcolare i logits (non salvati nella cache), e salta il prefill.
  - Se miss: `model_init_kvcache()` + `forward_prefill()` + `prefix_cache.store()`.

- **LRU Eviction**: quando la cache raggiunge `MAX_ENTRIES = 5`, rimuove l'entry con `last_used` più vecchio. Il limite è basso perché ogni entry copia l'intera KV cache (layer × pos × head × d_head float) → memoria cresce rapidamente.

- **Statistiche**: comandi `:cacheinfo` (shell) mostrano hit/miss e hit rate.

**Esempio**:
```bash
# Prima richiesta: miss (prefill completo)
curl -X POST ... -d '{"prompt":"Hello world",...}'

# Seconda richiesta: hit (skippato prefill)
curl -X POST ... -d '{"prompt":"Hello world",...}'
# Nel log server: [CACHE HIT] prompt in cache, saltato prefill
```

**Cosa si impara**:
- La KV cache è il bottleneck di memoria più grande dell'inferenza. Condividerla tra richieste riduce drasticamente il tempo di prefill ripetuti.
- La copia della KV cache (`model.kv_cache = entry.kv_cache`) è costosa ma fattibile per pochi entry. In produzione si userebbe memoria condivisa (reference counting o mmap).
- Il troncamento della cache avviene semplicemente reimpostando `n_cached`: i dati "extra" rimangono nell'array ma non vengono mai letti.

---

### Fase 21 — Attention Heatmap Export
**Obiettivo**: un endpoint che restituisce i pesi attention per ogni layer/head.

**Perché**: vedere quali token si guardano tra loro (es. il pronome "essa" guarda "Francia" 10 token prima) rende concreto il meccanismo della self-attention. I pesi attention sono numeri tra 0 e 1: la somma di ogni riga è 1.0. Un peso alto (vicino a 1.0) significa "questo token è molto rilevante per predire il prossimo".

**Implementazione** (`include/model.hpp`, `src/model.cpp`, `src/server.cpp`):

- **`AttentionSnapshot`** in `model.hpp`: struttura che memorizza i pesi attention in un vettore flat:
  ```cpp
  struct AttentionSnapshot {
      int n_layers, n_heads, seq_len;
      std::vector<float> data;  // [layer][head][q][k]
      void init(int nl, int nh, int sl);
      float& at(int layer, int head, int q_pos, int k_pos);
  };
  ```
  La memoria cresce come O(seq_len²), quindi limitiamo a 100 token.

- **`self_attention` e `self_attention_llama`**: aggiunto parametro opzionale `AttentionSnapshot* snap`. Dopo `softmax()`, se `snap != nullptr`, copiano `bufs.scores[0..pos]` nello snapshot per il layer e head correnti:
  ```cpp
  if (snap) {
      for (int t = 0; t <= pos; t++) {
          snap->at(layer, h, pos, t) = bufs.scores[t];
      }
  }
  ```

- **`forward()`**: aggiunto parametro `AttentionSnapshot* snap = nullptr` che viene passato alle funzioni di attention.

- **`inspect_attention()` in `model.cpp`**: esegue il forward sequenziale su tutti i token del prompt con `snap` attivo, poi formatta i dati in JSON:
  ```json
  {
    "tokens": ["The", " cat", " sat"],
    "layers": [
      {
        "layer": 0,
        "heads": [
          {
            "head": 0,
            "weights": [
              [1.0000, 0.0000, 0.0000],
              [0.7060, 0.2940, 0.0000],
              [0.6041, 0.1051, 0.2908]
            ]
          }
        ]
      }
    ]
  }
  ```
  Ogni riga `weights[q]` è la distribuzione di attenzione del token query `q` sui token precedenti. La causal mask fa sì che `weights[q][k] = 0` per `k > q`.

- **Endpoint `POST /v1/inspect/attention`**: accetta `{"prompt":"...","max_len":100}` e restituisce il JSON.
- **Comando shell `:inspect <testo>`**: stampa il JSON direttamente sulla console.

**Cosa si impara**:
- L'attention non è "magia", è una matrice di pesi che il modello apprende per puntare a token rilevanti.
- La causal mask è visibile nei dati: la parte superiore della matrice è triangolare inferiore (solo token precedenti).
- Alcune head sono "locali" (attendono solo ai token vicini), altre sono "globali" (attendono a token lontani). Questa specializzazione emerge dall'addestramento.

---

### Fase 22 — Continuous Batching
**Obiettivo**: processare più richieste concorrenti senza bloccare il thread HTTP.

**Perché**: nei server reali, le richieste arrivano in momenti diversi. Se il thread HTTP esegue direttamente l'inferenza, le richieste successive restano in attesa (bloccate) finché la prima non finisce. Il throughput crolla.

**Implementazione** (`src/server.cpp`):

- **`ServerRequest`**: struttura che contiene tutto lo stato di una richiesta:
  - `prompt`, `max_tokens`, `params` — parametri di input
  - `output_text`, `generated`, `prompt_tokens` — stato di output
  - `done`, `error`, `error_msg` — stato di completamento
  - `std::mutex mtx` + `std::condition_variable cv` — sincronizzazione tra thread HTTP e thread di inferenza

- **`RequestQueue`**: coda thread-safe con `push()`, `pop()`, `signal_shutdown()`. Il thread HTTP inserisce richieste con `push()`. Il thread di inferenza le preleva con `pop()` (bloccante se vuota).

- **`inference_thread()`**: loop infinito che:
  1. Preleva una richiesta dalla coda (`pop()` blocca se vuota)
  2. Esegue `generate_for_request()` — prefill + generazione
  3. Segnala `done = true` e notifica il thread HTTP con `cv.notify_all()`
  4. Ripete
  
  Questo thread è l'UNICO che tocca il modello, evitando race condition sulla KV cache.

- **`generate_for_request()`**: come `generate_text()` ma scrive direttamente in `ServerRequest::output_text`. Supporta prefix cache (lookup prima del prefill, store dopo).

- **`server_run()`**: avvia `std::thread(inference_thread, ...)` in background. Gli endpoint **non-streaming**:
  1. Creano una `ServerRequest`
  2. La mettono in coda con `queue.push(sreq)`
  3. Bloccano il thread HTTP su `sreq->cv.wait()` finché `done` non diventa `true`
  4. Costruiscono il JSON di risposta

  Gli endpoint **streaming** eseguono la generazione direttamente nel content provider (sincrono), per poter emettere token via SSE in tempo reale.

- **Graceful shutdown**: quando il server riceve `Ctrl+C`, `queue.signal_shutdown()` sveglia il thread di inferenza che esce dal loop, poi `inf_thread.join()` attende la terminazione.

**Architettura**:
```
Thread HTTP (N connessioni)
  → accetta richiesta
  → crea ServerRequest
  → queue.push(req)
  → cv.wait() finché done
  → risponde al client

Thread Inferenza (1 solo)
  → queue.pop() (bloccante)
  → generate_for_request()
  → done = true; cv.notify_all()
  → ripete
```

**Cosa si impara**:
- La separazione thread HTTP / thread inferenza è il pattern fondamentale nei server di produzione. Il thread HTTP deve essere libero di accettare connessioni senza essere bloccato dalla lentezza del modello.
- Il trade-off: la latenza di una singola richiesta aumenta leggermente (deve aspettare in coda), ma il **throughput totale** del server aumenta drasticamente perché non ci sono più richieste bloccate in attesa.
- In un engine reale (vLLM, TGI), il thread di inferenza farebbe UN forward pass batch con il token corrente di TUTTE le richieste attive, massimizzando il riutilizzo dei pesi in cache. La nostra implementazione è sequenziale (una richiesta alla volta) per semplicità, ma l'architettura della coda è identica.

---

## Licenza

Progetto didattico — MIT License.
