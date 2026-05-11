# EIE-LLM Web UI

Interfaccia web moderna per EIE-LLM, ispirata a OpenWebUI. Si connette direttamente al server HTTP e offre un'esperienza di chat completa nel browser, senza dipendenze da installare.

## Caratteristiche

- **Chat streaming in tempo reale** — i token arrivano man mano che vengono generati (SSE)
- **Cronologia conversazioni** — salvata automaticamente nel browser (`localStorage`)
- **Markdown + syntax highlighting** — messaggi formattati con marked.js e highlight.js
- **Pannello impostazioni** — temperatura, top-k, top-p, repetition penalty, greedy, streaming toggle, modalità chat
- **Attention Heatmap interattiva** — visualizza i pesi attention per ogni layer/head con canvas HTML5, tooltip al passaggio del mouse e valori numerici nelle celle
- **Tema chiaro/scuro** — toggle immediato con persistenza
- **Importa/Esporta chat** — formato JSON per backup e condivisione
- **Responsive** — funziona su desktop e mobile

## Avvio

1. Avvia il server EIE-LLM come al solito:
   ```bash
   ./build/eie-llm models/tinyllama.Q4_K_M.gguf --server 8080
   ```

2. Apri il browser all'indirizzo indicato:
   ```
   http://localhost:8080
   ```

Il server serve automaticamente i file statici dalla directory `webui/`.

## Utilizzo

### Chat
- Scrivi un messaggio nella casella in basso e premi **Invio** (o il pulsante verde)
- Lo streaming è attivo di default: vedrai i token apparire in tempo reale
- Il pulsante rosso **⏹** interrompe la generazione
- Ogni conversazione è salvata automaticamente nel browser

### Cronologia
- La sidebar sinistra mostra tutte le chat precedenti
- Clicca su una chat per riaprirla
- **+** per creare una nuova chat
- **Esporta** salva tutte le chat in un file JSON
- **Importa** carica un file JSON precedentemente esportato
- Clicca sulla **×** accanto a una chat per eliminarla

### Impostazioni
- Clicca **⚙️** nell'header per aprire il pannello
- Regola temperatura, max tokens, top-k, top-p, repetition penalty
- Attiva/disattiva streaming, greedy sampling, modalità chat
- Le impostazioni vengono salvate automaticamente

### Attention Heatmap
- Clicca **🔥** nell'header per aprire il pannello attention
- Inserisci un testo breve (max 100 token) e premi **Analizza**
- Scegli layer e head specifici, oppure visualizza la **media** su tutti
- La heatmap mostra con i colori (blu → rosso) quanto ogni token "guarda" gli altri

#### Come leggere la heatmap

La heatmap è una matrice quadrata dove:
- **Righe (asse Y, Query tokens)** — il token che sta "chiedendo" attenzione (il token corrente)
- **Colonne (asse X, Key tokens)** — il token a cui si rivolge (i token precedenti)
- **Colore** — da blu (basso peso, ~0.0) a rosso (alto peso, ~1.0)
- **Valore numerico** — visibile nelle celle quando sono abbastanza grandi, o nel tooltip al passaggio del mouse

Ogni riga somma a **1.0**: il token query distribuisce la sua "attenzione" tra i token precedenti. La parte superiore della matrice è triangolare inferiore perché la causal mask impedisce ai token di guardare al futuro.

#### Tooltip interattivo

Passando il mouse su una cella si visualizza un tooltip con:
- **Q:** il token query (riga)
- **K:** il token key (colonna)
- **Att:** il peso attention esatto con 4 cifre decimali (es. 0.2847)

#### Esempio pratico

Con il testo "The cat sat":
- Il token "sat" (riga) potrebbe avere un peso alto su "cat" (colonna) perché il verbo si riferisce al soggetto
- Il primo token "The" ha sempre peso 1.0 su se stesso (non ha precedenti)
- Token di punteggiatura spesso hanno pesi distribuiti uniformemente

#### Selezione layer/head

Il modello ha molteplici layer (strati) e head (teste d'attenzione). Ogni head specializzata su pattern diversi:
- **Head "locali"** — attendono solo ai token vicini (pattern sintattici)
- **Head "globali"** — attendono a token lontani (coreference, soggetto-verbo)
- **Media su tutti i layer/head** — dà una visione d'insieme stabile

## Architettura

- **Zero build step** — HTML/CSS/JS vanilla, nessun bundler
- **Zero dipendenze locali** — marked.js e highlight.js caricati da CDN
- **SPA (Single Page Application)** — tutta la logica è in `app.js`
- **API REST** — si connette agli endpoint `/v1/chat/completions`, `/v1/completions` e `/v1/inspect/attention`
- **File statici** — serviti nativamente da `httplib::Server::set_mount_point`

## File

```
webui/
├── index.html   — struttura DOM
├── style.css    — tema e layout responsive
├── app.js       — logica applicativa (chat, SSE, heatmap, storage)
└── README.md    — questo file
```
