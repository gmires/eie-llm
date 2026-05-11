# EIE-LLM Web UI

Interfaccia web moderna per EIE-LLM, ispirata a OpenWebUI. Si connette direttamente al server HTTP e offre un'esperienza di chat completa nel browser, senza dipendenze da installare.

## Caratteristiche

- **Chat streaming in tempo reale** — i token arrivano man mano che vengono generati (SSE)
- **Cronologia conversazioni** — salvata automaticamente nel browser (`localStorage`)
- **Markdown + syntax highlighting** — messaggi formattati con marked.js e highlight.js
- **Pannello impostazioni** — temperatura, top-k, top-p, repetition penalty, greedy, streaming toggle, modalità chat
- **Attention Heatmap** — visualizza i pesi attention per ogni layer/head con canvas HTML5
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

### Tema
- Clicca **🌙/☀️** per passare da tema scuro a chiaro

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
