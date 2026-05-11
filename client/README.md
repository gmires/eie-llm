# EIE-LLM Client Python

Client Python ufficiale per interrogare il server EIE-LLM via API HTTP.

## Installazione

```bash
cd client

# Crea virtual environment
python3 -m venv venv

# Attiva (Linux/Mac)
source venv/bin/activate

# Attiva (Windows)
venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt
```

## Uso

### Health Check

```bash
python client.py --mode health
```

### Completamento Testo

```bash
# Completamento semplice
python client.py --mode complete --prompt "The sky is"

# Con parametri di sampling
python client.py --mode complete --prompt "Hello" -t 0.8 --top-k 20 --max-tokens 100

# Greedy (deterministico)
python client.py --mode complete --prompt "Hello" --greedy

# Con chat template (TinyLlama)
python client.py --mode complete --prompt "Ciao!" --chat-template
```

### Chat

```bash
# Chat semplice
python client.py --mode chat --message "Qual è la capitale della Francia?"

# Con streaming
python client.py --mode chat --message "Ciao!" --stream
```

### Streaming SSE

```bash
# Streaming per completamento
python client.py --mode complete --prompt "Hello world" --stream

# Streaming per chat
python client.py --mode chat --message "Ciao!" --stream
```

### Attention Heatmap

```bash
# Plotta TUTTI i layer (media dei head)
python client.py --mode inspect --prompt "The cat sat" --plot

# Plotta un layer specifico (tutti i head)
python client.py --mode inspect --prompt "The cat sat" --plot --layer 0

# Plotta layer e head specifici
python client.py --mode inspect --prompt "The cat sat" --plot --layer 0 --head 0

# Salva su file invece di mostrare
python client.py --mode inspect --prompt "Hello" --plot --output attention.png
```

### Output Raw JSON

```bash
# Stampa il JSON completo della risposta
python client.py --mode complete --prompt "Hello" --raw
```

## Parametri

| Parametro | Descrizione | Default |
|---|---|---|
| `--host` | Host del server | `localhost` |
| `--port` | Porta del server | `8080` |
| `--mode` | `health`, `complete`, `chat`, `inspect` | (richiesto) |
| `--prompt` | Prompt per complete/inspect | — |
| `--message` | Messaggio per chat | — |
| `-t, --temperature` | Temperatura sampling | `1.0` |
| `--top-k` | Top-k sampling | `40` |
| `--top-p` | Nucleus sampling | `0.9` |
| `--max-tokens` | Token massimi | (server default) |
| `--repetition-penalty` | Penalità ripetizioni | `1.1` |
| `--greedy` | Sampling greedy | `False` |
| `--stream` | Attiva streaming SSE | `False` |
| `--plot` | Plotta attention heatmap | `False` |
| `--layer` | Layer da plottare | (tutti) |
| `--head` | Head da plottare | (tutti) |
| `--output` | File output immagine | — |
| `--raw` | Stampa JSON raw | `False` |

## Esempi Avanzati

```bash
# Server su porta diversa
python client.py --port 9090 --mode health

# Prompt lungo con inspect
python client.py --mode inspect --prompt "The quick brown fox jumps over the lazy dog" --plot

# Chat con temperatura alta (più creativo)
python client.py --mode chat --message "Raccontami una storia" -t 1.5 --max-tokens 200

# Streaming con greedy
python client.py --mode complete --prompt "Once upon a time" --stream --greedy
```

## Note

- Il server deve essere attivo prima di usare il client.
- Per `--plot` servono `matplotlib`, `seaborn` e `numpy`.
- Lo streaming (`--stream`) mostra i token man mano che arrivano.
