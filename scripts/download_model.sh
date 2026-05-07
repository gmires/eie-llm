#!/bin/bash
# ─────────────────────────────────────────────────────────
#  EIE-LLM — Script di download del modello GPT-2 GGUF
#  Scarica GPT-2 small (124M parametri, ~176MB) da HuggingFace
#  Uso: ./scripts/download_model.sh
# ─────────────────────────────────────────────────────────

set -e  # interrompi se un comando fallisce

MODEL_DIR="models"
MODEL_FILE="gpt2.Q8_0.gguf"
MODEL_URL="https://huggingface.co/igorbkz/gpt2-Q8_0-GGUF/resolve/main/gpt2.Q8_0.gguf"

mkdir -p "$MODEL_DIR"

# Controlla se il modello è già presente
if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "✓ Modello già presente in $MODEL_DIR/$MODEL_FILE"
    exit 0
fi

echo "→ Download GPT-2 Q8_0 (~176MB)..."

# Prova prima con wget, poi con curl come fallback
if command -v wget &> /dev/null; then
    wget -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
elif command -v curl &> /dev/null; then
    curl -L -o "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
else
    echo "[ERRORE] Installa wget o curl per scaricare il modello"
    exit 1
fi

# Verifica che il file sia un GGUF valido controllando il magic
MAGIC=$(xxd -p -l 4 "$MODEL_DIR/$MODEL_FILE")
if [ "$MAGIC" = "47475546" ]; then
    echo "✓ Modello scaricato e verificato: $MODEL_DIR/$MODEL_FILE"
else
    echo "[ERRORE] Il file scaricato non sembra un GGUF valido (magic: $MAGIC)"
    rm -f "$MODEL_DIR/$MODEL_FILE"
    exit 1
fi