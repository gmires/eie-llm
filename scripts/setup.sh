#!/bin/bash
# ─────────────────────────────────────────────
#  EIE-LLM — Script di setup completo
#  Scarica il modello GPT-2 e le dipendenze
#  Uso: ./scripts/setup.sh
# ─────────────────────────────────────────────

set -e

# ── Modello ───────────────────────────────────
MODEL_DIR="models"
MODEL_FILE="gpt2.Q8_0.gguf"
MODEL_URL="https://huggingface.co/igorbkz/gpt2-Q8_0-GGUF/resolve/main/gpt2.Q8_0.gguf"

mkdir -p "$MODEL_DIR"

if [ -f "$MODEL_DIR/$MODEL_FILE" ]; then
    echo "✓ Modello già presente"
else
    echo "→ Download GPT-2 Q8_0 (~176MB)..."
    if command -v wget &> /dev/null; then
        wget -O "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
    elif command -v curl &> /dev/null; then
        curl -L -o "$MODEL_DIR/$MODEL_FILE" "$MODEL_URL"
    else
        echo "[ERRORE] Installa wget o curl"
        exit 1
    fi

    MAGIC=$(xxd -p -l 4 "$MODEL_DIR/$MODEL_FILE")
    if [ "$MAGIC" = "47475546" ]; then
        echo "✓ Modello verificato"
    else
        echo "[ERRORE] File non valido"
        rm -f "$MODEL_DIR/$MODEL_FILE"
        exit 1
    fi
fi

# ── Dipendenze ────────────────────────────────
THIRD_PARTY="third_party"
mkdir -p "$THIRD_PARTY"

if [ -f "$THIRD_PARTY/httplib.h" ]; then
    echo "✓ httplib.h già presente"
else
    echo "→ Download cpp-httplib..."
    if command -v wget &> /dev/null; then
        wget -O "$THIRD_PARTY/httplib.h" \
          "https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h"
    elif command -v curl &> /dev/null; then
        curl -L -o "$THIRD_PARTY/httplib.h" \
          "https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h"
    fi
    echo "✓ httplib.h scaricato"
fi

echo ""
echo "✓ Setup completato. Ora:"
echo "  cmake -B build && cmake --build build"
echo "  ./build/eie-llm models/gpt2.Q8_0.gguf"