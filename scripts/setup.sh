#!/bin/bash
# ─────────────────────────────────────────────
#  EIE-LLM — Script di setup completo
#  Scarica i modelli e le dipendenze
#  Uso: ./scripts/setup.sh [--all] [--gpt2] [--llama]
#  Default: scarica solo GPT-2
# ─────────────────────────────────────────────

set -e

THIRD_PARTY="third_party"
MODEL_DIR="models"
mkdir -p "$MODEL_DIR" "$THIRD_PARTY"

# ── Argomenti ─────────────────────────────────
DOWNLOAD_GPT2=true
DOWNLOAD_LLAMA=false

for arg in "$@"; do
    case $arg in
        --all)   DOWNLOAD_GPT2=true;  DOWNLOAD_LLAMA=true  ;;
        --gpt2)  DOWNLOAD_GPT2=true;  DOWNLOAD_LLAMA=false ;;
        --llama) DOWNLOAD_GPT2=false; DOWNLOAD_LLAMA=true  ;;
    esac
done

# ── Helper download ───────────────────────────
download() {
    local url="$1"
    local dest="$2"
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$dest" "$url"
    else
        echo "[ERRORE] Installa wget o curl"
        exit 1
    fi
}

# ── Verifica magic GGUF ───────────────────────
verify_gguf() {
    local file="$1"
    local magic
    magic=$(xxd -p -l 4 "$file" 2>/dev/null || echo "")
    if [ "$magic" = "47475546" ]; then
        return 0
    else
        return 1
    fi
}

# ── GPT-2 small Q8_0 (~176MB) ─────────────────
if [ "$DOWNLOAD_GPT2" = true ]; then
    GPT2_FILE="$MODEL_DIR/gpt2.Q8_0.gguf"
    GPT2_URL="https://huggingface.co/igorbkz/gpt2-Q8_0-GGUF/resolve/main/gpt2.Q8_0.gguf"

    if [ -f "$GPT2_FILE" ]; then
        echo "✓ GPT-2 Q8_0 già presente"
    else
        echo "→ Download GPT-2 small Q8_0 (~176MB)..."
        download "$GPT2_URL" "$GPT2_FILE"
        if verify_gguf "$GPT2_FILE"; then
            echo "✓ GPT-2 verificato"
        else
            echo "[ERRORE] GPT-2: file non valido"
            rm -f "$GPT2_FILE"
            exit 1
        fi
    fi
fi

# ── TinyLlama 1.1B Q4_K_M (~670MB) ───────────
if [ "$DOWNLOAD_LLAMA" = true ]; then
    LLAMA_FILE="$MODEL_DIR/tinyllama.Q4_K_M.gguf"
    LLAMA_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

    if [ -f "$LLAMA_FILE" ]; then
        echo "✓ TinyLlama già presente"
    else
        echo "→ Download TinyLlama 1.1B Q4_K_M (~670MB)..."
        download "$LLAMA_URL" "$LLAMA_FILE"
        if verify_gguf "$LLAMA_FILE"; then
            echo "✓ TinyLlama verificato"
        else
            echo "[ERRORE] TinyLlama: file non valido"
            rm -f "$LLAMA_FILE"
            exit 1
        fi
    fi
fi

# ── httplib ───────────────────────────────────
if [ -f "$THIRD_PARTY/httplib.h" ]; then
    echo "✓ httplib.h già presente"
else
    echo "→ Download cpp-httplib..."
    download \
        "https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h" \
        "$THIRD_PARTY/httplib.h"
    echo "✓ httplib.h scaricato"
fi

# ── linenoise ─────────────────────────────────
if [ -f "$THIRD_PARTY/linenoise.hpp" ]; then
    echo "✓ linenoise.hpp già presente"
else
    echo "→ Download cpp-linenoise..."
    download \
        "https://raw.githubusercontent.com/yhirose/cpp-linenoise/master/linenoise.hpp" \
        "$THIRD_PARTY/linenoise.hpp"
    echo "✓ linenoise.hpp scaricato"
fi

# ── Riepilogo ─────────────────────────────────
echo ""
echo "✓ Setup completato!"
echo ""
echo "  Build:"
echo "    cmake -B build && cmake --build build"
echo ""
echo "  Uso:"
echo "    ./build/eie-llm models/gpt2.Q8_0.gguf        → GPT-2 shell"
echo "    ./build/eie-llm models/tinyllama.Q4_K_M.gguf → TinyLlama shell"
echo ""
echo "  Download modelli:"
echo "    ./scripts/setup.sh --gpt2    GPT-2 only"
echo "    ./scripts/setup.sh --llama   TinyLlama only"
echo "    ./scripts/setup.sh --all     entrambi"