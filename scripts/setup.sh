#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
#  EIE-LLM — Script di setup interattivo
#
#  Guida l'utente passo passo nel download dei modelli e delle dipendenze.
#  Se lanciato senza argomenti, mostra un menu interattivo.
#  Se lanciato con argomenti, opera in modalità non-interattiva:
#    ./scripts/setup.sh --gpt2       → solo GPT-2
#    ./scripts/setup.sh --llama      → solo TinyLlama
#    ./scripts/setup.sh --llama32    → solo Llama-3.2-3B
#    ./scripts/setup.sh --qwen25     → solo Qwen2.5-1.5B
#    ./scripts/setup.sh --qwen3      → solo Qwen3-1.7B
#    ./scripts/setup.sh --all        → tutti i modelli
#    ./scripts/setup.sh --libs       → solo aggiorna librerie
# ═════════════════════════════════════════════════════════════════════════════

set -e

THIRD_PARTY="third_party"
MODEL_DIR="models"
mkdir -p "$MODEL_DIR" "$THIRD_PARTY"

# ── Colori ────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Helper download ───────────────────────────
download() {
    local url="$1"
    local dest="$2"
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$dest" "$url"
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar -o "$dest" "$url"
    else
        echo -e "${RED}[ERRORE]${NC} Installa wget o curl"
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

# ── Download singolo modello ──────────────────
download_model() {
    local name="$1"
    local file="$2"
    local url="$3"
    local size="$4"

    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $name già presente"
        return 0
    fi

    echo -e "  ${BLUE}→${NC} Download $name (~$size)..."
    download "$url" "$file"
    if verify_gguf "$file"; then
        echo -e "  ${GREEN}✓${NC} $name verificato"
    else
        echo -e "  ${RED}[ERRORE]${NC} $name: file non valido"
        rm -f "$file"
        return 1
    fi
}

# ── Aggiorna librerie third_party ─────────────
update_libs() {
    echo ""
    echo "─────────────────────────────────────────────"
    echo "  Librerie third_party"
    echo "─────────────────────────────────────────────"

    local update_httplib=false
    local update_linenoise=false

    if [ -f "$THIRD_PARTY/httplib.h" ]; then
        read -rp "  Aggiornare httplib.h? [s/N] " ans
        [[ "$ans" =~ ^[Ss]$ ]] && update_httplib=true
    else
        update_httplib=true
    fi

    if [ -f "$THIRD_PARTY/linenoise.hpp" ]; then
        read -rp "  Aggiornare linenoise.hpp? [s/N] " ans
        [[ "$ans" =~ ^[Ss]$ ]] && update_linenoise=true
    else
        update_linenoise=true
    fi

    if [ "$update_httplib" = true ]; then
        echo -e "  ${BLUE}→${NC} Download cpp-httplib..."
        download \
            "https://raw.githubusercontent.com/yhirose/cpp-httplib/master/httplib.h" \
            "$THIRD_PARTY/httplib.h"
        echo -e "  ${GREEN}✓${NC} httplib.h aggiornato"
    fi

    if [ "$update_linenoise" = true ]; then
        echo -e "  ${BLUE}→${NC} Download cpp-linenoise..."
        download \
            "https://raw.githubusercontent.com/yhirose/cpp-linenoise/master/linenoise.hpp" \
            "$THIRD_PARTY/linenoise.hpp"
        echo -e "  ${GREEN}✓${NC} linenoise.hpp aggiornato"
    fi
}

# ── Menu interattivo modelli ──────────────────
menu_models() {
    echo ""
    echo "─────────────────────────────────────────────"
    echo "  Seleziona i modelli da scaricare"
    echo "─────────────────────────────────────────────"
    echo ""
    echo "  1) GPT-2 small Q8_0      (~176 MB) — base, inglese"
    echo "  2) TinyLlama 1.1B Q4_K_M  (~670 MB) — chat, multilingue"
    echo "  3) Llama-3.2-3B Q4_K_M   (~2.0 GB) — chat, italiano, più potente"
    echo "  4) Qwen2.5-1.5B Q4_K_M   (~941 MB) — chat, italiano, veloce"
    echo "  5) Qwen3-1.7B Q4_K_M     (~1.3 GB) — chat, italiano, thinking mode"
    echo "  6) Tutti i modelli"
    echo "  7) Nessuno (salta)"
    echo ""
    read -rp "  Scelta [1-7]: " choice

    case "$choice" in
        1) download_gpt2=true ;;
        2) download_llama=true ;;
        3) download_llama32=true ;;
        4) download_qwen25=true ;;
        5) download_qwen3=true ;;
        6) download_gpt2=true; download_llama=true; download_llama32=true; download_qwen25=true; download_qwen3=true ;;
        7) ;;
        *) echo -e "  ${YELLOW}!${NC} Scelta non valida, salto download modelli" ;;
    esac
}

# ── Modalità non-interattiva ──────────────────
DOWNLOAD_GPT2=false
DOWNLOAD_LLAMA=false
DOWNLOAD_LLAMA32=false
DOWNLOAD_QWEN25=false
DOWNLOAD_QWEN3=false
UPDATE_LIBS=false

if [ $# -gt 0 ]; then
    for arg in "$@"; do
        case $arg in
            --gpt2)    DOWNLOAD_GPT2=true ;;
            --llama)   DOWNLOAD_LLAMA=true ;;
            --llama32) DOWNLOAD_LLAMA32=true ;;
            --qwen25)  DOWNLOAD_QWEN25=true ;;
            --qwen3)   DOWNLOAD_QWEN3=true ;;
            --all)     DOWNLOAD_GPT2=true; DOWNLOAD_LLAMA=true; DOWNLOAD_LLAMA32=true; DOWNLOAD_QWEN25=true; DOWNLOAD_QWEN3=true ;;
            --libs)    UPDATE_LIBS=true ;;
        esac
    done
else
    # Modalità interattiva
    echo ""
    echo "╔═══════════════════════════════════════════╗"
    echo "║   EIE-LLM — Setup                         ║"
    echo "╠═══════════════════════════════════════════╣"
    echo "║  Questo script ti guida nel download dei  ║"
    echo "║  modelli e delle librerie necessarie.     ║"
    echo "╚═══════════════════════════════════════════╝"

    menu_models
    update_libs
fi

# ── Download modelli ──────────────────────────
if [ "$DOWNLOAD_GPT2" = true ]; then
    echo ""
    echo "─────────────────────────────────────────────"
    echo "  Download modelli"
    echo "─────────────────────────────────────────────"
    download_model \
        "GPT-2 small Q8_0" \
        "$MODEL_DIR/gpt2.Q8_0.gguf" \
        "https://huggingface.co/igorbkz/gpt2-Q8_0-GGUF/resolve/main/gpt2.Q8_0.gguf" \
        "176 MB"
fi

if [ "$DOWNLOAD_LLAMA" = true ]; then
    [ "$DOWNLOAD_GPT2" = false ] && echo "" && echo "─────────────────────────────────────────────" && echo "  Download modelli" && echo "─────────────────────────────────────────────"
    download_model \
        "TinyLlama 1.1B Q4_K_M" \
        "$MODEL_DIR/tinyllama.Q4_K_M.gguf" \
        "https://huggingface.co/second-state/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf" \
        "670 MB"
fi

if [ "$DOWNLOAD_LLAMA32" = true ]; then
    [ "$DOWNLOAD_GPT2" = false ] && [ "$DOWNLOAD_LLAMA" = false ] && echo "" && echo "─────────────────────────────────────────────" && echo "  Download modelli" && echo "─────────────────────────────────────────────"
    download_model \
        "Llama-3.2-3B-Instruct Q4_K_M" \
        "$MODEL_DIR/llama-3.2-3b.Q4_K_M.gguf" \
        "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
        "2.0 GB"
fi

if [ "$DOWNLOAD_QWEN25" = true ]; then
    [ "$DOWNLOAD_GPT2" = false ] && [ "$DOWNLOAD_LLAMA" = false ] && [ "$DOWNLOAD_LLAMA32" = false ] && echo "" && echo "─────────────────────────────────────────────" && echo "  Download modelli" && echo "─────────────────────────────────────────────"
    download_model \
        "Qwen2.5-1.5B-Instruct Q4_K_M" \
        "$MODEL_DIR/qwen2.5-1.5b.Q4_K_M.gguf" \
        "https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf" \
        "941 MB"
fi

if [ "$DOWNLOAD_QWEN3" = true ]; then
    [ "$DOWNLOAD_GPT2" = false ] && [ "$DOWNLOAD_LLAMA" = false ] && [ "$DOWNLOAD_LLAMA32" = false ] && [ "$DOWNLOAD_QWEN25" = false ] && echo "" && echo "─────────────────────────────────────────────" && echo "  Download modelli" && echo "─────────────────────────────────────────────"
    download_model \
        "Qwen3-1.7B Q4_K_M" \
        "$MODEL_DIR/qwen3-1.7b.Q4_K_M.gguf" \
        "https://huggingface.co/bartowski/Qwen_Qwen3-1.7B-GGUF/resolve/main/Qwen_Qwen3-1.7B-Q4_K_M.gguf" \
        "1.3 GB"
fi

# ── Riepilogo ─────────────────────────────────
echo ""
echo "╔═══════════════════════════════════════════╗"
echo "║   ✓ Setup completato!                     ║"
echo "╠═══════════════════════════════════════════╣"
echo "║                                           ║"
echo "║  Build:                                   ║"
echo "║    cmake -B build -DCMAKE_BUILD_TYPE=Release ║"
echo "║    cmake --build build -j$(nproc)         ║"
echo "║                                           ║"
echo "║  Uso:                                     ║"
[ -f "$MODEL_DIR/gpt2.Q8_0.gguf" ]        && echo "║    ./build/eie-llm models/gpt2.Q8_0.gguf           → GPT-2 shell       ║"
[ -f "$MODEL_DIR/tinyllama.Q4_K_M.gguf" ] && echo "║    ./build/eie-llm models/tinyllama.Q4_K_M.gguf    → TinyLlama shell  ║"
[ -f "$MODEL_DIR/llama-3.2-3b.Q4_K_M.gguf" ] && echo "║    ./build/eie-llm models/llama-3.2-3b.Q4_K_M.gguf → Llama-3.2 shell  ║"
[ -f "$MODEL_DIR/qwen2.5-1.5b.Q4_K_M.gguf" ] && echo "║    ./build/eie-llm models/qwen2.5-1.5b.Q4_K_M.gguf → Qwen2.5 shell  ║"
[ -f "$MODEL_DIR/qwen3-1.7b.Q4_K_M.gguf" ] && echo "║    ./build/eie-llm models/qwen3-1.7b.Q4_K_M.gguf   → Qwen3 shell    ║"
echo "║                                           ║"
echo "║  Server HTTP + Web UI:                    ║"
[ -f "$MODEL_DIR/llama-3.2-3b.Q4_K_M.gguf" ] && echo "║    ./build/eie-llm models/llama-3.2-3b.Q4_K_M.gguf --server 8080    ║"
[ -f "$MODEL_DIR/qwen2.5-1.5b.Q4_K_M.gguf" ] && echo "║    ./build/eie-llm models/qwen2.5-1.5b.Q4_K_M.gguf --server 8080    ║"
[ -f "$MODEL_DIR/qwen3-1.7b.Q4_K_M.gguf" ] && echo "║    ./build/eie-llm models/qwen3-1.7b.Q4_K_M.gguf   --server 8080    ║"
[ -f "$MODEL_DIR/llama-3.2-3b.Q4_K_M.gguf" ] || [ -f "$MODEL_DIR/qwen2.5-1.5b.Q4_K_M.gguf" ] || [ -f "$MODEL_DIR/qwen3-1.7b.Q4_K_M.gguf" ] || [ -f "$MODEL_DIR/tinyllama.Q4_K_M.gguf" ] && echo "║    # poi apri http://localhost:8080       ║"
echo "║                                           ║"
echo "╚═══════════════════════════════════════════╝"
echo ""
