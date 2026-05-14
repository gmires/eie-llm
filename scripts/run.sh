#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
#  EIE-LLM — Unified model runner
#
#  Usage:
#    ./scripts/run.sh [model]     → avvia il modello specificato
#    ./scripts/run.sh             → menu interattivo
#    ./scripts/run.sh --list      → elenca modelli disponibili
#
#  Esempi:
#    ./scripts/run.sh qwen3       → avvia Qwen3-1.7B
#    ./scripts/run.sh llama32     → avvia Llama-3.2-3B
# ═════════════════════════════════════════════════════════════════════════════
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODELS=(
    "gpt2:GPT-2 small 124M (Q8_0):models/gpt2.Q8_0.gguf:1.0:40:0.9:1.0"
    "tinyllama:TinyLlama 1.1B Chat (Q4_K_M):models/tinyllama.Q4_K_M.gguf:0.7:50:0.95:1.1"
    "llama32:Llama-3.2-3B-Instruct (Q4_K_M):models/llama-3.2-3b.Q4_K_M.gguf:0.6:40:0.9:1.1"
    "qwen25:Qwen2.5-1.5B-Instruct (Q4_K_M):models/qwen2.5-1.5b.Q4_K_M.gguf:0.7:40:0.8:1.05"
    "qwen3:Qwen3-1.7B-Instruct (Q8_0):models/qwen3-1.7b.Q8_0.gguf:0.6:20:0.95:1.5"
)

list_models() {
    echo "Modelli disponibili:"
    for m in "${MODELS[@]}"; do
        IFS=':' read -r key name path temp topk topp penalty <<< "$m"
        local full="$DIR/$path"
        if [ -f "$full" ]; then
            echo "  $key   $name  ✓"
        else
            echo "  $key   $name  ✗ (non scaricato)"
        fi
    done
    echo ""
    echo "Usa: ./scripts/run.sh <modello>"
}

menu() {
    echo ""
    echo "╔═══════════════════════════════════════════╗"
    echo "║   EIE-LLM — Model Launcher                ║"
    echo "╠═══════════════════════════════════════════╣"
    echo "║  Seleziona un modello:                    ║"
    local i=1
    for m in "${MODELS[@]}"; do
        IFS=':' read -r key name path temp topk topp penalty <<< "$m"
        local full="$DIR/$path"
        local status="✓"
        [ ! -f "$full" ] && status="✗ (non scaricato)"
        printf "║  %d) %-35s %s  ║\n" "$i" "$name" "$status"
        i=$((i+1))
    done
    echo "║                                           ║"
    echo "║  q) Esci                                   ║"
    echo "╚═══════════════════════════════════════════╝"
    read -rp "  Scelta [1-${#MODELS[@]}]: " choice
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#MODELS[@]}" ]; then
        local idx=$((choice-1))
        IFS=':' read -r key name path temp topk topp penalty <<< "${MODELS[$idx]}"
        run_model "$key" "$name" "$path" "$temp" "$topk" "$topp" "$penalty"
    fi
}

run_model() {
    local key="$1" name="$2" path="$3" temp="$4" topk="$5" topp="$6" penalty="$7"
    local full="$DIR/$path"

    if [ ! -f "$full" ]; then
        echo "  [ERRORE] Modello non trovato: $full"
        echo "  Esegui prima: ./scripts/setup.sh --${key/gpt2/gpt2}"
        exit 1
    fi

    echo ""
    echo "╔═══════════════════════════════════════════╗"
    echo "║  Avvio: $name"
    echo "║  Parametri: temp=$temp topk=$topk topp=$topp penalty=$penalty"
    if [ "$key" = "qwen3" ]; then
        echo "║  Thinking mode: ON (:think on)"
        echo "║  Per disabilitare: :think off"
    fi
    echo "╚═══════════════════════════════════════════╝"
    echo ""

    exec "$DIR/build/eie-llm" "$full"
}

if [ $# -eq 0 ]; then
    menu
else
    case "$1" in
        --list|-l) list_models ;;
        --help|-h) echo "Uso: ./scripts/run.sh [modello]"; list_models ;;
        *)
            found=false
            for m in "${MODELS[@]}"; do
                IFS=':' read -r key name path temp topk topp penalty <<< "$m"
                if [ "$key" = "$1" ]; then
                    run_model "$key" "$name" "$path" "$temp" "$topk" "$topp" "$penalty"
                    found=true
                    break
                fi
            done
            if [ "$found" = false ]; then
                echo "  [ERRORE] Modello sconosciuto: $1"
                list_models
                exit 1
            fi
            ;;
    esac
fi
