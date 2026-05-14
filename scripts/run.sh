#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
#  EIE-LLM — Unified model runner
#
#  Usage:
#    ./scripts/run.sh [model] [args...]     → avvia il modello con argomenti
#    ./scripts/run.sh --server [model] [port] → avvia il server HTTP
#    ./scripts/run.sh                       → menu interattivo
#    ./scripts/run.sh --list                → elenca modelli disponibili
#
#  Esempi:
#    ./scripts/run.sh qwen3                  → shell Qwen3-1.7B
#    ./scripts/run.sh qwen3 --server 8080    → server HTTP Qwen3-1.7B
#    ./scripts/run.sh --server qwen3 8080    → server HTTP (ordine alternativo)
#    ./scripts/run.sh llama32 --bench        → benchmark Llama-3.2
# ═════════════════════════════════════════════════════════════════════════════
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"

MODELS=(
    "gpt2:GPT-2 small 124M (Q8_0):models/gpt2.Q8_0.gguf:1.0:40:0.9:1.0"
    "tinyllama:TinyLlama 1.1B Chat (Q4_K_M):models/tinyllama.Q4_K_M.gguf:0.7:50:0.95:1.1"
    "llama32:Llama-3.2-3B-Instruct (Q4_K_M):models/llama-3.2-3b.Q4_K_M.gguf:0.6:40:0.9:1.1"
    "qwen25:Qwen2.5-1.5B-Instruct (Q4_K_M):models/qwen2.5-1.5b.Q4_K_M.gguf:0.7:40:0.8:1.05"
    "qwen3:Qwen3-1.7B-Instruct (Q4_K_M):models/qwen3-1.7b.Q4_K_M.gguf:0.6:20:0.95:1.5"
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
        local extra_args=()
        read -rp "  Server HTTP? (porta, es. 8080, o lascia vuoto per shell): " srvport
        if [ -n "$srvport" ]; then
            extra_args=("--server" "$srvport")
        fi
        run_model "$key" "$name" "$path" "$temp" "$topk" "$topp" "$penalty" "${extra_args[@]}"
    fi
}

run_model() {
    local key="$1" name="$2" path="$3" temp="$4" topk="$5" topp="$6" penalty="$7"
    shift 7  # rimuove i primi 7 argomenti, lascia extra_args in "$@"
    local full="$DIR/$path"

    if [ ! -f "$full" ]; then
        echo "  [ERRORE] Modello non trovato: $full"
        echo "  Esegui prima: ./scripts/setup.sh --${key/gpt2/gpt2}"
        exit 1
    fi

    local mode="shell"
    for arg in "$@"; do
        [ "$arg" = "--server" ] && mode="server"
    done

    echo ""
    echo "╔═══════════════════════════════════════════╗"
    echo "║  Avvio: $name"
    echo "║  Modalità: $mode"
    echo "║  Parametri: temp=$temp topk=$topk topp=$topp penalty=$penalty"
    if [ "$key" = "qwen3" ]; then
        echo "║  Thinking mode: ON (:think on)"
        echo "║  Per disabilitare: :think off"
    fi
    echo "╚═══════════════════════════════════════════╝"
    echo ""

    exec "$DIR/build/eie-llm" "$full" "$@"
}

_run_main() {
    if [ $# -eq 0 ]; then
        menu
        return
    fi

    local search_key=""
    local extra_args=()

    if [ "$1" = "--server" ]; then
        search_key="$2"
        extra_args=("--server")
        [ -n "$3" ] && extra_args+=("$3")
    elif [ "$1" = "--list" ] || [ "$1" = "-l" ]; then
        list_models
        return
    elif [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        echo "Uso: ./scripts/run.sh [modello] [--server [porta] | --bench N]"
        list_models
        return
    else
        search_key="$1"
        shift
        while [ $# -gt 0 ]; do
            local is_model=false
            for m in "${MODELS[@]}"; do
                local k n; IFS=':' read -r k n <<< "$m"
                [ "$k" = "$1" ] && is_model=true
            done
            [ "$is_model" = true ] && break
            extra_args+=("$1")
            shift
        done
    fi

    local found=false
    for m in "${MODELS[@]}"; do
        local key name path temp topk topp penalty
        IFS=':' read -r key name path temp topk topp penalty <<< "$m"
        if [ "$key" = "$search_key" ]; then
            run_model "$key" "$name" "$path" "$temp" "$topk" "$topp" "$penalty" "${extra_args[@]}"
            found=true
            break
        fi
    done
    if [ "$found" = false ]; then
        echo "  [ERRORE] Modello sconosciuto: $search_key" >&2
        list_models
        return 1
    fi
}
_run_main "$@"
