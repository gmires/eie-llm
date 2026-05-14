#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
#  EIE-LLM — Qwen2.5-1.5B-Instruct  (Q4_K_M)
#
#  Parametri consigliati (fonte: modello di riferimento Qwen2.5-7B):
#    :temp 0.7          → Temperatura moderata per chat
#    :topk 40           → Top-k di sicurezza
#    :topp 0.8          → Nucleus più stretto per coerenza
#    :penalty 1.05      → Penalità leggera ripetizioni
#
#  Qwen2.5 parte automaticamente in modalità chat.
#  Italiano: ottimo. Veloce e leggero (~1 GB).
# ═════════════════════════════════════════════════════════════════════════════
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "$DIR/build/eie-llm" "$DIR/models/qwen2.5-1.5b.Q4_K_M.gguf" "$@"
