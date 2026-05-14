#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
#  EIE-LLM — Llama-3.2-3B-Instruct  (Q4_K_M)
#
#  Parametri consigliati (fonte: Meta / best practices generali LLaMA):
#    :temp 0.6          → Temperatura medio-bassa per risposte precise
#    :topk 40           → Top-k di sicurezza
#    :topp 0.9          → Nucleus sampling bilanciato
#    :penalty 1.1       → Leggera penalità ripetizioni
#
#  Llama-3.2 parte automaticamente in modalità chat.
#  Italiano: buono (vocabolario multilingue migliorato).
# ═════════════════════════════════════════════════════════════════════════════
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "$DIR/build/eie-llm" "$DIR/models/llama-3.2-3b.Q4_K_M.gguf" "$@"
