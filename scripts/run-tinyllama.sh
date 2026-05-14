#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
#  EIE-LLM — TinyLlama 1.1B Chat  (Q4_K_M)
#
#  Parametri consigliati (fonte: model card HuggingFace + Cloudflare Workers):
#    :temp 0.7          → Temperatura moderata per chat
#    :topk 50           → Top-k di sicurezza
#    :topp 0.95         → Nucleus ampio per varietà
#    :penalty 1.1       → Leggera penalità ripetizioni
#
#  TinyLlama parte automaticamente in modalità chat
#  grazie al template ChatML nei metadata GGUF.
# ═════════════════════════════════════════════════════════════════════════════
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "$DIR/build/eie-llm" "$DIR/models/tinyllama.Q4_K_M.gguf" "$@"
