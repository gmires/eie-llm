#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
#  EIE-LLM — GPT-2 small  (124M, Q8_0)
#
#  Parametri consigliati per GPT-2 (modello base, completamento libero):
#    :temp 1.0          → Temperatura neutra (default)
#    :topk 40           → Top-k sampling predefinito
#    :topp 0.9          → Nucleus sampling
#    :penalty 1.0       → Nessuna penalità ripetizioni
#
#  GPT-2 non ha un chat template: parte in modalità raw.
#  Il prompt viene passato direttamente al modello per completamento.
# ═════════════════════════════════════════════════════════════════════════════
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "$DIR/build/eie-llm" "$DIR/models/gpt2.Q8_0.gguf"
