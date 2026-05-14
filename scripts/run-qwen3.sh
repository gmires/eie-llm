#!/bin/bash
# ═════════════════════════════════════════════════════════════════════════════
#  EIE-LLM — Qwen3-1.7B-Instruct  (Q8_0) — Thinking Mode
#
#  Qwen3 ha DUE modalità di funzionamento:
#
#  🔍 Thinking mode (default, consigliato per logica/codice)
#    :temp 0.6  :topk 20  :topp 0.95  :penalty 1.5  :think on
#    Il modello ragiona prima di rispondere (tag <think>...).
#    NON usare greedy decoding — causa ripetizioni infinite.
#    Fonte: model card ufficiale Qwen3 (HuggingFace)
#
#  ⚡ Non-thinking mode (chat veloce)
#    :temp 0.7  :topk 20  :topp 0.8  :penalty 1.5  :think off
#    Risposte dirette senza ragionamento esplicito.
#
#  Comandi utili:
#    :think on/off        Cambia modalità al volo
#    /think o /no_think   Nel prompt utente (soft switch per turno)
# ═════════════════════════════════════════════════════════════════════════════
set -e
DIR="$(cd "$(dirname "$0")/.." && pwd)"
exec "$DIR/build/eie-llm" "$DIR/models/qwen3-1.7b.Q8_0.gguf" "$@"
