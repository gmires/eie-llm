# EIE-LLM — Educational Inference Engine

Motore di inferenza LLM scritto in C++17 da zero, con scopo didattico.
Obiettivo: capire come funziona un inference engine moderno implementandolo
passo passo, dal parsing del file GGUF fino a un server HTTP minimale.

## Modello target

GPT-2 small (124M parametri) in formato GGUF — architettura semplice,
ideale per capire i meccanismi base prima di passare a modelli più grandi.

## Roadmap

- [x] Fase 1 — Parser GGUF: header + metadata KV
- [ ] Fase 2 — Lettura info tensori
- [ ] Fase 3 — Tokenizer BPE
- [ ] Fase 4 — Forward pass GPT-2
- [ ] Fase 5 — Shell interattiva
- [ ] Fase 6 — Server HTTP minimale

## Requisiti

- CMake >= 3.16
- GCC >= 13 o Clang >= 15
- C++17

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

## Download modello

```bash
chmod +x scripts/download_model.sh
./scripts/download_model.sh
```

## Uso

```bash
./build/eie-llm models/gpt2.Q8_0.gguf
```

## Struttura del progetto

```
eie-llm/
├── include/        # Header pubblici
├── src/            # Implementazioni
├── models/         # Modelli GGUF (non inclusi nel repo)
└── scripts/        # Utility
```