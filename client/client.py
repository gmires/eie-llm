#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EIE-LLM Client Python

Client ufficiale per interrogare il server EIE-LLM.
Supporta tutti gli endpoint:
  - GET  /health
  - POST /v1/completions
  - POST /v1/chat/completions
  - POST /v1/inspect/attention

E supporta lo streaming SSE per vedere i token
man mano che vengono generati.

Esempi:
  # Completamento semplice
  python client.py --mode complete --prompt "Hello"

  # Chat con TinyLlama
  python client.py --mode chat --message "Ciao!"

  # Streaming
  python client.py --mode complete --prompt "Hello" --stream

  # Attention heatmap
  python client.py --mode inspect --prompt "The cat sat" --plot

  # Layer e head specifici
  python client.py --mode inspect --prompt "Hello" --plot --layer 0 --head 0
"""

import argparse
import json
import sys
import os

# ── Dipendenze opzionali ──
# requests è necessario per tutte le chiamate HTTP
# matplotlib e seaborn solo per --plot
try:
    import requests
except ImportError:
    print("[ERRORE] 'requests' non installato.")
    print("  pip install requests")
    sys.exit(1)


# ═════════════════════════════════════════════
#  Classe principale: client API
# ═════════════════════════════════════════════

class EIEClient:
    """
    Client per l'API HTTP di EIE-LLM.

    Ogni metodo corrisponde a un endpoint del server.
    La base URL può essere configurata (default: localhost:8080).
    """

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    # ── GET /health ──────────────────────────
    def health(self) -> dict:
        """
        Verifica che il server sia attivo.
        Ritorna {"status": "ok", "model": "gpt2"}.
        """
        r = requests.get(f"{self.base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    # ── POST /v1/completions ─────────────────
    def complete(self, prompt: str, **kwargs) -> dict:
        """
        Completamento di testo.

        Parametri:
          prompt           : testo di input (obbligatorio)
          max_tokens       : token massimi da generare (default 50)
          temperature      : temperatura del sampling (default 1.0)
          top_k            : top-k sampling (default 40)
          top_p            : nucleus sampling (default 0.9)
          repetition_penalty : penalita ripetizioni (default 1.1)
          stream           : se True, ritorna un generator SSE
          chat             : se True, applica il chat template
          greedy           : se True, ignora temperatura e fa argmax
        """
        payload = {"prompt": prompt}
        payload.update(kwargs)

        if kwargs.get("stream"):
            # Streaming: ritorna un generator
            return self._stream_sse(
                f"{self.base_url}/v1/completions", payload
            )

        r = requests.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            timeout=300,
        )
        r.raise_for_status()
        return r.json()

    # ── POST /v1/chat/completions ────────────
    def chat_complete(self, message: str, **kwargs) -> dict:
        """
        Chat in stile OpenAI.

        Parametri:
          message          : messaggio dell'utente (obbligatorio)
          max_tokens       : token massimi da generare (default 100)
          temperature      : temperatura del sampling (default 1.0)
          top_k            : top-k sampling (default 40)
          top_p            : nucleus sampling (default 0.9)
          repetition_penalty : penalita ripetizioni (default 1.1)
          stream           : se True, ritorna un generator SSE
          greedy           : se True, ignora temperatura e fa argmax
        """
        payload = {
            "messages": [
                {"role": "user", "content": message}
            ]
        }
        payload.update(kwargs)

        if kwargs.get("stream"):
            return self._stream_sse(
                f"{self.base_url}/v1/chat/completions", payload
            )

        r = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=300,
        )
        r.raise_for_status()
        return r.json()

    # ── POST /v1/inspect/attention ───────────
    def inspect_attention(self, prompt: str, max_len: int = 100) -> dict:
        """
        Esporta gli attention scores per un prompt.
        Ritorna un dict con:
          - tokens: lista di token (stringhe)
          - layers: lista di layer, ognuno con lista di head,
            ognuno con matrice weights[q][k].
        """
        r = requests.post(
            f"{self.base_url}/v1/inspect/attention",
            json={"prompt": prompt, "max_len": max_len},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()

    # ── Streaming SSE (generator) ────────────
    def _stream_sse(self, url: str, payload: dict):
        """
        Generatore per lo streaming SSE.

        Yields ogni token man mano che arriva dal server.
        Formato:
          {"choices": [{"text": "...", "index": 0, "finish_reason": null}]}

        Quando finish_reason != null, lo yield come ultimo evento.
        """
        with requests.post(url, json=payload, stream=True, timeout=300) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                        yield parsed
                    except json.JSONDecodeError:
                        # Salta linee non-JSON (es. vuote)
                        pass


# ═════════════════════════════════════════════
#  Visualizzazione Attention Heatmap
# ═════════════════════════════════════════════

def plot_attention(data: dict, layer: int = None, head: int = None,
                   output_file: str = None):
    """
    Plotta l'attention heatmap usando matplotlib e seaborn.

    Parametri:
      data        : dict ritornato da inspect_attention()
      layer       : se specificato, mostra solo questo layer
      head        : se specificato, mostra solo questo head
      output_file : se specificato, salva l'immagine invece di mostrarla

    Comportamento:
      - Se layer e head sono entrambi specificati: mostra
        una singola heatmap per quella combinazione.
      - Se solo layer è specificato: mostra una griglia
        con tutti i head di quel layer.
      - Se nessuno è specificato: mostra la media di tutti
        i head per ogni layer (una heatmap per layer).
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
    except ImportError:
        print("[ERRORE] matplotlib/seaborn/numpy non installati.")
        print("  pip install matplotlib seaborn numpy")
        sys.exit(1)

    tokens = data["tokens"]
    layers = data["layers"]
    n_layers = len(layers)
    n_heads = len(layers[0]["heads"]) if layers else 0

    if layer is not None and head is not None:
        # ── Singola heatmap: layer X head ──
        weights = np.array(layers[layer]["heads"][head]["weights"])
        _plot_single_heatmap(weights, tokens,
                             f"Layer {layer}, Head {head}",
                             output_file)

    elif layer is not None:
        # ── Griglia: tutti i head di un layer ──
        heads_data = layers[layer]["heads"]
        n = len(heads_data)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for idx, hdata in enumerate(heads_data):
            r = idx // cols
            c = idx % cols
            weights = np.array(hdata["weights"])
            ax = axes[r][c]
            sns.heatmap(weights, ax=ax, cmap="viridis",
                        xticklabels=tokens, yticklabels=tokens,
                        cbar=False, square=True, vmin=0, vmax=1)
            ax.set_title(f"Head {hdata['head']}", fontsize=8)
            ax.tick_params(axis="both", labelsize=6)

        # Nascondi subplot vuoti
        for idx in range(n, rows * cols):
            r = idx // cols
            c = idx % cols
            axes[r][c].axis("off")

        fig.suptitle(f"Attention Heatmap — Layer {layer}", fontsize=12)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"Immagine salvata: {output_file}")
        else:
            plt.show()

    else:
        # ── Media per layer: tutti i layer ──
        n = n_layers
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        for idx, ldata in enumerate(layers):
            r = idx // cols
            c = idx % cols

            # Media di tutti i head per questo layer
            all_weights = [np.array(h["weights"]) for h in ldata["heads"]]
            avg_weights = np.mean(all_weights, axis=0)

            ax = axes[r][c]
            sns.heatmap(avg_weights, ax=ax, cmap="viridis",
                        xticklabels=tokens, yticklabels=tokens,
                        cbar=True, square=True, vmin=0, vmax=1)
            ax.set_title(f"Layer {ldata['layer']}", fontsize=8)
            ax.tick_params(axis="both", labelsize=6)

        # Nascondi subplot vuoti
        for idx in range(n, rows * cols):
            r = idx // cols
            c = idx % cols
            axes[r][c].axis("off")

        fig.suptitle("Attention Heatmap — Media per Layer", fontsize=12)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            print(f"Immagine salvata: {output_file}")
        else:
            plt.show()


def _plot_single_heatmap(weights, tokens, title, output_file=None):
    """Helper: plotta una singola heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 5))
    sns.heatmap(weights, cmap="viridis",
                xticklabels=tokens, yticklabels=tokens,
                cbar=True, square=True, vmin=0, vmax=1)
    plt.title(title)
    plt.xlabel("Key Token")
    plt.ylabel("Query Token")
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"Immagine salvata: {output_file}")
    else:
        plt.show()


# ═════════════════════════════════════════════
#  CLI principale
# ═════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Client Python per EIE-LLM Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Health check
  python client.py --mode health

  # Completamento
  python client.py --mode complete --prompt "The sky is"

  # Chat
  python client.py --mode chat --message "Ciao!"

  # Streaming
  python client.py --mode complete --prompt "Hello" --stream

  # Attention heatmap (tutti i layer)
  python client.py --mode inspect --prompt "The cat sat" --plot

  # Attention singolo layer/head
  python client.py --mode inspect --prompt "Hello" --plot --layer 0 --head 0

  # Parametri di sampling
  python client.py --mode complete --prompt "Hello" -t 0.8 --top-k 20 --max-tokens 100
        """,
    )

    parser.add_argument(
        "--host", default="localhost",
        help="Host del server (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Porta del server (default: 8080)",
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["health", "complete", "chat", "inspect"],
        help="Modalità di operazione",
    )

    # Input
    parser.add_argument("--prompt", help="Prompt per complete/inspect")
    parser.add_argument("--message", help="Messaggio per chat")

    # Parametri di sampling
    parser.add_argument(
        "-t", "--temperature", type=float, default=1.0,
        help="Temperatura del sampling (default: 1.0)",
    )
    parser.add_argument(
        "--top-k", type=int, default=40,
        help="Top-k sampling (default: 40)",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.9,
        help="Nucleus sampling p (default: 0.9)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help="Token massimi da generare",
    )
    parser.add_argument(
        "--repetition-penalty", type=float, default=1.1,
        help="Penalità ripetizioni (default: 1.1)",
    )
    parser.add_argument(
        "--greedy", action="store_true",
        help="Attiva sampling greedy (ignora temperatura)",
    )
    parser.add_argument(
        "--chat-template", action="store_true",
        help="Applica il chat template (solo --mode complete)",
    )

    # Streaming
    parser.add_argument(
        "--stream", action="store_true",
        help="Attiva streaming SSE (stampa token man mano)",
    )

    # Inspect
    parser.add_argument(
        "--plot", action="store_true",
        help="Plotta la heatmap di attention (solo --mode inspect)",
    )
    parser.add_argument(
        "--layer", type=int, default=None,
        help="Mostra solo questo layer (solo con --plot)",
    )
    parser.add_argument(
        "--head", type=int, default=None,
        help="Mostra solo questo head (solo con --plot)",
    )
    parser.add_argument(
        "--output", default=None,
        help="File di output per l'immagine (solo con --plot)",
    )
    parser.add_argument(
        "--max-len", type=int, default=100,
        help="Lunghezza massima prompt per inspect (default: 100)",
    )

    # Output
    parser.add_argument(
        "--raw", action="store_true",
        help="Stampa il JSON raw invece del testo formattato",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Crea il client
    base_url = f"http://{args.host}:{args.port}"
    client = EIEClient(base_url)

    # ═════════════════════════════════════════
    #  Modalità: health
    # ═════════════════════════════════════════
    if args.mode == "health":
        try:
            result = client.health()
            if args.raw:
                print(json.dumps(result, indent=2))
            else:
                print(f"Stato: {result.get('status', 'unknown')}")
                print(f"Modello: {result.get('model', 'unknown')}")
        except requests.ConnectionError:
            print(f"[ERRORE] Impossibile connettersi a {base_url}")
            print("  Assicurati che il server sia attivo:")
            print(f"    ./build/eie-llm models/gpt2.Q8_0.gguf --server {args.port}")
            sys.exit(1)

    # ═════════════════════════════════════════
    #  Modalità: complete
    # ═════════════════════════════════════════
    elif args.mode == "complete":
        if not args.prompt:
            print("[ERRORE] --prompt obbligatorio per --mode complete")
            sys.exit(1)

        params = {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "greedy": args.greedy,
            "stream": args.stream,
            "chat": args.chat_template,
        }
        if args.max_tokens is not None:
            params["max_tokens"] = args.max_tokens

        try:
            if args.stream:
                # Streaming: stampa token man mano
                print(args.prompt, end="", flush=True)
                for event in client.complete(args.prompt, **params):
                    choices = event.get("choices", [])
                    if not choices:
                        continue
                    text = choices[0].get("text", "")
                    delta = choices[0].get("delta", {})
                    finish = choices[0].get("finish_reason")

                    token = text or delta.get("content", "")
                    if token:
                        print(token, end="", flush=True)
                    if finish:
                        print()  # newline alla fine
                        break
            else:
                result = client.complete(args.prompt, **params)
                if args.raw:
                    print(json.dumps(result, indent=2))
                else:
                    text = result["choices"][0]["text"]
                    usage = result.get("usage", {})
                    print(f"{args.prompt}{text}")
                    print(f"\n[Usage] prompt: {usage.get('prompt_tokens')}, "
                          f"completion: {usage.get('completion_tokens')}, "
                          f"total: {usage.get('total_tokens')}")
        except requests.ConnectionError:
            print(f"[ERRORE] Impossibile connettersi a {base_url}")
            sys.exit(1)

    # ═════════════════════════════════════════
    #  Modalità: chat
    # ═════════════════════════════════════════
    elif args.mode == "chat":
        if not args.message:
            print("[ERRORE] --message obbligatorio per --mode chat")
            sys.exit(1)

        params = {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "greedy": args.greedy,
            "stream": args.stream,
        }
        if args.max_tokens is not None:
            params["max_tokens"] = args.max_tokens

        try:
            if args.stream:
                print("Assistente: ", end="", flush=True)
                for event in client.chat_complete(args.message, **params):
                    choices = event.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    finish = choices[0].get("finish_reason")

                    token = delta.get("content", "")
                    if token:
                        print(token, end="", flush=True)
                    if finish:
                        print()
                        break
            else:
                result = client.chat_complete(args.message, **params)
                if args.raw:
                    print(json.dumps(result, indent=2))
                else:
                    text = result["choices"][0]["text"]
                    print(f"Assistente: {text}")
        except requests.ConnectionError:
            print(f"[ERRORE] Impossibile connettersi a {base_url}")
            sys.exit(1)

    # ═════════════════════════════════════════
    #  Modalità: inspect
    # ═════════════════════════════════════════
    elif args.mode == "inspect":
        if not args.prompt:
            print("[ERRORE] --prompt obbligatorio per --mode inspect")
            sys.exit(1)

        try:
            result = client.inspect_attention(args.prompt, args.max_len)

            if args.raw:
                print(json.dumps(result, indent=2))
            else:
                print(f"Prompt: {args.prompt}")
                print(f"Token: {result['tokens']}")
                print(f"Layer: {len(result['layers'])}, "
                      f"Head per layer: {len(result['layers'][0]['heads'])}")

            if args.plot:
                plot_attention(result, args.layer, args.head, args.output)

        except requests.ConnectionError:
            print(f"[ERRORE] Impossibile connettersi a {base_url}")
            sys.exit(1)


if __name__ == "__main__":
    main()
