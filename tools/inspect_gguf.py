#!/usr/bin/env python3
# ─────────────────────────────────────────────
#  EIE-LLM — Inspector GGUF
#
#  Legge e stampa header, metadata e lista
#  tensori di qualsiasi file GGUF senza
#  dipendenze esterne (solo stdlib Python).
#
#  Uso:
#    python3 tools/inspect_gguf.py <file.gguf>
#    python3 tools/inspect_gguf.py <file.gguf> --tensors 50
# ─────────────────────────────────────────────

import struct
import sys
import os

def read_str(f):
    l = struct.unpack("<Q", f.read(8))[0]
    return f.read(l).decode("utf-8", errors="replace")

def read_val(f, t):
    if t == 0:  return struct.unpack("<B", f.read(1))[0]   # uint8
    if t == 1:  return struct.unpack("<b", f.read(1))[0]   # int8
    if t == 2:  return struct.unpack("<H", f.read(2))[0]   # uint16
    if t == 3:  return struct.unpack("<h", f.read(2))[0]   # int16
    if t == 4:  return struct.unpack("<I", f.read(4))[0]   # uint32
    if t == 5:  return struct.unpack("<i", f.read(4))[0]   # int32
    if t == 6:  return struct.unpack("<f", f.read(4))[0]   # float32
    if t == 7:  return bool(struct.unpack("<B", f.read(1))[0])  # bool
    if t == 8:  return read_str(f)                          # string
    if t == 9:                                              # array
        et, n = struct.unpack("<IQ", f.read(12))
        # Leggi tutti gli elementi ma mostra solo il conteggio
        for _ in range(n):
            read_val(f, et)
        return f"[array:{n} x type:{et}]"
    if t == 10: return struct.unpack("<Q", f.read(8))[0]   # uint64
    if t == 11: return struct.unpack("<q", f.read(8))[0]   # int64
    if t == 12: return struct.unpack("<d", f.read(8))[0]   # float64
    raise ValueError(f"Tipo sconosciuto: {t}")

# Mappa tipo tensore → nome leggibile
GGML_TYPE = {
    0:  "F32",
    1:  "F16",
    2:  "Q4_0",
    3:  "Q4_1",
    6:  "Q5_0",
    7:  "Q5_1",
    8:  "Q8_0",
    9:  "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
}

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 tools/inspect_gguf.py <file.gguf> [--tensors N]")
        sys.exit(1)

    path = sys.argv[1]
    max_tensors = 999999

    # Argomento opzionale --tensors N
    if "--tensors" in sys.argv:
        idx = sys.argv.index("--tensors")
        max_tensors = int(sys.argv[idx + 1])

    if not os.path.exists(path):
        print(f"[ERRORE] File non trovato: {path}")
        sys.exit(1)

    f = open(path, "rb")

    # ── Header ───────────────────────────────
    magic = f.read(4)
    if magic != b"GGUF":
        print(f"[ERRORE] Magic non valido: {magic}")
        sys.exit(1)

    version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

    print(f"\n{'═'*60}")
    print(f"  EIE-LLM — GGUF Inspector")
    print(f"{'═'*60}")
    print(f"  File     : {os.path.basename(path)}")
    print(f"  Size     : {os.path.getsize(path) / 1024 / 1024:.1f} MB")
    print(f"  Versione : {version}")
    print(f"  Tensori  : {n_tensors}")
    print(f"  Metadata : {n_kv}")
    print(f"{'─'*60}")

    # ── Metadata KV ──────────────────────────
    print(f"  METADATA")
    print(f"{'─'*60}")
    for i in range(n_kv):
        key = read_str(f)
        t   = struct.unpack("<I", f.read(4))[0]
        val = read_val(f, t)
        print(f"  {key:<45} = {val}")

    # ── Tensori ───────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  TENSORI (primi {min(max_tensors, n_tensors)})")
    print(f"{'─'*60}")
    print(f"  {'Nome':<45} {'Tipo':<8} Shape")
    print(f"{'─'*60}")

    for i in range(n_tensors):
        name   = read_str(f)
        n_dims = struct.unpack("<I", f.read(4))[0]
        shape  = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
        t_type = struct.unpack("<I", f.read(4))[0]
        offset = struct.unpack("<Q", f.read(8))[0]

        type_name = GGML_TYPE.get(t_type, f"type:{t_type}")
        shape_str = " x ".join(str(s) for s in shape)

        if i < max_tensors:
            print(f"  {name:<45} {type_name:<8} [{shape_str}]")

    print(f"{'═'*60}\n")
    f.close()

if __name__ == "__main__":
    main()