#!/usr/bin/env python3
"""
Verifica la dequantizzazione Q4_K contro l'algoritmo di riferimento ggml.
Legge i primi super-block di token_embd.weight e stampa i valori attesi.
"""
import struct, sys, math

def fp16(h):
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    if e == 0:
        v = m / (1 << 24)
    elif e == 31:
        v = float('inf')
    else:
        v = (1 + m / 1024.0) * (2 ** (e - 15))
    return -v if s else v

def get_scale_min_k4(j, sc):
    """get_scale_min_k4 da ggml-quants.c"""
    if j < 4:
        sv = sc[j]   & 0x3F
        mv = sc[j+4] & 0x3F
    else:
        sv = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4)
        mv = (sc[j+4] >>   4) | ((sc[j]   >> 6) << 4)
    return sv, mv

def dequant_q4k_block(block_bytes):
    """
    Dequantizza un super-block Q4_K (144 byte, 256 elementi).
    Algoritmo di riferimento da ggml-quants.c:
      for j in 0..3 (QK_K/64):
        for l in 0..31:
          y[j*64 + l]    = d1 * (q[j*32+l] & 0xF) - m1
          y[j*64 + 32+l] = d2 * (q[j*32+l] >>  4) - m2
    """
    assert len(block_bytes) == 144
    d    = fp16(struct.unpack_from('<H', block_bytes, 0)[0])
    dmin = fp16(struct.unpack_from('<H', block_bytes, 2)[0])
    sc   = list(block_bytes[4:16])
    qs   = list(block_bytes[16:144])

    scales = [0.0] * 8
    mins   = [0.0] * 8
    for b in range(8):
        sv, mv = get_scale_min_k4(b, sc)
        scales[b] = sv * d
        mins[b]   = mv * dmin

    out = [0.0] * 256
    for j in range(4):          # QK_K/64 = 4
        sc0 = scales[j*2];   m0 = mins[j*2]
        sc1 = scales[j*2+1]; m1 = mins[j*2+1]
        for l in range(32):
            byte = qs[j*32 + l]
            out[j*64 + l]    = (byte & 0xF) * sc0 - m0
            out[j*64 + 32+l] = (byte >>  4) * sc1 - m1
    return out, d, dmin, scales, mins

# ── Lettura GGUF ───────────────────────────────────────────────────────────
model_path = sys.argv[1] if len(sys.argv) > 1 else "models/tinyllama.Q4_K_M.gguf"

def read_str(f):
    l = struct.unpack('<Q', f.read(8))[0]
    return f.read(l).decode('utf-8', errors='replace')

def read_val(f, t):
    if t == 0:  return struct.unpack('<B', f.read(1))[0]
    if t == 1:  return struct.unpack('<b', f.read(1))[0]
    if t == 2:  return struct.unpack('<H', f.read(2))[0]
    if t == 3:  return struct.unpack('<h', f.read(2))[0]
    if t == 4:  return struct.unpack('<I', f.read(4))[0]
    if t == 5:  return struct.unpack('<i', f.read(4))[0]
    if t == 6:  return struct.unpack('<f', f.read(4))[0]
    if t == 7:  return bool(struct.unpack('<B', f.read(1))[0])
    if t == 8:  return read_str(f)
    if t == 9:
        et, n = struct.unpack('<IQ', f.read(12))
        return [read_val(f, et) for _ in range(n)]
    if t == 10: return struct.unpack('<Q', f.read(8))[0]
    if t == 11: return struct.unpack('<q', f.read(8))[0]
    if t == 12: return struct.unpack('<d', f.read(8))[0]

with open(model_path, 'rb') as f:
    f.read(4)  # magic
    version, n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))
    for _ in range(n_kv):
        read_str(f)
        t = struct.unpack('<I', f.read(4))[0]
        read_val(f, t)
    tensors = {}
    for _ in range(n_tensors):
        name = read_str(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        t_type, offset = struct.unpack('<IQ', f.read(12))
        tensors[name] = (shape, t_type, offset)
    pos = f.tell()
    data_offset = (pos + 31) // 32 * 32

    # ── token_embd.weight ───────────────────────────────────────────────
    shape, t_type, offset = tensors['token_embd.weight']
    print(f"token_embd.weight: type={t_type} (12=Q4_K) shape={shape}")
    n_embd = shape[0]  # 2048

    # Leggi i super-block per token 0 (embd[0..2047]) e token 1 (embd[2048..4095])
    # Ogni token ha 2048/256 = 8 super-block
    n_sb_per_token = n_embd // 256  # = 8

    all_vals = []
    for tok_id in range(3):  # prime 3 token
        tok_vals = []
        for sb in range(n_sb_per_token):
            global_sb = tok_id * n_sb_per_token + sb
            f.seek(data_offset + offset + global_sb * 144)
            block = f.read(144)
            vals, d, dmin, scales, mins = dequant_q4k_block(block)
            tok_vals.extend(vals)
            if tok_id == 1 and sb == 0:
                print(f"\n  Token 1 sb0: d={d:.6f} dmin={dmin:.6f}")
                print(f"  scales[0..3]={[f'{s:.4f}' for s in scales[:4]]}")
                print(f"  mins[0..3]  ={[f'{m:.4f}' for m in mins[:4]]}")
                print(f"  qs[0..7]    ={list(block[16:24])}")
                print(f"  vals[0..7]  ={[f'{v:.4f}' for v in vals[:8]]}")
        all_vals.append(tok_vals)

    for tok_id in range(3):
        v = all_vals[tok_id]
        rms = math.sqrt(sum(x*x for x in v) / len(v))
        print(f"\nToken {tok_id}: rms={rms:.4f}  vals[0..7]={[f'{x:.4f}' for x in v[:8]]}")

print("\n── Verifica get_scale_min_k4 per j=0..7 sul primo sb di token 1 ──")
with open(model_path, 'rb') as f:
    f.read(4)
    version, n_tensors, n_kv = struct.unpack('<IQQ', f.read(20))
    for _ in range(n_kv):
        read_str(f)
        t = struct.unpack('<I', f.read(4))[0]
        read_val(f, t)
    tensors2 = {}
    for _ in range(n_tensors):
        name = read_str(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        shape = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        t_type, offset = struct.unpack('<IQ', f.read(12))
        tensors2[name] = (shape, t_type, offset)
    pos = f.tell()
    data_offset = (pos + 31) // 32 * 32
    shape, t_type, offset = tensors2['token_embd.weight']
    # super-block 8 (primo sb di token 1)
    f.seek(data_offset + offset + 8 * 144)
    block = f.read(144)
d    = fp16(struct.unpack_from('<H', block, 0)[0])
dmin = fp16(struct.unpack_from('<H', block, 2)[0])
sc   = list(block[4:16])
print(f"d={d:.6f}  dmin={dmin:.6f}  scale_bytes={[hex(b) for b in sc]}")
for j in range(8):
    sv, mv = get_scale_min_k4(j, sc)
    print(f"  j={j}: sv={sv} scale={sv*d:.6f}  mv={mv} min={mv*dmin:.6f}")
