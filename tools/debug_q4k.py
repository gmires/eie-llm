import struct, sys

if len(sys.argv) < 2:
    print("Uso: python3 tools/debug_q4k.py <model.gguf>")
    sys.exit(1)

f = open(sys.argv[1], "rb")

magic = f.read(4)
version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))

def read_str(f):
    l = struct.unpack("<Q", f.read(8))[0]
    return f.read(l).decode("utf-8", errors="replace")

def read_val(f, t):
    if t == 0:  return struct.unpack("<B", f.read(1))[0]
    if t == 1:  return struct.unpack("<b", f.read(1))[0]
    if t == 2:  return struct.unpack("<H", f.read(2))[0]
    if t == 3:  return struct.unpack("<h", f.read(2))[0]
    if t == 4:  return struct.unpack("<I", f.read(4))[0]
    if t == 5:  return struct.unpack("<i", f.read(4))[0]
    if t == 6:  return struct.unpack("<f", f.read(4))[0]
    if t == 7:  return bool(struct.unpack("<B", f.read(1))[0])
    if t == 8:  return read_str(f)
    if t == 9:
        et, n = struct.unpack("<IQ", f.read(12))
        for _ in range(n): read_val(f, et)
        return None
    if t == 10: return struct.unpack("<Q", f.read(8))[0]
    if t == 11: return struct.unpack("<q", f.read(8))[0]
    if t == 12: return struct.unpack("<d", f.read(8))[0]

for _ in range(n_kv):
    read_str(f)
    t = struct.unpack("<I", f.read(4))[0]
    read_val(f, t)

tensors = {}
for _ in range(n_tensors):
    name   = read_str(f)
    n_dims = struct.unpack("<I", f.read(4))[0]
    shape  = [struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims)]
    t_type, offset = struct.unpack("<IQ", f.read(12))
    tensors[name] = (shape, t_type, offset)

pos = f.tell()
data_offset = (pos + 31) // 32 * 32

def fp16(h):
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    if e == 0:    v = m / (1 << 24)
    elif e == 31: v = float('inf')
    else:         v = (1 + m / 1024.0) * (2 ** (e - 15))
    return -v if s else v

name = "token_embd.weight"
shape, t_type, offset = tensors[name]
print(f"\nTensore : {name}")
print(f"Shape   : {shape}")
print(f"Tipo    : {t_type} (12=Q4_K)")
print(f"Offset  : {offset}")

f.seek(data_offset + offset)
block = f.read(144)

print(f"\n── Super-block 0 (144 byte) ──")
print(f"Bytes raw [0:16] : {[hex(b) for b in block[:16]]}")

d    = fp16(struct.unpack("<H", block[0:2])[0])
dmin = fp16(struct.unpack("<H", block[2:4])[0])
print(f"d    = {d:.6f}")
print(f"dmin = {dmin:.6f}")

sc_bytes = list(block[4:16])
print(f"Scale bytes      : {[hex(b) for b in sc_bytes]}")

print(f"\n── Scale/Min per sub-block ──")
for b in range(8):
    bit_off = b * 12
    byte_i  = bit_off // 8
    shift   = bit_off % 8
    raw = sc_bytes[byte_i]
    if byte_i + 1 < 12: raw |= sc_bytes[byte_i+1] << 8
    if byte_i + 2 < 12: raw |= sc_bytes[byte_i+2] << 16
    raw >>= shift
    scale_b = (raw & 0x3F) * d
    min_b   = ((raw >> 6) & 0x3F) * dmin
    print(f"  sub-block {b}: scale={scale_b:.6f}  min={min_b:.6f}")

raw0   = sc_bytes[0] | (sc_bytes[1] << 8)
scale0 = (raw0 & 0x3F) * d
min0   = ((raw0 >> 6) & 0x3F) * dmin

nibbles = block[16:]
print(f"\n── Primi 16 valori dequantizzati (sub-block 0) ──")
vals = []
for i in range(16):
    byte   = nibbles[i // 2]
    nibble = (byte & 0xF) if (i % 2 == 0) else (byte >> 4)
    vals.append(nibble * scale0 - min0)
print(f"  {[f'{v:.4f}' for v in vals]}")

print(f"\n── Range valori ──")
print(f"  min={min(vals):.4f}  max={max(vals):.4f}")
print(f"  (ragionevole: tra -2.0 e +2.0)")
