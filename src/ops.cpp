#include "ops.hpp"
#include "cpuinfo.hpp"
#include "ops_avx2.hpp"
#include <cstring>
#include <algorithm>
#include <cassert>
#include <iostream>

// ─────────────────────────────────────────────
//  Conversione float16 → float32
//
//  IEEE 754 float16: 1 bit segno, 5 bit esponente (bias 15),
//  10 bit mantissa. Tre casi distinti:
//
//  1) Esponente = 0, mantissa = 0 → ±zero
//
//  2) Esponente = 0, mantissa ≠ 0 → subnormale fp16
//     Valore = 2^(-14) * mantissa/1024
//     Si normalizza shiftando la mantissa a sinistra finché
//     il bit 10 è settato (trovando l'1 implicito), poi si
//     ribiasa per fp32 con: exponent_fp32 = 113 - k
//     dove k è il numero di shift eseguiti.
//     (113 = 127_bias_fp32 - 14_esponente_denormale_fp16)
//
//  3) Esponente = 31 → ±infinito o NaN
//
//  4) Caso normale: ribiasa l'esponente (bias 15 → bias 127)
//     ed estende la mantissa da 10 a 23 bit con shift di 13.
// ─────────────────────────────────────────────
float fp16_to_fp32(uint16_t h) {
    uint32_t sign     = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    uint32_t result;
    if (exponent == 0) {
        if (mantissa == 0) {
            // ±zero
            result = sign << 31;
        } else {
            // Subnormale: cerchiamo l'1 implicito shiftando a sinistra.
            // Ogni shift decrementa l'esponente (parte da 0 → diventa negativo
            // come uint32, poi viene ribilanciato nella formula).
            while (!(mantissa & 0x400)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x3FF;           // rimuove il bit implicito
            exponent = exponent + 127 - 14; // ribias: 113 - k (k shift eseguiti)
            result = (sign << 31) | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Infinito o NaN: esponente fp32 tutto a 1 (0xFF)
        result = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        // Normale: ribilancia il bias (15 → 127) ed estende la mantissa
        exponent = exponent + 127 - 15;
        result = (sign << 31) | (exponent << 23) | (mantissa << 13);
    }
    float f; memcpy(&f, &result, sizeof(f)); return f;
}


// ─────────────────────────────────────────────
//  Dequantizzazione Q8_0
//
//  Layout memoria per ogni blocco (34 byte):
//    byte 0-1  : scale in float16
//    byte 2-33 : 32 valori int8
//
//  Per ottenere il float originale:
//    f = int8_value * scale
//
//  Nota: il block size è sempre 32 per Q8_0
// ─────────────────────────────────────────────
void dequantize_q8_0(const uint8_t* src, float* dst, uint64_t n_elem) {
    // Ogni blocco contiene 32 elementi
    static constexpr int BLOCK_SIZE = 32;
    // Ogni blocco occupa 34 byte: 2 (scale f16) + 32 (int8)
    static constexpr int BLOCK_BYTES = 34;

    uint64_t n_blocks = n_elem / BLOCK_SIZE;

    for (uint64_t b = 0; b < n_blocks; b++) {
        const uint8_t* block = src + b * BLOCK_BYTES;

        // Leggi lo scale factor (float16, primi 2 byte del blocco)
        uint16_t scale_bits;
        memcpy(&scale_bits, block, sizeof(scale_bits));
        float scale = fp16_to_fp32(scale_bits);

        // Dequantizza i 32 valori int8 del blocco
        const int8_t* quants = reinterpret_cast<const int8_t*>(block + 2);
        float* out = dst + b * BLOCK_SIZE;

        for (int i = 0; i < BLOCK_SIZE; i++)
            out[i] = quants[i] * scale;
    }
}

// ─────────────────────────────────────────────
//  Dequantizzazione Q4_K  (da ggml-quants.c)
//
//  Formato super-block (256 elementi, 144 byte):
//
//  Offset  Size  Contenuto
//  0       2     d    — scale globale per gli scale dei sub-block (float16)
//  2       2     dmin — scale globale per i minimi dei sub-block  (float16)
//  4       12    scales: 8 scale + 8 minimi packed a 6 bit ciascuno
//  16      128   qs: nibble 4 bit × 256 elementi
//
//  Il super-block è diviso in 8 sub-block da 32 elementi.
//  Ogni coppia di sub-block (2j, 2j+1) condivide 32 byte di qs:
//    - low nibble  di qs[j*32 + l] → elemento l       del sub-block 2j
//    - high nibble di qs[j*32 + l] → elemento l       del sub-block 2j+1
//
//  Formula finale per ogni elemento:
//    val = nibble * (sv * d) - (mv * dmin)
//
//  I 6 bit di scale e min per ogni sub-block vengono estratti
//  con get_scale_min_k4, che usa i 12 byte di scala compressa.
// ─────────────────────────────────────────────
void dequantize_q4_k(const uint8_t* src, float* dst, uint64_t n_elem) {
    static constexpr int SUPER_BLOCK = 256;
    static constexpr int BLOCK_SIZE  = 32;
    static constexpr int N_SUB       = 8;
    static constexpr int SUPER_BYTES = 144;

    uint64_t n_super = n_elem / SUPER_BLOCK;

    for (uint64_t s = 0; s < n_super; s++) {
        const uint8_t* sb = src + s * SUPER_BYTES;

        uint16_t d_bits, dmin_bits;
        memcpy(&d_bits,    sb,     2);
        memcpy(&dmin_bits, sb + 2, 2);

        float d    = fp16_to_fp32(d_bits);
        float dmin = fp16_to_fp32(dmin_bits);

        const uint8_t* sc = sb + 4;
        float scales[N_SUB], mins[N_SUB];

        // Estrae scale e min per il sub-block j dal campo scales compresso a 6 bit.
        // Per j < 4: i bit 0-5 del byte j sono lo scale, i bit 0-5 del byte j+4 il min.
        // Per j >= 4: i bit 6-7 dei byte precedenti forniscono i 2 bit alti.
        // Formula identica a get_scale_min_k4 in ggml-quants.c.
        auto get_scale_min = [&](int j) -> std::pair<float,float> {
            uint8_t sv, mv;
            if (j < 4) {
                sv = sc[j]   & 0x3F;
                mv = sc[j+4] & 0x3F;
            } else {
                sv = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4);
                mv = (sc[j+4] >>   4) | ((sc[j  ] >> 6) << 4);
            }
            return {sv * d, mv * dmin};
        };

        for (int b = 0; b < N_SUB; b++) {
            auto [sc_b, min_b] = get_scale_min(b);
            scales[b] = sc_b;
            mins[b]   = min_b;
        }

        // Dequantizza i nibbles secondo il layout ggml Q4_K.
        // I 128 byte di qs sono divisi in 4 gruppi da 32 byte.
        // Il gruppo j copre i sub-block 2j e 2j+1 (64 elementi totali):
        //   elementi 0..31  → low nibble  di qs[j*32 + l]  (sub-block 2j)
        //   elementi 32..63 → high nibble di qs[j*32 + l]  (sub-block 2j+1)
        const uint8_t* nibbles = sb + 16;
        float* out = dst + s * SUPER_BLOCK;

        for (int j = 0; j < N_SUB / 2; j++) {
            float sc0 = scales[j * 2],     min0 = mins[j * 2];
            float sc1 = scales[j * 2 + 1], min1 = mins[j * 2 + 1];
            const uint8_t* q = nibbles + j * BLOCK_SIZE;
            float* o = out + j * 2 * BLOCK_SIZE;
            for (int l = 0; l < BLOCK_SIZE; l++) {
                o[l]              = (q[l] & 0xF) * sc0 - min0;  // sub-block 2j
                o[l + BLOCK_SIZE] = (q[l] >>  4) * sc1 - min1;  // sub-block 2j+1
            }
        }
    }
}

// ─────────────────────────────────────────────
//  Dequantizzazione Q6_K  (da ggml-quants.c)
//
//  Formato super-block (256 elementi, 210 byte):
//
//  Offset  Size  Contenuto
//  0       128   ql: 4 bit bassi di ogni elemento (nibble packed, 2 elem/byte)
//  128      64   qh: 2 bit alti di ogni elemento  (4 elem/byte, 2 bit ciascuno)
//  192      16   scales: 16 scale int8, una per gruppo di 16 elementi
//  208       2   d: super-scale float16 (fattore globale)
//
//  Il super-block è diviso in 2 metà da 128 elementi ciascuna (n = 0, 1).
//  Ogni metà usa:
//    - 64 byte di ql  (4 bit bassi per 128 elementi)
//    - 32 byte di qh  (2 bit alti per 128 elementi)
//    - 8 scale int8   (sc_n = scales + n*8)
//
//  I 6 bit di ogni elemento si ricostruiscono combinando ql e qh:
//    q1 = (ql[l]    & 0xF) | ((qh[l] >> 0) & 3) << 4  → elemento l+ 0
//    q2 = (ql[l+32] & 0xF) | ((qh[l] >> 2) & 3) << 4  → elemento l+32
//    q3 = (ql[l]    >> 4)  | ((qh[l] >> 4) & 3) << 4  → elemento l+64
//    q4 = (ql[l+32] >> 4)  | ((qh[l] >> 6) & 3) << 4  → elemento l+96
//  Centro: ogni q viene spostato di -32 (valori da 0..63 → -32..31).
//
//  Le 8 scale per metà coprono 8 gruppi di 16 elementi. L'indice is = l/16
//  seleziona la coppia di scale corretta per l in [0,15] e [16,31]:
//    y[l+ 0] = d * sc_n[is+0] * q1
//    y[l+32] = d * sc_n[is+2] * q2
//    y[l+64] = d * sc_n[is+4] * q3
//    y[l+96] = d * sc_n[is+6] * q4
//  Gli offset +0/+2/+4/+6 separano le scale dei 4 "slot" (q1..q4)
//  all'interno della metà, mentre is (0 o 1) scorre le 2 sotto-fasce.
//
//  BUG FIX (rispetto alla versione precedente):
//    - sc_n puntava a sc + n*4 (4 scale per metà): ERRATO.
//      Il formato prevede 8 scale per metà (16 totali / 2 metà).
//    - Gli indici delle scale erano fissi sc_n[0..3] per tutte le
//      32 iterazioni: ERRATO. Devono variare con is = l/16,
//      esattamente come in ggml-quants.c (ggml_dequantize_row_q6_K).
//    Effetto: tutti i tensori Q6_K (output.weight, ffn_down, attn_v)
//    producevano valori completamente errati → output non-sense.
// ─────────────────────────────────────────────
void dequantize_q6_k(const uint8_t* src, float* dst, uint64_t n_elem) {
    static constexpr int SUPER_BLOCK = 256;
    static constexpr int SUPER_BYTES = 210;

    uint64_t n_super = n_elem / SUPER_BLOCK;

    for (uint64_t s = 0; s < n_super; s++) {
        const uint8_t* sb = src + s * SUPER_BYTES;
        const uint8_t* ql = sb;
        const uint8_t* qh = sb + 128;
        const int8_t*  sc = reinterpret_cast<const int8_t*>(sb + 192);

        uint16_t d_bits;
        memcpy(&d_bits, sb + 208, 2);
        float d = fp16_to_fp32(d_bits);

        float* out = dst + s * SUPER_BLOCK;

        // Due metà da 128 elementi ciascuna. Ogni metà ha 8 scale (una per
        // gruppo di 16 elementi). L'indice is = l/16 seleziona la coppia
        // di scale corretta ogni 16 iterazioni, replicando la logica di
        // ggml-quants.c: sc[is+0], sc[is+2], sc[is+4], sc[is+6].
        for (int n = 0; n < 2; n++) {
            const uint8_t* ql_n = ql + n * 64;
            const uint8_t* qh_n = qh + n * 32;
            const int8_t*  sc_n = sc  + n * 8;
            float*          y   = out + n * 128;

            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql_n[l]    & 0xF) | (((qh_n[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql_n[l+32] & 0xF) | (((qh_n[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql_n[l]    >>  4) | (((qh_n[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql_n[l+32] >>  4) | (((qh_n[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc_n[is + 0] * q1;
                y[l + 32] = d * sc_n[is + 2] * q2;
                y[l + 64] = d * sc_n[is + 4] * q3;
                y[l + 96] = d * sc_n[is + 6] * q4;
            }
        }
    }
}

// ─────────────────────────────────────────────
//  Converti un tensore GGUF in vettore float32
//
//  Dispatcha sul tipo di quantizzazione del tensore e
//  restituisce tutti gli elementi come float32 non compressi.
//
//  Tipi supportati:
//    F32   → copia diretta
//    F16   → conversione elemento per elemento via fp16_to_fp32
//    Q8_0  → dequantize_q8_0  (1 scale fp16 ogni 32 int8)
//    Q4_K  → dequantize_q4_k  (super-block 256 elem, nibble 4 bit)
//    Q6_K  → dequantize_q6_k  (super-block 256 elem, 6 bit/elem)
//
//  Tipi non supportati: stampa un errore su stderr e restituisce
//  un vettore di zeri della dimensione corretta.
// ─────────────────────────────────────────────
std::vector<float> tensor_to_float(const GGUFTensor& t) {
    uint64_t n_elem = gguf_tensor_n_elements(t.info);
    std::vector<float> out(n_elem);

    switch (t.type()) {
        case GGMLType::F32:
            memcpy(out.data(), t.data.data(), n_elem * sizeof(float));
            break;

        case GGMLType::F16: {
            const uint16_t* src = reinterpret_cast<const uint16_t*>
                                  (t.data.data());
            for (uint64_t i = 0; i < n_elem; i++)
                out[i] = fp16_to_fp32(src[i]);
            break;
        }

        case GGMLType::Q8_0:
            dequantize_q8_0(t.data.data(), out.data(), n_elem);
            break;

        case GGMLType::Q4_K:
            dequantize_q4_k(t.data.data(), out.data(), n_elem);
            break;

        case GGMLType::Q6_K:
            dequantize_q6_k(t.data.data(), out.data(), n_elem);
            break;

        default:
            std::cerr << "[ERRORE] Tipo tensore non supportato: "
                      << ggml_type_name(t.type()) << "\n";
            break;
    }
    return out;
}

// ─────────────────────────────────────────────
//  matvec_q4k — y = A × x, A quantizzata Q4_K
//
//  Per ogni riga i di A (= output y[i]):
//    itera sui super-block lungo la riga,
//    decodifica nibbles + scale/min e accumula
//    il prodotto scalare con la finestra di x
//    corrispondente — senza materializzare float.
//
//  Dispatcher SIMD: se la CPU supporta AVX2+FMA,
//  delega a matvec_q4k_avx2 in ops_avx2.cpp.
//  Il rilevamento avviene una sola volta via
//  CPUFeatures cached in una static locale.
// ─────────────────────────────────────────────
void matvec_q4k(const uint8_t* A, const float* x, float* y, int out_dim, int in_dim) {
    static CPUFeatures f = cpu_features();
    if (f.avx2 && f.fma && avx2_enabled()) {
        matvec_q4k_avx2(A, x, y, out_dim, in_dim);
        return;
    }
    static constexpr int SUPER_BLOCK = 256;
    static constexpr int SUPER_BYTES = 144;
    static constexpr int N_SUB       = 8;
    static constexpr int BLOCK_SIZE  = 32;

    int n_super_per_row = in_dim / SUPER_BLOCK;
    int row_bytes       = n_super_per_row * SUPER_BYTES;

    #pragma omp parallel for if(out_dim > 512)
    for (int i = 0; i < out_dim; i++) {
        const uint8_t* row = A + i * row_bytes;
        float sum = 0.0f;

        for (int s = 0; s < n_super_per_row; s++) {
            const uint8_t* sb  = row + s * SUPER_BYTES;
            const float*   xb  = x   + s * SUPER_BLOCK;

            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits,    sb,     2);
            memcpy(&dmin_bits, sb + 2, 2);
            float d    = fp16_to_fp32(d_bits);
            float dmin = fp16_to_fp32(dmin_bits);

            const uint8_t* sc_bytes = sb + 4;

            auto get_scale_min = [&](int j) -> std::pair<float, float> {
                uint8_t sv, mv;
                if (j < 4) {
                    sv = sc_bytes[j]   & 0x3F;
                    mv = sc_bytes[j+4] & 0x3F;
                } else {
                    sv = (sc_bytes[j+4] & 0x0F) | ((sc_bytes[j-4] >> 6) << 4);
                    mv = (sc_bytes[j+4] >>   4) | ((sc_bytes[j  ] >> 6) << 4);
                }
                return {sv * d, mv * dmin};
            };

            const uint8_t* nibbles = sb + 16;

            for (int b = 0; b < N_SUB / 2; b++) {
                auto [sc0, min0] = get_scale_min(b * 2);
                auto [sc1, min1] = get_scale_min(b * 2 + 1);

                const uint8_t* q  = nibbles + b * BLOCK_SIZE;
                const float*   x0 = xb + b * 2 * BLOCK_SIZE;
                const float*   x1 = x0 + BLOCK_SIZE;

                for (int l = 0; l < BLOCK_SIZE; l++) {
                    sum += ((q[l] & 0xF) * sc0 - min0) * x0[l];
                    sum += ((q[l] >>  4) * sc1 - min1) * x1[l];
                }
            }
        }
        y[i] = sum;
    }
}

// ─────────────────────────────────────────────
//  matvec_q6k — y = A × x, A quantizzata Q6_K
//
//  Dispatcher SIMD: se AVX2+FMA sono disponibili,
//  delega al kernel AVX2 in ops_avx2.cpp.
// ─────────────────────────────────────────────
void matvec_q6k(const uint8_t* A, const float* x, float* y, int out_dim, int in_dim) {
    static CPUFeatures f = cpu_features();
    if (f.avx2 && f.fma && avx2_enabled()) {
        matvec_q6k_avx2(A, x, y, out_dim, in_dim);
        return;
    }
    static constexpr int SUPER_BLOCK = 256;
    static constexpr int SUPER_BYTES = 210;

    int n_super_per_row = in_dim / SUPER_BLOCK;
    int row_bytes       = n_super_per_row * SUPER_BYTES;

    #pragma omp parallel for if(out_dim > 512)
    for (int i = 0; i < out_dim; i++) {
        const uint8_t* row = A + i * row_bytes;
        float sum = 0.0f;

        for (int s = 0; s < n_super_per_row; s++) {
            const uint8_t* sb  = row + s * SUPER_BYTES;
            const float*   xb  = x   + s * SUPER_BLOCK;

            const uint8_t* ql = sb;
            const uint8_t* qh = sb + 128;
            const int8_t*  sc = reinterpret_cast<const int8_t*>(sb + 192);

            uint16_t d_bits;
            memcpy(&d_bits, sb + 208, 2);
            float d = fp16_to_fp32(d_bits);

            for (int n = 0; n < 2; n++) {
                const uint8_t* ql_n = ql + n * 64;
                const uint8_t* qh_n = qh + n * 32;
                const int8_t*  sc_n = sc  + n * 8;
                const float*   xn   = xb  + n * 128;

                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int8_t q1 = (int8_t)((ql_n[l]    & 0xF) | (((qh_n[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql_n[l+32] & 0xF) | (((qh_n[l] >> 2) & 3) << 4)) - 32;
                    int8_t q3 = (int8_t)((ql_n[l]    >>  4) | (((qh_n[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql_n[l+32] >>  4) | (((qh_n[l] >> 6) & 3) << 4)) - 32;

                    sum += (d * sc_n[is + 0] * q1) * xn[l +  0];
                    sum += (d * sc_n[is + 2] * q2) * xn[l + 32];
                    sum += (d * sc_n[is + 4] * q3) * xn[l + 64];
                    sum += (d * sc_n[is + 6] * q4) * xn[l + 96];
                }
            }
        }
        y[i] = sum;
    }
}

// ─────────────────────────────────────────────
//  matvec_q8_0 — y = A × x, A quantizzata Q8_0
//
//  Dispatcher SIMD: se AVX2+FMA sono disponibili,
//  delega al kernel AVX2 in ops_avx2.cpp.
// ─────────────────────────────────────────────
void matvec_q8_0(const uint8_t* A, const float* x, float* y, int out_dim, int in_dim) {
    static CPUFeatures f = cpu_features();
    if (f.avx2 && f.fma && avx2_enabled()) {
        matvec_q8_0_avx2(A, x, y, out_dim, in_dim);
        return;
    }
    static constexpr int BLOCK_SIZE  = 32;
    static constexpr int BLOCK_BYTES = 34;

    int n_blocks_per_row = in_dim / BLOCK_SIZE;
    int row_bytes        = n_blocks_per_row * BLOCK_BYTES;

    #pragma omp parallel for if(out_dim > 512)
    for (int i = 0; i < out_dim; i++) {
        const uint8_t* row = A + i * row_bytes;
        float sum = 0.0f;

        for (int b = 0; b < n_blocks_per_row; b++) {
            const uint8_t* block = row + b * BLOCK_BYTES;
            uint16_t scale_bits;
            memcpy(&scale_bits, block, sizeof(scale_bits));
            float scale = fp16_to_fp32(scale_bits);

            const int8_t* quants = reinterpret_cast<const int8_t*>(block + 2);
            const float*  xb     = x + b * BLOCK_SIZE;

            for (int j = 0; j < BLOCK_SIZE; j++)
                sum += quants[j] * scale * xb[j];
        }
        y[i] = sum;
    }
}

// ─────────────────────────────────────────────
//  matvec_f16 — y = A × x, A in float16
//  Converte ogni elemento inline durante il dot.
//
//  Dispatcher SIMD: se AVX2+FMA+F16C sono disponibili,
//  delega al kernel AVX2 che usa _mm256_cvtph_ps.
// ─────────────────────────────────────────────
void matvec_f16(const uint16_t* A, const float* x, float* y, int out_dim, int in_dim) {
    static CPUFeatures f = cpu_features();
    if (f.avx2 && f.fma && f.f16c && avx2_enabled()) {
        matvec_f16_avx2(A, x, y, out_dim, in_dim);
        return;
    }
    #pragma omp parallel for if(out_dim > 512)
    for (int i = 0; i < out_dim; i++) {
        const uint16_t* row = A + i * in_dim;
        float sum = 0.0f;
        for (int j = 0; j < in_dim; j++)
            sum += fp16_to_fp32(row[j]) * x[j];
        y[i] = sum;
    }
}

// ─────────────────────────────────────────────
//  matvec_quant — dispatch sul tipo di QuantTensor
// ─────────────────────────────────────────────
void matvec_quant(const QuantTensor& A, const float* x, float* y) {
    int out_dim = static_cast<int>(A.n_rows);
    int in_dim  = static_cast<int>(A.n_cols);

    switch (A.type) {
        case GGMLType::F32:
            matvec(reinterpret_cast<const float*>(A.data.data()), x, y, out_dim, in_dim);
            break;
        case GGMLType::F16:
            matvec_f16(reinterpret_cast<const uint16_t*>(A.data.data()), x, y, out_dim, in_dim);
            break;
        case GGMLType::Q8_0:
            matvec_q8_0(A.data.data(), x, y, out_dim, in_dim);
            break;
        case GGMLType::Q4_K:
            matvec_q4k(A.data.data(), x, y, out_dim, in_dim);
            break;
        case GGMLType::Q6_K:
            matvec_q6k(A.data.data(), x, y, out_dim, in_dim);
            break;
        default:
            std::cerr << "[ERRORE] matvec_quant: tipo non supportato: "
                      << ggml_type_name(A.type) << "\n";
            std::fill(y, y + out_dim, 0.0f);
    }
}

// ─────────────────────────────────────────────
//  matvec_quant_batch — prodotto matrice-vettore BATCH
//
//  Calcola y = A @ x^T dove x è una matrice
//  di N token invece di un singolo vettore.
//
//  Loop order ottimale per la cache:
//    per ogni riga i di A:
//      dequantizza la riga i
//      per ogni colonna j della riga:
//        per ogni token t del batch:
//          y[t][i] += w[i][j] * x[t][j]
//
//  Il peso w[i][j] viene caricato una sola volta
//  e riutilizzato per tutti i N token — questo
//  è il cuore del guadagno rispetto a N matvec
//  sequenziali, dove ogni peso veniva caricato N
//  volte (una per token).
// ─────────────────────────────────────────────
void matvec_quant_batch(const QuantTensor& A, const float* x_batch, float* y_batch, int N) {
    int out_dim = static_cast<int>(A.n_rows);
    int in_dim  = static_cast<int>(A.n_cols);
    std::fill(y_batch, y_batch + N * out_dim, 0.0f);

    switch (A.type) {
        case GGMLType::F32: {
            const float* W = reinterpret_cast<const float*>(A.data.data());
            #pragma omp parallel for if(out_dim > 512)
            for (int i = 0; i < out_dim; i++) {
                for (int j = 0; j < in_dim; j++) {
                    float w = W[i * in_dim + j];
                    for (int t = 0; t < N; t++) {
                        y_batch[t * out_dim + i] += w * x_batch[t * in_dim + j];
                    }
                }
            }
            break;
        }
        case GGMLType::F16:
        case GGMLType::Q8_0:
        case GGMLType::Q4_K:
        case GGMLType::Q6_K: {
            // Dequantizza una riga alla volta, accumula per tutti i token.
            // Il loop order j, t è cache-friendly: w è riutilizzato per tutti i token.
            //
            // BUG FIX: row deve essere PRIVATE per ogni thread OpenMP.
            // Prima era dichiarata fuori dal parallel for → shared → data race
            // e potenziale memory corruption con molti thread.
            #pragma omp parallel if(out_dim > 512)
            {
                std::vector<float> row(in_dim);
                #pragma omp for
                for (int i = 0; i < out_dim; i++) {
                    dequant_row(A, i, row.data());
                    for (int j = 0; j < in_dim; j++) {
                        float w = row[j];
                        for (int t = 0; t < N; t++) {
                            y_batch[t * out_dim + i] += w * x_batch[t * in_dim + j];
                        }
                    }
                }
            }
            break;
        }
        default:
            std::cerr << "[ERRORE] matvec_quant_batch: tipo non supportato: "
                      << ggml_type_name(A.type) << "\n";
            break;
    }
}

// ─────────────────────────────────────────────
//  dequant_row — dequantizza una singola riga
//
//  Usata per l'embedding lookup dove serve
//  solo una riga alla volta (non un matvec).
// ─────────────────────────────────────────────
void dequant_row(const QuantTensor& A, int row_idx, float* out) {
    uint64_t n_cols = A.n_cols;

    switch (A.type) {
        case GGMLType::F32: {
            const float* src = reinterpret_cast<const float*>(A.data.data())
                               + row_idx * n_cols;
            memcpy(out, src, n_cols * sizeof(float));
            break;
        }
        case GGMLType::F16: {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(A.data.data())
                                  + row_idx * n_cols;
            for (uint64_t j = 0; j < n_cols; j++)
                out[j] = fp16_to_fp32(src[j]);
            break;
        }
        case GGMLType::Q4_K: {
            int n_super = static_cast<int>(n_cols / 256);
            const uint8_t* row = A.data.data() + row_idx * n_super * 144;
            dequantize_q4_k(row, out, n_cols);
            break;
        }
        case GGMLType::Q6_K: {
            int n_super = static_cast<int>(n_cols / 256);
            const uint8_t* row = A.data.data() + row_idx * n_super * 210;
            dequantize_q6_k(row, out, n_cols);
            break;
        }
        case GGMLType::Q8_0: {
            int n_blocks = static_cast<int>(n_cols / 32);
            const uint8_t* row = A.data.data() + row_idx * n_blocks * 34;
            dequantize_q8_0(row, out, n_cols);
            break;
        }
        default:
            std::cerr << "[ERRORE] dequant_row: tipo non supportato: "
                      << ggml_type_name(A.type) << "\n";
            std::fill(out, out + n_cols, 0.0f);
    }
}

// ─────────────────────────────────────────────
//  Matrix multiplication C = A × B
//
//  A[m×k], B[k×n], C[m×n]
//  C[i][j] = Σ(k) A[i][k] * B[k][j]
//
//  Accesso row-major:
//    A[i][k] = A[i*k_dim + k]
//    B[k][j] = B[k*n + j]
//    C[i][j] = C[i*n + j]
//
//  Nota: loop order i,k,j è cache-friendly
//  per la lettura di A e B rispetto a i,j,k
// ─────────────────────────────────────────────
void matmul(const float* A, const float* B, float* C,
            int m, int k, int n) {
    // Inizializza output a zero
    memset(C, 0, m * n * sizeof(float));

    // Parallelizza sulle righe di output — ogni riga è indipendente
    #pragma omp parallel for if(m > 64)
    for (int i = 0; i < m; i++) {
        for (int kk = 0; kk < k; kk++) {
            float a = A[i * k + kk];
            for (int j = 0; j < n; j++) {
                // C[i,j] += A[i,kk] * B[kk,j]
                C[i * n + j] += a * B[kk * n + j];
            }
        }
    }
}

// ─────────────────────────────────────────────
//  Matrix-vector: y = A × x
//
//  A[out_dim × in_dim], x[in_dim], y[out_dim]
//  y[i] = Σ(j) A[i][j] * x[j]
//
//  Caso molto frequente nel transformer:
//  proiezioni Q/K/V, FFN, lm_head
//
//  Dispatcher SIMD: se AVX2+FMA sono disponibili,
//  delega a matvec_avx2 in ops_avx2.cpp.
// ─────────────────────────────────────────────
void matvec(const float* A, const float* x, float* y,
            int out_dim, int in_dim) {
    static CPUFeatures f = cpu_features();
    if (f.avx2 && f.fma && avx2_enabled()) {
        matvec_avx2(A, x, y, out_dim, in_dim);
        return;
    }
    // Versione scalare con OpenMP sulle righe (solo per matrici grandi)
    #pragma omp parallel for if(out_dim > 512)
    for (int i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        const float* row = A + i * in_dim;
        for (int j = 0; j < in_dim; j++)
            sum += row[j] * x[j];
        y[i] = sum;
    }
}

// ─────────────────────────────────────────────
//  Addizione elemento per elemento
// ─────────────────────────────────────────────
void vec_add(const float* a, const float* b, float* out, int n) {
    #pragma omp parallel for if(n > 1024)
    for (int i = 0; i < n; i++)
        out[i] = a[i] + b[i];
}

// ─────────────────────────────────────────────
//  Copia vettore
// ─────────────────────────────────────────────
void vec_copy(const float* src, float* dst, int n) {
    memcpy(dst, src, n * sizeof(float));
}

// ─────────────────────────────────────────────
//  Softmax numericamente stabile
//
//  Problema naive: exp(x) può andare in overflow
//  per x grandi. Soluzione: sottrai il massimo.
//  softmax(x) = softmax(x - max) (matematicamente
//  equivalente, numericamente più stabile)
// ─────────────────────────────────────────────
void softmax(float* x, int n) {
    // Trova il massimo per stabilità numerica
    float max_val = *std::max_element(x, x + n);

    // Calcola exp(x - max) e la loro somma
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    // Normalizza
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < n; i++)
        x[i] *= inv_sum;
}

// ─────────────────────────────────────────────
//  GELU activation (approssimazione GPT-2)
//
//  GPT-2 usa questa approssimazione invece
//  della formula esatta con erf() perché
//  è più veloce e sufficientemente precisa.
//
//  Costanti:
//    √(2/π) ≈ 0.7978845608
//    0.044715 = coefficiente cubico
// ─────────────────────────────────────────────
void gelu(float* x, int n) {
    static constexpr float SQRT_2_OVER_PI = 0.7978845608f;
    static constexpr float COEFF          = 0.044715f;

    #pragma omp parallel for if(n > 1024)
    for (int i = 0; i < n; i++) {
        float v = x[i];
        // Argomento della tanh
        float inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
        // GELU(x) = 0.5 * x * (1 + tanh(inner))
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}

// ─────────────────────────────────────────────
//  RMSNorm  (Root Mean Square Layer Normalization)
//
//  Alternativa a LayerNorm usata in LLaMA/GPT-NeoX.
//  Non calcola la media (mean-centering), solo l'RMS:
//
//    RMS(x) = sqrt( (1/n) * Σ xᵢ² + ε )
//    out[i]  = w[i] * x[i] / RMS(x)
//
//  Il parametro eps evita divisione per zero quando x ≈ 0.
//  Il vettore w (weight) è il parametro appreso della norma
//  (chiamato "gamma" in alcune implementazioni).
//
//  Parametri:
//    x   — vettore di input  [n]
//    w   — weight appresi    [n]
//    out — vettore di output [n]  (può coincidere con x)
//    n   — dimensione
//    eps — epsilon per stabilità numerica (tipicamente 1e-5)
// ─────────────────────────────────────────────
void rms_norm(const float* x, const float* w,
              float* out, int n, float eps) {
    // Calcola la mean square
    float ms = 0.0f;
    for (int i = 0; i < n; i++)
        ms += x[i] * x[i];
    ms /= n;

    // Normalizza e scala con il weight appreso
    float inv_rms = 1.0f / sqrtf(ms + eps);
    for (int i = 0; i < n; i++)
        out[i] = w[i] * x[i] * inv_rms;
}

// ─────────────────────────────────────────────
//  RoPE
//
//  Per ogni head h e ogni coppia di dimensioni i:
//    θ = pos / (freq_base ^ (2i / rope_dim))
//    x[h*d_head + 2i]   =  x[...] * cos(θ) - x[...+1] * sin(θ)
//    x[h*d_head + 2i+1] =  x[...] * sin(θ) + x[...+1] * cos(θ)
//
//  Applichiamo RoPE solo alle prime rope_dim
//  dimensioni di ogni head — le restanti
//  rimangono invariate.
// ─────────────────────────────────────────────
void rope(float* x, int pos,
           int n_heads, int d_head,
           int rope_dim, float freq_base) {
    // Numero di coppie su cui applicare RoPE
    int half_dim = rope_dim / 2;

    for (int h = 0; h < n_heads; h++) {
        float* xh = x + h * d_head;

        for (int i = 0; i < half_dim; i++) {
            // Calcola l'angolo per questa coppia
            float theta = (float)pos /
                powf(freq_base, (2.0f * i) / (float)rope_dim);

            float cos_t = cosf(theta);
            float sin_t = sinf(theta);

            // Leggi la coppia
            float x0 = xh[2 * i];
            float x1 = xh[2 * i + 1];

            // Applica la rotazione
            xh[2 * i]     = x0 * cos_t - x1 * sin_t;
            xh[2 * i + 1] = x0 * sin_t + x1 * cos_t;
        }
    }
}

// ─────────────────────────────────────────────
//  RoPE tipo NEOX (usato da Qwen2, Phi, Gemma)
//
//  Differenza rispetto a RoPE NORM (LLaMA):
//  Invece di ruotare coppie consecutive (0,1), (2,3)...
//  ruota coppie scambiate: (0, half_dim), (1, half_dim+1)...
//
//  Per ogni head h e ogni dimensione i in [0, half_dim):
//    θ = pos / (freq_base ^ (2i / rope_dim))
//    x[h*d_head + i]              = x[i] * cos(θ) - x[i + half_dim] * sin(θ)
//    x[h*d_head + i + half_dim]   = x[i] * sin(θ) + x[i + half_dim] * cos(θ)
// ─────────────────────────────────────────────
void rope_neox(float* x, int pos,
               int n_heads, int d_head,
               int rope_dim, float freq_base) {
    int half_dim = rope_dim / 2;

    for (int h = 0; h < n_heads; h++) {
        float* xh = x + h * d_head;

        for (int i = 0; i < half_dim; i++) {
            float theta = (float)pos /
                powf(freq_base, (2.0f * i) / (float)rope_dim);

            float cos_t = cosf(theta);
            float sin_t = sinf(theta);

            float x0 = xh[i];
            float x1 = xh[i + half_dim];

            xh[i]           = x0 * cos_t - x1 * sin_t;
            xh[i + half_dim] = x0 * sin_t + x1 * cos_t;
        }
    }
}

// ─────────────────────────────────────────────
//  SiLU activation  (Sigmoid Linear Unit)
//
//  Usata nel gate della FFN di LLaMA (SwiGLU):
//    SiLU(x) = x * σ(x) = x / (1 + e^(-x))
//
//  È equivalente a x * sigmoid(x). A differenza di ReLU,
//  SiLU è smooth e non azzera i valori negativi di colpo,
//  ma li attenua in modo continuo.
//
//  In LLaMA la FFN calcola: SiLU(gate) ⊙ up, dove ⊙ è il
//  prodotto elemento per elemento (SwiGLU gate mechanism).
// ─────────────────────────────────────────────
void silu(float* x, int n) {
    #pragma omp parallel for if(n > 1024)
    for (int i = 0; i < n; i++)
        x[i] = x[i] / (1.0f + expf(-x[i]));
}