#include "ops_avx2.hpp"
#include "ops.hpp"
#include <immintrin.h>
#include <cstring>

// ─────────────────────────────────────────────
//  Kernel matvec ottimizzati con AVX2 + FMA
//
//  Questo file contiene le implementazioni SIMD dei
//  kernel matvec per tutti i formati quantizzati.
//  Ogni funzione è marcata con:
//    __attribute__((target("avx2,fma")))
//  in modo che il compilatore generi codice VEX
//  (con i nuovi prefissi per le istruzioni a 256 bit)
//  solo all'interno di queste funzioni, senza
//  contaminare il resto del binario.
//
//  Strategia generale:
//  1. Loop esterno sulle righe (out_dim) — non
//     parallelizzato qui, lasciamo che sia ops.cpp
//     a decidere se usare OpenMP o meno.
//  2. Loop internore sulle colonne in blocchi di
//     8 float (256 bit / 32 bit = 8 elementi).
//  3. Accumulo in un registro __m256 con FMADD.
//  4. Riduzione orizzontale del registro in un
//     float scalare alla fine.
//  5. Prologo/epilogo scalare per gestire le
//     dimensioni non multipli di 8.
//
//  Per i formati quantizzati (Q4_K, Q6_K, Q8_0),
//  il flusso è leggermente diverso:
//  - Q8_0: scompattiamo 32 int8 in 4× __m256 di
//    float, moltiplicando per lo scale del blocco.
//  - Q4_K / Q6_K: dequantizziamo ogni sub-block
//    in un buffer float temporaneo allineato a 32
//    byte sullo stack, poi usiamo gli stessi FMADD
//    del caso F32. Questo è un compromesso tra
//    complessità e velocità: non è puro-SIMD, ma
//    elimina il loop scalare interno di 32 iter.
// ─────────────────────────────────────────────

// ─────────────────────────────────────────────
//  Riduzione orizzontale di un __m256 → float
//
//  Un registro AVX contiene 8 float. Per ottenere
//  la loro somma dobbiamo "piegare" il vettore
//  su se stesso fino a un singolo valore.
//
//  Passaggi:
//  1. Split del registro 256-bit in due 128-bit
//     (vlow e vhigh) e somma → 4 float.
//  2. _mm_movehdup_ps duplica gli elementi
//     dispari → somma parallela → 2 float.
//  3. _mm_movehl_ps sposta la metà alta sulla
//     bassa → somma finale → 1 float.
// ─────────────────────────────────────────────
__attribute__((target("avx2,fma")))
static inline float hsum256_ps_avx(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow  = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// ─────────────────────────────────────────────
//  matvec_f32 — prodotto matrice-vettore float32
//
//  A[out_dim × in_dim] · x[in_dim] = y[out_dim]
//
//  Per ogni riga i, accumuliamo il dot product
//  con x usando 8 float alla volta. I pesi GGUF
//  non sono garantiti allineati a 32 byte, quindi
//  usiamo _mm256_loadu_ps (load unaligned).
// ─────────────────────────────────────────────
__attribute__((target("avx2,fma")))
void matvec_avx2(const float* A, const float* x, float* y, int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; i++) {
        const float* row = A + i * in_dim;
        __m256 sum_vec = _mm256_setzero_ps();
        int j = 0;
        // Corpo SIMD: processa 8 elementi per iterazione
        for (; j + 7 < in_dim; j += 8) {
            __m256 a_vec = _mm256_loadu_ps(row + j);
            __m256 x_vec = _mm256_loadu_ps(x + j);
            sum_vec = _mm256_fmadd_ps(a_vec, x_vec, sum_vec);
        }
        // Riduci il registro SIMD in scalare
        float sum = hsum256_ps_avx(sum_vec);
        // Epilogo scalare per i rimanenti (0..7 elementi)
        for (; j < in_dim; j++) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

// ─────────────────────────────────────────────
//  matvec_f16 — prodotto matrice-vettore float16
//
//  Richiede F16C (bit 29 di ECX, CPUID leaf 1).
//  Carichiamo 8 half (128 bit) in un __m128i,
//  li convertiamo in 8 float con _mm256_cvtph_ps,
//  poi procediamo esattamente come il kernel F32.
// ─────────────────────────────────────────────
__attribute__((target("avx2,fma,f16c")))
void matvec_f16_avx2(const uint16_t* A, const float* x, float* y, int out_dim, int in_dim) {
    for (int i = 0; i < out_dim; i++) {
        const uint16_t* row = A + i * in_dim;
        __m256 sum_vec = _mm256_setzero_ps();
        int j = 0;
        for (; j + 7 < in_dim; j += 8) {
            __m128i a_half = _mm_loadu_si128((const __m128i*)(row + j));
            __m256 a_vec = _mm256_cvtph_ps(a_half);
            __m256 x_vec = _mm256_loadu_ps(x + j);
            sum_vec = _mm256_fmadd_ps(a_vec, x_vec, sum_vec);
        }
        float sum = hsum256_ps_avx(sum_vec);
        for (; j < in_dim; j++) {
            sum += fp16_to_fp32(row[j]) * x[j];
        }
        y[i] = sum;
    }
}

// ─────────────────────────────────────────────
//  matvec_q8_0 — prodotto con pesi Q8_0
//
//  Layout di ogni blocco Q8_0 (34 byte):
//    [2 byte: scale in float16]
//    [32 byte: 32 valori int8]
//
//  Strategia AVX2 per blocco:
//  1. Leggi lo scale fp16 e convertilo in float.
//  2. Broadcast dello scale in un registro __m256.
//  3. Carica i 32 int8 in due registri __m128i
//     (16+16 elementi).
//  4. Scompatta (sign-extend) ogni gruppo di 8 int8
//     in un __m256i di int32, usando _mm256_cvtepi8_epi32.
//     Questo richiede 4 operazioni (32 elem / 8 lane).
//  5. Converte ogni __m256i in __m256 float e moltiplicalo
//     per lo scale broadcastato.
//  6. Ora hai 4 registri __m256 con i 32 pesi dequantizzati;
//     accumula il dot product con i 4 chunk di x tramite
//     _mm256_fmadd_ps.
// ─────────────────────────────────────────────
__attribute__((target("avx2,fma")))
void matvec_q8_0_avx2(const uint8_t* A, const float* x, float* y, int out_dim, int in_dim) {
    static constexpr int BLOCK_SIZE  = 32;
    static constexpr int BLOCK_BYTES = 34;

    int n_blocks_per_row = in_dim / BLOCK_SIZE;
    int row_bytes        = n_blocks_per_row * BLOCK_BYTES;

    for (int i = 0; i < out_dim; i++) {
        const uint8_t* row = A + i * row_bytes;
        __m256 sum_vec = _mm256_setzero_ps();

        for (int b = 0; b < n_blocks_per_row; b++) {
            const uint8_t* block = row + b * BLOCK_BYTES;
            uint16_t scale_bits;
            memcpy(&scale_bits, block, sizeof(scale_bits));
            float scale = fp16_to_fp32(scale_bits);
            __m256 scale_vec = _mm256_set1_ps(scale);

            const int8_t* quants = reinterpret_cast<const int8_t*>(block + 2);
            const float*  xb     = x + b * BLOCK_SIZE;

            // Carica 32 int8 in due registri 128-bit, poi scompandi in 4× __m256i di int32
            __m128i q_lo  = _mm_loadu_si128((const __m128i*)quants);        // elementi 0..15
            __m128i q_hi  = _mm_loadu_si128((const __m128i*)(quants + 16)); // elementi 16..31

            // Ogni _mm256_cvtepi8_epi32 prende i primi 8 byte di un __m128i
            // e li espande a 8 int32. Dobbiamo shiftare di 8 byte per prendere
            // la seconda metà del __m128i.
            __m256i q0 = _mm256_cvtepi8_epi32(q_lo);
            __m256i q1 = _mm256_cvtepi8_epi32(_mm_srli_si128(q_lo, 8));
            __m256i q2 = _mm256_cvtepi8_epi32(q_hi);
            __m256i q3 = _mm256_cvtepi8_epi32(_mm_srli_si128(q_hi, 8));

            // Converte int32 → float32 e moltiplica per lo scale del blocco
            __m256 f0 = _mm256_mul_ps(_mm256_cvtepi32_ps(q0), scale_vec);
            __m256 f1 = _mm256_mul_ps(_mm256_cvtepi32_ps(q1), scale_vec);
            __m256 f2 = _mm256_mul_ps(_mm256_cvtepi32_ps(q2), scale_vec);
            __m256 f3 = _mm256_mul_ps(_mm256_cvtepi32_ps(q3), scale_vec);

            // Accumula il dot product con i 4 chunk di x
            sum_vec = _mm256_fmadd_ps(f0, _mm256_loadu_ps(xb),      sum_vec);
            sum_vec = _mm256_fmadd_ps(f1, _mm256_loadu_ps(xb + 8),  sum_vec);
            sum_vec = _mm256_fmadd_ps(f2, _mm256_loadu_ps(xb + 16), sum_vec);
            sum_vec = _mm256_fmadd_ps(f3, _mm256_loadu_ps(xb + 24), sum_vec);
        }
        y[i] = hsum256_ps_avx(sum_vec);
    }
}

// ─────────────────────────────────────────────
//  matvec_q4k — prodotto con pesi Q4_K
//
//  Il formato Q4_K usa super-block da 256 elementi
//  divisi in 8 sub-block da 32. Ogni sub-block ha
//  una propria scala e un minimo (6 bit ciascuno),
//  e i pesi sono memorizzati come nibble (4 bit).
//
//  Strategia "dequantizza-then-dot" (ibrida):
//  Invece di fare masking e shift in-SIMD (complesso
//  e lento per i nibble), dequantizziamo ogni
//  sub-block in un buffer float[32] allineato sullo
//  stack, poi usiamo il kernel F32 AVX2 su quel
//  buffer. Questo elimina il loop scalare interno
//  di 32 iterazioni, che è il collo di bottiglia.
// ─────────────────────────────────────────────
__attribute__((target("avx2,fma")))
void matvec_q4k_avx2(const uint8_t* A, const float* x, float* y, int out_dim, int in_dim) {
    static constexpr int SUPER_BLOCK = 256;
    static constexpr int SUPER_BYTES = 144;
    static constexpr int N_SUB       = 8;
    static constexpr int BLOCK_SIZE  = 32;

    int n_super_per_row = in_dim / SUPER_BLOCK;
    int row_bytes       = n_super_per_row * SUPER_BYTES;

    for (int i = 0; i < out_dim; i++) {
        const uint8_t* row = A + i * row_bytes;
        __m256 sum_vec = _mm256_setzero_ps();

        for (int s = 0; s < n_super_per_row; s++) {
            const uint8_t* sb  = row + s * SUPER_BYTES;
            const float*   xb  = x   + s * SUPER_BLOCK;

            // Legge scale globale (d) e scale dei minimi (dmin)
            uint16_t d_bits, dmin_bits;
            memcpy(&d_bits,    sb,     2);
            memcpy(&dmin_bits, sb + 2, 2);
            float d    = fp16_to_fp32(d_bits);
            float dmin = fp16_to_fp32(dmin_bits);

            const uint8_t* sc_bytes = sb + 4;
            const uint8_t* nibbles  = sb + 16;

            // Buffer temporaneo allineato per i 32 float dequantizzati
            alignas(32) float dequant[BLOCK_SIZE];

            for (int b = 0; b < N_SUB / 2; b++) {
                // Lambda per estrarre scale e minimo del sub-block j
                // dalla struttura compressa a 6 bit (vedi ops.cpp)
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

                auto [sc0, min0] = get_scale_min(b * 2);
                auto [sc1, min1] = get_scale_min(b * 2 + 1);

                const uint8_t* q = nibbles + b * BLOCK_SIZE;
                const float*   x0 = xb + b * 2 * BLOCK_SIZE;
                const float*   x1 = x0 + BLOCK_SIZE;

                // Dequantizza il primo sub-block (low nibble)
                for (int l = 0; l < BLOCK_SIZE; l++) {
                    dequant[l] = (q[l] & 0xF) * sc0 - min0;
                }
                // Dot product AVX2 sui 32 elementi
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant), _mm256_loadu_ps(x0), sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 8),  _mm256_loadu_ps(x0 + 8),  sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 16), _mm256_loadu_ps(x0 + 16), sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 24), _mm256_loadu_ps(x0 + 24), sum_vec);

                // Dequantizza il secondo sub-block (high nibble)
                for (int l = 0; l < BLOCK_SIZE; l++) {
                    dequant[l] = (q[l] >> 4) * sc1 - min1;
                }
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant), _mm256_loadu_ps(x1), sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 8),  _mm256_loadu_ps(x1 + 8),  sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 16), _mm256_loadu_ps(x1 + 16), sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 24), _mm256_loadu_ps(x1 + 24), sum_vec);
            }
        }
        y[i] = hsum256_ps_avx(sum_vec);
    }
}

// ─────────────────────────────────────────────
//  matvec_q6k — prodotto con pesi Q6_K
//
//  Il formato Q6_K usa 6 bit per elemento, con
//  i bit bassi (4 bit) in ql, i bit alti (2 bit)
//  in qh, e 16 scale int8 per super-block.
//
//  La strategia è identica a Q4_K: dequantizza
//  ogni gruppo di 32 elementi in un buffer float
//  temporaneo allineato, poi accumula con AVX2.
//  Per ogni metà da 128 elementi abbiamo 4 gruppi
//  di 32, quindi usiamo un buffer di 64 float per
//  ospitare due gruppi alla volta.
// ─────────────────────────────────────────────
__attribute__((target("avx2,fma")))
void matvec_q6k_avx2(const uint8_t* A, const float* x, float* y, int out_dim, int in_dim) {
    static constexpr int SUPER_BLOCK = 256;
    static constexpr int SUPER_BYTES = 210;

    int n_super_per_row = in_dim / SUPER_BLOCK;
    int row_bytes       = n_super_per_row * SUPER_BYTES;

    for (int i = 0; i < out_dim; i++) {
        const uint8_t* row = A + i * row_bytes;
        __m256 sum_vec = _mm256_setzero_ps();

        for (int s = 0; s < n_super_per_row; s++) {
            const uint8_t* sb  = row + s * SUPER_BYTES;
            const float*   xb  = x   + s * SUPER_BLOCK;

            const uint8_t* ql = sb;           // 4 bit bassi
            const uint8_t* qh = sb + 128;     // 2 bit alti
            const int8_t*  sc = reinterpret_cast<const int8_t*>(sb + 192); // 16 scale

            uint16_t d_bits;
            memcpy(&d_bits, sb + 208, 2);
            float d = fp16_to_fp32(d_bits);   // super-scale globale

            // Buffer da 64 float per ospitare due gruppi di 32
            alignas(32) float dequant[64];

            for (int n = 0; n < 2; n++) {
                const uint8_t* ql_n = ql + n * 64;
                const uint8_t* qh_n = qh + n * 32;
                const int8_t*  sc_n = sc  + n * 8;
                const float*   xn   = xb  + n * 128;

                // --- Primi due gruppi di 32 elementi (q1 e q2) ---
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;  // seleziona la coppia di scale (0 o 1)

                    // Ricostruisce il valore quantizzato a 6 bit dai nibbles
                    int8_t q1 = (int8_t)((ql_n[l]    & 0xF) | (((qh_n[l] >> 0) & 3) << 4)) - 32;
                    int8_t q2 = (int8_t)((ql_n[l+32] & 0xF) | (((qh_n[l] >> 2) & 3) << 4)) - 32;

                    // Dequantizza con le scale corrette (offset +0 e +2)
                    dequant[l]      = d * sc_n[is + 0] * q1;
                    dequant[l + 32] = d * sc_n[is + 2] * q2;
                }

                // Dot product AVX2 sui primi 32 elementi (0..31)  — q1
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant),     _mm256_loadu_ps(xn),      sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 8),  _mm256_loadu_ps(xn + 8),  sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 16), _mm256_loadu_ps(xn + 16), sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 24), _mm256_loadu_ps(xn + 24), sum_vec);

                // Dot product AVX2 sugli elementi 32..63 — q2
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 32), _mm256_loadu_ps(xn + 32), sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 40), _mm256_loadu_ps(xn + 40), sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 48), _mm256_loadu_ps(xn + 48), sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 56), _mm256_loadu_ps(xn + 56), sum_vec);

                // --- Secondi due gruppi di 32 elementi (q3 e q4) ---
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    int8_t q3 = (int8_t)((ql_n[l]    >>  4) | (((qh_n[l] >> 4) & 3) << 4)) - 32;
                    int8_t q4 = (int8_t)((ql_n[l+32] >>  4) | (((qh_n[l] >> 6) & 3) << 4)) - 32;
                    dequant[l]      = d * sc_n[is + 4] * q3;
                    dequant[l + 32] = d * sc_n[is + 6] * q4;
                }

                // Dot product AVX2 sugli elementi 32..63
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant),     _mm256_loadu_ps(xn + 64),  sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 8),  _mm256_loadu_ps(xn + 72),  sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 16), _mm256_loadu_ps(xn + 80),  sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 24), _mm256_loadu_ps(xn + 88),  sum_vec);

                // Dot product AVX2 sugli elementi 64..127
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 32), _mm256_loadu_ps(xn + 96),  sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 40), _mm256_loadu_ps(xn + 104), sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 48), _mm256_loadu_ps(xn + 112), sum_vec);
                sum_vec = _mm256_fmadd_ps(_mm256_loadu_ps(dequant + 56), _mm256_loadu_ps(xn + 120), sum_vec);
            }
        }
        y[i] = hsum256_ps_avx(sum_vec);
    }
}
