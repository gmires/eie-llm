#include "ops.hpp"
#include <cstring>
#include <algorithm>
#include <cassert>
#include <iostream>

// ─────────────────────────────────────────────
//  Conversione float16 → float32
//
//  IEEE 754 half precision (16 bit):
//    [s][eeeee][mmmmmmmmmm]
//     1    5        10
//
//  Casi speciali:
//    esponente = 0,    mantissa = 0 → zero
//    esponente = 0,    mantissa ≠ 0 → subnormale
//    esponente = 31,   mantissa = 0 → infinito
//    esponente = 31,   mantissa ≠ 0 → NaN
// ─────────────────────────────────────────────
float fp16_to_fp32(uint16_t h) {
    // Estrai i 3 campi dal float16
    uint32_t sign     = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;

    uint32_t result;

    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero (positivo o negativo)
            result = sign << 31;
        } else {
            // Numero subnormale → normalizza per float32
            // Trova la posizione del bit più significativo
            exponent = 1;
            while (!(mantissa & 0x400)) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            // Converti al bias float32 (127) da bias float16 (15)
            exponent = exponent + 127 - 15;
            result = (sign << 31) | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Infinito o NaN — propaga a float32
        result = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        // Numero normale: aggiusta il bias dell'esponente
        // float16: bias = 15,  float32: bias = 127
        exponent = exponent + 127 - 15;
        result = (sign << 31) | (exponent << 23) | (mantissa << 13);
    }

    // Reinterpreta i bit come float
    float f;
    memcpy(&f, &result, sizeof(f));
    return f;
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
//  Ottieni tensore come vettore float32
//
//  Gestisce i diversi tipi di quantizzazione.
//  F32: copia diretta (reinterpret cast)
//  Q8_0: dequantizza blocco per blocco
// ─────────────────────────────────────────────
std::vector<float> tensor_to_float(const GGUFTensor& t) {
    uint64_t n_elem = gguf_tensor_n_elements(t.info);
    std::vector<float> out(n_elem);

    switch (t.type()) {
        case GGMLType::F32:
            memcpy(out.data(), t.data.data(), n_elem * sizeof(float));
            break;

        case GGMLType::F16: {
            // Ogni elemento è un uint16 — convertiamo uno per uno
            // usando fp16_to_fp32 che abbiamo già implementato
            const uint16_t* src = reinterpret_cast<const uint16_t*>
                                  (t.data.data());
            for (uint64_t i = 0; i < n_elem; i++)
                out[i] = fp16_to_fp32(src[i]);
            break;
        }

        case GGMLType::Q8_0:
            dequantize_q8_0(t.data.data(), out.data(), n_elem);
            break;

        default:
            std::cerr << "[ERRORE] Tipo tensore non supportato: "
                      << ggml_type_name(t.type()) << "\n";
            break;
    }
    return out;
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
// ─────────────────────────────────────────────
void matvec(const float* A, const float* x, float* y,
            int out_dim, int in_dim) {
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

    for (int i = 0; i < n; i++) {
        float v = x[i];
        // Argomento della tanh
        float inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
        // GELU(x) = 0.5 * x * (1 + tanh(inner))
        x[i] = 0.5f * v * (1.0f + tanhf(inner));
    }
}