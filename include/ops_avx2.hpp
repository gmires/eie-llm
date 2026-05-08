#pragma once
#include <cstdint>

// ─────────────────────────────────────────────
//  Kernel AVX2/FMA per matvec quantizzati
//
//  Questo header dichiara le versioni ottimizzate
//  dei kernel matvec che sfruttano le istruzioni
//  SIMD a 256 bit (AVX2) e le operazioni fused
//  multiply-add (FMA).
//
//  Ogni funzione è implementata in ops_avx2.cpp e
//  marcata con l'attributo GCC/Clang:
//    __attribute__((target("avx2,fma")))
//  Questo permette di compilare solo quel file con
//  -mavx2, mantenendo il resto del binario compatibile
//  con CPU più vecchie. Il dispatcher in ops.cpp
//  decide a runtime se chiamare questi kernel o
//  la versione scalare di fallback.
//
//  Formati supportati:
//    - F32    : float32 nativi
//    - F16    : float16 con conversione F16C
//    - Q8_0   : 8 bit per peso, 1 scale fp16 ogni 32 pesi
//    - Q4_K   : 4 bit per peso, super-block 256 elementi
//    - Q6_K   : 6 bit per peso, super-block 256 elementi
// ─────────────────────────────────────────────

void matvec_avx2   (const float*    A, const float* x, float* y, int out_dim, int in_dim);
void matvec_f16_avx2(const uint16_t* A, const float* x, float* y, int out_dim, int in_dim);
void matvec_q8_0_avx2(const uint8_t* A, const float* x, float* y, int out_dim, int in_dim);
void matvec_q4k_avx2(const uint8_t*  A, const float* x, float* y, int out_dim, int in_dim);
void matvec_q6k_avx2(const uint8_t*  A, const float* x, float* y, int out_dim, int in_dim);
