#include "cpuinfo.hpp"

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

// ─────────────────────────────────────────────
//  Wrapper cross-platform per l'istruzione CPUID
//
//  CPUID è l'istruzione x86 che permette di interrogare
//  il processore per scoprire le sue caratteristiche.
//  Accetta un "function_id" (numero della richiesta)
//  e restituisce 4 registri (EAX, EBX, ECX, EDX) in
//  un array di 4 interi.
//
//  Su MSVC usiamo l'intrinsic __cpuid; su GCC/Clang
//  usiamo la builtin __cpuid (leaf semplice) o
//  __cpuid_count (leaf con sub-leaf).
// ─────────────────────────────────────────────

static void cpuid(int info[4], int function_id) {
#ifdef _MSC_VER
    __cpuid(info, function_id);
#else
    __cpuid(function_id, info[0], info[1], info[2], info[3]);
#endif
}

static void cpuidex(int info[4], int function_id, int subfunction_id) {
#ifdef _MSC_VER
    __cpuidex(info, function_id, subfunction_id);
#else
    __cpuid_count(function_id, subfunction_id, info[0], info[1], info[2], info[3]);
#endif
}

// ─────────────────────────────────────────────
//  Rilevamento runtime delle feature SIMD
//
//  Passaggi:
//  1. CPUID leaf 1 → controlliamo ECX:
//     - bit 27 (OSXSAVE): l'OS supporta XGETBV?
//     - bit 28 (AVX):     la CPU ha AVX?
//     - bit 12 (FMA):     la CPU ha FMA?
//     - bit 29 (F16C):    la CPU ha F16C?
//
//  2. CPUID leaf 7, sub-leaf 0 → controlliamo EBX:
//     - bit 5 (AVX2):     la CPU ha AVX2?
//
//  3. Se OSXSAVE E AVX sono presenti, usiamo XGETBV
//     per leggere il registro XCR0. I bit 1 (XMM)
//     e 2 (YMM) devono essere entrambi settati:
//     se l'OS non ha abilitato lo stato YMM, usare
//     istruzioni AVX causerebbe un crash (SIGILL).
//
//  Il risultato è memorizzato in una variabile static
//  locale: la prima chiamata fa il rilevamento, le
//  successive restituiscono il valore cached.
// ─────────────────────────────────────────────
CPUFeatures cpu_features() {
    static CPUFeatures features;
    static bool initialized = false;
    if (initialized) return features;

    int info[4];
    cpuid(info, 1);
    bool osxsave = (info[2] >> 27) & 1;
    bool avx     = (info[2] >> 28) & 1;
    features.fma  = (info[2] >> 12) & 1;
    features.f16c = (info[2] >> 29) & 1;

    cpuidex(info, 7, 0);
    features.avx2 = (info[1] >> 5) & 1;

    // Se l'OS non ha abilitato lo stato YMM (XGETBV), disabilita AVX/AVX2
    if (osxsave && avx) {
        unsigned long long xcr0;
#ifdef _MSC_VER
        xcr0 = _xgetbv(0);
#else
        __asm__ __volatile__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
#endif
        bool ymm_ok = (xcr0 & 0x6) == 0x6; // bit 1 (XMM) e bit 2 (YMM)
        if (!ymm_ok) {
            features.avx2 = false;
            features.fma  = false;
            features.f16c = false;
        }
    } else {
        features.avx2 = false;
        features.fma  = false;
        features.f16c = false;
    }

    initialized = true;
    return features;
}
