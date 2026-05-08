#pragma once
#include <cstdint>

// ─────────────────────────────────────────────
//  Rilevamento delle capability SIMD della CPU
//
//  Questo modulo fornisce una funzione per rilevare
//  a runtime quali estensioni SIMD sono disponibili
//  sulla CPU in esecuzione: AVX2, FMA, F16C.
//
//  Il rilevamento avviene tramite l'istruzione CPUID,
//  che è lo standard x86 per interrogare le feature
//  del processore. Per sicurezza, verifichiamo anche
//  che il sistema operativo abbia abilitato lo stato
//  dei registri YMM (via XGETBV), altrimenti anche se
//  la CPU supporta AVX2, il suo uso causerebbe un
//  SIGILL (illegal instruction).
//
//  Il risultato è cached in una static locale per
//  evitare di rifare il rilevamento ad ogni chiamata.
// ─────────────────────────────────────────────

struct CPUFeatures {
    bool avx2 = false;  // Advanced Vector Extensions 2 (256-bit integer e float)
    bool fma  = false;  // Fused Multiply-Add (a*b+c in un'unica istruzione)
    bool f16c = false;  // Half-precision Float Conversion (16↔32 bit)
};

// Rileva le capability della CPU (AVX2, FMA, F16C).
// Thread-safe dopo la prima chiamata; il risultato è cached.
CPUFeatures cpu_features();

// Controlla se l'utente ha disabilitato AVX2 via variabile d'ambiente
// EIE_NO_AVX2=1. Questo permette di forzare il fallback scalare
// per debug o compatibilità.
bool avx2_enabled();
