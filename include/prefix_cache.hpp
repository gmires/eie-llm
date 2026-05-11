#pragma once
#include "model.hpp"
#include <mutex>
#include <unordered_map>
#include <chrono>
#include <iostream>

// ─────────────────────────────────────────────
//  Prefix Cache — condivisione della KV cache
//
//  Nei server reali, molte richieste condividono
//  lo stesso prefisso (es. system prompt identico).
//  Ricalcolare la KV cache ogni volta è uno spreco.
//
//  Questa classe memorizza la KV cache di prompt
//  già visti in una mappa hash → KVCache.
//  Quando arriva una richiesta:
//    1. Calcola l'hash del prompt
//    2. Se presente in cache: copia la KV cache nel modello
//       e salta il prefill (risparmia O(n²) operazioni)
//    3. Se assente: fa il prefill normale e salva la KV cache
//
//  La cache usa politica LRU (Least Recently Used)
//  con un limite massimo di entry per controllare
//  l'uso di memoria.
// ─────────────────────────────────────────────

struct PrefixCacheEntry {
    // Copia completa della KV cache (k e v per ogni layer).
    // In produzione si userebbe memoria condivisa o reference
    // counting per evitare copie. Qui, per semplicità didattica,
    // copiamo i vettori float (costoso ma corretto).
    KVCache kv_cache;

    // Quanti token sono memorizzati in questa entry.
    // Deve corrispondere a kv_cache.n_cached.
    int n_tokens = 0;

    // Timestamp dell'ultimo accesso (per LRU eviction).
    std::chrono::steady_clock::time_point last_used;
};

class PrefixCache {
public:
    // Cerca il prompt nella cache.
    //
    // Se trovato:
    //   - Copia la KV cache salvata nel modello
    //   - Aggiorna il timestamp (touch LRU)
    //   - Ritorna true e imposta out_n_tokens
    //
    // Se non trovato:
    //   - Ritorna false
    //
    // Il parametro 'model' viene modificato solo se la lookup
    // ha successo: kv_cache viene sovrascritta con la copia
    // dalla cache e n_cached viene impostato.
    bool lookup(const std::string& prompt, Model& model, int& out_n_tokens);

    // Salva la KV cache corrente del modello nella cache.
    //
    // Chiamata dopo il prefill di un prompt non presente in cache.
    // Copia la KV cache del modello nella mappa globale.
    // Se la cache è piena, rimuove l'entry meno recentemente usata.
    void store(const std::string& prompt, const Model& model, int n_tokens);

    // Svuota tutta la cache.
    void clear();

    // Stampa statistiche (numero entry, hit/miss totali).
    void print_stats() const;

private:
    // Mutex per thread-safety: il server gestisce richieste
    // concorrenti e la cache è condivisa.
    mutable std::mutex mtx;

    // Mappa hash → entry. La chiave è std::hash<std::string>()
    // applicata al prompt completo.
    std::unordered_map<size_t, PrefixCacheEntry> entries;

    // Limite massimo di entry per controllare l'uso di memoria.
    // Ogni entry contiene la KV cache completa (layer × pos × head × d_head
    // float), quindi la memoria cresce rapidamente.
    static constexpr size_t MAX_ENTRIES = 5;

    // Statistiche per debug.
    size_t hits = 0;
    size_t misses = 0;

    // Rimuove l'entry con il timestamp più vecchio (LRU).
    void evict_lru();
};
