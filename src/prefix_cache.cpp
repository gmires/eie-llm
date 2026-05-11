#include "prefix_cache.hpp"
#include <iomanip>

// ─────────────────────────────────────────────
//  Calcola l'hash di una stringa.
//
//  Usiamo std::hash<std::string> della STL.
//  In produzione si userebbe un hash più robusto
//  (es. FNV-1a o xxhash) per evitare collisioni.
// ─────────────────────────────────────────────
static size_t hash_prompt(const std::string& prompt) {
    return std::hash<std::string>{}(prompt);
}

// ─────────────────────────────────────────────
//  Cerca il prompt nella cache.
//
//  Se il prompt è già stato visto, la sua KV cache
//  è stata salvata dopo il prefill. Copiamo quella
//  KV cache nel modello, così il prefill può saltare
//  tutti i token già processati.
//
//  Nota: copiamo i vettori k e v per ogni layer.
//  Questo è costoso in memoria e tempo, ma garantisce
//  che ogni richiesta abbia la propria KV cache isolata
//  (il modello non è thread-safe, ma la cache sì).
// ─────────────────────────────────────────────
bool PrefixCache::lookup(const std::string& prompt, Model& model, int& out_n_tokens) {
    std::lock_guard<std::mutex> lock(mtx);

    size_t h = hash_prompt(prompt);
    auto it = entries.find(h);
    if (it == entries.end()) {
        misses++;
        return false;
    }

    // Trovato! Copia la KV cache salvata nel modello.
    PrefixCacheEntry& entry = it->second;
    model.kv_cache = entry.kv_cache;  // copia profonda dei vector
    out_n_tokens = entry.n_tokens;

    // Aggiorna timestamp (touch per LRU)
    entry.last_used = std::chrono::steady_clock::now();
    hits++;
    return true;
}

// ─────────────────────────────────────────────
//  Salva la KV cache corrente nella cache globale.
//
//  Dopo il prefill di un prompt, la KV cache del modello
//  contiene K e V per tutti i token del prompt.
//  Salviamo una copia nella cache per le prossime richieste
//  con lo stesso prompt.
//
//  Se la cache ha raggiunto il limite massimo, rimuoviamo
//  l'entry meno recentemente usata (LRU eviction).
// ─────────────────────────────────────────────
void PrefixCache::store(const std::string& prompt, const Model& model, int n_tokens) {
    std::lock_guard<std::mutex> lock(mtx);

    size_t h = hash_prompt(prompt);

    // Se esiste già, aggiorna la entry
    auto it = entries.find(h);
    if (it != entries.end()) {
        it->second.kv_cache = model.kv_cache;
        it->second.n_tokens = n_tokens;
        it->second.last_used = std::chrono::steady_clock::now();
        return;
    }

    // Se la cache è piena, fai eviction LRU
    if (entries.size() >= MAX_ENTRIES) {
        evict_lru();
    }

    // Inserisci la nuova entry
    PrefixCacheEntry entry;
    entry.kv_cache = model.kv_cache;  // copia profonda
    entry.n_tokens = n_tokens;
    entry.last_used = std::chrono::steady_clock::now();
    entries[h] = std::move(entry);
}

// ─────────────────────────────────────────────
//  Svuota tutta la cache.
// ─────────────────────────────────────────────
void PrefixCache::clear() {
    std::lock_guard<std::mutex> lock(mtx);
    entries.clear();
    hits = 0;
    misses = 0;
}

// ─────────────────────────────────────────────
//  Stampa statistiche della cache.
// ─────────────────────────────────────────────
void PrefixCache::print_stats() const {
    std::lock_guard<std::mutex> lock(mtx);
    std::cout << "\n  Prefix Cache:\n"
              << "    entry attive: " << entries.size() << "/" << MAX_ENTRIES << "\n"
              << "    hit:  " << hits << "\n"
              << "    miss: " << misses << "\n";
    if (hits + misses > 0) {
        float rate = 100.0f * static_cast<float>(hits) / static_cast<float>(hits + misses);
        std::cout << "    hit rate: " << std::fixed << std::setprecision(1) << rate << "%\n";
    }
    std::cout << "\n";
}

// ─────────────────────────────────────────────
//  Eviction LRU — rimuove l'entry meno recente.
//
//  Scorre tutte le entry e trova quella con il
//  timestamp 'last_used' più vecchio.
//  Questo garantisce che i prompt usati di recente
//  restino in cache, mentre quelli obsoleti vengano
//  rimossi per liberare memoria.
// ─────────────────────────────────────────────
void PrefixCache::evict_lru() {
    if (entries.empty()) return;

    auto oldest = entries.begin();
    auto it = entries.begin();
    ++it;
    for (; it != entries.end(); ++it) {
        if (it->second.last_used < oldest->second.last_used) {
            oldest = it;
        }
    }
    entries.erase(oldest);
}
