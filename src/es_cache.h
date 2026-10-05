/*
 * Hot-expert RAM cache: fixed number of equal-size slots, one expert each.
 *
 * Eviction score = last_use + freq_weight * min(uses, 16)
 *   last_use     : "layer step" counter when the expert was last needed
 *   uses         : how often it has been needed since it was loaded
 * The slot with the lowest score is evicted. Slots that are still loading,
 * or are needed by the current or next layer (protected), are never
 * evicted. freq_weight = 0 gives plain LRU.
 */
#ifndef ES_CACHE_H
#define ES_CACHE_H

#include <stddef.h>
#include <stdint.h>

#include "es_io.h"

enum { ES_SLOT_EMPTY = 0, ES_SLOT_LOADING = 1, ES_SLOT_READY = 2 };

typedef struct {
    int      layer, expert;  /* key, -1 if empty */
    int      state;
    int      prefetched;     /* loaded by prefetch, not yet used */
    uint64_t last_use;
    uint32_t uses;
    uint64_t protect_until;  /* not evictable while step <= this */
    void    *buf;
    es_req   req;
} es_slot;

typedef struct {
    int      nslots;
    size_t   slot_bytes;
    es_slot *slots;
    int     *map;            /* layer * nexperts + expert -> slot, or -1 */
    int      nlayers, nexperts;
    uint64_t step;           /* advanced once per layer processed */
    double   freq_weight;
    /* stats */
    uint64_t hits, misses, prefetch_issued, prefetch_used, prefetch_wasted, evictions;
} es_cache;

int  es_cache_init(es_cache *c, int nslots, size_t slot_bytes, int nlayers, int nexperts,
                   double freq_weight);
void es_cache_free(es_cache *c);

/* Slot holding (layer, expert), or -1. */
int  es_cache_find(const es_cache *c, int layer, int expert);

/* Pick a victim slot, unmap it, and map it to (layer, expert) in state
 * LOADING. Returns the slot, or -1 if every slot is busy/protected.
 * pool is used to reap finished loads (es_done). */
int  es_cache_claim(es_cache *c, es_pool *pool, int layer, int expert);

#endif
