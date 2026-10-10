#include "es_cache.h"

#include <stdlib.h>
#include <string.h>

int es_cache_init(es_cache *c, int nslots, size_t slot_bytes, int nlayers, int nexperts,
                  double freq_weight) {
    memset(c, 0, sizeof *c);
    c->nslots = nslots;
    c->slot_bytes = slot_bytes;
    c->nlayers = nlayers;
    c->nexperts = nexperts;
    c->freq_weight = freq_weight;
    c->slots = calloc((size_t)nslots, sizeof(es_slot));
    c->map = malloc(sizeof(int) * (size_t)nlayers * (size_t)nexperts);
    if (!c->slots || !c->map) return -1;
    for (int i = 0; i < nlayers * nexperts; i++) c->map[i] = -1;
    for (int i = 0; i < nslots; i++) {
        c->slots[i].layer = c->slots[i].expert = -1;
        c->slots[i].buf = es_alloc(slot_bytes);
        if (!c->slots[i].buf) return -1;
        c->slots[i].req.done = 1;
    }
    return 0;
}

void es_cache_free(es_cache *c) {
    if (c->slots)
        for (int i = 0; i < c->nslots; i++) free(c->slots[i].buf);
    free(c->slots);
    free(c->map);
    memset(c, 0, sizeof *c);
}

int es_cache_find(const es_cache *c, int layer, int expert) {
    return c->map[layer * c->nexperts + expert];
}

int es_cache_claim(es_cache *c, es_pool *pool, int layer, int expert) {
    int best = -1;
    double best_score = 0;
    for (int i = 0; i < c->nslots; i++) {
        es_slot *s = &c->slots[i];
        if (s->state == ES_SLOT_LOADING) {
            int d = pool ? es_done(pool, &s->req) : __atomic_load_n(&s->req.done, __ATOMIC_ACQUIRE);
            if (!d) continue;
            s->state = ES_SLOT_READY; /* finished in the background */
        }
        if (s->state == ES_SLOT_EMPTY) { best = i; break; }
        if (s->protect_until >= c->step) continue;
        uint32_t u = s->uses < 16 ? s->uses : 16;
        double score = (double)s->last_use + c->freq_weight * u;
        if (best < 0 || score < best_score) { best = i; best_score = score; }
    }
    if (best < 0) return -1;
    es_slot *s = &c->slots[best];
    if (s->layer >= 0) {
        c->map[s->layer * c->nexperts + s->expert] = -1;
        c->evictions++;
        if (s->prefetched) c->prefetch_wasted++;
    }
    s->layer = layer;
    s->expert = expert;
    s->state = ES_SLOT_LOADING;
    s->prefetched = 0;
    s->uses = 0;
    s->last_use = c->step;
    s->protect_until = c->step + 1;
    c->map[layer * c->nexperts + expert] = best;
    return best;
}
