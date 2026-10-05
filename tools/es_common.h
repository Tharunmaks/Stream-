/* Shared helpers for the command-line tools. */
#ifndef ES_COMMON_H
#define ES_COMMON_H

#include <stdint.h>
#include <stdio.h>

static inline void es_expert_path(char *out, size_t n, const char *root,
                                  unsigned layer, unsigned expert) {
    snprintf(out, n, "%s/L%03u/E%04u.exp", root, layer, expert);
}

static inline uint64_t es_rng_next(uint64_t *s) { /* xorshift64* */
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1Dull;
}

/* manifest.txt: "layers <L>\nexperts <E>\nfile_bytes <B>\n" */
static inline int es_read_manifest(const char *root, unsigned *layers,
                                   unsigned *experts, unsigned long long *file_bytes) {
    char path[4096];
    snprintf(path, sizeof path, "%s/manifest.txt", root);
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    int ok = fscanf(f, "layers %u\nexperts %u\nfile_bytes %llu\n",
                    layers, experts, file_bytes) == 3;
    fclose(f);
    return ok ? 0 : -1;
}

#endif
