/*
 * es_bench: measure cold expert-load bandwidth and latency with es_io.
 *
 * Every timed read is cold: files are evicted from the page cache before
 * each run, and the tool checks afterwards (via mincore) that reads really
 * bypassed the cache. If O_DIRECT is silently turned into buffered I/O by
 * the filesystem (common with f2fs + Android file-based encryption), the
 * tool says so instead of reporting an inflated number.
 *
 *   ./es_bench -d ~/es_bench                 # thread sweep 1,2,4,6,8
 *   ./es_bench -d ~/es_bench -t 4 -c 512     # one config, 512 KiB chunks
 *   ./es_bench -d ~/es_bench -V              # verify all file hashes
 */
#define _GNU_SOURCE
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/es_format.h"
#include "../src/es_io.h"
#include "es_common.h"

static unsigned g_layers, g_experts;
static unsigned long long g_file_bytes;
static const char *g_root;

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static double pct(uint64_t *v, int n, double p) {
    int i = (int)(p * (n - 1) + 0.5);
    return v[i] / 1e6; /* ns -> ms */
}

static void evict_all(void) {
    char path[4096];
    for (unsigned l = 0; l < g_layers; l++)
        for (unsigned e = 0; e < g_experts; e++) {
            es_expert_path(path, sizeof path, g_root, l, e);
            es_evict(path);
        }
}

/* Average cached fraction over a sample of files. */
static double sample_cached(void) {
    char path[4096];
    double sum = 0;
    int n = 0;
    for (unsigned l = 0; l < g_layers; l++)
        for (unsigned e = 0; e < g_experts; e += 7) {
            es_expert_path(path, sizeof path, g_root, l, e);
            double f = es_cached_fraction(path);
            if (f >= 0) { sum += f; n++; }
        }
    return n ? sum / n : -1;
}

typedef struct {
    double gbps;          /* aggregate GB/s (1e9) over all batches */
    double p50, p90, p99; /* per-batch latency, ms */
    double cached_after;  /* sampled page-cache residency after the run */
    int direct;
    int errors;
} result;

/* Load `batches` groups of `k` distinct random experts from a random layer
 * (what one MoE layer needs per token). Returns stats. */
static result run(int threads, size_t chunk, int want_direct, int k, int batches,
                  uint8_t *bufs, uint64_t seed) {
    result res = {0};
    evict_all();
    es_pool *p = es_pool_create(threads, chunk, want_direct);
    if (!p) { fprintf(stderr, "pool create failed\n"); exit(1); }

    es_req *reqs = calloc((size_t)k, sizeof(es_req));
    uint64_t *lat = calloc((size_t)batches, sizeof(uint64_t));
    unsigned *perm = malloc(g_experts * sizeof(unsigned));
    char path[4096];
    uint64_t total_bytes = 0, t_start = es_now_ns();

    for (int b = 0; b < batches; b++) {
        unsigned layer = (unsigned)(es_rng_next(&seed) % g_layers);
        for (unsigned i = 0; i < g_experts; i++) perm[i] = i;
        for (int i = 0; i < k; i++) { /* partial Fisher-Yates */
            unsigned j = i + (unsigned)(es_rng_next(&seed) % (g_experts - i));
            unsigned t = perm[i]; perm[i] = perm[j]; perm[j] = t;
        }
        uint64_t t0 = es_now_ns();
        for (int i = 0; i < k; i++) {
            reqs[i].dst = bufs + (size_t)i * g_file_bytes;
            reqs[i].cap = g_file_bytes;
            es_expert_path(path, sizeof path, g_root, layer, perm[i]);
            if (es_submit(p, &reqs[i], path) != 0) res.errors++;
        }
        for (int i = 0; i < k; i++) {
            if (es_wait(p, &reqs[i]) != 0) res.errors++;
            else total_bytes += reqs[i].bytes;
        }
        lat[b] = es_now_ns() - t0;
        /* cheap sanity check outside the hot path: header only */
        for (int i = 0; i < k; i++)
            if (!reqs[i].err && es_validate(reqs[i].dst, reqs[i].bytes, 0)) res.errors++;
    }
    double secs = (es_now_ns() - t_start) / 1e9;
    res.direct = es_pool_is_direct(p);
    es_pool_destroy(p);

    res.cached_after = sample_cached();
    qsort(lat, (size_t)batches, sizeof(uint64_t), cmp_u64);
    res.gbps = total_bytes / 1e9 / secs;
    res.p50 = pct(lat, batches, 0.50);
    res.p90 = pct(lat, batches, 0.90);
    res.p99 = pct(lat, batches, 0.99);
    free(reqs); free(lat); free(perm);
    return res;
}

static int verify_all(void) {
    uint8_t *buf = es_alloc(g_file_bytes);
    es_pool *p = es_pool_create(4, 1 << 20, 1);
    char path[4096];
    int bad = 0;
    for (unsigned l = 0; l < g_layers; l++)
        for (unsigned e = 0; e < g_experts; e++) {
            es_req r = {.dst = buf, .cap = g_file_bytes};
            es_expert_path(path, sizeof path, g_root, l, e);
            const char *why = NULL;
            if (es_submit(p, &r, path) || es_wait(p, &r)) why = strerror(r.err);
            else why = es_validate(buf, r.bytes, 1);
            if (!why) {
                const es_header *h = (const es_header *)buf;
                if (h->layer != l || h->expert != e) why = "layer/expert id mismatch";
            }
            if (why) { printf("  BAD %s: %s\n", path, why); bad++; }
        }
    es_pool_destroy(p);
    free(buf);
    printf("verify: %u files, %d bad\n", g_layers * g_experts, bad);
    return bad;
}

static void usage(void) {
    fprintf(stderr,
            "usage: es_bench -d DIR [-t threads (default: sweep 1,2,4,6,8)]\n"
            "                [-c chunk_kib=1024] [-k experts_per_batch=8]\n"
            "                [-n batches=100] [-B buffered I/O] [-V verify]\n"
            "                [-M model_layers=94 for the per-token estimate]\n");
    exit(2);
}

int main(int argc, char **argv) {
    int threads = 0, k = 8, batches = 100, want_direct = 1, verify = 0, model_layers = 94;
    size_t chunk = 1 << 20;
    int opt;
    while ((opt = getopt(argc, argv, "d:t:c:k:n:BVM:h")) != -1) {
        switch (opt) {
        case 'd': g_root = optarg; break;
        case 't': threads = atoi(optarg); break;
        case 'c': chunk = (size_t)atoi(optarg) * 1024; break;
        case 'k': k = atoi(optarg); break;
        case 'n': batches = atoi(optarg); break;
        case 'B': want_direct = 0; break;
        case 'V': verify = 1; break;
        case 'M': model_layers = atoi(optarg); break;
        default: usage();
        }
    }
    if (!g_root || k < 1 || batches < 1 || chunk % ES_ALIGN) usage();
    if (es_read_manifest(g_root, &g_layers, &g_experts, &g_file_bytes) != 0) {
        fprintf(stderr, "no manifest in %s (run es_gen first)\n", g_root);
        return 1;
    }
    if ((unsigned)k > g_experts) k = (int)g_experts;

    char path[4096];
    es_expert_path(path, sizeof path, g_root, 0, 0);
    int direct_ok = es_probe_direct(path);
    printf("dataset : %u layers x %u experts x %.2f MiB = %.2f GB\n", g_layers,
           g_experts, g_file_bytes / 1048576.0,
           (double)g_file_bytes * g_layers * g_experts / 1e9);
    printf("O_DIRECT: %s%s\n", direct_ok ? "accepted" : "NOT available",
           want_direct ? "" : " (disabled with -B)");
    printf("batch   : %d experts from one random layer, %d batches, chunk %zu KiB\n\n",
           k, batches, chunk / 1024);

    if (verify && verify_all() != 0) return 1;

    uint8_t *bufs = es_alloc((size_t)k * g_file_bytes);
    if (!bufs) { perror("alloc"); return 1; }

    int sweep[] = {1, 2, 4, 6, 8};
    int nsweep = threads ? 1 : 5;
    double best = 0;
    int best_t = 0;
    printf("threads   GB/s   batch p50   p90      p99    cached-after  mode\n");
    for (int i = 0; i < nsweep; i++) {
        int t = threads ? threads : sweep[i];
        result r = run(t, chunk, want_direct, k, batches, bufs, 1234 + (uint64_t)t);
        printf("%5d   %6.2f  %7.1fms %7.1fms %7.1fms   %6.1f%%     %s%s\n", t, r.gbps,
               r.p50, r.p90, r.p99, r.cached_after * 100,
               r.direct ? "direct" : "buffered",
               r.errors ? "  ERRORS!" : "");
        fflush(stdout);
        if (r.errors) printf("        %d read/validate errors\n", r.errors);
        if (r.direct && r.cached_after > 0.10)
            printf("        WARNING: pages are in the page cache after O_DIRECT reads.\n"
                   "        The filesystem is silently buffering (likely f2fs + file\n"
                   "        encryption). Numbers are still cold reads, but RAM is used.\n");
        if (r.gbps > best) { best = r.gbps; best_t = t; }
        evict_all();
        sleep(2); /* let the flash and SoC cool a little between configs */
    }

    double layer_ms = g_file_bytes * k / (best * 1e9) * 1e3;
    printf("\nbest: %.2f GB/s at %d threads\n", best, best_t);
    printf("cold cost per MoE layer (%d experts): %.1f ms\n", k, layer_ms);
    printf("per-token streaming cost, %d MoE layers, all experts from flash:\n", model_layers);
    double hits[] = {0.0, 0.5, 0.7, 0.9};
    for (int i = 0; i < 4; i++)
        printf("  cache hit rate %3.0f%%  ->  %.2f s/token (I/O only, no overlap)\n",
               hits[i] * 100, layer_ms * model_layers * (1 - hits[i]) / 1e3);
    free(bufs);
    return 0;
}
