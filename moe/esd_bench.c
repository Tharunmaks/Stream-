/* esd_bench: decode speed of libesd (moe/esd.c) for an es-sparse-v1 shape, with every expert resident in RAM
 * (the phone case: the whole model fits in memory, nothing is read from flash during generation).
 *   esd_bench [--layers 16] [--dim 2048] [--ffn 1024] [--experts 64] [--bits 4|2] [--router-rank 0|RR] [--tokens 256]
 *             [--threads N] [--stored N] [--work DIR]
 * --stored N keeps only N experts per layer in the file (expert e reads stored expert e %% N) so a 70B-shaped model
 * (about 1,000 experts per layer: routers, LoRA and per-token reads at full size) can be timed on a machine without 17.5 GB
 * of RAM; the per-token expert read is the same size and never comes from the CPU cache.
 * Random parameters, same code path as Run, except routing is forced to a different expert every token (built with
 * -DESD_BENCH_SPREAD) so no expert is reused from the CPU cache: the worst case, as for a large E on a phone. Prints tok/s, bytes read per token and this machine's read bandwidth,
 * so the result can be scaled to another device: tok/s there ~ tok/s here * (its GB/s / this GB/s) when bandwidth-bound. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "es_sparse.h"

void *esd_open(const char *ckpt, const char *experts, long cache_bytes);
int esd_step(void *h, int token, float *logits);
void esd_reset(void *h); void esd_set(void *h, int prefetch_k, float sticky); void esd_close(void *h);
void esd_stats(void *h, long *o); int esd_threads(void *h, int n); const char *esd_kernel(void *h);

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ULL;
static uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float nrm(void) { double u = (xr() >> 11) * (1.0 / 9007199254740992.0) + 1e-12, v = (xr() >> 11) * (1.0 / 9007199254740992.0); return (float)(sqrt(-2 * log(u)) * cos(6.283185307 * v)); }
static const char *opt(int argc, char **argv, const char *n, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], n)) return argv[i + 1]; return d; }

int main(int argc, char **argv) {
    ess_hdr h = {0}; memcpy(h.magic, ESS_MAGIC, 8); h.version = 1; h.K = 1; h.T = 128; h.R = 4; h.lora_scale = 1;
    h.L = atoi(opt(argc, argv, "--layers", "16")); h.D = atoi(opt(argc, argv, "--dim", "2048")); h.F = atoi(opt(argc, argv, "--ffn", "1024")); h.E = atoi(opt(argc, argv, "--experts", "64")); h.reserved[1] = atoi(opt(argc, argv, "--router-rank", "0"));
    h.reserved[0] = atoi(opt(argc, argv, "--bits", "4"));
    h.scale = h.reserved[0] == 2 ? ess_default_scale_t(h.D, h.F) : ess_default_scale(h.D, h.F);
    const int ntok = atoi(opt(argc, argv, "--tokens", "256")), nth = atoi(opt(argc, argv, "--threads", "1")); const char *wd = opt(argc, argv, "--work", "."); int st_e = atoi(opt(argc, argv, "--stored", "0")); if (st_e <= 0 || st_e > (int)h.E) st_e = h.E;
    char al[16]; snprintf(al, sizeof al, "%d", st_e); setenv("ESD_BENCH_ALIAS", al, 1);
    char ck[4096], xf[4096]; snprintf(ck, sizeof ck, "%s/bench.ckpt", wd);
    snprintf(xf, sizeof xf, "%s/bench-L%uE%dD%uF%u-b%u.bin", wd, h.L, st_e, h.D, h.F, ess_bits(&h));
    const size_t need = (size_t)h.L * st_e * ess_expert_bytes(&h); struct stat st;
    if (stat(xf, &st) || (size_t)st.st_size < need) {   /* random experts (any byte is a valid int4 pair / ternary quad) */
        FILE *f = fopen(xf, "wb"); uint8_t *b = malloc(1 << 20);
        for (size_t done = 0; done < need; done += 1 << 20) { for (int i = 0; i < (1 << 20) / 8; i++) ((uint64_t *)b)[i] = xr(); fwrite(b, 1, need - done < (1 << 20) ? need - done : (1 << 20), f); }
        fclose(f); free(b);
    }
    const size_t pn = ess_params_n(&h); float *p = malloc(4 * pn);
    for (size_t i = 0; i < pn; i++) p[i] = 0.05f * nrm();
    FILE *f = fopen(ck, "wb"); fwrite(&h, sizeof h, 1, f); fwrite(p, 4, pn, f); fclose(f); free(p);

    /* this machine's streaming read bandwidth over a buffer much larger than the CPU caches */
    const size_t bn = (size_t)1 << 28; uint64_t *bb = malloc(bn); memset(bb, 1, bn); double bw = 0;
    for (int r = 0; r < 3; r++) { const double t0 = now(); uint64_t s = 0; for (size_t i = 0; i < bn / 8; i += 4) s += bb[i] ^ bb[i + 1] ^ bb[i + 2] ^ bb[i + 3]; const double g = bn / (now() - t0) / 1e9; if (g > bw) bw = g; if (s == 42) puts(""); }
    free(bb);
    /* and with 4 threads (phones are quoted at their all-core bandwidth) */
    double bw4 = 0; { const size_t n4 = (size_t)1 << 30; uint64_t *b4 = malloc(n4); memset(b4, 1, n4); volatile uint64_t sink = 0;
        for (int r = 0; r < 2; r++) { const double t0 = now();
            #pragma omp parallel for reduction(^:sink) num_threads(4)
            for (long i = 0; i < (long)(n4 / 8); i += 4) sink ^= b4[i] ^ b4[i + 1] ^ b4[i + 2] ^ b4[i + 3];
            const double g = n4 / (now() - t0) / 1e9; if (g > bw4) bw4 = g; }
        free(b4); }

    void *d = esd_open(ck, xf, (long)need + (1L << 20)); if (!d) return 1;
    esd_set(d, 0, 0); const int got = esd_threads(d, nth);
    float lg[256]; esd_reset(d);
    for (int t = 0; t < 8 * st_e + 256; t++) esd_step(d, (int)(xr() & 255), lg);   /* warm: pull experts into the cache */
    long s0[3]; esd_stats(d, s0); int tok = 'a';
    const double t0 = now();
    for (int t = 0; t < ntok; t++) { esd_step(d, tok, lg); int b = 0; for (int c = 1; c < 256; c++) if (lg[c] > lg[b]) b = c; tok = b; }
    const double dt = now() - t0; long s1[3]; esd_stats(d, s1);
    const double ex = (double)h.L * ess_expert_bytes(&h), rt = (double)ess_router_bytes(&h), lo = (double)h.L * ess_loe(&h) * 4, hb = 256.0 * h.D;
    printf("shape L=%u E=%u d=%u f=%u (%.1f B total), %u-bit experts, router rank %u, active/token %.1f M, threads %d, kernel %s, stored experts/layer %d\n", h.L, h.E, h.D, h.F,
           (double)h.L * h.E * 2 * h.D * h.F / 1e9, ess_bits(&h), h.reserved[1], (double)h.L * 2 * h.D * h.F / 1e6, got, esd_kernel(d), st_e);
    printf("decode %.1f tok/s (%.3f ms/token) | per token: experts %.1f MB + router %.1f MB + LoRA %.1f MB + head %.1f MB | cache reads during timing %ld\n",
           ntok / dt, 1e3 * dt / ntok, ex / 1e6, rt / 1e6, lo / 1e6, hb / 1e6, s1[1] - s0[1]);
    printf("effective %.1f GB/s of model bytes | this machine streams %.1f GB/s on 1 thread, %.1f GB/s on 4\n", (ex + rt + lo + hb) * ntok / dt / 1e9, bw, bw4);
    esd_close(d); return 0;
}
