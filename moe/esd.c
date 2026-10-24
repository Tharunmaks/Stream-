/* libesd.so: token-by-token decoder for es-sparse-v1 checkpoints (moe/es_sparse.h), the library es_common.esmoe.SparseRun
 * loads for the Run page. Built for phone CPUs: the per-token cost is reading one expert per layer, so experts are kept
 * packed (int4 or ternary, as in the file) and multiplied in place with int8 activations.
 *   - Experts are pread into an LRU cache and repacked once, output-major in blocks (int4: 32 weights in 16 bytes, low
 *     nibbles = first 16; ternary: 64 weights in 16 bytes, bit pair k = weights 16k..16k+15), so each output is one
 *     contiguous int8 dot product (SDOT / VNNI / PMADDUBSW shaped loops).
 *   - Activations are quantized to int8 per vector (symmetric, max-abs); routers and the head are int8 per row, LoRA float.
 *   - A pool of worker threads splits every matrix-vector product by output rows (ES_RUN_THREADS, default 4, at most
 *     the number of CPUs); workers spin briefly between jobs and then sleep.
 *   - Next-layer prefetch: before a layer's expert runs, the next layer's router guesses its expert and misses get
 *     posix_fadvise(WILLNEED); recall is counted. Sticky routing biases toward the previous token's expert (0 = off).
 *   void *esd_open(const char *ckpt, const char *experts_file, long cache_bytes)     NULL on error (message on stderr)
 *   int   esd_step(void *h, int token, float *logits256)                              <0 on read error
 *   void  esd_reset(void *h)  esd_set(void *h, int prefetch_k, float sticky)  int esd_slots(void *h)  int esd_threads(void *h, int n)
 *   void  esd_stats(void *h, long out[3] = hits, misses, bytes read)  esd_pfstats(void *h, long out[3] = checks, recalled, issued)
 *   void  esd_close(void *h)   const char *esd_kernel(void *h) ("avx2", "neon-sdot", "neon" or "scalar"; ES_RUN_SCALAR=1 forces scalar)
 * D and F must be multiples of 64. */
#define _GNU_SOURCE
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "es_sparse.h"
#include "esd_kern.h"

#if defined(__x86_64__) && defined(__GNUC__)
#define TC __attribute__((target_clones("arch=haswell", "default")))
#define PAUSE() __builtin_ia32_pause()
#elif defined(__aarch64__)
#define TC
#define PAUSE() __asm__ volatile("yield")
#else
#define TC
#define PAUSE() do {} while (0)
#endif
#define MAXTH 16

typedef struct { int key; uint64_t used; uint8_t *p; } slot;
typedef struct esd esd;
typedef struct { esd *d; int i; } warg;
typedef void (*job_fn)(esd *d, int lo, int hi);
struct esd {
    ess_hdr h; float *P; ess_view V; size_t loe, eb; int fd; unsigned bits; esd_kern k4, k2, k8; const char *kname; int8_t *wr8, *wp8, *hd8; float *wrs, *wps, *hds, *t; int rr, ex;
    slot *sl; int nsl, *where; uint64_t tick; long hits, miss, bytes; uint8_t *raw; int8_t *tmp;
    float *x, *hh, *s, *z, *u, *a, *y, *hprev, *lg; int8_t *xq; float xs; int xsum;
    int *prev, *pred, have; int pfk; float sticky; long pfc, pfr, pfi, ntok;
    /* current job */
    const uint8_t *jw; int jk, jl; const float *jin; float *jout; const int8_t *jw8; const float *jsc;
    /* worker pool */
    int nth; pthread_t th[MAXTH]; job_fn fn; int rows, quit, sleepers; unsigned gen; int done;
    pthread_mutex_t mu; pthread_cond_t cv; warg wa[MAXTH];
};

/* ---- pool: job rows are split into nth contiguous ranges (multiples of 8) */
static void slice(esd *d, int i, int *lo, int *hi) { const int per = ((d->rows + d->nth - 1) / d->nth + 7) & ~7; *lo = i * per < d->rows ? i * per : d->rows; *hi = *lo + per < d->rows ? *lo + per : d->rows; }
static void *worker(void *p) {
    esd *d = ((warg *)p)->d; const int i = ((warg *)p)->i; unsigned seen = 0;
    for (;;) {
        int spins = 0;
        while (__atomic_load_n(&d->gen, __ATOMIC_ACQUIRE) == seen) {
            if (++spins < 1 << 16) { PAUSE(); continue; }
            pthread_mutex_lock(&d->mu); d->sleepers++;
            while (__atomic_load_n(&d->gen, __ATOMIC_ACQUIRE) == seen) pthread_cond_wait(&d->cv, &d->mu);
            d->sleepers--; pthread_mutex_unlock(&d->mu);
        }
        seen = __atomic_load_n(&d->gen, __ATOMIC_ACQUIRE);
        if (d->quit) return NULL;
        int lo, hi; slice(d, i, &lo, &hi); if (lo < hi) d->fn(d, lo, hi);
        __atomic_add_fetch(&d->done, 1, __ATOMIC_ACQ_REL);
    }
}
static void run(esd *d, job_fn fn, int rows, int k) {   /* k: weights per row; small jobs stay on the calling thread */
    d->fn = fn; d->rows = rows;
    if (d->nth == 1 || (long)rows * k < (1L << 17)) { fn(d, 0, rows); return; }
    __atomic_store_n(&d->done, 0, __ATOMIC_RELEASE); __atomic_add_fetch(&d->gen, 1, __ATOMIC_ACQ_REL);
    if (__atomic_load_n(&d->sleepers, __ATOMIC_ACQUIRE)) { pthread_mutex_lock(&d->mu); pthread_cond_broadcast(&d->cv); pthread_mutex_unlock(&d->mu); }
    int lo, hi; slice(d, 0, &lo, &hi); if (lo < hi) fn(d, lo, hi);
    while (__atomic_load_n(&d->done, __ATOMIC_ACQUIRE) < d->nth - 1) PAUSE();
}
static void pool_stop(esd *d) {
    if (d->nth <= 1) { d->nth = 1; return; }
    d->quit = 1; __atomic_add_fetch(&d->gen, 1, __ATOMIC_ACQ_REL);
    pthread_mutex_lock(&d->mu); pthread_cond_broadcast(&d->cv); pthread_mutex_unlock(&d->mu);
    for (int i = 1; i < d->nth; i++) pthread_join(d->th[i], NULL);
    d->quit = 0; d->nth = 1;
}
int esd_threads(void *hd, int n) {
    esd *d = hd; pool_stop(d); const long nc = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    if (n > MAXTH) n = MAXTH;
    if (nc > 0 && n > nc) n = (int)nc;
    d->gen = 0;
    for (int i = 1; i < n; i++) { d->wa[i].d = d; d->wa[i].i = i; if (pthread_create(&d->th[i], NULL, worker, &d->wa[i])) { n = i; break; } }
    d->nth = n; return n;
}

/* ---- kernels */
static void quant(esd *d, const float *x, int n) {
    float m = 0; for (int i = 0; i < n; i++) { const float a = fabsf(x[i]); if (a > m) m = a; }
    const float s = m > 0 ? m / 127 : 1, r = 1 / s; int sum = 0;
    for (int i = 0; i < n; i++) { const int v = (int)lrintf(x[i] * r); d->xq[i] = (int8_t)v; sum += v; }
    d->xs = s; d->xsum = sum;
}
/* job: jout[j] = scale * W[j] . jin for packed output-major rows of jk weights (jin already quantized in xq) */
static void job_mv(esd *d, int lo, int hi) {
    const size_t rb = (size_t)d->jk * d->bits / 8; const float sc = d->h.scale * d->xs, bias = (d->bits == 2 ? 1.0f : 7.5f) * (float)d->xsum;
    int32_t acc[64]; const esd_kern kern = d->bits == 2 ? d->k2 : d->k4;
    for (int j = lo; j < hi; j += 64) { const int n = hi - j < 64 ? hi - j : 64; kern(d->jw + j * rb, rb, d->xq, d->jk, n, acc);
        for (int t = 0; t < n; t++) d->jout[j + t] = sc * ((float)acc[t] - bias); }
}
TC static float dotf(const float *a, const float *b, int n) { float s[8] = {0}; int i = 0; for (; i + 8 <= n; i += 8) for (int q = 0; q < 8; q++) s[q] += a[i + q] * b[i + q]; float r = 0; for (; i < n; i++) r += a[i] * b[i]; for (int q = 0; q < 8; q++) r += s[q]; return r; }
/* job: jout[j] = rowscale[j] * W8[j] . x for int8 rows of jk weights (routers, head; x quantized in xq) */
static void job_mv8(esd *d, int lo, int hi) {
    int32_t acc[64]; const int K = d->jk;
    for (int j = lo; j < hi; j += 64) { const int n = hi - j < 64 ? hi - j : 64; d->k8((const uint8_t *)(d->jw8 + (size_t)j * K), K, d->xq, K, n, acc);
        for (int t = 0; t < n; t++) d->jout[j + t] = d->jsc[j + t] * d->xs * (float)acc[t]; }
}
static void q8rows(const float *w, int rows, int k, int8_t *o, float *sc) {   /* per-row symmetric int8 */
    for (int j = 0; j < rows; j++) { const float *r = w + (size_t)j * k; float m = 0; for (int i = 0; i < k; i++) if (fabsf(r[i]) > m) m = fabsf(r[i]);
        const float s = m > 0 ? m / 127 : 1; sc[j] = s; for (int i = 0; i < k; i++) o[(size_t)j * k + i] = (int8_t)lrintf(r[i] / s); }
}

/* ---- expert cache: file layout -> output-major packed blocks */
static void pack(esd *d, const int8_t *t, int rows, int k, uint8_t *o) {   /* t: k x rows (row-major, file order); o: rows x k packed */
    const size_t rb = (size_t)k * d->bits / 8;
    for (int j = 0; j < rows; j++) { uint8_t *r = o + j * rb;
        if (d->bits == 2) for (int b = 0; b < k / 64; b++) for (int q = 0; q < 16; q++) { uint8_t v = 0;
            for (int p = 0; p < 4; p++) { const int w = t[(size_t)(64 * b + 16 * p + q) * rows + j]; v |= (uint8_t)((w > 0 ? 1 : w < 0 ? 2 : 0) << (2 * p)); } r[16 * b + q] = v; }
        else for (int b = 0; b < k / 32; b++) for (int q = 0; q < 16; q++)
            r[16 * b + q] = (uint8_t)((t[(size_t)(32 * b + q) * rows + j] + 8) | ((t[(size_t)(32 * b + 16 + q) * rows + j] + 8) << 4)); }
}
static void unpack_file(esd *d, const uint8_t *q, size_t n, int8_t *t) {   /* file order -> int8 (int4: nibble - 8, ternary: -1/0/1) */
    if (d->bits == 2) { for (size_t i = 0; i < n / 4; i++) for (int k = 0; k < 4; k++) { const int c = (q[i] >> (2 * k)) & 3; t[4 * i + k] = (int8_t)((c & 1) - (c >> 1)); } }
    else for (size_t i = 0; i < n / 2; i++) { t[2 * i] = (int8_t)((q[i] & 15) - 8); t[2 * i + 1] = (int8_t)((q[i] >> 4) - 8); }
}
static const uint8_t *get(esd *d, int l, int e) {
    const ess_hdr *h = &d->h; const int key = l * d->ex + e % d->ex; d->tick++;
    if (d->where[key] >= 0) { d->hits++; d->sl[d->where[key]].used = d->tick; return d->sl[d->where[key]].p; }
    int v = 0; for (int i = 1; i < d->nsl; i++) if (d->sl[i].used < d->sl[v].used) v = i;
    if (d->sl[v].key >= 0) d->where[d->sl[v].key] = -1;
    d->sl[v].key = -1; const off_t off = (off_t)key * d->eb; size_t got = 0;
    while (got < d->eb) { ssize_t n = pread(d->fd, d->raw + got, d->eb - got, off + got); if (n <= 0) return NULL; got += n; }
    const size_t half = d->eb / 2, n = (size_t)h->D * h->F;
    unpack_file(d, d->raw, n, d->tmp); pack(d, d->tmp, h->F, h->D, d->sl[v].p);          /* W1 (D x F) -> F rows of D */
    unpack_file(d, d->raw + half, n, d->tmp); pack(d, d->tmp, h->D, h->F, d->sl[v].p + half);   /* W2 (F x D) -> D rows of F */
    d->miss++; d->bytes += d->eb; d->sl[v].key = key; d->sl[v].used = d->tick; d->where[key] = v; return d->sl[v].p;
}

static float rmsn(float *o, const float *x, int n) { double s = 0; for (int i = 0; i < n; i++) s += (double)x[i] * x[i]; const float r = (float)(1 / sqrt(s / n + 1e-6)); for (int i = 0; i < n; i++) o[i] = x[i] * r; return r; }
/* router logits of layer l for hidden x (rms + token shift with that layer's previous h) */
static void router(esd *d, int l, const float *x, float *hh, float *s, float *z) {
    const int D = d->h.D; rmsn(hh, x, D); const float *m = d->V.mix[l], *hp = d->hprev + (size_t)l * D;
    for (int c = 0; c < D; c++) s[c] = d->have ? hh[c] + m[c] * hp[c] : hh[c];
    const int E = d->h.E, rr = d->rr; quant(d, s, D);
    if (rr) {   /* low rank: t = Wp s, z = Wr t */
        d->jw8 = d->wp8 + (size_t)l * rr * D; d->jsc = d->wps + (size_t)l * rr; d->jk = D; d->jout = d->t; run(d, job_mv8, rr, D);
        quant(d, d->t, rr); d->jw8 = d->wr8 + (size_t)l * E * rr; d->jsc = d->wrs + (size_t)l * E; d->jk = rr; d->jout = z; run(d, job_mv8, E, rr);
    } else { d->jw8 = d->wr8 + (size_t)l * E * D; d->jsc = d->wrs + (size_t)l * E; d->jk = D; d->jout = z; run(d, job_mv8, E, D); }
}

void esd_reset(void *hd) { esd *d = hd; d->have = 0; for (uint32_t l = 0; l < d->h.L; l++) { d->prev[l] = -1; d->pred[l * 64] = 0; } }
void esd_set(void *hd, int pfk, float sticky) { esd *d = hd; for (uint32_t l = 0; l < d->h.L; l++) d->pred[l * 64] = 0; d->pfk = pfk < 0 ? 0 : pfk > 63 ? 63 : pfk; d->sticky = sticky; }
int esd_slots(void *hd) { return ((esd *)hd)->nsl; }
const char *esd_kernel(void *hd) { return ((esd *)hd)->kname; }
void esd_stats(void *hd, long *o) { esd *d = hd; o[0] = d->hits; o[1] = d->miss; o[2] = d->bytes; }
void esd_pfstats(void *hd, long *o) { esd *d = hd; o[0] = d->pfc; o[1] = d->pfr; o[2] = d->pfi; }

void esd_close(void *hd) {
    esd *d = hd; if (!d) return;
    pool_stop(d); if (d->fd >= 0) close(d->fd);
    if (d->sl) for (int i = 0; i < d->nsl; i++) free(d->sl[i].p);
    free(d->sl); free(d->where); free(d->P); free(d->V.mix); free(d->V.Wr); free(d->V.Wp); free(d->wp8); free(d->wps); free(d->t); free(d->x); free(d->xq); free(d->raw); free(d->tmp); free(d->prev); free(d->pred); free(d->wr8); free(d->wrs); free(d->hd8); free(d->hds);
    pthread_mutex_destroy(&d->mu); pthread_cond_destroy(&d->cv); free(d);
}

void *esd_open(const char *ckpt, const char *experts, long cache_bytes) {
    esd *d = calloc(1, sizeof(esd)); d->fd = -1; d->nth = 1; pthread_mutex_init(&d->mu, NULL); pthread_cond_init(&d->cv, NULL);
    FILE *f = fopen(ckpt, "rb");
    if (!f || fread(&d->h, sizeof d->h, 1, f) != 1 || !ess_check_hdr(&d->h) || d->h.R > 64 || d->h.D % 64 || d->h.F % 64 || ess_rr(&d->h) % 32) {
        fprintf(stderr, "esd: %s is not an es-sparse-v1 checkpoint this decoder runs (needs LoRA rank <= 64, dim and ffn multiples of 64, router rank a multiple of 32)\n", ckpt);
        if (f) fclose(f);
        esd_close(d); return NULL;
    }
    const ess_hdr *h = &d->h; const size_t pn = ess_params_n(h); const int D = h->D, F = h->F, E = h->E, L = h->L;
    d->P = malloc(4 * pn); d->V.mix = malloc(sizeof(float *) * L); d->V.Wr = malloc(sizeof(float *) * L); d->V.Wp = malloc(sizeof(float *) * L);
    if (!d->P || fread(d->P, 4, pn, f) != pn) { fprintf(stderr, "esd: %s is truncated\n", ckpt); fclose(f); esd_close(d); return NULL; }
    fclose(f);
    ess_bind(h, d->P, &d->V); d->loe = ess_loe(h); d->eb = ess_expert_bytes(h); d->bits = ess_bits(h);
    d->fd = open(experts, O_RDONLY);
    d->ex = E;   /* experts per layer stored in the file */
#ifdef ESD_BENCH_SPREAD
    if (getenv("ESD_BENCH_ALIAS") && atoi(getenv("ESD_BENCH_ALIAS")) > 0 && atoi(getenv("ESD_BENCH_ALIAS")) < E) d->ex = atoi(getenv("ESD_BENCH_ALIAS"));   /* esd_bench: E routers, fewer stored experts */
#endif
    if (d->fd < 0 || lseek(d->fd, 0, SEEK_END) < (off_t)((size_t)L * d->ex * d->eb)) { fprintf(stderr, "esd: %s missing or smaller than the checkpoint's experts\n", experts); esd_close(d); return NULL; }
    long ns = cache_bytes / (long)d->eb; if (ns < 1) ns = 1; if (ns > (long)L * d->ex) ns = (long)L * d->ex;
    d->nsl = (int)ns; d->sl = calloc(d->nsl, sizeof(slot)); d->where = malloc(sizeof(int) * L * d->ex);
    d->raw = malloc(d->eb); d->tmp = malloc((size_t)D * F);
    if (!d->sl || !d->where || !d->raw || !d->tmp) { esd_close(d); return NULL; }
    for (int i = 0; i < L * d->ex; i++) d->where[i] = -1;
    for (int i = 0; i < d->nsl; i++) { d->sl[i].key = -1; if (!(d->sl[i].p = malloc(d->eb))) { esd_close(d); return NULL; } }
    const int M = D > F ? D : F, MX = M > E ? M : E;
    d->x = calloc((size_t)D * 4 + (size_t)MX * 3 + (size_t)L * D + 256, 4); d->xq = malloc(M);
    d->hh = d->x + D; d->s = d->hh + D; d->y = d->s + D; d->z = d->y + D; d->u = d->z + MX; d->a = d->u + MX; d->hprev = d->a + MX; d->lg = d->hprev + (size_t)L * D;
    d->prev = malloc(sizeof(int) * L); d->pred = calloc((size_t)L * 64, sizeof(int));
    const int rr = d->rr = (int)ess_rr(h), rk = rr ? rr : D;
    d->wr8 = malloc((size_t)L * E * rk); d->wrs = malloc(4 * (size_t)L * E); d->hd8 = malloc(256 * (size_t)D); d->hds = malloc(4 * 256);
    d->wp8 = malloc((size_t)L * rr * D + 1); d->wps = malloc(4 * (size_t)L * rr + 4); d->t = malloc(4 * (size_t)rr + 4);
    if (!d->wr8 || !d->wrs || !d->hd8 || !d->hds || !d->wp8 || !d->wps || !d->t) { esd_close(d); return NULL; }
    for (int l = 0; l < L; l++) { q8rows(d->V.Wr[l], E, rk, d->wr8 + (size_t)l * E * rk, d->wrs + (size_t)l * E);
        if (rr) q8rows(d->V.Wp[l], rr, D, d->wp8 + (size_t)l * rr * D, d->wps + (size_t)l * rr); }
    q8rows(d->V.head, 256, D, d->hd8, d->hds);
    esd_pick(&d->k4, &d->k2, &d->k8, &d->kname); esd_set(d, 2, 0); esd_reset(d);
    const char *t = getenv("ES_RUN_THREADS"); esd_threads(d, t ? atoi(t) : 4);
    return d;
}

int esd_step(void *hd, int tok, float *logits) {
    esd *d = hd; const ess_hdr *h = &d->h; const int D = h->D, F = h->F, E = h->E, L = h->L, R = h->R; const float ls = h->lora_scale;
    float *x = d->x, *z = d->z, zr[64]; memcpy(x, d->V.emb + (size_t)(tok & 255) * D, 4 * (size_t)D);
    for (int l = 0; l < L; l++) {
        router(d, l, x, d->hh, d->s, z);
        int k = 0; float best = -INFINITY;
        for (int e = 0; e < E; e++) { const float v = z[e] + (e == d->prev[l] ? d->sticky : 0); if (v > best) { best = v; k = e; } }
#ifdef ESD_BENCH_SPREAD
        k = (int)((d->ntok * 7 + l * 13) % E);   /* esd_bench: a different expert every token (no CPU-cache reuse) */
#endif
        float mx = -INFINITY; double sm = 0; for (int e = 0; e < E; e++) if (z[e] > mx) mx = z[e];
        for (int e = 0; e < E; e++) sm += exp((double)z[e] - mx);
        const float g = (float)(exp((double)z[k] - mx) / sm);
        if (d->pfk && l > 0 && d->pred[l * 64] > 0) { d->pfc++; for (int i = 1; i <= d->pred[l * 64]; i++) if (d->pred[l * 64 + i] == k) { d->pfr++; break; } }
        memcpy(d->hprev + (size_t)l * D, d->hh, 4 * (size_t)D);
        if (d->pfk && l + 1 < L && E > 1 && E <= 64) {   /* guess the next layer's expert from the hidden state before this expert's output */
            router(d, l + 1, x, d->y, d->u, d->a); const int kk = d->pfk < E ? d->pfk : E; int *pr = d->pred + (l + 1) * 64;
            for (int e = 0; e < E; e++) zr[e] = d->a[e];
            pr[0] = kk; for (int i = 1; i <= kk; i++) { int b = 0; for (int e = 1; e < E; e++) if (zr[e] > zr[b]) b = e; pr[i] = b; zr[b] = -INFINITY; }
            for (int i = 1; i <= kk; i++) { const int key = (l + 1) * d->ex + pr[i] % d->ex; if (d->where[key] < 0) { posix_fadvise(d->fd, (off_t)key * d->eb, d->eb, POSIX_FADV_WILLNEED); d->pfi++; } }
        }
        const uint8_t *q = get(d, l, k); if (!q) return -1;
        const float *Au = d->P + ess_dense_n(h) + ((size_t)l * E + k) * d->loe, *Bu = Au + (size_t)R * D, *Ad = Bu + (size_t)F * R, *Bd = Ad + (size_t)R * F;
        float *u = d->u, *a = d->a, *y = d->y, zz[64]; const float *s = d->s;
        quant(d, s, D); d->jw = q; d->jk = D; d->jout = u; run(d, job_mv, F, D);
        for (int r = 0; r < R; r++) zz[r] = ls * dotf(Au + (size_t)r * D, s, D);
        for (int j = 0; j < F; j++) { float v = u[j]; for (int r = 0; r < R; r++) v += Bu[(size_t)j * R + r] * zz[r]; a[j] = v / (1 + expf(-v)); }
        quant(d, a, F); d->jw = q + d->eb / 2; d->jk = F; d->jout = y; run(d, job_mv, D, F);
        for (int r = 0; r < R; r++) zz[r] = ls * dotf(Ad + (size_t)r * F, a, F);
        for (int c = 0; c < D; c++) { float v = y[c]; for (int r = 0; r < R; r++) v += Bd[(size_t)c * R + r] * zz[r]; x[c] += g * v; }
        d->prev[l] = k;
    }
    rmsn(d->hh, x, D); quant(d, d->hh, D); d->jw8 = d->hd8; d->jsc = d->hds; d->jk = D; d->jout = logits; run(d, job_mv8, 256, D);
    d->have = 1; d->ntok++; return 0;
}
