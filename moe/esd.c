/* libesd.so: token-by-token decoder for es-sparse-v1 checkpoints (moe/es_sparse.h), the library es_common.esmoe.SparseRun
 * loads for the Run page. Experts are pread from the experts file into an LRU cache of raw int4 bytes and multiplied in
 * place (no float copy). Next-layer prefetch: before a layer's expert runs, the next layer's router is applied to the
 * current hidden state and the top-k guesses not in the cache get posix_fadvise(WILLNEED); recall is counted.
 * Sticky routing adds a bias to the previous token's expert of the same layer (0 = off).
 *   void *esd_open(const char *ckpt, const char *experts_file, long cache_bytes)     NULL on error (message on stderr)
 *   int   esd_step(void *h, int token, float *logits256)                              <0 on read error
 *   void  esd_reset(void *h)  esd_set(void *h, int prefetch_k, float sticky)  int esd_slots(void *h)
 *   void  esd_stats(void *h, long out[3] = hits, misses, bytes read)  esd_pfstats(void *h, long out[3] = checks, recalled, issued)
 *   void  esd_close(void *h) */
#define _GNU_SOURCE
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "es_sparse.h"

#if defined(__x86_64__) && defined(__GNUC__)
#define TC __attribute__((target_clones("arch=haswell", "default")))
#else
#define TC
#endif

typedef struct { int key; uint64_t used; uint8_t *p; } slot;
typedef struct {
    ess_hdr h; float *P; ess_view V; size_t loe, eb; int fd;
    slot *sl; int nsl, *where; uint64_t tick; long hits, miss, bytes;
    float *x, *hh, *s, *z, *u, *a, *y, *acc, *hprev; int *prev, *pred, have; int pfk; float sticky; long pfc, pfr, pfi;
} esd;

static const uint8_t *get(esd *d, int l, int e) {
    const int key = l * d->h.E + e; d->tick++;
    if (d->where[key] >= 0) { d->hits++; d->sl[d->where[key]].used = d->tick; return d->sl[d->where[key]].p; }
    int v = 0; for (int i = 1; i < d->nsl; i++) if (d->sl[i].used < d->sl[v].used) v = i;
    if (d->sl[v].key >= 0) d->where[d->sl[v].key] = -1;
    const off_t off = (off_t)key * d->eb; size_t got = 0;
    while (got < d->eb) { ssize_t n = pread(d->fd, d->sl[v].p + got, d->eb - got, off + got); if (n <= 0) { d->sl[v].key = -1; return NULL; } got += n; }
    d->miss++; d->bytes += d->eb; d->sl[v].key = key; d->sl[v].used = d->tick; d->where[key] = v; return d->sl[v].p;
}
static float rmsn(float *o, const float *x, int n) { double s = 0; for (int i = 0; i < n; i++) s += (double)x[i] * x[i]; const float r = (float)(1 / sqrt(s / n + 1e-6)); for (int i = 0; i < n; i++) o[i] = x[i] * r; return r; }
static float dotf(const float *a, const float *b, int n) { float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
/* out[j] = scale * sum_i in[i] * (q[i][j] - 7.5), q rows of m nibbles; acc is m floats of scratch */
TC static void mv4(const uint8_t *q, const float *in, int n, int m, float scale, float *out, float *acc) {
    memset(acc, 0, 4 * (size_t)m); double sin = 0;
    for (int i = 0; i < n; i++) { const float v = in[i]; const uint8_t *r = q + (size_t)i * m / 2; sin += v;
        for (int j = 0; j < m / 2; j++) { acc[2 * j] += v * (float)(r[j] & 15); acc[2 * j + 1] += v * (float)(r[j] >> 4); } }
    const float c = (float)(7.5 * sin); for (int j = 0; j < m; j++) out[j] = scale * (acc[j] - c);
}
/* router logits of layer l for hidden x (rms + token shift with that layer's previous h) */
static void router(esd *d, int l, const float *x, float *hh, float *s, float *z) {
    const int D = d->h.D, E = d->h.E; rmsn(hh, x, D); const float *m = d->V.mix[l], *hp = d->hprev + (size_t)l * D;
    for (int c = 0; c < D; c++) s[c] = d->have ? hh[c] + m[c] * hp[c] : hh[c];
    for (int e = 0; e < E; e++) z[e] = dotf(d->V.Wr[l] + (size_t)e * D, s, D);
}

void esd_reset(void *hd) { esd *d = hd; d->have = 0; for (uint32_t l = 0; l < d->h.L; l++) { d->prev[l] = -1; d->pred[l * 64] = 0; } }
void esd_set(void *hd, int pfk, float sticky) { esd *d = hd; for (uint32_t l = 0; l < d->h.L; l++) d->pred[l * 64] = 0; d->pfk = pfk < 0 ? 0 : pfk > 63 ? 63 : pfk; d->sticky = sticky; }
int esd_slots(void *hd) { return ((esd *)hd)->nsl; }
void esd_stats(void *hd, long *o) { esd *d = hd; o[0] = d->hits; o[1] = d->miss; o[2] = d->bytes; }
void esd_pfstats(void *hd, long *o) { esd *d = hd; o[0] = d->pfc; o[1] = d->pfr; o[2] = d->pfi; }

void esd_close(void *hd) {
    esd *d = hd; if (!d) return;
    if (d->fd >= 0) close(d->fd);
    if (d->sl) for (int i = 0; i < d->nsl; i++) free(d->sl[i].p);
    free(d->sl); free(d->where); free(d->P); free(d->V.mix); free(d->V.Wr); free(d->x); free(d->prev); free(d->pred); free(d);
}

void *esd_open(const char *ckpt, const char *experts, long cache_bytes) {
    esd *d = calloc(1, sizeof(esd)); d->fd = -1; FILE *f = fopen(ckpt, "rb");
    if (!f || fread(&d->h, sizeof d->h, 1, f) != 1 || !ess_check_hdr(&d->h) || d->h.R > 64) { fprintf(stderr, "esd: %s is not an es-sparse-v1 checkpoint\n", ckpt); if (f) fclose(f); free(d); return NULL; }
    const ess_hdr *h = &d->h; const size_t pn = ess_params_n(h); const int D = h->D, F = h->F, E = h->E, L = h->L;
    d->P = malloc(4 * pn); if (!d->P || fread(d->P, 4, pn, f) != pn) { fprintf(stderr, "esd: %s is truncated\n", ckpt); fclose(f); free(d->P); free(d); return NULL; }
    fclose(f);
    d->V.mix = malloc(sizeof(float *) * L); d->V.Wr = malloc(sizeof(float *) * L); ess_bind(h, d->P, &d->V); d->loe = ess_loe(h); d->eb = ess_expert_bytes(h);
    d->fd = open(experts, O_RDONLY);
    if (d->fd < 0 || lseek(d->fd, 0, SEEK_END) < (off_t)ess_file_bytes(h)) { fprintf(stderr, "esd: %s missing or smaller than L*E*D*F bytes\n", experts); esd_close(d); return NULL; }
    long ns = cache_bytes / (long)d->eb; if (ns < 1) ns = 1; if (ns > (long)L * E) ns = (long)L * E;
    d->nsl = (int)ns; d->sl = calloc(d->nsl, sizeof(slot)); d->where = malloc(sizeof(int) * L * E);
    for (int i = 0; i < L * E; i++) d->where[i] = -1;
    for (int i = 0; i < d->nsl; i++) { d->sl[i].key = -1; if (!(d->sl[i].p = malloc(d->eb))) { esd_close(d); return NULL; } }
    const int M = D > F ? D : F, MX = M > E ? M : E;
    d->x = calloc((size_t)D * 4 + (size_t)MX * 6 + (size_t)L * D, 4);
    d->hh = d->x + D; d->s = d->hh + D; d->y = d->s + D; d->z = d->y + D; d->u = d->z + MX; d->a = d->u + MX; d->acc = d->a + MX; d->hprev = d->acc + 3 * (size_t)MX;
    d->prev = malloc(sizeof(int) * L); d->pred = calloc((size_t)L * 64, sizeof(int));
    esd_set(d, 2, 0); esd_reset(d); return d;
}

int esd_step(void *hd, int tok, float *logits) {
    esd *d = hd; const ess_hdr *h = &d->h; const int D = h->D, F = h->F, E = h->E, L = h->L, R = h->R; const float ls = h->lora_scale;
    float *x = d->x, *z = d->z, zr[64]; memcpy(x, d->V.emb + (size_t)(tok & 255) * D, 4 * (size_t)D);
    for (int l = 0; l < L; l++) {
        router(d, l, x, d->hh, d->s, z);
        int k = 0; float best = -INFINITY;
        for (int e = 0; e < E; e++) { const float v = z[e] + (e == d->prev[l] ? d->sticky : 0); if (v > best) { best = v; k = e; } }
        float mx = -INFINITY; double sm = 0; for (int e = 0; e < E; e++) if (z[e] > mx) mx = z[e];
        for (int e = 0; e < E; e++) sm += exp((double)z[e] - mx);
        const float g = (float)(exp((double)z[k] - mx) / sm);
        if (d->pfk && l > 0 && d->pred[l * 64] > 0) { d->pfc++; for (int i = 1; i <= d->pred[l * 64]; i++) if (d->pred[l * 64 + i] == k) { d->pfr++; break; } }
        memcpy(d->hprev + (size_t)l * D, d->hh, 4 * (size_t)D);
        if (d->pfk && l + 1 < L && E > 1) {   /* guess the next layer's expert from the hidden state before this expert's output */
            float *hn = d->y, *sn = d->u; router(d, l + 1, x, hn, sn, d->a); const int kk = d->pfk < E ? d->pfk : E; int *pr = d->pred + (l + 1) * 64;
            for (int e = 0; e < E && e < 64; e++) zr[e] = d->a[e];
            pr[0] = kk; for (int i = 1; i <= kk; i++) { int b = -1; for (int e = 0; e < E && e < 64; e++) if (zr[e] > -INFINITY && (b < 0 || zr[e] > zr[b])) b = e; pr[i] = b < 0 ? 0 : b; if (b >= 0) zr[b] = -INFINITY; }
            for (int i = 1; i <= kk; i++) { const int key = (l + 1) * E + pr[i]; if (d->where[key] < 0) { posix_fadvise(d->fd, (off_t)key * d->eb, d->eb, POSIX_FADV_WILLNEED); d->pfi++; } }
            if (E > 64) pr[0] = 0;
        }
        const uint8_t *q = get(d, l, k); if (!q) return -1;
        const float *Au = d->P + ess_dense_n(h) + ((size_t)l * E + k) * d->loe, *Bu = Au + (size_t)R * D, *Ad = Bu + (size_t)F * R, *Bd = Ad + (size_t)R * F;
        float *u = d->u, *a = d->a, *y = d->y, zz[64]; const float *s = d->s;
        mv4(q, s, D, F, h->scale, u, d->acc);
        for (int r = 0; r < R && r < 64; r++) zz[r] = ls * dotf(Au + (size_t)r * D, s, D);
        for (int j = 0; j < F; j++) { float v = u[j]; for (int r = 0; r < R && r < 64; r++) v += Bu[(size_t)j * R + r] * zz[r]; u[j] = v; a[j] = v / (1 + expf(-v)); }
        mv4(q + (size_t)D * F / 2, a, F, D, h->scale, y, d->acc);
        for (int r = 0; r < R && r < 64; r++) zz[r] = ls * dotf(Ad + (size_t)r * F, a, F);
        for (int c = 0; c < D; c++) { float v = y[c]; for (int r = 0; r < R && r < 64; r++) v += Bd[(size_t)c * R + r] * zz[r]; x[c] += g * v; }
        d->prev[l] = k;
    }
    rmsn(d->hh, x, D);
    for (int c = 0; c < 256; c++) logits[c] = dotf(d->V.head + (size_t)c * D, d->hh, D);
    d->have = 1; return 0;
}
