/* es_sparse: sparse-backprop trainer for es-sparse-v1 MoE models (`es_train moe --sparse [--p6] ...`, used by the
 * Trainer's Train MoE page). Model and file formats: moe/es_sparse.h.
 * Frozen packed-int4 experts are read in place from an mmap'd experts file (generated, seeded, when it does not exist),
 * routing is top-1, and only routed experts are decoded and get gradients: LoRA rank 4 on every routed expert, plus the
 * routers, token-shift mix, embedding and head. No BLAS needed.
 * Products through the frozen experts (forward and backward) run in int8 (moe/es_i8.h: activations quantized per row,
 * as libesd does, times the experts' exact int4/ternary values; AVX-VNNI, AVX2 or NEON SDOT) when D and F are multiples
 * of 16. --exact keeps them in float; --gradcheck uses float and also reports how close the int8 gradient is.
 *   plain (--sparse):      every layer keeps its activations; experts run one after another on one thread.
 *   --p6:                  checkpointed and expert-parallel: only layer inputs are kept, each layer is recomputed during
 *                          backward, and the experts of a layer are spread over --threads (default OMP_NUM_THREADS).
 * Output: "step N train_ce X ... train_tok/s Y", "eval step N heldout_loss X", and one final
 * "RESULT steps=N train_tok/s=Y heldout_before=A heldout_after=B ckpt=PATH". SIGTERM/SIGINT/SIGXCPU stop after the
 * current step, save the checkpoint and print RESULT. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include "es_sparse.h"
#include "es_i8.h"

#if defined(__x86_64__) && defined(__GNUC__)
#define TC __attribute__((target_clones("arch=haswell", "default")))
#else
#define TC
#endif

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ULL;
static uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float nrm(void) { double u = (xr() >> 11) * (1.0 / 9007199254740992.0) + 1e-12, v = (xr() >> 11) * (1.0 / 9007199254740992.0); return (float)(sqrt(-2 * log(u)) * cos(6.283185307 * v)); }
static void *fa(size_t n) { void *p = calloc(n ? n : 1, 4); if (!p) { fprintf(stderr, "out of memory (%zu floats)\n", n); exit(1); } return p; }

/* ---- matrix kernels, row-major. nn: C (n x m) = acc*C + al A (n x k) B (k x m); nt: B is (m x k); tn: C (a x b) += al A^T B, A (n x a), B (n x b) */
TC static void mm_nn(int n, int k, int m, float al, const float *A, const float *B, float *C, int acc) {
    if (!acc) memset(C, 0, 4 * (size_t)n * m);
    for (int j0 = 0; j0 < m; j0 += 256) {
        const int jn = m - j0 < 256 ? m - j0 : 256; int i = 0;
        for (; i + 4 <= n; i += 4) {
            float *c0 = C + (size_t)i * m + j0, *c1 = c0 + m, *c2 = c1 + m, *c3 = c2 + m; const float *a = A + (size_t)i * k;
            for (int kk = 0; kk < k; kk++) {
                const float a0 = al * a[kk], a1 = al * a[k + kk], a2 = al * a[2 * k + kk], a3 = al * a[3 * k + kk]; const float *b = B + (size_t)kk * m + j0;
                for (int j = 0; j < jn; j++) { const float bj = b[j]; c0[j] += a0 * bj; c1[j] += a1 * bj; c2[j] += a2 * bj; c3[j] += a3 * bj; }
            }
        }
        for (; i < n; i++) { float *c = C + (size_t)i * m + j0; for (int kk = 0; kk < k; kk++) { const float a = al * A[(size_t)i * k + kk]; const float *b = B + (size_t)kk * m + j0; for (int j = 0; j < jn; j++) c[j] += a * b[j]; } }
    }
}
TC static float dot(const float *a, const float *b, int k) {
    float s[8] = {0}; int i = 0; for (; i + 8 <= k; i += 8) for (int q = 0; q < 8; q++) s[q] += a[i + q] * b[i + q];
    float r = 0; for (; i < k; i++) r += a[i] * b[i]; for (int q = 0; q < 8; q++) r += s[q]; return r;
}
static void mm_nt(int n, int k, int m, float al, const float *A, const float *B, float *C, int acc) {
    if (k <= 8 && (size_t)k * m <= 32768) {   /* LoRA up-projections (k = rank): transpose B and go through mm_nn */
        float *bt = alloca(4 * (size_t)k * m); for (int j = 0; j < m; j++) for (int r = 0; r < k; r++) bt[(size_t)r * m + j] = B[(size_t)j * k + r];
        mm_nn(n, k, m, al, A, bt, C, acc); return; }
    for (int j0 = 0; j0 < m; j0 += 32) { const int j1 = j0 + 32 < m ? j0 + 32 : m;
        for (int i = 0; i < n; i++) { const float *a = A + (size_t)i * k; float *c = C + (size_t)i * m;
            for (int j = j0; j < j1; j++) { const float v = al * dot(a, B + (size_t)j * k, k); c[j] = acc ? c[j] + v : v; } } }
}
TC static void mm_tn_p(int n, int a, int b, int p0, int p1, float al, const float *A, const float *B, float *C) {   /* rows p0..p1 of C */
    for (int i = 0; i < n; i++) { const float *x = A + (size_t)i * a, *y = B + (size_t)i * b;
        for (int p = p0; p < p1; p++) { const float v = al * x[p]; if (v == 0) continue; float *c = C + (size_t)p * b; for (int q = 0; q < b; q++) c[q] += v * y[q]; } }
}
static void mm_tn(int n, int a, int b, float al, const float *A, const float *B, float *C) { mm_tn_p(n, a, b, 0, a, al, A, B, C); }
static float rms(float *o, const float *x, int n) { double s = 0; for (int i = 0; i < n; i++) s += (double)x[i] * x[i]; const float r = (float)(1 / sqrt(s / n + 1e-6)); for (int i = 0; i < n; i++) o[i] = x[i] * r; return r; }
static void rms_bwd(float *dx, const float *a, float r, const float *da, int n) { double s = 0; for (int i = 0; i < n; i++) s += (double)da[i] * a[i]; const float c = (float)(s / n); for (int i = 0; i < n; i++) dx[i] += r * (da[i] - a[i] * c); }

/* ---- config, params, experts */
static int L, E, D, F, R = 4, T = 128, B = 8, NT, NTH = 1, P6, RR; static unsigned BITS = 4; static size_t EBY;
static float SCALE, LS = 1.0f, AUX = 0.01f, LR = 2e-3f, LRL = 1e-3f;
static ess_hdr HD; static const uint8_t *XF;
static float *Pp, *Pg, *Pm, *Pv; static size_t PN, DN, LOE; static ess_view V, G; static uint8_t *touched;
#define LORA(base, l, e) ((base) + ((size_t)(l) * E + (e)) * LOE)

typedef struct { float *h, *rh, *s, *t, *p, *gate, *u, *zu, *zd, *y; int *sel; } act_t;
static act_t *ACT; static float **XIN, *DS, *DG, *DZ, *DT, *XS;
static int *cnt, *off, *lst;
typedef struct { float *S, *U, *A, *Y, *dA, *dS, *Zu, *Zd, *Tm, *T2, *W1, *W2, *Wt; int8_t *Bq, *Aq; int32_t *cs, *ar, *Ci; float *as; double bytes; } ws_t;
static ws_t *WS;
static double AUXL; static uint64_t RSIG;   /* hash of every routing decision in the last run (gradcheck skips entries that flip a route) */

/* parallel over an index range (token rows, or output rows of a weight gradient): fn(i0, i1, ctx) on --threads slices
 * with --p6. Every element is computed by the same code in the same order whatever the split, so results do not
 * depend on the thread count. */
typedef void (*rfn)(int, int, void *);
typedef struct { rfn fn; void *ctx; int i0, i1; } rarg;
static void *rworker(void *a) { rarg *r = a; r->fn(r->i0, r->i1, r->ctx); return NULL; }
static void par_range(rfn fn, int n, void *ctx) {
    const int nt = P6 && n >= 2 * NTH ? NTH : 1;
    if (nt == 1) { fn(0, n, ctx); return; }
    pthread_t th[64]; rarg a[64];
    for (int i = 0; i < nt; i++) { a[i] = (rarg){fn, ctx, (int)((long)n * i / nt), (int)((long)n * (i + 1) / nt)}; if (i) pthread_create(&th[i], NULL, rworker, &a[i]); }
    rworker(&a[0]); for (int i = 1; i < nt; i++) pthread_join(th[i], NULL);
}
typedef struct { int kind, n, k, m; float al; const float *A, *B; float *C; int acc; } mmc;
static void mm_part(int i0, int i1, void *c) { const mmc *x = c;
    if (x->kind == 2) { mm_tn_p(x->n, x->k, x->m, i0, i1, x->al, x->A, x->B, x->C); return; }
    (x->kind ? mm_nt : mm_nn)(i1 - i0, x->k, x->m, x->al, x->A + (size_t)i0 * x->k, x->B, x->C + (size_t)i0 * x->m, x->acc);
}
static void pmm_nn(int n, int k, int m, float al, const float *A, const float *B, float *C, int acc) { mmc x = {0, n, k, m, al, A, B, C, acc}; par_range(mm_part, n, &x); }
static void pmm_tn(int n, int a, int b, float al, const float *A, const float *B, float *C) { mmc x = {2, n, a, b, al, A, B, C, 0}; par_range(mm_part, a, &x); }
typedef struct { int l; act_t *a; const float *x; float *y; } tctx;

static void group(const int *sel) {
    memset(cnt, 0, sizeof(int) * E); for (int t = 0; t < NT; t++) cnt[sel[t]]++;
    off[0] = 0; for (int e = 1; e <= E; e++) off[e] = off[e - 1] + cnt[e - 1];
    int *pos = alloca(sizeof(int) * E); memcpy(pos, off, sizeof(int) * E); for (int t = 0; t < NT; t++) lst[pos[sel[t]]++] = t;
}
static const uint8_t *expert(int l, int e) { return XF + ((size_t)l * E + e) * EBY; }
static float silu(float u) { return u / (1 + expf(-u)); }

/* ---- int8 products through a frozen expert: weights are the exact int8 values (int4: nib - 8, the +0.5 folded in
 * through the row sums of the activations; ternary: -1, 0, 1), times SCALE */
static int FAST, I8CS; static i8_gemm_fn I8; static const char *I8N = "float";
static uint16_t LUT4[256]; static uint32_t LUT2[256];   /* packed byte -> its 2 (int4) or 4 (ternary) int8 weights */
static void i8_tables(void) {
    for (int b = 0; b < 256; b++) { int8_t v[4]; v[0] = (int8_t)((b & 15) - 8); v[1] = (int8_t)((b >> 4) - 8); memcpy(&LUT4[b], v, 2);
        for (int k = 0; k < 4; k++) { const int c = (b >> (2 * k)) & 3; v[k] = (int8_t)((c & 1) - (c >> 1)); } memcpy(&LUT2[b], v, 4); }
}
/* the kernel's B is [k/4][m][4]: one 32-bit word per (k-quad, output). Both decoders write it straight from the packed
 * matrix, 16 outputs at a time so the writes are contiguous. */
/* B for X W, W packed k x m row-major: word (iq, j) = W[4 iq + 0..3][j] */
static void vnni_nn(const uint8_t *q, int k, int m, uint32_t *Bo) {
    if (BITS == 2) { const int rb = m / 4;
        for (int iq = 0; iq < k / 4; iq++) { const uint8_t *r = q + (size_t)4 * iq * rb; uint32_t *o = Bo + (size_t)iq * m;
            for (int c = 0; c < rb; c++) { const uint32_t a = LUT2[r[c]], b = LUT2[r[rb + c]], d = LUT2[r[2 * rb + c]], e = LUT2[r[3 * rb + c]];
                for (int x = 0; x < 4; x++) o[4 * c + x] = ((a >> 8 * x) & 255) | ((b >> 8 * x) & 255) << 8 | ((d >> 8 * x) & 255) << 16 | ((e >> 8 * x) & 255) << 24; } }
        return; }
    const int rb = m / 2;
    for (int iq = 0; iq < k / 4; iq++) { const uint8_t *r = q + (size_t)4 * iq * rb; uint32_t *o = Bo + (size_t)iq * m;
        for (int c = 0; c < rb; c++) { const uint32_t a = LUT4[r[c]], b = LUT4[r[rb + c]], d = LUT4[r[2 * rb + c]], e = LUT4[r[3 * rb + c]];
            o[2 * c] = (a & 255) | (b & 255) << 8 | (d & 255) << 16 | (e & 255) << 24; o[2 * c + 1] = a >> 8 | (b >> 8) << 8 | (d >> 8) << 16 | (e >> 8) << 24; } }
}
/* B for X W^T, W packed m x k row-major: word (iq, j) = W[j][4 iq + 0..3] */
static void vnni_nt(const uint8_t *q, int m, int k, uint32_t *Bo) {
    const size_t rb = (size_t)k * BITS / 8;
    for (int j0 = 0; j0 < m; j0 += 16) for (int iq = 0; iq < k / 4; iq++) { uint32_t *o = Bo + (size_t)iq * m + j0; const uint8_t *r = q + (size_t)j0 * rb;
        if (BITS == 2) for (int j = 0; j < 16; j++) o[j] = LUT2[r[j * rb + iq]];
        else for (int j = 0; j < 16; j++) o[j] = LUT4[r[j * rb + 2 * iq]] | (uint32_t)LUT4[r[j * rb + 2 * iq + 1]] << 16; }
}
static void colsums(const uint32_t *Bo, int k, int m, int32_t *cs) {   /* only the AVX-VNNI / AVX2 kernels need these */
    memset(cs, 0, 4 * (size_t)m);
    for (int iq = 0; iq < k / 4; iq++) for (int j = 0; j < m; j++) { const uint32_t v = Bo[(size_t)iq * m + j]; cs[j] += (int8_t)v + (int8_t)(v >> 8) + (int8_t)(v >> 16) + (int8_t)(v >> 24); }
}
#define VNNI(how, q, a, b) do { vnni_##how(q, a, b, (uint32_t *)w->Bq); if (I8CS) colsums((uint32_t *)w->Bq, #how[1] == 'n' ? a : b, #how[1] == 'n' ? b : a, w->cs); } while (0)
/* out (n x m) = X (n x k) . (the matrix in w->Bq) */
static void i8mm(ws_t *w, int n, int k, int m, const float *X, float *out) {
    i8_quant_rows(X, n, k, w->Aq, w->as, w->ar); I8(n, m, k, w->Aq, w->Bq, w->cs, w->Ci);
    const float half = BITS == 2 ? 0 : 0.5f;
    for (int t = 0; t < n; t++) { const float sc = w->as[t] * SCALE, o = half * (float)w->ar[t]; const int32_t *c = w->Ci + (size_t)t * m; float *y = out + (size_t)t * m;
        for (int j = 0; j < m; j++) y[j] = sc * ((float)c[j] + o); }
}

/* parallel over experts: fn(tid, e, ctx) */
typedef void (*efn)(ws_t *, int, int, void *);
typedef struct { efn fn; int l; void *ctx; int next; ws_t *w; } pjob;
static void *pworker(void *a) { pjob *j = ((void **)a)[0]; ws_t *w = ((void **)a)[1]; int e; while ((e = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED)) < E) if (cnt[e]) j->fn(w, j->l, e, j->ctx); return NULL; }
static void par_experts(efn fn, int l, void *ctx) {
    pjob j = {fn, l, ctx, 0, NULL}; const int nt = P6 ? NTH : 1;
    if (nt == 1) { for (int e = 0; e < E; e++) if (cnt[e]) fn(&WS[0], l, e, ctx); return; }
    pthread_t th[64]; void *args[64][2];
    for (int i = 0; i < nt; i++) { args[i][0] = &j; args[i][1] = &WS[i]; if (i) pthread_create(&th[i], NULL, pworker, args[i]); }
    pworker(args[0]); for (int i = 1; i < nt; i++) pthread_join(th[i], NULL);
}

/* ---- forward */
typedef struct { act_t *a; float *xo; } fctx;
static void fwd_expert(ws_t *w, int l, int e, void *c) {
    act_t *a = ((fctx *)c)->a; float *xo = ((fctx *)c)->xo; const int n = cnt[e], *li = lst + off[e];
    const uint8_t *q = expert(l, e); w->bytes += (double)EBY;
    if (!FAST) { ess_decode(q, (size_t)D * F, w->W1, SCALE, BITS); ess_decode(q + EBY / 2, (size_t)F * D, w->W2, SCALE, BITS); }
    const float *Au = LORA(V.lo, l, e), *Bu = Au + (size_t)R * D, *Ad = Bu + (size_t)F * R, *Bd = Ad + (size_t)R * F;
    for (int i = 0; i < n; i++) memcpy(w->S + (size_t)i * D, a->s + (size_t)li[i] * D, 4 * (size_t)D);
    if (FAST) { VNNI(nn, q, D, F); i8mm(w, n, D, F, w->S, w->U); }
    else mm_nn(n, D, F, 1, w->S, w->W1, w->U, 0);
    if (R) { mm_nt(n, D, R, 1, w->S, Au, w->Zu, 0); mm_nt(n, R, F, LS, w->Zu, Bu, w->U, 1); }
    for (size_t i = 0; i < (size_t)n * F; i++) w->A[i] = silu(w->U[i]);
    if (FAST) { VNNI(nn, q + EBY / 2, F, D); i8mm(w, n, F, D, w->A, w->Y); }
    else mm_nn(n, F, D, 1, w->A, w->W2, w->Y, 0);
    if (R) { mm_nt(n, F, R, 1, w->A, Ad, w->Zd, 0); mm_nt(n, R, D, LS, w->Zd, Bd, w->Y, 1); }
    for (int i = 0; i < n; i++) {
        const int t = li[i]; const float g = a->gate[t], *y = w->Y + (size_t)i * D; float *x = xo + (size_t)t * D;
        memcpy(a->u + (size_t)t * F, w->U + (size_t)i * F, 4 * (size_t)F); memcpy(a->y + (size_t)t * D, y, 4 * (size_t)D);
        if (R) { memcpy(a->zu + (size_t)t * R, w->Zu + (size_t)i * R, 4 * R); memcpy(a->zd + (size_t)t * R, w->Zd + (size_t)i * R, 4 * R); }
        for (int c = 0; c < D; c++) x[c] += g * y[c];
    }
}
static void lf_rms(int t0, int t1, void *c) { const tctx *x = c; for (int t = t0; t < t1; t++) x->a->rh[t] = rms(x->a->h + (size_t)t * D, x->x + (size_t)t * D, D); }
static void lf_route(int t0, int t1, void *c) {   /* token shift, router, softmax for tokens t0..t1 (needs every h) */
    const int l = ((tctx *)c)->l, n = t1 - t0; act_t *a = ((tctx *)c)->a;
    for (int t = t0; t < t1; t++) { float *s = a->s + (size_t)t * D; const float *h = a->h + (size_t)t * D, *m = V.mix[l];
        if (t % T) { const float *hp = h - D; for (int c = 0; c < D; c++) s[c] = h[c] + m[c] * hp[c]; } else memcpy(s, h, 4 * (size_t)D); }
    float *s = a->s + (size_t)t0 * D, *p = a->p + (size_t)t0 * E;
    if (RR) { mm_nt(n, D, RR, 1, s, V.Wp[l], a->t + (size_t)t0 * RR, 0); mm_nt(n, RR, E, 1, a->t + (size_t)t0 * RR, V.Wr[l], p, 0); }
    else mm_nt(n, D, E, 1, s, V.Wr[l], p, 0);
    for (int t = t0; t < t1; t++) {
        float *z = a->p + (size_t)t * E; int k = 0; for (int e = 1; e < E; e++) if (z[e] > z[k]) k = e;
        double sm = 0; const float mx = z[k]; for (int e = 0; e < E; e++) { z[e] = expf(z[e] - mx); sm += z[e]; } for (int e = 0; e < E; e++) z[e] = (float)(z[e] / sm);
        a->sel[t] = k; a->gate[t] = z[k];
    }
}
static void layer_fwd(int l, const float *xi, float *xo, act_t *a) {
    tctx c0 = {l, a, xi, NULL}; par_range(lf_rms, NT, &c0); par_range(lf_route, NT, &c0);
    for (int t = 0; t < NT; t++) RSIG = RSIG * 1000003u + (uint64_t)a->sel[t] + 1;
    if (xo != xi) memcpy(xo, xi, 4 * (size_t)NT * D);
    group(a->sel); fctx c = {a, xo}; par_experts(fwd_expert, l, &c);
}

/* ---- backward: dx holds d(x_out) on entry and d(x_in) on return */
typedef struct { act_t *a; const float *dx; } bctx;
static void bwd_expert(ws_t *w, int l, int e, void *c) {
    act_t *a = ((bctx *)c)->a; const float *dx = ((bctx *)c)->dx; const int n = cnt[e], *li = lst + off[e];
    const uint8_t *q = expert(l, e); w->bytes += (double)EBY;
    /* transposed: W1T (F x D) = W1^T, W2T (D x F) = W2^T, so both products below are nn */
    if (!FAST) { ess_decode(q, (size_t)D * F, w->Wt, SCALE, BITS);
        for (int i = 0; i < D; i++) for (int j = 0; j < F; j++) w->W1[(size_t)j * D + i] = w->Wt[(size_t)i * F + j];
        ess_decode(q + EBY / 2, (size_t)F * D, w->Wt, SCALE, BITS);
        for (int j = 0; j < F; j++) for (int i = 0; i < D; i++) w->W2[(size_t)i * F + j] = w->Wt[(size_t)j * D + i]; }
    const float *Au = LORA(V.lo, l, e), *Bu = Au + (size_t)R * D, *Ad = Bu + (size_t)F * R, *Bd = Ad + (size_t)R * F;
    float *gAu = LORA(G.lo, l, e), *gBu = gAu + (size_t)R * D, *gAd = gBu + (size_t)F * R, *gBd = gAd + (size_t)R * F;
    for (int i = 0; i < n; i++) {
        const int t = li[i]; const float g = a->gate[t], *d = dx + (size_t)t * D; float *dy = w->Y + (size_t)i * D;
        for (int c = 0; c < D; c++) dy[c] = g * d[c];
        DG[t] = dot(d, a->y + (size_t)t * D, D);
        memcpy(w->S + (size_t)i * D, a->s + (size_t)t * D, 4 * (size_t)D); memcpy(w->U + (size_t)i * F, a->u + (size_t)t * F, 4 * (size_t)F);
        if (R) { memcpy(w->Zu + (size_t)i * R, a->zu + (size_t)t * R, 4 * R); memcpy(w->Zd + (size_t)i * R, a->zd + (size_t)t * R, 4 * R); }
    }
    for (size_t i = 0; i < (size_t)n * F; i++) w->A[i] = silu(w->U[i]);
    if (FAST) { VNNI(nt, q + EBY / 2, F, D); i8mm(w, n, D, F, w->Y, w->dA); }
    else mm_nn(n, D, F, 1, w->Y, w->W2, w->dA, 0);
    if (R) { mm_nn(n, D, R, 1, w->Y, Bd, w->Tm, 0); mm_nn(n, R, F, LS, w->Tm, Ad, w->dA, 1);
             mm_tn(n, D, R, LS, w->Y, w->Zd, gBd); mm_tn(n, R, F, LS, w->Tm, w->A, gAd); }
    for (size_t i = 0; i < (size_t)n * F; i++) { const float u = w->U[i], s = 1 / (1 + expf(-u)); w->dA[i] *= s * (1 + u * (1 - s)); }
    if (FAST) { VNNI(nt, q, D, F); i8mm(w, n, F, D, w->dA, w->dS); }
    else mm_nn(n, F, D, 1, w->dA, w->W1, w->dS, 0);
    if (R) { mm_nn(n, F, R, 1, w->dA, Bu, w->T2, 0); mm_nn(n, R, D, LS, w->T2, Au, w->dS, 1);
             mm_tn(n, F, R, LS, w->dA, w->Zu, gBu); mm_tn(n, R, D, LS, w->T2, w->S, gAu); }
    for (int i = 0; i < n; i++) memcpy(DS + (size_t)li[i] * D, w->dS + (size_t)i * D, 4 * (size_t)D);
    touched[l * E + e] = 1;
}
static void lb_dz(int t0, int t1, void *c) {   /* router logits gradient for tokens t0..t1; x = per-expert balance coefficients */
    const act_t *a = ((tctx *)c)->a; const float *ce = ((tctx *)c)->x;
    for (int t = t0; t < t1; t++) {
        const float *p = a->p + (size_t)t * E; float *z = DZ + (size_t)t * E; const int k = a->sel[t]; const float gk = DG[t] * p[k];
        double m = 0; for (int e = 0; e < E; e++) m += p[e] * ce[e];
        for (int e = 0; e < E; e++) z[e] = gk * ((e == k) - p[e]) + p[e] * (ce[e] - (float)m);
    }
}
static void lb_mix(int c0, int c1, void *c) {   /* mix gradient, channels c0..c1 */
    const act_t *a = ((tctx *)c)->a; float *gm = G.mix[((tctx *)c)->l];
    for (int t = 0; t < NT; t++) if (t % T) { const float *ds = DS + (size_t)t * D, *hp = a->h + (size_t)(t - 1) * D; for (int i = c0; i < c1; i++) gm[i] += ds[i] * hp[i]; }
}
static void lb_shift(int t0, int t1, void *c) {   /* d h (into s, no longer needed), then through rms into dx */
    act_t *a = ((tctx *)c)->a; float *dx = ((tctx *)c)->y; const float *m = V.mix[((tctx *)c)->l];
    for (int t = t0; t < t1; t++) {
        const float *ds = DS + (size_t)t * D; float *dh = a->s + (size_t)t * D;
        for (int i = 0; i < D; i++) dh[i] = ds[i];
        if (t % T != T - 1) { const float *dn = DS + (size_t)(t + 1) * D; for (int i = 0; i < D; i++) dh[i] += m[i] * dn[i]; }
        rms_bwd(dx + (size_t)t * D, a->h + (size_t)t * D, a->rh[t], dh, D);
    }
}
static void layer_bwd(int l, float *dx, act_t *a) {
    if (P6) layer_fwd(l, XIN[l], XS, a);   /* checkpointed: recompute this layer from its saved input */
    else group(a->sel);
    memset(DS, 0, 4 * (size_t)NT * D); memset(DG, 0, 4 * (size_t)NT);
    bctx c = {a, dx}; par_experts(bwd_expert, l, &c);
    /* router: gate = softmax(z)[sel]; Switch balance loss AUX * E * sum_e f_e * mean_t p_te (f_e constant) */
    float *ce = alloca(4 * (size_t)E); double bal = 0;
    for (int e = 0; e < E; e++) ce[e] = AUX * E * ((float)cnt[e] / NT) / NT;
    tctx c1 = {l, a, ce, dx}; par_range(lb_dz, NT, &c1);
    for (int e = 0; e < E; e++) { double pm = 0; for (int t = 0; t < NT; t++) pm += a->p[(size_t)t * E + e]; bal += ((double)cnt[e] / NT) * pm / NT; }
    AUXL += AUX * E * bal;
    if (RR) { pmm_tn(NT, E, RR, 1, DZ, a->t, G.Wr[l]); pmm_nn(NT, E, RR, 1, DZ, V.Wr[l], DT, 0); pmm_tn(NT, RR, D, 1, DT, a->s, G.Wp[l]); pmm_nn(NT, RR, D, 1, DT, V.Wp[l], DS, 1); }
    else { pmm_tn(NT, E, D, 1, DZ, a->s, G.Wr[l]); pmm_nn(NT, E, D, 1, DZ, V.Wr[l], DS, 1); }
    /* token shift s_t = h_t + mix h_(t-1), then rms */
    par_range(lb_mix, D, &c1); par_range(lb_shift, NT, &c1);
}

/* ---- whole model: tok is B x (T+1) bytes; returns mean cross-entropy, and accumulates gradients when train */
static float *XF_, *RF, *LG, *DX; static double *RL; static const uint8_t *TOK;
static void head_fwd(int t0, int t1, void *c) {   /* final rms, logits, per-token loss and d logits for tokens t0..t1 */
    (void)c; const int n = t1 - t0;
    for (int t = t0; t < t1; t++) RF[t] = rms(XF_ + (size_t)t * D, XIN[L] + (size_t)t * D, D);
    mm_nt(n, D, 256, 1, XF_ + (size_t)t0 * D, V.head, LG + (size_t)t0 * 256, 0);
    for (int i = t0; i < t1; i++) {
        float *z = LG + (size_t)i * 256, mx = -INFINITY; double sum = 0; const int y = TOK[(i / T) * (T + 1) + i % T + 1];
        for (int c = 0; c < 256; c++) if (z[c] > mx) mx = z[c];
        for (int c = 0; c < 256; c++) sum += exp((double)z[c] - mx);
        RL[i] = log(sum) + mx - z[y];
        for (int c = 0; c < 256; c++) z[c] = (float)(exp((double)z[c] - mx) / sum / NT);
        z[y] -= 1.0f / NT;
    }
}
static void head_bwd(int t0, int t1, void *c) {
    (void)c; float *dxf = XS + (size_t)t0 * D; mm_nn(t1 - t0, 256, D, 1, LG + (size_t)t0 * 256, V.head, dxf, 0); memset(DX + (size_t)t0 * D, 0, 4 * (size_t)(t1 - t0) * D);
    for (int t = t0; t < t1; t++) rms_bwd(DX + (size_t)t * D, XF_ + (size_t)t * D, RF[t], XS + (size_t)t * D, D);
}
static double run(const uint8_t *tok, int train) {
    RSIG = 0;
    for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) memcpy(XIN[0] + ((size_t)b * T + t) * D, V.emb + (size_t)tok[b * (T + 1) + t] * D, 4 * (size_t)D);
    for (int l = 0; l < L; l++) layer_fwd(l, XIN[l], XIN[l + 1], train && !P6 ? &ACT[l] : &ACT[0]);
    TOK = tok; par_range(head_fwd, NT, NULL);
    double loss = 0; for (int t = 0; t < NT; t++) loss += RL[t];
    if (!train) return loss / NT;
    pmm_tn(NT, 256, D, 1, LG, XF_, G.head);
    par_range(head_bwd, NT, NULL);
    for (int l = L - 1; l >= 0; l--) layer_bwd(l, DX, P6 ? &ACT[0] : &ACT[l]);
    for (int b = 0; b < B; b++) for (int t = 0; t < T; t++) { float *g = G.emb + (size_t)tok[b * (T + 1) + t] * D; const float *d = DX + ((size_t)b * T + t) * D; for (int c = 0; c < D; c++) g[c] += d[c]; }
    return loss / NT;
}

static int step_no;
static void adam(size_t a, size_t n, float lr) {
    const float b1 = 0.9f, b2 = 0.999f, c1 = 1 - powf(b1, step_no), c2 = 1 - powf(b2, step_no);
    for (size_t i = a; i < a + n; i++) { const float g = Pg[i]; Pm[i] = b1 * Pm[i] + (1 - b1) * g; Pv[i] = b2 * Pv[i] + (1 - b2) * g * g; Pp[i] -= lr * (Pm[i] / c1) / (sqrtf(Pv[i] / c2) + 1e-8f); Pg[i] = 0; }
}

static int save(const char *path, int steps) {
    char tmp[4096]; snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb"); if (!f) { perror(tmp); return -1; }
    HD.steps = steps; int ok = fwrite(&HD, sizeof HD, 1, f) == 1 && fwrite(Pp, 4, PN, f) == PN; ok &= fclose(f) == 0;
    if (!ok || rename(tmp, path)) { perror(path); unlink(tmp); return -1; }
    return 0;
}

static int gen_experts(const char *path, size_t bytes) {
    char tmp[4096]; snprintf(tmp, sizeof tmp, "%s.tmp.%d", path, (int)getpid());
    printf("generating frozen experts %s: %.2f GB of seeded random bytes\n", path, bytes / 1e9); fflush(stdout);
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (fd < 0) { perror(tmp); return -1; }
    uint64_t s = 0x9E3779B97F4A7C15ULL; const size_t CH = 1 << 20; uint8_t *buf = malloc(CH);
    for (size_t done = 0; done < bytes;) {
        const size_t n = bytes - done < CH ? bytes - done : CH;
        for (size_t i = 0; i < n; i += 8) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; memcpy(buf + i, &s, n - i < 8 ? n - i : 8); }
        if (write(fd, buf, n) != (ssize_t)n) { perror(tmp); close(fd); unlink(tmp); free(buf); return -1; }
        done += n;
    }
    free(buf); if (close(fd) || rename(tmp, path)) { perror(path); unlink(tmp); return -1; }
    return 0;
}

static volatile sig_atomic_t STOP;
static void on_sig(int s) { (void)s; STOP = 1; }
static const char *opt(int argc, char **argv, const char *n, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], n)) return argv[i + 1]; return d; }
static int flag(int argc, char **argv, const char *n) { for (int i = 1; i < argc; i++) if (!strcmp(argv[i], n)) return 1; return 0; }

int es_sparse_main(int argc, char **argv) {
    const char *xf = opt(argc, argv, "--experts-file", NULL), *data = opt(argc, argv, "--data", NULL), *ck = opt(argc, argv, "--save", NULL);
    L = atoi(opt(argc, argv, "--layers", "0")); E = atoi(opt(argc, argv, "--experts", "0")); D = atoi(opt(argc, argv, "--dim", "0")); F = atoi(opt(argc, argv, "--ffn", "0"));
    const int gc = flag(argc, argv, "--gradcheck");
    if (!xf || (!data && !gc) || L < 1 || E < 1 || D < 4 || F < 4 || D % 4 || F % 4) {
        fprintf(stderr, "usage: es_train moe --sparse [--p6] --layers L --experts E --dim D --ffn F --experts-file FILE --data FILE\n"
                        "         [--batch 8] [--seq 128] [--steps 100] [--eval-every 25] [--save OUT.ckpt] [--rank 4] [--lr 2e-3] [--lora-lr 1e-3]\n"
                        "         [--balance 0.01] [--bits 4|2 (2 = ternary)] [--router-rank 0|RR] [--scale S] [--threads N (default OMP_NUM_THREADS)] [--seed 1] [--exact] [--gradcheck]\n"
                        "  D and F must be multiples of 4 (64 to run in libesd). The experts file is created (seeded random bytes, L*E*D*F*bits/4) when it does not exist.\n");
        return 2;
    }
    P6 = flag(argc, argv, "--p6"); B = atoi(opt(argc, argv, "--batch", "8")); T = atoi(opt(argc, argv, "--seq", "128")); R = atoi(opt(argc, argv, "--rank", "4"));
    const int steps = atoi(opt(argc, argv, "--steps", "100")); int ev = atoi(opt(argc, argv, "--eval-every", "25")); if (ev < 1) ev = 1;
    LR = atof(opt(argc, argv, "--lr", "2e-3")); LRL = atof(opt(argc, argv, "--lora-lr", "1e-3")); AUX = atof(opt(argc, argv, "--balance", "0.01"));
    rs ^= strtoull(opt(argc, argv, "--seed", "1"), 0, 10) * 0x9E3779B97F4A7C15ULL; for (int i = 0; i < 8; i++) xr();
    NTH = atoi(opt(argc, argv, "--threads", getenv("OMP_NUM_THREADS") ? getenv("OMP_NUM_THREADS") : "1")); if (NTH < 1) NTH = 1; if (NTH > 64) NTH = 64;
    RR = atoi(opt(argc, argv, "--router-rank", "0")); if (RR < 0) RR = 0;
    BITS = flag(argc, argv, "--ternary") ? 2 : atoi(opt(argc, argv, "--bits", "4")); if (BITS != 2 && BITS != 4) { fprintf(stderr, "--bits must be 4 (int4) or 2 (ternary)\n"); return 2; }
    SCALE = atof(opt(argc, argv, "--scale", "0")); if (SCALE <= 0) SCALE = BITS == 2 ? ess_default_scale_t(D, F) : ess_default_scale(D, F);
    if (B < 1 || T < 2 || R < 0 || steps < 0) { fprintf(stderr, "bad --batch/--seq/--rank/--steps\n"); return 2; }
    NT = B * T;
    FAST = !flag(argc, argv, "--exact") && D % 16 == 0 && F % 16 == 0; if (FAST) { I8 = i8_pick(&I8N); i8_tables(); I8CS = !strcmp(I8N, "avx-vnni") || !strcmp(I8N, "avx2") || (!strcmp(I8N, "amx") && (D % 64 || F % 64)); }
    memcpy(HD.magic, ESS_MAGIC, 8); HD.version = 1; HD.L = L; HD.E = E; HD.D = D; HD.F = F; HD.R = R; HD.K = 1; HD.T = T; HD.scale = SCALE; HD.lora_scale = LS; HD.reserved[0] = BITS; HD.reserved[1] = RR; EBY = ess_expert_bytes(&HD);

    /* experts file: mmap'd in place (the page cache is the expert cache) */
    const size_t need = ess_file_bytes(&HD); struct stat st;
    if (stat(xf, &st) && (errno != ENOENT || gen_experts(xf, need))) { perror(xf); return 1; }
    if (stat(xf, &st) || (size_t)st.st_size < need) { fprintf(stderr, "%s: %lld bytes, this config needs %zu (L*E*D*F*bits/4)\n", xf, (long long)st.st_size, need); return 1; }
    int fd = open(xf, O_RDONLY); if (fd < 0) { perror(xf); return 1; }
    XF = mmap(NULL, need, PROT_READ, MAP_SHARED, fd, 0); if (XF == MAP_FAILED) { perror("mmap"); return 1; }
    madvise((void *)XF, need, MADV_RANDOM);

    LOE = ess_loe(&HD); DN = ess_dense_n(&HD); PN = ess_params_n(&HD);
    Pp = fa(PN); Pg = fa(PN); Pm = fa(PN); Pv = fa(PN); touched = calloc((size_t)L * E, 1);
    V.mix = malloc(sizeof(float *) * L); V.Wr = malloc(sizeof(float *) * L); V.Wp = malloc(sizeof(float *) * L); G.mix = malloc(sizeof(float *) * L); G.Wr = malloc(sizeof(float *) * L); G.Wp = malloc(sizeof(float *) * L);
    ess_bind(&HD, Pp, &V); ess_bind(&HD, Pg, &G);
    for (size_t i = 0; i < 256 * (size_t)D; i++) { V.emb[i] = nrm(); V.head[i] = 0.02f * nrm(); }
    for (int l = 0; l < L; l++) { for (int c = 0; c < D; c++) V.mix[l][c] = 0.5f; if (RR) { for (size_t i = 0; i < (size_t)RR * D; i++) V.Wp[l][i] = nrm() / sqrtf(D); for (size_t i = 0; i < (size_t)E * RR; i++) V.Wr[l][i] = nrm() / sqrtf(RR); }
        else for (size_t i = 0; i < (size_t)E * D; i++) V.Wr[l][i] = nrm() / sqrtf(D); }
    for (size_t le = 0; le < (size_t)L * E; le++) { float *a = V.lo + le * LOE, *ad = a + (size_t)R * D + (size_t)F * R;
        for (size_t i = 0; i < (size_t)R * D; i++) a[i] = nrm() / sqrtf(D);
        for (size_t i = 0; i < (size_t)R * F; i++) ad[i] = nrm() / sqrtf(F); }   /* B factors start at 0 */

    const size_t TD = (size_t)NT * D; const int na = P6 ? 1 : L; const int R1 = R ? R : 1;
    ACT = calloc(na, sizeof(act_t));
    for (int i = 0; i < na; i++) { act_t *a = &ACT[i]; a->h = fa(TD); a->rh = fa(NT); a->s = fa(TD); a->t = fa((size_t)NT * (RR ? RR : 1)); a->p = fa((size_t)NT * E); a->gate = fa(NT);
        a->u = fa((size_t)NT * F); a->zu = fa((size_t)NT * R1); a->zd = fa((size_t)NT * R1); a->y = fa(TD); a->sel = fa(NT); }
    XIN = malloc(sizeof(float *) * (L + 1)); for (int l = 0; l <= L; l++) XIN[l] = fa(TD);
    DS = fa(TD); DG = fa(NT); DZ = fa((size_t)NT * E); DT = fa((size_t)NT * (RR ? RR : 1)); XS = fa(TD); XF_ = fa(TD); RF = fa(NT); RL = calloc(NT, sizeof(double)); LG = fa((size_t)NT * 256); DX = fa(TD);
    cnt = fa(E); off = fa(E + 1); lst = fa(NT);
    const int nw = P6 ? NTH : 1; WS = calloc(nw, sizeof(ws_t));
    for (int i = 0; i < nw; i++) { ws_t *w = &WS[i]; w->S = fa(TD); w->U = fa((size_t)NT * F); w->A = fa((size_t)NT * F); w->Y = fa(TD); w->dA = fa((size_t)NT * F); w->dS = fa(TD);
        w->Zu = fa((size_t)NT * R1); w->Zd = fa((size_t)NT * R1); w->Tm = fa((size_t)NT * R1); w->T2 = fa((size_t)NT * R1); w->W1 = fa((size_t)D * F); w->W2 = fa((size_t)D * F); w->Wt = fa((size_t)D * F);
        const size_t mx = D > F ? D : F; w->Bq = (int8_t *)fa((size_t)D * F / 4); w->cs = (int32_t *)fa(mx); w->Aq = (int8_t *)fa((size_t)NT * mx / 4 + 1); w->Ci = (int32_t *)fa((size_t)NT * mx);
        w->ar = (int32_t *)fa(NT); w->as = fa(NT); }

    const double total = (double)L * E * 2 * D * F, active = (double)L * 2 * D * F;
    printf("cfg sparse mode=%s L=%d E=%d d=%d f=%d top1 lora_r=%d router_rank=%d | frozen %.3f B params (%.2f GB %s, %s) | active/token %.1f M | trainable %.2f M (dense %.2f M + LoRA %.2f M) | batch %dx%d threads %d | expert matmuls %s\n",
           P6 ? "p6" : "plain", L, E, D, F, R, RR, total / 1e9, need / 1e9, BITS == 2 ? "ternary" : "int4", xf, active / 1e6, PN / 1e6, DN / 1e6, (PN - DN) / 1e6, B, T, P6 ? NTH : 1, FAST ? I8N : "float");
    fflush(stdout);
    uint8_t *tok = malloc((size_t)B * (T + 1));

    if (gc) {   /* finite differences on random text against the analytic gradient, a few entries per group */
        for (size_t i = 0; i < (size_t)B * (T + 1); i++) tok[i] = (uint8_t)(32 + xr() % 95);
        for (size_t le = 0; le < (size_t)L * E; le++) { float *bu = V.lo + le * LOE + (size_t)R * D, *bd = bu + (size_t)F * R + (size_t)R * F;   /* nonzero B so A gets checked too */
            for (size_t i = 0; i < (size_t)F * R; i++) bu[i] = 0.1f * nrm();
            for (size_t i = 0; i < (size_t)D * R; i++) bd[i] = 0.1f * nrm(); }
        AUX = 0; double cosv = 1;
        if (FAST) {   /* the int8 path's gradient against the float one, then the finite differences below check the float one */
            run(tok, 1); float *g8 = malloc(4 * PN); memcpy(g8, Pg, 4 * PN); memset(Pg, 0, 4 * PN); FAST = 0; run(tok, 1);
            double ab = 0, aa = 0, bb = 0; for (size_t i = 0; i < PN; i++) { ab += (double)g8[i] * Pg[i]; aa += (double)g8[i] * g8[i]; bb += (double)Pg[i] * Pg[i]; }
            cosv = ab / sqrt(aa * bb + 1e-30); free(g8); printf("gc int8 (%s) gradient vs float: cosine %.5f\n", I8N, cosv);
        } else run(tok, 1);
        run(tok, 0); const uint64_t sig0 = RSIG; int skip = 0; int bad = 0, le = -1; for (int i = 0; i < L * E && le < 0; i++) if (touched[i]) le = i;
        const size_t lo0 = DN + (size_t)le * LOE, g0[9][2] = {{0, 256 * (size_t)D}, {256 * (size_t)D, 256 * (size_t)D}, {(size_t)(V.mix[0] - Pp), (size_t)D}, {(size_t)(V.Wr[L - 1] - Pp), (size_t)E * (RR ? RR : D)},
            {lo0, (size_t)R * D}, {lo0 + (size_t)R * D, (size_t)F * R}, {lo0 + (size_t)R * D + (size_t)F * R, (size_t)R * F}, {lo0 + (size_t)R * D + 2 * (size_t)F * R, (size_t)D * R}, {RR ? (size_t)(V.Wp[0] - Pp) : 0, (size_t)RR * D + 1}};
        const char *nm[9] = {"emb", "head", "mix0", "WrLast", "Au", "Bu", "Ad", "Bd", "Wp0"}; const int ng = RR ? 9 : 8;
        for (int g = 0; g < ng; g++) for (int k = 0; k < 4; k++) {
            size_t i = g0[g][0] + xr() % g0[g][1]; if (g == 0) i = (size_t)tok[xr() % (T)] * D + xr() % D;
            const float a = Pg[i], o = Pp[i], e = 1e-2f; Pp[i] = o + e; const double lp = run(tok, 0); const uint64_t sp = RSIG; Pp[i] = o - e; const double lm = run(tok, 0); Pp[i] = o;
            if (sp != sig0 || RSIG != sig0) { skip++; printf("gc %-6s [%zu] skipped: the perturbation changes a top-1 route (loss is not differentiable there)\n", nm[g], i); continue; }
            const double num = (lp - lm) / (2 * e), err = fabs(num - a) / fmax(fabs(num) + fabs(a), 1e-4);
            if (err > 0.05) bad++;
            printf("gc %-6s [%zu] analytic %+.6e numeric %+.6e rel %.3g\n", nm[g], i, a, num, err);
        }
        const int ok = bad <= 2 && skip < 2 * ng && cosv > 0.99; printf("gradcheck %s (%d of %d above 5%%, %d skipped for route flips, int8 gradient cosine %.4f)\n", ok ? "PASS" : "FAIL", bad, 4 * ng - skip, skip, cosv); return ok ? 0 : 1;
    }

    FILE *df = fopen(data, "rb"); if (!df) { perror(data); return 1; }
    fseek(df, 0, SEEK_END); const long dn = ftell(df); rewind(df);
    uint8_t *dat = malloc(dn > 0 ? dn : 1); if (dn <= 0 || fread(dat, 1, dn, df) != (size_t)dn) { fprintf(stderr, "%s: cannot read\n", data); return 1; } fclose(df);
    const long split = dn - dn / 10, hn = dn - split;
    if (split < T + 2 || hn < T + 2) { fprintf(stderr, "dataset too small for seq %d: need more than %d bytes in each of the 90%%/10%% parts\n", T, T + 1); return 1; }
    const int EVB = 2;   /* held-out: EVB batches at fixed, evenly spread offsets in the last 10% */
    double ev_loss_v;
#define EVAL() do { double s_ = 0; for (int r_ = 0; r_ < EVB; r_++) { for (int b_ = 0; b_ < B; b_++) { const int i_ = r_ * B + b_, n_ = EVB * B; \
        const long o_ = split + (long)((double)i_ * (hn - T - 1) / (n_ > 1 ? n_ - 1 : 1)); memcpy(tok + b_ * (T + 1), dat + o_, T + 1); } s_ += run(tok, 0); } ev_loss_v = s_ / EVB; } while (0)

    signal(SIGTERM, on_sig); signal(SIGINT, on_sig); signal(SIGXCPU, on_sig);
    EVAL(); const double h0 = ev_loss_v; double h1 = h0;
    printf("eval step 0 heldout_loss %.4f\n", h0); fflush(stdout);
    uint64_t ds = 12345; double tsum = 0, t1 = 0; int tn = 0, done = 0, last_ev = 0;
    for (step_no = 1; step_no <= steps && !STOP; step_no++) {
        for (int b = 0; b < B; b++) { ds = ds * 6364136223846793005ULL + 1442695040888963407ULL; const long o = (long)((ds >> 17) % (uint64_t)(split - T - 1)); memcpy(tok + b * (T + 1), dat + o, T + 1); }
        double by0 = 0; for (int i = 0; i < nw; i++) by0 += WS[i].bytes;
        const double t0 = now(); AUXL = 0; const double loss = run(tok, 1);
        double gn = 0; for (size_t i = 0; i < DN; i++) gn += (double)Pg[i] * Pg[i];
        for (size_t le = 0; le < (size_t)L * E; le++) if (touched[le]) for (size_t i = DN + le * LOE; i < DN + (le + 1) * LOE; i++) gn += (double)Pg[i] * Pg[i];
        gn = sqrt(gn); const float clip = gn > 1 ? (float)(1 / gn) : 1;
        if (clip < 1) { for (size_t i = 0; i < DN; i++) Pg[i] *= clip; for (size_t le = 0; le < (size_t)L * E; le++) if (touched[le]) for (size_t i = DN + le * LOE; i < DN + (le + 1) * LOE; i++) Pg[i] *= clip; }
        adam(0, DN, LR); int nt_ = 0;
        for (size_t le = 0; le < (size_t)L * E; le++) if (touched[le]) { adam(DN + le * LOE, LOE, LRL); touched[le] = 0; nt_++; }   /* lazy: only experts that saw tokens */
        const double dt = now() - t0; if (step_no == 1) t1 = dt; else { tsum += dt; tn++; }
        double by = -by0; for (int i = 0; i < nw; i++) by += WS[i].bytes;
        struct rusage ru; getrusage(RUSAGE_SELF, &ru);
        printf("step %d train_ce %.4f aux %.4f |g| %.3f train_tok/s %.1f step_s %.2f experts_touched %d experts_read_GB %.3f rss_MB %ld\n",
               step_no, loss, AUXL / L, gn, NT / dt, dt, nt_, by / 1e9, ru.ru_maxrss / 1024);
        fflush(stdout); done = step_no;
        if (step_no % ev == 0 && step_no < steps && !STOP) { EVAL(); h1 = ev_loss_v; last_ev = step_no; printf("eval step %d heldout_loss %.4f\n", step_no, h1); fflush(stdout); if (ck) save(ck, done); }
    }
    if (STOP) printf("[stop requested: saving after step %d]\n", done);
    if (ck && save(ck, done) == 0) printf("saved %s (%d steps, %.1f MB)\n", ck, done, (sizeof HD + 4.0 * PN) / 1e6);
    if (last_ev != done || done == 0) { EVAL(); h1 = ev_loss_v; printf("eval step %d heldout_loss %.4f\n", done, h1); }
    const double tps = tn ? NT * tn / tsum : t1 > 0 ? NT / t1 : 0;
    printf("RESULT steps=%d train_tok/s=%.1f heldout_before=%.4f heldout_after=%.4f mode=%s ckpt=%s\n", done, tps, h0, h1, P6 ? "p6" : "plain", ck ? ck : "-");
    fflush(stdout); return 0;
}
