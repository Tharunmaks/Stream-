/* es_moe: prototype trainer for a byte-level MoE transformer whose experts are frozen ternary (2-bit packed) weights
 * streamed from disk through an LRU cache under a RAM budget; trainable: embeddings, attention, router, per-expert LoRA.
 * Matmuls go through OpenBLAS (dlopen, 1 thread). Single file, local prototype.
 *   es_moe gen  FILE L E d f [seed]          write random ternary experts (2 bits/weight)
 *   es_moe train FILE L E d f k r steps cacheMB data [seqs seq]   */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

typedef void (*sg64_t)(int, int, int, long, long, long, float, const float *, long, const float *, long, float, float *, long);
static sg64_t SG; static double FLOPS;
static void mm(int ta, int tb, int M, int N, int K, float al, const float *A, int lda, const float *B, int ldb, float be, float *C, int ldc) {
    if (M <= 0 || N <= 0 || K <= 0) { if (be == 0 && M > 0 && N > 0) for (int i = 0; i < M; i++) memset(C + (size_t)i * ldc, 0, 4 * (size_t)N); return; }
    SG(101, ta ? 112 : 111, tb ? 112 : 111, M, N, K, al, A, lda, B, ldb, be, C, ldc); FLOPS += 2.0 * M * N * K;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ULL;
static uint64_t xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
static float nrm(void) { double u = (xr() >> 11) * (1.0 / 9007199254740992.0) + 1e-12, v = (xr() >> 11) * (1.0 / 9007199254740992.0); return (float)(sqrt(-2 * log(u)) * cos(6.283185307 * v)); }
static float *fa(size_t n) { float *p = calloc(n ? n : 1, 4); if (!p) { fprintf(stderr, "oom %zu\n", n); exit(1); } return p; }

/* ---- config / expert store */
static int L, E, D, F, K, R, H = 16, HD, S, T, NT; static float LS = 1.0f;
static size_t EB;            /* packed bytes per expert: Wu (F x D) then Wd (D x F), 4 weights per byte */
static int FD; static float LUTu[256][4], LUTd[256][4];
typedef struct { int key; uint64_t used; uint8_t *p; } slot;
static slot *SL; static int NSL, *where; static uint64_t tick; static double io_s, io_bytes, hits, miss;
static const uint8_t *expert(int l, int e) {
    const int key = l * E + e; tick++;
    if (where[key] >= 0) { hits++; SL[where[key]].used = tick; return SL[where[key]].p; }
    miss++; int v = 0; for (int i = 1; i < NSL; i++) if (SL[i].used < SL[v].used) v = i;
    if (SL[v].key >= 0) where[SL[v].key] = -1;
    const double t0 = now(); const off_t off = (off_t)key * EB; size_t got = 0;
    while (got < EB) { ssize_t n = pread(FD, SL[v].p + got, EB - got, off + got); if (n <= 0) { perror("pread"); exit(1); } got += n; }
    posix_fadvise(FD, off, EB, POSIX_FADV_DONTNEED);   /* keep the page cache from hiding the disk */
    io_s += now() - t0; io_bytes += EB;
    SL[v].key = key; SL[v].used = tick; where[key] = v; return SL[v].p;
}
static void decode(const uint8_t *p, size_t n, float *dst, float (*lut)[4]) { for (size_t i = 0; i < n / 4; i++) memcpy(dst + 4 * i, lut[p[i]], 16); }

/* ---- params */
typedef struct { float *p, *g, *m, *v; size_t n; } prm;
static prm P; static float *emb, **Wq, **Wk, **Wv, **Wo, **Wr, *lo; static float *gemb, **gWq, **gWk, **gWv, **gWo, **gWr, *glo;
static size_t LOE; static uint8_t *touched;   /* per (l,e) LoRA block: Au r x D, Bu F x r, Ad r x F, Bd D x r */
#define AU(l,e) (lo + ((size_t)(l) * E + (e)) * LOE)
#define GAU(l,e) (glo + ((size_t)(l) * E + (e)) * LOE)

/* ---- activations saved per layer */
typedef struct { float *xin, *a, *ra, *q, *k, *v, *Pr, *att, *xmid, *b, *rb, *gp, *pf, *u, *y, *zu, *zd; int *sel; } lay;
static lay *A; static float *xf, *rf, *logit;
static int *cnt, *lst;   /* per expert: token-slot lists */

static float rms(float *o, const float *x, int n) { double s = 0; for (int i = 0; i < n; i++) s += (double)x[i] * x[i]; const float r = (float)(1 / sqrt(s / n + 1e-6)); for (int i = 0; i < n; i++) o[i] = x[i] * r; return r; }
static void rms_bwd(float *dx, const float *a, float r, const float *da, int n) { double s = 0; for (int i = 0; i < n; i++) s += (double)da[i] * a[i]; const float c = (float)(s / n); for (int i = 0; i < n; i++) dx[i] += r * (da[i] - a[i] * c); }
static float silu(float u) { return u / (1 + expf(-u)); }
static float *Xg, *Ug, *Hg, *Yg, *Zu, *Zd, *Tm, *T2, *dXg, *Wu, *Wd, *SS, *dP;

static int RANDR; static void route(int l) {
    lay *y = &A[l]; float *lg = Tm;
    mm(0, 1, NT, E, D, 1, y->b, D, Wr[l], D, 0, lg, E);
    memset(cnt, 0, sizeof(int) * E);
    for (int t = 0; t < NT; t++) {
        float *z = lg + (size_t)t * E; int *s = y->sel + (size_t)t * K; float *g = y->gp + (size_t)t * K;
        if (RANDR) { for (int j = 0; j < K; j++) { int e; do { e = (int)(xr() % E); int u = 0; for (int i = 0; i < j; i++) u |= s[i] == e; if (!u) break; } while (1); s[j] = e; } }
        else for (int j = 0; j < K; j++) { int bi = -1; for (int e = 0; e < E; e++) { int u = 0; for (int i = 0; i < j; i++) u |= s[i] == e; if (!u && (bi < 0 || z[e] > z[bi])) bi = e; } s[j] = bi; }
        float mx = z[s[0]], sum = 0; for (int j = 0; j < K; j++) { g[j] = expf(z[s[j]] - mx); sum += g[j]; } for (int j = 0; j < K; j++) g[j] /= sum;
        for (int j = 0; j < K; j++) lst[(size_t)s[j] * NT * K + cnt[s[j]]++] = t * K + j;
        { float *pq = y->pf + (size_t)t * E, m2 = -INFINITY; double sm = 0; for (int e = 0; e < E; e++) if (z[e] > m2) m2 = z[e]; for (int e = 0; e < E; e++) { pq[e] = expf(z[e] - m2); sm += pq[e]; } for (int e = 0; e < E; e++) pq[e] /= sm; }
    }
}
static void layer_fwd(int l, float *x) {
    lay *y = &A[l]; const float sc = 1 / sqrtf((float)HD);
    memcpy(y->xin, x, 4 * (size_t)NT * D);
    for (int t = 0; t < NT; t++) y->ra[t] = rms(y->a + (size_t)t * D, x + (size_t)t * D, D);
    mm(0, 1, NT, D, D, 1, y->a, D, Wq[l], D, 0, y->q, D); mm(0, 1, NT, D, D, 1, y->a, D, Wk[l], D, 0, y->k, D); mm(0, 1, NT, D, D, 1, y->a, D, Wv[l], D, 0, y->v, D);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        const size_t o = (size_t)s * T * D + (size_t)h * HD; float *Pm = y->Pr + ((size_t)s * H + h) * T * T;
        mm(0, 1, T, T, HD, sc, y->q + o, D, y->k + o, D, 0, Pm, T);
        for (int i = 0; i < T; i++) { float *r = Pm + (size_t)i * T, mx = -INFINITY, sum = 0; for (int j = 0; j <= i; j++) if (r[j] > mx) mx = r[j]; for (int j = 0; j <= i; j++) { r[j] = expf(r[j] - mx); sum += r[j]; } for (int j = 0; j <= i; j++) r[j] /= sum; for (int j = i + 1; j < T; j++) r[j] = 0; }
        mm(0, 0, T, HD, T, 1, Pm, T, y->v + o, D, 0, y->att + o, D);
    }
    memcpy(y->xmid, x, 4 * (size_t)NT * D); mm(0, 1, NT, D, D, 1, y->att, D, Wo[l], D, 1, y->xmid, D);
    memcpy(x, y->xmid, 4 * (size_t)NT * D);
    for (int t = 0; t < NT; t++) y->rb[t] = rms(y->b + (size_t)t * D, y->xmid + (size_t)t * D, D);
    route(l);
    for (int e = 0; e < E; e++) {
        const int n = cnt[e]; if (!n) continue; const int *li = lst + (size_t)e * NT * K;
        const uint8_t *p = expert(l, e); decode(p, (size_t)F * D, Wu, LUTu); decode(p + (size_t)F * D / 4, (size_t)D * F, Wd, LUTd);
        const float *Au = AU(l, e), *Bu = Au + (size_t)R * D, *Ad = Bu + (size_t)F * R, *Bd = Ad + (size_t)R * F;
        for (int i = 0; i < n; i++) memcpy(Xg + (size_t)i * D, y->b + (size_t)(li[i] / K) * D, 4 * (size_t)D);
        mm(0, 1, n, F, D, 1, Xg, D, Wu, D, 0, Ug, F);
        if (R) { mm(0, 1, n, R, D, 1, Xg, D, Au, D, 0, Zu, R); mm(0, 1, n, F, R, LS, Zu, R, Bu, R, 1, Ug, F); }
        for (size_t i = 0; i < (size_t)n * F; i++) Hg[i] = silu(Ug[i]);
        mm(0, 1, n, D, F, 1, Hg, F, Wd, F, 0, Yg, D);
        if (R) { mm(0, 1, n, R, F, 1, Hg, F, Ad, F, 0, Zd, R); mm(0, 1, n, D, R, LS, Zd, R, Bd, R, 1, Yg, D); }
        for (int i = 0; i < n; i++) {
            const int ts = li[i], t = ts / K; const float g = y->gp[ts];
            memcpy(y->u + (size_t)ts * F, Ug + (size_t)i * F, 4 * (size_t)F); memcpy(y->y + (size_t)ts * D, Yg + (size_t)i * D, 4 * (size_t)D);
            if (R) { memcpy(y->zu + (size_t)ts * R, Zu + (size_t)i * R, 4 * R); memcpy(y->zd + (size_t)ts * R, Zd + (size_t)i * R, 4 * R); }
            float *xo = x + (size_t)t * D; const float *yy = Yg + (size_t)i * D; for (int c = 0; c < D; c++) xo[c] += g * yy[c];
        }
    }
}
static float AUX = 0.01f; static double AUXL;
static float *db, *dg, *da, *dq, *dk, *dv, *datt;
static void layer_bwd(int l, float *dx) {
    lay *y = &A[l]; const float sc = 1 / sqrtf((float)HD);
    memset(db, 0, 4 * (size_t)NT * D); memset(dg, 0, 4 * (size_t)NT * K);
    /* rebuild the expert lists of this layer */
    memset(cnt, 0, sizeof(int) * E);
    for (int ts = 0; ts < NT * K; ts++) { const int e = y->sel[ts]; lst[(size_t)e * NT * K + cnt[e]++] = ts; }
    for (int e = E - 1; e >= 0; e--) {
        const int n = cnt[e]; if (!n) continue; const int *li = lst + (size_t)e * NT * K;
        const uint8_t *p = expert(l, e); decode(p, (size_t)F * D, Wu, LUTu); decode(p + (size_t)F * D / 4, (size_t)D * F, Wd, LUTd);
        float *Au = AU(l, e), *Bu = Au + (size_t)R * D, *Ad = Bu + (size_t)F * R, *Bd = Ad + (size_t)R * F;
        float *gAu = GAU(l, e), *gBu = gAu + (size_t)R * D, *gAd = gBu + (size_t)F * R, *gBd = gAd + (size_t)R * F; touched[l * E + e] = 1;
        for (int i = 0; i < n; i++) {
            const int ts = li[i], t = ts / K; const float g = y->gp[ts]; const float *dxt = dx + (size_t)t * D, *yy = y->y + (size_t)ts * D;
            float *o = Yg + (size_t)i * D; double s = 0; for (int c = 0; c < D; c++) { o[c] = g * dxt[c]; s += (double)dxt[c] * yy[c]; } dg[ts] = (float)s;
            memcpy(Xg + (size_t)i * D, y->b + (size_t)t * D, 4 * (size_t)D); memcpy(Ug + (size_t)i * F, y->u + (size_t)ts * F, 4 * (size_t)F);
            if (R) { memcpy(Zu + (size_t)i * R, y->zu + (size_t)ts * R, 4 * R); memcpy(Zd + (size_t)i * R, y->zd + (size_t)ts * R, 4 * R); }
        }
        for (size_t i = 0; i < (size_t)n * F; i++) Hg[i] = silu(Ug[i]);
        float *dH = SS; mm(0, 0, n, F, D, 1, Yg, D, Wd, F, 0, dH, F);
        if (R) { mm(0, 0, n, R, D, 1, Yg, D, Bd, R, 0, Tm, R); mm(0, 0, n, F, R, LS, Tm, R, Ad, F, 1, dH, F);
                 mm(1, 0, D, R, n, LS, Yg, D, Zd, R, 1, gBd, R); mm(1, 0, R, F, n, LS, Tm, R, Hg, F, 1, gAd, F); }
        for (size_t i = 0; i < (size_t)n * F; i++) { const float u = Ug[i], s = 1 / (1 + expf(-u)); dH[i] *= s * (1 + u * (1 - s)); }
        mm(0, 0, n, D, F, 1, dH, F, Wu, D, 0, dXg, D);
        if (R) { mm(0, 0, n, R, F, 1, dH, F, Bu, R, 0, T2, R); mm(0, 0, n, D, R, LS, T2, R, Au, D, 1, dXg, D);
                 mm(1, 0, F, R, n, LS, dH, F, Zu, R, 1, gBu, R); mm(1, 0, R, D, n, LS, T2, R, Xg, D, 1, gAu, D); }
        for (int i = 0; i < n; i++) { float *o = db + (size_t)(li[i] / K) * D; const float *s = dXg + (size_t)i * D; for (int c = 0; c < D; c++) o[c] += s[c]; }
    }
    /* router: top-k renormalised softmax */
    float *dl = Tm; memset(dl, 0, 4 * (size_t)NT * E);
    for (int t = 0; t < NT; t++) { const float *g = y->gp + (size_t)t * K, *q = dg + (size_t)t * K; float m = 0; for (int j = 0; j < K; j++) m += g[j] * q[j]; for (int j = 0; j < K; j++) dl[(size_t)t * E + y->sel[t * K + j]] = g[j] * (q[j] - m); }
    if (AUX > 0) {   /* Switch-style balance loss AUX * E * sum_e f_e * mean_t p_te, f_e = routed fraction (constant) */
        for (int e = 0; e < E; e++) Tm[(size_t)NT * E + e] = AUX * E * ((float)cnt[e] / (NT * K)) / NT;
        const float *dPe = Tm + (size_t)NT * E; double bal = 0;
        for (int t = 0; t < NT; t++) { const float *pq = y->pf + (size_t)t * E; double m = 0; for (int e = 0; e < E; e++) m += pq[e] * dPe[e]; for (int e = 0; e < E; e++) dl[(size_t)t * E + e] += pq[e] * (dPe[e] - (float)m); }
        for (int e = 0; e < E; e++) { double pm = 0; for (int t = 0; t < NT; t++) pm += y->pf[(size_t)t * E + e]; bal += ((double)cnt[e] / (NT * K)) * pm / NT; } AUXL += AUX * E * bal;
    }
    mm(1, 0, E, D, NT, 1, dl, E, y->b, D, 1, gWr[l], D); mm(0, 0, NT, D, E, 1, dl, E, Wr[l], D, 1, db, D);
    for (int t = 0; t < NT; t++) rms_bwd(dx + (size_t)t * D, y->b + (size_t)t * D, y->rb[t], db + (size_t)t * D, D);
    /* attention */
    mm(0, 0, NT, D, D, 1, dx, D, Wo[l], D, 0, datt, D); mm(1, 0, D, D, NT, 1, dx, D, y->att, D, 1, gWo[l], D);
    for (int s = 0; s < S; s++) for (int h = 0; h < H; h++) {
        const size_t o = (size_t)s * T * D + (size_t)h * HD; const float *Pm = y->Pr + ((size_t)s * H + h) * T * T;
        mm(0, 1, T, T, HD, 1, datt + o, D, y->v + o, D, 0, dP, T);
        for (int i = 0; i < T; i++) { float *r = dP + (size_t)i * T; const float *pr = Pm + (size_t)i * T; double Dd = 0; for (int j = 0; j <= i; j++) Dd += pr[j] * r[j]; for (int j = 0; j < T; j++) r[j] = j <= i ? pr[j] * (r[j] - (float)Dd) * sc : 0; }
        mm(1, 0, T, HD, T, 1, Pm, T, datt + o, D, 0, dv + o, D);
        mm(0, 0, T, HD, T, 1, dP, T, y->k + o, D, 0, dq + o, D);
        mm(1, 0, T, HD, T, 1, dP, T, y->q + o, D, 0, dk + o, D);
    }
    mm(0, 0, NT, D, D, 1, dq, D, Wq[l], D, 0, da, D); mm(0, 0, NT, D, D, 1, dk, D, Wk[l], D, 1, da, D); mm(0, 0, NT, D, D, 1, dv, D, Wv[l], D, 1, da, D);
    mm(1, 0, D, D, NT, 1, dq, D, y->a, D, 1, gWq[l], D); mm(1, 0, D, D, NT, 1, dk, D, y->a, D, 1, gWk[l], D); mm(1, 0, D, D, NT, 1, dv, D, y->a, D, 1, gWv[l], D);
    for (int t = 0; t < NT; t++) rms_bwd(dx + (size_t)t * D, y->a + (size_t)t * D, y->ra[t], da + (size_t)t * D, D);
}
static float *X, *pe;
static double run(const uint8_t *tok, int backward) {   /* tok: S x (T+1) */
    for (int s = 0; s < S; s++) for (int t = 0; t < T; t++) memcpy(X + ((size_t)s * T + t) * D, emb + (size_t)tok[s * (T + 1) + t] * D, 4 * (size_t)D);
    for (size_t i = 0; i < (size_t)NT * D; i++) X[i] += pe[i % ((size_t)T * D)];
    for (int l = 0; l < L; l++) layer_fwd(l, X);
    for (int t = 0; t < NT; t++) rf[t] = rms(xf + (size_t)t * D, X + (size_t)t * D, D);
    mm(0, 1, NT, 256, D, 1, xf, D, emb, D, 0, logit, 256);
    double loss = 0;
    for (int s = 0; s < S; s++) for (int t = 0; t < T; t++) {
        float *z = logit + ((size_t)s * T + t) * 256, mx = -INFINITY; double sum = 0; const int y = tok[s * (T + 1) + t + 1];
        for (int c = 0; c < 256; c++) if (z[c] > mx) mx = z[c];
        for (int c = 0; c < 256; c++) sum += exp(z[c] - mx);
        loss += log(sum) + mx - z[y];
        for (int c = 0; c < 256; c++) z[c] = (float)(exp(z[c] - mx) / sum) / NT; z[y] -= 1.0f / NT;
    }
    if (!backward) return loss / NT;
    float *dx = fa((size_t)NT * D), *dxf = fa((size_t)NT * D);
    mm(0, 0, NT, D, 256, 1, logit, 256, emb, D, 0, dxf, D); mm(1, 0, 256, D, NT, 1, logit, 256, xf, D, 1, gemb, D);
    for (int t = 0; t < NT; t++) rms_bwd(dx + (size_t)t * D, xf + (size_t)t * D, rf[t], dxf + (size_t)t * D, D);
    for (int l = L - 1; l >= 0; l--) layer_bwd(l, dx);
    for (int s = 0; s < S; s++) for (int t = 0; t < T; t++) { float *g = gemb + (size_t)tok[s * (T + 1) + t] * D; const float *d = dx + ((size_t)s * T + t) * D; for (int c = 0; c < D; c++) g[c] += d[c]; }
    free(dx); free(dxf); return loss / NT;
}
static int step_no;
static void adam_range(size_t a, size_t n, float lr) {
    const float b1 = 0.9f, b2 = 0.999f, c1 = 1 - powf(b1, step_no), c2 = 1 - powf(b2, step_no);
    for (size_t i = a; i < a + n; i++) { const float g = P.g[i]; P.m[i] = b1 * P.m[i] + (1 - b1) * g; P.v[i] = b2 * P.v[i] + (1 - b2) * g * g; P.p[i] -= lr * (P.m[i] / c1) / (sqrtf(P.v[i] / c2) + 1e-8f); P.g[i] = 0; }
}
int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: see header\n"); return 2; }
    L = atoi(argv[3]); E = atoi(argv[4]); D = atoi(argv[5]); F = atoi(argv[6]); EB = (size_t)2 * D * F / 4;
    if (!strcmp(argv[1], "gen")) {
        rs ^= argc > 7 ? strtoull(argv[7], 0, 10) : 1; int fd = open(argv[2], O_WRONLY | O_CREAT | O_TRUNC, 0644); uint8_t *b = malloc(EB);
        for (long i = 0; i < (long)L * E; i++) { for (size_t j = 0; j < EB; j += 8) { uint64_t w = 0, r = xr(); for (int q = 0; q < 32; q++) { uint64_t c = (r >> (2 * q)) & 3; w |= (c == 3 ? 0 : c) << (2 * q); } memcpy(b + j, &w, 8); } if (write(fd, b, EB) != (ssize_t)EB) { perror("write"); return 1; } }
        close(fd); printf("wrote %ld experts x %zu bytes = %.2f GB, %.3f B frozen params\n", (long)L * E, EB, (double)L * E * EB / 1e9, (double)L * E * 2 * D * F / 1e9); return 0;
    }
    RANDR = getenv("ES_RANDROUTE") != NULL;   /* stress: uniform random top-k routing (worst case for the cache) */
    if (getenv("ES_AUX")) AUX = atof(getenv("ES_AUX"));
    K = atoi(argv[7]); R = atoi(argv[8]); const int steps = atoi(argv[9]); const double cmb = atof(argv[10]); const char *data = argv[11];
    S = argc > 12 ? atoi(argv[12]) : 4; T = argc > 13 ? atoi(argv[13]) : 128; NT = S * T; HD = D / H;
    void *h = dlopen(getenv("ES_BLAS"), RTLD_NOW); if (!h) { fprintf(stderr, "set ES_BLAS\n"); return 1; }
    SG = (sg64_t)dlsym(h, "scipy_cblas_sgemm64_"); void (*nt)(long) = (void (*)(long))dlsym(h, "scipy_openblas_set_num_threads64_"); if (nt) nt(1);
    FD = open(argv[2], O_RDONLY); if (FD < 0) { perror(argv[2]); return 1; }
    const float gu = sqrtf(1.5f / D), gd = sqrtf(1.5f / F) / sqrtf(2.0f * L);
    for (int b = 0; b < 256; b++) for (int q = 0; q < 4; q++) { const int c = (b >> (2 * q)) & 3; const float v = c == 1 ? 1 : c == 2 ? -1 : 0; LUTu[b][q] = v * gu; LUTd[b][q] = v * gd; }
    NSL = (int)(cmb * 1048576 / EB); if (NSL < 1) NSL = 1; if (NSL > L * E) NSL = L * E;
    SL = calloc(NSL, sizeof(slot)); for (int i = 0; i < NSL; i++) { SL[i].key = -1; SL[i].p = malloc(EB); memset(SL[i].p, 0, EB); }
    where = malloc(sizeof(int) * L * E); for (int i = 0; i < L * E; i++) where[i] = -1;
    LOE = (size_t)R * D + (size_t)F * R + (size_t)R * F + (size_t)D * R;
    const size_t nd = 256 * (size_t)D + (size_t)L * (4 * (size_t)D * D + (size_t)E * D), nl = (size_t)L * E * LOE;
    P.n = nd + nl; P.p = fa(P.n); P.g = fa(P.n); P.m = fa(P.n); P.v = fa(P.n); touched = calloc(L * E, 1);
    Wq = malloc(8 * L); Wk = malloc(8 * L); Wv = malloc(8 * L); Wo = malloc(8 * L); Wr = malloc(8 * L); gWq = malloc(8 * L); gWk = malloc(8 * L); gWv = malloc(8 * L); gWo = malloc(8 * L); gWr = malloc(8 * L);
    { size_t o = 0; emb = P.p; gemb = P.g; o += 256 * (size_t)D;
      for (int l = 0; l < L; l++) { Wq[l] = P.p + o; gWq[l] = P.g + o; o += (size_t)D * D; Wk[l] = P.p + o; gWk[l] = P.g + o; o += (size_t)D * D; Wv[l] = P.p + o; gWv[l] = P.g + o; o += (size_t)D * D; Wo[l] = P.p + o; gWo[l] = P.g + o; o += (size_t)D * D; Wr[l] = P.p + o; gWr[l] = P.g + o; o += (size_t)E * D; }
      lo = P.p + o; glo = P.g + o; }
    for (size_t i = 0; i < 256 * (size_t)D; i++) emb[i] = 0.02f * nrm();
    for (int l = 0; l < L; l++) { for (size_t i = 0; i < (size_t)D * D; i++) { Wq[l][i] = nrm() / sqrtf(D); Wk[l][i] = nrm() / sqrtf(D); Wv[l][i] = nrm() / sqrtf(D); Wo[l][i] = nrm() / sqrtf(D) / sqrtf(2.0f * L); } for (size_t i = 0; i < (size_t)E * D; i++) Wr[l][i] = 0.02f * nrm(); }
    for (int le = 0; le < L * E; le++) { float *a = lo + le * LOE; for (size_t i = 0; i < (size_t)R * D; i++) a[i] = nrm() / sqrtf(D); float *ad = a + (size_t)R * D + (size_t)F * R; for (size_t i = 0; i < (size_t)R * F; i++) ad[i] = nrm() / sqrtf(F); }
    A = calloc(L, sizeof(lay)); const size_t TD = (size_t)NT * D;
    for (int l = 0; l < L; l++) { lay *y = &A[l]; y->xin = fa(TD); y->a = fa(TD); y->ra = fa(NT); y->q = fa(TD); y->k = fa(TD); y->v = fa(TD); y->Pr = fa((size_t)S * H * T * T); y->att = fa(TD); y->xmid = fa(TD); y->b = fa(TD); y->rb = fa(NT);
        y->gp = fa((size_t)NT * K); y->pf = fa((size_t)NT * E); y->sel = malloc(sizeof(int) * NT * K); y->u = fa((size_t)NT * K * F); y->y = fa((size_t)NT * K * D); y->zu = fa((size_t)NT * K * (R ? R : 1)); y->zd = fa((size_t)NT * K * (R ? R : 1)); }
    xf = fa(TD); rf = fa(NT); logit = fa((size_t)NT * 256); X = fa(TD); pe = fa((size_t)T * D);
    for (int t = 0; t < T; t++) for (int i = 0; i < D; i += 2) { const double w = pow(10000, -(double)i / D); pe[t * D + i] = 0.1f * (float)sin(t * w); pe[t * D + i + 1] = 0.1f * (float)cos(t * w); }
    cnt = malloc(sizeof(int) * E); lst = malloc(sizeof(int) * (size_t)E * NT * K);
    const size_t NK = (size_t)NT * K; Xg = fa(NK * D); Ug = fa(NK * F); Hg = fa(NK * F); Yg = fa(NK * D); Zu = fa(NK * (R ? R : 1)); Zd = fa(NK * (R ? R : 1)); dXg = fa(NK * D); SS = fa(NK * F);
    Tm = fa((size_t)NT * (E > F ? E : F) + NK * F); T2 = fa(NK * (R ? R : 1)); Wu = fa((size_t)F * D); Wd = fa((size_t)D * F); dP = fa((size_t)T * T);
    db = fa(TD); dg = fa(NK); da = fa(TD); dq = fa(TD); dk = fa(TD); dv = fa(TD); datt = fa(TD);
    FILE *df = fopen(data, "rb"); fseek(df, 0, SEEK_END); long dn = ftell(df); rewind(df); uint8_t *dat = malloc(dn); if (fread(dat, 1, dn, df) != (size_t)dn) return 1; fclose(df);
    const long split = dn * 9 / 10; uint8_t *tok = malloc((size_t)S * (T + 1));
    const double act = 256.0 * D + L * (4.0 * D * D + (double)E * D + K * (2.0 * D * F + (double)R * 2 * (D + F)));
    printf("cfg L=%d E=%d d=%d f=%d k=%d r=%d | frozen expert params %.3f B (%.2f GB ternary on disk) | trainable %.1f M (dense %.1f M + expert LoRA %.1f M) | active/token %.1f M | cache %d slots (%.0f MB, %.1f%% of experts)\n",
           L, E, D, F, K, R, (double)L * E * 2 * D * F / 1e9, (double)L * E * EB / 1e9, P.n / 1e6, nd / 1e6, nl / 1e6, act / 1e6, NSL, NSL * EB / 1048576.0, 100.0 * NSL / (L * E));
    fflush(stdout);
    if (getenv("ES_GC")) {   /* finite-difference check of a few parameters in each group */
        for (int s = 0; s < S; s++) memcpy(tok + s * (T + 1), dat + s * 500, T + 1);
        run(tok, 1); const size_t lo_o = (size_t)(lo - P.p), grp[6][2] = {{0, 256 * (size_t)D}, {256 * (size_t)D, (size_t)D * D}, {256 * (size_t)D + 3 * (size_t)D * D, (size_t)D * D}, {256 * (size_t)D + 4 * (size_t)D * D, (size_t)E * D}, {lo_o, LOE * (size_t)E}, {lo_o + (size_t)R * D, (size_t)F * R}};
        const char *nm[6] = {"emb", "Wq0", "Wo0", "Wr0", "Au0", "Bu0"};
        for (int g = 0; g < 6; g++) for (int q = 0; q < (g == 3 ? 40 : 3); q++) {
            size_t i = grp[g][0] + xr() % grp[g][1]; const float a = P.g[i], o = P.p[i], e = getenv("ES_GCE") ? atof(getenv("ES_GCE")) : 1e-2f;
            P.p[i] = o + e; double lp = run(tok, 0); P.p[i] = o - e; double lm = run(tok, 0); P.p[i] = o;
            printf("gc %s[%zu] analytic %.6e numeric %.6e\n", nm[g], i, a, (lp - lm) / (2 * e));
        }
        return 0;
    }
    uint64_t ds = 12345; double tsum = 0; int tn = 0;
    for (step_no = 1; step_no <= steps; step_no++) {
        for (int s = 0; s < S; s++) { ds = ds * 6364136223846793005ULL + 1442695040888963407ULL; const long o = (long)((ds >> 17) % (uint64_t)(split - T - 2)); memcpy(tok + s * (T + 1), dat + o, T + 1); }
        const double t0 = now(), io0 = io_s, b0 = io_bytes, h0 = hits, m0 = miss, f0 = FLOPS;
        AUXL = 0; const double loss = run(tok, 1);
        double gn = 0; for (size_t i = 0; i < P.n; i++) gn += (double)P.g[i] * P.g[i]; gn = sqrt(gn); const float clip = gn > 1 ? (float)(1 / gn) : 1; for (size_t i = 0; i < P.n; i++) P.g[i] *= clip;
        adam_range(0, nd, 1e-3f); int nt_ = 0; for (int le = 0; le < L * E; le++) if (touched[le]) { adam_range(nd + le * LOE, LOE, 1e-3f); touched[le] = 0; nt_++; }   /* lazy: only experts that saw tokens */
        const double dt = now() - t0; if (step_no > 1) { tsum += dt; tn++; }
        struct rusage ru; getrusage(RUSAGE_SELF, &ru);
        printf("step %d loss %.4f aux %.3f |g| %.3f %.2fs %.1f tok/s io %.2fs %.1f MB (%.0f MB/s) hit %.1f%% experts-touched %d GFLOP %.1f (%.1f GFLOP/s) rss %ld MB\n", step_no, loss, AUXL / L, gn, dt, NT / dt, io_s - io0, (io_bytes - b0) / 1e6,
               io_s - io0 > 0 ? (io_bytes - b0) / 1e6 / (io_s - io0) : 0, 100 * (hits - h0) / (hits - h0 + miss - m0), nt_, (FLOPS - f0) / 1e9, (FLOPS - f0) / 1e9 / dt, ru.ru_maxrss / 1024);
        fflush(stdout);
    }
    double hl = 0; int hn = 4; for (int r = 0; r < hn; r++) { for (int s = 0; s < S; s++) { const long o = split + (long)((r * S + s) * 997 % (dn - split - T - 2)); memcpy(tok + s * (T + 1), dat + o, T + 1); } hl += run(tok, 0); }
    printf("held-out loss %.4f | mean step %.2fs = %.1f tok/s (steps 2..)\n", hl / hn, tsum / tn, tn ? NT * tn / tsum : 0);
    return 0;
}
