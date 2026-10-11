/* Flash Experts native kernels (compiled on the device the first time they are needed; see native.py).
 *
 * One expert = one page: magic u32 | pad u32 | scale f32[2mh] | acc f32[2mh] | int8 up[mh*d] | int8 down[mh*d] | padding
 * fx_forward : out = down^T silu(up x) for every (token, expert) pair, reading the 8-bit page directly
 * fx_backward: per expert, gradients of all its pairs, then row-wise Adagrad, 8-bit stochastic rounding and the page written back
 * The same maths as flashexperts.py (the NumPy reference); scripts/test_flash_experts.py checks they agree.
 * Callers split work into disjoint expert groups, so threads never touch the same page. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAGIC 0xE5E50001u

static inline float silu(float z) { return z / (1.0f + expf(-z)); }
static inline float dsilu(float z) { float s = 1.0f / (1.0f + expf(-z)); return s * (1.0f + z * (1.0f - s)); }

static inline uint64_t pool_row(uint64_t e, uint64_t u, uint64_t seed, uint64_t pool_n) {
    uint64_t h = (e * 0x9E3779B97F4A7C15ull + u * 0xC2B2AE3D27D4EB4Full + seed) * 0xFF51AFD7ED558CCDull;
    h ^= h >> 29;
    return h % pool_n;
}

/* dequantise one expert (or build a never-used one); returns 1 if it was never used */
static int load_expert(const uint8_t *base, int64_t page, int64_t e, int d, int mh, const float *pool, int64_t pool_n, uint64_t seed,
                       float *up, float *dn, float *acc) {
    const uint8_t *p = base + e * page;
    uint32_t magic; memcpy(&magic, p, 4);
    if (magic != MAGIC) {
        for (int u = 0; u < mh; u++) memcpy(up + (size_t)u * d, pool + pool_row((uint64_t)e, (uint64_t)u, seed, (uint64_t)pool_n) * d, sizeof(float) * d);
        memset(dn, 0, sizeof(float) * mh * d);
        if (acc) memset(acc, 0, sizeof(float) * 2 * mh);
        return 1;
    }
    float sc[2 * 64];
    memcpy(sc, p + 8, sizeof(float) * 2 * mh);
    if (acc) memcpy(acc, p + 8 + 8 * mh, sizeof(float) * 2 * mh);
    const int8_t *q = (const int8_t *)(p + 8 + 16 * mh);
    for (int r = 0; r < mh; r++) {
        const float s = sc[r]; const int8_t *qr = q + (size_t)r * d; float *o = up + (size_t)r * d;
        for (int k = 0; k < d; k++) o[k] = s * (float)qr[k];
    }
    q += (size_t)mh * d;
    for (int r = 0; r < mh; r++) {
        const float s = sc[mh + r]; const int8_t *qr = q + (size_t)r * d; float *o = dn + (size_t)r * d;
        for (int k = 0; k < d; k++) o[k] = s * (float)qr[k];
    }
    return 0;
}

static inline float dot(const float *a, const float *b, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0; int k = 0;
    for (; k + 4 <= n; k += 4) { s0 += a[k] * b[k]; s1 += a[k + 1] * b[k + 1]; s2 += a[k + 2] * b[k + 2]; s3 += a[k + 3] * b[k + 3]; }
    for (; k < n; k++) s0 += a[k] * b[k];
    return (s0 + s1) + (s2 + s3);
}

/* pairs [p0, p1) sorted by expert. z: (P, mh) pre-activations, o: (P, d) expert outputs (before gates). returns pages read */
int64_t fx_forward(const uint8_t *base, int64_t page, int d, int mh, const float *pool, int64_t pool_n, uint64_t seed,
                   int64_t p0, int64_t p1, const int64_t *eid, const int64_t *tok, const float *x, float *z, float *o) {
    if (mh > 64) return -1;
    float *up = malloc(sizeof(float) * 2 * mh * d), *dn = up + (size_t)mh * d; int64_t last = -1, reads = 0;
    for (int64_t p = p0; p < p1; p++) {
        if (eid[p] != last) { load_expert(base, page, eid[p], d, mh, pool, pool_n, seed, up, dn, NULL); last = eid[p]; reads++; }
        const float *xv = x + tok[p] * d; float *zp = z + p * mh, *op = o + p * d;
        for (int k = 0; k < d; k++) op[k] = 0;
        for (int m = 0; m < mh; m++) {
            float zz = dot(up + (size_t)m * d, xv, d); zp[m] = zz;
            const float a = silu(zz); const float *dr = dn + (size_t)m * d;
            for (int k = 0; k < d; k++) op[k] += a * dr[k];
        }
    }
    free(up);
    return reads;
}

static inline uint64_t rng_next(uint64_t *s) { uint64_t x = (*s += 0x9E3779B97F4A7C15ull); x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31); }

/* Expert groups [g0, g1): group g owns pairs [gstart[g], gstart[g+1]) of one expert.
 * dog: (P, d) gradient w.r.t. each pair's output times its gate. dxs: (P, d) out, gradient w.r.t. the pair's input.
 * update=1: Adagrad + 8-bit stochastic rounding + write the page. update=0: copy the gradients to gout (G, 2mh, d) for tests.
 * ema_up/ema_dn < 0: a never-updated row starts its accumulator at its own gradient energy, else at the given typical value.
 * stats[0..2] += sum of row gradient energy for up rows, for down rows, number of rows of each kind; returns experts created. */
int64_t fx_backward(uint8_t *base, int64_t page, int d, int mh, const float *pool, int64_t pool_n, uint64_t seed,
                    int64_t g0, int64_t g1, const int64_t *gstart, const int64_t *eid, const int64_t *tok,
                    const float *x, const float *z, const float *dog, float *dxs,
                    float lr, float ema_up, float ema_dn, uint64_t rng_seed, int update, float *gout, double *stats) {
    if (mh > 64) return -1;
    const size_t W = (size_t)mh * d;
    float *buf = malloc(sizeof(float) * (4 * W + 2 * mh));
    float *up = buf, *dn = up + W, *gup = dn + W, *gdn = gup + W, *acc = gdn + W;
    float a[64], dz[64];
    uint64_t rs = rng_seed ^ ((uint64_t)g0 * 0x9E3779B97F4A7C15ull);
    int64_t created = 0; double se_up = 0, se_dn = 0;
    for (int64_t g = g0; g < g1; g++) {
        const int64_t q0 = gstart[g], q1 = gstart[g + 1], e = eid[q0];
        const int fresh = load_expert(base, page, e, d, mh, pool, pool_n, seed, up, dn, acc);
        memset(gup, 0, sizeof(float) * 2 * W);
        for (int64_t p = q0; p < q1; p++) {
            const float *xv = x + tok[p] * d, *dg = dog + p * d, *zp = z + p * mh; float *dx = dxs + p * d;
            for (int m = 0; m < mh; m++) {
                a[m] = silu(zp[m]);
                dz[m] = dot(dn + (size_t)m * d, dg, d) * dsilu(zp[m]);
            }
            for (int k = 0; k < d; k++) dx[k] = 0;
            for (int m = 0; m < mh; m++) {
                const float am = a[m], dzm = dz[m]; const float *ur = up + (size_t)m * d;
                float *gd = gdn + (size_t)m * d, *gu = gup + (size_t)m * d;
                for (int k = 0; k < d; k++) { dx[k] += dzm * ur[k]; gd[k] += am * dg[k]; gu[k] += dzm * xv[k]; }
            }
        }
        if (!update) { memcpy(gout + (size_t)(g - g0) * 2 * W, gup, sizeof(float) * 2 * W); continue; }
        created += fresh;
        /* row-wise Adagrad on the 2mh rows (up rows then down rows), then re-quantise each row */
        uint8_t *pg = base + e * page; float sc[2 * 64];
        int8_t *qout = (int8_t *)(pg + 8 + 16 * mh);
        for (int r = 0; r < 2 * mh; r++) {
            float *w = (r < mh ? up : dn) + (size_t)(r % mh) * d; const float *gr = (r < mh ? gup : gdn) + (size_t)(r % mh) * d;
            const float ge = dot(gr, gr, d) / (float)d;
            if (r < mh) se_up += ge; else se_dn += ge;
            const float em = r < mh ? ema_up : ema_dn;
            const float a0 = acc[r] > 0 ? acc[r] : (em >= 0 ? em : ge);
            const float ac = a0 + ge; acc[r] = ac;
            const float step = ac > 0 ? lr / sqrtf(ac) : 0.0f;
            float amax = 0;
            for (int k = 0; k < d; k++) { w[k] -= step * gr[k]; float v = fabsf(w[k]); if (v > amax) amax = v; }
            const float s = (amax > 1e-12f ? amax : 1e-12f) / 127.0f, inv = 1.0f / s; sc[r] = s;
            int8_t *qr = qout + (size_t)r * d;
            for (int k = 0; k < d; k++) {
                float v = floorf(w[k] * inv + (float)(rng_next(&rs) >> 40) * (1.0f / 16777216.0f));
                qr[k] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
            }
        }
        const uint32_t mg = MAGIC; memcpy(pg, &mg, 4); memset(pg + 4, 0, 4);
        memcpy(pg + 8, sc, sizeof(float) * 2 * mh); memcpy(pg + 8 + 8 * mh, acc, sizeof(float) * 2 * mh);
    }
    stats[0] += se_up; stats[1] += se_dn; stats[2] += (double)(g1 - g0) * mh;
    free(buf);
    return created;
}

/* Adam with decoupled weight decay in one pass (the NumPy version makes ~10 temporary arrays per tensor).
 * g is scaled by sc (gradient clipping); bc1/bc2 are the bias corrections 1-b1^t and 1-b2^t. */
void fx_adam(float *w, const float *g, float *m, float *v, int64_t n, float sc, float b1, float b2, float bc1, float bc2, float lr, float wd, float eps) {
    const float ib1 = 1.0f / bc1, ib2 = 1.0f / bc2;
    for (int64_t i = 0; i < n; i++) {
        const float gi = g[i] * sc;
        const float mi = b1 * m[i] + (1.0f - b1) * gi, vi = b2 * v[i] + (1.0f - b2) * gi * gi;
        m[i] = mi; v[i] = vi;
        w[i] -= lr * ((mi * ib1) / (sqrtf(vi * ib2) + eps) + wd * w[i]);
    }
}
