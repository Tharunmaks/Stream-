/* Graft: a small trainable network on top of a frozen model's final hidden state (see es_graft.h). The engine runs the
 * forward pass at every token; tools/es_graft.c trains it with the backward pass. */
#define _GNU_SOURCE
#include "es_graft.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __EMSCRIPTEN__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define ERRF(...) do { if (err && cap) snprintf(err, cap, __VA_ARGS__); return -1; } while (0)
static const char MAGIC[8] = {'E', 'S', 'G', 'R', 'A', 'F', 'T', '1'};

static inline float dotf(const float *a, const float *b, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) { s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3]; }
    for (; i < n; i++) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}
static inline float sigm(float z) { return 1.0f / (1.0f + expf(-z)); }
static uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ull;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static double gauss(uint64_t *s) {   /* Box-Muller from a splitmix stream */
    *s += 1; const double u1 = ((mix64(*s) >> 11) + 1.0) / 9007199254740993.0;
    *s += 1; const double u2 = (mix64(*s) >> 11) / 9007199254740992.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* ---------------- memory candidates ---------------- */
uint64_t es_mem_key(int order, const int32_t *ctx) {
    uint64_t h = mix64(0x6d656d6f7279ull + (uint64_t)order);
    for (int i = 0; i < order; i++) {
        if (ctx[i] < 0) return 0;
        h = mix64(h ^ (uint32_t)ctx[i]);
    }
    return h ? h : 1;
}
const es_mem_ent *es_mem_find(const es_mem *mm, uint64_t key) {
    if (!mm->e || !key) return NULL;
    uint64_t lo = 0, hi = mm->n;
    while (lo < hi) { const uint64_t mid = (lo + hi) / 2; if (mm->e[mid].key < key) lo = mid + 1; else hi = mid; }
    return lo < mm->n && mm->e[lo].key == key ? &mm->e[lo] : NULL;
}
int es_mem_suggest(const es_mem *mm, const int32_t *ctx, int32_t minus_one, const int32_t *skip, int nskip, int32_t *out, int cap) {
    int n = 0;
    for (int o = 0; o < ES_MEM_ORDERS; o++) {
        const es_mem_ent *e = es_mem_find(mm, es_mem_key(ES_MEM_ORDER[o], ctx));
        if (!e) continue;
        uint32_t c[ES_MEM_KEEP]; int used[ES_MEM_KEEP] = {0};
        for (int i = 0; i < ES_MEM_KEEP; i++) c[i] = e->tok[i] < 0 ? 0 : e->cnt[i] - (e->tok[i] == minus_one ? 1u : 0u);
        for (int pick = 0; pick < ES_MEM_PER; pick++) {   /* most frequent first, ties by position */
            int b = -1;
            for (int i = 0; i < ES_MEM_KEEP; i++) if (!used[i] && c[i] > 0 && (b < 0 || c[i] > c[b])) b = i;
            if (b < 0) break;
            used[b] = 1;
            const int32_t t = e->tok[b];
            int dup = 0;
            for (int i = 0; i < nskip && !dup; i++) dup = skip[i] == t;
            for (int i = 0; i < n && !dup; i++) dup = out[i] == t;
            if (!dup && n < cap) out[n++] = t;
        }
    }
    return n;
}
void es_mem_counts(const es_mem *mm, const int32_t *ctx, int32_t minus_one, const int32_t *cand, int n, uint16_t *cnt, uint32_t *tot) {
    memset(cnt, 0, sizeof(uint16_t) * (size_t)n * ES_MEM_ORDERS);
    for (int o = 0; o < ES_MEM_ORDERS; o++) {
        tot[o] = 0;
        const es_mem_ent *e = es_mem_find(mm, es_mem_key(ES_MEM_ORDER[o], ctx));
        if (!e) continue;
        tot[o] = e->total - (minus_one >= 0 && e->total ? 1u : 0u);   /* a training position leaves itself out */
        for (int i = 0; i < ES_MEM_KEEP; i++) {
            if (e->tok[i] < 0) continue;
            const uint32_t c = e->cnt[i] - (e->tok[i] == minus_one ? 1u : 0u);
            if (!c) continue;
            for (int k = 0; k < n; k++) if (cand[k] == e->tok[i]) { cnt[(size_t)k * ES_MEM_ORDERS + o] = (uint16_t)(c < 65535 ? c : 65535); break; }
        }
    }
}
float es_mem_bonus(const float *mb, const float *mg, const uint16_t *cnt, const uint32_t *tot) {
    float b = 0;
    for (int o = 0; o < ES_MEM_ORDERS; o++) {
        const unsigned c = cnt[o];
        if (!c) continue;
        const float lc = logf((float)c), lp = tot[o] > c ? logf((float)c / (float)tot[o]) : 0.0f;
        const float *q = mb + ES_MEM_BONUS * o;
        b += (mg ? mg[o] : 1.0f) * (q[0] + q[1] * lc + q[2] * lp);
    }
    return b;
}
int es_mem_load(es_mem *mm, const char *path, char *err, size_t cap) {
    memset(mm, 0, sizeof *mm);
#ifdef __EMSCRIPTEN__
    (void)path; (void)err; (void)cap; return 0;
#else
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;   /* no memory: fine */
    struct stat st;
    if (fstat(fd, &st) || st.st_size < 16) { close(fd); ERRF("%s is not a graft memory", path); }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) ERRF("cannot map %s", path);
    uint64_t n; memcpy(&n, (const char *)p + 8, 8);
    if (memcmp(p, "ESMEM003", 8) || 16 + n * sizeof(es_mem_ent) != (uint64_t)st.st_size) { munmap(p, (size_t)st.st_size); ERRF("%s is not a graft memory", path); }
    mm->map = p; mm->len = (size_t)st.st_size; mm->n = n; mm->e = (const es_mem_ent *)((const char *)p + 16);
    return 0;
#endif
}
void es_mem_free(es_mem *mm) {
#ifndef __EMSCRIPTEN__
    if (mm->map) munmap(mm->map, mm->len);
#endif
    memset(mm, 0, sizeof *mm);
}

int64_t es_graft_slot(const es_graft *g, int order, const int32_t *ctx) {
    if (!g->E || g->n_exp == 0) return -1;
    uint64_t h = mix64((uint64_t)g->seed * 0x100000001b3ull + (uint64_t)order);
    for (int i = 0; i < order; i++) {
        if (ctx[i] < 0) return -1;
        h = mix64(h ^ (uint32_t)ctx[i]);
    }
    return (int64_t)(h % g->n_exp);
}

static void layout(es_graft *g, float *base) {
    g->P = base;
    g->W1 = g->P + (size_t)g->m * g->d;
    g->b1 = g->W1 + (size_t)g->f * g->m;
    g->W2 = g->b1 + g->f;
    g->Q = g->W2 + (size_t)g->m * g->f;
    g->mb = g->Q + (size_t)g->d * g->m;
    g->U = g->mb + ES_MEM_BONUS * ES_MEM_ORDERS;
}

static int map_experts(es_graft *g, const char *path, int create, int writable, char *err, size_t cap) {
    g->E = NULL; g->e_fd = -1; g->e_bytes = 0;
    if (g->n_exp == 0 || !path) return 0;
#ifdef __EMSCRIPTEN__
    (void)create; (void)writable;
    ERRF("memory experts need the native engine");
#else
    const size_t bytes = (size_t)g->n_exp * es_graft_rec(g->m) * sizeof(float);
    const int fd = open(path, (writable ? O_RDWR : O_RDONLY) | (create ? O_CREAT | O_TRUNC : 0), 0644);
    if (fd < 0) ERRF("cannot open %s: %s", path, strerror(errno));
    struct stat st;
    if (create) { if (ftruncate(fd, (off_t)bytes)) { close(fd); ERRF("cannot size %s: %s", path, strerror(errno)); } }
    else if (fstat(fd, &st) || (size_t)st.st_size != bytes) { close(fd); ERRF("%s does not match its graft (size)", path); }
    void *p = mmap(NULL, bytes, writable ? PROT_READ | PROT_WRITE : PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) { close(fd); ERRF("cannot map %s: %s", path, strerror(errno)); }
    madvise(p, bytes, MADV_RANDOM);   /* one expert at a time: do not read neighbouring pages */
    g->E = p; g->e_fd = fd; g->e_bytes = bytes; g->writable = writable;
    return 0;
#endif
}

int es_graft_new(es_graft *g, int d, int m, int f, int K, uint64_t n_exp, int n_vocab, uint32_t seed, const char *experts_path, char *err, size_t cap) {
    memset(g, 0, sizeof *g);
    if (d <= 0 || m <= 0 || f <= 0 || K <= 0 || K > 256) ERRF("bad graft shape");
    g->d = d; g->m = m; g->f = f; g->K = K; g->orders = ES_GRAFT_ORDERS; g->n_vocab = n_vocab; g->seed = seed; g->n_exp = n_exp; g->e_fd = -1;
    float *w = calloc(es_graft_dense_floats(g), sizeof(float));
    if (!w) ERRF("out of memory");
    layout(g, w);
    uint64_t s = seed;
    for (size_t i = 0; i < (size_t)m * d; i++) g->P[i] = (float)(gauss(&s) / sqrt((double)d));
    for (size_t i = 0; i < (size_t)f * m; i++) g->W1[i] = (float)(gauss(&s) / sqrt((double)m));
    for (size_t i = 0; i < (size_t)m * f; i++) g->W2[i] = (float)(gauss(&s) / sqrt((double)f));
    /* b1 = 0, Q = 0: the new graft changes nothing */
    if (map_experts(g, experts_path, 1, 1, err, cap)) { free(w); g->P = NULL; return -1; }
    return 0;
}

static int rd(FILE *fp, void *p, size_t n) { return fread(p, 1, n, fp) == n ? 0 : -1; }
int es_graft_load(es_graft *g, const char *path, int writable, char *err, size_t cap) {
    memset(g, 0, sizeof *g); g->e_fd = -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) ERRF("cannot open %s: %s", path, strerror(errno));
    char mg[8]; uint32_t h[8]; uint64_t q[4];
    if (rd(fp, mg, 8) || memcmp(mg, MAGIC, 8)) { fclose(fp); ERRF("%s is not a graft", path); }
    if (rd(fp, h, sizeof h) || rd(fp, q, sizeof q) || rd(fp, g->arch, sizeof g->arch) || rd(fp, g->name, sizeof g->name)) { fclose(fp); ERRF("%s is cut short", path); }
    if (h[0] != 1) { fclose(fp); ERRF("%s: graft version %u is not supported", path, h[0]); }
    g->d = (int)h[1]; g->m = (int)h[2]; g->f = (int)h[3]; g->K = (int)h[4]; g->orders = (int)h[5]; g->n_vocab = (int)h[6]; g->seed = h[7];
    g->n_exp = q[0]; g->fingerprint = q[1]; g->steps = q[2]; g->tokens = q[3];
    g->arch[sizeof g->arch - 1] = 0; g->name[sizeof g->name - 1] = 0;
    if (g->d <= 0 || g->m <= 0 || g->f <= 0 || g->K <= 0 || g->K > 256 || g->orders != ES_GRAFT_ORDERS || g->d > (1 << 16) || g->m > 4096 || g->f > 1 << 14) {
        fclose(fp); ERRF("%s: bad graft shape", path);
    }
    float *w = malloc(es_graft_dense_floats(g) * sizeof(float));
    if (!w) { fclose(fp); ERRF("out of memory"); }
    if (rd(fp, w, es_graft_dense_floats(g) * sizeof(float))) { free(w); fclose(fp); ERRF("%s is cut short", path); }
    fclose(fp);
    layout(g, w);
    char ep[4096];
    snprintf(ep, sizeof ep, "%s.experts", path);
    if (map_experts(g, ep, 0, writable, err, cap)) { free(w); g->P = NULL; return -1; }
    snprintf(ep, sizeof ep, "%s.mem", path);
    if (es_mem_load(&g->mem, ep, err, cap)) { es_graft_free(g); return -1; }
    return 0;
}

int es_graft_save(const es_graft *g, const char *path, char *err, size_t cap) {
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *fp = fopen(tmp, "wb");
    if (!fp) ERRF("cannot write %s: %s", tmp, strerror(errno));
    const uint32_t h[8] = {1, (uint32_t)g->d, (uint32_t)g->m, (uint32_t)g->f, (uint32_t)g->K, (uint32_t)g->orders, (uint32_t)g->n_vocab, g->seed};
    const uint64_t q[4] = {g->n_exp, g->fingerprint, g->steps, g->tokens};
    int bad = fwrite(MAGIC, 1, 8, fp) != 8 || fwrite(h, sizeof h, 1, fp) != 1 || fwrite(q, sizeof q, 1, fp) != 1 ||
              fwrite(g->arch, sizeof g->arch, 1, fp) != 1 || fwrite(g->name, sizeof g->name, 1, fp) != 1 ||
              fwrite(g->P, sizeof(float), es_graft_dense_floats(g), fp) != es_graft_dense_floats(g);
    bad |= fclose(fp) != 0;
    if (bad) { remove(tmp); ERRF("cannot write %s", tmp); }
#ifndef __EMSCRIPTEN__
    if (g->E && g->writable) msync(g->E, g->e_bytes, MS_SYNC);
#endif
    if (rename(tmp, path)) ERRF("cannot rename %s: %s", tmp, strerror(errno));
    return 0;
}

void es_graft_free(es_graft *g) {
    free(g->P);
    es_mem_free(&g->mem);
#ifndef __EMSCRIPTEN__
    if (g->E) munmap(g->E, g->e_bytes);
    if (g->e_fd >= 0) close(g->e_fd);
#endif
    memset(g, 0, sizeof *g); g->e_fd = -1;
}

int es_graft_tmp_init(es_graft_tmp *t, const es_graft *g) {
    memset(t, 0, sizeof *t);
    float *b = malloc(sizeof(float) * ((size_t)2 * g->d + 2 * g->m + 2 * g->f));
    if (!b) return -1;
    t->xh = b; t->delta = b + g->d; t->a = t->delta + g->d; t->u = t->a + g->m; t->z1 = t->u + g->m; t->h1 = t->z1 + g->f;
    return 0;
}
void es_graft_tmp_free(es_graft_tmp *t) { free(t->xh); memset(t, 0, sizeof *t); }

void es_graft_forward(const es_graft *g, const float *x, const int32_t *ctx, es_graft_tmp *t) {
    const int d = g->d, m = g->m, f = g->f;
    double ss = 0;
    for (int i = 0; i < d; i++) ss += (double)x[i] * x[i];
    t->rinv = (float)(1.0 / sqrt(ss / d + 1e-6));
    for (int i = 0; i < d; i++) t->xh[i] = x[i] * t->rinv;
    for (int r = 0; r < m; r++) t->a[r] = dotf(g->P + (size_t)r * d, t->xh, d);
    for (int j = 0; j < f; j++) { const float z = g->b1[j] + dotf(g->W1 + (size_t)j * m, t->a, m); t->z1[j] = z; t->h1[j] = z * sigm(z); }
    for (int r = 0; r < m; r++) t->u[r] = dotf(g->W2 + (size_t)r * f, t->h1, f);
    const float isq = 1.0f / sqrtf((float)m);
    for (int n = 0; n < ES_GRAFT_ORDERS; n++) {
        t->e[n] = es_graft_slot(g, n + 1, ctx);
        t->s[n] = 0; t->gate[n] = 0;
        if (t->e[n] < 0) continue;
        const float *rec = g->E + (size_t)t->e[n] * es_graft_rec(m), *k = rec, *v = rec + m;
        t->s[n] = dotf(t->a, k, m) * isq;
        t->gate[n] = sigm(t->s[n]);
        for (int r = 0; r < m; r++) t->u[r] += t->gate[n] * v[r];
    }
    for (int i = 0; i < d; i++) t->delta[i] = dotf(g->Q + (size_t)i * m, t->u, m);
    for (int o = 0; o < ES_MEM_ORDERS; o++) t->mg[o] = sigm(dotf(g->U + (size_t)o * m, t->a, m) + g->mb[ES_MEM_BONUS * o + 3]);
}

double es_graft_loss(int K, int d, const float *bl, double logr, int yin, float by, const float *const *W, const float *delta,
                     const uint16_t *cnt, const uint32_t *tot, const float *mb, const float *mg, float *g_delta, float *g_mb, float *g_mg) {
    double l[ES_GRAFT_MAXK], mx = isfinite(logr) ? logr : -INFINITY, mx0 = mx;
    for (int k = 0; k < K; k++) {
        l[k] = bl[k] + (W && delta ? dotf(W[k], delta, d) : 0.0f);
        if (cnt && mb) l[k] += es_mem_bonus(mb, mg, cnt + (size_t)k * ES_MEM_ORDERS, tot);
        if (l[k] > mx) mx = l[k];
        if (bl[k] > mx0) mx0 = bl[k];
    }
    double sum = isfinite(logr) ? exp(logr - mx) : 0.0;
    for (int k = 0; k < K; k++) sum += exp(l[k] - mx);
    const double logz = mx + log(sum);
    double logz0 = 0;
    if (yin == -2) {   /* anchor: the base model's own distribution is the target */
        double s0 = isfinite(logr) ? exp(logr - mx0) : 0.0;
        for (int k = 0; k < K; k++) s0 += exp(bl[k] - mx0);
        logz0 = mx0 + log(s0);
    }
    if (g_delta) memset(g_delta, 0, sizeof(float) * (size_t)d);
    if (g_mb) memset(g_mb, 0, sizeof(float) * ES_MEM_BONUS * ES_MEM_ORDERS);
    if (g_mg) memset(g_mg, 0, sizeof(float) * ES_MEM_ORDERS);
    double kl_lin = 0;
    for (int k = 0; k < K; k++) {
        const double pb = yin == -2 && isfinite(bl[k]) ? exp(bl[k] - logz0) : 0.0;
        if (yin == -2 && pb > 0) kl_lin += pb * (l[k] - bl[k]);   /* empty candidate slots (score -inf) take no part */
        if (!g_delta && !g_mb && !g_mg) continue;
        const float gk = (float)(exp(l[k] - logz) - (yin == -2 ? pb : k == yin ? 1.0 : 0.0));
        if (g_delta && W) { const float *w = W[k]; for (int i = 0; i < d; i++) g_delta[i] += gk * w[i]; }
        if (cnt) for (int o = 0; o < ES_MEM_ORDERS; o++) {
            const unsigned c = cnt[(size_t)k * ES_MEM_ORDERS + o];
            if (!c) continue;
            const float lc = logf((float)c), lp = tot[o] > c ? logf((float)c / (float)tot[o]) : 0.0f, gt = mg ? mg[o] : 1.0f;
            const float *q = mb + ES_MEM_BONUS * o;
            if (g_mb) { g_mb[ES_MEM_BONUS * o] += gk * gt; g_mb[ES_MEM_BONUS * o + 1] += gk * gt * lc; g_mb[ES_MEM_BONUS * o + 2] += gk * gt * lp; }
            if (g_mg) g_mg[o] += gk * (q[0] + q[1] * lc + q[2] * lp);
        }
    }
    if (yin == -2) return logz - logz0 - kl_lin;
    return logz - (yin >= 0 ? l[yin] : (double)by);
}

void es_graft_backward(const es_graft *g, const es_graft_tmp *t, const float *g_delta, const float *g_mg, float *grad, float *gu, float *ga, float *ek, float *ev) {
    const int d = g->d, m = g->m, f = g->f;
    float *gP = grad, *gW1 = gP + (size_t)m * d, *gb1 = gW1 + (size_t)f * m, *gW2 = gb1 + f, *gQ = gW2 + (size_t)m * f;
    memset(gu, 0, sizeof(float) * (size_t)m);
    for (int i = 0; i < d; i++) {   /* delta = Q u */
        const float gd = g_delta[i];
        if (gd == 0) continue;
        const float *q = g->Q + (size_t)i * m; float *gq = gQ + (size_t)i * m;
        for (int j = 0; j < m; j++) { gq[j] += gd * t->u[j]; gu[j] += gd * q[j]; }
    }
    memset(ga, 0, sizeof(float) * (size_t)m);
    if (g_mg) {   /* memory gates: sigmoid(U_o . a + bias_o) */
        float *gmb = gQ + (size_t)d * m, *gU = gmb + ES_MEM_BONUS * ES_MEM_ORDERS;
        for (int o = 0; o < ES_MEM_ORDERS; o++) {
            const float gp = g_mg[o] * t->mg[o] * (1 - t->mg[o]);
            if (gp == 0) continue;
            gmb[ES_MEM_BONUS * o + 3] += gp;
            const float *u = g->U + (size_t)o * m; float *gu2 = gU + (size_t)o * m;
            for (int r = 0; r < m; r++) { gu2[r] += gp * t->a[r]; ga[r] += gp * u[r]; }
        }
    }
    const float isq = 1.0f / sqrtf((float)m);
    for (int n = 0; n < ES_GRAFT_ORDERS; n++) {   /* u += sigmoid(a . k / sqrt m) v */
        float *gk = ek + (size_t)n * m, *gv = ev + (size_t)n * m;
        if (t->e[n] < 0) { memset(gk, 0, sizeof(float) * (size_t)m); memset(gv, 0, sizeof(float) * (size_t)m); continue; }
        const float *rec = g->E + (size_t)t->e[n] * es_graft_rec(m), *k = rec, *v = rec + m;
        const float gt = t->gate[n], ds = dotf(gu, v, m) * gt * (1 - gt) * isq;
        for (int r = 0; r < m; r++) { gv[r] = gt * gu[r]; gk[r] = ds * t->a[r]; ga[r] += ds * k[r]; }
    }
    float gh1[f];
    memset(gh1, 0, sizeof gh1);
    for (int r = 0; r < m; r++) {   /* u += W2 h1 */
        const float gr = gu[r];
        const float *w = g->W2 + (size_t)r * f; float *gw = gW2 + (size_t)r * f;
        for (int j = 0; j < f; j++) { gw[j] += gr * t->h1[j]; gh1[j] += gr * w[j]; }
    }
    for (int j = 0; j < f; j++) {   /* h1 = silu(W1 a + b1) */
        const float z = t->z1[j], s = sigm(z), gz = gh1[j] * s * (1 + z * (1 - s));
        if (gz == 0) continue;
        gb1[j] += gz;
        const float *w = g->W1 + (size_t)j * m; float *gw = gW1 + (size_t)j * m;
        for (int c = 0; c < m; c++) { gw[c] += gz * t->a[c]; ga[c] += gz * w[c]; }
    }
    for (int r = 0; r < m; r++) {   /* a = P xh */
        const float gr = ga[r];
        if (gr == 0) continue;
        float *gp = gP + (size_t)r * d;
        for (int c = 0; c < d; c++) gp[c] += gr * t->xh[c];
    }
}

int es_graft_topk(const float *v, int n, int k, int *idx) {
    int c = 0;
    for (int i = 0; i < n; i++) {
        const float x = v[i];
        if (c == k && !(x > v[idx[c - 1]])) continue;
        int j = c < k ? c++ : c - 1;
        while (j > 0 && v[idx[j - 1]] < x) { idx[j] = idx[j - 1]; j--; }
        idx[j] = i;
    }
    return c;
}

int es_graft_candidates(int K, const es_mem *mm, const float *logits, int n_vocab, const int32_t *ctx, int32_t minus_one, int32_t *out) {
    int idx[ES_GRAFT_MAXK];
    const int n = es_graft_topk(logits, n_vocab, K, idx);
    for (int i = 0; i < n; i++) out[i] = idx[i];
    return n + es_mem_suggest(mm, ctx, minus_one, out, n, out + n, ES_MEM_ORDERS * ES_MEM_PER);
}

void es_graft_apply(const es_graft *g, const float *x, const int32_t *ctx, float *logits, void (*row)(int tok, float *out, void *ud), void *ud,
                    es_graft_tmp *t, float *wbuf) {
    es_graft_forward(g, x, ctx, t);
    int32_t idx[ES_GRAFT_MAXK];
    const int n = es_graft_candidates(g->K, &g->mem, logits, g->n_vocab, ctx, -1, idx);   /* chosen from the base scores, as in training */
    uint16_t cnt[ES_GRAFT_MAXK * ES_MEM_ORDERS]; uint32_t tot[ES_MEM_ORDERS];
    es_mem_counts(&g->mem, ctx, -1, idx, n, cnt, tot);
    float add[ES_GRAFT_MAXK];
    for (int i = 0; i < n; i++) {
        row(idx[i], wbuf, ud);
        add[i] = dotf(wbuf, t->delta, g->d) + es_mem_bonus(g->mb, t->mg, cnt + (size_t)i * ES_MEM_ORDERS, tot);
    }
    for (int i = 0; i < n; i++) logits[idx[i]] += add[i];
    static int dbg = -1;
    if (dbg < 0) dbg = getenv("ES_GRAFT_DEBUG") != NULL;
    if (dbg) {   /* ES_GRAFT_DEBUG=1: what the graft did at each token */
        fprintf(stderr, "[graft ctx %d %d %d |", ctx[2], ctx[1], ctx[0]);
        for (int i = 0; i < n && i < 48; i++) {
            int mo = 0; for (int o = 0; o < ES_MEM_ORDERS; o++) if (cnt[(size_t)i * ES_MEM_ORDERS + o]) mo |= 1 << o;
            if (i < 6 || mo) fprintf(stderr, " %d:%+.2f%s", idx[i], add[i], mo ? (i >= g->K ? "M" : "m") : "");
        }
        fprintf(stderr, "]\n");
    }
}
