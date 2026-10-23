/* ExpertStream trainer: exact-gradient LoRA on a frozen, memory-mapped GGUF model (see es_train.h).
 *
 * One step over T tokens (S sequences packed one after another):
 *   forward   layer by layer; only each layer's input is kept (T x n_embd per layer)
 *   loss      final norm, output rows, cross-entropy on the learned positions, in tiles of 64 positions
 *   backward  for each layer from the last: recompute it from its saved input, then go back through it
 * Every weight matrix is read three times per step (forward, recompute, backward), always in layer order, so the file streams.
 * Matrix products dequantize a block of weight rows into floats and multiply them with a tile of tokens (2 tokens x 4 rows per
 * inner step); the backward products use the same rows transposed. Attention keeps no probabilities: the backward pass recomputes
 * them row by row from the saved log-sum-exp, so its memory grows with T, not T squared.
 * The math matches the engine's forward pass (rmsnorm with a double sum, GGML rotary pairs, biases, per-head q/k norms) except that
 * the engine rounds activations to 8 bits for its integer kernels; here everything is float (es_lora logitcheck measures the gap).
 */
#define _GNU_SOURCE
#include "es_train.h"
#include "es_gguf.h"
#include "es_quant.h"
#include "es_amx.h"
#include "es_cpu.h"

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define RB 32       /* weight rows dequantized together (forward) */
#define RBB 16      /* weight rows dequantized together (backward) */
#define TT 64       /* token tile (forward) */
#define TB 32       /* token tile (backward, token-parallel) */
#define LM_TILE 64  /* learned positions per output-layer tile */

typedef struct { uint32_t type; int rows, cols; const uint8_t *data; size_t rb; uint64_t nbytes; } tw;
typedef struct {
    float *attn_norm, *ffn_norm, *q_norm, *k_norm, *bq, *bk, *bv;
    int qn_len, kn_len;
    tw w[ES_LT_N];
} tlayer;

/* ---------------------------------------------------------------- thread pool */
typedef void (*job_fn)(void *ctx, int item, int tid);
typedef struct {
    int n; pthread_t *th;
    pthread_mutex_t mu; pthread_cond_t go;
    job_fn fn; void *ctx; int items; atomic_int next, busy; atomic_uint gen; atomic_int quit, sleepers;
} pool;
/* Workers poll for new work for a while before sleeping, and the caller polls for the end. Waking a sleeping thread
 * costs tens to hundreds of microseconds (much more in a VM, where the idle core itself has halted), and a training
 * pass runs tens of thousands of parallel sections with serial work in between: with plain sleeping workers, the
 * calling thread did most of the work alone. ES_POOL_SPIN sets the polling (default about a millisecond; 0 = sleep). */
static long pool_spin(void) { static long s = -1; if (s < 0) { const char *e = getenv("ES_POOL_SPIN"); s = e ? atol(e) : 200000; } return s; }
static inline void cpu_relax(void) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield");
#endif
}
static void pool_work(pool *p, int tid) {
    for (;;) {
        const int i = atomic_fetch_add(&p->next, 1);
        if (i >= p->items) break;
        p->fn(p->ctx, i, tid);
    }
}
typedef struct { pool *p; int tid; } worker_arg;
static void *pool_main(void *a_) {
    worker_arg *a = a_; pool *p = a->p; const int tid = a->tid; free(a);
    unsigned seen = 0;
    for (;;) {
        unsigned g = atomic_load_explicit(&p->gen, memory_order_acquire);
        for (long s = 0, n = pool_spin(); g == seen && s < n && !atomic_load(&p->quit); s++) { cpu_relax(); g = atomic_load_explicit(&p->gen, memory_order_acquire); }
        if (g == seen) {
            pthread_mutex_lock(&p->mu);
            atomic_fetch_add(&p->sleepers, 1);
            while ((g = atomic_load(&p->gen)) == seen && !atomic_load(&p->quit)) pthread_cond_wait(&p->go, &p->mu);
            atomic_fetch_sub(&p->sleepers, 1);
            pthread_mutex_unlock(&p->mu);
        }
        if (atomic_load(&p->quit)) return NULL;
        seen = g;
        pool_work(p, tid);
        atomic_fetch_sub_explicit(&p->busy, 1, memory_order_release);
    }
}
static void pool_init(pool *p, int n) {
    memset(p, 0, sizeof *p);
    es_unpin_self();   /* opened after the engine, this thread is pinned to one core, and the workers would inherit it */
    p->n = n < 1 ? 1 : n;
    pthread_mutex_init(&p->mu, NULL); pthread_cond_init(&p->go, NULL);
    p->th = calloc((size_t)p->n, sizeof(pthread_t));
    for (int i = 1; i < p->n; i++) { worker_arg *a = malloc(sizeof *a); a->p = p; a->tid = i; pthread_create(&p->th[i], NULL, pool_main, a); }
}
static void pool_run(pool *p, int items, job_fn fn, void *ctx) {
    if (items <= 0) return;
    if (p->n == 1 || items == 1) { for (int i = 0; i < items; i++) fn(ctx, i, 0); return; }
    p->fn = fn; p->ctx = ctx; p->items = items; atomic_store(&p->next, 0); atomic_store(&p->busy, p->n - 1);
    atomic_fetch_add_explicit(&p->gen, 1, memory_order_release);
    if (atomic_load(&p->sleepers)) { pthread_mutex_lock(&p->mu); pthread_cond_broadcast(&p->go); pthread_mutex_unlock(&p->mu); }
    pool_work(p, 0);
    for (long s = 0; atomic_load_explicit(&p->busy, memory_order_acquire) > 0; s++) { if (s < 100000) cpu_relax(); else sched_yield(); }
}
static void pool_free(pool *p) {
    pthread_mutex_lock(&p->mu); atomic_store(&p->quit, 1); pthread_cond_broadcast(&p->go); pthread_mutex_unlock(&p->mu);
    for (int i = 1; i < p->n; i++) pthread_join(p->th[i], NULL);
    free(p->th);
}

/* ---------------------------------------------------------------- the model */
struct es_tm {
    char arch[32];
    int L, d, H, KV, hd, qd, kvd, ff, V, n_rot, neox;
    float eps, rope_base, freq_scale, scale;
    float *rope_freqs;
    tw embd, out; float *out_norm;
    tlayer *ly;
    uint8_t *map; size_t map_len; uint64_t data_start;
    es_gguf g;
    pool P; int nt;
    float **scratch; size_t scratch_n;    /* per thread: dequantized rows, attention scores, private accumulators */
    /* LoRA */
    int rank; float alpha, ls; uint32_t mask;
    float *lp, *lg; size_t ln;
    size_t aoff[1][ES_LT_N]; size_t *off;   /* off[(l * ES_LT_N + t) * 2 + 0/1]: A and B of target t in layer l, or SIZE_MAX */
    /* buffers for up to Tcap tokens */
    int Tcap;
    float *X, *Xn, *Q, *K, *Vv, *Qp, *Kp, *Att, *Lse, *Xm, *Xn2, *G, *U, *Hh, *r1, *r2, *qr, *kr, *Z[ES_LT_N];
    float *keep; size_t keep_cap;   /* per-layer forward intermediates (no recompute in the backward pass) */
    float *dX, *dXn, *dQ, *dK, *dV, *dAtt, *dH, *dZ;
    float *ck; size_t ck_cap;                 /* saved layer inputs */
    float *cosr, *sinr; int rope_cap;
    float *lmbuf;                             /* LM_TILE x V logits */
    double bytes;
};

static int in_of(const es_tm *m, int t) { return t == ES_LT_O ? m->qd : t == ES_LT_DOWN ? m->ff : m->d; }
static int out_of(const es_tm *m, int t) {
    return t == ES_LT_Q ? m->qd : (t == ES_LT_K || t == ES_LT_V) ? m->kvd : t == ES_LT_O || t == ES_LT_DOWN ? m->d : m->ff;
}

static int tensor(es_tm *m, const char *name, tw *w, int req, char *err, size_t cap) {
    memset(w, 0, sizeof *w);
    const es_gguf_tensor *t = es_gguf_tensor_find(&m->g, name);
    if (!t) { if (req) snprintf(err, cap, "missing tensor %s", name); return req ? -1 : 0; }
    if (t->shard) { snprintf(err, cap, "split models are not supported by the trainer yet (%s)", name); return -1; }
    if (!es_quant_supported(t->type)) { snprintf(err, cap, "unsupported weight type %u in %s", t->type, name); return -1; }
    w->type = t->type; w->cols = (int)t->ne[0]; w->rows = t->n_dims > 1 ? (int)t->ne[1] : 1;
    w->data = m->map + m->data_start + t->offset; w->rb = es_row_bytes(t->type, w->cols); w->nbytes = t->nbytes;
    return 1;
}
static float *vec(es_tm *m, const char *name, int n, int req, char *err, size_t cap, int *len) {
    tw w; const int r = tensor(m, name, &w, req, err, cap);
    if (r <= 0) return NULL;
    const int cnt = w.cols * w.rows;
    if (n && cnt != n && !len) { snprintf(err, cap, "%s has %d values, expected %d", name, cnt, n); return NULL; }
    float *v = malloc(sizeof(float) * (size_t)cnt);
    es_dequant_row(w.type, w.data, v, cnt);
    if (len) *len = cnt;
    return v;
}
static float kv_float(const es_gguf *g, const char *key, float def) {
    const es_gguf_kv *kv = es_gguf_find(g, key);
    if (!kv) return def;
    if (kv->type == GGUF_T_F32 || kv->type == GGUF_T_F64) return (float)kv->v.f;
    return (float)es_gguf_int(g, key, (int64_t)def);
}

es_tm *es_tm_open(const char *path, int threads, char *err, size_t cap) {
    es_tm *m = calloc(1, sizeof *m);
    const char *e = NULL;
    if (es_gguf_open(&m->g, path, &e)) { snprintf(err, cap, "%s: %s", path, e ? e : "cannot open"); free(m); return NULL; }
    const char *arch = es_gguf_str(&m->g, "general.architecture");
    if (!arch) { snprintf(err, cap, "not a model file"); goto fail; }
    snprintf(m->arch, sizeof m->arch, "%s", arch);
    if (strcmp(arch, "llama") && strcmp(arch, "qwen2") && strcmp(arch, "qwen3")) {
        snprintf(err, cap, "the trainer supports dense llama, qwen2 and qwen3 models; this one is %s", arch); goto fail;
    }
    char key[160];
#define HI(nm, def) (snprintf(key, sizeof key, "%s." nm, arch), es_gguf_int(&m->g, key, def))
#define HF(nm, def) (snprintf(key, sizeof key, "%s." nm, arch), kv_float(&m->g, key, def))
    if (HI("expert_count", 0) > 0) { snprintf(err, cap, "mixture-of-experts models are not supported by the trainer yet"); goto fail; }
    m->L = (int)HI("block_count", 0); m->d = (int)HI("embedding_length", 0); m->H = (int)HI("attention.head_count", 0);
    m->KV = (int)HI("attention.head_count_kv", m->H); m->ff = (int)HI("feed_forward_length", 0);
    m->hd = (int)HI("attention.key_length", m->H ? m->d / m->H : 0);
    m->eps = HF("attention.layer_norm_rms_epsilon", 1e-5f); m->rope_base = HF("rope.freq_base", 10000.0f);
    m->n_rot = (int)HI("rope.dimension_count", m->hd);
    m->neox = strcmp(arch, "llama") != 0;
    m->qd = m->H * m->hd; m->kvd = m->KV * m->hd; m->scale = 1.0f / sqrtf((float)m->hd);
    snprintf(key, sizeof key, "%s.rope.scaling.type", arch);
    const char *st = es_gguf_str(&m->g, key);
    float factor = HF("rope.scaling.factor", 0.0f); if (factor == 0.0f) factor = HF("rope.scale_linear", 0.0f);
    if (st && !strcmp(st, "yarn")) { snprintf(err, cap, "YaRN rotary scaling is not supported by the trainer yet"); goto fail; }
    m->freq_scale = (factor == 0.0f || (st && !strcmp(st, "none"))) ? 1.0f : 1.0f / factor;
    if (HF("attention.temperature_scale", 0.0f) != 0.0f) { snprintf(err, cap, "attention temperature scaling is not supported"); goto fail; }
    if (!m->L || !m->d || !m->H || !m->ff || m->qd % 8 || m->kvd % 8 || m->d % 8 || m->ff % 8 || m->n_rot > m->hd || m->n_rot % 2) {
        snprintf(err, cap, "unsupported model shape"); goto fail;
    }
#undef HI
#undef HF
    struct stat sb;
    if (fstat(m->g.fd, &sb)) { snprintf(err, cap, "cannot stat %s", path); goto fail; }
    m->map_len = (size_t)sb.st_size;
    m->map = mmap(NULL, m->map_len, PROT_READ, MAP_SHARED, m->g.fd, 0);
    if (m->map == MAP_FAILED) { m->map = NULL; snprintf(err, cap, "cannot map %s: %s", path, strerror(errno)); goto fail; }
    madvise(m->map, m->map_len, MADV_SEQUENTIAL);
    m->data_start = m->g.data_start;
    if (tensor(m, "token_embd.weight", &m->embd, 1, err, cap) < 0) goto fail;
    m->V = m->embd.rows;
    if (tensor(m, "output.weight", &m->out, 0, err, cap) < 0) goto fail;
    if (!m->out.data) m->out = m->embd;   /* tied embeddings */
    if (!(m->out_norm = vec(m, "output_norm.weight", m->d, 1, err, cap, NULL))) goto fail;
    { int n = 0; m->rope_freqs = vec(m, "rope_freqs.weight", 0, 0, err, cap, &n); if (m->rope_freqs && n < m->n_rot / 2) { free(m->rope_freqs); m->rope_freqs = NULL; } }
    m->ly = calloc((size_t)m->L, sizeof(tlayer));
    static const char *wn[ES_LT_N] = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"};
    for (int l = 0; l < m->L; l++) {
        tlayer *y = &m->ly[l];
        snprintf(key, sizeof key, "blk.%d.attn_norm.weight", l); if (!(y->attn_norm = vec(m, key, m->d, 1, err, cap, NULL))) goto fail;
        snprintf(key, sizeof key, "blk.%d.ffn_norm.weight", l); if (!(y->ffn_norm = vec(m, key, m->d, 1, err, cap, NULL))) goto fail;
        snprintf(key, sizeof key, "blk.%d.attn_q_norm.weight", l); y->q_norm = vec(m, key, 0, 0, err, cap, &y->qn_len);
        snprintf(key, sizeof key, "blk.%d.attn_k_norm.weight", l); y->k_norm = vec(m, key, 0, 0, err, cap, &y->kn_len);
        snprintf(key, sizeof key, "blk.%d.attn_q.bias", l); y->bq = vec(m, key, m->qd, 0, err, cap, NULL);
        snprintf(key, sizeof key, "blk.%d.attn_k.bias", l); y->bk = vec(m, key, m->kvd, 0, err, cap, NULL);
        snprintf(key, sizeof key, "blk.%d.attn_v.bias", l); y->bv = vec(m, key, m->kvd, 0, err, cap, NULL);
        for (int t = 0; t < ES_LT_N; t++) {
            snprintf(key, sizeof key, "blk.%d.%s.weight", l, wn[t]);
            if (tensor(m, key, &y->w[t], 1, err, cap) < 0) goto fail;
            if (y->w[t].cols != in_of(m, t) || y->w[t].rows != out_of(m, t)) { snprintf(err, cap, "unexpected shape of %s", key); goto fail; }
        }
        if ((y->q_norm && y->qn_len != m->hd && y->qn_len != m->qd) || (y->k_norm && y->kn_len != m->hd && y->kn_len != m->kvd)) { snprintf(err, cap, "unexpected q/k norm size"); goto fail; }
    }
    if (threads <= 0) { const long c = sysconf(_SC_NPROCESSORS_ONLN); threads = c > 0 ? (int)c : 1; }
    m->nt = threads; pool_init(&m->P, threads);
    return m;
fail:
    es_tm_close(m);
    return NULL;
}

static void free_bufs(es_tm *m) {
    float **b[] = {&m->X, &m->Xn, &m->Q, &m->K, &m->Vv, &m->Qp, &m->Kp, &m->Att, &m->Lse, &m->Xm, &m->Xn2, &m->G, &m->U, &m->Hh, &m->r1, &m->r2, &m->qr, &m->kr,
                   &m->dX, &m->dXn, &m->dQ, &m->dK, &m->dV, &m->dAtt, &m->dH, &m->dZ, &m->lmbuf};
    for (size_t i = 0; i < sizeof b / sizeof *b; i++) { free(*b[i]); *b[i] = NULL; }
    for (int t = 0; t < ES_LT_N; t++) { free(m->Z[t]); m->Z[t] = NULL; }
    m->Tcap = 0;
}
static void weights_cache_free(void);
void es_tm_close(es_tm *m) {
    if (!m) return;
    weights_cache_free();   /* keyed by weight address: a model opened later could reuse the address */
    if (m->nt) pool_free(&m->P);
    free_bufs(m);
    if (m->scratch) { for (int i = 0; i < m->nt; i++) free(m->scratch[i]); free(m->scratch); }
    if (m->ly) for (int l = 0; l < m->L; l++) {
        tlayer *y = &m->ly[l];
        free(y->attn_norm); free(y->ffn_norm); free(y->q_norm); free(y->k_norm); free(y->bq); free(y->bk); free(y->bv);
    }
    free(m->ly); free(m->out_norm); free(m->rope_freqs); free(m->lp); free(m->lg); free(m->off); free(m->ck); free(m->cosr); free(m->sinr);
    if (m->map) munmap(m->map, m->map_len);
    es_gguf_close(&m->g);
    free(m);
}
void es_tm_dims(const es_tm *m, int *L, int *d, int *V) { if (L) *L = m->L; if (d) *d = m->d; if (V) *V = m->V; }
const char *es_tm_arch(const es_tm *m) { return m->arch; }
double es_tm_bytes_read(const es_tm *m) { return m->bytes; }

/* ---------------------------------------------------------------- buffers */
static float *fa(size_t n) { float *p = malloc(sizeof(float) * (n ? n : 1)); if (!p) { fprintf(stderr, "out of memory (%zu MB)\n", n * 4 >> 20); exit(1); } return p; }
static void ensure(es_tm *m, int T) {
    const size_t rk = m->rank > 0 ? (size_t)m->rank : 1;
    if (!m->scratch) {
        int mx = m->d > m->ff ? m->d : m->ff; if (m->qd > mx) mx = m->qd;
        m->scratch_n = (size_t)(RB > RBB ? RB : RBB) * mx;
        m->scratch = calloc((size_t)m->nt, sizeof(float *));
        for (int i = 0; i < m->nt; i++) m->scratch[i] = fa(m->scratch_n);
    }
    if (T > m->Tcap) {
        free_bufs(m);
        const size_t t = (size_t)T;
        m->X = fa(t * m->d); m->Xn = fa(t * m->d); m->Q = fa(t * m->qd); m->K = fa(t * m->kvd); m->Vv = fa(t * m->kvd);
        m->Qp = fa(t * m->qd); m->Kp = fa(t * m->kvd); m->Att = fa(t * m->qd); m->Lse = fa(t * m->H); m->Xm = fa(t * m->d); m->Xn2 = fa(t * m->d);
        m->G = fa(t * m->ff); m->U = fa(t * m->ff); m->Hh = fa(t * m->ff); m->r1 = fa(t); m->r2 = fa(t); m->qr = fa(t * m->H); m->kr = fa(t * m->KV);
        for (int k = 0; k < ES_LT_N; k++) m->Z[k] = fa(t * rk);
        m->dX = fa(t * m->d); m->dXn = fa(t * m->d); m->dQ = fa(t * m->qd); m->dK = fa(t * m->kvd); m->dV = fa(t * m->kvd);
        m->dAtt = fa(t * m->qd); m->dH = fa(t * m->ff); m->dZ = fa(t * rk);
        m->lmbuf = fa((size_t)LM_TILE * m->V);
        m->Tcap = T;
        /* private accumulators of the row-parallel backward product need T x max(in) per thread */
        int mx = m->d > m->ff ? m->d : m->ff; if (m->qd > mx) mx = m->qd;
        size_t need = (size_t)(RB > RBB ? RB : RBB) * mx;
        if ((size_t)LM_TILE * m->d + (size_t)RBB * mx > need) need = (size_t)LM_TILE * m->d + (size_t)RBB * mx;
        if (t + 64 > need) need = t + 64;
        const size_t few = t < (size_t)TB * m->nt ? t : (size_t)TB * m->nt;   /* the row-parallel backward product: T x width per thread */
        if (few * mx + (size_t)RBB * mx > need) need = few * mx + (size_t)RBB * mx;
        if ((size_t)m->qd + 64 > need) need = (size_t)m->qd + 64;
        if (need > m->scratch_n) {
            for (int i = 0; i < m->nt; i++) { free(m->scratch[i]); m->scratch[i] = fa(need); }
            m->scratch_n = need;
        }
    }
    const size_t ck = (size_t)(m->L + 1) * T * m->d;
    if (ck > m->ck_cap) { free(m->ck); m->ck = fa(ck); m->ck_cap = ck; }
    if (T > m->rope_cap) {
        free(m->cosr); free(m->sinr);
        const int half = m->n_rot / 2;
        m->cosr = fa((size_t)T * half); m->sinr = fa((size_t)T * half);
        for (int p = 0; p < T; p++)
            for (int i = 0; i < half; i++) {
                float th = p * powf(m->rope_base, -2.0f * i / m->n_rot);
                if (m->rope_freqs) th /= m->rope_freqs[i];
                th *= m->freq_scale;
                m->cosr[(size_t)p * half + i] = cosf(th); m->sinr[(size_t)p * half + i] = sinf(th);
            }
        m->rope_cap = T;
    }
}

/* ---------------------------------------------------------------- matrix kernels */
typedef float v8 __attribute__((vector_size(32), aligned(4)));
#define hsum(v) ((((v)[0] + (v)[1]) + ((v)[2] + (v)[3])) + (((v)[4] + (v)[5]) + ((v)[6] + (v)[7])))

/* Y[t][o0 + r] = X[t] . W[r] for r < nr, t < T (W: nr dequantized rows of length n) */
static inline __attribute__((always_inline)) void fwd_body(const float *X, int T, size_t ldx, const float *W, int nr, int n, float *Y, size_t ldy, int o0) {
    int r = 0;
    for (; r + 4 <= nr; r += 4) {
        const float *w0 = W + (size_t)r * n, *w1 = w0 + n, *w2 = w1 + n, *w3 = w2 + n;
        int t = 0;
        for (; t + 3 <= T; t += 3) {   /* 4 rows x 3 tokens: 12 accumulators, 7 loads per 12 multiply-adds */
            const float *x0 = X + (size_t)t * ldx, *x1 = x0 + ldx, *x2 = x1 + ldx;
            v8 a00 = {0}, a01 = {0}, a02 = {0}, a03 = {0}, a10 = {0}, a11 = {0}, a12 = {0}, a13 = {0}, a20 = {0}, a21 = {0}, a22 = {0}, a23 = {0};
            for (int i = 0; i < n; i += 8) {
                const v8 p = *(const v8 *)(x0 + i), q = *(const v8 *)(x1 + i), u = *(const v8 *)(x2 + i);
                v8 b = *(const v8 *)(w0 + i); a00 += p * b; a10 += q * b; a20 += u * b;
                b = *(const v8 *)(w1 + i); a01 += p * b; a11 += q * b; a21 += u * b;
                b = *(const v8 *)(w2 + i); a02 += p * b; a12 += q * b; a22 += u * b;
                b = *(const v8 *)(w3 + i); a03 += p * b; a13 += q * b; a23 += u * b;
            }
            float *y0 = Y + (size_t)t * ldy + o0 + r, *y1 = y0 + ldy, *y2 = y1 + ldy;
            y0[0] = hsum(a00); y0[1] = hsum(a01); y0[2] = hsum(a02); y0[3] = hsum(a03);
            y1[0] = hsum(a10); y1[1] = hsum(a11); y1[2] = hsum(a12); y1[3] = hsum(a13);
            y2[0] = hsum(a20); y2[1] = hsum(a21); y2[2] = hsum(a22); y2[3] = hsum(a23);
        }
        for (; t < T; t++) {
            const float *x0 = X + (size_t)t * ldx;
            v8 a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0};
            for (int i = 0; i < n; i += 8) {
                const v8 p = *(const v8 *)(x0 + i);
                a0 += p * *(const v8 *)(w0 + i); a1 += p * *(const v8 *)(w1 + i); a2 += p * *(const v8 *)(w2 + i); a3 += p * *(const v8 *)(w3 + i);
            }
            float *y0 = Y + (size_t)t * ldy + o0 + r;
            y0[0] = hsum(a0); y0[1] = hsum(a1); y0[2] = hsum(a2); y0[3] = hsum(a3);
        }
    }
    for (; r < nr; r++) {
        const float *w0 = W + (size_t)r * n;
        for (int t = 0; t < T; t++) {
            const float *x0 = X + (size_t)t * ldx; v8 a = {0};
            for (int i = 0; i < n; i += 8) a += *(const v8 *)(x0 + i) * *(const v8 *)(w0 + i);
            Y[(size_t)t * ldy + o0 + r] = hsum(a);
        }
    }
}
/* dX[t] += sum_r dY[t][o0 + r] * W[r] */
static inline __attribute__((always_inline)) void bwd_body(const float *dY, int T, size_t ldy, int o0, const float *W, int nr, int n, float *dX, size_t ldx) {
    int r = 0;
    for (; r + 4 <= nr; r += 4) {
        const float *w0 = W + (size_t)r * n, *w1 = w0 + n, *w2 = w1 + n, *w3 = w2 + n;
        int t = 0;
        for (; t + 2 <= T; t += 2) {
            const float *g0 = dY + (size_t)t * ldy + o0 + r, *g1 = g0 + ldy;
            const float c00 = g0[0], c01 = g0[1], c02 = g0[2], c03 = g0[3], c10 = g1[0], c11 = g1[1], c12 = g1[2], c13 = g1[3];
            if (c00 == 0 && c01 == 0 && c02 == 0 && c03 == 0 && c10 == 0 && c11 == 0 && c12 == 0 && c13 == 0) continue;
            float *d0 = dX + (size_t)t * ldx, *d1 = d0 + ldx;
            for (int i = 0; i < n; i += 8) {
                const v8 b0 = *(const v8 *)(w0 + i), b1 = *(const v8 *)(w1 + i), b2 = *(const v8 *)(w2 + i), b3 = *(const v8 *)(w3 + i);
                *(v8 *)(d0 + i) += c00 * b0 + c01 * b1 + c02 * b2 + c03 * b3;
                *(v8 *)(d1 + i) += c10 * b0 + c11 * b1 + c12 * b2 + c13 * b3;
            }
        }
        for (; t < T; t++) {
            const float *g0 = dY + (size_t)t * ldy + o0 + r;
            float *d0 = dX + (size_t)t * ldx;
            for (int i = 0; i < n; i += 8)
                *(v8 *)(d0 + i) += g0[0] * *(const v8 *)(w0 + i) + g0[1] * *(const v8 *)(w1 + i) + g0[2] * *(const v8 *)(w2 + i) + g0[3] * *(const v8 *)(w3 + i);
        }
    }
    for (; r < nr; r++) {
        const float *w0 = W + (size_t)r * n;
        for (int t = 0; t < T; t++) {
            const float c = dY[(size_t)t * ldy + o0 + r];
            if (c == 0) continue;
            float *d0 = dX + (size_t)t * ldx;
            for (int i = 0; i < n; i += 8) *(v8 *)(d0 + i) += c * *(const v8 *)(w0 + i);
        }
    }
}
static void fwd_generic(const float *X, int T, size_t ldx, const float *W, int nr, int n, float *Y, size_t ldy, int o0) { fwd_body(X, T, ldx, W, nr, n, Y, ldy, o0); }
static void bwd_generic(const float *dY, int T, size_t ldy, int o0, const float *W, int nr, int n, float *dX, size_t ldx) { bwd_body(dY, T, ldy, o0, W, nr, n, dX, ldx); }
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
__attribute__((target("avx2,fma"))) static void fwd_avx2(const float *X, int T, size_t ldx, const float *W, int nr, int n, float *Y, size_t ldy, int o0) { fwd_body(X, T, ldx, W, nr, n, Y, ldy, o0); }
__attribute__((target("avx2,fma"))) static void bwd_avx2(const float *dY, int T, size_t ldy, int o0, const float *W, int nr, int n, float *dX, size_t ldx) { bwd_body(dY, T, ldy, o0, W, nr, n, dX, ldx); }
#endif
typedef void (*fwd_k)(const float *, int, size_t, const float *, int, int, float *, size_t, int);
typedef void (*bwd_k)(const float *, int, size_t, int, const float *, int, int, float *, size_t);
static fwd_k FWD; static bwd_k BWD;
static void pick_kernels(void) {
    if (FWD) return;
    FWD = fwd_generic; BWD = bwd_generic;
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) { FWD = fwd_avx2; BWD = bwd_avx2; }
#endif
}

static void dequant_rows(const tw *w, int r0, int nr, float *dst) {
    for (int r = 0; r < nr; r++) es_dequant_row(w->type, w->data + w->rb * (size_t)(r0 + r), dst + (size_t)r * w->cols, w->cols);
}

/* ---------------------------------------------------------------- blocked GEMM
 * C (M x N, stride ldc) = beta*C + alpha * A B, with A element (i, k) at A[i*ai + k*ak] (so A or A^T) and B row-major
 * (K x N, stride ldb, N a multiple of 32). Register tiles: MR tokens x NR columns, the outer product of one broadcast A value
 * and NR/VL vectors of B per k. AVX-512: 8 x 32 (16 zmm accumulators); AVX2 and others (NEON on phones): 6 x 16.
 * Each C element is summed by one thread in a fixed order, so results do not depend on the thread count. */
typedef float v16 __attribute__((vector_size(64), aligned(4)));
#define GEMM_KERNEL(NAME, VT, VL, MR, ATTR) \
ATTR static void NAME(int mr, int K, const float *A, size_t ai, size_t ak, const float *B, size_t ldb, float *C, size_t ldc, float alpha, int beta) { \
    VT c0[MR], c1[MR]; \
    for (int i = 0; i < MR; i++) { c0[i] = (VT){0}; c1[i] = (VT){0}; } \
    if (mr == MR) { \
        for (int k = 0; k < K; k++) { \
            const VT b0 = *(const VT *)(B + (size_t)k * ldb), b1 = *(const VT *)(B + (size_t)k * ldb + VL); const float *a = A + (size_t)k * ak; \
            for (int i = 0; i < MR; i++) { const float s = a[i * ai]; c0[i] += s * b0; c1[i] += s * b1; } \
        } \
    } else { \
        for (int k = 0; k < K; k++) { \
            const VT b0 = *(const VT *)(B + (size_t)k * ldb), b1 = *(const VT *)(B + (size_t)k * ldb + VL); const float *a = A + (size_t)k * ak; \
            for (int i = 0; i < mr; i++) { const float s = a[i * ai]; c0[i] += s * b0; c1[i] += s * b1; } \
        } \
    } \
    for (int i = 0; i < mr; i++) { \
        VT *c = (VT *)(C + (size_t)i * ldc); \
        if (beta) { c[0] += alpha * c0[i]; c[1] += alpha * c1[i]; } else { c[0] = alpha * c0[i]; c[1] = alpha * c1[i]; } \
    } \
}
/* 16 columns (the LoRA rank): one vector per row, B read in place (row-major, K x 16) */
#define GEMM_KERNEL16(NAME, VT, MR, ATTR) \
ATTR static void NAME(int mr, int K, const float *A, size_t ai, size_t ak, const float *B, size_t ldb, float *C, size_t ldc, float alpha, int beta) { \
    VT c0[MR]; \
    for (int i = 0; i < MR; i++) c0[i] = (VT){0}; \
    if (mr == MR) { for (int k = 0; k < K; k++) { const VT b0 = *(const VT *)(B + (size_t)k * ldb); const float *a = A + (size_t)k * ak; for (int i = 0; i < MR; i++) c0[i] += a[i * ai] * b0; } } \
    else { for (int k = 0; k < K; k++) { const VT b0 = *(const VT *)(B + (size_t)k * ldb); const float *a = A + (size_t)k * ak; for (int i = 0; i < mr; i++) c0[i] += a[i * ai] * b0; } } \
    for (int i = 0; i < mr; i++) { VT *c = (VT *)(C + (size_t)i * ldc); if (beta) *c += alpha * c0[i]; else *c = alpha * c0[i]; } \
}
GEMM_KERNEL(gk_generic, v8, 8, 6, )
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
GEMM_KERNEL(gk_avx2, v8, 8, 6, __attribute__((target("avx2,fma"))))
GEMM_KERNEL(gk_avx512, v16, 16, 8, __attribute__((target("avx512f"))))
GEMM_KERNEL16(gk16_avx512, v16, 12, __attribute__((target("avx512f"))))
#endif
typedef void (*gemm_k)(int, int, const float *, size_t, size_t, const float *, size_t, float *, size_t, float, int);
static gemm_k GK, GK16; static int GMR, GNR, GMR16;
static int gemm_on(void) {   /* ES_GEMM=0: the older dot-product kernels, for comparison */
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("ES_GEMM"); on = !(e && !atoi(e));
        GK = gk_generic; GMR = 6; GNR = 16; GK16 = gk_generic; GMR16 = 6;   /* NR 16 kernels serve 16 columns as they are */
#if defined(__x86_64__) && (defined(__clang__) || defined(__GNUC__))
        __builtin_cpu_init();
        if (__builtin_cpu_supports("avx512f") && !getenv("ES_NO_AVX512")) { GK = gk_avx512; GMR = 8; GNR = 32; GK16 = gk16_avx512; GMR16 = 12; }
        else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) GK = GK16 = gk_avx2;
#endif
    }
    return on;
}
#define GMB 64    /* C rows per job */
#define GNB 128   /* C columns per job */
#ifndef GKB
#define GKB 64    /* terms summed in registers before they are added into C */
#endif
/* B is packed in panels of 32 columns, each K x 32 and contiguous: element (k, j) at B[(j / 32) * K * 32 + k * 32 + j % 32].
 * Read row by row with plain strides, a weight panel spans megabytes and every k lands on a new page; packed, it streams. */
typedef struct { int M, N, K; const float *A; size_t ai, ak; const float *B; float *C; size_t ldc; float alpha; int beta, nbn; } gm_ctx;
static void gm_job(void *c_, int item, int tid) {
    (void)tid; const gm_ctx *c = c_;
    const int i0 = (item / c->nbn) * GMB, j0 = (item % c->nbn) * GNB;
    const int i1 = i0 + GMB < c->M ? i0 + GMB : c->M, j1 = j0 + GNB < c->N ? j0 + GNB : c->N;
    /* K in blocks of GKB: partial sums of GKB terms are added into C, which keeps float32 rounding close to that of
     * 8-lane dot products (one running sum over thousands of terms is noticeably less accurate) */
    for (int k0 = 0; k0 < c->K; k0 += GKB) {
        const int kn = c->K - k0 < GKB ? c->K - k0 : GKB, beta = k0 ? 1 : c->beta;
        for (int j = j0; j < j1; j += GNR) {
            const float *Bj = c->B + (size_t)(j / 32) * c->K * 32 + (size_t)k0 * 32 + j % 32;
            for (int i = i0; i < i1; i += GMR)
                GK(i1 - i < GMR ? i1 - i : GMR, kn, c->A + (size_t)i * c->ai + (size_t)k0 * c->ak, c->ai, c->ak, Bj, 32, c->C + (size_t)i * c->ldc + j, c->ldc, c->alpha, beta);
        }
    }
}
typedef struct { int M, K; const float *A; size_t ai, ak; const float *B; size_t ldb; float *C; size_t ldc; float alpha; int beta; } g16_ctx;
static void g16_job(void *c_, int item, int tid) {   /* 48 rows of C, 16 columns */
    (void)tid; const g16_ctx *c = c_; const int i0 = item * 48, i1 = i0 + 48 < c->M ? i0 + 48 : c->M;
    for (int k0 = 0; k0 < c->K; k0 += GKB) {
        const int kn = c->K - k0 < GKB ? c->K - k0 : GKB, beta = k0 ? 1 : c->beta;
        for (int i = i0; i < i1; i += GMR16)
            GK16(i1 - i < GMR16 ? i1 - i : GMR16, kn, c->A + (size_t)i * c->ai + (size_t)k0 * c->ak, c->ai, c->ak, c->B + (size_t)k0 * c->ldb, c->ldb, c->C + (size_t)i * c->ldc, c->ldc, c->alpha, beta);
    }
}
static void gemm_packed(es_tm *m, int M, int N, int K, const float *A, size_t ai, size_t ak, const float *Bp, float *C, size_t ldc, float alpha, int beta) {
    gm_ctx c = {M, N, K, A, ai, ak, Bp, C, ldc, alpha, beta, (N + GNB - 1) / GNB};
    pool_run(&m->P, ((M + GMB - 1) / GMB) * c.nbn, gm_job, &c);
}
static float *GPAD; static size_t GPAD_N;
static float *gbuf(float **p, size_t *cap, size_t n) { if (n > *cap) { free(*p); *p = malloc(sizeof(float) * n); *cap = *p ? n : 0; } if (!*p) { fprintf(stderr, "out of memory\n"); exit(1); } return *p; }
typedef struct { const float *B; size_t ldb; int K, N; float *dst; } pk_ctx;
static void pk_job(void *c_, int item, int tid) {   /* panel item of row-major B, zero beyond N */
    (void)tid; const pk_ctx *c = c_; const int j0 = item * 32, w = c->N - j0 < 32 ? c->N - j0 : 32;
    float *d = c->dst + (size_t)item * c->K * 32;
    for (int k = 0; k < c->K; k++) { memcpy(d + (size_t)k * 32, c->B + (size_t)k * c->ldb + j0, sizeof(float) * w); if (w < 32) memset(d + (size_t)k * 32 + w, 0, sizeof(float) * (32 - w)); }
}
/* C (M x N) = beta C + alpha A B with B row-major (K x N, stride ldb): packs B, then the blocked product */
static void gemm(es_tm *m, int M, int N, int K, const float *A, size_t ai, size_t ak, const float *B, size_t ldb, float *C, size_t ldc, float alpha, int beta) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) { if (!beta) for (int i = 0; i < M; i++) memset(C + (size_t)i * ldc, 0, sizeof(float) * N); return; }
    if (N == 16) { g16_ctx c = {M, K, A, ai, ak, B, ldb, C, ldc, alpha, beta}; pool_run(&m->P, (M + 47) / 48, g16_job, &c); return; }
    const int Np = (N + 31) / 32 * 32;
    float *Bp = gbuf(&GPAD, &GPAD_N, (size_t)K * Np + (N == Np ? 0 : (size_t)M * Np));
    pk_ctx pc = {B, ldb, K, N, Bp};
    pool_run(&m->P, Np / 32, pk_job, &pc);
    if (N == Np) { gemm_packed(m, M, N, K, A, ai, ak, Bp, C, ldc, alpha, beta); return; }
    float *Cp = Bp + (size_t)K * Np;   /* N not a multiple of 32 (the LoRA rank): through a padded copy of C */
    gemm_packed(m, M, Np, K, A, ai, ak, Bp, Cp, (size_t)Np, alpha, 0);
    for (int i = 0; i < M; i++) {
        float *d = C + (size_t)i * ldc; const float *s2 = Cp + (size_t)i * Np;
        if (beta) for (int j = 0; j < N; j++) d[j] += s2[j]; else memcpy(d, s2, sizeof(float) * N);
    }
}

/* Frozen weights for the GEMM: a chunk of rows dequantized once, packed, and shared by all threads. */
static float *WCH; static size_t WCH_N;
static int chunk_rows(const tw *w) {   /* about 2 MB of fp32 weights per chunk, a multiple of 32 rows */
    int rc = (int)((2u << 20) / (sizeof(float) * (size_t)w->cols)) / 32 * 32;
    if (rc < 32) rc = 32;
    return rc > w->rows ? (w->rows + 31) / 32 * 32 : rc;
}
typedef struct { const tw *w; int r0, nr, fwd; float *dst; float **scr; } wc_ctx;
static void wc_job(void *c_, int item, int tid) {   /* 32 rows of the chunk: dequantize, then place into the packed panels */
    const wc_ctx *c = c_; const int a = item * 32, n = c->nr - a < 32 ? c->nr - a : 32, K = c->w->cols;
    float *t = c->scr[tid];
    dequant_rows(c->w, c->r0 + a, n, t);
    if (c->fwd) {   /* B = W^T (cols x rows): these 32 rows are panel 'item', element (k, r) */
        float *d = c->dst + (size_t)item * K * 32;
        for (int k = 0; k < K; k++) { float *dk = d + (size_t)k * 32; for (int r = 0; r < n; r++) dk[r] = t[(size_t)r * K + k]; for (int r = n; r < 32; r++) dk[r] = 0; }
    } else {        /* B = W (rows x cols): row a + r of the chunk is k, its columns spread over the panels */
        for (int p = 0; p < K / 32; p++) { float *d = c->dst + (size_t)p * c->nr * 32 + (size_t)a * 32; for (int r = 0; r < n; r++) memcpy(d + (size_t)r * 32, t + (size_t)r * K + p * 32, 128); }
    }
}
/* Packed weights are kept for the next pass while they fit in ES_WCACHE_MB (default: a quarter of the RAM): the
 * weights are frozen, so dequantizing and packing them again every pass is wasted work. Chunk r0 sits at r0 * cols. */
typedef struct { const void *key; int fwd; float *p; } wcache_ent;
static wcache_ent WCE[4096]; static int WCEN; static size_t WC_USED;
static size_t wc_budget(void) {
    static size_t b = (size_t)-1;
    if (b == (size_t)-1) {
        const char *e = getenv("ES_WCACHE_MB");
        if (e) b = (size_t)atol(e) << 20;
        else { const long pg = sysconf(_SC_PHYS_PAGES), ps = sysconf(_SC_PAGESIZE); b = pg > 0 && ps > 0 ? (size_t)pg * (size_t)ps / 4 : 0; }
    }
    return b;
}
/* the packed copy of w for this direction: cached (filled on first use), or NULL to pack chunk by chunk */
static float *wcache(const tw *w, int fwd, int *fresh) {
    *fresh = 0;
    for (int i = 0; i < WCEN; i++) if (WCE[i].key == w->data && WCE[i].fwd == fwd) return WCE[i].p;
    const size_t n = (size_t)((w->rows + 31) / 32 * 32) * w->cols * sizeof(float);
    if (WCEN >= 4096 || WC_USED + n > wc_budget()) return NULL;
    float *p = malloc(n); if (!p) return NULL;
    WCE[WCEN++] = (wcache_ent){w->data, fwd, p}; WC_USED += n; *fresh = 1;
    return p;
}
/* Y (T x rows, stride ldy) = X (T x cols) W^T */
static void gemm_fwd(es_tm *m, const tw *w, const float *X, int T, float *Y, size_t ldy) {
    const int K = w->cols, rc = chunk_rows(w);
    int fresh; float *cache = wcache(w, 1, &fresh), *Wt = cache ? NULL : gbuf(&WCH, &WCH_N, (size_t)rc * K);
    for (int r0 = 0; r0 < w->rows; r0 += rc) {
        const int nr = w->rows - r0 < rc ? w->rows - r0 : rc, np = (nr + 31) / 32 * 32;
        if (cache) Wt = cache + (size_t)r0 * K;
        if (!cache || fresh) { wc_ctx wc = {w, r0, nr, 1, Wt, m->scratch}; pool_run(&m->P, np / 32, wc_job, &wc); }
        if (np == nr) gemm_packed(m, T, nr, K, X, (size_t)K, 1, Wt, Y + r0, ldy, 1.0f, 0);
        else {   /* a row count that is not a multiple of 32: the padded columns go to a scratch copy */
            float *Yp = gbuf(&GPAD, &GPAD_N, (size_t)T * np);
            gemm_packed(m, T, np, K, X, (size_t)K, 1, Wt, Yp, (size_t)np, 1.0f, 0);
            for (int t = 0; t < T; t++) memcpy(Y + (size_t)t * ldy + r0, Yp + (size_t)t * np, sizeof(float) * nr);
        }
    }
    m->bytes += (double)w->nbytes;
}
/* dX (T x cols) += dY (T x rows, stride ldy) W; cols must be a multiple of 32 (checked by the caller) */
static void gemm_bwd(es_tm *m, const tw *w, const float *dY, int T, size_t ldy, float *dX) {
    const int K = w->cols, rc = chunk_rows(w);
    int fresh; float *cache = wcache(w, 0, &fresh), *Wc = cache ? NULL : gbuf(&WCH, &WCH_N, (size_t)rc * K);
    for (int r0 = 0; r0 < w->rows; r0 += rc) {
        const int nr = w->rows - r0 < rc ? w->rows - r0 : rc;
        if (cache) Wc = cache + (size_t)r0 * K;
        if (!cache || fresh) { wc_ctx wc = {w, r0, nr, 0, Wc, m->scratch}; pool_run(&m->P, (nr + 31) / 32, wc_job, &wc); }
        gemm_packed(m, T, K, nr, dY + r0, ldy, 1, Wc, dX, (size_t)K, 1.0f, 1);
    }
    m->bytes += (double)w->nbytes;
}


/* Optional BLAS path: ES_BLAS=/path/libopenblas.so (cblas_sgemm, or numpy's scipy_cblas_sgemm64_). The weights of one
 * product are dequantized whole into a buffer, then one sgemm runs over the whole micro-batch. Summation order differs
 * from the built-in kernels, so losses match to rounding, not bit for bit. */
typedef void (*sgemm32_t)(int, int, int, int, int, int, float, const float *, int, const float *, int, float, float *, int);
typedef void (*sgemm64_t)(int, int, int, long, long, long, float, const float *, long, const float *, long, float, float *, long);
static int BLAS_STATE = -1; static sgemm32_t SG32; static sgemm64_t SG64;
static int blas_init(int threads) {
    if (BLAS_STATE >= 0) return BLAS_STATE;
    BLAS_STATE = 0;
    const char *path = getenv("ES_BLAS"); if (!path || !*path) return 0;
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL); if (!h) { fprintf(stderr, "ES_BLAS: %s\n", dlerror()); return 0; }
    SG32 = (sgemm32_t)dlsym(h, "cblas_sgemm"); if (!SG32) SG64 = (sgemm64_t)dlsym(h, "scipy_cblas_sgemm64_");
    void (*nt32)(int) = (void (*)(int))dlsym(h, "openblas_set_num_threads"); void (*nt64)(long) = (void (*)(long))dlsym(h, "scipy_openblas_set_num_threads64_");
    /* BLAS threads default to 1: its own threads fight the trainer's spinning pool and collapse speed on a busy machine */
    const char *bt = getenv("ES_BLAS_THREADS"); threads = bt ? atoi(bt) : 1; if (threads < 1) threads = 1;
    if (nt32) nt32(threads); else if (nt64) nt64(threads);
    BLAS_STATE = SG32 || SG64; if (BLAS_STATE) fprintf(stderr, "ES_BLAS: using %s\n", path);
    return BLAS_STATE;
}
static void sgemm(int tb, int M, int N, int K, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    /* row-major C (M x N) = A (M x K) * op(B) (+ beta C); tb: B is N x K (transposed) */
    if (SG32) SG32(101, 111, tb ? 112 : 111, M, N, K, 1.0f, A, lda, B, ldb, beta, C, ldc);
    else SG64(101, 111, tb ? 112 : 111, M, N, K, 1.0f, A, lda, B, ldb, beta, C, ldc);
}
static void sgemm2(int ta, int tb, int M, int N, int K, float al, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    if (SG32) SG32(101, ta ? 112 : 111, tb ? 112 : 111, M, N, K, al, A, lda, B, ldb, beta, C, ldc);
    else SG64(101, ta ? 112 : 111, tb ? 112 : 111, M, N, K, al, A, lda, B, ldb, beta, C, ldc);
}
static int lora_blas(es_tm *m) { static int on = -1; if (on < 0) on = getenv("ES_LORA_BLAS") && atoi(getenv("ES_LORA_BLAS")) > 0; return on && m->nt == 1 && blas_init(1); }
typedef struct { const tw *w; float *dst; } dq_ctx;
static void dq_job(void *c_, int item, int tid) { (void)tid; dq_ctx *c = c_; const int r0 = item * 64, nr = c->w->rows - r0 < 64 ? c->w->rows - r0 : 64; dequant_rows(c->w, r0, nr, c->dst + (size_t)r0 * c->w->cols); }
static float *WBUF; static size_t WBUF_N;
static const float *blas_weights(es_tm *m, const tw *w) {
    const size_t n = (size_t)w->rows * w->cols;
    if (n > WBUF_N) { free(WBUF); WBUF = malloc(sizeof(float) * n); WBUF_N = WBUF ? n : 0; if (!WBUF) return NULL; }
    dq_ctx c = {w, WBUF}; pool_run(&m->P, (w->rows + 63) / 64, dq_job, &c);
    return WBUF;
}

/* ES_AMX=1: frozen-weight products on AMX bf16 tiles (fp32 accumulate). Each weight is packed to bf16 once per layout
 * (forward: W^T, backward: W) and kept: 2 bytes per weight per layout. Lossy (bf16 inputs); off by default. */
typedef struct { const void *key; int bt; uint16_t *p; } amx_ent;
static amx_ent AMXC[1024]; static int AMXN; static int AMX_ON = -1;
static int amx_on(void) { if (AMX_ON < 0) { const char *e = getenv("ES_AMX"); AMX_ON = e && atoi(e) && es_amx_available(); if (AMX_ON) fprintf(stderr, "ES_AMX: bf16 tiles\n"); } return AMX_ON; }
static const uint16_t *amx_packed(const tw *w, int bt) {
    for (int i = 0; i < AMXN; i++) if (AMXC[i].key == w->data && AMXC[i].bt == bt) return AMXC[i].p;
    if (AMXN >= 1024) return NULL;
    const int K = bt ? w->cols : w->rows, N = bt ? w->rows : w->cols, chunk = bt ? 16 : 32;
    uint16_t *p = aligned_alloc(64, (size_t)K * N * 2); float *part = malloc(sizeof(float) * (size_t)chunk * w->cols);
    if (!p || !part) { free(p); free(part); return NULL; }
    for (int r0 = 0; r0 < w->rows; r0 += chunk) { dequant_rows(w, r0, chunk, part); es_amx_pack_part(p, part, K, N, bt, r0); }
    free(part);
    AMXC[AMXN++] = (amx_ent){w->data, bt, p};
    return p;
}
static void weights_cache_free(void) {
    for (int i = 0; i < WCEN; i++) free(WCE[i].p);
    WCEN = 0; WC_USED = 0;
    for (int i = 0; i < AMXN; i++) free(AMXC[i].p);
    AMXN = 0;
}
typedef struct { const float *A; int lda; const uint16_t *B; int K, N; float *C; int ldc, M, beta; } amx_ctx;
static void amx_job(void *c_, int item, int tid) { (void)tid; amx_ctx *c = c_; const int m0 = item * 32, m1 = m0 + 32 < c->M ? m0 + 32 : c->M; es_amx_gemm(c->A, c->lda, c->B, c->K, c->N, c->C, c->ldc, m0, m1, c->beta); }
static int amx_mm(es_tm *m, const tw *w, int bt, const float *A, int lda, int T, float *C, int ldc, int beta) {
    if (!amx_on() || w->rows % 32 || w->cols % 32 || T < 16) return 0;
    const uint16_t *B = amx_packed(w, bt); if (!B) return 0;
    amx_ctx c = {A, lda, B, bt ? w->cols : w->rows, bt ? w->rows : w->cols, C, ldc, T, beta};
    pool_run(&m->P, (T + 31) / 32, amx_job, &c);
    m->bytes += (double)w->nbytes; return 1;
}
/* Y (T x rows, row stride ldy) = X (T x cols) W^T */
typedef struct { es_tm *m; const tw *w; const float *X; int T; float *Y; size_t ldy; } mmf_ctx;
static void mmf_job(void *c_, int item, int tid) {
    mmf_ctx *c = c_;
    const int r0 = item * RB, nr = c->w->rows - r0 < RB ? c->w->rows - r0 : RB;
    float *Wb = c->m->scratch[tid];
    dequant_rows(c->w, r0, nr, Wb);
    for (int t0 = 0; t0 < c->T; t0 += TT) {
        const int nt = c->T - t0 < TT ? c->T - t0 : TT;
        FWD(c->X + (size_t)t0 * c->w->cols, nt, (size_t)c->w->cols, Wb, nr, c->w->cols, c->Y + (size_t)t0 * c->ldy, c->ldy, r0);
    }
}
static void mm_fwd(es_tm *m, const tw *w, const float *X, int T, float *Y, size_t ldy) {
    if (amx_mm(m, w, 1, X, w->cols, T, Y, (int)ldy, 0)) return;
    if (gemm_on() && !blas_init(m->nt)) { gemm_fwd(m, w, X, T, Y, ldy); return; }
    const float *Wf;
    if (T >= 16 && blas_init(m->nt) && (Wf = blas_weights(m, w))) {
        sgemm(1, T, w->rows, w->cols, X, w->cols, Wf, w->cols, 0.0f, Y, (int)ldy); m->bytes += (double)w->nbytes; return;
    }
    mmf_ctx c = {m, w, X, T, Y, ldy};
    pool_run(&m->P, (w->rows + RB - 1) / RB, mmf_job, &c);
    m->bytes += (double)w->nbytes;
}

/* dX (T x cols) += dY (T x rows, stride ldy) W */
typedef struct { es_tm *m; const tw *w; const float *dY; int T; size_t ldy; float *dX; } mmb_ctx;
static void mmb_tok_job(void *c_, int item, int tid) {   /* a tile of tokens against every row */
    mmb_ctx *c = c_;
    const int t0 = item * TB, nt = c->T - t0 < TB ? c->T - t0 : TB, n = c->w->cols;
    float *Wb = c->m->scratch[tid];
    for (int r0 = 0; r0 < c->w->rows; r0 += RBB) {
        const int nr = c->w->rows - r0 < RBB ? c->w->rows - r0 : RBB;
        int any = 0;
        for (int t = 0; t < nt && !any; t++) for (int r = 0; r < nr; r++) if (c->dY[(size_t)(t0 + t) * c->ldy + r0 + r] != 0) { any = 1; break; }
        if (!any) continue;
        dequant_rows(c->w, r0, nr, Wb);
        BWD(c->dY + (size_t)t0 * c->ldy, nt, c->ldy, r0, Wb, nr, n, c->dX + (size_t)t0 * n, (size_t)n);
    }
}
typedef struct { es_tm *m; const tw *w; const float *dY; int T; size_t ldy; float **acc; int nblk; } mmr_ctx;
static void mmb_row_job(void *c_, int item, int tid) {   /* a block of rows against every token, into this thread's accumulator */
    mmr_ctx *c = c_;
    const int r0 = item * RBB, nr = c->w->rows - r0 < RBB ? c->w->rows - r0 : RBB, n = c->w->cols;
    float *Wb = c->m->scratch[tid] + (size_t)c->T * n;
    dequant_rows(c->w, r0, nr, Wb);
    BWD(c->dY, c->T, c->ldy, r0, Wb, nr, n, c->acc[tid], (size_t)n);
}
static void mm_bwd(es_tm *m, const tw *w, const float *dY, int T, size_t ldy, float *dX) {
    if (amx_mm(m, w, 0, dY, (int)ldy, T, dX, w->cols, 1)) return;
    if (gemm_on() && !blas_init(m->nt) && w->cols % 32 == 0) { gemm_bwd(m, w, dY, T, ldy, dX); return; }
    const float *Wf;
    if (T >= 16 && blas_init(m->nt) && (Wf = blas_weights(m, w))) {
        sgemm(0, T, w->cols, w->rows, dY, (int)ldy, Wf, w->cols, 1.0f, dX, w->cols); m->bytes += (double)w->nbytes; return;
    }
    if (T >= TB * m->nt) {
        mmb_ctx c = {m, w, dY, T, ldy, dX};
        pool_run(&m->P, (T + TB - 1) / TB, mmb_tok_job, &c);
    } else {   /* few tokens (the output layer's tiles): split the rows, each thread sums into its own copy of dX */
        const int n = w->cols;
        float *acc[256]; const int nt = m->nt < 256 ? m->nt : 256;
        for (int i = 0; i < nt; i++) { acc[i] = m->scratch[i]; memset(acc[i], 0, sizeof(float) * (size_t)T * n); }
        mmr_ctx c = {m, w, dY, T, ldy, acc, 0};
        pool_run(&m->P, (w->rows + RBB - 1) / RBB, mmb_row_job, &c);
        for (int i = 0; i < nt; i++) for (size_t k = 0; k < (size_t)T * n; k++) dX[k] += acc[i][k];
    }
    m->bytes += (double)w->nbytes;
}

/* ---------------------------------------------------------------- LoRA */
static size_t *lo(const es_tm *m, int l, int t) { return &m->off[((size_t)l * ES_LT_N + t) * 2]; }
size_t es_tm_lora_count(const es_tm *m) { return m->ln; }
float *es_tm_lora_params(es_tm *m) { return m->lp; }
float *es_tm_lora_grads(es_tm *m) { return m->lg; }
int es_tm_lora_where(const es_tm *m, size_t i, int *layer, int *target, int *is_b) {
    for (int l = 0; l < m->L; l++) for (int t = 0; t < ES_LT_N; t++) {
        const size_t *o = &m->off[((size_t)l * ES_LT_N + t) * 2];
        if (o[0] == SIZE_MAX) continue;
        const size_t end = o[1] + (size_t)out_of(m, t) * m->rank;
        if (i >= o[0] && i < end) { *layer = l; *target = t; *is_b = i >= o[1]; return 0; }
    }
    return -1;
}
void es_tm_lora_info(const es_tm *m, int *rank, float *alpha, uint32_t *mask) { if (rank) *rank = m->rank; if (alpha) *alpha = m->alpha; if (mask) *mask = m->mask; }

static uint64_t splitmix(uint64_t *s) { uint64_t z = (*s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
static int lora_layout(es_tm *m, int rank, float alpha, uint32_t mask) {
    if (rank < 1 || rank > 1024 || !(mask & ES_LT_ALL)) return -1;
    free(m->lp); free(m->lg); free(m->off);
    m->rank = rank; m->alpha = alpha > 0 ? alpha : (float)rank; m->ls = m->alpha / rank; m->mask = mask & ES_LT_ALL;
    m->off = malloc(sizeof(size_t) * (size_t)m->L * ES_LT_N * 2);
    size_t n = 0;
    for (int l = 0; l < m->L; l++)
        for (int t = 0; t < ES_LT_N; t++) {
            size_t *o = lo(m, l, t);
            if (!(m->mask >> t & 1)) { o[0] = o[1] = SIZE_MAX; continue; }
            o[0] = n; n += (size_t)rank * in_of(m, t);
            o[1] = n; n += (size_t)out_of(m, t) * rank;
        }
    m->ln = n; m->lp = calloc(n, sizeof(float)); m->lg = calloc(n, sizeof(float));
    m->Tcap = 0; free_bufs(m);   /* Z buffers depend on the rank */
    return m->lp && m->lg ? 0 : -1;
}
int es_tm_lora_new(es_tm *m, int rank, float alpha, uint32_t mask, uint64_t seed) {
    if (lora_layout(m, rank, alpha, mask)) return -1;
    uint64_t s = seed ? seed : 1;
    for (int l = 0; l < m->L; l++)
        for (int t = 0; t < ES_LT_N; t++) {
            const size_t *o = lo(m, l, t);
            if (o[0] == SIZE_MAX) continue;
            const int in = in_of(m, t);
            const float b = 1.0f / sqrtf((float)in);   /* uniform in [-1/sqrt(in), 1/sqrt(in)] (PyTorch's LoRA init for A); B starts at zero */
            for (size_t i = 0; i < (size_t)rank * in; i++) m->lp[o[0] + i] = b * (2.0f * (float)(splitmix(&s) >> 40) / (float)(1 << 24) - 1.0f);
        }
    return 0;
}

static uint64_t fingerprint(const es_tm *m) {
    uint64_t h = 1469598103934665603ull; const int v[] = {m->L, m->d, m->H, m->KV, m->hd, m->ff, m->V};
    for (size_t i = 0; i < sizeof v / sizeof *v; i++) { h ^= (uint64_t)v[i]; h *= 1099511628211ull; }
    for (const char *p = m->arch; *p; p++) { h ^= (uint8_t)*p; h *= 1099511628211ull; }
    return h;
}
typedef struct { char magic[8]; uint32_t version, n_layer, rank, mask; float alpha; uint32_t pad; uint64_t steps, fp, count; } lora_head;
int es_tm_lora_save(const es_tm *m, const char *path, uint64_t steps, char *err, size_t cap) {
    char tmp[4096]; snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) { snprintf(err, cap, "cannot write %s", tmp); return -1; }
    lora_head h = {{'E', 'S', 'L', 'O', 'R', 'A', '0', '1'}, 1, (uint32_t)m->L, (uint32_t)m->rank, m->mask, m->alpha, 0, steps, fingerprint(m), m->ln};
    const int ok = fwrite(&h, sizeof h, 1, f) == 1 && fwrite(m->lp, sizeof(float), m->ln, f) == m->ln;
    if (fclose(f) || !ok || rename(tmp, path)) { snprintf(err, cap, "cannot write %s", path); return -1; }
    return 0;
}
int es_tm_lora_load(es_tm *m, const char *path, uint64_t *steps, char *err, size_t cap) {
    FILE *f = fopen(path, "rb");
    if (!f) { snprintf(err, cap, "cannot open %s", path); return -1; }
    lora_head h;
    if (fread(&h, sizeof h, 1, f) != 1 || memcmp(h.magic, "ESLORA01", 8)) { fclose(f); snprintf(err, cap, "%s is not an ExpertStream LoRA", path); return -1; }
    if (h.fp != fingerprint(m)) { fclose(f); snprintf(err, cap, "%s was trained for a different model", path); return -1; }
    if (lora_layout(m, (int)h.rank, h.alpha, h.mask) || m->ln != h.count || fread(m->lp, sizeof(float), m->ln, f) != m->ln) { fclose(f); snprintf(err, cap, "%s is damaged", path); return -1; }
    fclose(f);
    if (steps) *steps = h.steps;
    return 0;
}

/* llama.cpp LoRA adapter: lora_a ne = [in, rank] (rank rows), lora_b ne = [rank, out] (out rows) */
static void wstr(FILE *f, const char *s) { const uint64_t n = strlen(s); fwrite(&n, 8, 1, f); fwrite(s, 1, n, f); }
int es_tm_lora_export_gguf(const es_tm *m, const char *path, char *err, size_t cap) {
    static const char *wn[ES_LT_N] = {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"};
    FILE *f = fopen(path, "wb");
    if (!f) { snprintf(err, cap, "cannot write %s", path); return -1; }
    uint64_t nt = 0; for (int l = 0; l < m->L; l++) for (int t = 0; t < ES_LT_N; t++) if (lo(m, l, t)[0] != SIZE_MAX) nt += 2;
    const uint32_t ver = 3, ts = GGUF_T_STRING, tf = GGUF_T_F32; const uint64_t nkv = 4;
    fwrite("GGUF", 1, 4, f); fwrite(&ver, 4, 1, f); fwrite(&nt, 8, 1, f); fwrite(&nkv, 8, 1, f);
    wstr(f, "general.architecture"); fwrite(&ts, 4, 1, f); wstr(f, m->arch);
    wstr(f, "general.type"); fwrite(&ts, 4, 1, f); wstr(f, "adapter");
    wstr(f, "adapter.type"); fwrite(&ts, 4, 1, f); wstr(f, "lora");
    wstr(f, "adapter.lora.alpha"); fwrite(&tf, 4, 1, f); fwrite(&m->alpha, 4, 1, f);
    uint64_t off = 0; char name[128];
    for (int l = 0; l < m->L; l++)
        for (int t = 0; t < ES_LT_N; t++) {
            if (lo(m, l, t)[0] == SIZE_MAX) continue;
            for (int ab = 0; ab < 2; ab++) {
                snprintf(name, sizeof name, "blk.%d.%s.weight.lora_%c", l, wn[t], ab ? 'b' : 'a');
                wstr(f, name);
                const uint32_t nd = 2, ty = 0;
                const uint64_t ne[2] = {ab ? (uint64_t)m->rank : (uint64_t)in_of(m, t), ab ? (uint64_t)out_of(m, t) : (uint64_t)m->rank};
                fwrite(&nd, 4, 1, f); fwrite(ne, 8, 2, f); fwrite(&ty, 4, 1, f); fwrite(&off, 8, 1, f);
                off += (ne[0] * ne[1] * 4 + 31) / 32 * 32;
            }
        }
    long pos = ftell(f); static const uint8_t z[32] = {0};
    fwrite(z, 1, (size_t)((32 - pos % 32) % 32), f);
    for (int l = 0; l < m->L; l++)
        for (int t = 0; t < ES_LT_N; t++) {
            const size_t *o = lo(m, l, t);
            if (o[0] == SIZE_MAX) continue;
            const size_t na = (size_t)m->rank * in_of(m, t), nb = (size_t)out_of(m, t) * m->rank;
            fwrite(m->lp + o[0], 4, na, f); fwrite(z, 1, (32 - na * 4 % 32) % 32, f);
            fwrite(m->lp + o[1], 4, nb, f); fwrite(z, 1, (32 - nb * 4 % 32) % 32, f);
        }
    if (fclose(f)) { snprintf(err, cap, "cannot write %s", path); return -1; }
    return 0;
}

static float *LT; static size_t LT_N;
/* forward part of target t: Z = X A^T (T x r); Y += ls * Z B^T */
typedef struct { es_tm *m; const float *X, *A, *B; float *Z, *Y; int T, in, out; size_t ldy; } lf_ctx;
static void lf_job(void *c_, int item, int tid) {
    (void)tid; lf_ctx *c = c_; const int r = c->m->rank;
    if (r % 8 == 0 && c->in % 8 == 0) {   /* SIMD path: Z = X A^T and Y += ls Z B^T with the matrix kernels */
        const int t0 = item * 16, nt = c->T - t0 < 16 ? c->T - t0 : 16;
        float *tmp = malloc(sizeof(float) * (size_t)nt * c->out);
        if (tmp) {
            FWD(c->X + (size_t)t0 * c->in, nt, (size_t)c->in, c->A, r, c->in, c->Z + (size_t)t0 * r, (size_t)r, 0);
            FWD(c->Z + (size_t)t0 * r, nt, (size_t)r, c->B, c->out, r, tmp, (size_t)c->out, 0);
            const float ls = c->m->ls;
            for (int t = 0; t < nt; t++) {
                float *y = c->Y + (size_t)(t0 + t) * c->ldy; const float *q = tmp + (size_t)t * c->out;
                for (int o = 0; o < c->out; o++) y[o] += ls * q[o];
            }
            free(tmp); return;
        }
    }
    for (int t = item * 16; t < c->T && t < item * 16 + 16; t++) {
        const float *x = c->X + (size_t)t * c->in; float *z = c->Z + (size_t)t * r;
        for (int k = 0; k < r; k++) { const float *a = c->A + (size_t)k * c->in; float s = 0; for (int i = 0; i < c->in; i++) s += a[i] * x[i]; z[k] = s; }
        float *y = c->Y + (size_t)t * c->ldy;
        for (int o = 0; o < c->out; o++) { const float *b = c->B + (size_t)o * r; float s = 0; for (int k = 0; k < r; k++) s += b[k] * z[k]; y[o] += c->m->ls * s; }
    }
}
static void lora_fwd(es_tm *m, int l, int t, const float *X, int T, float *Y, size_t ldy) {
    const size_t *o = lo(m, l, t);
    if (!m->lp || o[0] == SIZE_MAX) return;
    lf_ctx c = {m, X, m->lp + o[0], m->lp + o[1], m->Z[t], Y, T, in_of(m, t), out_of(m, t), ldy};
    if (gemm_on() && !lora_blas(m)) {   /* Z = X A^T; Y += ls Z B^T (A: r x in, B: out x r, both transposed into LT first) */
        const int r = m->rank;
        float *At = gbuf(&LT, &LT_N, (size_t)c.in * r + (size_t)r * c.out);
        float *Bt = At + (size_t)c.in * r;
        for (int k = 0; k < r; k++) for (int i = 0; i < c.in; i++) At[(size_t)i * r + k] = c.A[(size_t)k * c.in + i];
        for (int o2 = 0; o2 < c.out; o2++) for (int k = 0; k < r; k++) Bt[(size_t)k * c.out + o2] = c.B[(size_t)o2 * r + k];
        gemm(m, T, r, c.in, X, (size_t)c.in, 1, At, (size_t)r, c.Z, (size_t)r, 1.0f, 0);
        gemm(m, T, c.out, r, c.Z, (size_t)r, 1, Bt, (size_t)c.out, Y, ldy, m->ls, 1);
        return;
    }
    if (lora_blas(m)) {   /* ES_LORA_BLAS=1: two sgemms instead of the per-16-token jobs */
        sgemm2(0, 1, T, m->rank, c.in, 1.0f, X, c.in, c.A, c.in, 0.0f, c.Z, m->rank);
        sgemm2(0, 1, T, c.out, m->rank, m->ls, c.Z, m->rank, c.B, m->rank, 1.0f, Y, (int)ldy);
        return;
    }
    pool_run(&m->P, (T + 15) / 16, lf_job, &c);
}
/* backward part: dZ = ls dY B; dB += ls dY^T Z; dA += dZ^T X; dX += dZ A */
typedef struct { es_tm *m; const float *X, *Z, *dY, *A, *B; float *dA, *dB, *dX, *dZ; int T, in, out; size_t ldy; } lb_ctx;
static void lb_dz_job(void *c_, int item, int tid) {
    (void)tid; lb_ctx *c = c_; const int r = c->m->rank;
    for (int t = item * 16; t < c->T && t < item * 16 + 16; t++) {
        const float *g = c->dY + (size_t)t * c->ldy; float *dz = c->dZ + (size_t)t * r;
        for (int k = 0; k < r; k++) dz[k] = 0;
        for (int o = 0; o < c->out; o++) { const float go = g[o]; if (go == 0) continue; const float *b = c->B + (size_t)o * r; for (int k = 0; k < r; k++) dz[k] += go * b[k]; }
        for (int k = 0; k < r; k++) dz[k] *= c->m->ls;
        float *dx = c->dX + (size_t)t * c->in;
        for (int k = 0; k < r; k++) { const float s = dz[k]; if (s == 0) continue; const float *a = c->A + (size_t)k * c->in; for (int i = 0; i < c->in; i++) dx[i] += s * a[i]; }
    }
}
static void lb_db_job(void *c_, int item, int tid) {
    (void)tid; lb_ctx *c = c_; const int r = c->m->rank;
    for (int o = item * 32; o < c->out && o < item * 32 + 32; o++) {
        float acc[1024] = {0};
        for (int t = 0; t < c->T; t++) { const float go = c->dY[(size_t)t * c->ldy + o]; if (go == 0) continue; const float *z = c->Z + (size_t)t * r; for (int k = 0; k < r; k++) acc[k] += go * z[k]; }
        float *db = c->dB + (size_t)o * r;
        for (int k = 0; k < r; k++) db[k] += c->m->ls * acc[k];
    }
}
static void lb_da_job(void *c_, int item, int tid) {
    (void)tid; lb_ctx *c = c_; const int r = c->m->rank;
    const int i0 = item * 256, i1 = i0 + 256 < c->in ? i0 + 256 : c->in;
    for (int k = 0; k < r; k++) {
        float *da = c->dA + (size_t)k * c->in;
        for (int t = 0; t < c->T; t++) {
            const float s = c->dZ[(size_t)t * r + k]; if (s == 0) continue;
            const float *x = c->X + (size_t)t * c->in;
            for (int i = i0; i < i1; i++) da[i] += s * x[i];
        }
    }
}
static void lora_bwd(es_tm *m, int l, int t, const float *X, int T, const float *dY, size_t ldy, float *dX) {
    const size_t *o = lo(m, l, t);
    if (!m->lp || o[0] == SIZE_MAX) return;
    lb_ctx c = {m, X, m->Z[t], dY, m->lp + o[0], m->lp + o[1], m->lg + o[0], m->lg + o[1], dX, m->dZ, T, in_of(m, t), out_of(m, t), ldy};
    if (gemm_on() && !lora_blas(m)) {
        const int r = m->rank;
        gemm(m, T, r, c.out, dY, ldy, 1, c.B, (size_t)r, c.dZ, (size_t)r, m->ls, 0);          /* dZ = ls dY B */
        gemm(m, c.out, r, T, dY, 1, ldy, c.Z, (size_t)r, c.dB, (size_t)r, m->ls, 1);          /* dB += ls dY^T Z */
        gemm(m, r, c.in, T, c.dZ, 1, (size_t)r, X, (size_t)c.in, c.dA, (size_t)c.in, 1.0f, 1); /* dA += dZ^T X */
        gemm(m, T, c.in, r, c.dZ, (size_t)r, 1, c.A, (size_t)c.in, dX, (size_t)c.in, 1.0f, 1); /* dX += dZ A */
        return;
    }
    if (lora_blas(m)) {
        const int r = m->rank;
        sgemm2(0, 0, T, r, c.out, m->ls, dY, (int)ldy, c.B, r, 0.0f, c.dZ, r);            /* dZ = ls dY B */
        sgemm2(1, 0, c.out, r, T, m->ls, dY, (int)ldy, c.Z, r, 1.0f, c.dB, r);            /* dB += ls dY^T Z */
        sgemm2(1, 0, r, c.in, T, 1.0f, c.dZ, r, X, c.in, 1.0f, c.dA, c.in);              /* dA += dZ^T X */
        sgemm2(0, 0, T, c.in, r, 1.0f, c.dZ, r, c.A, c.in, 1.0f, dX, c.in);              /* dX += dZ A */
        return;
    }
    pool_run(&m->P, (T + 15) / 16, lb_dz_job, &c);
    pool_run(&m->P, (c.out + 31) / 32, lb_db_job, &c);
    pool_run(&m->P, (c.in + 255) / 256, lb_da_job, &c);
}

/* ---------------------------------------------------------------- small pieces */
static float rms_row(float *o, const float *x, const float *w, int n, float eps) {   /* as the engine: double sum */
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float s = (float)(1.0 / sqrt(ss / n + eps));
    for (int i = 0; i < n; i++) o[i] = x[i] * s * (w ? w[i] : 1.0f);
    return s;
}
/* dx += d(x*r*w)/dx . dy, with r = rstd of x */
static void rms_row_bwd(float *dx, const float *x, const float *w, float r, const float *dy, int n) {
    double dot = 0;
    for (int i = 0; i < n; i++) dot += (double)dy[i] * (w ? w[i] : 1.0f) * x[i];
    const float c = (float)(dot * r * r * r / n);
    for (int i = 0; i < n; i++) dx[i] += r * dy[i] * (w ? w[i] : 1.0f) - x[i] * c;
}
static void rope_row(const es_tm *m, float *v, int heads, int pos, int inverse) {
    const int half = m->n_rot / 2;
    const float *C = m->cosr + (size_t)pos * half, *Sn = m->sinr + (size_t)pos * half;
    for (int h = 0; h < heads; h++) {
        float *x = v + (size_t)h * m->hd;
        for (int i = 0; i < half; i++) {
            const int a = m->neox ? i : 2 * i, b = m->neox ? i + half : 2 * i + 1;
            const float c = C[i], s = inverse ? -Sn[i] : Sn[i], x0 = x[a], x1 = x[b];
            x[a] = x0 * c - x1 * s; x[b] = x0 * s + x1 * c;
        }
    }
}

/* the packed batch: sequence s occupies rows base[s] .. base[s] + len[s] - 1 */
typedef struct { int S; const int *len; int *base; int *pos; } batch;

typedef struct { es_tm *m; const batch *b; int l; } tok_ctx;
/* per token: q/k/v biases, q/k norms, rotation */
static void qkv_post_job(void *c_, int item, int tid) {
    (void)tid; tok_ctx *c = c_; es_tm *m = c->m; const tlayer *y = &m->ly[c->l];
    for (int t = item * 16; t < item * 16 + 16 && t < c->b->base[c->b->S]; t++) {
        float *q = m->Q + (size_t)t * m->qd, *k = m->K + (size_t)t * m->kvd, *v = m->Vv + (size_t)t * m->kvd;
        if (y->bq) for (int i = 0; i < m->qd; i++) q[i] += y->bq[i];
        if (y->bk) for (int i = 0; i < m->kvd; i++) k[i] += y->bk[i];
        if (y->bv) for (int i = 0; i < m->kvd; i++) v[i] += y->bv[i];
        if (y->q_norm) {
            memcpy(m->Qp + (size_t)t * m->qd, q, sizeof(float) * m->qd);
            if (y->qn_len == m->qd && m->qd != m->hd) m->qr[(size_t)t * m->H] = rms_row(q, q, y->q_norm, m->qd, m->eps);
            else for (int h = 0; h < m->H; h++) m->qr[(size_t)t * m->H + h] = rms_row(q + h * m->hd, q + h * m->hd, y->q_norm, m->hd, m->eps);
        }
        if (y->k_norm) {
            memcpy(m->Kp + (size_t)t * m->kvd, k, sizeof(float) * m->kvd);
            if (y->kn_len == m->kvd && m->kvd != m->hd) m->kr[(size_t)t * m->KV] = rms_row(k, k, y->k_norm, m->kvd, m->eps);
            else for (int h = 0; h < m->KV; h++) m->kr[(size_t)t * m->KV + h] = rms_row(k + h * m->hd, k + h * m->hd, y->k_norm, m->hd, m->eps);
        }
        rope_row(m, q, m->H, c->b->pos[t], 0); rope_row(m, k, m->KV, c->b->pos[t], 0);
    }
}
static void qk_bwd_job(void *c_, int item, int tid) {   /* undo rotation, norms (gradients); biases have none */
    tok_ctx *c = c_; es_tm *m = c->m; const tlayer *y = &m->ly[c->l];
    float *tmp = m->scratch[tid];
    for (int t = item * 16; t < item * 16 + 16 && t < c->b->base[c->b->S]; t++) {
        float *dq = m->dQ + (size_t)t * m->qd, *dk = m->dK + (size_t)t * m->kvd;
        rope_row(m, dq, m->H, c->b->pos[t], 1); rope_row(m, dk, m->KV, c->b->pos[t], 1);
        if (y->q_norm) {
            const float *qp = m->Qp + (size_t)t * m->qd;
            memcpy(tmp, dq, sizeof(float) * m->qd); memset(dq, 0, sizeof(float) * m->qd);
            if (y->qn_len == m->qd && m->qd != m->hd) rms_row_bwd(dq, qp, y->q_norm, m->qr[(size_t)t * m->H], tmp, m->qd);
            else for (int h = 0; h < m->H; h++) rms_row_bwd(dq + h * m->hd, qp + h * m->hd, y->q_norm, m->qr[(size_t)t * m->H + h], tmp + h * m->hd, m->hd);
        }
        if (y->k_norm) {
            const float *kp = m->Kp + (size_t)t * m->kvd;
            memcpy(tmp, dk, sizeof(float) * m->kvd); memset(dk, 0, sizeof(float) * m->kvd);
            if (y->kn_len == m->kvd && m->kvd != m->hd) rms_row_bwd(dk, kp, y->k_norm, m->kr[(size_t)t * m->KV], tmp, m->kvd);
            else for (int h = 0; h < m->KV; h++) rms_row_bwd(dk + h * m->hd, kp + h * m->hd, y->k_norm, m->kr[(size_t)t * m->KV + h], tmp + h * m->hd, m->hd);
        }
    }
}

/* attention of sequence s, key/value head g (and its query heads) */

/* ES_LORA_BLAS single-thread attention: per (sequence, head) S = Q K^T, P = softmax, O = P V as sgemms (exact causal mask) */
static float *ABUF; static size_t ABUF_N;
static float *abuf(size_t n) { if (n > ABUF_N) { free(ABUF); ABUF = malloc(sizeof(float) * n); ABUF_N = ABUF ? n : 0; } return ABUF; }
static int attn_blas_fwd(es_tm *m, const batch *b) {
    int mx_n = 0; for (int s = 0; s < b->S; s++) if (b->len[s] > mx_n) mx_n = b->len[s];
    float *S = abuf((size_t)mx_n * mx_n); if (!S) return 0;
    const int grp = m->H / m->KV, hd = m->hd;
    for (int s = 0; s < b->S; s++) {
        const int b0 = b->base[s], n = b->len[s];
        for (int h = 0; h < m->H; h++) {
            const int g = h / grp;
            const float *Q = m->Q + (size_t)b0 * m->qd + (size_t)h * hd, *K = m->K + (size_t)b0 * m->kvd + (size_t)g * hd, *V = m->Vv + (size_t)b0 * m->kvd + (size_t)g * hd;
            sgemm2(0, 1, n, n, hd, m->scale, Q, m->qd, K, m->kvd, 0.0f, S, n);
            for (int i = 0; i < n; i++) {
                float *r = S + (size_t)i * n, mx = -INFINITY;
                for (int j = 0; j <= i; j++) if (r[j] > mx) mx = r[j];
                double sum = 0; for (int j = 0; j <= i; j++) { r[j] = expf(r[j] - mx); sum += r[j]; }
                const float inv = (float)(1.0 / sum); for (int j = 0; j <= i; j++) r[j] *= inv;
                for (int j = i + 1; j < n; j++) r[j] = 0;
                m->Lse[(size_t)(b0 + i) * m->H + h] = mx + (float)log(sum);
            }
            sgemm2(0, 0, n, hd, n, 1.0f, S, n, V, m->kvd, 0.0f, m->Att + (size_t)b0 * m->qd + (size_t)h * hd, m->qd);
        }
    }
    return 1;
}
static int attn_blas_bwd(es_tm *m, const batch *b) {
    int mx_n = 0; for (int s = 0; s < b->S; s++) if (b->len[s] > mx_n) mx_n = b->len[s];
    float *P = abuf((size_t)2 * mx_n * mx_n + mx_n); if (!P) return 0;
    float *dP = P + (size_t)mx_n * mx_n, *D = dP + (size_t)mx_n * mx_n;
    const int grp = m->H / m->KV, hd = m->hd;
    for (int s = 0; s < b->S; s++) {
        const int b0 = b->base[s], n = b->len[s];
        for (int g = 0; g < m->KV; g++) for (int j = 0; j < n; j++) {
            memset(m->dK + (size_t)(b0 + j) * m->kvd + (size_t)g * hd, 0, sizeof(float) * hd);
            memset(m->dV + (size_t)(b0 + j) * m->kvd + (size_t)g * hd, 0, sizeof(float) * hd);
        }
        for (int h = 0; h < m->H; h++) {
            const int g = h / grp; const size_t qo = (size_t)b0 * m->qd + (size_t)h * hd, ko = (size_t)b0 * m->kvd + (size_t)g * hd;
            const float *Q = m->Q + qo, *K = m->K + ko, *V = m->Vv + ko, *dO = m->dAtt + qo, *O = m->Att + qo;
            sgemm2(0, 1, n, n, hd, m->scale, Q, m->qd, K, m->kvd, 0.0f, P, n);
            sgemm2(0, 1, n, n, hd, 1.0f, dO, m->qd, V, m->kvd, 0.0f, dP, n);
            for (int i = 0; i < n; i++) {
                float d = 0; for (int e = 0; e < hd; e++) d += dO[(size_t)i * m->qd + e] * O[(size_t)i * m->qd + e]; D[i] = d;
                const float lse = m->Lse[(size_t)(b0 + i) * m->H + h]; float *pr = P + (size_t)i * n, *dr = dP + (size_t)i * n;
                for (int j = 0; j <= i; j++) { pr[j] = expf(pr[j] - lse); dr[j] = pr[j] * (dr[j] - d) * m->scale; }
                for (int j = i + 1; j < n; j++) { pr[j] = 0; dr[j] = 0; }
            }
            sgemm2(1, 0, n, hd, n, 1.0f, P, n, dO, m->qd, 1.0f, m->dV + ko, m->kvd);     /* dV += P^T dO */
            sgemm2(0, 0, n, hd, n, 1.0f, dP, n, K, m->kvd, 0.0f, m->dQ + qo, m->qd);     /* dQ = dS K */
            sgemm2(1, 0, n, hd, n, 1.0f, dP, n, Q, m->qd, 1.0f, m->dK + ko, m->kvd);     /* dK += dS^T Q */
        }
    }
    return 1;
}
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#define ES_MV __attribute__((target_clones("arch=x86-64-v3", "default")))
#else
#define ES_MV
#endif
static inline __attribute__((always_inline)) float vdot(const float *a, const float *b, int n) {   /* 8 lanes, then a fixed-order sum */
    v8 acc = {0}; int e = 0;
    for (; e + 8 <= n; e += 8) acc += *(const v8 *)(a + e) * *(const v8 *)(b + e);
    float r = hsum(acc); for (; e < n; e++) r += a[e] * b[e];
    return r;
}
ES_MV static void attn_job(void *c_, int item, int tid) {
    tok_ctx *c = c_; es_tm *m = c->m;
    const int s = item / m->KV, g = item % m->KV, grp = m->H / m->KV, hd = m->hd, b0 = c->b->base[s], n = c->b->len[s];
    float *sc = m->scratch[tid];
    for (int h = g * grp; h < (g + 1) * grp; h++)
        for (int i = 0; i < n; i++) {
            const float *q = m->Q + (size_t)(b0 + i) * m->qd + (size_t)h * hd;
            float mx = -INFINITY;
            for (int j = 0; j <= i; j++) {
                const float *k = m->K + (size_t)(b0 + j) * m->kvd + (size_t)g * hd;
                const float d = vdot(q, k, hd);
                sc[j] = d * m->scale; if (sc[j] > mx) mx = sc[j];
            }
            double sum = 0;
            for (int j = 0; j <= i; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
            float *o = m->Att + (size_t)(b0 + i) * m->qd + (size_t)h * hd;
            for (int e = 0; e < hd; e++) o[e] = 0;
            const float inv = (float)(1.0 / sum);
            for (int j = 0; j <= i; j++) {
                const float p = sc[j] * inv; const float *v = m->Vv + (size_t)(b0 + j) * m->kvd + (size_t)g * hd;
                for (int e = 0; e < hd; e++) o[e] += p * v[e];
            }
            m->Lse[(size_t)(b0 + i) * m->H + h] = mx + (float)log(sum);
        }
}
ES_MV static void attn_bwd_job(void *c_, int item, int tid) {
    (void)tid; tok_ctx *c = c_; es_tm *m = c->m;
    const int s = item / m->KV, g = item % m->KV, grp = m->H / m->KV, hd = m->hd, b0 = c->b->base[s], n = c->b->len[s];
    for (int j = 0; j < n; j++) { memset(m->dK + (size_t)(b0 + j) * m->kvd + (size_t)g * hd, 0, sizeof(float) * hd); memset(m->dV + (size_t)(b0 + j) * m->kvd + (size_t)g * hd, 0, sizeof(float) * hd); }
    for (int h = g * grp; h < (g + 1) * grp; h++)
        for (int i = 0; i < n; i++) {
            const float *q = m->Q + (size_t)(b0 + i) * m->qd + (size_t)h * hd, *dO = m->dAtt + (size_t)(b0 + i) * m->qd + (size_t)h * hd;
            const float *O = m->Att + (size_t)(b0 + i) * m->qd + (size_t)h * hd;
            float *dq = m->dQ + (size_t)(b0 + i) * m->qd + (size_t)h * hd;
            const float D = vdot(dO, O, hd);
            for (int e = 0; e < hd; e++) dq[e] = 0;
            const float lse = m->Lse[(size_t)(b0 + i) * m->H + h];
            for (int j = 0; j <= i; j++) {
                const float *k = m->K + (size_t)(b0 + j) * m->kvd + (size_t)g * hd, *v = m->Vv + (size_t)(b0 + j) * m->kvd + (size_t)g * hd;
                const float d = vdot(q, k, hd), dp = vdot(dO, v, hd);
                const float p = expf(d * m->scale - lse), ds = p * (dp - D) * m->scale;
                float *dk = m->dK + (size_t)(b0 + j) * m->kvd + (size_t)g * hd, *dv = m->dV + (size_t)(b0 + j) * m->kvd + (size_t)g * hd;
                for (int e = 0; e < hd; e++) { dv[e] += p * dO[e]; dq[e] += ds * k[e]; dk[e] += ds * q[e]; }
            }
        }
}

typedef struct { es_tm *m; const batch *b; int l; const float *Xin; float *Xout; } layer_ctx;
static void norm1_job(void *c_, int item, int tid) {
    (void)tid; layer_ctx *c = c_; es_tm *m = c->m; const int T = c->b->base[c->b->S];
    for (int t = item * 16; t < item * 16 + 16 && t < T; t++) m->r1[t] = rms_row(m->Xn + (size_t)t * m->d, c->Xin + (size_t)t * m->d, m->ly[c->l].attn_norm, m->d, m->eps);
}
static void norm2_job(void *c_, int item, int tid) {
    (void)tid; layer_ctx *c = c_; es_tm *m = c->m; const int T = c->b->base[c->b->S];
    for (int t = item * 16; t < item * 16 + 16 && t < T; t++) m->r2[t] = rms_row(m->Xn2 + (size_t)t * m->d, m->Xm + (size_t)t * m->d, m->ly[c->l].ffn_norm, m->d, m->eps);
}

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
/* sigmoid on 16 lanes: exp via 2^n * poly (rel err ~1e-7), AVX-512F only */
__attribute__((target("avx512f"))) static inline __m512 vsig16(__m512 g) {
    __m512 x = _mm512_max_ps(_mm512_min_ps(_mm512_sub_ps(_mm512_setzero_ps(), g), _mm512_set1_ps(88.0f)), _mm512_set1_ps(-88.0f));
    __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
    r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
    __m512 p = _mm512_set1_ps(1.9875691500E-4f);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.3981999507E-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(8.3334519073E-3f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(4.1665795894E-2f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.6666665459E-1f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(5.0000001201E-1f));
    p = _mm512_fmadd_ps(p, _mm512_mul_ps(r, r), _mm512_add_ps(r, _mm512_set1_ps(1.0f)));
    __m512 e = _mm512_scalef_ps(p, n);
    return _mm512_div_ps(_mm512_set1_ps(1.0f), _mm512_add_ps(_mm512_set1_ps(1.0f), e));
}
__attribute__((target("avx512f"))) static size_t swiglu16(float *H, const float *G, const float *U, size_t a, size_t e) {
    size_t i = a;
    for (; i + 16 <= e; i += 16) { __m512 g = _mm512_loadu_ps(G + i); _mm512_storeu_ps(H + i, _mm512_mul_ps(_mm512_mul_ps(g, vsig16(g)), _mm512_loadu_ps(U + i))); }
    return i;
}
__attribute__((target("avx512f"))) static size_t swiglu_bwd16(float *G, float *U, const float *dH, size_t a, size_t e) {
    size_t i = a; const __m512 one = _mm512_set1_ps(1.0f);
    for (; i + 16 <= e; i += 16) {
        __m512 g = _mm512_loadu_ps(G + i), u = _mm512_loadu_ps(U + i), dh = _mm512_loadu_ps(dH + i), sg = vsig16(g);
        _mm512_storeu_ps(G + i, _mm512_mul_ps(_mm512_mul_ps(_mm512_mul_ps(dh, u), sg), _mm512_fmadd_ps(g, _mm512_sub_ps(one, sg), one)));
        _mm512_storeu_ps(U + i, _mm512_mul_ps(_mm512_mul_ps(dh, g), sg));
    }
    return i;
}
static int has512(void) { static int h = -1; if (h < 0) h = __builtin_cpu_supports("avx512f") && !getenv("ES_NO_VEXP"); return h; }
#else
static int has512(void) { return 0; }
#endif
static void swiglu_job(void *c_, int item, int tid) {
    (void)tid; layer_ctx *c = c_; es_tm *m = c->m; const int T = c->b->base[c->b->S];
    const size_t a = (size_t)item * 16 * m->ff, e = (size_t)(item * 16 + 16 < T ? item * 16 + 16 : T) * m->ff;
    size_t i0 = a;
#if defined(__x86_64__) && defined(__GNUC__)
    if (has512()) i0 = swiglu16(m->Hh, m->G, m->U, a, e);
#endif
    for (size_t i = i0; i < e; i++) { const float g = m->G[i]; m->Hh[i] = g / (1.0f + expf(-g)) * m->U[i]; }
}

/* one layer forward; leaves every intermediate the backward pass needs in the buffers */
static void layer_fwd(es_tm *m, const batch *b, int l, const float *Xin, float *Xout) {
    const int T = b->base[b->S], items = (T + 15) / 16;
    const tlayer *y = &m->ly[l];
    layer_ctx lc = {m, b, l, Xin, Xout};
    tok_ctx tc = {m, b, l};
    pool_run(&m->P, items, norm1_job, &lc);
    mm_fwd(m, &y->w[ES_LT_Q], m->Xn, T, m->Q, (size_t)m->qd); lora_fwd(m, l, ES_LT_Q, m->Xn, T, m->Q, (size_t)m->qd);
    mm_fwd(m, &y->w[ES_LT_K], m->Xn, T, m->K, (size_t)m->kvd); lora_fwd(m, l, ES_LT_K, m->Xn, T, m->K, (size_t)m->kvd);
    mm_fwd(m, &y->w[ES_LT_V], m->Xn, T, m->Vv, (size_t)m->kvd); lora_fwd(m, l, ES_LT_V, m->Xn, T, m->Vv, (size_t)m->kvd);
    pool_run(&m->P, items, qkv_post_job, &tc);
    if (!(lora_blas(m) && attn_blas_fwd(m, b))) pool_run(&m->P, b->S * m->KV, attn_job, &tc);
    mm_fwd(m, &y->w[ES_LT_O], m->Att, T, m->Xm, (size_t)m->d); lora_fwd(m, l, ES_LT_O, m->Att, T, m->Xm, (size_t)m->d);
    for (size_t i = 0; i < (size_t)T * m->d; i++) m->Xm[i] += Xin[i];
    pool_run(&m->P, items, norm2_job, &lc);
    mm_fwd(m, &y->w[ES_LT_GATE], m->Xn2, T, m->G, (size_t)m->ff); lora_fwd(m, l, ES_LT_GATE, m->Xn2, T, m->G, (size_t)m->ff);
    mm_fwd(m, &y->w[ES_LT_UP], m->Xn2, T, m->U, (size_t)m->ff); lora_fwd(m, l, ES_LT_UP, m->Xn2, T, m->U, (size_t)m->ff);
    pool_run(&m->P, items, swiglu_job, &lc);
    mm_fwd(m, &y->w[ES_LT_DOWN], m->Hh, T, Xout, (size_t)m->d); lora_fwd(m, l, ES_LT_DOWN, m->Hh, T, Xout, (size_t)m->d);
    for (size_t i = 0; i < (size_t)T * m->d; i++) Xout[i] += m->Xm[i];
}

typedef struct { es_tm *m; const batch *b; int l; const float *Xin; float *dX; } lb2_ctx;
static void swiglu_bwd_job(void *c_, int item, int tid) {   /* dH -> dG (into G) and dU (into U) */
    (void)tid; lb2_ctx *c = c_; es_tm *m = c->m; const int T = c->b->base[c->b->S];
    const size_t a = (size_t)item * 16 * m->ff, e = (size_t)(item * 16 + 16 < T ? item * 16 + 16 : T) * m->ff;
    size_t i0 = a;
#if defined(__x86_64__) && defined(__GNUC__)
    if (has512()) i0 = swiglu_bwd16(m->G, m->U, m->dH, a, e);
#endif
    for (size_t i = i0; i < e; i++) {
        const float g = m->G[i], u = m->U[i], dh = m->dH[i], sg = 1.0f / (1.0f + expf(-g));
        m->G[i] = dh * u * sg * (1.0f + g * (1.0f - sg));
        m->U[i] = dh * g * sg;
    }
}
static void norm2_bwd_job(void *c_, int item, int tid) {   /* dX (the gradient at Xm) += norm2 backward of dXn */
    (void)tid; lb2_ctx *c = c_; es_tm *m = c->m; const int T = c->b->base[c->b->S];
    for (int t = item * 16; t < item * 16 + 16 && t < T; t++)
        rms_row_bwd(c->dX + (size_t)t * m->d, m->Xm + (size_t)t * m->d, m->ly[c->l].ffn_norm, m->r2[t], m->dXn + (size_t)t * m->d, m->d);
}
static void norm1_bwd_job(void *c_, int item, int tid) {
    (void)tid; lb2_ctx *c = c_; es_tm *m = c->m; const int T = c->b->base[c->b->S];
    for (int t = item * 16; t < item * 16 + 16 && t < T; t++)
        rms_row_bwd(c->dX + (size_t)t * m->d, c->Xin + (size_t)t * m->d, m->ly[c->l].attn_norm, m->r1[t], m->dXn + (size_t)t * m->d, m->d);
}
/* dX: gradient at the layer output on entry, at the layer input on return (the layer was just recomputed by layer_fwd) */
static void layer_bwd(es_tm *m, const batch *b, int l, const float *Xin, float *dX) {
    const int T = b->base[b->S], items = (T + 15) / 16;
    const tlayer *y = &m->ly[l];
    lb2_ctx c = {m, b, l, Xin, dX};
    tok_ctx tc = {m, b, l};
    /* FFN: dX is the gradient at the down output and (residual) at Xm */
    memset(m->dH, 0, sizeof(float) * (size_t)T * m->ff);
    mm_bwd(m, &y->w[ES_LT_DOWN], dX, T, (size_t)m->d, m->dH); lora_bwd(m, l, ES_LT_DOWN, m->Hh, T, dX, (size_t)m->d, m->dH);
    pool_run(&m->P, items, swiglu_bwd_job, &c);
    memset(m->dXn, 0, sizeof(float) * (size_t)T * m->d);
    /* G and U now hold dG and dU; the LoRA inputs (Xn2) and Z are untouched */
    mm_bwd(m, &y->w[ES_LT_GATE], m->G, T, (size_t)m->ff, m->dXn); lora_bwd(m, l, ES_LT_GATE, m->Xn2, T, m->G, (size_t)m->ff, m->dXn);
    mm_bwd(m, &y->w[ES_LT_UP], m->U, T, (size_t)m->ff, m->dXn); lora_bwd(m, l, ES_LT_UP, m->Xn2, T, m->U, (size_t)m->ff, m->dXn);
    pool_run(&m->P, items, norm2_bwd_job, &c);
    /* attention: dX is now the gradient at Xm, so also at the o output and (residual) at Xin */
    memset(m->dAtt, 0, sizeof(float) * (size_t)T * m->qd);
    mm_bwd(m, &y->w[ES_LT_O], dX, T, (size_t)m->d, m->dAtt); lora_bwd(m, l, ES_LT_O, m->Att, T, dX, (size_t)m->d, m->dAtt);
    if (!(lora_blas(m) && attn_blas_bwd(m, b))) pool_run(&m->P, b->S * m->KV, attn_bwd_job, &tc);
    pool_run(&m->P, items, qk_bwd_job, &tc);
    memset(m->dXn, 0, sizeof(float) * (size_t)T * m->d);
    mm_bwd(m, &y->w[ES_LT_Q], m->dQ, T, (size_t)m->qd, m->dXn); lora_bwd(m, l, ES_LT_Q, m->Xn, T, m->dQ, (size_t)m->qd, m->dXn);
    mm_bwd(m, &y->w[ES_LT_K], m->dK, T, (size_t)m->kvd, m->dXn); lora_bwd(m, l, ES_LT_K, m->Xn, T, m->dK, (size_t)m->kvd, m->dXn);
    mm_bwd(m, &y->w[ES_LT_V], m->dV, T, (size_t)m->kvd, m->dXn); lora_bwd(m, l, ES_LT_V, m->Xn, T, m->dV, (size_t)m->kvd, m->dXn);
    pool_run(&m->P, items, norm1_bwd_job, &c);
}


/* ---------------------------------------------------------------- keep activations instead of recomputing */
#define NKB (17 + ES_LT_N)
static float **kb_ptr(es_tm *m, int i) {
    float **p[17] = {&m->Xn, &m->Q, &m->K, &m->Vv, &m->Qp, &m->Kp, &m->Att, &m->Lse, &m->Xm, &m->Xn2, &m->G, &m->U, &m->Hh, &m->r1, &m->r2, &m->qr, &m->kr};
    return i < 17 ? p[i] : &m->Z[i - 17];
}
static size_t kb_w(const es_tm *m, int i) {   /* floats per token of buffer i */
    const size_t rk = m->rank > 0 ? (size_t)m->rank : 1;
    const size_t w[17] = {m->d, m->qd, m->kvd, m->kvd, m->qd, m->kvd, m->qd, m->H, m->d, m->d, m->ff, m->ff, m->ff, 1, 1, m->H, m->KV};
    return i < 17 ? w[i] : rk;
}
static size_t kb_layer(const es_tm *m, int T) { size_t n = 0; for (int i = 0; i < NKB; i++) n += kb_w(m, i) * (size_t)T; return n; }
/* point the layer buffers at layer l's slice of m->keep (orig holds the normal buffers) */
static void kb_set(es_tm *m, int l, int T, float **orig) {
    float *base = l < 0 ? NULL : m->keep + (size_t)l * kb_layer(m, T);
    for (int i = 0; i < NKB; i++) { *kb_ptr(m, i) = l < 0 ? orig[i] : base; if (base) base += kb_w(m, i) * (size_t)T; }
}
static int keep_on(es_tm *m, int T) {
    static int env = -1; if (env < 0) { const char *e = getenv("ES_LORA_RECOMPUTE"); env = !(e && atoi(e)); }
    if (!env) return 0;
    const size_t need = kb_layer(m, T) * (size_t)m->L;
    if (need > m->keep_cap) {
        free(m->keep); m->keep = malloc(sizeof(float) * need); m->keep_cap = m->keep ? need : 0;
    }
    return m->keep != NULL;
}
/* ---------------------------------------------------------------- loss */
typedef struct { es_tm *m; int n; const int32_t *tgt; float gscale; double *loss; int backward; float *logits_out; } ce_ctx;
static void ce_job(void *c_, int item, int tid) {
    (void)tid; ce_ctx *c = c_; es_tm *m = c->m;
    float *lg = m->lmbuf + (size_t)item * m->V;
    if (c->logits_out) memcpy(c->logits_out + (size_t)item * m->V, lg, sizeof(float) * m->V);
    float mx = -INFINITY; for (int v = 0; v < m->V; v++) if (lg[v] > mx) mx = lg[v];
    double sum = 0; for (int v = 0; v < m->V; v++) sum += exp((double)lg[v] - mx);
    const double lse = mx + log(sum);
    const int tg = c->tgt ? c->tgt[item] : -1;
    c->loss[item] = tg >= 0 ? lse - lg[tg] : 0;
    if (c->backward) {
        for (int v = 0; v < m->V; v++) lg[v] = (float)exp((double)lg[v] - lse) * c->gscale;
        if (tg >= 0) lg[tg] -= c->gscale;
    }
}
typedef struct { es_tm *m; const int *rows; int n; const float *XL; float *xn, *rs; } fn_ctx;
static void finalnorm_job(void *c_, int item, int tid) {
    (void)tid; fn_ctx *c = c_; es_tm *m = c->m;
    c->rs[item] = rms_row(c->xn + (size_t)item * m->d, c->XL + (size_t)c->rows[item] * m->d, m->out_norm, m->d, m->eps);
}

/* ---------------------------------------------------------------- one run */
static double run(es_tm *m, int S, const int32_t *const *ids, const int *len, const int *first, int backward, float gscale, long *count, float *logits_out) {
    pick_kernels();
    int T = 0; for (int s = 0; s < S; s++) T += len[s];
    if (T <= 0) { if (count) *count = 0; return 0; }
    ensure(m, T);
    batch b = {S, len, malloc(sizeof(int) * (size_t)(S + 1)), malloc(sizeof(int) * (size_t)T)};
    b.base[0] = 0; for (int s = 0; s < S; s++) { b.base[s + 1] = b.base[s] + len[s]; for (int p = 0; p < len[s]; p++) b.pos[b.base[s] + p] = p; }
    m->bytes = 0;
    /* learned rows and their targets */
    int nl = 0; for (int s = 0; s < S; s++) for (int p = first[s] < 0 ? 0 : first[s]; p < len[s] - (logits_out ? 0 : 1); p++) nl++;
    int *rows = malloc(sizeof(int) * (size_t)(nl ? nl : 1)); int32_t *tgt = malloc(sizeof(int32_t) * (size_t)(nl ? nl : 1));
    nl = 0;
    for (int s = 0; s < S; s++) for (int p = first[s] < 0 ? 0 : first[s]; p < len[s] - (logits_out ? 0 : 1); p++) { rows[nl] = b.base[s] + p; tgt[nl] = p + 1 < len[s] ? ids[s][p + 1] : -1; nl++; }
    /* embeddings */
    float *X0 = m->ck;
    for (int s = 0; s < S; s++) for (int p = 0; p < len[s]; p++) {
        const int32_t id = ids[s][p] < 0 || ids[s][p] >= m->V ? 0 : ids[s][p];
        es_dequant_row(m->embd.type, m->embd.data + m->embd.rb * (size_t)id, X0 + (size_t)(b.base[s] + p) * m->d, m->d);
    }
    float *orig[NKB]; for (int i = 0; i < NKB; i++) orig[i] = *kb_ptr(m, i);
    const int keep = backward && keep_on(m, T);
    for (int l = 0; l < m->L; l++) {
        if (keep) kb_set(m, l, T, orig);
        layer_fwd(m, &b, l, m->ck + (size_t)l * T * m->d, m->ck + (size_t)(l + 1) * T * m->d);
    }
    if (keep) kb_set(m, -1, T, orig);
    const float *XL = m->ck + (size_t)m->L * T * m->d;
    /* loss, in tiles of learned positions; the backward pass of the output layer goes straight into dX */
    double loss = 0;
    if (backward) memset(m->dX, 0, sizeof(float) * (size_t)T * m->d);
    double *lt = malloc(sizeof(double) * LM_TILE);
    for (int i0 = 0; i0 < nl; i0 += LM_TILE) {
        const int n = nl - i0 < LM_TILE ? nl - i0 : LM_TILE;
        float *xn = m->Xn, *rs = m->r1;   /* layer buffers are free here */
        fn_ctx fc = {m, rows + i0, n, XL, xn, rs};
        pool_run(&m->P, n, finalnorm_job, &fc);
        mm_fwd(m, &m->out, xn, n, m->lmbuf, (size_t)m->V);
        ce_ctx cc = {m, n, tgt + i0, gscale, lt, backward, logits_out ? logits_out + (size_t)i0 * m->V : NULL};
        pool_run(&m->P, n, ce_job, &cc);
        for (int i = 0; i < n; i++) loss += lt[i];
        if (backward) {
            float *dxn = m->dXn; memset(dxn, 0, sizeof(float) * (size_t)n * m->d);
            mm_bwd(m, &m->out, m->lmbuf, n, (size_t)m->V, dxn);
            for (int i = 0; i < n; i++) rms_row_bwd(m->dX + (size_t)rows[i0 + i] * m->d, XL + (size_t)rows[i0 + i] * m->d, m->out_norm, rs[i], dxn + (size_t)i * m->d, m->d);
        }
    }
    free(lt);
    if (backward)
        for (int l = m->L - 1; l >= 0; l--) {
            const float *Xin = m->ck + (size_t)l * T * m->d;
            if (keep) kb_set(m, l, T, orig);   /* this layer's saved intermediates */
            else layer_fwd(m, &b, l, Xin, m->X);   /* recompute: refills every intermediate of this layer */
            layer_bwd(m, &b, l, Xin, m->dX);
        }
    if (keep) kb_set(m, -1, T, orig);
    if (count) { long c = 0; for (int i = 0; i < nl; i++) c += tgt[i] >= 0; *count = c; }
    free(rows); free(tgt); free(b.base); free(b.pos);
    return loss;
}
double es_tm_run(es_tm *m, int S, const int32_t *const *ids, const int *len, const int *first, int backward, float gscale, long *count) {
    return run(m, S, ids, len, first, backward, gscale, count, NULL);
}
int es_tm_logits(es_tm *m, const int32_t *ids, int n, float *out) {
    const int32_t *p = ids; const int f = 0;
    run(m, 1, &p, &n, &f, 0, 1.0f, NULL, out);
    return 0;
}
