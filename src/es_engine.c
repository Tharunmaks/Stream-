/*
 * ExpertStream engine (see es_engine.h). One forward pass for every supported model family:
 *  - attention: grouped-query attention with optional Q/K norms, biases, partial RoPE, YaRN scaling,
 *    sliding windows and attention sinks (gpt-oss), or DeepSeek's multi-head latent attention (MLA,
 *    with the absorbed K/V projections, so the cache holds one 576-wide latent row per token);
 *  - experts: softmax, sigmoid (with DeepSeek-V3 group selection and a selection bias) or
 *    softmax-over-the-chosen routing (gpt-oss), SwiGLU or gpt-oss' clamped SwiGLU, per-expert biases,
 *    shared experts, dense leading layers.
 * Tokens are processed in batches: every weight matrix, and every expert a batch needs, is read once
 * for the whole batch. Checked against llama.cpp on small random models of each family
 * (scripts/test_archs.py) and, for OLMoE, against an independent float64 NumPy implementation.
 */
#define _GNU_SOURCE
#include "es_engine.h"

#include <errno.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "es_cache.h"
#include "es_compute.h"
#include "es_cpu.h"
#include "es_format.h"
#include "es_gguf.h"
#include "es_io.h"
#include "es_quant.h"
#include "es_tok.h"
#include "../tools/es_common.h"

#define MAXK 32
#define MAXJOBS 16384
#define BMAX 16
static int BATCH_OK;

/* ---------------- weights ---------------- */
typedef struct { uint32_t type; int rows, cols; const void *data; } W;

enum { AR_STD, AR_GPTOSS, AR_DS2 };
enum { GATE_SOFTMAX, GATE_SIGMOID, GATE_SOFTMAX_CHOSEN };

typedef struct {
    W attn_norm, q, k, v, o, q_norm, k_norm, ffn_norm, gate_inp;
    W bq, bk, bv, bo;                         /* optional biases */
    W sinks;                                  /* gpt-oss: one extra softmax logit per head */
    W q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b;   /* DeepSeek MLA */
    W ffn_gate, ffn_up, ffn_down;             /* dense FFN (layers without experts) */
    W sh_gate, sh_up, sh_down, sh_gate_inp;   /* shared expert (and its sigmoid gate, Qwen2-MoE) */
    W gate_inp_b, exp_probs_b;                /* router bias (gpt-oss); expert selection bias (DeepSeek-V3) */
    W gate_exps_b, up_exps_b, down_exps_b;    /* per-expert biases (gpt-oss) */
    int moe, swa;
} layer;

typedef struct {
    const char *arch;
    int kind;
    int n_layer, n_embd, n_head, n_head_kv, head_dim, n_exp, n_used, n_vocab, ctx;
    float eps;
    /* rotary embeddings */
    int rope_neox, n_rot, n_ctx_orig;
    float rope_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, corr[2];
    const float *rope_freqs;                  /* Llama 3.1 style per-frequency factors, or NULL */
    float kq_scale;
    int n_swa;
    /* experts */
    int gating, norm_topk, act_oai, n_group, n_group_used;
    float w_scale;
    /* MLA */
    int q_lora, kv_lora, nope, rope_d, v_head;
    /* attention geometry: dk/dv per head inside attention, kvk/kvv floats per cached position, qd = Q floats per token, od = o input */
    int dk, dv, kvk, kvv, qd, od;
    W tok_embd, output, output_norm;
    layer *L;
    uint8_t **tptr;     /* RAM copy of each non-expert tensor, by tensor index */
    size_t core_bytes;
} model;

static es_gguf G;
static model M;
static es_tok *T;

static W load_w(const char *name, int required) {
    const es_gguf_tensor *t = es_gguf_tensor_find(&G, name);
    W w = {0};
    if (!t) {
        if (required) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
        return w;
    }
    if (!es_quant_supported(t->type)) {
        fprintf(stderr, "tensor %s uses %s, which this engine can't compute yet\n", name, es_ggml_type_name(t->type));
        exit(1);
    }
    w.type = t->type;
    w.cols = (int)t->ne[0];
    w.rows = (int)(t->ne[1] * t->ne[2]);
    w.data = M.tptr[t - G.t];
    return w;
}
/* rows [r0, r0 + n) of w as a matrix of its own */
static W rows_of(const W *w, int r0, int n) {
    W v = *w;
    v.rows = n;
    v.data = (const uint8_t *)w->data + es_row_bytes(w->type, w->cols) * (size_t)r0;
    return v;
}

long es_rss_mb(void) {
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "VmRSS:", 6)) { kb = atol(line + 6); break; }
    fclose(f);
    return kb / 1024;
}

/* ---------------- threads ---------------- */
static es_cpool *CP;
static int NT;
static float *SCRATCH[ES_MAX_CPUS];

typedef struct { uint32_t type; const void *w; int rows, cols; const es_act *a; float *y; int nb; size_t ys; } mv_job;
static mv_job JOBS[MAXJOBS];
static int NJOBS;

/* Rows are handed out in small chunks from a shared counter, so a slow core
 * (an A55, or a thread the OS parked) never holds up the others. */
static _Atomic long NEXT_ROW;
static long TOTAL_ROWS;
static void mv_task(int tid, int n, void *arg) {
    (void)arg; (void)n;
    const long CH = 16;
    int j = 0; long base = 0;
    for (;;) {
        long c = atomic_fetch_add(&NEXT_ROW, CH);
        if (c >= TOTAL_ROWS) break;
        long e = c + CH < TOTAL_ROWS ? c + CH : TOTAL_ROWS;
        while (c < e) {
            while (c >= base + JOBS[j].rows) { base += JOBS[j].rows; j++; }   /* chunks only move forward */
            long b = e < base + JOBS[j].rows ? e : base + JOBS[j].rows;
            /* nb > 1: several activations share this chunk of weight rows, so the weights are read from memory once */
            if (JOBS[j].nb > 1)   /* weights unpacked once for all activations (bit-identical to one at a time) */
                es_matvec_multi(JOBS[j].type, JOBS[j].w, JOBS[j].cols, (int)(c - base), (int)(b - base), JOBS[j].a, JOBS[j].nb, JOBS[j].y, JOBS[j].ys, SCRATCH[tid]);
            else
                es_matvec(JOBS[j].type, JOBS[j].w, JOBS[j].cols, (int)(c - base), (int)(b - base), JOBS[j].a, JOBS[j].y, SCRATCH[tid]);
            c = b;
        }
    }
}
static void run_jobs(void);
static void jobn(const W *w, const es_act *acts, float *y, size_t ystride, int nb) {
    if (NJOBS == MAXJOBS) run_jobs();   /* only when a batch needs more jobs than fit: run what is queued */
    JOBS[NJOBS++] = (mv_job){w->type, w->data, w->rows, w->cols, acts, y, nb, ystride};
}
static void run_jobs(void) {
    TOTAL_ROWS = 0;
    for (int j = 0; j < NJOBS; j++) TOTAL_ROWS += JOBS[j].rows;
    atomic_store(&NEXT_ROW, 0);
    if (NJOBS) {
        if (NT == 1) mv_task(0, 1, NULL); /* no thread pool (browser build) */
        else es_cpool_run(CP, mv_task, NULL);
    }
    NJOBS = 0;
}

/* ---------------- math helpers ---------------- */
static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float s = (float)(1.0 / sqrt(ss / n + eps));
    for (int i = 0; i < n; i++) o[i] = x[i] * s * (w ? w[i] : 1.0f);
}
static void add_bias(float *y, const W *b, int n) { if (b->data) { const float *p = b->data; for (int i = 0; i < n; i++) y[i] += p[i]; } }

/* Rotary embeddings (ggml's rope_yarn): cos/sin for one position, shared by every head and layer */
static float *RC, *RS; static int RPOS = -1;
static float yarn_ramp(float lo, float hi, int i0) { const float y = (i0 / 2 - lo) / fmaxf(0.001f, hi - lo); return 1 - fminf(1, fmaxf(0, y)); }
static void rope_cache(int pos) {
    if (pos == RPOS) return;
    const int half = M.n_rot / 2;
    for (int i = 0; i < half; i++) {
        float theta_extrap = pos * powf(M.rope_base, -2.0f * i / M.n_rot);
        if (M.rope_freqs) theta_extrap /= M.rope_freqs[i];
        const float theta_interp = M.freq_scale * theta_extrap;
        float theta = theta_interp, ms = M.attn_factor;
        if (M.ext_factor != 0.0f) {
            const float mix = yarn_ramp(M.corr[0], M.corr[1], 2 * i) * M.ext_factor;
            theta = theta_interp * (1 - mix) + theta_extrap * mix;
            ms *= 1.0f + 0.1f * logf(1.0f / M.freq_scale);
        }
        RC[i] = cosf(theta) * ms; RS[i] = sinf(theta) * ms;
    }
    RPOS = pos;
}
/* rotate the first n_rot values of n_heads vectors that are `stride` apart */
static void rope(float *v, int n_heads, int stride, int pos) {
    rope_cache(pos);
    const int nr = M.n_rot;
    for (int h = 0; h < n_heads; h++) {
        float *x = v + (size_t)h * stride;
        for (int i = 0; i < nr / 2; i++) {
            const float c = RC[i], s = RS[i];
            const int a = M.rope_neox ? i : 2 * i, b = M.rope_neox ? i + nr / 2 : 2 * i + 1;
            const float x0 = x[a], x1 = x[b];
            x[a] = x0 * c - x1 * s;
            x[b] = x0 * s + x1 * c;
        }
    }
}

/* ---------------- state ---------------- */
static float *LOGITS;
static float *KC, *VC;   /* [layer][ctx][kvk] and [layer][ctx][kvv] (no VC for MLA: values are the start of each key row) */
/* batch buffers: up to BMAX tokens */
static float *BX, *BXN, *BTMP, *BQ, *BQA, *BKV, *BQM, *BATL, *BATTN, *BGU, *BH, *BYE, *BROUT, *BLOG, *BV2;
static es_act BACT[BMAX], BACT2[BMAX];
/* Q8_K activation blocks come from a pool that is emptied together with the Q8_0 arena */
static es_block_q8_K *Q8K; static size_t Q8K_CAP, Q8K_USE;
static void arena_reset(void) { es_act_arena_reset(); Q8K_USE = 0; }
static void prep(es_act *a, const float *x, int cols) {
    es_block_q8_K *q = NULL;
    if (cols % 256 == 0 && Q8K_USE + (size_t)cols / 256 <= Q8K_CAP) { q = Q8K + Q8K_USE; Q8K_USE += (size_t)cols / 256; }
    es_act_prepare(a, x, cols, q);
}

/* attention of one query position, parallel over heads */
typedef struct { int layer, pos; const float *q; float *out; } att_arg;
static void att_task(int tid, int n, void *arg) {
    const att_arg *aa = arg;
    const layer *L = &M.L[aa->layer];
    const int mla = M.kind == AR_DS2, dk = M.dk, dv = M.dv, grp = M.n_head / M.n_head_kv;
    const float *kc = KC + (size_t)aa->layer * M.ctx * M.kvk;
    const float *vc = mla ? kc : VC + (size_t)aa->layer * M.ctx * M.kvv;
    const int vstride = mla ? M.kvk : M.kvv;
    float *sc = SCRATCH[tid];
    const float scale = M.kq_scale;
    const int t0 = L->swa && M.n_swa > 0 && aa->pos - M.n_swa + 1 > 0 ? aa->pos - M.n_swa + 1 : 0;
    int h0, h1;
    es_split(M.n_head, tid, n, &h0, &h1);
    for (int h = h0; h < h1; h++) {
        const float *q = aa->q + (size_t)h * dk;
        const int kh = mla ? 0 : h / grp;
        float mx = -INFINITY;
        for (int t = t0; t <= aa->pos; t++) {
            const float *k = kc + (size_t)t * M.kvk + (size_t)kh * dk;
            float s = 0;
            for (int i = 0; i < dk; i++) s += q[i] * k[i];
            sc[t] = s * scale;
            if (sc[t] > mx) mx = sc[t];
        }
        const float sink = L->sinks.data ? ((const float *)L->sinks.data)[h] : -INFINITY;
        if (sink > mx) mx = sink;
        float sum = 0;
        for (int t = t0; t <= aa->pos; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
        if (L->sinks.data) sum += expf(sink - mx);
        float *o = aa->out + (size_t)h * dv;
        memset(o, 0, sizeof(float) * (size_t)dv);
        for (int t = t0; t <= aa->pos; t++) {
            const float p = sc[t] / sum;
            const float *v = vc + (size_t)t * vstride + (size_t)kh * dv;
            for (int i = 0; i < dv; i++) o[i] += p * v[i];
        }
    }
}
static void attend(int l, int pos, const float *q, float *out) {
    att_arg aa = {l, pos, q, out};
    if (NT == 1) att_task(0, 1, &aa); else es_cpool_run(CP, att_task, &aa);
}

/* ---------------- experts ---------------- */
static const char *PACK;
static es_pool *IO;
static es_cache C;
static double STAT_WAIT_S, STAT_READ_MB;
static unsigned long long STAT_HITS, STAT_MISS;
static FILE *TRACE;
static int TRACE_TOKEN;

/* direct GGUF mode: the three expert tensors of each layer (one tensor per role holding every expert),
 * or, in older Mixtral files, one tensor per expert and role */
static const es_gguf_tensor **XT;   /* [layer*3 + {gate, up, down}] */
static const es_gguf_tensor **XTE;  /* [(layer*n_exp + expert)*3 + role] */
static size_t XOFF[3];              /* offsets of gate/up/down inside a slot */

static const es_tensor_desc *find_role(const es_header *h, uint32_t role) {
    for (uint32_t i = 0; i < h->n_tensors; i++)
        if (h->t[i].role == role) return &h->t[i];
    return NULL;
}

/* split models: shard 0 is the file given; later shards are extra files with their own data area */
static int SH_FD[256];
static uint64_t SH_DS[256];
static uint64_t shard_base(const es_gguf_tensor *t) { return t->shard ? SH_DS[t->shard] : G.data_start; }
static void read_at_fd(int fd, uint64_t off, void *dst, size_t n) {
    uint8_t *p = dst;
    while (n) {
        ssize_t r = es_gguf_pread(fd, p, n, off);
        if (r <= 0) { fprintf(stderr, "read failed at %llu\n", (unsigned long long)off); exit(1); }
        p += r; n -= (size_t)r; off += (uint64_t)r;
    }
}
static double SIM_S_PER_BYTE;   /* ES_SIM_FLASH_MBPS=500: pretend each reader gets that speed (to test slow storage on a fast disk) */
static void read_tensor(const es_gguf_tensor *t, uint64_t extra, void *dst, size_t n) {
    if (SIM_S_PER_BYTE > 0) { struct timespec ts = {0, (long)(n * SIM_S_PER_BYTE * 1e9)}; nanosleep(&ts, NULL); }
    read_at_fd(t->shard ? SH_FD[t->shard] : G.fd, shard_base(t) + t->offset + extra, dst, n);
}
/* expert e of layer l, role r (gate, up, down) into dst */
static void read_expert_part(int l, int e, int r, void *dst) {
    if (XTE) { const es_gguf_tensor *t = XTE[((size_t)l * M.n_exp + e) * 3 + r]; read_tensor(t, 0, dst, t->nbytes); return; }
    const es_gguf_tensor *x = XT[l * 3 + r];
    const uint64_t per = x->nbytes / x->ne[2];
    read_tensor(x, per * (uint64_t)e, dst, per);
}
static const es_gguf_tensor *expert_tensor(int l, int r) { return XTE ? XTE[(size_t)l * M.n_exp * 3 + r] : XT[l * 3 + r]; }

/* ================= ExpertStream Lookahead =================
 * 1. parallel loader: every missing expert of a layer is read by a small thread pool at once
 * 2. lookahead: the next layer's router is run on the current hidden state to guess its experts,
 *    which are loaded in the background while this layer computes
 * 3. adaptive skipping: experts with a tiny router weight are not loaded at all
 * 4. warm start: the experts used most last time are preloaded before the first question */
#ifndef __EMSCRIPTEN__
#include <pthread.h>
#include <sched.h>
static int GIO_ON, LOOKAHEAD;
static float SKIP_THR;
typedef struct { int slot, l, e; } gtask;
static struct { pthread_mutex_t mu; pthread_cond_t cv, cv_done; gtask *dq, *pq; int dh, dn, ph, pn, cap, stop; pthread_t th[16]; int nth; } GQ;
static unsigned long long STAT_PF_ISSUED, STAT_PF_USED, STAT_SKIPPED;
static unsigned *PROF;            /* per (layer, expert) use counts */
static char PROF_PATH[4096];

static void gio_load(const gtask *t) {
    es_slot *s = &C.slots[t->slot];
    for (int r = 0; r < 3; r++) read_expert_part(t->l, t->e, r, (uint8_t *)s->buf + XOFF[r]);
}
static void *gio_worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&GQ.mu);
        while (!GQ.stop && !GQ.dn && !GQ.pn) pthread_cond_wait(&GQ.cv, &GQ.mu);
        if (GQ.stop) { pthread_mutex_unlock(&GQ.mu); return NULL; }
        gtask t;
        if (GQ.dn) { t = GQ.dq[GQ.dh]; GQ.dh = (GQ.dh + 1) % GQ.cap; GQ.dn--; }
        else { t = GQ.pq[GQ.ph]; GQ.ph = (GQ.ph + 1) % GQ.cap; GQ.pn--; }
        pthread_mutex_unlock(&GQ.mu);
        gio_load(&t);
        __atomic_store_n(&C.slots[t.slot].req.done, 1, __ATOMIC_RELEASE);
        pthread_mutex_lock(&GQ.mu);
        pthread_cond_broadcast(&GQ.cv_done);
        pthread_mutex_unlock(&GQ.mu);
    }
}
static void gio_start(int nthreads, int nslots) {
    GQ.cap = nslots + 8; GQ.dq = malloc(sizeof(gtask) * (size_t)GQ.cap); GQ.pq = malloc(sizeof(gtask) * (size_t)GQ.cap);
    pthread_mutex_init(&GQ.mu, NULL); pthread_cond_init(&GQ.cv, NULL); pthread_cond_init(&GQ.cv_done, NULL);
    GQ.nth = nthreads > 16 ? 16 : nthreads < 1 ? 1 : nthreads;
    for (int i = 0; i < GQ.nth; i++) pthread_create(&GQ.th[i], NULL, gio_worker, NULL);
    GIO_ON = 1;
}
static void gio_enqueue(int slot, int l, int e, int urgent) {
    C.slots[slot].req.done = 0;
    pthread_mutex_lock(&GQ.mu);
    gtask t = {slot, l, e};
    if (urgent) { GQ.dq[(GQ.dh + GQ.dn) % GQ.cap] = t; GQ.dn++; } else { GQ.pq[(GQ.ph + GQ.pn) % GQ.cap] = t; GQ.pn++; }
    pthread_cond_signal(&GQ.cv);
    pthread_mutex_unlock(&GQ.mu);
}
static void gio_wait(int slot) {
    if (__atomic_load_n(&C.slots[slot].req.done, __ATOMIC_ACQUIRE)) return;
    pthread_mutex_lock(&GQ.mu);
    while (!__atomic_load_n(&C.slots[slot].req.done, __ATOMIC_ACQUIRE)) pthread_cond_wait(&GQ.cv_done, &GQ.mu);
    pthread_mutex_unlock(&GQ.mu);
}
static float *PFX, *PFR;
/* guess the experts layer l2 will choose from the hidden state x, and start loading them */
static void prefetch_layer(int l2, const float *x) {
    if (!GIO_ON || !LOOKAHEAD || l2 >= M.n_layer || !M.L[l2].moe) return;
    const int ne = M.n_exp, T = M.n_used + 2;
    rmsnorm(PFX, x, M.L[l2].ffn_norm.data, M.n_embd, M.eps);
    es_act a = {.x = PFX, .q8 = NULL, .q80 = NULL, .cols = M.n_embd};
    es_matvec(M.L[l2].gate_inp.type, M.L[l2].gate_inp.data, M.n_embd, 0, ne, &a, PFR, SCRATCH[0]);
    if (M.L[l2].exp_probs_b.data) { const float *b = M.L[l2].exp_probs_b.data; for (int e = 0; e < ne; e++) PFR[e] = (M.gating == GATE_SIGMOID ? 1.0f / (1.0f + expf(-PFR[e])) : PFR[e]) + b[e]; }
    for (int j = 0; j < T && j < ne; j++) {
        int best = -1;
        for (int e = 0; e < ne; e++) {
            if (PFR[e] == -INFINITY) continue;
            if (best < 0 || PFR[e] > PFR[best]) best = e;
        }
        if (best < 0) break;
        PFR[best] = -INFINITY;
        if (es_cache_find(&C, l2, best) >= 0) continue;
        const int s = es_cache_claim(&C, NULL, l2, best);
        if (s < 0) return;
        C.slots[s].prefetched = 1;
        C.slots[s].protect_until = C.step + 3;
        gio_enqueue(s, l2, best, 0);
        STAT_PF_ISSUED++; STAT_READ_MB += C.slot_bytes / 1048576.0;
    }
}
void es_engine_save_profile(void) {
    if (!PROF || !PROF_PATH[0]) return;
    FILE *f = fopen(PROF_PATH, "wb");
    if (!f) return;
    uint32_t hdr[3] = {0x50534545u, (uint32_t)M.n_layer, (uint32_t)M.n_exp};
    fwrite(hdr, 4, 3, f); fwrite(PROF, 4, (size_t)M.n_layer * (size_t)M.n_exp, f); fclose(f);
}
static void warm_start(int want) {
    if (!GIO_ON || !PROF) return;
    FILE *f = fopen(PROF_PATH, "rb");
    if (!f) return;
    uint32_t hdr[3];
    if (fread(hdr, 4, 3, f) != 3 || hdr[0] != 0x50534545u || (int)hdr[1] != M.n_layer || (int)hdr[2] != M.n_exp) { fclose(f); return; }
    const size_t n = (size_t)M.n_layer * (size_t)M.n_exp;
    if (fread(PROF, 4, n, f) != n) { fclose(f); return; }
    fclose(f);
    int loaded = 0, *order = malloc(sizeof(int) * n);
    for (size_t i = 0; i < n; i++) order[i] = (int)i;
    /* take the `want` most used (selection by repeated max is fine: want is a few thousand at most) */
    int *pick = malloc(sizeof(int) * (size_t)want);
    for (int w = 0; w < want; w++) {
        int b = -1;
        for (size_t i = 0; i < n; i++) if (order[i] >= 0 && PROF[order[i]] > 0 && (b < 0 || PROF[order[i]] > PROF[order[b]])) b = (int)i;
        if (b < 0) break;
        pick[loaded++] = order[b]; order[b] = -1;
    }
    for (int i = 0; i < loaded; i++) {
        const int l = pick[i] / M.n_exp, e = pick[i] % M.n_exp;
        const int s = es_cache_claim(&C, NULL, l, e);
        if (s < 0) break;
        C.slots[s].protect_until = 0; C.slots[s].uses = 1; C.slots[s].last_use = 0;
        gio_enqueue(s, l, e, 1);
    }
    for (int i = 0; i < loaded; i++) { const int s = es_cache_find(&C, pick[i] / M.n_exp, pick[i] % M.n_exp); if (s >= 0) { gio_wait(s); C.slots[s].state = ES_SLOT_READY; } }
    STAT_READ_MB += loaded * C.slot_bytes / 1048576.0;
    free(order); free(pick);
}
#else
#define GIO_ON 0
#endif

/* Make sure the k experts of layer l are in the RAM cache; fill W views. All k stay protected for this step. */
static void load_experts(int l, const int *sel, int k, W *g, W *u, W *d) {
    C.step++;
    int *slot = malloc(sizeof(int) * (size_t)k);
    char path[4096];
    for (int j = 0; j < k; j++) {
        int s = es_cache_find(&C, l, sel[j]);
        if (s >= 0) {
            STAT_HITS++;
#ifndef __EMSCRIPTEN__
            if (C.slots[s].prefetched) { STAT_PF_USED++; C.slots[s].prefetched = 0; }
#endif
        } else {
            STAT_MISS++;
            s = es_cache_claim(&C, IO, l, sel[j]);
            if (s < 0) { fprintf(stderr, "expert cache too small\n"); exit(1); }
            if (PACK) {
                es_expert_path(path, sizeof path, PACK, (unsigned)l, (unsigned)sel[j]);
                C.slots[s].req.dst = C.slots[s].buf;
                C.slots[s].req.cap = C.slot_bytes;
                if (es_submit(IO, &C.slots[s].req, path)) {
                    fprintf(stderr, "cannot read %s: %s\n", path, strerror(C.slots[s].req.err));
                    exit(1);
                }
            } else if (GIO_ON) {
#ifndef __EMSCRIPTEN__
                gio_enqueue(s, l, sel[j], 1);
#endif
            } else {
                uint64_t t0 = es_now_ns();
                for (int r = 0; r < 3; r++) read_expert_part(l, sel[j], r, (uint8_t *)C.slots[s].buf + XOFF[r]);
                STAT_WAIT_S += (es_now_ns() - t0) / 1e9;
                C.slots[s].state = ES_SLOT_READY;
            }
            STAT_READ_MB += C.slot_bytes / 1048576.0;
        }
        C.slots[s].uses++;
        C.slots[s].last_use = C.step;
        C.slots[s].protect_until = C.step;
        slot[j] = s;
    }
    uint64_t t0 = es_now_ns();
#ifndef __EMSCRIPTEN__
    if (GIO_ON) {
        for (int j = 0; j < k; j++) { gio_wait(slot[j]); C.slots[slot[j]].state = ES_SLOT_READY; if (PROF) PROF[(size_t)l * M.n_exp + sel[j]]++; }
        STAT_WAIT_S += (es_now_ns() - t0) / 1e9;
        t0 = es_now_ns();
    }
#endif
    for (int j = 0; j < k; j++) {
        es_slot *s = &C.slots[slot[j]];
        if (!PACK) {
            const es_gguf_tensor *tg = expert_tensor(l, 0), *tu = expert_tensor(l, 1), *td = expert_tensor(l, 2);
            g[j] = (W){tg->type, (int)tg->ne[1], (int)tg->ne[0], (uint8_t *)s->buf + XOFF[0]};
            u[j] = (W){tu->type, (int)tu->ne[1], (int)tu->ne[0], (uint8_t *)s->buf + XOFF[1]};
            d[j] = (W){td->type, (int)td->ne[1], (int)td->ne[0], (uint8_t *)s->buf + XOFF[2]};
            continue;
        }
        if (s->state == ES_SLOT_LOADING) {
            if (es_wait(IO, &s->req)) { fprintf(stderr, "expert read failed: %s\n", strerror(s->req.err)); exit(1); }
            const char *why = es_validate(s->buf, s->req.bytes, 0);
            if (why) { fprintf(stderr, "bad expert file L%d E%d: %s\n", l, sel[j], why); exit(1); }
            s->state = ES_SLOT_READY;
        }
        const es_header *h = s->buf;
        const es_tensor_desc *tg = find_role(h, ES_ROLE_GATE), *tu = find_role(h, ES_ROLE_UP), *td = find_role(h, ES_ROLE_DOWN);
        g[j] = (W){tg->qtype & 0xFFF, (int)tg->rows, (int)tg->cols, (uint8_t *)s->buf + tg->offset};
        u[j] = (W){tu->qtype & 0xFFF, (int)tu->rows, (int)tu->cols, (uint8_t *)s->buf + tu->offset};
        d[j] = (W){td->qtype & 0xFFF, (int)td->rows, (int)td->cols, (uint8_t *)s->buf + td->offset};
    }
    if (PACK) STAT_WAIT_S += (es_now_ns() - t0) / 1e9;
    free(slot);
}

/* ---------------- routing ---------------- */
/* choose the experts of one token from its router logits r[n_exp]; returns how many (k) */
static int route(int l, float *r, int *sel, float *w) {
    const layer *L = &M.L[l];
    const int ne = M.n_exp;
    int k = M.n_used;
    add_bias(r, &L->gate_inp_b, ne);
    if (M.gating == GATE_SOFTMAX) {
        float mx = -INFINITY, sum = 0;
        for (int e = 0; e < ne; e++) mx = r[e] > mx ? r[e] : mx;
        for (int e = 0; e < ne; e++) { r[e] = expf(r[e] - mx); sum += r[e]; }
        for (int e = 0; e < ne; e++) r[e] /= sum;
    } else if (M.gating == GATE_SIGMOID) {
        for (int e = 0; e < ne; e++) r[e] = 1.0f / (1.0f + expf(-r[e]));
    }
    /* selection scores: the probabilities plus the selection bias; whole groups may be ruled out first */
    float selp[1024];
    const float *bias = L->exp_probs_b.data;
    for (int e = 0; e < ne; e++) selp[e] = r[e] + (bias ? bias[e] : 0.0f);
    if (M.n_group > 1 && M.n_group_used < M.n_group) {
        const int per = ne / M.n_group;
        float gs[64]; int keep[64] = {0};
        for (int g = 0; g < M.n_group; g++) {   /* group score: its two best experts */
            float a = -INFINITY, b = -INFINITY;
            for (int e = g * per; e < (g + 1) * per; e++) { const float v = selp[e]; if (v > a) { b = a; a = v; } else if (v > b) b = v; }
            gs[g] = a + b;
        }
        for (int j = 0; j < M.n_group_used; j++) {
            int best = -1;
            for (int g = 0; g < M.n_group; g++) if (!keep[g] && (best < 0 || gs[g] > gs[best])) best = g;
            keep[best] = 1;
        }
        for (int g = 0; g < M.n_group; g++) if (!keep[g]) for (int e = g * per; e < (g + 1) * per; e++) selp[e] = -INFINITY;
    }
    float wsum = 0;
    for (int j = 0; j < k; j++) {
        int best = -1;
        for (int e = 0; e < ne; e++) {
            int used = 0;
            for (int i = 0; i < j; i++) used |= sel[i] == e;
            if (!used && (best < 0 || selp[e] > selp[best])) best = e;
        }
        sel[j] = best;
        w[j] = r[best];
        wsum += w[j];
    }
    if (M.gating == GATE_SOFTMAX_CHOSEN) {
        float mx = -INFINITY, s = 0;
        for (int j = 0; j < k; j++) mx = w[j] > mx ? w[j] : mx;
        for (int j = 0; j < k; j++) { w[j] = expf(w[j] - mx); s += w[j]; }
        for (int j = 0; j < k; j++) w[j] /= s;
        wsum = 1;
    }
#ifndef __EMSCRIPTEN__
    if (SKIP_THR > 0 && k > 1) {
        float mxw = 0; for (int j = 0; j < k; j++) if (w[j] > mxw) mxw = w[j];
        int kk = 0; wsum = 0;
        for (int j = 0; j < k; j++) if (w[j] >= SKIP_THR * mxw) { sel[kk] = sel[j]; w[kk] = w[j]; wsum += w[j]; kk++; }
        STAT_SKIPPED += (unsigned long long)(k - kk); k = kk;
    }
#endif
    if (M.norm_topk) { const float d = wsum > 6.103515625e-5f ? wsum : 6.103515625e-5f; for (int j = 0; j < k; j++) w[j] /= d; }
    if (M.w_scale != 0.0f && M.w_scale != 1.0f) for (int j = 0; j < k; j++) w[j] *= M.w_scale;
    return k;
}

static inline float act_oai(float g, float u) {   /* gpt-oss: clamped SwiGLU, alpha 1.702, limit 7 */
    const float x = g < 7.0f ? g : 7.0f, y = u < -7.0f ? -7.0f : u > 7.0f ? 7.0f : u;
    return x / (1.0f + expf(1.702f * -x)) * (y + 1.0f);
}

/* experts for N tokens: input BXN (prepared in BACT), output BTMP */
static void moe_n(int l, int N) {
    const layer *L = &M.L[l];
    const int ne = M.n_exp, n = M.n_embd;
    jobn(&L->gate_inp, BACT, BROUT, (size_t)ne, N);
    run_jobs();
    int sel[BMAX][MAXK], kk[BMAX];
    float w[BMAX][MAXK];
    for (int i = 0; i < N; i++) {
        kk[i] = route(l, BROUT + (size_t)i * ne, sel[i], w[i]);
        if (TRACE) {
            fprintf(TRACE, "%d %d", TRACE_TOKEN + i, l);
            for (int j = 0; j < kk[i]; j++) fprintf(TRACE, " %d", sel[i][j]);
            fputc('\n', TRACE);
        }
    }
#ifndef __EMSCRIPTEN__
    if (LOOKAHEAD && N == 1) prefetch_layer(l + 1, BX);
#endif
    /* the distinct experts, in order of first use; each assignment (token, choice) gets a row grouped by expert */
    int uex[BMAX * MAXK], nu = 0, rowof[BMAX][MAXK], tok_of[BMAX * MAXK];
    int cnt[BMAX * MAXK];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < kk[i]; j++) {
            int u = 0;
            while (u < nu && uex[u] != sel[i][j]) u++;
            if (u == nu) { uex[nu] = sel[i][j]; cnt[nu] = 0; nu++; }
            cnt[u]++;
        }
    int start[BMAX * MAXK], fill[BMAX * MAXK];
    for (int u = 0, r = 0; u < nu; u++) { start[u] = r; fill[u] = 0; r += cnt[u]; }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < kk[i]; j++) {
            int u = 0;
            while (uex[u] != sel[i][j]) u++;
            const int r = start[u] + fill[u]++;
            rowof[i][j] = r; tok_of[r] = i;
        }
    int ff = 0;
    static es_act xa[BMAX * MAXK], ha[BMAX * MAXK];
    static W g[BMAX * MAXK], u_[BMAX * MAXK], d[BMAX * MAXK];
    /* as many experts at a time as the cache can hold */
    const int chunk = C.nslots / 2 > 0 ? C.nslots / 2 : 1;
    for (int c0 = 0; c0 < nu; c0 += chunk) {
        const int cn = nu - c0 < chunk ? nu - c0 : chunk;
        load_experts(l, uex + c0, cn, g + c0, u_ + c0, d + c0);
        ff = g[c0].rows;
        for (int u = c0; u < c0 + cn; u++) {
            for (int q = 0; q < cnt[u]; q++) xa[start[u] + q] = BACT[tok_of[start[u] + q]];
            jobn(&g[u], xa + start[u], BGU + (size_t)start[u] * 2 * ff, (size_t)2 * ff, cnt[u]);
            jobn(&u_[u], xa + start[u], BGU + (size_t)start[u] * 2 * ff + ff, (size_t)2 * ff, cnt[u]);
        }
        run_jobs();
        for (int u = c0; u < c0 + cn; u++) {
            const int e = uex[u];
            for (int q = 0; q < cnt[u]; q++) {
                const int r = start[u] + q;
                float *gg = BGU + (size_t)r * 2 * ff, *uu = gg + ff, *hh = BH + (size_t)r * ff;
                if (L->gate_exps_b.data) { const float *b = (const float *)L->gate_exps_b.data + (size_t)e * ff; for (int i = 0; i < ff; i++) gg[i] += b[i]; }
                if (L->up_exps_b.data) { const float *b = (const float *)L->up_exps_b.data + (size_t)e * ff; for (int i = 0; i < ff; i++) uu[i] += b[i]; }
                if (M.act_oai) for (int i = 0; i < ff; i++) hh[i] = act_oai(gg[i], uu[i]);
                else for (int i = 0; i < ff; i++) hh[i] = gg[i] / (1.0f + expf(-gg[i])) * uu[i];
                prep(&ha[r], hh, ff);
            }
            jobn(&d[u], ha + start[u], BYE + (size_t)start[u] * n, (size_t)n, cnt[u]);
        }
        run_jobs();
        if (L->down_exps_b.data)
            for (int u = c0; u < c0 + cn; u++) {
                const float *b = (const float *)L->down_exps_b.data + (size_t)uex[u] * n;
                for (int q = 0; q < cnt[u]; q++) { float *y = BYE + (size_t)(start[u] + q) * n; for (int i = 0; i < n; i++) y[i] += b[i]; }
            }
    }
    for (int t = 0; t < N; t++) {
        float *out = BTMP + (size_t)t * n;
        for (int i = 0; i < n; i++) {
            float s = 0;
            for (int j = 0; j < kk[t]; j++) s += w[t][j] * BYE[(size_t)rowof[t][j] * n + i];
            out[i] = s;
        }
    }
}

/* dense SwiGLU FFN for N tokens: input prepared in acts, output out (stride n_embd) */
static void dense_n(const W *wg, const W *wu, const W *wd, const es_act *acts, int N, float *out) {
    const int ff = wg->rows;
    jobn(wg, acts, BGU, (size_t)2 * ff, N); jobn(wu, acts, BGU + ff, (size_t)2 * ff, N);
    run_jobs();
    for (int i = 0; i < N; i++) {
        const float *g = BGU + (size_t)i * 2 * ff; float *h = BH + (size_t)i * ff;
        for (int j = 0; j < ff; j++) h[j] = g[j] / (1.0f + expf(-g[j])) * g[ff + j];
        prep(&BACT2[i], h, ff);
    }
    jobn(wd, BACT2, out, (size_t)M.n_embd, N);
    run_jobs();
}

/* attention for N tokens at positions pos0.. : input BXN prepared in BACT, output BTMP */
static void attention_n(int l, int N, int pos0) {
    const layer *L = &M.L[l];
    const int n = M.n_embd;
    if (M.kind != AR_DS2) {
        const int hd = M.head_dim, kvd = M.kvk, qd = M.n_head * hd;
        jobn(&L->q, BACT, BQ, (size_t)qd, N); jobn(&L->k, BACT, BKV, (size_t)kvd, N); jobn(&L->v, BACT, BV2, (size_t)kvd, N);
        run_jobs();
        for (int i = 0; i < N; i++) {
            float *q = BQ + (size_t)i * qd, *k = BKV + (size_t)i * kvd, *v = BV2 + (size_t)i * kvd;
            add_bias(q, &L->bq, qd); add_bias(k, &L->bk, kvd); add_bias(v, &L->bv, kvd);
            if (L->q_norm.data) {
                if (L->q_norm.cols == n) rmsnorm(q, q, L->q_norm.data, qd, M.eps);
                else for (int h = 0; h < M.n_head; h++) rmsnorm(q + h * hd, q + h * hd, L->q_norm.data, hd, M.eps);
            }
            if (L->k_norm.data) {
                if (L->k_norm.cols == kvd && kvd != hd) rmsnorm(k, k, L->k_norm.data, kvd, M.eps);
                else for (int h = 0; h < M.n_head_kv; h++) rmsnorm(k + h * hd, k + h * hd, L->k_norm.data, hd, M.eps);
            }
            rope(q, M.n_head, hd, pos0 + i); rope(k, M.n_head_kv, hd, pos0 + i);
            memcpy(KC + ((size_t)l * M.ctx + pos0 + i) * kvd, k, sizeof(float) * (size_t)kvd);
            memcpy(VC + ((size_t)l * M.ctx + pos0 + i) * kvd, v, sizeof(float) * (size_t)kvd);
        }
        for (int i = 0; i < N; i++) attend(l, pos0 + i, BQ + (size_t)i * qd, BATTN + (size_t)i * qd);   /* causal: all earlier rows are written */
    } else {
        /* MLA: q = q_b(norm(q_a(x))) per head [nope | rope]; the cache row is [norm(c) | rope(k_pe)], values are c */
        const int H = M.n_head, nope = M.nope, rd = M.rope_d, kvl = M.kv_lora, qh = nope + rd, kvw = kvl + rd;
        if (L->q_a.data) jobn(&L->q_a, BACT, BQA, (size_t)M.q_lora, N); else jobn(&L->q, BACT, BQ, (size_t)H * qh, N);
        jobn(&L->kv_a, BACT, BKV, (size_t)kvw, N);
        run_jobs();
        if (L->q_a.data) {
            for (int i = 0; i < N; i++) { float *qa = BQA + (size_t)i * M.q_lora; rmsnorm(qa, qa, L->q_a_norm.data, M.q_lora, M.eps); prep(&BACT2[i], qa, M.q_lora); }
            jobn(&L->q_b, BACT2, BQ, (size_t)H * qh, N);
            run_jobs();
        }
        static es_act qn[BMAX * 256];
        for (int i = 0; i < N; i++) {
            float *kv = BKV + (size_t)i * kvw, *row = KC + ((size_t)l * M.ctx + pos0 + i) * M.kvk;
            rmsnorm(row, kv, L->kv_a_norm.data, kvl, M.eps);
            memcpy(row + kvl, kv + kvl, sizeof(float) * (size_t)rd);
            rope(row + kvl, 1, rd, pos0 + i);
            float *q = BQ + (size_t)i * H * qh;
            rope(q + nope, H, qh, pos0 + i);
            for (int h = 0; h < H; h++) {   /* absorbed: q_lat[h] = k_b[h] . q_nope[h] */
                prep(&qn[i * H + h], q + (size_t)h * qh, nope);
                W kb = rows_of(&L->k_b, h * kvl, kvl);
                jobn(&kb, &qn[i * H + h], BQM + ((size_t)i * H + h) * M.dk, 0, 1);
            }
        }
        run_jobs();
        for (int i = 0; i < N; i++)
            for (int h = 0; h < H; h++) memcpy(BQM + ((size_t)i * H + h) * M.dk + kvl, BQ + (size_t)i * H * qh + (size_t)h * qh + nope, sizeof(float) * (size_t)rd);
        for (int i = 0; i < N; i++) attend(l, pos0 + i, BQM + (size_t)i * H * M.dk, BATL + (size_t)i * H * kvl);
        arena_reset();
        for (int i = 0; i < N; i++)
            for (int h = 0; h < H; h++) {   /* back from the latent space: v_b[h] . out_lat[h] */
                prep(&qn[i * H + h], BATL + ((size_t)i * H + h) * kvl, kvl);
                W vb = rows_of(&L->v_b, h * M.v_head, M.v_head);
                jobn(&vb, &qn[i * H + h], BATTN + ((size_t)i * H + h) * M.v_head, 0, 1);
            }
        run_jobs();
    }
    arena_reset();
    for (int i = 0; i < N; i++) prep(&BACT2[i], BATTN + (size_t)i * M.od, M.od);
    jobn(&L->o, BACT2, BTMP, (size_t)n, N);
    run_jobs();
    for (int i = 0; i < N; i++) add_bias(BTMP + (size_t)i * n, &L->bo, n);
}

/* N tokens through the whole model; logits of tokens logits_from.. go to BLOG rows 0.. */
static void forward_n(const int32_t *toks, int N, int pos0, int logits_from) {
    const int n = M.n_embd;
    for (int i = 0; i < N; i++) es_dequant_row(M.tok_embd.type, (const uint8_t *)M.tok_embd.data + es_row_bytes(M.tok_embd.type, n) * (size_t)toks[i], BX + (size_t)i * n, n);
    TRACE_TOKEN = pos0;
    for (int l = 0; l < M.n_layer; l++) {
        layer *L = &M.L[l];
        arena_reset();
        for (int i = 0; i < N; i++) { rmsnorm(BXN + (size_t)i * n, BX + (size_t)i * n, L->attn_norm.data, n, M.eps); prep(&BACT[i], BXN + (size_t)i * n, n); }
        attention_n(l, N, pos0);
        for (size_t i = 0; i < (size_t)N * n; i++) BX[i] += BTMP[i];
        arena_reset();
        for (int i = 0; i < N; i++) { rmsnorm(BXN + (size_t)i * n, BX + (size_t)i * n, L->ffn_norm.data, n, M.eps); prep(&BACT[i], BXN + (size_t)i * n, n); }
        if (L->moe) moe_n(l, N);
        else dense_n(&L->ffn_gate, &L->ffn_up, &L->ffn_down, BACT, N, BTMP);
        if (L->sh_gate.data) {
            for (size_t i = 0; i < (size_t)N * n; i++) BX[i] += BTMP[i];
            dense_n(&L->sh_gate, &L->sh_up, &L->sh_down, BACT, N, BTMP);
            if (L->sh_gate_inp.data) {   /* Qwen2-MoE: the shared expert is scaled by sigmoid(gate . x) */
                float gt[BMAX];
                jobn(&L->sh_gate_inp, BACT, gt, 1, N);
                run_jobs();
                for (int i = 0; i < N; i++) { const float s = 1.0f / (1.0f + expf(-gt[i])); for (int j = 0; j < n; j++) BTMP[(size_t)i * n + j] *= s; }
            }
        }
        for (size_t i = 0; i < (size_t)N * n; i++) BX[i] += BTMP[i];
    }
    if (logits_from >= N) return;
    arena_reset();
    const int cnt = N - logits_from;
    for (int j = 0; j < cnt; j++) { rmsnorm(BXN + (size_t)j * n, BX + (size_t)(logits_from + j) * n, M.output_norm.data, n, M.eps); prep(&BACT[j], BXN + (size_t)j * n, n); }
    jobn(&M.output, BACT, BLOG, (size_t)M.n_vocab, cnt);
    run_jobs();
}
/* one token through the whole model; writes LOGITS if want_logits */
static void forward(int32_t tok, int pos, int want_logits) {
    forward_n(&tok, 1, pos, want_logits ? 0 : 1);
    if (want_logits) memcpy(LOGITS, BLOG, sizeof(float) * (size_t)M.n_vocab);
}

/* ---------------- sampling ---------------- */
static uint64_t RNG = 42;
/* the k largest logits, sorted: one pass over the vocabulary, insertion only when a value beats the current k-th */
static int topk_select(const float *lg, int nv, int k, int *idx) {
    int m = 0;
    for (int i = 0; i < nv; i++) {
        const float v = lg[i];
        if (m == k && v <= lg[idx[m - 1]]) continue;
        int j = m < k ? m++ : m - 1;
        while (j > 0 && lg[idx[j - 1]] < v) { idx[j] = idx[j - 1]; j--; }
        idx[j] = i;
    }
    return m;
}
static int S_IDX[1 << 18]; static float S_PR[1 << 18]; static int S_KEEP; static float S_CUM;
/* the distribution sample() draws from: temperature, top-k, then the smallest set whose mass reaches top-p */
static void build_dist(const float *lg, float temp, int top_k, float top_p) {
    const int nv = M.n_vocab;
    if (top_k <= 0 || top_k > nv) top_k = nv;
    const int n = topk_select(lg, nv, top_k, S_IDX);
    float mx = lg[S_IDX[0]], sum = 0;
    for (int i = 0; i < n; i++) { S_PR[i] = expf((lg[S_IDX[i]] - mx) / temp); sum += S_PR[i]; }
    float cum = 0; int keep = n;
    for (int i = 0; i < n; i++) { S_PR[i] /= sum; cum += S_PR[i]; if (cum >= top_p) { keep = i + 1; break; } }
    S_KEEP = keep; S_CUM = cum;
}
static float dist_prob(int tok) {
    for (int i = 0; i < S_KEEP; i++) if (S_IDX[i] == tok) return S_PR[i] / S_CUM;
    return 0.0f;
}
static float rng_unit(void) { return (float)((es_rng_next(&RNG) >> 11) * (1.0 / 9007199254740992.0)); }
static int argmax_of(const float *lg) {
    int best = 0;
    for (int i = 1; i < M.n_vocab; i++) if (lg[i] > lg[best]) best = i;
    return best;
}
static int sample(float temp, int top_k, float top_p) {
    if (temp <= 0) return argmax_of(LOGITS);
    build_dist(LOGITS, temp, top_k, top_p);
    float r = rng_unit() * S_CUM, acc = 0;
    for (int i = 0; i < S_KEEP; i++) { acc += S_PR[i]; if (r <= acc) return S_IDX[i]; }
    return S_IDX[S_KEEP - 1];
}

/* ---------------- prompt formatting ---------------- */
static const char *TEMPLATE_KIND = "raw";
#define DS_BOS "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>"
#define DS_EOS "<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>"
#define DS_USER "<\xef\xbd\x9cUser\xef\xbd\x9c>"
#define DS_ASSISTANT "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>"
#define KIMI_SYS "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|>"
static int TK(const char *k) { return !strcmp(TEMPLATE_KIND, k); }
static const char *template_kind_of(const char *t) {
    if (!t) return "raw";
    if (strstr(t, "<|im_middle|>")) return "kimi";
    if (strstr(t, "<|channel|>") || (strstr(t, "<|start|>") && strstr(t, "<|message|>"))) return "harmony";
    if (strstr(t, DS_USER)) return "deepseek";
    if (strstr(t, "<|im_start|>")) return "chatml";
    if (strstr(t, "<|start_header_id|>")) return "llama3";
    if (strstr(t, "<|user|>")) return "tulu";
    if (strstr(t, "' [INST] '")) return "mistral_sp";   /* newer Mistral templates: a space before [INST] and before each answer */
    if (strstr(t, "[INST]")) return "mistral";
    return "raw";
}
static void detect_template(void) { TEMPLATE_KIND = template_kind_of(es_gguf_str(&G, "tokenizer.chat_template")); }
static void harmony_system(char *out, size_t cap) {
    char date[32] = "2025-01-01";
    const time_t now = time(NULL);
    struct tm tmv;
    if (localtime_r(&now, &tmv)) strftime(date, sizeof date, "%Y-%m-%d", &tmv);
    snprintf(out, cap, "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: %s\n\n"
             "Reasoning: medium\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>", date);
}
static void format_turn(char *out, size_t cap, const char *msg, int first) {
    /* later turns first close the previous answer: its end token was
     * sampled but not fed back into the model */
    if (TK("chatml"))
        snprintf(out, cap, "%s<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", first ? "" : "<|im_end|>\n", msg);
    else if (TK("tulu"))
        snprintf(out, cap, "%s<|user|>\n%s\n<|assistant|>\n", first ? "<|endoftext|>" : "<|endoftext|>\n", msg);
    else if (TK("llama3"))
        snprintf(out, cap, "%s<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", first ? "<|begin_of_text|>" : "<|eot_id|>", msg);
    else if (TK("mistral"))
        snprintf(out, cap, "%s[INST] %s [/INST]", first ? "<s>" : "</s>", msg);
    else if (TK("mistral_sp"))
        snprintf(out, cap, "%s [INST] %s [/INST]", first ? "<s>" : "</s>", msg);
    else if (TK("deepseek"))
        snprintf(out, cap, "%s" DS_USER "%s" DS_ASSISTANT, first ? DS_BOS : DS_EOS, msg);
    else if (TK("kimi"))
        snprintf(out, cap, "%s<|im_user|>user<|im_middle|>%s<|im_end|><|im_assistant|>assistant<|im_middle|>", first ? KIMI_SYS : "<|im_end|>", msg);
    else if (TK("harmony")) {
        char sys[512] = "";
        if (first) harmony_system(sys, sizeof sys);
        snprintf(out, cap, "%s<|start|>user<|message|>%s<|end|><|start|>assistant", first ? sys : "<|end|>", msg);
    } else
        snprintf(out, cap, "%s", msg);
}


int es_engine_render(const es_msg *m, int n, char *out, size_t cap) {
    size_t o = 0;
#define PUT(...) do { int w_ = snprintf(out + o, o < cap ? cap - o : 0, __VA_ARGS__); if (w_ < 0 || o + (size_t)w_ >= cap) return -1; o += (size_t)w_; } while (0)
    const int chatml = TK("chatml"), tulu = TK("tulu"), llama3 = TK("llama3"), msp = TK("mistral_sp"), mistral = TK("mistral") || msp, ds = TK("deepseek"), kimi = TK("kimi"), harmony = TK("harmony");
    const char *sp = msp ? " " : "";
    if (tulu) PUT("<|endoftext|>");
    if (llama3) PUT("<|begin_of_text|>");
    if (mistral) PUT("<s>");
    if (ds) {
        PUT(DS_BOS);
        for (int i = 0; i < n; i++) if (m[i].role && (!strcmp(m[i].role, "system") || !strcmp(m[i].role, "developer"))) PUT("%s", m[i].content ? m[i].content : "");
    }
    if (kimi) { int has = 0; for (int i = 0; i < n; i++) has |= m[i].role && !strcmp(m[i].role, "system"); if (!has) PUT(KIMI_SYS); }
    if (harmony) { char sys[512]; harmony_system(sys, sizeof sys); PUT("%s", sys); }
    const char *pending_sys = NULL;   /* Mistral has no system role: it goes in front of the next user message */
    for (int i = 0; i < n; i++) {
        const char *r = m[i].role ? m[i].role : "user", *c = m[i].content ? m[i].content : "";
        const int sys = !strcmp(r, "system") || !strcmp(r, "developer"), as = !strcmp(r, "assistant");
        if (chatml) PUT("<|im_start|>%s\n%s<|im_end|>\n", sys ? "system" : as ? "assistant" : "user", c);
        else if (llama3) PUT("<|start_header_id|>%s<|end_header_id|>\n\n%s<|eot_id|>", sys ? "system" : as ? "assistant" : "user", c);
        else if (tulu) { if (as) PUT("<|assistant|>\n%s<|endoftext|>\n", c); else PUT("<|%s|>\n%s\n", sys ? "system" : "user", c); }
        else if (mistral) {
            if (sys) pending_sys = c;
            else if (as) PUT("%s%s</s>", sp, c);
            else { if (pending_sys) { PUT("%s[INST] %s\n\n%s [/INST]", sp, pending_sys, c); pending_sys = NULL; } else PUT("%s[INST] %s [/INST]", sp, c); }
        } else if (ds) { if (!sys) { if (as) PUT(DS_ASSISTANT "%s" DS_EOS, c); else PUT(DS_USER "%s", c); } }
        else if (kimi) {
            if (sys) PUT("<|im_system|>system<|im_middle|>%s<|im_end|>", c);
            else if (as) PUT("<|im_assistant|>assistant<|im_middle|>%s<|im_end|>", c);
            else PUT("<|im_user|>user<|im_middle|>%s<|im_end|>", c);
        } else if (harmony) {
            if (sys) PUT("<|start|>developer<|message|># Instructions\n\n%s\n\n<|end|>", c);
            else if (as) PUT("<|start|>assistant<|channel|>final<|message|>%s<|end|>", c);
            else PUT("<|start|>user<|message|>%s<|end|>", c);
        } else PUT("%s%s\n\n", sys ? "" : as ? "Assistant: " : "User: ", c);
    }
    if (chatml) PUT("<|im_start|>assistant\n");
    else if (llama3) PUT("<|start_header_id|>assistant<|end_header_id|>\n\n");
    else if (tulu) PUT("<|assistant|>\n");
    else if (ds) PUT(DS_ASSISTANT);
    else if (kimi) PUT("<|im_assistant|>assistant<|im_middle|>");
    else if (harmony) PUT("<|start|>assistant");
    else if (mistral) { if (pending_sys) PUT("%s[INST] %s [/INST]", sp, pending_sys); }
    else PUT("Assistant:");
#undef PUT
    return (int)o;
}

int es_engine_render_with(const char *chat_template, const es_msg *m, int n, char *out, size_t cap) {
    const char *keep = TEMPLATE_KIND;
    TEMPLATE_KIND = template_kind_of(chat_template);
    const int r = es_engine_render(m, n, out, cap);
    TEMPLATE_KIND = keep;
    return r;
}
const char *es_engine_template_kind(const char *chat_template) { return template_kind_of(chat_template); }

/* gpt-oss writes "<|channel|>analysis<|message|>thoughts<|end|><|start|>assistant<|channel|>final<|message|>answer".
 * The answer goes out as is, the thoughts inside <think>...</think> (as Qwen3 and DeepSeek-R1 write them), the markup not at all. */
static int H_START = -1, H_CHANNEL = -1, H_MESSAGE = -1, H_END = -1, H_STATE, H_THINK; static char H_CHAN[24]; static int H_CHANLEN;
enum { HS_ROLE, HS_CHAN, HS_MSG };
static void answer_start(void);
static int harmony_piece(int tok, const char *piece, int pl, char *buf, int cap) {
    if (tok == H_START) { H_STATE = HS_ROLE; return 0; }
    if (tok == H_CHANNEL) { H_STATE = HS_CHAN; H_CHANLEN = 0; return 0; }
    if (tok == H_MESSAGE) {
        H_STATE = HS_MSG; H_CHAN[H_CHANLEN] = 0;
        if (!strncmp(H_CHAN, "analysis", 8) && !H_THINK) { H_THINK = 1; return snprintf(buf, (size_t)cap, "<think>") < cap ? 7 : 0; }
        return 0;
    }
    if (tok == H_END) {
        if (H_THINK) { H_THINK = 0; H_STATE = HS_ROLE; return snprintf(buf, (size_t)cap, "</think>\n") < cap ? 9 : 0; }
        H_STATE = HS_ROLE; return 0;
    }
    if (H_STATE == HS_CHAN) { for (int i = 0; i < pl && H_CHANLEN < (int)sizeof H_CHAN - 1; i++) H_CHAN[H_CHANLEN++] = piece[i]; return 0; }
    if (H_STATE != HS_MSG) return 0;
    const int n = pl < cap ? pl : cap;
    memmove(buf, piece, (size_t)n);
    return n;
}

/* ---------------- public API ---------------- */
static int POS, FIRST = 1, GEN, MAX_NEW = 256, TOP_K = 40, DONE;
static int32_t *HIST;                /* every token of the conversation, by position */
static int SPEC_MAX = 7;             /* longest guess per step (0 = off) */
static int32_t ACCQ[BMAX + 1]; static int ACCP[BMAX + 1], ACCN, ACCI;
static unsigned long long STAT_SPEC_STEPS, STAT_SPEC_OK;
static float TEMP = 0.7f, TOP_P = 0.9f;
static es_engine_info INFO;
static es_turn_stats ST;
static uint64_t T_GEN0;
static unsigned long long H0, M0;
static double R0, W0;
static int32_t USER_TOK = -2;

void es_engine_get_info(es_engine_info *info) { *info = INFO; }
void es_engine_turn_stats(es_turn_stats *s) {
    *s = ST;
    s->gen_tokens = GEN;
    s->gen_s = T_GEN0 ? (es_now_ns() - T_GEN0) / 1e9 : 0;
    unsigned long long hits = STAT_HITS - H0, miss = STAT_MISS - M0;
    s->hit_pct = 100.0 * hits / (hits + miss ? hits + miss : 1);
    s->flash_mb = STAT_READ_MB - R0;
    s->wait_s = STAT_WAIT_S - W0;
    s->spec_steps = STAT_SPEC_STEPS; s->spec_accepted = STAT_SPEC_OK;
#ifndef __EMSCRIPTEN__
    s->pf_issued = STAT_PF_ISSUED; s->pf_used = STAT_PF_USED; s->skipped = STAT_SKIPPED;
#endif
}
void es_engine_set_sampling(float temp, int top_k, float top_p, int max_new, uint64_t seed) {
    TEMP = temp; TOP_K = top_k; TOP_P = top_p; MAX_NEW = max_new;
    if (seed) RNG = seed | 1;
}
void es_engine_reset(void) { POS = 0; FIRST = 1; DONE = 1; }

#define ERR(...) do { snprintf(err, errcap, __VA_ARGS__); return -1; } while (0)

/* expert weights are streamed, never loaded whole: merged "ffn_*_exps.weight" tensors, and the per-expert
 * "blk.N.ffn_gate.E.weight" tensors of older Mixtral files */
static int is_expert_tensor(const char *nm) {
    if (strstr(nm, "_exps.weight")) return 1;
    int l, e; char role[8], rest[16];
    return sscanf(nm, "blk.%d.ffn_%7[a-z].%d.%15s", &l, role, &e, rest) == 4 && !strcmp(rest, "weight");
}
static float gguf_float(const char *key, float def) {
    const es_gguf_kv *kv = es_gguf_find(&G, key);
    if (!kv) return def;
    if (kv->type == GGUF_T_F32 || kv->type == GGUF_T_F64) return (float)kv->v.f;
    return (float)es_gguf_int(&G, key, (int64_t)def);
}

int es_engine_init(const es_engine_opts *o, char *err, size_t errcap) {
    uint64_t t0 = es_now_ns();
    const char *e;
    char path[4096], key[256];
    PACK = o->pack;
    if (PACK) {
        snprintf(path, sizeof path, "%s/core.gguf", PACK);
        if (es_gguf_open(&G, path, &e)) ERR("%s: %s", path, e);
    } else if (o->gguf) {
        if (es_gguf_open(&G, o->gguf, &e)) ERR("%s: %s", o->gguf, e);
        /* split model: name-00001-of-00004.gguf -> open the other parts and merge their tensor tables */
        const char *dash = strstr(o->gguf, "-00001-of-");
        if (dash) {
            const int total = atoi(dash + 10);
            for (int part = 2; part <= total && part < 256; part++) {
                char sp[4096];
                snprintf(sp, sizeof sp, "%.*s-%05d-of-%05d%s", (int)(dash - o->gguf), o->gguf, part, total, dash + 15);
                es_gguf S;
                if (es_gguf_open(&S, sp, &e)) ERR("%s: %s", sp, e);
                SH_FD[part - 1] = S.fd; SH_DS[part - 1] = S.data_start;
                G.t = realloc(G.t, (G.n_tensors + S.n_tensors) * sizeof *G.t);
                for (uint64_t i = 0; i < S.n_tensors; i++) { G.t[G.n_tensors + i] = S.t[i]; G.t[G.n_tensors + i].shard = (uint32_t)(part - 1); }
                G.n_tensors += S.n_tensors;
            }
        }
    } else {
        if (es_gguf_open_fd(&G, o->gguf_fd, o->gguf_size, &e)) ERR("model file: %s", e);
    }
    (void)es_q2k_kernel_name();   /* pick the Q2_K kernel once, before worker threads exist */
    M.arch = es_gguf_str(&G, "general.architecture");
    if (!M.arch) ERR("not a model file (no general.architecture)");
#define HPI(nm, def) (snprintf(key, sizeof key, "%s." nm, M.arch), es_gguf_int(&G, key, def))
#define HPF(nm, def) (snprintf(key, sizeof key, "%s." nm, M.arch), gguf_float(key, def))
#define HPS(nm) (snprintf(key, sizeof key, "%s." nm, M.arch), es_gguf_str(&G, key))
    M.n_layer = (int)HPI("block_count", 0);
    M.n_embd = (int)HPI("embedding_length", 0);
    M.n_head = (int)HPI("attention.head_count", 0);
    M.n_head_kv = (int)HPI("attention.head_count_kv", M.n_head);
    M.n_exp = (int)HPI("expert_count", 0);
    M.n_used = (int)HPI("expert_used_count", 0);
    M.head_dim = (int)HPI("attention.key_length", M.n_head ? M.n_embd / M.n_head : 0);
    M.eps = HPF("attention.layer_norm_rms_epsilon", 1e-5f);
    M.rope_base = HPF("rope.freq_base", 10000.0f);
    const char *a = M.arch;
    M.kind = AR_STD; M.gating = GATE_SOFTMAX; M.norm_topk = 0;
    if (!strcmp(a, "olmoe")) { M.rope_neox = 1; M.norm_topk = 0; }
    else if (!strcmp(a, "qwen3moe")) { M.rope_neox = 1; M.norm_topk = 1; }
    else if (!strcmp(a, "qwen3") || !strcmp(a, "qwen2")) { M.rope_neox = 1; M.norm_topk = 0; }
    else if (!strcmp(a, "qwen2moe")) { M.rope_neox = 1; M.norm_topk = 0; }
    else if (!strcmp(a, "llama") || !strcmp(a, "mixtral")) { M.rope_neox = 0; M.norm_topk = 1; }
    else if (!strcmp(a, "gpt-oss")) { M.kind = AR_GPTOSS; M.rope_neox = 1; M.gating = GATE_SOFTMAX_CHOSEN; M.act_oai = 1; }
    else if (!strcmp(a, "deepseek2")) { M.kind = AR_DS2; M.rope_neox = 0; }
    else ERR("architecture '%s' is not supported yet (llama, mixtral, qwen2, qwen2moe, qwen3, qwen3moe, olmoe, gpt-oss and deepseek2 are)", a);
    snprintf(key, sizeof key, "%s.expert_weights_norm", a);
    if (es_gguf_find(&G, key)) M.norm_topk = (int)es_gguf_int(&G, key, M.norm_topk);
    { const int gf = (int)HPI("expert_gating_func", 0); if (gf == 1) M.gating = GATE_SOFTMAX; else if (gf == 2) M.gating = GATE_SIGMOID; else if (gf != 0) ERR("expert gating function %d is not supported yet", gf); }
    M.w_scale = HPF("expert_weights_scale", 0.0f);
    M.n_group = (int)HPI("expert_group_count", 0); M.n_group_used = (int)HPI("expert_group_used_count", 0);
    if (M.n_exp < 0 || M.n_exp > 1024 || M.n_used > MAXK || (M.n_exp > 0 && M.n_used <= 0) || M.n_group > 64) ERR("unsupported expert layout");

    /* attention geometry */
    if (M.kind == AR_DS2) {
        M.q_lora = (int)HPI("attention.q_lora_rank", 0);
        M.kv_lora = (int)HPI("attention.kv_lora_rank", 0);
        M.rope_d = (int)HPI("rope.dimension_count", 64);
        const int klen = (int)HPI("attention.key_length", 0), vlen = (int)HPI("attention.value_length", 0);   /* (HPI defaults must not nest) */
        const int kmla = (int)HPI("attention.key_length_mla", klen);
        M.v_head = (int)HPI("attention.value_length_mla", vlen);
        M.nope = kmla - M.rope_d;
        if (M.kv_lora <= 0 || M.nope <= 0 || M.v_head <= 0) ERR("incomplete DeepSeek attention shape in this file");
        M.n_rot = M.rope_d;
        M.dk = M.kv_lora + M.rope_d; M.dv = M.kv_lora; M.kvk = M.dk; M.kvv = 0;
        M.qd = M.n_head * (M.nope + M.rope_d); M.od = M.n_head * M.v_head; M.n_head_kv = 1;
    } else {
        M.n_rot = (int)HPI("rope.dimension_count", M.head_dim);
        if (M.n_rot > M.head_dim || M.n_rot % 2) ERR("unsupported rotary size %d for head size %d", M.n_rot, M.head_dim);
        M.dk = M.dv = M.head_dim; M.kvk = M.kvv = M.n_head_kv * M.head_dim;
        M.qd = M.od = M.n_head * M.head_dim;
    }
    if (M.kind == AR_GPTOSS) { M.n_swa = (int)HPI("attention.sliding_window", 0); }

    /* rotary scaling (llama.cpp's rules, so YaRN models behave the same) */
    {
        const char *st = HPS("rope.scaling.type");
        float factor = HPF("rope.scaling.factor", 0.0f);
        if (factor == 0.0f) factor = HPF("rope.scale_linear", 0.0f);
        const int none = st && !strcmp(st, "none"), yarn = st && !strcmp(st, "yarn");
        M.freq_scale = (factor == 0.0f || none) ? 1.0f : 1.0f / factor;
        const int ctx_train = (int)HPI("context_length", 4096);
        M.n_ctx_orig = (int)HPI("rope.scaling.original_context_length", ctx_train);
        M.beta_fast = HPF("rope.scaling.yarn_beta_fast", 32.0f);
        M.beta_slow = HPF("rope.scaling.yarn_beta_slow", 1.0f);
        M.ext_factor = yarn ? 1.0f : 0.0f;
        float log_mul = HPF("rope.scaling.yarn_log_multiplier", 0.0f);
        if (M.kind == AR_DS2) log_mul /= 0.1f;
        M.attn_factor = 1.0f;
        if (M.ext_factor != 0.0f) {
            const float f = 1.0f / M.freq_scale;
#define GETMS(s, m) ((s) <= 1.0f ? 1.0f : (0.1f * (m) * logf(s) + 1.0f))
            if (log_mul != 0.0f) {
                float ms = 1.0f;
                if (M.kind == AR_DS2 && log_mul != 1.0f) ms = log_mul;
                M.attn_factor = GETMS(f, ms) / GETMS(f, log_mul);
            } else M.attn_factor = GETMS(f, 1.0f);
#undef GETMS
            M.attn_factor *= 1.0f / (1.0f + 0.1f * logf(f));
        }
        M.attn_factor *= HPF("rope.scaling.attn_factor", 1.0f);
        const float nd = (float)M.n_rot;
#define CORR(beta) (nd * logf(M.n_ctx_orig / ((beta) * 2 * (float)M_PI)) / (2 * logf(M.rope_base)))
        M.corr[0] = fmaxf(0, floorf(CORR(M.beta_fast)));
        M.corr[1] = fminf(nd - 1, ceilf(CORR(M.beta_slow)));
        if (M.kind == AR_GPTOSS) {   /* gpt-oss' YaRN does not round the ramp ends ("truncate": false); verified against the reference model */
            M.corr[0] = fmaxf(0, CORR(M.beta_fast)); M.corr[1] = fminf(nd - 1, CORR(M.beta_slow));
        }
#undef CORR
        if (M.kind == AR_DS2) {
            const float org = M.attn_factor * (1.0f + 0.1f * logf(1.0f / M.freq_scale));
            const float ms = org * (1.0f + 0.1f * log_mul * logf(1.0f / M.freq_scale));
            M.kq_scale = 1.0f * ms * ms / sqrtf((float)(M.nope + M.rope_d));
        } else M.kq_scale = 1.0f / sqrtf((float)M.head_dim);
        if (getenv("ES_DEBUG")) fprintf(stderr, "rope: n_rot %d neox %d base %g freq_scale %g ext %g attn_factor %g corr %g..%g orig %d kq_scale %g\n",
                                        M.n_rot, M.rope_neox, M.rope_base, M.freq_scale, M.ext_factor, M.attn_factor, M.corr[0], M.corr[1], M.n_ctx_orig, M.kq_scale);
    }

    /* copy every non-expert tensor into RAM */
    M.tptr = calloc(G.n_tensors, sizeof(uint8_t *));
    for (uint64_t i = 0; i < G.n_tensors; i++) {
        const es_gguf_tensor *t = &G.t[i];
        if (is_expert_tensor(t->name)) continue;
        M.tptr[i] = es_alloc(t->nbytes ? t->nbytes : 1);
        if (!M.tptr[i]) ERR("out of memory loading %s", t->name);
        read_tensor(t, 0, M.tptr[i], t->nbytes);
        M.core_bytes += t->nbytes;
    }
    M.tok_embd = load_w("token_embd.weight", 1);
    M.output_norm = load_w("output_norm.weight", 1);
    M.output = load_w("output.weight", 0);
    if (!M.output.data) M.output = M.tok_embd;   /* tied embeddings */
    M.n_vocab = M.output.rows;
    { const W rf = load_w("rope_freqs.weight", 0); if (rf.data && rf.type == GGML_F32 && rf.cols >= M.n_rot / 2) M.rope_freqs = rf.data; }
    M.L = calloc((size_t)M.n_layer, sizeof(layer));
    const int swa_pat = (int)HPI("attention.sliding_window_pattern", M.kind == AR_GPTOSS ? 2 : 0);
    for (int l = 0; l < M.n_layer; l++) {
        layer *L = &M.L[l];
#define T_(field, name, req) do { snprintf(key, sizeof key, "blk.%d." name, l); L->field = load_w(key, req); } while (0)
        T_(attn_norm, "attn_norm.weight", 1);
        if (M.kind == AR_DS2) {
            T_(q_a, "attn_q_a.weight", 0); T_(q_a_norm, "attn_q_a_norm.weight", 0); T_(q_b, "attn_q_b.weight", 0);
            if (!L->q_a.data) T_(q, "attn_q.weight", 1);
            T_(kv_a, "attn_kv_a_mqa.weight", 1); T_(kv_a_norm, "attn_kv_a_norm.weight", 1);
            T_(k_b, "attn_k_b.weight", 0); T_(v_b, "attn_v_b.weight", 0);
            if (!L->k_b.data) {   /* older files: one kv_b matrix; split it (k part transposed per head, in float) */
                snprintf(key, sizeof key, "blk.%d.attn_kv_b.weight", l);
                const W kvb = load_w(key, 1);
                const int H = M.n_head, per = M.nope + M.v_head, kl = M.kv_lora;
                if (kvb.cols != kl || kvb.rows != H * per) ERR("unexpected attn_kv_b shape in layer %d", l);
                float *kb = malloc(sizeof(float) * (size_t)H * kl * M.nope), *rowf = malloc(sizeof(float) * (size_t)kl);
                const size_t rb = es_row_bytes(kvb.type, kl);
                uint8_t *vb = malloc(rb * (size_t)H * M.v_head);
                if (!kb || !rowf || !vb) ERR("out of memory splitting attn_kv_b");
                for (int h = 0; h < H; h++) {
                    for (int i = 0; i < M.nope; i++) {
                        es_dequant_row(kvb.type, (const uint8_t *)kvb.data + rb * (size_t)(h * per + i), rowf, kl);
                        for (int r = 0; r < kl; r++) kb[((size_t)h * kl + r) * M.nope + i] = rowf[r];
                    }
                    memcpy(vb + rb * (size_t)h * M.v_head, (const uint8_t *)kvb.data + rb * (size_t)(h * per + M.nope), rb * (size_t)M.v_head);
                }
                free(rowf);
                L->k_b = (W){GGML_F32, H * kl, M.nope, kb};
                L->v_b = (W){kvb.type, H * M.v_head, kl, vb};
            }
            if (L->q_a.data && (!L->q_a_norm.data || !L->q_b.data)) ERR("layer %d: incomplete q projection", l);
        } else {
            T_(q, "attn_q.weight", 1); T_(k, "attn_k.weight", 1); T_(v, "attn_v.weight", 1);
            T_(q_norm, "attn_q_norm.weight", 0); T_(k_norm, "attn_k_norm.weight", 0);
            T_(bq, "attn_q.bias", 0); T_(bk, "attn_k.bias", 0); T_(bv, "attn_v.bias", 0);
            if (!L->bq.data != !L->bk.data || !L->bq.data != !L->bv.data) ERR("layer %d has only some of the Q/K/V biases", l);
            T_(sinks, "attn_sinks.weight", 0);
        }
        T_(o, "attn_output.weight", 1); T_(bo, "attn_output.bias", 0);
        T_(ffn_norm, "ffn_norm.weight", 0);
        if (!L->ffn_norm.data) T_(ffn_norm, "post_attention_norm.weight", 1);
        T_(gate_inp, "ffn_gate_inp.weight", 0); T_(gate_inp_b, "ffn_gate_inp.bias", 0); T_(exp_probs_b, "exp_probs_b.bias", 0);
        T_(ffn_gate, "ffn_gate.weight", 0); T_(ffn_up, "ffn_up.weight", 0); T_(ffn_down, "ffn_down.weight", 0);
        T_(sh_gate, "ffn_gate_shexp.weight", 0); T_(sh_up, "ffn_up_shexp.weight", 0); T_(sh_down, "ffn_down_shexp.weight", 0);
        T_(sh_gate_inp, "ffn_gate_inp_shexp.weight", 0);
        T_(gate_exps_b, "ffn_gate_exps.bias", 0); T_(up_exps_b, "ffn_up_exps.bias", 0); T_(down_exps_b, "ffn_down_exps.bias", 0);
#undef T_
        L->moe = L->gate_inp.data != NULL;
        L->swa = swa_pat > 0 && (l % swa_pat) < swa_pat - 1;
        if (!L->moe && !L->ffn_gate.data) ERR("layer %d has neither experts nor a dense FFN", l);
        if (L->sinks.data && L->sinks.type != GGML_F32) ERR("attention sinks must be F32");
    }
    T = es_tok_load(&G, &e);
    if (!T) ERR("tokenizer: %s", e);
    detect_template();
    USER_TOK = es_tok_find(T, "<|user|>");
    if (TK("harmony")) {   /* gpt-oss ends a turn with <|return|> (or <|call|> for tools); <|end|> only ends its thinking */
        H_START = es_tok_find(T, "<|start|>"); H_CHANNEL = es_tok_find(T, "<|channel|>"); H_MESSAGE = es_tok_find(T, "<|message|>"); H_END = es_tok_find(T, "<|end|>");
        es_tok_set_eog(T, H_END, 0);
        es_tok_set_eog(T, es_tok_find(T, "<|return|>"), 1);
        es_tok_set_eog(T, es_tok_find(T, "<|call|>"), 1);
    }
    if (TK("kimi")) es_tok_set_eog(T, es_tok_find(T, "<|im_end|>"), 1);
    if (TK("deepseek")) es_tok_set_eog(T, es_tok_find(T, DS_EOS), 1);

    /* threads: compute on the fastest cores, readers on the rest */
#if defined(__EMSCRIPTEN__)
    NT = o->threads > 0 ? o->threads : 1;
#else
    NT = o->threads > 0 ? o->threads : es_auto_threads();   /* 0 = one thread per fast core */
#endif
    if (NT > ES_MAX_CPUS) NT = ES_MAX_CPUS;
#if !defined(__EMSCRIPTEN__)
    {   /* never run more compute threads than usable cores: the workers spin, so oversubscribing collapses speed */
        es_topo tp; int usable = 0;
        if (es_topo_read(&tp) == 0) for (int i = 0; i < tp.ncpu; i++) usable += tp.allowed[i] ? 1 : 0;
        if (usable > 0 && NT > usable) { fprintf(stderr, "note: %d threads requested but only %d cores usable; using %d\n", NT, usable, usable); NT = usable; }
    }
#endif
    if (NT > 1) {
        es_topo topo;
        es_topo_read(&topo);
        int ccpu[ES_MAX_CPUS], iocpu[ES_MAX_CPUS], nc = 0, nio = 0;
        for (int i = 0; i < topo.ncpu; i++) {
            int c = topo.order[i];
            if (!topo.allowed[c]) continue;
            if (nc < NT) ccpu[nc++] = c; else iocpu[nio++] = c;
        }
        CP = es_cpool_create(NT, nc == NT ? ccpu : NULL);
        if (PACK) IO = es_pool_create_on(4, 1 << 20, 1, iocpu, nio);
    } else if (PACK) {
        IO = es_pool_create(2, 1 << 20, 1);
    }
    M.ctx = o->ctx > 0 ? o->ctx : 2048;
    const int big = M.ctx > M.n_embd * 4 ? M.ctx : M.n_embd * 4;
    for (int i = 0; i < NT; i++) SCRATCH[i] = malloc(sizeof(float) * (size_t)(big + 4096));

    /* expert cache: slot size from the pack, or from the GGUF tensor shapes */
    unsigned long long slot_bytes = 0;
    int any_moe = 0, ffx = 0;
    for (int l = 0; l < M.n_layer; l++) if (M.L[l].moe) any_moe = 1;
    if (!any_moe) {
        slot_bytes = 4096;                      /* dense model: nothing to stream */
    } else if (PACK) {
        unsigned lay, ne;
        if (es_read_manifest(PACK, &lay, &ne, &slot_bytes)) ERR("no manifest.txt in %s", PACK);
    } else {
        static const char *kind[3] = {"gate", "up", "down"};
        snprintf(key, sizeof key, "blk.%d.ffn_gate.0.weight", 0);
        for (int l = 0; l < M.n_layer && !XTE; l++) if (M.L[l].moe) { snprintf(key, sizeof key, "blk.%d.ffn_gate.0.weight", l); if (es_gguf_tensor_find(&G, key)) XTE = calloc((size_t)M.n_layer * M.n_exp * 3, sizeof *XTE); break; }
        if (!XTE) XT = calloc((size_t)M.n_layer * 3, sizeof *XT);
        for (int l = 0; l < M.n_layer; l++) {
            if (!M.L[l].moe) continue;
            size_t off = 0;
            for (int r = 0; r < 3; r++) {
                const es_gguf_tensor *t;
                uint64_t per;
                if (XTE) {
                    for (int x = 0; x < M.n_exp; x++) {
                        snprintf(key, sizeof key, "blk.%d.ffn_%s.%d.weight", l, kind[r], x);
                        if (!(XTE[((size_t)l * M.n_exp + x) * 3 + r] = es_gguf_tensor_find(&G, key))) ERR("missing %s", key);
                    }
                    t = XTE[(size_t)l * M.n_exp * 3 + r]; per = t->nbytes;
                } else {
                    snprintf(key, sizeof key, "blk.%d.ffn_%s_exps.weight", l, kind[r]);
                    if (!(XT[l * 3 + r] = t = es_gguf_tensor_find(&G, key))) ERR("missing %s (fused gate+up expert layouts are not supported yet)", key);
                    per = t->nbytes / t->ne[2];
                }
                if (!es_quant_supported(t->type)) ERR("%s uses %s, not supported yet", key, es_ggml_type_name(t->type));
                if (r == 0 && (int)t->ne[1] > ffx) ffx = (int)t->ne[1];
                XOFF[r] = off;
                off += (per + 63) / 64 * 64;
            }
            if (off > slot_bytes) slot_bytes = off;
        }
    }
    int nslots = any_moe ? (int)(o->cache_mb * 1048576ull / slot_bytes) : 0;
    if (any_moe && nslots < 2 * M.n_used + 2) nslots = 2 * M.n_used + 2;
    if (any_moe) { if (es_cache_init(&C, nslots, slot_bytes, M.n_layer, M.n_exp, 1.0 * M.n_layer)) ERR("out of memory for the expert cache (%d slots of %.1f MB)", nslots, slot_bytes / 1048576.0); }
#ifndef __EMSCRIPTEN__
    if (any_moe) {
        const char *ev;
        if ((ev = getenv("ES_SIM_FLASH_MBPS")) && atof(ev) > 0) SIM_S_PER_BYTE = 1.0 / (atof(ev) * 1e6);
        SKIP_THR = (ev = getenv("ES_SKIP")) ? (float)atof(ev) : o->skip_thr;
        LOOKAHEAD = (ev = getenv("ES_LOOKAHEAD")) ? atoi(ev) : o->lookahead;
        int nio = (ev = getenv("ES_IOTHREADS")) ? atoi(ev) : (o->io_threads > 0 ? o->io_threads : 4);
        if (!PACK && nio > 0 && !getenv("ES_SYNC_IO")) {
            gio_start(nio, nslots);
            PFX = malloc(sizeof(float) * (size_t)M.n_embd); PFR = malloc(sizeof(float) * (size_t)(M.n_exp + 1));
            PROF = calloc((size_t)M.n_layer * (size_t)M.n_exp, sizeof(unsigned));
            if (o->gguf) snprintf(PROF_PATH, sizeof PROF_PATH, "%s.esprof", o->gguf);
            if (o->warm) warm_start(nslots * 7 / 10);
        }
    }
#endif

    /* activations and KV cache */
    const int n = M.n_embd;
    int ffmax = 0;
    for (int l = 0; l < M.n_layer; l++) {
        const layer *L = &M.L[l];
        if (L->ffn_gate.rows > ffmax) ffmax = L->ffn_gate.rows;
        if (L->sh_gate.rows > ffmax) ffmax = L->sh_gate.rows;
    }
    if (PACK && any_moe && ffx == 0) ffx = 8192;   /* packs: expert width is only known per file; this bounds it */
    const size_t K = (size_t)(any_moe ? M.n_used : 1);
    const size_t gu_rows = (size_t)BMAX * K;
    const int ffg = ffx > ffmax ? ffx : ffmax;
    const int H = M.n_head;
    const size_t qlen = M.kind == AR_DS2 ? (size_t)H * (M.nope + M.rope_d) : (size_t)M.qd;
    const size_t kvw = M.kind == AR_DS2 ? (size_t)(M.kv_lora + M.rope_d) : (size_t)M.kvk;
    BX = malloc(sizeof(float) * (size_t)n * BMAX); BXN = malloc(sizeof(float) * (size_t)n * BMAX); BTMP = malloc(sizeof(float) * (size_t)n * BMAX);
    BQ = malloc(sizeof(float) * qlen * BMAX); BKV = malloc(sizeof(float) * kvw * BMAX); BV2 = malloc(sizeof(float) * (size_t)M.kvk * BMAX);
    BATTN = malloc(sizeof(float) * (size_t)(M.od > M.qd ? M.od : M.qd) * BMAX);
    if (M.kind == AR_DS2) {
        BQA = malloc(sizeof(float) * (size_t)(M.q_lora + 1) * BMAX);
        BQM = malloc(sizeof(float) * (size_t)H * M.dk * BMAX);
        BATL = malloc(sizeof(float) * (size_t)H * M.kv_lora * BMAX);
        if (H > 256) ERR("too many attention heads (%d)", H);
    }
    BGU = malloc(sizeof(float) * 2 * (size_t)ffg * (gu_rows > BMAX ? gu_rows : BMAX));
    BH = malloc(sizeof(float) * (size_t)ffg * (gu_rows > BMAX ? gu_rows : BMAX));
    BYE = malloc(sizeof(float) * (size_t)n * gu_rows);
    BROUT = malloc(sizeof(float) * (size_t)(M.n_exp + 1) * BMAX);
    BLOG = malloc(sizeof(float) * (size_t)M.n_vocab * BMAX);
    LOGITS = malloc(sizeof(float) * (size_t)M.n_vocab);
    RC = malloc(sizeof(float) * (size_t)(M.n_rot / 2 + 1)); RS = malloc(sizeof(float) * (size_t)(M.n_rot / 2 + 1));
    {   /* everything prepared between two arena resets: one layer's activations of the whole batch */
        size_t fl = (size_t)BMAX * ((size_t)2 * n + qlen + (size_t)M.od + (size_t)M.q_lora + gu_rows / BMAX * ffg + ffg);
        if (M.kind == AR_DS2) fl += (size_t)BMAX * H * (size_t)(M.nope > M.kv_lora ? M.nope : M.kv_lora);
        es_act_arena_init(sizeof(es_blk_q80) * (fl / 32 + 4096));
        Q8K_CAP = fl / 256 + 1024;
        Q8K = malloc(sizeof(es_block_q8_K) * Q8K_CAP);
    }
    HIST = calloc((size_t)M.ctx + 64, sizeof(int32_t));
    { const char *ev = getenv("ES_SPEC"); SPEC_MAX = ev ? atoi(ev) : (o->spec == -2 ? 0 : o->spec > 0 ? o->spec : any_moe ? 0 : 7); }
    BATCH_OK = !getenv("ES_NO_BATCH");
    size_t kv_bytes = sizeof(float) * (size_t)M.n_layer * M.ctx * M.kvk;
    KC = malloc(kv_bytes);
    VC = M.kvv ? malloc(sizeof(float) * (size_t)M.n_layer * M.ctx * M.kvv) : NULL;
    if (!KC || (M.kvv && !VC) || !BGU || !BH || !BYE || !BLOG || !Q8K) ERR("out of memory for a %d-token context; try a smaller context", M.ctx);
    if (o->trace_path) TRACE = fopen(o->trace_path, "w");

    const char *nm = es_gguf_str(&G, "general.name");
    INFO = (es_engine_info){nm ? nm : M.arch, M.arch, M.n_layer, M.n_exp, M.n_used, M.ctx, nslots,
                            M.core_bytes / 1e6, any_moe ? nslots * (double)slot_bytes / 1e6 : 0.0, (kv_bytes + (M.kvv ? sizeof(float) * (size_t)M.n_layer * M.ctx * M.kvv : 0)) / 1e6,
                            (es_now_ns() - t0) / 1e9};
    DONE = 1;
    return 0;
}

int es_engine_begin(const char *msg, int raw, int32_t *ids, int ids_cap,
                    void (*progress)(int, int, void *), void *ud) {
    static char text[1 << 16];
    static int32_t toks[1 << 15];
    memset(&ST, 0, sizeof ST);
    if (raw) snprintf(text, sizeof text, "%s", msg);
    else format_turn(text, sizeof text, msg, FIRST);
    int nt = es_tok_encode(T, text, 1, toks, (int)(sizeof toks / sizeof toks[0]));
    if (raw && FIRST && es_tok_add_bos(T) && es_tok_bos(T) >= 0) {
        memmove(toks + 1, toks, sizeof(int32_t) * (size_t)nt);
        toks[0] = es_tok_bos(T);
        nt++;
    }
    if (POS + nt + 1 >= M.ctx) {           /* conversation full: start over */
        POS = 0; FIRST = 1; ST.ctx_reset = 1;
        if (!raw) {
            format_turn(text, sizeof text, msg, 1);
            nt = es_tok_encode(T, text, 1, toks, (int)(sizeof toks / sizeof toks[0]));
        }
        if (nt + 1 >= M.ctx) return -1;
    }
    for (int i = 0; i < nt && i < ids_cap; i++) ids[i] = toks[i];
    H0 = STAT_HITS; M0 = STAT_MISS; R0 = STAT_READ_MB; W0 = STAT_WAIT_S;
    if (progress) progress(0, nt, ud);   /* ids are ready: report them before the slow part */
    uint64_t tp = es_now_ns();
    for (int i = 0; i < nt; i++) HIST[POS + i] = toks[i];
    ACCN = ACCI = 0;
    if (BATCH_OK) {
        for (int i = 0; i < nt; i += BMAX) {
            const int cnt = nt - i < BMAX ? nt - i : BMAX, last = i + cnt == nt;
            forward_n(toks + i, cnt, POS, last ? cnt - 1 : cnt);
            POS += cnt;
            if (last) memcpy(LOGITS, BLOG, sizeof(float) * (size_t)M.n_vocab);
            if (progress) progress(i + cnt, nt, ud);
        }
    } else
    for (int i = 0; i < nt; i++) {
        forward(toks[i], POS++, i == nt - 1);
        if (progress) progress(i + 1, nt, ud);
    }
    ST.prompt_tokens = nt;
    ST.prompt_s = (es_now_ns() - tp) / 1e9;
    GEN = 0;
    DONE = 0;
    FIRST = 0;
    answer_start();
    T_GEN0 = es_now_ns();
    return nt;
}

/* Turbo: guess the next tokens by finding the latest earlier place where the last few tokens occurred and
 * copying what followed ("prompt lookup"). All guesses are checked in ONE batched pass, so the weights are read
 * once for several tokens. A wrong guess costs nothing but a little compute; the output is the same as without it
 * (exactly, when temperature is 0). */
static double SPEC_T1; static int SPEC_N1;      /* measured seconds per token of a plain step */
static int SPEC_COOL, SPEC_BACKOFF = 8;   /* after a useless guess, stop guessing for a while (grows while guesses keep failing) */
static int draft_tokens(int32_t *out, int max) {
    const int len = POS + 1;   /* HIST[POS] is the token about to be emitted */
    if (SPEC_COOL > 0) { SPEC_COOL--; return 0; }
    if (SPEC_N1 < 4) return 0;   /* learn the plain speed first */
    for (int g = 5; g >= 3; g--) {
        if (len < g + 1) continue;
        const int32_t *suf = HIST + len - g;
        for (int j = len - g - 1; j >= 0; j--) {
            int ok = 1;
            for (int t = 0; t < g; t++) if (HIST[j + t] != suf[t]) { ok = 0; break; }
            if (!ok) continue;
            int n = 0;
            const int lim = g >= 5 ? max : g == 4 ? (max < 5 ? max : 5) : (max < 3 ? max : 3);   /* longer match, bolder guess */
            while (n < lim && j + g + n < len) { out[n] = HIST[j + g + n]; n++; }
            if (n > 0) return n;
        }
    }
    return 0;
}

static int ANS_FRESH;   /* nothing of the answer shown yet (SentencePiece answers start with a space to drop) */
static void answer_start(void) { ANS_FRESH = 1; H_STATE = HS_ROLE; H_THINK = 0; }
/* the text of one answer token */
static int piece_out(int tok, char *buf, int cap) {
    int n = es_tok_piece(T, tok, 0, buf, cap);
    if (TK("harmony")) n = harmony_piece(tok, buf, n, buf, cap);
    if (ANS_FRESH && n > 0) {
        ANS_FRESH = 0;
        if (es_tok_is_spm(T) && buf[0] == ' ') { memmove(buf, buf + 1, (size_t)n - 1); n--; }
    }
    return n;
}
static int emit_piece(int tok, int tokpos, char *buf, int cap) {
    if (es_tok_is_eog(T, tok) || (tok == USER_TOK && USER_TOK >= 0)) { POS = tokpos; DONE = 1; return -1; }
    GEN++;
    return piece_out(tok, buf, cap);
}

int es_engine_next(char *buf, int cap) {
    if (ACCI < ACCN) {                              /* tokens accepted by an earlier check */
        if (GEN >= MAX_NEW) { DONE = 1; return -1; }
        const int k = ACCI++;
        return emit_piece(ACCQ[k], ACCP[k], buf, cap);
    }
    if (DONE || GEN >= MAX_NEW || POS >= M.ctx) { DONE = 1; return -1; }
    int next = sample(TEMP, TOP_K, TOP_P);
    if (es_tok_is_eog(T, next) || (next == USER_TOK && USER_TOK >= 0)) { DONE = 1; return -1; }
    HIST[POS] = next;
    int32_t draft[BMAX];
    int nd = 0;
    if (BATCH_OK && SPEC_MAX > 0 && POS + SPEC_MAX + 2 < M.ctx) nd = draft_tokens(draft, SPEC_MAX < BMAX - 1 ? SPEC_MAX : BMAX - 1);
    if (nd == 0) {
        int pl = piece_out(next, buf, cap);
        const uint64_t t0 = es_now_ns();
        forward(next, POS++, 1);
        const double dt = (es_now_ns() - t0) / 1e9;
        SPEC_T1 = SPEC_N1 == 0 ? dt : 0.8 * SPEC_T1 + 0.2 * dt; if (SPEC_N1 < 1000) SPEC_N1++;
        GEN++;
        return pl;
    }
    int32_t batch[BMAX];
    batch[0] = next;
    for (int i = 0; i < nd; i++) batch[i + 1] = draft[i];
    const uint64_t ts0 = es_now_ns();
    forward_n(batch, nd + 1, POS, 0);
    const double tstep = (es_now_ns() - ts0) / 1e9;
    int m = 0;
    for (; m < nd; m++) {                           /* does the model agree with guess m+1? */
        float *row = BLOG + (size_t)m * M.n_vocab;
        if (TEMP <= 0) { if (argmax_of(row) != draft[m]) break; }
        else {
            build_dist(row, TEMP, TOP_K, TOP_P);
            if (rng_unit() >= dist_prob(draft[m])) { row[draft[m]] = -INFINITY; break; }   /* next draw excludes the rejected guess */
        }
    }
    memcpy(LOGITS, BLOG + (size_t)m * M.n_vocab, sizeof(float) * (size_t)M.n_vocab);
    STAT_SPEC_STEPS++; STAT_SPEC_OK += (unsigned long long)m;
    /* keep guessing only while it is really faster per token than plain steps */
    if (tstep / (m + 1) > 0.9 * SPEC_T1) { SPEC_COOL = SPEC_BACKOFF; if (SPEC_BACKOFF < 128) SPEC_BACKOFF *= 2; } else SPEC_BACKOFF = 8;
    ACCN = ACCI = 0;
    for (int i = 1; i <= m; i++) { ACCQ[ACCN] = batch[i]; ACCP[ACCN] = POS + i; HIST[POS + i] = batch[i]; ACCN++; }
    POS += m + 1;
    GEN++;
    return piece_out(next, buf, cap);
}

/* ---------------- Instant: prefix reuse ----------------
 * Apps resend the whole conversation each turn. Reading it again costs time proportional to its length before the first word appears;
 * here only the part after the longest already-computed prefix is read. The KV rows of up to ES_KV_SLOTS other conversations are parked in
 * RAM (apps send side requests such as "write a title for this chat" between turns, which would otherwise evict the main conversation). */
typedef struct { int n; int32_t *ids; float *k, *v; uint64_t used; } kv_slot;
static kv_slot *SLOTS; static int NSLOTS = -1, PREFIX_ON = 1; static size_t SLOT_BYTES_MAX; static uint64_t SLOT_CLOCK;

static int lcp(const int32_t *a, int na, const int32_t *b, int nb) {
    int i = 0, m = na < nb ? na : nb;
    while (i < m && a[i] == b[i]) i++;
    return i;
}
static void slots_init(void) {
    if (NSLOTS >= 0) return;
    const char *e = getenv("ES_PREFIX_CACHE"); PREFIX_ON = !(e && !strcmp(e, "0"));
    e = getenv("ES_KV_SLOTS"); NSLOTS = e ? atoi(e) : 4; if (NSLOTS < 0) NSLOTS = 0; if (NSLOTS > 64) NSLOTS = 64;
    e = getenv("ES_KV_SLOT_MB"); SLOT_BYTES_MAX = (size_t)(e ? atol(e) : 1024) * 1048576u;
    SLOTS = calloc((size_t)(NSLOTS ? NSLOTS : 1), sizeof *SLOTS);
}
static size_t slots_bytes(void) { size_t b = 0; const size_t row = sizeof(float) * (size_t)M.n_layer * (M.kvk + M.kvv); for (int i = 0; i < NSLOTS; i++) b += (size_t)SLOTS[i].n * row; return b; }
/* copy the first n cached rows of every layer between the live context and a packed buffer (keys, then values) */
static void kv_copy(float *k, float *v, int n, int stride_n, int to_live) {
    for (int l = 0; l < M.n_layer; l++) {
        float *lk = KC + (size_t)l * M.ctx * M.kvk, *pk = k + (size_t)l * stride_n * M.kvk;
        if (to_live) memcpy(lk, pk, sizeof(float) * (size_t)n * M.kvk); else memcpy(pk, lk, sizeof(float) * (size_t)n * M.kvk);
        if (!M.kvv) continue;
        float *lv = VC + (size_t)l * M.ctx * M.kvv, *pv = v + (size_t)l * stride_n * M.kvv;
        if (to_live) memcpy(lv, pv, sizeof(float) * (size_t)n * M.kvv); else memcpy(pv, lv, sizeof(float) * (size_t)n * M.kvv);
    }
}
static void slot_free(kv_slot *s) { free(s->ids); free(s->k); free(s->v); memset(s, 0, sizeof *s); }
/* park the live conversation's first n positions (never into slot `keep`) */
static void slot_save(int n, int keep) {
    if (NSLOTS <= 0 || n < 32) return;
    const size_t perk = sizeof(float) * (size_t)M.n_layer * M.kvk, perv = sizeof(float) * (size_t)M.n_layer * M.kvv;
    for (int i = 0; i < NSLOTS; i++)                     /* already parked (or a longer version of it): just refresh */
        if (SLOTS[i].n >= n && lcp(SLOTS[i].ids, SLOTS[i].n, HIST, n) == n) { SLOTS[i].used = ++SLOT_CLOCK; return; }
    int w = -1;
    for (int i = 0; i < NSLOTS; i++) if (i != keep && (w < 0 || SLOTS[i].used < SLOTS[w].used)) w = i;
    if (w < 0) return;
    slot_free(&SLOTS[w]);
    while (slots_bytes() + (size_t)n * (perk + perv) > SLOT_BYTES_MAX) {   /* stay inside the RAM budget: drop the oldest */
        int o = -1;
        for (int i = 0; i < NSLOTS; i++) if (i != keep && SLOTS[i].n && (o < 0 || SLOTS[i].used < SLOTS[o].used)) o = i;
        if (o < 0) return;
        slot_free(&SLOTS[o]);
    }
    kv_slot *s = &SLOTS[w];
    s->ids = malloc(sizeof(int32_t) * (size_t)n); s->k = malloc(perk * (size_t)n); s->v = malloc(perv * (size_t)n + 1);
    if (!s->ids || !s->k || !s->v) { slot_free(s); return; }
    memcpy(s->ids, HIST, sizeof(int32_t) * (size_t)n);
    kv_copy(s->k, s->v, n, n, 0);
    s->n = n; s->used = ++SLOT_CLOCK;
}
static void slot_restore(int i, int upto) {
    kv_slot *s = &SLOTS[i]; const int n = upto < s->n ? upto : s->n;
    memcpy(HIST, s->ids, sizeof(int32_t) * (size_t)n);
    kv_copy(s->k, s->v, n, s->n, 1);
    s->used = ++SLOT_CLOCK; POS = n;
}

/* ---------------- Instant: prompt prefixes kept on disk ----------------
 * The KV rows of an expensive prompt (by default: at least 256 new tokens that took 0.5 s or more to read) are written in the
 * background to $ES_KV_DIR (default ~/.expertstream/kv). Any later request that starts the same way (same app system prompt, same long
 * document, the same chat after a restart) loads those rows instead of reading the text again. Lossless: the rows are stored as they are.
 * ES_KV_DISK=0 turns it off, ES_KV_DISK_MB is the space budget (2048, oldest files go first). */
#ifndef __EMSCRIPTEN__
#include <dirent.h>
#include <sys/stat.h>
#include <utime.h>
typedef struct { char path[1280]; int n; int32_t *ids; time_t mt; } kv_file;
static kv_file *DF; static int NDF, DF_ON = -1, DF_MIN = 256; static double DF_MIN_S = 0.5; static size_t DF_BUDGET;
static char DF_DIR[900]; static uint64_t FINGER; static pthread_mutex_t DF_MU = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long STAT_DISK_LOADS, STAT_DISK_SAVES;

static uint64_t fnv(uint64_t h, const void *p, size_t n) { const unsigned char *c = p; for (size_t i = 0; i < n; i++) { h ^= c[i]; h *= 0x100000001b3ull; } return h; }
static void df_init(void) {
    if (DF_ON >= 0) return;
    const char *e = getenv("ES_KV_DISK"); DF_ON = !(e && !strcmp(e, "0"));
    e = getenv("ES_KV_DISK_MB"); DF_BUDGET = (size_t)(e ? atol(e) : 2048) * 1048576u;
    e = getenv("ES_KV_DISK_MIN"); if (e) DF_MIN = atoi(e);
    e = getenv("ES_KV_DISK_MIN_S"); if (e) DF_MIN_S = atof(e);
    if (!DF_ON) return;
    /* the rows only fit the exact same weights: fingerprint the model by its shape, name and tensor table */
    uint64_t h = 0xcbf29ce484222325ull; const int shape[6] = {M.n_layer, M.n_embd, M.n_head, M.n_head_kv, M.head_dim, M.n_vocab};
    h = fnv(h, shape, sizeof shape); h = fnv(h, INFO.name, strlen(INFO.name)); h = fnv(h, &G.file_size, sizeof G.file_size);
    for (uint64_t i = 0; i < G.n_tensors; i++) { h = fnv(h, G.t[i].name, strlen(G.t[i].name)); h = fnv(h, &G.t[i].offset, sizeof G.t[i].offset); }
    FINGER = h;
    e = getenv("ES_KV_DIR");
    if (e) snprintf(DF_DIR, sizeof DF_DIR, "%s", e);
    else { const char *home = getenv("HOME"); char d[800]; snprintf(d, sizeof d, "%s/.expertstream", home ? home : "."); mkdir(d, 0700); snprintf(DF_DIR, sizeof DF_DIR, "%s/kv", d); }
    mkdir(DF_DIR, 0700);
    DIR *dir = opendir(DF_DIR); if (!dir) { DF_ON = 0; return; }
    struct dirent *de; char pre[32]; snprintf(pre, sizeof pre, "%016llx-", (unsigned long long)FINGER);
    while ((de = readdir(dir))) {
        if (strncmp(de->d_name, pre, 17) || !strstr(de->d_name, ".eskv")) continue;
        kv_file f = {0}; snprintf(f.path, sizeof f.path, "%s/%s", DF_DIR, de->d_name);
        FILE *fp = fopen(f.path, "rb"); if (!fp) continue;
        char mg[8]; uint64_t fg; int hdr[3];
        if (fread(mg, 1, 8, fp) == 8 && !memcmp(mg, "ESKV1\0\0\0", 8) && fread(&fg, 8, 1, fp) == 1 && fg == FINGER && fread(hdr, sizeof(int), 3, fp) == 3 &&
            hdr[1] == M.n_layer && hdr[2] == M.kvk && hdr[0] > 0 && hdr[0] <= 1 << 20) {
            f.n = hdr[0]; f.ids = malloc(sizeof(int32_t) * (size_t)f.n);
            if (fread(f.ids, sizeof(int32_t), (size_t)f.n, fp) == (size_t)f.n) {
                struct stat st; f.mt = stat(f.path, &st) ? 0 : st.st_mtime;
                DF = realloc(DF, sizeof *DF * (size_t)(NDF + 1)); DF[NDF++] = f;
            } else free(f.ids);
        }
        fclose(fp);
    }
    closedir(dir);
}
static size_t df_file_bytes(int n) { return 28 + 4 * (size_t)n + sizeof(float) * (size_t)n * M.n_layer * (M.kvk + M.kvv); }
typedef struct { int n; int32_t *ids; float *k, *v; } df_job;
static void *df_writer(void *arg) {               /* background: write, then add to the index and keep inside the budget */
    df_job *j = arg; const int kvd = M.kvk;
    uint64_t h = fnv(0xcbf29ce484222325ull, j->ids, sizeof(int32_t) * (size_t)j->n);
    char path[1100], tmp[1120]; snprintf(path, sizeof path, "%s/%016llx-%016llx.eskv", DF_DIR, (unsigned long long)FINGER, (unsigned long long)h);
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *fp = fopen(tmp, "wb"); int ok = fp != NULL;
    if (ok) {
        const int hdr[3] = {j->n, M.n_layer, kvd};
        ok = fwrite("ESKV1\0\0\0", 1, 8, fp) == 8 && fwrite(&FINGER, 8, 1, fp) == 1 && fwrite(hdr, sizeof(int), 3, fp) == 3 &&
             fwrite(j->ids, sizeof(int32_t), (size_t)j->n, fp) == (size_t)j->n &&
             fwrite(j->k, sizeof(float), (size_t)j->n * M.n_layer * kvd, fp) == (size_t)j->n * M.n_layer * kvd &&
             fwrite(j->v, sizeof(float), (size_t)j->n * M.n_layer * M.kvv, fp) == (size_t)j->n * M.n_layer * M.kvv;
        ok = (fclose(fp) == 0) && ok;
    }
    if (ok && rename(tmp, path) == 0) {
        pthread_mutex_lock(&DF_MU);
        for (int i = 0; i < NDF; i++)                                       /* an older, shorter version of the same prompt is now redundant */
            if (DF[i].n <= j->n && lcp(DF[i].ids, DF[i].n, j->ids, j->n) == DF[i].n && strcmp(DF[i].path, path)) { unlink(DF[i].path); free(DF[i].ids); DF[i--] = DF[--NDF]; }
        int dup = 0; for (int i = 0; i < NDF; i++) if (!strcmp(DF[i].path, path)) dup = 1;
        if (!dup) { DF = realloc(DF, sizeof *DF * (size_t)(NDF + 1)); kv_file f = {0}; snprintf(f.path, sizeof f.path, "%s", path); f.n = j->n; f.ids = j->ids; j->ids = NULL; f.mt = time(NULL); DF[NDF++] = f; }
        for (;;) {                                                          /* stay inside the disk budget: remove the least recently used */
            size_t tot = 0; int o = -1;
            for (int i = 0; i < NDF; i++) { tot += df_file_bytes(DF[i].n); if (o < 0 || DF[i].mt < DF[o].mt) o = i; }
            if (tot <= DF_BUDGET || NDF <= 1) break;
            unlink(DF[o].path); free(DF[o].ids); DF[o] = DF[--NDF];
        }
        STAT_DISK_SAVES++;
        pthread_mutex_unlock(&DF_MU);
    } else unlink(tmp);
    free(j->ids); free(j->k); free(j->v); free(j);
    return NULL;
}
static void df_save(int n) {                      /* copy the first n positions now (fast), write them later (slow) */
    if (!DF_ON || n < 32 || df_file_bytes(n) > DF_BUDGET) return;
    pthread_mutex_lock(&DF_MU);
    for (int i = 0; i < NDF; i++) if (DF[i].n >= n - 16 && lcp(DF[i].ids, DF[i].n, HIST, n) >= n - 16) { pthread_mutex_unlock(&DF_MU); return; }   /* already there */
    pthread_mutex_unlock(&DF_MU);
    df_job *j = calloc(1, sizeof *j); j->n = n;
    j->ids = malloc(sizeof(int32_t) * (size_t)n); j->k = malloc(sizeof(float) * (size_t)n * M.kvk * M.n_layer); j->v = malloc(sizeof(float) * (size_t)n * M.kvv * M.n_layer + 1);
    if (!j->ids || !j->k || !j->v) { free(j->ids); free(j->k); free(j->v); free(j); return; }
    memcpy(j->ids, HIST, sizeof(int32_t) * (size_t)n);
    kv_copy(j->k, j->v, n, n, 0);
    pthread_t th; if (pthread_create(&th, NULL, df_writer, j) == 0) pthread_detach(th); else df_writer(j);
}
/* longest stored prefix of toks: returns its length and the file index in *which */
static int df_best(const int32_t *toks, int nt, int *which) {
    int bl = 0; *which = -1;
    pthread_mutex_lock(&DF_MU);
    for (int i = 0; i < NDF; i++) { const int l = lcp(DF[i].ids, DF[i].n, toks, nt); if (l > bl) { bl = l; *which = i; } }
    pthread_mutex_unlock(&DF_MU);
    return bl;
}
static int df_load(int i, int upto) {             /* read the first upto rows of file i into the live context; 0 on success */
    pthread_mutex_lock(&DF_MU);
    if (i >= NDF) { pthread_mutex_unlock(&DF_MU); return -1; }
    kv_file f = DF[i]; int32_t *ids = malloc(sizeof(int32_t) * (size_t)f.n); memcpy(ids, DF[i].ids, sizeof(int32_t) * (size_t)f.n); DF[i].mt = time(NULL);
    pthread_mutex_unlock(&DF_MU);
    const int n = upto < f.n ? upto : f.n; int ok = 1;
    FILE *fp = fopen(f.path, "rb"); if (!fp) { free(ids); return -1; }
    const long base = 28 + 4L * f.n; const size_t layk = (size_t)f.n * M.kvk * sizeof(float), layv = (size_t)f.n * M.kvv * sizeof(float);
    for (int l = 0; l < M.n_layer && ok; l++) {
        ok = fseek(fp, base + (long)(l * layk), SEEK_SET) == 0 && fread(KC + (size_t)l * M.ctx * M.kvk, sizeof(float), (size_t)n * M.kvk, fp) == (size_t)n * M.kvk;
        if (M.kvv) ok = ok && fseek(fp, base + (long)(M.n_layer * layk + l * layv), SEEK_SET) == 0 && fread(VC + (size_t)l * M.ctx * M.kvv, sizeof(float), (size_t)n * M.kvv, fp) == (size_t)n * M.kvv;
    }
    fclose(fp);
    if (ok) { memcpy(HIST, ids, sizeof(int32_t) * (size_t)n); POS = n; utime(f.path, NULL); STAT_DISK_LOADS++; }
    free(ids);
    return ok ? 0 : -1;
}
#else
static void df_init(void) {}
static void df_save(int n) { (void)n; }
static int df_best(const int32_t *t, int n, int *w) { (void)t; (void)n; *w = -1; return 0; }
static int df_load(int i, int u) { (void)i; (void)u; return -1; }
static int DF_MIN = 256; static double DF_MIN_S = 0.5;
#endif

int es_engine_count(const char *text) {
    static int32_t tmp[1 << 16];
    return es_tok_encode(T, text, 1, tmp, (int)(sizeof tmp / sizeof tmp[0]));
}

int es_engine_prompt(const char *text, int *cached, int32_t *ids, int ids_cap, void (*progress)(int, int, void *), void *ud) {
    static int32_t *toks; static int cap;
    slots_init(); df_init();
    if (!toks) { cap = M.ctx + 64; toks = malloc(sizeof(int32_t) * (size_t)cap); }
    memset(&ST, 0, sizeof ST);
    int nt = 0;
    if (!strcmp(TEMPLATE_KIND, "raw") && es_tok_add_bos(T) && es_tok_bos(T) >= 0) toks[nt++] = es_tok_bos(T);   /* chat templates carry their own */
    const int m = es_tok_encode(T, text, 1, toks + nt, cap - nt);
    if (m < 0) return -1;
    nt += m;
    if (nt < 1 || nt + 1 >= M.ctx) return -1;
    int keep = 0;
    if (PREFIX_ON) {
        keep = lcp(HIST, POS, toks, nt);
        int best = -1, bl = keep;
        for (int i = 0; i < NSLOTS; i++) { const int l = SLOTS[i].n ? lcp(SLOTS[i].ids, SLOTS[i].n, toks, nt) : 0; if (l > bl + 16) { bl = l; best = i; } }
        int dfi; const int dl = df_best(toks, nt, &dfi);  /* a longer start saved on disk (same system prompt, or before a restart)? */
        const int use_disk = dfi >= 0 && dl > bl + 64;
        if (POS - keep >= 64 && (best >= 0 || use_disk || keep < POS)) slot_save(POS, best);   /* a different conversation is about to be overwritten: park it */
        if (use_disk && df_load(dfi, dl) == 0) keep = dl;
        else if (best >= 0) { slot_restore(best, bl); keep = bl; }
        if (keep >= nt) keep = nt - 1;                    /* the last prompt token is read again to get its next-word scores */
    }
    POS = keep;
    for (int i = keep; i < nt && i - keep < ids_cap; i++) ids[i - keep] = toks[i];   /* the ids that will be read now */
    if (progress) progress(0, nt - keep, ud);
    H0 = STAT_HITS; M0 = STAT_MISS; R0 = STAT_READ_MB; W0 = STAT_WAIT_S;
    const uint64_t tp = es_now_ns();
    for (int i = keep; i < nt; i++) HIST[i] = toks[i];
    ACCN = ACCI = 0;
    const int32_t *rest = toks + keep; const int nr = nt - keep;
    if (BATCH_OK) {
        for (int i = 0; i < nr; i += BMAX) {
            const int cnt = nr - i < BMAX ? nr - i : BMAX, last = i + cnt == nr;
            forward_n(rest + i, cnt, POS, last ? cnt - 1 : cnt);
            POS += cnt;
            if (last) memcpy(LOGITS, BLOG, sizeof(float) * (size_t)M.n_vocab);
            if (progress) progress(i + cnt, nr, ud);
        }
    } else
        for (int i = 0; i < nr; i++) { forward(rest[i], POS++, i == nr - 1); if (progress) progress(i + 1, nr, ud); }
    ST.prompt_tokens = nt;
    ST.prompt_s = (es_now_ns() - tp) / 1e9;
    if (PREFIX_ON && nr >= DF_MIN && ST.prompt_s >= DF_MIN_S) df_save(nt);   /* expensive to read: keep it on disk for next time */
    GEN = 0; DONE = 0; FIRST = 0;
    answer_start();
    T_GEN0 = es_now_ns();
    if (cached) *cached = keep;
    return nt;
}

int es_engine_dump_logits(const int32_t *ids, int n, FILE *f) {
    for (int i = 0; i < n; i++) {
        forward(ids[i], i, 1);
        fwrite(LOGITS, sizeof(float), (size_t)M.n_vocab, f);
    }
    return 0;
}

