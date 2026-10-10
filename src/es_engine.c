/*
 * ExpertStream engine (see es_engine.h). Forward pass: RMSNorm, attention
 * with optional Q/K norm and RoPE, softmax router, top-k experts (SwiGLU),
 * optional dense and shared-expert FFNs. Verified against an independent
 * float64 NumPy implementation of OLMoE to ~1e-6.
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

#define MAXK 16
#define MAXJOBS 40

/* ---------------- weights ---------------- */
typedef struct { uint32_t type; int rows, cols; const void *data; } W;

typedef struct {
    W attn_norm, q, k, v, o, q_norm, k_norm, ffn_norm, gate_inp;
    W bq, bk, bv;                            /* optional QKV biases (Qwen2) */
    W ffn_gate, ffn_up, ffn_down;            /* dense FFN (layers without experts) */
    W sh_gate, sh_up, sh_down;               /* shared expert, if any */
    int moe;
} layer;

typedef struct {
    const char *arch;
    int n_layer, n_embd, n_head, n_head_kv, head_dim, n_exp, n_used, n_vocab, ctx;
    float eps, rope_base;
    int rope_neox;      /* 1: rotate (i, i+d/2) pairs; 0: adjacent pairs */
    int norm_topk;      /* renormalize the top-k router weights */
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
            for (int q = 0; q < JOBS[j].nb; q++)
                es_matvec(JOBS[j].type, JOBS[j].w, JOBS[j].cols, (int)(c - base), (int)(b - base),
                          JOBS[j].a + q, JOBS[j].y + (size_t)q * JOBS[j].ys, SCRATCH[tid]);
            c = b;
        }
    }
}
static void job(const W *w, const es_act *a, float *y) {
    JOBS[NJOBS++] = (mv_job){w->type, w->data, w->rows, w->cols, a, y, 1, 0};
}
static void jobn(const W *w, const es_act *acts, float *y, size_t ystride, int nb) {
    JOBS[NJOBS++] = (mv_job){w->type, w->data, w->rows, w->cols, acts, y, nb, ystride};
}
static void run_jobs(void) {
    TOTAL_ROWS = 0;
    for (int j = 0; j < NJOBS; j++) TOTAL_ROWS += JOBS[j].rows;
    atomic_store(&NEXT_ROW, 0);
    if (NT == 1) mv_task(0, 1, NULL); /* no thread pool (browser build) */
    else es_cpool_run(CP, mv_task, NULL);
    NJOBS = 0;
}

/* ---------------- math helpers ---------------- */
static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    double ss = 0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float s = (float)(1.0 / sqrt(ss / n + eps));
    for (int i = 0; i < n; i++) o[i] = x[i] * s * (w ? w[i] : 1.0f);
}

/* cos/sin of the rotation angles are the same for every head and layer at one position: compute once per token */
static float *RC, *RS; static int RPOS = -1, RHD;
static void rope(float *v, int n_heads, int hd, int pos) {
    if (pos != RPOS || hd != RHD) {
        if (hd != RHD) { free(RC); free(RS); RC = malloc(sizeof(float) * (size_t)(hd / 2)); RS = malloc(sizeof(float) * (size_t)(hd / 2)); RHD = hd; }
        for (int i = 0; i < hd / 2; i++) { const float theta = pos * powf(M.rope_base, -2.0f * i / hd); RC[i] = cosf(theta); RS[i] = sinf(theta); }
        RPOS = pos;
    }
    for (int h = 0; h < n_heads; h++) {
        float *x = v + h * hd;
        for (int i = 0; i < hd / 2; i++) {
            const float c = RC[i], s = RS[i];
            const int a = M.rope_neox ? i : 2 * i, b = M.rope_neox ? i + hd / 2 : 2 * i + 1;
            const float x0 = x[a], x1 = x[b];
            x[a] = x0 * c - x1 * s;
            x[b] = x0 * s + x1 * c;
        }
    }
}

/* ---------------- state ---------------- */
static float *X, *XN, *Q, *K, *V, *ATT, *TMP, *LOGITS, *ROUTER;
static float *KC, *VC;   /* [layer][ctx][kv_dim] */
static float *GU, *H, *YE;
static es_block_q8_K *Q8A, *Q8H;

/* attention for one position, parallel over heads */
typedef struct { int layer, pos; } att_arg;
static void att_task(int tid, int n, void *arg) {
    const att_arg *aa = arg;
    const int hd = M.head_dim, kvd = M.n_head_kv * hd, grp = M.n_head / M.n_head_kv;
    const float *kc = KC + (size_t)aa->layer * M.ctx * kvd, *vc = VC + (size_t)aa->layer * M.ctx * kvd;
    float *sc = SCRATCH[tid];
    const float scale = 1.0f / sqrtf((float)hd);
    int h0, h1;
    es_split(M.n_head, tid, n, &h0, &h1);
    for (int h = h0; h < h1; h++) {
        const float *q = Q + h * hd;
        const int kh = h / grp;
        float mx = -INFINITY;
        for (int t = 0; t <= aa->pos; t++) {
            const float *k = kc + (size_t)t * kvd + kh * hd;
            float s = 0;
            for (int i = 0; i < hd; i++) s += q[i] * k[i];
            sc[t] = s * scale;
            if (sc[t] > mx) mx = sc[t];
        }
        float sum = 0;
        for (int t = 0; t <= aa->pos; t++) { sc[t] = expf(sc[t] - mx); sum += sc[t]; }
        float *o = ATT + h * hd;
        memset(o, 0, sizeof(float) * (size_t)hd);
        for (int t = 0; t <= aa->pos; t++) {
            const float p = sc[t] / sum;
            const float *v = vc + (size_t)t * kvd + kh * hd;
            for (int i = 0; i < hd; i++) o[i] += p * v[i];
        }
    }
}

/* ---------------- experts ---------------- */
static const char *PACK;
static es_pool *IO;
static es_cache C;
static double STAT_WAIT_S, STAT_READ_MB;
static unsigned long long STAT_HITS, STAT_MISS;
static FILE *TRACE;
static int TRACE_TOKEN;

/* direct GGUF mode: the three expert tensors of each layer */
static const es_gguf_tensor **XT;   /* [layer*3 + {gate, up, down}] */
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
    for (int r = 0; r < 3; r++) {
        const es_gguf_tensor *x = XT[t->l * 3 + r];
        const uint64_t per = x->nbytes / x->ne[2];
        read_tensor(x, per * (uint64_t)t->e, (uint8_t *)s->buf + XOFF[r], per);
    }
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
    for (int j = 0; j < T && j < ne; j++) {
        int best = -1;
        for (int e = 0; e < ne; e++) {
            if (PFR[e] == -INFINITY) continue;
            if (best < 0 || PFR[e] > PFR[best]) best = e;
        }
        if (best < 0) break;
        const float keep = PFR[best]; PFR[best] = -INFINITY; (void)keep;
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

/* Make sure the k experts of layer l are in the RAM cache; fill W views. */
static void load_experts(int l, const int *sel, int k, W *g, W *u, W *d) {
    C.step++;
    int slot[MAXK];
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
                for (int r = 0; r < 3; r++) {
                    const es_gguf_tensor *t = XT[l * 3 + r];
                    const uint64_t per = t->nbytes / t->ne[2];
                    read_tensor(t, per * (uint64_t)sel[j], (uint8_t *)C.slots[s].buf + XOFF[r], per);
                }
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
            const es_gguf_tensor *tg = XT[l * 3], *tu = XT[l * 3 + 1], *td = XT[l * 3 + 2];
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
}

static void moe(int l, const float *xn, float *out) {
    const int ne = M.n_exp;
    int k = M.n_used;
    es_act a;
    es_act_prepare(&a, xn, M.n_embd, M.n_embd % 256 ? NULL : Q8A);
    /* router: softmax over experts, take top-k */
    job(&M.L[l].gate_inp, &a, ROUTER);
    run_jobs();
    float mx = -INFINITY, sum = 0;
    for (int e = 0; e < ne; e++) mx = ROUTER[e] > mx ? ROUTER[e] : mx;
    for (int e = 0; e < ne; e++) { ROUTER[e] = expf(ROUTER[e] - mx); sum += ROUTER[e]; }
    for (int e = 0; e < ne; e++) ROUTER[e] /= sum;
    int sel[MAXK];
    float w[MAXK], wsum = 0;
    for (int j = 0; j < k; j++) {
        int best = -1;
        for (int e = 0; e < ne; e++) {
            int used = 0;
            for (int i = 0; i < j; i++) used |= sel[i] == e;
            if (!used && (best < 0 || ROUTER[e] > ROUTER[best])) best = e;
        }
        sel[j] = best;
        w[j] = ROUTER[best];
        wsum += w[j];
    }
#ifndef __EMSCRIPTEN__
    if (SKIP_THR > 0 && k > 1) {
        float mxw = 0; for (int j = 0; j < k; j++) if (w[j] > mxw) mxw = w[j];
        int kk = 0; wsum = 0;
        for (int j = 0; j < k; j++) if (w[j] >= SKIP_THR * mxw) { sel[kk] = sel[j]; w[kk] = w[j]; wsum += w[j]; kk++; }
        STAT_SKIPPED += (unsigned long long)(k - kk); k = kk;
    }
    if (LOOKAHEAD) prefetch_layer(l + 1, X);
#endif
    if (M.norm_topk) for (int j = 0; j < k; j++) w[j] /= wsum;
    if (TRACE) {
        fprintf(TRACE, "%d %d", TRACE_TOKEN, l);
        for (int j = 0; j < k; j++) fprintf(TRACE, " %d", sel[j]);
        fputc('\n', TRACE);
    }

    W g[MAXK], u[MAXK], d[MAXK];
    load_experts(l, sel, k, g, u, d);

    const int ff = g[0].rows;
    for (int j = 0; j < k; j++) {
        job(&g[j], &a, GU + (size_t)(2 * j) * ff);
        job(&u[j], &a, GU + (size_t)(2 * j + 1) * ff);
    }
    run_jobs();
    es_act ah[MAXK];
    for (int j = 0; j < k; j++) {
        const float *gg = GU + (size_t)(2 * j) * ff, *uu = gg + ff;
        float *hh = H + (size_t)j * ff;
        for (int i = 0; i < ff; i++) hh[i] = gg[i] / (1.0f + expf(-gg[i])) * uu[i];
        es_act_prepare(&ah[j], hh, ff, ff % 256 ? NULL : Q8H + (size_t)j * (ff / 256));
        job(&d[j], &ah[j], YE + (size_t)j * M.n_embd);
    }
    run_jobs();
    for (int i = 0; i < M.n_embd; i++) {
        float s = 0;
        for (int j = 0; j < k; j++) s += w[j] * YE[(size_t)j * M.n_embd + i];
        out[i] = s;
    }
}

static void dense_ffn(const W *wg, const W *wu, const W *wd, const float *xn, float *out) {
    es_act a, ah;
    es_act_prepare(&a, xn, M.n_embd, M.n_embd % 256 ? NULL : Q8A);
    const int ff = wg->rows;
    job(wg, &a, GU);
    job(wu, &a, GU + ff);
    run_jobs();
    for (int i = 0; i < ff; i++) H[i] = GU[i] / (1.0f + expf(-GU[i])) * GU[ff + i];
    es_act_prepare(&ah, H, ff, ff % 256 ? NULL : Q8H);
    job(wd, &ah, out);
    run_jobs();
}

/* one token through the whole model; writes LOGITS if want_logits */
static void forward(int32_t tok, int pos, int want_logits) {
    const int n = M.n_embd, hd = M.head_dim, kvd = M.n_head_kv * hd, qd_ = M.n_head * hd;
    es_dequant_row(M.tok_embd.type, (const uint8_t *)M.tok_embd.data + es_row_bytes(M.tok_embd.type, n) * (size_t)tok, X, n);
    for (int l = 0; l < M.n_layer; l++) {
        layer *L = &M.L[l];
        es_act_arena_reset();   /* activations of the previous layer are dead */
        rmsnorm(XN, X, L->attn_norm.data, n, M.eps);
        es_act a;
        es_act_prepare(&a, XN, n, n % 256 ? NULL : Q8A);
        job(&L->q, &a, Q);
        job(&L->k, &a, K);
        job(&L->v, &a, V);
        run_jobs();
        if (L->bq.data) {
            const float *b = L->bq.data; for (int i = 0; i < qd_; i++) Q[i] += b[i];
            b = L->bk.data; for (int i = 0; i < kvd; i++) K[i] += b[i];
            b = L->bv.data; for (int i = 0; i < kvd; i++) V[i] += b[i];
        }
        if (L->q_norm.data) {
            if (L->q_norm.cols == n) rmsnorm(Q, Q, L->q_norm.data, M.n_head * hd, M.eps);
            else for (int h = 0; h < M.n_head; h++) rmsnorm(Q + h * hd, Q + h * hd, L->q_norm.data, hd, M.eps);
        }
        if (L->k_norm.data) {
            if (L->k_norm.cols == kvd && kvd != hd) rmsnorm(K, K, L->k_norm.data, kvd, M.eps);
            else for (int h = 0; h < M.n_head_kv; h++) rmsnorm(K + h * hd, K + h * hd, L->k_norm.data, hd, M.eps);
        }
        rope(Q, M.n_head, hd, pos);
        rope(K, M.n_head_kv, hd, pos);
        memcpy(KC + ((size_t)l * M.ctx + pos) * kvd, K, sizeof(float) * (size_t)kvd);
        memcpy(VC + ((size_t)l * M.ctx + pos) * kvd, V, sizeof(float) * (size_t)kvd);
        att_arg aa = {l, pos};
        if (NT == 1) att_task(0, 1, &aa); else es_cpool_run(CP, att_task, &aa);
        es_act ao;
        es_act_prepare(&ao, ATT, M.n_head * hd, (M.n_head * hd) % 256 ? NULL : Q8H);
        job(&L->o, &ao, TMP);
        run_jobs();
        for (int i = 0; i < n; i++) X[i] += TMP[i];

        rmsnorm(XN, X, L->ffn_norm.data, n, M.eps);
        if (L->moe) moe(l, XN, TMP);
        else dense_ffn(&L->ffn_gate, &L->ffn_up, &L->ffn_down, XN, TMP);
        for (int i = 0; i < n; i++) X[i] += TMP[i];
        if (L->sh_gate.data) {
            dense_ffn(&L->sh_gate, &L->sh_up, &L->sh_down, XN, TMP);
            for (int i = 0; i < n; i++) X[i] += TMP[i];
        }
    }
    if (!want_logits) return;
    es_act_arena_reset();
    rmsnorm(XN, X, M.output_norm.data, n, M.eps);
    es_act a;
    es_act_prepare(&a, XN, n, n % 256 ? NULL : Q8A);
    job(&M.output, &a, LOGITS);
    run_jobs();
}


/* ================= batched forward (prompt reading and speculative verification) =================
 * N tokens go through the model together: every weight matrix is read from memory once for all N
 * (dense models; MoE models keep the one-token path). */
#define BMAX 16
static int BATCH_OK;
static float *BX, *BXN, *BQ, *BK, *BV, *BATTN, *BTMP, *BGU, *BH, *BLOG;
static es_block_q8_K *BQ8; static size_t BQ8STR;
static es_act BACT[BMAX];
static void forward_n(const int32_t *toks, int N, int pos0, int logits_from) {
    const int n = M.n_embd, hd = M.head_dim, kvd = M.n_head_kv * hd, qd_ = M.n_head * hd;
    for (int i = 0; i < N; i++) es_dequant_row(M.tok_embd.type, (const uint8_t *)M.tok_embd.data + es_row_bytes(M.tok_embd.type, n) * (size_t)toks[i], BX + (size_t)i * n, n);
#define PREP(src, cols, i) es_act_prepare(&BACT[i], (src), (cols), (cols) % 256 ? NULL : BQ8 + (size_t)(i) * BQ8STR)
    for (int l = 0; l < M.n_layer; l++) {
        layer *L = &M.L[l];
        es_act_arena_reset();
        for (int i = 0; i < N; i++) { rmsnorm(BXN + (size_t)i * n, BX + (size_t)i * n, L->attn_norm.data, n, M.eps); PREP(BXN + (size_t)i * n, n, i); }
        jobn(&L->q, BACT, BQ, (size_t)qd_, N); jobn(&L->k, BACT, BK, (size_t)kvd, N); jobn(&L->v, BACT, BV, (size_t)kvd, N);
        run_jobs();
        for (int i = 0; i < N; i++) {
            float *q = BQ + (size_t)i * qd_, *k = BK + (size_t)i * kvd, *v = BV + (size_t)i * kvd;
            if (L->bq.data) {
                for (int j = 0; j < qd_; j++) q[j] += ((const float *)L->bq.data)[j];
                for (int j = 0; j < kvd; j++) { k[j] += ((const float *)L->bk.data)[j]; v[j] += ((const float *)L->bv.data)[j]; }
            }
            if (L->q_norm.data) {
                if (L->q_norm.cols == n) rmsnorm(q, q, L->q_norm.data, M.n_head * hd, M.eps);
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
        for (int i = 0; i < N; i++) {   /* causal: token i sees positions <= pos0 + i, all already written */
            memcpy(Q, BQ + (size_t)i * qd_, sizeof(float) * (size_t)qd_);
            att_arg aa = {l, pos0 + i};
            if (NT == 1) att_task(0, 1, &aa); else es_cpool_run(CP, att_task, &aa);
            memcpy(BATTN + (size_t)i * qd_, ATT, sizeof(float) * (size_t)qd_);
        }
        es_act_arena_reset();
        for (int i = 0; i < N; i++) PREP(BATTN + (size_t)i * qd_, qd_, i);
        jobn(&L->o, BACT, BTMP, (size_t)n, N); run_jobs();
        for (size_t i = 0; i < (size_t)N * n; i++) BX[i] += BTMP[i];
        es_act_arena_reset();
        for (int i = 0; i < N; i++) { rmsnorm(BXN + (size_t)i * n, BX + (size_t)i * n, L->ffn_norm.data, n, M.eps); PREP(BXN + (size_t)i * n, n, i); }
        const int ff = L->ffn_gate.rows;
        jobn(&L->ffn_gate, BACT, BGU, (size_t)2 * ff, N); jobn(&L->ffn_up, BACT, BGU + ff, (size_t)2 * ff, N); run_jobs();
        for (int i = 0; i < N; i++) { const float *g = BGU + (size_t)i * 2 * ff; float *h = BH + (size_t)i * ff; for (int j = 0; j < ff; j++) h[j] = g[j] / (1.0f + expf(-g[j])) * g[ff + j]; }
        es_act_arena_reset();
        for (int i = 0; i < N; i++) PREP(BH + (size_t)i * ff, ff, i);
        jobn(&L->ffn_down, BACT, BTMP, (size_t)n, N); run_jobs();
        for (size_t i = 0; i < (size_t)N * n; i++) BX[i] += BTMP[i];
    }
    if (logits_from >= N) return;
    es_act_arena_reset();
    const int cnt = N - logits_from;
    for (int j = 0; j < cnt; j++) { rmsnorm(BXN + (size_t)j * n, BX + (size_t)(logits_from + j) * n, M.output_norm.data, n, M.eps); PREP(BXN + (size_t)j * n, n, j); }
    jobn(&M.output, BACT, BLOG, (size_t)M.n_vocab, cnt); run_jobs();
#undef PREP
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
static void detect_template(void) {
    const char *t = es_gguf_str(&G, "tokenizer.chat_template");
    if (!t) return;
    if (strstr(t, "<|im_start|>")) TEMPLATE_KIND = "chatml";
    else if (strstr(t, "<|user|>")) TEMPLATE_KIND = "tulu";
}
static void format_turn(char *out, size_t cap, const char *msg, int first) {
    /* later turns first close the previous answer: its end token was
     * sampled but not fed back into the model */
    if (!strcmp(TEMPLATE_KIND, "chatml"))
        snprintf(out, cap, "%s<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", first ? "" : "<|im_end|>\n", msg);
    else if (!strcmp(TEMPLATE_KIND, "tulu"))
        snprintf(out, cap, "%s<|user|>\n%s\n<|assistant|>\n", first ? "<|endoftext|>" : "<|endoftext|>\n", msg);
    else
        snprintf(out, cap, "%s", msg);
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
    M.n_layer = (int)HPI("block_count", 0);
    M.n_embd = (int)HPI("embedding_length", 0);
    M.n_head = (int)HPI("attention.head_count", 0);
    M.n_head_kv = (int)HPI("attention.head_count_kv", M.n_head);
    M.n_exp = (int)HPI("expert_count", 0);
    M.n_used = (int)HPI("expert_used_count", 0);
    snprintf(key, sizeof key, "%s.attention.key_length", M.arch);
    M.head_dim = (int)es_gguf_int(&G, key, M.n_head ? M.n_embd / M.n_head : 0);
    const es_gguf_kv *kv;
    snprintf(key, sizeof key, "%s.attention.layer_norm_rms_epsilon", M.arch);
    M.eps = (kv = es_gguf_find(&G, key)) ? (float)kv->v.f : 1e-5f;
    snprintf(key, sizeof key, "%s.rope.freq_base", M.arch);
    M.rope_base = (kv = es_gguf_find(&G, key)) ? (float)kv->v.f : 10000.0f;
    if (!strcmp(M.arch, "olmoe")) { M.rope_neox = 1; M.norm_topk = 0; }
    else if (!strcmp(M.arch, "qwen3moe")) { M.rope_neox = 1; M.norm_topk = 1; }
    else if (!strcmp(M.arch, "qwen3") || !strcmp(M.arch, "qwen2")) { M.rope_neox = 1; M.norm_topk = 0; }
    else if (!strcmp(M.arch, "llama") || !strcmp(M.arch, "mixtral")) { M.rope_neox = 0; M.norm_topk = 1; }
    else ERR("architecture '%s' is not supported yet (olmoe, qwen3moe, qwen3, qwen2, llama are)", M.arch);
    if (M.n_exp < 0 || M.n_used > MAXK || (M.n_exp > 0 && M.n_used <= 0)) ERR("unsupported expert layout");
    snprintf(key, sizeof key, "%s.rope.dimension_count", M.arch);
    { int rd = (int)es_gguf_int(&G, key, M.head_dim);
      if (rd != M.head_dim) ERR("partial rotary embeddings (%d of %d dims) are not supported yet", rd, M.head_dim); }
    snprintf(key, sizeof key, "%s.rope.scaling.type", M.arch);
    { const char *rs = es_gguf_str(&G, key); (void)rs; }

    /* copy every non-expert tensor into RAM */
    M.tptr = calloc(G.n_tensors, sizeof(uint8_t *));
    for (uint64_t i = 0; i < G.n_tensors; i++) {
        const es_gguf_tensor *t = &G.t[i];
        if (strstr(t->name, "_exps.")) continue;
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
    M.L = calloc((size_t)M.n_layer, sizeof(layer));
    for (int l = 0; l < M.n_layer; l++) {
        layer *L = &M.L[l];
#define T_(field, name, req) do { snprintf(key, sizeof key, "blk.%d." name ".weight", l); L->field = load_w(key, req); } while (0)
        T_(attn_norm, "attn_norm", 1); T_(q, "attn_q", 1); T_(k, "attn_k", 1); T_(v, "attn_v", 1);
        T_(o, "attn_output", 1); T_(q_norm, "attn_q_norm", 0); T_(k_norm, "attn_k_norm", 0);
        { snprintf(key, sizeof key, "blk.%d.attn_q.bias", l); L->bq = load_w(key, 0);
          snprintf(key, sizeof key, "blk.%d.attn_k.bias", l); L->bk = load_w(key, 0);
          snprintf(key, sizeof key, "blk.%d.attn_v.bias", l); L->bv = load_w(key, 0);
          if (!L->bq.data != !L->bk.data || !L->bq.data != !L->bv.data) ERR("layer %d has only some of the Q/K/V biases", l); }
        T_(ffn_norm, "ffn_norm", 1); T_(gate_inp, "ffn_gate_inp", 0);
        T_(ffn_gate, "ffn_gate", 0); T_(ffn_up, "ffn_up", 0); T_(ffn_down, "ffn_down", 0);
        T_(sh_gate, "ffn_gate_shexp", 0); T_(sh_up, "ffn_up_shexp", 0); T_(sh_down, "ffn_down_shexp", 0);
        L->moe = L->gate_inp.data != NULL;
        if (!L->moe && !L->ffn_gate.data) ERR("layer %d has neither experts nor a dense FFN", l);
    }
    T = es_tok_load(&G, &e);
    if (!T) ERR("tokenizer: %s", e);
    detect_template();
    USER_TOK = es_tok_find(T, "<|user|>");

    /* threads: compute on the fastest cores, readers on the rest */
    NT = o->threads > 0 ? o->threads : 1;
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
    int any_moe = 0;
    for (int l = 0; l < M.n_layer; l++) if (M.L[l].moe) any_moe = 1;
    if (!any_moe) {
        slot_bytes = 4096;                      /* dense model: nothing to stream */
    } else if (PACK) {
        unsigned lay, ne;
        if (es_read_manifest(PACK, &lay, &ne, &slot_bytes)) ERR("no manifest.txt in %s", PACK);
    } else {
        XT = calloc((size_t)M.n_layer * 3, sizeof *XT);
        static const char *kind[3] = {"gate", "up", "down"};
        for (int l = 0; l < M.n_layer; l++) {
            if (!M.L[l].moe) continue;
            size_t off = 0;
            for (int r = 0; r < 3; r++) {
                snprintf(key, sizeof key, "blk.%d.ffn_%s_exps.weight", l, kind[r]);
                XT[l * 3 + r] = es_gguf_tensor_find(&G, key);
                if (!XT[l * 3 + r]) ERR("missing %s (fused expert layouts are not supported yet)", key);
                if (!es_quant_supported(XT[l * 3 + r]->type)) ERR("%s uses %s, not supported yet", key, es_ggml_type_name(XT[l * 3 + r]->type));
                XOFF[r] = off;
                off += (XT[l * 3 + r]->nbytes / XT[l * 3 + r]->ne[2] + 63) / 64 * 64;
            }
            if (off > slot_bytes) slot_bytes = off;
        }
    }
    int nslots = any_moe ? (int)(o->cache_mb * 1048576ull / slot_bytes) : 0;
    if (any_moe && nslots < 2 * M.n_used + 1) nslots = 2 * M.n_used + 1;
    if (any_moe) { if (es_cache_init(&C, nslots, slot_bytes, M.n_layer, M.n_exp, 1.0 * M.n_layer)) ERR("out of memory for the expert cache"); }
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
    const int n = M.n_embd, kvd = M.n_head_kv * M.head_dim, qd = M.n_head * M.head_dim;
    int ffmax = 8192;
    for (int l = 0; l < M.n_layer; l++) if (M.L[l].ffn_gate.rows > ffmax) ffmax = M.L[l].ffn_gate.rows;
    X = malloc(sizeof(float) * n); XN = malloc(sizeof(float) * n); TMP = malloc(sizeof(float) * n);
    Q = malloc(sizeof(float) * qd); K = malloc(sizeof(float) * kvd); V = malloc(sizeof(float) * kvd);
    ATT = malloc(sizeof(float) * qd);
    LOGITS = malloc(sizeof(float) * (size_t)M.n_vocab);
    ROUTER = malloc(sizeof(float) * (size_t)(M.n_exp + 1));
    GU = malloc(sizeof(float) * 2 * (size_t)ffmax * MAXK);
    H = malloc(sizeof(float) * (size_t)ffmax * MAXK);
    YE = malloc(sizeof(float) * (size_t)n * MAXK);
    es_act_arena_init(sizeof(es_blk_q80) * ((size_t)(n + qd + 2 * ffmax * MAXK + 4096) / 32 + 64));
    Q8A = malloc(sizeof(es_block_q8_K) * (size_t)(n / 256 + 1 + qd / 256 + 1));
    Q8H = malloc(sizeof(es_block_q8_K) * (size_t)((ffmax / 256 + 1) * MAXK + qd / 256 + 1));
    HIST = calloc((size_t)M.ctx + 64, sizeof(int32_t));
    { const char *ev = getenv("ES_SPEC"); SPEC_MAX = ev ? atoi(ev) : (o->spec == -2 ? 0 : o->spec > 0 ? o->spec : 7); }
    BATCH_OK = !any_moe && !getenv("ES_NO_BATCH");
    if (BATCH_OK) {
        for (int l = 0; l < M.n_layer; l++) if (M.L[l].moe || M.L[l].sh_gate.data) BATCH_OK = 0;
        const int ffm = M.L[0].ffn_gate.rows > 0 ? M.L[0].ffn_gate.rows : ffmax;
        int mx = n > qd ? n : qd; if (ffm > mx) mx = ffm;
        BQ8STR = (size_t)(mx / 256 + 2);
        BQ8 = malloc(sizeof(es_block_q8_K) * BQ8STR * BMAX);
        BX = malloc(sizeof(float) * (size_t)n * BMAX); BXN = malloc(sizeof(float) * (size_t)n * BMAX); BTMP = malloc(sizeof(float) * (size_t)n * BMAX);
        BQ = malloc(sizeof(float) * (size_t)qd * BMAX); BATTN = malloc(sizeof(float) * (size_t)qd * BMAX);
        BK = malloc(sizeof(float) * (size_t)kvd * BMAX); BV = malloc(sizeof(float) * (size_t)kvd * BMAX);
        BGU = malloc(sizeof(float) * 2 * (size_t)ffm * BMAX); BH = malloc(sizeof(float) * (size_t)ffm * BMAX);
        BLOG = malloc(sizeof(float) * (size_t)M.n_vocab * BMAX);
        es_act_arena_init(sizeof(es_blk_q80) * ((size_t)BMAX * (mx + 64) / 32 + 4096));
    }
    size_t kv_bytes = sizeof(float) * (size_t)M.n_layer * M.ctx * kvd;
    KC = malloc(kv_bytes);
    VC = malloc(kv_bytes);
    if (!KC || !VC || !GU || !H) ERR("out of memory for a %d-token context; try a smaller context", M.ctx);
    if (o->trace_path) TRACE = fopen(o->trace_path, "w");

    const char *nm = es_gguf_str(&G, "general.name");
    INFO = (es_engine_info){nm ? nm : M.arch, M.arch, M.n_layer, M.n_exp, M.n_used, M.ctx, nslots,
                            M.core_bytes / 1e6, any_moe ? nslots * (double)slot_bytes / 1e6 : 0.0, 2 * kv_bytes / 1e6,
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
        TRACE_TOKEN = POS;
        forward(toks[i], POS++, i == nt - 1);
        if (progress) progress(i + 1, nt, ud);
    }
    ST.prompt_tokens = nt;
    ST.prompt_s = (es_now_ns() - tp) / 1e9;
    GEN = 0;
    DONE = 0;
    FIRST = 0;
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

static int emit_piece(int tok, int tokpos, char *buf, int cap) {
    if (es_tok_is_eog(T, tok) || (tok == USER_TOK && USER_TOK >= 0)) { POS = tokpos; DONE = 1; return -1; }
    GEN++;
    return es_tok_piece(T, tok, 0, buf, cap);
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
        int pl = es_tok_piece(T, next, 0, buf, cap);
        TRACE_TOKEN = POS;
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
    return es_tok_piece(T, next, 0, buf, cap);
}

int es_engine_dump_logits(const int32_t *ids, int n, FILE *f) {
    for (int i = 0; i < n; i++) {
        forward(ids[i], i, 1);
        fwrite(LOGITS, sizeof(float), (size_t)M.n_vocab, f);
    }
    return 0;
}
