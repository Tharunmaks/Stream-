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

typedef struct { uint32_t type; const void *w; int rows, cols; const es_act *a; float *y; } mv_job;
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
            es_matvec(JOBS[j].type, JOBS[j].w, JOBS[j].cols, (int)(c - base), (int)(b - base),
                      JOBS[j].a, JOBS[j].y, SCRATCH[tid]);
            c = b;
        }
    }
}
static void job(const W *w, const es_act *a, float *y) {
    JOBS[NJOBS++] = (mv_job){w->type, w->data, w->rows, w->cols, a, y};
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

static void rope(float *v, int n_heads, int hd, int pos) {
    for (int h = 0; h < n_heads; h++) {
        float *x = v + h * hd;
        for (int i = 0; i < hd / 2; i++) {
            const float theta = pos * powf(M.rope_base, -2.0f * i / hd);
            const float c = cosf(theta), s = sinf(theta);
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

static void read_at(uint64_t off, void *dst, size_t n) {
    uint8_t *p = dst;
    while (n) {
        ssize_t r = es_gguf_pread(G.fd, p, n, off);
        if (r <= 0) { fprintf(stderr, "read failed at %llu\n", (unsigned long long)off); exit(1); }
        p += r; n -= (size_t)r; off += (uint64_t)r;
    }
}

/* Make sure the k experts of layer l are in the RAM cache; fill W views. */
static void load_experts(int l, const int *sel, int k, W *g, W *u, W *d) {
    C.step++;
    int slot[MAXK];
    char path[4096];
    for (int j = 0; j < k; j++) {
        int s = es_cache_find(&C, l, sel[j]);
        if (s >= 0) {
            STAT_HITS++;
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
            } else {
                uint64_t t0 = es_now_ns();
                for (int r = 0; r < 3; r++) {
                    const es_gguf_tensor *t = XT[l * 3 + r];
                    const uint64_t per = t->nbytes / t->ne[2];
                    read_at(G.data_start + t->offset + per * (uint64_t)sel[j], (uint8_t *)C.slots[s].buf + XOFF[r], per);
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
    const int ne = M.n_exp, k = M.n_used;
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

/* ---------------- sampling ---------------- */
static uint64_t RNG = 42;
static int sample(float temp, int top_k, float top_p) {
    const int nv = M.n_vocab;
    if (temp <= 0) {
        int best = 0;
        for (int i = 1; i < nv; i++) if (LOGITS[i] > LOGITS[best]) best = i;
        return best;
    }
    static int idx[1 << 18];
    static float pr[1 << 18];
    int n = 0;
    for (int i = 0; i < nv; i++) idx[n++] = i;
    if (top_k <= 0 || top_k > nv) top_k = nv;
    /* partial selection sort for the top_k logits */
    for (int i = 0; i < top_k; i++) {
        int b = i;
        for (int j = i + 1; j < n; j++) if (LOGITS[idx[j]] > LOGITS[idx[b]]) b = j;
        int t = idx[i]; idx[i] = idx[b]; idx[b] = t;
    }
    float mx = LOGITS[idx[0]], sum = 0;
    for (int i = 0; i < top_k; i++) { pr[i] = expf((LOGITS[idx[i]] - mx) / temp); sum += pr[i]; }
    float cum = 0;
    int keep = top_k;
    for (int i = 0; i < top_k; i++) { pr[i] /= sum; cum += pr[i]; if (cum >= top_p) { keep = i + 1; break; } }
    float r = (float)((es_rng_next(&RNG) >> 11) * (1.0 / 9007199254740992.0)) * cum, acc = 0;
    for (int i = 0; i < keep; i++) { acc += pr[i]; if (r <= acc) return idx[i]; }
    return idx[keep - 1];
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
    } else {
        if (es_gguf_open_fd(&G, o->gguf_fd, o->gguf_size, &e)) ERR("model file: %s", e);
    }
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
        read_at(G.data_start + t->offset, M.tptr[i], t->nbytes);
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

int es_engine_next(char *buf, int cap) {
    if (DONE || GEN >= MAX_NEW || POS >= M.ctx) { DONE = 1; return -1; }
    int next = sample(TEMP, TOP_K, TOP_P);
    if (es_tok_is_eog(T, next) || (next == USER_TOK && USER_TOK >= 0)) { DONE = 1; return -1; }
    int pl = es_tok_piece(T, next, 0, buf, cap);
    TRACE_TOKEN = POS;
    forward(next, POS++, 1);
    GEN++;
    return pl;
}

int es_engine_dump_logits(const int32_t *ids, int n, FILE *f) {
    for (int i = 0; i < n; i++) {
        forward(ids[i], i, 1);
        fwrite(LOGITS, sizeof(float), (size_t)M.n_vocab, f);
    }
    return 0;
}
