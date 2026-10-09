/*
 * es_chat: real text generation from an ExpertStream pack.
 *
 * The core (embeddings, attention, norms, router, output head) is loaded
 * into RAM from core.gguf. Routed experts are read from the pack's expert
 * files on demand through the 4-reader flash loader and kept in a RAM
 * cache. Supported architectures: olmoe, qwen3moe, llama/mixtral-style MoE.
 *
 *   ./es_chat -m ~/packs/olmoe                       # interactive chat
 *   ./es_chat -m ~/packs/olmoe -p "Write a haiku about phones"
 *   ./es_chat -m ~/packs/olmoe -r -p "Once upon a time" -n 64   # raw text
 */
#define _GNU_SOURCE
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/es_cache.h"
#include "../src/es_compute.h"
#include "../src/es_cpu.h"
#include "../src/es_format.h"
#include "../src/es_gguf.h"
#include "../src/es_io.h"
#include "../src/es_quant.h"
#include "../src/es_tok.h"
#include "es_common.h"

#define MAXK 16
#define MAXJOBS 40

/* ---------------- weights ---------------- */
typedef struct { uint32_t type; int rows, cols; const void *data; } W;

typedef struct {
    W attn_norm, q, k, v, o, q_norm, k_norm, ffn_norm, gate_inp;
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
    uint8_t *core;
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
    w.data = M.core + t->offset;
    return w;
}

static long rss_mb(void) {
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

static void mv_task(int tid, int n, void *arg) {
    (void)arg;
    long total = 0;
    for (int j = 0; j < NJOBS; j++) total += JOBS[j].rows;
    long lo = total * tid / n, hi = total * (tid + 1) / n, base = 0;
    for (int j = 0; j < NJOBS; j++) {
        long a = lo > base ? lo : base, b = hi < base + JOBS[j].rows ? hi : base + JOBS[j].rows;
        if (a < b)
            es_matvec(JOBS[j].type, JOBS[j].w, JOBS[j].cols, (int)(a - base), (int)(b - base),
                      JOBS[j].a, JOBS[j].y, SCRATCH[tid]);
        base += JOBS[j].rows;
    }
}
static void job(const W *w, const es_act *a, float *y) {
    JOBS[NJOBS++] = (mv_job){w->type, w->data, w->rows, w->cols, a, y};
}
static void run_jobs(void) { es_cpool_run(CP, mv_task, NULL); NJOBS = 0; }

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
static unsigned long long FILE_BYTES;
static double STAT_WAIT_S, STAT_READ_MB;
static unsigned long long STAT_HITS, STAT_MISS;
static FILE *TRACE;
static int TRACE_TOKEN;

static const es_tensor_desc *find_role(const es_header *h, uint32_t role) {
    for (uint32_t i = 0; i < h->n_tensors; i++)
        if (h->t[i].role == role) return &h->t[i];
    return NULL;
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

    /* make sure the k experts are in RAM: submit all misses, then wait */
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
            es_expert_path(path, sizeof path, PACK, (unsigned)l, (unsigned)sel[j]);
            C.slots[s].req.dst = C.slots[s].buf;
            C.slots[s].req.cap = C.slot_bytes;
            if (es_submit(IO, &C.slots[s].req, path)) {
                fprintf(stderr, "cannot read %s: %s\n", path, strerror(C.slots[s].req.err));
                exit(1);
            }
            STAT_READ_MB += C.slot_bytes / 1048576.0;
        }
        C.slots[s].uses++;
        C.slots[s].last_use = C.step;
        C.slots[s].protect_until = C.step;
        slot[j] = s;
    }
    uint64_t t0 = es_now_ns();
    W g[MAXK], u[MAXK], d[MAXK];
    for (int j = 0; j < k; j++) {
        es_slot *s = &C.slots[slot[j]];
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
    STAT_WAIT_S += (es_now_ns() - t0) / 1e9;

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
    const int n = M.n_embd, hd = M.head_dim, kvd = M.n_head_kv * hd;
    es_dequant_row(M.tok_embd.type, (const uint8_t *)M.tok_embd.data + es_row_bytes(M.tok_embd.type, n) * (size_t)tok, X, n);
    for (int l = 0; l < M.n_layer; l++) {
        layer *L = &M.L[l];
        rmsnorm(XN, X, L->attn_norm.data, n, M.eps);
        es_act a;
        es_act_prepare(&a, XN, n, n % 256 ? NULL : Q8A);
        job(&L->q, &a, Q);
        job(&L->k, &a, K);
        job(&L->v, &a, V);
        run_jobs();
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
        es_cpool_run(CP, att_task, &aa);
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

/* ---------------- JSON-lines protocol (es_serve) ----------------
 * stdin : one prompt per line, "\\n" for newlines, "/reset" clears the chat
 * stdout: one JSON object per line: ready, tokens, prompt, tok, done, info */
static void json_str(const char *s, int n) {
    putchar('"');
    for (int i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)s[i];
        if (ch == '"' || ch == '\\') { putchar('\\'); putchar(ch); }
        else if (ch == '\n') fputs("\\n", stdout);
        else if (ch == '\r') fputs("\\r", stdout);
        else if (ch == '\t') fputs("\\t", stdout);
        else if (ch < 0x20) printf("\\u%04x", ch);
        else putchar(ch);
    }
    putchar('"');
}
static void unescape_line(char *s) {
    char *o = s;
    for (char *p = s; *p; p++) {
        if (*p == '\\' && p[1] == 'n') { *o++ = '\n'; p++; }
        else if (*p == '\\' && p[1] == '\\') { *o++ = '\\'; p++; }
        else *o++ = *p;
    }
    *o = 0;
}

/* ---------------- main ---------------- */
static void usage(void) {
    fprintf(stderr,
            "usage: es_chat -m PACKDIR [-p PROMPT] [-n max_tokens=256] [-r raw, no chat template]\n"
            "               [-t temperature=0.7 (0 = greedy)] [-k top_k=40] [-P top_p=0.9]\n"
            "               [-c context=2048] [-C cache_mb=1024] [-j threads=4] [-s seed]\n"
            "               [-T routing_trace.txt] [-q quiet: no stage log]\n"
            "               [-J JSON-lines protocol on stdin/stdout, used by es_serve]\n");
    exit(2);
}

int main(int argc, char **argv) {
    const char *prompt = NULL, *trace_path = NULL, *dump_ids = NULL, *dump_path = NULL;
    int max_new = 256, raw = 0, quiet = 0, top_k = 40, opt, json = 0;
    float temp = 0.7f, top_p = 0.9f;
    size_t cache_mb = 1024;
    M.ctx = 2048;
    NT = 4;
    while ((opt = getopt(argc, argv, "m:p:n:rt:k:P:c:C:j:s:T:qI:D:Jh")) != -1) {
        switch (opt) {
        case 'm': PACK = optarg; break;
        case 'p': prompt = optarg; break;
        case 'n': max_new = atoi(optarg); break;
        case 'r': raw = 1; break;
        case 't': temp = (float)atof(optarg); break;
        case 'k': top_k = atoi(optarg); break;
        case 'P': top_p = (float)atof(optarg); break;
        case 'c': M.ctx = atoi(optarg); break;
        case 'C': cache_mb = (size_t)atol(optarg); break;
        case 'j': NT = atoi(optarg); break;
        case 's': RNG = strtoull(optarg, NULL, 10) | 1; break;
        case 'T': trace_path = optarg; break;
        case 'q': quiet = 1; break;
        case 'I': dump_ids = optarg; break;   /* testing: comma-separated token ids */
        case 'D': dump_path = optarg; break;  /* testing: write every position's logits */
        case 'J': json = 1; quiet = 1; break; /* line protocol for es_serve */
        default: usage();
        }
    }
    if (!PACK) usage();
#define STAGE(...) do { if (!quiet) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while (0)

    /* ---- 1. load the core ("connect to the brain") ---- */
    uint64_t t0 = es_now_ns();
    char path[4096];
    snprintf(path, sizeof path, "%s/core.gguf", PACK);
    const char *err;
    if (es_gguf_open(&G, path, &err)) { fprintf(stderr, "%s: %s\n", path, err); return 1; }
    M.arch = es_gguf_str(&G, "general.architecture");
    char key[256];
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
    else if (!strcmp(M.arch, "qwen3moe") || !strcmp(M.arch, "qwen2moe")) { M.rope_neox = 1; M.norm_topk = !strcmp(M.arch, "qwen3moe"); }
    else if (!strcmp(M.arch, "llama") || !strcmp(M.arch, "mixtral")) { M.rope_neox = 0; M.norm_topk = 1; }
    else { fprintf(stderr, "architecture '%s' is not supported yet (olmoe, qwen3moe, llama/mixtral are)\n", M.arch); return 1; }
    if (M.n_used > MAXK) { fprintf(stderr, "too many experts per token\n"); return 1; }

    uint64_t data_bytes = G.file_size - G.data_start;
    M.core = es_alloc(data_bytes);
    M.core_bytes = data_bytes;
    for (uint64_t off = 0; off < data_bytes;) {
        ssize_t r = pread(G.fd, M.core + off, data_bytes - off > (1u << 26) ? (1u << 26) : data_bytes - off, (off_t)(G.data_start + off));
        if (r <= 0) { fprintf(stderr, "reading core.gguf failed\n"); return 1; }
        off += (uint64_t)r;
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
        T_(ffn_norm, "ffn_norm", 1); T_(gate_inp, "ffn_gate_inp", 0);
        T_(ffn_gate, "ffn_gate", 0); T_(ffn_up, "ffn_up", 0); T_(ffn_down, "ffn_down", 0);
        T_(sh_gate, "ffn_gate_shexp", 0); T_(sh_up, "ffn_up_shexp", 0); T_(sh_down, "ffn_down_shexp", 0);
        L->moe = L->gate_inp.data != NULL;
        if (!L->moe && !L->ffn_gate.data) { fprintf(stderr, "layer %d has neither experts nor a dense FFN\n", l); return 1; }
    }
    T = es_tok_load(&G, &err);
    if (!T) { fprintf(stderr, "tokenizer: %s\n", err); return 1; }
    detect_template();

    /* threads: compute on the fastest cores, readers on the rest */
    es_topo topo;
    es_topo_read(&topo);
    int ccpu[ES_MAX_CPUS], iocpu[ES_MAX_CPUS], nc = 0, nio = 0;
    for (int i = 0; i < topo.ncpu; i++) {
        int c = topo.order[i];
        if (!topo.allowed[c]) continue;
        if (nc < NT) ccpu[nc++] = c; else iocpu[nio++] = c;
    }
    CP = es_cpool_create(NT, nc == NT ? ccpu : NULL);
    IO = es_pool_create_on(4, 1 << 20, 1, iocpu, nio);
    const int big = M.ctx > M.n_embd * 4 ? M.ctx : M.n_embd * 4;
    for (int i = 0; i < NT; i++) SCRATCH[i] = malloc(sizeof(float) * (size_t)(big + 4096));

    /* expert cache */
    unsigned lay, ne;
    if (es_read_manifest(PACK, &lay, &ne, &FILE_BYTES)) { fprintf(stderr, "no manifest.txt in %s\n", PACK); return 1; }
    int nslots = (int)(cache_mb * 1048576ull / FILE_BYTES);
    if (nslots < 2 * M.n_used + 1) nslots = 2 * M.n_used + 1;
    if (es_cache_init(&C, nslots, FILE_BYTES, M.n_layer, M.n_exp, 1.0 * M.n_layer)) { fprintf(stderr, "out of memory for the expert cache\n"); return 1; }

    /* activations and KV cache */
    const int n = M.n_embd, kvd = M.n_head_kv * M.head_dim, qd = M.n_head * M.head_dim;
    int ffmax = 0;
    for (int l = 0; l < M.n_layer; l++) if (M.L[l].ffn_gate.rows > ffmax) ffmax = M.L[l].ffn_gate.rows;
    if (M.L[0].sh_gate.rows > ffmax) ffmax = M.L[0].sh_gate.rows;
    ffmax = ffmax > 8192 ? ffmax : 8192;
    X = malloc(sizeof(float) * n); XN = malloc(sizeof(float) * n); TMP = malloc(sizeof(float) * n);
    Q = malloc(sizeof(float) * qd); K = malloc(sizeof(float) * kvd); V = malloc(sizeof(float) * kvd);
    ATT = malloc(sizeof(float) * qd);
    LOGITS = malloc(sizeof(float) * (size_t)M.n_vocab);
    ROUTER = malloc(sizeof(float) * (size_t)(M.n_exp + 1));
    GU = malloc(sizeof(float) * 2 * (size_t)ffmax * MAXK);
    H = malloc(sizeof(float) * (size_t)ffmax * MAXK);
    YE = malloc(sizeof(float) * (size_t)n * MAXK);
    Q8A = malloc(sizeof(es_block_q8_K) * (size_t)(n / 256 + 1 + qd / 256 + 1));
    Q8H = malloc(sizeof(es_block_q8_K) * (size_t)((ffmax / 256 + 1) * MAXK + qd / 256 + 1));
    size_t kv_bytes = sizeof(float) * (size_t)M.n_layer * M.ctx * kvd;
    KC = malloc(kv_bytes);
    VC = malloc(kv_bytes);
    if (!KC || !VC) { fprintf(stderr, "out of memory for a %d-token context; try -c 1024\n", M.ctx); return 1; }
    if (trace_path) TRACE = fopen(trace_path, "w");

    STAGE("\033[1m[1/4] Connecting to the brain\033[0m  %s (%s), %d layers, %d experts, %d per token\n",
          es_gguf_str(&G, "general.name") ? es_gguf_str(&G, "general.name") : M.arch, M.arch, M.n_layer, M.n_exp, M.n_used);
    STAGE("      core.gguf %.0f MB loaded in %.2f s | expert cache %d slots (%zu MB) | context %d tokens (%.0f MB)\n",
          M.core_bytes / 1e6, (es_now_ns() - t0) / 1e9, nslots, cache_mb, M.ctx, 2 * kv_bytes / 1e6);
    STAGE("      threads: compute on %d fastest cores, 4 flash readers | RAM now %ld MB\n", NT, rss_mb());

    /* testing mode: run fixed token ids, dump logits of every position */
    if (dump_ids && dump_path) {
        FILE *df = fopen(dump_path, "wb");
        int p = 0;
        for (const char *s = dump_ids; *s;) {
            char *end;
            long id = strtol(s, &end, 10);
            if (end == s) { s++; continue; }
            forward((int32_t)id, p++, 1);
            fwrite(LOGITS, sizeof(float), (size_t)M.n_vocab, df);
            s = end;
        }
        fclose(df);
        STAGE("dumped logits for %d positions to %s\n", p, dump_path);
        return 0;
    }

    if (json) {
        printf("{\"ev\":\"ready\",\"model\":");
        const char *nm = es_gguf_str(&G, "general.name");
        json_str(nm ? nm : M.arch, (int)strlen(nm ? nm : M.arch));
        printf(",\"arch\":\"%s\",\"layers\":%d,\"experts\":%d,\"used\":%d,\"core_mb\":%.0f,\"cache_mb\":%zu,\"ctx\":%d,\"load_s\":%.2f,\"ram_mb\":%ld}\n",
               M.arch, M.n_layer, M.n_exp, M.n_used, M.core_bytes / 1e6, cache_mb, M.ctx, (es_now_ns() - t0) / 1e9, rss_mb());
        fflush(stdout);
    }

    /* ---- conversation loop ---- */
    int pos = 0, first = 1;
    static char line[1 << 15], text[1 << 16], piece[256];
    static int32_t toks[1 << 15];
    for (;;) {
        const char *msg = prompt;
        if (!msg) {
            if (!json) fprintf(stderr, "\n\033[1myou>\033[0m ");
            if (!fgets(line, sizeof line, stdin)) break;
            line[strcspn(line, "\n")] = 0;
            if (!line[0]) continue;
            if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) break;
            if (!strcmp(line, "/reset")) {
                pos = 0; first = 1;
                if (json) { printf("{\"ev\":\"info\",\"msg\":\"new chat\"}\n{\"ev\":\"done\"}\n"); fflush(stdout); }
                continue;
            }
            if (json) unescape_line(line);
            msg = line;
        }
        if (raw) snprintf(text, sizeof text, "%s", msg);
        else format_turn(text, sizeof text, msg, first);
        int nt = es_tok_encode(T, text, 1, toks, (int)(sizeof toks / sizeof toks[0]));
        if (raw && first && es_tok_add_bos(T) && es_tok_bos(T) >= 0) {
            memmove(toks + 1, toks, sizeof(int32_t) * (size_t)nt);
            toks[0] = es_tok_bos(T);
            nt++;
        }
        if (pos + nt + 1 >= M.ctx) {
            if (!json) { fprintf(stderr, "context full (%d tokens); restart or use -c\n", M.ctx); break; }
            /* start a fresh conversation rather than failing */
            pos = 0; first = 1;
            format_turn(text, sizeof text, msg, first);
            nt = es_tok_encode(T, text, 1, toks, (int)(sizeof toks / sizeof toks[0]));
            printf("{\"ev\":\"info\",\"msg\":\"context was full, started a new chat\"}\n");
            if (nt + 1 >= M.ctx) { printf("{\"ev\":\"info\",\"msg\":\"prompt too long\"}\n{\"ev\":\"done\"}\n"); fflush(stdout); continue; }
        }

        /* ---- 2. words to numbers ---- */
        STAGE("\033[1m[2/4] Turning words into numbers\033[0m  %d tokens:", nt);
        for (int i = 0; i < nt && i < 24; i++) STAGE(" %d", toks[i]);
        STAGE("%s\n", nt > 24 ? " ..." : "");
        if (json) {
            printf("{\"ev\":\"tokens\",\"n\":%d,\"ids\":[", nt);
            for (int i = 0; i < nt && i < 64; i++) printf("%s%d", i ? "," : "", toks[i]);
            printf("]}\n");
            fflush(stdout);
        }

        /* ---- 3. read the prompt ---- */
        unsigned long long h0 = STAT_HITS, m0 = STAT_MISS;
        double r0 = STAT_READ_MB, w0 = STAT_WAIT_S;
        uint64_t tp = es_now_ns();
        for (int i = 0; i < nt; i++) {
            TRACE_TOKEN = pos;
            forward(toks[i], pos++, i == nt - 1);
            STAGE("\r\033[1m[3/4] Reading the prompt\033[0m  %d/%d tokens", i + 1, nt);
            if (json && ((i + 1) % 4 == 0 || i + 1 == nt)) {
                printf("{\"ev\":\"prompt\",\"done\":%d,\"n\":%d,\"flash_mb\":%.0f,\"ram_mb\":%ld}\n", i + 1, nt, STAT_READ_MB - r0, rss_mb());
                fflush(stdout);
            }
        }
        double tps = (es_now_ns() - tp) / 1e9;
        STAGE("  %.1f s (%.2f tok/s) | flash %.0f MB | RAM %ld MB\n", tps, nt / tps, STAT_READ_MB - r0, rss_mb());

        /* ---- 4. answer ---- */
        STAGE("\033[1m[4/4] Answering\033[0m\n");
        uint64_t tg = es_now_ns();
        int gen = 0;
        for (; gen < max_new && pos < M.ctx; gen++) {
            int next = sample(temp, top_k, top_p);
            if (es_tok_is_eog(T, next)) break;
            int pl = es_tok_piece(T, next, 0, piece, sizeof piece);
            /* stop if the model starts writing the next user turn */
            if (!raw && !strcmp(TEMPLATE_KIND, "tulu") && next == es_tok_find(T, "<|user|>")) break;
            if (json) {
                printf("{\"ev\":\"tok\",\"t\":");
                json_str(piece, pl);
                printf("}\n");
            } else {
                fwrite(piece, 1, (size_t)pl, stdout);
            }
            fflush(stdout);
            TRACE_TOKEN = pos;
            forward(next, pos++, 1);
        }
        double tgs = (es_now_ns() - tg) / 1e9;
        unsigned long long hits = STAT_HITS - h0, miss = STAT_MISS - m0;
        if (json)
            printf("{\"ev\":\"done\",\"tokens\":%d,\"secs\":%.2f,\"tps\":%.2f,\"prompt_secs\":%.2f,\"hit\":%.1f,\"flash_mb\":%.0f,\"wait_s\":%.2f,\"ram_mb\":%ld}\n",
                   gen, tgs, gen / (tgs > 0 ? tgs : 1), tps, 100.0 * hits / (hits + miss ? hits + miss : 1),
                   STAT_READ_MB - r0, STAT_WAIT_S - w0, rss_mb());
        else
            printf("\n");
        fflush(stdout);
        STAGE("\033[2m      %d tokens in %.1f s = %.2f tok/s | experts %.0f%% from RAM, %.0f MB read from flash (%.1f s waiting) | RAM %ld MB\033[0m\n",
              gen, tgs, gen / (tgs > 0 ? tgs : 1), 100.0 * hits / (hits + miss ? hits + miss : 1),
              STAT_READ_MB - r0, STAT_WAIT_S - w0, rss_mb());
        first = 0;
        if (prompt) break;
    }
    if (TRACE) fclose(TRACE);
    es_cpool_destroy(CP);
    es_pool_destroy(IO);
    return 0;
}
