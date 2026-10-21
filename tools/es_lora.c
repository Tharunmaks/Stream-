/* es_lora: LoRA fine-tuning of a frozen GGUF model with exact gradients, on the CPU (phone or PC).
 *
 *   es_lora train -g MODEL.gguf --data FILE.jsonl|FILE.txt -o NAME.eslora [--rank 16] [--alpha 32] [--targets all]
 *                 [--lr 2e-4] [--epochs 1] [--steps N] [--seq 512] [--batch 2048] [--micro 1024] [--minutes M] [--resume]
 *   es_lora eval  -g MODEL.gguf --data FILE [--lora NAME.eslora]        held-out loss and perplexity
 *   es_lora gradcheck  -g MODEL.gguf [--data FILE]                      gradients vs finite differences
 *   es_lora logitcheck -g MODEL.gguf --data FILE [--lora NAME.eslora]   trainer's numbers vs the engine's
 *   es_lora export --lora NAME.eslora -g MODEL.gguf -o NAME.gguf        llama.cpp adapter (llama-cli --lora NAME.gguf)
 *   then: es_chat -g MODEL.gguf --lora NAME.eslora   (or es_serve --lora NAME.eslora)
 *
 * Data as es_graft: .txt (all text learned) or .jsonl ({"text"}, {"prompt","response"}, {"messages":[...]}: only answers learned).
 * --batch: tokens per optimizer step; --micro: tokens per pass (memory); every --val-every-th example is held out.
 * The model streams from disk layer by layer, three passes per step; memory is one layer's work plus the layer inputs of a pass.
 */
#define _GNU_SOURCE
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "../src/es_engine.h"
#include "../src/es_train.h"

static void die(const char *fmt, const char *a) { fprintf(stderr, fmt, a); fputc('\n', stderr); exit(1); }
#include "es_json.h"
#include "es_graft_anchor.h"
#include "es_data.h"

static const char *arg_of(int argc, char **argv, const char *name, const char *def) {
    for (int i = 0; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static int has_flag(int argc, char **argv, const char *name) { for (int i = 0; i < argc; i++) if (!strcmp(argv[i], name)) return 1; return 0; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static volatile sig_atomic_t STOP;
static void on_sig(int s) { (void)s; STOP = 1; }

static uint32_t parse_targets(const char *s) {
    if (!strcmp(s, "all")) return ES_LT_ALL;
    if (!strcmp(s, "attn")) return 1u << ES_LT_Q | 1u << ES_LT_K | 1u << ES_LT_V | 1u << ES_LT_O;
    if (!strcmp(s, "mlp")) return 1u << ES_LT_GATE | 1u << ES_LT_UP | 1u << ES_LT_DOWN;
    static const char *nm[ES_LT_N] = {"q", "k", "v", "o", "gate", "up", "down"};
    uint32_t m = 0; char buf[256]; snprintf(buf, sizeof buf, "%s", s);
    for (char *t = strtok(buf, ","); t; t = strtok(NULL, ",")) {
        int ok = 0;
        for (int i = 0; i < ES_LT_N; i++) if (!strcmp(t, nm[i])) { m |= 1u << i; ok = 1; }
        if (!strcmp(t, "qkvo")) { m |= 0xfu; ok = 1; }
        if (!ok) die("unknown target %s (q k v o gate up down, or all, attn, mlp)", t);
    }
    return m;
}

static es_tm *open_trainer(int argc, char **argv) {
    const char *g = arg_of(argc, argv, "-g", NULL);
    if (!g) die("%s", "need -g MODEL.gguf");
    char err[512];
    es_tm *m = es_tm_open(g, atoi(arg_of(argc, argv, "-j", "0")), err, sizeof err);
    if (!m) die("%s", err);
    return m;
}
static void open_engine(int argc, char **argv, int ctx) {   /* the engine: tokenizer, chat template, and the reference forward pass */
    es_engine_opts o = {.gguf_fd = -1, .ctx = ctx, .threads = atoi(arg_of(argc, argv, "-j", "0")), .lookahead = 1, .warm = 0, .spec = -2};
    o.gguf = arg_of(argc, argv, "-g", NULL);
    char err[512];
    if (es_engine_init(&o, err, sizeof err)) die("%s", err);
}

/* ---------------------------------------------------------------- batches */
typedef struct { const examples *X; const int *idx; int n; } exlist;
static long learned(const examples *X, int e) { return X->len[e] - 1 - (X->first[e] < 0 ? 0 : X->first[e]); }
/* the loss over a list of examples, in passes of about `micro` tokens; with backward, gradients of the mean loss (scale) */
static double pass_list(es_tm *m, const examples *X, const int *idx, int n, int micro, int backward, float gscale, long *count) {
    const int32_t **ids = malloc(sizeof(int32_t *) * (size_t)n); int *len = malloc(sizeof(int) * (size_t)n), *fst = malloc(sizeof(int) * (size_t)n);
    double loss = 0; long cnt = 0;
    for (int i = 0; i < n;) {
        int S = 0, T = 0;
        while (i < n && (S == 0 || T + X->len[idx[i]] <= micro)) { const int e = idx[i++]; ids[S] = X->pool + X->off[e]; len[S] = X->len[e]; fst[S] = X->first[e]; T += len[S]; S++; }
        long c = 0;
        loss += es_tm_run(m, S, ids, len, fst, backward, gscale, &c);
        cnt += c;
        if (STOP && !backward) break;
    }
    free(ids); free(len); free(fst);
    if (count) *count = cnt;
    return loss;
}
static double held_out(es_tm *m, const examples *X, const int *val, int nv, int micro, long *count) {
    long c = 0; const double l = pass_list(m, X, val, nv, micro, 0, 1.0f, &c);
    if (count) *count = c;
    return c ? l / c : 0;
}

/* ---------------------------------------------------------------- train */
static int cmd_train(int argc, char **argv) {
    const char *data = arg_of(argc, argv, "--data", NULL), *out = arg_of(argc, argv, "-o", NULL);
    if (!data || !out) die("%s", "usage: es_lora train -g MODEL.gguf --data FILE -o NAME.eslora [--rank 16] [--lr 2e-4] [--epochs 1] ...");
    const int seq = atoi(arg_of(argc, argv, "--seq", "512")), rank = atoi(arg_of(argc, argv, "--rank", "16"));
    const float alpha = (float)atof(arg_of(argc, argv, "--alpha", "0")), lr0 = (float)atof(arg_of(argc, argv, "--lr", "2e-4"));
    const double epochs = atof(arg_of(argc, argv, "--epochs", "1")), minutes = atof(arg_of(argc, argv, "--minutes", "0")), max_tok = atof(arg_of(argc, argv, "--tokens", "0"));
    const int batch = atoi(arg_of(argc, argv, "--batch", "2048")), micro = atoi(arg_of(argc, argv, "--micro", "1024")), val_every = atoi(arg_of(argc, argv, "--val-every", "20"));
    int steps = atoi(arg_of(argc, argv, "--steps", "0")); const int eval_every = atoi(arg_of(argc, argv, "--eval-every", "0"));
    const uint32_t mask = parse_targets(arg_of(argc, argv, "--targets", "all"));
    const float wd = (float)atof(arg_of(argc, argv, "--wd", "0")), clip = (float)atof(arg_of(argc, argv, "--clip", "1"));
    if (seq < 8 || micro < seq || batch < 1) die("%s", "need --seq >= 8 and --micro >= --seq");
    struct stat ds; if (stat(data, &ds)) die("cannot open %s", data);

    open_engine(argc, argv, seq + 16);
    examples X; collect(&X, data, "none", seq, max_tok, val_every);
    int *tr = malloc(sizeof(int) * (size_t)X.nex), *va = malloc(sizeof(int) * (size_t)X.nex), nt = 0, nv = 0; long trtok = 0;
    for (int e = 0; e < X.nex; e++) { if (X.val[e]) va[nv++] = e; else { tr[nt++] = e; trtok += learned(&X, e); } }
    if (!nt) die("%s", "no training examples in the data");
    es_tm *m = open_trainer(argc, argv);
    int L, d, V; es_tm_dims(m, &L, &d, &V);
    char err[512]; uint64_t done = 0;
    char opt[4096]; snprintf(opt, sizeof opt, "%s.opt", out);
    const int resume = has_flag(argc, argv, "--resume");
    if (resume) { if (es_tm_lora_load(m, out, &done, err, sizeof err)) die("%s", err); }
    else if (es_tm_lora_new(m, rank, alpha, mask, (uint64_t)atoll(arg_of(argc, argv, "--seed", "1")))) die("%s", "bad --rank or --targets");
    int r_; float a_; uint32_t m_; es_tm_lora_info(m, &r_, &a_, &m_);
    const size_t np = es_tm_lora_count(m);
    float *P = es_tm_lora_params(m), *Gr = es_tm_lora_grads(m), *M1 = calloc(np, sizeof(float)), *M2 = calloc(np, sizeof(float));
    if (resume) { FILE *f = fopen(opt, "rb"); if (f) { if (fread(M1, 4, np, f) != np || fread(M2, 4, np, f) != np) memset(M1, 0, np * 4), memset(M2, 0, np * 4); fclose(f); } }
    const int per_epoch = (int)((trtok + batch - 1) / batch);
    if (!steps) steps = (int)ceil(epochs * per_epoch);
    fprintf(stderr, "model %s: %d layers, width %d, vocabulary %d\n", es_tm_arch(m), L, d, V);
    fprintf(stderr, "LoRA rank %d, alpha %g, %zu trainable numbers (%.2f MB); %d training examples (%ld learned tokens), %d held out\n",
            r_, a_, np, np * 4 / 1e6, nt, trtok, nv);
    fprintf(stderr, "%d steps of about %d learned tokens (%d per epoch), lr %g, passes of %d tokens\n", steps, batch, per_epoch, lr0, micro);
    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
    const double t_start = now();
    if (nv && !resume) {
        long c; const double l0 = held_out(m, &X, va, nv, micro, &c);
        fprintf(stderr, "held-out loss before: %.4f (perplexity %.2f) over %ld tokens [%.1f s]\n", l0, exp(l0), c, now() - t_start);
    }
    /* example order: a fresh shuffle each epoch */
    uint64_t rng = 0x12345678u + done;
    int *ord = malloc(sizeof(int) * (size_t)nt), at = nt;
    double tok_done = 0, t_train = 0;
    for (int step = (int)done; step < steps && !STOP; step++) {
        if (minutes > 0 && now() - t_start > minutes * 60) { fprintf(stderr, "time limit reached\n"); break; }
        /* the examples of this step */
        int *bi = malloc(sizeof(int) * (size_t)nt), nb = 0; long bt = 0;
        while (bt < batch && nb < nt) {
            if (at >= nt) { for (int i = 0; i < nt; i++) ord[i] = tr[i]; for (int i = nt - 1; i > 0; i--) { rng = rng * 6364136223846793005ull + 1442695040888963407ull; const int j = (int)((rng >> 33) % (uint64_t)(i + 1)); const int t = ord[i]; ord[i] = ord[j]; ord[j] = t; } at = 0; }
            bi[nb++] = ord[at]; bt += learned(&X, ord[at]); at++;
        }
        memset(Gr, 0, np * sizeof(float));
        const double t0 = now();
        long c = 0; const double loss = pass_list(m, &X, bi, nb, micro, 1, 1.0f / (float)bt, &c);
        free(bi);
        /* AdamW with warm-up and cosine decay to 10%, global-norm clipping */
        double gn = 0; for (size_t i = 0; i < np; i++) gn += (double)Gr[i] * Gr[i];
        gn = sqrt(gn);
        const float cs = clip > 0 && gn > clip ? (float)(clip / gn) : 1.0f;
        const int warm = steps / 20 > 0 ? steps / 20 : 1;
        const double prog = (double)step / (steps > 1 ? steps - 1 : 1);
        const float lr = step < warm ? lr0 * (step + 1) / warm : (float)(lr0 * (0.1 + 0.9 * 0.5 * (1 + cos(M_PI * prog))));
        const float b1 = 0.9f, b2 = 0.999f, bc1 = 1 - powf(b1, step + 1), bc2 = 1 - powf(b2, step + 1);
        for (size_t i = 0; i < np; i++) {
            const float g = Gr[i] * cs;
            M1[i] = b1 * M1[i] + (1 - b1) * g; M2[i] = b2 * M2[i] + (1 - b2) * g * g;
            P[i] -= lr * ((M1[i] / bc1) / (sqrtf(M2[i] / bc2) + 1e-8f) + wd * P[i]);
        }
        const double dt = now() - t0; t_train += dt; tok_done += c;
        fprintf(stderr, "step %d/%d  loss %.4f  |grad| %.3f  lr %.2e  %.1f s  %.0f tokens/s  %.0f MB read\n",
                step + 1, steps, c ? loss / c : 0, gn, lr, dt, c / dt, es_tm_bytes_read(m) / 1e6);
        const int last = step + 1 == steps;
        if ((eval_every > 0 && (step + 1) % eval_every == 0) || last || STOP) {
            if (es_tm_lora_save(m, out, (uint64_t)step + 1, err, sizeof err)) die("%s", err);
            FILE *f = fopen(opt, "wb"); if (f) { fwrite(M1, 4, np, f); fwrite(M2, 4, np, f); fclose(f); }
            if (nv && !STOP) { long cv; const double lv = held_out(m, &X, va, nv, micro, &cv); fprintf(stderr, "held-out loss after step %d: %.4f (perplexity %.2f)\n", step + 1, lv, exp(lv)); }
        }
    }
    if (STOP) { if (es_tm_lora_save(m, out, 0, err, sizeof err) == 0) fprintf(stderr, "stopped: saved to %s (es_lora train ... --resume continues)\n", out); }
    fprintf(stderr, "trained %.0f tokens in %.0f s (%.1f tokens/s in training passes), saved %s\n", tok_done, now() - t_start, t_train > 0 ? tok_done / t_train : 0, out);
    es_tm_close(m);
    return 0;
}

/* ---------------------------------------------------------------- eval */
static int cmd_eval(int argc, char **argv) {
    const char *data = arg_of(argc, argv, "--data", NULL), *lora = arg_of(argc, argv, "--lora", NULL);
    if (!data) die("%s", "usage: es_lora eval -g MODEL.gguf --data FILE [--lora NAME.eslora]");
    const int seq = atoi(arg_of(argc, argv, "--seq", "512")), micro = atoi(arg_of(argc, argv, "--micro", "1024")), val_every = atoi(arg_of(argc, argv, "--val-every", "20"));
    open_engine(argc, argv, seq + 16);
    examples X; collect(&X, data, "none", seq, atof(arg_of(argc, argv, "--tokens", "0")), val_every);
    int *va = malloc(sizeof(int) * (size_t)X.nex), nv = 0;
    for (int e = 0; e < X.nex; e++) if (X.val[e] || val_every <= 0) va[nv++] = e;
    es_tm *m = open_trainer(argc, argv);
    char err[512];
    if (lora && es_tm_lora_load(m, lora, NULL, err, sizeof err)) die("%s", err);
    long c; const double t0 = now(), l = held_out(m, &X, va, nv, micro, &c);
    printf("held-out loss %.4f, perplexity %.3f, over %ld tokens of %d examples (%.1f s)\n", l, exp(l), c, nv, now() - t0);
    es_tm_close(m);
    return 0;
}

/* ---------------------------------------------------------------- gradient check */
static int cmd_gradcheck(int argc, char **argv) {
    es_tm *m = open_trainer(argc, argv);
    int L, d, V; es_tm_dims(m, &L, &d, &V);
    const int S = 2, n = atoi(arg_of(argc, argv, "--len", "24"));
    int32_t *ids[2]; int len[2] = {n, n - 5}, fst[2] = {0, 3};
    uint64_t s = 7;
    for (int k = 0; k < S; k++) { ids[k] = malloc(sizeof(int32_t) * (size_t)n); for (int i = 0; i < n; i++) { s = s * 6364136223846793005ull + 1; ids[k][i] = (int32_t)((s >> 33) % (uint64_t)V); } }
    if (es_tm_lora_new(m, 4, 8, ES_LT_ALL, 3)) die("%s", "lora");
    const size_t np = es_tm_lora_count(m);
    float *P = es_tm_lora_params(m), *G = es_tm_lora_grads(m);
    for (size_t i = 0; i < np; i++) { s = s * 6364136223846793005ull + 1; P[i] += 0.02f * ((float)(s >> 40) / (1 << 24) - 0.5f); }   /* B != 0: every gradient is non-zero */
    memset(G, 0, np * sizeof(float));
    long c; const double l0 = es_tm_run(m, S, (const int32_t *const *)ids, len, fst, 1, 1.0f, &c);
    printf("loss %.6f over %ld positions, %zu LoRA numbers\n", l0, c, np);
    /* directional derivatives: g . u vs (L(p + e u) - L(p - e u)) / 2e for random unit u */
    int bad = 0;
    float *u = malloc(np * sizeof(float)), *p0 = malloc(np * sizeof(float));
    memcpy(p0, P, np * sizeof(float));
    for (int trial = 0; trial < 3; trial++) {
        double nu = 0; for (size_t i = 0; i < np; i++) { s = s * 6364136223846793005ull + 1; u[i] = (float)(s >> 40) / (1 << 24) - 0.5f; nu += (double)u[i] * u[i]; }
        nu = sqrt(nu); double ga = 0; for (size_t i = 0; i < np; i++) { u[i] /= (float)nu; ga += (double)G[i] * u[i]; }
        const double e = atof(arg_of(argc, argv, "--eps", "0.3"));
        for (size_t i = 0; i < np; i++) P[i] = p0[i] + (float)e * u[i];
        const double lp = es_tm_run(m, S, (const int32_t *const *)ids, len, fst, 0, 1.0f, NULL);
        for (size_t i = 0; i < np; i++) P[i] = p0[i] - (float)e * u[i];
        const double lm = es_tm_run(m, S, (const int32_t *const *)ids, len, fst, 0, 1.0f, NULL);
        memcpy(P, p0, np * sizeof(float));
        const double fd = (lp - lm) / (2 * e), rel = fabs(fd - ga) / fmax(fabs(fd), fabs(ga));
        printf("random direction %d: analytic %+.6f  finite difference %+.6f  relative error %.2e\n", trial + 1, ga, fd, rel);
        bad += rel > 2e-2;
    }
    /* single numbers: the largest gradient of A and of B in the first and last layer and per target */
    int r; float a; uint32_t mk; es_tm_lora_info(m, &r, &a, &mk);
    const int nchk = atoi(arg_of(argc, argv, "--n", "14"));
    for (int k = 0; k < nchk; k++) {
        size_t best = 0; double bv = -1; const int want_t = k % ES_LT_N, want_b = k / ES_LT_N % 2;   /* every target, A and B */
        for (int tries = 0; tries < 200000 && bv < 0 + 1e30; tries++) {
            s = s * 6364136223846793005ull + 1; const size_t i = (size_t)(s >> 20) % np;
            int ly, tg, isb; es_tm_lora_where(m, i, &ly, &tg, &isb);
            if (tg == want_t && isb == want_b && fabs(G[i]) > bv) { bv = fabs(G[i]); best = i; }
            if (tries > 20000 && bv > 0) break;
        }
        /* the step: big enough that float rounding of the loss (about 1e-5) stays far below the change it causes */
        const double e = fmin(0.05, fmax(1e-3, 2e-3 / fabs(G[best])));
        P[best] = p0[best] + (float)e; const double lp = es_tm_run(m, S, (const int32_t *const *)ids, len, fst, 0, 1.0f, NULL);
        P[best] = p0[best] - (float)e; const double lm = es_tm_run(m, S, (const int32_t *const *)ids, len, fst, 0, 1.0f, NULL);
        P[best] = p0[best];
        const double fd = (lp - lm) / (2 * e), rel = fabs(fd - G[best]) / fmax(fabs(fd), fabs(G[best]));
        int ly, tg, isb; es_tm_lora_where(m, best, &ly, &tg, &isb);
        static const char *tn[ES_LT_N] = {"q", "k", "v", "o", "gate", "up", "down"};
        printf("layer %2d %-4s %s: analytic %+.5e  finite difference %+.5e  relative error %.2e\n", ly, tn[tg], isb ? "B" : "A", G[best], fd, rel);
        bad += rel > 5e-2;
    }
    printf(bad ? "GRADIENT CHECK FAILED (%d)\n" : "gradient check passed\n", bad);
    es_tm_close(m);
    return bad != 0;
}

/* ---------------------------------------------------------------- the trainer's forward pass vs the engine's */
typedef struct { float *lg; int V; } cap_ctx;
static void cap_cb(int pos, const float *x, const float *logits, void *ud) { (void)x; cap_ctx *c = ud; memcpy(c->lg + (size_t)pos * c->V, logits, sizeof(float) * (size_t)c->V); }
static int cmd_logitcheck(int argc, char **argv) {
    const char *data = arg_of(argc, argv, "--data", NULL), *lora = arg_of(argc, argv, "--lora", NULL);
    if (!data) die("%s", "usage: es_lora logitcheck -g MODEL.gguf --data FILE [--lora NAME.eslora]");
    const int seq = atoi(arg_of(argc, argv, "--seq", "128")), nex = atoi(arg_of(argc, argv, "--n", "4"));
    open_engine(argc, argv, seq + 16);
    char err[512];
    if (lora && es_engine_set_lora(lora, err, sizeof err)) die("%s", err);
    examples X; collect(&X, data, "none", seq, (double)seq * nex * 2, 0);
    es_tm *m = open_trainer(argc, argv);
    if (lora && es_tm_lora_load(m, lora, NULL, err, sizeof err)) die("%s", err);
    int L, d, V; es_tm_dims(m, &L, &d, &V);
    double maxd = 0, sumd = 0, le = 0, lt = 0; long cnt = 0, agree = 0, pos = 0;
    for (int e = 0; e < X.nex && e < nex; e++) {
        const int n = X.len[e]; const int32_t *ids = X.pool + X.off[e];
        float *a = malloc(sizeof(float) * (size_t)n * V), *b = malloc(sizeof(float) * (size_t)n * V);
        es_tm_logits(m, ids, n, a);
        cap_ctx cc = {b, V}; es_engine_features(ids, n, NULL, cap_cb, &cc);
        for (int p = 0; p < n; p++) {
            const float *x = a + (size_t)p * V, *y = b + (size_t)p * V;
            int ax = 0, bx = 0; double mx = -1e30, my = -1e30, sx = 0, sy = 0;
            for (int v = 0; v < V; v++) {
                const double dd = fabs(x[v] - y[v]); if (dd > maxd) maxd = dd; sumd += dd;
                if (x[v] > x[ax]) ax = v; if (y[v] > y[bx]) bx = v; if (x[v] > mx) mx = x[v]; if (y[v] > my) my = y[v];
            }
            for (int v = 0; v < V; v++) { sx += exp(x[v] - mx); sy += exp(y[v] - my); }
            agree += ax == bx; pos++;
            if (p + 1 < n) { const int t = ids[p + 1]; lt += mx + log(sx) - x[t]; le += my + log(sy) - y[t]; cnt++; }
        }
        free(a); free(b);
    }
    printf("%ld positions: largest logit difference %.4f, mean %.5f; same top token at %.1f%%\n", pos, maxd, sumd / ((double)pos * V), 100.0 * agree / pos);
    printf("perplexity: trainer (float) %.4f, engine (8-bit activations) %.4f\n", exp(lt / cnt), exp(le / cnt));
    es_tm_close(m);
    return 0;
}

static int cmd_export(int argc, char **argv) {
    const char *lora = arg_of(argc, argv, "--lora", NULL), *out = arg_of(argc, argv, "-o", NULL);
    if (!lora || !out) die("%s", "usage: es_lora export --lora NAME.eslora -g MODEL.gguf -o NAME.gguf");
    es_tm *m = open_trainer(argc, argv);
    char err[512];
    if (es_tm_lora_load(m, lora, NULL, err, sizeof err) || es_tm_lora_export_gguf(m, out, err, sizeof err)) die("%s", err);
    printf("wrote %s: llama-cli -m MODEL.gguf --lora %s\n", out, out);
    es_tm_close(m);
    return 0;
}

#ifdef ES_TRAIN_UNIFIED
int es_lora_main(int argc, char **argv) {
#else
int main(int argc, char **argv) {
#endif
    if (argc < 2) { fprintf(stderr, "usage: es_lora train|eval|gradcheck|logitcheck|export ... (see the top of tools/es_lora.c)\n"); return 1; }
    const char *c = argv[1];
    if (!strcmp(c, "train")) return cmd_train(argc, argv);
    if (!strcmp(c, "eval")) return cmd_eval(argc, argv);
    if (!strcmp(c, "gradcheck")) return cmd_gradcheck(argc, argv);
    if (!strcmp(c, "logitcheck")) return cmd_logitcheck(argc, argv);
    if (!strcmp(c, "export")) return cmd_export(argc, argv);
    fprintf(stderr, "unknown command %s\n", c);
    return 1;
}
