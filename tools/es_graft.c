/*
 * es_graft: teach a frozen model new things on a phone (src/es_graft.h explains the graft itself).
 *
 * The base model can be anything the engine runs, up to the 200B+ mixture-of-experts models. Its weights never change, so
 * there is no backward pass through it, no optimizer state for its billions of parameters and no GPU needed:
 *
 *   1. read   your text goes through the frozen model ONCE (at the engine's prompt-reading speed). For every position the
 *             final hidden state, the model's top-K candidates and their scores are saved in a feature file.
 *   2. train  the graft learns from the feature file only. The big model is not touched again, so a training step costs the
 *             same for a 1B and a 235B model: thousands of tokens per second on phone cores, for as many epochs as you like.
 *   3. use    es_chat -A my.esgraft / es_serve --graft my.esgraft run the frozen model with the graft on top.
 *
 *   es_graft read  -g MODEL.gguf --data notes.txt|chat.jsonl -o my.esfeat [--seq 512] [--k 32] [--tokens N] [--minutes M] [-j T]
 *   es_graft train my.esfeat -o my.esgraft [--epochs 3] [--minutes M] [--m 64] [--experts N] [--f 256] [-j T] [--resume]
 *   es_graft eval  my.esfeat my.esgraft          held-out loss of the base model and of the grafted model (exact)
 *   es_graft check -g MODEL.gguf my.esgraft --data text.txt [--tokens 200]   the engine applies the graft as training assumed
 *   es_graft selftest                            gradients against finite differences
 *
 * Data: .txt is plain text (every token is learned). .jsonl has one example per line: {"text": ...}, or
 * {"messages": [{"role": "user", "content": ...}, ..., {"role": "assistant", "content": ...}]}, or {"prompt": ..., "response": ...};
 * chat examples are written with the model's own chat template and only the answer is learned.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "../src/es_engine.h"
#include "../src/es_graft.h"
#include "../src/es_q2k.h"
#include "../src/es_quant.h"
#include "es_json.h"
#include "es_graft_anchor.h"

/* ------------------------------------------------------------------ feature file */
#define FEAT_OFF 256
typedef struct {
    char magic[8];
    uint32_t version, d, K, n_vocab, seq, val_every, flags, kc;   /* kc: candidates per position (top K + memory suggestions) */
    uint64_t fingerprint, n_rec, n_seq, n_tok, data_size, data_mtime, skipped, pad2;
    char arch[32], name[64];
} feat_hdr;
_Static_assert(sizeof(feat_hdr) <= FEAT_OFF, "feature header");
typedef struct { float logr, by; int32_t y, yin, seq, ctx[3]; } rec_head;   /* then int32 ids[kc] (-1 = none), float bl[kc], uint32 tot[orders], uint16 cnt[kc][orders] (memory counts), fp16 x[d]; yin -2 = anchor */
_Static_assert(sizeof(rec_head) == 32, "record head");
static size_t rec_size(int d, int K) { return sizeof(rec_head) + (size_t)8 * K + 4 * ES_MEM_ORDERS + (size_t)2 * ES_MEM_ORDERS * K + (size_t)2 * d; }

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static volatile sig_atomic_t STOP;
static void on_sig(int s) { (void)s; STOP = 1; }
static void die(const char *fmt, const char *a) { fprintf(stderr, fmt, a); fputc('\n', stderr); exit(1); }

static const char *arg_of(int argc, char **argv, const char *name, const char *def) {
    for (int i = 0; i < argc - 1; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static int has_flag(int argc, char **argv, const char *name) { for (int i = 0; i < argc; i++) if (!strcmp(argv[i], name)) return 1; return 0; }
/* the first argument that is not an option or an option's value */
static const char *positional(int argc, char **argv, int which) {
    static const char *with_value[] = {"-g", "--data", "-o", "--seq", "--k", "--tokens", "--minutes", "-j", "--epochs", "--experts", "--m", "--f",
                                       "--lr", "--elr", "--mlr", "--batch", "--val-every", "--seed", "-c", "-C", "--anchor", "--anchor-weight",
                                       "--sample", NULL};
    int k = 0;
    for (int i = 2; i < argc; i++) {
        int opt = argv[i][0] == '-' && argv[i][1];
        if (opt) { for (int j = 0; with_value[j]; j++) if (!strcmp(argv[i], with_value[j])) { i++; break; } continue; }
        if (k++ == which) return argv[i];
    }
    return NULL;
}

/* ------------------------------------------------------------------ 1. read */
typedef struct {
    FILE *fp; int d, K, kc, nv; int32_t seq_id; int train, anchor; const int32_t *ids; int n; uint8_t *buf;
    uint64_t written; const es_mem *mem;
} read_ctx;

/* the graft's context at pos, exactly as the engine computes it when it runs the graft */
static void ctx_of(const int32_t *ids, int pos, int32_t *ctx) {
    const int a = pos - 31 > 0 ? pos - 31 : 0;
    es_engine_canon_ctx(ids + a, pos - a + 1, ctx);
}

static void on_position(int pos, const float *x, const float *lg, void *ud) {
    read_ctx *c = ud;
    const int kc = c->kc, d = c->d;
    uint8_t *r = c->buf;
    rec_head *h = (rec_head *)r;
    int32_t *ids = (int32_t *)(r + sizeof(rec_head));
    float *bl = (float *)(ids + kc);
    uint32_t *tot = (uint32_t *)(bl + kc);
    uint16_t *cnt = (uint16_t *)(tot + ES_MEM_ORDERS), *xs = cnt + (size_t)kc * ES_MEM_ORDERS;
    const int32_t y = c->ids[pos + 1];
    int32_t ctx[ES_GRAFT_CTX]; ctx_of(c->ids, pos, ctx);
    /* the candidates: the base model's top K and what the memory suggests here (a training position leaves its own answer out) */
    const int k = es_graft_candidates(c->K, c->mem, lg, c->nv, ctx, c->train && !c->anchor ? y : -1, ids);
    h->yin = -1;
    double mx = lg[ids[0]];
    for (int i = 0; i < k; i++) if (lg[ids[i]] > mx) mx = lg[ids[i]];
    double all = 0, in = 0;
    for (int i = 0; i < c->nv; i++) all += exp((double)lg[i] - mx);
    for (int i = 0; i < kc; i++) {
        if (i < k) { bl[i] = lg[ids[i]]; in += exp((double)bl[i] - mx); if (ids[i] == y) h->yin = i; }
        else { ids[i] = -1; bl[i] = -INFINITY; }
    }
    es_mem_counts(c->mem, ctx, c->train && !c->anchor ? y : -1, ids, kc, cnt, tot);   /* padding ids are -1: no counts */
    const double rest = all - in;   /* log of the sum of exp(score) over every token that is not a candidate */
    h->logr = rest > all * 1e-12 ? (float)(mx + log(rest)) : -INFINITY;
    h->y = y; h->by = lg[y]; h->seq = c->train ? c->seq_id : -1 - c->seq_id;   /* held-out examples: negative */
    if (c->anchor) { h->yin = -2; h->y = -1; h->by = 0; }   /* the target is the base model itself */
    for (int i = 0; i < 3; i++) h->ctx[i] = ctx[i];
    for (int i = 0; i < d; i++) xs[i] = es_fp32_to_fp16(x[i]);
    if (fwrite(r, rec_size(d, kc), 1, c->fp) != 1) { fprintf(stderr, "cannot write the feature file\n"); exit(1); }
    c->written++;
}

/* ---- the memory: contexts of the training examples and what followed them ---- */
typedef struct { es_mem_ent *e; uint64_t cap, n; } memb;
static void memb_add(memb *b, uint64_t key, int32_t tok) {
    if (!key) return;
    if (b->n * 10 >= b->cap * 7) {   /* grow */
        memb nb = {calloc(b->cap ? b->cap * 2 : 1 << 16, sizeof(es_mem_ent)), b->cap ? b->cap * 2 : 1 << 16, 0};
        if (!nb.e) { fprintf(stderr, "out of memory (memory candidates)\n"); exit(1); }
        for (uint64_t i = 0; i < b->cap; i++) if (b->e[i].key) {
            uint64_t j = b->e[i].key & (nb.cap - 1);
            while (nb.e[j].key) j = (j + 1) & (nb.cap - 1);
            nb.e[j] = b->e[i]; nb.n++;
        }
        free(b->e); *b = nb;
    }
    uint64_t j = key & (b->cap - 1);
    while (b->e[j].key && b->e[j].key != key) j = (j + 1) & (b->cap - 1);
    es_mem_ent *e = &b->e[j];
    if (!e->key) { e->key = key; for (int i = 0; i < ES_MEM_KEEP; i++) { e->tok[i] = -1; e->cnt[i] = 0; } e->total = 0; b->n++; }
    e->total++;
    int lo = 0;
    for (int i = 0; i < ES_MEM_KEEP; i++) {
        if (e->tok[i] == tok) { e->cnt[i]++; return; }
        if (e->cnt[i] < e->cnt[lo]) lo = i;
    }
    e->tok[lo] = tok; e->cnt[lo]++;   /* empty slot (count 0), or the rarest one is replaced (space-saving count) */
}
static int ent_cmp(const void *a, const void *b) { const uint64_t x = ((const es_mem_ent *)a)->key, y = ((const es_mem_ent *)b)->key; return x < y ? -1 : x > y; }
static void memb_save(memb *b, const char *path) {
    uint64_t n = 0;
    for (uint64_t i = 0; i < b->cap; i++) if (b->e[i].key) b->e[n++] = b->e[i];
    qsort(b->e, n, sizeof(es_mem_ent), ent_cmp);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite("ESMEM003", 1, 8, f) != 8 || fwrite(&n, 8, 1, f) != 1 || (n && fwrite(b->e, sizeof(es_mem_ent), n, f) != n) || fclose(f)) die("cannot write %s", path);
    free(b->e); memset(b, 0, sizeof *b);
}

typedef struct {   /* where the examples come from */
    int jsonl; char *text; size_t len, at; int bos, L;
    int32_t *stream; int sn, sat, scap;
} source;

static char *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) die("cannot read %s", path);
    fclose(f); b[n] = 0; *len = (size_t)n;
    return b;
}
/* next example: its ids, and the first position whose NEXT token is learned (want[p] for p >= first) */
static int next_example(source *s, int32_t *ids, int cap, int *first, long *skipped) {
    if (!s->jsonl) {   /* plain text: a stream of tokens cut into windows of L tokens (BOS first when the model wants one) */
        while (s->sn - s->sat < s->L && s->at < s->len) {
            size_t e = s->at + 16384 < s->len ? s->at + 16384 : s->len;   /* tokenize piece by piece, cut at a line end */
            if (e < s->len) { size_t b = e; while (b > s->at + 1024 && s->text[b - 1] != '\n') b--; if (s->text[b - 1] == '\n') e = b; }
            const char save = s->text[e]; s->text[e] = 0;
            if (s->sat > 0) { memmove(s->stream, s->stream + s->sat, sizeof(int32_t) * (size_t)(s->sn - s->sat)); s->sn -= s->sat; s->sat = 0; }
            if (s->sn + (int)(e - s->at) + 16 > s->scap) { s->scap = s->sn + (int)(e - s->at) + 1024; s->stream = realloc(s->stream, sizeof(int32_t) * (size_t)s->scap); }
            const int m = es_engine_tokenize(s->text + s->at, s->stream + s->sn, s->scap - s->sn);
            s->text[e] = save;
            if (m > 0) s->sn += m;
            s->at = e;
        }
        const int avail = s->sn - s->sat;
        if (avail < 2) return 0;
        int n = 0;
        if (s->bos >= 0) ids[n++] = s->bos;
        const int take = avail < s->L ? avail : s->L;
        memcpy(ids + n, s->stream + s->sat, sizeof(int32_t) * (size_t)take);
        s->sat += take; n += take;
        *first = 0;
        return n;
    }
    while (s->at < s->len) {   /* JSON lines */
        char *line = s->text + s->at, *nl = memchr(line, '\n', s->len - s->at);
        const size_t ll = nl ? (size_t)(nl - line) : s->len - s->at;
        s->at += ll + (nl ? 1 : 0);
        if (ll < 2) continue;
        jparser P; jv *v = jparse(&P, line, ll);
        int n = -1;
        if (v && v->t == JOBJ) {
            const char *text = jstr(v, "text", NULL), *pr = jstr(v, "prompt", NULL), *rs = jstr(v, "response", jstr(v, "completion", NULL));
            const jv *msgs = jget(v, "messages");
            if (text) {
                n = 0;
                if (s->bos >= 0) ids[n++] = s->bos;
                const int m = es_engine_tokenize(text, ids + n, cap - n);
                n = m < 0 ? -1 : n + m;
                *first = 0;
            } else if (msgs && msgs->t == JARR && msgs->len >= 2) {
                es_msg mm[64]; int k = 0;
                for (const jv *c = msgs->kid; c && k < 64; c = c->next) { mm[k].role = jstr(c, "role", "user"); mm[k].content = jstr(c, "content", ""); k++; }
                if (k >= 2 && !strcmp(mm[k - 1].role, "assistant")) { int st; n = es_engine_chat_example(mm, k - 1, mm[k - 1].content, ids, cap, &st); *first = st - 1; }
            } else if (pr && rs) {
                es_msg mm[1] = {{"user", pr}}; int st;
                n = es_engine_chat_example(mm, 1, rs, ids, cap, &st); *first = st - 1;
            }
        }
        jfree(&P);
        if (n >= 2 && n <= cap) { if (*first < 0) *first = 0; return n; }
        (*skipped)++;
    }
    return 0;
}

static void write_rows(const char *feat, int d, int K, int nv) {   /* K: candidates per record */
    /* every token that was a candidate somewhere: its output row, as stored in the model (training needs these and nothing else of it) */
    int fd = open(feat, O_RDONLY);
    struct stat st; fstat(fd, &st);
    const size_t rs = rec_size(d, K), n = ((size_t)st.st_size - FEAT_OFF) / rs;
    uint8_t *used = calloc((size_t)nv, 1);
    uint8_t *buf = malloc(rs * 1024);
    for (size_t i = 0; i < n; i += 1024) {
        const size_t c = n - i < 1024 ? n - i : 1024;
        if (pread(fd, buf, rs * c, (off_t)(FEAT_OFF + rs * i)) != (ssize_t)(rs * c)) break;
        for (size_t j = 0; j < c; j++) { const int32_t *ids = (const int32_t *)(buf + rs * j + sizeof(rec_head)); for (int k = 0; k < K; k++) if (ids[k] >= 0 && ids[k] < nv) used[ids[k]] = 1;   /* -1: no candidate */ }
    }
    close(fd);
    uint32_t type; const uint8_t *data; size_t rb;
    es_engine_output_raw(&type, &data, &rb);
    char path[4096]; snprintf(path, sizeof path, "%s.rows", feat);
    FILE *f = fopen(path, "wb");
    if (!f) die("cannot write %s", path);
    uint32_t cnt = 0; for (int i = 0; i < nv; i++) cnt += used[i];
    const uint32_t h[4] = {type, (uint32_t)d, cnt, (uint32_t)rb};
    fwrite("ESROWS01", 1, 8, f); fwrite(h, sizeof h, 1, f);
    for (int i = 0; i < nv; i++) if (used[i]) { const int32_t id = i; fwrite(&id, 4, 1, f); }
    for (int i = 0; i < nv; i++) if (used[i]) fwrite(data + rb * (size_t)i, 1, rb, f);
    if (fclose(f)) die("cannot write %s", path);
    fprintf(stderr, "  output rows of %u candidate tokens saved (%.1f MB): training needs nothing else from the model\n", cnt, cnt * (double)rb / 1e6);
    free(used); free(buf);
}

static int init_engine(int argc, char **argv, int ctx) {
    es_engine_opts o = {.gguf_fd = -1, .ctx = ctx, .cache_mb = (size_t)atol(arg_of(argc, argv, "-C", "1024")), .threads = atoi(arg_of(argc, argv, "-j", "0")),
                        .lookahead = 1, .warm = 1, .spec = -2};
    o.gguf = arg_of(argc, argv, "-g", NULL);
    if (!o.gguf) die("%s", "need -g MODEL.gguf");
    char err[512];
    if (es_engine_init(&o, err, sizeof err)) die("%s", err);
    es_engine_info in; es_engine_get_info(&in);
    fprintf(stderr, "model %s (%s), %d layers%s | RAM %ld MB\n", in.name, in.arch, in.layers, in.experts ? ", mixture of experts" : "", es_rss_mb());
    if (in.remote) fprintf(stderr, "  %s\n", in.remote);
    return 0;
}

/* every example of the data (then the anchors) as tokens; the same cut every run, so an interrupted read resumes where it stopped */
typedef struct {
    int32_t *pool; size_t pn, pcap; size_t *off; int *len, *first, nex, excap; uint8_t *anc, *val;
    long skipped; double learn, anchor_tok, forward_tok; int ndata, nanc;
} examples;
static void collect(examples *X, const char *data, const char *anchor, int seq, double max_tok, int val_every) {
    memset(X, 0, sizeof *X);
    source src = {0};
    const size_t dl = strlen(data);
    src.jsonl = dl > 6 && !strcmp(data + dl - 6, ".jsonl");
    src.text = slurp(data, &src.len); src.bos = es_engine_bos(); src.L = seq - (src.bos >= 0 ? 1 : 0);
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)(seq + 16));
    source asrc = {0};
    if (strcmp(anchor, "none")) {   /* anchors: general questions and text, read through the model too; there the graft learns to keep the model's own answers */
        if (!strcmp(anchor, "auto")) { asrc.text = strdup(ES_GRAFT_ANCHOR); asrc.len = strlen(asrc.text); asrc.jsonl = 1; }
        else { asrc.text = slurp(anchor, &asrc.len); const size_t al = strlen(anchor); asrc.jsonl = al > 6 && !strcmp(anchor + al - 6, ".jsonl"); }
        asrc.bos = src.bos; asrc.L = src.L;
    }
    for (int pass = 0; pass < 2; pass++) {
        source *sr = pass ? &asrc : &src;
        if (!sr->text) continue;
        for (;;) {
            if (pass == 0 && max_tok > 0 && X->learn >= max_tok) break;
            int fst; long sk = 0; const int n = next_example(sr, ids, seq, &fst, &sk);
            if (pass == 0) X->skipped += sk;
            if (n <= 0) break;
            if (X->pn + (size_t)n > X->pcap) { X->pcap = (X->pn + (size_t)n) * 2; X->pool = realloc(X->pool, sizeof(int32_t) * X->pcap); }
            if (X->nex == X->excap) {
                X->excap = X->excap ? X->excap * 2 : 1024;
                X->off = realloc(X->off, sizeof(size_t) * X->excap); X->len = realloc(X->len, sizeof(int) * X->excap); X->first = realloc(X->first, sizeof(int) * X->excap);
                X->anc = realloc(X->anc, (size_t)X->excap); X->val = realloc(X->val, (size_t)X->excap);
            }
            if (!X->pool || !X->off || !X->len || !X->first || !X->anc || !X->val) die("%s", "out of memory");
            memcpy(X->pool + X->pn, ids, sizeof(int32_t) * (size_t)n);
            const int k = pass ? X->nanc++ : X->ndata++;   /* every val_every-th data example, and every val_every-th anchor, is held out */
            const int e = X->nex++;
            X->off[e] = X->pn; X->len[e] = n; X->first[e] = fst; X->anc[e] = (uint8_t)pass; X->val[e] = val_every > 0 && k % val_every == 0; X->pn += (size_t)n;
            if (pass) X->anchor_tok += n - 1 - fst; else X->learn += n - 1 - fst;
            X->forward_tok += n;
        }
        free(sr->text); free(sr->stream);
    }
    free(ids);
}

static int cmd_read(int argc, char **argv) {
    const char *data = arg_of(argc, argv, "--data", NULL), *out = arg_of(argc, argv, "-o", NULL);
    if (!data || !out) die("%s", "usage: es_graft read -g MODEL.gguf --data FILE.txt|FILE.jsonl -o NAME.esfeat [--seq 512] [--k 32] [--tokens N] [--minutes M]");
    const int seq = atoi(arg_of(argc, argv, "--seq", "512")), K = atoi(arg_of(argc, argv, "--k", "32"));
    const double max_tok = atof(arg_of(argc, argv, "--tokens", "0")), minutes = atof(arg_of(argc, argv, "--minutes", "0"));
    const int val_every = atoi(arg_of(argc, argv, "--val-every", "20")), use_mem = !has_flag(argc, argv, "--no-memory");
    const int kc = K + (use_mem ? ES_MEM_ORDERS * ES_MEM_PER : 0);
    if (seq < 16 || K < 2 || K > 256) die("%s", "--seq must be >= 16 and --k between 2 and 256");
    struct stat ds; if (stat(data, &ds)) die("cannot open %s", data);
    init_engine(argc, argv, seq + 16);
    int d, nv; es_engine_dims(&d, &nv);

    examples X;
    collect(&X, data, arg_of(argc, argv, "--anchor", "auto"), seq, max_tok, val_every);
    int32_t *pool = X.pool; size_t *off = X.off; int *len = X.len, *first = X.first, nex = X.nex, ndata = X.ndata, nanc = X.nanc;
    uint8_t *anc = X.anc, *val = X.val; const long skipped = X.skipped; const double learn = X.learn, anchor_tok = X.anchor_tok;
    if (!ndata) die("%s", "no examples in the data");
    if (skipped) fprintf(stderr, "  %ld examples skipped (unreadable, longer than --seq, or no assistant answer)\n", skipped);
    if (nanc) fprintf(stderr, "anchors: %d general examples (%.0f tokens) keep the model's general answers unchanged (--anchor none turns this off)\n", nanc, anchor_tok);

    /* 2. the memory: what followed each context in the TRAINING examples (held-out examples stay out of it) */
    char mpath[4096]; snprintf(mpath, sizeof mpath, "%s.mem", out);
    es_mem mem = {0};
    if (use_mem) {
        memb mb = {0};
        for (int e = 0; e < nex; e++) {
            if (val[e] || anc[e]) continue;   /* only what the graft should learn, and never the held-out part */
            const int32_t *t = pool + off[e];
            for (int p = first[e]; p < len[e] - 1; p++) {
                int32_t ctx[ES_GRAFT_CTX]; ctx_of(t, p, ctx);
                for (int o = 0; o < ES_MEM_ORDERS; o++) memb_add(&mb, es_mem_key(ES_MEM_ORDER[o], ctx), t[p + 1]);
            }
        }
        const uint64_t nctx = mb.n;
        memb_save(&mb, mpath);
        char err[256];
        if (es_mem_load(&mem, mpath, err, sizeof err)) die("%s", err);
        fprintf(stderr, "memory: %llu contexts from the training examples (%.1f MB)\n", (unsigned long long)nctx, nctx * sizeof(es_mem_ent) / 1e6);
    } else remove(mpath);

    /* 3. the frozen model reads every example once */
    feat_hdr H; memset(&H, 0, sizeof H);
    FILE *fp = fopen(out, "r+b");
    if (fp) {
        if (fread(&H, sizeof H, 1, fp) != 1 || memcmp(H.magic, "ESFEAT02", 8)) die("%s exists and is not a feature file", out);
        if (H.fingerprint != es_engine_fingerprint() || (int)H.K != K || (int)H.kc != kc || (int)H.seq != seq || H.data_size != (uint64_t)ds.st_size || H.data_mtime != (uint64_t)ds.st_mtime ||
            H.flags != (uint32_t)((use_mem ? 1 : 0) | (nanc ? 2 : 0)))
            die("%s was made from another model, data or settings: pick a new name", out);
        fprintf(stderr, "resuming: %llu examples (%llu positions) already read\n", (unsigned long long)H.n_seq, (unsigned long long)H.n_rec);
        if (ftruncate(fileno(fp), (off_t)(FEAT_OFF + rec_size(d, kc) * H.n_rec))) die("cannot resize %s", out);
        fseeko(fp, 0, SEEK_END);
    } else {
        fp = fopen(out, "w+b");
        if (!fp) die("cannot create %s", out);
        memcpy(H.magic, "ESFEAT02", 8);
        H.version = 2; H.d = (uint32_t)d; H.K = (uint32_t)K; H.kc = (uint32_t)kc; H.n_vocab = (uint32_t)nv; H.seq = (uint32_t)seq; H.val_every = (uint32_t)val_every;
        H.flags = (uint32_t)((use_mem ? 1 : 0) | (nanc ? 2 : 0));
        H.fingerprint = es_engine_fingerprint(); H.data_size = (uint64_t)ds.st_size; H.data_mtime = (uint64_t)ds.st_mtime;
        snprintf(H.arch, sizeof H.arch, "%s", es_engine_arch());
        es_engine_info in; es_engine_get_info(&in); snprintf(H.name, sizeof H.name, "%s", in.name);
        uint8_t z[FEAT_OFF] = {0}; memcpy(z, &H, sizeof H);
        if (fwrite(z, FEAT_OFF, 1, fp) != 1) die("cannot write %s", out);
    }
    H.skipped = (uint64_t)skipped;
    uint8_t *want = malloc((size_t)seq + 16);
    read_ctx c = {.fp = fp, .d = d, .K = K, .kc = kc, .nv = nv, .buf = malloc(rec_size(d, kc)), .mem = &mem};
    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
    const double t0 = now_s();
    double fwd = 0, fwd_left = 0;   /* tokens through the model: done this run, and still to do */
    for (int e = (int)H.n_seq; e < nex; e++) fwd_left += len[e];
    fprintf(stderr, "reading %d examples (%.0f tokens to learn from, %.0f anchor tokens) through the frozen model; every %d-th example is held out for checking\n",
            ndata, learn, anchor_tok, val_every);
    for (int e = (int)H.n_seq; e < nex; e++) {
        if (STOP || (minutes > 0 && now_s() - t0 > minutes * 60)) break;
        const int n = len[e];
        memset(want, 0, (size_t)n);
        int nw = 0;
        for (int p = first[e]; p < n - 1; p++) { want[p] = 1; nw++; }
        c.seq_id = e; c.train = !val[e]; c.anchor = anc[e]; c.ids = pool + off[e]; c.n = n; c.written = 0;
        if (nw && es_engine_features(c.ids, n, want, on_position, &c)) die("%s", "an example does not fit the context");
        fflush(fp);
        H.n_rec += c.written; H.n_seq = (uint64_t)e + 1; H.n_tok += (uint64_t)nw;
        if (pwrite(fileno(fp), &H, sizeof H, 0) != (ssize_t)sizeof H) die("cannot update %s", out);
        fwd += n; fwd_left -= n;
        const double el = now_s() - t0, rate = fwd / (el > 0 ? el : 1);
        fprintf(stderr, "\r  %llu/%d examples | the model reads %.2f tokens/s | about %.0f min left ", (unsigned long long)H.n_seq, nex, rate,
                rate > 0 ? fwd_left / rate / 60 : 0);
    }
    fprintf(stderr, "\n");
    fclose(fp);
    if (H.n_rec == 0) die("%s", "no training positions: is the data empty?");
    write_rows(out, d, kc, nv);
    if ((int)H.n_seq < nex) fprintf(stderr, "stopped early: run the same command again to continue\n");
    fprintf(stderr, "done: %s holds %llu positions (%.1f MB). Next: es_graft train %s -o NAME.esgraft\n", out, (unsigned long long)H.n_rec,
            (FEAT_OFF + rec_size(d, kc) * (double)H.n_rec) / 1e6, out);
    return 0;
}

/* ------------------------------------------------------------------ 2. train */
static float *ZERO_ROW;   /* the row of an empty candidate slot */
typedef struct {
    feat_hdr H; const uint8_t *map; size_t map_len, rs;
    int *row_of; float *rows;   /* output rows of the candidate tokens, fp32 */
} feats;

static void feats_open(feats *F, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("cannot open %s", path);
    struct stat st; fstat(fd, &st);
    if ((size_t)st.st_size < FEAT_OFF || pread(fd, &F->H, sizeof F->H, 0) != (ssize_t)sizeof F->H || memcmp(F->H.magic, "ESFEAT02", 8)) die("%s is not a feature file", path);
    F->rs = rec_size((int)F->H.d, (int)F->H.kc);
    if (FEAT_OFF + F->rs * F->H.n_rec > (size_t)st.st_size) die("%s is cut short", path);
    F->map_len = (size_t)st.st_size;
    F->map = mmap(NULL, F->map_len, PROT_READ, MAP_SHARED, fd, 0);
    if (F->map == MAP_FAILED) die("cannot map %s", path);
    close(fd);
    char rp[4096]; snprintf(rp, sizeof rp, "%s.rows", path);
    FILE *f = fopen(rp, "rb");
    if (!f) die("cannot open %s (run es_graft read to the end once)", rp);
    char mg[8]; uint32_t h[4];
    if (fread(mg, 8, 1, f) != 1 || memcmp(mg, "ESROWS01", 8) || fread(h, sizeof h, 1, f) != 1 || h[1] != F->H.d) die("%s is not a rows file", rp);
    const uint32_t cnt = h[2], rb = h[3];
    int32_t *ids = malloc(sizeof(int32_t) * cnt); uint8_t *raw = malloc(rb);
    F->rows = malloc(sizeof(float) * (size_t)cnt * F->H.d);
    F->row_of = malloc(sizeof(int) * F->H.n_vocab);
    ZERO_ROW = calloc(F->H.d, sizeof(float));
    if (!ids || !raw || !F->rows || !F->row_of) die("%s", "out of memory for the output rows");
    for (uint32_t i = 0; i < F->H.n_vocab; i++) F->row_of[i] = -1;
    if (fread(ids, 4, cnt, f) != cnt) die("%s is cut short", rp);
    for (uint32_t i = 0; i < cnt; i++) {
        if (fread(raw, 1, rb, f) != rb) die("%s is cut short", rp);
        es_dequant_row(h[0], raw, F->rows + (size_t)i * F->H.d, (int)F->H.d);
        if (ids[i] >= 0 && (uint32_t)ids[i] < F->H.n_vocab) F->row_of[ids[i]] = (int)i;
    }
    fclose(f); free(ids); free(raw);
}
static inline const rec_head *rec_at(const feats *F, size_t i) { return (const rec_head *)(F->map + FEAT_OFF + F->rs * i); }
static int is_val(const feats *F, const rec_head *r) { (void)F; return r->seq < 0; }

/* one position: forward, exact loss, and (when grad) the backward pass */
typedef struct {
    es_graft_tmp t; float *x, *gd, *grad, gu[4096], ga[4096];
    float *ek, *ev, *tek, *tev; int64_t *eslot; int ne;   /* expert gradients of this batch */
    double loss, base; long n; double kl; long na;   /* data positions; anchor positions (KL from the base model) */
    const float *W[ES_GRAFT_MAXK];
} worker;

static double ANCHOR_W = 1.0;
static double position(const es_graft *g, const feats *F, size_t i, worker *w, int grad, double *base) {
    const rec_head *r = rec_at(F, i);
    const int K = (int)F->H.kc, d = (int)F->H.d;
    const int32_t *ids = (const int32_t *)(r + 1);
    const float *bl = (const float *)(ids + K);
    const uint32_t *tot = (const uint32_t *)(bl + K);
    const uint16_t *cnt = (const uint16_t *)(tot + ES_MEM_ORDERS), *xs = cnt + (size_t)K * ES_MEM_ORDERS;
    for (int k = 0; k < d; k++) w->x[k] = es_fp16_to_fp32(xs[k]);
    es_graft_forward(g, w->x, r->ctx, &w->t);
    for (int k = 0; k < K; k++) w->W[k] = ids[k] >= 0 ? F->rows + (size_t)F->row_of[ids[k]] * d : ZERO_ROW;
    if (base) *base = es_graft_loss(K, d, bl, r->logr, r->yin, r->by, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    float gmb[ES_MEM_BONUS * ES_MEM_ORDERS], gmg[ES_MEM_ORDERS];
    const double L = es_graft_loss(K, d, bl, r->logr, r->yin, r->by, w->W, w->t.delta, cnt, tot, g->mb, w->t.mg, grad ? w->gd : NULL, grad ? gmb : NULL, grad ? gmg : NULL);
    if (grad) {
        if (r->yin == -2 && ANCHOR_W != 1.0) {
            for (int k = 0; k < d; k++) w->gd[k] *= (float)ANCHOR_W;
            for (int k = 0; k < ES_MEM_BONUS * ES_MEM_ORDERS; k++) gmb[k] *= (float)ANCHOR_W;
            for (int k = 0; k < ES_MEM_ORDERS; k++) gmg[k] *= (float)ANCHOR_W;
        }
        es_graft_backward(g, &w->t, w->gd, gmg, w->grad, w->gu, w->ga, w->tek, w->tev);
        float *gm = w->grad + (g->mb - g->P);
        for (int k = 0; k < ES_MEM_BONUS * ES_MEM_ORDERS; k++) gm[k] += gmb[k];
        for (int n = 0; n < ES_GRAFT_ORDERS; n++) if (w->t.e[n] >= 0) {   /* this batch's expert gradients, applied after the batch */
            memcpy(w->ek + (size_t)w->ne * g->m, w->tek + (size_t)n * g->m, sizeof(float) * g->m);
            memcpy(w->ev + (size_t)w->ne * g->m, w->tev + (size_t)n * g->m, sizeof(float) * g->m);
            w->eslot[w->ne++] = w->t.e[n];
        }
    }
    return L;
}

typedef struct {
    pthread_t th; int id; struct pool *P; worker w;
} slot;
typedef struct pool {
    int T; slot *S; pthread_mutex_t mu; pthread_cond_t go, done; int gen, left, quit;
    const es_graft *g; const feats *F; const size_t *idx; size_t n; int grad;
} pool;

static void *pool_main(void *arg) {
    slot *s = arg; pool *P = s->P; int seen = 0;
    for (;;) {
        pthread_mutex_lock(&P->mu);
        while (P->gen == seen && !P->quit) pthread_cond_wait(&P->go, &P->mu);
        if (P->quit) { pthread_mutex_unlock(&P->mu); return NULL; }
        seen = P->gen;
        pthread_mutex_unlock(&P->mu);
        worker *w = &s->w;
        const size_t a = P->n * (size_t)s->id / P->T, b = P->n * (size_t)(s->id + 1) / P->T;
        for (size_t i = a; i < b; i++) {
            double base;
            const double L = position(P->g, P->F, P->idx[i], w, P->grad, &base);
            if (rec_at(P->F, P->idx[i])->yin == -2) { w->kl += L; w->na++; }
            else { w->loss += L; w->base += base; w->n++; }
        }
        pthread_mutex_lock(&P->mu);
        if (--P->left == 0) pthread_cond_signal(&P->done);
        pthread_mutex_unlock(&P->mu);
    }
}
static void pool_run(pool *P, const size_t *idx, size_t n, int grad) {
    pthread_mutex_lock(&P->mu);
    P->idx = idx; P->n = n; P->grad = grad; P->left = P->T; P->gen++;
    pthread_cond_broadcast(&P->go);
    while (P->left) pthread_cond_wait(&P->done, &P->mu);
    pthread_mutex_unlock(&P->mu);
}

static void evaluate(pool *P, const size_t *val, size_t nval, double *base, double *loss, double *kl) {
    for (int t = 0; t < P->T; t++) { worker *w = &P->S[t].w; w->loss = w->base = w->kl = 0; w->n = w->na = 0; }
    for (size_t i = 0; i < nval; i += 4096) pool_run(P, val + i, nval - i < 4096 ? nval - i : 4096, 0);
    double b = 0, l = 0, k = 0; long n = 0, na = 0;
    for (int t = 0; t < P->T; t++) { const worker *w = &P->S[t].w; b += w->base; l += w->loss; n += w->n; k += w->kl; na += w->na; }
    *base = n ? b / n : 0; *loss = n ? l / n : 0; *kl = na ? k / na : 0;
}

static uint64_t RNGS = 88172645463325252ull;
static uint64_t rnd(void) { RNGS ^= RNGS << 13; RNGS ^= RNGS >> 7; RNGS ^= RNGS << 17; return RNGS; }

static pool *pool_new(const es_graft *g, const feats *F, int T, int per) {
    pool *P = calloc(1, sizeof *P);
    P->T = T; P->g = g; P->F = F; P->S = calloc((size_t)T, sizeof(slot));
    pthread_mutex_init(&P->mu, NULL); pthread_cond_init(&P->go, NULL); pthread_cond_init(&P->done, NULL);
    for (int t = 0; t < T; t++) {
        slot *s = &P->S[t]; s->id = t; s->P = P;
        if (es_graft_tmp_init(&s->w.t, g)) die("%s", "out of memory");
        s->w.x = malloc(sizeof(float) * g->d); s->w.gd = malloc(sizeof(float) * g->d);
        s->w.grad = calloc(es_graft_dense_floats(g), sizeof(float));
        s->w.ek = malloc(sizeof(float) * (size_t)g->m * ES_GRAFT_ORDERS * (per + 1));
        s->w.ev = malloc(sizeof(float) * (size_t)g->m * ES_GRAFT_ORDERS * (per + 1));
        s->w.eslot = malloc(sizeof(int64_t) * ES_GRAFT_ORDERS * (per + 1));
        s->w.tek = malloc(sizeof(float) * (size_t)g->m * ES_GRAFT_ORDERS); s->w.tev = malloc(sizeof(float) * (size_t)g->m * ES_GRAFT_ORDERS);
        if (!s->w.x || !s->w.grad || !s->w.ek || !s->w.ev || !s->w.eslot) die("%s", "out of memory");
        pthread_create(&s->th, NULL, pool_main, s);
    }
    return P;
}
static void pool_free(pool *P) {
    pthread_mutex_lock(&P->mu); P->quit = 1; pthread_cond_broadcast(&P->go); pthread_mutex_unlock(&P->mu);
    for (int t = 0; t < P->T; t++) pthread_join(P->S[t].th, NULL);
}

static int cpu_count(void) { long n = sysconf(_SC_NPROCESSORS_ONLN); return n > 0 ? (int)n : 1; }

static int cmd_train(int argc, char **argv) {
    const char *fpath = positional(argc, argv, 0), *out = arg_of(argc, argv, "-o", NULL);
    if (!fpath || !out) die("%s", "usage: es_graft train NAME.esfeat -o NAME.esgraft [--epochs 3] [--minutes M] [--m 64] [--experts N] [--f 256] [-j T] [--resume]");
    feats F; memset(&F, 0, sizeof F); feats_open(&F, fpath);
    const int d = (int)F.H.d, K = (int)F.H.K;
    const int m = atoi(arg_of(argc, argv, "--m", "64")), f = atoi(arg_of(argc, argv, "--f", "256"));
    const uint64_t n_exp = (uint64_t)atof(arg_of(argc, argv, "--experts", "0"));   /* hashed memory experts: experimental, no measured gain yet */
    const double epochs = atof(arg_of(argc, argv, "--epochs", "3")), minutes = atof(arg_of(argc, argv, "--minutes", "0"));
    const double lr = atof(arg_of(argc, argv, "--lr", "1e-3")), elr = atof(arg_of(argc, argv, "--elr", "0.02")), mlr = atof(arg_of(argc, argv, "--mlr", "0.05"));
    const int B = atoi(arg_of(argc, argv, "--batch", "64"));
    if (B < 1 || B > 4096) die("%s", "--batch must be 1..4096");
    ANCHOR_W = atof(arg_of(argc, argv, "--anchor-weight", "3"));
    int T = atoi(arg_of(argc, argv, "-j", "0")); if (T <= 0) T = cpu_count();
    RNGS ^= (uint64_t)atoll(arg_of(argc, argv, "--seed", "1")) * 0x9e3779b97f4a7c15ull;
    if (m > 4096 || m < 4 || f < 4) die("%s", "--m must be 4..4096");

    es_graft g; char err[512];
    char ep[4096]; snprintf(ep, sizeof ep, "%s.experts", out);
    const int resume = has_flag(argc, argv, "--resume");
    if (resume) { if (es_graft_load(&g, out, 1, err, sizeof err)) die("%s", err); if (g.d != d || g.K != K || g.fingerprint != F.H.fingerprint) die("%s does not belong to this feature file", out); }
    else {
        if (es_graft_new(&g, d, m, f, K, n_exp, (int)F.H.n_vocab, (uint32_t)rnd(), n_exp ? ep : NULL, err, sizeof err)) die("%s", err);
        g.fingerprint = F.H.fingerprint; snprintf(g.arch, sizeof g.arch, "%s", F.H.arch); snprintf(g.name, sizeof g.name, "%s", F.H.name);
    }
    if (!resume) {   /* the memory candidates travel with the graft (a resumed graft already has them, mapped) */
        char src[4096], dst[4096]; snprintf(src, sizeof src, "%s.mem", fpath); snprintf(dst, sizeof dst, "%s.mem", out);
        FILE *a = fopen(src, "rb"), *b = a ? fopen(dst, "wb") : NULL;
        if (a && b) { char buf[1 << 16]; size_t k; while ((k = fread(buf, 1, sizeof buf, a)) > 0) if (fwrite(buf, 1, k, b) != k) die("cannot write %s", dst); }
        if (b && fclose(b)) die("cannot write %s", dst);
        if (a) fclose(a); else remove(dst);
    }
    const size_t ND = es_graft_dense_floats(&g);
    float *am = calloc(ND, sizeof(float)), *av = calloc(ND, sizeof(float)), *gsum = malloc(ND * sizeof(float));
    char op[4096]; snprintf(op, sizeof op, "%s.opt", out);
    uint64_t step = 0;
    if (resume) { FILE *of = fopen(op, "rb"); if (of) { if (fread(&step, 8, 1, of) != 1 || fread(am, 4, ND, of) != ND || fread(av, 4, ND, of) != ND) step = 0; fclose(of); } }

    size_t *tr = malloc(sizeof(size_t) * F.H.n_rec), *va = malloc(sizeof(size_t) * F.H.n_rec), ntr = 0, nva = 0;
    for (size_t i = 0; i < F.H.n_rec; i++) { if (is_val(&F, rec_at(&F, i))) va[nva++] = i; else tr[ntr++] = i; }
    if (ntr == 0) die("%s", "no training positions");
    const double dense_p = (double)ND, exp_p = (double)n_exp * 2 * g.m;
    fprintf(stderr, "graft on %s: %d-dim hidden state, top %d candidates + memory | trainable: %.2fM parameters", F.H.name, d, K, dense_p / 1e6);
    if (n_exp) fprintf(stderr, " + %.2fB in hashed memory experts (on flash, sparse; experimental)", exp_p / 1e9);
    fprintf(stderr, "\n");
    fprintf(stderr, "data: %zu training and %zu held-out positions | %d threads, batch %d\n", ntr, nva, T, B);

    const int per = (B + T - 1) / T;
    pool *P = pool_new(&g, &F, T, per);
    double vb = 0, vl = 0;
    double vk = 0;
    if (nva) { evaluate(P, va, nva, &vb, &vl, &vk); fprintf(stderr, "held out before: base model %.4f (perplexity %.2f) | with graft %.4f\n", vb, exp(vb), vl); }
    const double base_held = vb;
    const uint64_t steps_epoch = (ntr + B - 1) / B, total = (uint64_t)(epochs * steps_epoch + 0.5) + (resume ? step : 0);
    const uint64_t warm = total / 20 > 50 ? 50 : total / 20 + 1;
    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
    const double t0 = now_s(); double tlast = t0, tsave = t0, run_loss = 0, run_base = 0, run_kl = 0; long run_n = 0, run_na = 0, done_tok = 0;
    size_t at = ntr;   /* position in the shuffled order */
    for (; step < total && !STOP; step++) {
        if (minutes > 0 && now_s() - t0 > minutes * 60) break;
        size_t batch[4096]; int nb = 0;
        for (; nb < B; nb++) {
            if (at >= ntr) { for (size_t i = ntr - 1; i > 0; i--) { const size_t j = rnd() % (i + 1); const size_t t = tr[i]; tr[i] = tr[j]; tr[j] = t; } at = 0; }
            batch[nb] = tr[at++];
        }
        for (int t = 0; t < T; t++) { worker *w = &P->S[t].w; memset(w->grad, 0, ND * sizeof(float)); w->ne = 0; w->loss = w->base = w->kl = 0; w->n = w->na = 0; }
        pool_run(P, batch, (size_t)nb, 1);
        /* dense weights: Adam on the batch mean; learning rate warms up, then falls along a cosine to 10% */
        memset(gsum, 0, ND * sizeof(float));
        for (int t = 0; t < T; t++) { const float *gw = P->S[t].w.grad; for (size_t i = 0; i < ND; i++) gsum[i] += gw[i]; run_loss += P->S[t].w.loss; run_base += P->S[t].w.base; run_n += P->S[t].w.n; run_kl += P->S[t].w.kl; run_na += P->S[t].w.na; }
        const double prog = (double)step / (total > 1 ? total - 1 : 1);
        const double lrs = lr * (step < warm ? (step + 1.0) / warm : 0.1 + 0.9 * 0.5 * (1 + cos(3.141592653589793 * prog)));
        const double b1 = 0.9, b2 = 0.999, c1 = 1 - pow(b1, step + 1.0), c2 = 1 - pow(b2, step + 1.0);
        float *W = g.P;
        const size_t imb = (size_t)(g.mb - g.P), jmb = imb + ES_MEM_BONUS * ES_MEM_ORDERS;   /* the memory bonus: a few scalars that need bigger steps */
        for (size_t i = 0; i < ND; i++) {
            const float gi = gsum[i] / nb;
            am[i] = (float)(b1 * am[i] + (1 - b1) * gi);
            av[i] = (float)(b2 * av[i] + (1 - b2) * gi * gi);
            W[i] -= (float)((i >= imb && i < jmb ? lrs / lr * mlr : lrs) * (am[i] / c1) / (sqrt(av[i] / c2) + 1e-8));
        }
        /* memory experts: row-wise Adagrad on each use (sparse: only the experts this batch touched) */
        const float elrs = (float)(elr * (step < warm ? (step + 1.0) / warm : 1.0));
        for (int t = 0; t < T; t++) {
            const worker *w = &P->S[t].w;
            for (int e = 0; e < w->ne; e++) {
                float *rec = g.E + (size_t)w->eslot[e] * es_graft_rec(g.m), *k = rec, *v = rec + g.m;
                const float *gk = w->ek + (size_t)e * g.m, *gv = w->ev + (size_t)e * g.m;
                float sk = 0, sv = 0;
                for (int r = 0; r < g.m; r++) { sk += gk[r] * gk[r]; sv += gv[r] * gv[r]; }
                if (sk > 0) { rec[2 * g.m] += sk / g.m; const float s = elrs / (sqrtf(rec[2 * g.m]) + 1e-10f); for (int r = 0; r < g.m; r++) k[r] -= s * gk[r]; }
                if (sv > 0) { rec[2 * g.m + 1] += sv / g.m; const float s = elrs / (sqrtf(rec[2 * g.m + 1]) + 1e-10f); for (int r = 0; r < g.m; r++) v[r] -= s * gv[r]; }
            }
        }
        done_tok += nb; g.tokens += (uint64_t)nb; g.steps++;
        const double tn = now_s();
        if (tn - tlast > 10 || step + 1 == total) {
            fprintf(stderr, "  step %llu/%llu | train loss %.4f (base %.4f) | change on general text %.4f | %.0f tokens/s | %.1f min\n", (unsigned long long)step + 1,
                    (unsigned long long)total, run_n ? run_loss / run_n : 0, run_n ? run_base / run_n : 0, run_na ? run_kl / run_na : 0, done_tok / (tn - t0), (tn - t0) / 60);
            run_loss = run_base = run_kl = 0; run_n = run_na = 0; tlast = tn;
        }
        if (tn - tsave > 300) { if (es_graft_save(&g, out, err, sizeof err)) die("%s", err); tsave = tn; }
    }
    const double secs = now_s() - t0;
    fprintf(stderr, "memory bonus per context length:");
    for (int o = 0; o < ES_MEM_ORDERS; o++) { const float *q = g.mb + ES_MEM_BONUS * o; fprintf(stderr, " | %d: %+.2f %+.2f log c %+.2f log(c/n), gate bias %+.2f", ES_MEM_ORDER[o], q[0], q[1], q[2], q[3]); }
    fprintf(stderr, "\n");
    if (es_graft_save(&g, out, err, sizeof err)) die("%s", err);
    FILE *of = fopen(op, "wb");
    if (of) { fwrite(&step, 8, 1, of); fwrite(am, 4, ND, of); fwrite(av, 4, ND, of); fclose(of); }
    fprintf(stderr, "trained %ld positions in %.1f s: %.0f tokens/s on %d threads\n", done_tok, secs, done_tok / (secs > 0 ? secs : 1), T);
    if (nva) {
        evaluate(P, va, nva, &vb, &vl, &vk);
        fprintf(stderr, "held out after: base model %.4f (perplexity %.2f) | with graft %.4f (perplexity %.2f) | %+.1f%% perplexity | change on general text %.4f\n",
                base_held, exp(base_held), vl, exp(vl), 100 * (exp(vl - base_held) - 1), vk);
        printf("RESULT positions=%ld seconds=%.1f tokens_per_s=%.0f held_out=%zu base_loss=%.5f graft_loss=%.5f anchor_kl=%.5f\n", done_tok, secs, done_tok / (secs > 0 ? secs : 1), nva, base_held, vl, vk);
    }
    fprintf(stderr, "saved %s. Use it: es_chat -g MODEL.gguf -A %s\n", out, out);
    pool_free(P);
    return 0;
}

static int cmd_eval(int argc, char **argv) {
    const char *fpath = positional(argc, argv, 0), *gpath = positional(argc, argv, 1);
    if (!fpath || !gpath) die("%s", "usage: es_graft eval NAME.esfeat NAME.esgraft");
    feats F; memset(&F, 0, sizeof F); feats_open(&F, fpath);
    es_graft g; char err[512];
    if (es_graft_load(&g, gpath, 0, err, sizeof err)) die("%s", err);
    if (g.d != (int)F.H.d || g.K != (int)F.H.K || g.fingerprint != F.H.fingerprint) die("%s does not belong to this feature file", gpath);
    size_t *va = malloc(sizeof(size_t) * F.H.n_rec), nva = 0;
    const int all = has_flag(argc, argv, "--all");
    for (size_t i = 0; i < F.H.n_rec; i++) if (all || is_val(&F, rec_at(&F, i))) va[nva++] = i;
    int T = atoi(arg_of(argc, argv, "-j", "0")); if (T <= 0) T = cpu_count();
    pool *P = pool_new(&g, &F, T, 1);
    double b, l, k;
    evaluate(P, va, nva, &b, &l, &k);
    printf("%zu %s positions: base model %.4f (perplexity %.2f) | with graft %.4f (perplexity %.2f) | %+.1f%% perplexity | change on general text %.4f\n", nva, all ? "" : "held-out",
           b, exp(b), l, exp(l), 100 * (exp(l - b) - 1), k);
    pool_free(P);
    return 0;
}

/* ------------------------------------------------------------------ plan: measured on this device before committing hours */
static struct { int K, nv, n; const int32_t *ids; double sink; } PL;
static void plan_pos(int pos, const float *x, const float *lg, void *ud) {   /* the same per-position work as read, without writing */
    (void)ud; (void)x;
    int top[ES_GRAFT_MAXK];
    const int k = es_graft_topk(lg, PL.nv, PL.K, top);
    double all = 0; for (int i = 0; i < PL.nv; i++) all += exp((double)lg[i] - lg[top[0]]);
    PL.sink += all + k + pos;
}
static double train_speed(int d, int kc, int m, int f, int T) {   /* positions per second of the training step, measured */
    es_graft g; char err[64];
    if (es_graft_new(&g, d, m, f, 32, 0, 1000, 1, NULL, err, sizeof err)) return 0;
    for (size_t i = 0; i < (size_t)g.d * g.m; i++) g.Q[i] = 0.01f;
    es_graft_tmp t; es_graft_tmp_init(&t, &g);
    /* candidates' output rows come from a table much bigger than the CPU caches, as in training (about 64 MB here) */
    const int nrows = (int)(((size_t)64 << 20) / ((size_t)d * 4)) > 4 * kc ? (int)(((size_t)64 << 20) / ((size_t)d * 4)) : 4 * kc;
    float *x = malloc(sizeof(float) * d), *rows = malloc(sizeof(float) * (size_t)nrows * d), *gd = malloc(sizeof(float) * d), *grad = calloc(es_graft_dense_floats(&g), sizeof(float));
    uint16_t *xs = malloc(sizeof(uint16_t) * d);
    float bl[ES_GRAFT_MAXK], gu[4096], ga[4096], ek[ES_GRAFT_ORDERS * 4096], ev[ES_GRAFT_ORDERS * 4096], gmb[64], gmg[8];
    const float *W[ES_GRAFT_MAXK]; uint16_t cnt[ES_GRAFT_MAXK * ES_MEM_ORDERS] = {0}; uint32_t tot[ES_MEM_ORDERS] = {0};
    for (int i = 0; i < d; i++) xs[i] = es_fp32_to_fp16((float)((i * 37 % 101) - 50) / 50.0f);
    for (size_t i = 0; i < (size_t)nrows * d; i++) rows[i] = (float)((int)(i * 13 % 29) - 14) / 100.0f;
    for (int k = 0; k < kc; k++) bl[k] = (float)(-k) * 0.1f;
    const int32_t ctx[ES_GRAFT_CTX] = {1, 2, 3};
    uint64_t r = 12345;
    const double t0 = now_s(); long n = 0;
    while (now_s() - t0 < 0.7) {
        for (int i = 0; i < d; i++) x[i] = es_fp16_to_fp32(xs[i]);
        for (int k = 0; k < kc; k++) { r = r * 6364136223846793005ull + 1442695040888963407ull; W[k] = rows + (size_t)((r >> 33) % (uint64_t)nrows) * d; }
        es_graft_forward(&g, x, ctx, &t);
        es_graft_loss(kc, d, bl, 0.5, 1, 0, W, t.delta, cnt, tot, g.mb, t.mg, gd, gmb, gmg);
        es_graft_backward(&g, &t, gd, gmg, grad, gu, ga, ek, ev);
        n++;
    }
    const double rate = n / (now_s() - t0);
    es_graft_free(&g); es_graft_tmp_free(&t); free(x); free(xs); free(rows); free(gd); free(grad);
    return rate * T * 0.4;   /* threads share memory bandwidth, then syncing, merging and the optimizer per batch: 0.4 per thread (calibrated on the test PC) */
}
static void say_time(char *b, size_t cap, double s) {
    if (s < 90) snprintf(b, cap, "%.0f seconds", s);
    else if (s < 5400) snprintf(b, cap, "%.0f minutes", s / 60);
    else snprintf(b, cap, "%.1f hours", s / 3600);
}
static int cmd_plan(int argc, char **argv) {
    const char *data = arg_of(argc, argv, "--data", NULL);
    if (!data) die("%s", "usage: es_graft plan -g MODEL.gguf --data FILE.txt|FILE.jsonl [--seq 512] [--k 32] [--tokens N] [--sample 128]");
    const int seq = atoi(arg_of(argc, argv, "--seq", "512")), K = atoi(arg_of(argc, argv, "--k", "32")), sample = atoi(arg_of(argc, argv, "--sample", "128"));
    const double max_tok = atof(arg_of(argc, argv, "--tokens", "0"));
    const int val_every = atoi(arg_of(argc, argv, "--val-every", "20"));
    init_engine(argc, argv, seq + 16);
    int d, nv; es_engine_dims(&d, &nv);
    examples X;
    collect(&X, data, arg_of(argc, argv, "--anchor", "auto"), seq, max_tok, val_every);
    if (!X.ndata) die("%s", "no examples in the data");
    /* reading speed: the first examples through the frozen model, exactly as read does it */
    PL.K = K; PL.nv = nv;
    double done = 0; const double t0 = now_s();
    for (int e = 0; e < X.nex && done < sample; e++) { es_engine_features(X.pool + X.off[e], X.len[e], NULL, plan_pos, NULL); done += X.len[e]; }
    const double rs = done / (now_s() - t0);
    int T = atoi(arg_of(argc, argv, "-j", "0")); if (T <= 0) T = cpu_count();
    const int kc = K + ES_MEM_ORDERS * ES_MEM_PER;
    const double ts = train_speed(d, kc, 64, 256, T);
    const double positions = X.learn + X.anchor_tok, feat_gb = positions * rec_size(d, kc) / 1e9;
    es_engine_info in; es_engine_get_info(&in);
    printf("plan: graft %s on this device\n", in.name);
    printf("  data: %d examples, %.0f tokens to learn from, plus %.0f anchor tokens (general text the graft must leave alone)%s\n", X.ndata, X.learn, X.anchor_tok,
           X.skipped ? " (some examples skipped)" : "");
    char t1[64], t2[64];
    say_time(t1, sizeof t1, X.forward_tok / rs); say_time(t2, sizeof t2, 3 * positions / ts);
    printf("  1. read: the frozen model reads %.0f tokens once. Measured here: %.2f tokens/s -> about %s (stop any time, run again to continue)\n",
           X.forward_tok, rs, t1);
    printf("     feature file: %.2f GB on this disk\n", feat_gb);
    printf("  2. train: about %.0f positions/s on %d threads -> 3 epochs in about %s (the big model is not used again)\n", ts, T, t2);
    printf("  3. use: es_chat -g MODEL -A NAME.esgraft (or es_serve --graft): the graft adds about 1-2 ms per token\n");
    printf("honest summary: the model's own weights stay frozen. The graft learns the facts, names, wording and style of your data, and reading\n"
           "the data through the big model once is the slow part. It cannot give the model abilities its weights do not have.\n");
    return 0;
}

/* ------------------------------------------------------------------ check: the engine runs the graft as training assumed */
static struct { int d, K, nv; const es_mem *mem; float *x; int32_t *top; int *ntop; uint16_t *cnt; uint32_t *tot; float *bl; double *logr; int *yin; float *by; const int32_t *ids; int n; } CK;
#define CKC (ES_GRAFT_MAXK)
static void ck_pos(int pos, const float *x, const float *lg, void *ud) {
    (void)ud;
    memcpy(CK.x + (size_t)pos * CK.d, x, sizeof(float) * CK.d);
    int32_t ctx[ES_GRAFT_CTX]; ctx_of(CK.ids, pos, ctx);
    int32_t *top = CK.top + (size_t)pos * CKC;
    const int k = es_graft_candidates(CK.K, CK.mem, lg, CK.nv, ctx, -1, top);
    double mx = lg[top[0]], all = 0, in = 0;
    for (int i = 0; i < k; i++) if (lg[top[i]] > mx) mx = lg[top[i]];
    for (int i = 0; i < CK.nv; i++) all += exp((double)lg[i] - mx);
    CK.yin[pos] = -1; CK.ntop[pos] = k;
    es_mem_counts(CK.mem, ctx, -1, top, k, CK.cnt + (size_t)pos * CKC * ES_MEM_ORDERS, CK.tot + (size_t)pos * ES_MEM_ORDERS);
    for (int i = 0; i < k; i++) { CK.bl[(size_t)pos * CKC + i] = lg[top[i]]; in += exp((double)lg[top[i]] - mx); if (top[i] == CK.ids[pos + 1]) CK.yin[pos] = i; }
    CK.logr[pos] = all - in > all * 1e-12 ? mx + log(all - in) : -INFINITY;
    CK.by[pos] = lg[CK.ids[pos + 1]];
}
static int cmd_check(int argc, char **argv) {
    const char *gpath = positional(argc, argv, 0), *data = arg_of(argc, argv, "--data", NULL);
    if (!gpath || !data) die("%s", "usage: es_graft check -g MODEL.gguf NAME.esgraft --data TEXT.txt [--tokens 200]");
    const int ntok = atoi(arg_of(argc, argv, "--tokens", "200"));
    init_engine(argc, argv, ntok + 32);
    int d, nv; es_engine_dims(&d, &nv);
    size_t len; char *text = slurp(data, &len);
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)(ntok + 8)); int n = 0;
    if (es_engine_bos() >= 0) ids[n++] = es_engine_bos();
    if (len > 1 << 20) text[1 << 20] = 0;
    int32_t *all = malloc(sizeof(int32_t) * (1 << 20));
    const int m = es_engine_tokenize(text, all, 1 << 20);
    for (int i = 0; i < m && n < ntok; i++) ids[n++] = all[i];
    es_graft g; char err[512];
    if (es_graft_load(&g, gpath, 0, err, sizeof err)) die("%s", err);
    CK.d = d; CK.K = g.K; CK.nv = nv; CK.ids = ids; CK.n = n; CK.mem = &g.mem;
    CK.x = malloc(sizeof(float) * (size_t)n * d); CK.top = malloc(sizeof(int32_t) * (size_t)n * CKC); CK.bl = malloc(sizeof(float) * (size_t)n * CKC);
    CK.ntop = malloc(sizeof(int) * n); CK.cnt = malloc(sizeof(uint16_t) * (size_t)n * CKC * ES_MEM_ORDERS); CK.tot = malloc(sizeof(uint32_t) * (size_t)n * ES_MEM_ORDERS); CK.logr = malloc(sizeof(double) * n); CK.yin = malloc(sizeof(int) * n); CK.by = malloc(sizeof(float) * n);
    /* A: the base model's outputs, then the graft's loss computed the way training computes it */
    uint8_t *want = malloc((size_t)n); memset(want, 1, (size_t)n); want[n - 1] = 0;   /* the last token has no next token to score */
    es_engine_features(ids, n, want, ck_pos, NULL);
    uint32_t type; const uint8_t *od; size_t rb; es_engine_output_raw(&type, &od, &rb);
    es_graft_tmp t; es_graft_tmp_init(&t, &g);
    float *rows = malloc(sizeof(float) * (size_t)CKC * d); const float *W[ES_GRAFT_MAXK];
    double *la = malloc(sizeof(double) * n), *l0 = malloc(sizeof(double) * n);
    int from_mem = 0;
    for (int p = 0; p < n - 1; p++) {
        int32_t ctx[ES_GRAFT_CTX]; ctx_of(ids, p, ctx);
        es_graft_forward(&g, CK.x + (size_t)p * d, ctx, &t);
        const int k = CK.ntop[p];
        for (int i = 0; i < k; i++) { es_dequant_row(type, od + rb * (size_t)CK.top[(size_t)p * CKC + i], rows + (size_t)i * d, d); W[i] = rows + (size_t)i * d; }
        if (CK.yin[p] >= g.K) from_mem++;
        la[p] = es_graft_loss(k, d, CK.bl + (size_t)p * CKC, CK.logr[p], CK.yin[p], CK.by[p], W, t.delta, CK.cnt + (size_t)p * CKC * ES_MEM_ORDERS, CK.tot + (size_t)p * ES_MEM_ORDERS, g.mb, t.mg, NULL, NULL, NULL);
        l0[p] = es_graft_loss(k, d, CK.bl + (size_t)p * CKC, CK.logr[p], CK.yin[p], CK.by[p], NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    }
    /* B: the engine with the graft on, one token at a time like generation; loss from its full list of scores */
    if (es_engine_set_graft(gpath, err, sizeof err)) die("%s", err);
    char tmp[] = "/tmp/es_graft_checkXXXXXX"; const int tfd = mkstemp(tmp);
    FILE *tf = fdopen(tfd, "w+b");
    es_engine_dump_logits(ids, n, tf);
    fflush(tf); rewind(tf);
    float *lg = malloc(sizeof(float) * nv);
    double worst = 0, sa = 0, sb = 0, s0 = 0;
    for (int p = 0; p < n - 1; p++) {
        if (fread(lg, sizeof(float), nv, tf) != (size_t)nv) die("%s", "cannot read back the scores");
        double mx = lg[0]; for (int i = 1; i < nv; i++) if (lg[i] > mx) mx = lg[i];
        double s = 0; for (int i = 0; i < nv; i++) s += exp((double)lg[i] - mx);
        const double lb = mx + log(s) - lg[ids[p + 1]];
        if (fabs(lb - la[p]) > worst) worst = fabs(lb - la[p]);
        sa += la[p]; sb += lb; s0 += l0[p];
    }
    fclose(tf); unlink(tmp);
    printf("%d positions: base %.4f | graft as trained %.4f | graft in the engine %.4f | largest difference per position %.2e | right token found only by the memory: %d\n",
           n - 1, s0 / (n - 1), sa / (n - 1), sb / (n - 1), worst, from_mem);
    printf("CHECK %s\n", worst < 2e-2 ? "OK" : "MISMATCH");
    return worst < 2e-2 ? 0 : 1;
}

/* ------------------------------------------------------------------ selftest: gradients against finite differences */
static int cmd_selftest(void) {
    char ep[] = "/tmp/es_graft_selftestXXXXXX"; const int fd = mkstemp(ep); close(fd);
    es_graft g; char err[256];
    const int d = 24, m = 8, f = 12, K = 6, NV = 40;
    if (es_graft_new(&g, d, m, f, K, 16, NV, 7, ep, err, sizeof err)) die("%s", err);
    #define RN() ((float)((int64_t)(rnd() % 2001) - 1000) / 1000.0f)
    for (size_t i = 0; i < es_graft_dense_floats(&g); i++) g.P[i] += 0.3f * RN();
    for (size_t i = 0; i < 16 * es_graft_rec(m); i++) g.E[i] = 0.5f * RN();
    float x[24], rowsb[ES_GRAFT_MAXK][24], bl[ES_GRAFT_MAXK]; const float *W[ES_GRAFT_MAXK];
    uint16_t cnt[ES_GRAFT_MAXK * ES_MEM_ORDERS];
    for (int i = 0; i < d; i++) x[i] = 2 * RN();
    for (int k = 0; k < K; k++) { bl[k] = 3 * RN(); for (int i = 0; i < d; i++) rowsb[k][i] = RN(); W[k] = rowsb[k]; for (int o = 0; o < ES_MEM_ORDERS; o++) cnt[k * ES_MEM_ORDERS + o] = (uint16_t)(rnd() % 4); }
    const int32_t ctx[ES_GRAFT_CTX] = {5, 9, 13, 2, 3, 4, 6, 7};
    es_graft_tmp t; es_graft_tmp_init(&t, &g);
    const size_t ND = es_graft_dense_floats(&g);
    float *grad = calloc(ND, sizeof(float)), gd[24], gu[8], ga[8], ek[3 * 8], ev[3 * 8], gmb[ES_MEM_BONUS * ES_MEM_ORDERS], gmg[ES_MEM_ORDERS];
    uint32_t tot[ES_MEM_ORDERS] = {9, 5, 3, 0};
    double worst = 0; int bad = 0;
    for (int trial = 0; trial < 3; trial++) {
        const int yin = trial == 0 ? 2 : trial == 1 ? -1 : -2; const float by = 0.7f; const double logr = 1.3;   /* right token among candidates, elsewhere, anchor */
        es_graft_forward(&g, x, ctx, &t);
        es_graft_loss(K, d, bl, logr, yin, by, W, t.delta, cnt, tot, g.mb, t.mg, gd, gmb, gmg);
        memset(grad, 0, sizeof(float) * ND);
        es_graft_backward(&g, &t, gd, gmg, grad, gu, ga, ek, ev);
        for (int i = 0; i < ES_MEM_BONUS * ES_MEM_ORDERS; i++) grad[(g.mb - g.P) + i] += gmb[i];
        #define LOSS() (es_graft_forward(&g, x, ctx, &t), es_graft_loss(K, d, bl, logr, yin, by, W, t.delta, cnt, tot, g.mb, t.mg, NULL, NULL, NULL))
        for (size_t i = 0; i < ND; i += (i + 7 < ND - ES_MEM_BONUS * ES_MEM_ORDERS ? 7 : 1)) {   /* every 7th dense weight, every memory bonus */
            const float o = g.P[i]; const float h = 1e-2f;
            g.P[i] = o + h; const double lp = LOSS(); g.P[i] = o - h; const double lm = LOSS(); g.P[i] = o;
            const double num = (lp - lm) / (2 * h), e = fabs(num - grad[i]) / (fabs(num) + fabs(grad[i]) + 5e-3);   /* float losses: ~1e-4 absolute noise */
            if (e > worst) worst = e;
            if (e > 2e-2) { bad++; if (getenv("ES_DEBUG")) fprintf(stderr, "trial %d dense %zu of %zu: numeric %g analytic %g\n", trial, i, ND, num, grad[i]); }
        }
        for (int n = 0; n < 3; n++) {
            const int64_t sl = es_graft_slot(&g, n + 1, ctx);
            float *rec = g.E + (size_t)sl * es_graft_rec(m);
            int shared = 0;   /* two orders can hash to the same expert: then the numeric gradient is the sum of both uses */
            for (int o2 = 0; o2 < 3; o2++) if (o2 != n && es_graft_slot(&g, o2 + 1, ctx) == sl) shared = 1;
            if (shared) continue;
            for (int r = 0; r < 2 * m; r++) {
                const float o = rec[r], h = 1e-2f;
                rec[r] = o + h; const double lp = LOSS(); rec[r] = o - h; const double lm = LOSS(); rec[r] = o;
                const double num = (lp - lm) / (2 * h), an = r < m ? ek[n * m + r] : ev[n * m + r - m];
                const double e = fabs(num - an) / (fabs(num) + fabs(an) + 5e-3);
                if (e > worst) worst = e;
                if (e > 2e-2) { bad++; if (getenv("ES_DEBUG")) fprintf(stderr, "trial %d expert order %d entry %d: numeric %g analytic %g\n", trial, n, r, num, an); }
            }
        }
    }
    /* the exact loss equals cross-entropy over the full list of scores when only the candidates change */
    float full[40]; for (int i = 0; i < NV; i++) full[i] = (float)(-2 + 0.05 * i);
    int top[ES_GRAFT_MAXK]; const int k = es_graft_topk(full, NV, K, top);
    float tb[ES_GRAFT_MAXK]; double mx = full[top[0]], all = 0, in = 0;
    for (int i = 0; i < NV; i++) all += exp(full[i] - mx);
    for (int i = 0; i < k; i++) { tb[i] = full[top[i]]; in += exp(tb[i] - mx); }
    es_graft_forward(&g, x, ctx, &t);
    double worst_ce = 0;
    for (int y = 0; y < NV; y += 3) {
        int yin = -1; for (int i = 0; i < k; i++) if (top[i] == y) yin = i;
        const double a = es_graft_loss(K, d, tb, mx + log(all - in), yin, full[y], W, t.delta, cnt, tot, g.mb, t.mg, NULL, NULL, NULL);
        float mod[40]; memcpy(mod, full, sizeof mod);
        for (int i = 0; i < k; i++) {
            float s2 = 0; for (int j = 0; j < d; j++) s2 += W[i][j] * t.delta[j];
            s2 += es_mem_bonus(g.mb, t.mg, cnt + i * ES_MEM_ORDERS, tot);
            mod[top[i]] += s2;
        }
        double m2 = mod[0], z = 0; for (int i = 1; i < NV; i++) if (mod[i] > m2) m2 = mod[i];
        for (int i = 0; i < NV; i++) z += exp(mod[i] - m2);
        const double b = m2 + log(z) - mod[y];
        if (fabs(a - b) > worst_ce) worst_ce = fabs(a - b);
    }
    {   /* anchor: KL(base || grafted) over the full vocabulary */
        const double a = es_graft_loss(K, d, tb, mx + log(all - in), -2, 0, W, t.delta, cnt, tot, g.mb, t.mg, NULL, NULL, NULL);
        float mod[40]; memcpy(mod, full, sizeof mod);
        for (int i = 0; i < k; i++) { float s2 = 0; for (int j = 0; j < d; j++) s2 += W[i][j] * t.delta[j]; mod[top[i]] += s2 + es_mem_bonus(g.mb, t.mg, cnt + i * ES_MEM_ORDERS, tot); }
        double z0 = 0, z1 = 0; for (int i = 0; i < NV; i++) { z0 += exp(full[i]); z1 += exp(mod[i]); }
        double kl = 0; for (int i = 0; i < NV; i++) { const double p = exp(full[i]) / z0; kl += p * (log(p) - (mod[i] - log(z1))); }
        if (fabs(a - kl) > worst_ce) worst_ce = fabs(a - kl);
    }
    es_graft_free(&g); unlink(ep);
    printf("gradient check: worst relative error %.2e (%d above 2e-2) | exact loss vs full cross-entropy and KL: %.2e\n", worst, bad, worst_ce);
    printf("SELFTEST %s\n", bad == 0 && worst_ce < 1e-4 ? "OK" : "FAILED");
    return bad == 0 && worst_ce < 1e-4 ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "es_graft: teach a frozen model (up to 200B+ parameters) new things on a phone\n"
                        "  es_graft plan  -g MODEL.gguf --data notes.txt|chat.jsonl          (measures this device: how long each step takes)\n"
                        "  es_graft read  -g MODEL.gguf --data notes.txt|chat.jsonl -o my.esfeat [--seq 512] [--k 32] [--tokens N] [--minutes M]\n"
                        "  es_graft train my.esfeat -o my.esgraft [--epochs 3] [--minutes M] [--m 64] [--experts N] [-j T]\n"
                        "  es_graft eval  my.esfeat my.esgraft\n"
                        "  es_graft check -g MODEL.gguf my.esgraft --data text.txt\n"
                        "  es_graft selftest\n"
                        "then: es_chat -g MODEL.gguf -A my.esgraft   or   es_serve -g MODEL.gguf --graft my.esgraft\n");
        return 2;
    }
    (void)jnum;
    if (!strcmp(argv[1], "read")) return cmd_read(argc, argv);
    if (!strcmp(argv[1], "train")) return cmd_train(argc, argv);
    if (!strcmp(argv[1], "eval")) return cmd_eval(argc, argv);
    if (!strcmp(argv[1], "check")) return cmd_check(argc, argv);
    if (!strcmp(argv[1], "plan")) return cmd_plan(argc, argv);
    if (!strcmp(argv[1], "selftest")) return cmd_selftest();
    die("unknown command %s", argv[1]);
    return 2;
}
