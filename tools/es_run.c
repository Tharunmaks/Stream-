/*
 * es_run: end-to-end MoE decode loop on synthetic weights.
 *
 * Every token goes through all model layers. Each layer does:
 *   1. attention-sized matvec on a resident buffer (compute cost of attention)
 *   2. routing to top-k experts (synthetic router, see below)
 *   3. expert lookup in the RAM cache, demand-load misses from flash
 *   4. router-ahead prefetch of the next layer's predicted experts
 *   5. the real expert FFN with NEON kernels on the big cores:
 *      gate/up matvec -> SiLU(gate) * up -> down matvec -> weighted sum
 *
 * What is real: file I/O (O_DIRECT, pinned readers), cache and eviction,
 * prefetch overlap, Q2_K kernels, thread pool, all timings.
 * What is SIMULATED: the weights are random and the router is synthetic.
 * Its locality (-r, -z) and the prefetch predictor accuracy (-p) are
 * parameters, not measurements, so treat hit rates as "what if". Real
 * values come from real routing traces in a later step.
 *
 *   ./es_run -d ~/es_bench                      # defaults, Qwen3-235B shape
 *   ./es_run -d ~/es_bench -p 0 -r 0            # worst case: no prefetch/locality
 *   ./es_run -d ~/es_bench -C 2048 -r 0.5 -p 0.8
 */
#define _GNU_SOURCE
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/es_cache.h"
#include "../src/es_compute.h"
#include "../src/es_cpu.h"
#include "../src/es_format.h"
#include "../src/es_io.h"
#include "../src/es_q2k.h"
#include "es_common.h"

#define MAXK 16

/* ---------------- options ---------------- */
static const char *o_dir;
static int o_layers = 94, o_k = 8, o_tokens = 32, o_warmup = 8, o_cthreads = 4,
           o_iothreads = 4, o_quiet = 0;
static size_t o_cache_mb = 1024, o_chunk = 1 << 20;
static double o_pred = 0.6, o_reuse = 0.3, o_zipf = 0.0, o_attn_mb = 23, o_freq = 1.0;
static uint64_t o_seed = 1;

/* ---------------- synthetic router ---------------- */
typedef struct {
    int nexp;
    int *perm;    /* [layers][nexp] popularity order per layer */
    double *cdf;  /* [nexp] zipf CDF over popularity ranks */
    int *route;   /* [layers][k] experts for the current token */
    int *prev;    /* [layers][k] previous token's experts */
    int *pred;    /* [layers][k] predicted experts (prefetch targets) */
    int have_prev;
    uint64_t rng;
} router;

static double urand(uint64_t *s) { return (es_rng_next(s) >> 11) * (1.0 / 9007199254740992.0); }

static void router_init(router *R, int layers, int nexp) {
    R->nexp = nexp;
    R->perm = malloc(sizeof(int) * layers * nexp);
    R->cdf = malloc(sizeof(double) * nexp);
    R->route = malloc(sizeof(int) * layers * o_k);
    R->prev = malloc(sizeof(int) * layers * o_k);
    R->pred = malloc(sizeof(int) * layers * o_k);
    R->have_prev = 0;
    R->rng = o_seed * 0x9E3779B97F4A7C15ull + 7;
    for (int l = 0; l < layers; l++) {
        int *p = R->perm + l * nexp;
        for (int e = 0; e < nexp; e++) p[e] = e;
        for (int e = nexp - 1; e > 0; e--) {
            int j = (int)(es_rng_next(&R->rng) % (uint64_t)(e + 1));
            int t = p[e]; p[e] = p[j]; p[j] = t;
        }
    }
    double sum = 0;
    for (int i = 0; i < nexp; i++) sum += 1.0 / pow(i + 1, o_zipf);
    double acc = 0;
    for (int i = 0; i < nexp; i++) {
        acc += 1.0 / pow(i + 1, o_zipf) / sum;
        R->cdf[i] = acc;
    }
}

static int zipf_rank(router *R) {
    double u = urand(&R->rng);
    int lo = 0, hi = R->nexp - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (R->cdf[mid] < u) lo = mid + 1; else hi = mid;
    }
    return lo;
}

static void router_next_token(router *R, int layers) {
    if (R->have_prev) memcpy(R->prev, R->route, sizeof(int) * layers * o_k);
    for (int l = 0; l < layers; l++) {
        int *r = R->route + l * o_k;
        for (int j = 0; j < o_k; j++) {
            int e, tries = 0, dup;
            do {
                if (R->have_prev && urand(&R->rng) < o_reuse)
                    e = R->prev[l * o_k + (int)(es_rng_next(&R->rng) % (uint64_t)o_k)];
                else
                    e = R->perm[l * R->nexp + zipf_rank(R)];
                dup = 0;
                for (int i = 0; i < j; i++) dup |= r[i] == e;
            } while (dup && ++tries < 1000);
            r[j] = e;
        }
        int *p = R->pred + l * o_k;
        for (int j = 0; j < o_k; j++)
            p[j] = urand(&R->rng) < o_pred ? r[j]
                                           : (int)(es_rng_next(&R->rng) % (uint64_t)R->nexp);
    }
    R->have_prev = 1;
}

/* ---------------- compute tasks ---------------- */
typedef struct {
    int k, hidden, inter;
    const es_block_q2_K *gate[MAXK], *up[MAXK], *down[MAXK];
    float wgt[MAXK];
    const es_block_q8_K *xq;
    es_block_q8_K *hq;  /* [k][inter/256] */
    float *gu;          /* [k][2][inter] */
    float *out;         /* [hidden] */
    /* attention stand-in */
    const es_block_q2_K *attn;
    int attn_rows;
    float *attn_out;
} job;

static void attn_task(int tid, int n, void *arg) {
    job *j = arg;
    int lo, hi;
    es_split(j->attn_rows, tid, n, &lo, &hi);
    es_matvec_q2_K(j->attn, j->hidden, lo, hi, j->xq, j->attn_out);
}

static void gate_up_task(int tid, int n, void *arg) {
    job *j = arg;
    int lo, hi;
    es_split(j->k * 2 * j->inter, tid, n, &lo, &hi);
    for (int seg = 0; seg < 2 * j->k; seg++) { /* seg = expert*2 + (0 gate, 1 up) */
        int s0 = seg * j->inter, s1 = s0 + j->inter;
        int a = lo > s0 ? lo : s0, b = hi < s1 ? hi : s1;
        if (a >= b) continue;
        int e = seg / 2;
        const es_block_q2_K *W = (seg & 1) ? j->up[e] : j->gate[e];
        es_matvec_q2_K(W, j->hidden, a - s0, b - s0, j->xq, j->gu + (size_t)seg * j->inter);
    }
}

static void down_task(int tid, int n, void *arg) {
    job *j = arg;
    int lo, hi;
    es_split(j->hidden, tid, n, &lo, &hi);
    const int nbi = j->inter / ES_QK_K;
    for (int r = lo; r < hi; r++) {
        float s = 0;
        for (int e = 0; e < j->k; e++)
            s += j->wgt[e] * es_dot_q2_K_q8_K(j->inter, j->down[e] + (size_t)r * nbi,
                                              j->hq + (size_t)e * nbi);
        j->out[r] = s;
    }
}

static const es_tensor_desc *find_role(const es_header *h, uint32_t role) {
    for (uint32_t i = 0; i < h->n_tensors; i++)
        if (h->t[i].role == role) return &h->t[i];
    return NULL;
}

/* ---------------- main ---------------- */
static void usage(void) {
    fprintf(stderr,
        "usage: es_run -d DIR [options]\n"
        "  -L layers=94     model MoE layers (files are reused if DIR has fewer;\n"
        "                   cache keys use the model layer, so no fake hits)\n"
        "  -k topk=8        experts per token per layer\n"
        "  -n tokens=32     tokens to generate   -w warmup=8 (excluded from summary)\n"
        "  -C cache_mb=1024 RAM for the hot-expert cache\n"
        "  -t threads=4     compute threads (fastest cores)   -i io=4 reader threads\n"
        "  -c chunk_kb=1024 read chunk size\n"
        "  -p 0.6           SIMULATED prefetch predictor accuracy (0 = no prefetch)\n"
        "  -r 0.3           SIMULATED chance an expert repeats from the previous token\n"
        "  -z 0.0           SIMULATED zipf skew of expert popularity (0 = uniform)\n"
        "  -a attn_mb=23    attention weights per layer, run from RAM (0 = skip)\n"
        "  -F 1.0           eviction: frequency weight, in tokens of recency per use\n"
        "  -S seed  -q quiet\n");
    exit(2);
}

int main(int argc, char **argv) {
    int opt;
    while ((opt = getopt(argc, argv, "d:L:k:n:w:C:t:i:c:p:r:z:a:F:S:qh")) != -1) {
        switch (opt) {
        case 'd': o_dir = optarg; break;
        case 'L': o_layers = atoi(optarg); break;
        case 'k': o_k = atoi(optarg); break;
        case 'n': o_tokens = atoi(optarg); break;
        case 'w': o_warmup = atoi(optarg); break;
        case 'C': o_cache_mb = (size_t)atol(optarg); break;
        case 't': o_cthreads = atoi(optarg); break;
        case 'i': o_iothreads = atoi(optarg); break;
        case 'c': o_chunk = (size_t)atoi(optarg) * 1024; break;
        case 'p': o_pred = atof(optarg); break;
        case 'r': o_reuse = atof(optarg); break;
        case 'z': o_zipf = atof(optarg); break;
        case 'a': o_attn_mb = atof(optarg); break;
        case 'F': o_freq = atof(optarg); break;
        case 'S': o_seed = strtoull(optarg, NULL, 10); break;
        case 'q': o_quiet = 1; break;
        default: usage();
        }
    }
    if (!o_dir || o_k < 1 || o_k > MAXK || o_layers < 1 || o_tokens < 1) usage();
    if (o_warmup >= o_tokens) o_warmup = o_tokens / 4;

    unsigned flayers, nexp;
    unsigned long long fbytes;
    if (es_read_manifest(o_dir, &flayers, &nexp, &fbytes) != 0) {
        fprintf(stderr, "no manifest in %s (run es_gen first)\n", o_dir);
        return 1;
    }
    if ((unsigned)o_k > nexp) { fprintf(stderr, "topk > experts in dataset\n"); return 1; }

    /* ---- threads: compute on the fastest cores, readers on the rest ---- */
    es_topo T;
    es_topo_read(&T);
    int ccpu[ES_MAX_CPUS], iocpu[ES_MAX_CPUS], nc = 0, nio = 0;
    for (int i = 0; i < T.ncpu; i++) {
        int c = T.order[i];
        if (!T.allowed[c]) continue;
        if (nc < o_cthreads) ccpu[nc++] = c; else iocpu[nio++] = c;
    }
    int pinned = nc == o_cthreads;
    es_cpool *cp = es_cpool_create(o_cthreads, pinned ? ccpu : NULL);
    es_pool *io = es_pool_create_on(o_iothreads, o_chunk, 1, iocpu, nio);
    if (!cp || !io) { fprintf(stderr, "thread pool creation failed\n"); return 1; }

    /* ---- shapes from the first expert file ---- */
    char path[4096];
    es_expert_path(path, sizeof path, o_dir, 0, 0);
    uint8_t *probe = es_alloc(fbytes);
    es_req pr = {.dst = probe, .cap = fbytes};
    if (es_submit(io, &pr, path) || es_wait(io, &pr) || es_validate(probe, pr.bytes, 0)) {
        fprintf(stderr, "cannot read/validate %s\n", path);
        return 1;
    }
    const es_header *h0 = (const es_header *)probe;
    const es_tensor_desc *tg = find_role(h0, ES_ROLE_GATE), *td = find_role(h0, ES_ROLE_DOWN);
    if (!tg || !td || tg->qtype != ES_QT_Q2_K) { fprintf(stderr, "unsupported expert layout\n"); return 1; }
    if ((h0->flags & ES_FLAG_SYNTHETIC) && !(h0->flags & ES_FLAG_SANE_SCALES)) {
        fprintf(stderr, "dataset was made by an older es_gen (random fp16 scales -> NaN).\n"
                        "regenerate it:  rm -rf %s && ./es_gen -d %s -L 8 -E 64\n", o_dir, o_dir);
        return 1;
    }
    const int hidden = (int)tg->cols, inter = (int)tg->rows;
    if (hidden % ES_QK_K || inter % ES_QK_K) { fprintf(stderr, "dims must be multiples of 256\n"); return 1; }
    free(probe);

    /* ---- cache ---- */
    int nslots = (int)(o_cache_mb * 1048576ull / fbytes);
    if (nslots < 2 * o_k + 1) {
        fprintf(stderr, "cache too small: need at least %d slots (%.0f MB)\n", 2 * o_k + 1,
                (2 * o_k + 1) * fbytes / 1048576.0);
        return 1;
    }
    es_cache C;
    if (es_cache_init(&C, nslots, fbytes, o_layers, (int)nexp, o_freq * o_layers) != 0) {
        fprintf(stderr, "out of memory allocating %d cache slots\n", nslots);
        return 1;
    }

    /* ---- attention stand-in, activations, buffers ---- */
    job J = {.k = o_k, .hidden = hidden, .inter = inter};
    const int nbh = hidden / ES_QK_K, nbi = inter / ES_QK_K;
    J.attn_rows = (int)(o_attn_mb * 1e6 / (nbh * sizeof(es_block_q2_K)));
    es_block_q2_K *attn = NULL;
    if (J.attn_rows > 0) {
        size_t ab = (size_t)J.attn_rows * nbh * sizeof(es_block_q2_K);
        attn = es_alloc(ab);
        uint64_t s = 99;
        for (size_t i = 0; i < ab / 8; i++) ((uint64_t *)attn)[i] = es_rng_next(&s);
        uint16_t d = es_fp32_to_fp16(0.01f);
        for (size_t i = 0; i < (size_t)J.attn_rows * nbh; i++) attn[i].d = attn[i].dmin = d;
        J.attn = attn;
        J.attn_out = calloc((size_t)J.attn_rows, sizeof(float));
    }
    float *x = malloc(sizeof(float) * hidden), *h = malloc(sizeof(float) * inter);
    es_block_q8_K *xq = malloc(sizeof(es_block_q8_K) * nbh);
    J.xq = xq;
    J.hq = malloc(sizeof(es_block_q8_K) * nbi * o_k);
    J.gu = malloc(sizeof(float) * 2 * inter * o_k);
    J.out = malloc(sizeof(float) * hidden);
    for (int e = 0; e < o_k; e++) J.wgt[e] = 1.0f / o_k;
    uint64_t xs = 5;
    for (int i = 0; i < hidden; i++) x[i] = (float)(urand(&xs) * 2 - 1);

    router R;
    router_init(&R, o_layers, (int)nexp);

    double tok_bytes = (double)o_layers * o_k * fbytes;
    printf("es_run: %d layers x top-%d of %u experts, %.2f MiB/expert, hidden %d, inter %d\n",
           o_layers, o_k, nexp, fbytes / 1048576.0, hidden, inter);
    printf("  cache %d slots (%zu MB) = %.1f%% of %d experts | attention %.0f MB/layer\n",
           nslots, o_cache_mb, 100.0 * nslots / (o_layers * nexp), o_layers * (int)nexp, o_attn_mb);
    printf("  compute: %d threads on cpu", o_cthreads);
    for (int i = 0; i < nc; i++) printf("%s%d", i ? "," : "", ccpu[i]);
    printf("%s | io: %d readers on cpu", pinned ? "" : " (NOT pinned)", o_iothreads);
    for (int i = 0; i < nio; i++) printf("%s%d", i ? "," : "", iocpu[i]);
    printf(" | kernel %s\n", es_q2k_kernel_name());
    printf("  SIMULATED router: reuse %.2f, zipf %.2f, predictor %.2f   (dataset has %u file layers)\n",
           o_reuse, o_zipf, o_pred, flayers);
    printf("  cold I/O per token if nothing were cached: %.2f GB\n\n", tok_bytes / 1e9);
    if (!o_quiet) printf("token    ms    io-wait  attn  experts | cached%%  prefetched%%  miss%% | MB read\n");

    double sum_ms = 0, sum_io = 0, sum_attn = 0, sum_exp = 0, sum_mb = 0;
    uint64_t sum_hit = 0, sum_pf = 0, sum_need = 0;
    int counted = 0, failed = 0;

    for (int t = 0; t < o_tokens && !failed; t++) {
        router_next_token(&R, o_layers);
        uint64_t hits0 = C.hits, pfu0 = C.prefetch_used;
        double io_ms = 0, attn_ms = 0, exp_ms = 0, mb = 0;
        uint64_t t0 = es_now_ns();

        for (int l = 0; l < o_layers && !failed; l++) {
            C.step++;
            const int *route = R.route + l * o_k;
            int slot[MAXK];

            /* 1. attention stand-in */
            es_quantize_q8_K(x, xq, hidden);
            uint64_t a0 = es_now_ns();
            if (attn) es_cpool_run(cp, attn_task, &J);
            attn_ms += (es_now_ns() - a0) / 1e6;

            /* 2-3. cache lookup, demand loads first (FIFO queue = priority) */
            for (int j = 0; j < o_k; j++) {
                int e = route[j], s = es_cache_find(&C, l, e);
                if (s >= 0) {
                    C.hits++;
                    if (C.slots[s].prefetched) { C.prefetch_used++; C.slots[s].prefetched = 0; }
                } else {
                    C.misses++;
                    s = es_cache_claim(&C, io, l, e);
                    if (s < 0) { fprintf(stderr, "cache full of busy slots\n"); failed = 1; break; }
                    es_expert_path(path, sizeof path, o_dir, (unsigned)(l % flayers), (unsigned)e);
                    C.slots[s].req.dst = C.slots[s].buf;
                    C.slots[s].req.cap = C.slot_bytes;
                    if (es_submit(io, &C.slots[s].req, path)) {
                        fprintf(stderr, "read %s: %s\n", path, strerror(C.slots[s].req.err));
                        failed = 1;
                        break;
                    }
                    mb += fbytes / 1048576.0;
                }
                C.slots[s].uses++;
                C.slots[s].last_use = C.step;
                if (C.slots[s].protect_until < C.step) C.slots[s].protect_until = C.step;
                slot[j] = s;
            }
            if (failed) break;

            /* 4. prefetch the next layer's predicted experts */
            if (o_pred > 0 && l + 1 < o_layers) {
                const int *pred = R.pred + (l + 1) * o_k;
                for (int j = 0; j < o_k; j++) {
                    if (es_cache_find(&C, l + 1, pred[j]) >= 0) continue;
                    int s = es_cache_claim(&C, io, l + 1, pred[j]);
                    if (s < 0) break;
                    es_expert_path(path, sizeof path, o_dir, (unsigned)((l + 1) % flayers),
                                   (unsigned)pred[j]);
                    C.slots[s].req.dst = C.slots[s].buf;
                    C.slots[s].req.cap = C.slot_bytes;
                    if (es_submit(io, &C.slots[s].req, path)) {
                        fprintf(stderr, "read %s: %s\n", path, strerror(C.slots[s].req.err));
                        failed = 1;
                        break;
                    }
                    C.slots[s].prefetched = 1;
                    C.prefetch_issued++;
                    mb += fbytes / 1048576.0;
                }
            }

            /* wait for this layer's experts */
            uint64_t w0 = es_now_ns();
            for (int j = 0; j < o_k && !failed; j++) {
                es_slot *s = &C.slots[slot[j]];
                if (s->state == ES_SLOT_LOADING) {
                    if (es_wait(io, &s->req) != 0) {
                        fprintf(stderr, "load failed: %s\n", strerror(s->req.err));
                        failed = 1;
                        break;
                    }
                    const char *why = es_validate(s->buf, s->req.bytes, 0);
                    if (why) { fprintf(stderr, "bad expert file: %s\n", why); failed = 1; break; }
                    s->state = ES_SLOT_READY;
                }
                const es_header *hh = s->buf;
                J.gate[j] = (const es_block_q2_K *)((uint8_t *)s->buf + find_role(hh, ES_ROLE_GATE)->offset);
                J.up[j] = (const es_block_q2_K *)((uint8_t *)s->buf + find_role(hh, ES_ROLE_UP)->offset);
                J.down[j] = (const es_block_q2_K *)((uint8_t *)s->buf + find_role(hh, ES_ROLE_DOWN)->offset);
            }
            io_ms += (es_now_ns() - w0) / 1e6;
            if (failed) break;

            /* 5. expert FFN */
            uint64_t e0 = es_now_ns();
            es_cpool_run(cp, gate_up_task, &J);
            for (int e = 0; e < o_k; e++) {
                const float *g = J.gu + (size_t)(2 * e) * inter, *u = g + inter;
                for (int i = 0; i < inter; i++) h[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
                es_quantize_q8_K(h, J.hq + (size_t)e * nbi, inter);
            }
            es_cpool_run(cp, down_task, &J);
            exp_ms += (es_now_ns() - e0) / 1e6;

            /* residual + RMSNorm keeps the random activations bounded */
            double ss = 0;
            for (int i = 0; i < hidden; i++) {
                x[i] += J.out[i] + (attn ? 1e-3f * J.attn_out[i % J.attn_rows] : 0);
                if (!isfinite(x[i])) x[i] = 0;
                ss += (double)x[i] * x[i];
            }
            float inv = (float)(1.0 / sqrt(ss / hidden + 1e-6));
            for (int i = 0; i < hidden; i++) x[i] *= inv;
        }
        if (failed) break;

        double ms = (es_now_ns() - t0) / 1e6;
        uint64_t hits = C.hits - hits0, pfu = C.prefetch_used - pfu0;
        uint64_t need = (uint64_t)o_layers * o_k;
        if (!o_quiet)
            printf("%4d%s %6.0f  %7.0f  %5.0f  %7.0f |  %5.1f     %5.1f      %5.1f | %6.0f\n", t,
                   t < o_warmup ? "w" : " ", ms, io_ms, attn_ms, exp_ms,
                   100.0 * (hits - pfu) / need, 100.0 * pfu / need, 100.0 * (need - hits) / need, mb);
        fflush(stdout);
        if (t >= o_warmup) {
            counted++;
            sum_ms += ms; sum_io += io_ms; sum_attn += attn_ms; sum_exp += exp_ms; sum_mb += mb;
            sum_hit += hits; sum_pf += pfu; sum_need += need;
        }
    }
    if (failed) return 1;

    double ms = sum_ms / counted;
    printf("\nsteady state over %d tokens (after %d warm-up):\n", counted, o_warmup);
    printf("  %.0f ms/token = %.2f tokens/s\n", ms, 1000 / ms);
    printf("  waiting on flash : %6.0f ms  (%.0f%%)\n", sum_io / counted, 100 * sum_io / sum_ms);
    printf("  attention compute: %6.0f ms\n", sum_attn / counted);
    printf("  expert compute   : %6.0f ms\n", sum_exp / counted);
    printf("  other            : %6.0f ms\n", (sum_ms - sum_io - sum_attn - sum_exp) / counted);
    printf("  experts needed: %.1f%% already cached, %.1f%% brought in by prefetch, %.1f%% demand-loaded\n",
           100.0 * (sum_hit - sum_pf) / sum_need, 100.0 * sum_pf / sum_need,
           100.0 * (sum_need - sum_hit) / sum_need);
    printf("  read %.0f MB/token (%.2f GB/s while generating)\n", sum_mb / counted,
           sum_mb / 1024 / (sum_ms / 1000));
    printf("  prefetches: %llu issued, %llu used, %llu evicted unused | evictions %llu\n",
           (unsigned long long)C.prefetch_issued, (unsigned long long)C.prefetch_used,
           (unsigned long long)C.prefetch_wasted, (unsigned long long)C.evictions);
    printf("  (hit rate and prefetch accuracy come from the SIMULATED router)\n");

    es_cpool_destroy(cp);
    es_pool_destroy(io);
    es_cache_free(&C);
    return 0;
}
