/*
 * es_kbench: CPU topology, Q2_K kernel correctness, and compute throughput.
 *
 * Answers: how fast can this phone turn 2-bit weights that are already in
 * RAM into matvec results, per core type and with N threads? Compare that
 * with es_bench's flash bandwidth to see which one limits tokens/s.
 *
 *   ./es_kbench              # default: 256 MB of weights
 *   ./es_kbench -m 512 -i 5  # bigger working set, more iterations
 */
#define _GNU_SOURCE
#include <getopt.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/es_cpu.h"
#include "../src/es_io.h"
#include "../src/es_q2k.h"
#include "es_common.h"

static es_topo T;

/* ---------- small helpers ---------- */

static float frand(uint64_t *s) { /* roughly N(0,1) */
    float a = 0;
    for (int i = 0; i < 4; i++) a += (float)(es_rng_next(s) >> 40) / (float)(1 << 24);
    return (a - 2.0f) * 1.7f;
}

static void print_meminfo(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return;
    char line[256];
    while (fgets(line, sizeof line, f))
        if (!strncmp(line, "MemTotal", 8) || !strncmp(line, "MemAvailable", 12) ||
            !strncmp(line, "MemFree", 7) || !strncmp(line, "SwapTotal", 9) ||
            !strncmp(line, "SwapFree", 8))
            printf("  %s", line);
    fclose(f);
}

typedef struct {
    atomic_int count, gen;
    int n;
} spin_barrier;

static void barrier_wait(spin_barrier *b) {
    int g = atomic_load(&b->gen);
    if (atomic_fetch_add(&b->count, 1) == b->n - 1) {
        atomic_store(&b->count, 0);
        atomic_fetch_add(&b->gen, 1);
        return;
    }
    for (int spins = 0; atomic_load(&b->gen) == g;)
        if (++spins > 2000) sched_yield();
}

/* ---------- parallel runner: pinned threads, timed from a common start ---------- */

typedef void (*work_fn)(int tid, int nthreads, void *arg);

typedef struct {
    int tid, nthreads, cpu;
    work_fn fn;
    void *arg;
    atomic_int *ready, *go;
    int pin_err;
} thread_ctx;

static void *thread_main(void *p) {
    thread_ctx *c = p;
    c->pin_err = c->cpu >= 0 ? es_pin_self(c->cpu) : 0;
    atomic_fetch_add(c->ready, 1);
    while (!atomic_load(c->go)) sched_yield();
    c->fn(c->tid, c->nthreads, c->arg);
    return NULL;
}

/* returns seconds; cpus[i] = cpu to pin thread i to (-1 = no pinning) */
static double run_parallel(int n, const int *cpus, work_fn fn, void *arg) {
    pthread_t th[ES_MAX_CPUS];
    thread_ctx ctx[ES_MAX_CPUS];
    atomic_int ready = 0, go = 0;
    for (int i = 0; i < n; i++) {
        ctx[i] = (thread_ctx){i, n, cpus ? cpus[i] : -1, fn, arg, &ready, &go, 0};
        pthread_create(&th[i], NULL, thread_main, &ctx[i]);
    }
    while (atomic_load(&ready) < n) sched_yield();
    uint64_t t0 = es_now_ns();
    atomic_store(&go, 1);
    for (int i = 0; i < n; i++) pthread_join(th[i], NULL);
    double secs = (es_now_ns() - t0) / 1e9;
    for (int i = 0; i < n; i++)
        if (ctx[i].pin_err)
            printf("    (warning: could not pin thread to cpu%d: %s)\n", ctx[i].cpu,
                   strerror(ctx[i].pin_err));
    return secs;
}

/* ---------- memory bandwidth ---------- */

typedef struct {
    const uint64_t *buf;
    size_t words;
    int iters;
    uint64_t sink[ES_MAX_CPUS * 8];
} mem_job;

static void mem_work(int tid, int n, void *arg) {
    mem_job *j = arg;
    size_t per = j->words / (size_t)n, lo = per * (size_t)tid, hi = lo + per;
    uint64_t a = 0, b = 0, c = 0, d = 0;
    for (int it = 0; it < j->iters; it++)
        for (size_t i = lo; i + 4 <= hi; i += 4) {
            a += j->buf[i]; b += j->buf[i + 1]; c += j->buf[i + 2]; d += j->buf[i + 3];
        }
    j->sink[tid * 8] = a ^ b ^ c ^ d;
}

/* ---------- matvec throughput ---------- */

typedef struct {
    const es_block_q2_K *W;
    int nmat, rows, cols, iters;
    const es_block_q8_K *xq;
    float *y;
    spin_barrier bar;
} mv_job;

/* Each matrix is split by rows across threads, with a barrier after every
 * matrix: the same sync granularity as running one expert projection. */
static void mv_work(int tid, int n, void *arg) {
    mv_job *j = arg;
    const size_t mat_blocks = (size_t)j->rows * (j->cols / ES_QK_K);
    int per = (j->rows + n - 1) / n, r0 = per * tid, r1 = r0 + per;
    if (r1 > j->rows) r1 = j->rows;
    for (int it = 0; it < j->iters; it++)
        for (int m = 0; m < j->nmat; m++) {
            if (r0 < r1)
                es_matvec_q2_K(j->W + (size_t)m * mat_blocks, j->cols, r0, r1, j->xq,
                               j->y + (size_t)m * j->rows);
            if (n > 1) barrier_wait(&j->bar);
        }
}

/* ---------- correctness ---------- */

static int test_kernels(void) {
    const int rows = 64, cols = 4096, nb = cols / ES_QK_K;
    uint64_t s = 42;
    float *w = malloc(sizeof(float) * rows * cols), *x = malloc(sizeof(float) * cols);
    float *wd = malloc(sizeof(float) * rows * cols);
    es_block_q2_K *wq = malloc(sizeof(es_block_q2_K) * rows * nb);
    es_block_q8_K *xq = malloc(sizeof(es_block_q8_K) * nb);
    for (int i = 0; i < rows * cols; i++) w[i] = frand(&s) * 0.02f;
    for (int i = 0; i < cols; i++) x[i] = frand(&s);
    for (int r = 0; r < rows; r++) es_quantize_q2_K_ref(w + r * cols, wq + r * nb, cols);
    es_dequantize_q2_K(wq, wd, rows * cols);
    es_quantize_q8_K(x, xq, cols);

    double qerr = 0, qref = 0, max_fast_rel = 0, act_err = 0, act_ref = 0;
    for (int i = 0; i < rows * cols; i++) {
        qerr += (w[i] - wd[i]) * (double)(w[i] - wd[i]);
        qref += (double)w[i] * w[i];
    }
    for (int r = 0; r < rows; r++) {
        double exact = 0;
        for (int c = 0; c < cols; c++) exact += (double)wd[r * cols + c] * x[c];
        float ref = es_dot_q2_K_q8_K_ref(cols, wq + r * nb, xq);
        float fast = es_dot_q2_K_q8_K(cols, wq + r * nb, xq);
        double rel = fabs(fast - ref) / (fabs(ref) + 1e-6);
        if (rel > max_fast_rel) max_fast_rel = rel;
        act_err += (ref - exact) * (ref - exact);
        act_ref += exact * exact;
    }
    double wq_rel = sqrt(qerr / qref), act_rel = sqrt(act_err / act_ref);
    int ok = max_fast_rel < 1e-5 && act_rel < 0.02 && wq_rel < 0.6;
    printf("correctness (%s):\n", es_q2k_kernel_name());
    printf("  fast kernel vs scalar reference : max rel err %.2e  %s\n", max_fast_rel,
           max_fast_rel < 1e-5 ? "OK" : "FAIL");
    printf("  int8 activations vs float dot   : rel err %.4f      %s\n", act_rel,
           act_rel < 0.02 ? "OK" : "FAIL");
    printf("  2-bit weight error (test quantizer, gaussian weights): rel %.3f\n", wq_rel);
    free(w); free(x); free(wd); free(wq); free(xq);
    return ok;
}

/* ---------- main ---------- */

static void usage(void) {
    fprintf(stderr,
            "usage: es_kbench [-m weight_MB=256] [-c cols=4096] [-r rows=1536]\n"
            "                 [-i iters=3] [-S (force scalar kernel)]\n"
            "                 [-A active_expert_params=14.19e9] [-D dense_params=7.3e9]\n");
    exit(2);
}

int main(int argc, char **argv) {
    size_t mb = 256;
    int cols = 4096, rows = 1536, iters = 3, opt;
    double expert_params = 14.19e9, dense_params = 7.3e9;
    while ((opt = getopt(argc, argv, "m:c:r:i:SA:D:h")) != -1) {
        switch (opt) {
        case 'm': mb = (size_t)atoi(optarg); break;
        case 'c': cols = atoi(optarg); break;
        case 'r': rows = atoi(optarg); break;
        case 'i': iters = atoi(optarg); break;
        case 'S': es_q2k_force_scalar(1); break;
        case 'A': expert_params = atof(optarg); break;
        case 'D': dense_params = atof(optarg); break;
        default: usage();
        }
    }
    if (cols % ES_QK_K || rows < 1 || iters < 1 || mb < 1) usage();

    es_topo_read(&T);
    printf("cpus (fastest first):\n");
    for (int i = 0; i < T.ncpu; i++) {
        int c = T.order[i];
        printf("  cpu%-2d %-18s part 0x%03x  max %4ld MHz  %s\n", c,
               es_part_name(T.impl[c], T.part[c]), T.part[c], T.max_khz[c] / 1000,
               T.allowed[c] ? "" : "(not allowed for this process)");
    }
    printf("features: dotprod=%s i8mm=%s   kernel: %s\n", es_has_dotprod() ? "yes" : "no",
           es_has_i8mm() ? "yes" : "no", es_q2k_kernel_name());
    printf("memory:\n");
    print_meminfo();
    printf("\n");

    if (!test_kernels()) {
        printf("\nKERNEL TEST FAILED - stopping.\n");
        return 1;
    }

    /* pinning orders: fastest-first list of allowed cpus, and one cpu per
     * distinct core type for the single-thread runs */
    int fast[ES_MAX_CPUS], nfast = 0, reps[ES_MAX_CPUS], nreps = 0;
    for (int i = 0; i < T.ncpu; i++) {
        int c = T.order[i];
        if (!T.allowed[c]) continue;
        fast[nfast++] = c;
        int seen = 0;
        for (int k = 0; k < nreps; k++)
            if (T.part[reps[k]] == T.part[c] && T.max_khz[reps[k]] == T.max_khz[c]) seen = 1;
        if (!seen) reps[nreps++] = c;
    }
    if (nfast == 0) { /* topology unreadable: run unpinned */
        nfast = T.ncpu > 0 ? T.ncpu : 1;
        for (int i = 0; i < nfast; i++) fast[i] = -1;
        reps[0] = -1;
        nreps = 1;
    }

    /* weights: random but valid Q2_K blocks, all pages touched */
    const int nb = cols / ES_QK_K;
    const size_t mat_bytes = (size_t)rows * nb * sizeof(es_block_q2_K);
    int nmat = (int)(mb * 1000000 / mat_bytes);
    if (nmat < 1) nmat = 1;
    size_t wbytes = (size_t)nmat * mat_bytes;
    es_block_q2_K *W = es_alloc(wbytes);
    float *y = es_alloc(sizeof(float) * (size_t)rows * nmat);
    float *x = malloc(sizeof(float) * cols);
    es_block_q8_K *xq = malloc(sizeof(es_block_q8_K) * nb);
    if (!W || !y || !x || !xq) { perror("alloc"); return 1; }
    uint64_t s = 7;
    uint64_t *w64 = (uint64_t *)W;
    for (size_t i = 0; i < wbytes / 8; i++) w64[i] = es_rng_next(&s);
    uint16_t dh = es_fp32_to_fp16(0.01f);
    for (size_t i = 0; i < wbytes / sizeof(es_block_q2_K); i++) W[i].d = W[i].dmin = dh;
    for (int i = 0; i < cols; i++) x[i] = frand(&s);
    es_quantize_q8_K(x, xq, cols);

    printf("\nworking set: %d matrices of %d x %d = %.0f MB of Q2_K weights\n", nmat, rows,
           cols, wbytes / 1e6);

    mem_job mj = {.buf = (const uint64_t *)W, .words = wbytes / 8, .iters = iters};
    mv_job vj = {.W = W, .nmat = nmat, .rows = rows, .cols = cols, .iters = iters,
                 .xq = xq, .y = y};
    const double bytes_done = (double)wbytes * iters;
    const double expert_bytes = 3.0 * mat_bytes;

    printf("\nsingle thread, per core type:\n");
    printf("  cpu    core               mem read GB/s   kernel GB/s   Gweights/s\n");
    for (int k = 0; k < nreps; k++) {
        int cpu = reps[k];
        double tm = run_parallel(1, &cpu, mem_work, &mj);
        double tk = run_parallel(1, &cpu, mv_work, &vj);
        printf("  cpu%-2d  %-18s %8.2f        %8.2f      %8.2f\n", cpu,
               cpu >= 0 ? es_part_name(T.impl[cpu], T.part[cpu]) : "?", bytes_done / tm / 1e9,
               bytes_done / tk / 1e9, bytes_done / tk / 1e9 * 256 / 84);
    }

    printf("\nthreads (fastest cores first), barrier after every matrix:\n");
    printf("  threads  cpus              mem GB/s   kernel GB/s   ms/expert\n");
    double best = 0;
    int best_n = 0;
    for (int n = 1; n <= nfast; n++) {
        if (n > 2 && n % 2 && n != nfast) continue; /* 1,2,4,6,8 + max */
        atomic_store(&vj.bar.count, 0);
        atomic_store(&vj.bar.gen, 0);
        vj.bar.n = n;
        double tm = run_parallel(n, fast, mem_work, &mj);
        double tk = run_parallel(n, fast, mv_work, &vj);
        char list[64] = "", tmp[8];
        for (int i = 0; i < n && strlen(list) < 50; i++) {
            snprintf(tmp, sizeof tmp, i ? ",%d" : "%d", fast[i]);
            strcat(list, tmp);
        }
        double gbps = bytes_done / tk / 1e9;
        printf("  %5d    %-16s  %8.2f    %8.2f      %7.2f\n", n, list, bytes_done / tm / 1e9,
               gbps, expert_bytes / (gbps * 1e9) * 1e3);
        if (gbps > best) { best = gbps; best_n = n; }
        fflush(stdout);
    }

    const double bpw = 84.0 / 256 / 1; /* bytes per weight */
    double t_exp = expert_params * bpw / (best * 1e9);
    double t_dense = dense_params * bpw / (best * 1e9);
    printf("\nbest kernel throughput: %.2f GB/s of Q2_K weights (%d threads)\n", best, best_n);
    printf("compute per token, weights already in RAM (Qwen3-235B-A22B shape):\n");
    printf("  routed experts %.1fB params : %.2f s\n", expert_params / 1e9, t_exp);
    printf("  dense part     %.1fB params : %.2f s   (attention + lm_head, if Q2_K)\n",
           dense_params / 1e9, t_dense);
    printf("  total                       : %.2f s/token compute floor\n", t_exp + t_dense);
    float sink = 0;
    for (int i = 0; i < rows * nmat; i += 997) sink += y[i];
    printf("(checksum %g)\n", sink);
    free(W); free(y); free(x); free(xq);
    return 0;
}
