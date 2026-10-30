/* esd_ktest: every libesd kernel this CPU has (moe/esd_kern.h) must give exactly the scalar result.
 *   esd_ktest            (on a phone, or under qemu-aarch64 -cpu max for the NEON and SDOT paths) */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esd_kern.h"
#include "es_i8.h"

static unsigned long long rs = 88172645463325252ULL;
static unsigned xr(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 11); }

static int check(const char *name, esd_kern k, esd_kern ref, int bits) {
    int bad = 0;
    for (int trial = 0; trial < 40; trial++) {
        const int k_ = bits == 1 ? 256 * (1 + xr() % 12) : 64 * (1 + xr() % 48), rows = 1 + xr() % 37; const size_t rb = bits == 1 ? (size_t)k_ / 256 * 52 : (size_t)k_ * bits / 8;
        uint8_t *w = malloc(rb * rows); int8_t *x = malloc(k_); int32_t *a = malloc(4 * rows), *b = malloc(4 * rows);
        for (size_t i = 0; i < rb * rows; i++) w[i] = (uint8_t)xr();
        if (bits == 1) for (size_t i = 0; i < rb * rows; i++) { const int tail = i % 52 >= 48, n = tail ? 81 : 243, v = (int)(xr() % n); w[i] = (uint8_t)((v * 256 + n - 1) / n); }
        if (bits == 8) for (size_t i = 0; i < rb * rows; i++) if (w[i] == 0x80) w[i] = 0x81;   /* int8 rows are in [-127, 127] */
        for (int i = 0; i < k_; i++) x[i] = (int8_t)(trial % 3 == 0 ? (xr() & 1 ? 127 : -127) : (int)(xr() % 255) - 127);
        k(w, rb, x, k_, rows, a); ref(w, rb, x, k_, rows, b);
        if (memcmp(a, b, 4 * rows)) bad++;
        free(w); free(x); free(a); free(b);
    }
    printf("%s %-12s %s\n", bad ? "FAIL" : "PASS", name, bad ? "differs from scalar" : "= scalar (40 random shapes)");
    return bad;
}

/* es_i8.h (the trainer's int8 GEMM): exact integer products, n x k by k x m in VNNI layout, B in [-8, 8] */
static int check_i8(const char *name, i8_gemm_fn g) {
    int bad = 0;
    for (int trial = 0; trial < 40; trial++) {   /* odd trials: the AMX shapes (k % 64, m % 32), with full and partial 32-token blocks */
        const int n = trial & 1 ? 1 + xr() % 70 : 1 + xr() % 23, k = trial & 1 ? 64 * (1 + xr() % 12) : 4 * (1 + xr() % 96), m = trial & 1 ? 32 * (1 + xr() % 6) : 16 * (1 + xr() % 12);
        int8_t *A = malloc((size_t)n * k), *B = malloc((size_t)k * m); int32_t *cs = calloc(m, 4), *c1 = malloc(4 * (size_t)n * m), *c2 = malloc(4 * (size_t)n * m);
        for (int i = 0; i < n * k; i++) A[i] = (int8_t)(trial % 3 == 0 ? (xr() & 1 ? 127 : -127) : (int)(xr() % 255) - 127);
        for (int i = 0; i < k * m; i++) { B[i] = (int8_t)(trial % 5 == 0 ? (xr() & 1 ? 8 : -8) : (int)(xr() % 17) - 8); cs[(i / 4) % m] += B[i]; }
        g(n, m, k, A, B, cs, c1); i8_gemm_scalar(n, m, k, A, B, cs, c2);
        if (memcmp(c1, c2, 4 * (size_t)n * m)) bad++;
        free(A); free(B); free(cs); free(c1); free(c2);
    }
    printf("%s %-12s %s\n", bad ? "FAIL" : "PASS", name, bad ? "differs from scalar" : "= scalar (40 random shapes)");
    return bad;
}

/* the activation quantizer: SIMD rows = the portable row, ties included */
static int check_quant(void) {
    int bad = 0;
    for (int trial = 0; trial < 40; trial++) {
        const int k = 32 * (1 + xr() % 64); float *x = malloc(4 * (size_t)k); int8_t *a = malloc(k), *b = malloc(k); float sa, sb; int32_t ra, rb;
        for (int i = 0; i < k; i++) x[i] = trial % 2 ? ((int)(xr() % 255) - 127) * 0.5f : ((int)(xr() % 20001) - 10000) * 1e-3f;   /* odd trials: exact .5 ties */
        if (trial % 2) x[0] = 63.5f, x[1] = -127.0f;
        i8_quant_rows(x, 1, k, a, &sa, &ra); i8_quant_row_c(x, k, b, &sb, &rb);
        if (memcmp(a, b, k) || sa != sb || ra != rb) bad++;
        free(x); free(a); free(b);
    }
    printf("%s train quantizer = portable (40 rows, with ties)\n", bad ? "FAIL" : "PASS"); return bad;
}

int main(void) {
    esd_kern k4, k2, k8, k1; const char *name; esd_pick(&k4, &k2, &k8, &k1, &name);
    char n[64]; int bad = 0, enc = 0;
    for (int nd = 4; nd <= 5; nd++) { const int n3 = nd == 5 ? 243 : 81;   /* every base-3 value round-trips through the byte code */
        for (int v = 0; v < n3; v++) { const uint8_t q = (uint8_t)((v * 256 + n3 - 1) / n3); int r = 0; for (int d = 0; d < nd; d++) r = r * 3 + tq_digit(q, d); enc += r != v; } }
    printf("%s tq1 byte code round-trips all 243 + 81 values\n", enc ? "FAIL" : "PASS"); bad += enc != 0;
    snprintf(n, sizeof n, "%s int4", name); bad += check(n, k4, k4_scalar, 4);
    snprintf(n, sizeof n, "%s ternary", name); bad += check(n, k2, k2_scalar, 2);
    snprintf(n, sizeof n, "%s int8", name); bad += check(n, k8, k8_scalar, 8);
    snprintf(n, sizeof n, "%s tq1", name); bad += check(n, k1, k1_scalar, 1);
    bad += check_quant();
    const char *gn; i8_gemm_fn g = i8_pick(&gn); snprintf(n, sizeof n, "train %s", gn); bad += check_i8(n, g);
#if defined(__x86_64__) && defined(__GNUC__)
    if (__builtin_cpu_supports("avx2")) bad += check_i8("train avx2", i8_gemm_avx2);
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("avxvnni")) bad += check_i8("train vnni", i8_gemm_vnni);
#endif
#if defined(__aarch64__)
    bad += check("neon tq1", k1_neon, k1_scalar, 1); bad += check("neon int4", k4_neon, k4_scalar, 4); bad += check("neon ternary", k2_neon, k2_scalar, 2); bad += check("neon int8", k8_neon, k8_scalar, 8);
#endif
    return bad != 0;
}
