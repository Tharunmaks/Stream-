/* esd_ktest: every libesd kernel this CPU has (moe/esd_kern.h) must give exactly the scalar result.
 *   esd_ktest            (on a phone, or under qemu-aarch64 -cpu max for the NEON and SDOT paths) */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "esd_kern.h"

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
#if defined(__aarch64__)
    bad += check("neon tq1", k1_neon, k1_scalar, 1); bad += check("neon int4", k4_neon, k4_scalar, 4); bad += check("neon ternary", k2_neon, k2_scalar, 2); bad += check("neon int8", k8_neon, k8_scalar, 8);
#endif
    return bad != 0;
}
