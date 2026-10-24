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
        const int k_ = 64 * (1 + xr() % 48), rows = 1 + xr() % 37; const size_t rb = (size_t)k_ * bits / 8;
        uint8_t *w = malloc(rb * rows); int8_t *x = malloc(k_); int32_t *a = malloc(4 * rows), *b = malloc(4 * rows);
        for (size_t i = 0; i < rb * rows; i++) w[i] = (uint8_t)xr();
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
    esd_kern k4, k2, k8; const char *name; esd_pick(&k4, &k2, &k8, &name);
    char n[64]; int bad = 0;
    snprintf(n, sizeof n, "%s int4", name); bad += check(n, k4, k4_scalar, 4);
    snprintf(n, sizeof n, "%s ternary", name); bad += check(n, k2, k2_scalar, 2);
    snprintf(n, sizeof n, "%s int8", name); bad += check(n, k8, k8_scalar, 8);
#if defined(__aarch64__)
    bad += check("neon int4", k4_neon, k4_scalar, 4); bad += check("neon ternary", k2_neon, k2_scalar, 2); bad += check("neon int8", k8_neon, k8_scalar, 8);
#endif
    return bad != 0;
}
