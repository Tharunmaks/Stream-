/* es_mtest: checks that es_matvec_multi (several activations per pass, used for prompt batches) gives bit-identical results to
 * es_matvec called once per activation, for every type it accelerates. Random weights and activations.   ./es_mtest */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/es_quant.h"
#include "../src/es_cpu.h"

static uint64_t R = 88172645463325252ull;
static uint32_t rnd(void) { R ^= R << 13; R ^= R >> 7; R ^= R << 17; return (uint32_t)R; }
static uint16_t f16(float f) {   /* positive normal floats only */
    uint32_t u; memcpy(&u, &f, 4); int e = (int)((u >> 23) & 255) - 127 + 15; return (uint16_t)((e << 10) | ((u >> 13) & 1023));
}
int main(void) {
    const struct { uint32_t t; const char *n; int bs, bb, dpos; } T[] = {
        {GGML_Q5_0, "Q5_0", 32, 22, 0}, {GGML_Q4_0, "Q4_0", 32, 18, 0}, {GGML_Q4_K, "Q4_K", 256, 144, 0}, {GGML_Q6_K, "Q6_K", 256, 210, 208}};
    const int cols = 1024, rows = 37, na = 13;
    es_act_arena_init(1 << 24);
    int bad = 0;
    printf("cpu: %s\n",
#if defined(__aarch64__)
           es_has_dotprod() ? "arm64 with SDOT (NEON kernels)" : "arm64 without SDOT"
#elif defined(__x86_64__)
           __builtin_cpu_supports("avx2") ? "x86-64 with AVX2 kernels" : "x86-64 without AVX2"
#else
           "generic"
#endif
    );
    for (size_t k = 0; k < sizeof T / sizeof T[0]; k++) {
        const size_t rb = (size_t)cols / T[k].bs * T[k].bb; uint8_t *W = malloc(rb * rows);
        for (size_t i = 0; i < rb * rows; i++) W[i] = (uint8_t)rnd();
        for (int r = 0; r < rows; r++) for (int b = 0; b < cols / T[k].bs; b++) {   /* sane block scales */
            uint8_t *blk = W + r * rb + (size_t)b * T[k].bb; uint16_t d = f16(0.001f + (rnd() % 1000) / 1e5f), dm = f16(0.0005f + (rnd() % 1000) / 1e5f);
            memcpy(blk + T[k].dpos, &d, 2); if (T[k].t == GGML_Q4_K) memcpy(blk + 2, &dm, 2);
        }
        es_act a[16]; es_block_q8_K *q8 = malloc(sizeof(es_block_q8_K) * (cols / 256 + 1) * na); float *x = malloc(sizeof(float) * cols * na);
        for (int i = 0; i < cols * na; i++) x[i] = ((int)(rnd() % 2001) - 1000) / 250.0f;
        for (int q = 0; q < na; q++) es_act_prepare(&a[q], x + (size_t)q * cols, cols, q8 + (size_t)q * (cols / 256 + 1));
        float *y1 = calloc((size_t)rows * na, sizeof(float)), *y2 = calloc((size_t)rows * na, sizeof(float)), *sc = malloc(sizeof(float) * cols);
        for (int q = 0; q < na; q++) es_matvec(T[k].t, W, cols, 3, rows, &a[q], y1 + (size_t)q * rows, sc);
        es_matvec_multi(T[k].t, W, cols, 3, rows, a, na, y2, rows, sc);
        int same = memcmp(y1, y2, sizeof(float) * rows * na) == 0;
        printf("%-5s %d activations x %d rows: %s (e.g. %.6f vs %.6f)\n", T[k].n, na, rows - 3, same ? "bit-identical" : "DIFFERENT", y1[rows + 5], y2[rows + 5]);
        bad |= !same;
        free(W); free(q8); free(x); free(y1); free(y2); free(sc);
    }
    puts(bad ? "FAIL" : "PASS");
    return bad;
}
