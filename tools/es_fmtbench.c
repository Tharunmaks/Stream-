/* es_fmtbench: speed of every weight format's kernel, one thread, random weights (GB of weights per second).
 *   ./es_fmtbench                  all formats
 *   ES_NO_LBNEON=1 ./es_fmtbench   phones: the previous low-bit path, for comparison
 *   ES_NO_AVX2=1 ./es_fmtbench     PCs: without the AVX2 kernels */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../src/es_quant.h"
#include "../src/es_gguf.h"
#if defined(__x86_64__)
#include <xmmintrin.h>
#endif
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(void) {
#if defined(__x86_64__)
    _mm_setcsr(_mm_getcsr() | 0x8040);   /* random scales include denormals; real models do not */
#endif
    const int cols = 4096, rows = 4096;
    const uint32_t types[] = {GGML_Q4_0, GGML_Q4_1, GGML_Q5_0, GGML_Q5_1, GGML_Q8_0, GGML_Q2_K, GGML_Q3_K, GGML_Q4_K, GGML_Q5_K, GGML_Q6_K,
                              GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ2_S, GGML_IQ3_XXS, GGML_IQ3_S, GGML_IQ1_S, GGML_IQ1_M, GGML_IQ4_NL, GGML_IQ4_XS,
                              GGML_TQ1_0, GGML_TQ2_0, GGML_MXFP4, GGML_BF16, GGML_F16, GGML_IQ1_XXS};
    float *x = malloc(sizeof(float) * cols), *y = malloc(sizeof(float) * rows), *scratch = malloc(sizeof(float) * cols * 4);
    for (int i = 0; i < cols; i++) x[i] = (float)rand() / (float)RAND_MAX - 0.5f;
    es_block_q8_K *q8 = malloc(sizeof(es_block_q8_K) * (cols / 256));
    es_act_arena_init(1 << 20);
    for (size_t t = 0; t < sizeof types / sizeof *types; t++) {
        if (!es_quant_supported(types[t])) continue;
        const size_t rb = es_row_bytes(types[t], cols), n = rb * rows;
        uint8_t *w = malloc(n);
        for (size_t i = 0; i < n; i++) w[i] = (uint8_t)rand();
        /* tame fp16 scales: set every 16-bit word with exponent >= 0x7c00 (inf/nan) to a small value */
        for (size_t i = 0; i + 1 < n; i += 2) { uint16_t v; memcpy(&v, w + i, 2); if ((v & 0x7c00) == 0x7c00) { v = 0x2000; memcpy(w + i, &v, 2); } }
        es_act_arena_reset();
        es_act a; es_act_prepare(&a, x, cols, q8);
        es_matvec(types[t], w, cols, 0, rows, &a, y, scratch);
        int it = 0; const double t0 = now(); double el;
        do { es_matvec(types[t], w, cols, 0, rows, &a, y, scratch); it++; } while ((el = now() - t0) < 0.4);
        printf("%-8s %6.2f GB/s  (%5.1f ms per 4096x4096)\n", es_ggml_type_name(types[t]), n * it / el / 1e9, el / it * 1e3);
        free(w);
    }
    return 0;
}
