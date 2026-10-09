/*
 * Quantized tensor rows: dequantize and matrix-vector product for the GGUF
 * types real models use (F32, F16, Q4_0..Q8_0, Q2_K..Q6_K). Layouts follow
 * llama.cpp exactly; every dequantizer is checked against the gguf Python
 * package by tools/es_qtest.
 */
#ifndef ES_QUANT_H
#define ES_QUANT_H

#include <stddef.h>
#include <stdint.h>

#include "es_q2k.h"

enum {
    GGML_F32 = 0, GGML_F16 = 1, GGML_Q4_0 = 2, GGML_Q4_1 = 3, GGML_Q5_0 = 6,
    GGML_Q5_1 = 7, GGML_Q8_0 = 8, GGML_Q2_K = 10, GGML_Q3_K = 11,
    GGML_Q4_K = 12, GGML_Q5_K = 13, GGML_Q6_K = 14,
};

/* 1 if this engine can compute with the type */
int  es_quant_supported(uint32_t type);
/* Dequantize n weights (n a multiple of the block size). */
void es_dequant_row(uint32_t type, const void *src, float *dst, int n);

/* Activation prepared once per matvec input: float copy plus Q8_K blocks
 * for the fast integer path. cols must be a multiple of 256. */
typedef struct {
    const float *x;
    es_block_q8_K *q8;
    int cols;
} es_act;
void es_act_prepare(es_act *a, const float *x, int cols, es_block_q8_K *q8_buf);

/* y[r] = dot(W[r], x) for r in [r0, r1). scratch holds `cols` floats. */
void es_matvec(uint32_t type, const void *W, int cols, int r0, int r1, const es_act *a,
               float *y, float *scratch);
/* Bytes of one row of `cols` weights of this type. */
size_t es_row_bytes(uint32_t type, int cols);

#endif
