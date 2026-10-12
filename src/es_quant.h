/*
 * Quantized tensor rows: dequantize and matrix-vector product for the GGUF
 * types real models use (F32, F16, BF16, Q4_0..Q8_0, Q2_K..Q6_K, IQ1_S..IQ4_XS, TQ1_0, TQ2_0, MXFP4, Q1_0, Q2_0, IQ1_XS..IQ1_XXXS). Layouts follow
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
    GGML_Q4_K = 12, GGML_Q5_K = 13, GGML_Q6_K = 14, GGML_IQ2_XXS = 16, GGML_IQ2_XS = 17, GGML_IQ3_XXS = 18,
    GGML_IQ1_S = 19, GGML_IQ4_NL = 20, GGML_IQ3_S = 21, GGML_IQ2_S = 22, GGML_IQ4_XS = 23, GGML_IQ1_M = 29,
    GGML_BF16 = 30, GGML_TQ1_0 = 34, GGML_TQ2_0 = 35, GGML_MXFP4 = 39, GGML_Q1_0 = 41, GGML_Q2_0 = 42,
    GGML_IQ1_XS = 64, GGML_IQ1_XXS = 65, GGML_IQ1_XXXS = 66,   /* Unsloth's narrow-index IQ1 (llama.cpp fork, branch iq1-narrow) */
};

/* 1 if this engine can compute with the type */
int  es_quant_supported(uint32_t type);
/* Dequantize n weights (n a multiple of the block size). */
void es_dequant_row(uint32_t type, const void *src, float *dst, int n);

/* Activation prepared once per matvec input: float copy plus Q8_K blocks
 * for the fast integer path. cols must be a multiple of 256. */
typedef struct { float d; int8_t qs[32]; } es_blk_q80;   /* activation block for the 32-wide formats */
typedef struct {
    const float *x;
    es_block_q8_K *q8;     /* NULL unless cols % 256 == 0 */
    es_blk_q80 *q80;       /* NULL unless the arena had room; cols % 32 == 0 */
    int cols;
} es_act;
/* Q8_0 activations live in an arena that is emptied after every batch of matvecs. */
void es_act_arena_init(size_t bytes);
void es_act_arena_reset(void);
void es_act_prepare(es_act *a, const float *x, int cols, es_block_q8_K *q8_buf);

/* y[r] = dot(W[r], x) for r in [r0, r1). scratch holds `cols` floats. */
void es_matvec(uint32_t type, const void *W, int cols, int r0, int r1, const es_act *a,
               float *y, float *scratch);
/* Several activations against the same rows (prompt batches): bit-identical to calling es_matvec for each, faster. */
void es_matvec_multi(uint32_t type, const void *W, int cols, int r0, int r1, const es_act *a, int na, float *y, size_t ystride, float *scratch);
/* Bytes of one row of `cols` weights of this type. */
size_t es_row_bytes(uint32_t type, int cols);

#endif
