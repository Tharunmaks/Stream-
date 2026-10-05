/*
 * 2-bit weights x 8-bit activations, Q2_K / Q8_K.
 *
 * es_block_q2_K is byte-identical to llama.cpp's block_q2_K (GGUF type 10),
 * so expert files can later be cut straight out of existing GGUF Q2_K
 * models without re-quantizing on the phone.
 *
 * One block = 256 weights in 16 groups of 16:
 *   w = d * (scales[g] & 15) * q  -  dmin * (scales[g] >> 4),   q in 0..3
 * Weight order inside a block (same as ggml dequantize_row_q2_K):
 *   for half h in 0,1:        qs[32h .. 32h+31]
 *     for shift s in 0,2,4,6:
 *       16 weights from (qs[32h + l]      >> s) & 3, l = 0..15
 *       16 weights from (qs[32h + 16 + l] >> s) & 3, l = 0..15
 */
#ifndef ES_Q2K_H
#define ES_Q2K_H

#include <stdint.h>
#include <string.h>

#define ES_QK_K 256

typedef struct {
    uint8_t  scales[ES_QK_K / 16]; /* low 4 bits: scale, high 4 bits: min */
    uint8_t  qs[ES_QK_K / 4];      /* 2-bit quants, 4 per byte */
    uint16_t d;                    /* fp16 super-block scale for scales */
    uint16_t dmin;                 /* fp16 super-block scale for mins */
} es_block_q2_K;

typedef struct {
    float   d;
    int8_t  qs[ES_QK_K];
    int16_t bsums[ES_QK_K / 16];   /* sum of each group of 16 qs */
} es_block_q8_K;

_Static_assert(sizeof(es_block_q2_K) == 84, "block_q2_K must be 84 bytes");
_Static_assert(sizeof(es_block_q8_K) == 292, "block_q8_K must be 292 bytes");

static inline float es_fp16_to_fp32(uint16_t h) {
#if defined(__aarch64__)
    __fp16 f;
    memcpy(&f, &h, 2);
    return (float)f;
#else
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else { /* subnormal: renormalize */
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            bits = sign | (exp << 23) | ((man & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
#endif
}

uint16_t es_fp32_to_fp16(float f);

/* Simple min/max Q2_K quantizer: for tests and benchmarks only. Real models
 * come pre-quantized (GGUF); quality quantization isn't done on the phone. */
void es_quantize_q2_K_ref(const float *x, es_block_q2_K *y, int n);
void es_dequantize_q2_K(const es_block_q2_K *x, float *y, int n);
void es_quantize_q8_K(const float *x, es_block_q8_K *y, int n);

/* dot(row of n weights, n activations); n is a multiple of 256 */
float es_dot_q2_K_q8_K_ref(int n, const es_block_q2_K *x, const es_block_q8_K *y);
float es_dot_q2_K_q8_K(int n, const es_block_q2_K *x, const es_block_q8_K *y);

/* y[r] = dot(W[r], x) for r in [r0, r1). W is row-major, cols/256 blocks
 * per row. Uses the fastest kernel this CPU supports. */
void es_matvec_q2_K(const es_block_q2_K *W, int cols, int r0, int r1,
                    const es_block_q8_K *xq, float *y);

/* "neon-dotprod" or "scalar" */
const char *es_q2k_kernel_name(void);
/* Force the scalar path (for A/B testing). */
void es_q2k_force_scalar(int on);

#endif
