#include "es_quant.h"
#include "es_cpu.h"
#include "es_iq_tables.h"
#if defined(__x86_64__) && !defined(__EMSCRIPTEN__)
#include <immintrin.h>
#endif

#include <stdlib.h>
#include <string.h>

typedef struct { uint16_t d; int8_t qs[32]; } blk_q8_0;                                   /* 34 */
typedef struct { uint16_t d; uint8_t qs[16]; } blk_q4_0;                                  /* 18 */
typedef struct { uint16_t d, m; uint8_t qs[16]; } blk_q4_1;                               /* 20 */
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; } blk_q5_0;                   /* 22 */
typedef struct { uint16_t d, m; uint8_t qh[4]; uint8_t qs[16]; } blk_q5_1;                /* 24 */
typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; } blk_q3_K; /* 110 */
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; } blk_q4_K;       /* 144 */
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } blk_q5_K; /* 176 */
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } blk_q6_K; /* 210 */

_Static_assert(sizeof(blk_q8_0) == 34, "q8_0");
_Static_assert(sizeof(blk_q3_K) == 110, "q3_K");
_Static_assert(sizeof(blk_q4_K) == 144, "q4_K");
_Static_assert(sizeof(blk_q5_K) == 176, "q5_K");
_Static_assert(sizeof(blk_q6_K) == 210, "q6_K");

int es_quant_supported(uint32_t t) {
    switch (t) {
    case GGML_F32: case GGML_F16: case GGML_Q4_0: case GGML_Q4_1: case GGML_Q5_0:
    case GGML_Q5_1: case GGML_Q8_0: case GGML_Q2_K: case GGML_Q3_K:
    case GGML_Q4_K: case GGML_Q5_K: case GGML_Q6_K: case GGML_IQ4_NL: case GGML_IQ4_XS: case GGML_BF16:
    case GGML_IQ2_XXS: case GGML_IQ2_XS: case GGML_IQ2_S: case GGML_IQ3_XXS: case GGML_IQ3_S: case GGML_IQ1_S: case GGML_IQ1_M:
    case GGML_TQ1_0: case GGML_TQ2_0: case GGML_MXFP4: case GGML_Q1_0: case GGML_Q2_0:
    case GGML_IQ1_XS: case GGML_IQ1_XXS: case GGML_IQ1_XXXS: return 1;
    }
    return 0;
}

size_t es_row_bytes(uint32_t t, int n) {
    switch (t) {
    case GGML_F32: return (size_t)n * 4;
    case GGML_F16: return (size_t)n * 2;
    case GGML_Q4_0: return (size_t)n / 32 * 18;
    case GGML_Q4_1: return (size_t)n / 32 * 20;
    case GGML_Q5_0: return (size_t)n / 32 * 22;
    case GGML_Q5_1: return (size_t)n / 32 * 24;
    case GGML_Q8_0: return (size_t)n / 32 * 34;
    case GGML_Q2_K: return (size_t)n / 256 * 84;
    case GGML_Q3_K: return (size_t)n / 256 * 110;
    case GGML_Q4_K: return (size_t)n / 256 * 144;
    case GGML_Q5_K: return (size_t)n / 256 * 176;
    case GGML_Q6_K: return (size_t)n / 256 * 210;
    case GGML_IQ4_NL: return (size_t)n / 32 * 18;
    case GGML_IQ4_XS: return (size_t)n / 256 * 136;
    case GGML_BF16: return (size_t)n * 2;
    case GGML_IQ2_XXS: return (size_t)n / 256 * 66;
    case GGML_IQ2_XS: return (size_t)n / 256 * 74;
    case GGML_IQ2_S: return (size_t)n / 256 * 82;
    case GGML_IQ3_XXS: return (size_t)n / 256 * 98;
    case GGML_IQ3_S: return (size_t)n / 256 * 110;
    case GGML_IQ1_S: return (size_t)n / 256 * 50;
    case GGML_IQ1_M: return (size_t)n / 256 * 56;
    case GGML_TQ1_0: return (size_t)n / 256 * 54;
    case GGML_TQ2_0: return (size_t)n / 256 * 66;
    case GGML_MXFP4: return (size_t)n / 32 * 17;
    case GGML_Q1_0: return (size_t)n / 128 * 18;
    case GGML_Q2_0: return (size_t)n / 64 * 18;
    case GGML_IQ1_XS: return (size_t)n / 256 * 46;
    case GGML_IQ1_XXS: return (size_t)n / 256 * 42;
    case GGML_IQ1_XXXS: return (size_t)n / 256 * 38;
    }
    return 0;
}

static inline void scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static void deq_q3_K(const blk_q3_K *x, float *y, int nb) {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    for (int i = 0; i < nb; i++) {
        const float d_all = es_fp16_to_fp32(x[i].d);
        const uint8_t *q = x[i].qs, *hm = x[i].hmask;
        uint8_t m = 1;
        uint32_t aux[4];
        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = (const int8_t *)aux;
        int is = 0;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; l++)
                    *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }
}

static void deq_q4_K(const blk_q4_K *x, float *y, int nb) {
    for (int i = 0; i < nb; i++) {
        const float d = es_fp16_to_fp32(x[i].d), min = es_fp16_to_fp32(x[i].dmin);
        const uint8_t *q = x[i].qs;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            scale_min_k4(is, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; l++) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

static void deq_q5_K(const blk_q5_K *x, float *y, int nb) {
    for (int i = 0; i < nb; i++) {
        const float d = es_fp16_to_fp32(x[i].d), min = es_fp16_to_fp32(x[i].dmin);
        const uint8_t *ql = x[i].qs, *qh = x[i].qh;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            scale_min_k4(is, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; l++) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

static void deq_q6_K(const blk_q6_K *x, float *y, int nb) {
    for (int i = 0; i < nb; i++) {
        const float d = es_fp16_to_fp32(x[i].d);
        const uint8_t *ql = x[i].ql, *qh = x[i].qh;
        const int8_t *sc = x[i].scales;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* ---- low-bit formats of the big models (layouts and arithmetic as in ggml's dequantize_row_*) ---- */
static inline float e8m0_half(uint8_t x) {   /* 0.5 * 2^(x-127) */
    uint32_t bits = x < 2 ? 0x00200000u << x : (uint32_t)(x - 1) << 23;
    float f; memcpy(&f, &bits, 4); return f;
}
static inline float h2f(const uint8_t *p) { uint16_t h; memcpy(&h, p, 2); return es_fp16_to_fp32(h); }
#define IQ1S_DELTA 0.125f
/* narrow-index IQ1 (IQ1_XS / XXS / XXXS): for 32-weight sub-block ib of a 256-weight block, the 4 grid rows, the odd scale
 * 2*ls+1 and the delta sign. Every other detail is IQ1_S's. */
static inline int iq1n_bytes(uint32_t t) { return t == GGML_IQ1_XS ? 46 : t == GGML_IQ1_XXS ? 42 : 38; }
static inline float iq1n_sub(uint32_t t, const uint8_t *p, int ib, const int8_t *rows[4], float *delta) {
    const uint8_t *qs = p + 2 + 4 * ib;
    int nib;
    if (t == GGML_IQ1_XXS) {
        const uint8_t h = p[34 + ib];
        for (int l = 0; l < 4; l++) rows[l] = (const int8_t *)(ES_IQ1XXS_GRID + (qs[l] | (((h >> l) & 1) << 8)));
        nib = ((h >> 4) & 7) | ((h >> 4) & 8);
    } else if (t == GGML_IQ1_XS) {
        const uint8_t h = p[34 + ib];
        for (int l = 0; l < 4; l++) rows[l] = (const int8_t *)(ES_IQ1XS_GRID + (qs[l] | (((h >> 2 * l) & 3) << 8)));
        nib = (p[42 + ib / 2] >> (4 * (ib & 1))) & 0xf;
    } else {
        for (int l = 0; l < 4; l++) rows[l] = (const int8_t *)(ES_IQ1XXXS_GRID + qs[l]);
        nib = (p[34 + ib / 2] >> (4 * (ib & 1))) & 0xf;
    }
    *delta = nib & 8 ? -IQ1S_DELTA : IQ1S_DELTA;
    return (float)(2 * (nib & 7) + 1);
}
static void deq_low(uint32_t t, const uint8_t *p, float *y, int n) {
    switch (t) {
    case GGML_IQ1_XS: case GGML_IQ1_XXS: case GGML_IQ1_XXXS: {
        const int bs = iq1n_bytes(t);
        for (int i = 0; i < n / 256; i++, p += bs) {
            const float d = h2f(p);
            for (int ib = 0; ib < 8; ib++) {
                const int8_t *rows[4]; float delta;
                const float dl = d * iq1n_sub(t, p, ib, rows, &delta);
                for (int l = 0; l < 4; l++) for (int j = 0; j < 8; j++) *y++ = dl * ((float)rows[l][j] + delta);
            }
        }
        return;
    }
    case GGML_Q1_0:   /* 128 weights: one bit each, +d or -d */
        for (int i = 0; i < n / 128; i++, p += 18) {
            const float d = h2f(p);
            for (int j = 0; j < 128; j++) *y++ = (p[2 + j / 8] >> (j % 8)) & 1 ? d : -d;
        }
        return;
    case GGML_Q2_0:   /* 64 weights: two bits each, (q - 1) * d */
        for (int i = 0; i < n / 64; i++, p += 18) {
            const float d = h2f(p);
            for (int j = 0; j < 64; j++) *y++ = (float)(((p[2 + j / 4] >> (2 * (j % 4))) & 3) - 1) * d;
        }
        return;
    case GGML_MXFP4:
        for (int i = 0; i < n / 32; i++, p += 17, y += 32) {
            const float d = e8m0_half(p[0]);
            for (int j = 0; j < 16; j++) { y[j] = ES_KVALUES_FP4[p[1 + j] & 15] * d; y[j + 16] = ES_KVALUES_FP4[p[1 + j] >> 4] * d; }
        }
        return;
    case GGML_TQ1_0: {
        static const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};
        for (int i = 0; i < n / 256; i++, p += 54) {
            const uint8_t *qs = p, *qh = p + 48;
            const float d = h2f(p + 52);
            for (int j = 0; j < 32; j += 32)
                for (int k = 0; k < 5; k++) for (int m = 0; m < 32; m++) { const uint8_t q = (uint8_t)(qs[j + m] * pow3[k]); *y++ = (float)((((uint16_t)q * 3) >> 8) - 1) * d; }
            for (int j = 32; j < 48; j += 16)
                for (int k = 0; k < 5; k++) for (int m = 0; m < 16; m++) { const uint8_t q = (uint8_t)(qs[j + m] * pow3[k]); *y++ = (float)((((uint16_t)q * 3) >> 8) - 1) * d; }
            for (int k = 0; k < 4; k++) for (int j = 0; j < 4; j++) { const uint8_t q = (uint8_t)(qh[j] * pow3[k]); *y++ = (float)((((uint16_t)q * 3) >> 8) - 1) * d; }
        }
        return;
    }
    case GGML_TQ2_0:
        for (int i = 0; i < n / 256; i++, p += 66) {
            const float d = h2f(p + 64);
            for (int j = 0; j < 64; j += 32) for (int l = 0; l < 4; l++) for (int m = 0; m < 32; m++) *y++ = (float)(((p[j + m] >> (l * 2)) & 3) - 1) * d;
        }
        return;
    case GGML_IQ2_XXS:
        for (int i = 0; i < n / 256; i++, p += 66) {
            const float d = h2f(p);
            for (int ib = 0; ib < 8; ib++) {
                uint32_t a[2]; memcpy(a, p + 2 + 8 * ib, 8);
                const uint8_t *a8 = (const uint8_t *)a;
                const float db = d * (0.5f + (float)(a[1] >> 28)) * 0.25f;
                for (int l = 0; l < 4; l++, y += 8) {
                    const uint8_t *grid = (const uint8_t *)(ES_IQ2XXS_GRID + a8[l]);
                    const uint8_t signs = ES_KSIGNS_IQ2XS[(a[1] >> 7 * l) & 127];
                    for (int j = 0; j < 8; j++) y[j] = db * grid[j] * (signs & ES_KMASK_IQ2XS[j] ? -1.f : 1.f);
                }
            }
        }
        return;
    case GGML_IQ2_XS:
        for (int i = 0; i < n / 256; i++, p += 74) {
            const float d = h2f(p);
            uint16_t qs[32]; memcpy(qs, p + 2, 64);
            const uint8_t *sc = p + 66;
            for (int ib = 0; ib < 8; ib++) {
                const float db[2] = {d * (0.5f + (sc[ib] & 0xf)) * 0.25f, d * (0.5f + (sc[ib] >> 4)) * 0.25f};
                for (int l = 0; l < 4; l++, y += 8) {
                    const uint8_t *grid = (const uint8_t *)(ES_IQ2XS_GRID + (qs[4 * ib + l] & 511));
                    const uint8_t signs = ES_KSIGNS_IQ2XS[qs[4 * ib + l] >> 9];
                    for (int j = 0; j < 8; j++) y[j] = db[l / 2] * grid[j] * (signs & ES_KMASK_IQ2XS[j] ? -1.f : 1.f);
                }
            }
        }
        return;
    case GGML_IQ2_S:
        for (int i = 0; i < n / 256; i++, p += 82) {
            const float d = h2f(p);
            const uint8_t *qs = p + 2, *signs = p + 2 + 32, *qh = p + 66, *sc = p + 74;
            for (int ib = 0; ib < 8; ib++) {
                const float db[2] = {d * (0.5f + (sc[ib] & 0xf)) * 0.25f, d * (0.5f + (sc[ib] >> 4)) * 0.25f};
                for (int l = 0; l < 4; l++, y += 8) {
                    const uint8_t *grid = (const uint8_t *)(ES_IQ2S_GRID + (qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300)));
                    for (int j = 0; j < 8; j++) y[j] = db[l / 2] * grid[j] * (signs[l] & ES_KMASK_IQ2XS[j] ? -1.f : 1.f);
                }
                qs += 4; signs += 4;
            }
        }
        return;
    case GGML_IQ3_XXS:
        for (int i = 0; i < n / 256; i++, p += 98) {
            const float d = h2f(p);
            const uint8_t *qs = p + 2, *ss = p + 2 + 64;
            for (int ib = 0; ib < 8; ib++) {
                uint32_t a; memcpy(&a, ss + 4 * ib, 4);
                const float db = d * (0.5f + (float)(a >> 28)) * 0.5f;
                for (int l = 0; l < 4; l++, y += 8) {
                    const uint8_t signs = ES_KSIGNS_IQ2XS[(a >> 7 * l) & 127];
                    const uint8_t *g1 = (const uint8_t *)(ES_IQ3XXS_GRID + qs[2 * l]), *g2 = (const uint8_t *)(ES_IQ3XXS_GRID + qs[2 * l + 1]);
                    for (int j = 0; j < 4; j++) {
                        y[j] = db * g1[j] * (signs & ES_KMASK_IQ2XS[j] ? -1.f : 1.f);
                        y[j + 4] = db * g2[j] * (signs & ES_KMASK_IQ2XS[j + 4] ? -1.f : 1.f);
                    }
                }
                qs += 8;
            }
        }
        return;
    case GGML_IQ3_S:
        for (int i = 0; i < n / 256; i++, p += 110) {
            const float d = h2f(p);
            const uint8_t *qs = p + 2, *qh = p + 66, *signs = p + 74, *sc = p + 106;
            for (int ib = 0; ib < 8; ib += 2) {
                for (int h = 0; h < 2; h++) {
                    const float db = d * (1 + 2 * (h ? sc[ib / 2] >> 4 : sc[ib / 2] & 0xf));
                    for (int l = 0; l < 4; l++, y += 8) {
                        const uint8_t *g1 = (const uint8_t *)(ES_IQ3S_GRID + (qs[2 * l] | ((qh[h] << (8 - 2 * l)) & 256)));
                        const uint8_t *g2 = (const uint8_t *)(ES_IQ3S_GRID + (qs[2 * l + 1] | ((qh[h] << (7 - 2 * l)) & 256)));
                        for (int j = 0; j < 4; j++) {
                            y[j] = db * g1[j] * (signs[l] & ES_KMASK_IQ2XS[j] ? -1.f : 1.f);
                            y[j + 4] = db * g2[j] * (signs[l] & ES_KMASK_IQ2XS[j + 4] ? -1.f : 1.f);
                        }
                    }
                    qs += 8; signs += 4;
                }
                qh += 2;
            }
        }
        return;
    case GGML_IQ1_S:
        for (int i = 0; i < n / 256; i++, p += 50) {
            const float d = h2f(p);
            const uint8_t *qs = p + 2;
            uint16_t qh[8]; memcpy(qh, p + 34, 16);
            for (int ib = 0; ib < 8; ib++) {
                const float dl = d * (2 * ((qh[ib] >> 12) & 7) + 1);
                const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
                for (int l = 0; l < 4; l++, y += 8) {
                    const int8_t *grid = (const int8_t *)(ES_IQ1S_GRID + (qs[l] | (((qh[ib] >> 3 * l) & 7) << 8)));
                    for (int j = 0; j < 8; j++) y[j] = dl * (grid[j] + delta);
                }
                qs += 4;
            }
        }
        return;
    case GGML_IQ1_M:
        for (int i = 0; i < n / 256; i++, p += 56) {
            const uint8_t *qs = p, *qh = p + 32;
            uint16_t sc[4]; memcpy(sc, p + 48, 8);
            const uint16_t s16 = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
            const float d = es_fp16_to_fp32(s16);
            for (int ib = 0; ib < 8; ib++) {
                const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
                const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
                const uint16_t idx[4] = {(uint16_t)(qs[0] | ((qh[0] << 8) & 0x700)), (uint16_t)(qs[1] | ((qh[0] << 4) & 0x700)),
                                         (uint16_t)(qs[2] | ((qh[1] << 8) & 0x700)), (uint16_t)(qs[3] | ((qh[1] << 4) & 0x700))};
                const float delta[4] = {qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA, qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA,
                                        qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA, qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA};
                for (int l = 0; l < 4; l++, y += 8) {
                    const int8_t *grid = (const int8_t *)(ES_IQ1S_GRID + idx[l]);
                    const float dl = l < 2 ? dl1 : dl2;
                    for (int j = 0; j < 8; j++) y[j] = dl * (grid[j] + delta[l]);
                }
                qs += 4; qh += 2;
            }
        }
        return;
    }
}

static const int8_t KV_IQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
void es_dequant_row(uint32_t t, const void *src, float *y, int n) {
    switch (t) {
    case GGML_BF16: {
        const uint16_t *h = src;
        for (int i = 0; i < n; i++) { uint32_t u = (uint32_t)h[i] << 16; memcpy(&y[i], &u, 4); }
        return;
    }
    case GGML_IQ4_NL: {
        const uint8_t *p = src;
        for (int i = 0; i < n / 32; i++, p += 18, y += 32) {
            uint16_t dh; memcpy(&dh, p, 2);
            const float d = es_fp16_to_fp32(dh);
            for (int j = 0; j < 16; j++) { y[j] = d * KV_IQ4NL[p[2 + j] & 15]; y[j + 16] = d * KV_IQ4NL[p[2 + j] >> 4]; }
        }
        return;
    }
    case GGML_IQ4_XS: {
        const uint8_t *p = src;
        for (int i = 0; i < n / 256; i++, p += 136) {
            uint16_t dh, sh; memcpy(&dh, p, 2); memcpy(&sh, p + 2, 2);
            const uint8_t *sl = p + 4, *qs = p + 8;
            const float d = es_fp16_to_fp32(dh);
            for (int ib = 0; ib < 8; ib++) {
                const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
                const float dl = d * (ls - 32);
                for (int j = 0; j < 16; j++) { y[j] = dl * KV_IQ4NL[qs[j] & 15]; y[j + 16] = dl * KV_IQ4NL[qs[j] >> 4]; }
                y += 32; qs += 16;
            }
        }
        return;
    }
    case GGML_F32: memcpy(y, src, sizeof(float) * (size_t)n); return;
    case GGML_F16: {
        const uint16_t *h = src;
        for (int i = 0; i < n; i++) y[i] = es_fp16_to_fp32(h[i]);
        return;
    }
    case GGML_Q8_0: {
        const blk_q8_0 *b = src;
        for (int i = 0; i < n / 32; i++) {
            const float d = es_fp16_to_fp32(b[i].d);
            for (int l = 0; l < 32; l++) *y++ = d * b[i].qs[l];
        }
        return;
    }
    case GGML_Q4_0: case GGML_Q4_1: {
        const int has_m = t == GGML_Q4_1;
        const uint8_t *p = src;
        const size_t bs = has_m ? 20 : 18;
        for (int i = 0; i < n / 32; i++, p += bs, y += 32) {
            uint16_t dh, mh = 0;
            memcpy(&dh, p, 2);
            if (has_m) memcpy(&mh, p + 2, 2);
            const float d = es_fp16_to_fp32(dh), m = has_m ? es_fp16_to_fp32(mh) : 0;
            const uint8_t *qs = p + (has_m ? 4 : 2);
            for (int j = 0; j < 16; j++) {
                const int x0 = qs[j] & 0xF, x1 = qs[j] >> 4;
                y[j] = has_m ? x0 * d + m : (x0 - 8) * d;
                y[j + 16] = has_m ? x1 * d + m : (x1 - 8) * d;
            }
        }
        return;
    }
    case GGML_Q5_0: case GGML_Q5_1: {
        const int has_m = t == GGML_Q5_1;
        const uint8_t *p = src;
        const size_t bs = has_m ? 24 : 22;
        for (int i = 0; i < n / 32; i++, p += bs, y += 32) {
            uint16_t dh, mh = 0;
            uint32_t qh;
            memcpy(&dh, p, 2);
            if (has_m) memcpy(&mh, p + 2, 2);
            memcpy(&qh, p + (has_m ? 4 : 2), 4);
            const float d = es_fp16_to_fp32(dh), m = has_m ? es_fp16_to_fp32(mh) : 0;
            const uint8_t *qs = p + (has_m ? 8 : 6);
            for (int j = 0; j < 16; j++) {
                const uint8_t xh0 = ((qh >> j) << 4) & 0x10;
                const uint8_t xh1 = (qh >> (j + 12)) & 0x10;
                const int x0 = (qs[j] & 0xF) | xh0, x1 = (qs[j] >> 4) | xh1;
                y[j] = has_m ? x0 * d + m : (x0 - 16) * d;
                y[j + 16] = has_m ? x1 * d + m : (x1 - 16) * d;
            }
        }
        return;
    }
    case GGML_Q2_K: es_dequantize_q2_K(src, y, n); return;
    case GGML_Q3_K: deq_q3_K(src, y, n / 256); return;
    case GGML_Q4_K: deq_q4_K(src, y, n / 256); return;
    case GGML_Q5_K: deq_q5_K(src, y, n / 256); return;
    case GGML_Q6_K: deq_q6_K(src, y, n / 256); return;
    case GGML_MXFP4: case GGML_TQ1_0: case GGML_TQ2_0: case GGML_IQ2_XXS: case GGML_IQ2_XS: case GGML_IQ2_S:
    case GGML_IQ3_XXS: case GGML_IQ3_S: case GGML_IQ1_S: case GGML_IQ1_M: case GGML_Q1_0: case GGML_Q2_0:
    case GGML_IQ1_XS: case GGML_IQ1_XXS: case GGML_IQ1_XXXS: deq_low(t, src, y, n); return;
    }
}

static es_blk_q80 *ARENA;
static size_t ARENA_CAP, ARENA_USE;   /* in blocks */
void es_act_arena_init(size_t bytes) { free(ARENA); ARENA_CAP = bytes / sizeof(es_blk_q80); ARENA = malloc(ARENA_CAP * sizeof(es_blk_q80)); ARENA_USE = 0; if (!ARENA) ARENA_CAP = 0; }
void es_act_arena_reset(void) { ARENA_USE = 0; }

static void quantize_q80(const float *x, es_blk_q80 *y, int n) {
    for (int b = 0; b < n / 32; b++, x += 32) {
        float amax = 0;
        for (int i = 0; i < 32; i++) { const float v = x[i] < 0 ? -x[i] : x[i]; amax = v > amax ? v : amax; }
        const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        y[b].d = d;
        for (int i = 0; i < 32; i++) { const float v = x[i] * id; y[b].qs[i] = (int8_t)(int)(v < 0 ? v - 0.5f : v + 0.5f); }
    }
}

void es_act_prepare(es_act *a, const float *x, int cols, es_block_q8_K *q8_buf) {
    static int float_only = -1; /* ES_FLOAT_ACT=1: exact float path (slow, for testing) */
    if (float_only < 0) float_only = getenv("ES_FLOAT_ACT") != NULL;
    if (float_only) q8_buf = NULL;
    a->x = x;
    a->cols = cols;
    a->q80 = NULL;
    if (!float_only && cols % 32 == 0 && ARENA_USE + (size_t)cols / 32 <= ARENA_CAP) {
        a->q80 = ARENA + ARENA_USE; ARENA_USE += (size_t)cols / 32;
        quantize_q80(x, a->q80, cols);
    }
    a->q8 = q8_buf;
    if (q8_buf && cols % 256 == 0) es_quantize_q8_K(x, q8_buf, cols);
}

static inline float dot_f32(const float *a, const float *b, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; i++) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}


/* ---- integer dot kernels (activations are int8; weights stay packed) ---- */
static inline float fp16f(uint16_t h) {   /* branch-free, no inf/nan (never appear in scales) */
    uint32_t u = (uint32_t)(h & 0x7fff) << 13; float f; memcpy(&f, &u, 4);
    f *= 0x1p112f;
    return (h & 0x8000) ? -f : f;
}
#ifdef __wasm_simd128__
#include <wasm_simd128.h>
/* 16 int8 x 16 int8 -> 4 partial int32 sums */
static inline v128_t wdot16(v128_t a, v128_t b) {
    return wasm_i32x4_add(wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(a), wasm_i16x8_extend_low_i8x16(b)),
                          wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(a), wasm_i16x8_extend_high_i8x16(b)));
}
static inline int whsum(v128_t v) { return wasm_i32x4_extract_lane(v, 0) + wasm_i32x4_extract_lane(v, 1) + wasm_i32x4_extract_lane(v, 2) + wasm_i32x4_extract_lane(v, 3); }
#define LD(p) wasm_v128_load((const void *)(p))
#endif
static inline int dot8(const int8_t *a, const int8_t *b, int n) {
#ifdef __wasm_simd128__
    v128_t acc = wasm_i32x4_splat(0);
    for (int i = 0; i < n; i += 16) acc = wasm_i32x4_add(acc, wdot16(LD(a + i), LD(b + i)));
    return whsum(acc);
#else
    int s = 0; for (int i = 0; i < n; i++) s += (int)a[i] * (int)b[i]; return s;
#endif
}


/* ---- the low-bit formats of the big models with 8-bit activations: every 32 weights are decoded to int8 (codebook values,
 * signs) and multiplied with the 32 activation values in integers; only the scales are float. Same numbers as
 * decoding to float, apart from the activations' 8-bit rounding (as for every other integer kernel here). ---- */
#if defined(__x86_64__) && !defined(__EMSCRIPTEN__)
static int LB_AVX2 = -1;
__attribute__((target("avx2"))) static int dot32_avx2(const int8_t *w, const int8_t *a) {
    const __m256i x = _mm256_loadu_si256((const __m256i *)w), y = _mm256_loadu_si256((const __m256i *)a);
    const __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x)), _mm256_set1_epi16(1));
    const __m128i h = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
    const __m128i q = _mm_add_epi32(h, _mm_shuffle_epi32(h, 0x4e));
    return _mm_cvtsi128_si32(_mm_add_epi32(q, _mm_shuffle_epi32(q, 0xb1)));
}
#endif

/* 32 weights from 16 bytes of 4-bit indices into a 16-entry table (MXFP4, IQ4): one table lookup instruction per 16 weights */
#if defined(__x86_64__) && !defined(__EMSCRIPTEN__)
__attribute__((target("ssse3"))) static void lut32_ssse3(const uint8_t *qs, const int8_t *lut, int8_t *q) {
    const __m128i t = _mm_loadu_si128((const __m128i *)lut), x = _mm_loadu_si128((const __m128i *)qs), m = _mm_set1_epi8(15);
    _mm_storeu_si128((__m128i *)q, _mm_shuffle_epi8(t, _mm_and_si128(x, m)));
    _mm_storeu_si128((__m128i *)(q + 16), _mm_shuffle_epi8(t, _mm_and_si128(_mm_srli_epi16(x, 4), m)));
}
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
static inline void lut32(const uint8_t *qs, const int8_t *lut, int8_t *q) {
#if defined(__x86_64__) && !defined(__EMSCRIPTEN__)
    if (LB_AVX2 > 0) { lut32_ssse3(qs, lut, q); return; }
#endif
#if defined(__aarch64__)
    const int8x16_t t = vld1q_s8(lut);
    const uint8x16_t x = vld1q_u8(qs);
    vst1q_s8(q, vqtbl1q_s8(t, vandq_u8(x, vdupq_n_u8(15))));
    vst1q_s8(q + 16, vqtbl1q_s8(t, vshrq_n_u8(x, 4)));
    return;
#endif
    for (int j = 0; j < 16; j++) { q[j] = lut[qs[j] & 15]; q[j + 16] = lut[qs[j] >> 4]; }
}
/* 8 codebook values with 8 sign bits, without branches: per byte (g ^ m) + (m & 1) negates where m = 0xFF.
 * The IQ2/IQ3 codebooks have no zero entries, so no carry crosses a byte. */
static uint64_t SIGNMASK[256];
static void signmask_init(void) {
    if (SIGNMASK[255]) return;
    for (int b = 0; b < 256; b++) { uint64_t m = 0; for (int j = 0; j < 8; j++) if (b >> j & 1) m |= 0xFFull << (8 * j); SIGNMASK[b] = m; }
}
static inline void sgrid8(int8_t *q, uint64_t g, uint8_t signs) {
    const uint64_t m = SIGNMASK[signs];
    g = (g ^ m) + (m & 0x0101010101010101ull);
    memcpy(q, &g, 8);
}
#if defined(__aarch64__)
static inline int dot32_neon(const int8_t *w, const int8_t *a) {
    const int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16), a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16);
    int16x8_t p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(a0)); p0 = vmlal_high_s8(p0, w0, a0);
    int16x8_t p1 = vmull_s8(vget_low_s8(w1), vget_low_s8(a1)); p1 = vmlal_high_s8(p1, w1, a1);
    return vaddvq_s32(vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}
#endif
static inline int dot_n(const int8_t *w, const int8_t *a, int n) {
#if defined(__x86_64__) && !defined(__EMSCRIPTEN__)
    if (n == 32 && LB_AVX2 > 0) return dot32_avx2(w, a);
#endif
#if defined(__aarch64__)
    if (n == 32) return dot32_neon(w, a);
#endif
    if (n % 16) { int t = 0; for (int i = 0; i < n; i++) t += (int)w[i] * (int)a[i]; return t; }
    return dot8(w, a, n);
}
static inline int asum(const int8_t *a, int n) { int s = 0; for (int i = 0; i < n; i++) s += a[i]; return s; }
#define ES_MULTI_MAX 16
static int getenv_cached_no_multi(void) { static int v = -1; if (v < 0) v = getenv("ES_NO_MULTI") != NULL; return v; }
static int lowbit_ok(uint32_t t) {
    return t == GGML_MXFP4 || t == GGML_IQ4_NL || t == GGML_IQ4_XS || t == GGML_IQ2_XXS || t == GGML_IQ2_XS || t == GGML_IQ2_S ||
           t == GGML_IQ3_XXS || t == GGML_IQ3_S || t == GGML_IQ1_S || t == GGML_IQ1_M || t == GGML_Q1_0 || t == GGML_Q2_0 ||
           t == GGML_IQ1_XS || t == GGML_IQ1_XXS || t == GGML_IQ1_XXXS;
}
static void mv_lowbit_n(uint32_t t, const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *const *ys, int na, float *out, size_t ostride) {
#if defined(__x86_64__) && !defined(__EMSCRIPTEN__)
    if (LB_AVX2 < 0) { __builtin_cpu_init(); LB_AVX2 = __builtin_cpu_supports("avx2") && !getenv("ES_NO_AVX2"); }
#endif
    const size_t rb = es_row_bytes(t, cols);
    int8_t q[32] __attribute__((aligned(32)));
    signmask_init();
    for (int r = r0; r < r1; r++) {
        const uint8_t *p = w + rb * (size_t)r;
        float s[ES_MULTI_MAX] = {0};
        switch (t) {
        case GGML_IQ1_XS: case GGML_IQ1_XXS: case GGML_IQ1_XXXS: {
            const int bs = iq1n_bytes(t);
            for (int i = 0; i < cols / 256; i++, p += bs) {
                const float d = h2f(p);
                for (int ib = 0; ib < 8; ib++) {
                    const int8_t *rows[4]; float delta;
                    const float dl = d * iq1n_sub(t, p, ib, rows, &delta);
                    for (int l = 0; l < 4; l++) memcpy(q + 8 * l, rows[l], 8);
                    for (int k = 0; k < na; k++) { const es_blk_q80 *a = &ys[k][i * 8 + ib]; s[k] += dl * a->d * ((float)dot_n(q, a->qs, 32) + delta * (float)asum(a->qs, 32)); }
                }
            }
            break;
        }
        case GGML_Q1_0:
            for (int i = 0; i < cols / 128; i++, p += 18) {
                const float d = h2f(p);
                for (int ib = 0; ib < 4; ib++) {
                    for (int l = 0; l < 4; l++) sgrid8(q + 8 * l, 0x0101010101010101ull, (uint8_t)~p[2 + 4 * ib + l]);   /* bit set: +1 */
                    for (int k = 0; k < na; k++) { const es_blk_q80 *a = &ys[k][i * 4 + ib]; s[k] += d * a->d * (float)dot_n(q, a->qs, 32); }
                }
            }
            break;
        case GGML_Q2_0:
            for (int i = 0; i < cols / 64; i++, p += 18) {
                const float d = h2f(p);
                for (int ib = 0; ib < 2; ib++) {
                    const uint8_t *qs = p + 2 + 8 * ib;
                    for (int j = 0; j < 32; j++) q[j] = (int8_t)(((qs[j / 4] >> (2 * (j % 4))) & 3) - 1);
                    for (int k = 0; k < na; k++) { const es_blk_q80 *a = &ys[k][i * 2 + ib]; s[k] += d * a->d * (float)dot_n(q, a->qs, 32); }
                }
            }
            break;
        case GGML_MXFP4:
            for (int b = 0; b < cols / 32; b++, p += 17) {
                lut32(p + 1, ES_KVALUES_FP4, q);
                { const float sc = e8m0_half(p[0]); for (int k = 0; k < na; k++) s[k] += sc * ys[k][b].d * (float)dot_n(q, ys[k][b].qs, 32); }
            }
            break;
        case GGML_IQ4_NL:
            for (int b = 0; b < cols / 32; b++, p += 18) {
                lut32(p + 2, KV_IQ4NL, q);
                { const float sc = h2f(p); for (int k = 0; k < na; k++) s[k] += sc * ys[k][b].d * (float)dot_n(q, ys[k][b].qs, 32); }
            }
            break;
        case GGML_IQ4_XS:
            for (int i = 0; i < cols / 256; i++, p += 136) {
                uint16_t sh; memcpy(&sh, p + 2, 2);
                const float d = h2f(p);
                const uint8_t *sl = p + 4, *qs = p + 8;
                for (int ib = 0; ib < 8; ib++, qs += 16) {
                    const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
                    for (int j = 0; j < 16; j++) { q[j] = KV_IQ4NL[qs[j] & 15]; q[j + 16] = KV_IQ4NL[qs[j] >> 4]; }
                                        for (int k = 0; k < na; k++) { const es_blk_q80 *a = &ys[k][i * 8 + ib]; s[k] += d * (float)(ls - 32) * a->d * (float)dot_n(q, a->qs, 32); }
                }
            }
            break;
        case GGML_IQ2_XXS:
            for (int i = 0; i < cols / 256; i++, p += 66) {
                const float d = h2f(p);
                for (int ib = 0; ib < 8; ib++) {
                    uint32_t a2[2]; memcpy(a2, p + 2 + 8 * ib, 8);
                    const uint8_t *a8 = (const uint8_t *)a2;
                    for (int l = 0; l < 4; l++) sgrid8(q + 8 * l, ES_IQ2XXS_GRID[a8[l]], ES_KSIGNS_IQ2XS[(a2[1] >> 7 * l) & 127]);
                                        for (int k = 0; k < na; k++) { const es_blk_q80 *a = &ys[k][i * 8 + ib]; s[k] += d * (0.5f + (float)(a2[1] >> 28)) * 0.25f * a->d * (float)dot_n(q, a->qs, 32); }
                }
            }
            break;
        case GGML_IQ2_XS:
            for (int i = 0; i < cols / 256; i++, p += 74) {
                const float d = h2f(p);
                uint16_t qs[32]; memcpy(qs, p + 2, 64);
                const uint8_t *sc = p + 66;
                for (int ib = 0; ib < 8; ib++) {
                    for (int l = 0; l < 4; l++) sgrid8(q + 8 * l, ES_IQ2XS_GRID[qs[4 * ib + l] & 511], ES_KSIGNS_IQ2XS[qs[4 * ib + l] >> 9]);
                                        for (int k = 0; k < na; k++) { const es_blk_q80 *a = &ys[k][i * 8 + ib]; s[k] += d * 0.25f * a->d * ((0.5f + (sc[ib] & 0xf)) * (float)dot_n(q, a->qs, 16) + (0.5f + (sc[ib] >> 4)) * (float)dot_n(q + 16, a->qs + 16, 16)); }
                }
            }
            break;
        case GGML_IQ2_S:
            for (int i = 0; i < cols / 256; i++, p += 82) {
                const float d = h2f(p);
                const uint8_t *qs = p + 2, *signs = p + 34, *qh = p + 66, *sc = p + 74;
                for (int ib = 0; ib < 8; ib++, qs += 4, signs += 4) {
                    for (int l = 0; l < 4; l++) sgrid8(q + 8 * l, ES_IQ2S_GRID[qs[l] | ((qh[ib] << (8 - 2 * l)) & 0x300)], signs[l]);
                                        for (int k = 0; k < na; k++) { const es_blk_q80 *a = &ys[k][i * 8 + ib]; s[k] += d * 0.25f * a->d * ((0.5f + (sc[ib] & 0xf)) * (float)dot_n(q, a->qs, 16) + (0.5f + (sc[ib] >> 4)) * (float)dot_n(q + 16, a->qs + 16, 16)); }
                }
            }
            break;
        case GGML_IQ3_XXS:
            for (int i = 0; i < cols / 256; i++, p += 98) {
                const float d = h2f(p);
                const uint8_t *qs = p + 2, *ss = p + 66;
                for (int ib = 0; ib < 8; ib++, qs += 8) {
                    uint32_t a2; memcpy(&a2, ss + 4 * ib, 4);
                    for (int l = 0; l < 4; l++)
                        sgrid8(q + 8 * l, (uint64_t)ES_IQ3XXS_GRID[qs[2 * l]] | (uint64_t)ES_IQ3XXS_GRID[qs[2 * l + 1]] << 32, ES_KSIGNS_IQ2XS[(a2 >> 7 * l) & 127]);
                                        for (int k = 0; k < na; k++) { const es_blk_q80 *a = &ys[k][i * 8 + ib]; s[k] += d * (0.5f + (float)(a2 >> 28)) * 0.5f * a->d * (float)dot_n(q, a->qs, 32); }
                }
            }
            break;
        case GGML_IQ3_S:
            for (int i = 0; i < cols / 256; i++, p += 110) {
                const float d = h2f(p);
                const uint8_t *qs = p + 2, *qh = p + 66, *signs = p + 74, *sc = p + 106;
                for (int ib = 0; ib < 8; ib++, qs += 8, signs += 4) {
                    const int h = ib & 1, hb = qh[ib];
                    for (int l = 0; l < 4; l++)
                        sgrid8(q + 8 * l, (uint64_t)ES_IQ3S_GRID[qs[2 * l] | ((hb << (8 - 2 * l)) & 256)] | (uint64_t)ES_IQ3S_GRID[qs[2 * l + 1] | ((hb << (7 - 2 * l)) & 256)] << 32, signs[l]);
                                        for (int k = 0; k < na; k++) { const es_blk_q80 *a = &ys[k][i * 8 + ib]; s[k] += d * (float)(1 + 2 * (h ? sc[ib / 2] >> 4 : sc[ib / 2] & 0xf)) * a->d * (float)dot_n(q, a->qs, 32); }
                }
            }
            break;
        case GGML_IQ1_S:
            for (int i = 0; i < cols / 256; i++, p += 50) {
                const float d = h2f(p);
                const uint8_t *qs = p + 2;
                uint16_t qh[8]; memcpy(qh, p + 34, 16);
                for (int ib = 0; ib < 8; ib++, qs += 4) {
                    for (int l = 0; l < 4; l++) memcpy(q + 8 * l, ES_IQ1S_GRID + (qs[l] | (((qh[ib] >> 3 * l) & 7) << 8)), 8);
                    const float delta = qh[ib] & 0x8000 ? -IQ1S_DELTA : IQ1S_DELTA;
                    for (int k = 0; k < na; k++) { const es_blk_q80 *a = &ys[k][i * 8 + ib]; s[k] += d * (float)(2 * ((qh[ib] >> 12) & 7) + 1) * a->d * ((float)dot_n(q, a->qs, 32) + delta * (float)asum(a->qs, 32)); }
                }
            }
            break;
        case GGML_IQ1_M:
            for (int i = 0; i < cols / 256; i++, p += 56) {
                const uint8_t *qs = p, *qh = p + 32;
                uint16_t sc[4]; memcpy(sc, p + 48, 8);
                const float d = es_fp16_to_fp32((uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)));
                for (int ib = 0; ib < 8; ib++, qs += 4, qh += 2) {
                    const int idx[4] = {qs[0] | ((qh[0] << 8) & 0x700), qs[1] | ((qh[0] << 4) & 0x700), qs[2] | ((qh[1] << 8) & 0x700), qs[3] | ((qh[1] << 4) & 0x700)};
                    const float delta[4] = {qh[0] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA, qh[0] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA,
                                            qh[1] & 0x08 ? -IQ1S_DELTA : IQ1S_DELTA, qh[1] & 0x80 ? -IQ1S_DELTA : IQ1S_DELTA};
                    for (int l = 0; l < 4; l++) memcpy(q + 8 * l, ES_IQ1S_GRID + idx[l], 8);
                    const float dl1 = (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1), dl2 = (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
                    for (int k = 0; k < na; k++) {
                        const es_blk_q80 *a = &ys[k][i * 8 + ib];
                        float part[2] = {0, 0};
                        for (int l = 0; l < 4; l++) part[l / 2] += (float)dot_n(q + 8 * l, a->qs + 8 * l, 8) + delta[l] * (float)asum(a->qs + 8 * l, 8);
                        s[k] += a->d * d * (dl1 * part[0] + dl2 * part[1]);
                    }
                }
            }
            break;
        }
        for (int k = 0; k < na; k++) out[(size_t)k * ostride + r] = s[k];
    }
}
static void mv_lowbit(uint32_t t, const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const es_blk_q80 *ys[1] = {y};
    mv_lowbit_n(t, w, cols, r0, r1, ys, 1, out, 0);
}

static void mv_q8_0(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 34;
    for (int r = r0; r < r1; r++) {
        const blk_q8_0 *x = (const blk_q8_0 *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) s += fp16f(x[b].d) * y[b].d * (float)dot8(x[b].qs, y[b].qs, 32);
        out[r] = s;
    }
}
static void mv_q4_0(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 18;
    for (int r = r0; r < r1; r++) {
        const blk_q4_0 *x = (const blk_q4_0 *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
#ifdef __wasm_simd128__
            const v128_t q = LD(x[b].qs), m = wasm_i8x16_splat(15), e = wasm_i8x16_splat(8);
            const v128_t lo = wasm_i8x16_sub(wasm_v128_and(q, m), e), hi = wasm_i8x16_sub(wasm_u8x16_shr(q, 4), e);
            const int a = whsum(wasm_i32x4_add(wdot16(lo, LD(y[b].qs)), wdot16(hi, LD(y[b].qs + 16))));
#else
            int a = 0;
            for (int j = 0; j < 16; j++) a += ((int)(x[b].qs[j] & 15) - 8) * y[b].qs[j] + ((int)(x[b].qs[j] >> 4) - 8) * y[b].qs[j + 16];
#endif
            s += fp16f(x[b].d) * y[b].d * (float)a;
        }
        out[r] = s;
    }
}
static void mv_q5_0(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 22;
    for (int r = r0; r < r1; r++) {
        const blk_q5_0 *x = (const blk_q5_0 *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            uint32_t qh; memcpy(&qh, x[b].qh, 4);
#ifdef __wasm_simd128__
            const v128_t qv = wasm_v128_load32_splat(x[b].qh), q = LD(x[b].qs), m15 = wasm_i8x16_splat(15), b16 = wasm_i8x16_splat(16), z = wasm_i8x16_splat(0);
            const v128_t bits = wasm_i8x16_make(1, 2, 4, 8, 16, 32, 64, (int8_t)128, 1, 2, 4, 8, 16, 32, 64, (int8_t)128);
            const v128_t h0 = wasm_v128_and(wasm_i8x16_ne(wasm_v128_and(wasm_i8x16_swizzle(qv, wasm_i8x16_make(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)), bits), z), b16);
            const v128_t h1 = wasm_v128_and(wasm_i8x16_ne(wasm_v128_and(wasm_i8x16_swizzle(qv, wasm_i8x16_make(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)), bits), z), b16);
            const v128_t lo = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(q, m15), h0), b16), hi = wasm_i8x16_sub(wasm_v128_or(wasm_u8x16_shr(q, 4), h1), b16);
            const int a = whsum(wasm_i32x4_add(wdot16(lo, LD(y[b].qs)), wdot16(hi, LD(y[b].qs + 16))));
#else
            int a = 0;
            for (int j = 0; j < 16; j++) {
                const int x0 = (x[b].qs[j] & 15) | (int)(((qh >> j) << 4) & 0x10), x1 = (x[b].qs[j] >> 4) | (int)((qh >> (j + 12)) & 0x10);
                a += (x0 - 16) * y[b].qs[j] + (x1 - 16) * y[b].qs[j + 16];
            }
#endif
            s += fp16f(x[b].d) * y[b].d * (float)a;
        }
        out[r] = s;
    }
}
static void mv_q4_K(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 144;
    for (int r = r0; r < r1; r++) {
        const blk_q4_K *x = (const blk_q4_K *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; j++) scale_min_k4(j, x[b].scales, &sc[j], &mn[j]);
            int dot = 0, mins = 0; const uint8_t *q = x[b].qs; const int8_t *q8 = y[b].qs;
            for (int c = 0; c < 4; c++, q += 32, q8 += 64) {
#ifdef __wasm_simd128__
                const v128_t m = wasm_i8x16_splat(15), qa = LD(q), qb = LD(q + 16);
                const int lo = whsum(wasm_i32x4_add(wdot16(wasm_v128_and(qa, m), LD(q8)), wdot16(wasm_v128_and(qb, m), LD(q8 + 16))));
                const int hi = whsum(wasm_i32x4_add(wdot16(wasm_u8x16_shr(qa, 4), LD(q8 + 32)), wdot16(wasm_u8x16_shr(qb, 4), LD(q8 + 48))));
#else
                int lo = 0, hi = 0;
                for (int l = 0; l < 32; l++) { lo += (q[l] & 15) * q8[l]; hi += (q[l] >> 4) * q8[l + 32]; }
#endif
                dot += sc[2 * c] * lo + sc[2 * c + 1] * hi;
                mins += mn[2 * c] * (y[b].bsums[4 * c] + y[b].bsums[4 * c + 1]) + mn[2 * c + 1] * (y[b].bsums[4 * c + 2] + y[b].bsums[4 * c + 3]);
            }
            s += y[b].d * (fp16f(x[b].d) * (float)dot - fp16f(x[b].dmin) * (float)mins);
        }
        out[r] = s;
    }
}
static void mv_q6_K(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 210;
    for (int r = r0; r < r1; r++) {
        const blk_q6_K *x = (const blk_q6_K *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            const uint8_t *ql = x[b].ql, *qh = x[b].qh; const int8_t *sc = x[b].scales, *q8 = y[b].qs;
            int tot = 0;
            for (int h = 0; h < 2; h++, ql += 64, qh += 32, sc += 8, q8 += 128) {
                int p[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#ifdef __wasm_simd128__
                const v128_t m15 = wasm_i8x16_splat(15), m3 = wasm_i8x16_splat(3), c32 = wasm_i8x16_splat(32);
                for (int is = 0; is < 2; is++) {
                    const int l = is * 16;
                    const v128_t a0 = LD(ql + l), a1 = LD(ql + l + 32), h = LD(qh + l);
                    const v128_t q1 = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(a0, m15), wasm_i8x16_shl(wasm_v128_and(h, m3), 4)), c32);
                    const v128_t q2 = wasm_i8x16_sub(wasm_v128_or(wasm_v128_and(a1, m15), wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(h, 2), m3), 4)), c32);
                    const v128_t q3 = wasm_i8x16_sub(wasm_v128_or(wasm_u8x16_shr(a0, 4), wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(h, 4), m3), 4)), c32);
                    const v128_t q4 = wasm_i8x16_sub(wasm_v128_or(wasm_u8x16_shr(a1, 4), wasm_i8x16_shl(wasm_u8x16_shr(h, 6), 4)), c32);
                    p[is] = whsum(wdot16(q1, LD(q8 + l))); p[is + 2] = whsum(wdot16(q2, LD(q8 + l + 32)));
                    p[is + 4] = whsum(wdot16(q3, LD(q8 + l + 64))); p[is + 6] = whsum(wdot16(q4, LD(q8 + l + 96)));
                }
#else
                for (int l = 0; l < 32; l++) {
                    const int is = l >> 4;
                    p[is]     += (((ql[l] & 15) | (((qh[l] >> 0) & 3) << 4)) - 32) * q8[l];
                    p[is + 2] += (((ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4)) - 32) * q8[l + 32];
                    p[is + 4] += (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32) * q8[l + 64];
                    p[is + 6] += (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32) * q8[l + 96];
                }
#endif
                for (int k = 0; k < 8; k++) tot += sc[k] * p[k];
            }
            s += y[b].d * fp16f(x[b].d) * (float)tot;
        }
        out[r] = s;
    }
}


static void mv_q3_K(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 110;
    for (int r = r0; r < r1; r++) {
        const blk_q3_K *x = (const blk_q3_K *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            uint32_t aux[4]; memcpy(aux, x[b].scales, 12);
            const uint32_t tmp = aux[2], k1 = 0x03030303, k2 = 0x0f0f0f0f;
            aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4); aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
            aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4); aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
            const int8_t *sc = (const int8_t *)aux; const uint8_t *q = x[b].qs, *hm = x[b].hmask; const int8_t *q8 = y[b].qs;
            int tot = 0, is = 0; uint8_t m = 1;
            for (int n = 0; n < 2; n++, q += 32) {
                for (int j = 0; j < 4; j++, m <<= 1) {
                    const int shift = 2 * j;
                    for (int half = 0; half < 2; half++, q8 += 16) {
                        const uint8_t *lq = q + half * 16, *lh = hm + half * 16; int s16;
#ifdef __wasm_simd128__
                        const v128_t t = wasm_v128_and(wasm_u8x16_shr(LD(lq), shift), wasm_i8x16_splat(3));
                        const v128_t no = wasm_i8x16_eq(wasm_v128_and(LD(lh), wasm_i8x16_splat((int8_t)m)), wasm_i8x16_splat(0));
                        s16 = whsum(wdot16(wasm_i8x16_sub(t, wasm_v128_and(no, wasm_i8x16_splat(4))), LD(q8)));
#else
                        s16 = 0;
                        for (int l = 0; l < 16; l++) s16 += (((lq[l] >> shift) & 3) - ((lh[l] & m) ? 0 : 4)) * q8[l];
#endif
                        tot += (sc[is++] - 32) * s16;
                    }
                }
            }
            s += y[b].d * fp16f(x[b].d) * (float)tot;
        }
        out[r] = s;
    }
}
static void mv_q5_K(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 176;
    for (int r = r0; r < r1; r++) {
        const blk_q5_K *x = (const blk_q5_K *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; j++) scale_min_k4(j, x[b].scales, &sc[j], &mn[j]);
            int dot = 0, mins = 0; const uint8_t *ql = x[b].qs, *qh = x[b].qh; const int8_t *q8 = y[b].qs; uint8_t u1 = 1, u2 = 2;
            for (int c = 0; c < 4; c++, ql += 32, q8 += 64, u1 <<= 2, u2 <<= 2) {
                int lo = 0, hi = 0;
#ifdef __wasm_simd128__
                for (int h = 0; h < 32; h += 16) {
                    const v128_t qa = LD(ql + h), qq = LD(qh + h), m15 = wasm_i8x16_splat(15), b16 = wasm_i8x16_splat(16);
                    const v128_t l5 = wasm_v128_or(wasm_v128_and(qa, m15), wasm_v128_and(wasm_i8x16_ne(wasm_v128_and(qq, wasm_i8x16_splat((int8_t)u1)), wasm_i8x16_splat(0)), b16));
                    const v128_t h5 = wasm_v128_or(wasm_u8x16_shr(qa, 4), wasm_v128_and(wasm_i8x16_ne(wasm_v128_and(qq, wasm_i8x16_splat((int8_t)u2)), wasm_i8x16_splat(0)), b16));
                    lo += whsum(wdot16(l5, LD(q8 + h))); hi += whsum(wdot16(h5, LD(q8 + 32 + h)));
                }
#else
                for (int l = 0; l < 32; l++) { lo += ((ql[l] & 15) + ((qh[l] & u1) ? 16 : 0)) * q8[l]; hi += ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) * q8[l + 32]; }
#endif
                dot += sc[2 * c] * lo + sc[2 * c + 1] * hi;
                mins += mn[2 * c] * (y[b].bsums[4 * c] + y[b].bsums[4 * c + 1]) + mn[2 * c + 1] * (y[b].bsums[4 * c + 2] + y[b].bsums[4 * c + 3]);
            }
            s += y[b].d * (fp16f(x[b].d) * (float)dot - fp16f(x[b].dmin) * (float)mins);
        }
        out[r] = s;
    }
}

#if defined(__x86_64__) && !defined(__EMSCRIPTEN__)
#include <immintrin.h>
#define AVX2 __attribute__((target("avx2,fma,f16c")))
AVX2 static inline float hsum_ps(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1)); a = _mm_add_ps(a, _mm_movehl_ps(a, a)); a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1)); return _mm_cvtss_f32(a);
}
AVX2 static inline int hsum_epi32(__m256i v) {
    __m128i a = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)); a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4e)); a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0xb1)); return _mm_cvtsi128_si32(a);
}
/* signed x signed int8 dot of 32 values -> 8 int32 partial sums */
AVX2 static inline __m256i dot32_ss(__m256i x, __m256i y) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(y, x)), _mm256_set1_epi16(1));
}
AVX2 static inline float fp16_hw(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }
/* 4 rows per pass: the int8 activations are loaded once and four independent
 * accumulators hide the FMA latency. Q8_0 x Q8_0 */
AVX2 static void avx_q8_0(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 34; int r = r0;
    for (; r + 4 <= r1; r += 4) {
        const blk_q8_0 *x0 = (const blk_q8_0 *)(w + rb * r), *x1 = (const blk_q8_0 *)(w + rb * (r + 1)),
                       *x2 = (const blk_q8_0 *)(w + rb * (r + 2)), *x3 = (const blk_q8_0 *)(w + rb * (r + 3));
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < nb; b++) {
            const __m256i yv = _mm256_loadu_si256((const __m256i *)y[b].qs); const float yd = y[b].d;
            a0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32_ss(_mm256_loadu_si256((const __m256i *)x0[b].qs), yv)), _mm256_set1_ps(fp16_hw(x0[b].d) * yd), a0);
            a1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32_ss(_mm256_loadu_si256((const __m256i *)x1[b].qs), yv)), _mm256_set1_ps(fp16_hw(x1[b].d) * yd), a1);
            a2 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32_ss(_mm256_loadu_si256((const __m256i *)x2[b].qs), yv)), _mm256_set1_ps(fp16_hw(x2[b].d) * yd), a2);
            a3 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot32_ss(_mm256_loadu_si256((const __m256i *)x3[b].qs), yv)), _mm256_set1_ps(fp16_hw(x3[b].d) * yd), a3);
        }
        out[r] = hsum_ps(a0); out[r + 1] = hsum_ps(a1); out[r + 2] = hsum_ps(a2); out[r + 3] = hsum_ps(a3);
    }
    for (; r < r1; r++) {
        const blk_q8_0 *x = (const blk_q8_0 *)(w + rb * r); __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const __m256i p = dot32_ss(_mm256_loadu_si256((const __m256i *)x[b].qs), _mm256_loadu_si256((const __m256i *)y[b].qs));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(fp16_hw(x[b].d) * y[b].d), acc);
        }
        out[r] = hsum_ps(acc);
    }
}
/* Q4_0: nibbles are used unsigned (maddubs) and the "-8" offset is removed
 * with 8 * sum(y), computed once per block and shared by the four rows. */
AVX2 static inline __m256i q4_nib(const blk_q4_0 *x, __m128i m) {
    const __m128i q = _mm_loadu_si128((const __m128i *)x->qs);
    return _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), m), _mm_and_si128(q, m));
}
AVX2 static inline __m256 q4_blk(const blk_q4_0 *x, __m128i m, __m256i yv, __m256i sy8, float yd, __m256 acc) {
    const __m256i p = _mm256_sub_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(q4_nib(x, m), yv), _mm256_set1_epi16(1)), sy8);
    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(fp16_hw(x->d) * yd), acc);
}
AVX2 static void avx_q4_0(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 18; const __m128i m = _mm_set1_epi8(15); const __m256i ones = _mm256_set1_epi8(1); int r = r0;
    for (; r + 4 <= r1; r += 4) {
        const blk_q4_0 *x0 = (const blk_q4_0 *)(w + rb * r), *x1 = (const blk_q4_0 *)(w + rb * (r + 1)),
                       *x2 = (const blk_q4_0 *)(w + rb * (r + 2)), *x3 = (const blk_q4_0 *)(w + rb * (r + 3));
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < nb; b++) {
            const __m256i yv = _mm256_loadu_si256((const __m256i *)y[b].qs); const float yd = y[b].d;
            const __m256i sy8 = _mm256_slli_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ones, yv), _mm256_set1_epi16(1)), 3);
            a0 = q4_blk(x0 + b, m, yv, sy8, yd, a0); a1 = q4_blk(x1 + b, m, yv, sy8, yd, a1);
            a2 = q4_blk(x2 + b, m, yv, sy8, yd, a2); a3 = q4_blk(x3 + b, m, yv, sy8, yd, a3);
        }
        out[r] = hsum_ps(a0); out[r + 1] = hsum_ps(a1); out[r + 2] = hsum_ps(a2); out[r + 3] = hsum_ps(a3);
    }
    for (; r < r1; r++) {
        const blk_q4_0 *x = (const blk_q4_0 *)(w + rb * r); __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const __m256i yv = _mm256_loadu_si256((const __m256i *)y[b].qs);
            const __m256i sy8 = _mm256_slli_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ones, yv), _mm256_set1_epi16(1)), 3);
            acc = q4_blk(x + b, m, yv, sy8, y[b].d, acc);
        }
        out[r] = hsum_ps(acc);
    }
}
/* Q5_0: 5-bit values (nibble | high bit << 4) used unsigned; "-16" removed via 16 * sum(y). */
AVX2 static inline __m256 q5_blk(const blk_q5_0 *x, __m128i m, __m256i shufc, __m256i bitm, __m256i yv, __m256i sy16, float yd, __m256 acc) {
    uint32_t qh; memcpy(&qh, x->qh, 4);
    const __m128i q = _mm_loadu_si128((const __m128i *)x->qs);
    const __m256i nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), m), _mm_and_si128(q, m));
    const __m256i sel = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)qh), shufc), bitm);
    const __m256i hb = _mm256_and_si256(_mm256_cmpeq_epi8(sel, bitm), _mm256_set1_epi8(16));
    const __m256i u = _mm256_or_si256(nib, hb);
    const __m256i p = _mm256_sub_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(u, yv), _mm256_set1_epi16(1)), sy16);
    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(fp16_hw(x->d) * yd), acc);
}
AVX2 static void avx_q5_0(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 22; const __m128i m = _mm_set1_epi8(15); const __m256i ones = _mm256_set1_epi8(1);
    const __m256i shufc = _mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);
    const __m256i bitm = _mm256_setr_epi8(1,2,4,8,16,32,64,(char)128,1,2,4,8,16,32,64,(char)128, 1,2,4,8,16,32,64,(char)128,1,2,4,8,16,32,64,(char)128);
    int r = r0;
    for (; r + 4 <= r1; r += 4) {
        const blk_q5_0 *x0 = (const blk_q5_0 *)(w + rb * r), *x1 = (const blk_q5_0 *)(w + rb * (r + 1)),
                       *x2 = (const blk_q5_0 *)(w + rb * (r + 2)), *x3 = (const blk_q5_0 *)(w + rb * (r + 3));
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < nb; b++) {
            const __m256i yv = _mm256_loadu_si256((const __m256i *)y[b].qs); const float yd = y[b].d;
            const __m256i sy16 = _mm256_slli_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ones, yv), _mm256_set1_epi16(1)), 4);
            a0 = q5_blk(x0 + b, m, shufc, bitm, yv, sy16, yd, a0); a1 = q5_blk(x1 + b, m, shufc, bitm, yv, sy16, yd, a1);
            a2 = q5_blk(x2 + b, m, shufc, bitm, yv, sy16, yd, a2); a3 = q5_blk(x3 + b, m, shufc, bitm, yv, sy16, yd, a3);
        }
        out[r] = hsum_ps(a0); out[r + 1] = hsum_ps(a1); out[r + 2] = hsum_ps(a2); out[r + 3] = hsum_ps(a3);
    }
    for (; r < r1; r++) {
        const blk_q5_0 *x = (const blk_q5_0 *)(w + rb * r); __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const __m256i yv = _mm256_loadu_si256((const __m256i *)y[b].qs);
            const __m256i sy16 = _mm256_slli_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ones, yv), _mm256_set1_epi16(1)), 4);
            acc = q5_blk(x + b, m, shufc, bitm, yv, sy16, y[b].d, acc);
        }
        out[r] = hsum_ps(acc);
    }
}
AVX2 static void avx_q4_K(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 144; const __m256i m = _mm256_set1_epi8(15);
    for (int r = r0; r < r1; r++) {
        const blk_q4_K *x = (const blk_q4_K *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; j++) scale_min_k4(j, x[b].scales, &sc[j], &mn[j]);
            __m256i acc = _mm256_setzero_si256(); int mins = 0; const uint8_t *q = x[b].qs; const int8_t *q8 = y[b].qs;
            for (int c = 0; c < 4; c++, q += 32, q8 += 64) {
                const __m256i qq = _mm256_loadu_si256((const __m256i *)q);
                const __m256i lo = _mm256_and_si256(qq, m), hi = _mm256_and_si256(_mm256_srli_epi16(qq, 4), m);
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(lo, _mm256_loadu_si256((const __m256i *)q8)), _mm256_set1_epi16(sc[2 * c])));
                acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(hi, _mm256_loadu_si256((const __m256i *)(q8 + 32))), _mm256_set1_epi16(sc[2 * c + 1])));
                mins += mn[2 * c] * (y[b].bsums[4 * c] + y[b].bsums[4 * c + 1]) + mn[2 * c + 1] * (y[b].bsums[4 * c + 2] + y[b].bsums[4 * c + 3]);
            }
            s += y[b].d * (fp16f(x[b].d) * (float)hsum_epi32(acc) - fp16f(x[b].dmin) * (float)mins);
        }
        out[r] = s;
    }
}
AVX2 static void avx_q6_K(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 210; const __m256i m15 = _mm256_set1_epi8(15), m3 = _mm256_set1_epi8(3);
    for (int r = r0; r < r1; r++) {
        const blk_q6_K *x = (const blk_q6_K *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            const uint8_t *ql = x[b].ql, *qh = x[b].qh; const int8_t *sc = x[b].scales, *q8 = y[b].qs;
            __m256i acc = _mm256_setzero_si256(); int corr = 0;
            for (int h = 0; h < 2; h++, ql += 64, qh += 32, sc += 8, q8 += 128) {
                const __m256i a0 = _mm256_loadu_si256((const __m256i *)ql), a1 = _mm256_loadu_si256((const __m256i *)(ql + 32)), hh = _mm256_loadu_si256((const __m256i *)qh);
                const __m256i q1 = _mm256_or_si256(_mm256_and_si256(a0, m15), _mm256_slli_epi16(_mm256_and_si256(hh, m3), 4));
                const __m256i q2 = _mm256_or_si256(_mm256_and_si256(a1, m15), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hh, 2), m3), 4));
                const __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(a0, 4), m15), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hh, 4), m3), 4));
                const __m256i q4 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(a1, 4), m15), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hh, 6), m3), 4));
                const __m256i qs[4] = {q1, q2, q3, q4};
                for (int k = 0; k < 4; k++) {
                    const __m256i p = _mm256_maddubs_epi16(qs[k], _mm256_loadu_si256((const __m256i *)(q8 + 32 * k)));
                    const __m256i sv = _mm256_set_m128i(_mm_set1_epi16(sc[2 * k + 1]), _mm_set1_epi16(sc[2 * k]));
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, sv));
                    const int g = (h * 8 + 2 * k);
                    corr += sc[2 * k] * y[b].bsums[g] + sc[2 * k + 1] * y[b].bsums[g + 1];
                }
            }
            s += y[b].d * fp16f(x[b].d) * (float)(hsum_epi32(acc) - 32 * corr);
        }
        out[r] = s;
    }
}
/* ---- several activations against the same rows (batched prompt reading, Turbo checks) ----
 * The weights of a few rows are unpacked once into thread-local scratch; then every activation runs exactly the arithmetic of the
 * single-activation kernel above (the same operations in the same order), so results are bit-identical. What is saved is unpacking the
 * same weights again for every activation, which is most of the work for 4-bit, 5-bit and 6-bit rows. */
static _Thread_local uint8_t *MB; static _Thread_local size_t MBCAP;
static uint8_t *mb_get(size_t n) {
    n = (n + 63) & ~(size_t)63;
    if (n > MBCAP) { free(MB); MB = aligned_alloc(64, n); MBCAP = MB ? n : 0; }
    return MB;
}
AVX2 static void avx_q5_0_multi(const uint8_t *w, int cols, int r0, int r1, const es_act *A, int na, float *Y, size_t ys) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 22; const __m128i m = _mm_set1_epi8(15); const __m256i ones = _mm256_set1_epi8(1);
    const __m256i shufc = _mm256_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3);
    const __m256i bitm = _mm256_setr_epi8(1,2,4,8,16,32,64,(char)128,1,2,4,8,16,32,64,(char)128, 1,2,4,8,16,32,64,(char)128,1,2,4,8,16,32,64,(char)128);
    uint8_t *U = mb_get((size_t)4 * nb * 32 + (size_t)4 * nb * sizeof(float)); float *D = (float *)(U + (size_t)4 * nb * 32);
    for (int r = r0; r < r1; r += 4) {
        const int nr = r1 - r < 4 ? r1 - r : 4;
        for (int i = 0; i < nr; i++) {
            const blk_q5_0 *x = (const blk_q5_0 *)(w + rb * (r + i));
            for (int b = 0; b < nb; b++) {
                uint32_t qh; memcpy(&qh, x[b].qh, 4);
                const __m128i q = _mm_loadu_si128((const __m128i *)x[b].qs);
                const __m256i nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), m), _mm_and_si128(q, m));
                const __m256i sel = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)qh), shufc), bitm);
                const __m256i hb = _mm256_and_si256(_mm256_cmpeq_epi8(sel, bitm), _mm256_set1_epi8(16));
                _mm256_store_si256((__m256i *)(U + ((size_t)i * nb + b) * 32), _mm256_or_si256(nib, hb));
                D[i * nb + b] = fp16_hw(x[b].d);
            }
        }
        for (int qa = 0; qa < na; qa++) {
            const es_blk_q80 *y = A[qa].q80; float *out = Y + (size_t)qa * ys;
            if (nr == 4) {
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                for (int b = 0; b < nb; b++) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i *)y[b].qs); const float yd = y[b].d;
                    const __m256i sy16 = _mm256_slli_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ones, yv), _mm256_set1_epi16(1)), 4);
#define Q5M(acc, i) acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_load_si256((const __m256i *)(U + ((size_t)(i) * nb + b) * 32)), yv), _mm256_set1_epi16(1)), sy16)), _mm256_set1_ps(D[(i) * nb + b] * yd), acc)
                    Q5M(a0, 0); Q5M(a1, 1); Q5M(a2, 2); Q5M(a3, 3);
                }
                out[r] = hsum_ps(a0); out[r + 1] = hsum_ps(a1); out[r + 2] = hsum_ps(a2); out[r + 3] = hsum_ps(a3);
            } else {
                for (int i = 0; i < nr; i++) {
                    __m256 acc = _mm256_setzero_ps();
                    for (int b = 0; b < nb; b++) {
                        const __m256i yv = _mm256_loadu_si256((const __m256i *)y[b].qs);
                        const __m256i sy16 = _mm256_slli_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ones, yv), _mm256_set1_epi16(1)), 4);
                        const float yd = y[b].d;
                        Q5M(acc, i);
                    }
                    out[r + i] = hsum_ps(acc);
                }
            }
#undef Q5M
        }
    }
}
AVX2 static void avx_q4_0_multi(const uint8_t *w, int cols, int r0, int r1, const es_act *A, int na, float *Y, size_t ys) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 18; const __m128i m = _mm_set1_epi8(15); const __m256i ones = _mm256_set1_epi8(1);
    uint8_t *U = mb_get((size_t)4 * nb * 32 + (size_t)4 * nb * sizeof(float)); float *D = (float *)(U + (size_t)4 * nb * 32);
    for (int r = r0; r < r1; r += 4) {
        const int nr = r1 - r < 4 ? r1 - r : 4;
        for (int i = 0; i < nr; i++) {
            const blk_q4_0 *x = (const blk_q4_0 *)(w + rb * (r + i));
            for (int b = 0; b < nb; b++) { _mm256_store_si256((__m256i *)(U + ((size_t)i * nb + b) * 32), q4_nib(x + b, m)); D[i * nb + b] = fp16_hw(x[b].d); }
        }
        for (int qa = 0; qa < na; qa++) {
            const es_blk_q80 *y = A[qa].q80; float *out = Y + (size_t)qa * ys;
#define Q4M(acc, i) acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_load_si256((const __m256i *)(U + ((size_t)(i) * nb + b) * 32)), yv), _mm256_set1_epi16(1)), sy8)), _mm256_set1_ps(D[(i) * nb + b] * yd), acc)
            if (nr == 4) {
                __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
                for (int b = 0; b < nb; b++) {
                    const __m256i yv = _mm256_loadu_si256((const __m256i *)y[b].qs); const float yd = y[b].d;
                    const __m256i sy8 = _mm256_slli_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ones, yv), _mm256_set1_epi16(1)), 3);
                    Q4M(a0, 0); Q4M(a1, 1); Q4M(a2, 2); Q4M(a3, 3);
                }
                out[r] = hsum_ps(a0); out[r + 1] = hsum_ps(a1); out[r + 2] = hsum_ps(a2); out[r + 3] = hsum_ps(a3);
            } else {
                for (int i = 0; i < nr; i++) {
                    __m256 acc = _mm256_setzero_ps();
                    for (int b = 0; b < nb; b++) {
                        const __m256i yv = _mm256_loadu_si256((const __m256i *)y[b].qs);
                        const __m256i sy8 = _mm256_slli_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ones, yv), _mm256_set1_epi16(1)), 3);
                        const float yd = y[b].d;
                        Q4M(acc, i);
                    }
                    out[r + i] = hsum_ps(acc);
                }
            }
#undef Q4M
        }
    }
}
/* K-quants: one row at a time like the single kernels; a group of 4 rows is unpacked, then every activation replays the row maths */
typedef struct { float d, dmin; uint8_t sc[8], mn[8]; } k4_meta;
AVX2 static void avx_q4_K_multi(const uint8_t *w, int cols, int r0, int r1, const es_act *A, int na, float *Y, size_t ys) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 144; const __m256i m = _mm256_set1_epi8(15);
    uint8_t *U = mb_get((size_t)4 * nb * 256 + (size_t)4 * nb * sizeof(k4_meta)); k4_meta *K = (k4_meta *)(U + (size_t)4 * nb * 256);
    for (int r = r0; r < r1; r += 4) {
        const int nr = r1 - r < 4 ? r1 - r : 4;
        for (int i = 0; i < nr; i++) {
            const blk_q4_K *x = (const blk_q4_K *)(w + rb * (r + i));
            for (int b = 0; b < nb; b++) {
                k4_meta *k = &K[i * nb + b];
                for (int j = 0; j < 8; j++) scale_min_k4(j, x[b].scales, &k->sc[j], &k->mn[j]);
                k->d = fp16f(x[b].d); k->dmin = fp16f(x[b].dmin);
                const uint8_t *q = x[b].qs; uint8_t *o = U + ((size_t)i * nb + b) * 256;
                for (int c = 0; c < 4; c++, q += 32) {
                    const __m256i qq = _mm256_loadu_si256((const __m256i *)q);
                    _mm256_store_si256((__m256i *)(o + 64 * c), _mm256_and_si256(qq, m));
                    _mm256_store_si256((__m256i *)(o + 64 * c + 32), _mm256_and_si256(_mm256_srli_epi16(qq, 4), m));
                }
            }
        }
        for (int qa = 0; qa < na; qa++) {
            const es_block_q8_K *y = A[qa].q8; float *out = Y + (size_t)qa * ys;
            for (int i = 0; i < nr; i++) {
                float s = 0;
                for (int b = 0; b < nb; b++) {
                    const k4_meta *k = &K[i * nb + b]; const uint8_t *o = U + ((size_t)i * nb + b) * 256; const int8_t *q8 = y[b].qs;
                    __m256i acc = _mm256_setzero_si256(); int mins = 0;
                    for (int c = 0; c < 4; c++, q8 += 64) {
                        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_load_si256((const __m256i *)(o + 64 * c)), _mm256_loadu_si256((const __m256i *)q8)), _mm256_set1_epi16(k->sc[2 * c])));
                        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_load_si256((const __m256i *)(o + 64 * c + 32)), _mm256_loadu_si256((const __m256i *)(q8 + 32))), _mm256_set1_epi16(k->sc[2 * c + 1])));
                        mins += k->mn[2 * c] * (y[b].bsums[4 * c] + y[b].bsums[4 * c + 1]) + k->mn[2 * c + 1] * (y[b].bsums[4 * c + 2] + y[b].bsums[4 * c + 3]);
                    }
                    s += y[b].d * (k->d * (float)hsum_epi32(acc) - k->dmin * (float)mins);
                }
                out[r + i] = s;
            }
        }
    }
}
AVX2 static void avx_q6_K_multi(const uint8_t *w, int cols, int r0, int r1, const es_act *A, int na, float *Y, size_t ys) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 210; const __m256i m15 = _mm256_set1_epi8(15), m3 = _mm256_set1_epi8(3);
    uint8_t *U = mb_get((size_t)4 * nb * 256 + (size_t)4 * nb * 20); uint8_t *M6 = U + (size_t)4 * nb * 256;   /* per block: 16 scales + d */
    for (int r = r0; r < r1; r += 4) {
        const int nr = r1 - r < 4 ? r1 - r : 4;
        for (int i = 0; i < nr; i++) {
            const blk_q6_K *x = (const blk_q6_K *)(w + rb * (r + i));
            for (int b = 0; b < nb; b++) {
                const uint8_t *ql = x[b].ql, *qh = x[b].qh; uint8_t *o = U + ((size_t)i * nb + b) * 256;
                for (int h = 0; h < 2; h++, ql += 64, qh += 32, o += 128) {
                    const __m256i a0 = _mm256_loadu_si256((const __m256i *)ql), a1 = _mm256_loadu_si256((const __m256i *)(ql + 32)), hh = _mm256_loadu_si256((const __m256i *)qh);
                    _mm256_store_si256((__m256i *)o, _mm256_or_si256(_mm256_and_si256(a0, m15), _mm256_slli_epi16(_mm256_and_si256(hh, m3), 4)));
                    _mm256_store_si256((__m256i *)(o + 32), _mm256_or_si256(_mm256_and_si256(a1, m15), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hh, 2), m3), 4)));
                    _mm256_store_si256((__m256i *)(o + 64), _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(a0, 4), m15), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hh, 4), m3), 4)));
                    _mm256_store_si256((__m256i *)(o + 96), _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(a1, 4), m15), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(hh, 6), m3), 4)));
                }
                uint8_t *mm = M6 + ((size_t)i * nb + b) * 20; memcpy(mm, x[b].scales, 16); const float d = fp16f(x[b].d); memcpy(mm + 16, &d, 4);
            }
        }
        for (int qa = 0; qa < na; qa++) {
            const es_block_q8_K *y = A[qa].q8; float *out = Y + (size_t)qa * ys;
            for (int i = 0; i < nr; i++) {
                float s = 0;
                for (int b = 0; b < nb; b++) {
                    const uint8_t *o = U + ((size_t)i * nb + b) * 256, *mm = M6 + ((size_t)i * nb + b) * 20; const int8_t *sc = (const int8_t *)mm, *q8 = y[b].qs;
                    float d; memcpy(&d, mm + 16, 4);
                    __m256i acc = _mm256_setzero_si256(); int corr = 0;
                    for (int h = 0; h < 2; h++, sc += 8, q8 += 128, o += 128) {
                        for (int k = 0; k < 4; k++) {
                            const __m256i p = _mm256_maddubs_epi16(_mm256_load_si256((const __m256i *)(o + 32 * k)), _mm256_loadu_si256((const __m256i *)(q8 + 32 * k)));
                            const __m256i sv = _mm256_set_m128i(_mm_set1_epi16(sc[2 * k + 1]), _mm_set1_epi16(sc[2 * k]));
                            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, sv));
                            const int g = (h * 8 + 2 * k);
                            corr += sc[2 * k] * y[b].bsums[g] + sc[2 * k + 1] * y[b].bsums[g + 1];
                        }
                    }
                    s += y[b].d * d * (float)(hsum_epi32(acc) - 32 * corr);
                }
                out[r + i] = s;
            }
        }
    }
}
static int HAVE_AVX2 = -1;
#endif

#if defined(__aarch64__)
void es_mv_q8_0_neon(const uint8_t *, int, int, int, const es_blk_q80 *, float *);
void es_mv_q4_0_neon(const uint8_t *, int, int, int, const es_blk_q80 *, float *);
void es_mv_q5_0_neon(const uint8_t *, int, int, int, const es_blk_q80 *, float *);
void es_mv_q4_K_neon(const uint8_t *, int, int, int, const es_block_q8_K *, float *);
void es_mv_q6_K_neon(const uint8_t *, int, int, int, const es_block_q8_K *, float *);
int es_mv_multi_neon(uint32_t, const uint8_t *, int, int, int, const es_act *, int, float *, size_t);
void es_mv_q3_K_neon(const uint8_t *, int, int, int, const es_block_q8_K *, float *);
void es_mv_q5_K_neon(const uint8_t *, int, int, int, const es_block_q8_K *, float *);
static int HAVE_DOT = -1;
#endif


#ifdef __wasm_simd128__
/* ---- WebAssembly kernels: vector accumulation (no lane extraction per block), four rows per pass for the
 * 32-wide formats. With relaxed SIMD (-mrelaxed-simd, picked at load time by the page) the int8 dot is one
 * instruction (SDOT on ARM phones, VNNI/pmaddubsw on x86). u = unsigned values 0..127, s = signed int8. */
static inline v128_t wdot_us(v128_t u, v128_t s, v128_t acc) {
#ifdef __wasm_relaxed_simd__
    return wasm_i32x4_relaxed_dot_i8x16_i7x16_add(s, u, acc);
#else
    return wasm_i32x4_add(acc, wasm_i32x4_add(wasm_i32x4_dot_i16x8(wasm_u16x8_extend_low_u8x16(u), wasm_i16x8_extend_low_i8x16(s)),
                                              wasm_i32x4_dot_i16x8(wasm_u16x8_extend_high_u8x16(u), wasm_i16x8_extend_high_i8x16(s))));
#endif
}
static inline float wsumf(v128_t v) { return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) + wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3); }
#define WACC(acc, t, xd, yd) (acc) = wasm_f32x4_add((acc), wasm_f32x4_mul(wasm_f32x4_convert_i32x4(t), wasm_f32x4_splat(fp16f(xd) * (yd))))

static inline v128_t w8_dot(const blk_q8_0 *x, v128_t y0, v128_t y1, v128_t ny0, v128_t ny1) {
    const v128_t z = wasm_i32x4_splat(0), x0 = LD(x->qs), x1 = LD(x->qs + 16);
#ifdef __wasm_relaxed_simd__
    const v128_t zz = wasm_i8x16_splat(0);   /* |x| as the i7 operand, the sign moved onto the activation */
    const v128_t m0 = wasm_i8x16_lt(x0, zz), m1 = wasm_i8x16_lt(x1, zz);
    return wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_v128_bitselect(ny1, y1, m1), wasm_i8x16_abs(x1),
           wasm_i32x4_relaxed_dot_i8x16_i7x16_add(wasm_v128_bitselect(ny0, y0, m0), wasm_i8x16_abs(x0), z));
#else
    (void)ny0; (void)ny1; (void)z;
    return wasm_i32x4_add(wdot16(x0, y0), wdot16(x1, y1));
#endif
}
static void wmv_q8_0(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 34; int r = r0;
    for (; r + 4 <= r1; r += 4) {
        const blk_q8_0 *x0 = (const blk_q8_0 *)(w + rb * r), *x1 = (const blk_q8_0 *)(w + rb * (r + 1)), *x2 = (const blk_q8_0 *)(w + rb * (r + 2)), *x3 = (const blk_q8_0 *)(w + rb * (r + 3));
        v128_t a0 = wasm_f32x4_splat(0), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < nb; b++) {
            const v128_t y0 = LD(y[b].qs), y1 = LD(y[b].qs + 16), ny0 = wasm_i8x16_neg(y0), ny1 = wasm_i8x16_neg(y1); const float yd = y[b].d;
            WACC(a0, w8_dot(x0 + b, y0, y1, ny0, ny1), x0[b].d, yd); WACC(a1, w8_dot(x1 + b, y0, y1, ny0, ny1), x1[b].d, yd);
            WACC(a2, w8_dot(x2 + b, y0, y1, ny0, ny1), x2[b].d, yd); WACC(a3, w8_dot(x3 + b, y0, y1, ny0, ny1), x3[b].d, yd);
        }
        out[r] = wsumf(a0); out[r + 1] = wsumf(a1); out[r + 2] = wsumf(a2); out[r + 3] = wsumf(a3);
    }
    for (; r < r1; r++) {
        const blk_q8_0 *x = (const blk_q8_0 *)(w + rb * r); v128_t a = wasm_f32x4_splat(0);
        for (int b = 0; b < nb; b++) { const v128_t y0 = LD(y[b].qs), y1 = LD(y[b].qs + 16); WACC(a, w8_dot(x + b, y0, y1, wasm_i8x16_neg(y0), wasm_i8x16_neg(y1)), x[b].d, y[b].d); }
        out[r] = wsumf(a);
    }
}
/* Q4_0: unsigned nibbles, "-8" removed through 8 * sum(y) (computed once per block, shared by the four rows) */
static inline v128_t w4_dot(const blk_q4_0 *x, v128_t y0, v128_t y1, v128_t sy8) {
    const v128_t q = LD(x->qs), m = wasm_i8x16_splat(15);
    return wasm_i32x4_sub(wdot_us(wasm_u8x16_shr(q, 4), y1, wdot_us(wasm_v128_and(q, m), y0, wasm_i32x4_splat(0))), sy8);
}
static void wmv_q4_0(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 18; int r = r0; const v128_t one = wasm_i8x16_splat(1);
    for (; r + 4 <= r1; r += 4) {
        const blk_q4_0 *x0 = (const blk_q4_0 *)(w + rb * r), *x1 = (const blk_q4_0 *)(w + rb * (r + 1)), *x2 = (const blk_q4_0 *)(w + rb * (r + 2)), *x3 = (const blk_q4_0 *)(w + rb * (r + 3));
        v128_t a0 = wasm_f32x4_splat(0), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < nb; b++) {
            const v128_t y0 = LD(y[b].qs), y1 = LD(y[b].qs + 16); const float yd = y[b].d;
            const v128_t sy8 = wasm_i32x4_shl(wdot_us(one, y1, wdot_us(one, y0, wasm_i32x4_splat(0))), 3);
            WACC(a0, w4_dot(x0 + b, y0, y1, sy8), x0[b].d, yd); WACC(a1, w4_dot(x1 + b, y0, y1, sy8), x1[b].d, yd);
            WACC(a2, w4_dot(x2 + b, y0, y1, sy8), x2[b].d, yd); WACC(a3, w4_dot(x3 + b, y0, y1, sy8), x3[b].d, yd);
        }
        out[r] = wsumf(a0); out[r + 1] = wsumf(a1); out[r + 2] = wsumf(a2); out[r + 3] = wsumf(a3);
    }
    for (; r < r1; r++) {
        const blk_q4_0 *x = (const blk_q4_0 *)(w + rb * r); v128_t a = wasm_f32x4_splat(0);
        for (int b = 0; b < nb; b++) {
            const v128_t y0 = LD(y[b].qs), y1 = LD(y[b].qs + 16);
            WACC(a, w4_dot(x + b, y0, y1, wasm_i32x4_shl(wdot_us(one, y1, wdot_us(one, y0, wasm_i32x4_splat(0))), 3)), x[b].d, y[b].d);
        }
        out[r] = wsumf(a);
    }
}
static inline v128_t w5_dot(const blk_q5_0 *x, v128_t y0, v128_t y1, v128_t sy16) {
    const v128_t qv = wasm_v128_load32_splat(x->qh), q = LD(x->qs), m15 = wasm_i8x16_splat(15), b16 = wasm_i8x16_splat(16), z = wasm_i8x16_splat(0);
    const v128_t bits = wasm_i8x16_make(1, 2, 4, 8, 16, 32, 64, (int8_t)128, 1, 2, 4, 8, 16, 32, 64, (int8_t)128);
    const v128_t h0 = wasm_v128_and(wasm_i8x16_ne(wasm_v128_and(wasm_i8x16_swizzle(qv, wasm_i8x16_make(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1)), bits), z), b16);
    const v128_t h1 = wasm_v128_and(wasm_i8x16_ne(wasm_v128_and(wasm_i8x16_swizzle(qv, wasm_i8x16_make(2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3)), bits), z), b16);
    return wasm_i32x4_sub(wdot_us(wasm_v128_or(wasm_u8x16_shr(q, 4), h1), y1, wdot_us(wasm_v128_or(wasm_v128_and(q, m15), h0), y0, wasm_i32x4_splat(0))), sy16);
}
static void wmv_q5_0(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 22; int r = r0; const v128_t one = wasm_i8x16_splat(1);
    for (; r + 4 <= r1; r += 4) {
        const blk_q5_0 *x0 = (const blk_q5_0 *)(w + rb * r), *x1 = (const blk_q5_0 *)(w + rb * (r + 1)), *x2 = (const blk_q5_0 *)(w + rb * (r + 2)), *x3 = (const blk_q5_0 *)(w + rb * (r + 3));
        v128_t a0 = wasm_f32x4_splat(0), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < nb; b++) {
            const v128_t y0 = LD(y[b].qs), y1 = LD(y[b].qs + 16); const float yd = y[b].d;
            const v128_t sy = wasm_i32x4_shl(wdot_us(one, y1, wdot_us(one, y0, wasm_i32x4_splat(0))), 4);
            WACC(a0, w5_dot(x0 + b, y0, y1, sy), x0[b].d, yd); WACC(a1, w5_dot(x1 + b, y0, y1, sy), x1[b].d, yd);
            WACC(a2, w5_dot(x2 + b, y0, y1, sy), x2[b].d, yd); WACC(a3, w5_dot(x3 + b, y0, y1, sy), x3[b].d, yd);
        }
        out[r] = wsumf(a0); out[r + 1] = wsumf(a1); out[r + 2] = wsumf(a2); out[r + 3] = wsumf(a3);
    }
    for (; r < r1; r++) {
        const blk_q5_0 *x = (const blk_q5_0 *)(w + rb * r); v128_t a = wasm_f32x4_splat(0);
        for (int b = 0; b < nb; b++) {
            const v128_t y0 = LD(y[b].qs), y1 = LD(y[b].qs + 16);
            WACC(a, w5_dot(x + b, y0, y1, wasm_i32x4_shl(wdot_us(one, y1, wdot_us(one, y0, wasm_i32x4_splat(0))), 4)), x[b].d, y[b].d);
        }
        out[r] = wsumf(a);
    }
}
static void wmv_q4_K(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 144; const v128_t m = wasm_i8x16_splat(15), z = wasm_i32x4_splat(0);
    for (int r = r0; r < r1; r++) {
        const blk_q4_K *x = (const blk_q4_K *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; j++) scale_min_k4(j, x[b].scales, &sc[j], &mn[j]);
            v128_t acc = z; const uint8_t *q = x[b].qs; const int8_t *q8 = y[b].qs;
            for (int c = 0; c < 4; c++, q += 32, q8 += 64) {
                const v128_t qa = LD(q), qb = LD(q + 16);
                const v128_t lo = wdot_us(wasm_v128_and(qb, m), LD(q8 + 16), wdot_us(wasm_v128_and(qa, m), LD(q8), z));
                const v128_t hi = wdot_us(wasm_u8x16_shr(qb, 4), LD(q8 + 48), wdot_us(wasm_u8x16_shr(qa, 4), LD(q8 + 32), z));
                acc = wasm_i32x4_add(acc, wasm_i32x4_add(wasm_i32x4_mul(lo, wasm_i32x4_splat(sc[2 * c])), wasm_i32x4_mul(hi, wasm_i32x4_splat(sc[2 * c + 1]))));
            }
            int mins = 0;
            for (int j = 0; j < 8; j++) mins += mn[j] * (y[b].bsums[2 * j] + y[b].bsums[2 * j + 1]);
            s += y[b].d * (fp16f(x[b].d) * (float)whsum(acc) - fp16f(x[b].dmin) * (float)mins);
        }
        out[r] = s;
    }
}
static void wmv_q6_K(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 210; const v128_t m15 = wasm_i8x16_splat(15), m3 = wasm_i8x16_splat(3), z = wasm_i32x4_splat(0);
    for (int r = r0; r < r1; r++) {
        const blk_q6_K *x = (const blk_q6_K *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            const uint8_t *ql = x[b].ql, *qh = x[b].qh; const int8_t *sc = x[b].scales, *q8 = y[b].qs; v128_t acc = z; int corr = 0;
            for (int g = 0; g < 16; g++) corr += sc[g] * y[b].bsums[g];
            for (int h = 0; h < 2; h++, ql += 64, qh += 32, sc += 8, q8 += 128) {
                for (int is = 0; is < 2; is++) {
                    const int l = is * 16;
                    const v128_t a0 = LD(ql + l), a1 = LD(ql + l + 32), hh = LD(qh + l);
                    const v128_t q1 = wasm_v128_or(wasm_v128_and(a0, m15), wasm_i8x16_shl(wasm_v128_and(hh, m3), 4));
                    const v128_t q2 = wasm_v128_or(wasm_v128_and(a1, m15), wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(hh, 2), m3), 4));
                    const v128_t q3 = wasm_v128_or(wasm_u8x16_shr(a0, 4), wasm_i8x16_shl(wasm_v128_and(wasm_u8x16_shr(hh, 4), m3), 4));
                    const v128_t q4 = wasm_v128_or(wasm_u8x16_shr(a1, 4), wasm_i8x16_shl(wasm_u8x16_shr(hh, 6), 4));
                    acc = wasm_i32x4_add(acc, wasm_i32x4_mul(wdot_us(q1, LD(q8 + l), z), wasm_i32x4_splat(sc[is])));
                    acc = wasm_i32x4_add(acc, wasm_i32x4_mul(wdot_us(q2, LD(q8 + l + 32), z), wasm_i32x4_splat(sc[is + 2])));
                    acc = wasm_i32x4_add(acc, wasm_i32x4_mul(wdot_us(q3, LD(q8 + l + 64), z), wasm_i32x4_splat(sc[is + 4])));
                    acc = wasm_i32x4_add(acc, wasm_i32x4_mul(wdot_us(q4, LD(q8 + l + 96), z), wasm_i32x4_splat(sc[is + 6])));
                }
            }
            s += y[b].d * fp16f(x[b].d) * (float)(whsum(acc) - 32 * corr);
        }
        out[r] = s;
    }
}
#endif

#if (defined(__x86_64__) && !defined(__EMSCRIPTEN__)) || defined(__aarch64__)
static int NO_MULTI = -1;   /* ES_NO_MULTI=1: one activation at a time (for comparisons) */
#endif
/* several activations against rows [r0, r1): y + q * ys receives activation q's results. Same numbers as calling es_matvec for each. */
void es_matvec_multi(uint32_t t, const void *W, int cols, int r0, int r1, const es_act *a, int na, float *y, size_t ys, float *scratch) {
    if (na > 1 && lowbit_ok(t) && !getenv_cached_no_multi()) {   /* low-bit formats: each block decoded once for every activation */
        int q80 = 1;
        for (int i = 0; i < na; i++) q80 &= a[i].q80 != NULL;
        if (q80) {
            for (int k0 = 0; k0 < na; k0 += ES_MULTI_MAX) {
                const es_blk_q80 *yy[ES_MULTI_MAX];
                const int nk = na - k0 < ES_MULTI_MAX ? na - k0 : ES_MULTI_MAX;
                for (int k = 0; k < nk; k++) yy[k] = a[k0 + k].q80;
                mv_lowbit_n(t, W, cols, r0, r1, yy, nk, y + (size_t)k0 * ys, ys);
            }
            return;
        }
    }
#if defined(__x86_64__) && !defined(__EMSCRIPTEN__)
    if (HAVE_AVX2 < 0) { __builtin_cpu_init(); HAVE_AVX2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c") && !getenv("ES_NO_AVX2"); }
    if (NO_MULTI < 0) NO_MULTI = getenv("ES_NO_MULTI") != NULL;
    if (HAVE_AVX2 && na > 1 && !NO_MULTI) {
        int q80 = 1, q8 = 1;
        for (int i = 0; i < na; i++) { q80 &= a[i].q80 != NULL; q8 &= a[i].q8 != NULL; }
        if (t == GGML_Q5_0 && q80) { if (mb_get((size_t)4 * (cols / 32) * 36)) { avx_q5_0_multi(W, cols, r0, r1, a, na, y, ys); return; } }
        else if (t == GGML_Q4_0 && q80) { if (mb_get((size_t)4 * (cols / 32) * 36)) { avx_q4_0_multi(W, cols, r0, r1, a, na, y, ys); return; } }
        else if (t == GGML_Q4_K && q8) { if (mb_get((size_t)4 * (cols / 256) * (256 + sizeof(k4_meta)))) { avx_q4_K_multi(W, cols, r0, r1, a, na, y, ys); return; } }
        else if (t == GGML_Q6_K && q8) { if (mb_get((size_t)4 * (cols / 256) * 276)) { avx_q6_K_multi(W, cols, r0, r1, a, na, y, ys); return; } }
    }
#endif
#if defined(__aarch64__)
    if (HAVE_DOT < 0) HAVE_DOT = es_has_dotprod();
    if (NO_MULTI < 0) NO_MULTI = getenv("ES_NO_MULTI") != NULL;
    if (HAVE_DOT && na > 1 && !NO_MULTI && es_mv_multi_neon(t, W, cols, r0, r1, a, na, y, ys)) return;
#endif
    for (int i = 0; i < na; i++) es_matvec(t, W, cols, r0, r1, a + i, y + (size_t)i * ys, scratch);
}

void es_matvec(uint32_t t, const void *W, int cols, int r0, int r1, const es_act *a, float *y,
               float *scratch) {
    const size_t rb = es_row_bytes(t, cols);
    const uint8_t *w = W;
    if (t == GGML_Q2_K && a->q8) {
        es_matvec_q2_K((const es_block_q2_K *)W, cols, r0, r1, a->q8, y);
        return;
    }
#if defined(__x86_64__) && !defined(__EMSCRIPTEN__)
    if (HAVE_AVX2 < 0) { __builtin_cpu_init(); HAVE_AVX2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c") && !getenv("ES_NO_AVX2"); }
    if (HAVE_AVX2) {
        if (a->q80) {
            if (t == GGML_Q8_0) { avx_q8_0(w, cols, r0, r1, a->q80, y); return; }
            if (t == GGML_Q4_0) { avx_q4_0(w, cols, r0, r1, a->q80, y); return; }
            if (t == GGML_Q5_0) { avx_q5_0(w, cols, r0, r1, a->q80, y); return; }
        }
        if (a->q8) {
            if (t == GGML_Q4_K) { avx_q4_K(w, cols, r0, r1, a->q8, y); return; }
            if (t == GGML_Q6_K) { avx_q6_K(w, cols, r0, r1, a->q8, y); return; }
        }
    }
#endif
#ifdef __wasm_simd128__
    if (a->q80) {
        if (t == GGML_Q8_0) { wmv_q8_0(w, cols, r0, r1, a->q80, y); return; }
        if (t == GGML_Q4_0) { wmv_q4_0(w, cols, r0, r1, a->q80, y); return; }
        if (t == GGML_Q5_0) { wmv_q5_0(w, cols, r0, r1, a->q80, y); return; }
    }
    if (a->q8) {
        if (t == GGML_Q4_K) { wmv_q4_K(w, cols, r0, r1, a->q8, y); return; }
        if (t == GGML_Q6_K) { wmv_q6_K(w, cols, r0, r1, a->q8, y); return; }
    }
#endif
#if defined(__aarch64__)
    if (HAVE_DOT < 0) HAVE_DOT = es_has_dotprod();
    if (HAVE_DOT) {
        if (a->q80) {
            if (t == GGML_Q8_0) { es_mv_q8_0_neon(w, cols, r0, r1, a->q80, y); return; }
            if (t == GGML_Q4_0) { es_mv_q4_0_neon(w, cols, r0, r1, a->q80, y); return; }
            if (t == GGML_Q5_0) { es_mv_q5_0_neon(w, cols, r0, r1, a->q80, y); return; }
        }
        if (a->q8) {
            if (t == GGML_Q4_K) { es_mv_q4_K_neon(w, cols, r0, r1, a->q8, y); return; }
            if (t == GGML_Q6_K) { es_mv_q6_K_neon(w, cols, r0, r1, a->q8, y); return; }
            if (t == GGML_Q3_K) { es_mv_q3_K_neon(w, cols, r0, r1, a->q8, y); return; }
            if (t == GGML_Q5_K) { es_mv_q5_K_neon(w, cols, r0, r1, a->q8, y); return; }
        }
    }
#endif
    if (a->q80 && lowbit_ok(t)) { mv_lowbit(t, w, cols, r0, r1, a->q80, y); return; }
    if (a->q80) {
        switch (t) {
        case GGML_Q8_0: mv_q8_0(w, cols, r0, r1, a->q80, y); return;
        case GGML_Q4_0: mv_q4_0(w, cols, r0, r1, a->q80, y); return;
        case GGML_Q5_0: mv_q5_0(w, cols, r0, r1, a->q80, y); return;
        default: break;
        }
    }
    if (a->q8) {
        if (t == GGML_Q4_K) { mv_q4_K(w, cols, r0, r1, a->q8, y); return; }
        if (t == GGML_Q6_K) { mv_q6_K(w, cols, r0, r1, a->q8, y); return; }
        if (t == GGML_Q3_K) { mv_q3_K(w, cols, r0, r1, a->q8, y); return; }
        if (t == GGML_Q5_K) { mv_q5_K(w, cols, r0, r1, a->q8, y); return; }
    }
    if (t == GGML_F32) {
        for (int r = r0; r < r1; r++) y[r] = dot_f32((const float *)(w + rb * r), a->x, cols);
        return;
    }
    for (int r = r0; r < r1; r++) {
        es_dequant_row(t, w + rb * (size_t)r, scratch, cols);
        y[r] = dot_f32(scratch, a->x, cols);
    }
}
