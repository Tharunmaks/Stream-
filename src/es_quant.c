#include "es_quant.h"

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
    case GGML_Q4_K: case GGML_Q5_K: case GGML_Q6_K: return 1;
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

void es_dequant_row(uint32_t t, const void *src, float *y, int n) {
    switch (t) {
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
    }
}

void es_act_prepare(es_act *a, const float *x, int cols, es_block_q8_K *q8_buf) {
    static int float_only = -1; /* ES_FLOAT_ACT=1: exact float path (slow, for testing) */
    if (float_only < 0) float_only = getenv("ES_FLOAT_ACT") != NULL;
    if (float_only) q8_buf = NULL;
    a->x = x;
    a->cols = cols;
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

void es_matvec(uint32_t t, const void *W, int cols, int r0, int r1, const es_act *a, float *y,
               float *scratch) {
    const size_t rb = es_row_bytes(t, cols);
    const uint8_t *w = W;
    if (t == GGML_Q2_K && a->q8) {
        es_matvec_q2_K((const es_block_q2_K *)W, cols, r0, r1, a->q8, y);
        return;
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
