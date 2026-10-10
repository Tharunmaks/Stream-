#include "es_quant.h"
#include "es_cpu.h"

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
    case GGML_Q4_K: case GGML_Q5_K: case GGML_Q6_K: case GGML_IQ4_NL: case GGML_IQ4_XS: case GGML_BF16: return 1;
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
static int HAVE_AVX2 = -1;
#endif

#if defined(__aarch64__)
void es_mv_q8_0_neon(const uint8_t *, int, int, int, const es_blk_q80 *, float *);
void es_mv_q4_0_neon(const uint8_t *, int, int, int, const es_blk_q80 *, float *);
void es_mv_q5_0_neon(const uint8_t *, int, int, int, const es_blk_q80 *, float *);
void es_mv_q4_K_neon(const uint8_t *, int, int, int, const es_block_q8_K *, float *);
void es_mv_q6_K_neon(const uint8_t *, int, int, int, const es_block_q8_K *, float *);
void es_mv_q3_K_neon(const uint8_t *, int, int, int, const es_block_q8_K *, float *);
void es_mv_q5_K_neon(const uint8_t *, int, int, int, const es_block_q8_K *, float *);
static int HAVE_DOT = -1;
#endif

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
