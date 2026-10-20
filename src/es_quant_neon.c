/* Integer dot-product kernels for Q8_0, Q4_0, Q5_0, Q4_K and Q6_K with NEON SDOT.
 * Compile with -march=armv8.2-a+dotprod; called only when the CPU reports asimddp. */
#if defined(__aarch64__)
#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

#include "es_q2k.h"
#include "es_quant.h"

typedef struct { uint16_t d; int8_t qs[32]; } nb_q8_0;
typedef struct { uint16_t d; uint8_t qs[16]; } nb_q4_0;
typedef struct { uint16_t d; uint8_t qh[4]; uint8_t qs[16]; } nb_q5_0;
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; } nb_q4_K;
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } nb_q6_K;
typedef struct { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; uint16_t d; } nb_q3_K;
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } nb_q5_K;

static inline int hsum(int32x4_t v) { return vaddvq_s32(v); }
static inline void scale_min(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

/* The legacy quants process four weight rows per pass: the int8 activations are loaded once, four independent
 * float accumulators hide the SDOT/FMA latency, and there is no horizontal sum per block (one per row at the end). */
#define DOT2(lo, hi, y0, y1) vdotq_s32(vdotq_s32(vdupq_n_s32(0), (lo), (y0)), (hi), (y1))
#define ACC(a, t, xd, yd) (a) = vfmaq_n_f32((a), vcvtq_f32_s32(t), es_fp16_to_fp32(xd) * (yd))

void es_mv_q8_0_neon(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 34; int r = r0;
    for (; r + 4 <= r1; r += 4) {
        const nb_q8_0 *x0 = (const nb_q8_0 *)(w + rb * r), *x1 = (const nb_q8_0 *)(w + rb * (r + 1)), *x2 = (const nb_q8_0 *)(w + rb * (r + 2)), *x3 = (const nb_q8_0 *)(w + rb * (r + 3));
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < nb; b++) {
            const int8x16_t y0 = vld1q_s8(y[b].qs), y1 = vld1q_s8(y[b].qs + 16); const float yd = y[b].d;
            ACC(a0, DOT2(vld1q_s8(x0[b].qs), vld1q_s8(x0[b].qs + 16), y0, y1), x0[b].d, yd);
            ACC(a1, DOT2(vld1q_s8(x1[b].qs), vld1q_s8(x1[b].qs + 16), y0, y1), x1[b].d, yd);
            ACC(a2, DOT2(vld1q_s8(x2[b].qs), vld1q_s8(x2[b].qs + 16), y0, y1), x2[b].d, yd);
            ACC(a3, DOT2(vld1q_s8(x3[b].qs), vld1q_s8(x3[b].qs + 16), y0, y1), x3[b].d, yd);
        }
        out[r] = vaddvq_f32(a0); out[r + 1] = vaddvq_f32(a1); out[r + 2] = vaddvq_f32(a2); out[r + 3] = vaddvq_f32(a3);
    }
    for (; r < r1; r++) {
        const nb_q8_0 *x = (const nb_q8_0 *)(w + rb * r); float32x4_t a = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) ACC(a, DOT2(vld1q_s8(x[b].qs), vld1q_s8(x[b].qs + 16), vld1q_s8(y[b].qs), vld1q_s8(y[b].qs + 16)), x[b].d, y[b].d);
        out[r] = vaddvq_f32(a);
    }
}
static inline int32x4_t q4_dot(const nb_q4_0 *x, uint8x16_t m, int8x16_t e, int8x16_t y0, int8x16_t y1) {
    const uint8x16_t q = vld1q_u8(x->qs);
    return DOT2(vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m)), e), vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), e), y0, y1);
}
void es_mv_q4_0_neon(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 18; const uint8x16_t m = vdupq_n_u8(15); const int8x16_t e = vdupq_n_s8(8); int r = r0;
    for (; r + 4 <= r1; r += 4) {
        const nb_q4_0 *x0 = (const nb_q4_0 *)(w + rb * r), *x1 = (const nb_q4_0 *)(w + rb * (r + 1)), *x2 = (const nb_q4_0 *)(w + rb * (r + 2)), *x3 = (const nb_q4_0 *)(w + rb * (r + 3));
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < nb; b++) {
            const int8x16_t y0 = vld1q_s8(y[b].qs), y1 = vld1q_s8(y[b].qs + 16); const float yd = y[b].d;
            ACC(a0, q4_dot(x0 + b, m, e, y0, y1), x0[b].d, yd); ACC(a1, q4_dot(x1 + b, m, e, y0, y1), x1[b].d, yd);
            ACC(a2, q4_dot(x2 + b, m, e, y0, y1), x2[b].d, yd); ACC(a3, q4_dot(x3 + b, m, e, y0, y1), x3[b].d, yd);
        }
        out[r] = vaddvq_f32(a0); out[r + 1] = vaddvq_f32(a1); out[r + 2] = vaddvq_f32(a2); out[r + 3] = vaddvq_f32(a3);
    }
    for (; r < r1; r++) {
        const nb_q4_0 *x = (const nb_q4_0 *)(w + rb * r); float32x4_t a = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) ACC(a, q4_dot(x + b, m, e, vld1q_s8(y[b].qs), vld1q_s8(y[b].qs + 16)), x[b].d, y[b].d);
        out[r] = vaddvq_f32(a);
    }
}
static const uint8_t Q5_BITS[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128}, Q5_I0[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}, Q5_I1[16] = {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
static inline int32x4_t q5_dot(const nb_q5_0 *x, uint8x16_t bits, uint8x16_t i0, uint8x16_t i1, int8x16_t y0, int8x16_t y1) {
    uint32x4_t qv32 = vdupq_n_u32(0); qv32 = vld1q_lane_u32((const uint32_t *)x->qh, qv32, 0);
    const uint8x16_t qv = vreinterpretq_u8_u32(qv32), q = vld1q_u8(x->qs), m15 = vdupq_n_u8(15), b16 = vdupq_n_u8(16);
    const uint8x16_t h0 = vandq_u8(vtstq_u8(vqtbl1q_u8(qv, i0), bits), b16), h1 = vandq_u8(vtstq_u8(vqtbl1q_u8(qv, i1), bits), b16);
    return DOT2(vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(vandq_u8(q, m15), h0), b16)), vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(vshrq_n_u8(q, 4), h1), b16)), y0, y1);
}
void es_mv_q5_0_neon(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 22; int r = r0;
    const uint8x16_t bits = vld1q_u8(Q5_BITS), i0 = vld1q_u8(Q5_I0), i1 = vld1q_u8(Q5_I1);
    for (; r + 4 <= r1; r += 4) {
        const nb_q5_0 *x0 = (const nb_q5_0 *)(w + rb * r), *x1 = (const nb_q5_0 *)(w + rb * (r + 1)), *x2 = (const nb_q5_0 *)(w + rb * (r + 2)), *x3 = (const nb_q5_0 *)(w + rb * (r + 3));
        float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < nb; b++) {
            const int8x16_t y0 = vld1q_s8(y[b].qs), y1 = vld1q_s8(y[b].qs + 16); const float yd = y[b].d;
            ACC(a0, q5_dot(x0 + b, bits, i0, i1, y0, y1), x0[b].d, yd); ACC(a1, q5_dot(x1 + b, bits, i0, i1, y0, y1), x1[b].d, yd);
            ACC(a2, q5_dot(x2 + b, bits, i0, i1, y0, y1), x2[b].d, yd); ACC(a3, q5_dot(x3 + b, bits, i0, i1, y0, y1), x3[b].d, yd);
        }
        out[r] = vaddvq_f32(a0); out[r + 1] = vaddvq_f32(a1); out[r + 2] = vaddvq_f32(a2); out[r + 3] = vaddvq_f32(a3);
    }
    for (; r < r1; r++) {
        const nb_q5_0 *x = (const nb_q5_0 *)(w + rb * r); float32x4_t a = vdupq_n_f32(0);
        for (int b = 0; b < nb; b++) ACC(a, q5_dot(x + b, bits, i0, i1, vld1q_s8(y[b].qs), vld1q_s8(y[b].qs + 16)), x[b].d, y[b].d);
        out[r] = vaddvq_f32(a);
    }
}
/* sum_j mn[j] * (bsums[2j] + bsums[2j+1]) for the 8 sub-blocks of a K-quant super-block, in vector form */
static inline int kmins(const uint8_t *mn, const int16_t *bs) {
    const int16x8_t p = vpaddq_s16(vld1q_s16(bs), vld1q_s16(bs + 8)), m = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(mn)));
    return vaddvq_s32(vaddq_s32(vmull_s16(vget_low_s16(p), vget_low_s16(m)), vmull_high_s16(p, m)));
}
void es_mv_q4_K_neon(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 144; const uint8x16_t m = vdupq_n_u8(15);
    for (int r = r0; r < r1; r++) {
        const nb_q4_K *x = (const nb_q4_K *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; j++) scale_min(j, x[b].scales, &sc[j], &mn[j]);
            int32x4_t accv = vdupq_n_s32(0); const uint8_t *q = x[b].qs; const int8_t *q8 = y[b].qs;
            for (int c = 0; c < 4; c++, q += 32, q8 += 64) {
                const uint8x16_t qa = vld1q_u8(q), qb = vld1q_u8(q + 16);
                int32x4_t lo = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(qa, m)), vld1q_s8(q8));
                lo = vdotq_s32(lo, vreinterpretq_s8_u8(vandq_u8(qb, m)), vld1q_s8(q8 + 16));
                int32x4_t hi = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8(qa, 4)), vld1q_s8(q8 + 32));
                hi = vdotq_s32(hi, vreinterpretq_s8_u8(vshrq_n_u8(qb, 4)), vld1q_s8(q8 + 48));
                accv = vmlaq_n_s32(accv, lo, sc[2 * c]); accv = vmlaq_n_s32(accv, hi, sc[2 * c + 1]);
            }
            s += y[b].d * (es_fp16_to_fp32(x[b].d) * (float)hsum(accv) - es_fp16_to_fp32(x[b].dmin) * (float)kmins(mn, y[b].bsums));
        }
        out[r] = s;
    }
}
void es_mv_q6_K_neon(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 210;
    const uint8x16_t m15 = vdupq_n_u8(15), m3 = vdupq_n_u8(3); const int8x16_t c32 = vdupq_n_s8(32);
    for (int r = r0; r < r1; r++) {
        const nb_q6_K *x = (const nb_q6_K *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            const uint8_t *ql = x[b].ql, *qh = x[b].qh; const int8_t *sc = x[b].scales, *q8 = y[b].qs; int32x4_t accv = vdupq_n_s32(0);
            for (int h = 0; h < 2; h++, ql += 64, qh += 32, sc += 8, q8 += 128) {
                for (int is = 0; is < 2; is++) {
                    const int l = is * 16;
                    const uint8x16_t a0 = vld1q_u8(ql + l), a1 = vld1q_u8(ql + l + 32), hh = vld1q_u8(qh + l);
                    const int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(a0, m15), vshlq_n_u8(vandq_u8(hh, m3), 4))), c32);
                    const int8x16_t q2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(a1, m15), vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 2), m3), 4))), c32);
                    const int8x16_t q3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(a0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 4), m3), 4))), c32);
                    const int8x16_t q4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(a1, 4), vshlq_n_u8(vshrq_n_u8(hh, 6), 4))), c32);
                    accv = vmlaq_n_s32(accv, vdotq_s32(vdupq_n_s32(0), q1, vld1q_s8(q8 + l)), sc[is]);
                    accv = vmlaq_n_s32(accv, vdotq_s32(vdupq_n_s32(0), q2, vld1q_s8(q8 + l + 32)), sc[is + 2]);
                    accv = vmlaq_n_s32(accv, vdotq_s32(vdupq_n_s32(0), q3, vld1q_s8(q8 + l + 64)), sc[is + 4]);
                    accv = vmlaq_n_s32(accv, vdotq_s32(vdupq_n_s32(0), q4, vld1q_s8(q8 + l + 96)), sc[is + 6]);
                }
            }
            s += y[b].d * es_fp16_to_fp32(x[b].d) * (float)hsum(accv);
        }
        out[r] = s;
    }
}

void es_mv_q3_K_neon(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 110; const uint8x16_t m3 = vdupq_n_u8(3), four = vdupq_n_u8(4);
    for (int r = r0; r < r1; r++) {
        const nb_q3_K *x = (const nb_q3_K *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            uint32_t aux[4]; memcpy(aux, x[b].scales, 12);
            const uint32_t tmp = aux[2], k1 = 0x03030303, k2 = 0x0f0f0f0f;
            aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4); aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
            aux[0] = (aux[0] & k2) | (((tmp >> 0) & k1) << 4); aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
            const int8_t *sc = (const int8_t *)aux; const uint8_t *q = x[b].qs, *hm = x[b].hmask; const int8_t *q8 = y[b].qs;
            int32x4_t accv = vdupq_n_s32(0); int is = 0; uint8_t m = 1;
            for (int n = 0; n < 2; n++, q += 32) {
                for (int j = 0; j < 4; j++, m <<= 1) {
                    const int8x16_t sh = vdupq_n_s8((int8_t)(-2 * j));
                    for (int half = 0; half < 2; half++, q8 += 16) {
                        const uint8x16_t t = vandq_u8(vshlq_u8(vld1q_u8(q + half * 16), sh), m3);
                        const uint8x16_t has = vtstq_u8(vld1q_u8(hm + half * 16), vdupq_n_u8(m));
                        const int8x16_t v = vreinterpretq_s8_u8(vsubq_u8(t, vbicq_u8(four, has)));
                        accv = vmlaq_n_s32(accv, vdotq_s32(vdupq_n_s32(0), v, vld1q_s8(q8)), sc[is++] - 32);
                    }
                }
            }
            s += y[b].d * es_fp16_to_fp32(x[b].d) * (float)hsum(accv);
        }
        out[r] = s;
    }
}
void es_mv_q5_K_neon(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 176; const uint8x16_t m15 = vdupq_n_u8(15), b16 = vdupq_n_u8(16);
    for (int r = r0; r < r1; r++) {
        const nb_q5_K *x = (const nb_q5_K *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; j++) scale_min(j, x[b].scales, &sc[j], &mn[j]);
            int32x4_t accv = vdupq_n_s32(0); const uint8_t *ql = x[b].qs, *qh = x[b].qh; const int8_t *q8 = y[b].qs; uint8_t u1 = 1, u2 = 2;
            for (int c = 0; c < 4; c++, ql += 32, q8 += 64, u1 <<= 2, u2 <<= 2) {
                int32x4_t lo = vdupq_n_s32(0), hi = vdupq_n_s32(0);
                for (int h = 0; h < 32; h += 16) {
                    const uint8x16_t qa = vld1q_u8(ql + h), qq = vld1q_u8(qh + h);
                    const uint8x16_t l5 = vorrq_u8(vandq_u8(qa, m15), vandq_u8(vtstq_u8(qq, vdupq_n_u8(u1)), b16));
                    const uint8x16_t h5 = vorrq_u8(vshrq_n_u8(qa, 4), vandq_u8(vtstq_u8(qq, vdupq_n_u8(u2)), b16));
                    lo = vdotq_s32(lo, vreinterpretq_s8_u8(l5), vld1q_s8(q8 + h)); hi = vdotq_s32(hi, vreinterpretq_s8_u8(h5), vld1q_s8(q8 + 32 + h));
                }
                accv = vmlaq_n_s32(accv, lo, sc[2 * c]); accv = vmlaq_n_s32(accv, hi, sc[2 * c + 1]);
            }
            s += y[b].d * (es_fp16_to_fp32(x[b].d) * (float)hsum(accv) - es_fp16_to_fp32(x[b].dmin) * (float)kmins(mn, y[b].bsums));
        }
        out[r] = s;
    }
}

/* ---- several activations against the same rows (batched prompt reading, Turbo checks) ----
 * A group of rows is unpacked once into thread-local scratch (the same signed bytes the kernels above build in registers); every
 * activation then runs the very same SDOT/FMA sequence, so results are bit-identical and the unpacking is not repeated per activation. */
#include <stdlib.h>
static _Thread_local uint8_t *NMB; static _Thread_local size_t NMBCAP;
static uint8_t *nmb(size_t n) {
    n = (n + 63) & ~(size_t)63;
    if (n > NMBCAP) { free(NMB); NMB = aligned_alloc(64, n); NMBCAP = NMB ? n : 0; }
    return NMB;
}
static void q4q5_multi(int five, const uint8_t *w, int cols, int r0, int r1, const es_act *A, int na, float *Y, size_t ys) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * (five ? 22 : 18);
    uint8_t *U = nmb((size_t)4 * nb * 32 + (size_t)4 * nb * sizeof(float)); if (!U) return;
    float *D = (float *)(U + (size_t)4 * nb * 32);
    const uint8x16_t m = vdupq_n_u8(15), bits = vld1q_u8(Q5_BITS), i0 = vld1q_u8(Q5_I0), i1 = vld1q_u8(Q5_I1), b16 = vdupq_n_u8(16); const int8x16_t e = vdupq_n_s8(8);
    for (int r = r0; r < r1; r += 4) {
        const int nr = r1 - r < 4 ? r1 - r : 4;
        for (int i = 0; i < nr; i++)
            for (int b = 0; b < nb; b++) {
                int8_t *o = (int8_t *)(U + ((size_t)i * nb + b) * 32);
                if (five) {
                    const nb_q5_0 *x = (const nb_q5_0 *)(w + rb * (r + i)) + b;
                    uint32x4_t qv32 = vdupq_n_u32(0); qv32 = vld1q_lane_u32((const uint32_t *)x->qh, qv32, 0);
                    const uint8x16_t qv = vreinterpretq_u8_u32(qv32), q = vld1q_u8(x->qs);
                    const uint8x16_t h0 = vandq_u8(vtstq_u8(vqtbl1q_u8(qv, i0), bits), b16), h1 = vandq_u8(vtstq_u8(vqtbl1q_u8(qv, i1), bits), b16);
                    vst1q_s8(o, vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(vandq_u8(q, m), h0), b16)));
                    vst1q_s8(o + 16, vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(vshrq_n_u8(q, 4), h1), b16)));
                    D[i * nb + b] = es_fp16_to_fp32(x->d);
                } else {
                    const nb_q4_0 *x = (const nb_q4_0 *)(w + rb * (r + i)) + b; const uint8x16_t q = vld1q_u8(x->qs);
                    vst1q_s8(o, vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m)), e)); vst1q_s8(o + 16, vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), e));
                    D[i * nb + b] = es_fp16_to_fp32(x->d);
                }
            }
#define UL(i, b) vld1q_s8((const int8_t *)(U + ((size_t)(i) * nb + (b)) * 32))
#define UH(i, b) vld1q_s8((const int8_t *)(U + ((size_t)(i) * nb + (b)) * 32 + 16))
#define MACC(a, i) (a) = vfmaq_n_f32((a), vcvtq_f32_s32(DOT2(UL(i, b), UH(i, b), y0, y1)), D[(i) * nb + b] * yd)
        for (int qa = 0; qa < na; qa++) {
            const es_blk_q80 *y = A[qa].q80; float *out = Y + (size_t)qa * ys;
            if (nr == 4) {
                float32x4_t a0 = vdupq_n_f32(0), a1 = a0, a2 = a0, a3 = a0;
                for (int b = 0; b < nb; b++) {
                    const int8x16_t y0 = vld1q_s8(y[b].qs), y1 = vld1q_s8(y[b].qs + 16); const float yd = y[b].d;
                    MACC(a0, 0); MACC(a1, 1); MACC(a2, 2); MACC(a3, 3);
                }
                out[r] = vaddvq_f32(a0); out[r + 1] = vaddvq_f32(a1); out[r + 2] = vaddvq_f32(a2); out[r + 3] = vaddvq_f32(a3);
            } else
                for (int i = 0; i < nr; i++) {
                    float32x4_t a = vdupq_n_f32(0);
                    for (int b = 0; b < nb; b++) { const int8x16_t y0 = vld1q_s8(y[b].qs), y1 = vld1q_s8(y[b].qs + 16); const float yd = y[b].d; MACC(a, i); }
                    out[r + i] = vaddvq_f32(a);
                }
        }
#undef UL
#undef UH
#undef MACC
    }
}
typedef struct { float d, dmin; uint8_t sc[8], mn[8]; } nk4_meta;
static void q4K_multi(const uint8_t *w, int cols, int r0, int r1, const es_act *A, int na, float *Y, size_t ys) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 144; const uint8x16_t m = vdupq_n_u8(15);
    uint8_t *U = nmb((size_t)4 * nb * 256 + (size_t)4 * nb * sizeof(nk4_meta)); if (!U) return;
    nk4_meta *K = (nk4_meta *)(U + (size_t)4 * nb * 256);
    for (int r = r0; r < r1; r += 4) {
        const int nr = r1 - r < 4 ? r1 - r : 4;
        for (int i = 0; i < nr; i++) {
            const nb_q4_K *x = (const nb_q4_K *)(w + rb * (r + i));
            for (int b = 0; b < nb; b++) {
                nk4_meta *k = &K[i * nb + b]; uint8_t *o = U + ((size_t)i * nb + b) * 256; const uint8_t *q = x[b].qs;
                for (int j = 0; j < 8; j++) scale_min(j, x[b].scales, &k->sc[j], &k->mn[j]);
                k->d = es_fp16_to_fp32(x[b].d); k->dmin = es_fp16_to_fp32(x[b].dmin);
                for (int c = 0; c < 4; c++, q += 32) {
                    const uint8x16_t qa = vld1q_u8(q), qb = vld1q_u8(q + 16);
                    vst1q_u8(o + 64 * c, vandq_u8(qa, m)); vst1q_u8(o + 64 * c + 16, vandq_u8(qb, m));
                    vst1q_u8(o + 64 * c + 32, vshrq_n_u8(qa, 4)); vst1q_u8(o + 64 * c + 48, vshrq_n_u8(qb, 4));
                }
            }
        }
        for (int qa = 0; qa < na; qa++) {
            const es_block_q8_K *y = A[qa].q8; float *out = Y + (size_t)qa * ys;
            for (int i = 0; i < nr; i++) {
                float s = 0;
                for (int b = 0; b < nb; b++) {
                    const nk4_meta *k = &K[i * nb + b]; const int8_t *o = (const int8_t *)(U + ((size_t)i * nb + b) * 256), *q8 = y[b].qs; int32x4_t accv = vdupq_n_s32(0);
                    for (int c = 0; c < 4; c++, q8 += 64) {
                        int32x4_t lo = vdotq_s32(vdupq_n_s32(0), vld1q_s8(o + 64 * c), vld1q_s8(q8));
                        lo = vdotq_s32(lo, vld1q_s8(o + 64 * c + 16), vld1q_s8(q8 + 16));
                        int32x4_t hi = vdotq_s32(vdupq_n_s32(0), vld1q_s8(o + 64 * c + 32), vld1q_s8(q8 + 32));
                        hi = vdotq_s32(hi, vld1q_s8(o + 64 * c + 48), vld1q_s8(q8 + 48));
                        accv = vmlaq_n_s32(accv, lo, k->sc[2 * c]); accv = vmlaq_n_s32(accv, hi, k->sc[2 * c + 1]);
                    }
                    s += y[b].d * (k->d * (float)hsum(accv) - k->dmin * (float)kmins(k->mn, y[b].bsums));
                }
                out[r + i] = s;
            }
        }
    }
}
static void q6K_multi(const uint8_t *w, int cols, int r0, int r1, const es_act *A, int na, float *Y, size_t ys) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 210;
    const uint8x16_t m15 = vdupq_n_u8(15), m3 = vdupq_n_u8(3); const int8x16_t c32 = vdupq_n_s8(32);
    uint8_t *U = nmb((size_t)4 * nb * 256 + (size_t)4 * nb * 20); if (!U) return;
    uint8_t *M6 = U + (size_t)4 * nb * 256;
    for (int r = r0; r < r1; r += 4) {
        const int nr = r1 - r < 4 ? r1 - r : 4;
        for (int i = 0; i < nr; i++) {
            const nb_q6_K *x = (const nb_q6_K *)(w + rb * (r + i));
            for (int b = 0; b < nb; b++) {
                const uint8_t *ql = x[b].ql, *qh = x[b].qh; int8_t *o = (int8_t *)(U + ((size_t)i * nb + b) * 256);
                for (int h = 0; h < 2; h++, ql += 64, qh += 32)
                    for (int is = 0; is < 2; is++, o += 64) {
                        const int l = is * 16;
                        const uint8x16_t a0 = vld1q_u8(ql + l), a1 = vld1q_u8(ql + l + 32), hh = vld1q_u8(qh + l);
                        vst1q_s8(o, vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(a0, m15), vshlq_n_u8(vandq_u8(hh, m3), 4))), c32));
                        vst1q_s8(o + 16, vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(a1, m15), vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 2), m3), 4))), c32));
                        vst1q_s8(o + 32, vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(a0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 4), m3), 4))), c32));
                        vst1q_s8(o + 48, vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(a1, 4), vshlq_n_u8(vshrq_n_u8(hh, 6), 4))), c32));
                    }
                uint8_t *mm = M6 + ((size_t)i * nb + b) * 20; memcpy(mm, x[b].scales, 16); const float d = es_fp16_to_fp32(x[b].d); memcpy(mm + 16, &d, 4);
            }
        }
        for (int qa = 0; qa < na; qa++) {
            const es_block_q8_K *y = A[qa].q8; float *out = Y + (size_t)qa * ys;
            for (int i = 0; i < nr; i++) {
                float s = 0;
                for (int b = 0; b < nb; b++) {
                    const int8_t *o = (const int8_t *)(U + ((size_t)i * nb + b) * 256); const uint8_t *mm = M6 + ((size_t)i * nb + b) * 20;
                    const int8_t *sc = (const int8_t *)mm, *q8 = y[b].qs; float d; memcpy(&d, mm + 16, 4); int32x4_t accv = vdupq_n_s32(0);
                    for (int h = 0; h < 2; h++, sc += 8, q8 += 128)
                        for (int is = 0; is < 2; is++, o += 64) {
                            const int l = is * 16;
                            accv = vmlaq_n_s32(accv, vdotq_s32(vdupq_n_s32(0), vld1q_s8(o), vld1q_s8(q8 + l)), sc[is]);
                            accv = vmlaq_n_s32(accv, vdotq_s32(vdupq_n_s32(0), vld1q_s8(o + 16), vld1q_s8(q8 + l + 32)), sc[is + 2]);
                            accv = vmlaq_n_s32(accv, vdotq_s32(vdupq_n_s32(0), vld1q_s8(o + 32), vld1q_s8(q8 + l + 64)), sc[is + 4]);
                            accv = vmlaq_n_s32(accv, vdotq_s32(vdupq_n_s32(0), vld1q_s8(o + 48), vld1q_s8(q8 + l + 96)), sc[is + 6]);
                        }
                    s += y[b].d * d * (float)hsum(accv);
                }
                out[r + i] = s;
            }
        }
    }
}
/* ---- ternary formats (TQ1_0, TQ2_0) against Q8_K activations: digit u in 0..2 for the value u - 1, so sum(u a) - sum(a) ---- */
void es_mv_tq2_0_neon(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 66; const uint8x16_t m3 = vdupq_n_u8(3);
    for (int r = r0; r < r1; r++) {
        const uint8_t *x = w + rb * r; float s = 0;
        for (int b = 0; b < nb; b++, x += 66) {
            const int8_t *q8 = y[b].qs; int32x4_t acc = vdupq_n_s32(0); int ys = 0;
            for (int g = 0; g < 16; g++) ys += y[b].bsums[g];
            for (int j = 0; j < 64; j += 32) {
                const uint8x16_t qa = vld1q_u8(x + j), qb = vld1q_u8(x + j + 16);
                for (int l = 0; l < 4; l++, q8 += 32) {
                    const int sh = 2 * l;
                    const uint8x16_t ua = vandq_u8(vshlq_u8(qa, vdupq_n_s8((int8_t)-sh)), m3), ub = vandq_u8(vshlq_u8(qb, vdupq_n_s8((int8_t)-sh)), m3);
                    acc = vdotq_s32(acc, vreinterpretq_s8_u8(ua), vld1q_s8(q8));
                    acc = vdotq_s32(acc, vreinterpretq_s8_u8(ub), vld1q_s8(q8 + 16));
                }
            }
            s += y[b].d * es_fp16_to_fp32(*(const uint16_t *)(x + 64)) * (float)(hsum(acc) - ys);
        }
        out[r] = s;
    }
}
/* digit k of each byte q: ((q * 3^k mod 256) * 3) >> 8 */
static inline uint8x16_t tq1_digit(uint8x16_t x, uint8_t c) {
    const uint8x16_t q = vmulq_u8(x, vdupq_n_u8(c));
    return vcombine_u8(vshrn_n_u16(vmull_u8(vget_low_u8(q), vdup_n_u8(3)), 8), vshrn_n_u16(vmull_high_u8(q, vdupq_n_u8(3)), 8));
}
void es_mv_tq1_0_neon(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    static const uint8_t pow3[5] = {1, 3, 9, 27, 81};
    const int nb = cols / 256; const size_t rb = (size_t)nb * 54;
    for (int r = r0; r < r1; r++) {
        const uint8_t *x = w + rb * r; float s = 0;
        for (int b = 0; b < nb; b++, x += 54) {
            const int8_t *q8 = y[b].qs; int32x4_t acc = vdupq_n_s32(0); int ys = 0, tail = 0;
            for (int g = 0; g < 16; g++) ys += y[b].bsums[g];
            const uint8x16_t qa = vld1q_u8(x), qb = vld1q_u8(x + 16), qc = vld1q_u8(x + 32);
            for (int k = 0; k < 5; k++) {
                acc = vdotq_s32(acc, vreinterpretq_s8_u8(tq1_digit(qa, pow3[k])), vld1q_s8(q8 + 32 * k));
                acc = vdotq_s32(acc, vreinterpretq_s8_u8(tq1_digit(qb, pow3[k])), vld1q_s8(q8 + 32 * k + 16));
                acc = vdotq_s32(acc, vreinterpretq_s8_u8(tq1_digit(qc, pow3[k])), vld1q_s8(q8 + 160 + 16 * k));
            }
            for (int k = 0; k < 4; k++) for (int j = 0; j < 4; j++) tail += ((((uint16_t)(uint8_t)(x[48 + j] * pow3[k])) * 3) >> 8) * q8[240 + 4 * k + j];
            s += y[b].d * es_fp16_to_fp32(*(const uint16_t *)(x + 52)) * (float)(hsum(acc) + tail - ys);
        }
        out[r] = s;
    }
}

/* ---- low-bit formats (IQ1/IQ2/IQ3/IQ4, MXFP4) against Q8_0 activations, one activation (generating) ----
 * The same decoding as mv_lowbit_n in es_quant.c, but codebook rows and signs are assembled in registers, each 32 weights take two
 * SDOT and one float multiply-add, each row's sum stays in a vector, and the activation sums for the IQ1 offsets are made once. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-const-variable"
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "es_iq_tables.h"
#pragma GCC diagnostic pop
static const int8_t N_IQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
#define N_IQ1S_DELTA 0.125f
static uint64_t NSM[256];   /* sign bits -> bytes of 0xFF where negative */
static void nsm_init(void) {
    if (NSM[255]) return;
    for (int b = 0; b < 256; b++) { uint64_t m = 0; for (int j = 0; j < 8; j++) if (b >> j & 1) m |= 0xFFull << (8 * j); NSM[b] = m; }
}
static inline float n_h2f(const uint8_t *p) { uint16_t h; memcpy(&h, p, 2); return es_fp16_to_fp32(h); }
static inline float n_e8m0_half(uint8_t x) { uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t)(x - 1) << 23); float f; memcpy(&f, &bits, 4); return f; }
static inline int8x16_t g2(uint64_t a, uint64_t b) { return vcombine_s8(vcreate_s8(a), vcreate_s8(b)); }
static inline int8x16_t sg2(uint64_t a, uint64_t b, uint8_t sa, uint8_t sb) {   /* two positive codebook rows with their 8 sign bits each */
    const int8x16_t m = g2(NSM[sa], NSM[sb]);
    return vsubq_s8(veorq_s8(g2(a, b), m), m);
}
static inline int8x16_t lut16(const int8_t *lut, uint8x16_t idx) { return vqtbl1q_s8(vld1q_s8(lut), idx); }
#define NACC(w0, w1, lo, hi) do { const int32x4_t a_ = vdotq_s32(vdupq_n_s32(0), (w0), vld1q_s8(A->qs)), b_ = vdotq_s32(vdupq_n_s32(0), (w1), vld1q_s8(A->qs + 16)); \
    acc = vfmaq_n_f32(acc, vcvtq_f32_s32(a_), (lo)); acc = vfmaq_n_f32(acc, vcvtq_f32_s32(b_), (hi)); } while (0)
static int nasum8(const int8_t *a) { int s = 0; for (int i = 0; i < 8; i++) s += a[i]; return s; }
int es_mv_lowbit_neon(uint32_t t, const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    if (!(t == GGML_IQ2_XXS || t == GGML_IQ2_XS || t == GGML_IQ2_S || t == GGML_IQ3_XXS || t == GGML_IQ3_S || t == GGML_IQ1_S || t == GGML_IQ1_M ||
          t == GGML_IQ4_NL || t == GGML_IQ4_XS || t == GGML_MXFP4 || t == GGML_IQ1_XS || t == GGML_IQ1_XXS || t == GGML_IQ1_XXXS)) return 0;
    const size_t rb = es_row_bytes(t, cols);
    static _Thread_local int *S8; static _Thread_local int S8CAP;
    const int iq1 = t == GGML_IQ1_S || t == GGML_IQ1_M || t == GGML_IQ1_XS || t == GGML_IQ1_XXS || t == GGML_IQ1_XXXS;
    if (iq1) {
        if (S8CAP < cols / 8) { free(S8); S8 = malloc(sizeof(int) * (size_t)(cols / 8)); S8CAP = S8 ? cols / 8 : 0; if (!S8) return 0; }
        for (int g = 0; g < cols / 8; g++) S8[g] = nasum8(y[g / 4].qs + 8 * (g % 4));
    }
    nsm_init();
    const uint8x16_t m15 = vdupq_n_u8(15);
    for (int r = r0; r < r1; r++) {
        const uint8_t *p = w + rb * (size_t)r;
        float32x4_t acc = vdupq_n_f32(0); float sd = 0;
        const es_blk_q80 *A;
        switch (t) {
        case GGML_IQ1_XS: case GGML_IQ1_XXS: case GGML_IQ1_XXXS: {
            const int bs = t == GGML_IQ1_XS ? 46 : t == GGML_IQ1_XXS ? 42 : 38;
            for (int i = 0; i < cols / 256; i++, p += bs) {
                const float d = n_h2f(p);
                for (int ib = 0; ib < 8; ib++) {
                    const uint8_t *qs = p + 2 + 4 * ib; int nib; uint64_t g[4];
                    if (t == GGML_IQ1_XXS) { const int h = p[34 + ib]; for (int l = 0; l < 4; l++) g[l] = ES_IQ1XXS_GRID[qs[l] | (((h >> l) & 1) << 8)]; nib = ((h >> 4) & 7) | ((h >> 4) & 8); }
                    else if (t == GGML_IQ1_XS) { const int h = p[34 + ib]; for (int l = 0; l < 4; l++) g[l] = ES_IQ1XS_GRID[qs[l] | (((h >> 2 * l) & 3) << 8)]; nib = (p[42 + ib / 2] >> (4 * (ib & 1))) & 0xf; }
                    else { for (int l = 0; l < 4; l++) g[l] = ES_IQ1XXXS_GRID[qs[l]]; nib = (p[34 + ib / 2] >> (4 * (ib & 1))) & 0xf; }
                    const float delta = nib & 8 ? -N_IQ1S_DELTA : N_IQ1S_DELTA;
                    A = &y[i * 8 + ib]; const float f = d * (float)(2 * (nib & 7) + 1) * A->d; const int *s8 = S8 + 4 * (i * 8 + ib);
                    NACC(g2(g[0], g[1]), g2(g[2], g[3]), f, f); sd += f * delta * (float)(s8[0] + s8[1] + s8[2] + s8[3]);
                }
            }
            break;
        }
        case GGML_MXFP4:
            for (int b = 0; b < cols / 32; b++, p += 17) {
                const uint8x16_t x = vld1q_u8(p + 1); A = &y[b]; const float f = n_e8m0_half(p[0]) * A->d;
                NACC(lut16(ES_KVALUES_FP4, vandq_u8(x, m15)), lut16(ES_KVALUES_FP4, vshrq_n_u8(x, 4)), f, f);
            }
            break;
        case GGML_IQ4_NL:
            for (int b = 0; b < cols / 32; b++, p += 18) {
                const uint8x16_t x = vld1q_u8(p + 2); A = &y[b]; const float f = n_h2f(p) * A->d;
                NACC(lut16(N_IQ4NL, vandq_u8(x, m15)), lut16(N_IQ4NL, vshrq_n_u8(x, 4)), f, f);
            }
            break;
        case GGML_IQ4_XS:
            for (int i = 0; i < cols / 256; i++, p += 136) {
                uint16_t sh; memcpy(&sh, p + 2, 2);
                const float d = n_h2f(p);
                const uint8_t *sl = p + 4, *qs = p + 8;
                for (int ib = 0; ib < 8; ib++, qs += 16) {
                    const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
                    const uint8x16_t x = vld1q_u8(qs); A = &y[i * 8 + ib]; const float f = d * (float)(ls - 32) * A->d;
                    NACC(lut16(N_IQ4NL, vandq_u8(x, m15)), lut16(N_IQ4NL, vshrq_n_u8(x, 4)), f, f);
                }
            }
            break;
        case GGML_IQ2_XXS:
            for (int i = 0; i < cols / 256; i++, p += 66) {
                const float d = n_h2f(p);
                for (int ib = 0; ib < 8; ib++) {
                    uint32_t a2[2]; memcpy(a2, p + 2 + 8 * ib, 8);
                    const uint8_t *a8 = (const uint8_t *)a2;
                    A = &y[i * 8 + ib]; const float f = d * (0.5f + (float)(a2[1] >> 28)) * 0.25f * A->d;
                    NACC(sg2(ES_IQ2XXS_GRID[a8[0]], ES_IQ2XXS_GRID[a8[1]], ES_KSIGNS_IQ2XS[a2[1] & 127], ES_KSIGNS_IQ2XS[(a2[1] >> 7) & 127]),
                         sg2(ES_IQ2XXS_GRID[a8[2]], ES_IQ2XXS_GRID[a8[3]], ES_KSIGNS_IQ2XS[(a2[1] >> 14) & 127], ES_KSIGNS_IQ2XS[(a2[1] >> 21) & 127]), f, f);
                }
            }
            break;
        case GGML_IQ2_XS:
            for (int i = 0; i < cols / 256; i++, p += 74) {
                const float d = n_h2f(p);
                uint16_t qs[32]; memcpy(qs, p + 2, 64);
                const uint8_t *sc = p + 66;
                for (int ib = 0; ib < 8; ib++) {
                    const uint16_t *e = qs + 4 * ib;
                    A = &y[i * 8 + ib]; const float f = d * 0.25f * A->d;
                    NACC(sg2(ES_IQ2XS_GRID[e[0] & 511], ES_IQ2XS_GRID[e[1] & 511], ES_KSIGNS_IQ2XS[e[0] >> 9], ES_KSIGNS_IQ2XS[e[1] >> 9]),
                         sg2(ES_IQ2XS_GRID[e[2] & 511], ES_IQ2XS_GRID[e[3] & 511], ES_KSIGNS_IQ2XS[e[2] >> 9], ES_KSIGNS_IQ2XS[e[3] >> 9]),
                         f * (0.5f + (sc[ib] & 0xf)), f * (0.5f + (sc[ib] >> 4)));
                }
            }
            break;
        case GGML_IQ2_S:
            for (int i = 0; i < cols / 256; i++, p += 82) {
                const float d = n_h2f(p);
                const uint8_t *qs = p + 2, *signs = p + 34, *qh = p + 66, *sc = p + 74;
                for (int ib = 0; ib < 8; ib++, qs += 4, signs += 4) {
                    const int hh = qh[ib];
                    A = &y[i * 8 + ib]; const float f = d * 0.25f * A->d;
                    NACC(sg2(ES_IQ2S_GRID[qs[0] | ((hh << 8) & 0x300)], ES_IQ2S_GRID[qs[1] | ((hh << 6) & 0x300)], signs[0], signs[1]),
                         sg2(ES_IQ2S_GRID[qs[2] | ((hh << 4) & 0x300)], ES_IQ2S_GRID[qs[3] | ((hh << 2) & 0x300)], signs[2], signs[3]),
                         f * (0.5f + (sc[ib] & 0xf)), f * (0.5f + (sc[ib] >> 4)));
                }
            }
            break;
        case GGML_IQ3_XXS:
            for (int i = 0; i < cols / 256; i++, p += 98) {
                const float d = n_h2f(p);
                const uint8_t *qs = p + 2, *ss = p + 66;
                for (int ib = 0; ib < 8; ib++, qs += 8) {
                    uint32_t a2; memcpy(&a2, ss + 4 * ib, 4);
#define G3X(j) ((uint64_t)ES_IQ3XXS_GRID[qs[2 * (j)]] | (uint64_t)ES_IQ3XXS_GRID[qs[2 * (j) + 1]] << 32)
                    A = &y[i * 8 + ib]; const float f = d * (0.5f + (float)(a2 >> 28)) * 0.5f * A->d;
                    NACC(sg2(G3X(0), G3X(1), ES_KSIGNS_IQ2XS[a2 & 127], ES_KSIGNS_IQ2XS[(a2 >> 7) & 127]),
                         sg2(G3X(2), G3X(3), ES_KSIGNS_IQ2XS[(a2 >> 14) & 127], ES_KSIGNS_IQ2XS[(a2 >> 21) & 127]), f, f);
#undef G3X
                }
            }
            break;
        case GGML_IQ3_S:
            for (int i = 0; i < cols / 256; i++, p += 110) {
                const float d = n_h2f(p);
                const uint8_t *qs = p + 2, *qh = p + 66, *signs = p + 74, *sc = p + 106;
                for (int ib = 0; ib < 8; ib++, qs += 8, signs += 4) {
                    const int h = ib & 1, hb = qh[ib];
#define G3S(l) ((uint64_t)ES_IQ3S_GRID[qs[2 * (l)] | ((hb << (8 - 2 * (l))) & 256)] | (uint64_t)ES_IQ3S_GRID[qs[2 * (l) + 1] | ((hb << (7 - 2 * (l))) & 256)] << 32)
                    A = &y[i * 8 + ib]; const float f = d * (float)(1 + 2 * (h ? sc[ib / 2] >> 4 : sc[ib / 2] & 0xf)) * A->d;
                    NACC(sg2(G3S(0), G3S(1), signs[0], signs[1]), sg2(G3S(2), G3S(3), signs[2], signs[3]), f, f);
#undef G3S
                }
            }
            break;
        case GGML_IQ1_S:
            for (int i = 0; i < cols / 256; i++, p += 50) {
                const float d = n_h2f(p);
                const uint8_t *qs = p + 2;
                uint16_t qh[8]; memcpy(qh, p + 34, 16);
                for (int ib = 0; ib < 8; ib++, qs += 4) {
                    const int hq = qh[ib];
                    const float delta = hq & 0x8000 ? -N_IQ1S_DELTA : N_IQ1S_DELTA;
                    A = &y[i * 8 + ib]; const float f = d * (float)(2 * ((hq >> 12) & 7) + 1) * A->d; const int *s8 = S8 + 4 * (i * 8 + ib);
                    NACC(g2(ES_IQ1S_GRID[qs[0] | ((hq & 7) << 8)], ES_IQ1S_GRID[qs[1] | ((hq >> 3 & 7) << 8)]),
                         g2(ES_IQ1S_GRID[qs[2] | ((hq >> 6 & 7) << 8)], ES_IQ1S_GRID[qs[3] | ((hq >> 9 & 7) << 8)]), f, f);
                    sd += f * delta * (float)(s8[0] + s8[1] + s8[2] + s8[3]);
                }
            }
            break;
        case GGML_IQ1_M:
            for (int i = 0; i < cols / 256; i++, p += 56) {
                const uint8_t *qs = p, *qh = p + 32;
                uint16_t sc[4]; memcpy(sc, p + 48, 8);
                const float d = es_fp16_to_fp32((uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)));
                for (int ib = 0; ib < 8; ib++, qs += 4, qh += 2) {
                    const float d0 = qh[0] & 0x08 ? -N_IQ1S_DELTA : N_IQ1S_DELTA, d1 = qh[0] & 0x80 ? -N_IQ1S_DELTA : N_IQ1S_DELTA;
                    const float d2 = qh[1] & 0x08 ? -N_IQ1S_DELTA : N_IQ1S_DELTA, d3 = qh[1] & 0x80 ? -N_IQ1S_DELTA : N_IQ1S_DELTA;
                    const float dl1 = (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1), dl2 = (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
                    A = &y[i * 8 + ib]; const float f = A->d * d; const int *s8 = S8 + 4 * (i * 8 + ib);
                    NACC(g2(ES_IQ1S_GRID[qs[0] | ((qh[0] << 8) & 0x700)], ES_IQ1S_GRID[qs[1] | ((qh[0] << 4) & 0x700)]),
                         g2(ES_IQ1S_GRID[qs[2] | ((qh[1] << 8) & 0x700)], ES_IQ1S_GRID[qs[3] | ((qh[1] << 4) & 0x700)]), f * dl1, f * dl2);
                    sd += f * (dl1 * (d0 * (float)s8[0] + d1 * (float)s8[1]) + dl2 * (d2 * (float)s8[2] + d3 * (float)s8[3]));
                }
            }
            break;
        default: return 0;
        }
        out[r] = vaddvq_f32(acc) + sd;
    }
    return 1;
}
#undef NACC

/* 1 if handled here (the caller falls back to one activation at a time otherwise) */
int es_mv_multi_neon(uint32_t t, const uint8_t *w, int cols, int r0, int r1, const es_act *A, int na, float *Y, size_t ys) {
    int q80 = 1, q8 = 1;
    for (int i = 0; i < na; i++) { q80 &= A[i].q80 != NULL; q8 &= A[i].q8 != NULL; }
    if (t == GGML_Q5_0 && q80 && nmb((size_t)4 * (cols / 32) * 36)) { q4q5_multi(1, w, cols, r0, r1, A, na, Y, ys); return 1; }
    if (t == GGML_Q4_0 && q80 && nmb((size_t)4 * (cols / 32) * 36)) { q4q5_multi(0, w, cols, r0, r1, A, na, Y, ys); return 1; }
    if (t == GGML_Q4_K && q8 && nmb((size_t)4 * (cols / 256) * (256 + sizeof(nk4_meta)))) { q4K_multi(w, cols, r0, r1, A, na, Y, ys); return 1; }
    if (t == GGML_Q6_K && q8 && nmb((size_t)4 * (cols / 256) * 276)) { q6K_multi(w, cols, r0, r1, A, na, Y, ys); return 1; }
    return 0;
}
#endif
