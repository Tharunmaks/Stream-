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

void es_mv_q8_0_neon(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 34;
    for (int r = r0; r < r1; r++) {
        const nb_q8_0 *x = (const nb_q8_0 *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            int32x4_t a = vdotq_s32(vdupq_n_s32(0), vld1q_s8(x[b].qs), vld1q_s8(y[b].qs));
            a = vdotq_s32(a, vld1q_s8(x[b].qs + 16), vld1q_s8(y[b].qs + 16));
            s += es_fp16_to_fp32(x[b].d) * y[b].d * (float)hsum(a);
        }
        out[r] = s;
    }
}
void es_mv_q4_0_neon(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 18; const uint8x16_t m = vdupq_n_u8(15); const int8x16_t e = vdupq_n_s8(8);
    for (int r = r0; r < r1; r++) {
        const nb_q4_0 *x = (const nb_q4_0 *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            const uint8x16_t q = vld1q_u8(x[b].qs);
            const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q, m)), e), hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), e);
            int32x4_t a = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(y[b].qs));
            a = vdotq_s32(a, hi, vld1q_s8(y[b].qs + 16));
            s += es_fp16_to_fp32(x[b].d) * y[b].d * (float)hsum(a);
        }
        out[r] = s;
    }
}
void es_mv_q5_0_neon(const uint8_t *w, int cols, int r0, int r1, const es_blk_q80 *y, float *out) {
    const int nb = cols / 32; const size_t rb = (size_t)nb * 22;
    for (int r = r0; r < r1; r++) {
        const nb_q5_0 *x = (const nb_q5_0 *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            uint32x4_t qv32 = vdupq_n_u32(0); qv32 = vld1q_lane_u32((const uint32_t *)x[b].qh, qv32, 0);
            const uint8x16_t qv = vreinterpretq_u8_u32(qv32), q = vld1q_u8(x[b].qs), m15 = vdupq_n_u8(15), b16 = vdupq_n_u8(16);
            static const uint8_t BITS[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128}, I0[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1}, I1[16] = {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
            const uint8x16_t bits = vld1q_u8(BITS);
            const uint8x16_t h0 = vandq_u8(vtstq_u8(vqtbl1q_u8(qv, vld1q_u8(I0)), bits), b16), h1 = vandq_u8(vtstq_u8(vqtbl1q_u8(qv, vld1q_u8(I1)), bits), b16);
            const int8x16_t lo = vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(vandq_u8(q, m15), h0), b16)), hi = vreinterpretq_s8_u8(vsubq_u8(vorrq_u8(vshrq_n_u8(q, 4), h1), b16));
            int32x4_t a = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(y[b].qs));
            a = vdotq_s32(a, hi, vld1q_s8(y[b].qs + 16));
            s += es_fp16_to_fp32(x[b].d) * y[b].d * (float)hsum(a);
        }
        out[r] = s;
    }
}
void es_mv_q4_K_neon(const uint8_t *w, int cols, int r0, int r1, const es_block_q8_K *y, float *out) {
    const int nb = cols / 256; const size_t rb = (size_t)nb * 144; const uint8x16_t m = vdupq_n_u8(15);
    for (int r = r0; r < r1; r++) {
        const nb_q4_K *x = (const nb_q4_K *)(w + rb * r); float s = 0;
        for (int b = 0; b < nb; b++) {
            uint8_t sc[8], mn[8];
            for (int j = 0; j < 8; j++) scale_min(j, x[b].scales, &sc[j], &mn[j]);
            int dot = 0, mins = 0; const uint8_t *q = x[b].qs; const int8_t *q8 = y[b].qs;
            for (int c = 0; c < 4; c++, q += 32, q8 += 64) {
                const uint8x16_t qa = vld1q_u8(q), qb = vld1q_u8(q + 16);
                int32x4_t lo = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(qa, m)), vld1q_s8(q8));
                lo = vdotq_s32(lo, vreinterpretq_s8_u8(vandq_u8(qb, m)), vld1q_s8(q8 + 16));
                int32x4_t hi = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8(qa, 4)), vld1q_s8(q8 + 32));
                hi = vdotq_s32(hi, vreinterpretq_s8_u8(vshrq_n_u8(qb, 4)), vld1q_s8(q8 + 48));
                dot += sc[2 * c] * hsum(lo) + sc[2 * c + 1] * hsum(hi);
                mins += mn[2 * c] * (y[b].bsums[4 * c] + y[b].bsums[4 * c + 1]) + mn[2 * c + 1] * (y[b].bsums[4 * c + 2] + y[b].bsums[4 * c + 3]);
            }
            s += y[b].d * (es_fp16_to_fp32(x[b].d) * (float)dot - es_fp16_to_fp32(x[b].dmin) * (float)mins);
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
            const uint8_t *ql = x[b].ql, *qh = x[b].qh; const int8_t *sc = x[b].scales, *q8 = y[b].qs; int tot = 0;
            for (int h = 0; h < 2; h++, ql += 64, qh += 32, sc += 8, q8 += 128) {
                for (int is = 0; is < 2; is++) {
                    const int l = is * 16;
                    const uint8x16_t a0 = vld1q_u8(ql + l), a1 = vld1q_u8(ql + l + 32), hh = vld1q_u8(qh + l);
                    const int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(a0, m15), vshlq_n_u8(vandq_u8(hh, m3), 4))), c32);
                    const int8x16_t q2 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(a1, m15), vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 2), m3), 4))), c32);
                    const int8x16_t q3 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(a0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hh, 4), m3), 4))), c32);
                    const int8x16_t q4 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(a1, 4), vshlq_n_u8(vshrq_n_u8(hh, 6), 4))), c32);
                    tot += sc[is] * hsum(vdotq_s32(vdupq_n_s32(0), q1, vld1q_s8(q8 + l))) + sc[is + 2] * hsum(vdotq_s32(vdupq_n_s32(0), q2, vld1q_s8(q8 + l + 32)))
                         + sc[is + 4] * hsum(vdotq_s32(vdupq_n_s32(0), q3, vld1q_s8(q8 + l + 64))) + sc[is + 6] * hsum(vdotq_s32(vdupq_n_s32(0), q4, vld1q_s8(q8 + l + 96)));
                }
            }
            s += y[b].d * es_fp16_to_fp32(x[b].d) * (float)tot;
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
            int tot = 0, is = 0; uint8_t m = 1;
            for (int n = 0; n < 2; n++, q += 32) {
                for (int j = 0; j < 4; j++, m <<= 1) {
                    const int8x16_t sh = vdupq_n_s8((int8_t)(-2 * j));
                    for (int half = 0; half < 2; half++, q8 += 16) {
                        const uint8x16_t t = vandq_u8(vshlq_u8(vld1q_u8(q + half * 16), sh), m3);
                        const uint8x16_t has = vtstq_u8(vld1q_u8(hm + half * 16), vdupq_n_u8(m));
                        const int8x16_t v = vreinterpretq_s8_u8(vsubq_u8(t, vbicq_u8(four, has)));
                        tot += (sc[is++] - 32) * hsum(vdotq_s32(vdupq_n_s32(0), v, vld1q_s8(q8)));
                    }
                }
            }
            s += y[b].d * es_fp16_to_fp32(x[b].d) * (float)tot;
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
            int dot = 0, mins = 0; const uint8_t *ql = x[b].qs, *qh = x[b].qh; const int8_t *q8 = y[b].qs; uint8_t u1 = 1, u2 = 2;
            for (int c = 0; c < 4; c++, ql += 32, q8 += 64, u1 <<= 2, u2 <<= 2) {
                int32x4_t lo = vdupq_n_s32(0), hi = vdupq_n_s32(0);
                for (int h = 0; h < 32; h += 16) {
                    const uint8x16_t qa = vld1q_u8(ql + h), qq = vld1q_u8(qh + h);
                    const uint8x16_t l5 = vorrq_u8(vandq_u8(qa, m15), vandq_u8(vtstq_u8(qq, vdupq_n_u8(u1)), b16));
                    const uint8x16_t h5 = vorrq_u8(vshrq_n_u8(qa, 4), vandq_u8(vtstq_u8(qq, vdupq_n_u8(u2)), b16));
                    lo = vdotq_s32(lo, vreinterpretq_s8_u8(l5), vld1q_s8(q8 + h)); hi = vdotq_s32(hi, vreinterpretq_s8_u8(h5), vld1q_s8(q8 + 32 + h));
                }
                dot += sc[2 * c] * hsum(lo) + sc[2 * c + 1] * hsum(hi);
                mins += mn[2 * c] * (y[b].bsums[4 * c] + y[b].bsums[4 * c + 1]) + mn[2 * c + 1] * (y[b].bsums[4 * c + 2] + y[b].bsums[4 * c + 3]);
            }
            s += y[b].d * (es_fp16_to_fp32(x[b].d) * (float)dot - es_fp16_to_fp32(x[b].dmin) * (float)mins);
        }
        out[r] = s;
    }
}
#endif
