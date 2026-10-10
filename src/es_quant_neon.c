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
            uint32_t qh; memcpy(&qh, x[b].qh, 4); int8_t w5[32];
            for (int j = 0; j < 16; j++) {
                w5[j] = (int8_t)(((x[b].qs[j] & 15) | (int)(((qh >> j) << 4) & 0x10)) - 16);
                w5[j + 16] = (int8_t)(((x[b].qs[j] >> 4) | (int)((qh >> (j + 12)) & 0x10)) - 16);
            }
            int32x4_t a = vdotq_s32(vdupq_n_s32(0), vld1q_s8(w5), vld1q_s8(y[b].qs));
            a = vdotq_s32(a, vld1q_s8(w5 + 16), vld1q_s8(y[b].qs + 16));
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
#endif
