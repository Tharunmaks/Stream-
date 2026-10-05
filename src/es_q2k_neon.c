/* Q2_K x Q8_K dot product with NEON + SDOT (armv8.2-a dotprod).
 * This file must be compiled with -march=armv8.2-a+dotprod. It is only
 * called when the CPU reports asimddp at runtime. */
#if defined(__aarch64__)
#include <arm_neon.h>

#include "es_q2k.h"

float es_dot_q2_K_q8_K_neon(int n, const es_block_q2_K *x, const es_block_q8_K *y);

/* one shift step: 32 weights = 2 groups of 16, each with its own scale.
 * QA/QB are the 16-byte quant vectors already shifted right by 0/2/4/6
 * (clang rejects vshrq_n_u8 with a shift of 0, so the caller passes q2a
 * directly for that case). */
#define Q2_STEP(QA, QB, IS)                                                   \
    do {                                                                      \
        const int8x16_t w0 = vreinterpretq_s8_u8(vandq_u8((QA), m3));         \
        const int8x16_t w1 = vreinterpretq_s8_u8(vandq_u8((QB), m3));         \
        const int8x16_t a0 = vld1q_s8(q8);                                    \
        const int8x16_t a1 = vld1q_s8(q8 + 16);                               \
        const int32x4_t p0 = vdotq_s32(vdupq_n_s32(0), w0, a0);               \
        const int32x4_t p1 = vdotq_s32(vdupq_n_s32(0), w1, a1);               \
        acc = vmlaq_n_s32(acc, p0, sc[(IS)] & 15);                            \
        acc = vmlaq_n_s32(acc, p1, sc[(IS) + 1] & 15);                        \
        q8 += 32;                                                             \
    } while (0)

float es_dot_q2_K_q8_K_neon(int n, const es_block_q2_K *x, const es_block_q8_K *y) {
    const int nb = n / ES_QK_K;
    const uint8x16_t m3 = vdupq_n_u8(3);
    float sumf = 0;

    for (int b = 0; b < nb; b++) {
        const uint8_t *sc = x[b].scales;
        const float dall = y[b].d * es_fp16_to_fp32(x[b].d);
        const float dmin = y[b].d * es_fp16_to_fp32(x[b].dmin);

        /* min correction: sum_g bsums[g] * (sc[g] >> 4) */
        const uint8x16_t mins = vshrq_n_u8(vld1q_u8(sc), 4);
        const int16x8_t m_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins)));
        const int16x8_t m_hi = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)));
        const int16x8_t bs_lo = vld1q_s16(y[b].bsums);
        const int16x8_t bs_hi = vld1q_s16(y[b].bsums + 8);
        int32x4_t ms = vmull_s16(vget_low_s16(bs_lo), vget_low_s16(m_lo));
        ms = vmlal_s16(ms, vget_high_s16(bs_lo), vget_high_s16(m_lo));
        ms = vmlal_s16(ms, vget_low_s16(bs_hi), vget_low_s16(m_hi));
        ms = vmlal_s16(ms, vget_high_s16(bs_hi), vget_high_s16(m_hi));
        const int summs = vaddvq_s32(ms);

        const uint8_t *q2 = x[b].qs;
        const int8_t *q8 = y[b].qs;
        int32x4_t acc = vdupq_n_s32(0);
        for (int half = 0; half < 2; half++) {
            const uint8x16_t q2a = vld1q_u8(q2 + 32 * half);
            const uint8x16_t q2b = vld1q_u8(q2 + 32 * half + 16);
            const int is = 8 * half;
            Q2_STEP(q2a, q2b, is + 0);
            Q2_STEP(vshrq_n_u8(q2a, 2), vshrq_n_u8(q2b, 2), is + 2);
            Q2_STEP(vshrq_n_u8(q2a, 4), vshrq_n_u8(q2b, 4), is + 4);
            /* >> 6 leaves only 2 bits, the & 3 in Q2_STEP is then a no-op */
            Q2_STEP(vshrq_n_u8(q2a, 6), vshrq_n_u8(q2b, 6), is + 6);
        }

        sumf += dall * (float)vaddvq_s32(acc) - dmin * (float)summs;
    }
    return sumf;
}
#endif
