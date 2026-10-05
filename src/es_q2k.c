#include "es_q2k.h"

#include <math.h>
#include <stdlib.h>

#include "es_cpu.h"

#if defined(__aarch64__)
float es_dot_q2_K_q8_K_neon(int n, const es_block_q2_K *x, const es_block_q8_K *y);
#endif

uint16_t es_fp32_to_fp16(float f) {
#if defined(__aarch64__)
    __fp16 h = (__fp16)f;
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
#else
    uint32_t b;
    memcpy(&b, &f, 4);
    uint32_t sign = (b >> 16) & 0x8000;
    int32_t exp = (int32_t)((b >> 23) & 0xff) - 127 + 15;
    uint32_t man = b & 0x7fffff;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half = man >> shift;
        if ((man >> (shift - 1)) & 1) half++;
        return (uint16_t)(sign | half);
    }
    uint32_t h = sign | ((uint32_t)exp << 10) | (man >> 13);
    if (man & 0x1000) h++; /* round half up; carries into exponent correctly */
    return (uint16_t)h;
#endif
}

/* position of weight (group g, element l) inside a 256 block, following the
 * layout described in es_q2k.h */
static inline void q2_pos(int g, int l, int *byte, int *shift) {
    int half = g / 8, within = g % 8;   /* 8 groups per half */
    *shift = (within / 2) * 2;
    *byte = 32 * half + (within & 1) * 16 + l;
}

void es_quantize_q2_K_ref(const float *x, es_block_q2_K *y, int n) {
    for (int b = 0; b < n / ES_QK_K; b++, x += ES_QK_K) {
        float scale[16], mn[16], max_scale = 0, max_min = 0;
        for (int g = 0; g < 16; g++) {
            float lo = x[16 * g], hi = x[16 * g];
            for (int l = 1; l < 16; l++) {
                float v = x[16 * g + l];
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
            if (lo > 0) lo = 0; /* mins are subtracted, so must be <= 0 */
            scale[g] = (hi - lo) / 3;
            mn[g] = -lo;
            if (scale[g] > max_scale) max_scale = scale[g];
            if (mn[g] > max_min) max_min = mn[g];
        }
        es_block_q2_K *blk = &y[b];
        memset(blk, 0, sizeof *blk);
        float d = max_scale / 15, dmin = max_min / 15;
        blk->d = es_fp32_to_fp16(d);
        blk->dmin = es_fp32_to_fp16(dmin);
        d = es_fp16_to_fp32(blk->d);       /* use the rounded values */
        dmin = es_fp16_to_fp32(blk->dmin);
        for (int g = 0; g < 16; g++) {
            int ls = d > 0 ? (int)lrintf(scale[g] / d) : 0;
            int lm = dmin > 0 ? (int)lrintf(mn[g] / dmin) : 0;
            ls = ls < 0 ? 0 : ls > 15 ? 15 : ls;
            lm = lm < 0 ? 0 : lm > 15 ? 15 : lm;
            blk->scales[g] = (uint8_t)(ls | (lm << 4));
            float dl = d * ls, ml = dmin * lm;
            for (int l = 0; l < 16; l++) {
                int q = dl > 0 ? (int)lrintf((x[16 * g + l] + ml) / dl) : 0;
                q = q < 0 ? 0 : q > 3 ? 3 : q;
                int byte, shift;
                q2_pos(g, l, &byte, &shift);
                blk->qs[byte] |= (uint8_t)(q << shift);
            }
        }
    }
}

void es_dequantize_q2_K(const es_block_q2_K *x, float *y, int n) {
    for (int b = 0; b < n / ES_QK_K; b++) {
        float d = es_fp16_to_fp32(x[b].d), dmin = es_fp16_to_fp32(x[b].dmin);
        for (int g = 0; g < 16; g++) {
            float dl = d * (x[b].scales[g] & 15), ml = dmin * (x[b].scales[g] >> 4);
            for (int l = 0; l < 16; l++) {
                int byte, shift;
                q2_pos(g, l, &byte, &shift);
                *y++ = dl * ((x[b].qs[byte] >> shift) & 3) - ml;
            }
        }
    }
}

void es_quantize_q8_K(const float *x, es_block_q8_K *y, int n) {
    for (int b = 0; b < n / ES_QK_K; b++, x += ES_QK_K) {
        float amax = 0;
        for (int i = 0; i < ES_QK_K; i++) {
            float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        if (amax == 0) {
            memset(&y[b], 0, sizeof y[b]);
            continue;
        }
        float id = 127.0f / amax;
        y[b].d = amax / 127.0f;
        for (int i = 0; i < ES_QK_K; i++) {
            int q = (int)lrintf(x[i] * id);
            y[b].qs[i] = (int8_t)(q > 127 ? 127 : q < -127 ? -127 : q);
        }
        for (int g = 0; g < 16; g++) {
            int s = 0;
            for (int l = 0; l < 16; l++) s += y[b].qs[16 * g + l];
            y[b].bsums[g] = (int16_t)s;
        }
    }
}

float es_dot_q2_K_q8_K_ref(int n, const es_block_q2_K *x, const es_block_q8_K *y) {
    float sumf = 0;
    for (int b = 0; b < n / ES_QK_K; b++) {
        const uint8_t *sc = x[b].scales;
        int summs = 0;
        for (int g = 0; g < 16; g++) summs += y[b].bsums[g] * (sc[g] >> 4);
        int isum = 0;
        for (int g = 0; g < 16; g++) {
            int s = 0;
            for (int l = 0; l < 16; l++) {
                int byte, shift;
                q2_pos(g, l, &byte, &shift);
                s += y[b].qs[16 * g + l] * ((x[b].qs[byte] >> shift) & 3);
            }
            isum += (sc[g] & 15) * s;
        }
        float dall = y[b].d * es_fp16_to_fp32(x[b].d);
        float dmin = y[b].d * es_fp16_to_fp32(x[b].dmin);
        sumf += dall * isum - dmin * summs;
    }
    return sumf;
}

typedef float (*dot_fn)(int, const es_block_q2_K *, const es_block_q8_K *);
static dot_fn g_dot;
static const char *g_name;
static int g_force_scalar;

static void pick_kernel(void) {
    g_dot = es_dot_q2_K_q8_K_ref;
    g_name = "scalar";
#if defined(__aarch64__)
    if (!g_force_scalar && es_has_dotprod()) {
        g_dot = es_dot_q2_K_q8_K_neon;
        g_name = "neon-dotprod";
    }
#endif
}

void es_q2k_force_scalar(int on) {
    g_force_scalar = on;
    pick_kernel();
}

const char *es_q2k_kernel_name(void) {
    if (!g_dot) pick_kernel();
    return g_name;
}

float es_dot_q2_K_q8_K(int n, const es_block_q2_K *x, const es_block_q8_K *y) {
    if (!g_dot) pick_kernel();
    return g_dot(n, x, y);
}

void es_matvec_q2_K(const es_block_q2_K *W, int cols, int r0, int r1,
                    const es_block_q8_K *xq, float *y) {
    if (!g_dot) pick_kernel();
    dot_fn dot = g_dot;
    const int nb = cols / ES_QK_K;
    for (int r = r0; r < r1; r++) y[r] = dot(cols, W + (size_t)r * nb, xq);
}
