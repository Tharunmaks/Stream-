/* esd_kern.h: int8-activation x packed-weight row dot products for libesd (moe/esd.c), one copy per CPU family.
 * Rows are output-major packed blocks (see esd.c). Every kernel returns biased sums so all paths are integer-exact
 * and give identical results:
 *   int4:    out[j] = sum_i x[i] * nibble[j][i]            (weight = nibble - 7.5; caller subtracts 7.5 * sum x)
 *   ternary: out[j] = sum_i x[i] * (value[j][i] + 1)       (value in -1/0/1; caller subtracts sum x)
 *   int8:    out[j] = sum_i x[i] * w[j][i]                 (routers and head, rows of k int8, k a multiple of 32)
 *   tq1:     ternary at 1.625 bits, same result as ternary (k a multiple of 256). Each 256-weight block is 52 bytes:
 *            48 bytes of 5 base-3 digits (weight 48 d + j is digit d of byte j) and 4 bytes of 4 digits (weight 240 + 4 d + j).
 *            Digit d of byte q is ((uint8_t)(q * 3^d) * 3) >> 8 (q = ceil(256 * v / 3^n) for the n-digit value v).
 * x: int8 in [-127, 127], k a multiple of 64. */
#ifndef ESD_KERN_H
#define ESD_KERN_H
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef void (*esd_kern)(const uint8_t *w, size_t rb, const int8_t *x, int k, int rows, int32_t *out);

static void k4_scalar(const uint8_t *w, size_t rb, const int8_t *x, int k, int rows, int32_t *out) {
    for (int j = 0; j < rows; j++) { const uint8_t *r = w + j * rb; int32_t s = 0;
        for (int b = 0; b < k / 32; b++) for (int q = 0; q < 16; q++) s += (int32_t)x[32 * b + q] * (r[16 * b + q] & 15) + (int32_t)x[32 * b + 16 + q] * (r[16 * b + q] >> 4);
        out[j] = s; }
}
static void k8_scalar(const uint8_t *wu, size_t rb, const int8_t *x, int k, int rows, int32_t *out) {
    for (int j = 0; j < rows; j++) { const int8_t *r = (const int8_t *)wu + j * rb; int32_t s = 0; for (int i = 0; i < k; i++) s += (int32_t)x[i] * r[i]; out[j] = s; }
}
static const uint8_t P3[5] = {1, 3, 9, 27, 81};
static inline int tq_digit(uint8_t q, int d) { return ((uint8_t)(q * P3[d]) * 3) >> 8; }
static void k1_scalar(const uint8_t *w, size_t rb, const int8_t *x, int k, int rows, int32_t *out) {
    for (int j = 0; j < rows; j++) { const uint8_t *r = w + j * rb; int32_t s = 0;
        for (int b = 0; b < k / 256; b++) { const uint8_t *q = r + 52 * b; const int8_t *xb = x + 256 * b;
            for (int i = 0; i < 48; i++) for (int d = 0; d < 5; d++) s += xb[48 * d + i] * tq_digit(q[i], d);
            for (int i = 0; i < 4; i++) for (int d = 0; d < 4; d++) s += xb[240 + 4 * d + i] * tq_digit(q[48 + i], d); }
        out[j] = s; }
}
static const int8_t T1[4] = {1, 2, 0, 1};   /* ternary code -> value + 1 (code 1 = +1, 2 = -1, 0 and 3 = 0) */
static void k2_scalar(const uint8_t *w, size_t rb, const int8_t *x, int k, int rows, int32_t *out) {
    for (int j = 0; j < rows; j++) { const uint8_t *r = w + j * rb; int32_t s = 0;
        for (int b = 0; b < k / 64; b++) for (int q = 0; q < 16; q++) { const uint8_t c = r[16 * b + q]; const int8_t *xb = x + 64 * b + q;
            s += xb[0] * T1[c & 3] + xb[16] * T1[(c >> 2) & 3] + xb[32] * T1[(c >> 4) & 3] + xb[48] * T1[c >> 6]; }
        out[j] = s; }
}

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define AVX2 __attribute__((target("avx2,fma")))
AVX2 static inline int32_t hsum8(__m256i v) { __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)); s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E)); s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1)); return _mm_cvtsi128_si32(s); }
AVX2 static inline __m256i nib32(const uint8_t *p) { const __m128i q = _mm_loadu_si128((const __m128i *)p); return _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 4), q), _mm256_set1_epi8(15)); }
AVX2 static void k4_avx2(const uint8_t *w, size_t rb, const int8_t *x, int k, int rows, int32_t *out) {
    const __m256i one = _mm256_set1_epi16(1); int j = 0;
    for (; j + 4 <= rows; j += 4) {
        const uint8_t *r0 = w + j * rb, *r1 = r0 + rb, *r2 = r1 + rb, *r3 = r2 + rb; __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
        for (int b = 0; b < k / 32; b++) { const __m256i xv = _mm256_loadu_si256((const __m256i *)(x + 32 * b));
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(nib32(r0 + 16 * b), xv), one));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(nib32(r1 + 16 * b), xv), one));
            a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(nib32(r2 + 16 * b), xv), one));
            a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(nib32(r3 + 16 * b), xv), one)); }
        out[j] = hsum8(a0); out[j + 1] = hsum8(a1); out[j + 2] = hsum8(a2); out[j + 3] = hsum8(a3);
    }
    if (j < rows) k4_scalar(w + j * rb, rb, x, k, rows - j, out + j);
}
AVX2 static inline void tri64(const uint8_t *p, __m256i *lo, __m256i *hi) {   /* 16 bytes -> value+1 for weights 0..31 and 32..63 */
    const __m128i q = _mm_loadu_si128((const __m128i *)p); const __m256i m = _mm256_set1_epi8(3);
    const __m256i lut = _mm256_setr_epi8(1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1);
    *lo = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 2), q), m));
    *hi = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(q, 6), _mm_srli_epi16(q, 4)), m));
}
AVX2 static void k2_avx2(const uint8_t *w, size_t rb, const int8_t *x, int k, int rows, int32_t *out) {
    const __m256i one = _mm256_set1_epi16(1); int j = 0;
    for (; j + 4 <= rows; j += 4) {
        const uint8_t *r = w + j * rb; __m256i a[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
        for (int b0 = 0; b0 < k / 64; b0 += 16) {   /* int16 partial sums: |lane| <= 16 blocks * 2 * 2 * 2 * 127 < 32768 */
            __m256i s[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
            const int b1 = b0 + 16 < k / 64 ? b0 + 16 : k / 64;
            for (int b = b0; b < b1; b++) { const __m256i x0 = _mm256_loadu_si256((const __m256i *)(x + 64 * b)), x1 = _mm256_loadu_si256((const __m256i *)(x + 64 * b + 32));
                for (int t = 0; t < 4; t++) { __m256i lo, hi; tri64(r + t * rb + 16 * b, &lo, &hi);
                    s[t] = _mm256_add_epi16(s[t], _mm256_add_epi16(_mm256_maddubs_epi16(lo, x0), _mm256_maddubs_epi16(hi, x1))); } }
            for (int t = 0; t < 4; t++) a[t] = _mm256_add_epi32(a[t], _mm256_madd_epi16(s[t], one)); }
        for (int t = 0; t < 4; t++) out[j + t] = hsum8(a[t]);
    }
    if (j < rows) k2_scalar(w + j * rb, rb, x, k, rows - j, out + j);
}
AVX2 static void k8_avx2(const uint8_t *wu, size_t rb, const int8_t *x, int k, int rows, int32_t *out) {
    const __m256i one = _mm256_set1_epi16(1); int j = 0;
    for (; j + 4 <= rows; j += 4) {
        __m256i a[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
        for (int i = 0; i < k; i += 32) { const __m256i xv = _mm256_loadu_si256((const __m256i *)(x + i));
            for (int t = 0; t < 4; t++) { const __m256i w = _mm256_loadu_si256((const __m256i *)(wu + (j + t) * rb + i));
                a[t] = _mm256_add_epi32(a[t], _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(xv, w)), one)); } }
        for (int t = 0; t < 4; t++) out[j + t] = hsum8(a[t]);
    }
    if (j < rows) k8_scalar(wu + j * rb, rb, x, k, rows - j, out + j);
}
AVX2 static inline __m128i tq16(__m128i q, __m256i mul) {   /* 16 bytes -> 16 digits for per-lane multipliers 3^d */
    const __m256i v = _mm256_and_si256(_mm256_mullo_epi16(_mm256_cvtepu8_epi16(q), mul), _mm256_set1_epi16(255));
    const __m256i dg = _mm256_srli_epi16(_mm256_mullo_epi16(v, _mm256_set1_epi16(3)), 8);
    return _mm_packus_epi16(_mm256_castsi256_si128(dg), _mm256_extracti128_si256(dg, 1));
}
AVX2 static void k1_avx2(const uint8_t *w, size_t rb, const int8_t *x, int k, int rows, int32_t *out) {
    const __m128i one = _mm_set1_epi16(1); const __m256i tail = _mm256_setr_epi16(1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27);
    for (int j = 0; j < rows; j++) { const uint8_t *r = w + j * rb; __m128i a = _mm_setzero_si128();
        for (int b = 0; b < k / 256; b++) { const uint8_t *q = r + 52 * b; const int8_t *xb = x + 256 * b; __m128i s = _mm_setzero_si128();
            const __m128i q0 = _mm_loadu_si128((const __m128i *)q), q1 = _mm_loadu_si128((const __m128i *)(q + 16)), q2 = _mm_loadu_si128((const __m128i *)(q + 32));
            for (int d = 0; d < 5; d++) { const __m256i m = _mm256_set1_epi16(P3[d]);
                s = _mm_add_epi16(s, _mm_maddubs_epi16(tq16(q0, m), _mm_loadu_si128((const __m128i *)(xb + 48 * d))));
                s = _mm_add_epi16(s, _mm_maddubs_epi16(tq16(q1, m), _mm_loadu_si128((const __m128i *)(xb + 48 * d + 16))));
                s = _mm_add_epi16(s, _mm_maddubs_epi16(tq16(q2, m), _mm_loadu_si128((const __m128i *)(xb + 48 * d + 32)))); }
            int32_t t4; memcpy(&t4, q + 48, 4);
            s = _mm_add_epi16(s, _mm_maddubs_epi16(tq16(_mm_set1_epi32(t4), tail), _mm_loadu_si128((const __m128i *)(xb + 240))));
            a = _mm_add_epi32(a, _mm_madd_epi16(s, one)); }   /* int16 lanes: at most 16 * 2 * 2 * 127 */
        a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4E)); a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0xB1)); out[j] = _mm_cvtsi128_si32(a); }
}
static void esd_pick(esd_kern *k4, esd_kern *k2, esd_kern *k8, esd_kern *k1, const char **name) {
    if (__builtin_cpu_supports("avx2") && !getenv("ES_RUN_SCALAR")) { *k4 = k4_avx2; *k2 = k2_avx2; *k8 = k8_avx2; *k1 = k1_avx2; *name = "avx2"; }
    else { *k4 = k4_scalar; *k2 = k2_scalar; *k8 = k8_scalar; *k1 = k1_scalar; *name = "scalar"; }
}

#elif defined(__aarch64__)
#include <arm_neon.h>
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#ifdef __clang__
#define DOTP __attribute__((target("dotprod")))
#else
#define DOTP __attribute__((target("+dotprod")))
#endif
static inline int32x4_t mac16(int32x4_t a, int8x16_t w, int8x16_t x) {   /* NEON without SDOT */
    return vpadalq_s16(vpadalq_s16(a, vmull_s8(vget_low_s8(w), vget_low_s8(x))), vmull_high_s8(w, x));
}
#define K4_BODY(MAC) \
    const uint8x16_t m = vdupq_n_u8(15); int j = 0; \
    for (; j + 4 <= rows; j += 4) { const uint8_t *r = w + j * rb; int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0; \
        for (int b = 0; b < k / 32; b++) { const int8x16_t x0 = vld1q_s8(x + 32 * b), x1 = vld1q_s8(x + 32 * b + 16); uint8x16_t q; \
            q = vld1q_u8(r + 16 * b);          a0 = MAC(MAC(a0, vreinterpretq_s8_u8(vandq_u8(q, m)), x0), vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), x1); \
            q = vld1q_u8(r + rb + 16 * b);     a1 = MAC(MAC(a1, vreinterpretq_s8_u8(vandq_u8(q, m)), x0), vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), x1); \
            q = vld1q_u8(r + 2 * rb + 16 * b); a2 = MAC(MAC(a2, vreinterpretq_s8_u8(vandq_u8(q, m)), x0), vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), x1); \
            q = vld1q_u8(r + 3 * rb + 16 * b); a3 = MAC(MAC(a3, vreinterpretq_s8_u8(vandq_u8(q, m)), x0), vreinterpretq_s8_u8(vshrq_n_u8(q, 4)), x1); } \
        out[j] = vaddvq_s32(a0); out[j + 1] = vaddvq_s32(a1); out[j + 2] = vaddvq_s32(a2); out[j + 3] = vaddvq_s32(a3); } \
    if (j < rows) k4_scalar(w + j * rb, rb, x, k, rows - j, out + j);
#define K2_BODY(MAC) \
    const uint8x16_t m = vdupq_n_u8(3); const int8x16_t lut = {1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1}; int j = 0; \
    for (; j + 2 <= rows; j += 2) { const uint8_t *r = w + j * rb; int32x4_t a0 = vdupq_n_s32(0), a1 = a0; \
        for (int b = 0; b < k / 64; b++) { const int8_t *xb = x + 64 * b; const int8x16_t x0 = vld1q_s8(xb), x1 = vld1q_s8(xb + 16), x2 = vld1q_s8(xb + 32), x3 = vld1q_s8(xb + 48); \
            for (int t = 0; t < 2; t++) { const uint8x16_t q = vld1q_u8(r + t * rb + 16 * b); int32x4_t a = t ? a1 : a0; \
                a = MAC(a, vqtbl1q_s8(lut, vandq_u8(q, m)), x0); a = MAC(a, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(q, 2), m)), x1); \
                a = MAC(a, vqtbl1q_s8(lut, vandq_u8(vshrq_n_u8(q, 4), m)), x2); a = MAC(a, vqtbl1q_s8(lut, vshrq_n_u8(q, 6)), x3); \
                if (t) a1 = a; else a0 = a; } } \
        out[j] = vaddvq_s32(a0); out[j + 1] = vaddvq_s32(a1); } \
    if (j < rows) k2_scalar(w + j * rb, rb, x, k, rows - j, out + j);
static void k4_neon(const uint8_t *w, size_t rb, const int8_t *x, int k, int rows, int32_t *out) { K4_BODY(mac16) }
static void k2_neon(const uint8_t *w, size_t rb, const int8_t *x, int k, int rows, int32_t *out) { K2_BODY(mac16) }
DOTP static void k4_sdot(const uint8_t *w, size_t rb, const int8_t *x, int k, int rows, int32_t *out) { K4_BODY(vdotq_s32) }
DOTP static void k2_sdot(const uint8_t *w, size_t rb, const int8_t *x, int k, int rows, int32_t *out) { K2_BODY(vdotq_s32) }
#define K8_BODY(MAC) \
    int j = 0; for (; j + 2 <= rows; j += 2) { const int8_t *r0 = (const int8_t *)wu + j * rb, *r1 = r0 + rb; int32x4_t a0 = vdupq_n_s32(0), a1 = a0; \
        for (int i = 0; i < k; i += 16) { const int8x16_t xv = vld1q_s8(x + i); a0 = MAC(a0, vld1q_s8(r0 + i), xv); a1 = MAC(a1, vld1q_s8(r1 + i), xv); } \
        out[j] = vaddvq_s32(a0); out[j + 1] = vaddvq_s32(a1); } \
    if (j < rows) k8_scalar(wu + j * rb, rb, x, k, rows - j, out + j);
#define K1_BODY(MAC) \
    const uint8x8_t three = vdup_n_u8(3); const uint8x16_t tm = {1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27}; \
    for (int j = 0; j < rows; j++) { const uint8_t *r = w + j * rb; int32x4_t a = vdupq_n_s32(0); \
        for (int b = 0; b < k / 256; b++) { const uint8_t *q = r + 52 * b; const int8_t *xb = x + 256 * b; \
            const uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16), q2 = vld1q_u8(q + 32); \
            for (int d = 0; d < 5; d++) { const uint8x16_t p = vdupq_n_u8(P3[d]); \
                a = MAC(a, TQD(vmulq_u8(q0, p)), vld1q_s8(xb + 48 * d)); a = MAC(a, TQD(vmulq_u8(q1, p)), vld1q_s8(xb + 48 * d + 16)); \
                a = MAC(a, TQD(vmulq_u8(q2, p)), vld1q_s8(xb + 48 * d + 32)); } \
            uint32_t t4; memcpy(&t4, q + 48, 4); a = MAC(a, TQD(vmulq_u8(vreinterpretq_u8_u32(vdupq_n_u32(t4)), tm)), vld1q_s8(xb + 240)); } \
        out[j] = vaddvq_s32(a); }
#define TQD(m) vreinterpretq_s8_u8(vcombine_u8(vshrn_n_u16(vmull_u8(vget_low_u8(m), three), 8), vshrn_n_u16(vmull_high_u8(m, vdupq_n_u8(3)), 8)))
static void k1_neon(const uint8_t *w, size_t rb, const int8_t *x, int k, int rows, int32_t *out) { K1_BODY(mac16) }
DOTP static void k1_sdot(const uint8_t *w, size_t rb, const int8_t *x, int k, int rows, int32_t *out) { K1_BODY(vdotq_s32) }
static void k8_neon(const uint8_t *wu, size_t rb, const int8_t *x, int k, int rows, int32_t *out) { K8_BODY(mac16) }
DOTP static void k8_sdot(const uint8_t *wu, size_t rb, const int8_t *x, int k, int rows, int32_t *out) { K8_BODY(vdotq_s32) }
static void esd_pick(esd_kern *k4, esd_kern *k2, esd_kern *k8, esd_kern *k1, const char **name) {
    if (getenv("ES_RUN_SCALAR")) { *k4 = k4_scalar; *k2 = k2_scalar; *k8 = k8_scalar; *k1 = k1_scalar; *name = "scalar"; }
    else if ((getauxval(AT_HWCAP) & HWCAP_ASIMDDP) && !getenv("ES_RUN_NO_SDOT")) { *k4 = k4_sdot; *k2 = k2_sdot; *k8 = k8_sdot; *k1 = k1_sdot; *name = "neon-sdot"; }
    else { *k4 = k4_neon; *k2 = k2_neon; *k8 = k8_neon; *k1 = k1_neon; *name = "neon"; }
}

#else
static void esd_pick(esd_kern *k4, esd_kern *k2, esd_kern *k8, esd_kern *k1, const char **name) { *k4 = k4_scalar; *k2 = k2_scalar; *k8 = k8_scalar; *k1 = k1_scalar; *name = "scalar"; }
#endif
#endif
