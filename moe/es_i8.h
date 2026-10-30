/* es_i8.h: int8 GEMM for es_train moe --sparse (moe/es_sparse.c): products through the frozen experts.
 *   C (n x m, int32) = A (n x k, int8 rows) . B, with B stored in "VNNI layout": [k/4][m][4] int8, i.e. for each group of
 *   4 reduction indices, the 4 bytes of output column j are adjacent. Every path returns the exact integer sum
 *   (A in [-127, 127], B in [-8, 8]; the AVX2 path needs |B| <= 64). k multiple of 4, m multiple of 16.
 * Paths: AVX-VNNI (vpdpbusd), AVX2 (pmaddubsw), NEON SDOT (sdot by lane), NEON, scalar; picked at runtime.
 * x86 multiplies (a + 128) by b and subtracts 128 * colsum(b), so callers pass colsum[j] = sum over k of B[.][j]. */
#ifndef ES_I8_H
#define ES_I8_H
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef void (*i8_gemm_fn)(int n, int m, int k, const int8_t *A, const int8_t *B, const int32_t *colsum, int32_t *C);

static void i8_gemm_scalar(int n, int m, int k, const int8_t *A, const int8_t *B, const int32_t *colsum, int32_t *C) {
    (void)colsum;
    for (int t = 0; t < n; t++) { int32_t *c = C + (size_t)t * m; memset(c, 0, 4 * (size_t)m); const int8_t *a = A + (size_t)t * k;
        for (int q = 0; q < k / 4; q++) { const int8_t *b = B + (size_t)q * m * 4; const int a0 = a[4 * q], a1 = a[4 * q + 1], a2 = a[4 * q + 2], a3 = a[4 * q + 3];
            for (int j = 0; j < m; j++) c[j] += a0 * b[4 * j] + a1 * b[4 * j + 1] + a2 * b[4 * j + 2] + a3 * b[4 * j + 3]; } }
}

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
/* 4 tokens x 16 outputs per tile; ACC(a, b, acc) adds dot4(a quad, b quads) into 8 int32 lanes */
#define I8_X86_BODY(ACC) \
    const __m256i off = _mm256_set1_epi8((char)0x80); int t = 0; \
    for (; t < n; t += 4) { const int nt = n - t < 4 ? n - t : 4; int32_t ab[4]; \
        for (int j = 0; j < m; j += 16) { __m256i c[4][2]; for (int r = 0; r < 4; r++) c[r][0] = c[r][1] = _mm256_setzero_si256(); \
            for (int q = 0; q < k / 4; q++) { const int8_t *b = B + ((size_t)q * m + j) * 4; \
                const __m256i b0 = _mm256_loadu_si256((const __m256i *)b), b1 = _mm256_loadu_si256((const __m256i *)(b + 32)); \
                for (int r = 0; r < nt; r++) { memcpy(&ab[r], A + (size_t)(t + r) * k + 4 * q, 4); \
                    const __m256i a = _mm256_xor_si256(_mm256_set1_epi32(ab[r]), off); c[r][0] = ACC(a, b0, c[r][0]); c[r][1] = ACC(a, b1, c[r][1]); } } \
            const __m256i s0 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(colsum + j)), 7), s1 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i *)(colsum + j + 8)), 7); \
            for (int r = 0; r < nt; r++) { _mm256_storeu_si256((__m256i *)(C + (size_t)(t + r) * m + j), _mm256_sub_epi32(c[r][0], s0)); \
                _mm256_storeu_si256((__m256i *)(C + (size_t)(t + r) * m + j + 8), _mm256_sub_epi32(c[r][1], s1)); } } }
__attribute__((target("avx2,avxvnni"))) static inline __m256i i8_acc_vnni(__m256i a, __m256i b, __m256i c) { return _mm256_dpbusd_avx_epi32(c, a, b); }
__attribute__((target("avx2"))) static inline __m256i i8_acc_avx2(__m256i a, __m256i b, __m256i c) { return _mm256_add_epi32(c, _mm256_madd_epi16(_mm256_maddubs_epi16(a, b), _mm256_set1_epi16(1))); }
__attribute__((target("avx2,avxvnni"))) static void i8_gemm_vnni(int n, int m, int k, const int8_t *A, const int8_t *B, const int32_t *colsum, int32_t *C) { I8_X86_BODY(i8_acc_vnni) }
__attribute__((target("avx2"))) static void i8_gemm_avx2(int n, int m, int k, const int8_t *A, const int8_t *B, const int32_t *colsum, int32_t *C) { I8_X86_BODY(i8_acc_avx2) }
static i8_gemm_fn i8_pick(const char **name) {
    if (getenv("ES_TRAIN_SCALAR")) { *name = "scalar"; return i8_gemm_scalar; }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("avxvnni") && !getenv("ES_TRAIN_NO_VNNI")) { *name = "avx-vnni"; return i8_gemm_vnni; }
    if (__builtin_cpu_supports("avx2")) { *name = "avx2"; return i8_gemm_avx2; }
    *name = "scalar"; return i8_gemm_scalar;
}

#elif defined(__aarch64__)
#include <arm_neon.h>
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#ifdef __clang__
#define I8_DOTP __attribute__((target("dotprod")))
#else
#define I8_DOTP __attribute__((target("+dotprod")))
#endif
/* 4 tokens (one 16-byte vector of their quads) x 16 outputs (4 vectors of 4 outputs x 4 bytes) per tile */
I8_DOTP static void i8_gemm_sdot(int n, int m, int k, const int8_t *A, const int8_t *B, const int32_t *colsum, int32_t *C) {
    (void)colsum;
    for (int t = 0; t < n; t += 4) { const int nt = n - t < 4 ? n - t : 4;
        for (int j = 0; j < m; j += 16) { int32x4_t c[4][4]; for (int r = 0; r < 4; r++) for (int v = 0; v < 4; v++) c[r][v] = vdupq_n_s32(0);
            for (int q = 0; q < k / 4; q++) { int32_t ab[4] = {0, 0, 0, 0}; for (int r = 0; r < nt; r++) memcpy(&ab[r], A + (size_t)(t + r) * k + 4 * q, 4);
                const int8x16_t a = vreinterpretq_s8_s32(vld1q_s32(ab)); const int8_t *b = B + ((size_t)q * m + j) * 4;
                const int8x16_t b0 = vld1q_s8(b), b1 = vld1q_s8(b + 16), b2 = vld1q_s8(b + 32), b3 = vld1q_s8(b + 48);
#define I8L(r) c[r][0] = vdotq_laneq_s32(c[r][0], b0, a, r); c[r][1] = vdotq_laneq_s32(c[r][1], b1, a, r); c[r][2] = vdotq_laneq_s32(c[r][2], b2, a, r); c[r][3] = vdotq_laneq_s32(c[r][3], b3, a, r);
                I8L(0) I8L(1) I8L(2) I8L(3)
#undef I8L
            }
            for (int r = 0; r < nt; r++) for (int v = 0; v < 4; v++) vst1q_s32(C + (size_t)(t + r) * m + j + 4 * v, c[r][v]); } }
}
static i8_gemm_fn i8_pick(const char **name) {
    if (!getenv("ES_TRAIN_SCALAR") && (getauxval(AT_HWCAP) & HWCAP_ASIMDDP)) { *name = "neon-sdot"; return i8_gemm_sdot; }
    *name = "scalar"; return i8_gemm_scalar;
}
#else
static i8_gemm_fn i8_pick(const char **name) { *name = "scalar"; return i8_gemm_scalar; }
#endif

/* per-row symmetric int8 (max-abs, round to nearest even: the same as libesd quant()): x (n x k floats) -> q, scale[n],
 * sum of q per row (needed for the int4 0.5 offset) */
__attribute__((unused)) static void i8_quant_rows(const float *x, int n, int k, int8_t *q, float *scale, int32_t *rsum) {
    for (int t = 0; t < n; t++) { const float *r = x + (size_t)t * k; float m = 0; for (int i = 0; i < k; i++) { const float a = r[i] < 0 ? -r[i] : r[i]; if (a > m) m = a; }
        const float s = m > 0 ? m / 127 : 1, inv = 1 / s; int32_t sum = 0; int8_t *o = q + (size_t)t * k;
        for (int i = 0; i < k; i++) { const int iv = (int)lrintf(r[i] * inv); o[i] = (int8_t)iv; sum += iv; }
        scale[t] = s; rsum[t] = sum; }
}
#endif
