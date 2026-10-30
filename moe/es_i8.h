/* es_i8.h: int8 GEMM for es_train moe --sparse (moe/es_sparse.c): products through the frozen experts.
 *   C (n x m, int32) = A (n x k, int8 rows) . B, with B stored in "VNNI layout": [k/4][m][4] int8, i.e. for each group of
 *   4 reduction indices, the 4 bytes of output column j are adjacent. Every path returns the exact integer sum
 *   (A in [-127, 127], B in [-8, 8]; the AVX2 path needs |B| <= 64). k multiple of 4, m multiple of 16.
 * Paths: AMX (tdpbssd, when k % 64 == 0 and m % 32 == 0; other shapes use the next path), AVX-VNNI (vpdpbusd),
 * AVX2 (pmaddubsw), NEON SDOT (sdot by lane), scalar; picked at runtime.
 * AVX-VNNI and AVX2 multiply (a + 128) by b and subtract 128 * colsum(b), so when i8_pick sets *colsum, callers pass
 * colsum[j] = sum over k of B[.][j] (the AMX path keeps it for its fallback shapes). */
#ifndef ES_I8_H
#define ES_I8_H
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
static i8_gemm_fn i8_fallback;
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11
#define I8_AMX 1
#include <sys/syscall.h>
#include <unistd.h>
typedef struct __attribute__((packed)) { uint8_t palette, start; uint8_t res[14]; uint16_t colsb[16]; uint8_t rows[16]; } i8_tcfg;
/* 32 tokens x 32 outputs per block: C tiles 0-3, A tiles 4-5 (16 tokens x 64 k), B tiles 6-7 (16 k-quads x 16 outputs x 4,
 * which is the VNNI layout read with a row stride of 4m bytes). A tail of fewer than 32 tokens goes through zeroed scratch. */
__attribute__((target("amx-tile,amx-int8"))) static void i8_gemm_amx(int n, int m, int k, const int8_t *A, const int8_t *B, const int32_t *colsum, int32_t *C) {
    if (k % 64 || m % 32) { i8_fallback(n, m, k, A, B, colsum, C); return; }
    static __thread int8_t *pa; static __thread int32_t *pc; static __thread size_t pan, pcn;
    i8_tcfg cfg; memset(&cfg, 0, sizeof cfg); cfg.palette = 1; for (int i = 0; i < 8; i++) { cfg.rows[i] = 16; cfg.colsb[i] = 64; }
    _tile_loadconfig(&cfg);
    const size_t sb = (size_t)m * 4;
    for (int t = 0; t < n; t += 32) {
        const int nr = n - t < 32 ? n - t : 32; const int8_t *a = A + (size_t)t * k; int32_t *c = C + (size_t)t * m;
        if (nr < 32) {
            if (pan < (size_t)32 * k) { free(pa); pan = (size_t)32 * k; pa = malloc(pan); }
            if (pcn < (size_t)32 * m) { free(pc); pcn = (size_t)32 * m; pc = malloc(4 * pcn); }
            memcpy(pa, a, (size_t)nr * k); memset(pa + (size_t)nr * k, 0, (size_t)(32 - nr) * k); a = pa; c = pc;
        }
        for (int j = 0; j < m; j += 32) {
            _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
            for (int q = 0; q < k; q += 64) { const int8_t *b = B + ((size_t)(q / 4) * m + j) * 4;
                _tile_loadd(4, a + q, k); _tile_loadd(5, a + (size_t)16 * k + q, k); _tile_loadd(6, b, sb); _tile_loadd(7, b + 64, sb);
                _tile_dpbssd(0, 4, 6); _tile_dpbssd(1, 4, 7); _tile_dpbssd(2, 5, 6); _tile_dpbssd(3, 5, 7); }
            _tile_stored(0, c + j, sb); _tile_stored(1, c + j + 16, sb); _tile_stored(2, c + (size_t)16 * m + j, sb); _tile_stored(3, c + (size_t)16 * m + j + 16, sb);
        }
        if (nr < 32) memcpy(C + (size_t)t * m, pc, 4 * (size_t)nr * m);
    }
    _tile_release();
}
#endif
static i8_gemm_fn i8_pick(const char **name) {
    if (getenv("ES_TRAIN_SCALAR")) { *name = "scalar"; return i8_gemm_scalar; }
    i8_fallback = i8_gemm_scalar; *name = "scalar";
    if (__builtin_cpu_supports("avx2")) { *name = "avx2"; i8_fallback = i8_gemm_avx2; }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("avxvnni") && !getenv("ES_TRAIN_NO_VNNI")) { *name = "avx-vnni"; i8_fallback = i8_gemm_vnni; }
#ifdef I8_AMX
    if (__builtin_cpu_supports("amx-int8") && __builtin_cpu_supports("amx-tile") && !getenv("ES_TRAIN_NO_AMX") && syscall(SYS_arch_prctl, 0x1023, 18) == 0) { *name = "amx"; return i8_gemm_amx; }
#endif
    return i8_fallback;
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
static void i8_quant_row_c(const float *r, int k, int8_t *o, float *scale, int32_t *rsum) {
    float m = 0; for (int i = 0; i < k; i++) { const float a = r[i] < 0 ? -r[i] : r[i]; m = a > m ? a : m; }
    const float s = m > 0 ? m / 127 : 1, inv = 1 / s; int32_t sum = 0;
    for (int i = 0; i < k; i++) { float v = r[i] * inv; v += 12582912.0f; v -= 12582912.0f;   /* = rintf for |v| <= 127 */
        const int iv = (int)v; o[i] = (int8_t)iv; sum += iv; }
    *scale = s; *rsum = sum;
}
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("avx2"))) static void i8_quant_row_avx2(const float *r, int k, int8_t *o, float *scale, int32_t *rsum) {
    if (k % 32) { i8_quant_row_c(r, k, o, scale, rsum); return; }
    const __m256 ab = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff)); __m256 mx = _mm256_setzero_ps();
    for (int i = 0; i < k; i += 8) mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_loadu_ps(r + i), ab));
    __m128 h = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1)); h = _mm_max_ps(h, _mm_movehl_ps(h, h)); h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
    const float m = _mm_cvtss_f32(h), s = m > 0 ? m / 127 : 1; const __m256 inv = _mm256_set1_ps(1 / s); __m256i sum = _mm256_setzero_si256();
    for (int i = 0; i < k; i += 32) {   /* cvtps rounds to nearest even (default MXCSR), like lrintf */
        const __m256i a = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(r + i), inv)), b = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(r + i + 8), inv));
        const __m256i c = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(r + i + 16), inv)), d = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(r + i + 24), inv));
        sum = _mm256_add_epi32(sum, _mm256_add_epi32(_mm256_add_epi32(a, b), _mm256_add_epi32(c, d)));
        const __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(a, b), _mm256_packs_epi32(c, d));   /* lanes interleaved: fix the order */
        _mm256_storeu_si256((__m256i *)(o + i), _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)));
    }
    __m128i t = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1)); t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4e)); t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xb1));
    *scale = s; *rsum = _mm_cvtsi128_si32(t);
}
#elif defined(__aarch64__)
static void i8_quant_row_neon(const float *r, int k, int8_t *o, float *scale, int32_t *rsum) {
    if (k % 16) { i8_quant_row_c(r, k, o, scale, rsum); return; }
    float32x4_t mx = vdupq_n_f32(0); for (int i = 0; i < k; i += 4) mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(r + i)));
    const float m = vmaxvq_f32(mx), s = m > 0 ? m / 127 : 1, inv = 1 / s; int32x4_t sum = vdupq_n_s32(0);
    for (int i = 0; i < k; i += 16) {   /* vcvtn: round to nearest even */
        const int32x4_t a = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(r + i), inv)), b = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(r + i + 4), inv));
        const int32x4_t c = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(r + i + 8), inv)), d = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(r + i + 12), inv));
        sum = vaddq_s32(sum, vaddq_s32(vaddq_s32(a, b), vaddq_s32(c, d)));
        vst1q_s8(o + i, vcombine_s8(vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b))), vqmovn_s16(vcombine_s16(vqmovn_s32(c), vqmovn_s32(d)))));
    }
    *scale = s; *rsum = vaddvq_s32(sum);
}
#endif
__attribute__((unused)) static void i8_quant_rows(const float *x, int n, int k, int8_t *q, float *scale, int32_t *rsum) {
#if defined(__x86_64__) && defined(__GNUC__)
    static int avx2 = -1; if (avx2 < 0) avx2 = __builtin_cpu_supports("avx2");
    if (avx2) { for (int t = 0; t < n; t++) i8_quant_row_avx2(x + (size_t)t * k, k, q + (size_t)t * k, scale + t, rsum + t); return; }
#elif defined(__aarch64__)
    for (int t = 0; t < n; t++) i8_quant_row_neon(x + (size_t)t * k, k, q + (size_t)t * k, scale + t, rsum + t); return;
#endif
    for (int t = 0; t < n; t++) i8_quant_row_c(x + (size_t)t * k, k, q + (size_t)t * k, scale + t, rsum + t);
}
#endif
