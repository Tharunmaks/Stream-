#define _GNU_SOURCE
#include "es_amx.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11
#include <immintrin.h>
#define HAVE_AMX_BUILD 1
#endif

static inline uint16_t bf16(float f) {   /* round to nearest even */
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)(u >> 16);
    u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16);
}

int es_amx_available(void) {
#ifdef HAVE_AMX_BUILD
    static int s = -1;
    if (s < 0) {
        __builtin_cpu_init();
        s = __builtin_cpu_supports("amx-bf16") && __builtin_cpu_supports("amx-tile") && syscall(SYS_arch_prctl, 0x1023, 18) == 0;
    }
    return s;
#else
    return 0;
#endif
}

/* Tile (kb, nb) holds B[kb*32 .. +32][nb*16 .. +16] as 16 rows of 16 pairs: row r = k-pair, element 2n+j = B[2r+j][n]. */
uint16_t *es_amx_pack(const float *src, int K, int N, int bt) {
    const int KB = K / 32, NB = N / 16;
    uint16_t *p = aligned_alloc(64, (size_t)K * N * 2);
    if (!p) return NULL;
    for (int kb = 0; kb < KB; kb++) for (int nb = 0; nb < NB; nb++) {
        uint16_t *t = p + ((size_t)kb * NB + nb) * 512;
        for (int r = 0; r < 16; r++) for (int n = 0; n < 16; n++) for (int j = 0; j < 2; j++) {
            const int k = kb * 32 + 2 * r + j, c = nb * 16 + n;
            t[r * 32 + 2 * n + j] = bf16(bt ? src[(size_t)c * K + k] : src[(size_t)k * N + c]);
        }
    }
    return p;
}

/* Pack from W row slices (so the whole matrix is never held in fp32). bt=1: B = W^T, W is N x K; part = 16 rows of W
 * starting at row n0 (n0 % 16 == 0). bt=0: B = W, W is K x N; part = 32 rows of W starting at row k0 (k0 % 32 == 0). */
void es_amx_pack_part(uint16_t *p, const float *part, int K, int N, int bt, int start) {
    const int KB = K / 32, NB = N / 16;
    if (bt) {
        const int nb = start / 16;
        for (int kb = 0; kb < KB; kb++) { uint16_t *t = p + ((size_t)kb * NB + nb) * 512;
            for (int r = 0; r < 16; r++) for (int n = 0; n < 16; n++) for (int j = 0; j < 2; j++) t[r * 32 + 2 * n + j] = bf16(part[(size_t)n * K + kb * 32 + 2 * r + j]); }
    } else {
        const int kb = start / 32;
        for (int nb = 0; nb < NB; nb++) { uint16_t *t = p + ((size_t)kb * NB + nb) * 512;
            for (int r = 0; r < 16; r++) for (int n = 0; n < 16; n++) for (int j = 0; j < 2; j++) t[r * 32 + 2 * n + j] = bf16(part[(size_t)(2 * r + j) * N + nb * 16 + n]); }
    }
}

#ifdef HAVE_AMX_BUILD
typedef struct { uint8_t palette, start; uint8_t res[14]; uint16_t colsb[16]; uint8_t rows[16]; } tcfg;
#define AMXT __attribute__((target("amx-tile,amx-bf16,avx512f")))
static int VERIFY = -1; long es_amx_redo;
#define REDO es_amx_redo
AMXT static void amx_block(const tcfg *cfg, const uint16_t *Ab, int K, const uint16_t *Bp, int KB, int NB, int nb, float *Cb) {
    _tile_loadconfig(cfg);
    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
    for (int kb = 0; kb < KB; kb++) {
        _tile_loadd(4, Ab + kb * 32, K * 2);
        _tile_loadd(5, Ab + (size_t)16 * K + kb * 32, K * 2);
        _tile_loadd(6, Bp + ((size_t)kb * NB + nb) * 512, 64);
        _tile_loadd(7, Bp + ((size_t)kb * NB + nb + 1) * 512, 64);
        _tile_dpbf16ps(0, 4, 6); _tile_dpbf16ps(1, 4, 7); _tile_dpbf16ps(2, 5, 6); _tile_dpbf16ps(3, 5, 7);
    }
    _tile_stored(0, Cb, 128); _tile_stored(1, Cb + 16, 128); _tile_stored(2, Cb + 512, 128); _tile_stored(3, Cb + 528, 128);
}
AMXT void es_amx_gemm(const float *A, int lda, const uint16_t *Bp, int K, int N, float *C, int ldc, int m0, int m1, int beta) {
    tcfg cfg; memset(&cfg, 0, sizeof cfg); cfg.palette = 1;
    for (int i = 0; i < 8; i++) { cfg.colsb[i] = 64; cfg.rows[i] = 16; }
    _tile_loadconfig(&cfg);
    const int KB = K / 32, NB = N / 16;
    uint16_t *Ab = aligned_alloc(64, (size_t)32 * K * 2);   /* 32 rows of A in bf16 */
    float Cb[32 * 32] __attribute__((aligned(64))), Cb2[32 * 32] __attribute__((aligned(64)));
    if (VERIFY < 0) { const char *e = getenv("ES_AMX_VERIFY"); VERIFY = !(e && !atoi(e)); }
    for (int mm = m0; mm < m1; mm += 32) {
        const int mr = m1 - mm < 32 ? m1 - mm : 32;
        for (int r = 0; r < 32; r++) {
            uint16_t *d = Ab + (size_t)r * K;
            if (r < mr) { const float *s = A + (size_t)(mm + r) * lda; for (int k = 0; k < K; k++) d[k] = bf16(s[k]); }
            else memset(d, 0, (size_t)K * 2);
        }
        for (int nb = 0; nb < NB; nb += 2) {
            /* On this VM a 32x32 block is occasionally corrupted (tile state lost when the thread is preempted, about 1 in
             * 10^4 blocks measured). Each block is computed twice; on a mismatch, again until two results agree. */
            amx_block(&cfg, Ab, K, Bp, KB, NB, nb, Cb);
            if (VERIFY) {
                for (int tries = 0; tries < 8; tries++) {
                    amx_block(&cfg, Ab, K, Bp, KB, NB, nb, Cb2);
                    if (!memcmp(Cb, Cb2, sizeof Cb)) break;
                    memcpy(Cb, Cb2, sizeof Cb); REDO++;
                }
            }
            for (int r = 0; r < mr; r++) {
                float *c = C + (size_t)(mm + r) * ldc + nb * 16; const float *s = Cb + r * 32;
                if (beta) for (int n = 0; n < 32; n++) c[n] += s[n]; else memcpy(c, s, 128);
            }
        }
    }
    free(Ab);
    _tile_release();
}
#else
void es_amx_gemm(const float *A, int lda, const uint16_t *Bp, int K, int N, float *C, int ldc, int m0, int m1, int beta) {
    (void)A; (void)lda; (void)Bp; (void)K; (void)N; (void)C; (void)ldc; (void)m0; (void)m1; (void)beta;
}
#endif
