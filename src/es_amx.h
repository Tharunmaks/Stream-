/* AMX bf16 GEMM for the trainer (x86 with amx_bf16). C (M x N) = A (M x K, fp32) * B, B pre-packed to bf16 tiles.
 * K % 32 == 0 and N % 32 == 0 are required; M is any size (padded internally). */
#ifndef ES_AMX_H
#define ES_AMX_H
#include <stddef.h>
#include <stdint.h>
int es_amx_available(void);                       /* CPU flag + kernel permission (asks once) */
/* Pack B (K x N) given as fp32: if bt, src is N x K row-major (B = src^T), else src is K x N row-major. Returns bf16 tiles. */
uint16_t *es_amx_pack(const float *src, int K, int N, int bt);
void es_amx_pack_part(uint16_t *p, const float *part, int K, int N, int bt, int start);
/* C rows m0..m1 (exclusive): C[m][n] = beta*C[m][n] + sum_k A[m][k] B[k][n]; beta is 0 or 1. Thread-safe per call. */
void es_amx_gemm(const float *A, int lda, const uint16_t *Bp, int K, int N, float *C, int ldc, int m0, int m1, int beta);
#endif
