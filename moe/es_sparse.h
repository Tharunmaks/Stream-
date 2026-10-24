/* es_sparse.h: the es-sparse-v1 model, shared by the trainer (moe/es_sparse.c, `es_train moe --sparse`) and the
 * Run decoder (moe/esd.c, libesd.so).
 *
 * Model (byte level, vocab 256), per token t, per layer l:
 *   h = rms(x)                       s = h + mix_l * h(t-1)        (token shift: the only mixing between positions)
 *   z = Wr_l s, p = softmax(z), e = argmax z, g = p[e]              (top-1 router)
 *   u = W1_e^T s + ls Bu Au s        y = W2_e^T silu(u) + ls Bd Ad silu(u)        x += g y
 *   logits = head rms(x)
 * W1_e (D x F) and W2_e (F x D) are frozen packed int4, read in place from the experts file ("lab packed int4" layout,
 * the same file es_common.esmoe reads): expert (l, e) is D*F bytes at offset (l*E + e)*D*F, W1 row-major then W2
 * row-major, two weights per byte, low nibble first, weight = (nibble - 7.5) * scale. Trainable: emb, head, mix,
 * Wr and per-expert LoRA (Au R x D, Bu F x R, Ad R x F, Bd D x R).
 *
 * Checkpoint (.ckpt): ess_hdr, then float32 params in ess_layout order: emb[256 D], head[256 D],
 * per layer {mix[D], Wr[E D]}, then per (l, e) LoRA {Au, Bu, Ad, Bd}. */
#ifndef ES_SPARSE_H
#define ES_SPARSE_H
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ESS_MAGIC "ESSPARS1"
typedef struct { char magic[8]; uint32_t version, L, E, D, F, R, K, T; float scale, lora_scale; uint64_t steps; uint32_t reserved[8]; } ess_hdr;

static inline size_t ess_loe(const ess_hdr *h) { return (size_t)h->R * h->D + (size_t)h->F * h->R + (size_t)h->R * h->F + (size_t)h->D * h->R; }
static inline size_t ess_dense_n(const ess_hdr *h) { return 2 * 256 * (size_t)h->D + (size_t)h->L * ((size_t)h->D + (size_t)h->E * h->D); }
static inline size_t ess_params_n(const ess_hdr *h) { return ess_dense_n(h) + (size_t)h->L * h->E * ess_loe(h); }
static inline size_t ess_expert_bytes(const ess_hdr *h) { return (size_t)h->D * h->F; }
static inline size_t ess_file_bytes(const ess_hdr *h) { return (size_t)h->L * h->E * ess_expert_bytes(h); }
/* default nibble scale: W1^T s has unit variance for unit-rms s (var of nibble - 7.5 is 255/12) */
static inline float ess_default_scale(int D, int F) { const int m = D > F ? D : F; return 1.0f / __builtin_sqrtf(21.25f * m); }

/* pointers into a flat parameter array */
typedef struct { float *emb, *head, *lo; float **mix, **Wr; } ess_view;
static inline void ess_bind(const ess_hdr *h, float *p, ess_view *v) {
    size_t o = 0; v->emb = p; o += 256 * (size_t)h->D; v->head = p + o; o += 256 * (size_t)h->D;
    for (uint32_t l = 0; l < h->L; l++) { v->mix[l] = p + o; o += h->D; v->Wr[l] = p + o; o += (size_t)h->E * h->D; }
    v->lo = p + o;
}

/* decode n packed int4 weights (n even) to floats */
static inline void ess_decode(const uint8_t *q, size_t n, float *w, float scale) {
    for (size_t i = 0; i < n / 2; i++) { w[2 * i] = ((float)(q[i] & 15) - 7.5f) * scale; w[2 * i + 1] = ((float)(q[i] >> 4) - 7.5f) * scale; }
}

static inline int ess_check_hdr(const ess_hdr *h) {
    return !memcmp(h->magic, ESS_MAGIC, 8) && h->version == 1 && h->L && h->E && h->D && h->F && h->K == 1 && h->D % 2 == 0 && h->F % 2 == 0;
}
#endif
