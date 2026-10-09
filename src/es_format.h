/*
 * ExpertStream on-disk expert file format, version 1.
 *
 * One file per (layer, expert):  <root>/L<layer:03d>/E<expert:04d>.exp
 *
 *   offset 0      : es_header (exactly 4096 bytes)
 *   offset 4096   : tensor 0 data, padded to a multiple of 4096
 *   ...           : tensor 1, tensor 2, ... each starting 4096-aligned
 *   file size     : multiple of 4096
 *
 * The whole file (header included) is read in ONE pass into a 4096-aligned
 * buffer, then the header is validated in place. That keeps it to one I/O
 * per expert and satisfies O_DIRECT alignment rules (offset, length and
 * buffer address all multiples of 4096).
 *
 * All integers are little-endian (aarch64 and x86-64 are both LE).
 */
#ifndef ES_FORMAT_H
#define ES_FORMAT_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#define ES_MAGIC        0x31505845u /* "EXP1" as bytes in the file */
#define ES_VERSION      1u
#define ES_ALIGN        4096u
#define ES_HEADER_BYTES 4096u
#define ES_MAX_TENSORS  8

enum es_qtype {
    ES_QT_RAW  = 0, /* opaque bytes */
    ES_QT_F16  = 1,
    ES_QT_Q8_0 = 2,
    ES_QT_Q2_K = 3, /* llama.cpp block_q2_K: 256 weights in 84 bytes */
    /* Imported tensors keep their GGUF type: qtype = ES_QT_GGML | ggml_type
     * (e.g. ES_QT_GGML | 10 is Q2_K, | 8 is Q8_0, | 12 is Q4_K). */
    ES_QT_GGML = 0x1000,
};

#define ES_GGML_Q2_K 10
static inline int es_qtype_is_q2k(uint32_t qt) {
    return qt == ES_QT_Q2_K || qt == (ES_QT_GGML | ES_GGML_Q2_K);
}

enum es_role {
    ES_ROLE_GATE  = 0,
    ES_ROLE_UP    = 1,
    ES_ROLE_DOWN  = 2,
    ES_ROLE_OTHER = 15,
};

enum es_flags {
    ES_FLAG_SYNTHETIC   = 1u << 0, /* random bytes, for benchmarking only */
    ES_FLAG_SANE_SCALES = 1u << 1, /* synthetic Q2_K blocks have finite fp16
                                      d/dmin, so the kernels can run on them */
};

typedef struct {
    uint32_t role;
    uint32_t qtype;
    uint32_t rows;   /* output features */
    uint32_t cols;   /* input features */
    uint64_t offset; /* from file start, multiple of ES_ALIGN */
    uint64_t nbytes; /* exact data size (unpadded) */
    uint64_t hash;   /* es_hash64(data, nbytes) */
    uint8_t  reserved[24];
} es_tensor_desc;    /* 64 bytes */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t header_bytes; /* ES_HEADER_BYTES */
    uint32_t n_tensors;
    uint32_t layer;
    uint32_t expert;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t file_bytes;   /* total file size, multiple of ES_ALIGN */
    uint64_t model_id;     /* hash of the model name: catches mixed packs */
    uint8_t  reserved1[64];
    es_tensor_desc t[ES_MAX_TENSORS];
    uint8_t  pad[ES_HEADER_BYTES - 112 - 64 * ES_MAX_TENSORS - 8];
    uint64_t header_hash;  /* es_hash64 of the preceding 4088 bytes */
} es_header;

_Static_assert(sizeof(es_tensor_desc) == 64, "tensor desc must be 64 bytes");
_Static_assert(sizeof(es_header) == ES_HEADER_BYTES, "header must be 4096 bytes");

static inline uint64_t es_round_up(uint64_t x, uint64_t a) {
    return (x + a - 1) / a * a;
}

/* Fast non-cryptographic 64-bit hash: detects truncation, bit rot and
 * wrong-file mistakes. Four independent lanes so it is not latency bound. */
static inline uint64_t es_hash64(const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    const uint64_t K = 0x9E3779B97F4A7C15ull;
    uint64_t h0 = 0x243F6A8885A308D3ull ^ n, h1 = 0x13198A2E03707344ull;
    uint64_t h2 = 0xA4093822299F31D0ull, h3 = 0x082EFA98EC4E6C89ull;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        uint64_t w0, w1, w2, w3;
        memcpy(&w0, p + i, 8);
        memcpy(&w1, p + i + 8, 8);
        memcpy(&w2, p + i + 16, 8);
        memcpy(&w3, p + i + 24, 8);
        h0 = (h0 ^ w0) * K; h0 ^= h0 >> 29;
        h1 = (h1 ^ w1) * K; h1 ^= h1 >> 29;
        h2 = (h2 ^ w2) * K; h2 ^= h2 >> 29;
        h3 = (h3 ^ w3) * K; h3 ^= h3 >> 29;
    }
    for (; i < n; i++) {
        h0 = (h0 ^ p[i]) * K;
        h0 ^= h0 >> 29;
    }
    uint64_t h = h0 ^ (h1 * 3) ^ (h2 * 5) ^ (h3 * 7);
    h ^= h >> 33; h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33; h *= 0xC4CEB9FE1A85EC53ull;
    h ^= h >> 33;
    return h;
}

static inline uint64_t es_header_hash(const es_header *h) {
    return es_hash64(h, offsetof(es_header, header_hash));
}

/* Validate a header that sits at the start of a fully loaded file buffer.
 * Returns NULL on success or a short reason string. If check_data is set,
 * every tensor's payload hash is checked too (costs one pass over memory). */
static inline const char *es_validate(const void *buf, size_t len, int check_data) {
    const es_header *h = (const es_header *)buf;
    if (len < ES_HEADER_BYTES) return "short file";
    if (h->magic != ES_MAGIC) return "bad magic";
    if (h->version != ES_VERSION) return "unsupported version";
    if (h->header_bytes != ES_HEADER_BYTES) return "bad header size";
    if (h->header_hash != es_header_hash(h)) return "header hash mismatch";
    if (h->n_tensors > ES_MAX_TENSORS) return "too many tensors";
    if (h->file_bytes != len) return "file size mismatch";
    for (uint32_t i = 0; i < h->n_tensors; i++) {
        const es_tensor_desc *t = &h->t[i];
        if (t->offset % ES_ALIGN) return "unaligned tensor";
        if (t->offset < ES_HEADER_BYTES || t->offset + t->nbytes > len)
            return "tensor out of bounds";
        if (check_data &&
            es_hash64((const uint8_t *)buf + t->offset, t->nbytes) != t->hash)
            return "tensor data hash mismatch";
    }
    return NULL;
}

#endif
