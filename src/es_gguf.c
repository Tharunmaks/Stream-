#define _GNU_SOURCE
#include "es_gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747u /* "GGUF" */

static const struct { uint32_t blck, tsize; const char *name; } k_types[] = {
    [0] = {1, 4, "F32"},       [1] = {1, 2, "F16"},       [2] = {32, 18, "Q4_0"},
    [3] = {32, 20, "Q4_1"},    [6] = {32, 22, "Q5_0"},    [7] = {32, 24, "Q5_1"},
    [8] = {32, 34, "Q8_0"},    [9] = {32, 36, "Q8_1"},    [10] = {256, 84, "Q2_K"},
    [11] = {256, 110, "Q3_K"}, [12] = {256, 144, "Q4_K"}, [13] = {256, 176, "Q5_K"},
    [14] = {256, 210, "Q6_K"}, [15] = {256, 292, "Q8_K"}, [16] = {256, 66, "IQ2_XXS"},
    [17] = {256, 74, "IQ2_XS"}, [18] = {256, 98, "IQ3_XXS"}, [19] = {256, 50, "IQ1_S"},
    [20] = {32, 18, "IQ4_NL"}, [21] = {256, 110, "IQ3_S"}, [22] = {256, 82, "IQ2_S"},
    [23] = {256, 136, "IQ4_XS"}, [24] = {1, 1, "I8"},     [25] = {1, 2, "I16"},
    [26] = {1, 4, "I32"},      [27] = {1, 8, "I64"},      [28] = {1, 8, "F64"},
    [29] = {256, 56, "IQ1_M"}, [30] = {1, 2, "BF16"},     [34] = {256, 54, "TQ1_0"},
    [35] = {256, 66, "TQ2_0"}, [39] = {32, 17, "MXFP4"},
};
#define N_TYPES (sizeof k_types / sizeof k_types[0])

int es_ggml_type_info(uint32_t type, uint32_t *blck, uint32_t *tsize) {
    if (type >= N_TYPES || k_types[type].blck == 0) return 0;
    *blck = k_types[type].blck;
    *tsize = k_types[type].tsize;
    return 1;
}

const char *es_ggml_type_name(uint32_t type) {
    return (type < N_TYPES && k_types[type].name) ? k_types[type].name : "unknown";
}

uint64_t es_ggml_row_bytes(uint32_t type, uint64_t ne0) {
    uint32_t b, s;
    if (!es_ggml_type_info(type, &b, &s) || ne0 % b) return 0;
    return ne0 / b * s;
}

/* ---- buffered sequential reader over the header ---- */
typedef struct {
    int fd;
    uint64_t pos, size;
    uint8_t buf[1 << 16];
    size_t blen, boff;
    int err;
} rd;

static int rd_bytes(rd *r, void *dst, size_t n) {
    uint8_t *d = dst;
    while (n) {
        if (r->boff == r->blen) {
            ssize_t got = pread(r->fd, r->buf, sizeof r->buf, (off_t)r->pos);
            if (got <= 0) { r->err = 1; return -1; }
            r->blen = (size_t)got;
            r->boff = 0;
        }
        size_t k = r->blen - r->boff < n ? r->blen - r->boff : n;
        memcpy(d, r->buf + r->boff, k);
        r->boff += k; r->pos += k; d += k; n -= k;
    }
    return 0;
}
/* r->pos counts bytes consumed; the next pread starts after the buffer */
static uint64_t rd_tell(const rd *r) { return r->pos; }

static uint64_t rd_u64(rd *r) { uint64_t v = 0; rd_bytes(r, &v, 8); return v; }
static uint32_t rd_u32(rd *r) { uint32_t v = 0; rd_bytes(r, &v, 4); return v; }

static char *rd_str(rd *r) {
    uint64_t n = rd_u64(r);
    if (r->err || n > (1u << 30)) { r->err = 1; return NULL; }
    char *s = malloc(n + 1);
    if (!s || rd_bytes(r, s, n)) { free(s); r->err = 1; return NULL; }
    s[n] = 0;
    return s;
}

static size_t scalar_size(uint32_t t) {
    switch (t) {
    case GGUF_T_U8: case GGUF_T_I8: case GGUF_T_BOOL: return 1;
    case GGUF_T_U16: case GGUF_T_I16: return 2;
    case GGUF_T_U32: case GGUF_T_I32: case GGUF_T_F32: return 4;
    case GGUF_T_U64: case GGUF_T_I64: case GGUF_T_F64: return 8;
    }
    return 0;
}

static void read_scalar(rd *r, uint32_t t, es_gguf_kv *kv) {
    uint8_t raw[8] = {0};
    size_t n = scalar_size(t);
    if (!n) { r->err = 1; return; }
    rd_bytes(r, raw, n);
    switch (t) {
    case GGUF_T_U8: case GGUF_T_BOOL: kv->v.u = raw[0]; break;
    case GGUF_T_I8: kv->v.i = (int8_t)raw[0]; break;
    case GGUF_T_U16: { uint16_t x; memcpy(&x, raw, 2); kv->v.u = x; } break;
    case GGUF_T_I16: { int16_t x; memcpy(&x, raw, 2); kv->v.i = x; } break;
    case GGUF_T_U32: { uint32_t x; memcpy(&x, raw, 4); kv->v.u = x; } break;
    case GGUF_T_I32: { int32_t x; memcpy(&x, raw, 4); kv->v.i = x; } break;
    case GGUF_T_F32: { float x; memcpy(&x, raw, 4); kv->v.f = x; } break;
    case GGUF_T_U64: memcpy(&kv->v.u, raw, 8); break;
    case GGUF_T_I64: memcpy(&kv->v.i, raw, 8); break;
    case GGUF_T_F64: memcpy(&kv->v.f, raw, 8); break;
    }
}

static void skip_array(rd *r, uint32_t t, uint64_t n) {
    if (t == GGUF_T_STRING) {
        for (uint64_t i = 0; i < n && !r->err; i++) free(rd_str(r));
        return;
    }
    size_t sz = scalar_size(t);
    if (!sz) { r->err = 1; return; }
    /* jump without reading: drop the buffer and move pos */
    uint64_t skip = sz * n, inbuf = r->blen - r->boff;
    if (skip <= inbuf) { r->boff += skip; r->pos += skip; return; }
    r->pos += skip;
    r->boff = r->blen = 0;
}

int es_gguf_open(es_gguf *g, const char *path, const char **err) {
    memset(g, 0, sizeof *g);
    g->fd = open(path, O_RDONLY | O_CLOEXEC);
    if (g->fd < 0) { *err = strerror(errno); return -1; }
    struct stat st;
    fstat(g->fd, &st);
    g->file_size = (uint64_t)st.st_size;

    rd *r = calloc(1, sizeof *r);
    r->fd = g->fd;
    if (rd_u32(r) != GGUF_MAGIC) { *err = "not a GGUF file"; goto fail; }
    g->version = rd_u32(r);
    if (g->version < 2 || g->version > 3) { *err = "unsupported GGUF version"; goto fail; }
    g->n_tensors = rd_u64(r);
    g->n_kv = rd_u64(r);
    if (g->n_kv > 1000000 || g->n_tensors > 10000000) { *err = "corrupt header"; goto fail; }
    g->kv_start = rd_tell(r);
    g->kv = calloc(g->n_kv ? g->n_kv : 1, sizeof *g->kv);
    g->alignment = 32;
    for (uint64_t i = 0; i < g->n_kv && !r->err; i++) {
        es_gguf_kv *kv = &g->kv[i];
        kv->raw_start = rd_tell(r);
        kv->key = rd_str(r);
        kv->type = rd_u32(r);
        if (kv->type == GGUF_T_STRING) {
            kv->str = rd_str(r);
        } else if (kv->type == GGUF_T_ARRAY) {
            kv->arr_type = rd_u32(r);
            kv->arr_n = rd_u64(r);
            skip_array(r, kv->arr_type, kv->arr_n);
        } else {
            read_scalar(r, kv->type, kv);
        }
        kv->raw_end = rd_tell(r);
        if (kv->key && !strcmp(kv->key, "general.alignment") && kv->v.u) g->alignment = kv->v.u;
    }
    g->kv_end = rd_tell(r);
    g->t = calloc(g->n_tensors ? g->n_tensors : 1, sizeof *g->t);
    for (uint64_t i = 0; i < g->n_tensors && !r->err; i++) {
        es_gguf_tensor *t = &g->t[i];
        t->name = rd_str(r);
        t->n_dims = rd_u32(r);
        if (t->n_dims < 1 || t->n_dims > 4) { *err = "bad tensor rank"; goto fail; }
        for (int d = 0; d < 4; d++) t->ne[d] = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) t->ne[d] = rd_u64(r);
        t->type = rd_u32(r);
        t->offset = rd_u64(r);
        uint64_t row = es_ggml_row_bytes(t->type, t->ne[0]);
        t->nbytes = row * t->ne[1] * t->ne[2] * t->ne[3];
    }
    if (r->err) { *err = "truncated or corrupt GGUF header"; goto fail; }
    g->data_start = (rd_tell(r) + g->alignment - 1) / g->alignment * g->alignment;
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (g->t[i].nbytes && g->data_start + g->t[i].offset + g->t[i].nbytes > g->file_size) {
            *err = "tensor data runs past end of file (incomplete download?)";
            goto fail;
        }
    free(r);
    return 0;
fail:
    free(r);
    es_gguf_close(g);
    return -1;
}

void es_gguf_close(es_gguf *g) {
    for (uint64_t i = 0; g->kv && i < g->n_kv; i++) { free(g->kv[i].key); free(g->kv[i].str); }
    for (uint64_t i = 0; g->t && i < g->n_tensors; i++) free(g->t[i].name);
    free(g->kv);
    free(g->t);
    if (g->fd > 0) close(g->fd);
    memset(g, 0, sizeof *g);
}

const es_gguf_kv *es_gguf_find(const es_gguf *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++)
        if (g->kv[i].key && !strcmp(g->kv[i].key, key)) return &g->kv[i];
    return NULL;
}

const es_gguf_tensor *es_gguf_tensor_find(const es_gguf *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++)
        if (!strcmp(g->t[i].name, name)) return &g->t[i];
    return NULL;
}

int64_t es_gguf_int(const es_gguf *g, const char *key, int64_t def) {
    const es_gguf_kv *kv = es_gguf_find(g, key);
    if (!kv) return def;
    switch (kv->type) {
    case GGUF_T_U8: case GGUF_T_U16: case GGUF_T_U32: case GGUF_T_U64: case GGUF_T_BOOL:
        return (int64_t)kv->v.u;
    case GGUF_T_I8: case GGUF_T_I16: case GGUF_T_I32: case GGUF_T_I64:
        return kv->v.i;
    }
    return def;
}

const char *es_gguf_str(const es_gguf *g, const char *key) {
    const es_gguf_kv *kv = es_gguf_find(g, key);
    return kv && kv->type == GGUF_T_STRING ? kv->str : NULL;
}
