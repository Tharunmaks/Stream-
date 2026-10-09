/*
 * Minimal GGUF (v2/v3) reader: metadata key/values and the tensor table.
 * Tensor data is not loaded; callers pread() what they need using
 * es_gguf.data_start + tensor.offset.
 */
#ifndef ES_GGUF_H
#define ES_GGUF_H

#include <stddef.h>
#include <stdint.h>

enum {
    GGUF_T_U8 = 0, GGUF_T_I8, GGUF_T_U16, GGUF_T_I16, GGUF_T_U32, GGUF_T_I32,
    GGUF_T_F32, GGUF_T_BOOL, GGUF_T_STRING, GGUF_T_ARRAY, GGUF_T_U64, GGUF_T_I64,
    GGUF_T_F64,
};

typedef struct {
    char    *key;
    uint32_t type;
    uint32_t arr_type;   /* GGUF_T_ARRAY only */
    uint64_t arr_n;      /* GGUF_T_ARRAY only */
    union { uint64_t u; int64_t i; double f; } v;   /* scalars */
    char    *str;        /* GGUF_T_STRING */
    uint64_t raw_start, raw_end; /* this entry's bytes in the file */
} es_gguf_kv;

typedef struct {
    char    *name;
    uint32_t n_dims;
    uint64_t ne[4];      /* ne[0] is the fastest-moving (row length) */
    uint32_t type;       /* ggml type */
    uint64_t offset;     /* relative to data_start */
    uint64_t nbytes;
} es_gguf_tensor;

typedef struct {
    int       fd;
    uint64_t  file_size;
    uint32_t  version;
    uint64_t  n_kv, n_tensors;
    es_gguf_kv     *kv;
    es_gguf_tensor *t;
    uint64_t  alignment;
    uint64_t  kv_start, kv_end;   /* byte range of the KV section (copyable) */
    uint64_t  data_start;
} es_gguf;

/* Returns 0 or sets err (static string) and returns -1. */
int  es_gguf_open(es_gguf *g, const char *path, const char **err);
void es_gguf_close(es_gguf *g);

const es_gguf_kv     *es_gguf_find(const es_gguf *g, const char *key);
const es_gguf_tensor *es_gguf_tensor_find(const es_gguf *g, const char *name);
/* Integer metadata value (any int/bool type), or def if missing. */
int64_t     es_gguf_int(const es_gguf *g, const char *key, int64_t def);
const char *es_gguf_str(const es_gguf *g, const char *key);

/* ggml type info: block size (weights) and bytes per block; 0 if unknown. */
int         es_ggml_type_info(uint32_t type, uint32_t *blck, uint32_t *tsize);
const char *es_ggml_type_name(uint32_t type);
/* Bytes of one row of ne0 weights; 0 if the type is unknown or ne0 is not
 * a multiple of the block size. */
uint64_t    es_ggml_row_bytes(uint32_t type, uint64_t ne0);

#endif
