/*
 * es_qtest: dump a dequantized GGUF tensor as raw float32, so the
 * dequantizers can be checked against an independent implementation.
 *
 *   es_qtest model.gguf blk.0.attn_v.weight out.f32 [max_rows]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "../src/es_gguf.h"
#include "../src/es_quant.h"

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: es_qtest FILE.gguf TENSOR OUT.f32 [max_rows]\n");
        return 2;
    }
    es_gguf g;
    const char *err;
    if (es_gguf_open(&g, argv[1], &err)) { fprintf(stderr, "%s\n", err); return 1; }
    const es_gguf_tensor *t = es_gguf_tensor_find(&g, argv[2]);
    if (!t) { fprintf(stderr, "no tensor %s\n", argv[2]); return 1; }
    if (!es_quant_supported(t->type)) { fprintf(stderr, "unsupported type %u\n", t->type); return 1; }
    uint64_t rows = t->ne[1] * t->ne[2] * t->ne[3];
    if (argc > 4 && (uint64_t)atoll(argv[4]) < rows) rows = (uint64_t)atoll(argv[4]);
    const int cols = (int)t->ne[0];
    const size_t rb = es_row_bytes(t->type, cols);
    void *row = malloc(rb);
    float *out = malloc(sizeof(float) * cols);
    FILE *f = fopen(argv[3], "wb");
    for (uint64_t r = 0; r < rows; r++) {
        if (pread(g.fd, row, rb, (off_t)(g.data_start + t->offset + r * rb)) != (ssize_t)rb) return 1;
        es_dequant_row(t->type, row, out, cols);
        fwrite(out, sizeof(float), (size_t)cols, f);
    }
    fclose(f);
    printf("%s %s: %llu rows x %d\n", argv[2], es_ggml_type_name(t->type), (unsigned long long)rows, cols);
    es_gguf_close(&g);
    return 0;
}
