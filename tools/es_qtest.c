/*
 * es_qtest: dump a dequantized GGUF tensor as raw float32, so the
 * dequantizers can be checked against an independent implementation.
 *
 *   es_qtest model.gguf blk.0.attn_v.weight out.f32 [max_rows]
 *   es_qtest --raw TYPE_ID blocks.bin out.f32      (raw blocks of one type, e.g. from scripts/test_quants.py)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../src/es_gguf.h"
#include "../src/es_quant.h"

static int raw_mode(int type, const char *in, const char *outp) {
    FILE *f = fopen(in, "rb");
    if (!f) { perror(in); return 1; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc((size_t)n);
    if (fread(b, 1, (size_t)n, f) != (size_t)n) return 1;
    fclose(f);
    if (!es_quant_supported((uint32_t)type)) { fprintf(stderr, "unsupported type %d\n", type); return 1; }
    const size_t r256 = es_row_bytes((uint32_t)type, 256);
    const int cols = (int)((size_t)n / r256 * 256);
    float *out = malloc(sizeof(float) * (size_t)cols);
    es_dequant_row((uint32_t)type, b, out, cols);
    FILE *o = fopen(outp, "wb");
    fwrite(out, sizeof(float), (size_t)cols, o);
    fclose(o);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 5 && !strcmp(argv[1], "--raw")) return raw_mode(atoi(argv[2]), argv[3], argv[4]);
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
