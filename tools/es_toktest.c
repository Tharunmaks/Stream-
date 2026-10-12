/* Tokenizer check against reference outputs.
 *   es_toktest VOCAB.gguf TESTS.inp TESTS.out   llama.cpp test files: inputs separated by "\n__ggml_vocab_test__\n",
 *                                               one line of expected ids per input
 *   es_toktest VOCAB.gguf TEXTFILE              print the ids of the file's text (special tokens parsed)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/es_gguf.h"
#include "../src/es_tok.h"

static char *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb");
    if (!f) { perror(p); exit(2); }
    fseek(f, 0, SEEK_END); long L = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)L + 1);
    *n = fread(b, 1, (size_t)L, f); b[*n] = 0; fclose(f);
    return b;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: es_toktest VOCAB.gguf TESTS.inp TESTS.out | es_toktest VOCAB.gguf TEXT\n"); return 2; }
    es_gguf g; const char *err;
    if (es_gguf_open(&g, argv[1], &err)) { fprintf(stderr, "%s: %s\n", argv[1], err); return 2; }
    es_tok *t = es_tok_load(&g, &err);
    if (!t) { fprintf(stderr, "tokenizer: %s\n", err); return 2; }
    static int32_t ids[1 << 20];
    size_t n;
    if (argc == 3) {
        char *text = slurp(argv[2], &n);
        const int k = es_tok_encode(t, text, 1, ids, 1 << 20);
        for (int i = 0; i < k; i++) printf("%d\n", ids[i]);
        return 0;
    }
    char *inp = slurp(argv[2], &n), *out = slurp(argv[3], &n);
    const char *sep = "\n__ggml_vocab_test__\n";
    int total = 0, bad = 0;
    char *ip = inp, *op = out;
    for (;;) {
        char *e = strstr(ip, sep);
        if (!e) break;
        *e = 0;
        char *nl = strchr(op, '\n');
        if (nl) *nl = 0;
        const int k = es_tok_encode(t, ip, 0, ids, 1 << 20);
        /* expected ids */
        int ok = 1, j = 0;
        char *q = op;
        for (;;) {
            while (*q == ' ') q++;
            if (!*q) break;
            const long v = strtol(q, &q, 10);
            if (j >= k || ids[j] != v) ok = 0;
            j++;
        }
        if (j != k) ok = 0;
        total++;
        if (!ok) {
            bad++;
            if (bad <= 5) {
                fprintf(stderr, "MISMATCH for \"%.60s\"\n  want: %s\n  got: ", ip, op);
                for (int i = 0; i < k; i++) fprintf(stderr, " %d", ids[i]);
                fprintf(stderr, "\n");
            }
        }
        ip = e + strlen(sep);
        if (!nl) break;
        op = nl + 1;
    }
    printf("%s: %d/%d tests match\n", argv[1], total - bad, total);
    return bad != 0;
}
