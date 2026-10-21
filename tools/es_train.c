/* es_train: one ExpertStream trainer, two modes.
 *   es_train finetune train|eval|gradcheck|logitcheck|export ...   LoRA on a frozen GGUF (was es_lora; same flags)
 *   es_train moe gen   --experts-file F --layers L --experts E --dim D --ffn F [--seed S]
 *   es_train moe train --experts-file F --layers L --experts E --dim D --ffn F --data FILE
 *                      [--topk 2] [--rank 8] [--steps 20] [--cache-mb 256] [--batch-seqs 4] [--seq 128]
 *                      [--balance 0.01 (router aux loss)] [--random-route] [--ternary (default; experts are always 2-bit ternary)]
 * Shared: -j threads (finetune), ES_BLAS=/path/libopenblas (both; required for moe), fast mode ES_AMX=1 ES_LORA_BLAS=1 (finetune).
 * Called as "es_lora" (symlink or copy), it behaves exactly like the old es_lora. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int es_lora_main(int argc, char **argv);
int es_moe_main(int argc, char **argv);

static const char *opt(int argc, char **argv, const char *n, const char *d) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], n)) return argv[i + 1];
    return d;
}
static int flag(int argc, char **argv, const char *n) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], n)) return 1; return 0; }
static void usage(void) {
    fprintf(stderr,
        "usage: es_train MODE ...\n"
        "  finetune  LoRA fine-tuning of a frozen GGUF model (llama/qwen). Same subcommands and flags as es_lora:\n"
        "            es_train finetune train -g MODEL.gguf --data FILE -o NAME.eslora [--rank 16] [--alpha 16] [--targets all]\n"
        "                     [--steps N] [--seq 512] [--batch 2048] [--micro 1024] [--lr 2e-4] [--eval-every N] [--resume] [-j 1]\n"
        "            es_train finetune eval|gradcheck|logitcheck|export ...\n"
        "  moe       from-scratch byte-level MoE prototype; frozen ternary experts streamed from disk through an LRU cache,\n"
        "            trainable embeddings/attention/router/per-expert LoRA\n"
        "            es_train moe gen   --experts-file F --layers L --experts E --dim D --ffn F [--seed 1]\n"
        "            es_train moe train --experts-file F --layers L --experts E --dim D --ffn F --data FILE [--topk 2] [--rank 8]\n"
        "                     [--steps 20] [--cache-mb 256] [--batch-seqs 4] [--seq 128] [--balance 0] [--random-route]\n"
        "  env: ES_BLAS=libopenblas.so (moe: required; finetune: optional), ES_AMX=1 ES_LORA_BLAS=1 (finetune fast mode, 1 thread),\n"
        "       ES_LORA_RECOMPUTE=1 (finetune low-RAM). Calling the binary as es_lora is the same as es_train finetune.\n");
}
static int moe(int argc, char **argv) {
    if (argc < 3 || (strcmp(argv[2], "gen") && strcmp(argv[2], "train"))) { usage(); return 2; }
    const char *f = opt(argc, argv, "--experts-file", NULL), *L = opt(argc, argv, "--layers", NULL), *E = opt(argc, argv, "--experts", NULL),
               *D = opt(argc, argv, "--dim", NULL), *F = opt(argc, argv, "--ffn", NULL);
    if (!f || !L || !E || !D || !F) { fprintf(stderr, "moe: need --experts-file --layers --experts --dim --ffn\n"); return 2; }
    if (!strcmp(argv[2], "gen")) {
        char *a[] = {"es_moe", "gen", (char *)f, (char *)L, (char *)E, (char *)D, (char *)F, (char *)opt(argc, argv, "--seed", "1"), NULL};
        return es_moe_main(8, a);
    }
    const char *data = opt(argc, argv, "--data", NULL);
    if (!data) { fprintf(stderr, "moe train: need --data FILE\n"); return 2; }
    if (flag(argc, argv, "--random-route")) setenv("ES_RANDROUTE", "1", 1);
    const char *bal = opt(argc, argv, "--balance", NULL); if (bal) setenv("ES_AUX", bal, 1);
    char *a[] = {"es_moe", "train", (char *)f, (char *)L, (char *)E, (char *)D, (char *)F, (char *)opt(argc, argv, "--topk", "2"),
                 (char *)opt(argc, argv, "--rank", "8"), (char *)opt(argc, argv, "--steps", "20"), (char *)opt(argc, argv, "--cache-mb", "256"),
                 (char *)data, (char *)opt(argc, argv, "--batch-seqs", "4"), (char *)opt(argc, argv, "--seq", "128"), NULL};
    return es_moe_main(14, a);
}
int main(int argc, char **argv) {
    const char *b = strrchr(argv[0], '/'); b = b ? b + 1 : argv[0];
    if (!strcmp(b, "es_lora")) return es_lora_main(argc, argv);
    if (argc < 2 || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) { usage(); return argc < 2; }
    if (!strcmp(argv[1], "finetune")) { argv[1] = "es_lora"; return es_lora_main(argc - 1, argv + 1); }
    if (!strcmp(argv[1], "moe")) return moe(argc, argv);
    fprintf(stderr, "unknown mode %s\n", argv[1]); usage(); return 2;
}
