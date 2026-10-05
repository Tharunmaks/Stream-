/*
 * es_gen: write synthetic expert files in the ExpertStream v1 format.
 *
 * The payload is random bytes sized exactly like a Q2_K-quantized SwiGLU
 * expert (gate, up: inter x hidden; down: hidden x inter). It is for I/O
 * benchmarking only; real conversion comes in a later step.
 *
 *   ./es_gen -d ~/es_bench -L 4 -E 64 -H 4096 -I 1536
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../src/es_format.h"
#include "../src/es_io.h"
#include "es_common.h"

static uint64_t q2k_bytes(uint64_t rows, uint64_t cols) {
    /* Q2_K: 256 weights per block, 84 bytes per block */
    return rows * ((cols + 255) / 256) * 84;
}

static void usage(void) {
    fprintf(stderr,
            "usage: es_gen -d DIR [-L layers=4] [-E experts=64] [-H hidden=4096]\n"
            "              [-I inter=1536] [-S seed=1] [-n (no fsync)]\n"
            "Presets: Qwen3-235B-A22B  -H 4096 -I 1536  (~5.8 MiB/expert)\n"
            "         DeepSeek-V3/Kimi -H 7168 -I 2048  (~13.8 MiB/expert)\n");
    exit(2);
}

int main(int argc, char **argv) {
    const char *root = NULL;
    unsigned layers = 4, experts = 64, hidden = 4096, inter = 1536;
    uint64_t seed = 1;
    int do_fsync = 1, opt;
    while ((opt = getopt(argc, argv, "d:L:E:H:I:S:nh")) != -1) {
        switch (opt) {
        case 'd': root = optarg; break;
        case 'L': layers = (unsigned)atoi(optarg); break;
        case 'E': experts = (unsigned)atoi(optarg); break;
        case 'H': hidden = (unsigned)atoi(optarg); break;
        case 'I': inter = (unsigned)atoi(optarg); break;
        case 'S': seed = strtoull(optarg, NULL, 10); break;
        case 'n': do_fsync = 0; break;
        default: usage();
        }
    }
    if (!root || !layers || !experts || !hidden || !inter) usage();

    struct { uint32_t role, rows, cols; } shapes[3] = {
        {ES_ROLE_GATE, inter, hidden},
        {ES_ROLE_UP, inter, hidden},
        {ES_ROLE_DOWN, hidden, inter},
    };
    uint64_t file_bytes = ES_HEADER_BYTES;
    for (int i = 0; i < 3; i++)
        file_bytes += es_round_up(q2k_bytes(shapes[i].rows, shapes[i].cols), ES_ALIGN);

    uint8_t *buf = es_alloc(file_bytes);
    if (!buf) { perror("alloc"); return 1; }

    if (mkdir(root, 0755) != 0 && errno != EEXIST) { perror(root); return 1; }

    double total_gb = (double)file_bytes * layers * experts / 1e9;
    printf("es_gen: %u layers x %u experts, %.2f MiB/file, %.2f GB total\n",
           layers, experts, file_bytes / 1048576.0, total_gb);

    uint64_t t0 = es_now_ns();
    char path[4096];
    for (unsigned l = 0; l < layers; l++) {
        snprintf(path, sizeof path, "%s/L%03u", root, l);
        if (mkdir(path, 0755) != 0 && errno != EEXIST) { perror(path); return 1; }
        for (unsigned e = 0; e < experts; e++) {
            memset(buf, 0, file_bytes);
            uint64_t s = seed * 0x9E3779B97F4A7C15ull + ((uint64_t)l << 32) + e + 1;
            es_header *h = (es_header *)buf;
            h->magic = ES_MAGIC;
            h->version = ES_VERSION;
            h->header_bytes = ES_HEADER_BYTES;
            h->n_tensors = 3;
            h->layer = l;
            h->expert = e;
            h->flags = ES_FLAG_SYNTHETIC;
            h->file_bytes = file_bytes;
            h->model_id = es_hash64("synthetic", 9);
            uint64_t off = ES_HEADER_BYTES;
            for (int i = 0; i < 3; i++) {
                es_tensor_desc *t = &h->t[i];
                t->role = shapes[i].role;
                t->qtype = ES_QT_Q2_K;
                t->rows = shapes[i].rows;
                t->cols = shapes[i].cols;
                t->offset = off;
                t->nbytes = q2k_bytes(t->rows, t->cols);
                uint64_t *w = (uint64_t *)(buf + off);
                for (uint64_t k = 0; k < t->nbytes / 8; k++) w[k] = es_rng_next(&s);
                t->hash = es_hash64(buf + off, t->nbytes);
                off += es_round_up(t->nbytes, ES_ALIGN);
            }
            h->header_hash = es_header_hash(h);

            es_expert_path(path, sizeof path, root, l, e);
            int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
            if (fd < 0) { perror(path); return 1; }
            size_t done = 0;
            while (done < file_bytes) {
                ssize_t n = write(fd, buf + done, file_bytes - done);
                if (n < 0) {
                    if (errno == EINTR) continue;
                    perror(path);
                    return 1;
                }
                done += (size_t)n;
            }
            if (do_fsync && fsync(fd) != 0) { perror("fsync"); return 1; }
            /* don't leave the new file in the page cache: it would fake reads */
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            close(fd);
        }
        double secs = (es_now_ns() - t0) / 1e9;
        printf("  layer %u/%u written  (%.0f MB/s incl. RNG%s)\n", l + 1, layers,
               (double)file_bytes * experts * (l + 1) / 1e6 / secs,
               do_fsync ? " + fsync" : "");
        fflush(stdout);
    }

    snprintf(path, sizeof path, "%s/manifest.txt", root);
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return 1; }
    fprintf(f, "layers %u\nexperts %u\nfile_bytes %llu\n", layers, experts,
            (unsigned long long)file_bytes);
    fclose(f);
    free(buf);
    printf("done: %s\n", root);
    return 0;
}
