/*
 * es_get: download a GGUF model from Hugging Face, either whole or only its core.
 *
 *   es_get REPO FILE [-d DIR] [--all | --plan]
 *   es_get unsloth/Kimi-K2-Instruct-GGUF UD-TQ1_0/Kimi-K2-Instruct-UD-TQ1_0-00001-of-00005.gguf -d ~/models
 *
 * Without --all only the core is downloaded: the file header, attention, embeddings, routers, shared experts and dense
 * layers. The files are created at full size but sparse, so the routed experts take no space until the engine fetches
 * each one the first time it is chosen (and keeps it, within ES_DISK_GB). Split models: name the first part; all parts
 * are handled. The engine finds the source URLs in FILE.esurl next to the first part. Uses curl; HF_TOKEN for gated repos.
 * Running it again resumes: parts of the core already downloaded are not fetched again.
 * --plan only reads the headers (a few MB) and prints how much would be downloaded now and later; nothing is kept.
 * --skeleton keeps the headers in full-size sparse files and nothing else: every weight reads as zero. The engine loads and
 * runs it (meaningless output), which checks that it handles the model's structure without downloading the model.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "../src/es_fetch.h"
#include "../src/es_gguf.h"
#include "../src/es_quant.h"

static int is_expert_tensor(const char *nm) {
    if (strstr(nm, "_exps.weight")) return 1;
    int l, e; char role[8], rest[16];
    return sscanf(nm, "blk.%d.ffn_%7[a-z].%d.%15s", &l, role, &e, rest) == 4 && !strcmp(rest, "weight");
}
typedef struct { uint64_t a, b; } range;
static int cmp_range(const void *x, const void *y) { const range *p = x, *q = y; return p->a < q->a ? -1 : p->a > q->a; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

/* fetch [a, b) of url into fd, in pieces, with progress */
static int fetch(const char *url, int fd, uint64_t a, uint64_t b, double *done_mb, double total_mb, double t0) {
    const uint64_t piece = 64ull << 20;
    for (uint64_t o = a; o < b; o += piece) {
        const uint64_t n = b - o < piece ? b - o : piece;
        if (es_http_range(url, o, n, fd, o)) { fprintf(stderr, "\ndownload failed at byte %llu of %s\n", (unsigned long long)o, url); return -1; }
        *done_mb += n / 1048576.0;
        const double el = now() - t0;
        fprintf(stderr, "\r  %.0f / %.0f MB  (%.1f MB/s)   ", *done_mb, total_mb, el > 0 ? *done_mb / el : 0);
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *repo = NULL, *file = NULL, *dir = ".";
    int all = 0, plan = 0, skel = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-d") && i + 1 < argc) dir = argv[++i];
        else if (!strcmp(argv[i], "--all")) all = 1;
        else if (!strcmp(argv[i], "--plan")) plan = 1;
        else if (!strcmp(argv[i], "--skeleton")) skel = 1;
        else if (!repo) repo = argv[i];
        else if (!file) file = argv[i];
    }
    if (!repo || !file) { fprintf(stderr, "usage: es_get REPO FILE [-d DIR] [--all | --plan | --skeleton]\n"); return 2; }
    mkdir(dir, 0755);
    int nparts = 1;
    const char *dash = strstr(file, "-00001-of-");
    if (dash) nparts = atoi(dash + 10);
    if (nparts < 1 || nparts > 255) { fprintf(stderr, "bad split count\n"); return 2; }
    const char *base1 = strrchr(file, '/') ? strrchr(file, '/') + 1 : file;
    char path1[4096], urlf[4200];
    snprintf(path1, sizeof path1, "%s/%s", dir, base1);
    snprintf(urlf, sizeof urlf, "%s.esurl", path1);
    FILE *uf = plan || skel ? fopen("/dev/null", "w") : fopen(urlf, "w");
    if (!uf) { perror(urlf); return 1; }
    double total_core = 0, total_all = 0, total_exp = 0;
    for (int part = 1; part <= nparts; part++) {
        char name[4096], url[8192], path[4400];
        if (dash) snprintf(name, sizeof name, "%.*s-%05d-of-%05d%s", (int)(dash - file), file, part, nparts, dash + 15);
        else snprintf(name, sizeof name, "%s", file);
        es_hf_url(url, sizeof url, repo, name);
        const char *b = strrchr(name, '/') ? strrchr(name, '/') + 1 : name;
        snprintf(path, sizeof path, "%s/%s", dir, b);
        if (plan) snprintf(path, sizeof path, "%s/.es_plan_%d.gguf", dir, (int)getpid());
        const int64_t size = es_http_size(url);
        if (size <= 0) { fprintf(stderr, "cannot reach %s (curl installed? the right file name?)\n", url); return 1; }
        fprintf(stderr, "%s: %.2f GB\n", b, size / 1e9);
        const int fd = open(path, O_RDWR | O_CREAT, 0644);
        struct stat st;
        if (fd < 0 || fstat(fd, &st)) { perror(path); return 1; }
        if (st.st_size != size && ftruncate(fd, size)) { perror("ftruncate"); return 1; }   /* sparse: holes take no space */
        char donef[4500];
        snprintf(donef, sizeof donef, "%s.escore", path);
        const int core_done = !plan && access(donef, F_OK) == 0;
        /* the header: fetch more until it parses and the tensor table is complete */
        es_gguf g; const char *err;
        uint64_t hdr = 16ull << 20;
        for (;;) {
            if (hdr > (uint64_t)size) hdr = (uint64_t)size;
            if (!core_done && es_http_range(url, 0, hdr, fd, 0)) { fprintf(stderr, "header download failed\n"); return 1; }
            if (es_gguf_open(&g, path, &err) == 0) {   /* a header cut short reads zeros: every tensor must have a name */
                int whole = g.data_start <= hdr;
                for (uint64_t i = 0; whole && i < g.n_tensors; i++) whole = g.t[i].name && g.t[i].name[0];
                if (whole) break;
                es_gguf_close(&g);
            }
            if (core_done || hdr >= (uint64_t)size || hdr >= (512ull << 20)) { fprintf(stderr, "%s: not a readable GGUF (%s)\n", b, err ? err : "header"); return 1; }
            hdr *= 2;
        }
        for (uint64_t i = 0; i < g.n_tensors; i++) if (!es_quant_supported(g.t[i].type)) {
            fprintf(stderr, "%s: %s uses %s (type %u), which the engine does not support yet; pick another quantization\n", b, g.t[i].name, es_ggml_type_name(g.t[i].type), g.t[i].type);
            return 1;
        }
        /* core tensors (everything but routed experts), merged into few ranges */
        range *rg = malloc(sizeof(range) * (g.n_tensors + 1));
        int nr = 0;
        double exp_bytes = 0;
        for (uint64_t i = 0; i < g.n_tensors; i++) {
            const uint64_t a = g.data_start + g.t[i].offset, e = a + g.t[i].nbytes;
            if (is_expert_tensor(g.t[i].name)) { exp_bytes += (double)g.t[i].nbytes; if (!all) continue; }
            rg[nr++] = (range){a, e};
        }
        qsort(rg, (size_t)nr, sizeof *rg, cmp_range);
        int m = 0;
        for (int i = 0; i < nr; i++) {
            if (m && rg[i].a <= rg[m - 1].b + (4u << 20)) { if (rg[i].b > rg[m - 1].b) rg[m - 1].b = rg[i].b; }
            else rg[m++] = rg[i];
        }
        double want = 0;
        for (int i = 0; i < m; i++) want += (rg[i].b - rg[i].a) / 1048576.0;
        total_all += size / 1e9; total_exp += exp_bytes / 1e9; total_core += want / 1024.0;
        if (plan || skel) {
            fprintf(stderr, "  core %.2f GB, routed experts %.2f GB\n", want / 1024.0, exp_bytes / 1e9);
            if (skel) { es_gguf_close(&g); close(fd); free(rg); continue; }
            es_gguf_close(&g); close(fd); unlink(path); free(rg);
            continue;
        }
        if (!core_done) {
            fprintf(stderr, "  downloading %s: %.0f MB of %.0f MB\n", all ? "everything" : "the core", want, size / 1048576.0);
            double done = 0; const double t0 = now();
            for (int i = 0; i < m; i++) if (fetch(url, fd, rg[i].a, rg[i].b, &done, want, t0)) return 1;
            fprintf(stderr, "\n");
            FILE *df = fopen(donef, "w"); if (df) { fprintf(df, "%s\n", all ? "all" : "core"); fclose(df); }
        } else fprintf(stderr, "  core already here\n");
        fprintf(uf, "%d %lld %s\n", part - 1, (long long)size, url);
        free(rg);
        es_gguf_close(&g);
        close(fd);
    }
    fprintf(uf, "%s\n", all ? "complete" : "experts-on-demand");
    fclose(uf);
    if (skel) { printf("skeleton: %s (%.1f GB, every weight zero)\n", path1, total_all); return 0; }
    if (plan) {
        printf("plan: model %.1f GB; download now (core) %.1f GB; routed experts %.1f GB, fetched when first used\n", total_all, total_core, total_exp);
        return 0;
    }
    printf("ready: %s\n", path1);
    printf("  model %.1f GB in total, of which routed experts %.1f GB%s; downloaded now: %.1f GB\n", total_all, total_exp, all ? "" : " (fetched when first used)", total_core);
    printf("  run: ./es_chat -g %s     (ES_DISK_GB=N caps the space experts may take; ES_OFFLINE=1 uses only experts already here)\n", path1);
    return 0;
}
