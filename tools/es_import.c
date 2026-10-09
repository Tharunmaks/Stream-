/*
 * es_import: turn a GGUF Mixture-of-Experts model into an ExpertStream pack.
 *
 *   OUT/manifest.txt        layers, experts, sizes (read by es_run)
 *   OUT/core.gguf           everything that is not a routed expert:
 *                           embeddings, attention, norms, router, shared
 *                           experts, output head, tokenizer + all metadata.
 *                           A valid GGUF file, so standard tools can read it.
 *   OUT/L###/E####.exp      one file per routed expert (gate, up, down),
 *                           bytes copied unchanged from the GGUF.
 *
 * Usage:
 *   es_import -i model.gguf                      # inspect only, write nothing
 *   es_import -o ~/packs/qwen3 model-00001-of-00003.gguf model-00002-... ...
 *   es_import -o OUT -x coding.pack model.gguf   # only the experts listed
 *
 * A pack file (-x) lists experts to keep, one per line: "LAYER EXPERT", or
 * "LAYER *" for a whole layer. Lines starting with # are comments.
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
#include "../src/es_gguf.h"
#include "../src/es_io.h"
#include "es_common.h"

#define MAX_SPLITS 64

static es_gguf G[MAX_SPLITS];
static int NG;

typedef struct { int split; const es_gguf_tensor *t; } tref;

static tref find_tensor(const char *name) {
    for (int s = 0; s < NG; s++) {
        const es_gguf_tensor *t = es_gguf_tensor_find(&G[s], name);
        if (t) return (tref){s, t};
    }
    return (tref){-1, NULL};
}

static int is_expert_tensor(const char *name) {
    return strstr(name, "_exps.") != NULL;
}

static int read_full(int fd, void *dst, size_t n, uint64_t off) {
    uint8_t *d = dst;
    while (n) {
        ssize_t k = pread(fd, d, n, (off_t)off);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return -1;
        d += k; n -= (size_t)k; off += (uint64_t)k;
    }
    return 0;
}

static int write_full(int fd, const void *src, size_t n) {
    const uint8_t *s = src;
    while (n) {
        ssize_t k = write(fd, s, n);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return -1;
        s += k; n -= (size_t)k;
    }
    return 0;
}

static int write_u32(int fd, uint32_t v) { return write_full(fd, &v, 4); }
static int write_u64(int fd, uint64_t v) { return write_full(fd, &v, 8); }
static int write_str(int fd, const char *s) {
    uint64_t n = strlen(s);
    return write_u64(fd, n) || write_full(fd, s, n);
}

static int pad_to(int fd, uint64_t *pos, uint64_t align) {
    static const uint8_t zeros[4096];
    uint64_t target = (*pos + align - 1) / align * align;
    while (*pos < target) {
        size_t k = target - *pos > sizeof zeros ? sizeof zeros : (size_t)(target - *pos);
        if (write_full(fd, zeros, k)) return -1;
        *pos += k;
    }
    return 0;
}

static int copy_range(int dst, int src, uint64_t off, uint64_t n) {
    static uint8_t buf[1 << 22];
    while (n) {
        size_t k = n > sizeof buf ? sizeof buf : (size_t)n;
        if (read_full(src, buf, k, off) || write_full(dst, buf, k)) return -1;
        off += k; n -= k;
    }
    return 0;
}

/* mkdir -p: create every missing parent directory too */
static int mkdir_p(const char *p) {
    char tmp[4096];
    size_t n = strlen(p);
    if (n == 0 || n >= sizeof tmp) { errno = ENAMETOOLONG; return -1; }
    memcpy(tmp, p, n + 1);
    for (char *s = tmp + 1; *s; s++) {
        if (*s != '/') continue;
        *s = 0;
        if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return -1;
        *s = '/';
    }
    return mkdir(tmp, 0755) == 0 || errno == EEXIST ? 0 : -1;
}

/* ---- core.gguf: metadata (minus split.*) + every non-expert tensor ---- */
static int write_core(const char *out) {
    char path[4096];
    snprintf(path, sizeof path, "%s/core.gguf", out);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) { perror(path); return -1; }

    uint64_t n_kv = 0, n_t = 0;
    for (uint64_t i = 0; i < G[0].n_kv; i++)
        if (strncmp(G[0].kv[i].key, "split.", 6)) n_kv++;
    for (int s = 0; s < NG; s++)
        for (uint64_t i = 0; i < G[s].n_tensors; i++)
            if (!is_expert_tensor(G[s].t[i].name)) n_t++;

    const uint64_t align = G[0].alignment;
    uint64_t pos = 0;
    int bad = write_u32(fd, 0x46554747u) || write_u32(fd, 3) || write_u64(fd, n_t) ||
              write_u64(fd, n_kv);
    pos = 24;
    for (uint64_t i = 0; i < G[0].n_kv && !bad; i++) {
        const es_gguf_kv *kv = &G[0].kv[i];
        if (!strncmp(kv->key, "split.", 6)) continue;
        bad = copy_range(fd, G[0].fd, kv->raw_start, kv->raw_end - kv->raw_start);
        pos += kv->raw_end - kv->raw_start;
    }
    /* tensor table with new offsets */
    uint64_t off = 0;
    for (int s = 0; s < NG && !bad; s++)
        for (uint64_t i = 0; i < G[s].n_tensors && !bad; i++) {
            const es_gguf_tensor *t = &G[s].t[i];
            if (is_expert_tensor(t->name)) continue;
            bad = write_str(fd, t->name) || write_u32(fd, t->n_dims);
            pos += 8 + strlen(t->name) + 4;
            for (uint32_t d = 0; d < t->n_dims && !bad; d++) { bad = write_u64(fd, t->ne[d]); pos += 8; }
            bad = bad || write_u32(fd, t->type) || write_u64(fd, off);
            pos += 12;
            off = (off + t->nbytes + align - 1) / align * align;
        }
    bad = bad || pad_to(fd, &pos, align);
    const uint64_t data0 = pos;
    (void)data0;
    for (int s = 0; s < NG && !bad; s++)
        for (uint64_t i = 0; i < G[s].n_tensors && !bad; i++) {
            const es_gguf_tensor *t = &G[s].t[i];
            if (is_expert_tensor(t->name)) continue;
            bad = copy_range(fd, G[s].fd, G[s].data_start + t->offset, t->nbytes);
            pos += t->nbytes;
            /* data0 is aligned, so absolute alignment = offset alignment */
            bad = bad || pad_to(fd, &pos, align);
        }
    if (bad || fsync(fd)) { fprintf(stderr, "writing %s failed: %s\n", path, strerror(errno)); close(fd); return -1; }
    close(fd);
    printf("  core.gguf: %llu tensors, %.1f MB\n", (unsigned long long)n_t, pos / 1e6);
    return 0;
}

/* ---- pack filter ---- */
static unsigned char *g_keep; /* [layer * n_exp + e] */
static int load_pack(const char *path, int n_layer, int n_exp) {
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }
    g_keep = calloc((size_t)n_layer * n_exp, 1);
    char line[256];
    int n = 0;
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        int l, e;
        char star;
        if (sscanf(line, "%d %d", &l, &e) == 2 && l >= 0 && l < n_layer && e >= 0 && e < n_exp) {
            g_keep[l * n_exp + e] = 1; n++;
        } else if (sscanf(line, "%d %c", &l, &star) == 2 && star == '*' && l >= 0 && l < n_layer) {
            for (e = 0; e < n_exp; e++) g_keep[l * n_exp + e] = 1;
            n += n_exp;
        }
    }
    fclose(f);
    printf("  pack %s: %d experts selected\n", path, n);
    return 0;
}

static void usage(void) {
    fprintf(stderr,
            "usage: es_import -i FILE.gguf [more splits...]        inspect only\n"
            "       es_import -o OUTDIR [-x pack.txt] FILE.gguf [more splits...]\n");
    exit(2);
}

int main(int argc, char **argv) {
    const char *out = NULL, *pack = NULL;
    int inspect = 0, opt;
    while ((opt = getopt(argc, argv, "o:x:ih")) != -1) {
        switch (opt) {
        case 'o': out = optarg; break;
        case 'x': pack = optarg; break;
        case 'i': inspect = 1; break;
        default: usage();
        }
    }
    if (optind >= argc || (!out && !inspect)) usage();
    for (int i = optind; i < argc; i++) {
        if (NG == MAX_SPLITS) { fprintf(stderr, "too many files\n"); return 1; }
        const char *err;
        if (es_gguf_open(&G[NG], argv[i], &err)) { fprintf(stderr, "%s: %s\n", argv[i], err); return 1; }
        NG++;
    }
    int64_t want_splits = es_gguf_int(&G[0], "split.count", 1);
    if (want_splits != NG)
        fprintf(stderr, "warning: model says %lld split files, %d given\n", (long long)want_splits, NG);

    const char *arch = es_gguf_str(&G[0], "general.architecture");
    if (!arch) { fprintf(stderr, "no general.architecture\n"); return 1; }
    char key[256];
#define HP(name) (snprintf(key, sizeof key, "%s." name, arch), es_gguf_int(&G[0], key, 0))
    const int n_layer = (int)HP("block_count"), n_exp = (int)HP("expert_count"),
              n_used = (int)HP("expert_used_count"), n_embd = (int)HP("embedding_length");
    const char *name = es_gguf_str(&G[0], "general.name");

    /* sizes */
    double core_b = 0, exp_b = 0, core_p = 0, exp_p = 0;
    for (int s = 0; s < NG; s++)
        for (uint64_t i = 0; i < G[s].n_tensors; i++) {
            const es_gguf_tensor *t = &G[s].t[i];
            double p = (double)t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3];
            if (is_expert_tensor(t->name)) { exp_b += t->nbytes; exp_p += p; }
            else { core_b += t->nbytes; core_p += p; }
            if (!t->nbytes)
                fprintf(stderr, "warning: tensor %s has unknown type %u\n", t->name, t->type);
        }
    printf("model   : %s (%s)\n", name ? name : "?", arch);
    printf("shape   : %d layers, hidden %d, %d experts, %d used per token\n", n_layer, n_embd,
           n_exp, n_used);
    printf("params  : %.2fB total = %.2fB core + %.2fB routed experts\n",
           (core_p + exp_p) / 1e9, core_p / 1e9, exp_p / 1e9);
    printf("size    : %.2f GB = %.2f GB core (stays in RAM) + %.2f GB experts (streamed)\n",
           (core_b + exp_b) / 1e9, core_b / 1e9, exp_b / 1e9);
    if (n_exp > 0) {
        double per_token = exp_b / n_exp * n_used;
        printf("per token: %.2f GB of expert weights (%.2fB active expert params)\n",
               per_token / 1e9, exp_p / n_exp * n_used / 1e9);
    }

    /* per-layer expert tensors */
    int max_file = 0, moe_layers = 0;
    uint64_t max_bytes = 0;
    tref L3[512][3];
    static const char *kind[3] = {"gate", "up", "down"};
    static const uint32_t role[3] = {ES_ROLE_GATE, ES_ROLE_UP, ES_ROLE_DOWN};
    if (n_layer > 512) { fprintf(stderr, "too many layers\n"); return 1; }
    for (int l = 0; l < n_layer; l++) {
        int have = 0;
        uint64_t fb = ES_HEADER_BYTES;
        for (int k = 0; k < 3; k++) {
            snprintf(key, sizeof key, "blk.%d.ffn_%s_exps.weight", l, kind[k]);
            L3[l][k] = find_tensor(key);
            if (L3[l][k].t) {
                have++;
                const es_gguf_tensor *t = L3[l][k].t;
                if (t->n_dims != 3 || (int)t->ne[2] != n_exp) {
                    fprintf(stderr, "%s: unexpected shape\n", key);
                    return 1;
                }
                fb += es_round_up(t->nbytes / t->ne[2], ES_ALIGN);
            }
        }
        if (have == 0) continue;
        if (have != 3) {
            fprintf(stderr, "layer %d: only %d of gate/up/down expert tensors (fused layouts are not supported yet)\n", l, have);
            return 1;
        }
        moe_layers++;
        if (fb > max_bytes) max_bytes = fb;
        if (l == 0 || !max_file) {
            printf("experts : ");
            for (int k = 0; k < 3; k++)
                printf("%s %s [%llu x %llu]%s", kind[k], es_ggml_type_name(L3[l][k].t->type),
                       (unsigned long long)L3[l][k].t->ne[1], (unsigned long long)L3[l][k].t->ne[0],
                       k < 2 ? ", " : "\n");
            max_file = 1;
        }
    }
    printf("layout  : %d MoE layers, %.2f MiB per expert file\n", moe_layers, max_bytes / 1048576.0);
    if (inspect || !n_exp) {
        if (!n_exp) printf("this model has no routed experts; nothing to stream\n");
        return 0;
    }

    if (pack && load_pack(pack, n_layer, n_exp)) return 1;
    if (mkdir_p(out)) { perror(out); return 1; }
    printf("writing %s\n", out);
    if (write_core(out)) return 1;

    uint8_t *buf = es_alloc(max_bytes);
    char path[4096];
    uint64_t t0 = es_now_ns(), written = 0;
    int files = 0;
    for (int l = 0; l < n_layer; l++) {
        if (!L3[l][0].t) continue;
        snprintf(path, sizeof path, "%s/L%03d", out, l);
        if (mkdir_p(path)) { perror(path); return 1; }
        for (int e = 0; e < n_exp; e++) {
            if (g_keep && !g_keep[l * n_exp + e]) continue;
            memset(buf, 0, ES_HEADER_BYTES);
            es_header *h = (es_header *)buf;
            h->magic = ES_MAGIC;
            h->version = ES_VERSION;
            h->header_bytes = ES_HEADER_BYTES;
            h->n_tensors = 3;
            h->layer = (uint32_t)l;
            h->expert = (uint32_t)e;
            h->model_id = es_hash64(name ? name : arch, strlen(name ? name : arch));
            uint64_t off = ES_HEADER_BYTES;
            for (int k = 0; k < 3; k++) {
                const es_gguf_tensor *t = L3[l][k].t;
                const es_gguf *g = &G[L3[l][k].split];
                uint64_t per = t->nbytes / t->ne[2];
                es_tensor_desc *d = &h->t[k];
                d->role = role[k];
                d->qtype = ES_QT_GGML | t->type;
                d->rows = (uint32_t)t->ne[1];
                d->cols = (uint32_t)t->ne[0];
                d->offset = off;
                d->nbytes = per;
                memset(buf + off, 0, es_round_up(per, ES_ALIGN));
                if (read_full(g->fd, buf + off, per, g->data_start + t->offset + per * (uint64_t)e)) {
                    fprintf(stderr, "read error in layer %d expert %d\n", l, e);
                    return 1;
                }
                d->hash = es_hash64(buf + off, per);
                off += es_round_up(per, ES_ALIGN);
            }
            h->file_bytes = off;
            h->header_hash = es_header_hash(h);
            es_expert_path(path, sizeof path, out, (unsigned)l, (unsigned)e);
            int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
            if (fd < 0 || write_full(fd, buf, off) || fsync(fd)) {
                fprintf(stderr, "%s: %s\n", path, strerror(errno));
                return 1;
            }
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            close(fd);
            written += off;
            files++;
        }
        double s = (es_now_ns() - t0) / 1e9;
        printf("  layer %3d done  %6d files  %8.2f GB  %6.0f MB/s\r", l, files, written / 1e9,
               written / 1e6 / (s > 0 ? s : 1));
        fflush(stdout);
    }
    printf("\n");

    snprintf(path, sizeof path, "%s/manifest.txt", out);
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return 1; }
    fprintf(f, "layers %d\nexperts %d\nfile_bytes %llu\n", n_layer, n_exp,
            (unsigned long long)max_bytes);
    fprintf(f, "used %d\narch %s\nname %s\nhidden %d\nmoe_layers %d\npartial %d\n", n_used, arch,
            name ? name : "?", n_embd, moe_layers, g_keep != NULL);
    fclose(f);
    printf("done: %d expert files, %.2f GB, in %.0f s\n", files, written / 1e9,
           (es_now_ns() - t0) / 1e9);
    free(buf);
    for (int s = 0; s < NG; s++) es_gguf_close(&G[s]);
    return 0;
}
