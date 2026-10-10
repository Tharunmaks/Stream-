/*
 * es_chat: real text generation with the ExpertStream engine.
 *
 *   ./es_chat -m ~/packs/olmoe                       # interactive chat (pack)
 *   ./es_chat -g ~/models/model.gguf                 # same, straight from a GGUF
 *   ./es_chat -m ~/packs/olmoe -p "Write a haiku about phones"
 *   ./es_chat -m ~/packs/olmoe -r -p "Once upon a time" -n 64   # raw text
 */
#define _GNU_SOURCE
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/es_engine.h"

static int QUIET, JSON;
#define STAGE(...) do { if (!QUIET) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while (0)

/* ---------------- JSON-lines protocol (es_serve) ----------------
 * stdin : one prompt per line, "\\n" for newlines, "/reset" clears the chat
 * stdout: one JSON object per line: ready, tokens, prompt, tok, done, info */
static void json_str(const char *s, int n) {
    putchar('"');
    for (int i = 0; i < n; i++) {
        unsigned char ch = (unsigned char)s[i];
        if (ch == '"' || ch == '\\') { putchar('\\'); putchar(ch); }
        else if (ch == '\n') fputs("\\n", stdout);
        else if (ch == '\r') fputs("\\r", stdout);
        else if (ch == '\t') fputs("\\t", stdout);
        else if (ch < 0x20) printf("\\u%04x", ch);
        else putchar(ch);
    }
    putchar('"');
}
static void unescape_line(char *s) {
    char *o = s;
    for (char *p = s; *p; p++) {
        if (*p == '\\' && p[1] == 'n') { *o++ = '\n'; p++; }
        else if (*p == '\\' && p[1] == '\\') { *o++ = '\\'; p++; }
        else *o++ = *p;
    }
    *o = 0;
}

static void on_progress(int done, int n, void *ud) {
    if (done == 0) {   /* token ids are known: show them first */
        const int32_t *ids = ud;
        if (JSON) {
            printf("{\"ev\":\"tokens\",\"n\":%d,\"ids\":[", n);
            for (int i = 0; i < n && i < 64; i++) printf("%s%d", i ? "," : "", ids[i]);
            printf("]}\n");
            fflush(stdout);
        }
        STAGE("\033[1m[2/4] Turning words into numbers\033[0m  %d tokens:", n);
        for (int i = 0; i < n && i < 24; i++) STAGE(" %d", ids[i]);
        STAGE("%s\n", n > 24 ? " ..." : "");
        return;
    }
    STAGE("\r\033[1m[3/4] Reading the prompt\033[0m  %d/%d tokens", done, n);
    if (JSON && (done % 4 == 0 || done == n)) {
        es_turn_stats s;
        es_engine_turn_stats(&s);
        printf("{\"ev\":\"prompt\",\"done\":%d,\"n\":%d,\"flash_mb\":%.0f,\"ram_mb\":%ld}\n", done, n, s.flash_mb, es_rss_mb());
        fflush(stdout);
    }
}

static void usage(void) {
    fprintf(stderr,
            "usage: es_chat (-m PACKDIR | -g FILE.gguf) [-p PROMPT] [-n max_tokens=256] [-r raw]\n"
            "               [-t temperature=0.7 (0 = greedy)] [-k top_k=40] [-P top_p=0.9]\n"
            "               [-c context=2048] [-C cache_mb=1024] [-j threads=4] [-s seed]\n"
            "               [-T routing_trace.txt] [-q quiet] [-J JSON-lines protocol for es_serve]\n"
            "               Lookahead: [-L 1 predict next layer (default on)] [-E 0.15 skip weak experts] [-W 1 warm start (default on)] [-i 4 readers]\n");
    exit(2);
}

int main(int argc, char **argv) {
    es_engine_opts o = {.gguf_fd = -1, .ctx = 2048, .cache_mb = 1024, .threads = 4, .lookahead = 1, .warm = 1};
    const char *prompt = NULL, *dump_ids = NULL, *dump_path = NULL;
    int max_new = 256, raw = 0, top_k = 40, opt;
    float temp = 0.7f, top_p = 0.9f;
    uint64_t seed = 42;
    while ((opt = getopt(argc, argv, "m:g:p:n:rt:k:P:c:C:j:s:T:qI:D:JhE:L:W:i:")) != -1) {
        switch (opt) {
        case 'm': o.pack = optarg; break;
        case 'g': o.gguf = optarg; break;
        case 'p': prompt = optarg; break;
        case 'n': max_new = atoi(optarg); break;
        case 'r': raw = 1; break;
        case 't': temp = (float)atof(optarg); break;
        case 'k': top_k = atoi(optarg); break;
        case 'P': top_p = (float)atof(optarg); break;
        case 'c': o.ctx = atoi(optarg); break;
        case 'C': o.cache_mb = (size_t)atol(optarg); break;
        case 'j': o.threads = atoi(optarg); break;
        case 's': seed = strtoull(optarg, NULL, 10); break;
        case 'T': o.trace_path = optarg; break;
        case 'q': QUIET = 1; break;
        case 'I': dump_ids = optarg; break;   /* testing: comma-separated token ids */
        case 'D': dump_path = optarg; break;  /* testing: write every position's logits */
        case 'J': JSON = 1; QUIET = 1; break;
        case 'E': o.skip_thr = (float)atof(optarg); break;   /* skip experts below this share of the best */
        case 'L': o.lookahead = atoi(optarg); break;         /* 1 = predict + preload next layer */
        case 'W': o.warm = atoi(optarg); break;             /* 1 = preload last run's hot experts */
        case 'i': o.io_threads = atoi(optarg); break;
        default: usage();
        }
    }
    if (!o.pack == !o.gguf) usage();

    char err[512];
    if (es_engine_init(&o, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    es_engine_set_sampling(temp, top_k, top_p, max_new, seed);
    es_engine_info in;
    es_engine_get_info(&in);
    STAGE("\033[1m[1/4] Connecting to the brain\033[0m  %s (%s), %d layers, %d experts, %d per token\n",
          in.name, in.arch, in.layers, in.experts, in.used);
    STAGE("      core %.0f MB loaded in %.2f s | expert cache %d slots (%.0f MB) | context %d tokens (%.0f MB)\n",
          in.core_mb, in.load_s, in.cache_slots, in.cache_mb, in.ctx, in.kv_mb);
    STAGE("      %s | RAM now %ld MB\n", o.pack ? "experts from the pack" : "experts read straight from the GGUF", es_rss_mb());

    if (dump_ids && dump_path) {
        static int32_t ids[1 << 14];
        int n = 0;
        for (const char *s = dump_ids; *s && n < (int)(sizeof ids / sizeof ids[0]);) {
            char *end;
            long id = strtol(s, &end, 10);
            if (end == s) { s++; continue; }
            ids[n++] = (int32_t)id;
            s = end;
        }
        FILE *df = fopen(dump_path, "wb");
        es_engine_dump_logits(ids, n, df);
        fclose(df);
        STAGE("dumped logits for %d positions to %s\n", n, dump_path);
        return 0;
    }
    if (JSON) {
        printf("{\"ev\":\"ready\",\"model\":");
        json_str(in.name, (int)strlen(in.name));
        printf(",\"arch\":\"%s\",\"layers\":%d,\"experts\":%d,\"used\":%d,\"core_mb\":%.0f,\"cache_mb\":%.0f,\"ctx\":%d,\"load_s\":%.2f,\"ram_mb\":%ld}\n",
               in.arch, in.layers, in.experts, in.used, in.core_mb, in.cache_mb, in.ctx, in.load_s, es_rss_mb());
        fflush(stdout);
    }

    static char line[1 << 15], piece[256];
    static int32_t ids[1 << 15];
    for (;;) {
        const char *msg = prompt;
        if (!msg) {
            if (!JSON) fprintf(stderr, "\n\033[1myou>\033[0m ");
            if (!fgets(line, sizeof line, stdin)) break;
            line[strcspn(line, "\n")] = 0;
            if (!line[0]) continue;
            if (!strcmp(line, "/exit") || !strcmp(line, "/quit")) break;
            if (!strcmp(line, "/reset")) {
                es_engine_reset();
                if (JSON) { printf("{\"ev\":\"info\",\"msg\":\"new chat\"}\n{\"ev\":\"done\"}\n"); fflush(stdout); }
                continue;
            }
            if (JSON) unescape_line(line);
            msg = line;
        }
        int nt = es_engine_begin(msg, raw, ids, (int)(sizeof ids / sizeof ids[0]), on_progress, ids);
        if (nt < 0) {
            if (JSON) { printf("{\"ev\":\"info\",\"msg\":\"prompt too long for the context\"}\n{\"ev\":\"done\"}\n"); fflush(stdout); continue; }
            fprintf(stderr, "prompt too long for the context\n");
            break;
        }
        es_turn_stats s;
        es_engine_turn_stats(&s);
        if (JSON && s.ctx_reset) { printf("{\"ev\":\"info\",\"msg\":\"context was full, started a new chat\"}\n"); fflush(stdout); }
        STAGE("  %.1f s (%.2f tok/s) | flash %.0f MB | RAM %ld MB\n\033[1m[4/4] Answering\033[0m\n", s.prompt_s, nt / (s.prompt_s > 0 ? s.prompt_s : 1), s.flash_mb, es_rss_mb());

        int pl;
        while ((pl = es_engine_next(piece, sizeof piece)) >= 0) {
            if (JSON) { printf("{\"ev\":\"tok\",\"t\":"); json_str(piece, pl); printf("}\n"); }
            else fwrite(piece, 1, (size_t)pl, stdout);
            fflush(stdout);
        }
        es_engine_turn_stats(&s);
        double tps = s.gen_tokens / (s.gen_s > 0 ? s.gen_s : 1);
        if (JSON)
            printf("{\"ev\":\"done\",\"tokens\":%d,\"secs\":%.2f,\"tps\":%.2f,\"prompt_secs\":%.2f,\"hit\":%.1f,\"flash_mb\":%.0f,\"wait_s\":%.2f,\"ram_mb\":%ld,\"pf_issued\":%llu,\"pf_used\":%llu,\"skipped\":%llu}\n",
                   s.gen_tokens, s.gen_s, tps, s.prompt_s, s.hit_pct, s.flash_mb, s.wait_s, es_rss_mb(), s.pf_issued, s.pf_used, s.skipped);
        else
            printf("\n");
        fflush(stdout);
        STAGE("\033[2m      %d tokens in %.1f s = %.2f tok/s | experts %.0f%% from RAM, %.0f MB read from flash (%.1f s waiting) | RAM %ld MB\033[0m\n",
              s.gen_tokens, s.gen_s, tps, s.hit_pct, s.flash_mb, s.wait_s, es_rss_mb());
        if (prompt) break;
    }
    return 0;
}
