/*
 * Browser build of the ExpertStream engine (Emscripten, runs in a Web Worker).
 * The model is the user's own GGUF file: every read goes to JavaScript,
 * which slices the File with FileReaderSync. Nothing is uploaded anywhere.
 */
#include <emscripten.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/es_engine.h"
#include "../src/es_gguf.h"

EM_JS(int, js_read_at, (void *buf, int n, double off), {
    return Module.readAt(buf, n, off);
});

static ssize_t wasm_pread(int fd, void *buf, size_t n, uint64_t off) {
    (void)fd;
    return js_read_at(buf, (int)n, (double)off);
}

EM_JS(void, js_progress, (int done, int total), {
    Module.onProgress(done, total);
});
static void progress(int done, int total, void *ud) {
    (void)ud;
    js_progress(done, total);
}

static char ERR[512], JSONBUF[1024], PIECE[512];
static int32_t IDS[4096];
static int NIDS;

EMSCRIPTEN_KEEPALIVE int esw_init(double size, int ctx, int cache_mb, int threads) {
    es_gguf_pread = wasm_pread;
    es_engine_opts o = {.gguf_fd = 3, .gguf_size = (uint64_t)size, .ctx = ctx, .cache_mb = (size_t)cache_mb, .threads = threads > 0 ? threads : 1};
    return es_engine_init(&o, ERR, sizeof ERR);
}
EMSCRIPTEN_KEEPALIVE const char *esw_error(void) { return ERR; }

EMSCRIPTEN_KEEPALIVE const char *esw_info(void) {
    es_engine_info in;
    es_engine_get_info(&in);
    snprintf(JSONBUF, sizeof JSONBUF,
             "{\"name\":\"%s\",\"arch\":\"%s\",\"layers\":%d,\"experts\":%d,\"used\":%d,\"ctx\":%d,"
             "\"slots\":%d,\"core_mb\":%.0f,\"cache_mb\":%.0f,\"kv_mb\":%.0f,\"load_s\":%.2f}",
             in.name, in.arch, in.layers, in.experts, in.used, in.ctx, in.cache_slots, in.core_mb, in.cache_mb, in.kv_mb, in.load_s);
    return JSONBUF;
}

EMSCRIPTEN_KEEPALIVE void esw_sampling(float temp, int top_k, float top_p, int max_new) {
    es_engine_set_sampling(temp, top_k, top_p, max_new, 0);
}
EMSCRIPTEN_KEEPALIVE void esw_reset(void) { es_engine_reset(); }

/* returns the prompt token count (-1: too long); ids via esw_ids() */
EMSCRIPTEN_KEEPALIVE int esw_begin(const char *msg, int raw) {
    NIDS = es_engine_begin(msg, raw, IDS, 4096, progress, NULL);
    return NIDS;
}
EMSCRIPTEN_KEEPALIVE const int32_t *esw_ids(void) { return IDS; }

/* next piece as a NUL-terminated string, or NULL when the answer is done */
EMSCRIPTEN_KEEPALIVE const char *esw_next(void) {
    int n = es_engine_next(PIECE, (int)sizeof PIECE - 1);
    if (n < 0) return NULL;
    PIECE[n] = 0;
    return PIECE;
}

EMSCRIPTEN_KEEPALIVE const char *esw_stats(void) {
    es_turn_stats s;
    es_engine_turn_stats(&s);
    snprintf(JSONBUF, sizeof JSONBUF,
             "{\"prompt_tokens\":%d,\"prompt_s\":%.2f,\"tokens\":%d,\"secs\":%.2f,\"tps\":%.2f,"
             "\"hit\":%.1f,\"read_mb\":%.0f,\"read_s\":%.2f,\"ctx_reset\":%d}",
             s.prompt_tokens, s.prompt_s, s.gen_tokens, s.gen_s, s.gen_tokens / (s.gen_s > 0 ? s.gen_s : 1),
             s.hit_pct, s.flash_mb, s.wait_s, s.ctx_reset);
    return JSONBUF;
}
