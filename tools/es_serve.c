/*
 * es_serve: ExpertStream Instant. One local server that any browser or app can use, with fast answers.
 *
 *   ./es_serve -g ~/models/model.gguf          open http://127.0.0.1:8080, or point any app at http://127.0.0.1:8080/v1
 *   ./es_serve -m ~/packs/olmoe                 an ExpertStream pack
 *   ./es_serve -g model.gguf --lan              other devices on the network too (an API key is created and required)
 *
 * Speaks three protocols on one port:
 *   OpenAI   GET /v1/models, POST /v1/chat/completions, POST /v1/completions   (streaming with SSE)
 *   Ollama   GET /api/tags, /api/version, POST /api/chat, /api/generate, /api/show
 *   web app  GET /, GET /api/status, POST /api/chat (text body), POST /api/reset
 *
 * Why answers come fast:
 *   - prefix reuse: apps resend the whole conversation every turn; the engine keeps what it already read (and parks up to 4 other
 *     conversations in RAM), so only the new message is read before the first word (es_engine_prompt)
 *   - Turbo: guessed words are checked in one batched pass (prompt lookup, in the engine)
 *   - streaming: every piece is sent the moment it exists
 *
 * Safety: listens on 127.0.0.1 unless --lan. Browsers: only local pages, https://tharunmaks.github.io and --cors origins may call it
 * (CORS + Private Network Access); other websites get 403. With --lan, every request from another device needs the API key.
 */
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "../src/es_engine.h"
#include "../src/es_io.h"
#include "es_json.h"

static pthread_mutex_t g_engine = PTHREAD_MUTEX_INITIALIZER;
static char g_web[PATH_MAX], g_model[256], g_ready[2048], g_key[128], g_cors[2048] = "https://tharunmaks.github.io";
static int g_lan, g_ctx = 4096, g_max = 512, g_topk = 40;
static float g_temp = 0.7f, g_topp = 0.9f;
static es_engine_info g_info;
static long long g_file_size;

/* ------------------------------------------------------------------ small growable buffer */
typedef struct { char *p; size_t n, cap; } buf;
static void bput(buf *b, const char *s, size_t n) {
    if (b->n + n + 1 > b->cap) { b->cap = (b->n + n + 1) * 2 + 256; b->p = realloc(b->p, b->cap); }
    memcpy(b->p + b->n, s, n); b->n += n; b->p[b->n] = 0;
}
static void bstr(buf *b, const char *s) { bput(b, s, strlen(s)); }
static void bfmt(buf *b, const char *f, ...) {
    char t[1024]; va_list a; va_start(a, f); int n = vsnprintf(t, sizeof t, f, a); va_end(a);
    if (n > 0) bput(b, t, (size_t)n < sizeof t ? (size_t)n : sizeof t - 1);
}
static void bjson(buf *b, const char *s, size_t n) {          /* a JSON string literal */
    bput(b, "\"", 1);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c == '"' || c == '\\') { char e[2] = {'\\', (char)c}; bput(b, e, 2); }
        else if (c == '\n') bput(b, "\\n", 2); else if (c == '\r') bput(b, "\\r", 2); else if (c == '\t') bput(b, "\\t", 2);
        else if (c < 0x20) { char e[8]; snprintf(e, sizeof e, "\\u%04x", c); bput(b, e, 6); }
        else if (c < 0x80) bput(b, (const char *)&s[i], 1);
        else {   /* a whole, valid UTF-8 character, or U+FFFD: JSON must be valid UTF-8 whatever bytes the model wrote */
            const int k = c >= 0xC2 && c <= 0xDF ? 2 : c >= 0xE0 && c <= 0xEF ? 3 : c >= 0xF0 && c <= 0xF4 ? 4 : 0;
            int ok = k > 0 && i + (size_t)k <= n;
            for (int j = 1; ok && j < k; j++) ok = ((unsigned char)s[i + j] & 0xC0) == 0x80;
            if (ok && k == 3) { const unsigned char c1 = (unsigned char)s[i + 1]; ok = !(c == 0xE0 && c1 < 0xA0) && !(c == 0xED && c1 >= 0xA0); }
            if (ok && k == 4) { const unsigned char c1 = (unsigned char)s[i + 1]; ok = !(c == 0xF0 && c1 < 0x90) && !(c == 0xF4 && c1 >= 0x90); }
            if (ok) { bput(b, (const char *)&s[i], (size_t)k); i += (size_t)k - 1; }
            else bput(b, "\xef\xbf\xbd", 3);
        }
    }
    bput(b, "\"", 1);
}

/* ------------------------------------------------------------------ HTTP */
static int send_all(int fd, const void *p0, size_t n) {
    const char *p = p0;
    while (n) {
        ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
        if (k < 0 && errno == EINTR) continue;
        if (k <= 0) return -1;
        p += k; n -= (size_t)k;
    }
    return 0;
}
typedef struct { int fd, loopback; char method[8], path[512]; char *hdr; char *body; size_t blen; char origin[256]; int cors_ok; } req;

static const char *hget(const req *r, const char *name, char *out, size_t cap) {   /* header value or NULL */
    char pat[64]; snprintf(pat, sizeof pat, "\r\n%s:", name);
    const char *h = strcasestr(r->hdr, pat);
    if (!h) return NULL;
    h += strlen(pat); while (*h == ' ') h++;
    size_t i = 0; while (h[i] && h[i] != '\r' && i + 1 < cap) { out[i] = h[i]; i++; }
    out[i] = 0; return out;
}
static void cors_headers(const req *r, buf *h) {
    if (!r->origin[0] || !r->cors_ok) return;
    bfmt(h, "Access-Control-Allow-Origin: %s\r\nVary: Origin\r\nAccess-Control-Allow-Headers: Authorization, Content-Type\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\nAccess-Control-Allow-Private-Network: true\r\nAccess-Control-Max-Age: 600\r\n", r->origin);
}
static void reply(const req *r, int code, const char *type, const char *body, size_t n) {
    buf h = {0};
    bfmt(&h, "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %zu\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n"
             "Referrer-Policy: no-referrer\r\nConnection: close\r\n", code,
         code == 200 ? "OK" : code == 204 ? "No Content" : code == 400 ? "Bad Request" : code == 401 ? "Unauthorized" : code == 403 ? "Forbidden" : code == 404 ? "Not Found" : "Error", type, n);
    if (!strcmp(type, "text/html; charset=utf-8")) bstr(&h, "X-Frame-Options: DENY\r\n");
    cors_headers(r, &h);
    bstr(&h, "\r\n");
    send_all(r->fd, h.p, h.n); if (n) send_all(r->fd, body, n);
    free(h.p);
}
static void reply_err(const req *r, int code, const char *msg) {     /* OpenAI-style error object (Ollama clients read "error" too) */
    buf b = {0}; bstr(&b, "{\"error\":{\"message\":"); bjson(&b, msg, strlen(msg)); bstr(&b, ",\"type\":\"invalid_request_error\"}}");
    reply(r, code, "application/json", b.p, b.n); free(b.p);
}
static int stream_start(const req *r, const char *type) {
    buf h = {0};
    bfmt(&h, "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n", type);
    cors_headers(r, &h); bstr(&h, "\r\n");
    int rc = send_all(r->fd, h.p, h.n); free(h.p); return rc;
}

/* ------------------------------------------------------------------ access rules */
static int local_name(const char *v) {
    while (*v == ' ') v++;
    return !strncmp(v, "127.0.0.1", 9) || !strncmp(v, "localhost", 9) || !strncmp(v, "[::1]", 5);
}
static int origin_allowed(const char *o) {
    if (!strncmp(o, "http://", 7) && local_name(o + 7)) return 1;
    if (!strncmp(o, "https://", 8) && local_name(o + 8)) return 1;
    if (!strcmp(o, "null")) return 0;
    char list[2048]; snprintf(list, sizeof list, "%s", g_cors);
    for (char *t = strtok(list, ", "); t; t = strtok(NULL, ", ")) if (!strcmp(t, "*") || !strcmp(t, o)) return 1;
    return 0;
}
/* 0 = ok, else HTTP status to refuse with */
static int check_access(req *r) {
    char v[512];
    if (!g_lan) {   /* local only: refuse DNS-rebinding tricks (a foreign name that resolves to 127.0.0.1) */
        if (!hget(r, "host", v, sizeof v) || !local_name(v)) return 403;
    }
    if (hget(r, "origin", v, sizeof v)) {
        snprintf(r->origin, sizeof r->origin, "%s", v);
        r->cors_ok = origin_allowed(v);
        if (!r->cors_ok) return 403;
    }
    if (g_key[0] && !r->loopback) {
        char a[256];
        if (!hget(r, "authorization", a, sizeof a) || strncmp(a, "Bearer ", 7) || strcmp(a + 7, g_key)) return 401;
    }
    return 0;
}

/* ------------------------------------------------------------------ generation shared by every protocol */
typedef struct {
    int (*emit)(void *ud, const char *s, size_t n);     /* nonzero = the client went away */
    void *ud;
    char *stop[8]; int nstop;
    int max_new, top_k; float temp, top_p; uint64_t seed;
} gen_opts;
typedef struct { buf text; int prompt_tokens, cached, gen_tokens, finish_length, client_gone; double prompt_s, gen_s, ttft_s, total_s; } gen_result;

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }
static size_t utf8_ready(const char *s, size_t n) {            /* bytes that end on a whole UTF-8 character */
    size_t i = n, k = 0;
    while (i > 0 && k < 4 && ((unsigned char)s[i - 1] & 0xC0) == 0x80) { i--; k++; }
    if (i == 0) return n;
    unsigned char c = (unsigned char)s[i - 1]; int need = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
    return (size_t)need == k + 1 ? n : i - 1;
}
/* run one request: text is the fully formatted prompt. Returns 0, or -1 if the prompt does not fit */
static int generate(const char *text, gen_opts *o, gen_result *res, void (*progress)(int, int, void *), void *pud, int32_t *ids, int ids_cap) {
    const double t0 = now_s();
    memset(res, 0, sizeof *res);
    es_engine_set_sampling(o->temp, o->top_k, o->top_p, o->max_new, o->seed);
    int cached = 0;
    const int nt = es_engine_prompt(text, &cached, ids, ids_cap, progress, pud);
    if (nt < 0) return -1;
    es_turn_stats st; es_engine_turn_stats(&st);
    res->prompt_tokens = nt; res->cached = cached; res->prompt_s = st.prompt_s;
    size_t hold = 0;                                               /* hold back what could still become a stop string */
    for (int i = 0; i < o->nstop; i++) { size_t l = strlen(o->stop[i]); if (l > hold) hold = l; }
    hold = hold ? hold - 1 : 0;
    size_t sent = 0; char piece[512]; int pl, stopped = 0;
    const double tg = now_s();
    while ((pl = es_engine_next(piece, sizeof piece)) >= 0) {
        if (!res->ttft_s && pl > 0) res->ttft_s = now_s() - t0;
        bput(&res->text, piece, (size_t)pl);
        for (int i = 0; i < o->nstop && !stopped; i++) {
            const size_t sl = strlen(o->stop[i]); if (!sl) continue;
            const size_t from = res->text.n > sl + (size_t)pl ? res->text.n - sl - (size_t)pl : 0;
            char *hit = memmem(res->text.p + from, res->text.n - from, o->stop[i], sl);
            if (hit) { res->text.n = (size_t)(hit - res->text.p); res->text.p[res->text.n] = 0; stopped = 1; }
        }
        size_t upto = stopped ? res->text.n : (res->text.n > hold ? res->text.n - hold : 0);
        upto = utf8_ready(res->text.p, upto);
        if (upto > sent && o->emit) { if (o->emit(o->ud, res->text.p + sent, upto - sent)) { res->client_gone = 1; break; } sent = upto; }
        if (stopped) break;
    }
    if (!res->client_gone && res->text.n > sent && o->emit) o->emit(o->ud, res->text.p + sent, res->text.n - sent);
    es_engine_turn_stats(&st);
    res->gen_tokens = st.gen_tokens; res->gen_s = now_s() - tg; res->total_s = now_s() - t0;
    res->finish_length = !stopped && st.gen_tokens >= o->max_new;
    if (!res->text.p) bput(&res->text, "", 0);
    return 0;
}

/* read sampling options from an OpenAI body or an Ollama "options" object */
static void read_opts(const jv *b, gen_opts *o, int ollama) {
    const jv *src = ollama ? jget(b, "options") : b;
    o->temp = (float)jnum(src, "temperature", g_temp); o->top_p = (float)jnum(src, "top_p", g_topp); o->top_k = (int)jnum(src, "top_k", g_topk);
    o->seed = (uint64_t)jnum(src, "seed", 0);
    o->max_new = (int)(ollama ? jnum(src, "num_predict", g_max) : jnum(b, "max_completion_tokens", jnum(b, "max_tokens", g_max)));
    if (o->max_new <= 0 || o->max_new > g_ctx) o->max_new = g_ctx;
    const jv *st = jget(src, "stop");
    if (st && st->t == JSTR) o->stop[o->nstop++] = st->s;
    else if (st && st->t == JARR) for (jv *c = st->kid; c && o->nstop < 8; c = c->next) if (c->t == JSTR) o->stop[o->nstop++] = c->s;
}
/* text of a message "content": a string, or OpenAI parts [{type:text,text:...}] */
static const char *content_text(const jv *c, buf *tmp) {
    if (!c) return "";
    if (c->t == JSTR) return c->s;
    if (c->t == JARR) {
        tmp->n = 0; bput(tmp, "", 0);
        for (jv *p = c->kid; p; p = p->next) { const char *t = jstr(p, "text", NULL); if (t) { if (tmp->n) bstr(tmp, "\n"); bstr(tmp, t); } }
        return tmp->p;
    }
    return "";
}
/* render messages, dropping the oldest non-system ones until prompt + room for the answer fits in the context */
static char *render_fit(es_msg *m, int n, int *max_new, const char **err) {
    size_t cap = 256; for (int i = 0; i < n; i++) cap += strlen(m[i].content) + 64;
    char *text = malloc(cap);
    int first = 0;
    for (;;) {
        int k = 0; es_msg *mm = malloc(sizeof(es_msg) * (size_t)(n ? n : 1));
        for (int i = 0; i < n; i++) if (i >= first || !strcmp(m[i].role, "system")) mm[k++] = m[i];
        const int len = es_engine_render(mm, k, text, cap); free(mm);
        if (len < 0) { *err = "could not format the conversation"; free(text); return NULL; }
        const int nt = es_engine_count(text);
        const int room = g_ctx - nt - 2;
        if (room >= 16 || first >= n - 1) {
            if (room < 1) { *err = "the last message alone is longer than the context; start the server with a larger -c"; free(text); return NULL; }
            if (*max_new > room) *max_new = room;
            return text;
        }
        first++;
        while (first < n - 1 && !strcmp(m[first].role, "system")) first++;
    }
}

/* ------------------------------------------------------------------ OpenAI */
typedef struct { const req *r; const char *id; long created; int chat; int sent_role; } sse_ctx;
static int sse_emit(void *ud, const char *s, size_t n) {
    sse_ctx *c = ud; buf b = {0};
    bfmt(&b, "data: {\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", c->id, c->chat ? "chat.completion.chunk" : "text_completion", c->created);
    bjson(&b, g_model, strlen(g_model));
    if (c->chat) { bstr(&b, ",\"choices\":[{\"index\":0,\"delta\":{"); if (!c->sent_role) { bstr(&b, "\"role\":\"assistant\","); c->sent_role = 1; } bstr(&b, "\"content\":"); bjson(&b, s, n); bstr(&b, "},\"finish_reason\":null}]}\n\n"); }
    else { bstr(&b, ",\"choices\":[{\"index\":0,\"text\":"); bjson(&b, s, n); bstr(&b, ",\"finish_reason\":null}]}\n\n"); }
    int rc = send_all(c->r->fd, b.p, b.n); free(b.p); return rc;
}
static void usage_json(buf *b, const gen_result *g) {
    bfmt(b, "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d,\"prompt_tokens_details\":{\"cached_tokens\":%d}},"
            "\"x_expertstream\":{\"cached_tokens\":%d,\"read_tokens\":%d,\"prompt_ms\":%.0f,\"ttft_ms\":%.0f,\"tokens_per_s\":%.1f}",
         g->prompt_tokens, g->gen_tokens, g->prompt_tokens + g->gen_tokens, g->cached, g->cached, g->prompt_tokens - g->cached,
         g->prompt_s * 1000, g->ttft_s * 1000, g->gen_tokens / (g->gen_s > 0 ? g->gen_s : 1));
}
static void openai(req *r, int chat) {
    jparser P; jv *b = jparse(&P, r->body, r->blen);
    if (!b || b->t != JOBJ) { reply_err(r, 400, "the body must be a JSON object"); jfree(&P); return; }
    gen_opts o = {0}; read_opts(b, &o, 0);
    char *text = NULL; const char *err = NULL;
    buf tmp = {0};
    if (chat) {
        const jv *ms = jget(b, "messages");
        if (!ms || ms->t != JARR || !ms->len) { reply_err(r, 400, "messages must be a non-empty array"); jfree(&P); return; }
        es_msg *m = calloc((size_t)ms->len, sizeof *m); buf *parts = calloc((size_t)ms->len, sizeof *parts); int n = 0;
        for (jv *x = ms->kid; x; x = x->next, n++) { m[n].role = jstr(x, "role", "user"); m[n].content = content_text(jget(x, "content"), &parts[n]); }
        text = render_fit(m, n, &o.max_new, &err);
        for (int i = 0; i < n; i++) free(parts[i].p);
        free(m); free(parts);
    } else {
        const jv *p = jget(b, "prompt"); const char *pt = p && p->t == JSTR ? p->s : p && p->t == JARR && p->kid && p->kid->t == JSTR ? p->kid->s : NULL;
        if (!pt) { reply_err(r, 400, "prompt must be a string"); jfree(&P); return; }
        text = strdup(pt);
        const int room = g_ctx - es_engine_count(text) - 2; if (o.max_new > room) o.max_new = room;
        if (room < 1) { free(text); text = NULL; err = "the prompt is longer than the context"; }
    }
    if (!text) { reply_err(r, 400, err); jfree(&P); free(tmp.p); return; }
    const int stream = (int)jnum(b, "stream", 0);
    const jv *so = jget(b, "stream_options"); const int inc_usage = so && jnum(so, "include_usage", 0);
    char id[48]; snprintf(id, sizeof id, "%s-%08lx%04x", chat ? "chatcmpl" : "cmpl", (unsigned long)time(NULL), (unsigned)(es_now_ns() & 0xffff));
    sse_ctx sc = {r, id, (long)time(NULL), chat, 0};
    gen_result g;
    pthread_mutex_lock(&g_engine);
    if (stream) { stream_start(r, "text/event-stream"); o.emit = sse_emit; o.ud = &sc; if (chat) sse_emit(&sc, "", 0); }
    const int rc = generate(text, &o, &g, NULL, NULL, NULL, 0);
    pthread_mutex_unlock(&g_engine);
    free(text);
    if (rc < 0) { if (stream) send_all(r->fd, "data: {\"error\":{\"message\":\"prompt too long\"}}\n\n", 47); else reply_err(r, 400, "the prompt does not fit in the context"); jfree(&P); free(g.text.p); return; }
    const char *fin = g.finish_length ? "length" : "stop";
    buf out = {0};
    if (stream) {
        if (!g.client_gone) {
            bfmt(&out, "data: {\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", id, chat ? "chat.completion.chunk" : "text_completion", sc.created); bjson(&out, g_model, strlen(g_model));
            if (chat) bfmt(&out, ",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"%s\"}]", fin); else bfmt(&out, ",\"choices\":[{\"index\":0,\"text\":\"\",\"finish_reason\":\"%s\"}]", fin);
            if (!inc_usage) { bstr(&out, ","); usage_json(&out, &g); }      /* with include_usage the totals come in their own last chunk, as OpenAI does */
            bstr(&out, "}\n\n");
            if (inc_usage) { bfmt(&out, "data: {\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", id, chat ? "chat.completion.chunk" : "text_completion", sc.created); bjson(&out, g_model, strlen(g_model)); bstr(&out, ",\"choices\":[],"); usage_json(&out, &g); bstr(&out, "}\n\n"); }
            bstr(&out, "data: [DONE]\n\n");
            send_all(r->fd, out.p, out.n);
        }
    } else {
        bfmt(&out, "{\"id\":\"%s\",\"object\":\"%s\",\"created\":%ld,\"model\":", id, chat ? "chat.completion" : "text_completion", sc.created); bjson(&out, g_model, strlen(g_model));
        if (chat) { bstr(&out, ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":"); bjson(&out, g.text.p, g.text.n); bfmt(&out, "},\"finish_reason\":\"%s\"}],", fin); }
        else { bstr(&out, ",\"choices\":[{\"index\":0,\"text\":"); bjson(&out, g.text.p, g.text.n); bfmt(&out, ",\"finish_reason\":\"%s\"}],", fin); }
        usage_json(&out, &g); bstr(&out, "}");
        reply(r, 200, "application/json", out.p, out.n);
    }
    free(out.p); free(g.text.p); free(tmp.p); jfree(&P);
}
static void openai_models(req *r) {
    buf b = {0}; bstr(&b, "{\"object\":\"list\",\"data\":[{\"id\":"); bjson(&b, g_model, strlen(g_model));
    bfmt(&b, ",\"object\":\"model\",\"created\":%ld,\"owned_by\":\"expertstream\"}]}", (long)time(NULL));
    reply(r, 200, "application/json", b.p, b.n); free(b.p);
}

/* ------------------------------------------------------------------ Ollama */
static void iso_now(char *out, size_t cap) { time_t t = time(NULL); struct tm g; gmtime_r(&t, &g); strftime(out, cap, "%Y-%m-%dT%H:%M:%SZ", &g); }
typedef struct { const req *r; int chat; } nd_ctx;
static int nd_emit(void *ud, const char *s, size_t n) {
    nd_ctx *c = ud; buf b = {0}; char ts[32]; iso_now(ts, sizeof ts);
    bstr(&b, "{\"model\":"); bjson(&b, g_model, strlen(g_model)); bfmt(&b, ",\"created_at\":\"%s\",", ts);
    if (c->chat) { bstr(&b, "\"message\":{\"role\":\"assistant\",\"content\":"); bjson(&b, s, n); bstr(&b, "}"); } else { bstr(&b, "\"response\":"); bjson(&b, s, n); }
    bstr(&b, ",\"done\":false}\n");
    int rc = send_all(c->r->fd, b.p, b.n); free(b.p); return rc;
}
static void ollama_final(buf *b, const gen_result *g, int chat, const char *full, size_t fn) {
    char ts[32]; iso_now(ts, sizeof ts);
    bstr(b, "{\"model\":"); bjson(b, g_model, strlen(g_model)); bfmt(b, ",\"created_at\":\"%s\",", ts);
    if (chat) { bstr(b, "\"message\":{\"role\":\"assistant\",\"content\":"); bjson(b, full, fn); bstr(b, "},"); } else { bstr(b, "\"response\":"); bjson(b, full, fn); bstr(b, ","); }
    bfmt(b, "\"done\":true,\"done_reason\":\"%s\",\"total_duration\":%.0f,\"load_duration\":0,\"prompt_eval_count\":%d,\"prompt_eval_duration\":%.0f,"
            "\"eval_count\":%d,\"eval_duration\":%.0f,\"x_expertstream\":{\"cached_tokens\":%d,\"ttft_ms\":%.0f}}\n",
         g->finish_length ? "length" : "stop", g->total_s * 1e9, g->prompt_tokens, g->prompt_s * 1e9, g->gen_tokens, g->gen_s * 1e9, g->cached, g->ttft_s * 1000);
}
static void ollama(req *r, int chat) {
    jparser P; jv *b = jparse(&P, r->body, r->blen);
    if (!b || b->t != JOBJ) { reply_err(r, 400, "the body must be a JSON object"); jfree(&P); return; }
    gen_opts o = {0}; read_opts(b, &o, 1);
    const jv *sv = jget(b, "stream"); const int stream = sv ? (int)sv->n : 1;    /* Ollama streams unless told not to */
    char *text = NULL; const char *err = NULL;
    if (chat) {
        const jv *ms = jget(b, "messages");
        if (!ms || ms->t != JARR || !ms->len) { reply_err(r, 400, "messages must be a non-empty array"); jfree(&P); return; }
        es_msg *m = calloc((size_t)ms->len, sizeof *m); int n = 0;
        for (jv *x = ms->kid; x; x = x->next, n++) { m[n].role = jstr(x, "role", "user"); m[n].content = jstr(x, "content", ""); }
        text = render_fit(m, n, &o.max_new, &err); free(m);
    } else {
        const char *pr = jstr(b, "prompt", NULL), *sys = jstr(b, "system", NULL);
        if (!pr) { reply_err(r, 400, "prompt must be a string"); jfree(&P); return; }
        if (jnum(b, "raw", 0)) { text = strdup(pr); const int room = g_ctx - es_engine_count(text) - 2; if (o.max_new > room) o.max_new = room; if (room < 1) { free(text); text = NULL; err = "the prompt is longer than the context"; } }
        else { es_msg m[2]; int n = 0; if (sys) m[n++] = (es_msg){"system", sys}; m[n++] = (es_msg){"user", pr}; text = render_fit(m, n, &o.max_new, &err); }
    }
    if (!text) { reply_err(r, 400, err); jfree(&P); return; }
    nd_ctx nc = {r, chat}; gen_result g;
    pthread_mutex_lock(&g_engine);
    if (stream) { stream_start(r, "application/x-ndjson"); o.emit = nd_emit; o.ud = &nc; }
    const int rc = generate(text, &o, &g, NULL, NULL, NULL, 0);
    pthread_mutex_unlock(&g_engine);
    free(text);
    if (rc < 0) { if (stream) send_all(r->fd, "{\"error\":\"prompt too long\"}\n", 28); else reply_err(r, 400, "the prompt does not fit in the context"); jfree(&P); free(g.text.p); return; }
    buf out = {0};
    if (stream) { if (!g.client_gone) { ollama_final(&out, &g, chat, "", 0); send_all(r->fd, out.p, out.n); } }
    else { ollama_final(&out, &g, chat, g.text.p, g.text.n); reply(r, 200, "application/json", out.p, out.n); }
    free(out.p); free(g.text.p); jfree(&P);
}
static void ollama_tags(req *r) {
    buf b = {0}; char ts[32]; iso_now(ts, sizeof ts);
    bstr(&b, "{\"models\":[{\"name\":"); bjson(&b, g_model, strlen(g_model)); bstr(&b, ",\"model\":"); bjson(&b, g_model, strlen(g_model));
    bfmt(&b, ",\"modified_at\":\"%s\",\"size\":%lld,\"digest\":\"\",\"details\":{\"format\":\"gguf\",\"family\":\"%s\",\"families\":[\"%s\"],\"parameter_size\":\"\",\"quantization_level\":\"\"}}]}",
         ts, g_file_size, g_info.arch, g_info.arch);
    reply(r, 200, "application/json", b.p, b.n); free(b.p);
}
static void ollama_show(req *r) {
    buf b = {0};
    bfmt(&b, "{\"modelfile\":\"\",\"parameters\":\"\",\"template\":\"\",\"details\":{\"format\":\"gguf\",\"family\":\"%s\"},\"model_info\":{\"general.architecture\":\"%s\",\"%s.context_length\":%d}}",
         g_info.arch, g_info.arch, g_info.arch, g_ctx);
    reply(r, 200, "application/json", b.p, b.n); free(b.p);
}

/* ------------------------------------------------------------------ the bundled web app (its own conversation, kept server-side) */
typedef struct { char *role, *content; } wmsg;
static wmsg *W; static int WN, WCAP;
static void web_add(const char *role, const char *content, size_t n) {
    if (WN == WCAP) { WCAP = WCAP ? WCAP * 2 : 16; W = realloc(W, sizeof *W * (size_t)WCAP); }
    W[WN].role = strdup(role); W[WN].content = strndup(content, n); WN++;
}
static void web_clear(void) { for (int i = 0; i < WN; i++) { free(W[i].role); free(W[i].content); } WN = 0; }
static void chunk(int fd, const char *s, size_t n) { char h[32]; int hl = snprintf(h, sizeof h, "%zx\r\n", n); send_all(fd, h, (size_t)hl); send_all(fd, s, n); send_all(fd, "\r\n", 2); }
typedef struct { const req *r; const int32_t *ids; } wp_ctx;
static void web_progress(int done, int n, void *ud) {
    wp_ctx *c = ud; buf b = {0};
    if (done == 0) { bfmt(&b, "{\"ev\":\"tokens\",\"n\":%d,\"ids\":[", n); for (int i = 0; i < n && i < 64; i++) bfmt(&b, "%s%d", i ? "," : "", c->ids[i]); bstr(&b, "]}\n"); }
    else if (done % 4 == 0 || done == n) { es_turn_stats s; es_engine_turn_stats(&s); bfmt(&b, "{\"ev\":\"prompt\",\"done\":%d,\"n\":%d,\"flash_mb\":%.0f,\"ram_mb\":%ld}\n", done, n, s.flash_mb, es_rss_mb()); }
    if (b.n) chunk(c->r->fd, b.p, b.n);
    free(b.p);
}
static int web_emit(void *ud, const char *s, size_t n) {
    const req *r = ud; buf b = {0}; bstr(&b, "{\"ev\":\"tok\",\"t\":"); bjson(&b, s, n); bstr(&b, "}\n");
    chunk(r->fd, b.p, b.n); free(b.p); return 0;
}
static void web_chat(req *r) {
    if (!r->blen) { reply(r, 400, "text/plain", "empty prompt\n", 13); return; }
    pthread_mutex_lock(&g_engine);
    { buf hh = {0}; bstr(&hh, "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nCache-Control: no-store\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n");
      cors_headers(r, &hh); bstr(&hh, "\r\n"); send_all(r->fd, hh.p, hh.n); free(hh.p); }
    web_add("user", r->body, r->blen);
    gen_opts o = {.emit = web_emit, .ud = r, .max_new = g_max, .top_k = g_topk, .temp = g_temp, .top_p = g_topp};
    es_msg *m = malloc(sizeof(es_msg) * (size_t)WN); for (int i = 0; i < WN; i++) m[i] = (es_msg){W[i].role, W[i].content};
    const char *err = NULL; char *text = render_fit(m, WN, &o.max_new, &err); free(m);
    static int32_t ids[64]; wp_ctx pc = {r, ids}; gen_result g; memset(&g, 0, sizeof g); int rc = -1;
    if (text) rc = generate(text, &o, &g, web_progress, &pc, ids, 64);
    free(text);
    buf b = {0};
    if (rc < 0) { WN--; free(W[WN].role); free(W[WN].content); bstr(&b, "{\"ev\":\"info\",\"msg\":\"message too long for the context\"}\n{\"ev\":\"done\"}\n"); }
    else {
        web_add("assistant", g.text.p, g.text.n);
        es_turn_stats s; es_engine_turn_stats(&s);
        bfmt(&b, "{\"ev\":\"done\",\"tokens\":%d,\"secs\":%.2f,\"tps\":%.2f,\"prompt_secs\":%.2f,\"cached\":%d,\"read\":%d,\"ttft\":%.3f,\"hit\":%.1f,\"flash_mb\":%.0f,\"wait_s\":%.2f,\"ram_mb\":%ld,"
                 "\"spec_steps\":%llu,\"spec_ok\":%llu,\"fetched\":%llu,\"net_mb\":%.0f}\n", g.gen_tokens, g.gen_s, g.gen_tokens / (g.gen_s > 0 ? g.gen_s : 1), g.prompt_s, g.cached, g.prompt_tokens - g.cached, g.ttft_s,
             s.hit_pct, s.flash_mb, s.wait_s, es_rss_mb(), s.spec_steps, s.spec_accepted, s.fetched, s.net_mb);
    }
    chunk(r->fd, b.p, b.n); send_all(r->fd, "0\r\n\r\n", 5);
    free(b.p); free(g.text.p);
    pthread_mutex_unlock(&g_engine);
}
static void web_reset(req *r) {
    pthread_mutex_lock(&g_engine); web_clear(); pthread_mutex_unlock(&g_engine);
    const char *s = "{\"ev\":\"info\",\"msg\":\"new chat\"}\n{\"ev\":\"done\"}\n";
    reply(r, 200, "application/x-ndjson", s, strlen(s));
}
static void serve_file(req *r, const char *name, const char *type) {
    char path[PATH_MAX + 64]; snprintf(path, sizeof path, "%s/%s", g_web, name);
    FILE *f = fopen(path, "rb");
    if (!f) { reply(r, 404, "text/plain", "not found\n", 10); return; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1); size_t got = fread(b, 1, (size_t)n, f); fclose(f);
    reply(r, 200, type, b, got); free(b);
}

/* ------------------------------------------------------------------ connection */
static void *client(void *arg) {
    int fd = (int)(intptr_t)arg;
    struct sockaddr_storage pa; socklen_t pl = sizeof pa; req r = {.fd = fd};
    if (!getpeername(fd, (struct sockaddr *)&pa, &pl)) {
        if (pa.ss_family == AF_INET) r.loopback = (ntohl(((struct sockaddr_in *)&pa)->sin_addr.s_addr) >> 24) == 127;
        else if (pa.ss_family == AF_INET6) { const struct in6_addr *a = &((struct sockaddr_in6 *)&pa)->sin6_addr; r.loopback = IN6_IS_ADDR_LOOPBACK(a) || (IN6_IS_ADDR_V4MAPPED(a) && a->s6_addr[12] == 127); }
    }
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    struct timeval tv = {30, 0}; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    enum { HDR_MAX = 1 << 16, BODY_MAX = 16 << 20 };
    char *b = malloc(HDR_MAX + 1); size_t n = 0; char *end = NULL;
    while (n < HDR_MAX) {
        ssize_t k = recv(fd, b + n, HDR_MAX - n, 0);
        if (k <= 0) break;
        n += (size_t)k; b[n] = 0;
        if ((end = strstr(b, "\r\n\r\n"))) break;
    }
    if (!end) { close(fd); free(b); return NULL; }
    const size_t hl = (size_t)(end - b) + 4;
    r.hdr = strndup(b, hl - 2);   /* keeps the last "\r\n" so every header starts with "\r\n" */
    sscanf(b, "%7s %511s", r.method, r.path);
    char *q = strchr(r.path, '?'); if (q) *q = 0;
    char v[64]; long clen = hget(&r, "content-length", v, sizeof v) ? atol(v) : 0;
    if (clen < 0 || clen > BODY_MAX) { reply(&r, 413, "text/plain", "too large\n", 10); goto out; }
    r.body = malloc((size_t)clen + 1); r.blen = n - hl < (size_t)clen ? n - hl : (size_t)clen; memcpy(r.body, b + hl, r.blen);
    while (r.blen < (size_t)clen) { ssize_t k = recv(fd, r.body + r.blen, (size_t)clen - r.blen, 0); if (k <= 0) break; r.blen += (size_t)k; }
    r.body[r.blen] = 0;
    const int deny = check_access(&r);
    if (!strcmp(r.method, "OPTIONS")) { reply(&r, deny ? 403 : 204, "text/plain", "", 0); goto out; }
    if (deny == 401) { reply_err(&r, 401, "missing or wrong API key (Authorization: Bearer <key>)"); goto out; }
    if (deny) { reply(&r, 403, "text/plain", "forbidden: this origin may not use the local model (start es_serve with --cors ORIGIN)\n", 86); goto out; }
    const int get = !strcmp(r.method, "GET"), post = !strcmp(r.method, "POST");
    char ct[128]; const int hasct = hget(&r, "content-type", ct, sizeof ct) != NULL;
    /* the web app sends text/plain (so a message that starts with '{' stays a message); API clients send JSON, some without a type */
    const int json = hasct ? strstr(ct, "json") != NULL || (!strstr(ct, "text/plain") && r.blen && r.body[0] == '{') : (r.blen && r.body[0] == '{');
    if (get && (!strcmp(r.path, "/") || !strcmp(r.path, "/index.html"))) serve_file(&r, "index.html", "text/html; charset=utf-8");
    else if (get && (!strcmp(r.path, "/health") || !strcmp(r.path, "/api/status"))) reply(&r, 200, "application/json", g_ready, strlen(g_ready));
    else if (get && (!strcmp(r.path, "/v1/models") || !strcmp(r.path, "/models"))) openai_models(&r);
    else if (post && (!strcmp(r.path, "/v1/chat/completions") || !strcmp(r.path, "/chat/completions"))) openai(&r, 1);
    else if (post && (!strcmp(r.path, "/v1/completions") || !strcmp(r.path, "/completions"))) openai(&r, 0);
    else if (post && !strcmp(r.path, "/api/chat")) { if (json) ollama(&r, 1); else web_chat(&r); }
    else if (post && !strcmp(r.path, "/api/generate")) ollama(&r, 0);
    else if (post && !strcmp(r.path, "/api/reset")) web_reset(&r);
    else if (get && !strcmp(r.path, "/api/tags")) ollama_tags(&r);
    else if (post && !strcmp(r.path, "/api/show")) ollama_show(&r);
    else if (get && !strcmp(r.path, "/api/version")) reply(&r, 200, "application/json", "{\"version\":\"0.6.0\"}", 19);
    else if (get && !strcmp(r.path, "/api/ps")) ollama_tags(&r);
    else reply(&r, 404, "text/plain", "not found\n", 10);
out:
    close(fd); free(b); free(r.hdr); free(r.body);
    return NULL;
}

static void make_key(void) {
    const char *home = getenv("HOME"); char dir[PATH_MAX], path[PATH_MAX + 16];
    snprintf(dir, sizeof dir, "%s/.expertstream", home ? home : "."); mkdir(dir, 0700);
    snprintf(path, sizeof path, "%s/key", dir);
    FILE *f = fopen(path, "r");
    if (f) { if (fgets(g_key, sizeof g_key, f)) g_key[strcspn(g_key, "\r\n")] = 0; fclose(f); if (strlen(g_key) >= 16) return; }
    unsigned char rnd[24]; int rf = open("/dev/urandom", O_RDONLY); if (rf < 0 || read(rf, rnd, sizeof rnd) != (ssize_t)sizeof rnd) { fprintf(stderr, "no random source\n"); exit(1); } close(rf);
    strcpy(g_key, "es-"); for (int i = 0; i < 24; i++) snprintf(g_key + 3 + 2 * i, 3, "%02x", rnd[i]);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600); if (fd >= 0) { if (write(fd, g_key, strlen(g_key)) < 0) {} close(fd); }
}

static void usage(void) {
    fprintf(stderr,
            "usage: es_serve (-m PACKDIR | -g FILE.gguf) [-p port=8080] [-w webdir] [--lan] [--key KEY] [--cors ORIGIN[,ORIGIN...]|*]\n"
            "                [-c context=4096] [-C expert_cache_mb=auto: 60%% of free RAM] [-j threads=fast cores] [-n max_tokens=512] [-t temperature=0.7]\n"
            "                [-S Turbo draft length, 0 = off] [-E skip] [-L lookahead] [-W warm] [-i readers] [--graft NAME.esgraft] [--adapter NAME.esadapt] [--cache-route 0.2]\n"
            "  OpenAI base URL http://127.0.0.1:PORT/v1 (any API key), Ollama URL http://127.0.0.1:PORT, web app http://127.0.0.1:PORT\n"
            "  ES_PREFIX_CACHE=0 turns prefix reuse off, ES_KV_SLOTS=N parked conversations (4), ES_KV_SLOT_MB budget (1024)\n");
    exit(2);
}

int main(int argc, char **argv) {
    es_engine_opts o = {.gguf_fd = -1, .ctx = 4096, .cache_mb = 0, .lookahead = 1, .warm = 1};
    int port = 8080;
    const char *graft = NULL, *adapter = NULL;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i]; const char *nx = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--")) continue;                         /* old style: es_chat options after -- */
        else if (!strcmp(a, "--lan")) g_lan = 1;
        else if (!strcmp(a, "--key") && nx) { snprintf(g_key, sizeof g_key, "%s", nx); i++; }
        else if (!strcmp(a, "--graft") && nx) { graft = nx; i++; }       /* trained with es_graft: runs on top of the frozen model */
        else if (!strcmp(a, "--adapter") && nx) { adapter = nx; i++; }
        else if (!strcmp(a, "--cache-route") && nx) { o.cache_route = (float)atof(nx); i++; }   /* prefer experts already in RAM */   /* trained with es_graft zo: inside every layer */
        else if (!strcmp(a, "--cors") && nx) { snprintf(g_cors + strlen(g_cors), sizeof g_cors - strlen(g_cors), ",%s", nx); i++; }
        else if (a[0] == '-' && a[1] && !a[2] && nx) {
            i++;
            switch (a[1]) {
            case 'm': o.pack = nx; break;
            case 'g': o.gguf = nx; break;
            case 'p': port = atoi(nx); break;
            case 'w': snprintf(g_web, sizeof g_web, "%s", nx); break;
            case 'c': o.ctx = atoi(nx); break;
            case 'C': o.cache_mb = (size_t)atol(nx); break;
            case 'j': o.threads = atoi(nx); break;
            case 'n': g_max = atoi(nx); break;
            case 't': g_temp = (float)atof(nx); break;
            case 'k': g_topk = atoi(nx); break;
            case 'P': g_topp = (float)atof(nx); break;
            case 'S': o.spec = atoi(nx); if (o.spec == 0) o.spec = -2; break;
            case 'E': o.skip_thr = (float)atof(nx); break;
            case 'L': o.lookahead = atoi(nx); break;
            case 'W': o.warm = atoi(nx); break;
            case 'i': o.io_threads = atoi(nx); break;
            default: usage();
            }
        } else usage();
    }
    if (!o.pack == !o.gguf) usage();
    if (g_lan && !g_key[0]) make_key();
    if (!g_web[0]) {   /* web/ next to this binary */
        char self[PATH_MAX]; ssize_t sl = readlink("/proc/self/exe", self, sizeof self - 1);
        if (sl > 0) { self[sl] = 0; *strrchr(self, '/') = 0; snprintf(g_web, sizeof g_web, "%s/web", self); }
    }
    fprintf(stderr, "Loading %s ...\n", o.pack ? o.pack : o.gguf);
    char err[512];
    if (es_engine_init(&o, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    if (adapter && es_engine_adapter_load(adapter, NULL, NULL, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    if (graft && es_engine_set_graft(graft, err, sizeof err)) { fprintf(stderr, "%s\n", err); return 1; }
    es_engine_get_info(&g_info); g_ctx = g_info.ctx;
    snprintf(g_model, sizeof g_model, "%s%s%s", g_info.name, adapter ? " + adapter" : "", graft ? " + graft" : "");
    { struct stat sb; g_file_size = o.gguf && !stat(o.gguf, &sb) ? (long long)sb.st_size : 0; }
    { buf b = {0}; bstr(&b, "{\"ev\":\"ready\",\"model\":"); bjson(&b, g_model, strlen(g_model));
      bfmt(&b, ",\"arch\":\"%s\",\"layers\":%d,\"experts\":%d,\"used\":%d,\"core_mb\":%.0f,\"cache_mb\":%.0f,\"ctx\":%d,\"load_s\":%.2f,\"ram_mb\":%ld,\"api\":[\"openai\",\"ollama\"],\"instant\":true}",
           g_info.arch, g_info.layers, g_info.experts, g_info.used, g_info.core_mb, g_info.cache_mb, g_info.ctx, g_info.load_s, es_rss_mb());
      snprintf(g_ready, sizeof g_ready, "%s", b.p); free(b.p); }
    signal(SIGPIPE, SIG_IGN);
    int s = socket(AF_INET, SOCK_STREAM, 0), one = 1;
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons((uint16_t)port)};
    a.sin_addr.s_addr = htonl(g_lan ? INADDR_ANY : INADDR_LOOPBACK);
    if (bind(s, (struct sockaddr *)&a, sizeof a) || listen(s, 64)) { fprintf(stderr, "cannot listen on port %d: %s\n", port, strerror(errno)); return 1; }
    fprintf(stderr, "ExpertStream Instant is ready (%s, %.1f s to load)\n"
                    "  web app        http://127.0.0.1:%d\n"
                    "  OpenAI apps    base URL http://127.0.0.1:%d/v1   model \"%s\"   API key: %s\n"
                    "  Ollama apps    http://127.0.0.1:%d\n",
            g_model, g_info.load_s, port, port, g_model, g_key[0] ? "only needed from other devices (below)" : "anything", port);
    if (g_lan) fprintf(stderr, "  other devices  http://<this device's IP>:%d   with  Authorization: Bearer %s   (saved in ~/.expertstream/key)\n", port, g_key);
    fprintf(stderr, "  websites allowed to call it: local pages, %s\n", g_cors);
    for (;;) {
        int c = accept(s, NULL, NULL);
        if (c < 0) { if (errno == EINTR) continue; perror("accept"); break; }
        pthread_t th;
        if (pthread_create(&th, NULL, client, (void *)(intptr_t)c) == 0) pthread_detach(th); else close(c);
    }
    return 0;
}
