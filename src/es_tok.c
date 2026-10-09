#include "es_tok.h"

#include <stdlib.h>
#include <string.h>

enum { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER = 4, TT_UNUSED = 5, TT_BYTE = 6 };

/* ---- string -> int hash map (open addressing) ---- */
typedef struct { const char *s; uint32_t len; int32_t val; } hent;
typedef struct { hent *e; uint32_t cap; } hmap;

static uint64_t fnv(const char *s, uint32_t n) {
    uint64_t h = 1469598103934665603ull;
    for (uint32_t i = 0; i < n; i++) { h ^= (uint8_t)s[i]; h *= 1099511628211ull; }
    return h;
}
static void hm_init(hmap *m, uint64_t n) {
    m->cap = 16;
    while (m->cap < n * 2) m->cap <<= 1;
    m->e = calloc(m->cap, sizeof(hent));
}
static void hm_put(hmap *m, const char *s, uint32_t len, int32_t v) {
    uint32_t i = (uint32_t)fnv(s, len) & (m->cap - 1);
    while (m->e[i].s) {
        if (m->e[i].len == len && !memcmp(m->e[i].s, s, len)) return; /* keep first */
        i = (i + 1) & (m->cap - 1);
    }
    m->e[i] = (hent){s, len, v};
}
static int32_t hm_get(const hmap *m, const char *s, uint32_t len) {
    uint32_t i = (uint32_t)fnv(s, len) & (m->cap - 1);
    while (m->e[i].s) {
        if (m->e[i].len == len && !memcmp(m->e[i].s, s, len)) return m->e[i].val;
        i = (i + 1) & (m->cap - 1);
    }
    return -1;
}

struct es_tok {
    uint64_t n;
    char **tok;
    uint32_t *len;
    int32_t *type;
    hmap vocab, merges;
    char **mstr;              /* merge strings "a b" (kept alive for the map) */
    int32_t *special;         /* control + user-defined ids, longest first */
    int n_special;
    int32_t bos, eos, eot;
    int add_bos;
    char b2u[256][4];         /* byte -> UTF-8 of its GPT-2 unicode char */
    uint8_t b2u_len[256];
    int16_t u2b[512];         /* codepoint -> byte, -1 if none */
};

static int utf8_put(uint32_t cp, char *o) {
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 63)); return 2; }
    o[0] = (char)(0xE0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 63)); o[2] = (char)(0x80 | (cp & 63));
    return 3;
}
static int utf8_len(uint8_t c) {
    return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
}
static uint32_t utf8_cp(const uint8_t *s, int n) {
    if (n == 1) return s[0];
    if (n == 2) return ((s[0] & 31u) << 6) | (s[1] & 63u);
    if (n == 3) return ((s[0] & 15u) << 12) | ((s[1] & 63u) << 6) | (s[2] & 63u);
    return ((s[0] & 7u) << 18) | ((s[1] & 63u) << 12) | ((s[2] & 63u) << 6) | (s[3] & 63u);
}

static void build_byte_map(es_tok *t) {
    int n = 0;
    for (int i = 0; i < 512; i++) t->u2b[i] = -1;
    for (int b = 0; b < 256; b++) {
        int keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        uint32_t cp = keep ? (uint32_t)b : 256u + (uint32_t)n++;
        t->b2u_len[b] = (uint8_t)utf8_put(cp, t->b2u[b]);
        t->u2b[cp] = (int16_t)b;
    }
}

int es_tok_n_vocab(const es_tok *t) { return (int)t->n; }
int32_t es_tok_bos(const es_tok *t) { return t->bos; }
int32_t es_tok_eos(const es_tok *t) { return t->eos; }
int es_tok_add_bos(const es_tok *t) { return t->add_bos; }
int32_t es_tok_find(const es_tok *t, const char *s) { return hm_get(&t->vocab, s, (uint32_t)strlen(s)); }
int es_tok_is_eog(const es_tok *t, int32_t id) { return id == t->eos || (t->eot >= 0 && id == t->eot); }

static const es_tok *g_sort_tok;
static int cmp_special(const void *a, const void *b) {
    const es_tok *t = g_sort_tok;
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    if (t->len[x] != t->len[y]) return t->len[x] > t->len[y] ? -1 : 1;
    return x < y ? -1 : x > y;
}

es_tok *es_tok_load(const es_gguf *g, const char **err) {
    const char *model = es_gguf_str(g, "tokenizer.ggml.model");
    if (!model || strcmp(model, "gpt2")) { *err = "only byte-level BPE (tokenizer.ggml.model = gpt2) is supported so far"; return NULL; }
    es_tok *t = calloc(1, sizeof *t);
    uint64_t n_types = 0, n_merges = 0;
    t->tok = es_gguf_strings(g, "tokenizer.ggml.tokens", &t->n, &t->len);
    t->type = es_gguf_ints(g, "tokenizer.ggml.token_type", &n_types);
    uint32_t *mlen = NULL;
    t->mstr = es_gguf_strings(g, "tokenizer.ggml.merges", &n_merges, &mlen);
    if (!t->tok || !t->mstr) { *err = "tokenizer arrays missing"; free(t); return NULL; }
    if (!t->type || n_types != t->n) {
        free(t->type);
        t->type = malloc(t->n * sizeof(int32_t));
        for (uint64_t i = 0; i < t->n; i++) t->type[i] = TT_NORMAL;
    }
    hm_init(&t->vocab, t->n);
    for (uint64_t i = 0; i < t->n; i++) hm_put(&t->vocab, t->tok[i], t->len[i], (int32_t)i);
    hm_init(&t->merges, n_merges);
    for (uint64_t i = 0; i < n_merges; i++) hm_put(&t->merges, t->mstr[i], mlen[i], (int32_t)i);
    free(mlen);
    build_byte_map(t);

    t->bos = (int32_t)es_gguf_int(g, "tokenizer.ggml.bos_token_id", -1);
    t->eos = (int32_t)es_gguf_int(g, "tokenizer.ggml.eos_token_id", -1);
    t->eot = (int32_t)es_gguf_int(g, "tokenizer.ggml.eot_token_id", -1);
    t->add_bos = (int)es_gguf_int(g, "tokenizer.ggml.add_bos_token", 0);
    if (t->eot < 0) {
        static const char *eots[] = {"<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", NULL};
        for (int i = 0; eots[i] && t->eot < 0; i++) t->eot = es_tok_find(t, eots[i]);
    }

    t->special = malloc(t->n * sizeof(int32_t));
    for (uint64_t i = 0; i < t->n; i++)
        if ((t->type[i] == TT_CONTROL || t->type[i] == TT_USER) && t->len[i] > 0)
            t->special[t->n_special++] = (int32_t)i;
    g_sort_tok = t;
    qsort(t->special, (size_t)t->n_special, sizeof(int32_t), cmp_special);
    return t;
}

void es_tok_free(es_tok *t) {
    if (!t) return;
    free(t->tok); free(t->len); free(t->type); free(t->mstr);
    free(t->vocab.e); free(t->merges.e); free(t->special);
    free(t);
}

/* ---- character classes for the GPT-2 pre-tokenizer ---- */
enum { C_SPACE, C_LETTER, C_NUMBER, C_OTHER };
static int cclass(uint32_t cp) {
    if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f') return C_SPACE;
    if (cp < 0x80) {
        if ((cp | 32) >= 'a' && (cp | 32) <= 'z') return C_LETTER;
        if (cp >= '0' && cp <= '9') return C_NUMBER;
        return C_OTHER;
    }
    if (cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000)
        return C_SPACE;
    return C_LETTER;
}

/* BPE-merge one pre-tokenized word (raw bytes) and append ids */
static int bpe_word(const es_tok *t, const uint8_t *w, int wn, int32_t *out, int max) {
    /* symbol buffer: each byte becomes its GPT-2 unicode char */
    int cap = wn * 4 + 1;
    char *buf = malloc((size_t)cap * 2);
    int *st = malloc(sizeof(int) * (size_t)wn), *ln = malloc(sizeof(int) * (size_t)wn);
    int ns = 0, p = 0;
    for (int i = 0; i < wn; i++) {
        st[ns] = p;
        memcpy(buf + p, t->b2u[w[i]], t->b2u_len[w[i]]);
        ln[ns] = t->b2u_len[w[i]];
        p += ln[ns];
        ns++;
    }
    char *pair = buf + cap;
    for (;;) {
        int best = -1, best_rank = 0x7fffffff;
        for (int i = 0; i + 1 < ns; i++) {
            int a = ln[i], b = ln[i + 1];
            if (a + b + 1 > cap) continue;
            memcpy(pair, buf + st[i], (size_t)a);
            pair[a] = ' ';
            memcpy(pair + a + 1, buf + st[i + 1], (size_t)b);
            int r = hm_get(&t->merges, pair, (uint32_t)(a + b + 1));
            if (r >= 0 && r < best_rank) { best_rank = r; best = i; }
        }
        if (best < 0) break;
        /* symbols are contiguous in buf, so merging just extends the left one */
        ln[best] += ln[best + 1];
        memmove(st + best + 1, st + best + 2, sizeof(int) * (size_t)(ns - best - 2));
        memmove(ln + best + 1, ln + best + 2, sizeof(int) * (size_t)(ns - best - 2));
        ns--;
    }
    int n = 0;
    for (int i = 0; i < ns && n < max; i++) {
        int32_t id = hm_get(&t->vocab, buf + st[i], (uint32_t)ln[i]);
        if (id >= 0) { out[n++] = id; continue; }
        /* fall back to single byte symbols */
        for (int j = 0; j < ln[i] && n < max;) {
            int k = utf8_len((uint8_t)buf[st[i] + j]);
            int32_t b = hm_get(&t->vocab, buf + st[i] + j, (uint32_t)k);
            if (b >= 0) out[n++] = b;
            j += k;
        }
    }
    free(buf); free(st); free(ln);
    return n;
}

/* GPT-2 pre-tokenizer, same rules as llama.cpp's unicode_regex_split_custom_gpt2 */
static int encode_raw(const es_tok *t, const uint8_t *s, int n, int32_t *out, int max) {
    /* decode to codepoints with byte offsets */
    uint32_t *cp = malloc(sizeof(uint32_t) * (size_t)(n + 1));
    int *off = malloc(sizeof(int) * (size_t)(n + 1));
    int m = 0;
    for (int i = 0; i < n;) {
        int k = utf8_len(s[i]);
        if (i + k > n) k = 1;
        cp[m] = utf8_cp(s + i, k);
        off[m++] = i;
        i += k;
    }
    off[m] = n;
#define CL(i) ((i) < m ? cclass(cp[i]) : -1)
    int count = 0;
    for (int pos = 0; pos < m && count < max;) {
        int start = pos;
        uint32_t c = cp[pos];
        if (c == '\'' && pos + 1 < m) {
            uint32_t c1 = cp[pos + 1];
            if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') { pos += 2; goto emit; }
            if (pos + 2 < m) {
                uint32_t c2 = cp[pos + 2];
                if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) { pos += 3; goto emit; }
            }
        }
        {
            /* ' ?\p{L}+' | ' ?\p{N}+' | ' ?[^\s\p{L}\p{N}]+' */
            const int sp = c == ' ';
            const int k2 = sp ? CL(pos + 1) : CL(pos);
            if (k2 == C_LETTER || k2 == C_NUMBER || k2 == C_OTHER) {
                pos += sp;
                while (CL(pos) == k2) pos++;
                goto emit;
            }
        }
        {
            int nw = 0;
            while (CL(pos + nw) == C_SPACE) nw++;
            if (nw > 1 && pos + nw < m) { pos += nw - 1; goto emit; }
            if (nw > 0) { pos += nw; goto emit; }
        }
        pos++;
    emit:
        count += bpe_word(t, s + off[start], off[pos] - off[start], out + count, max - count);
    }
#undef CL
    free(cp); free(off);
    return count;
}

int es_tok_encode(const es_tok *t, const char *text, int parse_special, int32_t *out, int max) {
    /* fragments: either a special token id or a raw text span */
    typedef struct { int id; int a, b; } frag;
    int n = (int)strlen(text), cap = 64, nf = 1;
    frag *f = malloc(sizeof(frag) * (size_t)cap);
    f[0] = (frag){-1, 0, n};
    for (int si = 0; si < t->n_special; si++) {
        int32_t id = t->special[si];
        if (!parse_special && t->type[id] == TT_CONTROL) continue;
        const char *pat = t->tok[id];
        int pl = (int)t->len[id];
        for (int fi = 0; fi < nf; fi++) {
            if (f[fi].id >= 0 || f[fi].b - f[fi].a < pl) continue;
            const char *base = text + f[fi].a;
            const char *hit = NULL;
            for (int i = 0; i + pl <= f[fi].b - f[fi].a; i++)
                if (base[i] == pat[0] && !memcmp(base + i, pat, (size_t)pl)) { hit = base + i; break; }
            if (!hit) continue;
            if (nf + 2 > cap) { cap *= 2; f = realloc(f, sizeof(frag) * (size_t)cap); }
            int a = f[fi].a, h = (int)(hit - text), b = f[fi].b;
            memmove(f + fi + 3, f + fi + 1, sizeof(frag) * (size_t)(nf - fi - 1));
            f[fi] = (frag){-1, a, h};
            f[fi + 1] = (frag){id, h, h + pl};
            f[fi + 2] = (frag){-1, h + pl, b};
            nf += 2;
            fi++; /* continue scanning in the remainder (fi+2 next iteration) */
        }
    }
    int count = 0;
    for (int fi = 0; fi < nf && count < max; fi++) {
        if (f[fi].id >= 0) out[count++] = f[fi].id;
        else if (f[fi].b > f[fi].a)
            count += encode_raw(t, (const uint8_t *)text + f[fi].a, f[fi].b - f[fi].a, out + count, max - count);
    }
    free(f);
    return count;
}

int es_tok_piece(const es_tok *t, int32_t id, int show_special, char *buf, int cap) {
    if (id < 0 || (uint64_t)id >= t->n) return 0;
    int ty = t->type[id];
    if (ty == TT_CONTROL || ty == TT_UNUSED) {
        if (!show_special) return 0;
        int n = (int)t->len[id] < cap ? (int)t->len[id] : cap;
        memcpy(buf, t->tok[id], (size_t)n);
        return n;
    }
    if (ty == TT_USER) {
        int n = (int)t->len[id] < cap ? (int)t->len[id] : cap;
        memcpy(buf, t->tok[id], (size_t)n);
        return n;
    }
    const uint8_t *s = (const uint8_t *)t->tok[id];
    int n = 0;
    for (uint32_t i = 0; i < t->len[id] && n < cap;) {
        int k = utf8_len(s[i]);
        uint32_t c = utf8_cp(s + i, k);
        int b = c < 512 ? t->u2b[c] : -1;
        if (b >= 0) buf[n++] = (char)b;
        else for (int j = 0; j < k && n < cap; j++) buf[n++] = (char)s[i + j];
        i += (uint32_t)k;
    }
    return n;
}
