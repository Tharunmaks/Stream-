#include "es_tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "es_regex.h"

enum { TT_NORMAL = 1, TT_UNKNOWN = 2, TT_CONTROL = 3, TT_USER = 4, TT_UNUSED = 5, TT_BYTE = 6 };
enum { TK_BPE, TK_SPM };

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

#define MAX_RX 6
#define MAX_EOG 16
struct es_tok {
    int kind;                 /* TK_BPE (byte-level BPE, GPT-2 style) or TK_SPM (SentencePiece, Llama/Mistral style) */
    uint64_t n;
    char **tok;
    uint32_t *len;
    int32_t *type;
    float *score;             /* SPM merge scores */
    hmap vocab, merges;
    char **mstr;              /* merge strings "a b" (kept alive for the map) */
    int32_t *special;         /* control + user-defined ids, longest first */
    int n_special;
    int32_t bos, eos, eot, unk;
    int32_t eog[MAX_EOG]; int n_eog;
    int add_bos, add_space_prefix, ignore_merges;
    es_rx *rx[MAX_RX]; int n_rx;   /* pre-tokenizer: applied one after another */
    int32_t byte_id[256];     /* SPM <0xXX> tokens */
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
int es_tok_is_spm(const es_tok *t) { return t->kind == TK_SPM; }
int32_t es_tok_find(const es_tok *t, const char *s) { return hm_get(&t->vocab, s, (uint32_t)strlen(s)); }
int es_tok_is_eog(const es_tok *t, int32_t id) {
    for (int i = 0; i < t->n_eog; i++) if (t->eog[i] == id) return 1;
    return 0;
}
void es_tok_set_eog(es_tok *t, int32_t id, int on) {
    if (id < 0) return;
    for (int i = 0; i < t->n_eog; i++)
        if (t->eog[i] == id) { if (!on) t->eog[i] = t->eog[--t->n_eog]; return; }
    if (on && t->n_eog < MAX_EOG) t->eog[t->n_eog++] = id;
}

static const es_tok *g_sort_tok;
static int cmp_special(const void *a, const void *b) {
    const es_tok *t = g_sort_tok;
    int32_t x = *(const int32_t *)a, y = *(const int32_t *)b;
    if (t->len[x] != t->len[y]) return t->len[x] > t->len[y] ? -1 : 1;
    return x < y ? -1 : x > y;
}

/* ---- pre-tokenizers: the same pattern lists as llama.cpp (src/llama-vocab.cpp), by tokenizer.ggml.pre ---- */
#define RX_GPT2 "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)"
#define RX_CONTR "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
#define RX_LLAMA3 RX_CONTR "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
#define RX_QWEN2 RX_CONTR "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
#define RX_UPPER "((?=[\\p{L}])([^a-z]))"
#define RX_LOWER "((?=[\\p{L}])([^A-Z]))"
typedef struct { const char *names; int ignore_merges; const char *rx[MAX_RX]; } pre_def;
static const pre_def PRE[] = {
    {"|llama3|llama-v3|llama-bpe|falcon3|falcon-h1|pixtral|midm-2.0|lfm2|jina-v5-nano|", 1, {RX_LLAMA3}},
    {"|dbrx|smaug-bpe|", 0, {RX_LLAMA3}},
    {"|chatglm-bpe|", 0, {RX_LLAMA3}},
    {"|glm4|glm5|", 1, {RX_LLAMA3}},
    {"|qwen2|deepseek-r1-qwen|kormo|f2llmv2|stablelm2|hunyuan|solar-open|megrez|grok-2|", 0, {RX_QWEN2}},
    {"|qwen35|", 0, {RX_CONTR "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}},
    {"|gpt-2|phi-2|olmo|mpt|jais|trillion|granite-docling|exaone4|roberta-bpe|jina-v1-en|jina-v2-code|a.x-4.0|mellum|modern-bert|gigachat|", 0, {RX_GPT2}},
    {"|starcoder|refact|command-r|smollm|codeshell|exaone|minerva-7b|mellum2|", 0, {"\\p{N}", RX_GPT2}},
    {"|deepseek-v3|hunyuan-dense|joyai-llm|hy_v4|", 0, {"\\p{N}{1,3}", "[\xe4\xb8\x80-\xe9\xbe\xa5\xe3\x81\x80-\xe3\x82\x9f\xe3\x82\xa0-\xe3\x83\xbf]+",
        "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"}},
    {"|deepseek-llm|", 0, {"[\r\n]", "\\s?[A-Za-z\xc2\xb5\xc3\x80-\xc3\x96\xc3\x98-\xc3\xb6\xc3\xb8-\xc6\xba\xc6\xbc-\xc6\xbf\xc7\x84-\xca\x93\xca\x95-\xca\xaf\xcd\xb0-\xcd\xb3\xcd\xb6\xcd\xb7\xcd\xbb-\xcd\xbd\xcd\xbf\xce\x86\xce\x88-\xce\x8a\xce\x8c\xce\x8e-\xce\xa1\xce\xa3-\xcf\xb5\xcf\xb7-\xd2\x81\xd2\x8a-\xd4\xaf\xd4\xb1-\xd5\x96\xe1\x82\xa0-\xe1\x83\x85\xe1\x8e\xa0-\xe1\x8f\xb5\xe1\x8f\xb8-\xe1\x8f\xbd\xe1\xb2\x90-\xe1\xb2\xba\xe1\xb2\xbd-\xe1\xb2\xbf\xe1\xb4\x80-\xe1\xb4\xab\xe1\xb5\xab-\xe1\xb5\xb7\xe1\xb5\xb9-\xe1\xb6\x9a\xe1\xb8\x80-\xe1\xbc\x95\xe1\xbc\x98-\xe1\xbc\x9d\xe1\xbc\xa0-\xe1\xbd\x85\xe1\xbd\x88-\xe1\xbd\x8d\xe1\xbd\x90-\xe1\xbd\x97\xe1\xbd\x99\xe1\xbd\x9b\xe1\xbd\x9d\xe1\xbd\x9f-\xe1\xbd\xbd\xe1\xbe\x80-\xe1\xbe\xb4\xe1\xbe\xb6-\xe1\xbe\xbc\xe1\xbe\xbe\xe1\xbf\x82-\xe1\xbf\x84\xe1\xbf\x86-\xe1\xbf\x8c\xe1\xbf\x90-\xe1\xbf\x93\xe1\xbf\x96-\xe1\xbf\x9b\xe1\xbf\xa0-\xe1\xbf\xac\xe1\xbf\xb2-\xe1\xbf\xb4\xe1\xbf\xb6-\xe1\xbf\xbc\xe2\x84\x82\xe2\x84\x87\xe2\x84\x8a-\xe2\x84\x93\xe2\x84\x95\xe2\x84\x99-\xe2\x84\x9d\xe2\x84\xa4\xe2\x84\xa6\xe2\x84\xa8\xe2\x84\xaa-\xe2\x84\xad\xe2\x84\xaf-\xe2\x84\xb4\xe2\x84\xb9\xe2\x84\xbc-\xe2\x84\xbf\xe2\x85\x85-\xe2\x85\x89\xe2\x85\x8e\xe2\x86\x83\xe2\x86\x84\xe2\xb0\x80-\xe2\xb1\xbb\xe2\xb1\xbe-\xe2\xb3\xa4\xe2\xb3\xab-\xe2\xb3\xae\xe2\xb3\xb2\xe2\xb3\xb3\xea\x99\x80-\xea\x99\xad\xea\x9a\x80-\xea\x9a\x9b\xea\x9c\xa2-\xea\x9d\xaf\xea\x9d\xb1-\xea\x9e\x87\xea\x9e\x8b-\xea\x9e\x8e\xea\xad\xb0-\xea\xae\xbf\xef\xac\x80-\xef\xac\x86\xef\xac\x93-\xef\xac\x97\xef\xbc\xa1-\xef\xbc\xba\xef\xbd\x81-\xef\xbd\x9a\xf0\x90\x90\x80-\xf0\x90\x91\x8f\xf0\x90\x92\xb0-\xf0\x90\x93\x93\xf0\x90\x93\x98-\xf0\x90\x93\xbb\xf0\x90\xb2\x80-\xf0\x90\xb2\xb2\xf0\x90\xb3\x80-\xf0\x90\xb3\xb2\xf0\x91\xa2\xa0-\xf0\x91\xa3\x9f\xf0\x9e\xa4\x80-\xf0\x9e\xa5\x83]+", "\\s?[!-/:-~\xef\xbc\x81-\xef\xbc\x8f\xef\xbc\x9a-\xef\xbd\x9e\xe2\x80\x98-\xe2\x80\x9f\xe3\x80\x80-\xe3\x80\x82]+", "\\s+$", "[\xe4\xb8\x80-\xe9\xbe\xa5\xe0\xa0\x80-\xe4\xb8\x80\xea\xb0\x80-\xed\x9f\xbf]+", "\\p{N}+"}},
    {"|falcon|", 0, {"[\\p{P}\\$\\+<=>\\^~\\|`]+", RX_GPT2, "[0-9][0-9][0-9]"}},
    {"|deepseek-coder|", 0, {"[\r\n]", "\\s?\\p{L}+", "\\s?\\p{P}+", "[\xe4\xb8\x80-\xe9\xbe\xa5\xe0\xa0\x80-\xe4\xb8\x80\xea\xb0\x80-\xed\x9f\xbf]+", "\\p{N}"}},
    {"|tekken|", 1, {"[^\\r\\n\\p{L}\\p{N}]?" RX_UPPER "*" RX_LOWER "+|[^\\r\\n\\p{L}\\p{N}]?" RX_UPPER "+" RX_LOWER "*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}},
    {"|gpt-4o|llama4|kanana2|talkie|minimax-m2|", 0, {"[^\\r\\n\\p{L}\\p{N}]?" RX_UPPER "*" RX_LOWER "+" RX_CONTR "?|[^\\r\\n\\p{L}\\p{N}]?" RX_UPPER "+" RX_LOWER "*" RX_CONTR "?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}},
    /* Kimi K2: llama.cpp's own splitter (letters exclude Han characters, marks are not letters) written as a pattern */
    {"|kimi-k2|", 0, {"\\p{Han}+|[^\\r\\n\\p{L}\\p{N}]?(?:(?!\\p{Han})\\p{L})+" RX_CONTR "?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}},
    {"|bailingmoe|bailingmoe2|llada-moe|", 0, {"'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"}},
    {"|seed-coder|", 0, {RX_CONTR "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}},
    {"|ufakzeka|", 0, {"[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}},
    {"|minicpm5|", 1, {"\\p{N}{1,3}", RX_CONTR "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}+| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}},
    {"|k2-horizon|", 0, {RX_CONTR "|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}|\\p{M}|\\u200C|\\u200D)+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}},
    {"|laguna|", 0, {"[^\\n]+|[\\n]+", RX_QWEN2}},
    {"|exaone-moe|", 0, {RX_CONTR "|[^\\r\\n\\p{L}\\p{N}]?(?:\\p{L}\\p{M}*(?: \\p{L}\\p{M}*)*)+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]?|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"}},
    {"|jais-2|", 0, {RX_CONTR "|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s{512}(?!\\S)|\\s{256}(?!\\S)|\\s{128}(?!\\S)|\\s{64}(?!\\S)|\\s{32}(?!\\S)|\\s{16}(?!\\S)|\\s{8}(?!\\S)|\\s{4}(?!\\S)|\\s{1,2}(?!\\S)|\\s{1}"}},
    {NULL, 0, {"[\\p{P}\\$\\+<=>\\^~\\|]+", RX_GPT2, "\\p{N}+", "[0-9][0-9][0-9]"}},   /* default */
};
static const pre_def *find_pre(const char *pre) {
    char key[96];
    if (pre && strlen(pre) < sizeof key - 2) {
        snprintf(key, sizeof key, "|%s|", pre);
        for (int i = 0; PRE[i].names; i++) if (strstr(PRE[i].names, key)) return &PRE[i];
    }
    for (int i = 0;; i++) if (!PRE[i].names) return &PRE[i];
}
int es_tok_pre_known(const char *pre) { return find_pre(pre)->names != NULL; }

es_tok *es_tok_load(const es_gguf *g, const char **err) {
    const char *model = es_gguf_str(g, "tokenizer.ggml.model");
    int kind;
    if (model && !strcmp(model, "gpt2")) kind = TK_BPE;
    else if (model && !strcmp(model, "llama")) kind = TK_SPM;
    else { *err = "only byte-level BPE (gpt2) and SentencePiece (llama) tokenizers are supported"; return NULL; }
    es_tok *t = calloc(1, sizeof *t);
    t->kind = kind;
    uint64_t n_types = 0, n_merges = 0, n_scores = 0;
    t->tok = es_gguf_strings(g, "tokenizer.ggml.tokens", &t->n, &t->len);
    t->type = es_gguf_ints(g, "tokenizer.ggml.token_type", &n_types);
    if (!t->tok) { *err = "tokenizer arrays missing"; free(t); return NULL; }
    if (!t->type || n_types != t->n) {
        free(t->type);
        t->type = malloc(t->n * sizeof(int32_t));
        for (uint64_t i = 0; i < t->n; i++) t->type[i] = TT_NORMAL;
    }
    hm_init(&t->vocab, t->n);
    for (uint64_t i = 0; i < t->n; i++) hm_put(&t->vocab, t->tok[i], t->len[i], (int32_t)i);
    if (kind == TK_BPE) {
        uint32_t *mlen = NULL;
        t->mstr = es_gguf_strings(g, "tokenizer.ggml.merges", &n_merges, &mlen);
        if (!t->mstr) { *err = "tokenizer merges missing"; free(t); return NULL; }
        hm_init(&t->merges, n_merges);
        for (uint64_t i = 0; i < n_merges; i++) hm_put(&t->merges, t->mstr[i], mlen[i], (int32_t)i);
        free(mlen);
        build_byte_map(t);
        const pre_def *p = find_pre(es_gguf_str(g, "tokenizer.ggml.pre"));
        t->ignore_merges = p->ignore_merges;
        for (int i = 0; i < MAX_RX && p->rx[i]; i++) {
            const char *e = NULL;
            t->rx[t->n_rx] = es_rx_compile(p->rx[i], &e);
            if (!t->rx[t->n_rx]) { *err = "internal: bad pre-tokenizer pattern"; return NULL; }
            t->n_rx++;
        }
    } else {
        t->score = es_gguf_floats(g, "tokenizer.ggml.scores", &n_scores);
        if (!t->score || n_scores != t->n) { *err = "SentencePiece scores missing"; free(t); return NULL; }
        for (int b = 0; b < 256; b++) {
            char nm[8];
            snprintf(nm, sizeof nm, "<0x%02X>", b);
            t->byte_id[b] = es_tok_find(t, nm);
        }
    }

    t->bos = (int32_t)es_gguf_int(g, "tokenizer.ggml.bos_token_id", -1);
    t->eos = (int32_t)es_gguf_int(g, "tokenizer.ggml.eos_token_id", -1);
    t->eot = (int32_t)es_gguf_int(g, "tokenizer.ggml.eot_token_id", -1);
    t->unk = (int32_t)es_gguf_int(g, "tokenizer.ggml.unknown_token_id", kind == TK_SPM ? 0 : -1);
    t->add_bos = (int)es_gguf_int(g, "tokenizer.ggml.add_bos_token", kind == TK_SPM);
    t->add_space_prefix = (int)es_gguf_int(g, "tokenizer.ggml.add_space_prefix", kind == TK_SPM);
    if (t->eot < 0) {
        static const char *eots[] = {"<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", NULL};
        for (int i = 0; eots[i] && t->eot < 0; i++) t->eot = es_tok_find(t, eots[i]);
    }
    es_tok_set_eog(t, t->eos, 1);
    es_tok_set_eog(t, t->eot, 1);
    es_tok_set_eog(t, (int32_t)es_gguf_int(g, "tokenizer.ggml.eom_token_id", -1), 1);

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
    for (int i = 0; i < t->n_rx; i++) es_rx_free(t->rx[i]);
    free(t->tok); free(t->len); free(t->type); free(t->mstr); free(t->score);
    free(t->vocab.e); free(t->merges.e); free(t->special);
    free(t);
}

/* ---- byte-level BPE ---- */
/* BPE-merge one pre-tokenized word (raw bytes) and append ids */
static int bpe_word(const es_tok *t, const uint8_t *w, int wn, int32_t *out, int max) {
    /* symbol buffer: each byte becomes its GPT-2 unicode char */
    int cap = wn * 4 + 1;
    char *buf = malloc((size_t)cap * 2);
    int *st = malloc(sizeof(int) * (size_t)wn), *ln = malloc(sizeof(int) * (size_t)wn);
    int ns = 0, p = 0;
    for (int i = 0; i < wn; i++) {
        memcpy(buf + p, t->b2u[w[i]], t->b2u_len[w[i]]);
        p += t->b2u_len[w[i]];
    }
    if (t->ignore_merges) {   /* a whole word that is in the vocabulary is one token */
        const int32_t id = hm_get(&t->vocab, buf, (uint32_t)p);
        if (id >= 0) { free(buf); free(st); free(ln); if (max > 0) { out[0] = id; return 1; } return 0; }
    }
    p = 0;
    for (int i = 0; i < wn; i++) {   /* one symbol per UTF-8 character of the mapped text */
        st[ns] = p;
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

static int encode_bpe(const es_tok *t, const uint8_t *s, int n, int32_t *out, int max) {
    /* decode to code points with byte offsets */
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
    /* pieces as end positions; each pattern splits every piece further */
    int cap = m + 8, np = 1;
    int *ends = malloc(sizeof(int) * (size_t)cap), *tmp = malloc(sizeof(int) * (size_t)cap);
    ends[0] = m;
    for (int r = 0; r < t->n_rx && m > 0; r++) {
        int nt = 0, a = 0;
        for (int i = 0; i < np; i++) { nt += es_rx_split(t->rx[r], cp, a, ends[i], tmp + nt, cap - nt); a = ends[i]; }
        int *sw = ends; ends = tmp; tmp = sw; np = nt;
    }
    int count = 0, a = 0;
    for (int i = 0; i < np && m > 0 && count < max; i++) {
        count += bpe_word(t, s + off[a], off[ends[i]] - off[a], out + count, max - count);
        a = ends[i];
    }
    free(ends); free(tmp); free(cp); free(off);
    return count;
}

/* ---- SentencePiece: merge the adjacent pair whose token scores highest, until none is left ---- */
typedef struct { int left, right; float score; int size; } bigram;
typedef struct { bigram *h; int n, cap; } bheap;
static int bg_less(const bigram *a, const bigram *b) {   /* a below b: lower score, or same score and later */
    return a->score < b->score || (a->score == b->score && a->left > b->left);
}
static void bh_push(bheap *q, bigram x) {
    if (q->n == q->cap) { q->cap = q->cap ? 2 * q->cap : 64; q->h = realloc(q->h, sizeof(bigram) * (size_t)q->cap); }
    int i = q->n++;
    while (i > 0) { const int p = (i - 1) / 2; if (!bg_less(&q->h[p], &x)) break; q->h[i] = q->h[p]; i = p; }
    q->h[i] = x;
}
static bigram bh_pop(bheap *q) {
    bigram top = q->h[0], x = q->h[--q->n];
    int i = 0;
    for (;;) {
        int c = 2 * i + 1;
        if (c >= q->n) break;
        if (c + 1 < q->n && bg_less(&q->h[c], &q->h[c + 1])) c++;
        if (!bg_less(&x, &q->h[c])) break;
        q->h[i] = q->h[c]; i = c;
    }
    if (q->n) q->h[i] = x;
    return top;
}
typedef struct { int start, n, prev, next; } sym;
static void spm_try(const es_tok *t, const char *txt, const sym *sy, int l, int r, bheap *q) {
    if (l < 0 || r < 0) return;
    const int32_t id = hm_get(&t->vocab, txt + sy[l].start, (uint32_t)(sy[l].n + sy[r].n));
    if (id < 0) return;
    bh_push(q, (bigram){l, r, t->score[id], sy[l].n + sy[r].n});
}
static int encode_spm(const es_tok *t, const char *raw, int rn, int prefix, int32_t *out, int max) {
    /* spaces become U+2581, with one in front when asked */
    char *txt = malloc((size_t)(rn + 1) * 3 + 4);
    int n = 0;
    if (prefix) { memcpy(txt, "\xe2\x96\x81", 3); n = 3; }
    for (int i = 0; i < rn; i++) {
        if (raw[i] == ' ') { memcpy(txt + n, "\xe2\x96\x81", 3); n += 3; }
        else txt[n++] = raw[i];
    }
    sym *sy = malloc(sizeof(sym) * (size_t)(n + 1));
    int ns = 0;
    for (int o = 0; o < n;) {
        int k = utf8_len((uint8_t)txt[o]);
        if (o + k > n) k = n - o;
        sy[ns] = (sym){o, k, ns - 1, o + k == n ? -1 : ns + 1};
        ns++; o += k;
    }
    bheap q = {0};
    for (int i = 1; i < ns; i++) spm_try(t, txt, sy, i - 1, i, &q);
    while (q.n) {
        const bigram b = bh_pop(&q);
        sym *L = &sy[b.left], *R = &sy[b.right];
        if (L->n == 0 || R->n == 0 || L->n + R->n != b.size) continue;   /* stale */
        L->n += R->n; R->n = 0;
        L->next = R->next;
        if (R->next >= 0) sy[R->next].prev = b.left;
        spm_try(t, txt, sy, L->prev, b.left, &q);
        spm_try(t, txt, sy, b.left, L->next, &q);
    }
    int cnt = 0;
    for (int i = ns ? 0 : -1; i >= 0 && cnt < max; i = sy[i].next) {
        const int32_t id = hm_get(&t->vocab, txt + sy[i].start, (uint32_t)sy[i].n);
        if (id >= 0) { out[cnt++] = id; continue; }
        for (int j = 0; j < sy[i].n && cnt < max; j++) {   /* unknown: its bytes */
            const int32_t b = t->byte_id[(uint8_t)txt[sy[i].start + j]];
            out[cnt++] = b >= 0 ? b : t->unk;
        }
    }
    free(q.h); free(sy); free(txt);
    return cnt;
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
    int count = 0, prev_special = 1;
    for (int fi = 0; fi < nf && count < max; fi++) {
        if (f[fi].id >= 0) { out[count++] = f[fi].id; prev_special = 1; continue; }
        if (f[fi].b <= f[fi].a) continue;
        if (t->kind == TK_SPM) {
            count += encode_spm(t, text + f[fi].a, f[fi].b - f[fi].a, t->add_space_prefix && prev_special, out + count, max - count);
            prev_special = 0;
        } else {
            count += encode_bpe(t, (const uint8_t *)text + f[fi].a, f[fi].b - f[fi].a, out + count, max - count);
        }
    }
    free(f);
    return count;
}

int es_tok_piece(const es_tok *t, int32_t id, int show_special, char *buf, int cap) {
    if (id < 0 || (uint64_t)id >= t->n || cap <= 0) return 0;
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
    if (t->kind == TK_SPM) {
        if (ty == TT_BYTE && t->len[id] == 6) {   /* <0xXX> */
            const char h[3] = {(char)s[3], (char)s[4], 0};
            buf[0] = (char)strtol(h, NULL, 16);
            return 1;
        }
        if (ty == TT_UNKNOWN) { const char *u = "\xe2\x96\x85"; for (; n < 3 && n < cap; n++) buf[n] = u[n]; return n; }
        for (uint32_t i = 0; i < t->len[id] && n < cap;) {
            if (i + 3 <= t->len[id] && s[i] == 0xE2 && s[i + 1] == 0x96 && s[i + 2] == 0x81) { buf[n++] = ' '; i += 3; }
            else buf[n++] = (char)s[i++];
        }
        return n;
    }
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
