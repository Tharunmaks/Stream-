/* Training examples for es_graft and es_lora: plain text cut into windows, or JSON lines ({"text"}, {"prompt", "response"},
 * {"messages": [...]}) in the model's chat template with only the answer learned. Every val_every-th example is held out.
 * The includer defines die(fmt, arg) and includes es_engine.h, es_json.h and es_graft_anchor.h first. */
#ifndef ES_DATA_H
#define ES_DATA_H

typedef struct {   /* where the examples come from */
    int jsonl; char *text; size_t len, at; int bos, L;
    int32_t *stream; int sn, sat, scap;
} source;

static char *slurp(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) die("cannot open %s", path);
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) die("cannot read %s", path);
    fclose(f); b[n] = 0; *len = (size_t)n;
    return b;
}
/* next example: its ids, and the first position whose NEXT token is learned (want[p] for p >= first) */
static int next_example(source *s, int32_t *ids, int cap, int *first, long *skipped) {
    if (!s->jsonl) {   /* plain text: a stream of tokens cut into windows of L tokens (BOS first when the model wants one) */
        while (s->sn - s->sat < s->L && s->at < s->len) {
            size_t e = s->at + 16384 < s->len ? s->at + 16384 : s->len;   /* tokenize piece by piece, cut at a line end */
            if (e < s->len) { size_t b = e; while (b > s->at + 1024 && s->text[b - 1] != '\n') b--; if (s->text[b - 1] == '\n') e = b; }
            const char save = s->text[e]; s->text[e] = 0;
            if (s->sat > 0) { memmove(s->stream, s->stream + s->sat, sizeof(int32_t) * (size_t)(s->sn - s->sat)); s->sn -= s->sat; s->sat = 0; }
            if (s->sn + (int)(e - s->at) + 16 > s->scap) { s->scap = s->sn + (int)(e - s->at) + 1024; s->stream = realloc(s->stream, sizeof(int32_t) * (size_t)s->scap); }
            const int m = es_engine_tokenize(s->text + s->at, s->stream + s->sn, s->scap - s->sn);
            s->text[e] = save;
            if (m > 0) s->sn += m;
            s->at = e;
        }
        const int avail = s->sn - s->sat;
        if (avail < 2) return 0;
        int n = 0;
        if (s->bos >= 0) ids[n++] = s->bos;
        const int take = avail < s->L ? avail : s->L;
        memcpy(ids + n, s->stream + s->sat, sizeof(int32_t) * (size_t)take);
        s->sat += take; n += take;
        *first = 0;
        return n;
    }
    while (s->at < s->len) {   /* JSON lines */
        char *line = s->text + s->at, *nl = memchr(line, '\n', s->len - s->at);
        const size_t ll = nl ? (size_t)(nl - line) : s->len - s->at;
        s->at += ll + (nl ? 1 : 0);
        if (ll < 2) continue;
        jparser P; jv *v = jparse(&P, line, ll);
        int n = -1;
        if (v && v->t == JOBJ) {
            const char *text = jstr(v, "text", NULL), *pr = jstr(v, "prompt", NULL), *rs = jstr(v, "response", jstr(v, "completion", NULL));
            const jv *msgs = jget(v, "messages");
            if (text) {
                n = 0;
                if (s->bos >= 0) ids[n++] = s->bos;
                const int m = es_engine_tokenize(text, ids + n, cap - n);
                n = m < 0 ? -1 : n + m;
                *first = 0;
            } else if (msgs && msgs->t == JARR && msgs->len >= 2) {
                es_msg mm[64]; int k = 0;
                for (const jv *c = msgs->kid; c && k < 64; c = c->next) { mm[k].role = jstr(c, "role", "user"); mm[k].content = jstr(c, "content", ""); k++; }
                if (k >= 2 && !strcmp(mm[k - 1].role, "assistant")) { int st; n = es_engine_chat_example(mm, k - 1, mm[k - 1].content, ids, cap, &st); *first = st - 1; }
            } else if (pr && rs) {
                es_msg mm[1] = {{"user", pr}}; int st;
                n = es_engine_chat_example(mm, 1, rs, ids, cap, &st); *first = st - 1;
            }
        }
        jfree(&P);
        if (n >= 2 && n <= cap) { if (*first < 0) *first = 0; return n; }
        (*skipped)++;
    }
    return 0;
}

/* every example of the data (then the anchors) as tokens; the same cut every run, so an interrupted read resumes where it stopped */
typedef struct {
    int32_t *pool; size_t pn, pcap; size_t *off; int *len, *first, nex, excap; uint8_t *anc, *val;
    long skipped; double learn, anchor_tok, forward_tok; int ndata, nanc;
} examples;
static void collect(examples *X, const char *data, const char *anchor, int seq, double max_tok, int val_every) {
    memset(X, 0, sizeof *X);
    source src = {0};
    const size_t dl = strlen(data);
    src.jsonl = dl > 6 && !strcmp(data + dl - 6, ".jsonl");
    src.text = slurp(data, &src.len); src.bos = es_engine_bos(); src.L = seq - (src.bos >= 0 ? 1 : 0);
    int32_t *ids = malloc(sizeof(int32_t) * (size_t)(seq + 16));
    source asrc = {0};
    if (strcmp(anchor, "none")) {   /* anchors: general questions and text, read through the model too; there the graft learns to keep the model's own answers */
        if (!strcmp(anchor, "auto")) { asrc.text = strdup(ES_GRAFT_ANCHOR); asrc.len = strlen(asrc.text); asrc.jsonl = 1; }
        else { asrc.text = slurp(anchor, &asrc.len); const size_t al = strlen(anchor); asrc.jsonl = al > 6 && !strcmp(anchor + al - 6, ".jsonl"); }
        asrc.bos = src.bos; asrc.L = src.L;
    }
    for (int pass = 0; pass < 2; pass++) {
        source *sr = pass ? &asrc : &src;
        if (!sr->text) continue;
        for (;;) {
            if (pass == 0 && max_tok > 0 && X->learn >= max_tok) break;
            int fst; long sk = 0; const int n = next_example(sr, ids, seq, &fst, &sk);
            if (pass == 0) X->skipped += sk;
            if (n <= 0) break;
            if (X->pn + (size_t)n > X->pcap) { X->pcap = (X->pn + (size_t)n) * 2; X->pool = realloc(X->pool, sizeof(int32_t) * X->pcap); }
            if (X->nex == X->excap) {
                X->excap = X->excap ? X->excap * 2 : 1024;
                X->off = realloc(X->off, sizeof(size_t) * X->excap); X->len = realloc(X->len, sizeof(int) * X->excap); X->first = realloc(X->first, sizeof(int) * X->excap);
                X->anc = realloc(X->anc, (size_t)X->excap); X->val = realloc(X->val, (size_t)X->excap);
            }
            if (!X->pool || !X->off || !X->len || !X->first || !X->anc || !X->val) die("%s", "out of memory");
            memcpy(X->pool + X->pn, ids, sizeof(int32_t) * (size_t)n);
            const int k = pass ? X->nanc++ : X->ndata++;   /* every val_every-th data example, and every val_every-th anchor, is held out */
            const int e = X->nex++;
            X->off[e] = X->pn; X->len[e] = n; X->first[e] = fst; X->anc[e] = (uint8_t)pass; X->val[e] = val_every > 0 && k % val_every == 0; X->pn += (size_t)n;
            if (pass) X->anchor_tok += n - 1 - fst; else X->learn += n - 1 - fst;
            X->forward_tok += n;
        }
        free(sr->text); free(sr->stream);
    }
    free(ids);
}

#endif
