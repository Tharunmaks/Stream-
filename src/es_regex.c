#include "es_regex.h"

#include <stdlib.h>
#include <string.h>

#include "es_unicode.h"

/* ---------------- code point properties ---------------- */
int es_uni_flags(uint32_t cp) {
    if (cp > 0x10FFFF) return 0;
    int lo = 0, hi = (int)(sizeof ES_UNI_RANGES / sizeof ES_UNI_RANGES[0]) - 1;
    while (lo < hi) {   /* last range whose start <= cp */
        const int mid = (lo + hi + 1) / 2;
        if ((ES_UNI_RANGES[mid] >> 8) <= cp) lo = mid; else hi = mid - 1;
    }
    return (int)(ES_UNI_RANGES[lo] & 0xFF);
}
int es_uni_is_space(uint32_t cp) {
    return (cp >= 9 && cp <= 13) || cp == ' ' || cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}
int es_uni_is_han(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0x2CEB0 && cp <= 0x2EBEF) || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}
/* the category \p{...} sees (llama.cpp: fixed ASCII classes; non-ASCII white space has no category) */
static int pcat(uint32_t cp) {
    if (cp < 128) {
        if (cp >= '0' && cp <= '9') return ESU_NUMBER;
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return ESU_LETTER;
        if (strchr("!\"#%&'()*,-./:;?@[\\]_{}", (int)cp) && cp) return ESU_PUNCT;
        if (strchr("$+<=>^`|~", (int)cp) && cp) return ESU_SYMBOL;
        return 0;
    }
    if (es_uni_is_space(cp)) return 0;
    return es_uni_flags(cp) & (ESU_NUMBER | ESU_LETTER | ESU_PUNCT | ESU_MARK | ESU_SYMBOL);   /* others match no \p{} class */
}
static int is_s(uint32_t cp) { return cp < 128 ? (cp == ' ' || (cp >= 9 && cp <= 13)) : es_uni_is_space(cp); }

/* ---------------- classes ---------------- */
enum { I_RANGE, I_CAT, I_SPACE, I_DIGIT, I_WORD, I_HAN };
typedef struct { uint8_t kind, neg; uint32_t lo, hi; } item;
typedef struct { item *it; int n, cap; int neg; } cls;

static int item_has(const item *i, uint32_t c) {
    int r;
    switch (i->kind) {
    case I_RANGE: r = c >= i->lo && c <= i->hi; break;
    case I_CAT: r = (pcat(c) & (int)i->lo) != 0; break;
    case I_SPACE: r = is_s(c); break;
    case I_DIGIT: r = c >= '0' && c <= '9'; break;
    case I_WORD: r = (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_'; break;
    default: r = es_uni_is_han(c); break;
    }
    return i->neg ? !r : r;
}
static int cls_has(const cls *k, uint32_t c) {
    int r = 0;
    for (int i = 0; i < k->n && !r; i++) r = item_has(&k->it[i], c);
    return k->neg ? !r : r;
}

/* ---------------- syntax tree ---------------- */
enum { N_CLASS, N_ANY, N_BOL, N_EOL, N_GROUP, N_ALT, N_LOOK, N_REP };
typedef struct node node;
struct node {
    uint8_t t, neg;
    int min, max;            /* N_REP */
    node *a;                 /* body: N_GROUP, N_LOOK, N_REP */
    node **alts; int nalt;   /* N_ALT */
    cls *c;                  /* N_CLASS */
    node *next;
    /* N_REP whose body reads exactly one character: all conditions must hold */
    const cls **cond; uint8_t *cneg; int ncond, simple;
};

struct es_rx {
    node *root;
    node **all; int nall, capall;
    cls **cls; int ncls, capcls;
};

typedef struct { es_rx *r; const uint32_t *p; int n, i; const char *err; } parser;

static node *mk(parser *ps, int t) {
    es_rx *r = ps->r;
    if (r->nall == r->capall) { r->capall = r->capall ? 2 * r->capall : 32; r->all = realloc(r->all, sizeof(node *) * (size_t)r->capall); }
    node *n = calloc(1, sizeof *n);
    n->t = (uint8_t)t;
    r->all[r->nall++] = n;
    return n;
}
static cls *mkcls(parser *ps) {
    es_rx *r = ps->r;
    if (r->ncls == r->capcls) { r->capcls = r->capcls ? 2 * r->capcls : 16; r->cls = realloc(r->cls, sizeof(cls *) * (size_t)r->capcls); }
    cls *c = calloc(1, sizeof *c);
    r->cls[r->ncls++] = c;
    return c;
}
static void add(cls *c, int kind, int neg, uint32_t lo, uint32_t hi) {
    if (c->n == c->cap) { c->cap = c->cap ? 2 * c->cap : 8; c->it = realloc(c->it, sizeof(item) * (size_t)c->cap); }
    c->it[c->n++] = (item){(uint8_t)kind, (uint8_t)neg, lo, hi};
}
static int hexv(uint32_t c) { return c >= '0' && c <= '9' ? (int)c - '0' : (c | 32) >= 'a' && (c | 32) <= 'f' ? (int)(c | 32) - 'a' + 10 : -1; }

/* an escape after '\': adds to c and returns 1, or returns 0 with *lit set to a literal code point */
static int escape(parser *ps, cls *c, uint32_t *lit) {
    if (ps->i >= ps->n) { ps->err = "trailing backslash"; return 0; }
    const uint32_t e = ps->p[ps->i++];
    switch (e) {
    case 's': add(c, I_SPACE, 0, 0, 0); return 1;
    case 'S': add(c, I_SPACE, 1, 0, 0); return 1;
    case 'd': add(c, I_DIGIT, 0, 0, 0); return 1;
    case 'D': add(c, I_DIGIT, 1, 0, 0); return 1;
    case 'w': add(c, I_WORD, 0, 0, 0); return 1;
    case 'W': add(c, I_WORD, 1, 0, 0); return 1;
    case 'p': case 'P': {
        if (ps->i >= ps->n || ps->p[ps->i] != '{') { ps->err = "\\p needs {...}"; return 0; }
        char name[16]; int k = 0;
        ps->i++;
        while (ps->i < ps->n && ps->p[ps->i] != '}' && k < 15) name[k++] = (char)ps->p[ps->i++];
        name[k] = 0;
        if (ps->i >= ps->n) { ps->err = "unclosed \\p{"; return 0; }
        ps->i++;
        const int neg = e == 'P';
        int m = 0;
        if (!strcmp(name, "Han")) { add(c, I_HAN, neg, 0, 0); return 1; }
        if (name[0] == 'L') m = ESU_LETTER;   /* L, Lu, Ll, Lt, Lm, Lo: llama.cpp does not split letters by case */
        else if (!strcmp(name, "N")) m = ESU_NUMBER;
        else if (!strcmp(name, "P")) m = ESU_PUNCT;
        else if (!strcmp(name, "M")) m = ESU_MARK;
        else if (!strcmp(name, "S")) m = ESU_SYMBOL;
        else if (!strcmp(name, "Z")) m = ESU_SEPARATOR;
        else if (!strcmp(name, "C")) m = ESU_CONTROL;
        else { ps->err = "unknown \\p{} class"; return 0; }
        add(c, I_CAT, neg, (uint32_t)m, 0);
        return 1;
    }
    case 'r': *lit = '\r'; return 0;
    case 'n': *lit = '\n'; return 0;
    case 't': *lit = '\t'; return 0;
    case 'f': *lit = '\f'; return 0;
    case 'v': *lit = '\v'; return 0;
    case 'x': case 'u': {
        const int nd = e == 'x' ? 2 : 4; uint32_t v = 0;
        for (int k = 0; k < nd; k++) {
            const int h = ps->i < ps->n ? hexv(ps->p[ps->i]) : -1;
            if (h < 0) { ps->err = "bad hex escape"; return 0; }
            v = v * 16 + (uint32_t)h; ps->i++;
        }
        *lit = v; return 0;
    }
    default: *lit = e; return 0;
    }
}

static node *parse_alt(parser *ps);

static node *parse_atom(parser *ps) {
    const uint32_t c = ps->p[ps->i];
    if (c == '(') {
        ps->i++;
        int t = N_GROUP, neg = 0;
        if (ps->i + 1 < ps->n && ps->p[ps->i] == '?') {
            const uint32_t k = ps->p[ps->i + 1];
            if (k == ':') ps->i += 2;
            else if (k == '=' || k == '!') { t = N_LOOK; neg = k == '!'; ps->i += 2; }
            else { ps->err = "unsupported group"; return NULL; }
        }
        node *g = mk(ps, t);
        g->neg = (uint8_t)neg;
        g->a = parse_alt(ps);
        if (ps->err) return NULL;
        if (ps->i >= ps->n || ps->p[ps->i] != ')') { ps->err = "missing )"; return NULL; }
        ps->i++;
        return g;
    }
    if (c == '[') {
        ps->i++;
        node *nd = mk(ps, N_CLASS);
        cls *k = nd->c = mkcls(ps);
        if (ps->i < ps->n && ps->p[ps->i] == '^') { k->neg = 1; ps->i++; }
        int first = 1;
        while (ps->i < ps->n && (ps->p[ps->i] != ']' || first)) {
            first = 0;
            uint32_t lo = ps->p[ps->i++];
            if (lo == '\\') { if (escape(ps, k, &lo)) continue; if (ps->err) return NULL; }
            uint32_t hi = lo;
            if (ps->i + 1 < ps->n && ps->p[ps->i] == '-' && ps->p[ps->i + 1] != ']') {
                ps->i++;
                hi = ps->p[ps->i++];
                if (hi == '\\') { cls tmp = {0}; if (escape(ps, &tmp, &hi)) { free(tmp.it); ps->err = "class in a range"; return NULL; } if (ps->err) return NULL; }
            }
            add(k, I_RANGE, 0, lo, hi);
        }
        if (ps->i >= ps->n) { ps->err = "missing ]"; return NULL; }
        ps->i++;
        return nd;
    }
    ps->i++;
    if (c == '.') { node *nd = mk(ps, N_CLASS); nd->c = mkcls(ps); nd->c->neg = 1; add(nd->c, I_RANGE, 0, '\n', '\n'); return nd; }
    if (c == '^') return mk(ps, N_BOL);
    if (c == '$') return mk(ps, N_EOL);
    node *nd = mk(ps, N_CLASS);
    nd->c = mkcls(ps);
    uint32_t lit = c;
    if (c == '\\') { if (escape(ps, nd->c, &lit)) return nd; if (ps->err) return NULL; }
    add(nd->c, I_RANGE, 0, lit, lit);
    return nd;
}

static int number(parser *ps, int *v) {
    int k = 0, x = 0;
    while (ps->i < ps->n && ps->p[ps->i] >= '0' && ps->p[ps->i] <= '9') { x = x * 10 + (int)(ps->p[ps->i++] - '0'); k++; }
    *v = x;
    return k;
}

/* a sequence of atoms with quantifiers, linked by next; stops at | or ) */
static node *parse_seq(parser *ps) {
    node *head = NULL, *tail = NULL;
    while (ps->i < ps->n && ps->p[ps->i] != '|' && ps->p[ps->i] != ')') {
        node *a = parse_atom(ps);
        if (!a) return NULL;
        if (ps->i < ps->n) {
            const uint32_t q = ps->p[ps->i];
            int mn = -1, mx = -1;
            if (q == '?') { mn = 0; mx = 1; ps->i++; }
            else if (q == '*') { mn = 0; mx = 1 << 30; ps->i++; }
            else if (q == '+') { mn = 1; mx = 1 << 30; ps->i++; }
            else if (q == '{' && ps->i + 1 < ps->n && ps->p[ps->i + 1] >= '0' && ps->p[ps->i + 1] <= '9') {
                ps->i++;
                number(ps, &mn); mx = mn;
                if (ps->i < ps->n && ps->p[ps->i] == ',') { ps->i++; if (!number(ps, &mx)) mx = 1 << 30; }
                if (ps->i >= ps->n || ps->p[ps->i] != '}') { ps->err = "bad {n,m}"; return NULL; }
                ps->i++;
            }
            if (mn >= 0) {
                if (ps->i < ps->n && (ps->p[ps->i] == '?' || ps->p[ps->i] == '+')) { ps->err = "lazy/possessive quantifiers are not supported"; return NULL; }
                node *r = mk(ps, N_REP);
                r->min = mn; r->max = mx; r->a = a;
                a = r;
            }
        }
        if (tail) tail->next = a; else head = a;
        tail = a;
    }
    return head;
}

static node *parse_alt(parser *ps) {
    node *first = parse_seq(ps);
    if (ps->err) return NULL;
    if (ps->i >= ps->n || ps->p[ps->i] != '|') return first;
    node *alt = mk(ps, N_ALT);
    int cap = 4;
    alt->alts = malloc(sizeof(node *) * (size_t)cap);
    alt->alts[alt->nalt++] = first;
    while (ps->i < ps->n && ps->p[ps->i] == '|') {
        ps->i++;
        node *s = parse_seq(ps);
        if (ps->err) return NULL;
        if (alt->nalt == cap) { cap *= 2; alt->alts = realloc(alt->alts, sizeof(node *) * (size_t)cap); }
        alt->alts[alt->nalt++] = s;
    }
    return alt;
}

/* does sequence s read exactly one character, so that a repeat of it is a loop over characters?
 * Collects the class conditions (lookaheads of one character, and the class that reads it). */
static int one_char(const node *s, const cls **cond, uint8_t *cneg, int *nc, int *reads) {
    for (; s; s = s->next) {
        if (s->t == N_CLASS) {
            if (*reads || *nc >= 8) return 0;
            cond[*nc] = s->c; cneg[*nc] = 0; (*nc)++; *reads = 1;
        } else if (s->t == N_GROUP) {
            if (!one_char(s->a, cond, cneg, nc, reads)) return 0;
        } else if (s->t == N_LOOK) {
            const node *b = s->a;
            if (!b || b->t != N_CLASS || b->next || *nc >= 8 || *reads) return 0;
            cond[*nc] = b->c; cneg[*nc] = s->neg; (*nc)++;
        } else return 0;
    }
    return 1;
}
static void prepare(es_rx *r) {
    for (int i = 0; i < r->nall; i++) {
        node *n = r->all[i];
        if (n->t != N_REP) continue;
        const cls *cond[8]; uint8_t cneg[8]; int nc = 0, reads = 0;
        if (one_char(n->a, cond, cneg, &nc, &reads) && reads) {
            n->simple = 1; n->ncond = nc;
            n->cond = malloc(sizeof(cls *) * (size_t)nc); n->cneg = malloc((size_t)nc);
            memcpy(n->cond, cond, sizeof(cls *) * (size_t)nc); memcpy(n->cneg, cneg, (size_t)nc);
        }
    }
}

es_rx *es_rx_compile(const char *pattern, const char **err) {
    /* UTF-8 pattern -> code points */
    const uint8_t *s = (const uint8_t *)pattern;
    const int L = (int)strlen(pattern);
    uint32_t *cp = malloc(sizeof(uint32_t) * (size_t)(L + 1));
    int n = 0;
    for (int i = 0; i < L;) {
        const uint8_t c = s[i];
        int k = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (i + k > L) k = 1;
        uint32_t v = k == 1 ? c : k == 2 ? (c & 31u) : k == 3 ? (c & 15u) : (c & 7u);
        for (int j = 1; j < k; j++) v = (v << 6) | (s[i + j] & 63u);
        cp[n++] = v;
        i += k;
    }
    es_rx *r = calloc(1, sizeof *r);
    parser ps = {r, cp, n, 0, NULL};
    r->root = parse_alt(&ps);
    if (!ps.err && ps.i < n) ps.err = "unbalanced )";
    free(cp);
    if (ps.err) { if (err) *err = ps.err; es_rx_free(r); return NULL; }
    prepare(r);
    return r;
}

void es_rx_free(es_rx *r) {
    if (!r) return;
    for (int i = 0; i < r->nall; i++) { free(r->all[i]->alts); free(r->all[i]->cond); free(r->all[i]->cneg); free(r->all[i]); }
    for (int i = 0; i < r->ncls; i++) { free(r->cls[i]->it); free(r->cls[i]); }
    free(r->all); free(r->cls); free(r);
}

/* ---------------- matching (backtracking with continuations) ---------------- */
typedef struct frame { const node *n; int rep, cnt, start; const struct frame *up; } frame;
typedef struct { const uint32_t *cp; int begin, end; } mctx;

static int m(const mctx *x, const node *n, int pos, const frame *k);
static int rep(const mctx *x, const node *n, int pos, int cnt, const frame *k) {
    if (cnt < n->max) {
        frame f = {n, 1, cnt + 1, pos, k};
        const int r = m(x, n->a, pos, &f);
        if (r >= 0) return r;
    }
    return cnt >= n->min ? m(x, n->next, pos, k) : -1;
}
static int m(const mctx *x, const node *n, int pos, const frame *k) {
    for (;;) {
        if (!n) {
            if (!k) return pos;
            if (k->rep) {
                if (pos == k->start) return m(x, k->n->next, pos, k->up);   /* an empty iteration ends the loop */
                return rep(x, k->n, pos, k->cnt, k->up);
            }
            n = k->n; k = k->up;
            continue;
        }
        switch (n->t) {
        case N_CLASS:
            if (pos >= x->end || !cls_has(n->c, x->cp[pos])) return -1;
            pos++; n = n->next; continue;
        case N_BOL:
            if (pos != x->begin) return -1;
            n = n->next; continue;
        case N_EOL:
            if (pos != x->end) return -1;
            n = n->next; continue;
        case N_GROUP: {
            frame f = {n->next, 0, 0, 0, k};
            return m(x, n->a, pos, &f);
        }
        case N_ALT: {
            frame f = {n->next, 0, 0, 0, k};
            for (int i = 0; i < n->nalt; i++) { const int r = m(x, n->alts[i], pos, &f); if (r >= 0) return r; }
            return -1;
        }
        case N_LOOK: {
            const int ok = m(x, n->a, pos, NULL) >= 0;
            if (ok == n->neg) return -1;
            n = n->next; continue;
        }
        case N_REP:
            if (n->simple) {
                int run = 0;
                while (run < n->max && pos + run < x->end) {
                    const uint32_t c = x->cp[pos + run];
                    int ok = 1;
                    for (int i = 0; i < n->ncond && ok; i++) ok = cls_has(n->cond[i], c) != n->cneg[i];
                    if (!ok) break;
                    run++;
                }
                for (int i = run; i >= n->min; i--) { const int r = m(x, n->next, pos + i, k); if (r >= 0) return r; }
                return -1;
            }
            return rep(x, n, pos, 0, k);
        default: return -1;
        }
    }
}

int es_rx_match(const es_rx *r, const uint32_t *cp, int n, int pos) {
    const mctx x = {cp, 0, n};
    return m(&x, r->root, pos, NULL);
}

int es_rx_split(const es_rx *r, const uint32_t *cp, int a, int b, int *ends, int cap) {
    /* the fragment is its own text: ^, $ and look-aheads stop at b */
    const mctx x = {cp, a, b};
    int cnt = 0, prev = a;
    for (int s = a; s < b;) {
        const int e = m(&x, r->root, s, NULL);
        if (e < 0) { s++; continue; }
        if (s > prev && cnt < cap) ends[cnt++] = s;
        if (e > s) { if (cnt < cap) ends[cnt++] = e; prev = e; s = e; }
        else { prev = s; s++; }   /* empty match: move on */
    }
    if (prev < b && cnt < cap) ends[cnt++] = b;
    return cnt;
}
