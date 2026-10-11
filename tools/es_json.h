/* Small JSON reader for es_serve (requests are small; one arena per request). Parses objects, arrays, strings (all escapes, \u with
 * surrogate pairs -> UTF-8), numbers, true/false/null. Not a validator: returns NULL on malformed input. */
#ifndef ES_JSON_H
#define ES_JSON_H
#include <stdlib.h>
#include <string.h>

enum { JNULL, JBOOL, JNUM, JSTR, JARR, JOBJ };
typedef struct jv { int t; double n; char *s; struct jv *kid, *next; char *key; int len; } jv;
typedef struct { const char *p, *e; jv *pool; int used, cap; char *str; size_t sused, scap; int depth; } jparser;

static jv *jnew(jparser *P, int t) {
    if (P->used >= P->cap) return NULL;
    jv *v = &P->pool[P->used++]; memset(v, 0, sizeof *v); v->t = t; return v;
}
static void jws(jparser *P) { while (P->p < P->e && (*P->p == ' ' || *P->p == '\t' || *P->p == '\n' || *P->p == '\r')) P->p++; }
static int jhex(const char *s) { int v = 0; for (int i = 0; i < 4; i++) { char c = s[i]; v <<= 4; if (c >= '0' && c <= '9') v |= c - '0'; else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10; else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10; else return -1; } return v; }
static char *jstring(jparser *P) {           /* P->p at the opening quote */
    if (P->p >= P->e || *P->p != '"') return NULL;
    P->p++;
    char *out = P->str + P->sused; size_t o = 0;
    while (P->p < P->e && *P->p != '"') {
        if (P->sused + o + 8 >= P->scap) return NULL;
        unsigned char c = (unsigned char)*P->p++;
        if (c != '\\') { out[o++] = (char)c; continue; }
        if (P->p >= P->e) return NULL;
        c = (unsigned char)*P->p++;
        if (c == 'n') out[o++] = '\n'; else if (c == 't') out[o++] = '\t'; else if (c == 'r') out[o++] = '\r';
        else if (c == 'b') out[o++] = '\b'; else if (c == 'f') out[o++] = '\f';
        else if (c == 'u') {
            if (P->e - P->p < 4) return NULL;
            long cp = jhex(P->p); P->p += 4; if (cp < 0) return NULL;
            if (cp >= 0xD800 && cp < 0xDC00 && P->e - P->p >= 6 && P->p[0] == '\\' && P->p[1] == 'u') {
                int lo = jhex(P->p + 2);
                if (lo >= 0xDC00 && lo < 0xE000) { cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00); P->p += 6; }
            }
            if (cp < 0x80) out[o++] = (char)cp;
            else if (cp < 0x800) { out[o++] = (char)(0xC0 | (cp >> 6)); out[o++] = (char)(0x80 | (cp & 63)); }
            else if (cp < 0x10000) { out[o++] = (char)(0xE0 | (cp >> 12)); out[o++] = (char)(0x80 | ((cp >> 6) & 63)); out[o++] = (char)(0x80 | (cp & 63)); }
            else { out[o++] = (char)(0xF0 | (cp >> 18)); out[o++] = (char)(0x80 | ((cp >> 12) & 63)); out[o++] = (char)(0x80 | ((cp >> 6) & 63)); out[o++] = (char)(0x80 | (cp & 63)); }
        } else out[o++] = (char)c;   /* \" \\ \/ */
    }
    if (P->p >= P->e) return NULL;
    P->p++;
    out[o] = 0; P->sused += o + 1;
    return out;
}
static jv *jvalue(jparser *P) {
    jws(P);
    if (P->p >= P->e || ++P->depth > 64) return NULL;
    jv *v = NULL; char c = *P->p;
    if (c == '{' || c == '[') {
        const int obj = c == '{'; v = jnew(P, obj ? JOBJ : JARR); if (!v) return NULL;
        P->p++; jws(P); jv **tail = &v->kid;
        if (P->p < P->e && *P->p == (obj ? '}' : ']')) { P->p++; P->depth--; return v; }
        for (;;) {
            char *k = NULL;
            if (obj) { jws(P); k = jstring(P); if (!k) return NULL; jws(P); if (P->p >= P->e || *P->p != ':') return NULL; P->p++; }
            jv *x = jvalue(P); if (!x) return NULL;
            x->key = k; *tail = x; tail = &x->next; v->len++;
            jws(P);
            if (P->p < P->e && *P->p == ',') { P->p++; continue; }
            if (P->p < P->e && *P->p == (obj ? '}' : ']')) { P->p++; break; }
            return NULL;
        }
    } else if (c == '"') { v = jnew(P, JSTR); if (!v || !(v->s = jstring(P))) return NULL; }
    else if (!strncmp(P->p, "true", 4)) { v = jnew(P, JBOOL); if (v) v->n = 1; P->p += 4; }
    else if (!strncmp(P->p, "false", 5)) { v = jnew(P, JBOOL); P->p += 5; }
    else if (!strncmp(P->p, "null", 4)) { v = jnew(P, JNULL); P->p += 4; }
    else { char *end; double d = strtod(P->p, &end); if (end == P->p) return NULL; v = jnew(P, JNUM); if (v) v->n = d; P->p = end; }
    P->depth--;
    return v;
}
/* parse text[0..n); memory is owned by the parser (free with jfree) */
static jv *jparse(jparser *P, const char *text, size_t n) {
    memset(P, 0, sizeof *P);
    P->p = text; P->e = text + n; P->cap = 4096 + (int)(n / 4); P->pool = malloc(sizeof(jv) * (size_t)P->cap);
    P->scap = n + 64; P->str = malloc(P->scap);
    if (!P->pool || !P->str) return NULL;
    jv *v = jvalue(P); jws(P);
    return v;
}
static void jfree(jparser *P) { free(P->pool); free(P->str); P->pool = NULL; P->str = NULL; }
static jv *jget(const jv *o, const char *k) { if (!o || o->t != JOBJ) return NULL; for (jv *c = o->kid; c; c = c->next) if (c->key && !strcmp(c->key, k)) return c; return NULL; }
static const char *jstr(const jv *o, const char *k, const char *def) { jv *v = jget(o, k); return v && v->t == JSTR ? v->s : def; }
static double jnum(const jv *o, const char *k, double def) { jv *v = jget(o, k); return v && (v->t == JNUM || v->t == JBOOL) ? v->n : def; }
#endif
