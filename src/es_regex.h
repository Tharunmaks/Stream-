/*
 * Small backtracking regex engine for tokenizer pre-splitting. Works on Unicode code points.
 * Supports what tokenizer patterns use: | ( ) (?: ) (?= ) (?! ) [...] [^...] ranges . ^ $
 * ? * + {n} {n,} {n,m}, escapes \s \S \d \D \w \W \r \n \t \f \v \xHH \uHHHH and \p{L N P M S Z C Lu Ll Lt Lm Lo Han}
 * (and \P{...}). Alternatives are tried in order (first match wins), like ECMAScript and PCRE.
 *
 * Character classes follow llama.cpp's conventions, so GGUF models split text exactly as there:
 * ASCII characters use fixed ASCII classes, other characters their Unicode general category, and
 * a non-ASCII white-space character only matches \s.
 */
#ifndef ES_REGEX_H
#define ES_REGEX_H

#include <stdint.h>

typedef struct es_rx es_rx;

es_rx *es_rx_compile(const char *pattern, const char **err);
void   es_rx_free(es_rx *r);
/* match anchored at pos inside cp[0..n); returns the end of the match or -1 */
int    es_rx_match(const es_rx *r, const uint32_t *cp, int n, int pos);
/* split [a, b) into pieces: every match and every gap between matches is a piece (llama.cpp semantics).
 * Appends piece ends to ends[] (cap entries); returns the number appended. */
int    es_rx_split(const es_rx *r, const uint32_t *cp, int a, int b, int *ends, int cap);

/* code point properties used by the tokenizer */
int es_uni_flags(uint32_t cp);      /* ESU_* category flags of es_unicode.h */
int es_uni_is_space(uint32_t cp);
int es_uni_is_han(uint32_t cp);

#endif
