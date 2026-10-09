/*
 * Byte-level BPE tokenizer (GPT-2 family: OLMo, Qwen, Llama-3 style vocab)
 * loaded from GGUF metadata. Mirrors llama.cpp: special and user-defined
 * tokens are matched first (longest first), then the GPT-2 pre-tokenizer
 * splits the rest, then BPE merges by rank.
 *
 * Limitation: non-ASCII characters are all treated as letters by the
 * pre-tokenizer (no Unicode tables), so text with non-ASCII punctuation
 * can split differently from llama.cpp. ASCII text matches exactly.
 */
#ifndef ES_TOK_H
#define ES_TOK_H

#include <stdint.h>

#include "es_gguf.h"

typedef struct es_tok es_tok;

es_tok *es_tok_load(const es_gguf *g, const char **err);
void    es_tok_free(es_tok *t);

/* Encode UTF-8 text. parse_special: also match control tokens such as
 * <|endoftext|> written in the text. Returns token count (<= max). */
int  es_tok_encode(const es_tok *t, const char *text, int parse_special, int32_t *out, int max);
/* Bytes of token id (control tokens give "" unless show_special). */
int  es_tok_piece(const es_tok *t, int32_t id, int show_special, char *buf, int cap);
int  es_tok_n_vocab(const es_tok *t);
int32_t es_tok_bos(const es_tok *t);
int32_t es_tok_eos(const es_tok *t);
int  es_tok_add_bos(const es_tok *t);
/* id of an exact vocab string, or -1 */
int32_t es_tok_find(const es_tok *t, const char *s);
/* 1 if the token is an end-of-generation token (eos / eot / im_end ...) */
int  es_tok_is_eog(const es_tok *t, int32_t id);

#endif
