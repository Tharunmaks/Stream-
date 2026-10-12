/*
 * Tokenizers loaded from GGUF metadata, mirroring llama.cpp:
 * - byte-level BPE (tokenizer.ggml.model = gpt2: Qwen, Llama 3, DeepSeek, Kimi, gpt-oss, GLM ...):
 *   special tokens first (longest first), then the model's pre-tokenizer patterns (es_regex.c, Unicode
 *   categories from es_unicode.h), then BPE merges by rank;
 * - SentencePiece (tokenizer.ggml.model = llama: Llama 1/2, Mistral, Mixtral ...): spaces become U+2581,
 *   adjacent pieces merge by score, unknown characters fall back to <0xXX> byte tokens.
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
void es_tok_set_eog(es_tok *t, int32_t id, int on);
/* 1 for a SentencePiece vocabulary (spaces are U+2581, a space is added in front of the text) */
int  es_tok_is_spm(const es_tok *t);
/* 1 if tokenizer.ggml.pre names a pre-tokenizer this engine knows (otherwise the default rules are used) */
int  es_tok_pre_known(const char *pre);

#endif
