/*
 * ExpertStream inference engine: MoE text generation with the core in RAM
 * and routed experts read on demand into a RAM cache.
 *
 * Experts come from one of two sources:
 *   - an ExpertStream pack (es_import output): fast parallel O_DIRECT reads
 *   - a GGUF file read directly, no import needed (used by the browser build)
 */
#ifndef ES_ENGINE_H
#define ES_ENGINE_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    const char *pack;        /* pack directory, or NULL */
    const char *gguf;        /* GGUF file path, or NULL */
    int         gguf_fd;     /* >= 0: GGUF handle already open via es_gguf_pread hook */
    uint64_t    gguf_size;   /* size for gguf_fd */
    int         ctx;         /* context length in tokens */
    size_t      cache_mb;    /* expert cache size (0 = auto: 60% of the free RAM) */
    int         threads;     /* compute threads (pinned to the fastest cores) */
    const char *trace_path;  /* optional routing trace file */
    /* ExpertStream Lookahead (native engine, GGUF experts) */
    float       skip_thr;    /* drop experts whose router weight < thr * best (0 = off) */
    int         lookahead;   /* 1 = predict the next layer's experts and load them while computing */
    int         warm;        /* 1 = preload the experts this model used most last time */
    int         io_threads;  /* parallel expert readers (0 = 4) */
    float       cache_route; /* prefer experts already in RAM when nearly as good (bonus = this x best score; 0 = off, 0.2 measured) */
    int         spec;        /* Turbo: guess up to this many next tokens from the text so far and check them together (-1 = default 7, 0 = off) */
} es_engine_opts;

typedef struct {
    const char *name, *arch;
    int layers, experts, used, ctx, cache_slots;
    double core_mb, cache_mb, kv_mb, load_s;
    const char *remote;      /* experts on demand (a model fetched with es_get): what is on this device, else NULL */
} es_engine_info;

typedef struct {
    int prompt_tokens, gen_tokens, ctx_reset;
    double prompt_s, gen_s, hit_pct, flash_mb, wait_s;
    unsigned long long pf_issued, pf_used, skipped, spec_steps, spec_accepted;
    unsigned long long fetched;   /* experts downloaded on demand this session, and their size */
    double net_mb;
} es_turn_stats;

/* 0 on success, else writes a message to err */
int  es_engine_init(const es_engine_opts *o, char *err, size_t errcap);
void es_engine_get_info(es_engine_info *info);
void es_engine_set_sampling(float temp, int top_k, float top_p, int max_new, uint64_t seed);
/* Forget the conversation. */
void es_engine_reset(void);
/* Tokenize msg (chat template unless raw) and feed it to the model.
 * ids receive the prompt's token ids. progress(0, total) is called once the
 * ids are known, then progress(done, total) as tokens are read. Returns the token count, or -1 if it can't fit. */
int  es_engine_begin(const char *msg, int raw, int32_t *ids, int ids_cap,
                     void (*progress)(int done, int total, void *ud), void *ud);
/* Next piece of the answer into buf. Returns its length (may be 0), or -1
 * when the answer is complete. */
int  es_engine_next(char *buf, int cap);
void es_engine_turn_stats(es_turn_stats *s);
/* ---- Instant: serve apps that resend the whole conversation every turn ----
 * Render a conversation with the model's own chat template, ending where the assistant answers. roles: system, user, assistant.
 * Returns the text length, or -1 if it does not fit in cap. */
typedef struct { const char *role, *content; } es_msg;
int  es_engine_render(const es_msg *m, int n, char *out, size_t cap);
/* The same for a given tokenizer.chat_template text (no model needed; for tests), and the family it is recognised as. */
int  es_engine_render_with(const char *chat_template, const es_msg *m, int n, char *out, size_t cap);
const char *es_engine_template_kind(const char *chat_template);
/* Make the context exactly `text` (already formatted; BOS added when the model wants it), reusing the longest prefix that is already
 * computed: the live conversation, or one of a few parked ones (ES_KV_SLOTS, default 4; ES_PREFIX_CACHE=0 turns reuse off).
 * Only the new tokens are read. *cached = prompt tokens reused. Returns the prompt token count, or -1 if it does not fit. */
int  es_engine_prompt(const char *text, int *cached, int32_t *ids, int ids_cap, void (*progress)(int done, int total, void *ud), void *ud);
/* Tokens in a text (for trimming a conversation to the context). */
int  es_engine_count(const char *text);
/* Testing: feed ids one by one and write each position's logits. */
int  es_engine_dump_logits(const int32_t *ids, int n, FILE *f);
long es_rss_mb(void);
/* ---- Graft (es_graft.h): a small trained network on top of the frozen model ---- */
/* load a graft trained for this model (NULL or "" removes it); 0 on success */
int  es_engine_set_graft(const char *path, char *err, size_t cap);
/* graft training: the base model's final hidden state and scores for every wanted position of one sequence */
int  es_engine_features(const int32_t *ids, int n, const uint8_t *want, void (*cb)(int pos, const float *x, const float *logits, void *ud), void *ud);
int  es_engine_chat_example(const es_msg *m, int n, const char *resp, int32_t *ids, int cap, int *start);
int  es_engine_tokenize(const char *text, int32_t *ids, int cap);   /* raw text, no BOS */
/* the graft's context at a position: win = the last n (<= 32) tokens up to it, re-split as the tokenizer would (ctx[0] = last) */
void es_engine_canon_ctx(const int32_t *win, int n, int32_t *ctx);
int  es_engine_bos(void);                                          /* BOS to put in front of raw text, or -1 */
int  es_engine_piece(int32_t id, char *buf, int cap);
uint64_t es_engine_fingerprint(void);
void es_engine_dims(int *n_embd, int *n_vocab);
const char *es_engine_arch(void);
int  es_engine_output_raw(uint32_t *type, const uint8_t **data, size_t *row_bytes);
/* ---- Adapter: a small residual adapter after every layer, trained without backpropagation (es_graft zo) ---- */
int    es_engine_adapter_new(int rank, uint64_t seed);                 /* A random from seed, B = 0 (changes nothing yet) */
int    es_engine_adapter_load(const char *path, uint64_t *seed, uint64_t *steps, char *err, size_t cap);
int    es_engine_adapter_save(const char *path, uint64_t steps, char *err, size_t cap);
size_t es_engine_adapter_count(void);                                 /* its numbers ... */
float *es_engine_adapter_params(void);                                /* ... which may be changed in place, then: */
void   es_engine_adapter_changed(void);                               /* forget everything computed with the old numbers */
int    es_engine_adapter_rank(void);
float  es_engine_gauss(uint64_t seed, uint64_t i);                    /* N(0,1), identical on every device */
void   es_engine_adapter_lanes(float *lane0, float *lane1);           /* adapter versions for es_engine_features_multi (NULL = the adapter) */
/* Several sequences through the model layer by layer: each layer's weights and experts are read once for all of them (a model
 * bigger than RAM is streamed once per group instead of once per 64 tokens). cb(s, pos, ...) as es_engine_features. lane[s]
 * picks the adapter version of sequence s (NULL: all the same). Same numbers as one sequence at a time. */
int    es_engine_features_multi(int S, const int32_t *const *ids, const int *len, const uint8_t *const *want, const int *lane,
                                void (*cb)(int s, int pos, const float *x, const float *logits, void *ud), void *ud);
double es_engine_flash_mb(void);                                      /* expert data read from storage so far, MB */
/* Remember which experts were used (read back by warm start). */
void es_engine_save_profile(void);

#endif
