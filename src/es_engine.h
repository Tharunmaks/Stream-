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
    size_t      cache_mb;    /* expert cache size */
    int         threads;     /* compute threads (pinned to the fastest cores) */
    const char *trace_path;  /* optional routing trace file */
    /* ExpertStream Lookahead (native engine, GGUF experts) */
    float       skip_thr;    /* drop experts whose router weight < thr * best (0 = off) */
    int         lookahead;   /* 1 = predict the next layer's experts and load them while computing */
    int         warm;        /* 1 = preload the experts this model used most last time */
    int         io_threads;  /* parallel expert readers (0 = 4) */
    int         spec;        /* Turbo: guess up to this many next tokens from the text so far and check them together (-1 = default 7, 0 = off) */
} es_engine_opts;

typedef struct {
    const char *name, *arch;
    int layers, experts, used, ctx, cache_slots;
    double core_mb, cache_mb, kv_mb, load_s;
} es_engine_info;

typedef struct {
    int prompt_tokens, gen_tokens, ctx_reset;
    double prompt_s, gen_s, hit_pct, flash_mb, wait_s;
    unsigned long long pf_issued, pf_used, skipped, spec_steps, spec_accepted;
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
/* Testing: feed ids one by one and write each position's logits. */
int  es_engine_dump_logits(const int32_t *ids, int n, FILE *f);
long es_rss_mb(void);
/* Remember which experts were used (read back by warm start). */
void es_engine_save_profile(void);

#endif
