/*
 * Graft: teach a frozen big model new things on a phone.
 *
 * The base model (any size the engine runs, a 200B mixture-of-experts included) stays frozen. A small trainable graft reads
 * the model's final hidden state x (after the output norm, the vector the output matrix multiplies) and the last 1-3
 * tokens, and moves the scores of the model's own top-K candidate tokens:
 *
 *     xh = x / rms(x)                     a = P xh                   (m numbers: the graft's working space)
 *     u  = W2 silu(W1 a + b1)                                         (a small dense network)
 *        + sum over n = 1..3 of  sigmoid(a . k_e / sqrt m) * v_e,     e = hash(last n tokens)   (memory experts on flash)
 *     delta = Q u                          score_c += w_c . delta      for the K tokens the base model ranks highest
 *
 * w_c is row c of the model's output matrix. Q starts at zero, so a new graft is exactly the base model. Every other token
 * keeps its base score, so training can compute the exact loss of the model that later runs (see es_graft_loss).
 * Memory experts live in a sparse file: untouched experts take no space.
 */
#ifndef ES_GRAFT_H
#define ES_GRAFT_H

#include <stddef.h>
#include <stdint.h>

#define ES_GRAFT_MAXK 320
#define ES_GRAFT_ORDERS 3
#define ES_GRAFT_CTX 16      /* tokens of context the graft looks at (the current one and 15 before it) */

/* Memory candidates: an exact n-gram memory of the training text. For a context seen in training, the tokens that followed
 * it there join the candidates whose scores the graft may change, so the graft can say things the base model would not
 * have proposed (a name, a number, a word from your notes). Contexts of 2, 4, 8 and 16 tokens (the long ones finish names
 * and sentences from the training text), up to 4 tokens each. */
#define ES_MEM_ORDERS 4
#define ES_MEM_PER 4         /* suggestions per context */
#define ES_MEM_KEEP 6        /* continuations remembered per context (most frequent) */
#define ES_MEM_BONUS 4       /* learned numbers per memory order: a, b, e and the gate's bias */
typedef struct { uint64_t key; int32_t tok[ES_MEM_KEEP]; uint32_t cnt[ES_MEM_KEEP]; uint32_t total, pad; } es_mem_ent;   /* total: how often the context occurred */
typedef struct { const es_mem_ent *e; uint64_t n; void *map; size_t len; } es_mem;
static const int ES_MEM_ORDER[ES_MEM_ORDERS] = {2, 4, 8, 16};
/* key of the context of `order` tokens ending at ctx[0], or 0 when the context is shorter */
uint64_t es_mem_key(int order, const int32_t *ctx);
const es_mem_ent *es_mem_find(const es_mem *mm, uint64_t key);
/* the suggested tokens for ctx (no duplicates, none of skip[0..nskip)); minus_one: leave one count of that token out
 * (training: the position's own answer must not suggest itself). Returns the count written to out. */
int es_mem_suggest(const es_mem *mm, const int32_t *ctx, int32_t minus_one, const int32_t *skip, int nskip, int32_t *out, int cap);
/* how often the memory saw each candidate after this context, per memory order (cnt: n x ES_MEM_ORDERS), and how often it saw
 * the context (tot: ES_MEM_ORDERS) */
void es_mem_counts(const es_mem *mm, const int32_t *ctx, int32_t minus_one, const int32_t *cand, int n, uint16_t *cnt, uint32_t *tot);
/* the memory's score bonus for one candidate (cnt: its ES_MEM_ORDERS counts; mg: the gates, NULL = 1) */
float es_mem_bonus(const float *mb, const float *mg, const uint16_t *cnt, const uint32_t *tot);
int es_mem_load(es_mem *mm, const char *path, char *err, size_t cap);
void es_mem_free(es_mem *mm);

typedef struct {
    int d, m, f, K, orders, n_vocab;
    uint32_t seed;
    uint64_t n_exp;          /* memory experts (0 = none) */
    uint64_t fingerprint;    /* of the base model: a graft only runs on the model it was trained on */
    uint64_t steps, tokens;  /* training so far */
    char arch[32], name[64];
    float *P, *W1, *b1, *W2, *Q;   /* m x d, f x m, f, m x f, d x m (row-major) */
    float *mb;               /* memory bonus, per memory order: a token seen c times after a context seen n times gets
                              *   gate * (a + b log c + e log(c/n)),  gate = sigmoid(U_o . a + bias): how much to trust this memory here */
    float *U;                /* gate weights: ES_MEM_ORDERS x m */
    float *E;                /* n_exp records of es_graft_rec(m) floats: key[m] val[m] acc_k acc_v; mapped from the experts file */
    size_t e_bytes;
    int e_fd, writable;
    es_mem mem;              /* memory candidates (path.mem), may be empty */
} es_graft;

static inline size_t es_graft_rec(int m) { return (size_t)2 * m + 2; }
static inline size_t es_graft_dense_floats(const es_graft *g) {
    return (size_t)g->m * g->d + (size_t)g->f * g->m + g->f + (size_t)g->m * g->f + (size_t)g->d * g->m + ES_MEM_BONUS * ES_MEM_ORDERS + (size_t)ES_MEM_ORDERS * g->m;
}

/* per-thread scratch of one position's forward pass (kept for the backward pass) */
typedef struct {
    float *xh, *a, *z1, *h1, *u, *delta;   /* d, m, f, f, m, d */
    float rinv;
    int64_t e[ES_GRAFT_ORDERS];            /* expert slot per order, -1 when the context is too short */
    float s[ES_GRAFT_ORDERS], gate[ES_GRAFT_ORDERS];
    float mg[ES_MEM_ORDERS];               /* memory gates */
} es_graft_tmp;

/* new graft with random P, W1, W2 and Q = 0; experts file created sparse at experts_path (NULL or n_exp 0: no experts) */
int  es_graft_new(es_graft *g, int d, int m, int f, int K, uint64_t n_exp, int n_vocab, uint32_t seed, const char *experts_path, char *err, size_t cap);
/* load path (and path.experts, path.mem); writable maps the experts for training */
int  es_graft_load(es_graft *g, const char *path, int writable, char *err, size_t cap);
int  es_graft_save(const es_graft *g, const char *path, char *err, size_t cap);
void es_graft_free(es_graft *g);
int  es_graft_tmp_init(es_graft_tmp *t, const es_graft *g);
void es_graft_tmp_free(es_graft_tmp *t);

/* expert slot of the last `order` tokens (ctx[0] = the current token, ctx[1] the one before, ...), or -1 */
int64_t es_graft_slot(const es_graft *g, int order, const int32_t *ctx);
/* forward: t->delta = the change of the hidden state (d numbers) */
void es_graft_forward(const es_graft *g, const float *x, const int32_t *ctx, es_graft_tmp *t);

/* The exact loss of the grafted model at one position. The K candidates get
 *     score_c = base_c + w_c . delta + es_mem_bonus(cnt_c, tot, gates)
 * and every other token keeps its base score.
 * bl: the candidates' base scores; logr = log of the sum of exp(score) over all other tokens; W: the K output rows (pointers to
 * d floats each), may be NULL when delta is zero; cnt/tot: memory counts (may be NULL).
 * yin >= 0: the right token is candidate yin; -1: it is another token, whose base score is by. Returns -log p(right token).
 * yin == -2: an anchor position (general text the graft must leave alone): returns KL(base || grafted), exact.
 * mg: the memory gates of this position (es_graft_tmp.mg). Writes dL/d delta (d numbers), dL/d mb and dL/d gates when not NULL. */
double es_graft_loss(int K, int d, const float *bl, double logr, int yin, float by, const float *const *W, const float *delta,
                     const uint16_t *cnt, const uint32_t *tot, const float *mb, const float *mg, float *g_delta, float *g_mb, float *g_mg);

/* backward of es_graft_forward for the dense weights: adds into grad (same layout as P..U, see es_graft_dense_floats);
 * g_mg: dL/d memory gates (NULL = none); the experts' gradients go to ek/ev (orders x m each) */
void es_graft_backward(const es_graft *g, const es_graft_tmp *t, const float *g_delta, const float *g_mg, float *grad, float *gu, float *ga, float *ek, float *ev);

/* the tokens whose scores the graft changes at one position: the base model's top K, then the memory's suggestions.
 * minus_one: see es_mem_suggest (-1 at inference). Returns the count (at most K + ES_MEM_ORDERS * ES_MEM_PER). */
int es_graft_candidates(int K, const es_mem *mm, const float *logits, int n_vocab, const int32_t *ctx, int32_t minus_one, int32_t *out);
/* Inference: change the scores of one logits row. ctx: ES_GRAFT_CTX tokens (ctx[0] the current one, -1 before the start).
 * row(tok, out) writes output row tok (d floats). */
void es_graft_apply(const es_graft *g, const float *x, const int32_t *ctx, float *logits, void (*row)(int tok, float *out, void *ud), void *ud,
                    es_graft_tmp *t, float *wbuf);
/* the k largest of v[0..n), sorted descending; returns the count */
int  es_graft_topk(const float *v, int n, int k, int *idx);

#endif
