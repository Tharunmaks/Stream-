/* ExpertStream trainer: LoRA fine-tuning of a frozen GGUF model with exact gradients, on the CPU.
 *
 * The model's weights stay quantized and memory-mapped; every step streams them layer by layer (forward, then the backward pass
 * recomputes each layer from its saved input and goes back through it), so a model larger than RAM trains with the memory of one
 * layer plus the saved layer inputs. Only the LoRA matrices (A: rank x in, B: out x rank, on q k v o gate up down) are trained;
 * their gradients are exact (checked against finite differences by `es_lora gradcheck`).
 *
 * Supported: dense llama, qwen2 and qwen3 models (Llama 2/3, Mistral, CodeLlama, DeepSeek-Coder, SmolLM, Qwen2.5 and
 * Qwen2.5-Coder, Qwen3), every weight format the engine reads.
 */
#ifndef ES_TRAIN_H
#define ES_TRAIN_H
#include <stddef.h>
#include <stdint.h>

enum { ES_LT_Q, ES_LT_K, ES_LT_V, ES_LT_O, ES_LT_GATE, ES_LT_UP, ES_LT_DOWN, ES_LT_N };
#define ES_LT_ALL 0x7fu

typedef struct es_tm es_tm;

/* Opens the model (read-only memory map). threads 0 = all cores. NULL and a message in err on failure. */
es_tm *es_tm_open(const char *gguf, int threads, char *err, size_t cap);
void   es_tm_close(es_tm *m);
void   es_tm_dims(const es_tm *m, int *n_layer, int *n_embd, int *n_vocab);
const char *es_tm_arch(const es_tm *m);

/* LoRA on the targets in mask (bit ES_LT_*): A random (seeded), B = 0, so a new LoRA changes nothing. */
int    es_tm_lora_new(es_tm *m, int rank, float alpha, uint32_t mask, uint64_t seed);
int    es_tm_lora_save(const es_tm *m, const char *path, uint64_t steps, char *err, size_t cap);
int    es_tm_lora_load(es_tm *m, const char *path, uint64_t *steps, char *err, size_t cap);
size_t es_tm_lora_count(const es_tm *m);
float *es_tm_lora_params(es_tm *m);
float *es_tm_lora_grads(es_tm *m);        /* accumulated by es_tm_run(backward=1); zero it before a step */
int    es_tm_lora_where(const es_tm *m, size_t i, int *layer, int *target, int *is_b);   /* which matrix number i belongs to */
void   es_tm_lora_info(const es_tm *m, int *rank, float *alpha, uint32_t *mask);
/* Export as a llama.cpp LoRA adapter (GGUF, F32): llama-cli -m MODEL.gguf --lora FILE */
int    es_tm_lora_export_gguf(const es_tm *m, const char *path, char *err, size_t cap);

/* Forward (and backward) over S sequences, each starting at position 0. Learned: positions p >= first[s] predicting ids[p+1].
 * Returns the summed cross-entropy (nats); *count gets the number of learned positions. With backward, d(gscale * loss)/d(LoRA)
 * is added to the gradients. */
double es_tm_run(es_tm *m, int S, const int32_t *const *ids, const int *len, const int *first, int backward, float gscale, long *count);
/* Logits of every position of one sequence (n x n_vocab), for checks. */
int    es_tm_logits(es_tm *m, const int32_t *ids, int n, float *out);
/* Weight bytes read from the model file by the last es_tm_run (streamed passes). */
double es_tm_bytes_read(const es_tm *m);

#endif
