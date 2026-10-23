# LoRA with exact gradients: fine-tune a frozen model on a CPU

## What it is

`es_lora` trains a **LoRA** (low-rank adapter) inside every layer of a GGUF model the engine runs, with **exact gradients**
(backpropagation through the frozen model), on the CPU of a PC or phone. The model's own weights are never changed and stay
quantized on disk; only the small LoRA matrices are trained (on q, k, v, o, gate, up and down of every layer, rank 16 by default).

It closes the two gaps that kept training slow here:

| Gap | Before | Now |
|---|---|---|
| Train inside an existing model | Only MeZO adapters (`es_graft zo`): gradients estimated from forward passes, about 10x slower to learn than the graft | Exact gradients through the whole model, like PyTorch LoRA |
| Stream the model's weights during training | The engine streamed layers for inference and MeZO only | Every training step reads the model layer by layer (forward, then the backward pass recomputes each layer from its saved input), so a model bigger than RAM can train |

It does **not** make a CPU as fast as a GPU. A training GPU does hundreds of TFLOPS; a phone does well under one. What it makes
possible is exact fine-tuning, at the CPU's speed, with the memory of one layer's work plus the saved layer inputs.

## Commands

```sh
make es_lora es_chat es_serve
./es_lora train -g MODEL.gguf --data code.jsonl -o mycoder.eslora --epochs 2      # Ctrl-C saves; --resume continues
./es_lora eval  -g MODEL.gguf --data code.jsonl --lora mycoder.eslora              # held-out loss and perplexity
./es_chat -g MODEL.gguf --lora mycoder.eslora                                      # or: es_serve -g MODEL.gguf --lora mycoder.eslora
./es_lora export --lora mycoder.eslora -g MODEL.gguf -o mycoder-lora.gguf          # llama.cpp: llama-cli -m MODEL.gguf --lora mycoder-lora.gguf
```

Data is the same as `es_graft`: `.txt` (all text learned) or `.jsonl` with `{"text": ...}`, `{"prompt": ..., "response": ...}`
or `{"messages": [...]}` (chat template of the model; only the answers are learned). Every 20th example is held out.

Options: `--rank 16` `--alpha` (default: the rank; the LoRA is scaled by alpha/rank) `--targets all|attn|mlp|q,k,v,o,gate,up,down` `--lr 2e-4`
`--epochs 1` or `--steps N`, `--batch 2048` (learned tokens per optimizer step), `--micro 1024` (tokens per pass: memory),
`--seq 512` (example length), `--minutes M`, `--eval-every N`, `-j THREADS`.

Models: dense llama, qwen2 and qwen3 (Llama 2/3, Mistral, CodeLlama, DeepSeek-Coder, SmolLM, Qwen2.5 and Qwen2.5-Coder, Qwen3),
every weight format the engine reads. Mixture-of-experts and YaRN models are not supported by the trainer yet.

## How it works

One optimizer step over T tokens (several examples packed together):

1. **Forward**, layer by layer. Only each layer's input is kept (T x width floats per layer).
2. **Loss**: final norm, output rows and cross-entropy on the learned positions, in tiles of 64 positions.
3. **Backward**, from the last layer down: recompute the layer from its saved input, then go back through it. Attention keeps no
   probability matrix: the backward pass recomputes probabilities row by row from the saved log-sum-exp, so memory grows with T,
   not T squared.
4. **AdamW** on the LoRA numbers (warm-up, cosine decay to 10%, gradient clipping at norm 1).

Every weight matrix is read three times per step, always in layer order. Products are a blocked GEMM: a chunk of weight rows
(about 2 MB) is dequantized once, packed into 32-column panels shared by all threads, and multiplied with register tiles (AVX-512:
8 tokens x 32 outputs, AVX2 and NEON: 6 x 16), summing 64 terms in registers at a time. Packed weights are kept for the next pass
while they fit in `ES_WCACHE_MB` (default a quarter of the RAM; 0 turns it off). The LoRA products use the same GEMM. The frozen
weights get no gradients, so there is no weight-gradient or optimizer memory for them.

**Fast mode** (`ES_AMX=1`): on Intel Xeon 4th gen and later, the frozen products run on AMX bf16 tiles (float32 sums). Losses
match float32 to about 3 decimals. Each tile block is computed twice and compared (a guard; 0 mismatches in our runs);
`ES_AMX_VERIFY=0` skips the check for about 10% more speed. Other machines run the exact float32 path.

**Threads**: all cores by default (`-j N` to choose). The engine, opened for its tokenizer, pins its calling thread to one core;
the trainer resets that before starting its threads (before this fix, every training run used one core whatever `-j` said).

## Checked

* `es_lora gradcheck`: analytic gradients vs finite differences, for every target (A and B), first to last layer, and along random
  directions through all LoRA numbers.
* `es_lora logitcheck`: the trainer's forward pass vs the engine's.

Results:

| Check | SmolLM2-135M (llama) | Qwen2.5-0.5B (qwen2: biases) | Llama-3.2-1B (llama 3 rotary factors) | tiny random Qwen3 (q/k norms, GQA) |
|---|---|---|---|---|
| gradients vs finite differences, 3 random directions | 0.01-0.2% | 0.02-0.09% | 0.006-0.06% | 0.007-0.1% |
| gradients of single numbers (every target, A and B) | within 1.5% | within 1.7% | within 1.6% | within 0.05% |
| trainer vs engine, both in float: largest logit difference | 0.0001 | 0.0002 | 0.0001 | |
| trainer vs engine (8-bit activations): perplexity | 13.21 vs 13.23 | 12.29 vs 12.22 | 8.44 vs 8.43 | |
| engine running a trained LoRA vs trainer (float) | 0.0001, same top token 100% | | | |

Finite differences of a float32 loss are noisy; the single-number errors shrink as the step grows (rounding, not a wrong
gradient), and on the tiny model, where rounding is small, they are 0.005-0.05%.

## Measured: learning, against MeZO and the graft

The same benchmark as [zo.md](zo.md): SmolLM2-135M (Q4_K_M), 20,000 tokens of TinyStories in examples of 128 tokens, the same 8
held-out examples (1,016 tokens).

| Method | Held-out perplexity | Time |
|---|---|---|
| MeZO adapters (`es_graft zo`), best run | 12.35 -> 12.15 (-1.6%) | 83 min (1,500 steps) |
| Graft (`es_graft`) | 12.35 -> 10.21 (-17.3%) | 74 s |
| **LoRA, exact gradients (`es_lora`)**, rank 16, lr 1e-3, 20 steps of 2,048 tokens (2 epochs) | 12.29 -> **8.31 (-32.4%)** | 18 min |

(The LoRA's starting 12.29 is the trainer's float forward pass; the others are the engine's, with 8-bit activations.)
The LoRA keeps learning past the graft because it changes the model's layers, not only its final scores; MeZO, with gradients
estimated from forward passes, barely moved in four times as long.

Speed, trained tokens per second on a 4-core Xeon (AVX-512 + AMX), `--seq 256 --batch 2048 --micro 1024`, steady state:

| model | before | float32 (exact) | Fast mode (AMX bf16) |
|---|---|---|---|
| SmolLM2-135M Q8_0 | 34 | ~190 (5.5x) | ~345 (10x) |
| Qwen2.5-0.5B Q8_0 | 10.5 | ~61 (5.8x) | ~134 (12.8x) |

"Before" was one core (the pinning above) and dot-product kernels. Cost grows with the model: roughly 4-6 x parameters x tokens
floating-point operations per step, so a 7B model is about 14x slower per token than the 0.5B one, and a 70B model about 140x;
a CPU cannot train 70B at interactive speed (that takes a datacenter GPU). Use `--minutes` to bound a run and `--resume` to continue.

