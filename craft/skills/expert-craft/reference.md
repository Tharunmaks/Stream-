# Expert Craft reference

## How it works (Flash Experts)

A dense 70B model costs 6 x 70e9 x tokens FLOPs to train: about 4e20 per billion tokens, hundreds of years on a phone CPU. So the model
is sparse, and only what each word touches is computed, read and written.

| Part | Where | What |
|---|---|---|
| Dense backbone | RAM | Llama-style transformer (RMSNorm, RoPE, attention, SwiGLU) from a fixed ladder: width 96, 128, 192, 256, 320, 384 or 512 (0.6M to 39M parameters). Hand-written, gradient-checked backprop; Adam (fused C kernel). |
| Flash Experts | one sparse file on storage | One extra residual layer in the middle of the backbone. Up to ~18 million experts; each is a tiny MLP (e.g. 10 hidden units at width 192 = 3,840 parameters) whose 8-bit weights, per-row scales and optimizer state fill exactly one 4,096-byte page. |
| Router | RAM | Hashed contexts: the last 1, 2 and 3 tokens each pick one expert, so the same context always trains the same page. A small learned sigmoid gate (from the hidden state) weighs each one. `--route pk` uses a learned product-key router instead. |
| Native kernels | compiled on the device | One pass per expert: read the page, dequantise, forward, backward, row-wise Adagrad, 8-bit stochastic rounding, write the page back. Threads over disjoint experts, fast cores only. |

A never-used expert takes no disk space (sparse file); it is created on first use with its up-projection from a shared random pool and its
down-projection at zero, so a new layer is an exact no-op and the file grows with training. The file is opened for random access
(`MADV_RANDOM`), which stops the operating system from reading 16-32 neighbouring pages for every page touched.

Related published work: hashed n-gram memories (Engram, DeepSeek-AI, arXiv 2601.07372; hash n-gram embeddings in the Byte Latent
Transformer, arXiv 2412.09871), product-key memories (Lample et al. 2019) and PEER (He 2024, "Mixture of A Million Experts").
Here the slot is a trainable expert stored in flash and trained in place on the device.

## Measured (4-core x86 Linux PC, TinyStories, the full 70B-parameter expert file, 8 minutes of training each)

| Version | Dense width | Tokens/s | Steps in 8 min | Validation loss | Time to reach v1's 8-minute loss (2.70) |
|---|---|---|---|---|---|
| v1 (NumPy experts) | 192 | ~1,150 | 622 | 2.691 | 463 s |
| v2 (native kernels) | 192 | 3,113 | 1,460 | 2.408 | ~220 s |
| v2 | 128 | 5,397 | 2,515 | 2.314 | 140 s |
| v2 | 96 | 6,534 | 2,973 | 2.323 | 110 s |

- Speed at the same architecture: 2.7x. Time to the same quality with the width the planner picks for short runs: 4.2x faster.
- Where it came from (one 1,024-token step, width 192): expert forward 348 ms -> 17 ms (native kernel + random-access hint),
  expert backward and update 296 ms -> 50 ms, Adam 26 ms -> 7 ms. The dense backbone already runs near the CPU's BLAS speed.
- Quality, same data and same 2,000 steps, dense width 192: dense only 2.611 (perplexity 13.6); with 262,144 hashed experts 2.367 (10.7).
- Flash written: the trainer counts every page rewrite (3 pages per token), but the system merges repeated writes to the same page; on the
  test PC 4.6 GB really reached the disk while 13.7 GB of page rewrites were counted. `phone-status` reports the real amount when the system
  provides it (`/proc/self/io`).
- Not yet measured on a phone. `phone-plan` measures the device it runs on.

## Command reference

`EC = python3 <skill folder>/scripts/expertcraft.py`

| Command | Flags |
|---|---|
| `$EC phone-plan SIZE` | `--minutes M` (600), `--oversubscribe`, `--width W`, `--route hash\|pk`, `--orders 1,2,3`, `--json` |
| `$EC phone-train NAME` | first run: `--data PATH` or `--dataset tinystories`, `--params SIZE`, `--vocab 2048`; every run: `--minutes M`, `--steps N`, `--min-battery %`, `--max-temp C`, `--max-wear-gb GB`, `--min-free-gb GB` |
| `$EC phone-sample NAME` | `--prompt TEXT`, `--tokens N` (80), `--temp T` (0.8) |
| `$EC phone-status NAME` | step, loss, val_loss trend, tokens/s, experts used, flash written, disk used |
| `$EC plan SIZE` / `design SIZE` | free GPU planning for dense models |
| `$EC scaffold SIZE NAME` | PyTorch project + Colab and Kaggle notebooks |
| `$EC run NAME data\|train\|export\|sample [args] [--background]`, `status NAME`, `stop NAME` | run the scaffolded project |

Environment: `EXPERTCRAFT_HOME` (projects, default `~/expertcraft`), `EXPERTCRAFT_CACHE` (compiled kernels, default `~/.cache/expertcraft`),
`EXPERTCRAFT_NATIVE=0` (NumPy only), `OPENBLAS_NUM_THREADS` (default: number of fast cores), `CC` (compiler).

## Project folder

`~/expertcraft/NAME/`: `phone.json` (architecture, step, schedule), `tok.json` (tokenizer), `ids.npy` (tokenised text), `model.npz`
(dense weights, router, Adam state; saved every 2 minutes and on exit), `experts_L.bin` (the expert file), `log.jsonl` (one line per 20 steps).

## Troubleshooting

| Problem | Fix |
|---|---|
| `No module named numpy` | Termux: `pkg install python-numpy`; elsewhere `pip install numpy` |
| Slow, and `native.why()` says no compiler | Termux: `pkg install clang`; Debian: `apt install build-essential`. Check: `python3 -c "import sys; sys.path.insert(0,'<skill>/scripts/phone'); import native; print(native.why())"` |
| `does not support sparse files` | Use internal storage (Termux home), not an SD card or `/sdcard` |
| `disk almost full ... stopped safely` | Free space, or plan a smaller size (`phone-plan` without `--oversubscribe`) |
| Training stops when the screen is off | Termux: `termux-wake-lock`; keep the phone charging |
| Loss is NaN or flat | Start a new project; with your own data use at least a few MB of text |
| `paused: phone is warm` | The `--max-temp` guard; let it cool or raise the limit |
