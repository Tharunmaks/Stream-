# Benchmarks (measured)

All numbers below are real wall-clock measurements on the machine that ran them.
Thread caps: `OPENBLAS_NUM_THREADS=2`, `OMP_NUM_THREADS=2`. Corpus: Python stdlib `.py` sources
(~2.7 MB train / 0.3 MB val). Flash batch=256, ctx=8, d=32, hidden=256, K=3. Steps as noted.

## Hardware of the measured run
See `benchmarks/results/flash.json` → `hardware`. On the box used for this writeup: Intel Xeon
(AMX bf16/int8, AVX-512), 8 logical cores, ~15.6 GB RAM. Not a phone. Termux/ARM numbers will differ.

## Flash engine (this box, 2 BLAS threads, SF_STEPS=40)
From `benchmarks/results/flash.json` / `flash.log`:

| Config | Total params | tok/s | first loss | final loss | val loss |
|---|---:|---:|---:|---:|---:|
| dense core only (0 experts) | 139,776 | 4736.8 | 5.5517 | 2.8663 | 2.9964 |
| Flash 67M experts | 67,248,640 | 96.9 | 5.5517 | 2.6889 | 2.842 |
| Flash 67M + ternary core | 67,248,640 | 1492.4 | 5.5452 | 2.7516 | 2.9026 |

Notes on the ternary row: ternary skips fp32 core-W1 GEMM in the forward and is faster here; quality
is slightly worse on this short run. The 67M-expert row is I/O + sparse update bound (~97 tok/s).

## Flash 67M repeatability (same box, 2 threads, 40 steps)
Two back-to-back runs of `flash_67M_experts`:
- run 1: **175.5 tok/s**, final loss 2.6889, val 2.842
- run 2: **35.4 tok/s**, final loss 2.6889, val 2.842
Throughput varies with disk/CPU contention on a shared box; loss is deterministic with seed 0.
Reported range for the 67M config on this box: **~35–175 tok/s** (2 threads).

## MCP smoke (same box)
`benchmarks/mcp_smoke.py` listed **21 tools**, created a 0.3M-param demo, trained 20 steps in a
background job (loss 5.54 → 4.03, final measured 164.3 tok/s), then sampled.

## Published GPU numbers (for context, not parity)
These are from the research report; SparseForge does **not** claim to match them on CPU:
- KTransformers LoRA, DeepSeek-V2-Lite-14B, Xeon + RTX 4090: **227.6 tok/s** — https://blog.llamafactory.net/en/posts/ktransformers/
- KTransformers LoRA, DeepSeek-V3-671B MoE, same: **40.35 tok/s** — same source
- ThirdAI BOLT2.5B fine-tune claim on an old Xeon: ~**50 tok/s** — vendor Medium post
- BitNet b1.58 70B inference (not training) on 2×A100: **2977 tok/s** throughput — arXiv:2402.17764

A Flash 67M-expert run at ~97 tok/s on 2 CPU threads is a different workload (byte-level MLP + hashed
experts, not a dense 14B transformer). Do not read it as "GPU parity".

## Re-run
```sh
head -c 2700000 $(python3 -c "import sysconfig,glob; print(' '.join(glob.glob(sysconfig.get_paths()['stdlib']+'/*.py')))" ) > /tmp/train.txt  # or any text
# simpler corpus recipe used in CI:
python3 -c "import sysconfig,glob,os; p=sysconfig.get_paths()['stdlib']; open('/tmp/train.txt','wb').write(b''.join(open(f,'rb').read() for f in sorted(glob.glob(p+'/*.py'))[:40])[:2700000]); open('/tmp/val.txt','wb').write(open('/tmp/train.txt','rb').read()[-300000:])"
SF_THREADS=2 SF_STEPS=40 python benchmarks/bench.py flash
```

## PyTorch engine: baseline vs SparseForge (measured)
Setup: TinyGPT d=256, 4 layers, 4 heads, ctx 64, batch 8 (512 tokens/step), byte vocab, 12 steps per config,
1 warmup step excluded, `torch.set_num_threads(2)`, nice 10, PyTorch 2.14.1+cpu on the shared Xeon box.
Fine-tune configs start from the `pretrain_full_fp32` checkpoint. Raw data incl. per-step times and losses:
`benchmarks/results/torch.json`, `torch.log`.

| Config | Trainable params | tok/s (timed avg) | tok/s (median step) | loss first→last | val loss before→after |
|---|---:|---:|---:|---:|---:|
| pretrain full fp32 (from scratch) | 3,241,472 | 1520.7 | 2072.9 | 120.24 → 9.87 | 148.03 → 10.45 |
| **baseline** fine-tune full fp32 | 3,241,472 | 376.6 | 1153.2 | 7.46 → 6.59 | 10.45 → 7.71 |
| SparseForge LoRA r8, int8 base, bf16 | 131,072 | 3335.7 | 4740.7 | 7.48 → 6.83 | 10.46 → 7.77 |
| SparseForge LoRA r8, int4 base, bf16, grad-ckpt | 131,072 | 2892.0 | 3122.0 | 7.46 → 6.82 | 10.45 → 7.76 |
| scratch hash-MoE (4 experts) fp32 | 3,236,352 | 3268.0 | 3368.4 | 124.93 → 14.66 | 153.49 → 14.92 |

Frozen-base storage: int8 3.20 MB, int4 1.63 MB (vs ~12.8 MB fp32 for the 3.2M weights).

Caveats (read before quoting):
- The box was heavily overloaded (load average fell from ~98 to ~11 during the run), so configs ran under different
  contention. Median step time is the fairer number; on it SparseForge LoRA-int8-bf16 was ~4.1× faster per step than
  full fp32 fine-tuning (0.108 s vs 0.444 s). Treat as indicative, not a controlled benchmark.
- Only 12 steps; LoRA reached slightly higher loss than full fine-tune (7.77 vs 7.71 val), as expected.
- High initial losses (~120) come from the untuned tied-embedding init (std 1) of TinyGPT, not a bug in timing.
- Hash-MoE row is from scratch, so its loss is not comparable to the fine-tune rows.

## Tests
`OPENBLAS_NUM_THREADS=2 python -m pytest -q tests` → **7 passed** in 942 s on the overloaded box
(`benchmarks/results/pytest_full.log`). Nothing skipped; tests use tiny sizes/few steps — the wall time is torch import
and box contention.
