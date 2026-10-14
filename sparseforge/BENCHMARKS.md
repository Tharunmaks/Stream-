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
