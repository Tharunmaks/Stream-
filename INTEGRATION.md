# integration-main: what was merged and measured (Oct 21 2026, box CPU, 1 thread)
Base: origin/claude/expertstream-moe-android-oegt59 (7ace4f7). Merged: es-moe (contains es-speedup 16a5ee6..19c1352), es-speedup-trainer-app, sparseforge-phone-trainer. No conflicts.
Unified trainer: `es_train finetune ...` (= es_lora, same flags) and `es_train moe gen|train ...` (= moe/es_moe.c, named flags). The es_lora binary is still built, and es_train behaves as es_lora when invoked under that name.
Makefile: -ldl, src/es_amx.c in es_lora, new es_train target. moe/es_moe binary removed from git.
Results:
- es_train finetune, fast mode (ES_AMX=1 ES_LORA_BLAS=1 ES_BLAS=openblas), SmolLM2-135M q8_0, rank 16, seq 128, batch 640, 10 steps, -j 1:
  steps 2-10 at 288-302 tok/s; 273.1 tok/s averaged over all training passes; held-out 4.5752 (same as the es-speedup reference).
- es_train finetune gradcheck: passed.
- es_train moe, 1.07 B-total prototype (L8 E64 d1024 f1024 k2 r8, 64 MB cache, balance 0.01), 10 steps: 301.4 tok/s mean (steps 2..), held-out 3.2949, RSS ~1.04 GB.
- es_mtest: PASS. sparseforge pytest: 7 passed.

## es_train moe --sparse (Oct 24 2026): the Trainer's Train MoE engine, now in this repo
The Trainer (Tharunmaks/Expert-Stream-Trainer, Train MoE page) runs `es_train moe --sparse [--p6] --layers --experts --dim --ffn --batch --seq
--steps --eval-every --data --experts-file --save`. That mode used to exist only in a private prebuilt binary; it is now `moe/es_sparse.c`,
with the Run decoder for its checkpoints in `moe/esd.c` (`make es_train libesd.so`, install both into the same `ES_BIN_DIR`; the
checkpoint format is new, so do not mix this es_train with an older private libesd.so). Model and file formats: `moe/es_sparse.h`.
No ES_BLAS needed. Missing preset expert files are generated (seeded). Test: `python3 scripts/test_sparse.py`.
Measured on a 4-core x86 box (gcc -O3), seq 128:
- tiny (L2 E4 d512 f256, batch 8): ~5,000 tok/s plain, ~6,500 tok/s --p6 x4; held-out 5.65 -> 2.67 in 60 steps on this repo's .md text.
- small 0.27B (L8 E32 d1024 f512, batch 16): 340 tok/s plain, 645 tok/s --p6 x4, RSS 375 MB.
- 2B (L16 E32 d2048 f1024, batch 16): 84 tok/s --p6 x4, RSS 1.8 GB, 3.2 GB of experts decoded per step; held-out 5.99 -> 3.66 in 3 steps.
- plain and --p6 checkpoints are bit-identical; gradcheck passes for every parameter group; libesd.so reproduces the trainer's held-out loss.

## Phone-speed decoding for 70B-total MoE (libesd rewrite, Oct 24 2026)
Goal: 1000 tok/s generation on a phone CPU for a 70B-total model. Dense 70B cannot do it (35 GB read per token at 4-bit);
a 70B-total / 67M-active top-1 MoE can in principle: L16 d2048 f1024, 1043 experts per layer, ternary experts.
- `es_train moe --sparse --bits 2`: ternary experts (17.5 GB for 70B total instead of 35 GB int4). Trains as well as int4 on
  the tiny preset (held-out 2.674 vs 2.670 after 60 steps).
- `--router-rank 64`: low-rank router. With 1043 experts a full router reads 34 MB per token (twice the expert); rank 64 reads 3.2 MB.
- libesd: experts repacked output-major once per cache fill, int8 activations, int8 routers and head, AVX2 / NEON / NEON-SDOT
  kernels (runtime-picked, bit-identical to scalar: `make esd_ktest`, also run under qemu-aarch64 for both ARM paths), worker
  pool over output rows (ES_RUN_THREADS, default 4). Still reproduces the trainer's held-out loss (int4 2.8731 = 2.8731,
  ternary + low-rank router 2.8405 vs 2.8399).
- `make esd_bench`: decode tok/s with every expert in RAM and a different expert every token (no CPU-cache reuse).

Measured here (4-core Xeon 2.1 GHz, AVX2, ~27 GB/s all-core read), 67M active per token:
| shape | experts | router | threads | tok/s |
|---|---|---|---|---|
| L16 E64 (old libesd: scalar int4, float router, 1 thread) | int4 | full | 1 | 55 |
| L16 E64 | int4 | full | 4 | ~430 |
| L16 E64 | ternary | full | 1 | 228 |
| L16 E64 | ternary | full | 4 | ~520-560 |
| **L16 E1043 (70.0 B total)** | ternary | full | 4 | 438 (router fits this box's 260 MB L3; a phone would be slower) |
| **L16 E1043 (70.0 B total)** | ternary | rank 64 | 4 | **~540** |
The 70B rows keep 64 experts per layer in the file (`--stored 64`; this box has 15 GB of RAM): each token still reads a full
1 MB expert per layer from RAM, and routers and LoRA are full size.

Phone, by estimate only (not measured): a flagship phone has about 2x this box's memory bandwidth and SDOT int8 cores, and the
kernel here is compute-bound (12 of 27 GB/s used), so 500-1200 tok/s is the expected range. Measure in Termux with
`make esd_bench && ./esd_bench --experts 1043 --stored 64 --router-rank 64 --bits 2 --threads 4`.
Still needed for a real 70B on a phone: 24 GB RAM (17.5 GB experts + 1.6 GB LoRA), a pre-packed expert file that can be mmap'd
without repacking or a second copy, and trained experts (the frozen random experts only give LoRA-level quality).
