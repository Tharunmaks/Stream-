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
