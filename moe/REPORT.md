# es_moe prototype: measured results (1 thread, shared box with load ~7; Lite PID 127257 running)
Experts: frozen random-init ternary (2 bits/weight) on disk, LRU cache, pread + POSIX_FADV_DONTNEED to stop the page cache hiding reads.
Trainable: byte embeddings, attention, router, per-expert LoRA (lazy Adam on touched experts). Data: data.txt (byte level), seq 128.
Experts were never trained (random init), so loss drops come from attention, router and LoRA only. See logs/ for every step.
