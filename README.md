# ExpertStream

CPU-only inference for very large Mixture-of-Experts models on an Android
phone (Termux, aarch64, no root, no GPU). The expert weights are streamed
from flash instead of being held in RAM.

Status: **Step 1, expert file format + parallel loader + bandwidth benchmark.**

## Layout

```
src/es_format.h       on-disk expert format (header, tensor descriptors, hash)
src/es_io.{h,c}       parallel chunked loader (pthreads, O_DIRECT, fallback)
tools/es_gen.c        writes synthetic expert files in the real format
tools/es_bench.c      cold-read bandwidth/latency benchmark with cache checks
scripts/dd_baseline.sh  dd cross-check (fio doesn't work in Termux)
docs/expert_format.md format spec and design notes
```

## Step 1: build and benchmark (Termux)

```sh
pkg install -y clang make git coreutils findutils
git clone <this repo> && cd Stream-
git checkout claude/expertstream-moe-android-oegt59
make

# CPU features, needed for the kernel step (please paste the output)
grep -m1 -i features /proc/cpuinfo; grep -m1 -i 'CPU part' /proc/cpuinfo

# Keep data in Termux $HOME (internal /data). NOT /sdcard: that is a FUSE
# mount, where O_DIRECT fails and throughput is much lower.
./es_gen -d ~/es_bench -L 8 -E 64          # 512 experts x 5.9 MiB = 3.2 GB
./es_bench -d ~/es_bench -V                # verify + thread sweep 1..8
./es_bench -d ~/es_bench -t 4 -k 1 -n 200  # single-expert latency
./es_bench -d ~/es_bench -t 4 -c 256       # smaller chunks
./es_bench -d ~/es_bench -t 4 -c 4096      # larger chunks
bash scripts/dd_baseline.sh ~/es_bench 4   # dd cross-check

# Bigger experts (DeepSeek-V3 / Kimi-K2 shape, 13.8 MiB each)
./es_gen -d ~/es_bench_big -H 7168 -I 2048 -L 4 -E 64
./es_bench -d ~/es_bench_big -M 61
```

The dataset (3.2 GB) is bigger than your ~2.9 GB of free RAM, so even when
the filesystem buffers I/O, the page cache can't hold all of it. On top of
that, es_bench evicts every file before each run and checks residency
afterwards with `mincore()`.

Clean up afterwards with `rm -rf ~/es_bench ~/es_bench_big`.

## What the numbers mean (back-of-envelope, I/O only)

Qwen3-235B-A22B shape at Q2_K: 94 MoE layers × 8 experts × 5.9 MiB is
**4.6 GB of expert reads per token** with a cold cache.

| flash BW     | 0% hits | 50% hits | 70% hits | 90% hits |
|--------------|---------|----------|----------|----------|
| 1.8 GB/s     | 2.6 s   | 1.3 s    | 0.78 s   | 0.26 s   |

- A 2–4 GB RAM cache holds about 350–700 of the model's 12,032 experts
  (3–6%). Whether that gets a 50%+ hit rate depends on how skewed real
  routing is. We'll measure it on real traces in a later step and won't
  assume it.
- Compute isn't in that table. About 22B active params × ~2.6 bits is
  ~7 GB of weight traffic per token through the cores, so the matmul time
  is the same order as the I/O. Prefetching overlaps the two, but it doesn't
  make either one disappear.
- The whole Qwen3-235B model at Q2_K is about 75 GB, so it **fits in your
  151 GB of local flash**. No cloud tier is needed for that class of model.
- The cloud tier (20–100 MB/s, high latency) costs 0.15–0.7 s or more per
  expert miss. It must **never be on the per-token critical path**. Only
  fill local flash from it in the background.
