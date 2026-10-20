# ExpertStream

**Website (no install):** open `docs/index.html` on GitHub Pages
(Settings → Pages → Branch `claude/expertstream-moe-android-oegt59`, folder `/docs`),
or open `web/index.html` from `http://127.0.0.1` / any normal browser tab. In **Models**
search Hugging Face, download a GGUF (saved on the device, resumable), or pick a file from the
phone; then **Chat** or use the **CLI** tab. The engine runs in the page as WebAssembly.
It needs a normal browser tab: pages inside other apps (sandboxed viewers) block downloads
and workers; **Testing → Can this page run models?** says exactly what is blocked.

**Faster, native (Termux):** one command installs everything, downloads a
model and opens the same app backed by the native engine
(`es_serve -g model.gguf` or `-m pack`):

```sh
curl -fsSL https://raw.githubusercontent.com/tharunmaks/stream-/claude/expertstream-moe-android-oegt59/scripts/setup.sh | bash
```

**ExpertStream Instant: fast answers on a phone or PC, in any browser or app.** One local server (`es_serve`) speaks the OpenAI API
(`/v1/chat/completions`, `/v1/completions`, `/v1/models`, streaming), the Ollama API (`/api/chat`, `/api/generate`, `/api/tags`) and the web app's
own protocol. Point any app that lets you set an OpenAI base URL or an Ollama server at it; the website (any browser, also when opened from
GitHub Pages) connects to it by itself.

```sh
make -s && ./es_serve -g ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf     # or: curl -fsSL .../scripts/setup.sh | MODEL=fast bash
# OpenAI apps: base URL http://127.0.0.1:8080/v1 (any API key)   Ollama apps: http://127.0.0.1:8080   web app: http://127.0.0.1:8080
./es_serve -g model.gguf --lan                                          # other devices too; prints an API key they must send
```

Why answers start fast:
- **Prefix reuse.** Chat apps resend the whole conversation every turn. The engine keeps what it has already read and reads only the new part;
  up to 4 other conversations stay parked in RAM, so side requests ("write a title for this chat") do not throw the main one away.
- **Prompt cache on disk.** An expensive prompt start (a long system prompt or document) is saved losslessly in `~/.expertstream/kv` in the
  background; any later request that starts the same way, also after a restart, loads it instead of reading it again (2 GB budget, oldest first).
- **Faster prompt reading.** In batches, each weight row is unpacked once for all tokens instead of once per token (AVX2 and ARM SDOT kernels
  for Q4_0, Q5_0, Q4_K, Q6_K). Results are bit-identical (`./es_mtest`); 1.8x faster prompt reading for Qwen2.5 0.5B and Llama 3.2 1B.
- **Turbo** speculative decoding and **streaming** of every piece as soon as it exists.

Measured on a 4-core x86 PC, a chat app session (840-token system prompt, the conversation resent each turn, a title request after each
answer), time until the first word appears:

| Model | Turn 2+ without reuse | Turn 2+ with Instant | First turn | First turn after a restart |
|---|---|---|---|---|
| Qwen2.5 0.5B Q4_K_M | 15.6 s (old engine) / 8.1 s | **0.29 s** | 7.6 s (was 13 s) | **0.12 s** |
| SmolLM2 135M Q4_K_M | 6.1 s | **0.23 s** | 4.7 s | **0.075 s** |
| OLMoE 1B-7B Q2_K (experts streamed) | about 5 min (like turn 1) | **3.4-4.0 s** | 305 s | **0.44 s** |

Reuse never changes an answer: with greedy sampling the answers with and without reuse, and after loading from disk, are identical
(`python3 scripts/test_instant.py MODEL.gguf`, 21 checks: OpenAI and Ollama clients, streaming, stop strings, context trimming, CORS and
Private Network Access, DNS-rebinding refusal, LAN key). Honest limits: the first time a prompt is seen it is read at the engine's normal
speed (big mixture-of-experts models read prompts slowly), it serves one request at a time, and Safari may block a https website from reaching
127.0.0.1. Websites other than local pages and this project's site are refused unless started with `--cors ORIGIN`.

**Graft: teach a 200B model your data, on your phone.** Fine-tuning a 200-billion-parameter model the usual way needs terabytes of memory
and data-centre GPUs; no phone can do that at any speed. A graft takes another way. The big model (any GGUF this engine runs, Qwen3 235B
and DeepSeek R1 included) stays **frozen**, and a small graft on its final hidden state learns your facts, names, wording and style.
The graft has three parts: a low-rank network, an exact n-gram memory of your text that can propose words the model never would, and a
learned gate that decides when to trust it. Built-in general "anchor" text teaches it to leave everything else alone.

```sh
make es_graft es_chat
./es_graft plan  -g MODEL.gguf --data notes.txt               # measured on this device: hours to read, minutes to train
./es_graft read  -g MODEL.gguf --data notes.txt -o my.esfeat  # the big model reads your data ONCE (resumable)
./es_graft train my.esfeat -o my.esgraft                      # the big model is not used again: thousands of tokens/s on CPU cores
./es_chat  -g MODEL.gguf -A my.esgraft                        # or: ./es_serve -g MODEL.gguf --graft my.esgraft (OpenAI/Ollama API)
```

Measured on a 4-core PC (30 made-up companies with 4 facts each). Questions were asked in a wording the graft never saw, and answered
greedily. Right facts, base model -> with the graft: Qwen2.5-0.5B **0 -> 119** of 120, OLMoE 7B (mixture of experts) **0 -> 118** of 120.
General questions outside the data: 23 -> 23 and 24 -> 24 of 24. Plain text: held-out perplexity -17%. Training: 9,500 and 4,800 tokens/s;
reading through the frozen model: about 90 and 9 tokens/s. Models above 7B were not run end to end here (this machine's disk was too
slow to stream a 20B model's experts). The loss is exact (only
the candidates' scores change), and `es_graft check` shows the engine reproduces it during generation (largest difference per position
about 1e-6). Honest limits: reading the data through a 200B model on a phone takes hours (`plan` measures it), and a graft teaches
facts and style, not new abilities. Details: [`craft/skills/expert-craft/graft.md`](craft/skills/expert-craft/graft.md); test:
`python3 scripts/test_graft.py MODEL.gguf`.

**ExpertStream Cloud & OmniRoute Gateway.** Pool your own machines into one OpenAI-compatible endpoint with free `sk-expert-...`
keys: `gateway/omni_router.py` (OpenAI API with streaming, prompt compression, response cache, scoring router with failover),
`cloud/worker_agent.py` (one command per machine, works behind home routers), an MCP server (`mcp/expertstream_mcp.py`) and a
dashboard. Every node runs the native engine. Details and limits: [`docs/cloud.md`](docs/cloud.md); test: `python3 scripts/test_cloud.py MODEL.gguf`.

**Faster on PCs; streaming tuned for phones.** On a 4-core x86 PC without a GPU, Qwen3-30B-A3B (30B parameters, Q2_K) generates
8.0 words/s from RAM (it was 3.65; llama.cpp: 7.6), after new AVX2 kernels for the formats big models use (Q2_K, Q3_K, TQ1_0/TQ2_0, all
IQ formats, MXFP4: 4-15x faster each, every one checked against exact float math). With phone-like storage (1 GB/s) and only 1-3 GB of
RAM for experts it streams at 3.8-6.4 words/s. The expert cache now sizes itself to the free RAM, and `-R 0.2` (es_serve
`--cache-route 0.2`) prefers experts already in RAM: +4-13% speed, perplexity unchanged within noise. Not measured: a real phone (ARM
uses its own NEON kernels).

**Tested on 218 models (`scripts/test_200.py`).** 105 small random models covering 14 designs, compared logit by logit with
PyTorch (Kimi K3: with llama.cpp); 43 real GGUF files in many quantizations, compared with llama.cpp; and 70 real giant files from 14B
to 2.4T parameters (9.1 TB in all), loaded and run from their headers alone (`es_get --skeleton`, every weight zero: this proves
support, not answer quality). All 218 pass. The run found and fixed a Qwen3 q-norm bug and added Kimi K3. Details: [`docs/test-200.md`](docs/test-200.md).

**Layer-by-layer reading and training inside the model (MeZO).** For models bigger than RAM, `es_graft read` and `es_graft zo` run
groups of examples through the model layer by layer (as FlexGen does): each layer's experts are read from flash once for the whole group.
Outputs are bit-identical to one sequence at a time on every architecture, and the bytes read fell 10.4x for OLMoE (`es_graft multicheck`).
`es_graft zo` trains small adapters inside every layer with MeZO: two forward passes per step, no backpropagation, the memory of running the
model. Every step is 16 bytes (seed, slope, step size), so devices can merge their training by replaying each other's step logs
(`es_graft zo-merge`, bit-identical). Measured honestly: on SmolLM2-135M it lowered held-out perplexity by 1.9% in 17 minutes (1.6% after
83 minutes with the safe default, levelling off after about 1,000 steps), where the graft gave 17.3% in 74 seconds. Details: [`craft/skills/expert-craft/zo.md`](craft/skills/expert-craft/zo.md).

**LoRA with exact gradients** (`es_lora`): fine-tunes q, k, v, o, gate, up and down of every layer of a frozen GGUF model on the
CPU, streaming the model layer by layer (forward, then recompute and backward per layer) so it needs about one layer's memory.
Gradients checked against finite differences; the trainer matches the engine to 1e-4. On the TinyStories benchmark above:
held-out perplexity 12.29 -> 8.31 (-32%) in 18 minutes, against -17% for the graft and -1.6% for MeZO. Then
`es_chat -g MODEL.gguf --lora NAME.eslora`, or export for llama.cpp. Details: [`craft/skills/expert-craft/lora.md`](craft/skills/expert-craft/lora.md).

**Expert Craft: train a model with up to 70 billion parameters on your phone, free, as a skill for your AI (no MCP).**
Claude Code (PC or Termux): `/plugin install expert-craft --marketplace Tharunmaks/Stream-`. Any agent that reads a skills folder:
`curl -fsSL https://raw.githubusercontent.com/Tharunmaks/Stream-/claude/expertstream-moe-android-oegt59/scripts/install-skill.sh | sh`.
Or tell your AI: *"Install the Expert Craft skill from github.com/Tharunmaks/Stream- (folder craft/skills/expert-craft) and train a 70B model on this phone."*

How: a small dense model stays in RAM; up to 70 billion more parameters live in ~18 million page-sized experts (one 4 KB page each, 8-bit) in a
sparse file. The last 1, 2 and 3 tokens each pick an expert by hash, and only those 3 pages per word are read, trained and written back by native
C kernels the device compiles for itself (NumPy fallback). v2 measured on a 4-core PC with the full 70B-parameter file: 6,534 tokens/s (v1: ~1,150),
and v1's 8-minute quality reached in 110 s instead of 463 s. Experts lower validation perplexity by 21% against the same dense model alone.
Honest limits: each word uses under 0.005% of the parameters, quality follows the tokens processed and the dense model rather than the file
size, training rewrites flash pages (capped by `--max-wear-gb`), and it is not yet measured on a phone. Details: [`craft/README.md`](craft/README.md).

**Speed (measured, 4-core x86 test PC, SmolLM2-135M, greedy, `es_chat -j 4`, default build, no `-march=native`)**:
Q4_K_M 107-111 tok/s, Q4_0 103-114 tok/s (127-138 with Turbo on), Q8_0 about 90 tok/s (memory-bandwidth bound).
The same file in the browser (WebAssembly, relaxed-SIMD build, 3 threads) runs at about 42 tok/s (Q4_0) / 35 tok/s (Q4_K_M)
on that PC. Phone numbers differ: run `bash scripts/speedtest.sh` in Termux (it builds, downloads both files, prints a table)
and use `es_chat -j 0` (the default: one thread per fast core, pinned; never more threads than usable cores).
How: runtime-dispatched AVX2/FMA integer kernels (Q8_0/Q4_0/Q5_0/Q4_K/Q6_K, 4 rows per pass; `ES_NO_AVX2=1` turns
them off), NEON SDOT on ARM, batched prompt processing, and **Turbo** (prompt-lookup speculative decoding: the engine
guesses the next few words from the text so far and checks them in one batched pass; greedy output stays identical,
`-S 0` or `ES_SPEC=0` turns it off, it switches itself off when it does not help).

**ExpertStream Lookahead** (native engine): big mixture-of-experts models stream experts from storage, and
plain streaming waits for storage on every layer. Lookahead (default on in `es_chat`/`es_serve`) loads all
missing experts of a layer in parallel, runs the next layer's router early to guess and preload its experts while the
CPU computes (`-L 1`), can skip experts the router barely uses (`-E 0.15`, a speed/quality knob, off by default),
and remembers hot experts for a warm start next run (`-W 1`, stored in `<model>.esprof`). On OLMoE with storage
simulated at 500 MB/s per reader (`ES_SIM_FLASH_MBPS=500`): 1.46 -> 2.94 tok/s, experts found in RAM 49% -> 91%,
identical output. It removes waiting, not reading: storage speed still bounds models that need many GB per word.

Rebuild the site: `source emsdk_env.sh && bash scripts/build_web.sh` (`UI_ONLY=1` reuses the last
WebAssembly build). Sources: `web/src/{template.html,style.css,js/,workers/}`.

Supported model families: OLMoE, Qwen3 / Qwen3-MoE, Qwen2/2.5, Llama 3-style (byte-level BPE),
quantisations F32/F16/Q4_0–Q8_0/Q2_K–Q6_K. Not yet: SentencePiece tokenizers (Mistral, Llama 2,
Gemma), IQ/BF16/MXFP4 quants, MLA attention (DeepSeek, Kimi), split GGUF inside the page.

CPU-only inference for very large Mixture-of-Experts models on an Android
phone (Termux, aarch64, no root, no GPU). The expert weights are streamed
from flash instead of being held in RAM.

Status:
- **Step 1 (done):** expert file format, parallel loader, flash benchmark.
  Measured: 1.75 GB/s cold with 4–6 readers (real O_DIRECT, 0% cached),
  28 ms per 8-expert layer, 5.5 ms for a single 5.9 MiB expert.
- **Step 2 (done):** GGUF-compatible Q2_K NEON SDOT kernel. Measured on
  4x Cortex-A78: 9.8 GB/s of weights (memory read tops out at 13.8 GB/s),
  so about 0.72 s/token of compute for the Qwen3-235B shape. Adding the
  A55 cores makes it slower, so compute uses only the big cores.
- **Step 3 (done):** `es_run`, an end-to-end decode loop: hot-expert
  cache, router-ahead prefetch, readers on the little cores, compute on the
  big cores. Weights are synthetic and routing is simulated.
- **Step 4 (done):** real models. `scripts/hf_get.sh` downloads GGUF
  files from Hugging Face. `es_import` turns any GGUF MoE (single file or
  split) into a pack: `core.gguf` plus one file per expert. Tested
  byte-for-byte against the official `gguf` reader on 3 real models. `-x`
  imports only chosen experts, which is how skill packs (coding, chat) will
  be stored.
- **Step 5 (current):** real text. `es_chat` runs OLMoE-1B-7B from a pack:
  tokenizer, attention, router and streamed experts. With float activations
  its logits match an independent float64 NumPy reference to about 1e-6
  over 6 tokens. With the default 8-bit activations (like llama.cpp) the
  top-1 token matches llama.cpp at 14 of 15 positions.

## Layout

```
src/es_format.h       on-disk expert format (header, tensor descriptors, hash)
src/es_io.{h,c}       parallel chunked loader (pthreads, O_DIRECT, fallback)
src/es_cpu.{h,c}      core types, max clocks, pinning, dotprod detection
src/es_q2k.{h,c}      GGUF-compatible Q2_K weights, Q8_K activations, scalar ref
src/es_q2k_neon.c     NEON + SDOT dot product (armv8.2-a dotprod)
src/es_compute.{h,c}  pinned compute pool (spin, then sleep)
src/es_cache.{h,c}    hot-expert RAM cache, frequency + recency eviction
tools/es_gen.c        writes synthetic expert files in the real format
tools/es_bench.c      cold-read bandwidth/latency benchmark with cache checks
tools/es_kbench.c     kernel correctness + compute throughput per core type
tools/es_run.c        end-to-end decode loop (cache + prefetch + kernels)
tools/es_import.c     GGUF MoE -> ExpertStream pack (core.gguf + expert files)
src/es_gguf.{h,c}     GGUF v2/v3 reader (metadata, tensor table, splits)
src/es_quant.{h,c}    dequantize + matvec for F32/F16/Q4_0..Q8_0/Q2_K..Q6_K
src/es_tok.{h,c}      byte-level BPE tokenizer read from GGUF metadata
tools/es_chat.c       text generation: core in RAM, experts streamed + cached
tools/es_qtest.c      dumps dequantized tensors for checking
tools/es_serve.c      ExpertStream Instant: OpenAI + Ollama + web app API, prefix reuse, CORS (127.0.0.1 unless --lan)
tools/es_json.h       small JSON reader for the server
tools/es_mtest.c      checks the batched kernels are bit-identical to one-at-a-time
scripts/test_instant.py  end-to-end tests of the server
web/index.html        the app (generated, self-contained, engine embedded)
web/src/              app source: template.html, style.css, js/ modules, workers/
web/es_wasm.c         WebAssembly entry points (Emscripten, runs in a Worker)
src/es_engine.{h,c}   the engine shared by es_chat, es_serve and the browser
docs/index.html       copy of web/index.html for GitHub Pages
scripts/setup.sh      install, build, download, import, start the app
scripts/hf_get.sh     resumable Hugging Face download, all parts of a split
scripts/dd_baseline.sh  dd cross-check (fio doesn't work in Termux)
craft/skills/expert-craft/  Expert Craft skill: SKILL.md, reference.md, scripts/ (CLI, Flash Experts, C kernels, trainer)
.claude-plugin/       plugin marketplace (Claude Code: /plugin install expert-craft --marketplace Tharunmaks/Stream-)
scripts/install-skill.sh  installs the skill for any agent that reads a skills folder
scripts/test_flash_experts.py  tests: gradients, 8-bit store, resume, 70B sparse file
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

## Step 2: kernel benchmark (Termux)

```sh
cd ~/Stream- && git pull && make
grep -E 'processor|CPU part' /proc/cpuinfo | paste - - | awk '{print $3, $NF}'
cat /proc/self/status | grep Cpus_allowed_list
./es_kbench                 # topology, correctness, single-core + scaling
./es_kbench -S -m 128       # same with the scalar kernel, for comparison
```

Keep the screen on and Termux in the foreground. Android moves background
apps onto the little cores, which would make the numbers look worse than
they are.

`es_q2k` uses exactly the llama.cpp `block_q2_K` layout. It was checked
bit for bit against ggml's `dequantize_row_q2_K`, so expert files can be
cut from existing GGUF Q2_K models.

## Step 3: run the engine (Termux)

```sh
cd ~/Stream- && git pull && make
rm -rf ~/es_bench && ./es_gen -d ~/es_bench -L 8 -E 128   # 6.3 GB, 128 experts like Qwen3
bash scripts/step3.sh                            # all four runs below, saved to step3_results.txt
./es_run -d ~/es_bench                           # defaults
./es_run -d ~/es_bench -q -p 0 -r 0              # worst case: no prefetch, no locality
./es_run -d ~/es_bench -q -r 0.5 -p 0.8 -C 1536  # optimistic routing, bigger cache
./es_run -d ~/es_bench -q -a 0                   # without the attention cost
```

The model has 94 layers but the dataset only 8. Model layer `l` reads file
layer `l % 8`, but the cache key is the model layer, and reads are O_DIRECT,
so reusing files can't produce fake cache hits.

`-r`, `-z` and `-p` are **assumptions**, not measurements. Real values need
real routing traces (later step).

## Step 4: import a real model (Termux)

```sh
cd ~/Stream- && git pull && make
bash scripts/hf_get.sh bartowski/OLMoE-1B-7B-0924-Instruct-GGUF Q2_K.gguf ~/models
./es_import -i ~/models/OLMoE-1B-7B-0924-Instruct-Q2_K.gguf          # inspect
./es_import -o ~/packs/olmoe ~/models/OLMoE-1B-7B-0924-Instruct-Q2_K.gguf
```

What `es_import -i` reports for this model: 6.9B parameters, of which
0.48B are core (0.23 GB, kept in RAM) and 6.44B are routed experts
(2.33 GB, streamed). Each token needs 0.29 GB of expert weights.

Note: "Q2_K" files usually mix quant types. In this one the down
projection is Q3_K, so the engine needs a Q3_K kernel as well as Q2_K.
Generating real text is the next step: tokenizer, attention, router and
the remaining kernels.

Disk: the GGUF and the pack each take the model's full size. Delete the
GGUF after importing if space is tight.

## Step 5: chat with a real model (Termux)

```sh
cd ~/Stream- && git pull && make
./es_chat -m ~/packs/olmoe -p "Write a short Python function that checks if a number is prime."
./es_chat -m ~/packs/olmoe          # interactive chat, /exit to quit
./es_chat -m ~/packs/olmoe -r -t 0 -p "The capital of France is" -n 32   # raw text, greedy
```

Options: `-n` max new tokens, `-t` temperature (0 = greedy), `-C` expert
cache in MB (default 1024), `-c` context length (default 2048, 537 MB of
RAM for this model), `-T file` writes the experts picked for every token
and layer (input for building skill packs).

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


## Legal and trust checks (the 20 must-haves before launch)
`scripts/legal_checklist.py` holds the 20 items (privacy policy, terms, refund and cookie policies, consent banner, form consents, data minimisation, third-party SDK
audit, dark patterns, hidden fees, fake reviews, unsupported claims, alt text, colour contrast, keyboard navigation, business details, age consent,
unsubscribe link, licensed fonts and images, data deletion) with why, how and how to verify. `python3 scripts/legal_checklist.py audit FOLDER` scans a project folder and returns
PASS / FAIL / NEEDS-REVIEW / N/A per item with evidence. 15 agents in the "Legal & trust" team of the website's agent library help with the items. This is an
engineering checklist, not legal advice.
The website now has `legal.html` (privacy, terms, refunds, cookies and storage, data deletion, accessibility, operator, licences, and the status of all 20 checks on this
site), a privacy notice, Settings > Privacy > "Delete all my data", consent text on the Hugging Face sign-in, a skip link, keyboard fixes and contrast fixes.
Fill `web/legal/operator.json` (e-mail, postal address, governing law, code licence) and run `python3 scripts/build_legal.py` to publish your own details.
