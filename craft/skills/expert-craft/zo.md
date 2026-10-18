# Training a 200B model inside a phone: what is built, what is measured

The plan this answers: treat the phone's flash (256 GB to 1 TB) as the weight store and its RAM (12 to 24 GB) as a passing buffer.
It has five parts: an ultra-sparse 1.58-bit MoE, a flash-to-RAM expert streamer, layer-wise zeroth-order updates (MeZO) on small
adapters, and federated merging of what each device trained.

The memory arithmetic behind it is right. Full 16-bit training of 200B parameters with Adam needs about 16 bytes per parameter:
weights, gradients, fp32 master weights and two Adam moments, or 3.2 TB. A phone can hold none of that.

## The parts, one by one

| Part of the plan | In ExpertStream | Status |
|---|---|---|
| 1.58-bit (ternary) sparse MoE base model | The engine runs ternary weights (TQ1_0, TQ2_0), 1-bit formats (IQ1_S, IQ1_M, Q1_0, IQ1_XXXS) and every major MoE architecture. | Runtime: done. Model: **no 200B ternary MoE exists to download.** Pretraining one takes trillions of tokens on data-centre hardware; a phone can adapt one, not create it. The closest real files are 1-bit quantizations of existing MoEs (Qwen3.5 397B in 107 GB, DeepSeek R1 in 162 GB). |
| Flash-to-RAM expert streamer, contiguous experts | Expert packs (each expert contiguous, parallel `O_DIRECT` reads), GGUF experts read in place, `MADV_RANDOM` (sequential readahead wastes reads when experts are scattered), next-layer prefetch, experts downloaded on demand. | Done earlier. |
| **New:** stream each layer once for many sequences | `es_engine_features_multi`: all sequences go through layer 1, then all through layer 2, and so on (FlexGen-style). Each layer's experts are read once for the whole group instead of once per 64 tokens. | Done. Bit-identical outputs on every architecture (Llama 3, Mixtral, OLMoE, gpt-oss, DeepSeek V3 MLA, Qwen3.5 DeltaNet). OLMoE forced to stream: **10.4x fewer bytes read** for the same 2,000 tokens. |
| Layer-wise zeroth-order updates (MeZO) on adapters | `es_graft zo`: a rank-r adapter after every layer, trained from two forward passes per step. The random direction comes from one seed and is never stored. Both passes share one layer-by-layer pass, so each expert is read once per step. Memory: the model's inference memory plus the adapter. | Done; measured below. |
| Federated / split merging | Each step is fully described by `(seed, slope, lr)`: 16 bytes. `es_graft zo-merge START A.zolog B.zolog ...` replays other devices' steps. No weights and no data leave a device. | Done. A replayed log reproduces the trained adapter **bit for bit**. |

## Measured: MeZO adapters vs the graft (4-core PC, CPU)

Same data (TinyStories, 20,000 tokens), same model (SmolLM2-135M); held-out perplexity at the start: 12.35.

| Method | Settings | Held-out perplexity | Time |
|---|---|---|---|
| **Graft** (exact gradients on top; the model reads the data once) | defaults | 12.35 -> **10.21 (-17.3%)** | 74 s (70 s reading, 4 s training) |
| MeZO adapters, plain (as published) | rank 8, lr 1e-3, 300 steps | 12.35 -> 12.11 (-1.9%) | 17 min |
| MeZO adapters, plain | rank 8, lr 3e-3 | diverged (perplexity 7,741) | |
| MeZO adapters, normalized slope (default) | rank 8, lr 1e-3, 200 steps | 12.35 -> 12.25 (-0.8%) | 11 min |
| MeZO adapters, normalized | rank 8, lr 3e-3, 200 steps | +0.3% (noise) | 11 min |
| MeZO adapters, normalized | rank 1, lr 3e-3 / 1e-2, 200 steps | +0.9% / diverged (+87%) | 10 min each |

Each MeZO step ran both perturbed versions of 4 examples in one layer-by-layer pass: about 3.1 s per step here. The held-out set
is small (8 examples, 1,016 tokens), so differences under about 1% are noise. Plain MeZO on these 276,480 adapter numbers learns at
lr 1e-3 and diverges at 3e-3. Dividing the slope by its running size (the default) keeps it from diverging at moderate step sizes,
but does not make it learn faster.

## What this means for 200B on a phone

* **Memory is solved.** MeZO needs only the memory of running the model, and the layer-by-layer pass reads each expert once per step.
  A 200B MoE that runs on the phone can be trained this way.
* **Time is the problem.** Every MeZO step is two passes over its batch through the whole model. For a 200B model on a phone that is
  seconds to minutes per step, depending on how much of the model the batch touches and on storage speed. MeZO is known to need
  thousands to tens of thousands of steps (Malladi et al. used up to 100,000), and in our measurements it learned an order of
  magnitude slower than the graft. `es_graft plan` prints the measured cost of one step on your device.
* **The practical route today is the graft.** The big model reads your data once and never again; the graft trains with exact
  gradients in seconds. MeZO adapters remain available (`es_graft zo`) for changing the model's inside and for federated training,
  where their 16-byte steps shine.

## Commands

```sh
./es_graft zo -g MODEL.gguf --data notes.txt -o my.esadapt --steps 1000 [--rank 8] [--lr 1e-3] [--batch 4] [--min-battery 30] [--resume]
./es_chat -g MODEL.gguf -Z my.esadapt                                 # or es_serve --adapter my.esadapt
# federated: every device starts from the same --init-seed and uses its own --seed (its own random directions); then anyone can merge
./es_graft zo-merge my.esadapt.start phoneA.esadapt.zolog phoneB.esadapt.zolog -o merged.esadapt [--scale 0.5]
./es_graft multicheck -g MODEL.gguf --data text.txt --adapter        # layer by layer = one sequence at a time, bit for bit
```
