# Graft: teach a frozen big model your data on a phone

## What it is (and what it is not)

Training a 200B model the normal way means a backward pass through all of it and optimizer state for every weight: terabytes of
memory and data-centre GPUs. **That is impossible on a phone, and no software trick changes it.** A phone CPU does well under one TFLOPS;
a training GPU does hundreds.

A graft avoids that cost:

* The big model (any GGUF the ExpertStream engine runs: Qwen3 235B, gpt-oss 120B, DeepSeek R1, Kimi K2, Qwen3.8 2.4T ...) stays
  **frozen**. Its weights are never changed and never need gradients.
* A small **graft** sits on top of the model's final hidden state and changes the scores of a few candidate next tokens. It has two
  parts:
  - a low-rank network (about 0.3M to 1.2M parameters, depending on the model's width),
  - an exact **n-gram memory** of your text. When a context from your data comes up again, what followed it there becomes a candidate,
    even if the base model would never have proposed it (a name, a number, a word from your notes).
* A learned **gate** decides from the hidden state how much to trust each memory length.
* **Anchors**, a built-in set of general questions and text, teach the graft to leave everything else alone. You can give your own
  general examples instead (`--anchor FILE`).

So the graft teaches the model *your* facts, names, wording and style. It cannot give the model abilities its weights do not have.

## The three steps

| Step | What runs | Cost |
|---|---|---|
| 1. `es_graft read` | The frozen model reads your data (and the anchors) once. Examples go through it in groups of up to 2,048 tokens, layer by layer, so a model bigger than RAM streams its experts once per group, not once per 64 tokens (measured: 10x fewer bytes read). For every position it saves the final hidden state and the base model's top-32 candidates and scores in a feature file. | The model's prompt-reading speed. This is the slow part for a 200B model, and it is resumable. |
| 2. `es_graft train` | The graft trains from the feature file only; the big model is not used again. | The same for a 1B and a 235B model. Measured: about 9,500 tokens/s on 4 cores of a PC. |
| 3. `es_chat -A` / `es_serve --graft` | The frozen model plus the graft on every token. | About 1-2 ms per token: Qwen2.5-0.5B generated 33.3 -> 31.5 tokens/s; for OLMoE (7B) the difference was within noise. |

The training loss is **exact**, not an estimate. Only the candidate tokens change score; every other token keeps its base score, so
the probability of the right token is computed exactly. `es_graft check` verifies that the engine computes the same numbers while
generating (largest difference per position: about 1e-6).

## Commands

```sh
cd ~/Stream- && make es_graft es_chat es_serve es_get        # the engine (expertcraft.py graft-setup does this)
./es_graft plan  -g MODEL.gguf --data notes.txt               # measured on this device: hours to read, minutes to train, disk
./es_graft read  -g MODEL.gguf --data notes.txt -o my.esfeat  # Ctrl-C any time; the same command continues
./es_graft train my.esfeat -o my.esgraft                      # prints held-out loss before/after and the change on general text
./es_chat  -g MODEL.gguf -A my.esgraft                        # or: ./es_serve -g MODEL.gguf --graft my.esgraft
```

Data: `.txt` is plain text (every token is learned). `.jsonl` has one example per line: `{"text": ...}`,
`{"prompt": ..., "response": ...}`, or `{"messages": [{"role": "user", "content": ...}, ..., {"role": "assistant", "content": ...}]}`.
Chat examples use the model's own chat template, and only the answer is learned. Write several wordings of each question you care
about: every 20th example (`--val-every`) is held out to measure what the graft really learned.

Options:
* `read`: `--seq 512` (example length), `--k 32` (candidates), `--tokens N`, `--minutes M`, `--anchor auto|none|FILE`, `--no-memory`, `--group 2048` (tokens per layer-by-layer group).
* `train`: `--epochs 3`, `--minutes M`, `--anchor-weight 3`, `--m 64` (graft width), `--resume`, `--experts N` (hashed memory experts on flash; experimental, see below).

## Measured (4-core x86 PC; phone not measured yet)

The test: 30 made-up companies with 4 facts each. The graft learned 3 wordings of every question; the test asked the 4th wording,
never seen in training, and the answer was generated greedily. General questions are outside both the data and the anchors.

| Model | Right facts (120 held-out questions), base -> graft | General questions, base -> graft | Held-out answer perplexity | Change on general text |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct (dense) | 0 -> **119** | 23 -> **23** of 24 | 12.58 -> 1.01 | 0.008 nats |
| OLMoE-1B-7B-Instruct (6.9B mixture of experts, Q2_K) | 0 -> **118** | 24 -> **24** of 24 | 158.4 -> 1.01 | 0.002 nats |
| Qwen2.5-0.5B without anchors or memory gates (earlier version) | 0 -> 118 | 12 -> 6 of 12 | | |

The last row is why the anchors and the gates exist. Plain text (TinyStories, 20,000 tokens read through Qwen2.5-0.5B): held-out
perplexity 9.28 -> 7.70 (-17%), with 0.015 nats of change on general text.

Speed on the test PC:

| | Qwen2.5-0.5B | OLMoE 6.9B MoE |
|---|---|---|
| read (the frozen model, batches of 64) | 82-100 tokens/s | about 9 tokens/s |
| train (3 epochs, 4 threads) | 9,500 tokens/s | 4,800 tokens/s |
| generation without -> with graft | 33.3 -> 31.5 tokens/s | 8.0 -> 8.3 (noise) |

Not run end to end here: models above 7B. The test machine's disk gives about 10 MB/s of random reads, so one 64-token batch through
gpt-oss-20b, which streams most of its 12 GB of experts, would take about 17 minutes. The code path is the same for every model the
engine runs; how long reading takes on your device is what `plan` measures.

Hashed memory experts in a sparse file on flash (`--experts N`, Engram-style) are implemented and gradient-checked, but they gave no
measurable gain in any test above (facts: the same 119/120 and 23/24, held-out loss 0.0054 vs 0.0055). They are off by default.

## Changing the inside of the model

The graft works on the model's output. To change the layers themselves without backpropagation, see [zo.md](zo.md):
MeZO adapters, layer-by-layer reading and federated merging, with measurements.

## Honest limits

* Reading the data through a big model is slow on a phone. For a model whose experts do not fit in RAM, every batch streams most of
  the model from flash. Run `plan` first: it measures this.
* A graft changes the scores of up to 48 candidates per token: the base model's top 32 and up to 16 from the memory. A word that
  neither proposes cannot appear.
* Facts are recalled through the memory, so the answer's wording should match your data. The tests used held-out *question* wordings,
  with answers written like the training answers.
* The checks above show general questions keep their answers; they do not prove that nothing at all changed. Train-time output
  reports "change on general text" (KL from the base model on held-out anchors); keep it small.
