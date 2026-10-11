# Expert Craft

Train a language model with **up to 70 billion parameters on your own phone, for free**: no GPU, no account, no server, no MCP.
Expert Craft is an **agent skill**: a folder with instructions (`SKILL.md`) and the trainer, which an AI agent reads and runs on your device.
Website with a planner: https://tharunmaks.github.io/Stream-/craft.html

## Install it into your AI

**Claude Code** (PC, or Termux on Android):
```
/plugin install expert-craft --marketplace Tharunmaks/Stream-
```
Older Claude Code: `/plugin marketplace add Tharunmaks/Stream-` then `/plugin install expert-craft@expertstream`.
No GitHub SSH key on the device: start Claude Code with `CLAUDE_CODE_PLUGIN_PREFER_HTTPS=1`.

**Any agent that reads a skills folder** (copies the skill to `~/.claude/skills/expert-craft`, or to the folder you pass):
```sh
curl -fsSL https://raw.githubusercontent.com/Tharunmaks/Stream-/claude/expertstream-moe-android-oegt59/scripts/install-skill.sh | sh
```
**Apps that take a skill upload**: `docs/expert-craft-skill.zip` (also on the website).

**Or just tell your AI**: *"Install the Expert Craft skill from github.com/Tharunmaks/Stream- (folder craft/skills/expert-craft), then use
it to plan and train a 70B model on this phone and explain every step to me."*

Then ask: *"Use expert-craft to train a 70B model on this phone for 10 hours."* The skill makes the AI set up Python, NumPy and a C compiler,
measure the device, explain the plan honestly, train in the background, watch the loss and let you talk to the result.

## By hand
```sh
pkg install -y python python-numpy clang                    # Termux; Linux: python3-numpy + gcc; macOS: Xcode tools + pip install numpy
EC="python3 ~/Stream-/craft/expertcraft.py"                  # or <skill folder>/scripts/expertcraft.py
$EC phone-plan 70e9 --minutes 600
$EC phone-train mymodel --dataset tinystories --params 70e9 --minutes 600 --min-battery 25 --max-temp 42 --max-wear-gb 2000
$EC phone-sample mymodel --prompt "Once upon a time"
$EC phone-status mymodel
```

## The technology: Flash Experts (v2)

* A small **dense transformer in RAM** (0.6M to 39M parameters, sized from the measured speed and the minutes you give).
* **Up to ~18 million page-sized experts in a sparse file** on storage: each expert is a tiny MLP whose 8-bit weights and optimizer state fill
  exactly one 4 KB page, 70 billion parameters in all. Unused experts take no space; the file grows as training uses them.
* **Hashed contexts route each word**: the last 1, 2 and 3 tokens each pick one expert, so the same context always trains the same page.
* **Native kernels** (C, compiled on the device on first use, NumPy fallback): one pass per expert reads the page, runs forward and backward,
  applies row-wise Adagrad, re-quantises with stochastic rounding and writes the page back, on the fast CPU cores in parallel.
  The expert file is opened for random access, so the system does not read 16-32 neighbouring pages for each one used.

Details, all flags and troubleshooting: [`skills/expert-craft/reference.md`](skills/expert-craft/reference.md).

## Measured (4-core Linux PC, TinyStories, full 70B-parameter expert file, 8 minutes each)

| Version | Dense width | Tokens/s | Validation loss after 8 min | Time to reach v1's 8-minute loss |
|---|---|---|---|---|
| v1 (NumPy experts) | 192 | ~1,150 | 2.691 | 463 s |
| v2 | 192 | 3,113 | 2.408 | ~220 s |
| v2 | 128 | 5,397 | 2.314 | 140 s |
| v2 | 96 | 6,534 | 2.323 | **110 s (4.2x faster)** |

Experts vs no experts, same data and 2,000 steps at width 192: validation loss 2.611 without experts, 2.367 with 262,144 hashed experts
(perplexity 13.6 -> 10.7). A real 69.98-billion-parameter run of v1 on this PC: 1.83M tokens in 23 minutes, validation loss 2.345, a 74.6 GB
sparse file of which 2.15 GB was on disk.

## Honest limits

* All 70B parameters are really trained on the device, but each word uses under 0.005% of them. **Quality follows the tokens your device can
  process and the dense model, not the file size.** It is not comparable to 70B models trained on trillions of tokens in data centres.
* Each token rewrites up to 3 pages (12 KB). The plan prints this upper bound; the system merges repeated writes, and `phone-status` shows the
  real amount (on the test PC about a third). Cap it with `--max-wear-gb`.
* Speed was measured on a PC, not yet on a phone; `phone-plan` measures your device. A trained model is used with `phone-sample` (NumPy).

## Free GPUs for dense models
`$EC plan 125e6`, `$EC scaffold 125e6 NAME` (PyTorch, FSDP, resumable, `colab.ipynb`, `kaggle.ipynb`), `$EC run NAME data|train|export|sample`.
A dense 70B model needs about 1.1 TB of training memory and cannot be trained on free GPUs.

## Layout
```
craft/.claude-plugin/plugin.json      plugin manifest (the marketplace is .claude-plugin/marketplace.json at the repository root)
craft/skills/expert-craft/SKILL.md    what the AI reads
craft/skills/expert-craft/reference.md
craft/skills/expert-craft/scripts/    expertcraft.py (CLI), phone/ (flashexperts.py, fx_kernels.c, native.py, builder.py, phonetrain.py, layout.py), template/ (GPU project)
craft/expertcraft.py                  shortcut to the CLI
```
Tests: `python3 scripts/test_flash_experts.py` (gradients, 8-bit store, resume, 70B sparse file, native kernels against NumPy).
