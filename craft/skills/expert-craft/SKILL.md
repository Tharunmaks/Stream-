---
name: expert-craft
description: Train a language model from scratch, for free, on the user's own phone or computer, with up to 70 billion parameters (Flash Experts - a small dense transformer in RAM plus millions of page-sized experts in a sparse file on flash, trained in place by native C kernels). Use when the user wants to make, build or train their own AI / LLM / language model, train on a phone or in Termux, asks for a 1B-70B model without a GPU or money, wants to know what their device can train, or wants a free Colab/Kaggle GPU training project.
compatibility: Python 3.8+ and NumPy. A C compiler (clang or gcc) makes training several times faster but is optional. Android (Termux), Linux, macOS, WSL. Needs 1 GB+ free internal storage; a full 70B expert file can grow to 75 GB.
---

# Expert Craft: train up to 70B parameters on a phone, free

You are operating a real trainer. Everything runs on the user's device: no GPU, no account, no network except optional downloads.
All commands below use `EC`, the command line tool bundled with this skill:

```sh
EC="python3 ${CLAUDE_SKILL_DIR}/scripts/expertcraft.py"     # Claude Code fills in CLAUDE_SKILL_DIR
# other agents: EC="python3 <folder that contains this SKILL.md>/scripts/expertcraft.py"
```

Projects live in `~/expertcraft/<name>` (set `EXPERTCRAFT_HOME` to change; the tool refuses paths outside it).

## 1. Set up (once)

Check what is there, then install only what is missing:

```sh
python3 -c "import numpy; print('numpy', numpy.__version__)"; (command -v cc || command -v clang || command -v gcc) && echo compiler-ok
```

| Device | Install |
|---|---|
| Android | Termux from F-Droid, then `pkg install -y python python-numpy clang` |
| Debian/Ubuntu/WSL | `sudo apt install -y python3-numpy build-essential` (or `pip install numpy`) |
| macOS | `xcode-select --install; pip3 install numpy` |

The C kernels compile by themselves on first use (cached in `~/.cache/expertcraft`). Without a compiler training still works, only slower;
say so if you could not install one. Train on internal storage (Termux home), never on an SD card or `/sdcard` (no sparse files there).

## 2. Plan before training (always)

```sh
$EC phone-plan 70e9 --minutes 600      # the size the user asked for, and how long they can train
```

It measures this device (speed, storage speed, free disk, RAM) and prints the architecture, tokens per second, how many times each expert
will be used, file size, flash written per day and an "Honest summary". Show the user the key numbers and read the honest summary to them
in plain words. If free disk is too small it shrinks the size; `--oversubscribe` keeps the full size (the file only grows as experts are
used, and training stops safely before the disk is full).

## 3. Data

Ask what text to learn from. Use only text the user owns or may use. Options:
- `--data FILE_OR_FOLDER` (txt, md, code ... at least 20,000 characters; more is better)
- `--dataset tinystories` (free 22 MB sample of simple English stories, good first test)

## 4. Train (in the background, it can take hours)

```sh
mkdir -p ~/expertcraft
nohup $EC phone-train mymodel --dataset tinystories --params 70e9 --minutes 600 \
      --min-battery 25 --max-temp 42 --max-wear-gb 2000 > ~/expertcraft/mymodel.out 2>&1 &
```

- The first run prepares the tokenizer and data, then trains. Running the same command again continues where it stopped.
- `--minutes` is the time budget; `--steps N` sets an exact number of steps instead.
- Guards: pauses below `--min-battery` % when not charging, above `--max-temp` C; stops at `--max-wear-gb` of flash written, or when disk runs low.
- Optional: `--width 96|128|192|256|320|384|512` forces the dense model size (default: chosen from speed x minutes);
  `--route pk` uses a learned product-key router instead of hashed contexts (hash is better in our tests).
- On a phone, tell the user to keep it plugged in, and in Termux to run `termux-wake-lock` so Android does not stop the job.

## 5. Watch it

```sh
$EC phone-status mymodel           # step, loss, val_loss trend, tokens/s, experts used, flash written, disk used
tail -n 3 ~/expertcraft/mymodel.out
```

Report the numbers it prints. Loss should fall quickly in the first few hundred steps. If `val_loss` stops improving while `loss` keeps
falling, the model is memorising: suggest more or different text.

## 6. Use it

```sh
$EC phone-sample mymodel --prompt "Once upon a time" --tokens 80 --temp 0.8
```

## 7. Rules you must follow

1. Never call a model trained until `phone-status` shows the run finished (or stopped) with a lower `val_loss` than at the start.
2. Report only numbers the tool printed. Do not invent speeds, losses or quality claims.
3. Always tell the user plainly: all 70B parameters are really trained on the device, but each word uses under 0.005% of them, and
   the text quality follows the tokens the device can process and the dense model, not the file size. It is not a 70B-quality model
   like those trained on trillions of tokens in data centres.
4. Tell the user about flash wear (the plan prints GB per hour) before a long run, and use `--max-wear-gb`.
5. Do not delete `~/expertcraft/<name>` or the expert file unless the user asks; it is their trained model.

## Other routes

- Dense models on free GPUs (Colab, Kaggle), 1M to about 1B parameters: `$EC plan 125e6`, then `$EC scaffold 125e6 mymodel` writes a
  PyTorch project with `colab.ipynb` and `kaggle.ipynb`; `$EC run mymodel data|train|export|sample [args] [--background]`.
- How it works, measured results, all flags and troubleshooting: [reference.md](reference.md).
