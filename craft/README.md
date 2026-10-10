# Expert Craft

An MCP server that lets Claude, ChatGPT, Cursor, Gemini CLI or any MCP-capable AI build a language
model **from scratch**, from 1 million up to 70 billion parameters, using free compute.

Website: https://tharunmaks.github.io/Stream-/craft.html (has a size planner and copy-paste setup).

## Connect
```sh
git clone https://github.com/Tharunmaks/Stream-.git ~/Stream-
claude mcp add expertcraft -- python3 ~/Stream-/craft/expertcraft_mcp.py      # Claude Code, PC or Termux
python3 ~/Stream-/craft/expertcraft_mcp.py connect                              # snippets for every other app
python3 ~/Stream-/craft/expertcraft_mcp.py --http 8766 --public                 # URL for phone apps (cloudflared)
```
The server needs only Python 3. Training needs `pip install torch numpy tokenizers datasets`.

Then ask your AI: *"use expertcraft to build a 30M model on TinyStories and train it"*.

## Tools
| tool | does |
|---|---|
| `craft_guide` | workflow and rules |
| `design_model` | config for 1M–70B with cost |
| `estimate_model` | params, memory, FLOPs, hours for any config |
| `free_compute_plan` | phone vs Colab T4 vs Kaggle 2×T4: fits, hours, largest model |
| `scaffold_project` | model.py, data.py, train.py, export.py, sample.py, colab.ipynb, kaggle.ipynb |
| `run_step` | data / train / export / sample (background for long jobs, `gpus>1` = FSDP) |
| `job_status`, `stop_job` | loss trend, val loss, checkpoints; stop with a saved checkpoint |
| `connect_guide` | setup for Claude Code/Desktop, Cursor, VS Code, Codex, Gemini CLI, phone apps |

## What the generated project does
- Llama-style model: RMSNorm, RoPE, grouped-query attention, SwiGLU.
- `data.py`: streams free datasets (TinyStories, FineWeb-Edu, WikiText) or your own text, trains a byte-level BPE tokenizer.
- `train.py`: one device or many GPUs with FSDP2, meta-device init (no machine holds the whole model),
  activation checkpointing, bf16/fp16 automatically (T4 → fp16), resumes automatically,
  `--max_minutes` saves before a free session ends.
- `export.py` → one `model.pt`; `sample.py` to talk to it.

Tested: a full scaffold → data → train → export → sample run through the MCP tools, and 2-process FSDP
training with sharded checkpoint, resume and export.

## Honest limits
70B models can be designed, scaffolded and trained with this code, but proper training needs about
1.1 TB of GPU memory and roughly 400k H100-hours, so it cannot be done on free hardware. Free GPUs
handle roughly 30M–1B; a phone handles tiny models.

Projects live in `~/expertcraft` (set `EXPERTCRAFT_HOME` to change); the server refuses paths outside it.
