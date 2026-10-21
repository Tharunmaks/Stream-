# Expert Stream Trainer (web app)
Phone-friendly Flask UI styled after the ExpertStream site (palette, Bricolage Grotesque + IBM Plex fonts, bundled offline in static/fonts.css) for es_lora: pick GGUF / HF GGUF repo, upload Q&A JSONL (auto-converted to {"prompt","response"}), LoRA options, live tok/s, RSS, loss chart, ETA, stop/resume, chat (es_chat -J --lora), download .eslora and llama.cpp LoRA GGUF (es_lora export).
Run: `./run.sh` (PORT=8765). Env: ES_BIN_DIR (es_lora/es_chat dir), ES_MODEL_DIR (GGUFs), ES_BLAS (OpenBLAS for fast mode).
Termux: `pkg install python clang git && pip install flask huggingface_hub`, build es_lora/es_chat with build.sh, then `ES_BIN_DIR=... ES_MODEL_DIR=... ./run.sh`, open http://localhost:8765. Untick Fast mode (AMX is x86 only).
Limits: resume keeps LoRA weights + Adam state but es_lora saves step=0 on stop, so the app restarts with the remaining steps and the LR warm-up/cosine schedule restarts. HF safetensors-only repos are rejected (convert with llama.cpp convert_hf_to_gguf.py). One training run at a time; no auth (LAN only).

## Modes (unified es_train)
The app drives `es_train` (ES_BIN_DIR, default the repo root (set ES_BIN_DIR)) and falls back to `es_lora` if es_train is missing.
- Fine-tune: `es_train finetune train ...` (identical to es_lora). Stop/resume, chat, .eslora + GGUF export.
- MoE from scratch: `es_train moe gen/train` with layers, experts, width, expert size, top-k, cache MB, routing balance, random routing; experts are always ternary. The expert file is generated in runs/ on first use. JSONL is converted to plain "Q:/A:" text (the MoE prototype is byte-level).
  Limits: no checkpoint (stop only, no resume), no chat or download, since the prototype saves no weights.
