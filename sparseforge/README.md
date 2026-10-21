# SparseForge

CPU-first language-model training for **phones (Termux/Android, ARM) and small low-RAM machines, no GPU**,
built on the ExpertStream / Expert Craft idea: a **small dense core in RAM + many tiny hashed experts on disk**,
with sparse expert updates.

Two engines:

| Engine | Needs | Use |
|---|---|---|
| `sparseforge.flash` | Python + NumPy only | Phones / low-RAM CPUs. Targets ~70M total params (mostly on-disk experts). |
| `sparseforge.train` (torch) | PyTorch | PCs: TinyGPT transformer, int8/int4 frozen base + LoRA, bf16, grad checkpointing, hash-MoE FFN, sampled-softmax output. |

Plus `sparseforge-mcp`, a stdio MCP server (official `mcp` Python SDK) so AI assistants can create, train, monitor and sample models.

## How the Flash engine works (honest description)
- Byte-level LM. **Dense core in RAM**: byte embeddings for the last `ctx` bytes → `W1` → GELU → output layer. Trained with Adam.
  The optimizer state is saved as fp16 to halve its size on disk.
- **Hashed experts on disk** (`np.memmap`): `n_experts × hidden` int8 rows + an fp16 scale and fp16 momentum per row. For each position,
  the last 1..K bytes are hashed to K expert rows, which are added to the hidden pre-activation (Hash-Layers-style routing, no learned router).
- **Sparse updates**: each step reads and writes only the rows it touched (≤ batch×K). The rest stay on disk. The files are created sparse,
  so disk use only grows as experts are actually used.
- Expert optimizer: Lion-style momentum-sign update, requantized to int8 with **stochastic rounding**.
- Optional **ternary core** (`--ternary`): BitNet-style {-1,0,1}·scale weights for `W1` in the forward pass, trained with a straight-through estimator.
- **Streaming data**: the corpus is read in 1 MB chunks, never loaded whole.
- Threads: the large matmuls use NumPy's BLAS (OpenBLAS uses NEON on ARM and AVX on x86). Cap threads with `OPENBLAS_NUM_THREADS`.
- Not in Flash: attention and gradient checkpointing. The core is a single MLP layer, so checkpointing would not help. Both live in the torch engine.
  Flash is a fast n-gram-plus-MLP style model. It is **not** a 70M-parameter transformer in quality terms.

## Install on Android (Termux)
```sh
pkg update && pkg install -y python python-numpy git
git clone https://github.com/Tharunmaks/sparseforge && cd sparseforge     # once pushed
pip install --no-deps -e .                                                # numpy comes from pkg
export OPENBLAS_NUM_THREADS=2
python -m sparseforge.flash create ~/sfm/demo --experts 262144 --hidden 256   # ~67M expert params + 0.14M core
python -m sparseforge.flash train ~/sfm/demo mytext.txt --steps 500 --batch 256
python -m sparseforge.flash sample ~/sfm/demo --prompt "Once upon a time" -n 200
```
MCP server on Termux: `pip install mcp` (pulls pydantic-core; on Termux this may need `pkg install rust` to build), then `sparseforge-mcp`.

## Install on PC
```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[mcp]" numpy        # add torch for the torch engine
```
MCP client config:
```json
{"mcpServers": {"sparseforge": {"command": "python", "args": ["-m", "sparseforge.mcp_server"],
  "env": {"OPENBLAS_NUM_THREADS": "2", "PYTHONPATH": "/path/to/sparseforge"}}}}
```

## MCP tools
Flash / phone: `detect_hardware`, `benchmark_kernels`, `estimate_memory`, `plan_training`, `create_model`, `list_models`,
`train_model` (background job), `get_job_status`, `list_jobs`, `stop_job`, `tail_logs`, `sample_model`, `evaluate_model`.
Torch / PC: `estimate_throughput`, `start_finetune`, `quantize_base`, `export_adapter`, `merge_adapter`, `evaluate`, `mnx_integrate`,
`offload_to_gpu` (writes a Kaggle/Colab notebook; nothing runs remotely).

## Benchmarks
`SF_THREADS=2 python benchmarks/bench.py flash` and `... torch`. Results go to `benchmarks/results/*.json`. See `BENCHMARKS.md`.
`python benchmarks/mcp_smoke.py` starts the MCP server over stdio, lists its tools, and runs create → train → status → sample.

## Tests
`OPENBLAS_NUM_THREADS=2 python -m pytest -q tests`
