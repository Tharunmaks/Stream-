"""sparseforge-mcp: stdio MCP server. Flash tools are NumPy-only (Termux); torch tools load lazily."""
import json, os
try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP
from . import hardware, estimators, jobs, notebook

mcp = FastMCP("sparseforge-mcp")
MODELS = os.environ.get("SPARSEFORGE_MODELS", os.path.expanduser("~/.sparseforge/models"))

def _mpath(name):
    return name if os.path.isabs(name) else os.path.join(MODELS, name)

# ---------- hardware / planning ----------
@mcp.tool()
def detect_hardware() -> dict:
    """CPU model, cores, NUMA, RAM, NEON/AMX/AVX-512/AVX2 flags, GPU presence."""
    return hardware.detect_hardware()

@mcp.tool()
def benchmark_kernels(size: int = 512, iters: int = 10) -> dict:
    """Measure real fp32 GEMM GFLOPS on this device (NumPy BLAS)."""
    return estimators.benchmark_kernels_numpy(size, iters)

@mcp.tool()
def estimate_memory(n_experts: int = 262144, hidden: int = 256, ctx: int = 8, d: int = 32, batch: int = 256) -> dict:
    """Estimate RAM and max disk for a Flash model (dense core in RAM + hashed experts on disk)."""
    return estimators.estimate_flash_memory(n_experts, hidden, ctx, d, batch)

@mcp.tool()
def estimate_throughput(n_params: float, sustained_tflops: float, method: str = "lora", active_frac: float = 1.0, mfu: float = 0.4) -> dict:
    """Roofline tokens/s estimate for transformer training (estimate, not a measurement)."""
    return estimators.estimate_throughput(int(n_params), sustained_tflops, method, active_frac, mfu)

@mcp.tool()
def plan_training(target_params: float = 70e6, ram_gb: float = 0) -> dict:
    """Suggest a Flash config with ~target_params total, sized to fit RAM (ram_gb=0: detect)."""
    hw = hardware.detect_hardware(); ram = ram_gb or hw["ram_gb"] or 2
    hidden = 256; n_exp = max(1024, int(target_params // hidden))
    batch = 256 if ram >= 2 else 64
    return {"create": {"hidden": hidden, "n_experts": n_exp, "ctx": 8, "d": 32}, "train": {"batch": batch, "threads": min(4, hw["logical_cores"] or 2)},
            "memory": estimators.estimate_flash_memory(n_exp, hidden, batch=batch)}

# ---------- Flash engine: create / train / status / sample ----------
@mcp.tool()
def create_model(name: str, n_experts: int = 262144, hidden: int = 256, ctx: int = 8, d: int = 32, k_hash: int = 3, ternary: bool = False) -> dict:
    """Create a Flash model (byte-level LM). 262144 experts x 256 hidden ~= 67M expert params on disk."""
    from .flash.engine import create_model as cm
    return cm(_mpath(name), ctx, d, hidden, n_experts, k_hash, ternary)

@mcp.tool()
def list_models() -> list:
    """List models in the models folder."""
    return sorted(os.listdir(MODELS)) if os.path.isdir(MODELS) else []

@mcp.tool()
def train_model(name: str, corpus_path: str, steps: int = 200, batch: int = 256, lr: float = 0.002,
                expert_lr: float = 0.002, threads: int = 2, eval_corpus: str = "") -> dict:
    """Start background training of a Flash model on a UTF-8/bytes text file (streamed). Returns job_id."""
    return jobs.start_flash_job(_mpath(name), corpus_path, steps, batch, lr, expert_lr, threads, eval_corpus or None)

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """Job state, step, loss, measured tokens/s, or final result."""
    return jobs.get_status(job_id)

@mcp.tool()
def list_jobs() -> list:
    """All job ids."""
    return jobs.list_jobs()

@mcp.tool()
def stop_job(job_id: str) -> dict:
    """Stop a job (Flash jobs save weights first)."""
    return jobs.stop_job(job_id)

@mcp.tool()
def tail_logs(job_id: str, n: int = 50) -> str:
    """Last n log lines of a job."""
    return jobs.tail_logs(job_id, n)

@mcp.tool()
def sample_model(name: str, prompt: str = "", n_tokens: int = 200, temperature: float = 0.8) -> str:
    """Generate text from a Flash model."""
    from .flash.engine import sample
    return sample(_mpath(name), prompt, n_tokens, temperature)

@mcp.tool()
def evaluate_model(name: str, corpus_path: str) -> dict:
    """Mean loss (nats/byte) on random windows of a text file."""
    from .flash.engine import evaluate
    return {"eval_loss": evaluate(_mpath(name), corpus_path)}

@mcp.tool()
def offload_to_gpu(config_json: str, target: str = "kaggle", out_path: str = "sparseforge_job.ipynb", repo_url: str = "") -> dict:
    """Write a Kaggle/Colab notebook for a torch-engine config (nothing runs remotely)."""
    nb = notebook.make_notebook(json.loads(config_json), target, repo_url or None)
    json.dump(nb, open(out_path, "w"), indent=1)
    return {"notebook": os.path.abspath(out_path), "target": target}

# ---------- torch engine (PC/laptop; needs PyTorch) ----------
@mcp.tool()
def start_finetune(config_json: str) -> dict:
    """[torch] Background TinyGPT run. keys: d,n_layers,n_heads,ctx,batch,steps,lr,mode(full|lora),bits(8|4),lora_r,bf16,grad_ckpt,moe,threads,base_ckpt,corpus."""
    return jobs.start_job(json.loads(config_json))

@mcp.tool()
def quantize_base(checkpoint: str, out_path: str, bits: int = 8) -> dict:
    """[torch] Quantize a TinyGPT checkpoint's linear layers to int8/int4 (frozen base for LoRA)."""
    import torch
    from .model import TinyGPT, apply_lora_quant, count_params
    sd = torch.load(checkpoint); cfg = {k: v for k, v in _infer_cfg(sd).items()}
    m = TinyGPT(**cfg); m.load_state_dict(sd); apply_lora_quant(m, bits=bits, r=0)
    torch.save(m.state_dict(), out_path)
    return {"out_path": out_path, "bytes": os.path.getsize(out_path), **count_params(m)}

def _infer_cfg(sd):
    d = sd["tok.weight"].shape[1]; L = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    return dict(vocab=sd["tok.weight"].shape[0], d=d, n_layers=L, ctx=sd["pos.weight"].shape[0], n_heads=max(1, d // 64))

@mcp.tool()
def export_adapter(job_id: str, out_path: str) -> dict:
    """Save only LoRA adapter tensors (PEFT-style names) from a finished job."""
    import torch
    sd = torch.load(os.path.join(jobs.ROOT, job_id, "model.pt"))
    ad = {k: v for k, v in sd.items() if "lora_" in k}
    torch.save(ad, out_path)
    return {"out_path": out_path, "tensors": len(ad), "bytes": os.path.getsize(out_path)}

@mcp.tool()
def merge_adapter(job_id: str, out_path: str) -> dict:
    """Merge LoRA into dequantized base weights -> plain fp32 TinyGPT checkpoint."""
    import torch
    sd = torch.load(os.path.join(jobs.ROOT, job_id, "model.pt")); out = {}
    cfg = json.load(open(os.path.join(jobs.ROOT, job_id, "config.json")))
    scaling = 16 / cfg.get("lora_r", 8)
    for k, v in sd.items():
        if k.endswith(".qweight"):
            p = k[:-8]; s = sd[p + ".scale"].float(); q = v
            if q.dtype == torch.uint8:
                lo = (q & 0xF).to(torch.int8) - 8; hi = (q >> 4).to(torch.int8) - 8
                q = torch.stack([lo, hi], -1).reshape(q.shape[0], -1)
            w = q.float() * s
            if p + ".lora_A" in sd:
                w = w + sd[p + ".lora_B"] @ sd[p + ".lora_A"] * scaling
            out[p + ".weight"] = w
            if sd.get(p + ".bias") is not None: out[p + ".bias"] = sd[p + ".bias"]
        elif not any(k.endswith(s) for s in (".scale", ".lora_A", ".lora_B", ".bias")) or k.rsplit(".", 1)[0] + ".qweight" not in sd:
            out[k] = v
    torch.save(out, out_path)
    return {"out_path": out_path, "tensors": len(out)}

@mcp.tool()
def evaluate(checkpoint: str, corpus: str = "", batches: int = 8) -> dict:
    """Validation loss (nats/byte) of a plain TinyGPT checkpoint on the corpus' last 10%."""
    import torch
    from .model import TinyGPT
    from .data import load_corpus
    from .train import evaluate as ev
    sd = torch.load(checkpoint); cfg = _infer_cfg(sd); m = TinyGPT(**cfg); m.load_state_dict(sd)
    data = load_corpus(corpus or None); val = data[int(data.numel() * .9):]
    loss = ev(m, val, {"batch": 8, "ctx": cfg["ctx"], "bf16": False, "eval_batches": batches}, torch.Generator().manual_seed(99))
    return {"val_loss": round(loss, 4)}

@mcp.tool()
def mnx_integrate(mnx_path: str) -> dict:
    """Scan an Mnx repo for nn.Linear usage and return a patch plan for LoRA/quant/MoE integration (read-only)."""
    hits = []
    for root, _, files in os.walk(mnx_path):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(root, f)
                try:
                    for i, line in enumerate(open(p, errors="ignore"), 1):
                        if "nn.Linear(" in line: hits.append(f"{p}:{i}: {line.strip()}")
                except OSError: pass
    return {"linear_layers": hits[:200], "plan": "After loading Mnx weights, replace these nn.Linear modules with "
            "sparseforge.layers.QuantLoRALinear(layer.weight, layer.bias, bits, r); optionally swap FFN for HashMoEFFN."}

def main():
    mcp.run()

if __name__ == "__main__":
    main()
