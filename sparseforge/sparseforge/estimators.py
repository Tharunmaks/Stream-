"""Rough analytic estimators. Outputs are estimates, not measurements."""
BYTES = {  # bytes per base parameter (weights + grads + optimizer as applicable)
    "full_adam_bf16": 16.0, "full_adam_fp32": 16.0, "lora_bf16": 2.0,
    "lora_int8": 1.0, "qlora_int4": 0.55,
}

def estimate_memory(n_params, method="qlora_int4", seq_len=512, batch=1, n_layers=None,
                    d_model=None, lora_frac=0.005, checkpointing=True):
    base = n_params * BYTES[method]
    adapters = 0 if method.startswith("full") else n_params * lora_frac * 16
    if n_layers is None:
        n_layers = max(1, int(round((n_params / 12) ** (1/3) / 8)))  # crude
    if d_model is None:
        d_model = int((n_params / (12 * n_layers)) ** 0.5)
    per_layer_act = seq_len * batch * d_model * 2 * 34  # ~34 activations/token/layer, bf16
    act = per_layer_act * (n_layers ** 0.5 if checkpointing else n_layers)
    total = base + adapters + act
    return {"method": method, "base_gb": round(base/1e9, 2), "adapters_gb": round(adapters/1e9, 3),
            "activations_gb": round(act/1e9, 2), "total_gb": round(total/1e9, 2), "note": "estimate"}

def estimate_throughput(n_params, sustained_tflops, method="lora", active_frac=1.0, mfu=0.4):
    """tokens/s ~= mfu*FLOPS / (flops per token). Full training ~6N, LoRA ~4N (no weight grads)."""
    per_tok = (6 if method == "full" else 4) * n_params * active_frac
    return {"tokens_per_s": round(sustained_tflops * 1e12 * mfu / per_tok, 2),
            "assumptions": {"mfu": mfu, "flops_per_token": per_tok}, "note": "estimate; calibrate with benchmark_kernels"}

def benchmark_kernels_numpy(size=512, iters=10):
    """Measured fp32 GEMM GFLOPS via NumPy BLAS (works in Termux)."""
    import time, numpy as np
    a = np.random.rand(size, size).astype(np.float32); b = np.random.rand(size, size).astype(np.float32); a @ b
    t = time.perf_counter()
    for _ in range(iters): a @ b
    return {"float32_gflops_measured": round(2 * size**3 * iters / (time.perf_counter() - t) / 1e9, 1), "size": size}

def estimate_flash_memory(n_experts=262144, hidden=256, ctx=8, d=32, batch=256, k_hash=3):
    core = 256 * d + ctx * d * hidden + hidden + hidden * 256 + 256
    ram = core * 4 * 3 + batch * (ctx * d + 3 * hidden + 2 * 256) * 4 + batch * k_hash * hidden * 4 * 2
    disk = n_experts * (hidden * 3 + 2)
    return {"core_params": core, "expert_params": n_experts * hidden, "ram_mb_est": round(ram / 1e6, 1),
            "disk_mb_max": round(disk / 1e6, 1), "note": "estimate; experts file is sparse, grows as experts are used"}

def benchmark_kernels(size=1024, iters=10, dtypes=("float32", "bfloat16")):
    import time, torch
    out = {}
    for dt in dtypes:
        d = getattr(torch, dt)
        a = torch.randn(size, size, dtype=d); b = torch.randn(size, size, dtype=d)
        a @ b
        t = time.perf_counter()
        for _ in range(iters):
            a @ b
        el = time.perf_counter() - t
        out[dt] = {"gflops_measured": round(2 * size**3 * iters / el / 1e9, 1)}
    out["threads"] = torch.get_num_threads()
    return out
