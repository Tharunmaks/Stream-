"""Measured CPU benchmarks (thread-capped). Usage: SF_THREADS=2 python benchmarks/bench.py flash|torch
All numbers written to benchmarks/results/*.json are real wall-clock measurements on the machine that ran this."""
import json, os, sys, time
T = os.environ.get("SF_THREADS", "2")
for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"): os.environ[k] = T
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "benchmarks", "results"); os.makedirs(OUT, exist_ok=True)
from sparseforge.hardware import detect_hardware
TRAIN = os.environ.get("SF_TRAIN", "/tmp/train.txt"); VAL = os.environ.get("SF_VAL", "/tmp/val.txt")
STEPS = int(os.environ.get("SF_STEPS", 60))

def flash():
    from sparseforge.flash.engine import create_model, train, evaluate, sample
    from sparseforge.estimators import benchmark_kernels_numpy
    res = {"hardware": detect_hardware(), "threads": T, "gemm": benchmark_kernels_numpy(512, 10)}
    for name, kw in [("dense_core_only", dict(n_experts=0)),
                     ("flash_67M_experts", dict(n_experts=262144)),
                     ("flash_67M_ternary_core", dict(n_experts=262144, ternary=True))]:
        p = os.path.join("/tmp/sf_bench", name); os.system(f"rm -rf {p}")
        info = create_model(p, **kw); t = time.time()
        r = train(p, TRAIN, steps=STEPS, batch=256, lr=3e-3, expert_lr=2e-3, save_every=0)
        r["val_loss"] = evaluate(p, VAL); r["wall_s"] = round(time.time() - t, 1)
        r["disk_bytes_used"] = sum(os.stat(os.path.join(p, f)).st_blocks * 512 for f in os.listdir(p))
        r["sample"] = sample(p, "def ", 80, 0.7)
        res[name] = r; print(name, json.dumps({k: r[k] for k in ("tokens_per_s", "first_loss", "final_loss", "val_loss", "total_params")}), flush=True)
        json.dump(res, open(os.path.join(OUT, "flash.json"), "w"), indent=1)

def torch_bench():
    import torch; torch.set_num_threads(int(T))
    from sparseforge.train import run
    base = dict(d=256, n_layers=4, n_heads=4, ctx=64, batch=8, warmup=2, eval_batches=2, threads=int(T), corpus=TRAIN)
    ck = "/tmp/sf_bench/torch_base.pt"; os.makedirs("/tmp/sf_bench", exist_ok=True)
    res = {"threads": T}
    for name, c in [("pretrain_full_fp32", dict(steps=STEPS, mode="full", lr=1e-3, save=ck)),
                    ("ft_baseline_full_fp32", dict(steps=STEPS // 2, mode="full", base_ckpt=ck)),
                    ("ft_sf_lora_int8_bf16", dict(steps=STEPS // 2, mode="lora", bits=8, bf16=True, base_ckpt=ck, lr=1e-3)),
                    ("ft_sf_lora_int4_bf16_ckpt", dict(steps=STEPS // 2, mode="lora", bits=4, bf16=True, grad_ckpt=True, base_ckpt=ck, lr=1e-3)),
                    ("scratch_hashmoe4_fp32", dict(steps=STEPS // 2, mode="full", moe=4, lr=1e-3))]:
        r = run({**base, **c}); r.pop("config")
        res[name] = r; print(name, json.dumps({k: r[k] for k in ("tokens_per_s", "first_loss", "last_loss", "val_loss_after", "total_params", "trainable_params")}), flush=True)
        json.dump(res, open(os.path.join(OUT, "torch.json"), "w"), indent=1)

if __name__ == "__main__":
    {"flash": flash, "torch": torch_bench}[sys.argv[1] if len(sys.argv) > 1 else "flash"]()
