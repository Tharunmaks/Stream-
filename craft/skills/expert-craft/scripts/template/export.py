"""Turn a training run into one weights file: run/model.pt (fp16, config inside).
Works for single-device runs (ckpt.pt) and FSDP runs (sharded dcp/ folder).
  python export.py --out run [--dtype fp16|bf16|fp32]"""
import argparse, json, os, torch

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="run")
ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
a = ap.parse_args()
dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
state = json.load(open(os.path.join(a.out, "state.json")))

if os.path.isdir(os.path.join(a.out, "dcp")):
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    tmp = os.path.join(a.out, "full.tmp")
    dcp_to_torch_save(os.path.join(a.out, "dcp"), tmp)
    sd = torch.load(tmp, map_location="cpu", weights_only=False)["model"]
    os.remove(tmp)
else:
    sd = torch.load(os.path.join(a.out, "ckpt.pt"), map_location="cpu", weights_only=False)["model"]

sd = {k: (v.to(dt) if torch.is_floating_point(v) else v) for k, v in sd.items()}
dst = os.path.join(a.out, "model.pt")
torch.save({"model": sd, "config": state["config"], "step": state["step"]}, dst)
n = sum(v.numel() for k, v in sd.items() if not (state["config"].get("tie_embeddings") and k == "head.weight"))
print(f"wrote {dst}: {n/1e6:.2f}M params, step {state['step']}, {os.path.getsize(dst)/1e6:.1f} MB")
