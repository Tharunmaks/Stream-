"""Training entrypoint. Measures real wall-clock tokens/s."""
import argparse, json, os, time, torch
from .model import TinyGPT, apply_lora_quant, count_params
from .data import load_corpus, batches

DEFAULT = dict(d=384, n_layers=6, n_heads=6, ctx=128, batch=16, steps=30, lr=3e-4,
               mode="full", bits=8, lora_r=8, bf16=False, grad_ckpt=False, moe=0,
               sparse_out_neg=0, threads=None, base_ckpt=None, save=None, corpus=None,
               seed=0, eval_batches=4, status_file=None, warmup=2)

def _status(cfg, **kw):
    if cfg.get("status_file"):
        with open(cfg["status_file"] + ".tmp", "w") as f:
            json.dump(kw, f)
        os.replace(cfg["status_file"] + ".tmp", cfg["status_file"])

@torch.no_grad()
def evaluate(model, data, cfg, gen):
    model.eval(); it = batches(data, cfg["batch"], cfg["ctx"], gen); tot = 0
    ctx = torch.autocast("cpu", dtype=torch.bfloat16) if cfg["bf16"] else torch.autocast("cpu", enabled=False)
    for _ in range(cfg["eval_batches"]):
        x, y = next(it)
        with ctx:
            tot += model(x, y).item()
    model.train(); return tot / cfg["eval_batches"]

def run(cfg):
    cfg = {**DEFAULT, **cfg}
    torch.manual_seed(cfg["seed"])
    if cfg["threads"]:
        torch.set_num_threads(int(cfg["threads"]))
    data = load_corpus(cfg["corpus"]); n = int(data.numel() * 0.9)
    train_d, val_d = data[:n], data[n:]
    model = TinyGPT(256, cfg["d"], cfg["n_layers"], cfg["n_heads"], ctx=cfg["ctx"], moe=cfg["moe"])
    if cfg["base_ckpt"]:
        model.load_state_dict(torch.load(cfg["base_ckpt"]), strict=True)
    if cfg["mode"] == "lora":
        apply_lora_quant(model, bits=cfg["bits"], r=cfg["lora_r"],
                         compute_dtype=torch.bfloat16 if cfg["bf16"] else torch.float32)
    model.grad_ckpt = cfg["grad_ckpt"]; model.sparse_out_neg = cfg["sparse_out_neg"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    val_before = evaluate(model, val_d, cfg, torch.Generator().manual_seed(99))
    it = batches(train_d, cfg["batch"], cfg["ctx"], gen)
    ac = torch.autocast("cpu", dtype=torch.bfloat16) if cfg["bf16"] else torch.autocast("cpu", enabled=False)
    losses, t_start, timed_tokens = [], None, 0
    for step in range(cfg["steps"]):
        if step == cfg["warmup"]:
            t_start = time.perf_counter()
        x, y = next(it)
        with ac:
            loss = model(x, y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(round(loss.item(), 4))
        if t_start is not None:
            timed_tokens += x.numel()
        _status(cfg, state="running", step=step + 1, steps=cfg["steps"], loss=losses[-1])
    elapsed = time.perf_counter() - t_start
    val_after = evaluate(model, val_d, cfg, torch.Generator().manual_seed(99))
    if cfg["save"]:
        torch.save(model.state_dict(), cfg["save"])
    res = dict(config=cfg, **count_params(model), threads=torch.get_num_threads(),
               tokens_per_s=round(timed_tokens / elapsed, 1), timed_steps=cfg["steps"] - cfg["warmup"],
               elapsed_s=round(elapsed, 2), first_loss=losses[0], last_loss=losses[-1],
               val_loss_before=round(val_before, 4), val_loss_after=round(val_after, 4))
    _status(cfg, state="done", result=res)
    return res

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--out")
    a = ap.parse_args(); cfg = json.load(open(a.config))
    try:
        res = run(cfg)
    except Exception as e:
        _status(cfg, state="failed", error=repr(e)); raise
    s = json.dumps(res, indent=2); print(s)
    if a.out:
        open(a.out, "w").write(s)

if __name__ == "__main__":
    main()
