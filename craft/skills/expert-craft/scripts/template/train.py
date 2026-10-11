"""Expert Craft trainer: one device, or many GPUs with FSDP2 (sharded) for big models.

One device (phone CPU, laptop, Colab T4):
  python train.py --steps 2000 --batch 8 --out run
Several GPUs (Kaggle 2x T4, cloud):
  torchrun --nproc_per_node 2 train.py --fsdp --steps 5000 --out run
Huge models (13B-70B): add --grad_ckpt and use many GPUs; the model is built on the 'meta'
device so no single machine ever holds all the weights.

It resumes automatically from --out, and --max_minutes stops and saves cleanly before a free
session (Colab/Kaggle) runs out. Point --out at Google Drive or /kaggle/working to keep runs.
"""
import argparse, json, math, os, signal, time
import numpy as np
import torch
import torch.distributed as dist
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.json")
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="run")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--batch", type=int, default=8, help="sequences per device per micro-step")
ap.add_argument("--accum", type=int, default=1, help="gradient accumulation micro-steps")
ap.add_argument("--seq", type=int, default=0, help="sequence length (default: config max_seq_len)")
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--min_lr_frac", type=float, default=0.1)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--eval_every", type=int, default=200)
ap.add_argument("--eval_iters", type=int, default=20)
ap.add_argument("--save_every", type=int, default=200)
ap.add_argument("--log_every", type=int, default=10)
ap.add_argument("--max_minutes", type=float, default=0, help="stop + save after this many minutes (0 = no limit)")
ap.add_argument("--fsdp", action="store_true", help="shard params/grads/optimizer across GPUs (needs torchrun)")
ap.add_argument("--grad_ckpt", action="store_true", help="recompute activations: much less memory, ~30%% slower")
ap.add_argument("--compile", action="store_true")
ap.add_argument("--device", default="auto")
ap.add_argument("--seed", type=int, default=1337)
a = ap.parse_args()

# ---------------------------------------------------------------- setup
world = int(os.environ.get("WORLD_SIZE", 1))
distributed = world > 1
if distributed:
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
rank = dist.get_rank() if distributed else 0
local = int(os.environ.get("LOCAL_RANK", 0))
master = rank == 0

if a.device != "auto":
    dev = torch.device(a.device)
elif torch.cuda.is_available():
    dev = torch.device("cuda", local); torch.cuda.set_device(dev)
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    dev = torch.device("mps")
else:
    dev = torch.device("cpu")
if dev.type == "cuda":
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4/P100 -> fp16
    torch.backends.cuda.matmul.allow_tf32 = True
else:
    dtype = torch.float32


def log(*x):
    if master:
        print(*x, flush=True)


cfg = json.load(open(a.config))
cfg["grad_ckpt"] = a.grad_ckpt
T = a.seq or cfg["max_seq_len"]
meta = json.load(open(os.path.join(a.data, "meta.json")))
assert meta["vocab_size"] <= cfg["vocab_size"], "run data.py first (it sets config vocab_size)"
os.makedirs(a.out, exist_ok=True)

# ---------------------------------------------------------------- model
torch.manual_seed(a.seed)
use_fsdp = distributed and a.fsdp
if use_fsdp:
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    with torch.device("meta"):
        model = GPT(cfg)
    mp = MixedPrecisionPolicy(param_dtype=dtype, reduce_dtype=torch.float32)
    for b in model.blocks:
        fully_shard(b, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    model.to_empty(device=dev)
    model.init_weights()
    if cfg.get("tie_embeddings"):
        model.head.weight = model.emb.weight
    autocast = False
else:
    model = GPT(cfg).to(dev)
    autocast = dtype != torch.float32
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if dev.type == "cuda" else None)
raw = model.module if hasattr(model, "module") else model
n_params = sum(p.numel() for p in {id(p): p for p in raw.parameters()}.values())
log(f"Expert Craft | {n_params/1e6:.2f}M params | {world} x {dev.type} | {str(dtype).split('.')[-1]} | "
    f"fsdp={use_fsdp} grad_ckpt={a.grad_ckpt}")
if a.compile:
    model = torch.compile(model)

decay = [p for n, p in raw.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in raw.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=dev.type == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))

# ---------------------------------------------------------------- data
ndt = np.dtype(meta["dtype"])
train = np.memmap(os.path.join(a.data, "train.bin"), dtype=ndt, mode="r")
vpath = os.path.join(a.data, "val.bin")
val = np.memmap(vpath, dtype=ndt, mode="r") if os.path.getsize(vpath) > (T + 1) * ndt.itemsize * 4 else None
assert len(train) > T + 1, f"not enough tokens ({len(train)}) for seq {T}; get more data or lower --seq"


def get_batch(src, rng):
    ix = rng.integers(0, len(src) - T - 1, a.batch)
    x = np.stack([src[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([src[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    if dev.type == "cuda":
        return x.pin_memory().to(dev, non_blocking=True), y.pin_memory().to(dev, non_blocking=True)
    return x.to(dev), y.to(dev)


def fwd(x, y):
    with torch.autocast(dev.type, dtype=dtype, enabled=autocast):
        return model(x, y)[1]


@torch.no_grad()
def evaluate(step):
    if val is None:
        return None
    model.eval()
    rng = np.random.default_rng(10_000 + rank)
    tot = torch.zeros(1, device=dev)
    for _ in range(a.eval_iters):
        tot += fwd(*get_batch(val, rng)).float()
    if distributed:
        dist.all_reduce(tot)
    model.train()
    return tot.item() / (a.eval_iters * world)


# ---------------------------------------------------------------- checkpoints
def save(step, why):
    state = {"step": step, "config": cfg, "n_params": n_params, "why": why, "time": time.time()}
    if use_fsdp:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict
        msd, osd = get_state_dict(model, opt)
        dcp.save({"model": msd, "optim": osd}, checkpoint_id=os.path.join(a.out, "dcp"))
    elif master:
        torch.save({"model": raw.state_dict(), "optim": opt.state_dict(), "scaler": scaler.state_dict(), **state},
                   os.path.join(a.out, "ckpt.tmp"))
        os.replace(os.path.join(a.out, "ckpt.tmp"), os.path.join(a.out, "ckpt.pt"))
    if master:
        json.dump(state, open(os.path.join(a.out, "state.json"), "w"), indent=2)
        log(f"saved step {step} ({why}) -> {a.out}")
    if distributed:
        dist.barrier()


def load():
    sp = os.path.join(a.out, "state.json")
    if not os.path.exists(sp):
        return 0
    step = json.load(open(sp))["step"]
    if use_fsdp:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
        msd, osd = get_state_dict(model, opt)
        sd = {"model": msd, "optim": osd}
        dcp.load(sd, checkpoint_id=os.path.join(a.out, "dcp"))
        set_state_dict(model, opt, model_state_dict=sd["model"], optim_state_dict=sd["optim"])
    else:
        ck = torch.load(os.path.join(a.out, "ckpt.pt"), map_location=dev, weights_only=False)
        raw.load_state_dict(ck["model"]); opt.load_state_dict(ck["optim"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
    log(f"resumed from step {step}")
    return step


def lr_at(step):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


stop_flag = {"v": False}
signal.signal(signal.SIGTERM, lambda *_: stop_flag.update(v=True))
step = load()
logf = open(os.path.join(a.out, "log.jsonl"), "a") if master else None
t_start = t_log = time.time()
tokens_per_step = a.batch * a.accum * T * world
model.train()

while step < a.steps:
    rng = np.random.default_rng(a.seed * 1_000_003 + step * 997 + rank)
    lr = lr_at(step)
    for g in opt.param_groups:
        g["lr"] = lr
    loss_acc = 0.0
    for micro in range(a.accum):
        x, y = get_batch(train, rng)
        if distributed and not use_fsdp:
            model.require_backward_grad_sync = micro == a.accum - 1
        loss = fwd(x, y) / a.accum
        scaler.scale(loss).backward()
        loss_acc += loss.item()
    scaler.unscale_(opt)
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update()
    opt.zero_grad(set_to_none=True)
    step += 1

    if step % a.log_every == 0 or step == 1:
        now = time.time()
        tps = tokens_per_step * (a.log_every if step > 1 else 1) / max(1e-9, now - t_log)
        t_log = now
        rec = {"step": step, "loss": round(loss_acc, 4), "lr": lr, "grad_norm": round(float(gnorm), 3),
               "tok_per_s": round(tps), "minutes": round((now - t_start) / 60, 2)}
        log(f"step {step}/{a.steps} | loss {loss_acc:.4f} | lr {lr:.2e} | {tps:,.0f} tok/s | {rec['minutes']} min")
        if logf:
            logf.write(json.dumps(rec) + "\n"); logf.flush()
    if step % a.eval_every == 0 or step == a.steps:
        vl = evaluate(step)
        if vl is not None:
            log(f"step {step} | val loss {vl:.4f} | ppl {math.exp(min(vl, 20)):.1f}")
            if logf:
                logf.write(json.dumps({"step": step, "val_loss": round(vl, 4)}) + "\n"); logf.flush()
    out_of_time = a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes
    if distributed:
        f = torch.tensor([1.0 if (out_of_time or stop_flag["v"]) else 0.0], device=dev)
        dist.all_reduce(f, op=dist.ReduceOp.MAX)
        out_of_time = f.item() > 0
    if out_of_time or stop_flag["v"]:
        save(step, "time limit"); break
    if step % a.save_every == 0 and step < a.steps:
        save(step, "periodic")

if step >= a.steps:
    save(step, "finished")
    log("done. next: python export.py --out " + a.out + " && python sample.py --out " + a.out)
if distributed:
    dist.destroy_process_group()
