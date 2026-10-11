#!/usr/bin/env python3
"""Expert Craft: build language models from scratch with free compute, including a 70-billion-parameter model trained on a phone.

Part of ExpertStream: https://github.com/Tharunmaks/Stream-   Python 3.8+. Phone commands need NumPy (Termux: pkg install python python-numpy).

PHONE (Flash Experts: sparse training, experts live in a file on flash)
  expertcraft.py phone-plan 70e9 [--minutes 600]        what 70B means on THIS device: disk, speed, wear, what you get
  expertcraft.py phone-train NAME --data text.txt --params 70e9 --minutes 600   train (resumes by itself; Ctrl-C saves)
  (add --oversubscribe to plan a file bigger than the free disk: it grows only as experts are used and training stops safely when space runs low)
  expertcraft.py phone-sample NAME --prompt "Once upon a time"
  expertcraft.py phone-status NAME

FREE GPUs (Colab / Kaggle) and planning
  expertcraft.py plan 125e6          can free hardware train this size, and how long
  expertcraft.py design 30e6         architecture and cost
  expertcraft.py scaffold 30e6 NAME  write a ready-to-train PyTorch project with Colab and Kaggle notebooks
  expertcraft.py run NAME data|train|export|sample [args ...] [--background]
  expertcraft.py status NAME | stop NAME

All project folders live under EXPERTCRAFT_HOME (default ~/expertcraft).
"""
import json, math, os, re, shutil, signal, subprocess, sys, time

VERSION = "2.0.0"
APP = "Expert Craft"
REPO = "https://github.com/Tharunmaks/Stream-"
SITE = "https://tharunmaks.github.io/Stream-/craft.html"
HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "template")
HOME = os.path.abspath(os.path.expanduser(os.environ.get("EXPERTCRAFT_HOME", "~/expertcraft")))
MIN_PARAMS, MAX_PARAMS = 1e6, 70e9

# ======================================================================= model math
def count_params(c):
    d, L, h = c["d_model"], c["n_layers"], c["n_heads"]
    kv, ff, V = c["n_kv_heads"], c["d_ff"], c["vocab_size"]
    hd = d // h
    per_layer = 2 * d * h * hd + 2 * d * kv * hd + 3 * d * ff + 2 * d
    return L * per_layer + V * d + (0 if c.get("tie_embeddings") else V * d) + d


def fmt(n):
    for u, s in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if n >= u:
            return f"{n / u:.2f}{s}"
    return str(int(n))


def design(target, vocab=None, seq=None):
    target = float(target)
    if not MIN_PARAMS <= target <= MAX_PARAMS:
        raise ValueError(f"target_params must be between {fmt(MIN_PARAMS)} and {fmt(MAX_PARAMS)}")
    vocab = int(vocab or (4096 if target < 1e7 else 8192 if target < 5e8 else 32000))
    tie = target < 2e9
    best = None
    for d in range(64, 16385, 64):
        hd = 32 if d < 256 else (64 if d < 2048 else 128)
        if d % hd:
            continue
        h = d // hd
        if h >= 8 and h % 4:
            continue  # keep heads divisible by 4 so grouped-query attention works
        kv = h // 4 if h >= 8 else h
        if h >= 32:
            kv = 8 if h % 8 == 0 else kv  # Llama-style: 8 KV heads on big models
        ff = max(128, round(8 * d / 3 / 128) * 128)
        pref = 64 if d < 1024 else 100  # preferred width/depth ratio
        for L in range(2, 129):
            c = dict(vocab_size=vocab, d_model=d, n_layers=L, n_heads=h, n_kv_heads=kv, d_ff=ff,
                     tie_embeddings=tie, max_seq_len=int(seq or (512 if target < 5e7 else 1024 if target < 1e9 else 4096)),
                     rope_base=10000.0 if target < 1e9 else 500000.0)
            n = count_params(c)
            if n > target * 1.3:
                break
            score = abs(n - target) / target + 0.15 * abs(math.log((d / L) / pref))
            if best is None or score < best[0]:
                best = (score, c)
    return best[1]


# Effective (not peak) training speed. Assumptions are printed with every plan.
HARDWARE = [
    {"id": "phone", "name": "Your phone (CPU, Termux)", "free": True, "gpus": 1, "mem_gb": 3.0,
     "eff_tflops": 0.02, "limit": "no limit, but slow; keep it plugged in",
     "how": "pkg install python python-numpy; then use the phone commands (python3 craft/expertcraft.py phone-plan 70e9). "
            "They train sparse models with Flash Experts, which is how a phone reaches 70B parameters."},
    {"id": "colab_t4", "name": "Google Colab free (1x T4 16 GB)", "free": True, "gpus": 1, "mem_gb": 15.0,
     "eff_tflops": 15.0, "limit": "sessions end after a few hours and GPU time is not guaranteed; resume from Drive",
     "how": "Open colab.ipynb from the project in colab.research.google.com, Runtime > Change runtime type > T4 GPU."},
    {"id": "kaggle_2xt4", "name": "Kaggle free (2x T4 16 GB)", "free": True, "gpus": 2, "mem_gb": 15.0,
     "eff_tflops": 15.0, "limit": "about 30 GPU-hours per week, sessions up to about 12 h (check kaggle.com for current quota)",
     "how": "Upload kaggle.ipynb at kaggle.com/code, Settings > Accelerator > GPU T4 x2, Internet on. Uses FSDP on both GPUs."},
    {"id": "rented_8xh100", "name": "Rented 8x H100 80 GB (paid, for reference)", "free": False, "gpus": 8,
     "mem_gb": 80.0, "eff_tflops": 400.0, "limit": "pay per hour",
     "how": "torchrun --nproc_per_node 8 train.py --fsdp --grad_ckpt"},
]


def train_mem_gb(n, gpus, seq, d, L, grad_ckpt, batch=1):
    states = 16 * n / gpus  # fp32 weights + grads + Adam m,v, sharded by FSDP
    act = batch * seq * d * L * (4 if grad_ckpt else 34) / 1e9
    return states / 1e9 + act + 1.0  # +1 GB runtime overhead


def estimate(c, tokens=None, eff_tflops=400.0, gpus=1):
    n = count_params(c)
    tokens = float(tokens or 20 * n)
    flops = 6 * n * tokens
    hours = flops / (eff_tflops * 1e12 * gpus) / 3600
    gb = lambda b: round(b / 1e9, 2)
    return {
        "params": n, "params_h": fmt(n), "train_tokens_h": fmt(tokens),
        "tokens_rule": "20 tokens per parameter (Chinchilla compute-optimal); small models improve well past this",
        "train_flops": f"{flops:.2e}",
        "memory_gb": {"training_state_fp32_adam": gb(16 * n), "weights_fp16": gb(2 * n),
                      "weights_int8": gb(n), "weights_4bit": gb(0.5 * n)},
        "hours_at": {"gpus": gpus, "effective_tflops_each": eff_tflops, "hours": round(hours, 1),
                     "days": round(hours / 24, 1)},
    }


def free_plan(target, tokens=None):
    c = design(target) if not isinstance(target, dict) else target
    n = count_params(c)
    tokens = float(tokens or 20 * n)
    flops = 6 * n * tokens
    rows = []
    for hw in HARDWARE:
        g = hw["gpus"]
        need = train_mem_gb(n, g, min(c["max_seq_len"], 1024), c["d_model"], c["n_layers"], grad_ckpt=n > 3e8)
        fits = need <= hw["mem_gb"] * 0.9
        hours = flops / (hw["eff_tflops"] * 1e12 * g) / 3600
        # largest model that fits this hardware (binary search on the design curve)
        lo, hi = MIN_PARAMS, MAX_PARAMS
        for _ in range(40):
            mid = (lo * hi) ** 0.5
            cm = design(mid)
            ok = train_mem_gb(count_params(cm), g, 1024, cm["d_model"], cm["n_layers"], True) <= hw["mem_gb"] * 0.9
            lo, hi = (mid, hi) if ok else (lo, mid)
        tok_12h = hw["eff_tflops"] * 1e12 * g * 12 * 3600 / (6 * n)
        rows.append({"option": hw["name"], "id": hw["id"], "free": hw["free"], "fits_in_memory": fits,
                     "memory_needed_gb_per_device": round(need, 1), "memory_available_gb": hw["mem_gb"],
                     "hours_for_full_training": round(hours, 1),
                     "tokens_per_12h_session": fmt(tok_12h),
                     "largest_trainable_model_here": fmt(lo), "limit": hw["limit"], "how": hw["how"]})
    free_fit = [r for r in rows if r["free"] and r["fits_in_memory"]]
    best = min(free_fit, key=lambda r: r["hours_for_full_training"]) if free_fit else None
    if best and best["hours_for_full_training"] <= 60:
        verdict = f"Trainable for free: about {best['hours_for_full_training']} h on {best['option']}."
    elif best:
        verdict = (f"Fits on {best['option']} but needs about {best['hours_for_full_training']} h. That is "
                   f"{round(best['hours_for_full_training'] / 30, 1)} weeks of Kaggle quota. Train on fewer tokens "
                   f"(still useful), or pick a smaller model.")
    else:
        verdict = (f"{fmt(n)} parameters do not fit on any free hardware for training "
                   f"({fmt(16 * n)}B of training state). You can still design it, scaffold it and test the code on a "
                   "tiny copy for free, then train on rented GPUs. Free route to big compute: apply to Google's "
                   "TPU Research Cloud (free Cloud TPUs for accepted research, needs a JAX/XLA port).")
    return {"model": {"params": fmt(n), "config": c}, "train_tokens": fmt(tokens), "train_flops": f"{flops:.2e}",
            "verdict": verdict, "options": rows,
            "assumptions": "effective TFLOPS: phone 0.02, T4 15, H100 400 per device; memory = 16 bytes/param "
                           "(sharded across GPUs with FSDP) + activations at batch 1, seq 1024."}


# ======================================================================= workspace
def ws_path(p):
    """Resolve a project path inside the workspace; refuse anything outside it."""
    p = os.path.expanduser(str(p or ""))
    full = os.path.abspath(p if os.path.isabs(p) else os.path.join(HOME, p))
    if full != HOME and not full.startswith(HOME + os.sep):
        raise ValueError(f"path must be inside the workspace {HOME} (set EXPERTCRAFT_HOME to change it)")
    return full


def notebook(cells):
    nb_cells = []
    for kind, src in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    return {"cells": nb_cells, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "nbformat": 4, "nbformat_minor": 5}


def make_notebooks(name, target, dataset, vocab, steps):
    gc = " --grad_ckpt" if float(target) > 3e8 else ""
    common = f"""!git clone --depth 1 {REPO}.git expertstream
!python expertstream/craft/expertcraft.py scaffold {target:.4g} {name} --vocab {vocab} --dataset {dataset} --into .
%cd {name}
!pip -q install tokenizers datasets"""
    colab = notebook([
        ("markdown", f"# {APP}: train `{name}` on a free Colab T4\nRuntime > Change runtime type > **T4 GPU**, then run all. "
                     f"Checkpoints go to Google Drive, so if the session ends just run all again and it resumes.\n\n{REPO}"),
        ("code", "from google.colab import drive\ndrive.mount('/content/drive')"),
        ("code", common),
        ("code", f"!python data.py --dataset {dataset} --vocab {vocab} --max_tokens 200e6"),
        ("code", f"!python train.py --out /content/drive/MyDrive/expertcraft/{name} --steps {steps} --batch 16 --accum 4 --max_minutes 200{gc}"),
        ("code", f"!python export.py --out /content/drive/MyDrive/expertcraft/{name}\n"
                 f"!python sample.py --out /content/drive/MyDrive/expertcraft/{name} --prompt 'Once upon a time'"),
    ])
    kaggle = notebook([
        ("markdown", f"# {APP}: train `{name}` on free Kaggle 2x T4\nSettings: **Accelerator GPU T4 x2**, **Internet on**. "
                     f"Uses FSDP to split the model over both GPUs. Output is saved in /kaggle/working; to resume in a new "
                     f"session, add the previous version's output as input and copy it back to /kaggle/working/run.\n\n{REPO}"),
        ("code", common),
        ("code", f"!python data.py --dataset {dataset} --vocab {vocab} --max_tokens 300e6"),
        ("code", f"!torchrun --nproc_per_node 2 train.py --fsdp --out /kaggle/working/run --steps {steps} --batch 16 --accum 2 --max_minutes 690{gc}"),
        ("code", "!python export.py --out /kaggle/working/run\n!python sample.py --out /kaggle/working/run --prompt 'Once upon a time'"),
    ])
    return colab, kaggle


def scaffold(path, target, name=None, vocab=None, dataset="tinystories", seq=None):
    full = ws_path(path)
    name = name or os.path.basename(full) or "model"
    c = design(target, vocab, seq)
    n = count_params(c)
    os.makedirs(full, exist_ok=True)
    for fn in ("model.py", "train.py", "data.py", "export.py", "sample.py"):
        shutil.copy(os.path.join(TEMPLATE, fn), os.path.join(full, fn))
    json.dump(c, open(os.path.join(full, "config.json"), "w"), indent=2)
    steps = 2000 if n < 5e7 else 6000 if n < 3e8 else 20000
    colab, kaggle = make_notebooks(name, target, dataset, c["vocab_size"], steps)
    json.dump(colab, open(os.path.join(full, "colab.ipynb"), "w"), indent=1)
    json.dump(kaggle, open(os.path.join(full, "kaggle.ipynb"), "w"), indent=1)
    plan = free_plan(c)
    json.dump({"app": APP, "version": VERSION, "name": name, "target_params": target, "params": n,
               "dataset": dataset, "created": time.strftime("%Y-%m-%d %H:%M")}, open(os.path.join(full, "craft.json"), "w"), indent=2)
    readme = f"""# {name}

Built with **{APP}** ({REPO}). {fmt(n)} parameters, Llama-style (RMSNorm, RoPE, GQA, SwiGLU).

{plan['verdict']}

## Train
```sh
pip install torch numpy tokenizers datasets
python data.py --dataset {dataset} --vocab {c['vocab_size']} --max_tokens 50e6   # free data + BPE tokenizer
python train.py --out run --steps {steps}                                       # resumes automatically
python export.py --out run && python sample.py --out run --prompt "Once upon a time"
```
Multi-GPU (Kaggle 2x T4 or cloud): `torchrun --nproc_per_node 2 train.py --fsdp --out run`.
Big models: add `--grad_ckpt`. Free GPUs: open `colab.ipynb` or `kaggle.ipynb`.

## Free compute for this model
| option | fits | hours | largest model there |
|---|---|---|---|
""" + "".join(f"| {r['option']} | {'yes' if r['fits_in_memory'] else 'no'} | {r['hours_for_full_training']} | {r['largest_trainable_model_here']} |\n"
              for r in plan["options"])
    open(os.path.join(full, "README.md"), "w").write(readme)
    out = {"path": full, "name": name, "params": fmt(n), "config": c,
           "files": sorted(os.listdir(full)), "verdict": plan["verdict"],
           "next": [f"run_step(path='{path}', step='data', args=['--dataset','{dataset}','--max_tokens','20e6'])",
                    f"run_step(path='{path}', step='train', args=['--steps','{steps}'], background=true)",
                    f"job_status(path='{path}')"]}
    if n > 3e8:
        out["note"] = ("Too big to train on a phone. Train it on Kaggle/Colab (notebooks included) or rented GPUs; "
                       "on the phone, test the pipeline with a tiny copy first (scaffold 5e6).")
    return out


# ======================================================================= running steps
STEPS = {"data": "data.py", "train": "train.py", "export": "export.py", "sample": "sample.py"}
PATH_FLAGS = {"--file", "--out", "--data", "--config"}


def check_args(args, proj):
    args = [str(x) for x in (args or [])]
    for i, x in enumerate(args):
        if x.startswith("--") and not re.fullmatch(r"--[a-z_]+", x):
            raise ValueError(f"bad flag {x}")
        if i and args[i - 1] in PATH_FLAGS:
            p = os.path.abspath(os.path.join(proj, os.path.expanduser(x)))
            if not (p == HOME or p.startswith(HOME + os.sep)):
                raise ValueError(f"{args[i - 1]} must point inside the workspace {HOME}")
    return args


def jobs_file(proj):
    return os.path.join(proj, ".jobs.json")


def run_step(path, step, args=None, background=False, gpus=1, timeout=900):
    proj = ws_path(path)
    if step not in STEPS:
        raise ValueError(f"step must be one of {list(STEPS)}")
    if not os.path.exists(os.path.join(proj, STEPS[step])):
        raise ValueError(f"no project at {proj}; call scaffold_project first")
    args = check_args(args, proj)
    gpus = int(gpus or 1)
    if step == "train" and gpus > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(gpus), "train.py", "--fsdp"] + args
    else:
        cmd = [sys.executable, STEPS[step]] + args
    os.makedirs(os.path.join(proj, "logs"), exist_ok=True)
    logp = os.path.join(proj, "logs", f"{step}.log")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if background:
        lf = open(logp, "a")
        lf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd[1:])}\n"); lf.flush()
        p = subprocess.Popen(cmd, cwd=proj, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        jobs = json.load(open(jobs_file(proj))) if os.path.exists(jobs_file(proj)) else {}
        jobs[step] = {"pid": p.pid, "cmd": cmd[1:], "started": time.time(), "log": logp}
        json.dump(jobs, open(jobs_file(proj), "w"), indent=2)
        return {"started": step, "pid": p.pid, "log": logp, "check_with": f"job_status(path='{path}')"}
    try:
        r = subprocess.run(cmd, cwd=proj, capture_output=True, text=True, timeout=float(timeout), env=env)
        out = (r.stdout + r.stderr)
        open(logp, "a").write(out)
        return {"step": step, "exit_code": r.returncode, "output_tail": out[-4000:]}
    except subprocess.TimeoutExpired:
        return {"step": step, "error": f"took longer than {timeout}s; run it with background=true"}


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # zombie check on Linux/Android
        return open(f"/proc/{pid}/stat").read().split()[2] != "Z"
    except Exception:
        return True


def job_status(path):
    proj = ws_path(path)
    jobs = json.load(open(jobs_file(proj))) if os.path.exists(jobs_file(proj)) else {}
    out = {"path": proj, "jobs": {}}
    for step, j in jobs.items():
        tail = ""
        if os.path.exists(j["log"]):
            tail = "".join(open(j["log"], errors="replace").readlines()[-15:])
        out["jobs"][step] = {"running": alive(j["pid"]), "pid": j["pid"],
                             "minutes": round((time.time() - j["started"]) / 60, 1), "log_tail": tail}
    for run in sorted(os.listdir(proj)) if os.path.isdir(proj) else []:
        lp = os.path.join(proj, run, "log.jsonl")
        if os.path.exists(lp):
            recs = [json.loads(x) for x in open(lp).read().splitlines()[-200:] if x.strip()]
            tr = [r for r in recs if "loss" in r]
            va = [r for r in recs if "val_loss" in r]
            out.setdefault("runs", {})[run] = {
                "last": tr[-1] if tr else None, "best_val_loss": min((r["val_loss"] for r in va), default=None),
                "loss_trend": [r["loss"] for r in tr[-10:]],
                "saved": json.load(open(os.path.join(proj, run, "state.json"))) if os.path.exists(os.path.join(proj, run, "state.json")) else None}
    return out


def stop_job(path, step="train"):
    proj = ws_path(path)
    jobs = json.load(open(jobs_file(proj))) if os.path.exists(jobs_file(proj)) else {}
    if step not in jobs or not alive(jobs[step]["pid"]):
        return {"stopped": False, "reason": "not running"}
    os.killpg(jobs[step]["pid"], signal.SIGTERM)  # trainer saves a checkpoint on SIGTERM
    return {"stopped": True, "note": "the trainer saves a checkpoint before exiting; run train again to resume"}




# ======================================================================= phone: Flash Experts (up to 70B parameters, on the device)
def _phone():
    sys.path.insert(0, os.path.join(HERE, "phone"))
    try:
        import phonetrain
    except ImportError as e:
        raise SystemExit(f"{e}\nThe phone commands need NumPy. Termux: pkg install python python-numpy   PC: pip install numpy")
    return phonetrain


def _flag(argv, name, default=None, cast=str):
    if name in argv:
        i = argv.index(name)
        if i + 1 >= len(argv):
            raise SystemExit(f"{name} needs a value")
        return cast(argv[i + 1])
    return default


def _positional(argv, n):
    """first n arguments that are not flags or flag values"""
    out, skip = [], False
    for i, a in enumerate(argv):
        if skip:
            skip = False
            continue
        if a.startswith("--"):
            skip = a not in ("--background", "--oversubscribe", "--json")
            continue
        out.append(a)
    if len(out) < n:
        raise SystemExit("missing argument; run: python3 craft/expertcraft.py help")
    return out


def _orders(argv):
    return tuple(int(x) for x in _flag(argv, "--orders", "1,2,3").split(","))


def phone_plan(argv):
    pt = _phone()
    target = float(_positional(argv, 1)[0])
    p = pt.plan(target, _flag(argv, "--minutes", 600.0, float), HOME, vocab=_flag(argv, "--vocab", 2048, int), layers=_flag(argv, "--layers", 1, int),
                topk=_flag(argv, "--topk", 8, int), heads=_flag(argv, "--heads", 1, int), oversubscribe="--oversubscribe" in argv, route=_flag(argv, "--route", "hash"), orders=_orders(argv), width=_flag(argv, "--width", None, int))
    if "--json" in argv:
        print(json.dumps(p, indent=2, default=str))
        return
    d, e, c = p["device"], p["experts"], p["backbone"]["config"]
    print(f"Device: {d['ram_gb']} GB RAM, {d['cores']} cores, {d['disk_free_gb']} GB free disk" + (f", battery {d['battery_pct']}%" if d["battery_pct"] is not None else ""))
    print(f"Plan for {pt.fmt(target)} parameters in {p['minutes']:.0f} minutes:\n")
    print(f"  dense backbone (RAM):   {pt.fmt(c['params'])}  ({c['layers']} layers, width {c['dim']})")
    print(f"  Flash Experts (disk):   {e['experts_per_layer']:,} experts x {e['layers']} layer(s), {pt.fmt(e['params_per_expert'])} parameters each = {pt.fmt(e['total_params'])}")
    print(f"  total parameters:       {pt.fmt(p['total_params'])}   (file up to {p['file_gb']} GB, grows as experts are used; {d['disk_free_gb']} GB free)")
    print(f"  used per word:          {pt.fmt(p['active_params_per_token'])}  ({100 * p['active_fraction']:.4f}% of the total)")
    print(f"  speed (measured):       {p['tokens_per_second']:.0f} tokens/s  ->  {pt.fmt(p['tokens_affordable'])} tokens in {p['minutes']:.0f} min ({p['tokens_per_backbone_param']} per backbone parameter)")
    print(f"  expert visits:          each expert is used about {p['visits_per_expert']} times")
    print(f"  flash traffic:          {p['pages_per_token']} pages x 4 KB read and written per token; about {p['wear_tb_per_day']} TB written per day of nonstop training")
    if p.get("io"):
        print(f"  storage speed (random 4 KB): read {p['io']['read_mb_s']:.0f} MB/s, write {p['io']['write_mb_s']:.0f} MB/s; training needs {p['flash_mb_s_needed']} MB/s" + (" -> STORAGE LIMITS SPEED" if p["flash_limits_speed"] else ""))
    if e.get("note"):
        print(f"\n  NOTE: {e['note']}")
    print("\nHonest summary:")
    for line in p["honest"]:
        print("  - " + line)
    print(f"\nStart:  python3 craft/expertcraft.py phone-train mymodel --data your.txt --params {target / 1e9:g}e9 --minutes {p['minutes']:.0f}")


def phone_train(argv):
    pt = _phone()
    name = _positional(argv, 1)[0]
    proj = ws_path(name)
    if not os.path.exists(os.path.join(proj, "phone.json")):
        data = _flag(argv, "--data") or _flag(argv, "--dataset")
        if not data:
            raise SystemExit("first run needs --data FILE_OR_FOLDER (or --dataset tinystories)")
        p, meta = pt.create(proj, data, _flag(argv, "--params", 70e9, float), _flag(argv, "--minutes", 600.0, float), vocab=_flag(argv, "--vocab", 2048, int),
                            layers=_flag(argv, "--layers", 1, int), topk=_flag(argv, "--topk", 8, int), heads=_flag(argv, "--heads", 1, int), oversubscribe="--oversubscribe" in argv, route=_flag(argv, "--route", "hash"), orders=_orders(argv), width=_flag(argv, "--width", None, int))
        for line in p["honest"]:
            print("  - " + line)
    res = pt.train(proj, minutes=_flag(argv, "--minutes", None, float), steps=_flag(argv, "--steps", None, int), min_battery=_flag(argv, "--min-battery", 0, int),
                   max_temp=_flag(argv, "--max-temp", 0, float), max_wear_gb=_flag(argv, "--max-wear-gb", None, float), min_free_gb=_flag(argv, "--min-free-gb", None, float))
    print(json.dumps(res, indent=2))


def phone_sample(argv):
    pt = _phone()
    name = _positional(argv, 1)[0]
    print(pt.sample(ws_path(name), _flag(argv, "--prompt", "Once upon a time"), _flag(argv, "--tokens", 80, int), _flag(argv, "--temp", 0.8, float)))


def phone_status(argv):
    print(json.dumps(_phone().status(ws_path(_positional(argv, 1)[0])), indent=2))


# ======================================================================= CLI
def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if "--version" in argv:
        print(APP, VERSION)
    elif cmd == "phone-plan":
        phone_plan(rest)
    elif cmd == "phone-train":
        phone_train(rest)
    elif cmd == "phone-sample":
        phone_sample(rest)
    elif cmd == "phone-status":
        phone_status(rest)
    elif cmd == "plan":
        print(json.dumps(free_plan(float(_positional(rest, 1)[0])), indent=2))
    elif cmd == "design":
        c = design(float(_positional(rest, 1)[0]), _flag(rest, "--vocab"))
        print(json.dumps({"config": c, "estimate": estimate(c)}, indent=2))
    elif cmd == "scaffold":
        global HOME
        into = _flag(rest, "--into")
        if into:  # notebooks scaffold into the current folder
            HOME = os.path.abspath(into)
        pos = _positional(rest, 1)
        out = scaffold(pos[1] if len(pos) > 1 else "model", float(pos[0]), None, _flag(rest, "--vocab"), _flag(rest, "--dataset", "tinystories"))
        print(json.dumps({k: out[k] for k in ("path", "params", "verdict", "files")}, indent=2))
    elif cmd == "run":
        pos = _positional(rest, 2)
        extra = [a for a in rest[2:] if a != "--background"]
        print(json.dumps(run_step(pos[0], pos[1], extra, background="--background" in rest), indent=2))
    elif cmd == "status":
        print(json.dumps(job_status(_positional(rest, 1)[0]), indent=2))
    elif cmd == "stop":
        print(json.dumps(stop_job(_positional(rest, 1)[0]), indent=2))
    else:
        raise SystemExit(f"unknown command {cmd!r}; run: python3 craft/expertcraft.py help")


if __name__ == "__main__":
    main(sys.argv[1:])
