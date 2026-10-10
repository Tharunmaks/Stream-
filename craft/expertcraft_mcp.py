#!/usr/bin/env python3
"""Expert Craft: an MCP server that lets any AI (Claude, ChatGPT, Cursor, Gemini CLI, ...) build
language models from scratch, from 1 million up to 70 billion parameters, using free compute.

Part of ExpertStream: https://github.com/Tharunmaks/Stream-

Run it (Python 3.8+, standard library only, works in Termux):
  python3 expertcraft_mcp.py                      MCP over stdio (Claude Code, Claude Desktop, Cursor, ...)
  python3 expertcraft_mcp.py --http 8766          MCP over HTTP at http://127.0.0.1:8766/s/<secret>/mcp
  python3 expertcraft_mcp.py --http 8766 --public same, plus a free https address (cloudflared) for phone apps
  python3 expertcraft_mcp.py plan 125e6           CLI: free-compute plan for a model size
  python3 expertcraft_mcp.py scaffold 30e6 mymodel  CLI: write a ready-to-train project

Tools: craft_guide, design_model, estimate_model, free_compute_plan, scaffold_project,
       run_step, job_status, stop_job, connect_guide
All project folders live under the workspace (EXPERTCRAFT_HOME, default ~/expertcraft).
"""
import json, math, os, re, secrets, shutil, signal, subprocess, sys, threading, time

VERSION = "1.0.0"
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
     "how": "pkg install python python-numpy; pip install torch may not be available on Termux. "
            "If not, use ExpertStream's NumPy builder (mcp/builder.py) for tiny models."},
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
!python expertstream/craft/expertcraft_mcp.py scaffold {target:.4g} {name} --vocab {vocab} --dataset {dataset} --into .
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


# ======================================================================= guides
GUIDE = f"""{APP} {VERSION}: build a language model from scratch, 1M to 70B parameters, on free compute.

Workflow for the AI:
 1. design_model(target_params) and free_compute_plan(target_params): tell the user honestly what it costs.
 2. scaffold_project(path, target_params, dataset): writes model.py, data.py, train.py, export.py, sample.py,
    config.json and colab.ipynb / kaggle.ipynb.
 3. run_step(path, 'data', ['--dataset','tinystories','--max_tokens','20e6'])  free data + BPE tokenizer.
 4. run_step(path, 'train', ['--steps','2000','--out','run'], background=true), then job_status(path).
    Loss must go down. If it is NaN or flat, lower --lr (e.g. 1e-4).
 5. run_step(path, 'export', ['--out','run']) then run_step(path, 'sample', ['--out','run','--prompt','Hello']).

Rules:
 R1 Never claim a model is trained until job_status shows it finished and val loss went down.
 R2 Always show free_compute_plan before training anything above 100M parameters.
 R3 Start small: test the full pipeline with a 5M model, then scale.
 R4 70B is supported for design, code and multi-GPU training, but it cannot be trained on free hardware;
    say so plainly (about 1.1 TB of training state, ~5.6e23 FLOPs).
 R5 Free GPUs: Colab (1x T4) and Kaggle (2x T4, --fsdp). Phones: tiny models only.
 R6 Use --max_minutes so free sessions end with a saved checkpoint; training resumes automatically.

Workspace: {HOME}   Website: {SITE}   Code: {REPO}
"""


def connect_guide(client="all"):
    me = os.path.abspath(__file__)
    py = "python3"
    snippets = {
        "claude_code": f"claude mcp add expertcraft -- {py} {me}",
        "termux_claude_code": f"pkg install python && claude mcp add expertcraft -- {py} {me}",
        "claude_desktop": {"mcpServers": {"expertcraft": {"command": py, "args": [me]}}},
        "cursor_windsurf_cline": {"mcpServers": {"expertcraft": {"command": py, "args": [me]}}},
        "vscode": {"servers": {"expertcraft": {"type": "stdio", "command": py, "args": [me]}}},
        "codex_cli": f'[mcp_servers.expertcraft]\ncommand = "{py}"\nargs = ["{me}"]',
        "gemini_cli": {"mcpServers": {"expertcraft": {"command": py, "args": [me]}}},
        "phone_apps_by_url": (f"{py} {me} --http 8766 --public   -> prints https://<name>.trycloudflare.com/s/<secret>/mcp. "
                              "Paste that link where the app lets you add a custom connector / MCP server. "
                              "cloudflared is free: pkg install cloudflared (Termux) or see developers.cloudflare.com. "
                              "Keep the link private: the secret in it is the password."),
    }
    if client != "all" and client in snippets:
        return {client: snippets[client]}
    return snippets


# ======================================================================= MCP protocol
CFG = {k: {"type": "integer"} for k in ("d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff", "vocab_size", "max_seq_len")}
CFG["tie_embeddings"] = {"type": "boolean"}
TOOLS = [
    {"name": "craft_guide", "description": f"{APP}: how to build a model from scratch with these tools, and the rules. Read first.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "design_model", "description": "Design a Llama-style config for a parameter target (1e6 to 70e9). Returns config + cost estimate.",
     "inputSchema": {"type": "object", "properties": {"target_params": {"type": "number", "description": "e.g. 30e6, 1e9, 70e9"},
                                                      "vocab_size": {"type": "integer"}, "max_seq_len": {"type": "integer"}},
                     "required": ["target_params"]}},
    {"name": "estimate_model", "description": "Exact parameter count, memory, FLOPs and training time for a given config.",
     "inputSchema": {"type": "object", "properties": {**CFG, "tokens": {"type": "number"}, "gpus": {"type": "integer"},
                                                      "effective_tflops": {"type": "number"}},
                     "required": ["d_model", "n_layers", "n_heads", "n_kv_heads", "d_ff", "vocab_size"]}},
    {"name": "free_compute_plan", "description": "Where this model can be trained for free (phone, Colab T4, Kaggle 2xT4), how long it takes, and the largest model each option can train.",
     "inputSchema": {"type": "object", "properties": {"target_params": {"type": "number"}, "tokens": {"type": "number"}},
                     "required": ["target_params"]}},
    {"name": "scaffold_project", "description": "Create a ready-to-train project folder (PyTorch model, BPE data prep, FSDP trainer, export, sampler, Colab and Kaggle notebooks).",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string", "description": "folder name inside the workspace, e.g. 'tiny-stories'"},
         "target_params": {"type": "number"}, "name": {"type": "string"}, "vocab_size": {"type": "integer"},
         "max_seq_len": {"type": "integer"},
         "dataset": {"type": "string", "enum": ["tinystories", "fineweb-edu", "wikitext", "text"]}},
         "required": ["path", "target_params"]}},
    {"name": "run_step", "description": "Run a project step: data | train | export | sample. Long steps: background=true then job_status. gpus>1 trains with FSDP via torchrun.",
     "inputSchema": {"type": "object", "properties": {
         "path": {"type": "string"}, "step": {"type": "string", "enum": list(STEPS)},
         "args": {"type": "array", "items": {"type": "string"}, "description": "CLI flags, e.g. ['--steps','500','--out','run']"},
         "background": {"type": "boolean"}, "gpus": {"type": "integer"}, "timeout": {"type": "number"}},
         "required": ["path", "step"]}},
    {"name": "job_status", "description": "Is training running, latest loss, loss trend, best validation loss, saved checkpoints.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "stop_job", "description": "Stop a background step; the trainer saves a checkpoint first.",
     "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}, "step": {"type": "string"}}, "required": ["path"]}},
    {"name": "connect_guide", "description": "Config snippets to connect Expert Craft to Claude Code, Claude Desktop, Cursor, VS Code, Codex, Gemini CLI, or phone apps by URL.",
     "inputSchema": {"type": "object", "properties": {"client": {"type": "string"}}}},
]


def call(name, a):
    if name == "craft_guide":
        return GUIDE
    if name == "design_model":
        c = design(a["target_params"], a.get("vocab_size"), a.get("max_seq_len"))
        return {"config": c, "estimate": estimate(c), "free_compute": free_plan(c)["verdict"]}
    if name == "estimate_model":
        c = {k: a[k] for k in CFG if k in a}
        c.setdefault("tie_embeddings", False)
        return estimate(c, a.get("tokens"), a.get("effective_tflops", 400.0), a.get("gpus", 1))
    if name == "free_compute_plan":
        return free_plan(a["target_params"], a.get("tokens"))
    if name == "scaffold_project":
        return scaffold(a["path"], a["target_params"], a.get("name"), a.get("vocab_size"),
                        a.get("dataset", "tinystories"), a.get("max_seq_len"))
    if name == "run_step":
        return run_step(a["path"], a["step"], a.get("args"), a.get("background", False), a.get("gpus", 1), a.get("timeout", 900))
    if name == "job_status":
        return job_status(a["path"])
    if name == "stop_job":
        return stop_job(a["path"], a.get("step", "train"))
    if name == "connect_guide":
        return connect_guide(a.get("client", "all"))
    raise ValueError(f"unknown tool {name}")


def handle(r):
    """One JSON-RPC message -> response dict, or None for notifications."""
    m, i = r.get("method"), r.get("id")
    if i is None:
        return None
    ok = lambda res: {"jsonrpc": "2.0", "id": i, "result": res}
    try:
        if m == "initialize":
            return ok({"protocolVersion": (r.get("params") or {}).get("protocolVersion", "2025-06-18"),
                       "capabilities": {"tools": {"listChanged": False}},
                       "serverInfo": {"name": "expertcraft", "title": APP, "version": VERSION},
                       "instructions": GUIDE})
        if m == "tools/list":
            return ok({"tools": TOOLS})
        if m == "tools/call":
            p = r.get("params") or {}
            try:
                out = call(p.get("name"), p.get("arguments") or {})
                text = out if isinstance(out, str) else json.dumps(out, indent=2)
                return ok({"content": [{"type": "text", "text": text}]})
            except Exception as e:
                return ok({"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
        if m == "ping":
            return ok({})
        return {"jsonrpc": "2.0", "id": i, "error": {"code": -32601, "message": f"method not found: {m}"}}
    except Exception as e:
        return {"jsonrpc": "2.0", "id": i, "error": {"code": -32603, "message": str(e)}}


def serve_stdio():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}) + "\n")
            sys.stdout.flush(); continue
        msgs = r if isinstance(r, list) else [r]
        outs = [o for o in (handle(x) for x in msgs) if o]
        if outs:
            sys.stdout.write(json.dumps(outs if isinstance(r, list) else outs[0]) + "\n"); sys.stdout.flush()


def get_secret():
    p = os.path.join(HOME, ".secret")
    os.makedirs(HOME, exist_ok=True)
    if not os.path.exists(p):
        with open(p, "w") as f:
            f.write(secrets.token_urlsafe(24))
        os.chmod(p, 0o600)
    return open(p).read().strip()


def serve_http(port, host="127.0.0.1", public=False):
    import http.server
    secret = get_secret()
    route = f"/s/{secret}/mcp"

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body=b"", ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.split("?")[0] == route:
                return self._send(405, b'{"error":"use POST (streamable HTTP, JSON responses)"}')
            if self.path == "/":
                return self._send(200, f"{APP} MCP server is running. {SITE}".encode(), "text/plain")
            self._send(404, b"{}")

        def do_DELETE(self):
            self._send(200 if self.path.split("?")[0] == route else 404, b"{}")

        def do_POST(self):
            if self.path.split("?")[0] != route:
                return self._send(404, b'{"error":"not found"}')
            n = int(self.headers.get("Content-Length") or 0)
            if n > 2_000_000:
                return self._send(413, b"{}")
            try:
                r = json.loads(self.rfile.read(n) or b"null")
            except json.JSONDecodeError:
                return self._send(400, b'{"jsonrpc":"2.0","id":null,"error":{"code":-32700,"message":"parse error"}}')
            msgs = r if isinstance(r, list) else [r]
            outs = [o for o in (handle(x) for x in msgs if isinstance(x, dict)) if o]
            if not outs:
                return self._send(202)
            self._send(200, json.dumps(outs if isinstance(r, list) else outs[0]).encode())

    srv = http.server.ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    print(f"{APP} MCP (HTTP) on http://{host}:{port}{route}", flush=True)
    if public:
        threading.Thread(target=tunnel, args=(port, route), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def tunnel(port, route):
    exe = shutil.which("cloudflared")
    if not exe:
        print("cloudflared not found. Install it (Termux: pkg install cloudflared) or use a free ssh tunnel:\n"
              f"  ssh -R 80:localhost:{port} nokey@localhost.run   then add {route} to the https address it prints", flush=True)
        return
    p = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m:
            print(f"\nPublic MCP link (paste into your AI app, keep it private):\n  {m.group(0)}{route}\n", flush=True)


# ======================================================================= CLI
def main(argv):
    def opt(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default

    if argv and argv[0] == "plan":
        print(json.dumps(free_plan(float(argv[1])), indent=2)); return
    if argv and argv[0] == "design":
        c = design(float(argv[1]), opt("--vocab"))
        print(json.dumps({"config": c, "estimate": estimate(c)}, indent=2)); return
    if argv and argv[0] == "scaffold":
        global HOME
        into = opt("--into")
        if into:  # notebooks scaffold into the current folder
            HOME = os.path.abspath(into)
        out = scaffold(argv[2] if len(argv) > 2 else "model", float(argv[1]), None, opt("--vocab"), opt("--dataset", "tinystories"))
        print(json.dumps({k: out[k] for k in ("path", "params", "verdict", "files")}, indent=2)); return
    if argv and argv[0] == "connect":
        print(json.dumps(connect_guide(), indent=2)); return
    if argv and argv[0] in ("-h", "--help", "help"):
        print(__doc__); return
    if "--version" in argv:
        print(APP, VERSION); return
    if "--http" in argv:
        serve_http(int(opt("--http")), opt("--host", "127.0.0.1"), "--public" in argv); return
    serve_stdio()


if __name__ == "__main__":
    main(sys.argv[1:])
