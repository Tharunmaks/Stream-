"""Expert Stream LoRA Trainer: tiny Flask app driving es_lora / es_chat. Works on Linux and Termux."""
import os, re, json, glob, time, signal, subprocess, threading
from flask import Flask, request, jsonify, send_file, send_from_directory
HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("ES_BIN_DIR", "/workspace/es-speedup")
MODELS = os.environ.get("ES_MODEL_DIR", "/workspace/es-bench")
RUNS = os.path.join(HERE, "runs"); DATA = os.path.join(HERE, "data")
FAST_ENV = {"ES_AMX": "1", "ES_LORA_BLAS": "1",
            "ES_BLAS": os.environ.get("ES_BLAS", "/usr/local/lib/python3.13/dist-packages/numpy.libs/libscipy_openblas64_-f48b354e.so")}
app = Flask(__name__, static_folder=None)
JOB = {"proc": None, "run": None, "cfg": None, "lines": [], "steps": [], "heldout": [], "state": "idle", "rss": 0, "t0": 0, "done_offset": 0}
STEP = re.compile(r"step (\d+)/(\d+)\s+loss ([\d.]+).*?lr ([\de.+-]+)\s+([\d.]+) s\s+(\d+) tokens/s")
HELD = re.compile(r"held-out loss (?:before|after step (\d+)): ([\d.]+)")

def rss(pid):
    try:
        for l in open(f"/proc/{pid}/status"):
            if l.startswith("VmRSS"): return int(l.split()[1]) // 1024
    except Exception: pass
    return 0

def reader(p):
    for raw in p.stderr:
        l = raw.decode(errors="replace").rstrip()
        JOB["lines"] = (JOB["lines"] + [l])[-200:]
        m = STEP.search(l)
        if m:
            JOB["steps"].append({"step": int(m[1]) + JOB["done_offset"], "of": int(m[2]) + JOB["done_offset"], "loss": float(m[3]),
                                 "lr": float(m[4]), "secs": float(m[5]), "tps": int(m[6]), "rss": JOB["rss"]})
        h = HELD.search(l)
        if h: JOB["heldout"].append({"step": int(h[1] or 0) + (JOB["done_offset"] if h[1] else 0), "loss": float(h[2])})
    p.wait()
    JOB["state"] = "stopped" if JOB["state"] == "stopping" else ("finished" if p.returncode == 0 else f"error ({p.returncode})")

def sampler():
    while True:
        p = JOB["proc"]
        if p and p.poll() is None: JOB["rss"] = rss(p.pid)
        time.sleep(1)
threading.Thread(target=sampler, daemon=True).start()

def convert_dataset(src, dst):
    """Accepts .txt, or JSONL with text / prompt+response / messages / any two columns (question,answer, q,a, input,output...)."""
    if src.endswith(".txt"):
        os.replace(src, dst + ".txt"); return dst + ".txt", "text"
    out, n = [], 0
    for line in open(src, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line: continue
        o = json.loads(line)
        if "text" in o or "messages" in o or ("prompt" in o and ("response" in o or "completion" in o)): out.append(o)
        else:
            ks = [k for k, v in o.items() if isinstance(v, str)]
            pk = next((k for k in ks if k.lower() in ("question", "q", "input", "instruction", "prompt", "query")), ks[0] if ks else None)
            rk = next((k for k in ks if k.lower() in ("answer", "a", "output", "response", "completion", "reply")), ks[1] if len(ks) > 1 else None)
            if not pk or not rk: raise ValueError("need two text columns per line")
            out.append({"prompt": o[pk], "response": o[rk]})
        n += 1
    with open(dst + ".jsonl", "w") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    return dst + ".jsonl", f"{n} examples"

@app.get("/")
def index(): return send_from_directory(os.path.join(HERE, "static"), "index.html")

@app.get("/static/<path:f>")
def st(f): return send_from_directory(os.path.join(HERE, "static"), f)

@app.get("/api/models")
def models(): return jsonify(sorted(glob.glob(os.path.join(MODELS, "*.gguf")) + glob.glob(os.path.join(HERE, "models", "*.gguf"))))

@app.post("/api/hf")
def hf():
    repo = request.json.get("repo", "").strip()
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
        files = list_repo_files(repo)
    except Exception as e: return jsonify(error=f"cannot list {repo}: {e}"), 400
    g = [f for f in files if f.endswith(".gguf")]
    if not g:
        st = any(f.endswith(".safetensors") for f in files)
        return jsonify(error=("This repo has only safetensors weights. Unsupported here: es_lora needs a GGUF. "
                              "Convert with llama.cpp convert_hf_to_gguf.py, or use a '-GGUF' repo.") if st else "no GGUF files in repo"), 400
    pick = next((f for f in g if "q8_0" in f.lower()), sorted(g, key=len)[0])
    path = hf_hub_download(repo, pick, local_dir=os.path.join(HERE, "models"))
    return jsonify(path=path)

@app.get("/api/datasets")
def datasets(): return jsonify(sorted(glob.glob(os.path.join(DATA, "*.jsonl")) + glob.glob(os.path.join(DATA, "*.txt"))))

@app.post("/api/upload")
def upload():
    f = request.files["file"]; name = re.sub(r"[^\w.-]", "_", f.filename)
    tmp = os.path.join(DATA, "raw_" + name); f.save(tmp)
    try: path, info = convert_dataset(tmp, os.path.join(DATA, os.path.splitext(name)[0]))
    except Exception as e: return jsonify(error=str(e)), 400
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    return jsonify(path=path, info=info)

def start(cfg, resume=False):
    run = cfg["run"]; out = os.path.join(RUNS, run + ".eslora")
    steps = int(cfg["steps"])
    if resume:
        done = JOB["steps"][-1]["step"] if JOB["steps"] and JOB["run"] == run else 0
        steps = max(1, steps - done); JOB["done_offset"] = done
    else:
        JOB.update(steps=[], heldout=[], done_offset=0)
    seq = int(cfg["seq"]); batch = max(seq, int(cfg.get("batch", 640)))
    cmd = [os.path.join(BIN, "es_lora"), "train", "-g", cfg["model"], "--data", cfg["data"], "--rank", str(cfg["rank"]),
           "--alpha", str(cfg["alpha"]), "--targets", cfg["targets"], "--steps", str(steps), "--seq", str(seq),
           "--batch", str(batch), "--micro", str(batch), "--lr", str(cfg["lr"]), "-j", str(cfg["threads"]),
           "--eval-every", "5", "-o", out] + (["--resume"] if resume else [])
    env = dict(os.environ)
    if cfg.get("fast"): env.update(FAST_ENV)
    if cfg.get("recompute"): env["ES_LORA_RECOMPUTE"] = "1"
    JOB.update(cfg=cfg, run=run, state="running", t0=time.time(), rss=0, lines=["$ " + " ".join(cmd)])
    JOB["proc"] = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, cwd=RUNS)
    threading.Thread(target=reader, args=(JOB["proc"],), daemon=True).start()

@app.post("/api/train")
def train():
    if JOB["proc"] and JOB["proc"].poll() is None: return jsonify(error="a run is active"), 409
    cfg = request.json; cfg["run"] = re.sub(r"[^\w-]", "_", cfg.get("run") or time.strftime("run_%H%M%S"))
    start(cfg); return jsonify(ok=True, run=cfg["run"])

@app.post("/api/stop")
def stop():
    p = JOB["proc"]
    if p and p.poll() is None: JOB["state"] = "stopping"; p.send_signal(signal.SIGINT)
    return jsonify(ok=True)

@app.post("/api/resume")
def resume():
    if JOB["proc"] and JOB["proc"].poll() is None: return jsonify(error="a run is active"), 409
    if not JOB["cfg"]: return jsonify(error="nothing to resume"), 400
    start(JOB["cfg"], resume=True); return jsonify(ok=True)

@app.get("/api/status")
def status():
    s = JOB["steps"]; cfg = JOB["cfg"] or {}
    eta = None
    if s and JOB["state"] == "running":
        avg = sum(x["secs"] for x in s[-5:]) / len(s[-5:]); eta = round(avg * (s[-1]["of"] - s[-1]["step"]))
    return jsonify(state=JOB["state"], run=JOB["run"], cfg=cfg, steps=s, heldout=JOB["heldout"], rss=JOB["rss"],
                   log=JOB["lines"][-12:], eta=eta, elapsed=round(time.time() - JOB["t0"]) if JOB["t0"] else 0)

@app.get("/api/runs")
def runs():
    r = []
    for f in sorted(glob.glob(os.path.join(RUNS, "*.eslora")), key=os.path.getmtime, reverse=True):
        n = os.path.basename(f)[:-7]; g = os.path.join(RUNS, n + ".gguf")
        r.append({"run": n, "eslora_mb": round(os.path.getsize(f) / 1e6, 2), "gguf": os.path.exists(g)})
    return jsonify(r)

@app.post("/api/export/<run>")
def export(run):
    run = re.sub(r"[^\w-]", "_", run); model = (JOB["cfg"] or {}).get("model") or request.json.get("model")
    p = subprocess.run([os.path.join(BIN, "es_lora"), "export", "--lora", os.path.join(RUNS, run + ".eslora"), "-g", model, "-j", "1",
                        "-o", os.path.join(RUNS, run + ".gguf")], capture_output=True, text=True)
    return (jsonify(ok=True) if p.returncode == 0 else (jsonify(error=p.stderr[-500:]), 500))

@app.get("/dl/<path:name>")
def dl(name):
    name = os.path.basename(name)
    return send_file(os.path.join(RUNS, name), as_attachment=True)

CHAT = {"proc": None, "key": None, "lock": threading.Lock()}
@app.post("/api/chat")
def chat():
    j = request.json; run = re.sub(r"[^\w-]", "_", j.get("run") or ""); model = j["model"]
    key = (model, run)
    with CHAT["lock"]:
        p = CHAT["proc"]
        if not p or p.poll() is not None or CHAT["key"] != key:
            if p and p.poll() is None: p.kill()
            cmd = [os.path.join(BIN, "es_chat"), "-g", model, "-J", "-j", str(j.get("threads", 1)), "-n", str(j.get("max", 96)), "-t", "0.7", "-C", "512"]
            if run: cmd += ["--lora", os.path.join(RUNS, run + ".eslora")]
            p = CHAT["proc"] = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            CHAT["key"] = key
            while True:
                l = p.stdout.readline()
                if not l: return jsonify(error="es_chat exited"), 500
                if '"ready"' in l: break
        if j.get("reset"):
            p.stdin.write("/reset\n"); p.stdin.flush()
            while '"done"' not in (p.stdout.readline() or '"done"'): pass
        p.stdin.write(j["msg"].replace("\\", "\\\\").replace("\n", "\\n") + "\n"); p.stdin.flush()
        text, stats = "", {}
        while True:
            l = p.stdout.readline()
            if not l: break
            try: e = json.loads(l)
            except Exception: continue
            if e.get("ev") == "tok": text += e["t"]
            elif e.get("ev") == "done":
                stats = e; break
        return jsonify(reply=text, tps=stats.get("tps"), ram_mb=stats.get("ram_mb"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8765)), threaded=True)
