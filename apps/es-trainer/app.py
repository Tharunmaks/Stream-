"""Expert Stream LoRA Trainer: tiny Flask app driving es_lora / es_chat. Works on Linux and Termux.

Alone (./run.sh) it is a single-user app. Inside ExpertStream Cloud (gateway/omni_router.py mounts it at /training) every request
carries the signed-in user (environ "es.user") and their plan's limits ("es.limits"): each user gets their own job, chat, uploads
and runs, models come only from the model library, datasets only from the user's own uploads or the shared samples, and at most
ES_TRAIN_MAX_JOBS trainings run on the server at once."""
import os, re, shlex, shutil, json, glob, time, signal, subprocess, threading
from flask import Flask, request, jsonify, send_file, send_from_directory
HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("ES_BIN_DIR", "/workspace/es-int")
def trainer():
    """Prefer the unified es_train (modes finetune/moe); fall back to the old es_lora."""
    t = os.path.join(BIN, "es_train")
    return ([t, "finetune"] if os.path.exists(t) else [os.path.join(BIN, "es_lora")])
MODELS = os.environ.get("ES_MODEL_DIR", "/workspace/es-bench")
RUNS = os.environ.get("ES_TRAIN_RUNS") or os.path.join(HERE, "runs")       # per user: RUNS/<user>/
SAMPLES = os.path.join(HERE, "data")                                         # sample datasets, shared and read-only
DATA = os.environ.get("ES_TRAIN_DATA") or SAMPLES                            # uploads, per user: DATA/<user>/
FAST_ENV = {"ES_AMX": "1"}   # bf16 AMX tiles on Intel Xeon (4th gen and later); exact float32 everywhere else
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("ES_TRAIN_MAX_UPLOAD_MB", "64")) << 20
MAX_JOBS = int(os.environ.get("ES_TRAIN_MAX_JOBS", "1"))     # trainings running at once on this server (CPU bound)
MAX_CHATS = int(os.environ.get("ES_TRAIN_MAX_CHATS", "2"))   # es_chat processes kept alive (each holds a model in RAM)
JOBS = {}   # user -> job

def new_job():
    return {"proc": None, "run": None, "cfg": None, "lines": [], "steps": [], "heldout": [], "state": "idle", "rss": 0, "t0": 0, "done_offset": 0, "mode": "finetune"}

def who():
    """(user, limits): the signed-in Cloud user and their plan's limits; 'local' with no limits when run alone."""
    u = request.environ.get("es.user")
    return (re.sub(r"[^\w.@-]", "_", str(u)) if u else "local"), (request.environ.get("es.limits") or {})

def job():
    return JOBS.setdefault(who()[0], new_job())

def udir(base):
    d = os.path.join(base, who()[0]); os.makedirs(d, exist_ok=True); return d

def running():
    return sum(1 for j in JOBS.values() if j["proc"] and j["proc"].poll() is None)

def allowed_model(path):
    rp = os.path.realpath(path or "")
    return any(os.path.realpath(m) == rp for m in model_list())

def allowed_data(path):
    rp = os.path.realpath(path or ""); ok = (os.path.realpath(SAMPLES) + os.sep, os.path.realpath(udir(DATA)) + os.sep)
    return os.path.isfile(rp) and (os.path.dirname(rp) + os.sep in ok)
STEP = re.compile(r"step (\d+)/(\d+)\s+loss ([\d.]+).*?lr ([\de.+-]+)\s+([\d.]+) s\s+(\d+) tokens/s")
MOESTEP = re.compile(r"^step (\d+) loss ([\d.]+) aux ([\d.]+).*? ([\d.]+)s ([\d.]+) tok/s.*?hit ([\d.]+)%.*?rss (\d+) MB")
MOEHELD = re.compile(r"^held-out loss ([\d.]+)")
HELD = re.compile(r"held-out loss (?:before|after step (\d+)): ([\d.]+)")

def rss(pid):
    try:
        for l in open(f"/proc/{pid}/status"):
            if l.startswith("VmRSS"): return int(l.split()[1]) // 1024
    except Exception: pass
    return 0

def reader(p, JOB):
    for raw in p.stderr:
        l = raw.decode(errors="replace").rstrip()
        JOB["lines"] = (JOB["lines"] + [l])[-200:]
        if JOB["mode"] == "moe":
            m = MOESTEP.search(l)
            if m: JOB["steps"].append({"step": int(m[1]), "of": int(JOB["cfg"]["steps"]), "loss": float(m[2]), "aux": float(m[3]),
                                       "secs": float(m[4]), "tps": round(float(m[5])), "hit": float(m[6]), "rss": int(m[7])})
            h = MOEHELD.search(l)
            if h: JOB["heldout"].append({"step": int(JOB["cfg"]["steps"]), "loss": float(h[1])})
            continue
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
        for J in list(JOBS.values()):
            p = J["proc"]
            if p and p.poll() is None: J["rss"] = rss(p.pid)
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

def model_list(): return sorted(glob.glob(os.path.join(MODELS, "*.gguf")) + glob.glob(os.path.join(HERE, "models", "*.gguf")))

@app.get("/api/models")
def models(): return jsonify(model_list())

@app.post("/api/hf")
def hf():
    if who()[1].get("hf_download") is False:
        return jsonify(error="Adding models from Hugging Face needs a paid plan. Use a model already in the library, or upgrade."), 402
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
def datasets():   # the shared samples, then this user's uploads
    return jsonify(sorted(glob.glob(os.path.join(SAMPLES, "*.jsonl")) + glob.glob(os.path.join(SAMPLES, "*.txt")))
                   + sorted(glob.glob(os.path.join(udir(DATA), "*.jsonl")) + glob.glob(os.path.join(udir(DATA), "*.txt"))))

@app.post("/api/upload")
def upload():
    f = request.files["file"]; name = re.sub(r"[^\w.-]", "_", f.filename); d = udir(DATA)
    tmp = os.path.join(d, "raw_" + name); f.save(tmp)
    try: path, info = convert_dataset(tmp, os.path.join(d, os.path.splitext(name)[0]))
    except Exception as e: return jsonify(error=str(e)), 400
    finally:
        if os.path.exists(tmp): os.remove(tmp)
    return jsonify(path=path, info=info)

def moe_text(path):
    """es_train moe reads raw bytes; turn JSONL Q&A into plain 'Q: .. A: ..' text once."""
    if not path.endswith(".jsonl"): return path
    out = path[:-6] + ".moe.txt"
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(path):
        with open(out, "w") as f:
            for line in open(path):
                if not line.strip(): continue
                o = json.loads(line)
                if "text" in o: f.write(o["text"] + "\n\n")
                elif "messages" in o: f.write("\n".join(f'{m.get("role","")}: {m.get("content","")}' for m in o["messages"]) + "\n\n")
                else: f.write(f'Q: {o.get("prompt","")}\nA: {o.get("response", o.get("completion",""))}\n\n')
    return out

def start_moe(cfg, JOB, runs):
    L, E, D, F = (str(int(cfg[k])) for k in ("layers", "experts", "dim", "ffn"))
    ef = os.path.join(RUNS, f"experts_L{L}_E{E}_d{D}_f{F}.bin")   # random frozen experts: one shared file per shape
    tr = os.path.join(BIN, "es_train")
    gen = [tr, "moe", "gen", "--experts-file", ef, "--layers", L, "--experts", E, "--dim", D, "--ffn", F]
    cmd = [tr, "moe", "train", "--experts-file", ef, "--layers", L, "--experts", E, "--dim", D, "--ffn", F,
           "--topk", str(int(cfg["topk"])), "--rank", str(int(cfg["rank"])), "--steps", str(int(cfg["steps"])),
           "--cache-mb", str(int(cfg["cache_mb"])), "--seq", str(int(cfg["seq"])), "--batch-seqs", str(int(cfg.get("batch_seqs", 4))),
           "--balance", str(float(cfg.get("balance", 0))), "--data", moe_text(cfg["data"])] + (["--random-route"] if cfg.get("random_route") else [])
    sh = ("" if os.path.exists(ef) else " ".join(map(shlex.quote, gen)) + " && ") + "exec " + ("stdbuf -oL " if shutil.which("stdbuf") else "") + " ".join(map(shlex.quote, cmd))
    env = dict(os.environ)
    JOB.update(cfg=cfg, run=cfg["run"], mode="moe", state="running", t0=time.time(), rss=0, steps=[], heldout=[], done_offset=0, lines=["$ " + sh])
    JOB["proc"] = subprocess.Popen(["sh", "-c", sh], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=runs, start_new_session=True)
    p = JOB["proc"]; p.stderr = p.stdout
    threading.Thread(target=reader, args=(p, JOB), daemon=True).start()

def start(cfg, JOB, runs, resume=False):
    if cfg.get("mode") == "moe": return start_moe(cfg, JOB, runs)
    JOB["mode"] = "finetune"
    run = cfg["run"]; out = os.path.join(runs, run + ".eslora")
    steps = int(cfg["steps"])
    if resume:
        done = JOB["steps"][-1]["step"] if JOB["steps"] and JOB["run"] == run else 0
        steps = max(1, steps - done); JOB["done_offset"] = done
    else:
        JOB.update(steps=[], heldout=[], done_offset=0)
    seq = int(cfg["seq"]); batch = max(seq, int(cfg.get("batch", 640)))
    cmd = trainer() + ["train", "-g", cfg["model"], "--data", cfg["data"], "--rank", str(cfg["rank"]),
           "--alpha", str(cfg["alpha"]), "--targets", cfg["targets"], "--steps", str(steps), "--seq", str(seq),
           "--batch", str(batch), "--micro", str(batch), "--lr", str(cfg["lr"]), "-j", str(cfg["threads"]),
           "--eval-every", "5", "-o", out] + (["--resume"] if resume else [])
    env = dict(os.environ)
    if cfg.get("fast"): env.update(FAST_ENV)
    if cfg.get("recompute"): env.update(ES_LORA_RECOMPUTE="1", ES_WCACHE_MB="0")   # low RAM: no kept activations, no kept weights
    JOB.update(cfg=cfg, run=run, state="running", t0=time.time(), rss=0, lines=["$ " + " ".join(cmd)])
    JOB["proc"] = subprocess.Popen(cmd, env=env, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, cwd=runs)
    threading.Thread(target=reader, args=(JOB["proc"], JOB), daemon=True).start()

def check(cfg):
    """A reason the run is not allowed (plan limits, files that are not this user's), or None."""
    lim = who()[1]
    if running() >= MAX_JOBS: return "the server is training for someone else right now: try again in a few minutes", 429
    if cfg.get("mode") == "moe":
        if lim.get("moe") is False: return "training a mixture-of-experts model needs a paid plan", 402
    elif not allowed_model(cfg.get("model")): return "pick a model from the model library", 400
    if not allowed_data(cfg.get("data")): return "pick one of your datasets (or a sample)", 400
    if lim.get("max_steps") and int(cfg.get("steps") or 0) > lim["max_steps"]:
        return "your plan allows up to %d steps per run: lower the steps or upgrade" % lim["max_steps"], 402
    t = int(cfg.get("threads") or 0)   # 0: every core (the plan's limit in the Cloud)
    if lim.get("max_threads"): t = min(t, int(lim["max_threads"])) if t > 0 else int(lim["max_threads"])
    cfg["threads"] = max(t, 0)
    return None

@app.post("/api/train")
def train():
    JOB = job()
    if JOB["proc"] and JOB["proc"].poll() is None: return jsonify(error="a run is active"), 409
    cfg = request.json; cfg["run"] = re.sub(r"[^\w-]", "_", cfg.get("run") or time.strftime("run_%H%M%S"))
    bad = check(cfg)
    if bad: return jsonify(error=bad[0]), bad[1]
    start(cfg, JOB, udir(RUNS)); return jsonify(ok=True, run=cfg["run"])

@app.post("/api/stop")
def stop():
    JOB = job(); p = JOB["proc"]
    if p and p.poll() is None:
        JOB["state"] = "stopping"
        if JOB["mode"] == "moe": os.killpg(p.pid, signal.SIGTERM)
        else: p.send_signal(signal.SIGINT)
    return jsonify(ok=True)

@app.post("/api/resume")
def resume():
    JOB = job()
    if JOB["proc"] and JOB["proc"].poll() is None: return jsonify(error="a run is active"), 409
    if not JOB["cfg"]: return jsonify(error="nothing to resume"), 400
    if JOB["mode"] == "moe": return jsonify(error="MoE prototype runs keep no checkpoint, so they can't resume. Start a new run."), 400
    if running() >= MAX_JOBS: return jsonify(error="the server is training for someone else right now: try again in a few minutes"), 429
    start(JOB["cfg"], JOB, udir(RUNS), resume=True); return jsonify(ok=True)

@app.get("/api/status")
def status():
    JOB = job(); s = JOB["steps"]; cfg = JOB["cfg"] or {}
    eta = None
    if s and JOB["state"] == "running":
        avg = sum(x["secs"] for x in s[-5:]) / len(s[-5:]); eta = round(avg * (s[-1]["of"] - s[-1]["step"]))
    return jsonify(state=JOB["state"], mode=JOB["mode"], run=JOB["run"], cfg=cfg, steps=s, heldout=JOB["heldout"], rss=JOB["rss"],
                   log=JOB["lines"][-12:], eta=eta, elapsed=round(time.time() - JOB["t0"]) if JOB["t0"] else 0,
                   limits=who()[1], busy=running() >= MAX_JOBS and not (JOB["proc"] and JOB["proc"].poll() is None))

@app.get("/api/runs")
def runs():
    r = []; runs_dir = udir(RUNS)
    for f in sorted(glob.glob(os.path.join(runs_dir, "*.eslora")), key=os.path.getmtime, reverse=True):
        n = os.path.basename(f)[:-7]; g = os.path.join(runs_dir, n + ".gguf")
        r.append({"run": n, "eslora_mb": round(os.path.getsize(f) / 1e6, 2), "gguf": os.path.exists(g)})
    return jsonify(r)

@app.post("/api/export/<run>")
def export(run):
    run = re.sub(r"[^\w-]", "_", run); model = (job()["cfg"] or {}).get("model") or (request.json or {}).get("model")
    if not allowed_model(model): return jsonify(error="pick a model from the model library"), 400
    runs_dir = udir(RUNS)
    p = subprocess.run(trainer() + ["export", "--lora", os.path.join(runs_dir, run + ".eslora"), "-g", model, "-j", "1",
                        "-o", os.path.join(runs_dir, run + ".gguf")], capture_output=True, text=True)
    return (jsonify(ok=True) if p.returncode == 0 else (jsonify(error=p.stderr[-500:]), 500))

@app.get("/dl/<path:name>")
def dl(name):
    name = os.path.basename(name)
    return send_file(os.path.join(udir(RUNS), name), as_attachment=True)

CHATS = {}; CHAT_LOCK = threading.Lock()   # user -> {"proc", "key", "lock", "used"}
def chat_slot():
    """This user's chat process slot; when MAX_CHATS are alive, the least recently used other one is closed."""
    u = who()[0]
    with CHAT_LOCK:
        c = CHATS.get(u)
        if not c:
            alive = [(k, v) for k, v in CHATS.items() if v["proc"] and v["proc"].poll() is None]
            while len(alive) >= MAX_CHATS:
                k, v = min(alive, key=lambda kv: kv[1]["used"]); v["proc"].kill(); del CHATS[k]; alive.remove((k, v))
            c = CHATS[u] = {"proc": None, "key": None, "lock": threading.Lock(), "used": time.time()}
        c["used"] = time.time()
        return c

@app.post("/api/chat")
def chat():
    j = request.json; run = re.sub(r"[^\w-]", "_", j.get("run") or ""); model = j["model"]
    if not allowed_model(model): return jsonify(error="pick a model from the model library"), 400
    if run and not os.path.exists(os.path.join(udir(RUNS), run + ".eslora")): return jsonify(error="no such adapter"), 404
    key = (model, run); CHAT = chat_slot()
    with CHAT["lock"]:
        p = CHAT["proc"]
        if not p or p.poll() is not None or CHAT["key"] != key:
            if p and p.poll() is None: p.kill()
            cmd = [os.path.join(BIN, "es_chat"), "-g", model, "-J", "-j", str(j.get("threads", 1)), "-n", str(j.get("max", 96)), "-t", "0.7", "-C", "512"]
            if run: cmd += ["--lora", os.path.join(udir(RUNS), run + ".eslora")]
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

os.makedirs(RUNS, exist_ok=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8765)), threaded=True)
