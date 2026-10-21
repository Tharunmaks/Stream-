"""Minimal subprocess job manager. Jobs live under ~/.sparseforge/jobs/<id>/."""
import json, os, signal, subprocess, sys, uuid
ROOT = os.environ.get("SPARSEFORGE_HOME", os.path.expanduser("~/.sparseforge/jobs"))

def _dir(job_id): return os.path.join(ROOT, job_id)

def start_flash_job(model_path, corpus, steps=200, batch=256, lr=2e-3, expert_lr=2e-3, threads=2, eval_corpus=None):
    """Background Flash (NumPy) training job with capped BLAS threads."""
    job_id = uuid.uuid4().hex[:10]; d = _dir(job_id); os.makedirs(d)
    st = os.path.join(d, "status.json")
    argv = [sys.executable, "-m", "sparseforge.flash", "train", model_path, corpus, "--steps", str(steps), "--batch", str(batch),
            "--lr", str(lr), "--expert-lr", str(expert_lr), "--status-file", st, "--stop-file", os.path.join(d, "STOP")]
    if eval_corpus: argv += ["--eval-corpus", eval_corpus]
    env = {**os.environ, **{k: str(threads) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
    json.dump({"engine": "flash", "model_path": model_path, "argv": argv}, open(os.path.join(d, "config.json"), "w"))
    log = open(os.path.join(d, "log.txt"), "w")
    p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    open(os.path.join(d, "pid"), "w").write(str(p.pid))
    return {"job_id": job_id, "pid": p.pid, "dir": d}

def start_job(config):
    job_id = uuid.uuid4().hex[:10]; d = _dir(job_id); os.makedirs(d)
    config = {**config, "status_file": os.path.join(d, "status.json")}
    config.setdefault("save", os.path.join(d, "model.pt"))
    json.dump(config, open(os.path.join(d, "config.json"), "w"))
    log = open(os.path.join(d, "log.txt"), "w")
    p = subprocess.Popen([sys.executable, "-m", "sparseforge.train", "--config", os.path.join(d, "config.json"),
                          "--out", os.path.join(d, "result.json")], stdout=log, stderr=subprocess.STDOUT,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))), start_new_session=True)
    open(os.path.join(d, "pid"), "w").write(str(p.pid))
    return {"job_id": job_id, "pid": p.pid, "dir": d}

def _alive(pid):
    try: os.kill(pid, 0); return True
    except OSError: return False

def get_status(job_id):
    d = _dir(job_id)
    if not os.path.isdir(d): return {"error": "unknown job"}
    st = {"state": "starting"}
    if os.path.exists(os.path.join(d, "status.json")):
        try: st = json.load(open(os.path.join(d, "status.json")))
        except ValueError: pass
    pid = int(open(os.path.join(d, "pid")).read())
    if st.get("state") in ("running", "starting") and not _alive(pid):
        st["state"] = "stopped_or_crashed"
    return {"job_id": job_id, **st}

def stop_job(job_id):
    open(os.path.join(_dir(job_id), "STOP"), "w").close()  # graceful for flash jobs (saves weights)
    pid = int(open(os.path.join(_dir(job_id), "pid")).read())
    import time; time.sleep(2)
    if not _alive(pid): return {"stopped": True, "graceful": True}
    try: os.killpg(pid, signal.SIGTERM); return {"stopped": True}
    except OSError as e: return {"stopped": False, "error": str(e)}

def tail_logs(job_id, n=50):
    p = os.path.join(_dir(job_id), "log.txt")
    return "".join(open(p).readlines()[-n:]) if os.path.exists(p) else ""

def list_jobs():
    return sorted(os.listdir(ROOT)) if os.path.isdir(ROOT) else []
