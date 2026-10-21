#!/usr/bin/env python3
"""Expert Stream site: accounts, plans + Razorpay (TEST mode), Run (streamed MoE inference). Port 8780.
The notebook lives in the separate expert-stream-trainer app; both share es_common (DB, JWT, subscriptions, model library)."""
import hashlib, hmac, json, os, resource, signal, subprocess, sys, tempfile, threading, time, uuid
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, stream_with_context
import socket, urllib.request
import es_common as C
from es_common import esmoe

app = Flask(__name__)
KEY_ID = os.environ.get("RAZORPAY_KEY_ID", "rzp_test_PLACEHOLDER")
KEY_SECRET = os.environ.get("RAZORPAY_KEY_SECRET", "placeholder_secret")
LIVE_KEYS = KEY_ID.startswith("rzp_test_") and KEY_ID != "rzp_test_PLACEHOLDER"
PLANS = {
    "es-starter": dict(product="Expert Stream", name="Starter", inr=0, items=["Run MoE models streamed from disk", "MCP server, 1 API key", "Notebook: 1 job, 30 min CPU"]),
    "one-starter": dict(product="Expert One", name="Starter", inr=2000, items=["Expert Stream, Cloud, Build, Craft", "Notebook: 2 jobs, 4 h CPU each", "Models up to 10B"]),
    "one-mid": dict(product="Expert One", name="Mid", inr=5000, items=["Everything in Starter", "Notebook: 4 jobs, 12 h CPU each", "Models up to 30B"]),
    "one-top": dict(product="Expert One", name="Top", inr=10000, items=["Everything in Mid", "Notebook: 8 jobs, 24 h CPU each", "Models up to 70B, priority queue"]),
}
ORDERS = {}
TRAINER_URL = os.environ.get("TRAINER_URL", "http://127.0.0.1:8781")
_run_cache = {}
app.jinja_env.globals.update(trainer_url=TRAINER_URL)

@app.context_processor
def _ctx(): u = C.user_from_request(request); return dict(user=u, sub=C.subscription(u["id"]) if u else None)

@app.get("/")
def home(): return render_template("plans.html", plans=PLANS, key_id=KEY_ID)

@app.route("/signup", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def auth():
    mode = request.path[1:]; err = None
    if request.method == "POST":
        try:
            f = request.form; uid = (C.signup if mode == "signup" else C.login)(f["email"], f["password"])
            r = redirect(request.args.get("next") or "/"); r.set_cookie(C.COOKIE, C.make_token(uid), httponly=True, samesite="Lax",
                                                                     domain=os.environ.get("ES_COOKIE_DOMAIN")); return r
        except ValueError as e: err = str(e)
    return render_template("auth.html", mode=mode, err=err)

@app.get("/logout")
def logout(): r = redirect("/"); r.delete_cookie(C.COOKIE, domain=os.environ.get("ES_COOKIE_DOMAIN")); return r

@app.get("/run")
def run_page(): return render_template("run.html", models=C.list_models())

@app.post("/api/run_moe")
def run_moe():
    d = request.json or {}; m = C.get_model(d.get("model", "")) or abort(404)
    if m["id"] not in _run_cache: _run_cache[m["id"]] = esmoe.load_run(m["path"], cache_mb=int(os.environ.get("ES_RUN_CACHE_MB", 64)))
    if d.get("stream"):   # SSE: one event per token, then the stats, so time to first token is real
        it = _run_cache[m["id"]].stream(d.get("prompt", "")[:2000] or " ", max_new=min(int(d.get("max_new", 64)), 256), temperature=float(d.get("temperature", 0.6)),
                                         **({k: d[k] for k in ("prefetch", "sticky") if k in d} if isinstance(_run_cache[m["id"]], esmoe.SparseRun) else {}))
        def gen():
            for kind, v in it:
                yield "data: " + json.dumps({"token": bytes([v]).decode("latin-1")} if kind == "token" else {"done": v}) + "\n\n"
        return Response(stream_with_context(gen()), mimetype="text/event-stream", headers={"Cache-Control": "no-cache"})
    return jsonify(_run_cache[m["id"]].generate(d.get("prompt", "")[:2000] or " ", max_new=min(int(d.get("max_new", 64)), 256), temperature=float(d.get("temperature", 0.6))))

# ---- GGUF chat backend: prebuilt es_serve per (model, threads), proxied as SSE so TTFT is real ----
_gguf_srv = {}; _gguf_lock = threading.Lock()
def _gguf_server(m, threads):
    key = (m["id"], threads)
    with _gguf_lock:
        s = _gguf_srv.get(key)
        if s and s["p"].poll() is None: return s
        so = socket.socket(); so.bind(("127.0.0.1", 0)); port = so.getsockname()[1]; so.close()
        nice = int(os.environ.get("ES_CHAT_NICE", "10"))
        env = dict(os.environ, ES_PREFIX_CACHE=os.environ.get("ES_PREFIX_CACHE", "0"))
        mt = m.get("meta") or {}; mt = json.loads(mt) if isinstance(mt, str) else mt
        llama = mt.get("backend") == "llama.cpp"
        if llama:   # llama.cpp backend: persistent llama-server, model stays resident (mmap + mlock-free), max 4 threads
            lt = max(2, min(4, os.cpu_count() - int(os.getloadavg()[0]))) if threads == "auto" else min(int(threads), 4)
            cmd = [os.environ.get("ES_LLAMA_SERVER", "/workspace/llamabin/llama-b11583/llama-server"), "-m", m["path"], "--host", "127.0.0.1",
                   "--port", str(port), "-t", str(lt), "-tb", str(lt), "-c", "2048", "-np", "1", "--no-webui"]
        else:
            cmd = [C.es_bin("es_serve"), "-g", m["path"], "-p", str(port), "-j", str(threads), "-c", "2048"]
        p = subprocess.Popen(["nice", "-n", str(nice)] + cmd,
                             env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(600):
            try: urllib.request.urlopen(f"http://127.0.0.1:{port}/" + ("health" if llama else "v1/models"), timeout=1); break
            except Exception:
                if p.poll() is not None: raise RuntimeError("es_serve exited")
                time.sleep(0.1)
        s = _gguf_srv[key] = dict(p=p, port=port); return s

@app.post("/api/chat")
def chat_api():
    d = request.json or {}; m = C.get_model(d.get("model", "")) or abort(404)
    if m.get("source") != "gguf": abort(400, "not a GGUF model; use /api/run_moe")
    tv = str(d.get("threads", os.environ.get("ES_CHAT_THREADS", 2)))
    threads = "auto" if tv == "auto" else max(1, min(int(tv), 8))   # auto: es_serve picks free cores by load, cap 4 (es-run-speed engine)
    msgs = d.get("messages") or [{"role": "user", "content": d.get("prompt", "")[:8000] or " "}]
    s = _gguf_server(m, threads)
    body = json.dumps(dict(model=m["name"], messages=msgs, stream=True, stream_options={"include_usage": True},
                           max_tokens=min(int(d.get("max_new", 128)), 1024), temperature=float(d.get("temperature", 0.6)))).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{s['port']}/v1/chat/completions", body, {"Content-Type": "application/json"})
    def gen():
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                if not line.strip(): continue
                t = line.decode()
                if t.startswith("data:") and '"usage"' in t and "x_expertstream" not in t:
                    try:   # llama.cpp: map its timings to the usage extension Run reads
                        j = json.loads(t[5:]); tm = j.get("timings") or {}
                        if j.get("usage"):
                            j["x_expertstream"] = dict(tokens_per_s=round(tm.get("predicted_per_second", 0), 1), ttft_ms=round(tm.get("prompt_ms", 0), 1), backend="llama.cpp")
                            t = "data: " + json.dumps(j) + "\n"
                    except Exception: pass
                yield t + "\n"
    return Response(stream_with_context(gen()), mimetype="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/create")
def create_page(): return render_template("create.html")

@app.post("/api/plan")
def plan_api():
    d = request.json or {}
    try: return jsonify(esmoe.plan(float(d["total_params"]), int(d.get("experts") or 0) or None, int(d.get("topk") or 2)))
    except (ValueError, KeyError) as e: return jsonify(error=str(e)), 400

@app.post("/api/create")
def create_api():
    u = C.user_from_request(request)
    if not u: return jsonify(error="login"), 401
    d = request.json or {}; lim = C.PLAN_LIMITS[C.subscription(u["id"])]
    name = "".join(ch for ch in (d.get("name") or "model") if ch.isalnum() or ch in "-_")[:40] or "model"
    try: cfg = esmoe.plan(float(d["total_params"]), int(d.get("experts") or 0) or None, int(d.get("topk") or 2))
    except ValueError as e: return jsonify(error=str(e)), 400
    if cfg["total_params"] > 7e10: return jsonify(error="Create goes up to 70B total params"), 400
    steps = int(d.get("train_steps") or 0)
    if steps and cfg["total_params"] > min(lim["max_params"], 2e7): return jsonify(error="training on Create is limited to your plan size (max 20M here); use the Trainer for longer jobs"), 403
    path = os.path.join(C.MODELS_DIR, f"{u['id'][:8]}-{name}-{uuid.uuid4().hex[:6]}")
    if steps:
        text = open(os.path.join(os.path.dirname(__file__), "data/sample.txt"), errors="replace").read()
        c = esmoe.train_torch(path, cfg, text, steps=min(steps, lim["max_steps"], 1000), threads=2, log=lambda s: None)
    else: c = esmoe.create(path, cfg)
    mid = C.register_model(u["id"], name, path, c, "create")
    return jsonify(id=mid, name=name, **{k: v for k, v in c.items() if k in ("total_params", "active_params", "layers", "experts", "dim", "ffn", "topk", "materialized", "train")})

@app.get("/mcp")
def mcp_page(): return render_template("mcp.html")

@app.post("/api/order")
def order():
    u = C.user_from_request(request)
    if not u: return jsonify(error="login"), 401
    pid = (request.json or {}).get("plan"); p = PLANS.get(pid)
    if not p or p["inr"] == 0: abort(400)
    if LIVE_KEYS:
        import razorpay
        o = razorpay.Client(auth=(KEY_ID, KEY_SECRET)).order.create(dict(amount=p["inr"] * 100, currency="INR", receipt=uuid.uuid4().hex[:12]))
    else:  # no real test keys supplied: local stub order so the flow can be exercised
        o = dict(id="order_stub_" + uuid.uuid4().hex[:14], amount=p["inr"] * 100, currency="INR")
    ORDERS[o["id"]] = (pid, u["id"])
    return jsonify(order_id=o["id"], amount=o["amount"], currency="INR", key_id=KEY_ID, stub=not LIVE_KEYS, plan=p["product"] + " " + p["name"])

@app.post("/api/verify")
def verify():
    d = request.json or {}
    oid, pay, sig = d.get("razorpay_order_id", ""), d.get("razorpay_payment_id", ""), d.get("razorpay_signature", "")
    if oid not in ORDERS: return jsonify(ok=False, error="unknown order"), 400
    exp = hmac.new(KEY_SECRET.encode(), f"{oid}|{pay}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(exp, sig): return jsonify(ok=False, error="bad signature"), 400
    pid, uid = ORDERS.pop(oid); C.set_subscription(uid, pid, oid, pay)
    return jsonify(ok=True, plan=pid)

if __name__ == "__main__": C.db(); app.run("0.0.0.0", int(os.environ.get("PORT", 8780)), threaded=True)
