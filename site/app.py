#!/usr/bin/env python3
"""Expert Stream site: accounts, plans + Razorpay (TEST mode), Run (streamed MoE inference). Port 8780.
The notebook lives in the separate expert-stream-trainer app; both share es_common (DB, JWT, subscriptions, model library)."""
import hashlib, hmac, json, os, resource, signal, subprocess, sys, tempfile, threading, time, uuid
from flask import Flask, abort, jsonify, redirect, render_template, request
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
    if m["id"] not in _run_cache: _run_cache[m["id"]] = esmoe.StreamedMoE(m["path"], cache_mb=64)
    return jsonify(_run_cache[m["id"]].generate(d.get("prompt", "")[:2000] or " ", max_new=min(int(d.get("max_new", 64)), 256), temperature=float(d.get("temperature", 0.6))))

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
