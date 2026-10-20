#!/usr/bin/env python3
"""OmniRoute Gateway: an OpenAI-compatible front door to the ExpertCloud nodes.

    python3 gateway/omni_router.py [--host 0.0.0.0] [--port 8000] [--open-signup] [--similar-cache 0.9]

* POST /v1/chat/completions (JSON or stream=true Server-Sent Events), GET /v1/models; models: a model name served by a node, or
  auto (the biggest model online), expertstream-fast (the smallest warm one), expertstream-coder (a coder model, else auto).
* Virtual API keys sk-expert-..., free and unlimited (a per-key request rate limit guards against runaway clients; contributors
  are exempt). Keys are made by the admin (dashboard, POST /admin/keys, the MCP tool) or by anyone with --open-signup.
* Token Killer: every prompt is compacted (whitespace, repeated lines and messages; code untouched); "compress": "aggressive"
  also drops filler phrases. Saved tokens are counted per key.
* Response cache in SQLite: an identical request at temperature 0 is answered from the cache in milliseconds. "cache": "similar"
  (or --similar-cache T for every request) also reuses the answer of a prompt whose words overlap at least T (FTS5 + Jaccard);
  off by default, because a near-identical question can need a different answer.
* Auto-Combo router: every request goes to the online node with the best score (free capacity, time to first token, free RAM for
  the model and context, weights already warm, measured speed); if a node drops or fails before its first token, the next one
  takes the request at once. When every node is busy, requests wait, contributors first.
* Workers join at /cloud/ws with the join token (dashboard, or GET /admin/join).

The admin token and join token are made on first start, stored in expertcloud.db and written to ~/.expertcloud/ (mode 600).
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
import uuid

import secrets
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cloud.controller import ALIASES, Controller, Node  # noqa: E402
from gateway.compress import compress_messages, est_tokens  # noqa: E402
from gateway.store import Store, default_db, key_hash  # noqa: E402

OPTS = dict(open_signup=False, similar=0.0, rate_per_min=120, queue_timeout=120.0)
app = FastAPI(title="ExpertStream OmniRoute Gateway", version="1.0")
STORE = None
CTRL = None
ADMIN = None
JOIN = None
RATE = {}   # key hash -> recent request times


def err(status, message, kind="invalid_request_error"):
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind}})


def bearer(request):
    h = request.headers.get("authorization") or ""
    return h[7:].strip() if h.lower().startswith("bearer ") else (request.headers.get("x-api-key") or "").strip()


def is_admin(token):
    return bool(token) and secrets.compare_digest(token, ADMIN)


def auth(request):
    """(key hash or 'admin', None) or (None, error response)"""
    tok = bearer(request)
    if is_admin(tok): return "admin", None
    h = STORE.check_key(tok)
    if not h: return None, err(401, "missing or unknown API key (Authorization: Bearer sk-expert-...)", "authentication_error")
    if OPTS["rate_per_min"] > 0 and STORE.contribution(h) == 0:
        now = time.time()
        q = [t for t in RATE.get(h, []) if now - t < 60]
        if len(q) >= OPTS["rate_per_min"]: return None, err(429, "rate limit: %d requests per minute (contributing a node lifts it)" % OPTS["rate_per_min"], "rate_limit_error")
        q.append(now); RATE[h] = q
    return h, None


def admin_only(request):
    return None if is_admin(bearer(request) or request.headers.get("x-admin-token", "")) else err(401, "admin token required", "authentication_error")


# ---------------------------------------------------------------------------------------------------------------- OpenAI API
@app.get("/v1/models")
async def v1_models(request: Request):
    h, e = auth(request)
    if e: return e
    data = [dict(id=a, object="model", owned_by="expertcloud", alias_for=CTRL.resolve(a)) for a in sorted(ALIASES)] + CTRL.models()
    return {"object": "list", "data": data}


def cache_key(model, messages, params):
    blob = json.dumps(dict(model=model, messages=messages, p={k: params.get(k) for k in ("max_tokens", "top_p", "top_k", "seed", "stop")}), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def prompt_text(messages):
    return "\n".join("%s: %s" % (m.get("role"), m.get("content")) for m in messages if isinstance(m.get("content"), str))


def chunk(cid, model, created, delta=None, finish=None, usage=None, extra=None):
    c = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
         "choices": [] if delta is None and finish is None else [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}
    if usage is not None: c["usage"] = usage
    if extra: c["x_expertcloud"] = extra
    return "data: " + json.dumps(c) + "\n\n"


@app.post("/v1/chat/completions")
async def chat(request: Request):
    h, e = auth(request)
    if e: return e
    try:
        body = await request.json()
    except ValueError:
        return err(400, "body must be JSON")
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
        return err(400, "messages must be a non-empty list of {role, content}")
    model = str(body.get("model") or "auto")
    params = {k: body.get(k) for k in ("max_tokens", "temperature", "top_p", "top_k", "seed", "stop") if body.get(k) is not None}
    if "max_completion_tokens" in body and "max_tokens" not in params: params["max_tokens"] = body["max_completion_tokens"]
    mode = str(body.get("compress") or request.headers.get("x-compress") or "safe").lower()
    if mode != "off":
        messages, before, after = compress_messages(messages, aggressive=mode == "aggressive")
    else:
        before = after = sum(est_tokens(m.get("content") or "") for m in messages if isinstance(m.get("content"), str))
    saved = max(0, before - after)
    target = CTRL.resolve(model) or model
    temp = float(params.get("temperature", 1.0) if params.get("temperature") is not None else 1.0)
    cmode = body.get("cache")
    use_cache = cmode is not False and cmode != "off" and (temp == 0 or cmode in ("force", "similar"))
    ck = cache_key(target, messages, params)
    stream = bool(body.get("stream"))
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())
    acct = h if h != "admin" else None

    cached = None
    if use_cache:
        t0 = time.perf_counter()
        cached = STORE.cache_get(ck)
        if cached is None and (cmode == "similar" or OPTS["similar"] > 0):
            cached = STORE.cache_similar(target, prompt_text(messages), OPTS["similar"] if OPTS["similar"] > 0 else 0.9)
        if cached is not None:
            ms = (time.perf_counter() - t0) * 1000
            if acct: STORE.account(acct, saved=saved + cached.get("usage", {}).get("completion_tokens", 0), cache_hit=True)
            extra = dict(cache="hit", cache_ms=round(ms, 2), tokens_saved=saved, node=None)
            usage = dict(prompt_tokens=after, completion_tokens=0, total_tokens=after)
            if stream:
                async def replay():
                    yield chunk(cid, target, created, {"role": "assistant", "content": ""})
                    yield chunk(cid, target, created, {"content": cached["text"]})
                    yield chunk(cid, target, created, {}, "stop", usage, extra)
                    yield "data: [DONE]\n\n"
                return StreamingResponse(replay(), media_type="text/event-stream")
            return {"id": cid, "object": "chat.completion", "created": created, "model": target, "usage": usage, "x_expertcloud": extra,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": cached["text"]}, "finish_reason": "stop"}]}

    prio = 0 if (acct and STORE.contribution(acct) > 0) else 1
    gen = CTRL.run(model, messages, params, priority=prio, queue_timeout=OPTS["queue_timeout"])

    async def finish(text, info):
        usage = dict(info.get("usage") or {})
        pt, ct = int(usage.get("prompt_tokens") or after), int(usage.get("completion_tokens") or est_tokens(text))
        usage = dict(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
        if acct: STORE.account(acct, pt, ct, saved)
        if use_cache and text: STORE.cache_put(ck, info.get("model") or target, prompt_text(messages), dict(text=text, usage=usage))
        extra = dict(cache="miss" if use_cache else "off", tokens_saved=saved, node=info.get("node"), ttft_s=info.get("ttft"), tps=info.get("tps"))
        return usage, extra

    if stream:
        async def sse():
            text, sent_role = "", False
            try:
                async for kind, val in gen:
                    if kind == "delta":
                        if not sent_role: yield chunk(cid, target, created, {"role": "assistant", "content": ""}); sent_role = True
                        text += val
                        yield chunk(cid, target, created, {"content": val})
                    else:
                        usage, extra = await finish(text, val)
                        yield chunk(cid, val.get("model") or target, created, {}, "stop", usage, extra)
            except Exception as ex:   # no node, or a node failed mid-answer
                yield "data: " + json.dumps({"error": {"message": str(ex), "type": "server_error"}}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    text, info = "", {}
    try:
        async for kind, val in gen:
            if kind == "delta": text += val
            else: info = val
    except RuntimeError as ex:
        return err(503, str(ex), "server_error")
    usage, extra = await finish(text, info)
    return {"id": cid, "object": "chat.completion", "created": created, "model": info.get("model") or target, "usage": usage, "x_expertcloud": extra,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}


@app.get("/me")
async def me(request: Request):
    h, e = auth(request)
    if e: return e
    return STORE.key_stats(h) if h != "admin" else {"admin": True}


@app.post("/keys")
async def open_signup(request: Request):
    if not OPTS["open_signup"]: return err(403, "sign-up is closed: ask the admin for a key", "permission_error")
    try: body = await request.json()
    except ValueError: body = {}
    return {"key": STORE.new_key(str((body or {}).get("label") or "self-service")[:64])}


# ---------------------------------------------------------------------------------------------------------------- workers
@app.websocket("/cloud/ws")
async def cloud_ws(ws: WebSocket):
    await ws.accept()
    if not secrets.compare_digest(ws.query_params.get("token", ""), JOIN):
        await ws.send_text(json.dumps({"type": "rejected", "message": "wrong join token"})); await ws.close(code=4401); return
    try:
        hello = json.loads(await asyncio.wait_for(ws.receive_text(), 30))
    except Exception:
        await ws.close(); return
    if hello.get("type") != "hello": await ws.close(); return
    node = Node(ws, hello)
    owner = hello.get("owner_key")
    node.owner_hash = STORE.check_key(owner) if owner else None
    STORE.node_seen(node.id, node.name, node.owner_hash)
    CTRL.add(node)
    print("node joined: %s (%s, %d models)" % (node.name, node.id[:8], len(node.models)), flush=True)
    try:
        while True:
            raw = await ws.receive_text()
            try: msg = json.loads(raw)
            except ValueError: continue
            CTRL.on_message(node, msg)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        CTRL.drop(node)
        print("node left: %s" % node.name, flush=True)


# ---------------------------------------------------------------------------------------------------------------- admin
def join_command(request):
    host = request.headers.get("host") or "localhost:8000"
    scheme = "wss" if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https" else "ws"
    return "python3 cloud/worker_agent.py --master %s://%s --token %s" % (scheme, host, JOIN)


@app.post("/admin/keys")
async def admin_new_key(request: Request):
    e = admin_only(request)
    if e: return e
    try: body = await request.json()
    except ValueError: body = {}
    return {"key": STORE.new_key(str((body or {}).get("label") or "")[:64])}


@app.get("/admin/keys")
async def admin_keys(request: Request):
    return admin_only(request) or {"keys": STORE.list_keys()}


@app.delete("/admin/keys/{prefix}")
async def admin_revoke(prefix: str, request: Request):
    return admin_only(request) or {"revoked": STORE.revoke(prefix)}


@app.get("/admin/nodes")
async def admin_nodes(request: Request):
    return admin_only(request) or {"nodes": [n.view() for n in CTRL.online()]}


@app.post("/admin/deploy")
async def admin_deploy(request: Request):
    e = admin_only(request)
    if e: return e
    b = await request.json()
    return await CTRL.deploy(str(b.get("node_id")), str(b.get("repo")), str(b.get("file")))


@app.get("/admin/join")
async def admin_join(request: Request):
    return admin_only(request) or {"command": join_command(request), "token": JOIN}


@app.get("/admin/live")
async def admin_live(request: Request):
    e = admin_only(request)
    if e: return e
    return {"nodes": [n.view() for n in CTRL.online()], "models": CTRL.models(), "speed": CTRL.live_tps(), "totals": STORE.totals(),
            "cache": STORE.cache_stats(), "join": join_command(request), "open_signup": OPTS["open_signup"]}


@app.get("/")
@app.get("/dashboard")
async def dashboard():
    return FileResponse(os.path.join(ROOT, "dashboard", "index.html"))


@app.get("/health")
async def health():
    return {"ok": True, "nodes": len(CTRL.online()), "models": [m["id"] for m in CTRL.models()]}


def init(db_path):
    global STORE, CTRL, ADMIN, JOIN
    STORE = Store(db_path)
    CTRL = Controller(STORE)
    ADMIN = STORE.setting("admin_token", lambda: "adm-" + secrets.token_urlsafe(24))
    JOIN = STORE.setting("join_token", lambda: "join-" + secrets.token_urlsafe(18))
    d = os.path.expanduser("~/.expertcloud"); os.makedirs(d, exist_ok=True)
    for name, val in (("admin_token", ADMIN), ("join_token", JOIN)):
        p = os.path.join(d, name)
        with open(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f: f.write(val)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept nodes and users from other machines")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--db", default=default_db())
    ap.add_argument("--open-signup", action="store_true", help="anyone may make a key with POST /keys")
    ap.add_argument("--similar-cache", type=float, default=0.0, help="reuse answers of prompts with at least this word overlap (0 = exact only)")
    ap.add_argument("--rate", type=int, default=120, help="requests per minute per key (0 = no limit; contributors are never limited)")
    ap.add_argument("--queue-timeout", type=float, default=120.0, help="seconds a request may wait for a free node")
    a = ap.parse_args()
    OPTS.update(open_signup=a.open_signup, similar=a.similar_cache, rate_per_min=a.rate, queue_timeout=a.queue_timeout)
    init(a.db)
    print("OmniRoute gateway on http://%s:%d  (dashboard: /dashboard)" % (a.host, a.port))
    print("  admin token: %s" % ADMIN)
    print("  join a node: python3 cloud/worker_agent.py --master ws://%s:%d --token %s" % (a.host if a.host != "0.0.0.0" else "THIS_HOST", a.port, JOIN), flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
