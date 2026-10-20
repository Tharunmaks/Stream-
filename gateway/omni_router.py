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
import socket
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cloud.controller import ALIASES, Controller, Node  # noqa: E402
from gateway.compress import compress_messages, est_tokens  # noqa: E402
from gateway.store import Store, default_db, key_hash  # noqa: E402

OPTS = dict(open_signup=False, similar=0.0, rate_per_min=120, queue_timeout=120.0, host="127.0.0.1", port=8000)
STARTED = time.time()
app = FastAPI(title="ExpertStream OmniRoute Gateway", version="1.0")
STORE = None
CTRL = None
ADMIN = None
JOIN = None
RATE = {}   # key hash -> recent request times
ADMIN_USE = dict(requests=0, completion_tokens=0, tokens_saved=0, cache_hits=0)   # requests made with the admin token (dashboard Playground)


def err(status, message, kind="invalid_request_error"):
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind}})


QUOTES = " \t\r\n\"'`<>\u2018\u2019\u201c\u201d\u00ab\u00bb"


def clean_token(t):
    """A token as people paste it: surrounding spaces, quotes (straight or curly), <>, a 'Bearer ' prefix, a trailing newline."""
    t = t or ""
    try: t = t.encode("latin-1").decode("utf-8")   # HTTP headers arrive as latin-1: recover pasted curly quotes
    except (UnicodeEncodeError, UnicodeDecodeError): pass
    t = t.strip(QUOTES)
    while t[:7].lower() == "bearer ": t = t[7:].strip(QUOTES)
    return t


def same_secret(a, b):
    """Constant-time comparison that also works for non-ASCII input (str compare_digest raises TypeError on it)."""
    return bool(a) and bool(b) and secrets.compare_digest(a.encode("utf-8", "replace"), b.encode("utf-8", "replace"))


def bearer(request):
    """The token of a request: Authorization (with or without 'Bearer '), x-api-key or x-admin-token."""
    for name in ("authorization", "x-api-key", "x-admin-token"):
        t = clean_token(request.headers.get(name))
        if t: return t
    return ""


def is_admin(token):
    return same_secret(clean_token(token), ADMIN)


def token_hint(tok, want):
    """Why a token was refused, in words that say what to do."""
    where = "it is in ~/.expertcloud/admin_token, printed when the gateway starts"
    if not tok: return "no token sent: use the header Authorization: Bearer <token> (%s)" % ("an sk-expert- key" if want == "key" else "the admin token; " + where)
    if same_secret(tok, JOIN): return "that is the join token (for worker nodes), not the admin token; the admin token starts with adm- (%s)" % where
    if want == "admin" and tok.startswith("sk-expert-"): return "that is a user API key; the admin pages need the admin token (adm-...; %s)" % where
    if tok.startswith("adm-"): return "wrong admin token: it changes when the gateway is started with --rotate-tokens or a different --db (%s)" % where
    if want == "admin": return "not the admin token (%s)" % where
    return "unknown or revoked API key (Authorization: Bearer sk-expert-...)"


def auth(request):
    """(key hash or 'admin', None) or (None, error response)"""
    tok = bearer(request)
    if is_admin(tok): return "admin", None
    h = STORE.check_key(tok) if tok else None
    if not h: return None, err(401, token_hint(tok, "key"), "authentication_error")
    if OPTS["rate_per_min"] > 0 and STORE.contribution(h) == 0:
        now = time.time()
        q = [t for t in RATE.get(h, []) if now - t < 60]
        if len(q) >= OPTS["rate_per_min"]: return None, err(429, "rate limit: %d requests per minute (contributing a node lifts it)" % OPTS["rate_per_min"], "rate_limit_error")
        q.append(now); RATE[h] = q
    return h, None


def admin_only(request):
    tok = bearer(request)
    return None if is_admin(tok) else err(401, token_hint(tok, "admin"), "authentication_error")


# ---------------------------------------------------------------------------------------------------------------- OpenAI API
@app.get("/v1/models")
async def v1_models(request: Request):
    h, e = auth(request)
    if e: return e
    data = [dict(id=a, object="model", owned_by="expertcloud", alias_for=CTRL.resolve(a)) for a in sorted(ALIASES)] + CTRL.models()
    return {"object": "list", "data": data}


def admin_use(completion_tokens=0, saved=0, cache_hits=0):
    ADMIN_USE["requests"] += 1; ADMIN_USE["completion_tokens"] += completion_tokens
    ADMIN_USE["tokens_saved"] += saved; ADMIN_USE["cache_hits"] += cache_hits


def totals():
    t = STORE.totals()
    for k, v in ADMIN_USE.items(): t[k] += v
    return t


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
            else: admin_use(saved=saved + cached.get("usage", {}).get("completion_tokens", 0), cache_hits=1)
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
        else: admin_use(completion_tokens=ct, saved=saved)
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
    tok = clean_token(ws.query_params.get("token"))
    if not (same_secret(tok, JOIN) or is_admin(tok)):
        await ws.send_text(json.dumps({"type": "rejected", "message": "wrong join token (it starts with join-: GET /admin/join or the dashboard's Connect page)"}))
        await ws.close(code=4401); return
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
def lan_ips():
    """This machine's addresses other devices on the network can reach (no packet is sent: UDP connect only picks a route)."""
    ips = []
    for probe in ("10.255.255.255", "192.168.255.255", "8.8.8.8"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect((probe, 1)); ip = s.getsockname()[0]; s.close()
            if not ip.startswith("127.") and ip not in ips: ips.append(ip)
        except OSError:
            pass
    return ips


def join_command(request, host=None):
    host = host or request.headers.get("host") or "localhost:%d" % OPTS["port"]
    scheme = "wss" if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https" else "ws"
    return "python3 cloud/worker_agent.py --master %s://%s --token %s" % (scheme, host, JOIN)


def connect_info(request):
    """Join commands and API base URLs for this device and for other devices on the network."""
    lan = [dict(ip=ip, join=join_command(request, "%s:%d" % (ip, OPTS["port"])), api="http://%s:%d/v1" % (ip, OPTS["port"])) for ip in lan_ips()]
    return dict(join=join_command(request), local_join="python3 cloud/worker_agent.py --master ws://127.0.0.1:%d --token %s" % (OPTS["port"], JOIN),
                lan=lan, lan_reachable=OPTS["host"] not in ("127.0.0.1", "localhost", "::1"), bind_host=OPTS["host"], port=OPTS["port"])


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
    return admin_only(request) or dict(connect_info(request), command=join_command(request), token=JOIN)


@app.get("/admin/live")
async def admin_live(request: Request):
    e = admin_only(request)
    if e: return e
    return {"nodes": [n.view() for n in CTRL.online()], "models": CTRL.models(), "speed": CTRL.live_tps(), "totals": totals(),
            "cache": STORE.cache_stats(), "join": join_command(request), "open_signup": OPTS["open_signup"], "connect": connect_info(request),
            "options": dict(rate_per_min=OPTS["rate_per_min"], similar_cache=OPTS["similar"], queue_timeout=OPTS["queue_timeout"]),
            "uptime_s": round(time.time() - STARTED)}


@app.get("/auth/check")
async def auth_check(request: Request):
    """For the dashboard's sign-in: which kind of token this is, or a hint why it is refused."""
    tok = bearer(request)
    if is_admin(tok): return {"ok": True, "role": "admin"}
    if tok and STORE.check_key(tok): return {"ok": True, "role": "key", "hint": token_hint(tok, "admin")}
    return err(401, token_hint(tok, "admin"), "authentication_error")


@app.get("/")
@app.get("/dashboard")
async def dashboard():
    return FileResponse(os.path.join(ROOT, "dashboard", "index.html"), headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})


@app.get("/health")
async def health():
    return {"ok": True, "nodes": len(CTRL.online()), "models": [m["id"] for m in CTRL.models()]}


def init(db_path, rotate=False):
    global STORE, CTRL, ADMIN, JOIN
    STORE = Store(db_path)
    CTRL = Controller(STORE)
    if rotate:
        STORE.set_setting("admin_token", "adm-" + secrets.token_urlsafe(24))
        STORE.set_setting("join_token", "join-" + secrets.token_urlsafe(18))
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
    ap.add_argument("--rotate-tokens", action="store_true", help="make a new admin token and join token (old ones stop working; nodes must rejoin)")
    a = ap.parse_args()
    OPTS.update(open_signup=a.open_signup, similar=a.similar_cache, rate_per_min=a.rate, queue_timeout=a.queue_timeout, host=a.host, port=a.port)
    init(a.db, a.rotate_tokens)
    local = "http://127.0.0.1:%d" % a.port
    print("OmniRoute gateway on http://%s:%d%s" % (a.host, a.port, "  (new tokens made)" if a.rotate_tokens else ""))
    print("  dashboard (signs in by itself): %s/dashboard#token=%s" % (local, ADMIN))
    print("  admin token: %s   (also in ~/.expertcloud/admin_token)" % ADMIN)
    print("  join a node here: python3 cloud/worker_agent.py --master ws://127.0.0.1:%d --token %s" % (a.port, JOIN))
    if OPTS["host"] not in ("127.0.0.1", "localhost", "::1"):
        for ip in lan_ips(): print("  from other devices: http://%s:%d/dashboard   join: --master ws://%s:%d" % (ip, a.port, ip, a.port))
    else:
        print("  only this device can connect; start with --host 0.0.0.0 to let other devices join")
    sys.stdout.flush()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
