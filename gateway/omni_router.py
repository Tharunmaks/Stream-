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
import re
import sys
import time
import uuid

import secrets
import socket
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cloud.controller import ALIASES, Controller, Node  # noqa: E402
from gateway.compress import compress_messages, est_tokens  # noqa: E402
from gateway.store import Store, default_db, key_hash  # noqa: E402
from gateway import billing, plans as plans_mod  # noqa: E402
from gateway.database import DbError, DocStore  # noqa: E402

OPTS = dict(open_signup=False, similar=0.0, rate_per_min=120, queue_timeout=120.0, host="127.0.0.1", port=8000)
STARTED = time.time()
app = FastAPI(title="ExpertStream OmniRoute Gateway", version="1.0")
# Apps on other sites (the ExpertStream app, a user's web app) call the API from the browser. Auth is a bearer token, never a
# cookie, so allowing any origin does not let one site act as a visitor of another.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["Authorization", "Content-Type", "x-api-key"])
PLANS = plans_mod.PLANS
DOCS = None
sys.path.insert(0, os.path.join(ROOT, "es-common"))
_ESC = []


def esc():
    """es_common (ExpertStream accounts, Expert One subscriptions), shared with the ExpertStream site; None if unavailable."""
    if not _ESC:
        try:
            import es_common
            es_common.db()
            _ESC.append(es_common)
        except Exception as ex:   # noqa: BLE001 - the Cloud works alone without the site's library
            print("ExpertStream accounts off: %s" % ex, flush=True); _ESC.append(None)
    return _ESC[0]


def site_url():
    return (STORE.setting("expertstream_url") or os.environ.get("EXPERTSTREAM_URL") or "http://127.0.0.1:8780").rstrip("/")
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


def expert_one(uid):
    """The Expert One plan id of a user (from the ExpertStream site), or None."""
    e, es_uid = esc(), STORE.es_uid(uid)
    if not e or not es_uid: return None
    try: return e.subscription(es_uid)
    except Exception: return None   # noqa: BLE001


def user_plan(uid, product="cloud"):
    if product == "expertone":
        _, p = plans_mod.find(PLANS, expert_one(uid) or "es-starter")
        return dict(p or plans_mod.free_plan(PLANS, "expertone"), expires=None, source="ExpertStream")
    best = dict(plans_mod.free_plan(PLANS, product), expires=None, source=None)
    sub = STORE.subscription(uid, product)
    if sub:
        _, p = plans_mod.find(PLANS, sub[0])
        if p: best = dict(p, expires=sub[1], source=None)
    one = plans_mod.UNLOCKS.get(expert_one(uid) or "") if product == "cloud" else None
    if one:   # Expert One includes Expert Cloud: the better of the two counts
        _, p = plans_mod.find(PLANS, one)
        if p and p["price_inr"] > best["price_inr"]: best = dict(p, expires=None, source="Expert One")
    return best


def auth(request, count=False):
    """(key hash or 'admin', None) or (None, error response). Sets request.state.uid and .plan for user accounts and their
    keys. count=True also checks the plan's daily limit (chat requests; they are counted once answered)."""
    request.state.uid, request.state.plan = None, None
    tok = bearer(request)
    if is_admin(tok): return "admin", None
    su = STORE.session_user(tok) if tok.startswith("ses-") else None
    h = su[2] if su else (STORE.check_key(tok) if tok else None)
    if not h: return None, err(401, "signed out: sign in again" if tok.startswith("ses-") else token_hint(tok, "key"), "authentication_error")
    uid = su[0] if su else STORE.key_owner(h)
    rate = OPTS["rate_per_min"]
    if uid:
        plan = user_plan(uid)
        request.state.uid, request.state.plan = uid, plan
        # the plan's per-minute limit is for AI requests; database and other API calls share a roomier one
        rate = plan["limits"].get("requests_per_min", rate) if count else max(120, 2 * plan["limits"].get("requests_per_min", 0))
        if count and STORE.requests_today(uid) >= plan["limits"].get("requests_per_day", 1 << 30):
            return None, err(429, "you used today's %d requests on the %s plan: upgrade for more, or wait until midnight UTC"
                             % (plan["limits"]["requests_per_day"], plan["name"]), "rate_limit_error")
    contributed = STORE.user_usage(uid)["contributed_tokens"] if uid else STORE.contribution(h)
    if rate > 0 and contributed == 0:
        now = time.time()
        bucket = (h, count)
        q = [t for t in RATE.get(bucket, []) if now - t < 60]
        if len(q) >= rate: return None, err(429, "rate limit: %d %srequests per minute on your plan" % (rate, "AI " if count else ""), "rate_limit_error")
        q.append(now); RATE[bucket] = q
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
    h, e = auth(request, count=True)
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
    plan = request.state.plan
    if plan:   # the plan caps answer length
        cap = int(plan["limits"].get("max_tokens") or 0)
        if cap: params["max_tokens"] = min(int(params.get("max_tokens") or cap), cap)
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
            if request.state.uid: STORE.count_request(request.state.uid)
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

    # queue order: 0 first (admin, Max plan, contributors), then Pro, then Free
    prio = 0 if not acct or STORE.contribution(acct) > 0 else int(plan["limits"].get("priority", 1)) if plan else 1
    gen = CTRL.run(model, messages, params, priority=prio, queue_timeout=OPTS["queue_timeout"])

    async def finish(text, info):
        usage = dict(info.get("usage") or {})
        pt, ct = int(usage.get("prompt_tokens") or after), int(usage.get("completion_tokens") or est_tokens(text))
        usage = dict(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
        if acct: STORE.account(acct, pt, ct, saved)
        if request.state.uid: STORE.count_request(request.state.uid, ct)
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



# ---------------------------------------------------------------------------------------------------------------- public cloud
USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
ATTEMPTS = {}   # (ip, kind) -> recent times, against password guessing and sign-up floods


def setting_on(name, default):
    v = STORE.setting(name)
    return default if v is None else v == "1"


def too_many(request, kind, limit, window):
    ip = request.client.host if request.client else "?"
    now = time.time()
    q = [t for t in ATTEMPTS.get((ip, kind), []) if now - t < window]
    ATTEMPTS[(ip, kind)] = q
    return len(q) >= limit, q


def user_of(request):
    """(user id, username, console key hash), or None."""
    tok = bearer(request)
    return STORE.session_user(tok) if tok.startswith("ses-") else None


async def body_of(request):
    try:
        b = await request.json()
        return b if isinstance(b, dict) else {}
    except ValueError:
        return {}


COOKIE = "ec_session"   # the Cloud's sign-in cookie (the Training page needs it); the site's is es_common.COOKIE (es_session)


def signed_in(resp, uid, token, es_uid=None):
    """Sets the Cloud cookie, and the ExpertStream cookie so the site sees the same person signed in."""
    resp.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=30 * 86400, path="/")
    e = esc()
    if e and es_uid:
        resp.set_cookie(e.COOKIE, e.make_token(es_uid), httponly=True, samesite="lax", max_age=7 * 86400, path="/",
                        domain=os.environ.get("ES_COOKIE_DOMAIN") or None)
    return resp


def session_reply(uid, es_uid=None):
    row = STORE.db.execute("SELECT username FROM users WHERE id=?", (uid,)).fetchone()
    tok = STORE.new_session(uid)
    return signed_in(JSONResponse({"token": tok, "username": row[0]}), uid, tok, es_uid)


@app.post("/auth/signup")
async def signup(request: Request):
    """One account for ExpertStream and ExpertStream Cloud: {"email", "password"} makes it on both. ({"username"} still works for a
    Cloud-only account when the ExpertStream library is not installed.)"""
    if not setting_on("signup", True): return err(403, "sign-up is closed on this cloud", "permission_error")
    full, q = too_many(request, "signup", 5, 3600)
    if full: return err(429, "too many new accounts from this address; try again in an hour", "rate_limit_error")
    b = await body_of(request)
    email, pw = str(b.get("email") or b.get("username") or "").strip(), str(b.get("password") or "")
    if len(pw) < 8: return err(400, "password: at least 8 characters")
    e = esc()
    if "@" in email and e:
        try: es_uid = e.signup(email, pw)
        except ValueError as ex: return err(409 if "registered" in str(ex) else 400, str(ex))
        uid = STORE.linked_user(es_uid, email.lower())
        q.append(time.time())
        return session_reply(uid, es_uid)
    if e: return err(400, "enter your email address: one account works on ExpertStream and ExpertStream Cloud")
    if not USERNAME.match(email): return err(400, "username: 3 to 32 letters, digits, dots, dashes or underscores")
    uid = STORE.create_user(email, pw)
    if uid is None: return err(409, "that username is taken")
    q.append(time.time())
    return session_reply(uid)


@app.post("/auth/login")
async def login(request: Request):
    full, q = too_many(request, "login", 10, 600)
    if full: return err(429, "too many wrong passwords; wait 10 minutes", "rate_limit_error")
    b = await body_of(request)
    name, pw = str(b.get("email") or b.get("username") or "").strip(), str(b.get("password") or "")
    e = esc()
    if "@" in name and e:   # an ExpertStream account: the password is checked by ExpertStream
        try: es_uid = e.login(name, pw)
        except ValueError:
            q.append(time.time()); return err(401, "wrong email or password", "authentication_error")
        return session_reply(STORE.linked_user(es_uid, name.lower()), es_uid)
    uid = STORE.check_password(name, pw) if name else None
    if not uid:
        q.append(time.time()); return err(401, "wrong email or password", "authentication_error")
    return session_reply(uid, STORE.es_uid(uid))


@app.post("/auth/sso")
async def sso(request: Request):
    """Signed in on the ExpertStream site (its es_session cookie): the same person, signed in here."""
    e = esc()
    tok = request.cookies.get(e.COOKIE) if e else None
    es_uid = e.verify_token(tok) if tok else None
    if not es_uid: return err(401, "not signed in on ExpertStream", "authentication_error")
    row = e.db().q("SELECT email FROM users WHERE id=?", (es_uid,))
    if not row: return err(401, "not signed in on ExpertStream", "authentication_error")
    return session_reply(STORE.linked_user(es_uid, row[0]["email"]), es_uid)


@app.post("/auth/cookie")
async def cookie(request: Request):
    """A console signed in before cookies existed: set them from its session token."""
    tok = bearer(request)
    su = STORE.session_user(tok) if tok.startswith("ses-") else None
    if not su: return err(401, "sign in first", "authentication_error")
    return signed_in(JSONResponse({"ok": True}), su[0], tok, STORE.es_uid(su[0]))


@app.post("/auth/logout")
async def logout(request: Request):
    tok = bearer(request) or request.cookies.get(COOKIE) or ""
    if tok.startswith("ses-"): STORE.end_session(tok)
    r = JSONResponse({"ok": True}); r.delete_cookie(COOKIE, path="/")
    e = esc()
    if e: r.delete_cookie(e.COOKIE, path="/", domain=os.environ.get("ES_COOKIE_DOMAIN") or None)
    return r


def public_stats():
    nodes = CTRL.online()
    return dict(nodes=len(nodes), cpu_cores=sum(int((n.caps or {}).get("cpu_cores") or 0) for n in nodes),
                ram_gb=round(sum(float((n.caps or {}).get("ram_mb") or 0) for n in nodes) / 1024, 1),
                busy=sum(n.active for n in nodes), capacity=sum(n.capacity for n in nodes),
                models=[dict(id=m["id"], size_mb=m.get("size_mb"), nodes=m["nodes"], warm=m["warm_nodes"] > 0) for m in CTRL.models()],
                speed=CTRL.live_tps(), served=totals(), users=STORE.user_count(),
                signup=setting_on("signup", True), public_compute=setting_on("public_compute", False), rate_per_min=OPTS["rate_per_min"])


@app.get("/public/stats")
async def stats():
    """Live numbers anyone may see: no names, addresses or tokens."""
    return public_stats()


@app.get("/account")
async def account(request: Request):
    u = user_of(request)
    if not u: return err(401, "sign in first", "authentication_error")
    use = STORE.user_usage(u[0])
    row = STORE.db.execute("SELECT created FROM users WHERE id=?", (u[0],)).fetchone()
    cp = user_plan(u[0])
    return dict(username=u[1], created=row[0], usage=use, today=STORE.requests_today(u[0]), db_bytes=DOCS.used_bytes(u[0]),
                storage=dict(quota_bytes=quota(u[0], cp["limits"]), plan_bytes=int(cp["limits"].get("db_mb", 1024)) << 20, extra_gb=STORE.extra_gb(u[0]),
                             addons=STORE.storage_addons(u[0]), price_inr_per_gb=plans_mod.STORAGE["price_inr_per_gb"]),
                plan=dict(id=cp["id"], name=cp["name"], expires=cp["expires"], limits=cp["limits"], contributor=use["contributed_tokens"] > 0),
                plans={p: plan_view(u[0], p) for p in PLANS})


@app.get("/account/keys")
async def account_keys(request: Request):
    u = user_of(request)
    return {"keys": STORE.user_keys(u[0])} if u else err(401, "sign in first", "authentication_error")


@app.post("/account/keys")
async def account_new_key(request: Request):
    u = user_of(request)
    if not u: return err(401, "sign in first", "authentication_error")
    cap = int(user_plan(u[0])["limits"].get("api_keys", 20))
    if len([k for k in STORE.user_keys(u[0]) if not k["revoked"]]) >= cap:
        return err(402, "your plan allows %d API key%s: revoke one or upgrade" % (cap, "" if cap == 1 else "s"))
    b = await body_of(request)
    return {"key": STORE.new_key(str(b.get("label") or "my key")[:64], u[0])}


@app.delete("/account/keys/{prefix}")
async def account_revoke(prefix: str, request: Request):
    u = user_of(request)
    return {"revoked": STORE.user_revoke(u[0], prefix)} if u else err(401, "sign in first", "authentication_error")


def my_nodes(uid):
    mine = set(STORE.user_key_hashes(uid))
    out = []
    for n in CTRL.online():
        if n.owner_hash in mine:
            c = n.caps or {}
            out.append(dict(name=n.name, cpu_cores=c.get("cpu_cores"), ram_mb=c.get("ram_mb"), models=[m for m in n.models], active=n.active,
                            served=n.served, tps=n.tps, online_s=round(time.time() - n.connected)))
    return out


@app.get("/account/compute")
async def account_compute(request: Request):
    u = user_of(request)
    if not u: return err(401, "sign in first", "authentication_error")
    return dict(enabled=setting_on("public_compute", False), nodes=my_nodes(u[0]), contributed=STORE.user_usage(u[0])["contributed_tokens"])


@app.post("/account/compute")
async def account_join(request: Request):
    """A join command for the user's own computer; its work is credited to a key of theirs."""
    u = user_of(request)
    if not u: return err(401, "sign in first", "authentication_error")
    if not setting_on("public_compute", False): return err(403, "this cloud does not take shared computers yet", "permission_error")
    key = STORE.new_key("my computers", u[0])
    host = request.headers.get("host") or "localhost:%d" % OPTS["port"]
    public = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    base = ("https://" if public else "http://") + host
    raw = "https://raw.githubusercontent.com/tharunmaks/stream-/claude/expertstream-moe-android-oegt59/deploy/join.sh"
    return dict(command="curl -sL %s | EC_KEY=%s bash -s -- %s %s" % (raw, key, base, JOIN),
                python="python3 cloud/worker_agent.py --master %s --token %s --key %s" % (("wss://" if public else "ws://") + host, JOIN, key))


@app.get("/admin/users")
async def admin_users(request: Request):
    e = admin_only(request)
    if e: return e
    rows = STORE.db.execute("SELECT id, username, created, last_login, disabled FROM users ORDER BY created DESC LIMIT 500").fetchall()
    return {"users": [dict(id=r[0], username=r[1], created=r[2], last_login=r[3], disabled=bool(r[4]), usage=STORE.user_usage(r[0]),
                           today=STORE.requests_today(r[0]), plans={p: plan_view(r[0], p) for p in PLANS},
                           db_bytes=DOCS.used_bytes(r[0]), extra_gb=STORE.extra_gb(r[0])) for r in rows],
            "plans": PLANS}


@app.post("/admin/settings")
async def admin_settings(request: Request):
    e = admin_only(request)
    if e: return e
    b = await body_of(request)
    for name in ("signup", "public_compute"):
        if name in b: STORE.set_setting(name, "1" if b[name] else "0")
    return {"signup": setting_on("signup", True), "public_compute": setting_on("public_compute", False)}



# ---------------------------------------------------------------------------------------------------------------- plans and billing
def plan_view(uid, product):
    p = user_plan(uid, product)
    return dict(id=p["id"], name=p["name"], price_inr=p["price_inr"], expires=p["expires"], features=p["features"], source=p.get("source"))


def rzp_keys():
    return dict(key_id=os.environ.get("RAZORPAY_KEY_ID") or STORE.setting("rzp_key_id") or "",
                key_secret=os.environ.get("RAZORPAY_KEY_SECRET") or STORE.setting("rzp_key_secret") or "",
                webhook_secret=os.environ.get("RAZORPAY_WEBHOOK_SECRET") or STORE.setting("rzp_webhook_secret") or "")


def apps_view():
    return {"expertstream": {"title": "ExpertStream", "url": site_url(), "plans_url": site_url() + "/", "accounts": bool(esc())},
            "training": {"url": "/training/", "on": bool(TRAINER)}}


@app.get("/plans")
async def list_plans():
    """Every plan of both products, in INR a month, and whether online payment is switched on."""
    return {"currency": "INR", "period_days": plans_mod.PERIOD_DAYS, "products": PLANS, "storage": plans_mod.STORAGE,
            "payments": billing.configured(rzp_keys()), "apps": apps_view()}


@app.get("/account/plans")
async def account_plans(request: Request):
    """The caller's plan in each product. For the ExpertStream app: sign the user in (POST /auth/login), then call this."""
    h, e = auth(request)
    if e: return e
    uid = request.state.uid
    if not uid: return err(403, "plans belong to user accounts", "permission_error")
    return {p: plan_view(uid, p) for p in PLANS}


def activate(order_id, payment_id):
    pay = STORE.payment(order_id)
    if not pay or not STORE.mark_paid(order_id, payment_id): return False
    if pay["plan"].startswith("storage:"):   # extra database space for 30 days
        STORE.add_storage(pay["user_id"], int(pay["plan"].split(":", 1)[1]), plans_mod.PERIOD_DAYS, order_id)
        print("payment %s: %s for user %d" % (payment_id, pay["plan"], pay["user_id"]), flush=True)
        return True
    product, plan = plans_mod.find(PLANS, pay["plan"])
    if not plan: return False
    STORE.set_subscription(pay["user_id"], product, plan["id"], plans_mod.PERIOD_DAYS, "razorpay")
    print("payment %s: %s for user %d" % (payment_id, plan["id"], pay["user_id"]), flush=True)
    return True


@app.post("/billing/checkout")
async def checkout(request: Request):
    u = user_of(request)
    if not u: return err(401, "sign in first", "authentication_error")
    b = await body_of(request)
    st = plans_mod.STORAGE
    if b.get("storage_gb") is not None:
        try: gb = int(b["storage_gb"])
        except (TypeError, ValueError): return err(400, "storage_gb must be a whole number")
        if not st["min_gb"] <= gb <= st["max_gb"]: return err(400, "buy between %d and %d GB" % (st["min_gb"], st["max_gb"]))
        item, price, desc = "storage:%d" % gb, gb * st["price_inr_per_gb"], "%d GB extra database space (30 days)" % gb
    else:
        product, plan = plans_mod.find(PLANS, str(b.get("plan") or ""))
        if product and PLANS[product].get("sold_on_site"): return err(400, "Expert One is bought on ExpertStream: %s/" % site_url())
        if not plan or plan["price_inr"] <= 0: return err(400, "pick a paid plan")
        item, price, desc = plan["id"], plan["price_inr"], "%s %s (30 days)" % (PLANS[product]["title"], plan["name"])
    keys = rzp_keys()
    if not billing.configured(keys): return err(503, "online payment is not switched on yet: contact the owner", "server_error")
    amount = int(price * 100)
    try:
        order = await billing.create_order(keys, amount, "u%d-%s" % (u[0], item.replace(":", "-")), {"user": u[1], "item": item})
    except billing.PaymentError as ex:
        return err(502, str(ex), "server_error")
    STORE.new_payment(order["id"], u[0], item, amount)
    return dict(order_id=order["id"], amount=amount, currency="INR", key_id=keys["key_id"], plan=item, description=desc, username=u[1])


@app.post("/billing/verify")
async def verify(request: Request):
    u = user_of(request)
    if not u: return err(401, "sign in first", "authentication_error")
    b = await body_of(request)
    order_id, payment_id = str(b.get("order_id") or ""), str(b.get("payment_id") or "")
    pay = STORE.payment(order_id)
    if not pay or pay["user_id"] != u[0]: return err(404, "no such order")
    if not billing.payment_signature_ok(rzp_keys(), order_id, payment_id, b.get("signature")): return err(400, "the payment could not be verified")
    activate(order_id, payment_id)   # False when the webhook was first: the plan is already on
    if pay["plan"].startswith("storage:"): return {"ok": True, "extra_gb": STORE.extra_gb(u[0])}
    product, _ = plans_mod.find(PLANS, pay["plan"])
    return {"ok": True, "plan": plan_view(u[0], product)}


@app.post("/billing/webhook")
async def webhook(request: Request):
    ev = billing.webhook_event(rzp_keys()["webhook_secret"], await request.body(), request.headers.get("x-razorpay-signature"))
    if ev is None: return err(400, "bad signature")
    pl = ev.get("payload") or {}
    pay = ((pl.get("payment") or {}).get("entity")) or {}
    order_id = pay.get("order_id") or ((pl.get("order") or {}).get("entity") or {}).get("id")
    if ev.get("event") in ("payment.captured", "order.paid") and order_id and pay.get("id"): activate(order_id, pay["id"])
    return {"ok": True}


@app.get("/billing/payments")
async def my_payments(request: Request):
    u = user_of(request)
    return {"payments": STORE.payments(u[0])} if u else err(401, "sign in first", "authentication_error")


@app.get("/admin/billing")
async def admin_billing(request: Request):
    e = admin_only(request)
    if e: return e
    k = rzp_keys()
    return dict(configured=billing.configured(k), key_id=k["key_id"], webhook_set=bool(k["webhook_secret"]), test_mode=k["key_id"].startswith("rzp_test_"),
                webhook_url=str(request.base_url).rstrip("/") + "/billing/webhook", payments=STORE.payments(), apps=apps_view())


@app.post("/admin/billing")
async def admin_billing_set(request: Request):
    e = admin_only(request)
    if e: return e
    b = await body_of(request)
    for field, name in (("key_id", "rzp_key_id"), ("key_secret", "rzp_key_secret"), ("webhook_secret", "rzp_webhook_secret"), ("expertstream_url", "expertstream_url")):
        if field in b: STORE.set_setting(name, str(b[field] or "").strip())
    return {"ok": True, "configured": billing.configured(rzp_keys())}


@app.post("/admin/users/{username}/storage")
async def admin_add_storage(username: str, request: Request):
    """Give a user extra database space by hand: {"gb": 10, "days": 30}."""
    e = admin_only(request)
    if e: return e
    uid = STORE.user_id(username)
    if not uid: return err(404, "no such user")
    b = await body_of(request)
    try: gb, days = int(b.get("gb")), float(b.get("days") or plans_mod.PERIOD_DAYS)
    except (TypeError, ValueError): return err(400, "gb must be a whole number")
    if gb < 1 or days <= 0: return err(400, "gb and days must be positive")
    STORE.add_storage(uid, gb, days, "owner")
    return {"ok": True, "extra_gb": STORE.extra_gb(uid)}


@app.post("/admin/users/{username}/plan")
async def admin_set_plan(username: str, request: Request):
    """Give a user a plan by hand (paid another way, a friend, a test): {"plan": "cloud_pro", "days": 30}; days null = no end;
    {"plan": null, "product": "cloud"} ends it."""
    e = admin_only(request)
    if e: return e
    uid = STORE.user_id(username)
    if not uid: return err(404, "no such user")
    b = await body_of(request)
    if not b.get("plan"):
        STORE.end_subscription(uid, str(b.get("product") or "cloud")); return {"ok": True}
    product, plan = plans_mod.find(PLANS, str(b["plan"]))
    if not plan: return err(400, "no such plan")
    if product == "expertone":   # written where the site keeps it
        e, es_uid = esc(), STORE.es_uid(uid)
        if not e or not es_uid: return err(400, "this account has no ExpertStream login (it signed up with a username)")
        e.set_subscription(es_uid, None if plan["price_inr"] == 0 else plan["id"], "owner", "owner")
        return {"ok": True}
    if plan["price_inr"] == 0: STORE.end_subscription(uid, product); return {"ok": True}
    days = b.get("days", plans_mod.PERIOD_DAYS)
    return {"ok": True, "expires": STORE.set_subscription(uid, product, plan["id"], None if days is None else float(days), "owner")}


# ---------------------------------------------------------------------------------------------------------------- database
def quota(uid, limits):
    """Database space of a user: the plan's, plus extra space bought, in bytes."""
    return (int(limits.get("db_mb", 1024)) << 20) + (int(STORE.extra_gb(uid)) << 30)


def db_user(request):
    """(user id, plan limits, quota bytes) or an error response."""
    h, e = auth(request)
    if e: return None, e
    if not request.state.uid: return None, err(403, "the database belongs to user accounts: sign in, or use a key made in your console", "permission_error")
    lim = request.state.plan["limits"]
    return (request.state.uid, lim, quota(request.state.uid, lim)), None


def db_call(fn):
    try:
        return fn()
    except DbError as ex:
        return err(ex.status, str(ex))


def jarg(text, what, kind=dict):
    if not text: return kind()
    try: v = json.loads(text)
    except ValueError: raise DbError(400, "%s must be JSON" % what)
    if not isinstance(v, kind): raise DbError(400, "%s must be a JSON %s" % (what, "object" if kind is dict else "list"))
    return v


@app.get("/db")
async def db_list(request: Request):
    u, e = db_user(request)
    if e: return e
    return dict(databases=DOCS.list_dbs(u[0]), used_bytes=DOCS.used_bytes(u[0]), quota_bytes=u[2], plan_bytes=int(u[1].get("db_mb", 1024)) << 20,
                extra_gb=STORE.extra_gb(u[0]), max_databases=u[1].get("databases", 1), max_indexes=u[1].get("indexes", 5),
                max_snapshots=u[1].get("snapshots", 1), price_inr_per_gb=plans_mod.STORAGE["price_inr_per_gb"])


@app.post("/db")
async def db_create(request: Request):
    u, e = db_user(request)
    if e: return e
    name = str((await body_of(request)).get("name") or "")
    return db_call(lambda: (DOCS.create_db(u[0], name, int(u[1].get("databases", 1))), {"ok": True, "name": name})[1])


@app.delete("/db/{db}")
async def db_drop(db: str, request: Request):
    u, e = db_user(request)
    return e or {"deleted": DOCS.drop_db(u[0], db)}


# database-level actions first, so their names are never taken for collections (collection names cannot start with _)
@app.post("/db/{db}/_batch")
async def db_batch(db: str, request: Request):
    """A transaction: {"ops": [{"op": "put"|"patch"|"delete", "collection": ..., "doc"|"id"|"changes": ...}, ...]}: all, or none."""
    u, e = db_user(request)
    if e: return e
    b = await body_of(request)
    return db_call(lambda: {"results": DOCS.batch(u[0], db, b.get("ops"), u[2])})


@app.get("/db/{db}/_snapshots")
async def db_snapshots(db: str, request: Request):
    u, e = db_user(request)
    return e or db_call(lambda: {"snapshots": DOCS.snapshots(u[0], db), "max": u[1].get("snapshots", 1)})


@app.post("/db/{db}/_snapshots")
async def db_snapshot(db: str, request: Request):
    u, e = db_user(request)
    if e: return e
    label = (await body_of(request)).get("label")
    return db_call(lambda: DOCS.snapshot(u[0], db, label, u[2], int(u[1].get("snapshots", 1))))


@app.post("/db/{db}/_snapshots/{sid}/restore")
async def db_restore(db: str, sid: str, request: Request):
    u, e = db_user(request)
    return e or db_call(lambda: DOCS.restore(u[0], db, sid))


@app.delete("/db/{db}/_snapshots/{sid}")
async def db_snapshot_drop(db: str, sid: str, request: Request):
    u, e = db_user(request)
    return e or {"deleted": DOCS.drop_snapshot(u[0], db, sid)}


@app.get("/db/{db}/_export")
async def db_export(db: str, request: Request):
    """The whole database as one SQLite file (table docs: coll, id, data JSON)."""
    u, e = db_user(request)
    if e: return e
    try: path = DOCS.export_file(u[0], db)
    except DbError as ex: return err(ex.status, str(ex))
    from starlette.background import BackgroundTask
    return FileResponse(path, filename=db + ".sqlite", media_type="application/vnd.sqlite3", background=BackgroundTask(os.remove, path))


@app.delete("/db/{db}/{coll}")
async def db_drop_coll(db: str, coll: str, request: Request):
    u, e = db_user(request)
    return e or db_call(lambda: {"deleted": DOCS.drop_coll(u[0], db, coll)})


@app.get("/db/{db}/{coll}")
async def db_find(db: str, coll: str, request: Request, limit: int = 50, offset: int = 0, where: str = "", sort: str = "", fields: str = "",
                  search: str = ""):
    """Documents. where: a JSON filter, e.g. {"status": "paid", "total": {"$gte": 100}, "tags": {"$contains": "new"}};
    sort: fields separated by commas, - for descending (e.g. -total,name); fields: the fields to return; search: text anywhere."""
    u, e = db_user(request)
    if e: return e
    return db_call(lambda: DOCS.find(u[0], db, coll, jarg(where, "where"), limit, offset, [x for x in sort.split(",") if x],
                                     [x for x in fields.split(",") if x], search))


@app.post("/db/{db}/{coll}/_query")
async def db_query(db: str, coll: str, request: Request):
    """The same as GET, with the query in the body: {"where", "sort": [...], "fields": [...], "search", "limit", "offset", "explain"}."""
    u, e = db_user(request)
    if e: return e
    b = await body_of(request)

    def run():
        if b.get("explain"): return {"plan": DOCS.explain(u[0], db, coll, b.get("where") or {})}
        return DOCS.find(u[0], db, coll, b.get("where") or {}, b.get("limit", 50), b.get("offset", 0), b.get("sort") or [], b.get("fields") or [], b.get("search"))
    return db_call(run)


@app.get("/db/{db}/{coll}/_indexes")
async def db_indexes(db: str, coll: str, request: Request):
    u, e = db_user(request)
    return e or db_call(lambda: {"indexes": DOCS.indexes(u[0], db, coll), "max_per_database": u[1].get("indexes", 5)})


@app.post("/db/{db}/{coll}/_indexes")
async def db_index_create(db: str, coll: str, request: Request):
    u, e = db_user(request)
    if e: return e
    field = str((await body_of(request)).get("field") or "")
    return db_call(lambda: (DOCS.create_index(u[0], db, coll, field, int(u[1].get("indexes", 5))), {"ok": True, "field": field})[1])


@app.delete("/db/{db}/{coll}/_indexes/{field}")
async def db_index_drop(db: str, coll: str, field: str, request: Request):
    u, e = db_user(request)
    return e or db_call(lambda: {"deleted": DOCS.drop_index(u[0], db, coll, field)})


@app.post("/db/{db}/{coll}")
async def db_insert(db: str, coll: str, request: Request):
    """One document (a JSON object) or many (a JSON list: an export of another database). An "id" field replaces that document."""
    u, e = db_user(request)
    if e: return e
    try: b = await request.json()
    except ValueError: return err(400, "body must be JSON")
    docs = b if isinstance(b, list) else [b]
    if len(docs) > 10000: return err(413, "10,000 documents at most per request")
    return db_call(lambda: {"ids": DOCS.put_many(u[0], db, coll, docs, u[2])})


@app.get("/db/{db}/{coll}/{doc_id}")
async def db_get(db: str, coll: str, doc_id: str, request: Request):
    u, e = db_user(request)
    return e or db_call(lambda: DOCS.get(u[0], db, coll, doc_id))


@app.put("/db/{db}/{coll}/{doc_id}")
async def db_put(db: str, coll: str, doc_id: str, request: Request):
    u, e = db_user(request)
    if e: return e
    try: b = await request.json()
    except ValueError: return err(400, "body must be JSON")
    if not isinstance(b, dict): return err(400, "a document is a JSON object")
    return db_call(lambda: {"id": DOCS.put_many(u[0], db, coll, [dict(b, id=doc_id)], u[2])[0]})


@app.patch("/db/{db}/{coll}/{doc_id}")
async def db_patch(db: str, coll: str, doc_id: str, request: Request):
    """Change part of a document: {"status": "paid", "address.city": "Pune", "$inc": {"stock": -1}, "$push": {"tags": "new"}, "$unset": ["tmp"]}"""
    u, e = db_user(request)
    if e: return e
    b = await body_of(request)
    return db_call(lambda: DOCS.patch(u[0], db, coll, doc_id, b, u[2]))


@app.delete("/db/{db}/{coll}/{doc_id}")
async def db_delete(db: str, coll: str, doc_id: str, request: Request):
    u, e = db_user(request)
    return e or db_call(lambda: (DOCS.delete(u[0], db, coll, doc_id), {"ok": True})[1])


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
    return dict(join=join_command(request), join_token=JOIN, public=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https", local_join="python3 cloud/worker_agent.py --master ws://127.0.0.1:%d --token %s" % (OPTS["port"], JOIN),
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
            "uptime_s": round(time.time() - STARTED), "users": STORE.user_count(),
            "settings": dict(signup=setting_on("signup", True), public_compute=setting_on("public_compute", False))}


@app.get("/auth/check")
async def auth_check(request: Request):
    """For the dashboard's sign-in: which kind of token this is, or a hint why it is refused."""
    tok = bearer(request)
    if is_admin(tok): return {"ok": True, "role": "admin"}
    if tok.startswith("ses-") and STORE.session_user(tok): return {"ok": True, "role": "user", "hint": "that is a user sign-in; the admin pages need the admin token"}
    if tok and STORE.check_key(tok): return {"ok": True, "role": "key", "hint": token_hint(tok, "admin")}
    return err(401, token_hint(tok, "admin"), "authentication_error")


# ---------------------------------------------------------------------------------------------------------------- training
TRAINER = None


def training_app():
    """The Training app (apps/es-trainer), each request carrying the signed-in user and their plan's training limits."""
    import importlib.util
    from http.cookies import SimpleCookie
    os.environ.setdefault("ES_BIN_DIR", ROOT)                                     # es_train, es_lora, es_chat built here
    os.environ.setdefault("ES_MODEL_DIR", os.path.expanduser("~/models"))
    os.environ.setdefault("ES_TRAIN_RUNS", os.path.join(os.path.dirname(os.path.abspath(STORE.path)), "training", "runs"))
    os.environ.setdefault("ES_TRAIN_DATA", os.path.join(os.path.dirname(os.path.abspath(STORE.path)), "training", "data"))
    spec = importlib.util.spec_from_file_location("es_trainer_app", os.path.join(ROOT, "apps", "es-trainer", "app.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    flask_app = mod.app

    def guarded(environ, start_response):
        c = SimpleCookie(); c.load(environ.get("HTTP_COOKIE", ""))
        su = STORE.session_user(c[COOKIE].value) if COOKIE in c else None
        if not su:
            body = (b'<!doctype html><meta charset="utf-8"><body style="font:15px system-ui;background:#0b0d12;color:#e8ebf2;padding:40px">'
                    b'Sign in to ExpertStream Cloud to use Training. <a style="color:#7c6cff" href="/" target="_top">Sign in</a></body>')
            start_response("401 Unauthorized", [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))])
            return [body]
        lim = user_plan(su[0])["limits"]
        environ["es.user"] = "u%d" % su[0]
        environ["es.limits"] = {"max_steps": lim.get("train_steps"), "moe": lim.get("train_moe"), "max_threads": lim.get("train_threads"),
                               "hf_download": bool(lim.get("train_moe"))}   # adding models to the shared library: paid plans
        return flask_app(environ, start_response)
    return guarded


PAGE_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY", "X-Content-Type-Options": "nosniff"}


@app.get("/")
async def console():
    """The public cloud: sign up, models, playground, API keys, sharing a computer."""
    return FileResponse(os.path.join(ROOT, "dashboard", "console.html"), headers=PAGE_HEADERS)


@app.get("/dashboard")
@app.get("/admin")
async def dashboard():
    """The owner's control room (admin token)."""
    return FileResponse(os.path.join(ROOT, "dashboard", "index.html"), headers=PAGE_HEADERS)


@app.get("/ui.css")
@app.get("/ui.js")
async def ui_asset(request: Request):
    name = request.url.path.lstrip("/")
    return FileResponse(os.path.join(ROOT, "dashboard", name), media_type="text/css" if name.endswith(".css") else "text/javascript",
                        headers={"Cache-Control": "no-cache"})


DEPLOY_FILES = {"Dockerfile": "deploy/huggingface/Dockerfile", "README.md": "deploy/huggingface/README.md"}


@app.get("/deploy/huggingface/{name}")
async def deploy_file(name: str):
    """The two files of a Hugging Face Space that hosts this cloud for free (no secrets in them)."""
    if name not in DEPLOY_FILES: return err(404, "no such file")
    return FileResponse(os.path.join(ROOT, DEPLOY_FILES[name]), media_type="text/plain; charset=utf-8")


@app.get("/health")
async def health():
    return {"ok": True, "nodes": len(CTRL.online()), "models": [m["id"] for m in CTRL.models()]}


def init(db_path, rotate=False):
    global STORE, CTRL, ADMIN, JOIN
    global PLANS, DOCS
    STORE = Store(db_path)
    CTRL = Controller(STORE)
    base = os.path.dirname(os.path.abspath(db_path))
    PLANS = plans_mod.load(os.environ.get("EXPERTCLOUD_PLANS") or os.path.join(base, "plans.json"))
    global TRAINER
    if TRAINER is None and os.environ.get("ES_TRAINING", "1") != "0":
        try:
            from starlette.middleware.wsgi import WSGIMiddleware
            TRAINER = training_app()
            app.mount("/training", WSGIMiddleware(TRAINER))
        except Exception as ex:   # noqa: BLE001 - Flask missing: the Cloud runs without Training
            print("Training off: %s (pip install flask)" % ex, flush=True)
    DOCS = DocStore(os.environ.get("EXPERTCLOUD_DATA") or os.path.join(base, "clouddata"), legacy=os.path.join(base, "clouddata.db"))
    if rotate:
        STORE.set_setting("admin_token", "adm-" + secrets.token_urlsafe(24))
        STORE.set_setting("join_token", "join-" + secrets.token_urlsafe(18))
    for name, env in (("admin_token", "EXPERTCLOUD_ADMIN_TOKEN"), ("join_token", "EXPERTCLOUD_JOIN_TOKEN")):
        if os.environ.get(env, "").strip(): STORE.set_setting(name, clean_token(os.environ[env]))   # fixed tokens for cloud hosts (secrets)
    ADMIN = STORE.setting("admin_token", lambda: "adm-" + secrets.token_urlsafe(24))
    JOIN = STORE.setting("join_token", lambda: "join-" + secrets.token_urlsafe(18))
    d = os.path.expanduser("~/.expertcloud"); os.makedirs(d, exist_ok=True)
    for name, val in (("admin_token", ADMIN), ("join_token", JOIN)):
        p = os.path.join(d, name)
        with open(os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f: f.write(val)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept nodes and users from other machines")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT") or 8000))
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
    print("  public page (sign up, models, API keys): %s/" % local)
    print("  owner dashboard (signs in by itself): %s/dashboard#token=%s" % (local, ADMIN))
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
