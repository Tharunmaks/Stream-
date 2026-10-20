"""ExpertCloud control plane: the registry of worker nodes, the Auto-Combo scoring router and job dispatch with failover.

Workers dial out to the gateway over a WebSocket (cloud/worker_agent.py), so nodes behind home routers need no port forwarding.
Protocol, one JSON object per message:
  worker -> controller  hello {node_id, name, caps, models, capacity, owner_key?}, heartbeat {load, models, tps, ttft},
                        chunk {job, delta}, done {job, usage, ttft, tps}, error {job, message}, deploy_result {req, ok, message, models}
  controller -> worker  infer {job, model, messages, params}, cancel {job}, deploy {req, repo, file}
"""
import asyncio
import itertools
import json
import time
import uuid

FIRST_TOKEN_TIMEOUT = 600.0     # a cold big model can take minutes to load its core; a dropped node fails over at once
ALIASES = {"auto", "expertstream-fast", "expertstream-coder"}


class Node:
    def __init__(self, ws, hello):
        self.ws = ws
        self.id = str(hello.get("node_id") or uuid.uuid4())
        self.name = str(hello.get("name") or self.id[:8])[:64]
        self.caps = hello.get("caps") or {}
        self.models = {m["name"]: m for m in hello.get("models") or [] if isinstance(m, dict) and m.get("name")}
        self.capacity = max(1, int(hello.get("capacity") or 1))
        self.load = {}
        self.active = 0
        self.ttft = None     # rolling averages (EWMA), seconds and tokens/s
        self.tps = None
        self.served = 0
        self.failures = 0
        self.last_seen = time.time()
        self.connected = time.time()
        self.owner_hash = None
        self.jobs = {}       # job id -> asyncio.Queue of events
        self.deploys = {}    # request id -> Future
        self.send_lock = asyncio.Lock()

    async def send(self, obj):
        async with self.send_lock:
            await self.ws.send_text(json.dumps(obj))

    def view(self):
        return dict(id=self.id, name=self.name, caps=self.caps, load=self.load, capacity=self.capacity, active=self.active,
                    models=list(self.models.values()), ttft_s=self.ttft, tps=self.tps, served=self.served, failures=self.failures,
                    last_seen=self.last_seen, online_s=round(time.time() - self.connected), ping_ms=self.load.get("ping_ms"))


def ewma(old, new, a=0.3):
    return new if old is None else (1 - a) * old + a * new


class Controller:
    def __init__(self, store):
        self.store = store
        self.nodes = {}
        self.waiters = []                    # (priority, seq, future): requests waiting for a free slot
        self.seq = itertools.count()
        self.tps_window = []                 # (time, tokens) of recent answers, for the live speedometer

    # ---- registry ----
    def add(self, node):
        old = self.nodes.get(node.id)
        if old: self.drop(old, "replaced by a new connection")
        self.nodes[node.id] = node

    def drop(self, node, why="disconnected"):
        if self.nodes.get(node.id) is node: del self.nodes[node.id]
        for q in node.jobs.values(): q.put_nowait(("lost", why))
        for f in node.deploys.values():
            if not f.done(): f.set_result({"ok": False, "message": why})
        self.wake()

    def on_message(self, node, msg):
        node.last_seen = time.time()
        t = msg.get("type")
        if t == "heartbeat":
            node.load = msg.get("load") or {}
            if isinstance(msg.get("models"), list): node.models = {m["name"]: m for m in msg["models"] if isinstance(m, dict) and m.get("name")}
            if msg.get("capacity"): node.capacity = max(1, int(msg["capacity"]))
        elif t in ("chunk", "done", "error"):
            q = node.jobs.get(msg.get("job"))
            if q: q.put_nowait((t, msg))
        elif t == "deploy_result":
            f = node.deploys.pop(msg.get("req"), None)
            if f and not f.done(): f.set_result(msg)
            if isinstance(msg.get("models"), list): node.models = {m["name"]: m for m in msg["models"] if isinstance(m, dict) and m.get("name")}

    def online(self):
        now = time.time()
        return [n for n in self.nodes.values() if now - n.last_seen < 20]

    def models(self):
        seen = {}
        for n in self.online():
            for name, m in n.models.items():
                e = seen.setdefault(name, dict(id=name, object="model", owned_by="expertcloud", nodes=0, warm_nodes=0, size_mb=m.get("size_mb")))
                e["nodes"] += 1; e["warm_nodes"] += 1 if m.get("warm") else 0
        return list(seen.values())

    # ---- Auto-Combo scoring ----
    def resolve(self, model):
        """The concrete model for an alias: auto = the biggest model online (warm preferred), fast = the smallest warm one,
        coder = a model with 'coder' in its name (else auto)."""
        ms = self.models()
        if not ms: return None
        if model not in ALIASES: return model if any(m["id"] == model for m in ms) else None
        if model == "expertstream-coder":
            c = [m for m in ms if "coder" in m["id"].lower()]
            if c: ms = c
        warm = [m for m in ms if m["warm_nodes"]] or ms
        key = lambda m: m.get("size_mb") or 0
        return (min(warm, key=key) if model == "expertstream-fast" else max(warm, key=key))["id"]

    def score(self, node, model, ctx_tokens):
        """Higher is better; None if the node cannot take the request now."""
        m = node.models.get(model)
        if not m or node.active >= node.capacity: return None
        free_mb = float(node.load.get("ram_free_mb") or 0)
        need_mb = (0 if m.get("warm") else float(m.get("core_mb") or 0)) + ctx_tokens / 1000 * 64   # core to load + KV cache
        s = 3.0 * (1 - node.active / node.capacity)                            # 1. concurrent requests vs capacity
        s += 2.0 / (1 + (node.ttft if node.ttft is not None else 2.0))         # 2. rolling time to first token
        s += 1.0 if free_mb >= need_mb else -3.0                               # 3. RAM for the model and the context
        s += 3.0 if m.get("warm") else 0.0                                     # 4. weights already warm on that node
        s += 0.5 * min((node.tps or 0) / 20, 1)                                # measured speed
        s -= 1.0 * min(node.failures, 3) / 3                                   # recent failures
        s -= 0.5 * float(node.load.get("cpu_pct") or 0) / 100                  # busy with other work
        return s

    def ranked(self, model, ctx_tokens, exclude=()):
        c = [(self.score(n, model, ctx_tokens), n) for n in self.online() if n.id not in exclude]
        return [n for s, n in sorted((x for x in c if x[0] is not None), key=lambda x: -x[0])]

    # ---- waiting for a slot (contributors first) ----
    def wake(self):
        while self.waiters:
            prio, seq, f = self.waiters[0]
            if f.done(): self.waiters.pop(0); continue
            f.set_result(True); self.waiters.pop(0); break

    async def wait_slot(self, priority, timeout):
        f = asyncio.get_running_loop().create_future()
        self.waiters.append((priority, next(self.seq), f)); self.waiters.sort(key=lambda w: (w[0], w[1]))
        try:
            await asyncio.wait_for(f, timeout)
        except asyncio.TimeoutError:
            pass

    # ---- running a request ----
    async def run(self, model, messages, params, priority=1, queue_timeout=120.0):
        """Async generator of ("delta", text) and finally ("done", info). Fails over to the next best node if one drops or errors
        before its first token; raises RuntimeError when no node can serve."""
        target = self.resolve(model)
        if not target: raise RuntimeError("no online node has model %r" % model)
        ctx = sum(len(m.get("content") or "") for m in messages if isinstance(m.get("content"), str)) // 4 + int(params.get("max_tokens") or 512)
        tried, deadline, last_err = set(), time.time() + queue_timeout, "busy"
        while True:
            cands = self.ranked(target, ctx, tried)
            if not cands:
                if not any(n.models.get(target) for n in self.online() if n.id not in tried): raise RuntimeError("no node left for %s (%s)" % (target, last_err))
                if time.time() > deadline: raise RuntimeError("all nodes busy for %.0f s" % queue_timeout)
                await self.wait_slot(priority, max(0.05, deadline - time.time()))
                continue
            node = cands[0]
            job = uuid.uuid4().hex
            q = asyncio.Queue()
            node.jobs[job] = q; node.active += 1
            t0, first, ntok = time.time(), None, 0
            try:
                await node.send(dict(type="infer", job=job, model=target, messages=messages, params=params))
                while True:
                    try:
                        kind, msg = await asyncio.wait_for(q.get(), FIRST_TOKEN_TIMEOUT if first is None else 300.0)
                    except asyncio.TimeoutError:
                        kind, msg = "lost", "timed out"
                    if kind == "chunk":
                        if first is None: first = time.time() - t0
                        ntok += 1
                        yield ("delta", msg.get("delta") or "")
                    elif kind == "done":
                        usage = msg.get("usage") or {}
                        node.ttft = ewma(node.ttft, first if first is not None else time.time() - t0)
                        if msg.get("tps"): node.tps = ewma(node.tps, float(msg["tps"]))
                        node.served += 1; node.failures = max(0, node.failures - 1)
                        gen = int(usage.get("completion_tokens") or ntok)
                        self.store.credit_node(node.id, node.owner_hash, gen)
                        gtps = float(msg.get("tps") or 0)
                        self.tps_window.append((time.time(), gen, gen / gtps if gtps > 0 else time.time() - t0 - (first or 0)))   # generation time
                        yield ("done", dict(node=node.name, node_id=node.id, model=target, usage=usage, ttft=first, tps=msg.get("tps")))
                        return
                    else:   # error or lost connection
                        node.failures += 1
                        last_err = (msg.get("message") if isinstance(msg, dict) else msg) or "error"
                        if first is not None: raise RuntimeError("node %s failed mid-answer: %s" % (node.name, last_err))
                        tried.add(node.id)
                        break   # fail over to the next node
            finally:
                node.jobs.pop(job, None); node.active = max(0, node.active - 1)
                self.wake()

    def live_tps(self, window=60.0):
        now = time.time()
        self.tps_window = [x for x in self.tps_window if now - x[0] < window]
        toks = sum(x[1] for x in self.tps_window); secs = sum(x[2] for x in self.tps_window)
        return dict(tokens_last_minute=toks, answers_last_minute=len(self.tps_window), avg_tps=(toks / secs) if secs > 0 else 0.0,   # generation speed
                    cluster_tps=sum((n.tps or 0) for n in self.online() if n.active))

    async def deploy(self, node_id, repo, file, timeout=3600.0):
        node = self.nodes.get(node_id)
        if not node: return {"ok": False, "message": "no such node"}
        req = uuid.uuid4().hex
        f = asyncio.get_running_loop().create_future()
        node.deploys[req] = f
        await node.send(dict(type="deploy", req=req, repo=repo, file=file))
        try:
            return await asyncio.wait_for(f, timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "message": "deploy timed out"}
