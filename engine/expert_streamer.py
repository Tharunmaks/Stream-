"""ExpertStream 2.0 inference worker: drives the native ExpertStream engine (es_serve) for the cloud.

The weight streaming itself is native code (src/es_engine.c): the model file is memory-mapped, only the experts a token routes to are
read (with the next layer's likely experts prefetched while the current one computes), an expert cache sized to the free RAM keeps
the hot ones, and Turbo checks several guessed tokens in one batched pass (speculative decoding with drafts from the text so far;
a separate draft model is not implemented). This module manages engine processes and streams their answers:

* every model file in the models directory is offered; a model is "warm" while an engine process holds it,
* up to `slots` requests run at once, each in its own engine process (a mapped model's weights are shared through the page cache,
  so extra processes cost little more than their KV cache); more requests wait for a free process,
* deploy downloads a model from Hugging Face with es_get: only its core, experts arrive the first time they are used.
"""
import asyncio
import json
import os
import re
import shutil
import socket
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAFE_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SAFE_FILE = re.compile(r"^[A-Za-z0-9_./-]+\.gguf$")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Instance:
    def __init__(self, model, proc, port):
        self.model, self.proc, self.port = model, proc, port
        self.busy = False
        self.last_used = time.time()


class ExpertStreamer:
    def __init__(self, models_dir, slots=1, threads=0, ctx=4096, cache_mb=0, turbo=True, cache_route=0.0, experts=0, bin_dir=ROOT, idle_s=900):
        self.models_dir = os.path.abspath(os.path.expanduser(models_dir))
        os.makedirs(self.models_dir, exist_ok=True)
        self.slots, self.ctx, self.cache_mb, self.turbo = max(1, slots), ctx, cache_mb, turbo
        self.cache_route, self.experts, self.idle_s = cache_route, experts, idle_s
        self.threads = threads or max(1, (os.cpu_count() or 2) // self.slots)
        self.serve_bin = os.path.join(bin_dir, "es_serve")
        self.get_bin = os.path.join(bin_dir, "es_get")
        self.instances = []
        self.cond = asyncio.Condition()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0))

    # ---- models on this node ----
    def model_files(self):
        out = {}
        for dirpath, _, files in os.walk(self.models_dir):
            for f in files:
                if not f.endswith(".gguf") or f.startswith("."): continue
                m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", f)
                if m and m.group(1) != "00001": continue
                name = f[:m.start()] if m else f[:-5]
                path = os.path.join(dirpath, f)
                parts = [path]
                if m:
                    parts = [os.path.join(dirpath, "%s-%05d-of-%s.gguf" % (name, i, m.group(2))) for i in range(1, int(m.group(2)) + 1)]
                size = sum(os.path.getsize(p) for p in parts if os.path.exists(p))
                out[name] = dict(path=path, size_mb=round(size / 2**20))
        return out

    def models(self):
        warm = {i.model for i in self.instances if i.proc.returncode is None}
        return [dict(name=n, size_mb=v["size_mb"], core_mb=round(v["size_mb"] * 0.25), warm=n in warm) for n, v in sorted(self.model_files().items())]

    def active(self):
        return sum(1 for i in self.instances if i.busy)

    # ---- engine processes ----
    async def _start(self, model):
        files = self.model_files()
        if model not in files: raise RuntimeError("model %s is not on this node" % model)
        if not os.path.exists(self.serve_bin): raise RuntimeError("es_serve not built (make es_serve)")
        port = free_port()
        cmd = [self.serve_bin, "-g", files[model]["path"], "-p", str(port), "-c", str(self.ctx), "-j", str(self.threads)]
        if self.cache_mb: cmd += ["-C", str(self.cache_mb)]
        if not self.turbo: cmd += ["-S", "0"]
        if self.cache_route > 0: cmd += ["--cache-route", str(self.cache_route)]
        if self.experts > 0: cmd += ["--experts", str(self.experts)]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE)
        inst = Instance(model, proc, port)
        t0 = time.time()
        while True:   # a big model maps or reads its core first; wait until it answers
            if proc.returncode is not None:
                err = (await proc.stderr.read()).decode(errors="replace")[-400:]
                raise RuntimeError("engine exited while loading %s: %s" % (model, err.strip()))
            try:
                r = await self.client.get("http://127.0.0.1:%d/health" % port, timeout=2.0)
                if r.status_code == 200: break
            except httpx.HTTPError:
                pass
            if time.time() - t0 > 1800: proc.kill(); raise RuntimeError("engine did not come up within 30 min")
            await asyncio.sleep(0.25)
        asyncio.create_task(self._drain(proc))
        return inst

    async def _drain(self, proc):   # keep the engine's log pipe from filling up
        try:
            while await proc.stderr.read(65536): pass
        except Exception:
            pass

    async def acquire(self, model):
        async with self.cond:
            while True:
                self.instances = [i for i in self.instances if i.proc.returncode is None]
                for i in self.instances:
                    if i.model == model and not i.busy:
                        i.busy = True; return i
                if len(self.instances) < self.slots:
                    break
                idle = [i for i in self.instances if not i.busy]
                if idle:   # a process holding another model is free: replace it
                    victim = min(idle, key=lambda i: i.last_used)
                    victim.proc.terminate(); self.instances.remove(victim)
                    break
                await self.cond.wait()
            placeholder = Instance(model, type("P", (), {"returncode": None})(), 0); placeholder.busy = True
            self.instances.append(placeholder)
        try:
            inst = await self._start(model)
        except Exception:
            async with self.cond:
                self.instances.remove(placeholder); self.cond.notify_all()
            raise
        async with self.cond:
            self.instances[self.instances.index(placeholder)] = inst
            inst.busy = True
        return inst

    async def release(self, inst):
        async with self.cond:
            inst.busy = False; inst.last_used = time.time()
            self.cond.notify_all()

    async def reap_idle(self):
        """Stop engine processes unused for idle_s seconds (their RAM goes back to the system)."""
        while True:
            await asyncio.sleep(30)
            async with self.cond:
                for i in list(self.instances):
                    if not i.busy and i.port and time.time() - i.last_used > self.idle_s:
                        i.proc.terminate(); self.instances.remove(i)

    # ---- inference ----
    async def chat_stream(self, model, messages, params):
        """Async generator: ("delta", text) ..., then ("done", {usage, ttft, tps})."""
        inst = await self.acquire(model)
        try:
            body = dict(model=model, messages=messages, stream=True, stream_options={"include_usage": True})
            for k in ("max_tokens", "temperature", "top_p", "top_k", "seed", "stop"):
                if params.get(k) is not None: body[k] = params[k]
            t0, first, n, usage = time.time(), None, 0, {}
            async with self.client.stream("POST", "http://127.0.0.1:%d/v1/chat/completions" % inst.port, json=body) as r:
                if r.status_code != 200:
                    raise RuntimeError("engine HTTP %d: %s" % (r.status_code, (await r.aread()).decode(errors="replace")[:300]))
                async for line in r.aiter_lines():
                    if not line.startswith("data: "): continue
                    data = line[6:].strip()
                    if data == "[DONE]": break
                    try: ev = json.loads(data)
                    except ValueError: continue
                    if ev.get("error"): raise RuntimeError(str(ev["error"].get("message", ev["error"])))
                    if ev.get("usage"): usage = ev["usage"]
                    for ch in ev.get("choices") or []:
                        d = (ch.get("delta") or {}).get("content")
                        if d:
                            if first is None: first = time.time() - t0
                            n += 1
                            yield ("delta", d)
            gen_s = time.time() - t0 - (first or 0)
            toks = int(usage.get("completion_tokens") or n)
            yield ("done", dict(usage=usage or dict(completion_tokens=n), ttft=first, tps=(toks / gen_s) if gen_s > 0 and toks > 1 else None))
        finally:
            await self.release(inst)

    # ---- deployment ----
    async def deploy(self, repo, file):
        """Download a model's core from Hugging Face (experts arrive on first use)."""
        if not SAFE_REPO.match(repo or "") or not SAFE_FILE.match(file or "") or ".." in file:
            return dict(ok=False, message="repo must be owner/name and file a .gguf path")
        if not os.path.exists(self.get_bin): return dict(ok=False, message="es_get not built (make es_get)")
        if shutil.disk_usage(self.models_dir).free < 2 * 2**30: return dict(ok=False, message="less than 2 GB free on this node")
        proc = await asyncio.create_subprocess_exec(self.get_bin, repo, file, "-d", self.models_dir, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await proc.communicate()
        ok = proc.returncode == 0
        tail = (out.decode(errors="replace") + err.decode(errors="replace")).strip().splitlines()
        return dict(ok=ok, message=(tail[-1] if tail else "") [:300])

    async def close(self):
        for i in self.instances:
            if i.port and i.proc.returncode is None: i.proc.terminate()
        await self.client.aclose()
