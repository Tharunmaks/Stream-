#!/usr/bin/env python3
"""ExpertCloud worker: turns this machine into a node of the cloud.

    python3 cloud/worker_agent.py --master ws://GATEWAY:8000 --token JOIN_TOKEN [--key sk-expert-...] [--models ~/models] [--slots 1]

It dials out to the gateway over a WebSocket (works behind home routers and NAT, no port forwarding), reports this machine's
CPU cores, RAM, GPU memory, disk read speed and load every 5 seconds, and runs inference on the native ExpertStream engine
(engine/expert_streamer.py). --key credits the tokens this node generates to your API key (contribute-to-use: contributors get
priority when the cloud is busy).

Security: the gateway can only ask this node to (1) answer a chat request with a model already here and (2) download a .gguf
model from Hugging Face into the models directory. It cannot run commands. Note that the prompts you serve pass through this
machine: join only a gateway you trust, and run it over TLS (wss://) on the internet.
"""
import argparse
import asyncio
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import urllib.parse
import uuid

try:
    import psutil
except ImportError:   # Termux: pkg install python-psutil; without it, /proc is read directly
    psutil = None
import websockets

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.expert_streamer import ExpertStreamer  # noqa: E402

STATE = os.path.expanduser("~/.expertcloud")


def node_id():
    os.makedirs(STATE, exist_ok=True)
    p = os.path.join(STATE, "node_id")
    if not os.path.exists(p):
        with open(p, "w") as f: f.write(uuid.uuid4().hex)
    return open(p).read().strip()


def gpu_info():
    if not shutil.which("nvidia-smi"): return dict(gpu=None, vram_mb=0)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        name, mem = out.strip().splitlines()[0].split(",")
        return dict(gpu=name.strip(), vram_mb=int(float(mem)))
    except Exception:
        return dict(gpu=None, vram_mb=0)


def disk_read_mbps(path):
    """Read speed of the models disk: 64 MB from the biggest model file (uncached if possible), else a temporary file."""
    target, size = None, 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            p = os.path.join(dirpath, f)
            if f.endswith(".gguf") and os.path.getsize(p) > size: target, size = p, os.path.getsize(p)
    tmp = None
    if not target or size < 64 << 20:
        tmp = target = os.path.join(path, ".es_disk_test")
        with open(tmp, "wb") as f: f.write(os.urandom(64 << 20))
        size = 64 << 20
    try:
        fd = os.open(target, os.O_RDONLY)
        try:
            if hasattr(os, "posix_fadvise"): os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            off = max(0, size // 2 - (32 << 20)); t0 = time.time(); n = 0
            while n < 64 << 20:
                b = os.pread(fd, 4 << 20, off + n)
                if not b: break
                n += len(b)
            return round(n / 2**20 / max(time.time() - t0, 1e-6))
        finally:
            os.close(fd)
    finally:
        if tmp and os.path.exists(tmp): os.remove(tmp)


def meminfo():
    """(total, available) bytes."""
    if psutil:
        vm = psutil.virtual_memory(); return vm.total, vm.available
    m = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1); m[k] = int(v.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    total = m.get("MemTotal") or os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return total, m.get("MemAvailable", m.get("MemFree", 0))


CPU_LAST = [None]


def cpu_pct():
    if psutil: return psutil.cpu_percent(interval=None)
    try:
        f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    except (OSError, ValueError):
        return os.getloadavg()[0] / (os.cpu_count() or 1) * 100 if hasattr(os, "getloadavg") else 0.0
    idle, total, last = f[3] + (f[4] if len(f) > 4 else 0), sum(f), CPU_LAST[0]
    CPU_LAST[0] = (idle, total)
    return 0.0 if not last or total == last[1] else round(100 * (1 - (idle - last[0]) / (total - last[1])), 1)


def caps(models_dir):
    total, _ = meminfo()
    return dict(os=platform.system(), arch=platform.machine(), cpu_cores=(psutil.cpu_count(logical=False) if psutil else None) or os.cpu_count(),
                cpu_threads=os.cpu_count(), ram_mb=total // 2**20, disk_read_mbps=disk_read_mbps(models_dir),
                disk_free_mb=shutil.disk_usage(models_dir).free // 2**20, engine="cpu", **gpu_info())


class Worker:
    def __init__(self, args):
        self.args = args
        self.engine = ExpertStreamer(args.models, slots=args.slots, threads=args.threads, ctx=args.ctx, cache_mb=args.cache_mb,
                                     turbo=not args.no_turbo, cache_route=args.cache_route, experts=args.experts)
        self.tasks = {}
        self.ws = None
        self.send_lock = asyncio.Lock()
        self.last_tps = None
        self.last_ping = None

    async def send(self, obj):
        async with self.send_lock:
            await self.ws.send(json.dumps(obj))

    def load(self, ping_ms=None):
        total, avail = meminfo()
        return dict(cpu_pct=cpu_pct(), ram_free_mb=avail // 2**20, ram_mb=total // 2**20,
                    active=self.engine.active(), ping_ms=ping_ms, tps=self.last_tps)

    async def heartbeat(self):
        """Keeps the node listed as online. It never waits for a ping: on a phone that is busy answering, a ping can take
        seconds, and a late heartbeat made the node look offline in the middle of an answer."""
        while True:
            try:
                await self.send(dict(type="heartbeat", load=self.load(self.last_ping), models=self.engine.models(), capacity=self.args.slots))
            except Exception:
                return   # the connection is gone; session() reconnects
            await asyncio.sleep(5)

    async def pinger(self):
        while True:
            t0 = time.time()
            try:
                await asyncio.wait_for(await self.ws.ping(), 20)
                self.last_ping = round((time.time() - t0) * 1000, 1)
            except Exception:
                self.last_ping = None
            await asyncio.sleep(10)

    async def infer(self, msg):
        job = msg.get("job")
        try:
            async for kind, val in self.engine.chat_stream(msg.get("model"), msg.get("messages") or [], msg.get("params") or {}):
                if kind == "delta": await self.send(dict(type="chunk", job=job, delta=val))
                else:
                    if val.get("tps"): self.last_tps = round(val["tps"], 2)
                    await self.send(dict(type="done", job=job, usage=val.get("usage"), ttft=val.get("ttft"), tps=val.get("tps")))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            try: await self.send(dict(type="error", job=job, message=str(e)[:300]))
            except Exception: pass
        finally:
            self.tasks.pop(job, None)

    async def deploy(self, msg):
        res = await self.engine.deploy(msg.get("repo"), msg.get("file"))
        await self.send(dict(type="deploy_result", req=msg.get("req"), models=self.engine.models(), **res))

    async def session(self):
        master = self.args.master.strip().rstrip("/")
        if master.startswith("http"): master = "ws" + master[4:]        # http://host:8000 -> ws://host:8000
        if "://" not in master: master = "ws://" + master
        token = self.args.token.strip(" \t\r\n\"'`<>\u2018\u2019\u201c\u201d")
        url = master + "/cloud/ws?token=" + urllib.parse.quote(token, safe="")
        async with websockets.connect(url, max_size=2**24, ping_interval=None) as ws:
            self.ws = ws
            hello = dict(type="hello", node_id=node_id() + (("-" + self.args.name) if self.args.suffix_name else ""), name=self.args.name,
                         caps=caps(self.engine.models_dir), models=self.engine.models(), capacity=self.args.slots, owner_key=self.args.key)
            await self.send(hello)
            print("joined %s as %s with %d model(s)" % (self.args.master, self.args.name, len(hello["models"])), flush=True)
            hb = asyncio.create_task(self.heartbeat())
            pg = asyncio.create_task(self.pinger())
            try:
                async for raw in ws:
                    try: msg = json.loads(raw)
                    except ValueError: continue
                    t = msg.get("type")
                    if t == "infer": self.tasks[msg.get("job")] = asyncio.create_task(self.infer(msg))
                    elif t == "cancel":
                        task = self.tasks.pop(msg.get("job"), None)
                        if task: task.cancel()
                    elif t == "deploy": asyncio.create_task(self.deploy(msg))
                    elif t == "rejected": print("rejected by the gateway: %s" % msg.get("message"), flush=True); return False
            finally:
                hb.cancel(); pg.cancel()
                for task in self.tasks.values(): task.cancel()
        return True

    async def run(self):
        asyncio.create_task(self.engine.reap_idle())
        delay = 1
        while True:
            try:
                if await self.session() is False: return
                delay = 1
            except (OSError, websockets.WebSocketException) as e:
                print("gateway unreachable (%s); retrying in %d s" % (e, delay), flush=True)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", required=True, help="gateway URL, ws://host:8000 or wss://host")
    ap.add_argument("--token", required=True, help="the cloud's join token (shown on the dashboard)")
    ap.add_argument("--key", default=None, help="your sk-expert-... key, credited with the tokens this node generates")
    ap.add_argument("--models", default="~/models", help="directory of .gguf models")
    ap.add_argument("--name", default=socket.gethostname())
    ap.add_argument("--suffix-name", action="store_true", help=argparse.SUPPRESS)   # several test workers on one machine
    ap.add_argument("--slots", type=int, default=1, help="requests served at once (each in its own engine process)")
    ap.add_argument("--threads", type=int, default=0, help="threads per engine process (default: cores / slots)")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--cache-mb", type=int, default=0, help="expert cache per process (0 = auto)")
    ap.add_argument("--cache-route", type=float, default=0.0, help="prefer experts already in RAM (0.2 measured: +4..13%% speed)")
    ap.add_argument("--experts", type=int, default=0, help="experts per token for MoE models (0 = as trained)")
    ap.add_argument("--no-turbo", action="store_true", help="turn off speculative checking of guessed tokens")
    args = ap.parse_args()
    try:
        asyncio.run(Worker(args).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
