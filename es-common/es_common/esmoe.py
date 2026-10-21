"""Expert Stream MoE format, model creation, and streamed inference (pure numpy + optional torch for training).

Layout of a model directory:
  config.json                      architecture + param counts
  dense.npz                        embeddings, attention, norms, routers, head (always in RAM)
  experts/L{l:03d}/E{e:05d}.i8     one file per expert: int8 [w1 (f x d) | w2 (d x f)] + float32 row scales
Only the top-k experts per token are read from disk (pread via np.memmap), through an LRU cache with a byte budget,
so total params can be far larger than RAM (format scales to 2T total; tested here only on small models).
"""
import json, math, os, time
from collections import OrderedDict
import numpy as np

VOCAB = 256  # byte-level

def plan(total_params, experts=None, topk=2):
    """Pick a MoE config whose total param count is close to total_params (1M .. 70B+)."""
    t = float(total_params)
    if t < 1e6 or t > 2.2e12: raise ValueError("total_params must be between 1M and 2.2T")
    E = experts or (8 if t < 1e8 else 16 if t < 1e10 else 64 if t < 1e11 else 256)
    best = None
    for d in [64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192]:
        f = 2 * d
        for L in range(2, 129, 2):
            n = count(L, E, d, f)
            if best is None or abs(math.log(n / t)) < abs(math.log(best[0] / t)): best = (n, L, d, f)
    n, L, d, f = best
    return dict(layers=L, experts=E, dim=d, ffn=f, topk=min(topk, E), heads=max(1, d // 64), vocab=VOCAB,
                total_params=n, active_params=count(L, E, d, f, active=min(topk, E)))

def count(L, E, d, f, active=None):
    dense = 2 * VOCAB * d + L * (4 * d * d + E * d + 2 * d) + d
    return dense + L * (active if active else E) * 2 * d * f

def _q8(w):
    s = np.abs(w).max(axis=1, keepdims=True) / 127.0 + 1e-12
    return np.clip(np.round(w / s), -127, 127).astype(np.int8), s.astype(np.float32).ravel()

def write_expert(path, w1, w2):
    q1, s1 = _q8(w1); q2, s2 = _q8(w2)
    with open(path, "wb") as fh:
        fh.write(q1.tobytes()); fh.write(q2.tobytes()); fh.write(s1.tobytes()); fh.write(s2.tobytes())

def create(out, cfg, seed=0, materialize_max=2e9, dense=None, experts=None):
    """Write a model directory. Sizes above materialize_max get config + dense only and lazily-initialised
    experts (sparse placeholder files are not written; see config 'materialized')."""
    os.makedirs(out, exist_ok=True)
    L, E, d, f = cfg["layers"], cfg["experts"], cfg["dim"], cfg["ffn"]
    rng = np.random.default_rng(seed)
    mat = cfg["total_params"] <= materialize_max
    if dense is None:
        r = lambda *s: (rng.standard_normal(s) / math.sqrt(s[-1])).astype(np.float32)
        dense = {"emb": r(VOCAB, d), "head": r(VOCAB, d), "lnf": np.ones(d, np.float32)}
        for l in range(L):
            for k in "qkvo": dense[f"{k}{l}"] = r(d, d)
            dense[f"router{l}"] = r(E, d); dense[f"ln1_{l}"] = np.ones(d, np.float32); dense[f"ln2_{l}"] = np.ones(d, np.float32)
    if cfg["total_params"] <= 5e9: np.savez(os.path.join(out, "dense.npz"), **dense)
    if mat:
        for l in range(L):
            os.makedirs(os.path.join(out, "experts", f"L{l:03d}"), exist_ok=True)
            for e in range(E):
                if experts is not None: w1, w2 = experts[l][e]
                else:
                    w1 = (rng.standard_normal((f, d)) / math.sqrt(d)).astype(np.float32)
                    w2 = (rng.standard_normal((d, f)) / math.sqrt(f)).astype(np.float32)
                write_expert(os.path.join(out, "experts", f"L{l:03d}", f"E{e:05d}.i8"), w1, w2)
    cfg = dict(cfg, format="es-moe-v1", materialized=mat, expert_dtype="int8+rowscale", seed=seed)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    return cfg

def _rms(x, g): return x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6) * g

class StreamedMoE:
    def __init__(self, path, cache_mb=256):
        self.p = path; self.cfg = json.load(open(os.path.join(path, "config.json")))
        if not self.cfg.get("materialized", True): raise ValueError("experts not materialized for this size; run_moe needs expert files on disk")
        self.W = dict(np.load(os.path.join(path, "dense.npz")))
        self.cache = OrderedDict(); self.budget = cache_mb << 20; self.used = 0
        self.stats = dict(reads=0, hits=0, bytes_read=0)

    def expert(self, l, e):
        key = (l, e)
        if key in self.cache: self.cache.move_to_end(key); self.stats["hits"] += 1; return self.cache[key]
        d, f = self.cfg["dim"], self.cfg["ffn"]
        fn = os.path.join(self.p, "experts", f"L{l:03d}", f"E{e:05d}.i8")
        with open(fn, "rb") as fh: raw = fh.read()
        q = np.frombuffer(raw, np.int8, 2 * d * f); s = np.frombuffer(raw, np.float32, f + d, 2 * d * f)
        w1 = q[:f * d].reshape(f, d).astype(np.float32) * s[:f, None]
        w2 = q[f * d:].reshape(d, f).astype(np.float32) * s[f:, None]
        sz = w1.nbytes + w2.nbytes
        self.stats["reads"] += 1; self.stats["bytes_read"] += len(raw)
        while self.cache and self.used + sz > self.budget:
            _, (a, b) = self.cache.popitem(last=False); self.used -= a.nbytes + b.nbytes
        self.cache[key] = (w1, w2); self.used += sz
        return w1, w2

    def forward(self, ids):
        c, W = self.cfg, self.W; d, H = c["dim"], c["heads"]; hd = d // H; T = len(ids)
        x = W["emb"][ids]
        mask = np.triu(np.full((T, T), -1e9, np.float32), 1)
        for l in range(c["layers"]):
            h = _rms(x, W[f"ln1_{l}"])
            q, k, v = (h @ W[f"{n}{l}"].T for n in "qkv")
            q = q.reshape(T, H, hd).transpose(1, 0, 2); k = k.reshape(T, H, hd).transpose(1, 0, 2); v = v.reshape(T, H, hd).transpose(1, 0, 2)
            a = q @ k.transpose(0, 2, 1) / math.sqrt(hd) + mask; a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
            x = x + (a @ v).transpose(1, 0, 2).reshape(T, d) @ W[f"o{l}"].T
            h = _rms(x, W[f"ln2_{l}"]); g = h @ W[f"router{l}"].T
            top = np.argsort(-g, axis=1)[:, :c["topk"]]
            gv = np.take_along_axis(g, top, 1); gv = np.exp(gv - gv.max(1, keepdims=True)); gv /= gv.sum(1, keepdims=True)
            y = np.zeros_like(x)
            for e in np.unique(top):
                rows, slot = np.nonzero(top == e); w1, w2 = self.expert(l, int(e))
                z = h[rows] @ w1.T; z = z / (1 + np.exp(-z))  # silu
                y[rows] += (z @ w2.T) * gv[rows, slot][:, None]
            x = x + y
        return _rms(x, W["lnf"]) @ W["head"].T

    def generate(self, prompt, max_new=64, temperature=0.8, seed=0, ctx=128):
        rng = np.random.default_rng(seed); ids = list(prompt.encode("utf-8", "replace")) or [32]
        t0 = time.time()
        for _ in range(max_new):
            lg = self.forward(np.array(ids[-ctx:]))[-1]
            if temperature <= 0: n = int(lg.argmax())
            else: p = np.exp((lg - lg.max()) / temperature); p /= p.sum(); n = int(rng.choice(VOCAB, p=p))
            ids.append(n)
        dt = time.time() - t0
        out = bytes(ids[len(prompt.encode()):]).decode("utf-8", "replace")
        return dict(text=out, new_tokens=max_new, seconds=round(dt, 3), tok_per_s=round(max_new / dt, 2),
                    cache_mb=round(self.used / 2**20, 1), **self.stats)

def train_torch(out, cfg, text, steps=300, seq=64, batch=16, threads=2, lr=3e-3, log=print):
    """Actually train a small byte-level MoE (torch, CPU) and export it in es-moe-v1 format."""
    import torch, torch.nn.functional as F
    torch.set_num_threads(threads); torch.manual_seed(0)
    L, E, d, f, K, H = cfg["layers"], cfg["experts"], cfg["dim"], cfg["ffn"], cfg["topk"], cfg["heads"]
    P = lambda *s: torch.nn.Parameter(torch.randn(*s) / math.sqrt(s[-1]))
    ps = {"emb": P(VOCAB, d), "head": P(VOCAB, d), "lnf": torch.nn.Parameter(torch.ones(d))}
    for l in range(L):
        for k in "qkvo": ps[f"{k}{l}"] = P(d, d)
        ps[f"router{l}"] = P(E, d); ps[f"ln1_{l}"] = torch.nn.Parameter(torch.ones(d)); ps[f"ln2_{l}"] = torch.nn.Parameter(torch.ones(d))
        ps[f"w1_{l}"] = P(E, f, d); ps[f"w2_{l}"] = P(E, d, f)
    rms = lambda x, g: x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6) * g
    def fwd(ids):
        B, T = ids.shape; x = ps["emb"][ids]; hd = d // H
        for l in range(L):
            h = rms(x, ps[f"ln1_{l}"]); q, k, v = (h @ ps[f"{n}{l}"].T for n in "qkv")
            sp = lambda t: t.view(B, T, H, hd).transpose(1, 2)
            a = F.scaled_dot_product_attention(sp(q), sp(k), sp(v), is_causal=True).transpose(1, 2).reshape(B, T, d)
            x = x + a @ ps[f"o{l}"].T
            h = rms(x, ps[f"ln2_{l}"]).reshape(-1, d); g = h @ ps[f"router{l}"].T
            gv, top = g.topk(K, -1); gv = gv.softmax(-1); y = torch.zeros_like(h)
            prob = g.softmax(-1); load = F.one_hot(top[:, 0], E).float().mean(0)
            aux = E * (prob.mean(0) * load).sum()
            for e in range(E):
                r, s = (top == e).nonzero(as_tuple=True)
                if len(r): y.index_add_(0, r, (F.silu(h[r] @ ps[f"w1_{l}"][e].T) @ ps[f"w2_{l}"][e].T) * gv[r, s, None])
            x = x + y.view(B, T, d); fwd.aux = fwd.aux + aux
        return rms(x, ps["lnf"]) @ ps["head"].T
    data = torch.tensor(list(text.encode("utf-8", "replace")), dtype=torch.long)
    opt = torch.optim.AdamW(ps.values(), lr=lr); hist = []; t0 = time.time()
    for s in range(steps):
        ix = torch.randint(0, len(data) - seq - 1, (batch,)); xb = torch.stack([data[i:i + seq] for i in ix]); yb = torch.stack([data[i + 1:i + seq + 1] for i in ix])
        fwd.aux = 0.0; lo = F.cross_entropy(fwd(xb).view(-1, VOCAB), yb.view(-1)); (lo + 0.01 * fwd.aux).backward(); opt.step(); opt.zero_grad()
        if s % 25 == 0 or s == steps - 1:
            hist.append((s, round(lo.item(), 4))); log(f"step {s} loss {lo.item():.4f} tok/s {(s+1)*batch*seq/(time.time()-t0):.0f}")
    dense = {k: v.detach().numpy() for k, v in ps.items() if not k.startswith("w")}
    ex = [[(ps[f"w1_{l}"][e].detach().numpy(), ps[f"w2_{l}"][e].detach().numpy()) for e in range(E)] for l in range(L)]
    c = create(out, cfg, dense=dense, experts=ex); c["train"] = dict(steps=steps, loss_history=hist, seconds=round(time.time() - t0, 1),
                                                                    train_tok_per_s=round(steps * batch * seq / (time.time() - t0), 1))
    json.dump(c, open(os.path.join(out, "config.json"), "w"), indent=1); return c
