"""SparseForge Flash engine (NumPy only, Termux/ARM friendly).

Model (byte-level LM):
  core (RAM, fp32, Adam):  emb[V,d] for last C bytes -> concat -> W1[C*d,H] -> +experts -> GELU -> Wout[H,V]
  experts (disk, np.memmap): N rows of H int8 weights + fp16 scale + fp16 momentum.
    Each position looks up K hashed contexts (last 1..K bytes, hashed with the layer salt) -> K expert rows,
    summed into the hidden pre-activation. Only touched rows are read/updated (sparse update).
  Expert optimizer: momentum-sign (Lion-style) update, requantized to int8 with stochastic rounding.
  Optional ternary (BitNet-style, straight-through) quantization of W1 in the forward pass.
Large matmuls go through NumPy BLAS (OpenBLAS/Accelerate use NEON/AVX). Set threads with SF_THREADS.
"""
import json, os, time
import numpy as np

V = 256

def _gelu(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

def _gelu_grad(x):
    t = np.tanh(0.7978845608 * (x + 0.044715 * x ** 3))
    return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * 0.7978845608 * (1 + 3 * 0.044715 * x * x)

def _ternary(w):
    s = np.abs(w).mean() + 1e-8
    return np.clip(np.round(w / s), -1, 1) * s

class FlashModel:
    def __init__(self, path):
        self.path = path
        self.cfg = json.load(open(os.path.join(path, "config.json")))
        c = self.cfg
        core = np.load(os.path.join(path, "core.npz"))
        self.p = {k: core[k].astype(np.float32) for k in ["emb", "W1", "b1", "Wout", "bout"]}
        self.m = {k: core["m_" + k].astype(np.float32) for k in self.p} if "m_emb" in core else {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: core["v_" + k].astype(np.float32) for k in self.p} if "v_emb" in core else {k: np.zeros_like(v) for k, v in self.p.items()}
        self.step = int(core["step"]) if "step" in core else 0
        N, H = c["n_experts"], c["hidden"]
        if N:
            self.xw = np.memmap(os.path.join(path, "experts_w.i8"), dtype=np.int8, mode="r+", shape=(N, H))
            self.xs = np.memmap(os.path.join(path, "experts_s.f16"), dtype=np.float16, mode="r+", shape=(N,))
            self.xm = np.memmap(os.path.join(path, "experts_m.f16"), dtype=np.float16, mode="r+", shape=(N, H))
        self.rng = np.random.default_rng(self.step + 1)

    # ----- routing -----
    def expert_ids(self, ctx):
        """ctx: [B, C] byte contexts -> [B, K] expert ids (hash of last k bytes, k=1..K)."""
        c = self.cfg; B = ctx.shape[0]; ids = np.empty((B, c["k_hash"]), np.int64)
        h = np.full(B, 1469598103934665603, dtype=np.uint64)
        with np.errstate(over="ignore"):
            for k in range(c["k_hash"]):
                h = (h ^ ctx[:, -1 - k].astype(np.uint64)) * np.uint64(1099511628211)
                ids[:, k] = ((h ^ np.uint64(k * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF)) % np.uint64(c["n_experts"])).astype(np.int64)
        return ids

    def _expert_vecs(self, ids):
        w = self.xw[ids.reshape(-1)].astype(np.float32) * self.xs[ids.reshape(-1)].astype(np.float32)[:, None]
        return w.reshape(ids.shape[0], ids.shape[1], -1).sum(1)

    # ----- forward/backward -----
    def forward(self, ctx, train=False):
        c, p = self.cfg, self.p
        B = ctx.shape[0]
        x = p["emb"][ctx].reshape(B, -1)                      # [B, C*d]
        W1 = _ternary(p["W1"]) if c.get("ternary") else p["W1"]
        pre = x @ W1 + p["b1"]
        ids = None
        if c["n_experts"]:
            ids = self.expert_ids(ctx); pre = pre + self._expert_vecs(ids)
        h = _gelu(pre)
        logits = h @ p["Wout"] + p["bout"]
        return logits, (x, pre, h, ids)

    def loss_and_grads(self, ctx, y):
        logits, (x, pre, h, ids) = self.forward(ctx, True)
        B = len(y); z = logits - logits.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1)); loss = float((lse - z[np.arange(B), y]).mean())
        g = np.exp(z - lse[:, None]); g[np.arange(B), y] -= 1; g /= B
        p = self.p; gr = {}
        gr["Wout"] = h.T @ g; gr["bout"] = g.sum(0)
        dh = (g @ p["Wout"].T) * _gelu_grad(pre)
        W1 = _ternary(p["W1"]) if self.cfg.get("ternary") else p["W1"]
        gr["W1"] = x.T @ dh; gr["b1"] = dh.sum(0)
        dx = (dh @ W1.T).reshape(B, self.cfg["ctx"], -1)
        ge = np.zeros_like(p["emb"]); np.add.at(ge, ctx, dx); gr["emb"] = ge
        return loss, gr, dh, ids

    def update(self, gr, dh, ids, lr, elr):
        self.step += 1; b1, b2 = 0.9, 0.99
        for k, g in gr.items():
            self.m[k] = b1 * self.m[k] + (1 - b1) * g; self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.step); vh = self.v[k] / (1 - b2 ** self.step)
            self.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if ids is not None:  # sparse expert update: only touched rows
            flat = ids.reshape(-1); gx = np.repeat(dh, ids.shape[1], axis=0)
            u, inv = np.unique(flat, return_inverse=True)
            G = np.zeros((len(u), dh.shape[1]), np.float32); np.add.at(G, inv, gx)
            m = self.xm[u].astype(np.float32)
            upd = np.sign(0.9 * m + 0.1 * G)            # Lion-style
            m = 0.99 * m + 0.01 * G
            w = self.xw[u].astype(np.float32) * self.xs[u].astype(np.float32)[:, None] - elr * upd
            s = np.maximum(np.abs(w).max(1), 1e-6) / 127.0
            q = w / s[:, None]; q = np.floor(q + self.rng.random(q.shape, dtype=np.float32))   # stochastic rounding
            self.xw[u] = np.clip(q, -127, 127).astype(np.int8); self.xs[u] = s.astype(np.float16)
            self.xm[u] = m.astype(np.float16)
            return len(u)
        return 0

    def save(self):
        d = {k: v for k, v in self.p.items()}
        d.update({"m_" + k: v.astype(np.float16) for k, v in self.m.items()})
        d.update({"v_" + k: v.astype(np.float16) for k, v in self.v.items()})
        tmp = os.path.join(self.path, "core.tmp.npz"); np.savez(tmp, step=self.step, **d)
        os.replace(tmp, os.path.join(self.path, "core.npz"))
        if self.cfg["n_experts"]:
            self.xw.flush(); self.xs.flush(); self.xm.flush()

    def param_counts(self):
        core = sum(v.size for v in self.p.values())
        ex = self.cfg["n_experts"] * self.cfg["hidden"]
        return {"core_params": int(core), "expert_params": int(ex), "total_params": int(core + ex)}

def create_model(path, ctx=8, d=32, hidden=256, n_experts=262144, k_hash=3, ternary=False, seed=0):
    os.makedirs(path, exist_ok=True)
    r = np.random.default_rng(seed)
    cfg = dict(ctx=ctx, d=d, hidden=hidden, n_experts=n_experts, k_hash=k_hash, ternary=ternary, vocab=V)
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"), indent=1)
    np.savez(os.path.join(path, "core.npz"),
             emb=(r.standard_normal((V, d)) * 0.1).astype(np.float32),
             W1=(r.standard_normal((ctx * d, hidden)) / np.sqrt(ctx * d)).astype(np.float32),
             b1=np.zeros(hidden, np.float32),
             Wout=(r.standard_normal((hidden, V)) / np.sqrt(hidden)).astype(np.float32),
             bout=np.zeros(V, np.float32))
    if n_experts:  # sparse files: unused experts use no disk blocks on most filesystems
        for name, nbytes in [("experts_w.i8", n_experts * hidden), ("experts_s.f16", n_experts * 2),
                             ("experts_m.f16", n_experts * hidden * 2)]:
            with open(os.path.join(path, name), "wb") as f:
                f.truncate(nbytes)
    m = FlashModel(path)
    return {"path": path, "config": cfg, **m.param_counts()}

def stream_batches(corpus, ctx, batch, chunk=1 << 20, seed=0):
    """Stream a file in chunks (no full load); random windows within the current chunk."""
    r = np.random.default_rng(seed); size = os.path.getsize(corpus)
    with open(corpus, "rb") as f:
        while True:
            off = int(r.integers(0, max(1, size - chunk)))
            f.seek(off); buf = np.frombuffer(f.read(chunk), np.uint8)
            if len(buf) <= ctx + 1:
                f.seek(0); buf = np.frombuffer(f.read(chunk), np.uint8)
            for _ in range(64):
                ix = r.integers(0, len(buf) - ctx - 1, batch)
                win = buf[ix[:, None] + np.arange(ctx + 1)]
                yield win[:, :ctx].astype(np.int64), win[:, ctx].astype(np.int64)

def train(path, corpus, steps=200, batch=256, lr=2e-3, expert_lr=2e-3, save_every=100, status_file=None,
          eval_corpus=None, log_every=10, stop_file=None):
    m = FlashModel(path); c = m.cfg
    it = stream_batches(corpus, c["ctx"], batch, seed=m.step)
    hist = []; t0 = time.perf_counter(); toks = 0; t_warm = None
    for s in range(steps):
        if stop_file and os.path.exists(stop_file):
            break
        if s == 2: t_warm = time.perf_counter(); toks = 0
        x, y = next(it)
        loss, gr, dh, ids = m.loss_and_grads(x, y)
        touched = m.update(gr, dh, ids, lr, expert_lr)
        toks += len(y)
        if s % log_every == 0 or s == steps - 1:
            el = time.perf_counter() - (t_warm or t0)
            st = dict(state="running", step=m.step, local_step=s + 1, steps=steps, loss=round(loss, 4),
                      tokens_per_s=round(toks / el, 1) if el > 0 else None, experts_touched=touched)
            hist.append(st)
            if status_file:
                json.dump(st, open(status_file + ".tmp", "w")); os.replace(status_file + ".tmp", status_file)
        if save_every and (s + 1) % save_every == 0:
            m.save()
    m.save()
    el = time.perf_counter() - (t_warm or t0)
    res = dict(state="done", steps_done=s + 1, total_step=m.step, final_loss=hist[-1]["loss"], first_loss=hist[0]["loss"],
               tokens_per_s=round(toks / el, 1), elapsed_s=round(el, 2), **m.param_counts())
    if eval_corpus:
        res["eval_loss"] = evaluate(path, eval_corpus)
    if status_file:
        json.dump(res, open(status_file + ".tmp", "w")); os.replace(status_file + ".tmp", status_file)
    return res

def evaluate(path, corpus, n=4096, seed=123):
    m = FlashModel(path); c = m.cfg
    x, y = next(stream_batches(corpus, c["ctx"], n, seed=seed))
    logits, _ = m.forward(x); z = logits - logits.max(1, keepdims=True)
    return round(float((np.log(np.exp(z).sum(1)) - z[np.arange(n), y]).mean()), 4)

def sample(path, prompt="", n=200, temperature=0.8, seed=0):
    m = FlashModel(path); c = m.cfg; r = np.random.default_rng(seed)
    toks = list(prompt.encode("utf-8", "ignore")) or [32]
    for _ in range(n):
        ctx = np.array([([32] * c["ctx"] + toks)[-c["ctx"]:]], np.int64)
        lg = m.forward(ctx)[0][0] / max(temperature, 1e-3); p = np.exp(lg - lg.max()); p /= p.sum()
        toks.append(int(r.choice(V, p=p)))
    return bytes(toks).decode("utf-8", "replace")
