"""ExpertStream Builder: create language models from scratch (1 thousand to 37 billion parameters).

 * design_config  - pick layers/width/heads to hit a parameter target (hard limits enforced here, not by prompt)
 * BPE tokenizer  - trained from your text, byte-level, same format the engine reads
 * Trainer        - real training (manual backprop, Adam) for models that fit a CPU: Llama-style transformer
 * grow_depth     - function-preserving growth: a trained small model becomes a deeper one that gives the same
                    answers, then training continues (the cheap path to large models)
 * write_gguf     - streams any size (up to 37B) to disk in F16/F32, loadable by the ExpertStream engine
Needs NumPy for training/growing (pkg install python-numpy). Design, estimates and writing big files need none.
"""
import json, math, os, re, struct, time

MIN_PARAMS, MAX_PARAMS = 1_000, 37_000_000_000
try:
    import numpy as np
except Exception:
    np = None

# ---------------------------------------------------------------- configuration
def count_params(c):
    D, F, L, V = c['dim'], c['ffn'], c['layers'], c['vocab']
    kv = c['kv_heads'] * (D // c['heads'])
    per = D * D + 2 * D * kv + D * D + 3 * D * F + 2 * D
    return V * D + L * per + D + (0 if c.get('tied', True) else V * D)

def design_config(target, vocab=4096, ctx=1024, tied=None):
    """Smallest-surprise Llama-style shape for a target size. Raises if outside 1e3..37e9."""
    if not (MIN_PARAMS <= target <= MAX_PARAMS):
        raise ValueError('target must be between %d and %d parameters (37 billion is the supported maximum)' % (MIN_PARAMS, MAX_PARAMS))
    tied = tied if tied is not None else target < 3e9
    best = None
    for D in sorted({int(x) for x in [16 * 2 ** (i / 4.0) for i in range(0, 80)]}):
        D = max(8, (D // 8) * 8)
        if D > 16384: break
        heads = max(1, D // 64) if D >= 64 else max(1, D // 8)
        while D % heads: heads -= 1
        for L in range(1, 129):
            ffn = int(round(D * 8 / 3 / 32)) * 32 or 32
            kvh = heads if D < 1024 else max(1, heads // 4 if heads % 4 == 0 else heads)
            c = dict(dim=D, layers=L, heads=heads, kv_heads=kvh, ffn=ffn, vocab=min(vocab, max(260, vocab if target > 1e6 else 260)), ctx=ctx, tied=tied, rope_base=10000.0, eps=1e-5)
            err = abs(count_params(c) - target) / target
            # prefer depth about proportional to width (L ~ D/64 .. D/96 for big models)
            aspect = abs(math.log((L + 1) / (D / 96.0 + 1)))
            score = err + 0.02 * aspect
            if best is None or score < best[0]: best = (score, c)
    c = best[1]; c['params'] = count_params(c)
    return c

def estimate(c, tokens, flops=2e11, ram_gb=4.0, disk_free_gb=100.0):
    n = count_params(c); train_flops = 6.0 * n * tokens
    bytes_f16, bytes_train = 2 * n, 16 * n   # weights+grads+Adam moments in fp32 ~ 16 B/param
    return dict(params=n, tokens=tokens, train_flops=train_flops, seconds=train_flops / flops, days=train_flops / flops / 86400,
                file_gb_f16=bytes_f16 / 1e9, train_state_gb=bytes_train / 1e9, fits_in_ram_to_train=bytes_train / 1e9 < ram_gb,
                fits_on_disk=bytes_f16 / 1e9 < disk_free_gb, chinchilla_tokens=20 * n,
                note='6*N*D FLOPs; flops = sustained device FLOP/s (phone CPU ~1e11, laptop CPU ~5e11, 1 GPU ~1e13+)')

# ---------------------------------------------------------------- tokenizer (byte-level BPE)
def _b2u():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256)); cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}
B2U = _b2u(); U2B = {v: k for k, v in B2U.items()}
SPLIT = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+")

def train_bpe(text, vocab_size=1024, specials=('<|endoftext|>',)):
    vocab_size = max(vocab_size, 256 + len(specials) + 1)
    words = {}
    for w in SPLIT.findall(text):
        k = tuple(B2U[b] for b in w.encode('utf-8')); words[k] = words.get(k, 0) + 1
    merges = []; vocab = [B2U[b] for b in range(256)]
    seqs = {k: list(k) for k in words}
    while len(vocab) + len(specials) < vocab_size:
        pc = {}
        for k, s in seqs.items():
            f = words[k]
            for a, b in zip(s, s[1:]): pc[(a, b)] = pc.get((a, b), 0) + f
        if not pc: break
        (a, b), n = max(pc.items(), key=lambda kv: (kv[1], kv[0]))
        if n < 2: break
        merges.append(a + ' ' + b); vocab.append(a + b)
        for k, s in seqs.items():
            i = 0; out = []
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b: out.append(a + b); i += 2
                else: out.append(s[i]); i += 1
            seqs[k] = out
    vocab += list(specials)
    return dict(tokens=vocab, merges=merges, specials=list(specials))

class Tok:
    def __init__(self, t):
        self.t = t; self.id = {s: i for i, s in enumerate(t['tokens'])}; self.rank = {tuple(m.split(' ')): i for i, m in enumerate(t['merges'])}; self.cache = {}
        self.eos = self.id[t['specials'][0]] if t['specials'] else -1
    def _bpe(self, w):
        if w in self.cache: return self.cache[w]
        s = list(w)
        while len(s) > 1:
            best = min(((self.rank.get((a, b), 1e18), i) for i, (a, b) in enumerate(zip(s, s[1:]))), default=None)
            if best is None or best[0] >= 1e18: break
            i = best[1]; s[i:i + 2] = [s[i] + s[i + 1]]
        self.cache[w] = s; return s
    def encode(self, text):
        out = []
        for w in SPLIT.findall(text):
            for p in self._bpe(''.join(B2U[b] for b in w.encode('utf-8'))): out.append(self.id[p])
        return out
    def decode(self, ids):
        s = ''.join(self.t['tokens'][i] for i in ids if i < len(self.t['tokens']) and self.t['tokens'][i] not in self.t['specials'])
        return bytes(U2B[c] for c in s if c in U2B).decode('utf-8', 'replace')

# ---------------------------------------------------------------- model (numpy)
def need_np():
    if np is None: raise RuntimeError('NumPy is required for training (Termux: pkg install python-numpy; PC: pip install numpy)')

def init_params(c, seed=1):
    need_np(); r = np.random.default_rng(seed); D, F, V, L = c['dim'], c['ffn'], c['vocab'], c['layers']
    kv = c['kv_heads'] * (D // c['heads']); s = 0.02; so = 0.02 / math.sqrt(2 * L)
    n = lambda *sh, sc=s: (r.standard_normal(sh) * sc).astype(np.float32)
    P = {'tok': n(V, D), 'norm': np.ones(D, np.float32)}
    if not c.get('tied', True): P['out'] = n(V, D)
    for l in range(L):
        P.update({f'{l}.an': np.ones(D, np.float32), f'{l}.q': n(D, D), f'{l}.k': n(kv, D), f'{l}.v': n(kv, D), f'{l}.o': n(D, D, sc=so),
                  f'{l}.fn': np.ones(D, np.float32), f'{l}.g': n(F, D), f'{l}.u': n(F, D), f'{l}.d': n(D, F, sc=so)})
    return P

def _rope_tab(T, hd, base):
    i = np.arange(hd // 2); th = np.arange(T)[:, None] * base ** (-2.0 * i / hd)
    return np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)
def _rope(x, cs, sn, inv=False):          # x: (B,H,T,hd) interleaved pairs, like the engine (non-neox)
    a, b = x[..., 0::2], x[..., 1::2]; c, s = cs[None, None], (-sn if inv else sn)[None, None]
    o = np.empty_like(x); o[..., 0::2] = a * c - b * s; o[..., 1::2] = a * s + b * c; return o
def _rms(x, g, eps):
    r = 1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps); return x * r * g, r
def _rms_back(dy, x, g, r):
    u = dy * g; D = x.shape[-1]
    dx = r * u - x * (r ** 3) * (u * x).sum(-1, keepdims=True) / D
    return dx, (dy * x * r).reshape(-1, D).sum(0)

def forward(P, c, ids, targets=None, cache=False):
    B, T = ids.shape; D, H, KH = c['dim'], c['heads'], c['kv_heads']; hd = D // H; rep = H // KH; eps = c['eps']
    cs, sn = _rope_tab(T, hd, c['rope_base']); x = P['tok'][ids]; C = []
    mask = np.triu(np.full((T, T), -1e30, np.float32), 1)
    for l in range(c['layers']):
        h, r1 = _rms(x, P[f'{l}.an'], eps)
        q = (h @ P[f'{l}.q'].T).reshape(B, T, H, hd).transpose(0, 2, 1, 3); k = (h @ P[f'{l}.k'].T).reshape(B, T, KH, hd).transpose(0, 2, 1, 3); v = (h @ P[f'{l}.v'].T).reshape(B, T, KH, hd).transpose(0, 2, 1, 3)
        q, k = _rope(q, cs, sn), _rope(k, cs, sn)
        kk, vv = (np.repeat(k, rep, 1), np.repeat(v, rep, 1)) if rep > 1 else (k, v)
        s = q @ kk.transpose(0, 1, 3, 2) / math.sqrt(hd) + mask; s -= s.max(-1, keepdims=True); p = np.exp(s); p /= p.sum(-1, keepdims=True)
        ctx = (p @ vv).transpose(0, 2, 1, 3).reshape(B, T, D); x1 = x + ctx @ P[f'{l}.o'].T
        h2, r2 = _rms(x1, P[f'{l}.fn'], eps); g = h2 @ P[f'{l}.g'].T; u = h2 @ P[f'{l}.u'].T; sg = 1 / (1 + np.exp(-g)); a = g * sg * u
        x2 = x1 + a @ P[f'{l}.d'].T
        if cache: C.append((x, h, r1, q, kk, vv, p, ctx, x1, h2, r2, g, u, sg, a))
        x = x2
    hf, rf = _rms(x, P['norm'], eps); W = P.get('out', P['tok']); logits = hf @ W.T
    if targets is None: return logits, None, None
    m = logits.max(-1, keepdims=True); e = np.exp(logits - m); pr = e / e.sum(-1, keepdims=True)
    loss = -np.log(pr[np.arange(B)[:, None], np.arange(T)[None], targets] + 1e-30).mean()
    return logits, float(loss), (C, x, hf, rf, pr, ids, targets, cs, sn)

def backward(P, c, st):
    C, xf, hf, rf, pr, ids, targets, cs, sn = st; B, T = ids.shape; D, H, KH = c['dim'], c['heads'], c['kv_heads']; hd = D // H; rep = H // KH; eps = c['eps']
    G = {k: np.zeros_like(v) for k, v in P.items()}
    dl = pr.copy(); dl[np.arange(B)[:, None], np.arange(T)[None], targets] -= 1; dl /= (B * T)
    W = P.get('out', P['tok']); gw = G['out'] if 'out' in P else G['tok']
    gw += dl.reshape(-1, dl.shape[-1]).T @ hf.reshape(-1, D); dhf = dl @ W
    dx, dg = _rms_back(dhf, xf, P['norm'], rf); G['norm'] += dg
    for l in reversed(range(c['layers'])):
        x, h, r1, q, kk, vv, p, ctx, x1, h2, r2, g, u, sg, a = C[l]
        da = dx @ P[f'{l}.d']; G[f'{l}.d'] += dx.reshape(-1, D).T @ a.reshape(-1, a.shape[-1])
        dg_ = da * u * (sg * (1 + g * (1 - sg))); du = da * g * sg
        G[f'{l}.g'] += dg_.reshape(-1, dg_.shape[-1]).T @ h2.reshape(-1, D); G[f'{l}.u'] += du.reshape(-1, du.shape[-1]).T @ h2.reshape(-1, D)
        dh2 = dg_ @ P[f'{l}.g'] + du @ P[f'{l}.u']; dx1, dgn = _rms_back(dh2, x1, P[f'{l}.fn'], r2); G[f'{l}.fn'] += dgn; dx1 = dx1 + dx
        dctx = dx1 @ P[f'{l}.o']; G[f'{l}.o'] += dx1.reshape(-1, D).T @ ctx.reshape(-1, D)
        dctx = dctx.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        dp = dctx @ vv.transpose(0, 1, 3, 2); dvv = p.transpose(0, 1, 3, 2) @ dctx
        ds = p * (dp - (dp * p).sum(-1, keepdims=True)) / math.sqrt(hd)
        dq = ds @ kk; dkk = ds.transpose(0, 1, 3, 2) @ q
        if rep > 1:
            dkk = dkk.reshape(B, KH, rep, T, hd).sum(2); dvv = dvv.reshape(B, KH, rep, T, hd).sum(2)
        dq, dkk = _rope(dq, cs, sn, True), _rope(dkk, cs, sn, True)
        mq = dq.transpose(0, 2, 1, 3).reshape(B, T, D); mk = dkk.transpose(0, 2, 1, 3).reshape(B, T, KH * hd); mv = dvv.transpose(0, 2, 1, 3).reshape(B, T, KH * hd)
        G[f'{l}.q'] += mq.reshape(-1, D).T @ h.reshape(-1, D); G[f'{l}.k'] += mk.reshape(-1, KH * hd).T @ h.reshape(-1, D); G[f'{l}.v'] += mv.reshape(-1, KH * hd).T @ h.reshape(-1, D)
        dh = mq @ P[f'{l}.q'] + mk @ P[f'{l}.k'] + mv @ P[f'{l}.v']; dxa, dga = _rms_back(dh, x, P[f'{l}.an'], r1); G[f'{l}.an'] += dga; dx = dxa + dx1
    np.add.at(G['tok'], ids.reshape(-1), dx.reshape(-1, D))
    return G

# ---------------------------------------------------------------- training
def get_batch(data, B, T, rng):
    ix = rng.integers(0, len(data) - T - 1, B); x = np.stack([data[i:i + T] for i in ix]); y = np.stack([data[i + 1:i + T + 1] for i in ix]); return x.astype(np.int64), y.astype(np.int64)

class Adam:
    def __init__(self, P, lr=3e-3, wd=0.01, b1=0.9, b2=0.95):
        self.m = {k: np.zeros_like(v) for k, v in P.items()}; self.v = {k: np.zeros_like(v) for k, v in P.items()}; self.t = 0; self.lr, self.wd, self.b1, self.b2 = lr, wd, b1, b2
    def step(self, P, G, lr, clip=1.0):
        gn = math.sqrt(sum(float((g * g).sum()) for g in G.values())); sc = min(1.0, clip / (gn + 1e-9)); self.t += 1
        for k in P:
            g = G[k] * sc; self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g; self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t); vh = self.v[k] / (1 - self.b2 ** self.t)
            P[k] -= lr * (mh / (np.sqrt(vh) + 1e-8) + (self.wd * P[k] if P[k].ndim == 2 else 0))
        return gn

def cosine_lr(step, total, peak, warm=50, floor=0.1):
    if step < warm: return peak * (step + 1) / warm
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(1, total - warm)))))

def evaluate(P, c, data, batches=8, B=8, T=None, seed=0):
    rng = np.random.default_rng(seed); T = T or min(c['ctx'], 128); ls = []
    for _ in range(batches):
        x, y = get_batch(data, B, T, rng); ls.append(forward(P, c, x, y)[1])
    l = float(np.mean(ls)); return dict(loss=l, perplexity=math.exp(min(l, 30)))

def train(P, c, data, steps, B=8, T=64, lr=3e-3, seed=0, log=None, ckpt=None, every=100, stop=lambda: False, opt=None):
    need_np(); rng = np.random.default_rng(seed); opt = opt or Adam(P, lr); t0 = time.time(); first = last = None
    for s in range(steps):
        if stop(): break
        x, y = get_batch(data, B, T, rng); _, loss, st = forward(P, c, x, y, cache=True); G = backward(P, c, st); gn = opt.step(P, G, cosine_lr(s, steps, lr))
        first = loss if first is None else first; last = loss
        if log and (s % 10 == 0 or s == steps - 1): log(dict(step=s + 1, steps=steps, loss=round(loss, 4), grad_norm=round(gn, 3), tokens=(s + 1) * B * T, seconds=round(time.time() - t0, 1)))
        if ckpt and (s + 1) % every == 0: save_ckpt(ckpt, P, c)
    if ckpt: save_ckpt(ckpt, P, c)
    return dict(first_loss=first, last_loss=last, seconds=time.time() - t0)

def save_ckpt(path, P, c, tok=None):
    np.savez(path, **P, __config__=np.frombuffer(json.dumps(dict(c, tok=tok) if tok else c).encode(), np.uint8))
def load_ckpt(path):
    z = np.load(path); c = json.loads(bytes(z['__config__']).decode()); return {k: z[k] for k in z.files if k != '__config__'}, c

def generate(P, c, tok, prompt, n=60, temp=0.8, seed=0):
    rng = np.random.default_rng(seed); ids = tok.encode(prompt) or [0]
    for _ in range(n):
        x = np.array([ids[-c['ctx']:]], np.int64); lg = forward(P, c, x)[0][0, -1]
        if temp <= 0: nxt = int(lg.argmax())
        else:
            p = np.exp((lg - lg.max()) / temp); p /= p.sum(); nxt = int(rng.choice(len(p), p=p))
        ids.append(nxt)
    return tok.decode(ids)

# ---------------------------------------------------------------- growth (function preserving)
def grow_depth(P, c, new_layers):
    """Insert identity layers (output projections zero) so logits are unchanged; training then makes them useful.
    Layers are duplicated from the existing ones (stacking), which is the cheap way to make a deeper model."""
    need_np(); L = c['layers']; assert new_layers > L
    nc = dict(c, layers=new_layers); Q = {k: v for k, v in P.items() if not re.match(r'^\d+\.', k)}
    src = [i % L for i in range(new_layers)] if False else None
    order = list(range(L)) + [L - 1 - (i % L) for i in range(new_layers - L)]   # old layers keep their place; copies of top layers appended
    for new_l, old_l in enumerate(order):
        for nm in ('an', 'q', 'k', 'v', 'o', 'fn', 'g', 'u', 'd'):
            w = P[f'{old_l}.{nm}'].copy()
            if new_l >= L and nm in ('o', 'd'): w[:] = 0       # identity at the start
            Q[f'{new_l}.{nm}'] = w
    return Q, nc

# ---------------------------------------------------------------- GGUF writer (streams, any size)
GT_F32, GT_F16 = 0, 1
def _s(b): return struct.pack('<Q', len(b)) + b
def _kv(k, t, v):
    k = _s(k.encode()); 
    if t == 'u32': return k + struct.pack('<II', 4, v)
    if t == 'f32': return k + struct.pack('<If', 6, v)
    if t == 'str': return k + struct.pack('<I', 8) + _s(v.encode())
    if t == 'arr_str': return k + struct.pack('<IIQ', 9, 8, len(v)) + b''.join(_s(x.encode()) for x in v)
    if t == 'arr_i32': return k + struct.pack('<IIQ', 9, 5, len(v)) + struct.pack('<%di' % len(v), *v)
def tensor_list(c):
    D, F, V, L = c['dim'], c['ffn'], c['vocab'], c['layers']; kv = c['kv_heads'] * (D // c['heads']); T = [('token_embd.weight', (V, D), 'tok', 2), ('output_norm.weight', (D,), 'norm', 1)]
    if not c.get('tied', True): T.append(('output.weight', (V, D), 'out', 2))
    for l in range(L):
        T += [(f'blk.{l}.attn_norm.weight', (D,), f'{l}.an', 1), (f'blk.{l}.attn_q.weight', (D, D), f'{l}.q', 2), (f'blk.{l}.attn_k.weight', (kv, D), f'{l}.k', 2), (f'blk.{l}.attn_v.weight', (kv, D), f'{l}.v', 2),
              (f'blk.{l}.attn_output.weight', (D, D), f'{l}.o', 2), (f'blk.{l}.ffn_norm.weight', (D,), f'{l}.fn', 1), (f'blk.{l}.ffn_gate.weight', (F, D), f'{l}.g', 2), (f'blk.{l}.ffn_up.weight', (F, D), f'{l}.u', 2), (f'blk.{l}.ffn_down.weight', (D, F), f'{l}.d', 2)]
    return T
def write_gguf(path, c, tok, P=None, dtype='f16', name='expertstream-model', seed=0, progress=None, sparse=False):
    """P=None writes a randomly initialised model (valid at any size, streamed so RAM stays small)."""
    if count_params(c) > MAX_PARAMS: raise ValueError('over the 37B limit')
    tl = tensor_list(c); es = 2 if dtype == 'f16' else 4; D = c['dim']
    meta = [('general.architecture', 'str', 'llama'), ('general.name', 'str', name), ('general.description', 'str', 'Created from scratch with ExpertStream Builder' + ('' if P is not None else ' (random initialisation, not trained)')),
            ('llama.context_length', 'u32', c['ctx']), ('llama.embedding_length', 'u32', D), ('llama.block_count', 'u32', c['layers']), ('llama.feed_forward_length', 'u32', c['ffn']),
            ('llama.attention.head_count', 'u32', c['heads']), ('llama.attention.head_count_kv', 'u32', c['kv_heads']), ('llama.attention.layer_norm_rms_epsilon', 'f32', c['eps']),
            ('llama.rope.freq_base', 'f32', c['rope_base']), ('llama.rope.dimension_count', 'u32', D // c['heads']), ('llama.vocab_size', 'u32', c['vocab']),
            ('tokenizer.ggml.model', 'str', 'gpt2'), ('tokenizer.ggml.pre', 'str', 'gpt-2'), ('tokenizer.ggml.tokens', 'arr_str', tok['tokens'] + ['<pad%d>' % i for i in range(c['vocab'] - len(tok['tokens']))]),
            ('tokenizer.ggml.token_type', 'arr_i32', [1] * 256 + [1] * (len(tok['tokens']) - 256 - len(tok['specials'])) + [3] * len(tok['specials']) + [1] * (c['vocab'] - len(tok['tokens']))),
            ('tokenizer.ggml.merges', 'arr_str', tok['merges']), ('tokenizer.ggml.eos_token_id', 'u32', len(tok['tokens']) - 1), ('tokenizer.ggml.add_bos_token', 'u32', 0)]
    if len(tok['tokens']) > c['vocab']: raise ValueError('tokenizer larger than model vocab')
    head = b'GGUF' + struct.pack('<IQQ', 3, len(tl), len(meta)) + b''.join(_kv(*m) for m in meta)
    infos, off = [], 0
    for n, sh, _, _ in tl:
        cnt = int(np.prod(sh)) if np else math.prod(sh); infos.append(_s(n.encode()) + struct.pack('<I', len(sh)) + b''.join(struct.pack('<Q', d) for d in reversed(sh)) + struct.pack('<IQ', GT_F16 if (dtype == 'f16' and len(sh) == 2) else GT_F32, off))
        sz = cnt * (es if len(sh) == 2 and dtype == 'f16' else 4); off += (sz + 31) // 32 * 32
    head += b''.join(infos); pad = (-len(head)) % 32; head += b'\0' * pad
    r = np.random.default_rng(seed) if np else None; written = 0; total = off
    with open(path, 'wb') as f:
        f.write(head)
        for n, sh, key, nd in tl:
            cnt = math.prod(sh); half = dtype == 'f16' and nd == 2; dt = (np.float16 if half else np.float32) if np else None
            if P is not None: f.write(P[key].astype(dt).tobytes() if True else b'')
            elif sparse and nd == 2: f.seek(cnt * (es if half else 4), 1)
            elif nd == 1: f.write(np.ones(cnt, np.float32).tobytes() if np else struct.pack('<f', 1.0) * cnt)
            else:
                sc = 0.02 / (math.sqrt(2 * c['layers']) if key.endswith(('.o', '.d')) else 1.0); left = cnt
                while left > 0:
                    k = min(left, 1 << 24)
                    if np: f.write((r.standard_normal(k) * sc).astype(dt).tobytes())
                    else: f.truncate(f.tell() + k * (2 if half else 4)); f.seek(0, 2)
                    left -= k
            sz = cnt * (es if half else 4); padn = (-sz) % 32
            if padn and not (sparse and P is None and nd == 2): f.write(b'\0' * padn)
            written += sz + padn
            if sparse and P is None and nd == 2: f.seek(padn, 1)
            if progress: progress(written / max(1, total))
        if sparse: f.truncate(len(head) + total)
    return os.path.getsize(path)
