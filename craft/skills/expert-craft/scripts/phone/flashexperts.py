"""Flash Experts: a layer with millions of page-sized experts that live in a file on flash and are trained in place.

Why this exists: a 70B-parameter model cannot be trained densely on a phone (6 x 70e9 x tokens FLOPs is decades of phone time),
but a SPARSE model can: if every word touches only a few percent of a percent of the parameters, the compute per word is that
of a small model, and only the touched parameters have to be read from and written back to storage.

 * An expert is a tiny MLP (a few hidden units) whose weights fill exactly one 4096-byte flash page (8-bit weights, per-row scales,
   per-row Adagrad state). One expert = one page read and one page write per use, no wasted bytes.
 * A router with PRODUCT KEYS (Lample et al. 2019, "Large Memory Layers with Product Keys"; the same idea scales to millions of experts in
   He 2024, "Mixture of A Million Experts") picks the top-K experts out of nk*nk with only 2*nk key comparisons per word.
 * Training updates only the experts that were used (row-wise Adagrad, stochastic rounding to 8 bit), like the huge embedding tables of
   recommender systems. Untouched experts cost nothing: the file is sparse and a never-used expert is created on first use from a
   shared random pool (up-projection) and zeros (down-projection), so the layer starts as an exact no-op and grows with training.
 * All of the arithmetic is NumPy, so it runs in Termux on a phone with `pkg install python-numpy`.
"""
import json, math, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

MAGIC = 0xE5E50001
POOL = 1 << 15
from layout import PAGE, hidden_per_page, expert_params, plan_experts   # noqa: F401  (pure-Python layout maths lives in layout.py)


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _dsilu(z):
    s = 1.0 / (1.0 + np.exp(-z)); return s * (1.0 + z * (1.0 - s))


def _topk(s, k):
    """per-row top-k of s (n,m): indices and values, sorted descending"""
    if k >= s.shape[1]: idx = np.argsort(-s, 1)[:, :k]
    else: idx = np.argpartition(s, s.shape[1] - k, axis=1)[:, -k:]
    v = np.take_along_axis(s, idx, 1); o = np.argsort(-v, 1)
    return np.take_along_axis(idx, o, 1), np.take_along_axis(v, o, 1)


def seg_add(dst, idx, vals):
    """dst[idx] += vals where idx may repeat (sort + reduceat, much faster than np.add.at); dst: (m, ...) vals: (len(idx), ...)"""
    if len(idx) == 0: return
    o = np.argsort(idx, kind='stable'); s = idx[o]; first = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    dst[s[first]] += np.add.reduceat(vals[o], first, axis=0)


# ----------------------------------------------------------------------------------------------------------- stores
class Ex:
    """experts loaded for one step: ids (u,), up/dn (u,mh,d) float, acc (u,2mh) (row-wise Adagrad state), fresh (u,) never-used flags"""
    __slots__ = ('ids', 'up', 'dn', 'acc', 'fresh')
    def __init__(self, ids, up, dn, acc, fresh): self.ids, self.up, self.dn, self.acc, self.fresh = ids, up, dn, acc, fresh


class _Base:
    """shared behaviour: virgin experts, Adagrad step"""
    def __init__(self, n_experts, d, mh, seed, dtype=np.float32):
        self.E, self.d, self.mh, self.seed, self.dtype = int(n_experts), int(d), int(mh), int(seed), dtype
        self.pool = (np.random.default_rng(seed + 7919).standard_normal((POOL, d)) / math.sqrt(d)).astype(dtype)
        self.stats = dict(pages_read=0, pages_written=0, loads=0, applies=0, experts_created=0)
        self.ema = np.zeros(2); self.ema_n = 0      # typical row gradient energy (up rows, down rows): start value for new rows' Adagrad state

    def _virgin_up(self, ids):
        # unit u of expert e takes row hash(e, u) of a shared random pool: cheap, deterministic, no storage
        u = np.arange(self.mh, dtype=np.uint64)[None, :]; e = ids.astype(np.uint64)[:, None]
        h = (e * np.uint64(0x9E3779B97F4A7C15) + u * np.uint64(0xC2B2AE3D27D4EB4F) + np.uint64(self.seed)) * np.uint64(0xFF51AFD7ED558CCD)
        h ^= h >> np.uint64(29)
        return self.pool[(h % np.uint64(POOL)).astype(np.int64)]

    def _step(self, ex, gup, gdn, lr):
        """row-wise Adagrad on every weight row of every used expert -> new float weights (u,2mh,d), new acc (u,2mh).
        A row that has never been updated starts its accumulator at the typical gradient energy, so the first step scales with the
        size of its gradient instead of always being a full-size step."""
        mh = self.mh; ge = np.empty((len(ex.ids), 2 * mh), self.dtype)
        np.einsum('umd,umd->um', gup, gup, out=ge[:, :mh]); np.einsum('umd,umd->um', gdn, gdn, out=ge[:, mh:]); ge /= self.d
        start = ge if self.ema_n == 0 else np.repeat(self.ema, mh).astype(self.dtype)[None, :]
        acc0 = np.where(ex.acc > 0, ex.acc, start)
        if len(ge): self.ema_update(ge[:, :mh].mean(), ge[:, mh:].mean())
        acc = acc0 + ge; inv = lr / (np.sqrt(acc) + 1e-30)
        w = np.empty((len(ex.ids), 2 * mh, self.d), self.dtype)
        np.subtract(ex.up, gup * inv[:, :mh, None], out=w[:, :mh]); np.subtract(ex.dn, gdn * inv[:, mh:, None], out=w[:, mh:])
        return w, acc

    def ema_update(self, up, dn):
        cur = np.array([up, dn]); self.ema = cur if self.ema_n == 0 else 0.98 * self.ema + 0.02 * cur; self.ema_n += 1

    def meta(self): return dict(ema=self.ema.tolist(), ema_n=self.ema_n, stats=self.stats)
    def set_meta(self, m): self.ema = np.array(m['ema']); self.ema_n = m['ema_n']; self.stats.update(m.get('stats', {}))


class DenseStore(_Base):
    """float32 experts in RAM: for tests, gradient checks and small experiments"""
    def __init__(self, n_experts, d, mh, seed=0, dtype=np.float32):
        super().__init__(n_experts, d, mh, seed, dtype)
        self.up = np.zeros((self.E, mh, d), dtype); self.dn = np.zeros((self.E, mh, d), dtype)
        self.acc = np.zeros((self.E, 2 * mh), dtype); self.mat = np.zeros(self.E, bool)

    def load(self, ids, count=True):
        up = self.up[ids].copy(); dn = self.dn[ids].copy(); v = ~self.mat[ids]
        if v.any(): up[v] = self._virgin_up(ids[v])
        if count: self.stats['loads'] += 1; self.stats['pages_read'] += len(ids)
        return Ex(ids, up, dn, self.acc[ids].copy(), v)

    def apply(self, ex, gup, gdn, lr):
        w, acc = self._step(ex, gup, gdn, lr); mh = self.mh
        self.up[ex.ids] = w[:, :mh]; self.dn[ex.ids] = w[:, mh:]; self.acc[ex.ids] = acc
        self.stats['experts_created'] += int(ex.fresh.sum()); self.mat[ex.ids] = True
        self.stats['applies'] += 1; self.stats['pages_written'] += len(ex.ids)

    def flush(self): pass
    def close(self): pass


class FlashStore(_Base):
    """the real thing: one 4096-byte page per expert in a sparse file, 8-bit weights with per-row scale, memory-mapped.
    layout of a page: magic u32 | pad u32 | scale f32[2mh] | acc f32[2mh] | int8 up[mh*d] | int8 down[mh*d] | zero padding"""
    def __init__(self, path, n_experts, d, mh, seed=0, create=False, threads=None, page=PAGE):
        super().__init__(n_experts, d, mh, seed)
        self.path, self.page = path, page
        need = 8 + 16 * mh + 2 * mh * d
        if need > page: raise ValueError('expert needs %d bytes but the page is %d' % (need, page))
        size = self.E * page
        if create or not os.path.exists(path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, 'wb') as f: f.truncate(size)
            # a file system without sparse-file support would allocate everything: detect it before the disk fills
            if size > (1 << 30) and os.stat(path).st_blocks * 512 > (1 << 28):
                os.remove(path); raise OSError("this folder does not support sparse files (SD card / FAT?). Use the phone's internal storage (Termux home).")
        elif os.path.getsize(path) != size: raise ValueError('%s has the wrong size for %d experts' % (path, self.E))
        self.mm = np.memmap(path, dtype=np.uint8, mode='r+', shape=(self.E, page))
        # experts are read at random: turn off the kernel's readahead, which would otherwise read (or zero-fill) 16-32 neighbouring pages
        # for every page we touch and multiply flash traffic by that much
        try:
            import mmap as _m
            self.mm._mmap.madvise(_m.MADV_RANDOM)
        except Exception:
            pass
        if threads is None:
            from layout import fast_cores
            threads = fast_cores()
        self.pool_threads = ThreadPoolExecutor(threads) if threads > 1 else None
        self.so, self.ao, self.po = 8, 8 + 8 * mh, 8 + 16 * mh      # byte offsets of scales, acc, payload

    # --- raw page IO (threaded so several flash reads are in flight at once)
    def _read(self, ids):
        if self.pool_threads is None or len(ids) < 512: return self.mm[ids]
        parts = np.array_split(ids, self.pool_threads._max_workers * 2)
        return np.concatenate(list(self.pool_threads.map(lambda p: self.mm[p], parts)))

    def _write(self, ids, raw):
        if self.pool_threads is None or len(ids) < 512: self.mm[ids] = raw; return
        parts = np.array_split(np.arange(len(ids)), self.pool_threads._max_workers * 2)
        list(self.pool_threads.map(lambda p: self.mm.__setitem__(ids[p], raw[p]), parts))

    def load(self, ids, count=True):
        mh, d = self.mh, self.d; raw = self._read(ids); n = len(ids)
        mat = raw[:, :4].copy().view('<u4')[:, 0] == MAGIC
        sc = np.ascontiguousarray(raw[:, self.so:self.so + 8 * mh]).view('<f4').reshape(n, 2 * mh)
        acc = np.ascontiguousarray(raw[:, self.ao:self.ao + 8 * mh]).view('<f4').reshape(n, 2 * mh).copy()
        q = raw[:, self.po:self.po + 2 * mh * d].view(np.int8).reshape(n, 2 * mh, d)
        w = q.astype(np.float32) * sc[..., None]; up, dn = w[:, :mh], w[:, mh:]
        v = ~mat
        if v.any(): up[v] = self._virgin_up(ids[v]); dn[v] = 0; acc[v] = 0
        if count: self.stats['loads'] += 1; self.stats['pages_read'] += n
        return Ex(ids, up, dn, acc, v)

    def apply(self, ex, gup, gdn, lr, rng=None):
        mh, d = self.mh, self.d; w, acc = self._step(ex, gup, gdn, lr); n = len(ex.ids)
        rng = rng or np.random.default_rng()
        sc = np.maximum(np.abs(w).max(-1), 1e-12) / 127.0
        w /= sc[..., None]; w += rng.random(w.shape, dtype=np.float32)                              # stochastic rounding: floor(x + U[0,1))
        np.floor(w, out=w); np.clip(w, -127, 127, out=w)
        raw = np.zeros((n, self.page), np.uint8)
        raw[:, :4] = np.frombuffer(np.uint32(MAGIC).tobytes(), np.uint8)
        raw[:, self.so:self.so + 8 * mh] = sc.astype('<f4').view(np.uint8).reshape(n, -1)
        raw[:, self.ao:self.ao + 8 * mh] = acc.astype('<f4').view(np.uint8).reshape(n, -1)
        raw[:, self.po:self.po + 2 * mh * d] = w.astype(np.int8).view(np.uint8).reshape(n, -1)
        self._write(ex.ids, raw)
        self.stats['experts_created'] += int(ex.fresh.sum()); self.stats['applies'] += 1; self.stats['pages_written'] += n

    def native_args(self):
        """base pointer, page size, d, mh, shared pool, pool rows, seed: what the C kernels need to read and write pages"""
        return (self.mm.ctypes.data, self.page, self.d, self.mh, self.pool.ctypes.data, len(self.pool), self.seed)

    def materialized_bytes(self):
        return os.stat(self.path).st_blocks * 512

    def flush(self): self.mm.flush()
    def close(self):
        self.mm.flush()
        if self.pool_threads: self.pool_threads.shutdown()


# ----------------------------------------------------------------------------------------------------------- the layer
def _mix64(h):
    """splitmix64 finaliser (vectorised, wraps silently)"""
    h = h ^ (h >> np.uint64(30)); h = h * np.uint64(0xBF58476D1CE4E5B9)
    h = h ^ (h >> np.uint64(27)); h = h * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


class FlashExpertLayer:
    """Residual sublayer that adds the output of a few experts per token. Two ways to choose the experts:

      route='pk'    product-key router (learned): top-K of nk*nk experts per head, softmax gates, mean over heads
      route='hash'  hashed n-gram router (no learning): the last n tokens, for each n in `orders`, pick one expert, so every occurrence
                    of the same context trains the same page; context-aware sigmoid gates decide how much each order contributes

    Router parameters (norm gain, query/keys or gate weights) are ordinary RAM parameters stored in P under `<prefix>.*`; the experts live
    in `store`. backward() trains them in place, a chunk of experts at a time, so memory stays small:
    load -> gradients -> Adagrad -> 8-bit stochastic rounding -> write back."""
    def __init__(self, prefix, d, nk, store, heads=1, topk=8, kc=32, chunk=1024, route='pk', orders=(2, 3), vocab=0):
        self.p, self.d, self.nk, self.H, self.K, self.kc, self.store, self.chunk = prefix, d, nk, heads, topk, kc, store, chunk
        self.route_kind, self.orders, self.vocab = route, tuple(orders), vocab
        self.mh = store.mh; self.last = None; self.lr_e = 0.05; self.update = True; self.grads = None; self.ctx_ids = None
        self.native = None          # None: use the C kernels when the store is a FlashStore and they compile; False: always NumPy
        self.M = len(self.orders) if route == 'hash' else heads * topk       # experts used per token

    def init_params(self, rng):
        d, H, kc, nk = self.d, self.H, self.kc, self.nk
        P = {f'{self.p}.mn': np.ones(d, np.float32)}
        if self.route_kind == 'hash':
            P[f'{self.p}.mg'] = (rng.standard_normal((len(self.orders), d)) * 0.02).astype(np.float32); P[f'{self.p}.mb'] = np.zeros(len(self.orders), np.float32)
        else:
            P[f'{self.p}.mq'] = (rng.standard_normal((H * 2 * kc, d)) * 0.5 / math.sqrt(d)).astype(np.float32)
            P[f'{self.p}.mk1'] = rng.standard_normal((H, nk, kc)).astype(np.float32); P[f'{self.p}.mk2'] = rng.standard_normal((H, nk, kc)).astype(np.float32)
        return P

    # --- product-key router: top-K experts per (token, head)
    def route(self, P, h):
        n, H, K, kc, nk = len(h), self.H, self.K, self.kc, self.nk
        q = (h @ P[f'{self.p}.mq'].T).reshape(n, H, 2, kc); K1, K2 = P[f'{self.p}.mk1'], P[f'{self.p}.mk2']
        I = np.empty((n, H, K), np.int64); J = np.empty((n, H, K), np.int64); S = np.empty((n, H, K), h.dtype)
        kk = min(K, nk)
        for hd in range(H):
            for a in range(0, n, 256):
                qa = q[a:a + 256, hd]; i1, v1 = _topk(qa[:, 0] @ K1[hd].T, kk); i2, v2 = _topk(qa[:, 1] @ K2[hd].T, kk)
                comb = (v1[:, :, None] + v2[:, None, :]).reshape(len(qa), -1); ci, cv = _topk(comb, K)
                I[a:a + 256, hd] = np.take_along_axis(i1, ci // kk, 1); J[a:a + 256, hd] = np.take_along_axis(i2, ci % kk, 1); S[a:a + 256, hd] = cv
        return q, I, J, S

    # --- hashed n-gram router: expert id of the context made of the last n tokens
    def hash_ids(self, ids):
        """ids: (B,T) token ids -> (B*T, len(orders)) expert ids; positions before the start of a sequence use a pad symbol"""
        ids = np.asarray(ids, np.int64); B, T = ids.shape; out = np.empty((B * T, len(self.orders)), np.int64); pad = np.int64(self.vocab)
        for oi, o in enumerate(self.orders):
            h = np.full((B, T), 0x51ED270B + o, np.uint64)
            for j in range(o):
                sh = np.full((B, T), pad, np.int64)
                if j == 0: sh = ids
                elif j < T: sh[:, j:] = ids[:, :T - j]
                h = _mix64(h * np.uint64(0x9E3779B97F4A7C15) + (sh.astype(np.uint64) + np.uint64(1)))
            out[:, oi] = (h % np.uint64(self.store.E)).astype(np.int64).reshape(-1)
        return out

    def _groups(self, ids):
        """sort the (token, slot) pairs by expert: returns order, unique experts, group starts, per-pair local expert index"""
        order = np.argsort(ids, kind='stable'); s = ids[order]
        first = np.flatnonzero(np.r_[True, s[1:] != s[:-1]]); uniq = s[first]
        inv = np.repeat(np.arange(len(uniq)), np.diff(np.r_[first, len(s)]))
        return order, uniq, first, inv

    def _chunks(self, uniq, first, npairs):
        for u0 in range(0, len(uniq), self.chunk):
            u1 = min(len(uniq), u0 + self.chunk); yield u0, u1, int(first[u0]), int(first[u1]) if u1 < len(uniq) else npairs

    # --- native (C) path: same maths, one pass per expert, threads over disjoint experts
    def _lib(self):
        if self.native is False or not isinstance(self.store, FlashStore): return None
        import native
        return native.lib()

    def _split(self, first, npairs):
        """contiguous group ranges with about the same number of pairs, one per thread"""
        T = self.store.pool_threads._max_workers if self.store.pool_threads else 1; G = len(first)
        cuts = np.unique(np.searchsorted(first, np.linspace(0, npairs, T + 1)[1:-1]))
        b = [0] + [int(c) for c in cuts if 0 < c < G] + [G]
        return [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] > b[i]]

    def _run(self, fn, jobs):
        pool = self.store.pool_threads
        return list(pool.map(fn, jobs)) if pool and len(jobs) > 1 else [fn(j) for j in jobs]

    def _native_forward(self, L, x, eid):
        n, M, d, mh = len(x), eid.shape[1], self.d, self.mh
        ids = eid.reshape(-1); order = np.argsort(ids, kind='stable'); s = np.ascontiguousarray(ids[order], np.int64)
        first = np.flatnonzero(np.r_[True, s[1:] != s[:-1]]); tok = np.ascontiguousarray(order // M, np.int64)
        x = np.ascontiguousarray(x, np.float32); P = len(ids); z = np.empty((P, mh), np.float32); o = np.empty((P, d), np.float32)
        gs = np.r_[first, P].astype(np.int64); A = self.store.native_args()
        def job(r):
            return L.fx_forward(*A, int(gs[r[0]]), int(gs[r[1]]), s.ctypes.data, tok.ctypes.data, x.ctypes.data, z.ctypes.data, o.ctypes.data)
        reads = sum(self._run(job, self._split(first, P)))
        st = self.store.stats; st['loads'] += 1; st['pages_read'] += int(reads)
        ou = np.empty_like(o); ou[order] = o
        return ou.reshape(n, M, d), ('native', order, s, gs, tok, z, x)

    def _native_backward(self, L, cache, do):
        _, order, s, gs, tok, z, x = cache; n, M, d = do.shape; mh = self.mh; P = len(order); G = len(gs) - 1
        dog = np.ascontiguousarray(do.reshape(-1, d)[order], np.float32); dxs = np.empty((P, d), np.float32)
        st = self.store; A = st.native_args(); em = st.ema if st.ema_n > 0 else np.array([-1.0, -1.0])
        seed = int(np.random.default_rng().integers(1, 2 ** 63)); upd = 1 if self.update else 0
        gout = np.empty((G, 2 * mh, d), np.float32) if not upd else None
        jobs = self._split(gs[:-1], P); stats = [np.zeros(3) for _ in jobs]
        def job(k):
            g0, g1 = jobs[k]; go = gout[g0:g1].ctypes.data if gout is not None else None
            return L.fx_backward(*A, g0, g1, gs.ctypes.data, s.ctypes.data, tok.ctypes.data, x.ctypes.data, z.ctypes.data, dog.ctypes.data, dxs.ctypes.data,
                                 float(self.lr_e), float(em[0]), float(em[1]), seed, upd, go, stats[k].ctypes.data)
        created = sum(self._run(job, list(range(len(jobs)))))
        tot = np.sum(stats, 0)
        if upd:
            if tot[2] > 0: st.ema_update(tot[0] / tot[2], tot[1] / tot[2])
            st.stats['experts_created'] += int(created); st.stats['applies'] += 1; st.stats['pages_written'] += G
        else:
            self.grads = [(s[gs[:-1]], np.ascontiguousarray(gout[:, :mh]), np.ascontiguousarray(gout[:, mh:]))]
        dxu = np.empty_like(dxs); dxu[order] = dxs
        return dxu.reshape(n, M, d).sum(1)

    # --- expert evaluation shared by both routers
    def _experts_forward(self, x, eid):
        """x (n,d), eid (n,M) expert ids -> outputs ou (n,M,d) in token order, plus what backward needs"""
        L = self._lib() if x.dtype == np.float32 else None
        if L is not None: return self._native_forward(L, x, eid)
        n, M, d, mh = len(x), eid.shape[1], self.d, self.mh
        ids = eid.reshape(-1); order, uniq, first, inv = self._groups(ids); tok = order // M
        dt = x.dtype; o = np.empty((len(ids), d), dt); z = np.empty((len(ids), mh), dt)
        for u0, u1, p0, p1 in self._chunks(uniq, first, len(ids)):
            ex = self.store.load(uniq[u0:u1]); same = (u1 - u0) == (p1 - p0)
            up = ex.up if same else ex.up[inv[p0:p1] - u0]; dn = ex.dn if same else ex.dn[inv[p0:p1] - u0]
            zz = np.matmul(up, x[tok[p0:p1]][:, :, None])[:, :, 0]; z[p0:p1] = zz; o[p0:p1] = np.matmul(_silu(zz)[:, None, :], dn)[:, 0]
        ou = np.empty_like(o); ou[order] = o
        return ou.reshape(n, M, d), (order, uniq, first, inv, tok, z, o)

    def _experts_backward(self, x, cache, do):
        """do (n,M,d): gradient w.r.t. each expert's output. Trains the experts in place; returns dx (n,d) through the experts"""
        if isinstance(cache[0], str): return self._native_backward(self._lib(), cache, do)
        order, uniq, first, inv, tok, z, o = cache; n, M, d = do.shape; dt = x.dtype
        dos = do.reshape(-1, d)[order]; dxs = np.empty((len(order), d), dt); self.grads = []
        for u0, u1, p0, p1 in self._chunks(uniq, first, len(order)):
            ex = self.store.load(uniq[u0:u1], count=False); same = (u1 - u0) == (p1 - p0)
            up = ex.up if same else ex.up[inv[p0:p1] - u0]; dn = ex.dn if same else ex.dn[inv[p0:p1] - u0]
            dg = dos[p0:p1]; zz = z[p0:p1]; aa = _silu(zz)
            dz = np.matmul(dn, dg[:, :, None])[:, :, 0] * _dsilu(zz)
            dxs[p0:p1] = np.matmul(dz[:, None, :], up)[:, 0]                                         # uses the OLD weights
            gdn = aa[:, :, None] * dg[:, None, :]; gup = dz[:, :, None] * x[tok[p0:p1]][:, None, :]
            if not same:
                st = first[u0:u1] - p0; gdn = np.add.reduceat(gdn, st, axis=0); gup = np.add.reduceat(gup, st, axis=0)
            if self.update: self.store.apply(ex, gup, gdn, self.lr_e)
            else: self.grads.append((ex.ids, gup, gdn))
        dxu = np.empty_like(dxs); dxu[order] = dxs
        return dxu.reshape(n, M, d).sum(1)

    def forward(self, P, x, train=False, ids=None):
        """x: (n,d) already normalised; ids: (B,T) token ids (needed by route='hash'). Returns y (n,d)."""
        n, d, H, K = len(x), self.d, self.H, self.K
        if self.route_kind == 'hash':
            eid = self.hash_ids(ids); ou, ec = self._experts_forward(x, eid)
            zg = x @ P[f'{self.p}.mg'].T + P[f'{self.p}.mb']; g = 1.0 / (1.0 + np.exp(-zg))                 # (n, orders) sigmoid gates
            y = (ou * g[:, :, None]).sum(1)
            if train: self.last = ('hash', ec, g, ou, x)
            return y
        q, I, J, S = self.route(P, x)
        g = np.exp(S - S.max(-1, keepdims=True)); g /= g.sum(-1, keepdims=True)                      # (n,H,K) softmax gates
        ou, ec = self._experts_forward(x, (I * self.nk + J).reshape(n, H * K))
        y = (ou * g.reshape(n, H * K, 1)).sum(1) / H
        if train: self.last = ('pk', ec, q, I, J, g, ou, x)
        return y

    def backward(self, P, G, dy):
        """returns dx (n,d); router gradients go into G; the experts are updated in place (or, if self.update is False, their
        gradients are kept in self.grads for tests)"""
        last = self.last; self.last = None; n, d = dy.shape
        if last[0] == 'hash':
            _, ec, g, ou, x = last
            dg = (ou * dy[:, None, :]).sum(-1); dz = dg * g * (1 - g)
            dx = self._experts_backward(x, ec, dy[:, None, :] * g[:, :, None])
            G[f'{self.p}.mg'] += dz.T @ x; G[f'{self.p}.mb'] += dz.sum(0); dx += dz @ P[f'{self.p}.mg']
            return dx
        _, ec, q, I, J, g, ou, x = last; H, K, kc = self.H, self.K, self.kc
        dyH = dy / H; dg = (ou.reshape(n, H, K, d) * dyH[:, None, None, :]).sum(-1)
        dx = self._experts_backward(x, ec, dyH[:, None, :] * g.reshape(n, H * K, 1))
        # router: softmax backward, then scatter into the product keys
        ds = g * (dg - (g * dg).sum(-1, keepdims=True))
        K1, K2 = P[f'{self.p}.mk1'], P[f'{self.p}.mk2']; dK1 = np.zeros_like(K1); dK2 = np.zeros_like(K2); dq = np.zeros_like(q)
        for hd in range(H):
            for part, (idx, Kt, dKt) in enumerate(((I[:, hd], K1[hd], dK1[hd]), (J[:, hd], K2[hd], dK2[hd]))):
                dsh = ds[:, hd]
                dq[:, hd, part] = (dsh[:, :, None] * Kt[idx]).sum(1)
                np.add.at(dKt, idx.reshape(-1), (dsh[:, :, None] * q[:, hd, part][:, None, :]).reshape(-1, kc))
        G[f'{self.p}.mk1'] += dK1; G[f'{self.p}.mk2'] += dK2
        dqf = dq.reshape(n, H * 2 * kc); G[f'{self.p}.mq'] += dqf.T @ x; dx += dqf @ P[f'{self.p}.mq']
        return dx

    def commit(self, lr=None): pass       # kept for callers of the earlier API: updates now happen inside backward()
