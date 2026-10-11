#!/usr/bin/env python3
"""Tests for Expert Craft's Flash Experts (NumPy only).   python3 scripts/test_flash_experts.py [--big]

 1. gradients of the expert layer (inputs, router, expert weights) match finite differences, for both routers (hashed n-gram and product-key)
 2. the layer inside the transformer: backbone and router gradients still match finite differences
 3. the 8-bit flash store round-trips: one update then reload agrees with the float result to one quantisation step
 4. a new layer is an exact no-op (loss identical to the model without it) and training with it lowers the loss
 5. a project can be created, trained, stopped, resumed and sampled (the phone workflow)
 6. a 70-billion-parameter expert file is a sparse file: apparent size ~75 GB, disk use only for experts that were touched
 7. the native C kernels (compiled on this device) give the same outputs, gradients, updates and Adam steps as the NumPy reference
"""
import json, os, shutil, sys, tempfile, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'craft', 'skills', 'expert-craft', 'scripts', 'phone'))
try:
    import numpy as np
except ImportError:
    sys.exit('SKIP: needs numpy')
import builder as B, flashexperts as F, phonetrain as PT, native

ok = True
def check(name, cond, detail=''):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name + (('  ' + detail) if detail else ''), flush=True)

def fd(arr, idx, f, eps=1e-6):
    o = arr[idx]; arr[idx] = o + eps; a = f(); arr[idx] = o - eps; b = f(); arr[idx] = o; return (a - b) / (2 * eps)

# ---- 1. layer gradients
rng = np.random.default_rng(0); d, nk, H, K, kc, mh, n = 16, 8, 2, 4, 8, 3, 5
st = F.DenseStore(nk * nk, d, mh, dtype=np.float64); st.up[:] = rng.standard_normal(st.up.shape) * .3; st.dn[:] = rng.standard_normal(st.dn.shape) * .3; st.mat[:] = True
L = F.FlashExpertLayer('t', d, nk, st, heads=H, topk=K, kc=kc, chunk=7); L.update = False
P = {k: v.astype(np.float64) for k, v in L.init_params(rng).items()}; x = rng.standard_normal((n, d)); R = rng.standard_normal((n, d))
loss = lambda: float((L.forward(P, x) * R).sum())
L.forward(P, x, train=True); G = {k: np.zeros_like(v) for k, v in P.items()}; dx = L.backward(P, G, R.copy())
ids = np.concatenate([a for a, _, _ in L.grads]); gup = np.concatenate([b for _, b, _ in L.grads]); gdn = np.concatenate([c for _, _, c in L.grads]); worst = 0.0
rel = lambda a, b: abs(a - b) / (abs(a) + abs(b) + 1e-9)
for _ in range(25):
    i = tuple(rng.integers(0, s) for s in x.shape); worst = max(worst, rel(dx[i], fd(x, i, loss)))
for k in ('t.mq', 't.mk1', 't.mk2'):
    for _ in range(25):
        i = tuple(rng.integers(0, s) for s in P[k].shape); worst = max(worst, rel(G[k][i], fd(P[k], i, loss)))
for _ in range(40):
    u = rng.integers(0, len(ids)); m = rng.integers(0, mh); c = rng.integers(0, d)
    worst = max(worst, rel(gup[u, m, c], fd(st.up, (ids[u], m, c), loss)), rel(gdn[u, m, c], fd(st.dn, (ids[u], m, c), loss)))
check('1 expert-layer gradients match finite differences', worst < 1e-4, 'worst relative error %.1e' % worst)

# ---- 1b. hashed n-gram router
stH = F.DenseStore(50, d, mh, dtype=np.float64); stH.up[:] = rng.standard_normal(stH.up.shape) * .3; stH.dn[:] = rng.standard_normal(stH.dn.shape) * .3; stH.mat[:] = True
LH = F.FlashExpertLayer('h', d, 7, stH, route='hash', orders=(2, 3), vocab=40, chunk=5); LH.update = False
PH = {k: v.astype(np.float64) for k, v in LH.init_params(rng).items()}; PH['h.mg'] = rng.standard_normal(PH['h.mg'].shape) * .5; PH['h.mb'][:] = .3
idsH = rng.integers(0, 12, (2, 3)); xH = rng.standard_normal((6, d)); RH = rng.standard_normal((6, d)); lossH = lambda: float((LH.forward(PH, xH, ids=idsH) * RH).sum())
LH.forward(PH, xH, train=True, ids=idsH); GH = {k: np.zeros_like(v) for k, v in PH.items()}; dxH = LH.backward(PH, GH, RH.copy()); worst = 0.0
idsA = np.concatenate([a for a, _, _ in LH.grads]); gupH = np.concatenate([b for _, b, _ in LH.grads]); gdnH = np.concatenate([c for _, _, c in LH.grads])
for _ in range(20):
    i = tuple(rng.integers(0, sz) for sz in xH.shape); worst = max(worst, rel(dxH[i], fd(xH, i, lossH)))
for k in ('h.mg', 'h.mb'):
    for _ in range(10):
        i = tuple(rng.integers(0, sz) for sz in PH[k].shape); worst = max(worst, rel(GH[k][i], fd(PH[k], i, lossH)))
for _ in range(30):
    u = rng.integers(0, len(idsA)); m = rng.integers(0, mh); c = rng.integers(0, d)
    worst = max(worst, rel(gupH[u, m, c], fd(stH.up, (idsA[u], m, c), lossH)), rel(gdnH[u, m, c], fd(stH.dn, (idsA[u], m, c), lossH)))
check('1b hashed n-gram router gradients match finite differences', worst < 1e-4, 'worst relative error %.1e' % worst)
h1 = LH.hash_ids(np.array([[5, 6, 7], [9, 6, 7]]))
check('1c the same two-token context picks the same expert in different sequences', h1[2, 0] == h1[5, 0] and h1[2, 1] != h1[5, 1])

# ---- 2. inside the transformer
c = dict(dim=16, layers=2, heads=2, kv_heads=2, ffn=32, vocab=40, ctx=16, tied=True, rope_base=10000.0, eps=1e-5)
Pm = {k: v.astype(np.float64) for k, v in B.init_params(c, seed=3).items()}
st2 = F.DenseStore(64, 16, 3, dtype=np.float64); st2.up[:] = rng.standard_normal(st2.up.shape) * .3; st2.dn[:] = rng.standard_normal(st2.dn.shape) * .3; st2.mat[:] = True
L2 = F.FlashExpertLayer('1', 16, 8, st2, heads=2, topk=4, kc=8); L2.update = False; Pm.update({k: v.astype(np.float64) for k, v in L2.init_params(rng).items()}); mem = {1: L2}
ids_in = rng.integers(0, 40, (2, 8)); tg = rng.integers(0, 40, (2, 8)); f = lambda: B.forward(Pm, c, ids_in, tg, mem=mem)[1]
_, l0, stt = B.forward(Pm, c, ids_in, tg, cache=True, mem=mem); Gm = B.backward(Pm, c, stt); worst = 0.0
for k in ('tok', '0.q', '1.d', '1.mn', '1.mq', '1.mk1', '1.mk2', '1.g'):
    for _ in range(8):
        i = tuple(rng.integers(0, s) for s in Pm[k].shape); worst = max(worst, rel(Gm[k][i], fd(Pm[k], i, f)))
check('2 backbone + router gradients through the full model match finite differences', worst < 5e-4, 'worst relative error %.1e' % worst)

# ---- 3. 8-bit flash store round trip
tmp = tempfile.mkdtemp(prefix='fx_test_')
try:
    d3, mh3 = 64, F.hidden_per_page(64); fs = F.FlashStore(os.path.join(tmp, 'e.bin'), 256, d3, mh3, seed=1, create=True)
    ids3 = np.array([3, 7, 200]); ex = fs.load(ids3); check('3a never-used experts load as shared-pool up-projection and zero down-projection', ex.fresh.all() and not ex.dn.any() and ex.up.std() > 0)
    g = lambda s: rng.standard_normal((3, mh3, d3)).astype(np.float32) * s
    gu, gd = g(1e-3), g(1e-3); e0 = (fs.ema.copy(), fs.ema_n); w, acc = fs._step(ex, gu, gd, 0.05); fs.ema, fs.ema_n = e0; fs.apply(ex, gu, gd, 0.05, rng=np.random.default_rng(2)); back = fs.load(ids3)
    step = np.abs(w).max(-1, keepdims=True) / 127.0; err = np.abs(np.concatenate([back.up, back.dn], 1) - w)
    check('3b one update then reload agrees with the float result to one 8-bit step', (err <= step * 1.001 + 1e-9).all() and not back.fresh.any() and np.allclose(back.acc, acc),
          'max error %.2e, step %.2e' % (err.max(), step.max()))
    fs.close(); fs2 = F.FlashStore(os.path.join(tmp, 'e.bin'), 256, d3, mh3, seed=1); b2 = fs2.load(ids3); check('3c data survives closing and reopening the file', np.allclose(b2.up, back.up) and np.allclose(b2.dn, back.dn)); fs2.close()

    # ---- 4. no-op at start, then learning
    text = open(os.path.join(ROOT, 'README.md'), encoding='utf-8').read()
    tk = B.train_bpe(text, vocab_size=300); tok = B.Tok(tk); data = np.array(tok.encode(text), np.int64)
    c4 = dict(dim=64, layers=2, heads=2, kv_heads=2, ffn=128, vocab=len(tk['tokens']), ctx=64, tied=True, rope_base=10000.0, eps=1e-5)
    def run(use, steps, lre, route='pk'):
        P4 = B.init_params(c4, seed=5); mem4 = None
        if use:
            s4 = F.FlashStore(os.path.join(tmp, 'm%d.bin' % int(time.time() * 1000)), 64 * 64, 64, F.hidden_per_page(64), seed=2, create=True)
            l4 = F.FlashExpertLayer('1', 64, 64, s4, heads=1, topk=4, route=route, orders=(2, 3), vocab=c4['vocab']); l4.lr_e = lre; P4.update(l4.init_params(np.random.default_rng(9))); mem4 = {1: l4}
        opt = B.Adam(P4, 3e-3); r = np.random.default_rng(0); out = []
        for s in range(steps):
            x4, y4 = B.get_batch(data, 8, 32, r); _, l, stt = B.forward(P4, c4, x4, y4, cache=True, mem=mem4); G4 = B.backward(P4, c4, stt); opt.step(P4, G4, 3e-3); out.append(l)
        if use: s4.close()
        return out
    a = run(False, 8, 0)
    for route in ('pk', 'hash'):
        b = run(True, 8, 0.0, route)
        check('4a (%s) a new Flash Experts layer is an exact no-op (loss identical to the model without it)' % route, np.allclose(a, b, atol=1e-6), 'max difference %.1e' % np.abs(np.array(a) - np.array(b)).max())
    base = run(False, 150, 0); t = run(True, 150, 0.003, 'hash'); check('4b training with hashed Flash Experts lowers the loss and beats the same model without experts', np.mean(t[-10:]) < np.mean(t[:10]) - 0.5 and np.mean(t[-10:]) < np.mean(base[-10:]), '%.2f -> %.2f (without experts %.2f)' % (np.mean(t[:10]), np.mean(t[-10:]), np.mean(base[-10:])))

    # ---- 5. phone workflow
    proj = os.path.join(tmp, 'proj'); os.makedirs(proj); src = os.path.join(proj, 'text.txt'); open(src, 'w').write(text * 3)
    p, meta = PT.create(os.path.join(proj, 'm'), src, 5e8, 1.0, vocab=300, log=lambda *_: None)
    r1 = PT.train(os.path.join(proj, 'm'), steps=40, log=lambda *_: None); r2 = PT.train(os.path.join(proj, 'm'), steps=40, log=lambda *_: None)
    check('5a a project trains, is saved, and resumes where it stopped', r1['step'] == 40 and r2['step'] == 80 and r2['last_loss'] < r1['first_loss'], 'steps %d -> %d, loss %.2f -> %.2f' % (r1['step'], r2['step'], r1['first_loss'], r2['last_loss']))
    s = PT.sample(os.path.join(proj, 'm'), 'The ', n=20); check('5b the trained model generates text', isinstance(s, str) and len(s) > 4)
    stt = PT.status(os.path.join(proj, 'm')); check('5c status reports parameters and flash used', stt['total_params'] > 4e8 and stt['experts_disk_gb'] >= 0, '%.2f B parameters' % (stt['total_params'] / 1e9))
    # ---- 6. 70B sparse file
    free = shutil.disk_usage(tmp).free / 1e9
    if free < 3: print('SKIP 6 (needs 3 GB free)')
    else:
        lay = F.plan_experts(70e9, 192); d6 = 192; path = os.path.join(tmp, 'big.bin'); big = F.FlashStore(path, lay['experts_per_layer'], d6, lay['hidden'], seed=3, create=True)
        apparent = os.path.getsize(path) / 1e9; l6 = F.FlashExpertLayer('0', d6, lay['nk'], big, route='hash', orders=(2, 3), vocab=2048); P6 = l6.init_params(np.random.default_rng(1)); l6.lr_e = .003
        for _ in range(3):
            xx = np.random.default_rng(4).standard_normal((256, d6)).astype(np.float32); l6.forward(P6, xx, train=True, ids=np.random.default_rng(7).integers(0, 2048, (4, 64))); l6.backward(P6, {k: np.zeros_like(v) for k, v in P6.items()}, np.ones((256, d6), np.float32))
        used = big.materialized_bytes() / 1e9; touched = big.stats['pages_written'] * 4096 / 1e9
        check('6 a %.0fB-parameter expert file is sparse' % (lay['total_params'] / 1e9), 70 < apparent < 80 and used < 0.1 and used <= touched * 1.5 + 0.01,
              'apparent %.1f GB, on disk %.3f GB, %d experts created' % (apparent, used, big.stats['experts_created'])); big.close()
    # ---- 7. native kernels against the NumPy reference
    if native.lib() is None: print('SKIP 7 (%s)' % native.why())
    else:
        d7 = 64; mh7 = F.hidden_per_page(d7); r7 = np.random.default_rng(3)
        A7 = F.FlashStore(os.path.join(tmp, 'n1.bin'), 4096, d7, mh7, seed=5, create=True)
        L0 = F.FlashExpertLayer('t', d7, 64, A7, route='hash', orders=(1, 2, 3), vocab=50); L0.native = False; L0.lr_e = .01
        P7 = L0.init_params(r7); P7['t.mg'][:] = r7.standard_normal(P7['t.mg'].shape) * .3
        for _ in range(3):
            L0.forward(P7, r7.standard_normal((256, d7)).astype(np.float32), train=True, ids=r7.integers(0, 50, (8, 32)))
            L0.backward(P7, {k: np.zeros_like(v) for k, v in P7.items()}, r7.standard_normal((256, d7)).astype(np.float32))
        A7.flush(); shutil.copy(os.path.join(tmp, 'n1.bin'), os.path.join(tmp, 'n2.bin'))
        B7 = F.FlashStore(os.path.join(tmp, 'n2.bin'), 4096, d7, mh7, seed=5); B7.ema = A7.ema.copy(); B7.ema_n = A7.ema_n
        LN = F.FlashExpertLayer('t', d7, 64, A7, route='hash', orders=(1, 2, 3), vocab=50); LN.lr_e = .01
        LP = F.FlashExpertLayer('t', d7, 64, B7, route='hash', orders=(1, 2, 3), vocab=50); LP.native = False; LP.lr_e = .01
        ids7 = r7.integers(0, 60, (8, 32)); x7 = r7.standard_normal((256, d7)).astype(np.float32); dy7 = r7.standard_normal((256, d7)).astype(np.float32)
        LN.update = LP.update = False
        yN, yP = LN.forward(P7, x7, train=True, ids=ids7), LP.forward(P7, x7, train=True, ids=ids7)
        GN = {k: np.zeros_like(v) for k, v in P7.items()}; GP = {k: np.zeros_like(v) for k, v in P7.items()}; dN, dP = LN.backward(P7, GN, dy7), LP.backward(P7, GP, dy7)
        rr = lambda a, b: float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))
        gN = {e: (u, w) for e, u, w in zip(LN.grads[0][0].tolist(), LN.grads[0][1], LN.grads[0][2])}
        gP = [(e, u, w) for ids_, us, ws in LP.grads for e, u, w in zip(ids_.tolist(), us, ws)]
        eg = max(max(rr(gN[e][0], u), rr(gN[e][1], w)) for e, u, w in gP)
        check('7a native forward, input gradient and expert gradients equal the NumPy reference', rr(yN, yP) < 1e-5 and rr(dN, dP) < 1e-5 and eg < 1e-4 and len(gN) == len(gP),
              'y %.1e, dx %.1e, expert grads %.1e' % (rr(yN, yP), rr(dN, dP), eg))
        LN.update = LP.update = True; u7 = np.unique(LN.hash_ids(ids7))
        for L_ in (LN, LP): L_.forward(P7, x7, train=True, ids=ids7); L_.backward(P7, {k: np.zeros_like(v) for k, v in P7.items()}, dy7)
        a7, b7 = A7.load(u7, count=False), B7.load(u7, count=False); wa, wb = np.concatenate([a7.up, a7.dn], 1), np.concatenate([b7.up, b7.dn], 1)
        stp = np.abs(wb).max(-1, keepdims=True) / 127
        check('7b a native update equals the NumPy update to within 8-bit stochastic rounding', bool((np.abs(wa - wb) <= 2 * stp + 1e-7).all()) and rr(a7.acc, b7.acc) < 1e-5,
              'max difference %.2f quantisation steps' % float((np.abs(wa - wb) / (stp + 1e-12)).max())); A7.close(); B7.close()
        Pa = {'w': r7.standard_normal((300, 70)).astype(np.float32), 'b': r7.standard_normal(70).astype(np.float32)}; Pb = {k: v.copy() for k, v in Pa.items()}; oa, ob = B.Adam(Pa), B.Adam(Pb)
        for _ in range(5):
            Ga = {k: r7.standard_normal(v.shape).astype(np.float32) for k, v in Pa.items()}; oa.step(Pa, Ga, 1e-2)
            sv = B._native; B._native = lambda: None; ob.step(Pb, Ga, 1e-2); B._native = sv
        check('7c native Adam step equals NumPy Adam', max(float(np.abs(Pa[k] - Pb[k]).max()) for k in Pa) < 1e-5)
finally:
    shutil.rmtree(tmp, ignore_errors=True)
print('ALL PASSED' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
