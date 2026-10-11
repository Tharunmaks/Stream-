"""Expert Craft on a phone: train a model with up to 70 billion parameters, for free, on the device.

How (see flashexperts.py): a small dense transformer lives in RAM and is trained normally. One or more layers get a huge pool of
page-sized experts that live in a SPARSE FILE on flash. Each word touches only a handful of experts; only those are read, trained
(row-wise Adagrad, 8-bit stochastic rounding) and written back. Compute per word is that of the small model; flash traffic per word
is (pages per word) x 4 KB each way. The file starts empty and grows as experts are first used.

Honest limits are part of the output of `plan`: how many tokens the device can process, how many times each expert will be used,
how much flash will be written. A 70B-parameter file is not a 70B-quality model: quality follows tokens seen and the dense backbone.
"""
import json, math, os, shutil, signal, sys, time, urllib.request
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import builder as B
import flashexperts as F

MAX_PARAMS = 70e9
ASSUMED_TBW = 500.0     # TB of lifetime writes assumed for a 256 GB phone flash; real rating is not published, used only to scale the wear estimate


def fmt(n):
    for u, s in ((1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'K')):
        if n >= u: return '%.2f%s' % (n / u, s)
    return str(int(n))


# ------------------------------------------------------------------------------------------------------------- device
def _read(path, default=None):
    try:
        with open(path) as f: return f.read().strip()
    except Exception: return default


def device_info(home):
    mem = {}
    for line in (_read('/proc/meminfo', '') or '').splitlines():
        k, _, v = line.partition(':'); mem[k] = int(v.split()[0]) * 1024 if v.split() else 0
    probe = home
    while not os.path.exists(probe) and probe != os.path.dirname(probe): probe = os.path.dirname(probe)
    du = shutil.disk_usage(probe)
    bat = '/sys/class/power_supply/battery'
    cap = _read(bat + '/capacity'); temp = _read(bat + '/temp')
    return dict(ram_gb=round(mem.get('MemTotal', 0) / 1e9, 1), ram_free_gb=round(mem.get('MemAvailable', 0) / 1e9, 1), cores=os.cpu_count() or 1,
                disk_free_gb=round(du.free / 1e9, 1), disk_total_gb=round(du.total / 1e9, 1),
                battery_pct=int(cap) if cap and cap.isdigit() else None, battery_status=_read(bat + '/status'),
                battery_temp_c=int(temp) / 10.0 if temp and temp.lstrip('-').isdigit() else None)


def power_ok(min_battery, max_temp):
    """(ok, reason). Unreadable sensors never block training."""
    i = device_info('/')
    if i['battery_temp_c'] is not None and max_temp and i['battery_temp_c'] >= max_temp: return False, 'phone is warm (%.1f C)' % i['battery_temp_c']
    if i['battery_pct'] is not None and min_battery and i['battery_pct'] < min_battery and (i['battery_status'] or '').lower() not in ('charging', 'full'):
        return False, 'battery %d%% and not charging' % i['battery_pct']
    return True, ''


# ------------------------------------------------------------------------------------------------------------- planning
def backbone_for(tokens, vocab, ctx=128, width=None):
    """the largest rung of the backbone ladder that still gets about 20 tokens per parameter (never below the smallest rung); width forces a rung"""
    rungs = B.ladder(vocab, ctx); best = rungs[0]
    if width:
        for c in rungs:
            if c['dim'] == int(width): return dict(c)
        raise ValueError('width must be one of %s' % [c['dim'] for c in rungs])
    for c in rungs:
        if c['params'] <= tokens / 20.0: best = c
    return dict(best)


def expert_layout(c, target_params, layers=1, topk=8, heads=1, free_gb=None, route='hash', orders=(1, 2, 3)):
    """experts per layer for a total-parameter target; shrinks the target to what the free disk can hold"""
    d = c['dim']; lay = F.plan_experts(max(target_params - c['params'], 1e6), d, layers)
    note = None
    if free_gb is not None and lay['file_bytes'] / 1e9 > free_gb * 0.85:
        fit = free_gb * 0.85 * 1e9 / F.PAGE * lay['params_per_expert']
        lay = F.plan_experts(fit, d, layers); note = 'free disk (%.0f GB) limits this phone to about %s parameters right now' % (free_gb, fmt(lay['total_params'] + c['params']))
    lay.update(layers=layers, topk=topk, heads=heads, kc=32, note=note, route=route, orders=list(orders), per_token=len(orders) if route == 'hash' else heads * topk)
    lay['total_with_backbone'] = lay['total_params'] + c['params']
    return lay


def io_bench(folder, pages=20000, size_mb=512):
    """random 4 KB reads/writes on the real storage (page cache bypassed as far as the OS allows): the number that decides speed and wear"""
    os.makedirs(folder, exist_ok=True); p = os.path.join(folder, '.iobench'); n = size_mb * 256
    try:
        with open(p, 'wb') as f:
            blk = os.urandom(1 << 20)
            for _ in range(size_mb): f.write(blk)
            f.flush(); os.fsync(f.fileno())
        try:
            fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
        except Exception: pass
        rng = np.random.default_rng(1); idx = rng.integers(0, n, pages); mm = np.memmap(p, dtype=np.uint8, mode='r+', shape=(n, 4096))
        t = time.time(); x = mm[np.sort(idx)]; r = time.time() - t
        ids = np.sort(rng.integers(0, n, pages)); t = time.time(); mm[ids] = x; mm.flush(); w = time.time() - t
        del mm
        return dict(read_mb_s=round(pages * 4096 / 1e6 / max(r, 1e-6), 0), write_mb_s=round(pages * 4096 / 1e6 / max(w, 1e-6), 0))
    finally:
        try: os.remove(p)
        except OSError: pass


def bench(c, lay, seconds=4.0, folder='.'):
    """tokens/second of the REAL trainer (dense backbone + expert layer, small file) on this device"""
    d = c['dim']; mh = F.hidden_per_page(d); nk = 128; path = os.path.join(folder, '.bench_experts.bin')
    st = F.FlashStore(path, nk * nk, d, mh, create=True)
    try:
        P = B.init_params(c); L = F.FlashExpertLayer(str(c['layers'] // 2), d, nk, st, heads=lay['heads'], topk=lay['topk'], route=lay['route'], orders=tuple(lay['orders']), vocab=c['vocab'])
        P.update(L.init_params(np.random.default_rng(0))); mem = {c['layers'] // 2: L}; opt = B.Adam(P, 1e-3)
        data = np.random.default_rng(0).integers(0, c['vocab'], 20000); rng = np.random.default_rng(0); Bt, T = 16, 64
        x, y = B.get_batch(data, Bt, T, rng); _, _, s = B.forward(P, c, x, y, cache=True, mem=mem); B.backward(P, c, s)   # warm up (compiles the kernels once)
        n = 0; t0 = time.time(); k = 0
        while time.time() - t0 < seconds or k < 3:
            x, y = B.get_batch(data, Bt, T, rng); _, _, s = B.forward(P, c, x, y, cache=True, mem=mem); G = B.backward(P, c, s); opt.step(P, G, 1e-3); k += 1; n += Bt * T
        return n / (time.time() - t0)
    finally:
        st.close()
        try: os.remove(path)
        except OSError: pass


def plan(target_params, minutes=600.0, home='.', vocab=2048, layers=1, topk=8, heads=1, run_bench=True, io=True, oversubscribe=False, route='hash', orders=(1, 2, 3), width=None):
    target_params = float(target_params)
    if not (1e6 <= target_params <= MAX_PARAMS): raise ValueError('target must be between 1M and 70B parameters')
    dev = device_info(home); out = dict(device=dev, target_params=target_params, minutes=minutes)
    free = None if oversubscribe else dev['disk_free_gb']
    rungs = B.ladder(vocab, 128)
    if width or not run_bench:
        c = backbone_for(400.0 * dev['cores'] / 4 * minutes * 60, vocab, width=width); lay = expert_layout(c, target_params, layers, topk, heads, free, route, orders)
        tps = bench(c, lay, folder=home) if run_bench else 400.0 * dev['cores'] / 4
    else:
        # speed depends on the dense width: time per token ~ a + b * params. Measure two widths, then pick the largest width that still gets
        # about 20 tokens per parameter in the time given (compute-optimal), and measure that one for the final numbers
        lo, hi = rungs[0], rungs[3]
        t_lo = 1.0 / bench(lo, expert_layout(lo, target_params, layers, topk, heads, free, route, orders), folder=home)
        t_hi = 1.0 / bench(hi, expert_layout(hi, target_params, layers, topk, heads, free, route, orders), folder=home)
        b_ = max((t_hi - t_lo) / (hi['params'] - lo['params']), 1e-15); a_ = max(t_lo - b_ * lo['params'], 0.0)
        c = rungs[0]
        for r in rungs:
            if minutes * 60 / (a_ + b_ * r['params']) >= 20 * r['params']: c = r
        c = dict(c); lay = expert_layout(c, target_params, layers, topk, heads, free, route, orders)
        tps = 1.0 / t_lo if c['dim'] == lo['dim'] else (1.0 / t_hi if c['dim'] == hi['dim'] else bench(c, lay, folder=home))
    tokens = tps * minutes * 60
    pages_tok = layers * lay['per_token']; bytes_tok = pages_tok * F.PAGE
    active = c['params'] + pages_tok * lay['params_per_expert']
    day_tb = tps * 86400 * bytes_tok / 1e12
    io_res = io_bench(home) if io else None
    out.update(backbone=dict(config=c, params=c['params']), experts=lay, tokens_per_second=round(tps, 0), tokens_affordable=int(tokens),
               tokens_per_backbone_param=round(tokens / c['params'], 1),
               total_params=lay['total_with_backbone'], active_params_per_token=active, active_fraction=active / lay['total_with_backbone'],
               file_gb=round(lay['file_bytes'] / 1e9, 1), disk_free_gb=dev['disk_free_gb'],
               pages_per_token=pages_tok, flash_kb_per_token_each_way=bytes_tok / 1024.0,
               visits_per_expert=round(tokens * lay['per_token'] / lay['experts_per_layer'], 2),
               flash_written_gb=round(tokens * bytes_tok / 1e9, 1), wear_tb_per_day=round(day_tb, 2), wear_percent_of_assumed_life_per_day=round(100 * day_tb / ASSUMED_TBW, 2), io=io_res)
    if io_res:
        need = tps * bytes_tok / 1e6
        out['flash_mb_s_needed'] = round(need, 1)
        out['flash_limits_speed'] = need > 0.5 * min(io_res['read_mb_s'], io_res['write_mb_s'])
    out['honest'] = honest(out)
    return out


def honest(p):
    return [
        '%s parameters are trained on this device: %s dense in RAM and %s in %s experts on flash. Each word uses about %s of them (%.4f%%).' % (
            fmt(p['total_params']), fmt(p['backbone']['params']), fmt(p['experts']['total_params']), fmt(p['experts']['experts_per_layer'] * p['experts']['layers']),
            fmt(p['active_params_per_token']), 100 * p['active_fraction']),
        'In %.0f minutes this phone processes about %s tokens (%s per second). Each expert is used about %.1f times in that time.' % (
            p['minutes'], fmt(p['tokens_affordable']), fmt(p['tokens_per_second']), p['visits_per_expert']),
        'Quality follows the tokens seen and the dense backbone, not the file size: expect a small-model level of text. More minutes and more text help; the huge expert pool adds capacity that fills up as training continues.',
        'The file grows as experts are first used, up to %.0f GB. Training rewrites up to %.0f GB of pages per hour (%.1f TB per day of nonstop training, %.2f%% of the life of a flash rated for %d TB). That is an upper bound: the system merges repeated writes to the same page, and phone-status shows the real amount written (on a Linux test PC it was about a third). Use --max-wear-gb to cap it.' % (
            p['file_gb'], p['tokens_per_second'] * 3600 * p['flash_kb_per_token_each_way'] * 1024 / 1e9, p['wear_tb_per_day'], p['wear_percent_of_assumed_life_per_day'], ASSUMED_TBW),
    ]


# ------------------------------------------------------------------------------------------------------------- project
def _paths(proj):
    j = lambda n: os.path.join(proj, n)
    return dict(meta=j('phone.json'), tok=j('tok.json'), ids=j('ids.npy'), model=j('model.npz'), log=j('log.jsonl'), experts=j('experts_%d.bin'))


def read_text(src):
    if os.path.isdir(src):
        parts = []
        for root, _, files in os.walk(src):
            for fn in sorted(files):
                if fn.endswith(('.txt', '.md', '.py', '.c', '.h', '.js', '.json', '.html', '.css', '.csv')) and not fn.startswith('.'):
                    try: parts.append(open(os.path.join(root, fn), encoding='utf-8', errors='replace').read())
                    except OSError: pass
        return '\n<|endoftext|>\n'.join(parts)
    return open(src, encoding='utf-8', errors='replace').read()


DATASETS = {'tinystories': 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt'}


def fetch(url, dest):
    if not url.startswith('https://'): raise ValueError('only https:// downloads are allowed')
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, 'wb') as f: shutil.copyfileobj(r, f)


def create(proj, data, target_params, minutes, vocab=2048, layers=1, topk=8, heads=1, lr=3e-3, lr_e=0.003, batch=16, seq=64, seed=1, log=print, oversubscribe=False, route='hash', orders=(1, 2, 3), width=None):
    """prepare data + tokenizer, choose the architecture for this device, make an empty expert file"""
    os.makedirs(proj, exist_ok=True); pth = _paths(proj)
    if os.path.exists(pth['meta']): raise ValueError('%s already exists; use phone-train to continue it' % proj)
    if data in DATASETS or (isinstance(data, str) and data.startswith('https://')):
        dst = os.path.join(proj, 'data.txt'); log('downloading %s ...' % data); fetch(DATASETS.get(data, data), dst); data = dst
    text = read_text(data)
    if len(text) < 20000: raise ValueError('need at least 20,000 characters of text to train on')
    log('training the tokenizer on %d characters ...' % min(len(text), 1_500_000))
    tk = B.train_bpe(text[:1_500_000], vocab_size=vocab); tok = B.Tok(tk)
    log('tokenizing ...'); ids = []
    for a in range(0, len(text), 500_000): ids += tok.encode(text[a:a + 500_000])
    ids = np.array(ids, np.int32); np.save(pth['ids'], ids); json.dump(tk, open(pth['tok'], 'w'))
    p = plan(target_params, minutes, proj, vocab=len(tk['tokens']), layers=layers, topk=topk, heads=heads, io=False, oversubscribe=oversubscribe, route=route, orders=orders, width=width)
    c = p['backbone']['config']; lay = p['experts']
    meta = dict(version=2, created=time.strftime('%Y-%m-%d %H:%M'), config=c, experts=dict(lay, layer_ids=[c['layers'] // 2 + i for i in range(layers)]),
                target_params=target_params, tokens=int(len(ids)), batch=batch, seq=seq, lr=lr, lr_e=lr_e, seed=seed, step=0,
                horizon=max(200, int(p['tokens_affordable'] / (batch * seq))), plan=dict(tokens_per_second=p['tokens_per_second'], total_params=p['total_params']))
    json.dump(meta, open(pth['meta'], 'w'), indent=1)
    log('architecture: dense %s + %s experts per layer on flash = %s parameters' % (fmt(c['params']), fmt(lay['experts_per_layer']), fmt(p['total_params'])))
    return p, meta


def _open(proj, meta, create_store):
    c = meta['config']; ex = meta['experts']; pth = _paths(proj); stores, mem = [], {}
    P = B.init_params(c, seed=meta['seed'])
    for i, l in enumerate(ex['layer_ids']):
        if not create_store and not os.path.exists(pth['experts'] % l): raise ValueError('expert file %s is missing' % (pth['experts'] % l))
        st = F.FlashStore(pth['experts'] % l, ex['nk'] ** 2, c['dim'], ex['hidden'], seed=meta['seed'] + l)
        L = F.FlashExpertLayer(str(l), c['dim'], ex['nk'], st, heads=ex['heads'], topk=ex['topk'], kc=ex['kc'], route=ex['route'], orders=tuple(ex['orders']), vocab=c['vocab'])
        P.update(L.init_params(np.random.default_rng(meta['seed'] + l))); mem[l] = L; stores.append(st)
    return c, P, mem, stores


def _save(proj, meta, P, opt, stores):
    for s in stores: s.flush()
    pth = _paths(proj); arrs = {'P/' + k: v for k, v in P.items()}
    arrs.update({'m/' + k: v for k, v in opt.m.items()}); arrs.update({'v/' + k: v for k, v in opt.v.items()})
    arrs['adam_t'] = np.array(opt.t); arrs['store_meta'] = np.frombuffer(json.dumps([s.meta() for s in stores]).encode(), np.uint8)
    tmp = pth['model'] + '.tmp.npz'; np.savez(tmp, **arrs); os.replace(tmp, pth['model'])
    json.dump(meta, open(pth['meta'] + '.tmp', 'w'), indent=1); os.replace(pth['meta'] + '.tmp', pth['meta'])


def _load_state(proj, P, opt, stores):
    pth = _paths(proj)
    if not os.path.exists(pth['model']): return False
    z = np.load(pth['model'])
    for k in P: P[k] = z['P/' + k].copy()
    for k in opt.m: opt.m[k] = z['m/' + k].copy(); opt.v[k] = z['v/' + k].copy()
    opt.t = int(z['adam_t'])
    for s, m in zip(stores, json.loads(bytes(z['store_meta']).decode())): s.set_meta(m)
    return True


def io_written():
    """bytes this process really sent to storage (Linux/Android /proc/self/io); repeated writes to a page before the OS flushes it count once"""
    try:
        for line in open('/proc/self/io'):
            if line.startswith('write_bytes:'): return int(line.split()[1])
    except Exception:
        pass
    return None


def train(proj, minutes=None, steps=None, min_battery=0, max_temp=0, max_wear_gb=None, save_every=120.0, log=print, stop=lambda: False, min_free_gb=None):
    pth = _paths(proj); meta = json.load(open(pth['meta'])); ids = np.load(pth['ids']).astype(np.int64)
    nv = max(20000, len(ids) // 50); tr, va = ids[:-nv], ids[-nv:]
    c, P, mem, stores = _open(proj, meta, True); opt = B.Adam(P, meta['lr']); resumed = _load_state(proj, P, opt, stores)
    for L in mem.values(): L.update = True
    if steps: meta['horizon'] = int(meta['step'] + steps)
    horizon = meta['horizon']; Bt, T = meta['batch'], meta['seq']; step = meta['step']
    reserve = min_free_gb if min_free_gb is not None else max(1.0, 0.02 * shutil.disk_usage(proj).total / 1e9)
    t0 = time.time(); t_save = t0; stopped = 'finished'; tok0 = step * Bt * T; first_loss = last_loss = None
    io0 = io_written(); pw0 = sum(s.stats['pages_written'] for s in stores); before = meta.get('flash_written_gb', 0.0)
    def wear_gb():     # all sessions: real bytes written to storage when the system reports it, else every page write counted (an upper bound)
        io = io_written()
        now = (io - io0) / 1e9 if io is not None and io0 is not None else (sum(s.stats['pages_written'] for s in stores) - pw0) * F.PAGE / 1e9
        meta['flash_written_gb'] = round(before + now, 3)
        return before + now
    log('%s: %s parameters (%s on flash). step %d of %d%s' % (os.path.basename(proj), fmt(meta['plan']['total_params']), fmt(meta['experts']['total_params']), step, horizon, ' (resumed)' if resumed else ''))
    def val():
        r = np.random.default_rng(5); ls = []
        for _ in range(8):
            x, y = B.get_batch(va, 16, min(T, 64), r); ls.append(B.forward(P, c, x, y, mem=mem)[1])
        return float(np.mean(ls))
    try:
        while step < horizon:
            if stop(): stopped = 'stopped'; break
            if minutes and time.time() - t0 >= minutes * 60: stopped = 'time budget reached'; break
            free_gb = shutil.disk_usage(proj).free / 1e9      # a full disk kills a memory-mapped writer (SIGBUS), so stop while there is still room
            if free_gb < reserve: stopped = 'disk almost full (%.1f GB free); stopped safely, free some space and run again' % free_gb; break
            w = wear_gb()
            if max_wear_gb and w >= max_wear_gb: stopped = 'flash write budget reached (%.1f GB)' % w; break
            ok, why = power_ok(min_battery, max_temp)
            if not ok:
                log('paused: %s' % why)
                while not ok and not stop(): time.sleep(30); ok, why = power_ok(min_battery, max_temp)
                continue
            rng = np.random.default_rng(meta['seed'] * 1000003 + step); x, y = B.get_batch(tr, Bt, T, rng)
            f = B.cosine_lr(step, horizon, 1.0, warm=50, floor=0.1)
            for L in mem.values(): L.lr_e = meta['lr_e'] * f
            _, loss, st = B.forward(P, c, x, y, cache=True, mem=mem); G = B.backward(P, c, st); opt.step(P, G, meta['lr'] * f)
            step += 1; meta['step'] = step; first_loss = loss if first_loss is None else first_loss; last_loss = loss
            if step % 20 == 0 or step == horizon:
                el = time.time() - t0; pr = sum(s.stats['pages_read'] for s in stores); pw = sum(s.stats['pages_written'] for s in stores)
                rec = dict(step=step, loss=round(loss, 4), tokens=step * Bt * T, tok_s=round((step * Bt * T - tok0) / max(el, 1e-9), 0), pages_read=pr, pages_written=pw,
                           experts_used=sum(s.stats['experts_created'] for s in stores), page_writes_gb=round(pw * F.PAGE / 1e9, 2), flash_written_gb=round(wear_gb(), 2), disk_gb=round(sum(s.materialized_bytes() for s in stores) / 1e9, 2))
                if step % 100 == 0 or step == horizon: rec['val_loss'] = round(val(), 4)
                open(pth['log'], 'a').write(json.dumps(rec) + '\n'); log(json.dumps(rec))
            if time.time() - t_save > save_every: _save(proj, meta, P, opt, stores); t_save = time.time()
    except KeyboardInterrupt:
        stopped = 'interrupted'
    wear_gb(); _save(proj, meta, P, opt, stores)
    res = dict(status=stopped, step=step, horizon=horizon, first_loss=first_loss, last_loss=last_loss, val_loss=round(val(), 4), seconds=round(time.time() - t0, 1),
               flash_written_gb=round(wear_gb(), 2), page_writes_gb=round(sum(s.stats['pages_written'] for s in stores) * F.PAGE / 1e9, 2), disk_gb=round(sum(s.materialized_bytes() for s in stores) / 1e9, 2),
               experts_used=sum(s.stats['experts_created'] for s in stores))
    for s in stores: s.close()
    return res


def load_model(proj):
    pth = _paths(proj); meta = json.load(open(pth['meta'])); tk = json.load(open(pth['tok']))
    c, P, mem, stores = _open(proj, meta, False); opt = B.Adam(P, meta['lr'])
    if not _load_state(proj, P, opt, stores): raise ValueError('no trained model in %s yet; run phone-train first' % proj)
    for L in mem.values(): L.update = False
    return c, P, mem, stores, B.Tok(tk), meta


def sample(proj, prompt, n=80, temp=0.8, seed=0):
    c, P, mem, stores, tok, meta = load_model(proj)
    try: return B.generate(P, c, tok, prompt, n=n, temp=temp, seed=seed, mem=mem)
    finally:
        for s in stores: s.close()


def status(proj):
    pth = _paths(proj); meta = json.load(open(pth['meta'])); recs = []
    if os.path.exists(pth['log']): recs = [json.loads(x) for x in open(pth['log']).read().splitlines()[-400:] if x.strip()]
    vals = [r for r in recs if 'val_loss' in r]
    disk = sum(os.stat(pth['experts'] % l).st_blocks * 512 for l in meta['experts']['layer_ids'] if os.path.exists(pth['experts'] % l))
    return dict(name=os.path.basename(proj), total_params=meta['plan']['total_params'], step=meta['step'], horizon=meta['horizon'], last=recs[-1] if recs else None,
                best_val_loss=min((r['val_loss'] for r in vals), default=None), val_trend=[r['val_loss'] for r in vals[-8:]], experts_disk_gb=round(disk / 1e9, 2),
                experts_file_gb=round(meta['experts']['file_bytes'] / 1e9, 1))
