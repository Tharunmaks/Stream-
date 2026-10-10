#!/usr/bin/env python3
"""ExpertStream MCP server: lets an AI coder (Claude Code, etc.) download, inspect, load and run
GGUF language models locally through the ExpertStream engine, with no browser.

  claude mcp add expertstream -- python3 ~/Stream-/mcp/expertstream_mcp.py

Transport: MCP over stdio (newline-delimited JSON-RPC). With --http PORT it also answers on
127.0.0.1:PORT (POST /mcp for JSON-RPC, GET /health) so the website's Settings page can see it.
Pure standard library. Environment: ES_MODELS (default ~/models), HF_TOKEN, ES_MAX_PARAMS_B (default 250).
"""
import json, os, sys, struct, threading, subprocess, time, shutil, urllib.request, urllib.parse, urllib.error, http.server, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.expanduser(os.environ.get('ES_MODELS', '~/models'))
MAX_B = float(os.environ.get('ES_MAX_PARAMS_B', '250'))
CFG = os.path.expanduser('~/.expertstream/config.json')
VERSION = '1.0.0'

def cfg():
    try: return json.load(open(CFG))
    except Exception: return {}
def hf_token(): return os.environ.get('HF_TOKEN') or cfg().get('hf_token', '')

# ---------------- GGUF header ----------------
GT = {0:('F32',1,4),1:('F16',1,2),2:('Q4_0',32,18),3:('Q4_1',32,20),6:('Q5_0',32,22),7:('Q5_1',32,24),8:('Q8_0',32,34),9:('Q8_1',32,36),10:('Q2_K',256,84),11:('Q3_K',256,110),12:('Q4_K',256,144),13:('Q5_K',256,176),14:('Q6_K',256,210),15:('Q8_K',256,292),16:('IQ2_XXS',256,66),17:('IQ2_XS',256,74),18:('IQ3_XXS',256,98),19:('IQ1_S',256,50),20:('IQ4_NL',32,18),21:('IQ3_S',256,110),22:('IQ2_S',256,82),23:('IQ4_XS',256,136),24:('I8',1,1),25:('I16',1,2),26:('I32',1,4),27:('I64',1,8),28:('F64',1,8),29:('IQ1_M',256,56),30:('BF16',1,2)}
ENGINE_TYPES = {0,1,2,3,6,7,8,10,11,12,13,14}
ARCH = {'olmoe':'verified','qwen3moe':'verified','qwen3':'verified','qwen2':'verified','llama':'ok','mixtral':'experimental'}
class NeedMore(Exception): pass
def parse_gguf(buf):
    p = [0]
    def need(n):
        if p[0] + n > len(buf): raise NeedMore()
    def rd(fmt):
        n = struct.calcsize(fmt); need(n); v = struct.unpack_from('<' + fmt, buf, p[0])[0]; p[0] += n; return v
    def st():
        n = rd('Q'); need(n); s = buf[p[0]:p[0]+n].decode('utf8', 'replace'); p[0] += n; return s
    SC = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
    need(24)
    if buf[:4] != b'GGUF': raise ValueError('not a GGUF file')
    p[0] = 4; ver = rd('I')
    nT = rd('Q'); nKV = rd('Q'); kv = {}; alen = {}
    for _ in range(nKV):
        k = st(); t = rd('I')
        if t == 8: kv[k] = st()
        elif t == 9:
            et = rd('I'); n = rd('Q'); alen[k] = n
            if et == 8:
                for _ in range(n): m = rd('Q'); need(m); p[0] += m
            else:
                sz = struct.calcsize('<' + SC[et]); need(sz * n)
                if n <= 16: kv[k] = [struct.unpack_from('<' + SC[et], buf, p[0] + i * sz)[0] for i in range(n)]
                p[0] += sz * n
        else: kv[k] = rd(SC[t])
    tens = []
    for _ in range(nT):
        name = st(); nd = rd('I'); dims = [rd('Q') for _ in range(nd)]; ty = rd('I'); off = rd('Q')
        n = 1
        for d in dims: n *= d
        g = GT.get(ty); by = -(-n // g[1]) * g[2] if g else 0
        tens.append((name, ty, n, by))
    return kv, alen, tens
def summarize(kv, alen, tens, size):
    a = kv.get('general.architecture', '?'); k = lambda s: kv.get(a + '.' + s)
    L = k('block_count') or 0; emb = k('embedding_length') or 0; nh = k('attention.head_count') or 0
    nkv = k('attention.head_count_kv'); nkv = nkv[0] if isinstance(nkv, list) else (nkv or nh)
    hd = k('attention.key_length') or (emb // nh if nh else 0)
    nexp = k('expert_count') or 0; used = k('expert_used_count') or 0
    exp = core = params = 0; types = {}
    for n, ty, c, by in tens:
        params += c; types[ty] = types.get(ty, 0) + by
        if '_exps.' in n: exp += by
        else: core += by
    return dict(arch=a, name=kv.get('general.name', ''), size=size, layers=L, emb=emb, heads=nh, kv_heads=nkv, head_dim=hd, experts=nexp, used=used, ctx_train=k('context_length'), vocab=alen.get('tokenizer.ggml.tokens', 0), tok_model=kv.get('tokenizer.ggml.model', ''), tok_pre=kv.get('tokenizer.ggml.pre', ''), rope_dim=k('rope.dimension_count'), expert_bytes=exp, core_bytes=core, params=params, quant_mix={GT[t][0] if t in GT else str(t): b for t, b in types.items()}, types=list(types), moe=bool(nexp > 1 and exp))
def assess(s, ctx=1024, cache_mb=384, free_ram_mb=None, file_name=''):
    blockers, warns = [], []
    if s['arch'] not in ARCH: blockers.append('architecture %r is not supported (supported: olmoe, qwen3moe, qwen3, qwen2, llama)' % s['arch'])
    if s['tok_model'] and s['tok_model'] != 'gpt2': blockers.append('tokenizer %r (SentencePiece) is not supported; only byte-level BPE' % s['tok_model'])
    bad = [GT[t][0] if t in GT else str(t) for t in s['types'] if t not in ENGINE_TYPES]
    if bad: blockers.append('unsupported quantisation: ' + ', '.join(bad))
    if s['rope_dim'] and s['head_dim'] and s['rope_dim'] != s['head_dim']: blockers.append('partial rotary embeddings are not supported')
    if re.search(r'-\d{5}-of-\d{5}', file_name): warns.append('split GGUF: import with es_import first (import_pack tool)')
    if ARCH.get(s['arch']) == 'experimental': warns.append('mixtral routing is not verified against llama.cpp')
    kv_mb = s['layers'] * 2 * s['kv_heads'] * s['head_dim'] * ctx * 4 / 1048576
    core_mb = (s['core_bytes'] if s['moe'] else s['size']) / 1048576
    need = core_mb + kv_mb + (cache_mb if s['moe'] else 0) + 120
    if free_ram_mb and need > free_ram_mb: blockers.append('needs about %d MB of RAM but only %d MB is free' % (need, free_ram_mb))
    return dict(runs=not blockers, blockers=blockers, warnings=warns, core_mb=round(core_mb), kv_mb=round(kv_mb), needs_ram_mb=round(need), expert_mb=round(s['expert_bytes'] / 1048576))
def read_header(path_or_url, headers=None):
    for mb in (4, 16, 48):
        n = mb << 20
        if path_or_url.startswith('http'):
            req = urllib.request.Request(path_or_url, headers=dict(headers or {}, Range='bytes=0-%d' % (n - 1)))
            with urllib.request.urlopen(req, timeout=60) as r:
                buf = r.read(); cr = r.headers.get('Content-Range'); size = int(cr.split('/')[1]) if cr else len(buf)
        else:
            size = os.path.getsize(path_or_url)
            with open(path_or_url, 'rb') as f: buf = f.read(n)
        try: return summarize(*parse_gguf(buf), size)
        except NeedMore:
            if mb == 48: raise
    raise ValueError('header too large')

# ---------------- device ----------------
def meminfo():
    d = {}
    try:
        for l in open('/proc/meminfo'):
            k, v = l.split(':'); d[k] = int(v.split()[0]) // 1024
    except Exception: pass
    return d
def device():
    m = meminfo(); os.makedirs(MODELS, exist_ok=True)
    du = shutil.disk_usage(MODELS)
    return dict(cpus=os.cpu_count(), ram_total_mb=m.get('MemTotal'), ram_free_mb=m.get('MemAvailable'), disk_free_gb=round(du.free / 1e9, 1), models_dir=MODELS, engine=bool(find_engine()), max_params_b=MAX_B)
def find_engine():
    for p in (os.path.join(ROOT, 'es_chat'), shutil.which('es_chat') or ''):
        if p and os.access(p, os.X_OK): return p
    return None

# ---------------- HF ----------------
def hf(url, **kw):
    h = {'User-Agent': 'expertstream-mcp'}
    if hf_token(): h['Authorization'] = 'Bearer ' + hf_token()
    h.update(kw.get('headers', {}))
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=60)
def hf_json(url): return json.load(hf(url))
JOBS = {}
def download_job(jid, repo, file, dest):
    j = JOBS[jid]; url = 'https://huggingface.co/%s/resolve/main/%s?download=true' % (repo, urllib.parse.quote(file))
    try:
        have = os.path.getsize(dest + '.part') if os.path.exists(dest + '.part') else 0
        while True:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'expertstream-mcp', 'Range': 'bytes=%d-' % have, **({'Authorization': 'Bearer ' + hf_token()} if hf_token() else {})})
                with urllib.request.urlopen(req, timeout=60) as r:
                    cr = r.headers.get('Content-Range'); total = int(cr.split('/')[1]) if cr else int(r.headers.get('Content-Length', 0)) + (have if r.status == 206 else 0)
                    if r.status == 200 and have: have = 0
                    j['total'] = total; t0 = time.time(); b0 = have
                    with open(dest + '.part', 'ab' if have and r.status == 206 else 'wb') as f:
                        while True:
                            c = r.read(1 << 20)
                            if not c: break
                            f.write(c); have += len(c); j['done'] = have; j['mbps'] = round((have - b0) / 1e6 / max(1e-3, time.time() - t0), 1)
                            if j.get('cancel'): j['status'] = 'cancelled'; return
                if total and have < total: continue
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code < 500: raise
                j['note'] = 'retrying: %s' % e; time.sleep(3)
        os.replace(dest + '.part', dest); j['status'] = 'done'
    except Exception as e:
        j['status'] = 'error'; j['error'] = str(e) + (' (gated or private repo: connect a Hugging Face token)' if '401' in str(e) or '403' in str(e) else '')

# ---------------- engine process ----------------
class Engine:
    def __init__(self): self.p = None; self.path = None; self.info = None; self.lock = threading.Lock()
    def load(self, path, ctx, cache, temp=0.7, max_new=256):
        self.unload(); exe = find_engine()
        if not exe: raise RuntimeError('es_chat not found. Run `make` in %s (see README).' % ROOT)
        self.p = subprocess.Popen([exe, '-g', path, '-J', '-c', str(ctx), '-C', str(cache), '-t', str(temp), '-n', str(max_new)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        t0 = time.time()
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError('engine exited: ' + (self.p.stderr.read() or '')[-500:])
            try: e = json.loads(line)
            except Exception: continue
            if e.get('ev') == 'ready': self.info = e; self.path = path; e['load_wall_s'] = round(time.time() - t0, 2); return e
    def unload(self):
        if self.p:
            try: self.p.stdin.close(); self.p.terminate()
            except Exception: pass
        self.p = self.path = self.info = None
    def ask(self, text, max_tokens=256, temperature=0.7, raw=False):
        if not self.p: raise RuntimeError('no model loaded; call load_model first')
        with self.lock:
            # sampling is fixed per process; restart is not needed for these small controls, so apply by command line on load
            self.p.stdin.write(text.replace('\\', '\\\\').replace('\n', '\\n') + '\n'); self.p.stdin.flush()
            out, st = [], {}
            while True:
                line = self.p.stdout.readline()
                if not line: raise RuntimeError('engine stopped')
                try: e = json.loads(line)
                except Exception: continue
                if e['ev'] == 'tok': out.append(e['t'])
                elif e['ev'] == 'done': st = e; break
            return ''.join(out), st
ENG = Engine()

# ---------------- catalog for recommendations ----------------
CATALOG = [
 dict(id='smollm2-135m', params_b=0.135, active_b=0.135, repo='bartowski/SmolLM2-135M-Instruct-GGUF', file='SmolLM2-135M-Instruct-Q8_0.gguf', skill='demo, smoke tests', quality=1),
 dict(id='qwen2.5-0.5b', params_b=0.5, active_b=0.5, repo='Qwen/Qwen2.5-0.5B-Instruct-GGUF', file='qwen2.5-0.5b-instruct-q4_k_m.gguf', skill='small assistant, multilingual', quality=2),
 dict(id='llama3.2-1b', params_b=1.2, active_b=1.2, repo='bartowski/Llama-3.2-1B-Instruct-GGUF', file='Llama-3.2-1B-Instruct-Q4_K_M.gguf', skill='general chat', quality=3),
 dict(id='qwen3-1.7b', params_b=1.7, active_b=1.7, repo='Qwen/Qwen3-1.7B-GGUF', file='Qwen3-1.7B-Q8_0.gguf', skill='reasoning, general chat', quality=4),
 dict(id='olmoe-1b-7b', params_b=6.9, active_b=1.3, repo='bartowski/OLMoE-1B-7B-0924-Instruct-GGUF', file='OLMoE-1B-7B-0924-Instruct-Q2_K.gguf', skill='MoE showcase, general chat', quality=4),
 dict(id='qwen3-30b-a3b', params_b=30.5, active_b=3.3, repo='unsloth/Qwen3-30B-A3B-GGUF', file='Qwen3-30B-A3B-Q2_K.gguf', skill='strong general/coding MoE', quality=7),
 dict(id='qwen3-235b-a22b', params_b=235, active_b=22, repo='unsloth/Qwen3-235B-A22B-GGUF', file=None, skill='frontier-class MoE (split files: use import_pack)', quality=9),
]
def plan(params_b, bits, active_b=None, cache_gb=2.0, flash_gbs=1.75, disk_free_gb=None):
    if params_b > MAX_B: return dict(ok=False, error='%.0fB exceeds the supported maximum of %.0fB parameters' % (params_b, MAX_B))
    active_b = active_b or params_b
    disk = params_b * bits / 8; act = active_b * bits / 8
    hit = min(.9, cache_gb / max(1, disk) * 12) if params_b != active_b else 1.0
    sec = act * (1 - hit) / flash_gbs + act / 9.8
    free = disk_free_gb if disk_free_gb is not None else device()['disk_free_gb']
    return dict(ok=True, file_gb=round(disk, 1), fits_on_disk=disk <= free, free_disk_gb=free, read_per_token_gb=round(act, 2), est_cache_hit=round(hit, 2), seconds_per_token=round(sec, 2), tokens_per_second=round(1 / sec, 3), note='rough estimate; measure with benchmark')

# ---------------- tools ----------------
def t_status(a):
    d = device(); d['loaded'] = ENG.path; d['server_version'] = VERSION; d['hf_connected'] = bool(hf_token()); d['downloads'] = {k: dict(file=v['file'], status=v['status'], done=v.get('done', 0), total=v.get('total', 0)) for k, v in JOBS.items()}
    return d
def t_list_models(a):
    out = []
    os.makedirs(MODELS, exist_ok=True)
    for f in sorted(os.listdir(MODELS)):
        if f.endswith('.gguf'):
            p = os.path.join(MODELS, f)
            try: s = read_header(p); ass = assess(s, free_ram_mb=meminfo().get('MemAvailable'), file_name=f); out.append(dict(file=f, size_gb=round(s['size'] / 1e9, 2), arch=s['arch'], moe=s['moe'], runs=ass['runs'], problems=ass['blockers']))
            except Exception as e: out.append(dict(file=f, error=str(e)))
    return dict(models_dir=MODELS, models=out)
def t_search(a):
    q = urllib.parse.quote(a.get('query', '')); n = min(int(a.get('limit', 10)), 25)
    r = hf_json('https://huggingface.co/api/models?filter=gguf&sort=downloads&direction=-1&limit=%d&expand%%5B%%5D=gguf&expand%%5B%%5D=downloads%s' % (n, '&search=' + q if q else ''))
    return [dict(repo=m['id'], arch=(m.get('gguf') or {}).get('architecture'), params=(m.get('gguf') or {}).get('total'), downloads=m.get('downloads'), supported=(m.get('gguf') or {}).get('architecture') in ARCH) for m in r]
def t_files(a):
    out = []
    def walk(path, d):
        for f in hf_json('https://huggingface.co/api/models/%s/tree/main%s' % (a['repo'], '/' + path if path else '')):
            if f['type'] == 'file' and f['path'].lower().endswith('.gguf'): out.append(dict(file=f['path'], size_gb=round(((f.get('lfs') or {}).get('size') or f.get('size', 0)) / 1e9, 2)))
            elif f['type'] == 'directory' and d < 1: walk(f['path'], d + 1)
    walk('', 0); return sorted(out, key=lambda x: x['size_gb'])
def t_inspect(a):
    if a.get('repo'):
        url = 'https://huggingface.co/%s/resolve/main/%s' % (a['repo'], urllib.parse.quote(a['file']))
        s = read_header(url, {'Authorization': 'Bearer ' + hf_token()} if hf_token() else None)
        name = a['file']
    else:
        p = a['path'] if os.path.isabs(a['path']) else os.path.join(MODELS, a['path']); s = read_header(p); name = p
    if s['size'] and s['params']:
        pass
    return dict(summary={k: v for k, v in s.items() if k not in ('types',)}, verdict=assess(s, int(a.get('ctx', 1024)), int(a.get('cache_mb', 384)), meminfo().get('MemAvailable'), name))
def t_plan(a): return plan(float(a['params_b']), float(a.get('bits', 2.5)), float(a['active_b']) if a.get('active_b') else None, float(a.get('cache_gb', 2)))
def t_download(a):
    repo, file = a['repo'], a['file']; os.makedirs(MODELS, exist_ok=True)
    try:
        s = read_header('https://huggingface.co/%s/resolve/main/%s' % (repo, urllib.parse.quote(file)), {'Authorization': 'Bearer ' + hf_token()} if hf_token() else None)
        v = assess(s, file_name=file)
        if not v['runs'] and not a.get('force'): return dict(started=False, reason='not runnable', verdict=v)
        if s['params'] / 1e9 > MAX_B: return dict(started=False, reason='%.0fB parameters exceeds the %.0fB limit' % (s['params'] / 1e9, MAX_B))
        if s['size'] / 1e9 > device()['disk_free_gb'] * 0.95: return dict(started=False, reason='not enough free disk space for %.1f GB' % (s['size'] / 1e9))
    except urllib.error.HTTPError as e:
        return dict(started=False, reason='HTTP %d from Hugging Face%s' % (e.code, ' (connect a token for gated/private repos)' if e.code in (401, 403) else ''))
    jid = 'dl%d' % (len(JOBS) + 1); dest = os.path.join(MODELS, os.path.basename(file))
    JOBS[jid] = dict(file=file, repo=repo, status='downloading', done=0, total=s['size'])
    threading.Thread(target=download_job, args=(jid, repo, file, dest), daemon=True).start()
    return dict(started=True, job=jid, saving_to=dest, size_gb=round(s['size'] / 1e9, 2), next='poll download_status, then load_model')
def t_dl_status(a):
    j = JOBS.get(a.get('job'))
    if not j: return dict(error='unknown job', jobs=list(JOBS))
    return dict(j, percent=round(100 * j.get('done', 0) / max(1, j.get('total', 1)), 1))
def t_load(a):
    p = a['model'] if os.path.isabs(a['model']) else os.path.join(MODELS, a['model'])
    if not os.path.exists(p): raise RuntimeError('model not found: ' + p)
    s = read_header(p); v = assess(s, int(a.get('ctx', 2048)), int(a.get('cache_mb', 1024)), meminfo().get('MemAvailable'), p)
    if not v['runs']: return dict(loaded=False, verdict=v)
    e = ENG.load(p, int(a.get('ctx', 2048)), int(a.get('cache_mb', 1024)), float(a.get('temperature', 0.7)), int(a.get('max_tokens', 256)))
    return dict(loaded=True, model=e.get('model'), arch=e.get('arch'), layers=e.get('layers'), experts=e.get('experts'), core_mb=e.get('core_mb'), ram_mb=e.get('ram_mb'), load_seconds=e.get('load_wall_s'))
def t_unload(a): ENG.unload(); return dict(unloaded=True)
def t_generate(a):
    text, st = ENG.ask(a['prompt'], raw=bool(a.get('raw')))
    return dict(text=text, tokens=st.get('tokens'), seconds=st.get('secs'), tokens_per_second=st.get('tps'), expert_cache_hit_pct=st.get('hit'), flash_mb_read=st.get('flash_mb'), ram_mb=st.get('ram_mb'))
def t_reset(a):
    if ENG.p:
        with ENG.lock:
            ENG.p.stdin.write('/reset\n'); ENG.p.stdin.flush()
            while json.loads(ENG.p.stdout.readline() or '{}').get('ev') != 'done': pass
    return dict(reset=True)
def t_bench(a):
    t0 = time.time(); text, st = ENG.ask('Write a short story about a robot who learns to paint.')
    return dict(tokens=st.get('tokens'), tokens_per_second=st.get('tps'), prompt_seconds=st.get('prompt_secs'), wall_seconds=round(time.time() - t0, 2), ram_mb=st.get('ram_mb'))
def t_recommend(a):
    d = device(); mx = min(float(a.get('max_params_b', MAX_B)), MAX_B); out = []
    for m in CATALOG:
        if m['params_b'] > mx: continue
        bits = 2.7 if m['params_b'] > 20 else 4.5 if m['params_b'] > 3 else 5
        p = plan(m['params_b'], bits, m['active_b'], 2.0, disk_free_gb=d['disk_free_gb'])
        out.append(dict(id=m['id'], skill=m['skill'], quality=m['quality'], plan=p, repo=m['repo'], file=m['file'], local=bool(m['file'] and os.path.exists(os.path.join(MODELS, m['file'])))))
    out.sort(key=lambda x: (-(x['plan']['fits_on_disk'] and x['plan']['seconds_per_token'] < float(a.get('max_seconds_per_token', 15))), -x['quality']))
    return dict(device=d, task=a.get('task', ''), recommendations=out, how_to_build_a_system=['recommend_models -> pick one', 'download_model(repo,file)', 'download_status until done', 'load_model(file)', 'generate(prompt) in a loop; chat_reset between tasks', 'benchmark to confirm speed', 'for split GGUF (235B class) use import_pack then load the pack with es_serve -m'])
def t_import(a):
    exe = os.path.join(ROOT, 'es_import')
    if not os.access(exe, os.X_OK): raise RuntimeError('es_import not built; run make')
    out = os.path.expanduser(a.get('out') or '~/packs/' + os.path.splitext(os.path.basename(a['gguf']))[0])
    p = subprocess.run([exe, '-o', out, a['gguf'] if os.path.isabs(a['gguf']) else os.path.join(MODELS, a['gguf'])], capture_output=True, text=True, timeout=3600)
    return dict(ok=p.returncode == 0, pack=out, log=(p.stdout + p.stderr)[-800:])
def t_serve(a):
    exe = os.path.join(ROOT, 'es_serve'); port = int(a.get('port', 8080))
    if not os.access(exe, os.X_OK): raise RuntimeError('es_serve not built; run make')
    m = a['model'] if os.path.isabs(a['model']) else os.path.join(MODELS, a['model'])
    subprocess.Popen([exe, '-g', m, '-p', str(port), '-w', os.path.join(ROOT, 'web')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dict(started=True, url='http://127.0.0.1:%d' % port)
def t_set_token(a):
    os.makedirs(os.path.dirname(CFG), exist_ok=True); c = cfg(); c['hf_token'] = a['token']; json.dump(c, open(CFG, 'w')); os.chmod(CFG, 0o600)
    try: who = hf_json('https://huggingface.co/api/whoami-v2'); return dict(saved=True, user=who.get('name'))
    except Exception as e: return dict(saved=True, warning='token saved but validation failed: %s' % e)

def S(**p): return dict(type='object', properties=p, required=[k for k, v in p.items() if v.pop('req', False)])
def P(t, d, **k): return dict(type=t, description=d, **k)
TOOLS = [
 ('expertstream_status', 'Device RAM/disk/CPU, whether the engine is built, the loaded model, running downloads. Call first.', S(), t_status),
 ('recommend_models', 'Rank models that fit this device (max 250B parameters) for a task, with the exact tool calls to set one up. Use this to design an on-device AI system.', S(task=P('string', 'what the system should do'), max_params_b=P('number', 'upper bound on parameters in billions, default and hard cap 250'), max_seconds_per_token=P('number', 'default 15')), t_recommend),
 ('plan_capacity', 'Estimate whether a model of a given size fits flash and how fast it would run. Refuses above the 250B cap.', S(params_b=P('number', 'total parameters in billions', req=True), bits=P('number', 'bits per weight, e.g. 2.7 for Q2_K, 4.5 for Q4_K_M'), active_b=P('number', 'parameters used per token for MoE'), cache_gb=P('number', 'expert cache in RAM')), t_plan),
 ('list_local_models', 'GGUF files in the models folder with a can-it-run verdict.', S(), t_list_models),
 ('search_huggingface', 'Search Hugging Face for GGUF models; flags architectures the engine supports.', S(query=P('string', 'search text'), limit=P('integer', 'max results, default 10')), t_search),
 ('list_repo_files', 'List the .gguf files (with sizes) in a Hugging Face repo.', S(repo=P('string', 'owner/name', req=True)), t_files),
 ('inspect_model', 'Read a GGUF header (local file or Hugging Face file) and report architecture, sizes, RAM need and whether the engine can run it. Does not download the model.', S(repo=P('string', 'owner/name (for remote)'), file=P('string', 'file in the repo'), path=P('string', 'local file name or path'), ctx=P('integer', 'context tokens'), cache_mb=P('integer', 'expert cache MB')), t_inspect),
 ('download_model', 'Start a resumable background download from Hugging Face into the models folder. Checks compatibility, disk space and the 250B cap first.', S(repo=P('string', 'owner/name', req=True), file=P('string', 'gguf file name', req=True), force=P('boolean', 'download even if the check fails')), t_download),
 ('download_status', 'Progress of a download job.', S(job=P('string', 'job id from download_model', req=True)), t_dl_status),
 ('load_model', 'Load a local GGUF into the ExpertStream engine (core in RAM, experts streamed from flash). One model at a time.', S(model=P('string', 'file name in the models folder, or absolute path', req=True), ctx=P('integer', 'context tokens, default 2048'), cache_mb=P('integer', 'expert cache MB, default 1024'), temperature=P('number', '0 = greedy, default 0.7'), max_tokens=P('integer', 'longest answer, default 256')), t_load),
 ('generate', 'Send a prompt to the loaded model and return its answer with speed stats. Keeps conversation context until chat_reset.', S(prompt=P('string', 'user message', req=True), raw=P('boolean', 'continue raw text without chat template')), t_generate),
 ('chat_reset', 'Forget the conversation so far.', S(), t_reset),
 ('benchmark', 'Generate a fixed prompt on the loaded model and report tokens per second and memory.', S(), t_bench),
 ('unload_model', 'Free the loaded model\'s memory.', S(), t_unload),
 ('import_pack', 'Convert a (split) GGUF MoE into an ExpertStream pack for very large models.', S(gguf=P('string', 'gguf file', req=True), out=P('string', 'output folder')), t_import),
 ('open_web_ui', 'Start the ExpertStream website backed by a local model (es_serve) and return its URL.', S(model=P('string', 'gguf file', req=True), port=P('integer', 'default 8080')), t_serve),
 ('connect_huggingface', 'Save a Hugging Face read token (stored in ~/.expertstream/config.json, mode 600) so gated/private models can be downloaded.', S(token=P('string', 'hf_... token', req=True)), t_set_token),
]
TOOLMAP = {n: f for n, d, s, f in TOOLS}

# ---------------- JSON-RPC ----------------
def handle(req):
    m, i = req.get('method'), req.get('id'); params = req.get('params') or {}
    if m == 'initialize':
        return dict(jsonrpc='2.0', id=i, result=dict(protocolVersion=params.get('protocolVersion', '2024-11-05'), capabilities=dict(tools=dict(listChanged=False)), serverInfo=dict(name='expertstream', version=VERSION), instructions='Run and manage local LLMs (up to %.0fB parameters) with the ExpertStream engine. Start with expertstream_status, then recommend_models.' % MAX_B))
    if m == 'ping': return dict(jsonrpc='2.0', id=i, result={})
    if m == 'tools/list': return dict(jsonrpc='2.0', id=i, result=dict(tools=[dict(name=n, description=d, inputSchema=s) for n, d, s, f in TOOLS]))
    if m == 'tools/call':
        f = TOOLMAP.get(params.get('name'))
        if not f: return dict(jsonrpc='2.0', id=i, error=dict(code=-32602, message='unknown tool'))
        try: res = f(params.get('arguments') or {}); return dict(jsonrpc='2.0', id=i, result=dict(content=[dict(type='text', text=json.dumps(res, indent=1))]))
        except Exception as e: return dict(jsonrpc='2.0', id=i, result=dict(isError=True, content=[dict(type='text', text='%s: %s' % (type(e).__name__, e))]))
    if i is None: return None   # notification
    return dict(jsonrpc='2.0', id=i, error=dict(code=-32601, message='method not found'))
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _cors(self): self.send_header('Access-Control-Allow-Origin', '*'); self.send_header('Access-Control-Allow-Headers', 'content-type'); self.send_header('Access-Control-Allow-Private-Network', 'true')
    def do_OPTIONS(self): self.send_response(204); self._cors(); self.end_headers()
    def do_GET(self):
        b = json.dumps(dict(ok=True, server='expertstream-mcp', version=VERSION, tools=len(TOOLS), max_params_b=MAX_B, device=device())).encode()
        self.send_response(200); self._cors(); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(b)
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0)); r = handle(json.loads(self.rfile.read(n) or b'{}')); b = json.dumps(r).encode() if r else b''
        self.send_response(200 if r else 202); self._cors(); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(b)
def main():
    if '--http' in sys.argv:
        port = int(sys.argv[sys.argv.index('--http') + 1]); srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), H)
        if '--stdio' not in sys.argv:
            print('ExpertStream MCP on http://127.0.0.1:%d/mcp (GET /health)' % port, file=sys.stderr); srv.serve_forever(); return
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try: r = handle(json.loads(line))
        except Exception as e: r = dict(jsonrpc='2.0', id=None, error=dict(code=-32700, message=str(e)))
        if r: sys.stdout.write(json.dumps(r) + '\n'); sys.stdout.flush()
if __name__ == '__main__': main()
