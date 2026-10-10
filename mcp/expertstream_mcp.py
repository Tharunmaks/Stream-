#!/usr/bin/env python3
"""ExpertStream MCP server: lets an AI coder (Claude Code, etc.) download, inspect, load and run
GGUF language models locally through the ExpertStream engine, with no browser.

  claude mcp add expertstream -- python3 ~/Stream-/mcp/expertstream_mcp.py

Transports: MCP over stdio; or --http PORT for streamable HTTP (POST /mcp) and legacy SSE (GET /sse),
plus an OpenAI-compatible API (POST /v1/chat/completions, model "agent:<id>") and GET /health.
Secure by default: HTTP needs a token (auto-created in ~/.expertstream/token, mode 600), loopback servers reject
foreign Host/Origin headers, bodies are capped, model paths are confined to the models folder, downloads only
follow Hugging Face hosts. --host 0.0.0.0 shares it on the network (use --cert/--key for TLS); --no-auth is loopback only.
--public opens a free https address (cloudflared or ssh) and prints a connector URL for apps that want a URL instead of a config file; --public-url https://your.domain does the same for a domain/proxy you own.
250+ agent presets (mcp/agents.json) run on the loaded local model.
Pure standard library. Environment: ES_MODELS (default ~/models), HF_TOKEN, ES_MAX_PARAMS_B (default 250).
"""
import json, os, sys, struct, threading, subprocess, time, shutil, urllib.request, urllib.parse, urllib.error, http.server, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.expanduser(os.environ.get('ES_MODELS', '~/models'))
MAX_B = float(os.environ.get('ES_MAX_PARAMS_B', '250'))
CFG = os.path.expanduser('~/.expertstream/config.json')
VERSION = '3.0.0'

def cfg():
    try: return json.load(open(CFG))
    except Exception: return {}
def hf_token(): return os.environ.get('HF_TOKEN') or cfg().get('hf_token', '')


# ---------------- security helpers ----------------
RE_REPO = re.compile(r'^[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}$')
RE_FILE = re.compile(r'^[\w][\w.\- /]{0,200}\.gguf$')
RE_HFTOKEN = re.compile(r'^hf_[A-Za-z0-9]{20,200}$')
ALLOW_ANY_PATH = os.environ.get('ES_ALLOW_ANY_PATH') == '1'
MAX_BODY = 4 << 20
def need_repo(r):
    if not isinstance(r, str) or not RE_REPO.match(r): raise ValueError('repo must look like owner/name')
    return r
def need_file(f):
    if not isinstance(f, str) or not RE_FILE.match(f) or '..' in f or f.startswith('/'): raise ValueError('file must be a plain .gguf name from the repository')
    return f
def under(path, root):
    path, root = os.path.realpath(path), os.path.realpath(root)
    return path == root or path.startswith(root + os.sep)
def safe_model_path(name):
    p = os.path.expanduser(name); p = p if os.path.isabs(p) else os.path.join(MODELS, p)
    if not ALLOW_ANY_PATH and not under(p, MODELS): raise ValueError('model files must be inside %s (set ES_ALLOW_ANY_PATH=1 to allow other places)' % MODELS)
    if not p.lower().endswith('.gguf'): raise ValueError('only .gguf files can be loaded')
    return p
class _HFRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        h = urllib.parse.urlparse(newurl)
        if h.scheme != 'https' or not (h.hostname == 'huggingface.co' or h.hostname.endswith('.huggingface.co') or h.hostname.endswith('.hf.co') or h.hostname == 'hf.co'):
            raise urllib.error.URLError('refusing to follow a redirect to %s' % h.hostname)
        return super().redirect_request(req, fp, code, msg, headers, newurl)
HFO = urllib.request.build_opener(_HFRedirect())

# ---------------- GGUF header ----------------
GT = {0:('F32',1,4),1:('F16',1,2),2:('Q4_0',32,18),3:('Q4_1',32,20),6:('Q5_0',32,22),7:('Q5_1',32,24),8:('Q8_0',32,34),9:('Q8_1',32,36),10:('Q2_K',256,84),11:('Q3_K',256,110),12:('Q4_K',256,144),13:('Q5_K',256,176),14:('Q6_K',256,210),15:('Q8_K',256,292),16:('IQ2_XXS',256,66),17:('IQ2_XS',256,74),18:('IQ3_XXS',256,98),19:('IQ1_S',256,50),20:('IQ4_NL',32,18),21:('IQ3_S',256,110),22:('IQ2_S',256,82),23:('IQ4_XS',256,136),24:('I8',1,1),25:('I16',1,2),26:('I32',1,4),27:('I64',1,8),28:('F64',1,8),29:('IQ1_M',256,56),30:('BF16',1,2)}
ENGINE_TYPES = {0,1,2,3,6,7,8,10,11,12,13,14,20,23,30}
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
    if re.search(r'-\d{5}-of-\d{5}', file_name): warns.append('split GGUF: the native engine loads the first part directly once every part is downloaded')
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
            with HFO.open(req, timeout=60) as r:
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
    return HFO.open(urllib.request.Request(url, headers=h), timeout=60)
def hf_json(url): return json.load(hf(url))
JOBS = {}
def download_job(jid, repo, file, dest):
    m = re.match(r'^(.*)-00001-of-(\d{5})(\.gguf)$', file)
    if not m: return _dl_one(jid, repo, file, dest, True)
    for i in range(1, int(m.group(2)) + 1):
        f = '%s-%05d-of-%s%s' % (m.group(1), i, m.group(2), m.group(3)); JOBS[jid]['file'] = f
        _dl_one(jid, repo, f, os.path.join(os.path.dirname(dest), os.path.basename(f)), i == int(m.group(2)))
        if JOBS[jid]['status'] in ('error', 'cancelled'): return
def _dl_one(jid, repo, file, dest, last):
    j = JOBS[jid]; url = 'https://huggingface.co/%s/resolve/main/%s?download=true' % (repo, urllib.parse.quote(file))
    try:
        have = os.path.getsize(dest + '.part') if os.path.exists(dest + '.part') else 0
        while True:
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'expertstream-mcp', 'Range': 'bytes=%d-' % have, **({'Authorization': 'Bearer ' + hf_token()} if hf_token() else {})})
                with HFO.open(req, timeout=60) as r:
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
        os.replace(dest + '.part', dest)
        if last: j['status'] = 'done'
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
    def ask(self, text, max_tokens=256, temperature=0.7, raw=False, on_tok=None, fresh=False):
        if not self.p: raise RuntimeError('no model loaded; call load_model first')
        with self.lock:
            if fresh: self._reset()
            self.p.stdin.write(text.replace('\\', '\\\\').replace('\n', '\\n') + '\n'); self.p.stdin.flush()
            out, st = [], {}
            while True:
                line = self.p.stdout.readline()
                if not line: raise RuntimeError('engine stopped')
                try: e = json.loads(line)
                except Exception: continue
                if e['ev'] == 'tok':
                    out.append(e['t'])
                    if on_tok:
                        try: on_tok(e['t'])
                        except Exception: on_tok = None   # client went away: keep draining so the engine stays in sync
                elif e['ev'] == 'done': st = e; break
            return ''.join(out), st
    def _reset(self):
        self.p.stdin.write('/reset\n'); self.p.stdin.flush()
        while json.loads(self.p.stdout.readline() or '{}').get('ev') != 'done': pass
    def reset(self):
        if self.p:
            with self.lock: self._reset()
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


# ---------------- agents: 250+ presets that run on the loaded model ----------------
def load_agents():
    for p in (os.path.join(HERE, 'agents.json'),):
        try: return json.load(open(p, encoding='utf-8'))
        except Exception: pass
    return dict(count=0, categories={}, agents=[])
AGENTS = load_agents(); AGENT_BY_ID = {a['id']: a for a in AGENTS['agents']}
def agent_brief(a): return dict(id=a['id'], name=a['name'], category=a['category_name'], description=a['description'])
STOP = set('a an the and or of to in on for with my me i we you it is are be should would could can how do does what why when use using need want make get your our this that from into as at by if so then about some any new small big'.split())
def toks(t): return [w for w in re.findall(r'[a-z0-9+#]+', t.lower()) if w not in STOP and len(w) > 1]
_DOC = {}
def _doc(a):
    if a['id'] not in _DOC: _DOC[a['id']] = (set(toks(a['id'].replace('-', ' ') + ' ' + a['name'])), set(toks(a['description'] + ' ' + a['category_name'])), set(toks(a.get('tags', ''))))
    return _DOC[a['id']]
_DF = {}
def _idf(w):
    if not _DF:
        for x in AGENTS['agents']:
            for t in _doc(x)[0] | _doc(x)[1] | _doc(x)[2]: _DF[t] = _DF.get(t, 0) + 1
    import math
    return math.log(1 + len(AGENTS['agents']) / (1 + _DF.get(w, 0)))
def score_agent(a, words):
    names, desc, tags = _doc(a)
    return sum(_idf(w) * (3 if (w in names or w in tags) else 1) for w in words if w in names or w in desc or w in tags)
def find_agents(query='', category='', limit=20):
    ws = toks(query)
    pool = [a for a in AGENTS['agents'] if not category or category.lower() in (a['category'], a['category_name'].lower())]
    if ws: pool = sorted([a for a in pool if score_agent(a, ws) > 0], key=lambda a: -score_agent(a, ws))
    return pool[:limit]
def agent_prompt(a, task, context=''):
    return a['system'] + '\n\n' + ('Context:\n' + context.strip() + '\n\n' if context and context.strip() else '') + 'Task:\n' + task.strip()
def run_agent_text(agent_id, task, context='', on_tok=None, keep=False):
    a = AGENT_BY_ID.get(agent_id)
    if not a: raise RuntimeError('unknown agent %r; use list_agents or recommend_agents' % agent_id)
    text, st = ENG.ask(agent_prompt(a, task, context), on_tok=on_tok, fresh=not keep)
    return a, text, st
def t_list_agents(a):
    ags = find_agents(a.get('query', ''), a.get('category', ''), int(a.get('limit', 25)))
    return dict(total=AGENTS['count'], categories=AGENTS['categories'], shown=len(ags), agents=[agent_brief(x) for x in ags])
def t_get_agent(a):
    x = AGENT_BY_ID.get(a['agent_id'])
    return x or dict(error='unknown agent')
def t_recommend_agents(a):
    ags = find_agents(a['task'], '', int(a.get('limit', 5)))
    return dict(task=a['task'], suggested=[agent_brief(x) for x in ags], next='run_agent(agent_id, task) or run_pipeline([...], task)')
def t_run_agent(a):
    ag, text, st = run_agent_text(a['agent_id'], a['task'], a.get('context', ''), keep=bool(a.get('keep_context')))
    return dict(agent=ag['name'], answer=text, tokens=st.get('tokens'), tokens_per_second=st.get('tps'))
def t_pipeline(a):
    ids = a['agent_ids']; cur = a['task']; steps = []
    for i in ids[:8]:
        ag, text, st = run_agent_text(i, cur if not steps else 'Previous agent output:\n' + steps[-1]['answer'] + '\n\nOriginal task:\n' + a['task'] + '\n\nDo your part.')
        steps.append(dict(agent=ag['name'], answer=text))
    return dict(steps=steps, final=steps[-1]['answer'] if steps else '')
def t_council(a):
    ids = a.get('agent_ids') or [x['id'] for x in find_agents(a['task'], '', 3)]
    views = []
    for i in ids[:5]:
        ag, text, st = run_agent_text(i, a['task']); views.append((ag['name'], text))
    merged = '\n\n'.join('### %s\n%s' % v for v in views)
    ag, final, st = run_agent_text('synthesizer', 'Task:\n' + a['task'] + '\n\nAgent answers:\n' + merged + '\n\nMerge into one answer and note disagreements.')
    return dict(members=[v[0] for v in views], answers=[dict(agent=n, answer=t) for n, t in views], synthesis=final)

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
    q = urllib.parse.quote(str(a.get('query', ''))[:100]); n = min(int(a.get('limit', 10)), 25)
    r = hf_json('https://huggingface.co/api/models?filter=gguf&sort=downloads&direction=-1&limit=%d&expand%%5B%%5D=gguf&expand%%5B%%5D=downloads%s' % (n, '&search=' + q if q else ''))
    return [dict(repo=m['id'], arch=(m.get('gguf') or {}).get('architecture'), params=(m.get('gguf') or {}).get('total'), downloads=m.get('downloads'), supported=(m.get('gguf') or {}).get('architecture') in ARCH) for m in r]
def t_files(a):
    need_repo(a['repo']); out = []
    def walk(path, d):
        for f in hf_json('https://huggingface.co/api/models/%s/tree/main%s' % (a['repo'], '/' + path if path else '')):
            if f['type'] == 'file' and f['path'].lower().endswith('.gguf'): out.append(dict(file=f['path'], size_gb=round(((f.get('lfs') or {}).get('size') or f.get('size', 0)) / 1e9, 2)))
            elif f['type'] == 'directory' and d < 1: walk(f['path'], d + 1)
    walk('', 0); return sorted(out, key=lambda x: x['size_gb'])
def t_inspect(a):
    if a.get('repo'):
        need_repo(a['repo']); need_file(a['file'])
        url = 'https://huggingface.co/%s/resolve/main/%s' % (a['repo'], urllib.parse.quote(a['file']))
        s = read_header(url, {'Authorization': 'Bearer ' + hf_token()} if hf_token() else None)
        name = a['file']
    else:
        p = safe_model_path(a['path']); s = read_header(p); name = p
    if s['size'] and s['params']:
        pass
    return dict(summary={k: v for k, v in s.items() if k not in ('types',)}, verdict=assess(s, int(a.get('ctx', 1024)), int(a.get('cache_mb', 384)), meminfo().get('MemAvailable'), name))
def t_plan(a): return plan(float(a['params_b']), float(a.get('bits', 2.5)), float(a['active_b']) if a.get('active_b') else None, float(a.get('cache_gb', 2)))
def t_download(a):
    repo, file = need_repo(a['repo']), need_file(a['file']); os.makedirs(MODELS, exist_ok=True)
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
    p = safe_model_path(a['model'])
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
    ENG.reset(); return dict(reset=True)
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
    if not under(out, os.path.expanduser('~/packs')): raise ValueError('packs must be created inside ~/packs')
    p = subprocess.run([exe, '-o', out, safe_model_path(a['gguf'])], capture_output=True, text=True, timeout=3600)
    return dict(ok=p.returncode == 0, pack=out, log=(p.stdout + p.stderr)[-800:])
def t_serve(a):
    exe = os.path.join(ROOT, 'es_serve'); port = int(a.get('port', 8080))
    if not 1024 <= port <= 65535: raise ValueError('port must be 1024-65535')
    if not os.access(exe, os.X_OK): raise RuntimeError('es_serve not built; run make')
    m = safe_model_path(a['model'])
    subprocess.Popen([exe, '-g', m, '-p', str(port), '-w', os.path.join(ROOT, 'web')], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dict(started=True, url='http://127.0.0.1:%d' % port)
def oauth_post(path, data, js=False):
    body = json.dumps(data).encode() if js else urllib.parse.urlencode(data).encode()
    req = urllib.request.Request('https://huggingface.co' + path, data=body, headers={'Content-Type': 'application/json' if js else 'application/x-www-form-urlencoded', 'User-Agent': 'expertstream-mcp'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r: return json.load(r)
    except urllib.error.HTTPError as e:
        try: return json.load(e)
        except Exception: return dict(error='HTTP %d' % e.code)
DEVICE = {}
def t_connect(a):
    if a.get('token'): return t_set_token(a)
    c = cfg(); cid = c.get('oauth_client')
    if not cid:
        r = oauth_post('/oauth/register', dict(client_name='ExpertStream MCP', redirect_uris=['http://127.0.0.1/callback'], grant_types=['urn:ietf:params:oauth:grant-type:device_code', 'refresh_token'], token_endpoint_auth_method='none', scope='openid profile read-repos'), js=True)
        cid = r.get('client_id')
        if not cid: raise RuntimeError('Hugging Face did not register the client: %s' % r)
        c['oauth_client'] = cid; os.makedirs(os.path.dirname(CFG), exist_ok=True); json.dump(c, open(CFG, 'w')); os.chmod(CFG, 0o600)
    d = oauth_post('/oauth/device', dict(client_id=cid, scope='openid profile read-repos'))
    if 'device_code' not in d: raise RuntimeError('device sign-in failed: %s' % d)
    DEVICE.update(d, client_id=cid, t=time.time())
    return dict(step='Tell the user: open %s and enter the code %s, then click Allow. Then call connect_huggingface_finish.' % (d.get('verification_uri', 'https://hf.co/oauth/device'), d['user_code']), url=d.get('verification_uri'), code=d['user_code'], expires_in_s=d.get('expires_in'))
def t_connect_finish(a):
    if not DEVICE: return dict(error='call connect_huggingface first')
    deadline = time.time() + float(a.get('wait_s', 60))
    while time.time() < deadline:
        r = oauth_post('/oauth/token', dict(grant_type='urn:ietf:params:oauth:grant-type:device_code', device_code=DEVICE['device_code'], client_id=DEVICE['client_id']))
        if r.get('access_token'):
            c = cfg(); c['hf_token'] = r['access_token']; c['hf_refresh'] = r.get('refresh_token'); json.dump(c, open(CFG, 'w')); os.chmod(CFG, 0o600); DEVICE.clear()
            try: who = hf_json('https://huggingface.co/api/whoami-v2').get('name')
            except Exception: who = None
            return dict(connected=True, user=who)
        if r.get('error') not in ('authorization_pending', 'slow_down'): return dict(connected=False, error=r)
        time.sleep(5)
    return dict(connected=False, pending=True, hint='the user has not approved yet; call again')
def t_set_token(a):
    if not RE_HFTOKEN.match(a.get('token', '')): raise ValueError('that does not look like a Hugging Face token (hf_...)')
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
 ('connect_huggingface', 'Link the user\'s Hugging Face account (for private/gated models). Without arguments it starts a one-step sign-in: give the user the URL and code it returns, then call connect_huggingface_finish. Or pass a token directly.', S(token=P('string', 'optional hf_... token')), t_connect),
 ('connect_huggingface_finish', 'Wait for the user to approve the Hugging Face sign-in started by connect_huggingface and save the access.', S(wait_s=P('number', 'seconds to wait, default 60')), t_connect_finish),
]
TOOLS += [
 ('list_agents', 'Browse the 250+ built-in agent presets (coding languages, DevOps, security, data/ML, product, writing, daily life, phone/Termux, agent teams). Each runs on the loaded local model.', S(query=P('string', 'search words, e.g. "rust" or "postgres"'), category=P('string', 'category id: languages, engineering, devops, security, data, product, learning, life, mobile, orchestrators'), limit=P('integer', 'default 25')), t_list_agents),
 ('get_agent', 'Full definition (system prompt) of one agent.', S(agent_id=P('string', 'agent id', req=True)), t_get_agent),
 ('recommend_agents', 'Suggest the best agents for a task described in words.', S(task=P('string', 'what you need done', req=True), limit=P('integer', 'default 5')), t_recommend_agents),
 ('run_agent', 'Run one agent on a task using the loaded local model. Fresh context each call unless keep_context is true.', S(agent_id=P('string', 'agent id', req=True), task=P('string', 'the task or question', req=True), context=P('string', 'extra material such as code or notes'), keep_context=P('boolean', 'continue the previous conversation')), t_run_agent),
 ('run_pipeline', 'Run agents in sequence; each receives the previous agent\'s output (max 8). Example: planner -> python -> test-writer -> code-reviewer.', S(agent_ids=dict(type='array', items=dict(type='string'), description='agent ids in order'), task=P('string', 'the overall task', req=True)), t_pipeline),
 ('agent_council', 'Ask up to 5 agents the same question and merge their answers with the synthesizer agent.', S(task=P('string', 'question or task', req=True), agent_ids=dict(type='array', items=dict(type='string'), description='optional agent ids; default = best 3 matches')), t_council),
]
# ---------------- model builder: create language models from scratch (1k .. 37B parameters) ----------------
BUILDS = os.path.expanduser(os.environ.get('ES_BUILDS', '~/builds'))
BJOBS = {}
MAX_BUILD_B = 37.0
RULES = [
 "R1 SIZE: a model must have between 1,000 and 37,000,000,000 parameters. The tools refuse anything outside this range; do not try to work around it.",
 "R2 HONESTY: report only numbers returned by tools (loss, perplexity, tokens, seconds, parameters). Never invent results, never call a model 'trained' unless a train_model job finished and evaluate_model shows a lower loss than the untrained baseline.",
 "R3 LABELS: a model made by create_model is a random initialisation. Call it UNTRAINED. A model that has seen few tokens is 'barely trained'. Say what it can and cannot do.",
 "R4 DATA: train only on text the user wrote, owns, or has a licence to use. prepare_data needs i_have_the_right_to_use_this_data=true and writes a manifest (files, sizes, hashes). Do not scrape or use private data without consent. Never include secrets or personal data.",
 "R5 ORDER: design_model -> estimate_training -> prepare_data -> train_model -> evaluate_model -> (grow_model -> train_model again) -> export_model -> load_model + generate to test. Do not skip evaluate_model before export_model.",
 "R6 COST: before a long job, show estimate_training (time, memory, disk) and get the user's OK. Training uses 6*N*D FLOPs: say plainly when a target is too slow for this device and offer smaller models, growth from a small trained model, or running the same checkpoint on stronger hardware.",
 "R7 SAFETY: never overwrite a user's files; builds live in ~/builds/<name>; check free disk and RAM first; stop jobs with stop_job when asked.",
 "R8 PROVENANCE: export_model writes a model card (size, data manifest, steps, loss, date, UNTRAINED/trained status). Keep it with the model and quote it when describing the model.",
 "R9 NO MISUSE: do not build models intended to deceive, harass, or produce malware; refuse and explain.",
 "R10 REPORT: finish with a short report listing what was built, the real metrics, what the model is good for, and the next step to improve it.",
]
def _np():
    try: import builder as B
    except Exception as e: raise RuntimeError('builder module failed to load: %s' % e)
    if B.np is None: raise RuntimeError('NumPy is needed for training. Termux: pkg install python-numpy   PC: pip install numpy')
    return B
def _bdir(name):
    if not re.match(r'^[A-Za-z0-9][\w.-]{0,63}$', str(name)): raise ValueError('name must be letters, digits, - _ . (max 64)')
    d = os.path.join(BUILDS, name); os.makedirs(d, exist_ok=True); return d
def _flops():
    try:
        B = _np(); a = B.np.random.rand(512, 512).astype('float32'); t = time.time(); n = 0
        while time.time() - t < 0.4: a @ a; n += 1
        return 2 * 512 ** 3 * n / (time.time() - t)
    except Exception: return 5e10
def _cfg(a, vocab=None):
    import builder as B
    if a.get('config'): c = dict(a['config']); c['params'] = B.count_params(c); return c
    return B.design_config(int(float(a['target_params'])), vocab=int(vocab or a.get('vocab_size', 4096)), ctx=int(a.get('context', 512)))
def t_rules(a): return dict(rules=RULES, tools=['design_model', 'estimate_training', 'prepare_data', 'train_model', 'job_status', 'stop_job', 'evaluate_model', 'grow_model', 'create_model', 'export_model'], builder_agents=[x['id'] for x in AGENTS['agents'] if x['category'] == 'builder'], limits=dict(min_params=1000, max_params=37_000_000_000))
def t_design(a):
    import builder as B; c = _cfg(a)
    est = B.estimate(c, int(float(a.get('tokens', 20 * c['params']))), flops=_flops(), ram_gb=(meminfo().get('MemAvailable') or 4000) / 1024, disk_free_gb=device()['disk_free_gb'])
    return dict(config=c, estimate=est, next='estimate_training / prepare_data / create_model')
def t_estimate(a):
    import builder as B; c = _cfg(a); fl = float(a.get('flops') or _flops())
    return B.estimate(c, int(float(a['tokens'])), flops=fl, ram_gb=(meminfo().get('MemAvailable') or 4000) / 1024, disk_free_gb=device()['disk_free_gb'])
def t_prepare(a):
    B = _np()
    if a.get('i_have_the_right_to_use_this_data') is not True: raise ValueError('Rule R4: set i_have_the_right_to_use_this_data=true only for text the user owns or may use.')
    d = _bdir(a['name']); texts, man = [], []
    import hashlib
    for pth in a['paths'][:200]:
        p = os.path.realpath(os.path.expanduser(pth))
        if not (under(p, os.path.expanduser('~')) or ALLOW_ANY_PATH): raise ValueError('data must be inside your home folder')
        if any(part.startswith('.') for part in os.path.relpath(p, os.path.expanduser('~')).split(os.sep)): raise ValueError('hidden folders (such as .ssh or .config) are never used as training data (Rule R4)')
        files = []
        if os.path.isdir(p):
            for r, ds, fs in os.walk(p):
                ds[:] = [d for d in ds if not d.startswith('.') and d not in ('node_modules', '__pycache__')]
                files += [os.path.join(r, f) for f in fs]
        else: files = [p]
        for f in sorted(files)[:5000]:
            b = os.path.basename(f).lower()
            if os.path.getsize(f) > 200 << 20 or b.startswith('.') or re.search(r'(id_rsa|id_ed25519|\.pem$|\.key$|\.p12$|secret|credential|passw|token|\.env)', b): continue
            try: t = open(f, encoding='utf-8', errors='ignore').read()
            except Exception: continue
            if len(t) < 20: continue
            texts.append(t); man.append(dict(file=f, bytes=len(t.encode('utf-8', 'ignore')), sha256=hashlib.sha256(t.encode('utf-8', 'ignore')).hexdigest()[:16]))
    text = '\n'.join(texts)
    if len(text) < 2000: raise ValueError('need at least 2,000 characters of text')
    tk = B.train_bpe(text[:int(a.get('tokenizer_sample_chars', 2_000_000))], vocab_size=int(a.get('vocab_size', 1024))); tok = B.Tok(tk)
    ids = B.np.array(tok.encode(text), dtype=B.np.int32); ids = B.np.concatenate([ids, B.np.array([tok.eos], dtype=B.np.int32)])
    B.np.save(os.path.join(d, 'data.npy'), ids); json.dump(tk, open(os.path.join(d, 'tokenizer.json'), 'w')); json.dump(dict(files=man, chars=len(text)), open(os.path.join(d, 'data_manifest.json'), 'w'), indent=1)
    ok = tok.decode(tok.encode(text[:2000])) == text[:2000]
    return dict(build_dir=d, files=len(man), characters=len(text), vocab=len(tk['tokens']), tokens=int(len(ids)), tokens_per_char=round(len(ids) / len(text), 3), tokenizer_roundtrip_exact=ok, next='train_model(name, target_params, steps)')
def _train_job(jid, B, name, c, steps, bs, T, lr, minutes, resume):
    j = BJOBS[jid]; d = os.path.join(BUILDS, name); data = B.np.load(os.path.join(d, 'data.npy')).astype('int64'); tk = json.load(open(os.path.join(d, 'tokenizer.json')))
    n = int(len(data) * 0.95); tr, va = data[:n], data[n:] if len(data) - n > T + 2 else data
    try:
        ck = os.path.join(d, 'ckpt.npz')
        if resume and os.path.exists(ck): P, c0 = B.load_ckpt(ck); c = {k: v for k, v in c0.items() if k != 'tok'}
        else: P = B.init_params(c, 1)
        j['baseline'] = B.evaluate(P, c, va, batches=4, B=4, T=min(T, 64)); t0 = time.time()
        def log(x): j.update(progress=x)
        r = B.train(P, c, tr, steps, B=bs, T=T, lr=lr, log=log, ckpt=None, stop=lambda: j.get('stop') or time.time() - t0 > minutes * 60)
        B.save_ckpt(ck, P, c); j['final'] = B.evaluate(P, c, va, batches=6, B=4, T=min(T, 64)); j['train'] = r
        json.dump(dict(config=c, steps_done=(j.get('progress') or {}).get('step', 0), tokens_seen=(j.get('progress') or {}).get('tokens', 0), baseline=j['baseline'], final=j['final'], trained_at=time.strftime('%Y-%m-%d %H:%M:%S')), open(os.path.join(d, 'train_state.json'), 'w'), indent=1)
        j['status'] = 'done'
    except Exception as e: j['status'] = 'error'; j['error'] = '%s: %s' % (type(e).__name__, e)
def t_train(a):
    B = _np(); name = a['name']; d = _bdir(name)
    if not os.path.exists(os.path.join(d, 'data.npy')): raise ValueError('run prepare_data first (Rule R5)')
    tk = json.load(open(os.path.join(d, 'tokenizer.json'))); resume = bool(a.get('resume')) and os.path.exists(os.path.join(d, 'ckpt.npz'))
    c = _cfg(a, vocab=len(tk['tokens'])) if (a.get('target_params') or a.get('config')) else None
    if c is None and not resume: raise ValueError('give target_params (or config), or resume=true')
    if c: c['vocab'] = len(tk['tokens']); c['params'] = B.count_params(c)
    n = c['params'] if c else 0; need_gb = 16 * n / 1e9; free_gb = (meminfo().get('MemAvailable') or 4000) / 1024
    if need_gb > free_gb * 0.8: raise ValueError('training needs about %.1f GB RAM (weights+gradients+Adam) but only %.1f GB is free. Use a smaller model, grow a smaller trained model, or export a checkpoint to stronger hardware (Rule R6).' % (need_gb, free_gb))
    bs, T, steps = int(a.get('batch_size', 16)), int(a.get('seq_len', 64)), int(a.get('steps', 300))
    jid = 'tr%d' % (len(BJOBS) + 1); BJOBS[jid] = dict(status='running', name=name, params=n, started=time.time())
    threading.Thread(target=_train_job, args=(jid, B, name, c, steps, bs, T, float(a.get('lr', 3e-3)), float(a.get('max_minutes', 20)), resume), daemon=True).start()
    return dict(started=True, job=jid, params=n, ram_needed_gb=round(need_gb, 2), next='job_status(job) until done, then evaluate_model')
def t_job(a):
    j = BJOBS.get(a['job'])
    if not j: return dict(error='unknown job', jobs=list(BJOBS))
    return {k: v for k, v in j.items() if k != 'stop'} | dict(elapsed_s=round(time.time() - j.get('started', time.time()), 1))
def t_stop(a):
    j = BJOBS.get(a['job']); 
    if j: j['stop'] = True
    return dict(stopping=bool(j))
def _load(name):
    B = _np(); d = os.path.join(BUILDS, name); P, c = B.load_ckpt(os.path.join(d, 'ckpt.npz')); c = {k: v for k, v in c.items() if k != 'tok'}; return B, d, P, c
def t_eval(a):
    B, d, P, c = _load(a['name']); tk = json.load(open(os.path.join(d, 'tokenizer.json'))); tok = B.Tok(tk); data = B.np.load(os.path.join(d, 'data.npy')).astype('int64'); va = data[int(len(data) * 0.95):]
    va = va if len(va) > 130 else data; ev = B.evaluate(P, c, va, batches=8, B=4, T=min(64, c['ctx']))
    st = json.load(open(os.path.join(d, 'train_state.json'))) if os.path.exists(os.path.join(d, 'train_state.json')) else {}
    base = (st.get('baseline') or {}).get('loss'); samples = [B.generate(P, c, tok, p, 40, 0.7, seed=i)[:160] for i, p in list(enumerate(a.get('prompts') or ['The', 'In the']))[:3]]
    return dict(params=B.count_params(c), heldout_loss=round(ev['loss'], 4), perplexity=round(ev['perplexity'], 2), untrained_baseline_loss=base, improved=(base is not None and ev['loss'] < base - 0.05), tokens_seen=st.get('tokens_seen'), samples=samples)
def t_grow(a):
    B, d, P, c = _load(a['name']); new = _bdir(a['new_name']); nl = int(a['new_layers'])
    if nl <= c['layers']: raise ValueError('new_layers must exceed %d' % c['layers'])
    Q, nc = B.grow_depth(P, c, nl); nc['params'] = B.count_params(nc)
    if nc['params'] > MAX_BUILD_B * 1e9: raise ValueError('grown model would exceed 37B parameters')
    ids = B.np.random.default_rng(0).integers(0, c['vocab'], (1, 16)); diff = float(B.np.abs(B.forward(P, c, ids)[0] - B.forward(Q, nc, ids)[0]).max())
    B.save_ckpt(os.path.join(new, 'ckpt.npz'), Q, nc)
    for f in ('tokenizer.json', 'data.npy', 'data_manifest.json'):
        if os.path.exists(os.path.join(d, f)): shutil.copy(os.path.join(d, f), new)
    return dict(new_params=nc['params'], layers=nc['layers'], max_logit_change=diff, function_preserved=diff < 1e-4, next='train_model(new_name, resume=true) so the new layers learn')
def t_create(a):
    B = _np(); c = _cfg(a, vocab=int(a.get('vocab_size', 32000)))
    if c['params'] > MAX_BUILD_B * 1e9: raise ValueError('over 37B')
    dtype = a.get('dtype', 'f16'); sz = c['params'] * (2 if dtype == 'f16' else 4); free = shutil.disk_usage(MODELS if os.path.isdir(MODELS) else os.path.expanduser('~')).free
    if sz * 1.05 > free: raise ValueError('needs %.1f GB of free disk, only %.1f GB free (Rule R7)' % (sz / 1e9, free / 1e9))
    name = a['name']; _bdir(name); path = os.path.join(MODELS, name + '.gguf'); os.makedirs(MODELS, exist_ok=True)
    if os.path.exists(path) and not a.get('overwrite'): raise ValueError('%s exists (Rule R7); choose another name' % path)
    tk = B_tok(c['vocab']); size = __import__('builder').write_gguf(path, c, tk, None, dtype, name)
    json.dump(dict(config=c, status='UNTRAINED random initialisation', created=time.strftime('%Y-%m-%d %H:%M:%S')), open(os.path.join(BUILDS, name, 'card.json'), 'w'), indent=1)
    return dict(path=path, params=c['params'], size_gb=round(size / 1e9, 2), status='UNTRAINED (random weights). It loads and runs but outputs noise until trained.', next='load_model, or prepare_data + train_model on a smaller proxy then grow_model')
def B_tok(vocab):
    import builder as B; tk = B.train_bpe('abcdefghijklmnopqrstuvwxyz ' * 40, vocab_size=300); tk['tokens'] = tk['tokens'][:-1] + ['<|endoftext|>']; return tk
def t_export(a):
    B, d, P, c = _load(a['name']); st = json.load(open(os.path.join(d, 'train_state.json'))) if os.path.exists(os.path.join(d, 'train_state.json')) else {}
    dtype = a.get('dtype', 'f16'); tk = json.load(open(os.path.join(d, 'tokenizer.json'))); path = os.path.join(MODELS, a['name'] + '.gguf'); os.makedirs(MODELS, exist_ok=True)
    if os.path.exists(path) and not a.get('overwrite'): raise ValueError('%s exists (Rule R7)' % path)
    trained = bool(st.get('steps_done')); size = B.write_gguf(path, c, tk, P, dtype, a['name'])
    card = '# %s\n\n- parameters: %s\n- status: %s\n- tokens seen: %s\n- baseline loss: %s -> final loss: %s\n- architecture: llama-style, %d layers, dim %d, %d heads, context %d\n- data manifest: %s\n- built with ExpertStream Builder on %s\n' % (a['name'], f"{B.count_params(c):,}", 'TRAINED' if trained else 'UNTRAINED (random)', st.get('tokens_seen'), (st.get('baseline') or {}).get('loss'), (st.get('final') or {}).get('loss'), c['layers'], c['dim'], c['heads'], c['ctx'], os.path.join(d, 'data_manifest.json'), time.strftime('%Y-%m-%d'))
    open(os.path.join(d, 'MODEL_CARD.md'), 'w').write(card)
    return dict(path=path, size_mb=round(size / 1e6, 2), trained=trained, model_card=os.path.join(d, 'MODEL_CARD.md'), next='load_model(%s.gguf) then generate (use raw=true for a base model)' % a['name'])
def t_workflow(a):
    n = float(a.get('target_params', 5e6)); nm = a.get('name', 'my-model')
    if not 1000 <= n <= 37e9: raise ValueError('target must be 1e3..37e9 (Rule R1)')
    steps = [dict(agent='model-architect', tool='design_model', args=dict(target_params=n)), dict(agent='build-safety-reviewer', tool='builder_rules', note='read the rules and confirm data rights'),
             dict(agent='data-curator', tool='prepare_data', args=dict(name=nm, paths=['<user text files>'], vocab_size=1024, i_have_the_right_to_use_this_data='<true only with consent>')),
             dict(agent='training-engineer', tool='train_model', args=dict(name=nm, target_params=min(n, 2e7), steps=500)), dict(agent='model-evaluator', tool='evaluate_model', args=dict(name=nm))]
    if n > 2e7: steps += [dict(agent='model-scaler', tool='grow_model', args=dict(name=nm, new_name=nm + '-big', new_layers='<more layers>')), dict(agent='training-engineer', tool='train_model', args=dict(name=nm + '-big', resume=True))]
    steps += [dict(agent='model-exporter', tool='export_model', args=dict(name=nm)), dict(agent='build-lead', tool='load_model + generate', note='test it and write the R10 report')]
    return dict(goal=a.get('goal', ''), target_params=n, steps=steps, rules='call builder_rules; follow R1-R10', honesty=('Training cost ~ 6 x params x tokens FLOPs. For %.2g params at the Chinchilla ratio that is %.2g FLOPs; use estimate_training for this device.' % (n, 6 * n * 20 * n)))
TOOLS += [
 ('builder_rules', 'READ FIRST. The rules (R1-R10) an AI must follow when creating a language model from scratch with these tools: size limits 1,000 to 37 billion parameters, honesty about training, data rights, order of steps, reporting.', S(), t_rules),
 ('build_model_workflow', 'Step-by-step build plan for a target size, naming the builder agent and tool call for every step.', S(target_params=P('number', 'parameters, 1e3..3.7e10', req=True), name=P('string', 'build name'), goal=P('string', 'what the model is for')), t_workflow),
 ('design_model', 'Design a Llama-style architecture (layers, width, heads, FFN, vocab) for a parameter target between 1,000 and 37 billion, with memory/time estimates for this device.', S(target_params=P('number', 'target parameters', req=True), vocab_size=P('integer', 'default 4096'), context=P('integer', 'default 512'), tokens=P('number', 'training tokens to estimate for')), t_design),
 ('estimate_training', 'Estimate training time, RAM and disk (6*N*D FLOPs) for a model size and token budget on this device.', S(target_params=P('number', 'parameters'), tokens=P('number', 'training tokens', req=True), flops=P('number', 'override measured FLOP/s')), t_estimate),
 ('prepare_data', 'Build a training set from text files you have the right to use: trains a byte-level BPE tokenizer, tokenizes, writes a data manifest. Requires i_have_the_right_to_use_this_data=true.', S(name=P('string', 'build name', req=True), paths=dict(type='array', items=dict(type='string'), description='text files or folders inside your home'), vocab_size=P('integer', 'default 1024'), i_have_the_right_to_use_this_data=P('boolean', 'must be true', req=True)), t_prepare),
 ('train_model', 'Train a model from scratch (real backprop + Adam, runs in the background). Needs NumPy and prepare_data first. Poll with job_status.', S(name=P('string', 'build name', req=True), target_params=P('number', 'size, or use resume'), steps=P('integer', 'default 300'), batch_size=P('integer', 'default 16'), seq_len=P('integer', 'default 64'), lr=P('number', 'default 0.003'), max_minutes=P('number', 'stop after this long, default 20'), resume=P('boolean', 'continue from the checkpoint')), t_train),
 ('job_status', 'Progress of a training job: step, loss, tokens, seconds, final metrics.', S(job=P('string', 'job id', req=True)), t_job),
 ('stop_job', 'Stop a running training job (a checkpoint is saved).', S(job=P('string', 'job id', req=True)), t_stop),
 ('evaluate_model', 'Held-out loss, perplexity, improvement over the untrained baseline, and sample generations.', S(name=P('string', 'build name', req=True), prompts=dict(type='array', items=dict(type='string'))), t_eval),
 ('grow_model', 'Make a deeper model from a trained one without changing its answers (function-preserving), then continue training it. The practical path to larger models.', S(name=P('string', 'trained build', req=True), new_name=P('string', 'name of the grown build', req=True), new_layers=P('integer', 'total layers, more than now', req=True)), t_grow),
 ('create_model', 'Create a randomly initialised model of any size from 1,000 up to 37 billion parameters as a loadable GGUF (streamed to disk; checks free space). It is UNTRAINED.', S(target_params=P('number', 'parameters', req=True), name=P('string', 'file name', req=True), dtype=P('string', 'f16 or f32'), vocab_size=P('integer', 'default 32000'), overwrite=P('boolean', 'replace existing')), t_create),
 ('export_model', 'Export a trained checkpoint to GGUF in the models folder with a model card; load it with load_model.', S(name=P('string', 'build name', req=True), dtype=P('string', 'f16 or f32'), overwrite=P('boolean', 'replace existing')), t_export),
]

TOOLMAP = {n: f for n, d, s, f in TOOLS}

# ---------------- JSON-RPC ----------------
def handle(req):
    m, i = req.get('method'), req.get('id'); params = req.get('params') or {}
    if m == 'initialize':
        return dict(jsonrpc='2.0', id=i, result=dict(protocolVersion=params.get('protocolVersion', '2024-11-05'), capabilities=dict(tools=dict(listChanged=False), prompts=dict(listChanged=False)), serverInfo=dict(name='expertstream', version=VERSION), instructions='Run and manage local LLMs (up to %.0fB parameters) with the ExpertStream engine. Start with expertstream_status, then recommend_models. To CREATE a model from scratch (1 thousand to 37 billion parameters) call builder_rules first and follow rules R1-R10.' % MAX_B))
    if m == 'ping': return dict(jsonrpc='2.0', id=i, result={})
    if m == 'prompts/list': return dict(jsonrpc='2.0', id=i, result=dict(prompts=[dict(name='build-a-model', title='Build a language model from scratch', description='Rules and workflow for creating a model with the builder tools (1k to 37B parameters)', arguments=[dict(name='goal', description='what the model is for and its size', required=True)])] + [dict(name=a['id'], title=a['name'], description=a['description'], arguments=[dict(name='task', description='what you want this agent to do', required=True)]) for a in AGENTS['agents']]))
    if m == 'prompts/get' and params.get('name') == 'build-a-model':
        g = (params.get('arguments') or {}).get('goal', '')
        return dict(jsonrpc='2.0', id=i, result=dict(description='Build a model from scratch', messages=[dict(role='user', content=dict(type='text', text='You are the Build Lead. Goal: ' + g + '\n\nYou MUST follow these rules:\n' + '\n'.join(RULES) + '\n\nStart by calling builder_rules, then build_model_workflow, then follow the workflow with the other builder agents (list_agents category=builder). Ask the user before long jobs.'))]))
    if m == 'prompts/get':
        a = AGENT_BY_ID.get(params.get('name'))
        if not a: return dict(jsonrpc='2.0', id=i, error=dict(code=-32602, message='unknown prompt'))
        t = (params.get('arguments') or {}).get('task', '')
        return dict(jsonrpc='2.0', id=i, result=dict(description=a['description'], messages=[dict(role='user', content=dict(type='text', text=agent_prompt(a, t or '(describe your task here)')))]))
    if m == 'resources/list': return dict(jsonrpc='2.0', id=i, result=dict(resources=[]))
    if m == 'tools/list': return dict(jsonrpc='2.0', id=i, result=dict(tools=[dict(name=n, description=d, inputSchema=s) for n, d, s, f in TOOLS]))
    if m == 'tools/call':
        f = TOOLMAP.get(params.get('name'))
        if not f: return dict(jsonrpc='2.0', id=i, error=dict(code=-32602, message='unknown tool'))
        try: res = f(params.get('arguments') or {}); return dict(jsonrpc='2.0', id=i, result=dict(content=[dict(type='text', text=json.dumps(res, indent=1))]))
        except Exception as e: return dict(jsonrpc='2.0', id=i, result=dict(isError=True, content=[dict(type='text', text='%s: %s' % (type(e).__name__, e))]))
    if i is None: return None   # notification
    return dict(jsonrpc='2.0', id=i, error=dict(code=-32601, message='method not found'))
# ---------------- HTTP: MCP (streamable + legacy SSE), OpenAI-compatible API, health ----------------
import queue, secrets, socket
TOKEN = os.environ.get('ES_MCP_TOKEN', '')
TOKEN_FILE = os.path.expanduser('~/.expertstream/token')
ALLOWED_HOSTS = None      # set in main(): loopback binds only accept loopback Host headers (DNS-rebinding defence)
EXTRA_ORIGINS = [o for o in os.environ.get('ES_MCP_ORIGINS', '').split(',') if o]
FAILS = {}                # ip -> [count, first_time]
def load_token():
    try: t = open(TOKEN_FILE).read().strip()
    except Exception: t = ''
    if len(t) < 24:
        t = secrets.token_urlsafe(32); os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
        fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600); os.write(fd, t.encode()); os.close(fd)
    return t
def origin_ok(o):
    if not o: return True
    if o == 'https://tharunmaks.github.io' or o in EXTRA_ORIGINS: return True
    u = urllib.parse.urlparse(o)
    return u.scheme == 'http' and u.hostname in ('127.0.0.1', 'localhost', '::1')
SESSIONS = {}   # legacy SSE sessions: id -> queue
def flat(c):
    if isinstance(c, list): return '\n'.join(x.get('text', '') for x in c if isinstance(x, dict))
    return c or ''
def oa_prompt(messages, model):
    sysm = [flat(m.get('content')) for m in messages if m.get('role') in ('system', 'developer')]
    convo = [m for m in messages if m.get('role') not in ('system', 'developer')]
    parts = []
    ag = AGENT_BY_ID.get(model[6:]) if str(model).startswith('agent:') else None
    if ag: parts.append(ag['system'])
    if sysm: parts.append('Instructions:\n' + '\n'.join(sysm))
    if len(convo) > 1: parts.append('Conversation so far:\n' + '\n'.join('%s: %s' % (m.get('role', 'user').capitalize(), flat(m.get('content'))) for m in convo[:-1]))
    parts.append(flat(convo[-1].get('content')) if convo else '')
    return '\n\n'.join(p for p in parts if p)
def lan_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('10.255.255.255', 1)); ips.add(s.getsockname()[0]); s.close()
    except Exception: pass
    return sorted(ips)
class H(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'
    def log_message(self, *a): pass
    timeout = 30
    def _cors(self):
        o = self.headers.get('Origin', '')
        if o and origin_ok(o): self.send_header('Access-Control-Allow-Origin', o); self.send_header('Vary', 'Origin')
        self.send_header('X-Content-Type-Options', 'nosniff'); self.send_header('Cache-Control', 'no-store'); self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Access-Control-Allow-Headers', 'content-type, authorization, mcp-session-id, mcp-protocol-version, accept, last-event-id')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS'); self.send_header('Access-Control-Allow-Private-Network', 'true'); self.send_header('Access-Control-Expose-Headers', 'Mcp-Session-Id')
    def _send(self, code, obj=None, ctype='application/json', extra=None):
        b = b'' if obj is None else (obj if isinstance(obj, bytes) else json.dumps(obj).encode())
        self.send_response(code); self._cors(); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(b)))
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)
    def _path_token(self):
        """Connector-style URLs carry the secret in the path: https://host/s/<token>/mcp (apps that only accept a URL)."""
        self._pathauth = False
        if self.path.startswith('/s/'):
            parts = self.path.split('/', 3)
            tok = parts[2] if len(parts) > 2 else ''
            if TOKEN and secrets.compare_digest(tok.encode(), TOKEN.encode()):
                self._pathauth = True; self.path = '/' + (parts[3] if len(parts) > 3 else '')
            else:
                ip = self.client_address[0]; f = FAILS.setdefault(ip, [0, time.time()]); f[0] += 1; time.sleep(0.4)
                self.path = '/__bad_token__'
    def _guard(self):
        """Host and Origin checks that stop other web pages and DNS-rebinding from reaching a local server."""
        if ALLOWED_HOSTS is not None:
            host = self.headers.get('Host', '').rsplit(':', 1)[0].strip('[]') if not self.headers.get('Host', '').startswith('[') else '::1'
            if host not in ALLOWED_HOSTS: self._send(421, dict(error='unexpected Host header')); return False
        if not origin_ok(self.headers.get('Origin', '')) and urllib.parse.urlparse(self.path).path != '/health':
            self._send(403, dict(error='origin not allowed')); return False
        return True
    def _auth(self, count=True):
        if not TOKEN or getattr(self, '_pathauth', False): return True
        ip = self.client_address[0]; f = FAILS.get(ip)
        if f and time.time() - f[1] > 300: FAILS.pop(ip, None); f = None
        if f and f[0] >= 10: return False
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        hdr = self.headers.get('Authorization', '')
        got = (hdr[7:] if hdr.startswith('Bearer ') else '') or (q.get('token') or [''])[0]
        if secrets.compare_digest(got.encode(), TOKEN.encode()): FAILS.pop(ip, None); return True
        if count:
            f = FAILS.setdefault(ip, [0, time.time()]); f[0] += 1; time.sleep(0.4)
        return False
    def _deny(self):
        ip = self.client_address[0]
        if FAILS.get(ip, [0])[0] >= 10: return self._send(429, dict(error='too many wrong tokens; wait 5 minutes'))
        return self._send(401, dict(error='missing or wrong token (Authorization: Bearer <token>)'))
    def do_OPTIONS(self):
        self._path_token()
        if not self._guard(): return
        self.send_response(204); self._cors(); self.end_headers()
    def do_DELETE(self):
        self._path_token()
        if not self._guard(): return
        if not self._auth(): return self._deny()
        self._send(200, {})
    def do_GET(self):
        self._path_token()
        if not self._guard(): return
        u = urllib.parse.urlparse(self.path); path = u.path.rstrip('/') or '/'
        if path == '/health':
            info = dict(ok=True, server='expertstream-mcp', version=VERSION, tools=len(TOOLS), agents=AGENTS['count'], max_params_b=MAX_B, auth=bool(TOKEN))
            if self._auth(count=False): info['device'] = device(); info['loaded'] = ENG.path
            return self._send(200, info)
        if path.startswith('/.well-known/'): return self._send(404, dict(error='no OAuth here; the secret is in the connector URL'))
        if not self._auth(): return self._deny()
        if path == '/': return self._send(200, ('ExpertStream MCP %s\n\nMCP (streamable HTTP):  POST /mcp\nMCP (legacy SSE):       GET /sse  +  POST /messages\nOpenAI-compatible API:  POST /v1/chat/completions   GET /v1/models\nAgents:                 GET /agents   (model name "agent:<id>")\nHealth:                 GET /health\n' % VERSION).encode(), 'text/plain')
        if path == '/agents': return self._send(200, dict(total=AGENTS['count'], categories=AGENTS['categories'], agents=[agent_brief(a) for a in AGENTS['agents']]))
        if path == '/v1/models': return self._send(200, dict(object='list', data=[dict(id='expertstream', object='model', owned_by='local')] + [dict(id='agent:' + a['id'], object='model', owned_by='expertstream-agents') for a in AGENTS['agents']]))
        if path == '/sse':
            if len(SESSIONS) >= 32: return self._send(503, dict(error='too many open sessions'))
            sid = secrets.token_hex(16); qu = queue.Queue(); SESSIONS[sid] = qu
            self.send_response(200); self._cors(); self.send_header('Content-Type', 'text/event-stream'); self.send_header('Cache-Control', 'no-cache'); self.end_headers()
            try:
                self.wfile.write(('event: endpoint\ndata: /messages?session_id=%s\n\n' % sid).encode()); self.wfile.flush()
                while True:
                    try: m = qu.get(timeout=15); self.wfile.write(('event: message\ndata: %s\n\n' % json.dumps(m)).encode())
                    except queue.Empty: self.wfile.write(b': ping\n\n')
                    self.wfile.flush()
            except Exception: pass
            finally: SESSIONS.pop(sid, None)
            return
        if path == '/mcp': return self._send(405, dict(error='use POST'))
        if path.startswith('/.well-known/'): return self._send(404, dict(error='no OAuth here; the secret is in the connector URL'))
        self._send(404, dict(error='not found'))
    def _body(self):
        n = int(self.headers.get('Content-Length', 0))
        if n < 0 or n > MAX_BODY: raise OverflowError('body too large')
        return json.loads(self.rfile.read(n) or b'{}')
    def do_POST(self):
        self._path_token()
        if not self._guard(): return
        if not self._auth(): return self._deny()
        u = urllib.parse.urlparse(self.path); path = u.path.rstrip('/')
        try: body = self._body()
        except OverflowError: return self._send(413, dict(error='request too large (limit 4 MB)'))
        except Exception: return self._send(400, dict(error='bad JSON'))
        if path == '/mcp':
            batch = isinstance(body, list); rs = [r for r in (handle(x) for x in (body if batch else [body])) if r]
            extra = {'Mcp-Session-Id': secrets.token_hex(16)} if not batch and body.get('method') == 'initialize' else {}
            if not rs: return self._send(202, None, extra=extra)
            return self._send(200, rs if batch else rs[0], extra=extra)
        if path == '/messages':
            qu = SESSIONS.get((urllib.parse.parse_qs(u.query).get('session_id') or [''])[0])
            if not qu: return self._send(404, dict(error='unknown session'))
            r = handle(body)
            if r: qu.put(r)
            return self._send(202, None)
        if path == '/v1/chat/completions': return self._openai(body)
        self._send(404, dict(error='not found'))
    def _openai(self, body):
        model = body.get('model') or 'expertstream'
        if not ENG.p: return self._send(503, dict(error=dict(message='no model is loaded. Start the server with --model FILE or call load_model first.', type='unavailable')))
        prompt = oa_prompt(body.get('messages') or [], model); cid = 'chatcmpl-' + secrets.token_hex(6); now = int(time.time())
        if body.get('stream'):
            self.send_response(200); self._cors(); self.send_header('Content-Type', 'text/event-stream'); self.send_header('Cache-Control', 'no-cache'); self.end_headers()
            def chunk(delta, fin=None): self.wfile.write(('data: %s\n\n' % json.dumps(dict(id=cid, object='chat.completion.chunk', created=now, model=model, choices=[dict(index=0, delta=delta, finish_reason=fin)]))).encode()); self.wfile.flush()
            try:
                chunk(dict(role='assistant', content=''))
                ENG.ask(prompt, on_tok=lambda t: chunk(dict(content=t)), fresh=True)
                chunk({}, 'stop'); self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
            except Exception: pass
            return
        try: text, st = ENG.ask(prompt, fresh=True)
        except Exception as e: return self._send(500, dict(error=dict(message=str(e))))
        n = st.get('tokens') or 0
        self._send(200, dict(id=cid, object='chat.completion', created=now, model=model, choices=[dict(index=0, message=dict(role='assistant', content=text), finish_reason='stop')], usage=dict(prompt_tokens=0, completion_tokens=n, total_tokens=n)))
def start_tunnel(port, ready):
    """Open a public HTTPS address for this server (needed by apps that want a domain, not a config file)."""
    def reader(proc, rx):
        for line in proc.stderr if rx[1] else proc.stdout:
            m = re.search(rx[0], line)
            if m: ready(m.group(0)); return
    tries = []
    if shutil.which('cloudflared'): tries.append((['cloudflared', 'tunnel', '--url', 'http://127.0.0.1:%d' % port, '--no-autoupdate'], (r'https://[a-z0-9-]+\.trycloudflare\.com', True)))
    if shutil.which('ssh'): tries.append((['ssh', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ServerAliveInterval=30', '-R', '80:127.0.0.1:%d' % port, 'nokey@localhost.run'], (r'https://[a-z0-9]+\.lhr\.life', False)))
    if not tries: raise RuntimeError('no tunnel tool found. Termux: pkg install cloudflared   PC: install cloudflared (free) or use ssh')
    cmd, rx = tries[0]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT if not rx[1] else subprocess.PIPE, text=True)
    threading.Thread(target=reader, args=(proc, rx), daemon=True).start(); return proc
def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
def main():
    global TOKEN, ALLOWED_HOSTS
    host = arg('--host', '127.0.0.1'); loop = host in ('127.0.0.1', 'localhost', '::1')
    if arg('--token'):
        TOKEN = arg('--token')
        if len(TOKEN) < 16: sys.exit('--token must be at least 16 characters')
    elif '--no-auth' in sys.argv:
        if not loop: sys.exit('--no-auth is only allowed on 127.0.0.1')
        TOKEN = ''
    elif not TOKEN: TOKEN = load_token()
    public = '--public' in sys.argv or arg('--public-url')
    if public:
        if not re.match(r'^[A-Za-z0-9_-]{16,}$', TOKEN or ''): TOKEN = load_token() if not TOKEN else TOKEN
        if not re.match(r'^[A-Za-z0-9_-]{16,}$', TOKEN): sys.exit('--public needs a token of letters, digits, - and _ (16+ characters)')
    ALLOWED_HOSTS = None if (public or not loop) else {'127.0.0.1', 'localhost', '::1'}
    if arg('--model'):
        try: e = ENG.load(safe_model_path(arg('--model')), int(arg('--ctx', 4096)), int(arg('--cache', 1024)), float(arg('--temp', 0.7)), int(arg('--max-tokens', 512))); print('Loaded model %s' % e.get('model'), file=sys.stderr)
        except Exception as ex: print('Could not load model: %s' % ex, file=sys.stderr)
    if '--http' in sys.argv:
        port = int(arg('--http')); srv = http.server.ThreadingHTTPServer((host, port), H); srv.daemon_threads = True
        scheme = 'http'
        if arg('--cert') and arg('--key'):
            import ssl; ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.minimum_version = ssl.TLSVersion.TLSv1_2; ctx.load_cert_chain(arg('--cert'), arg('--key')); srv.socket = ctx.wrap_socket(srv.socket, server_side=True); scheme = 'https'
        elif not loop: print('WARNING: plain http on a network. Anyone on it can read the token. Use --cert/--key, a VPN, or an SSH tunnel.', file=sys.stderr)
        for ip in (lan_ips() if host == '0.0.0.0' else [host]):
            print('ExpertStream MCP  %s://%s:%d/mcp   (OpenAI API: %s://%s:%d/v1)' % (scheme, ip, port, scheme, ip, port), file=sys.stderr)
        if TOKEN: print('Access token is in %s%s (send it as "Authorization: Bearer <token>")' % (TOKEN_FILE, '' if TOKEN == load_token() else ' / your --token'), file=sys.stderr)
        def show(base):
            base = base.rstrip('/')
            print('\n  PASTE THIS INTO YOUR AI APP (custom connector / MCP server URL):\n    %s/s/%s/mcp\n  OpenAI-compatible base URL:\n    %s/s/%s/v1\n  Keep it private: the link contains your access token.\n' % (base, TOKEN, base, TOKEN), file=sys.stderr)
        if arg('--public-url'): show(arg('--public-url'))
        elif '--public' in sys.argv:
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            try: start_tunnel(port, show)
            except Exception as ex: sys.exit('Could not open a public address: %s' % ex)
            if '--stdio' not in sys.argv:
                while True: time.sleep(3600)
        if '--stdio' not in sys.argv: srv.serve_forever(); return
        if '--public' not in sys.argv: threading.Thread(target=srv.serve_forever, daemon=True).start()
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try: r = handle(json.loads(line))
        except Exception as e: r = dict(jsonrpc='2.0', id=None, error=dict(code=-32700, message=str(e)))
        if r: sys.stdout.write(json.dumps(r) + '\n'); sys.stdout.flush()
if __name__ == '__main__': main()
