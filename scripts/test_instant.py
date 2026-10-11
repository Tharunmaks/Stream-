#!/usr/bin/env python3
"""Tests for ExpertStream Instant (es_serve). Python standard library only.

    make es_serve && python3 scripts/test_instant.py MODEL.gguf

Starts its own servers on free ports, then checks:
  1. OpenAI API: models, chat (plain and streaming), text completions, stop strings, max_tokens, content parts
  2. Ollama API: tags, version, chat (streaming NDJSON), generate
  3. prefix reuse: the second turn reads only the new tokens, and the answers are the same as with reuse turned off
  4. disk cache: after a restart the same prompt is reused from disk, and the answer is the same
  5. a conversation longer than the context is trimmed (oldest messages first) instead of failing
  6. access rules: other websites get 403, the project site and local pages get CORS + Private Network Access, foreign Host names are
     refused (DNS rebinding), with --lan other devices need the API key
  7. the bundled web app protocol (/api/chat text, /api/reset) still works
"""
import json, os, shutil, socket, subprocess, sys, tempfile, time, urllib.error, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVE = os.path.join(ROOT, 'es_serve')
if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]): sys.exit('usage: test_instant.py MODEL.gguf  (a small chat model, e.g. SmolLM2-135M-Instruct)')
if not os.path.exists(SERVE): sys.exit('build es_serve first: make es_serve')
MODEL = sys.argv[1]
ok = True
def check(name, cond, detail=''):
    global ok; ok &= bool(cond); print(('PASS ' if cond else 'FAIL ') + name + (('  ' + str(detail)) if detail else ''), flush=True)

def free_port():
    s = socket.socket(); s.bind(('127.0.0.1', 0)); p = s.getsockname()[1]; s.close(); return p

def start(*args, env=None):
    port = free_port(); e = dict(os.environ, **(env or {}))
    p = subprocess.Popen([SERVE, '-g', MODEL, '-p', str(port), '-c', '2048'] + list(args), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=e)
    for _ in range(600):
        try: urllib.request.urlopen('http://127.0.0.1:%d/health' % port, timeout=1).read(); return p, port
        except Exception: time.sleep(0.2)
    p.kill(); sys.exit('server did not start: ' + p.stderr.read().decode()[-500:])

def http(port, path, body=None, headers=None, method=None, host='127.0.0.1', raw=False):
    data = None if body is None else (body.encode() if isinstance(body, str) else json.dumps(body).encode())
    h = {'Content-Type': 'application/json'} if body is not None and not isinstance(body, str) else {}
    h.update(headers or {})
    req = urllib.request.Request('http://%s:%d%s' % (host, port, path), data=data, headers=h, method=method)
    try:
        r = urllib.request.urlopen(req, timeout=600); b = r.read().decode(); return r.status, dict(r.headers), (b if raw else (json.loads(b) if b.startswith(('{', '[')) else b))
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()

def chat(port, msgs, **kw):
    return http(port, '/v1/chat/completions', dict(model='m', messages=msgs, temperature=0, max_tokens=kw.pop('max_tokens', 32), **kw))[2]

def sse(port, body):
    s, h, text = http(port, '/v1/chat/completions', body, raw=True)
    events = [l[6:] for l in text.split('\n') if l.startswith('data: ')]
    return events

tmp = tempfile.mkdtemp(prefix='instant_test_')
procs = []
try:
    env = {'ES_KV_DIR': os.path.join(tmp, 'kv'), 'ES_KV_DISK_MIN': '64', 'ES_KV_DISK_MIN_S': '0'}
    p, port = start(env=env); procs.append(p)
    # ---- 1. OpenAI
    s, _, m = http(port, '/v1/models'); check('1a GET /v1/models lists the model', s == 200 and m['data'][0]['id'], m['data'][0]['id'])
    r = chat(port, [{'role': 'user', 'content': 'What is the capital of France?'}])
    txt = r['choices'][0]['message']['content']
    check('1b chat completion answers with usage', bool(txt.strip()) and r['usage']['completion_tokens'] > 0 and r['choices'][0]['finish_reason'] in ('stop', 'length'), repr(txt[:60]))
    ev = sse(port, dict(model='m', messages=[{'role': 'user', 'content': 'Say hello.'}], stream=True, max_tokens=16, temperature=0))
    chunks = [json.loads(e) for e in ev if e != '[DONE]']
    check('1c streaming: role first, content pieces, finish reason, [DONE]', ev[-1] == '[DONE]' and chunks[0]['choices'][0]['delta'].get('role') == 'assistant'
          and any(c['choices'][0]['delta'].get('content') for c in chunks if c['choices']) and chunks[-1]['choices'][0]['finish_reason'] is not None, '%d chunks' % len(chunks))
    s, _, c = http(port, '/v1/completions', dict(model='m', prompt='1, 2, 3, 4, 5, 6,', max_tokens=40, temperature=0, stop=['10']))
    check('1d text completion with a stop string ends before it', s == 200 and '10' not in c['choices'][0]['text'] and c['choices'][0]['finish_reason'] == 'stop', repr(c['choices'][0]['text']))
    r = chat(port, [{'role': 'user', 'content': 'Write a long story about a cat.'}], max_tokens=5)
    check('1e max_tokens stops with finish_reason length', r['choices'][0]['finish_reason'] == 'length' and r['usage']['completion_tokens'] == 5)
    r = chat(port, [{'role': 'user', 'content': [{'type': 'text', 'text': 'Say yes.'}]}])
    check('1f content given as parts is accepted', bool(r['choices'][0]['message']['content']))
    # ---- 2. Ollama
    s, _, t = http(port, '/api/tags'); s2, _, v = http(port, '/api/version')
    check('2a GET /api/tags and /api/version', s == 200 and t['models'] and s2 == 200 and 'version' in v)
    s, _, text = http(port, '/api/chat', dict(model='m', messages=[{'role': 'user', 'content': 'Say hi'}], options=dict(num_predict=12, temperature=0)), raw=True)
    lines = [json.loads(l) for l in text.splitlines() if l.strip()]
    check('2b Ollama chat streams NDJSON and ends with done:true', s == 200 and len(lines) > 1 and lines[-1]['done'] and ''.join(l['message']['content'] for l in lines).strip(), '%d lines' % len(lines))
    s, _, g = http(port, '/api/generate', dict(model='m', prompt='Name a color.', stream=False, options=dict(num_predict=8, temperature=0)))
    check('2c Ollama generate (not streaming)', s == 200 and g['done'] and g['eval_count'] > 0, repr(g.get('response', '')[:40]))
    # ---- 3. prefix reuse: fewer tokens read, same answers
    system = 'You are a careful assistant. ' + ' '.join('Rule %d: be brief and kind.' % i for i in range(40))
    conv = [{'role': 'system', 'content': system}]
    answers = []
    for q in ('Hi there!', 'What is two plus two?', 'Name one planet.'):
        conv.append({'role': 'user', 'content': q}); r = chat(port, conv); a = r['choices'][0]['message']['content']; answers.append((a, r['usage']))
        conv.append({'role': 'assistant', 'content': a})
    u2 = answers[1][1]
    check('3a turn 2 reuses the conversation and reads only the new tokens', u2['prompt_tokens_details']['cached_tokens'] >= u2['prompt_tokens'] - 32,
          '%d of %d prompt tokens reused' % (u2['prompt_tokens_details']['cached_tokens'], u2['prompt_tokens']))
    p2, port2 = start(env={'ES_PREFIX_CACHE': '0', 'ES_KV_DISK': '0'}); procs.append(p2)
    conv2 = [{'role': 'system', 'content': system}]; same = True
    for i, q in enumerate(('Hi there!', 'What is two plus two?', 'Name one planet.')):
        conv2.append({'role': 'user', 'content': q}); a = chat(port2, conv2)['choices'][0]['message']['content']; same &= a == answers[i][0]
        conv2.append({'role': 'assistant', 'content': a})
    check('3b answers with reuse are identical to answers without it (greedy)', same)
    side = chat(port, [{'role': 'user', 'content': 'Write a 3-word title.'}], max_tokens=6)
    conv.append({'role': 'user', 'content': 'Thanks!'}); r = chat(port, conv)
    check('3c a side request in between does not evict the conversation (parked in RAM)', r['usage']['prompt_tokens_details']['cached_tokens'] >= r['usage']['prompt_tokens'] - 32,
          '%d of %d reused' % (r['usage']['prompt_tokens_details']['cached_tokens'], r['usage']['prompt_tokens']))
    # ---- 4. disk cache across a restart
    time.sleep(1.5)
    files = [f for f in os.listdir(env['ES_KV_DIR']) if f.endswith('.eskv')]
    p.terminate(); p.wait(); procs.remove(p)
    p, port = start(env=env); procs.append(p)
    r = chat(port, [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'Hi there!'}])
    check('4 after a restart the prompt is loaded from disk and the answer is the same', files and r['usage']['prompt_tokens_details']['cached_tokens'] > 100 and r['choices'][0]['message']['content'] == answers[0][0],
          '%d files, %d tokens from disk' % (len(files), r['usage']['prompt_tokens_details']['cached_tokens']))
    # ---- 5. too long for the context: oldest messages are dropped
    big = [{'role': 'user' if i % 2 == 0 else 'assistant', 'content': 'This is message number %d. ' % i * 20} for i in range(41)]
    s, _, r = http(port, '/v1/chat/completions', dict(model='m', messages=big, max_tokens=8, temperature=0))
    check('5 a conversation longer than the context still gets an answer (oldest messages dropped)', s == 200 and r['usage']['prompt_tokens'] < 2048, 'prompt tokens used %s' % (r.get('usage', {}).get('prompt_tokens') if isinstance(r, dict) else r))
    # ---- 6. access rules
    s, _, _ = http(port, '/v1/models', headers={'Origin': 'https://evil.example'})
    check('6a another website is refused (403)', s == 403)
    s, h, _ = http(port, '/v1/models', headers={'Origin': 'https://tharunmaks.github.io'})
    check('6b the project website gets CORS', s == 200 and h.get('Access-Control-Allow-Origin') == 'https://tharunmaks.github.io')
    s, h, _ = http(port, '/v1/chat/completions', method='OPTIONS', headers={'Origin': 'http://localhost:3000', 'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Private-Network': 'true'})
    check('6c a local page passes the preflight, with Private Network Access', s == 204 and h.get('Access-Control-Allow-Private-Network') == 'true')
    s, _, _ = http(port, '/v1/models', headers={'Host': 'evil.example:%d' % port})
    check('6d a foreign Host name is refused (DNS rebinding)', s == 403)
    ip = None
    try:
        u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); u.connect(('10.255.255.255', 1)); ip = u.getsockname()[0]; u.close()
    except OSError: pass
    if ip and not ip.startswith('127.'):
        p3, port3 = start('--lan', '--key', 'test-key-123', env={'ES_KV_DISK': '0'}); procs.append(p3)
        s1, _, _ = http(port3, '/v1/models', host=ip)
        s2, _, _ = http(port3, '/v1/models', host=ip, headers={'Authorization': 'Bearer test-key-123'})
        s3, _, _ = http(port3, '/v1/models')
        check('6e --lan: another device needs the key, this device does not', s1 == 401 and s2 == 200 and s3 == 200, 'without key %d, with key %d, local %d' % (s1, s2, s3))
    else: print('SKIP 6e (no network address)')
    # ---- 7. the bundled web app
    s, _, text = http(port, '/api/chat', 'Hello!', headers={'Content-Type': 'text/plain'}, raw=True)
    evs = [json.loads(l) for l in text.splitlines() if l.strip()]
    kinds = [e['ev'] for e in evs]
    check('7a web app protocol: tokens, prompt, tok and done events', s == 200 and 'tokens' in kinds and 'tok' in kinds and kinds[-1] == 'done' and 'tps' in evs[-1], ','.join(sorted(set(kinds))))
    s, _, text = http(port, '/api/reset', '', method='POST', raw=True)
    check('7b web app reset', s == 200 and 'new chat' in text)
finally:
    for p in procs:
        p.terminate()
        try: p.wait(5)
        except Exception: p.kill()
    shutil.rmtree(tmp, ignore_errors=True)
print('ALL PASSED' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
