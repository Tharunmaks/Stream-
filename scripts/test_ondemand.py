#!/usr/bin/env python3
"""Experts on demand (models bigger than the device's storage): es_get + es_chat against a local HTTP server.

    make es_chat es_get && python3 scripts/test_ondemand.py MOE_MODEL.gguf     (e.g. OLMoE-1B-7B Q2_K, 2.6 GB)

The model is served over HTTP with byte ranges (as Hugging Face does). es_get downloads only the core; es_chat then
downloads each routed expert the first time it is chosen. Checks: the core is much smaller than the model, the logits are
bit-identical to the full local file, ES_OFFLINE runs without downloading, and ES_DISK_GB keeps the experts within budget.
"""
import http.server
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]): sys.exit(__doc__)
SRC = os.path.abspath(sys.argv[1])
NAME = os.path.basename(SRC)


class Range(http.server.BaseHTTPRequestHandler):
    def _file(self):
        return SRC if os.path.basename(self.path) == NAME else None

    def do_HEAD(self):
        if not self._file(): self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header('Content-Length', str(os.path.getsize(SRC))); self.end_headers()

    def do_GET(self):
        if not self._file(): self.send_response(404); self.end_headers(); return
        size = os.path.getsize(SRC)
        m = re.match(r'bytes=(\d+)-(\d+)', self.headers.get('Range', ''))
        a, b = (int(m[1]), min(int(m[2]), size - 1)) if m else (0, size - 1)
        self.send_response(206 if m else 200); self.send_header('Content-Length', str(b - a + 1)); self.end_headers()
        with open(SRC, 'rb') as f:
            f.seek(a); n = b - a + 1
            while n > 0:
                d = f.read(min(n, 1 << 20)); self.wfile.write(d); n -= len(d)

    def log_message(self, *a): pass


srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Range)
threading.Thread(target=srv.serve_forever, daemon=True).start()
env = dict(os.environ, HF_ENDPOINT='http://127.0.0.1:%d' % srv.server_address[1], ES_SKIP='0', ES_FLOAT_ACT='1')
work = tempfile.mkdtemp(prefix='es_ondemand_')
bad = 0


def check(name, ok, info=''):
    global bad
    bad += not ok
    print('%s %s  %s' % ('PASS' if ok else 'FAIL', name, info))


def chat(model, extra_env=None, ids='1,2,3,4,5,6,7,8', dump=None):
    e = dict(env, **(extra_env or {}))
    cmd = [os.path.join(ROOT, 'es_chat'), '-g', model, '-W', '0', '-j', '4', '-I', ids, '-D', dump]
    return subprocess.run(cmd, env=e, capture_output=True, text=True)


def used_bytes(p): return os.stat(p).st_blocks * 512


def experts_here(p):   # the engine's map of downloaded experts
    try: return sum(open(p + '.eshave', 'rb').read())
    except OSError: return 0


try:
    r = subprocess.run([os.path.join(ROOT, 'es_get'), 'test/repo', NAME, '-d', work], env=env, capture_output=True, text=True)
    local = os.path.join(work, NAME)
    check('1 es_get downloads only the core', r.returncode == 0 and used_bytes(local) < 0.5 * os.path.getsize(SRC),
          '%.0f MB of %.0f MB' % (used_bytes(local) / 1e6, os.path.getsize(SRC) / 1e6))
    full, od = os.path.join(work, 'full.bin'), os.path.join(work, 'od.bin')
    chat(SRC, dump=full)
    r = chat(local, dump=od)
    same = os.path.exists(od) and open(full, 'rb').read() == open(od, 'rb').read()
    check('2 logits identical to the full model', same, r.stderr.strip().splitlines()[-1] if not same and r.stderr else '')
    before = experts_here(local)
    r = chat(local, {'ES_OFFLINE': '1'}, ids='9,10,11,12,13,14,15,16,17,18,19,20', dump=od)
    check('3 ES_OFFLINE downloads nothing', r.returncode == 0 and before > 0 and experts_here(local) == before, '%d experts here' % before)
    shutil.rmtree(work); os.makedirs(work)
    subprocess.run([os.path.join(ROOT, 'es_get'), 'test/repo', NAME, '-d', work], env=env, capture_output=True)
    core = used_bytes(local)
    budget = 0.15 * (os.path.getsize(SRC) - core)
    r = chat(local, {'ES_DISK_GB': '%.3f' % (budget / 1e9)}, ids=','.join(str(i) for i in range(100, 140)), dump=od)
    check('4 ES_DISK_GB keeps experts within budget', r.returncode == 0 and used_bytes(local) - core <= budget * 1.05 + 64e6,
          'experts on disk %.0f MB, budget %.0f MB' % ((used_bytes(local) - core) / 1e6, budget / 1e6))
finally:
    srv.shutdown()
    shutil.rmtree(work, ignore_errors=True)
print('ALL PASSED' if not bad else '%d FAILED' % bad)
sys.exit(1 if bad else 0)
