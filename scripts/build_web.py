#!/usr/bin/env python3
"""Assemble the website: web/src/template.html + style + JS modules + the WebAssembly engine
-> web/index.html and docs/index.html (single self-contained file).
usage: build_web.py ENGINE_JS [--verify-q3 "text"]"""
import glob, sys, os, re
root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
rd = lambda p: open(p if os.path.isabs(p) else os.path.join(root, p), encoding='utf-8').read()
engine = rd(sys.argv[1]) + '\n' + rd('web/src/workers/engine_driver.js')
dl = rd('web/src/workers/dl_worker.js')
engine_mt = (rd(sys.argv[sys.argv.index('--mt') + 1]) + '\n' + rd('web/src/workers/engine_driver.js')) if '--mt' in sys.argv else ''
def opt(flag):
    return (rd(sys.argv[sys.argv.index(flag) + 1]) + '\n' + rd('web/src/workers/engine_driver.js')) if flag in sys.argv else ''
engine_rs, engine_mt_rs = opt('--rs'), opt('--mt-rs')
js = '\n'.join(rd(p) for p in sorted(glob.glob(os.path.join(root, 'web/src/js/*.js'))))
q3 = 'not compared yet'
if '--verify-q3' in sys.argv: q3 = sys.argv[sys.argv.index('--verify-q3') + 1]
js = js.replace('@@Q3VERIFY@@', q3)
import json
ag = json.load(open(os.path.join(root, 'mcp/agents.json'), encoding='utf-8'))
ag['agents'] = [{k: a[k] for k in ('id', 'name', 'category', 'category_name', 'description', 'system', 'tags') if k in a} for a in ag['agents']]
js = js.replace('@@AGENTS@@', json.dumps(ag, ensure_ascii=False).replace('</', '<\\/'))
for name, txt in (('engine mt', engine_mt), ('engine rs', engine_rs), ('engine mt rs', engine_mt_rs), ('engine', engine), ('dl worker', dl), ('app js', js)):
    assert '</script' not in txt.lower() and '\0' not in txt, name + ' contains </script or NUL'
page = rd('web/src/template.html').replace('@@STYLE@@', rd('web/src/style.css')).replace('@@DLWORKER@@', dl).replace('@@ENGINE_MT_RS@@', engine_mt_rs).replace('@@ENGINE_RS@@', engine_rs).replace('@@ENGINE_MT@@', engine_mt).replace('@@ENGINE@@', engine).replace('@@JS@@', js)
import hashlib, base64
page = page.replace('@@FONTS@@', rd('web/src/fonts.css'))
hashes = ["'sha256-" + base64.b64encode(hashlib.sha256(m.encode('utf-8')).digest()).decode() + "'" for m in re.findall(r'<script>(.*?)</script>', page, re.S)]
CSP = "; ".join(["default-src 'none'", "script-src 'self' 'wasm-unsafe-eval' " + ' '.join(hashes), "worker-src 'self' blob:", "style-src 'unsafe-inline'", "font-src data:",
  "img-src 'self' data: blob: https://*.huggingface.co https://*.hf.co", "connect-src 'self' https://huggingface.co https://*.huggingface.co https://*.hf.co https://*.trycloudflare.com https://*.lhr.life http://127.0.0.1:* http://localhost:* http://*:8765",
  "base-uri 'none'", "form-action 'none'", "object-src 'none'", "frame-src 'none'", "manifest-src 'none'", "media-src 'none'"])
assert '"' not in CSP
page = page.replace('@@CSP@@', CSP)
for out in ('web/index.html', 'docs/index.html'):
    open(os.path.join(root, out), 'w', encoding='utf-8').write(page)
import shutil
sw = rd('web/src/coi-sw.js').replace('@@CSP@@', CSP + "; frame-ancestors 'none'")
for d in ('web', 'docs'): open(os.path.join(root, d, 'coi-sw.js'), 'w').write(sw)
os.makedirs(os.path.join(root, 'docs/.well-known'), exist_ok=True)
shutil.copy(os.path.join(root, 'web/src/wellknown/security.txt'), os.path.join(root, 'docs/.well-known/security.txt'))
print('CSP:', CSP[:140], '...')
print('built web/index.html + docs/index.html:', len(page.encode()) // 1024, 'KB')
