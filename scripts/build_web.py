#!/usr/bin/env python3
"""Assemble the website: web/src/template.html + style + JS modules + the WebAssembly engine
-> web/index.html and docs/index.html (single self-contained file).
usage: build_web.py ENGINE_JS [--verify-q3 "text"]"""
import glob, sys, os
root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
rd = lambda p: open(p if os.path.isabs(p) else os.path.join(root, p), encoding='utf-8').read()
engine = rd(sys.argv[1]) + '\n' + rd('web/src/workers/engine_driver.js')
dl = rd('web/src/workers/dl_worker.js')
engine_mt = (rd(sys.argv[sys.argv.index('--mt') + 1]) + '\n' + rd('web/src/workers/engine_driver.js')) if '--mt' in sys.argv else ''
js = '\n'.join(rd(p) for p in sorted(glob.glob(os.path.join(root, 'web/src/js/*.js'))))
q3 = 'not compared yet'
if '--verify-q3' in sys.argv: q3 = sys.argv[sys.argv.index('--verify-q3') + 1]
js = js.replace('@@Q3VERIFY@@', q3)
for name, txt in (('engine mt', engine_mt), ('engine', engine), ('dl worker', dl), ('app js', js)):
    assert '</script' not in txt.lower() and '\0' not in txt, name + ' contains </script or NUL'
page = rd('web/src/template.html').replace('@@STYLE@@', rd('web/src/style.css')).replace('@@DLWORKER@@', dl).replace('@@ENGINE_MT@@', engine_mt).replace('@@ENGINE@@', engine).replace('@@JS@@', js)
for out in ('web/index.html', 'docs/index.html'):
    open(os.path.join(root, out), 'w', encoding='utf-8').write(page)
import shutil
for d in ('web', 'docs'): shutil.copy(os.path.join(root, 'web/src/coi-sw.js'), os.path.join(root, d, 'coi-sw.js'))
print('built web/index.html + docs/index.html:', len(page.encode()) // 1024, 'KB')
