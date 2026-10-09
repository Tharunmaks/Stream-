#!/usr/bin/env bash
# Build the website: compile the engine to WebAssembly (Emscripten) and embed
# it in web/index.html, which works as a single self-contained file.
# docs/index.html is the same file, for GitHub Pages.
#
#   source /path/to/emsdk/emsdk_env.sh && bash scripts/build_web.sh
set -euo pipefail
cd "$(dirname "$0")/.."
command -v emcc >/dev/null || { echo "emcc not found: install and activate emsdk first"; exit 1; }
OUT=$(mktemp -d)
emcc -O3 -msimd128 -std=c11 -Wall -Wextra \
    web/es_wasm.c src/es_engine.c src/es_gguf.c src/es_quant.c src/es_tok.c src/es_q2k.c \
    src/es_cache.c src/es_io.c src/es_cpu.c src/es_compute.c \
    -o "$OUT/es_engine.js" \
    -sMODULARIZE=1 -sEXPORT_NAME=ESEngine -sENVIRONMENT=worker -sSINGLE_FILE=1 \
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sINITIAL_MEMORY=64MB -sSTACK_SIZE=2MB \
    -sEXPORTED_FUNCTIONS=_esw_init,_esw_error,_esw_info,_esw_sampling,_esw_reset,_esw_begin,_esw_ids,_esw_next,_esw_stats,_malloc,_free \
    -sEXPORTED_RUNTIME_METHODS=UTF8ToString,stringToNewUTF8,HEAPU8,HEAP32
python3 - "$OUT/es_engine.js" <<'PY'
import sys
eng = open(sys.argv[1], encoding='utf-8').read() + '\n' + open('web/worker_driver.js', encoding='utf-8').read()
assert '</script' not in eng
page = open('web/app_template.html', encoding='utf-8').read().replace('@@ENGINE_JS@@', eng)
for out in ('web/index.html', 'docs/index.html'):
    open(out, 'w', encoding='utf-8').write(page)
print('web/index.html and docs/index.html:', len(page.encode()) // 1024, 'KB')
PY
rm -rf "$OUT"
