#!/usr/bin/env bash
# Build the website: compile the engine to WebAssembly (Emscripten), then assemble
# web/index.html (and docs/index.html for GitHub Pages) from web/src.
#   source /path/to/emsdk/emsdk_env.sh && bash scripts/build_web.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/wasm
mkdir -p "$OUT"
if [ -z "${UI_ONLY:-}" ]; then
command -v emcc >/dev/null || { echo "emcc not found: install and activate emsdk first"; exit 1; }
emcc -O3 -msimd128 -std=c11 -Wall -Wextra \
    web/es_wasm.c src/es_engine.c src/es_gguf.c src/es_quant.c src/es_tok.c src/es_q2k.c \
    src/es_cache.c src/es_io.c src/es_cpu.c src/es_compute.c \
    -o "$OUT/es_engine.js" \
    -sMODULARIZE=1 -sEXPORT_NAME=ESEngine -sENVIRONMENT=worker -sSINGLE_FILE=1 -sSINGLE_FILE_BINARY_ENCODE=0 \
    -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=2GB -sINITIAL_MEMORY=64MB -sSTACK_SIZE=2MB \
    -sEXPORTED_FUNCTIONS=_esw_init,_esw_error,_esw_info,_esw_sampling,_esw_reset,_esw_begin,_esw_ids,_esw_next,_esw_stats,_malloc,_free \
    -sEXPORTED_RUNTIME_METHODS=UTF8ToString,stringToNewUTF8,HEAPU8,HEAP32,wasmMemory
# multi-threaded build (used when the page is cross-origin isolated)
emcc -O3 -msimd128 -pthread -std=c11 -Wall -Wextra \
    web/es_wasm.c src/es_engine.c src/es_gguf.c src/es_quant.c src/es_tok.c src/es_q2k.c \
    src/es_cache.c src/es_io.c src/es_cpu.c src/es_compute.c \
    -o "$OUT/es_engine_mt.js" \
    -sMODULARIZE=1 -sEXPORT_NAME=ESEngine -sENVIRONMENT=worker -sSINGLE_FILE=1 -sSINGLE_FILE_BINARY_ENCODE=0 \
    -sALLOW_MEMORY_GROWTH=1 -sPTHREAD_POOL_SIZE=7 -sMAXIMUM_MEMORY=2GB -sINITIAL_MEMORY=64MB -sSTACK_SIZE=2MB \
    -sEXPORTED_FUNCTIONS=_esw_init,_esw_error,_esw_info,_esw_sampling,_esw_reset,_esw_begin,_esw_ids,_esw_next,_esw_stats,_malloc,_free \
    -sEXPORTED_RUNTIME_METHODS=UTF8ToString,stringToNewUTF8,HEAPU8,HEAP32,wasmMemory
fi   # UI_ONLY=1 reuses build/wasm/es_engine.js
python3 scripts/build_web.py "$OUT/es_engine.js" --mt "$OUT/es_engine_mt.js" "$@"
