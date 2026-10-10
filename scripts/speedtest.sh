#!/usr/bin/env bash
# One-command speed check for Termux / Linux: builds the engine, downloads two small models
# (SmolLM2-135M, Q4_0 = 92 MB and Q4_K_M = 105 MB) and prints tokens/second for each setting.
#   bash scripts/speedtest.sh
# Paste the whole output back so the kernels can be tuned for your exact phone.
set -euo pipefail
cd "$(dirname "$0")/.."
M=${ES_MODELS:-$HOME/models}
REPO=bartowski/SmolLM2-135M-Instruct-GGUF
make es_chat >/dev/null
echo "== device"; uname -m; grep -m1 -i features /proc/cpuinfo || true
grep -i 'CPU part' /proc/cpuinfo | sort | uniq -c || true
for q in Q4_0 Q4_K_M; do
    [ -s "$M/SmolLM2-135M-Instruct-$q.gguf" ] || bash scripts/hf_get.sh "$REPO" "-$q.gguf" "$M" >/dev/null
done
P="Write a short story about a robot who learns to paint, in detail."
run() {  # run MODEL LABEL ARGS...
    local f=$1 label=$2; shift 2
    local out; out=$(./es_chat -g "$f" -p "$P" -n 160 -t 0 "$@" 2>&1 >/dev/null | grep -o 'tokens in [0-9.]* s = [0-9.]*' | sed 's/.*= //' | tr '\n' ' ')
    printf '%-10s %-28s %s tok/s\n' "$(basename "$f" .gguf | sed 's/SmolLM2-135M-Instruct-//')" "$label" "$out"
}
echo "== speed (3 runs each; greedy; threads: auto = one per fast core)"
for q in Q4_0 Q4_K_M; do
    f="$M/SmolLM2-135M-Instruct-$q.gguf"
    run "$f" "auto threads, Turbo off" -S 0; run "$f" "auto threads, Turbo off" -S 0; run "$f" "auto threads, Turbo off" -S 0
    run "$f" "auto threads, Turbo on";  run "$f" "auto threads, Turbo on"
    run "$f" "4 threads, Turbo off" -j 4 -S 0
    run "$f" "8 threads, Turbo off" -j 8 -S 0
done
