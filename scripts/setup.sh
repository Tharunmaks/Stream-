#!/usr/bin/env bash
# One-command setup for Termux: install, build, download + import OLMoE,
# start the web app and open it in the phone's browser.
#
#   curl -fsSL https://raw.githubusercontent.com/tharunmaks/stream-/claude/expertstream-moe-android-oegt59/scripts/setup.sh | bash
#   or, inside the repo:  bash scripts/setup.sh
#
# Safe to run again: finished steps are skipped.
set -euo pipefail
REPO_URL=https://github.com/tharunmaks/stream-.git
BRANCH=claude/expertstream-moe-android-oegt59
SRC=$HOME/Stream-
MODELS=$HOME/models
PACK=$HOME/packs/olmoe
GGUF=OLMoE-1B-7B-0924-Instruct-Q2_K.gguf
PORT=${PORT:-8080}

step() { printf '\n\033[1m[%s] %s\033[0m\n' "$1" "$2"; }

step 1/4 "Installing tools (clang, make, git, curl)"
if command -v pkg >/dev/null 2>&1; then
    pkg install -y clang make git curl >/dev/null
else
    echo "not Termux: make sure clang (or gcc), make, git and curl are installed"
fi

step 2/4 "Getting and building ExpertStream"
if [ ! -d "$SRC/.git" ]; then
    git clone -q -b "$BRANCH" "$REPO_URL" "$SRC"
else
    git -C "$SRC" checkout -q -- . 2>/dev/null || true
    git -C "$SRC" checkout -q "$BRANCH"
    git -C "$SRC" pull -q --ff-only origin "$BRANCH"
fi
make -s -C "$SRC"

step 3/4 "Getting the model (OLMoE-1B-7B, 2.6 GB download, done once)"
if [ -f "$PACK/manifest.txt" ] && [ -f "$PACK/core.gguf" ]; then
    echo "already imported: $PACK"
else
    [ -f "$MODELS/$GGUF" ] || bash "$SRC/scripts/hf_get.sh" bartowski/OLMoE-1B-7B-0924-Instruct-GGUF "$GGUF" "$MODELS"
    "$SRC/es_import" -o "$PACK" "$MODELS/$GGUF"
fi

step 4/4 "Starting the web app"
URL="http://127.0.0.1:$PORT"
"$SRC/es_serve" -m "$PACK" -p "$PORT" "$@" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT INT TERM
for _ in $(seq 1 120); do
    curl -fs "$URL/api/status" >/dev/null 2>&1 && break
    kill -0 $SERVER 2>/dev/null || { echo "the server stopped; see the message above"; exit 1; }
    sleep 0.5
done
if command -v termux-open-url >/dev/null 2>&1; then
    termux-open-url "$URL"
else
    echo "open $URL in a browser"
fi
echo "Keep Termux open while you chat. Press Ctrl+C here to stop."
wait $SERVER
