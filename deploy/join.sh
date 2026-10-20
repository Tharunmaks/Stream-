#!/usr/bin/env bash
# Adds this machine to your ExpertCloud as a node: free machines such as a Kaggle or Colab notebook, GitHub Codespaces or an
# Oracle Cloud Always Free VM give your cloud more computers.
#
#   bash join.sh https://YOUR-NAME-expertcloud.hf.space join-TOKEN [MODEL_REPO MODEL_FILE]
#
# In a notebook cell:
#   !curl -sL https://raw.githubusercontent.com/tharunmaks/stream-/claude/expertstream-moe-android-oegt59/deploy/join.sh | bash -s -- URL TOKEN
set -e
MASTER=${1:-}; TOKEN=${2:-}
REPO=${3:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}; FILE=${4:-qwen2.5-1.5b-instruct-q4_k_m.gguf}
if [ -z "$MASTER" ] || [ -z "$TOKEN" ]; then sed -n 2,9p "$0" 2>/dev/null || echo "usage: join.sh CLOUD_URL JOIN_TOKEN"; exit 1; fi
DIR=${EC_DIR:-$HOME/Stream-}
if [ ! -d "$DIR/.git" ]; then
  echo "getting the code"; git clone -q --depth 1 -b claude/expertstream-moe-android-oegt59 https://github.com/tharunmaks/stream- "$DIR"
else git -C "$DIR" pull -q || true; fi
cd "$DIR"
if [ ! -x ./es_serve ]; then
  echo "building the engine (a minute or two)"
  if command -v clang >/dev/null; then make -s -j"$(nproc)" es_serve es_get; else make -s -j"$(nproc)" CC=gcc es_serve es_get; fi
fi
python3 -c "import websockets, httpx" 2>/dev/null || pip install -q websockets httpx
python3 -c "import psutil" 2>/dev/null || pip install -q psutil 2>/dev/null || true
MODELS=${EC_MODELS:-$HOME/models}; mkdir -p "$MODELS"
if ! ls "$MODELS"/*.gguf >/dev/null 2>&1; then echo "downloading $FILE"; ./es_get "$REPO" "$FILE" -d "$MODELS" --all; fi
NAME=${EC_NAME:-$(hostname | cut -c1-20)}
echo "joining $MASTER as $NAME (keep this running; stop with the stop button / Ctrl+C)"
while true; do
  python3 cloud/worker_agent.py --master "$MASTER" --token "$TOKEN" --models "$MODELS" --name "$NAME" ${EC_KEY:+--key "$EC_KEY"} || true
  echo "node stopped, restarting in 5 s"; sleep 5
done
