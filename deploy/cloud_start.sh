#!/usr/bin/env bash
# The whole ExpertCloud on one cloud machine (a Hugging Face Space, a free VM, any Linux server): the gateway on a public
# port, and a node on the same machine with a model, both restarted if they stop.
#
#   PORT                    port to serve (Hugging Face Spaces: 7860, set by the Space)
#   EXPERTCLOUD_ADMIN_TOKEN fixed admin token (set it as a secret; otherwise a new one is made on every restart and printed in the logs)
#   EXPERTCLOUD_JOIN_TOKEN  fixed join token for extra nodes (same)
#   EC_MODEL_REPO, EC_MODEL_FILE   the model downloaded on first start (default Qwen2.5 1.5B Instruct, 1.1 GB)
#   EC_NODE=0               gateway only, no node on this machine
cd "$(dirname "$0")/.." || exit 1
PORT=${PORT:-7860}
if [ -z "${EC_MODELS:-}" ]; then if [ -d /data ] && [ -w /data ]; then EC_MODELS=/data/models; export EXPERTCLOUD_DB=${EXPERTCLOUD_DB:-/data/expertcloud.db}; else EC_MODELS=$HOME/models; fi; fi
REPO=${EC_MODEL_REPO:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}
FILE=${EC_MODEL_FILE:-qwen2.5-1.5b-instruct-q4_k_m.gguf}
mkdir -p "$EC_MODELS"

python3 gateway/omni_router.py --host 0.0.0.0 --port "$PORT" &
GW=$!
for _ in $(seq 1 60); do python3 -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2)" 2>/dev/null && break; sleep 0.5; done

if [ "${EC_NODE:-1}" != 0 ]; then
  (
    if ! ls "$EC_MODELS"/*.gguf >/dev/null 2>&1; then
      echo "downloading $REPO / $FILE (first start only)"
      ./es_get "$REPO" "$FILE" -d "$EC_MODELS" --all || echo "model download failed: add one from the dashboard's Nodes page"
    fi
    while true; do
      python3 cloud/worker_agent.py --master "ws://127.0.0.1:$PORT" --token "$(cat "$HOME/.expertcloud/join_token")" \
        --models "$EC_MODELS" --name "${EC_NAME:-cloud-1}" --slots "${EC_SLOTS:-1}"
      echo "node stopped, restarting in 5 s"; sleep 5
    done
  ) &
fi
wait "$GW"   # if the gateway ever exits, so does the container, and the host restarts it
