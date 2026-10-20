#!/usr/bin/env bash
# One command to run your own ExpertCloud: the gateway and a worker node, kept alive in the background.
#
#   scripts/expertcloud.sh start        start both (and print the dashboard sign-in link)
#   scripts/expertcloud.sh stop         stop both
#   scripts/expertcloud.sh status       what is running
#   scripts/expertcloud.sh link         print the dashboard sign-in link again
#   scripts/expertcloud.sh logs         follow the logs (Ctrl+C leaves them running)
#
# Options for start (or set the variable): --lan (EC_LAN=1, let other devices connect), --port N (EC_PORT), --models DIR (EC_MODELS,
# default ~/models), --name NAME (EC_NAME), --no-worker (EC_NO_WORKER=1, gateway only), --rotate (make new tokens).
# On Termux it also takes a wake lock, so Android does not freeze everything when you switch to another app.
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd); STATE="$HOME/.expertcloud"; mkdir -p "$STATE"
PORT=${EC_PORT:-$(cat "$STATE/port" 2>/dev/null || echo 8000)}; MODELS=${EC_MODELS:-$HOME/models}; NAME=${EC_NAME:-$(hostname 2>/dev/null || echo node)}; LAN=${EC_LAN:-0}; NOWORKER=${EC_NO_WORKER:-0}; ROTATE=0
cmd=${1:-start}; shift || true
while [ $# -gt 0 ]; do case "$1" in
  --lan) LAN=1;; --port) PORT=$2; shift;; --models) MODELS=$2; shift;; --name) NAME=$2; shift;; --no-worker) NOWORKER=1;; --rotate) ROTATE=1;;
  *) echo "unknown option $1"; exit 1;; esac; shift; done

alive() { [ -s "$STATE/$1.pid" ] && kill -0 "$(cat "$STATE/$1.pid")" 2>/dev/null; }
healthy() { python3 - "$PORT" <<'PY' 2>/dev/null
import sys, urllib.request
urllib.request.urlopen("http://127.0.0.1:%s/health" % sys.argv[1], timeout=2); 
PY
}
link() { [ -s "$STATE/admin_token" ] && echo "http://127.0.0.1:$PORT/dashboard#token=$(cat "$STATE/admin_token")"; }
launch() {   # name, command...: detached from this terminal, survives closing the session
  local n=$1; shift
  if command -v setsid >/dev/null; then nohup setsid "$@" >>"$STATE/$n.log" 2>&1 </dev/null & else nohup "$@" >>"$STATE/$n.log" 2>&1 </dev/null & fi
  echo $! >"$STATE/$n.pid"
}

case "$cmd" in
start)
  echo "$PORT" >"$STATE/port"
  python3 -c "import fastapi, uvicorn, websockets" 2>/dev/null || { echo "Installing Python packages (one time)..."; pip install -q fastapi uvicorn websockets httpx || exit 1; }
  if [ "$NOWORKER" = 0 ] && [ ! -x ./es_serve ]; then echo "Building the engine (one time, a few minutes)..."; make es_serve es_get >"$STATE/build.log" 2>&1 || { echo "Build failed, see $STATE/build.log"; exit 1; }; fi
  command -v termux-wake-lock >/dev/null && termux-wake-lock && echo "Wake lock on: Android keeps this running in the background."
  if alive gateway && [ $ROTATE = 0 ]; then echo "Gateway already running."; else
    alive gateway && { kill "$(cat "$STATE/gateway.pid")"; sleep 1; }
    host=127.0.0.1; [ "$LAN" = 1 ] && host=0.0.0.0
    args=(--host $host --port "$PORT"); [ $ROTATE = 1 ] && args+=(--rotate-tokens)
    launch gateway python3 gateway/omni_router.py "${args[@]}"
    for _ in $(seq 1 40); do healthy && break; sleep 0.5; done
    healthy || { echo "The gateway did not start. Last lines of $STATE/gateway.log:"; tail -n 8 "$STATE/gateway.log"; exit 1; }
  fi
  if [ "$NOWORKER" = 0 ]; then
    mkdir -p "$MODELS"
    alive worker && { kill "$(cat "$STATE/worker.pid")"; sleep 1; }
    launch worker python3 cloud/worker_agent.py --master "ws://127.0.0.1:$PORT" --token "$(cat "$STATE/join_token")" --models "$MODELS" --name "$NAME"
    echo "Node '$NAME' started (models folder: $MODELS)."
    ls "$MODELS"/*.gguf >/dev/null 2>&1 || echo "  No models yet: put .gguf files in $MODELS, or deploy one from the dashboard (Nodes page)."
  fi
  echo; echo "Open this link in your browser; it signs you in:"; echo "  $(link)"
  if [ "$LAN" = 1 ]; then ip=$(python3 -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(('10.255.255.255',1));print(s.getsockname()[0])" 2>/dev/null); [ -n "$ip" ] && echo "Other devices on your Wi-Fi: http://$ip:$PORT/dashboard"; fi
  echo; echo "It keeps running when you switch apps or tabs. Stop it with: scripts/expertcloud.sh stop";;
stop)
  for n in worker gateway; do alive $n && kill "$(cat "$STATE/$n.pid")" && echo "stopped $n"; rm -f "$STATE/$n.pid"; done
  command -v termux-wake-unlock >/dev/null && termux-wake-unlock;;
status)
  alive gateway && healthy && echo "gateway: running on port $PORT" || echo "gateway: stopped"
  alive worker && echo "node:    running" || echo "node:    stopped"
  alive gateway && link;;
link) link || echo "No token yet: run start first.";;
logs) touch "$STATE/gateway.log" "$STATE/worker.log"; tail -n 20 -f "$STATE/gateway.log" "$STATE/worker.log";;
*) sed -n 2,12p "$0"; exit 1;;
esac
