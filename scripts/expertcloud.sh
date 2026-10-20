#!/usr/bin/env bash
# ExpertCloud on your own device, as one command. The gateway and this device's node run in the background and restart
# themselves if they crash. Works on Termux and on a PC.
#
#   cloud start        start (safe to run again)
#   cloud status       live numbers from the cloud: nodes, speed, requests
#   cloud open         print the dashboard link (it signs you in)
#   cloud stop         stop everything
#   cloud logs         show what is happening (Ctrl+C returns, nothing stops)
#   cloud shortcut     make the word "cloud" work in any new terminal
#
# start options: --lan (let other devices join), --models DIR (default ~/models), --port N (default 8000), --name NAME,
#                --gateway-only, --rotate (new tokens; old ones stop working)
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd); STATE="$HOME/.expertcloud"; mkdir -p "$STATE"
PORT=${EC_PORT:-$(cat "$STATE/port" 2>/dev/null || echo 8000)}
MODELS=${EC_MODELS:-$HOME/models}; NAME=${EC_NAME:-$(cat "$STATE/name" 2>/dev/null || hostname 2>/dev/null || echo "my-device")}
LAN=${EC_LAN:-0}; GATEWAY_ONLY=0; ROTATE=0
cmd=${1:-status}; shift || true
while [ $# -gt 0 ]; do case "$1" in
  --lan) LAN=1;; --port) PORT=$2; shift;; --models) MODELS=$2; shift;; --name) NAME=$2; shift;;
  --gateway-only) GATEWAY_ONLY=1;; --rotate) ROTATE=1;; *) echo "Unknown option: $1"; exit 1;; esac; shift; done
say() { printf '%s\n' "$*"; }

running() { [ -s "$STATE/$1.pid" ] && kill -0 "$(cat "$STATE/$1.pid")" 2>/dev/null; }
up() { python3 -c "import urllib.request,sys;urllib.request.urlopen('http://127.0.0.1:$PORT/health',timeout=2)" >/dev/null 2>&1; }
token() { cat "$STATE/admin_token" 2>/dev/null; }
open_link() { [ -s "$STATE/admin_token" ] && say "http://127.0.0.1:$PORT/dashboard#token=$(token)" || say "Not started yet: run  cloud start"; }

# Runs a command forever: if it stops or crashes, it starts again after a few seconds. Its loop pid is kept for stop.
keep() {   # name, command...
  local n=$1; shift
  local cmdline; printf -v cmdline '%q ' "$@"
  nohup bash -c 'echo $$ > "$0"; while true; do '"$cmdline"' >>"$1" 2>&1; echo "[$(date "+%H:%M:%S")] $2 stopped, restarting in 5 s" >>"$1"; sleep 5; done' \
    "$STATE/$n.pid" "$STATE/$n.log" "$n" </dev/null >/dev/null 2>&1 &
  for _ in $(seq 1 20); do [ -s "$STATE/$n.pid" ] && break; sleep 0.1; done
}
stop_one() {   # name: its loop, then its program
  running "$1" && kill "$(cat "$STATE/$1.pid")" 2>/dev/null
  rm -f "$STATE/$1.pid"
}

case "$cmd" in
start)
  if [ "$ROTATE" = 1 ]; then stop_one worker; stop_one gateway; fi
  echo "$PORT" >"$STATE/port"; echo "$NAME" >"$STATE/name"
  if ! python3 -c "import fastapi, uvicorn, websockets, httpx" 2>/dev/null; then
    say "Installing the Python packages (first time only, a minute or two)..."
    pip install -q fastapi uvicorn websockets httpx psutil 2>/dev/null || pip install -q fastapi uvicorn websockets httpx || { say "Could not install packages. Check your internet connection."; exit 1; }
  fi
  if [ "$GATEWAY_ONLY" = 0 ] && [ ! -x ./es_serve ]; then
    say "Building the engine (first time only, a few minutes)..."
    make es_serve es_get >"$STATE/build.log" 2>&1 || { say "Build failed. Details: $STATE/build.log"; exit 1; }
  fi
  command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock >/dev/null 2>&1
  if running gateway && up; then say "Cloud already running."
  else
    stop_one gateway
    host=127.0.0.1; [ "$LAN" = 1 ] && host=0.0.0.0
    args=(--host $host --port "$PORT"); [ "$ROTATE" = 1 ] && args+=(--rotate-tokens)
    keep gateway python3 gateway/omni_router.py "${args[@]}"
    for _ in $(seq 1 60); do up && break; sleep 0.5; done
    up || { say "The gateway did not start. Last lines:"; tail -n 6 "$STATE/gateway.log"; exit 1; }
    say "Gateway running on port $PORT."
  fi
  if [ "$GATEWAY_ONLY" = 0 ]; then
    mkdir -p "$MODELS"
    stop_one worker
    keep worker python3 cloud/worker_agent.py --master "ws://127.0.0.1:$PORT" --token "$(cat "$STATE/join_token")" --models "$MODELS" --name "$NAME"
    say "This device is now a node called \"$NAME\" (models folder: $MODELS)."
    ls "$MODELS"/*.gguf >/dev/null 2>&1 || say "  No models yet. Put .gguf files in $MODELS, or add one from the dashboard (Nodes page)."
  fi
  say ""
  say "Your cloud's public page (people sign up and use it here):"
  say "  http://127.0.0.1:$PORT/"
  say "Your owner dashboard (signs you in by itself):"
  say "  $(open_link)"
  if [ "$LAN" = 1 ]; then ip=$(python3 -c "import socket;s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);s.connect(('10.255.255.255',1));print(s.getsockname()[0])" 2>/dev/null)
    [ -n "$ip" ] && say "Other devices on the same Wi-Fi open: http://$ip:$PORT/"; fi
  say ""; say "It keeps running in the background. Check it any time with:  cloud status"
  ;;
stop)
  stop_one worker; stop_one gateway
  pkill -f "cloud/worker_agent.py --master" 2>/dev/null; pkill -f "gateway/omni_router.py --host" 2>/dev/null
  command -v termux-wake-unlock >/dev/null 2>&1 && termux-wake-unlock >/dev/null 2>&1
  say "Stopped."
  ;;
status)
  if ! up; then say "The cloud is not running. Start it with:  cloud start"; exit 0; fi
  python3 - "$PORT" "$(token)" "$NAME" <<'PY'
import json, sys, urllib.request
port, tok, me = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    req = urllib.request.Request("http://127.0.0.1:%s/admin/live" % port, headers={"Authorization": "Bearer " + tok})
    d = json.load(urllib.request.urlopen(req, timeout=5))
except Exception as e:
    print("Gateway is up, but the status could not be read: %s" % e); sys.exit(0)
nodes, sp, t = d.get("nodes", []), d.get("speed", {}), d.get("totals", {})
print("ExpertCloud on port %s  (up %s)" % (port, "%d min" % (d.get("uptime_s", 0) // 60)))
print("  Nodes online:      %d" % len(nodes))
for n in nodes:
    c, l = n.get("caps") or {}, n.get("load") or {}
    mark = "this device" if n["name"] == me else ""
    print("    - %-14s %s  %d/%d busy  %s models  %s" % (n["name"], "working" if n.get("active") else "idle   ", n.get("active", 0), n.get("capacity", 1),
          len(n.get("models") or []), mark))
print("  Speed (last min):  %.1f tokens/s" % sp.get("avg_tps", 0))
print("  Requests served:   %d   tokens generated: %d" % (t.get("requests", 0), t.get("completion_tokens", 0)))
print("  Models:            %s" % (", ".join(m["id"] for m in d.get("models", [])) or "none yet"))
PY
  running worker && say "This device's node: running." || say "This device's node: stopped (cloud start)."
  ;;
open) open_link ;;
logs) touch "$STATE/gateway.log" "$STATE/worker.log"; say "Ctrl+C returns to the prompt; the cloud keeps running."; tail -n 25 -f "$STATE/gateway.log" "$STATE/worker.log" ;;
shortcut)
  line="alias cloud='$ROOT/scripts/expertcloud.sh'"
  for rc in "$HOME/.bashrc" "$HOME/.profile"; do
    grep -qF "$line" "$rc" 2>/dev/null || echo "$line" >>"$rc"
  done
  say "Done. Open a new terminal (or type: $line), then use:  cloud start" ;;
*) sed -n 2,12p "$0"; exit 1 ;;
esac
