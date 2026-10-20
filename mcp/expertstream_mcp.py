#!/usr/bin/env python3
"""ExpertStream MCP server: lets AI agents run the ExpertCloud through the OmniRoute gateway.

    python3 mcp/expertstream_mcp.py --stdio                 (Claude Code, Claude Desktop, any stdio MCP client)
    python3 mcp/expertstream_mcp.py --http 8765             (MCP over HTTP: POST JSON-RPC to http://127.0.0.1:8765/mcp)
    python3 mcp/expertstream_mcp.py --http 8765 --stdio     (both)

Tools: list_cloud_nodes, deploy_model_to_node, generate_free_api_key, route_inference.
The gateway is http://127.0.0.1:8000 unless EXPERTCLOUD_GATEWAY says otherwise; the admin token comes from EXPERTCLOUD_ADMIN_TOKEN
or ~/.expertcloud/admin_token (written by the gateway on start). Standard library only.
"""
import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GATEWAY = os.environ.get("EXPERTCLOUD_GATEWAY", "http://127.0.0.1:8000").rstrip("/")
PROTOCOL = "2025-06-18"


def admin_token():
    t = os.environ.get("EXPERTCLOUD_ADMIN_TOKEN")
    if t: return t.strip()
    p = os.path.expanduser("~/.expertcloud/admin_token")
    return open(p).read().strip() if os.path.exists(p) else ""


def call(method, path, body=None, timeout=600):
    req = urllib.request.Request(GATEWAY + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": "Bearer " + admin_token(), "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try: return json.loads(e.read())
        except ValueError: return {"error": {"message": "HTTP %d" % e.code}}
    except urllib.error.URLError as e:
        return {"error": {"message": "gateway not reachable at %s (%s): start python3 gateway/omni_router.py" % (GATEWAY, e.reason)}}


TOOLS = [
    dict(name="list_cloud_nodes", description="List the ExpertCloud worker nodes online: hardware (CPU, RAM, GPU, disk speed), load, models (warm or not), "
         "requests in flight, time to first token and tokens per second.", inputSchema={"type": "object", "properties": {}}),
    dict(name="deploy_model_to_node", description="Download a GGUF model from Hugging Face onto a node (only its core now; experts arrive on first use). "
         "Get node_id from list_cloud_nodes.", inputSchema={"type": "object", "required": ["node_id", "repo", "file"], "properties": {
             "node_id": {"type": "string"}, "repo": {"type": "string", "description": "e.g. unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"},
             "file": {"type": "string", "description": "e.g. Qwen3-30B-A3B-Instruct-2507-Q2_K.gguf (the first part of a split model)"}}}),
    dict(name="generate_free_api_key", description="Make a free sk-expert-... API key for the OpenAI-compatible endpoint of the gateway.",
         inputSchema={"type": "object", "properties": {"label": {"type": "string"}}}),
    dict(name="route_inference", description="Answer a prompt on the cloud: the gateway picks the best node. model: auto, expertstream-fast, "
         "expertstream-coder or a model name from list_cloud_nodes.", inputSchema={"type": "object", "required": ["prompt"], "properties": {
             "prompt": {"type": "string"}, "system": {"type": "string"}, "model": {"type": "string", "default": "auto"},
             "max_tokens": {"type": "integer", "default": 512}, "temperature": {"type": "number", "default": 0.7}}}),
]


def run_tool(name, args):
    if name == "list_cloud_nodes":
        r = call("GET", "/admin/nodes")
        if "nodes" not in r: return r, True
        lines = []
        for n in r["nodes"]:
            c, l = n.get("caps") or {}, n.get("load") or {}
            ms = ", ".join("%s%s" % (m["name"], " (warm)" if m.get("warm") else "") for m in n.get("models") or []) or "no models"
            lines.append("%s [%s]: %s cores, %s MB RAM (%s free), GPU %s, disk %s MB/s, CPU %s%%, %d/%d busy, ttft %s s, %s tok/s | %s" % (
                n["name"], n["id"], c.get("cpu_cores"), c.get("ram_mb"), l.get("ram_free_mb"), c.get("gpu") or "none", c.get("disk_read_mbps"),
                l.get("cpu_pct"), n.get("active", 0), n.get("capacity", 1), None if n.get("ttft_s") is None else round(n["ttft_s"], 2),
                None if n.get("tps") is None else round(n["tps"], 1), ms))
        return ("\n".join(lines) or "no nodes online (join one with the command from the dashboard)"), False
    if name == "deploy_model_to_node":
        r = call("POST", "/admin/deploy", {k: args.get(k) for k in ("node_id", "repo", "file")}, timeout=7200)
        return r, not r.get("ok")
    if name == "generate_free_api_key":
        r = call("POST", "/admin/keys", {"label": args.get("label") or "mcp"})
        return (("key: %s\nOpenAI base URL: %s/v1" % (r["key"], GATEWAY)) if "key" in r else r), "key" not in r
    if name == "route_inference":
        msgs = ([{"role": "system", "content": args["system"]}] if args.get("system") else []) + [{"role": "user", "content": args.get("prompt", "")}]
        r = call("POST", "/v1/chat/completions", {"model": args.get("model") or "auto", "messages": msgs,
                                                   "max_tokens": int(args.get("max_tokens") or 512), "temperature": args.get("temperature", 0.7)})
        if "choices" not in r: return r, True
        x = r.get("x_expertcloud") or {}
        return "%s\n\n[model %s, node %s, cache %s, %s tokens, %s tok/s]" % (r["choices"][0]["message"]["content"], r.get("model"), x.get("node"),
                                                                            x.get("cache"), r["usage"].get("completion_tokens"), x.get("tps")), False
    return {"error": "unknown tool %s" % name}, True


def handle(msg):
    """One JSON-RPC message -> response dict, or None for notifications."""
    mid, method = msg.get("id"), msg.get("method")
    if method == "initialize":
        res = {"protocolVersion": (msg.get("params") or {}).get("protocolVersion") or PROTOCOL, "capabilities": {"tools": {}},
               "serverInfo": {"name": "expertstream", "version": "2.0"}}
    elif method == "tools/list":
        res = {"tools": TOOLS}
    elif method == "tools/call":
        p = msg.get("params") or {}
        out, is_err = run_tool(p.get("name"), p.get("arguments") or {})
        res = {"content": [{"type": "text", "text": out if isinstance(out, str) else json.dumps(out, indent=1)}], "isError": bool(is_err)}
    elif method == "ping":
        res = {}
    elif mid is None:
        return None   # notifications (initialized, cancelled)
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found: %s" % method}}
    return None if mid is None else {"jsonrpc": "2.0", "id": mid, "result": res}


def serve_stdio():
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        try: msg = json.loads(line)
        except ValueError:
            print(json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}), flush=True); continue
        out = [r for r in (handle(m) for m in (msg if isinstance(msg, list) else [msg])) if r]
        if out: print(json.dumps(out if isinstance(msg, list) else out[0]), flush=True)


class Http(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.rstrip("/") not in ("", "/mcp"): self.send_error(404); return
        try: msg = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)))
        except ValueError: self.send_error(400); return
        out = [r for r in (handle(m) for m in (msg if isinstance(msg, list) else [msg])) if r]
        if not out: self.send_response(202); self.end_headers(); return
        data = json.dumps(out if isinstance(msg, list) else out[0]).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        data = json.dumps({"server": "expertstream-mcp", "tools": [t["name"] for t in TOOLS], "gateway": GATEWAY}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--http", type=int, default=0, help="also serve MCP over HTTP on this port (127.0.0.1)")
    ap.add_argument("--stdio", action="store_true")
    a = ap.parse_args()
    if not a.http and not a.stdio: a.stdio = True
    if a.http:
        srv = ThreadingHTTPServer(("127.0.0.1", a.http), Http)
        if a.stdio: threading.Thread(target=srv.serve_forever, daemon=True).start()
        else: print("MCP over HTTP: http://127.0.0.1:%d/mcp (gateway %s)" % (a.http, GATEWAY), file=sys.stderr, flush=True); srv.serve_forever()
    if a.stdio: serve_stdio()


if __name__ == "__main__":
    main()
