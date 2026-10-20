#!/usr/bin/env python3
"""ExpertCloud end to end: a gateway, two worker nodes and the MCP server on this machine, a real model on the native engine.

    make es_serve es_get && pip install fastapi uvicorn httpx psutil websockets && python3 scripts/test_cloud.py MODEL.gguf

Checks: nodes join and report hardware; keys and authentication; /v1/models; a JSON answer; the same request answered from the
cache; a streamed answer (SSE); Token Killer savings; two requests at once on one free slot (queueing); failover when the node that
answered is killed; the four MCP tools over HTTP. Runs in a temporary HOME, so ~/.expertcloud is not touched.
"""
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]): sys.exit(__doc__)
MODEL = os.path.abspath(sys.argv[1])
NAME = os.path.basename(MODEL)[:-5]
bad = 0


def check(name, ok, info=""):
    global bad
    bad += not ok
    print("%s %-48s %s" % ("PASS" if ok else "FAIL", name, info), flush=True)


def port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


tmp = tempfile.mkdtemp(prefix="expertcloud_test_")
env = dict(os.environ, HOME=tmp, PYTHONUNBUFFERED="1")
procs = []


def start(args, name):
    log = open(os.path.join(tmp, name + ".log"), "w")
    p = subprocess.Popen([sys.executable] + args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(p); return p


def stop(p):
    try: os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError: pass


try:
    gp, mp = port(), port()
    G = "http://127.0.0.1:%d" % gp
    start(["gateway/omni_router.py", "--port", str(gp), "--db", os.path.join(tmp, "expertcloud.db")], "gateway")
    for _ in range(100):
        try:
            if httpx.get(G + "/health", timeout=1).status_code == 200: break
        except httpx.HTTPError: time.sleep(0.2)
    admin = open(os.path.join(tmp, ".expertcloud", "admin_token")).read().strip()
    join = open(os.path.join(tmp, ".expertcloud", "join_token")).read().strip()
    A = {"Authorization": "Bearer " + admin}
    workers = {}
    for w in ("alpha", "beta"):
        d = os.path.join(tmp, "models_" + w); os.makedirs(d); os.symlink(MODEL, os.path.join(d, os.path.basename(MODEL)))
        workers[w] = start(["cloud/worker_agent.py", "--master", "ws://127.0.0.1:%d" % gp, "--token", join, "--models", d, "--name", w,
                            "--suffix-name", "--slots", "1", "--threads", "2", "--ctx", "2048"], "worker_" + w)
    nodes = []
    for _ in range(150):
        nodes = httpx.get(G + "/admin/nodes", headers=A).json().get("nodes", [])
        if len(nodes) == 2: break
        time.sleep(0.2)
    caps = nodes[0]["caps"] if nodes else {}
    check("1 two nodes join and report hardware", len(nodes) == 2 and caps.get("ram_mb", 0) > 0 and caps.get("cpu_cores"),
          "%d nodes; %s cores, %s MB RAM, disk %s MB/s" % (len(nodes), caps.get("cpu_cores"), caps.get("ram_mb"), caps.get("disk_read_mbps")))
    r = httpx.post(G + "/admin/keys", headers=A, json={"label": "test"}).json()
    key = r.get("key", "")
    K = {"Authorization": "Bearer " + key}
    bad_auth = httpx.post(G + "/v1/chat/completions", headers={"Authorization": "Bearer sk-expert-nope"}, json={"messages": [{"role": "user", "content": "hi"}]}).status_code
    no_admin = httpx.post(G + "/admin/keys", headers=K, json={}).status_code
    check("2 keys: made by the admin, unknown keys refused", key.startswith("sk-expert-") and bad_auth == 401 and no_admin == 401,
          "unknown key -> %d, user key on admin API -> %d" % (bad_auth, no_admin))
    models = [m["id"] for m in httpx.get(G + "/v1/models", headers=K).json().get("data", [])]
    check("3 /v1/models lists the model and aliases", NAME in models and "auto" in models, ", ".join(models))

    req = {"model": "auto", "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}], "temperature": 0, "max_tokens": 16}
    t0 = time.time(); r1 = httpx.post(G + "/v1/chat/completions", headers=K, json=req, timeout=600).json(); t1 = time.time() - t0
    txt = r1.get("choices", [{}])[0].get("message", {}).get("content", "")
    x1 = r1.get("x_expertcloud", {})
    check("4 JSON answer from a node", bool(txt.strip()) and x1.get("node") in ("alpha", "beta"),
          "%r from %s in %.2f s (%s completion tokens)" % (txt.strip()[:40], x1.get("node"), t1, r1.get("usage", {}).get("completion_tokens")))
    t0 = time.time(); r2 = httpx.post(G + "/v1/chat/completions", headers=K, json=req, timeout=60).json(); t2 = time.time() - t0
    x2 = r2.get("x_expertcloud", {})
    check("5 same request answered from the cache", x2.get("cache") == "hit" and r2["choices"][0]["message"]["content"] == txt,
          "lookup %.2f ms inside the gateway, %.1f ms round trip" % (x2.get("cache_ms", -1), t2 * 1000))

    sreq = dict(req, stream=True, temperature=0.7, messages=[{"role": "user", "content": "Say hello in three words."}])
    parts, done, final = [], False, {}
    with httpx.stream("POST", G + "/v1/chat/completions", headers=K, json=sreq, timeout=600) as s:
        for line in s.iter_lines():
            if not line.startswith("data: "): continue
            if line == "data: [DONE]": done = True; break
            ev = json.loads(line[6:])
            for c in ev.get("choices", []):
                if c.get("delta", {}).get("content"): parts.append(c["delta"]["content"])
            if ev.get("x_expertcloud"): final = ev
    check("6 streamed answer (SSE)", done and len(parts) > 1 and final.get("usage", {}).get("completion_tokens", 0) > 0,
          "%d chunks: %r" % (len(parts), "".join(parts).strip()[:40]))

    messy = "Could you please   tell me,\n\n\n\n   what 2 + 2 is?   \n" + ("Please answer the question above carefully and briefly.\n" * 6)
    r3 = httpx.post(G + "/v1/chat/completions", headers=K, json={"messages": [{"role": "user", "content": messy}, {"role": "user", "content": messy}],
                                                                 "max_tokens": 8, "compress": "aggressive"}, timeout=600).json()
    saved = r3.get("x_expertcloud", {}).get("tokens_saved", 0)
    check("7 Token Killer saves prompt tokens", saved > 0, "%d of about %d tokens saved" % (saved, 2 * len(messy) // 4))

    res = {}
    def one(i):
        res[i] = httpx.post(G + "/v1/chat/completions", headers=K, json={"messages": [{"role": "user", "content": "Count to %d." % (3 + i)}], "max_tokens": 12},
                            timeout=600).json()
    ths = [threading.Thread(target=one, args=(i,)) for i in range(4)]
    for t in ths: t.start()
    for t in ths: t.join()
    served = [res[i].get("x_expertcloud", {}).get("node") for i in range(4)]
    check("8 four requests at once on two one-slot nodes", all(res[i].get("choices") for i in range(4)) and set(served) == {"alpha", "beta"},
          "served by %s" % served)

    victim = x1.get("node") or "alpha"
    stop(workers[victim]); time.sleep(0.5)
    r4 = httpx.post(G + "/v1/chat/completions", headers=K, json={"messages": [{"role": "user", "content": "Name a color."}], "max_tokens": 8}, timeout=600).json()
    other = r4.get("x_expertcloud", {}).get("node")
    check("9 failover after a node is killed", bool(r4.get("choices")) and other and other != victim, "killed %s, answered by %s" % (victim, other))

    menv = dict(env, EXPERTCLOUD_GATEWAY=G)
    mlog = open(os.path.join(tmp, "mcp.log"), "w")
    procs.append(subprocess.Popen([sys.executable, "mcp/expertstream_mcp.py", "--http", str(mp)], cwd=ROOT, env=menv, stdout=mlog, stderr=subprocess.STDOUT, start_new_session=True))
    M = "http://127.0.0.1:%d/mcp" % mp
    for _ in range(50):
        try: httpx.get(M, timeout=1); break
        except httpx.HTTPError: time.sleep(0.1)
    rpc = lambda i, method, params=None: httpx.post(M, json={"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}}, timeout=600).json()
    init = rpc(1, "initialize", {"protocolVersion": "2025-06-18"})
    tools = [t["name"] for t in rpc(2, "tools/list")["result"]["tools"]]
    nodes_txt = rpc(3, "tools/call", {"name": "list_cloud_nodes", "arguments": {}})["result"]["content"][0]["text"]
    key_txt = rpc(4, "tools/call", {"name": "generate_free_api_key", "arguments": {"label": "mcp-test"}})["result"]["content"][0]["text"]
    inf = rpc(5, "tools/call", {"name": "route_inference", "arguments": {"prompt": "What is 3 + 4?", "max_tokens": 10, "temperature": 0}})["result"]
    dep = rpc(6, "tools/call", {"name": "deploy_model_to_node", "arguments": {"node_id": "nope", "repo": "a/b", "file": "c.gguf"}})["result"]
    check("10 MCP over HTTP: the four tools", init.get("result", {}).get("serverInfo") and len(tools) == 4 and other in nodes_txt and "sk-expert-" in key_txt
          and not inf.get("isError") and dep.get("isError"), "tools %s; route_inference: %r" % (tools, inf["content"][0]["text"].split("\n")[0][:40]))
finally:
    for p in procs: stop(p)
    if bad: print("logs kept in", tmp)
    else: shutil.rmtree(tmp, ignore_errors=True)
print("ALL PASSED" if not bad else "%d FAILED" % bad)
sys.exit(1 if bad else 0)
