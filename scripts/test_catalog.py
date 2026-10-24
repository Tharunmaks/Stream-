#!/usr/bin/env python3
"""The model catalog, credits and compute tiers without a real computer: stand-in nodes answer over the same messages a worker sends
(chunk, done, deploy_result), so the gateway's routing, billing and deploy code run end to end.

    python3 scripts/test_catalog.py
"""
import asyncio
import hashlib
import hmac
import json
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
tmp = tempfile.mkdtemp(prefix="eccat_")
os.environ["HOME"] = tmp
for k in ("RAZORPAY_KEY_ID", "RAZORPAY_KEY_SECRET", "RAZORPAY_WEBHOOK_SECRET", "EXPERTCLOUD_ADMIN_TOKEN"): os.environ.pop(k, None)

from fastapi.testclient import TestClient  # noqa: E402
from cloud.controller import Controller, Node  # noqa: E402
from gateway import billing, catalog, omni_router as gw  # noqa: E402

gw.init(os.path.join(tmp, "expertcloud.db"))
c = TestClient(gw.app); c.__enter__()   # one event loop for all requests, so deploys run in the background
A = {"Authorization": "Bearer " + gw.ADMIN}
bad = 0


def check(name, ok, info=""):
    global bad
    bad += not ok
    print("%s %-62s %s" % ("PASS" if ok else "FAIL", name, info))


class FakeWS:
    """A worker on the other end of the WebSocket: answers infer with 3 chunks and 2,000 prompt / 3,000 completion tokens."""
    def __init__(self): self.node = None; self.sent = []

    async def send_text(self, raw):
        msg = json.loads(raw); self.sent.append(msg)
        loop = asyncio.get_running_loop()
        if msg["type"] == "infer":
            for w in ("Hello", " from ", self.node.name):
                loop.call_soon(gw.CTRL.on_message, self.node, dict(type="chunk", job=msg["job"], delta=w))
            loop.call_soon(gw.CTRL.on_message, self.node, dict(type="done", job=msg["job"], usage=dict(prompt_tokens=2000, completion_tokens=3000), tps=40))
        elif msg["type"] == "deploy":
            name = catalog.served_name(msg["file"])
            models = [dict(name=k, size_mb=1) for k in list(self.node.models) + [name]]
            loop.call_later(0.05, gw.CTRL.on_message, self.node, dict(type="deploy_result", req=msg["req"], ok=True, message="saved", models=models))


def node(name, ram_gb, models, tps=None):
    ws = FakeWS()
    n = Node(ws, dict(node_id=name, name=name, caps=dict(ram_mb=ram_gb * 1024, disk_free_mb=200000), capacity=2,
                      models=[dict(name=m, size_mb=100, warm=True) for m in models]))
    n.load = dict(ram_free_mb=ram_gb * 900); n.tps = tps; ws.node = n
    gw.CTRL.add(n); return n


def user(name):
    r = c.post("/auth/signup", json={"username": name, "password": "password123"}).json()
    return {"Authorization": "Bearer " + r["token"]}


def chat(headers, model, **extra):
    gw.RATE.clear()   # the Free plan's 5 a minute is not what these checks test
    return c.post("/v1/chat/completions", headers=headers, json=dict(model=model, messages=[{"role": "user", "content": "hi"}], **extra))


# ---- the catalog
cat = c.get("/catalog").json()
ids = {m["id"]: m for m in cat["models"]}
check("1 catalog: 54 models, 35 free, prices in rupees", cat["count"] == 54 and cat["free"] == 35 and cat["currency"] == "INR"
      and ids["llama-3.3-70b"]["price_in_inr"] == 50 and ids["qwen3-8b"]["free"], "%d models, %d free" % (cat["count"], cat["free"]))
check("1b every model is a Q4_K_M GGUF on Hugging Face with a size", all(m["file"].endswith("Q4_K_M.gguf") and "/" in m["repo"] and m["size_gb"] > 0
                                                                       for m in gw.CAT.values()))
check("2 nothing runs yet: a catalog model says so", chat(A, "qwen3-8b").status_code == 503 and cat["online"] == 0)

# ---- deploy from the dashboard: the best computer is picked by RAM
small = node("phone", 8, [])
big = node("server", 64, [])
r = c.post("/admin/catalog/llama-3.3-70b/deploy", headers=A).json()
check("3 deploy picks a computer with enough RAM (70B -> 64 GB server)", r.get("node") == "server", str(r))
sent = [m for m in big.ws.sent if m["type"] == "deploy"]
check("3b the computer is told the Hugging Face repo and file", sent and sent[0]["repo"] == "bartowski/Llama-3.3-70B-Instruct-GGUF"
      and sent[0]["file"] == "Llama-3.3-70B-Instruct-Q4_K_M.gguf")
time.sleep(0.3)
adm = c.get("/admin/catalog", headers=A).json()
row = next(m for m in adm["models"] if m["id"] == "llama-3.3-70b")
check("4 the dashboard shows the deploy finished and where the model runs", row["deploy"]["state"] == "done" and row["running_on"] == ["server"], str(row["deploy"]))
r = c.post("/admin/catalog/qwen2.5-72b/deploy", headers=A, json={"node_id": "nope"})
check("4b deploying to an unknown computer is refused", r.status_code == 404)
c.post("/admin/catalog/qwen3-8b/deploy", headers=A, json={"node_id": "phone"}); time.sleep(0.3)
listed = {m["id"]: m for m in c.get("/v1/models", headers=A).json()["data"]}
check("5 /v1/models lists running catalog models by short id, with prices", listed.get("llama-3.3-70b", {}).get("online") and "qwen3-8b" in listed
      and listed["llama-3.3-70b"]["price_out_inr"] == 100 and "qwen2.5-72b" not in listed)

# ---- free models: no credits needed
U = user("asha")
key = c.post("/account/keys", headers=U, json={"label": "app"}).json()["key"]
K = {"Authorization": "Bearer " + key}
r = chat(K, "qwen3-8b").json()
check("6 a free model answers with an API key, costs nothing", r["choices"][0]["message"]["content"] == "Hello from phone" and r["x_expertcloud"]["cost_inr"] == 0)

# ---- paid models: credits
r = chat(K, "llama-3.3-70b")
check("7 a paid model without credits: 402 with the price", r.status_code == 402 and "credits" in r.json()["error"]["message"], r.json()["error"]["message"])
c.post("/admin/billing", headers=A, json={"key_id": "rzp_test_abc", "key_secret": "sekret"})


ORDERS = []


async def fake_order(keys, amount, receipt, notes):   # stands in for POST https://api.razorpay.com/v1/orders
    ORDERS.append(amount); return {"id": "order_cr%d" % len(ORDERS), "amount": amount, "currency": "INR"}
billing.create_order = fake_order
co = c.post("/billing/checkout", headers=U, json={"credits_inr": 500}).json()
too_small = c.post("/billing/checkout", headers=U, json={"credits_inr": 10}).status_code
sig = hmac.new(b"sekret", b"order_cr1|pay_cr1", hashlib.sha256).hexdigest()
ok = c.post("/billing/verify", headers=U, json={"order_id": "order_cr1", "payment_id": "pay_cr1", "signature": sig}).json()
c.post("/billing/verify", headers=U, json={"order_id": "order_cr1", "payment_id": "pay_cr1", "signature": sig})
check("8 buy Rs 500 of credits with Razorpay (once, however often verified)", co["amount"] == 50000 and ok.get("credits_inr") == 500 and too_small == 400
      and c.get("/account", headers=U).json()["credits_inr"] == 500)
r = chat(K, "llama-3.3-70b").json()
want = (2000 * 50 + 3000 * 100) / 1e6   # Rs 0.40
check("9 a paid model answers and is charged per token", r["choices"][0]["message"]["content"] == "Hello from server" and abs(r["x_expertcloud"]["cost_inr"] - want) < 1e-9,
      "Rs %.2f for 2,000 in + 3,000 out" % r["x_expertcloud"]["cost_inr"])
cr = c.get("/account/credits", headers=U).json()
check("10 balance, purchase log and usage by model", abs(cr["credits_inr"] - (500 - want)) < 1e-9 and cr["log"][0]["kind"] == "purchase"
      and {x["model"]: x["requests"] for x in cr["usage"]} == {"llama-3.3-70b": 1, "qwen3-8b": 1}, "balance Rs %.2f" % cr["credits_inr"])
r = chat(K, "Llama-3.3-70B-Instruct-Q4_K_M").json()
check("10b the file name also works and is billed the same", abs(r["x_expertcloud"]["cost_inr"] - want) < 1e-9)
c.post("/admin/users/asha/credits", headers=A, json={"inr": -499.20})   # leaves exactly Rs 0
r = chat(K, "llama-3.3-70b")
check("11 an empty balance stops paid models; free ones still answer", r.status_code == 402 and chat(K, "qwen3-8b").status_code == 200)
r = chat({"Authorization": "Bearer " + c.post("/admin/keys", headers=A, json={}).json()["key"]}, "llama-3.3-70b")
check("12 a key without an account cannot use paid models", r.status_code == 402)
check("13 cost rounding: a tiny paid request still costs at least a paisa", catalog.cost_paise(gw.CAT["phi-4"], 1, 1) == 1 and catalog.cost_paise(gw.CAT["qwen3-8b"], 10**6, 10**6) == 0)
adm = c.get("/admin/catalog", headers=A).json()
check("14 the owner sees usage and earnings by model", {x["model"]: x["inr"] for x in adm["usage"]}.get("llama-3.3-70b") == round(2 * want, 2))

# ---- compute tiers
V = user("ravi")
co = c.post("/billing/checkout", headers=V, json={"plan": "compute_ultra"}).json()
c.post("/billing/verify", headers=V, json={"order_id": co["order_id"], "payment_id": "pay_c1", "signature": hmac.new(b"sekret", ("%s|pay_c1" % co["order_id"]).encode(), hashlib.sha256).hexdigest()})
acc = c.get("/account", headers=V).json()
check("15 Compute Ultra (Rs 2,499): +200 GB storage", acc["plans"]["compute"]["id"] == "compute_ultra" and acc["storage"]["quota_bytes"] == (1024 << 20) + (200 << 30))
ctl = Controller(gw.STORE)
for n in (node("a-phone", 8, ["m"], 30), node("b-pc", 32, ["m"], 10), node("c-server", 64, ["m"], 18)): ctl.nodes[n.id] = n
plain = [n.name for n in ctl.ranked("m", 500)]
ultra = [n.name for n in ctl.ranked("m", 500, min_ram_mb=32 * 1024, fast=True)]
check("16 a compute tier uses only 32 GB+ computers, fastest first", plain[0] == "a-phone" and ultra == ["c-server", "b-pc"], "any: %s, Ultra: %s" % (plain, ultra))
print("ALL PASSED" if not bad else "%d FAILED" % bad)
sys.exit(1 if bad else 0)
