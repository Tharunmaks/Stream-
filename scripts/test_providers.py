#!/usr/bin/env python3
"""AI providers and compute tiers, without any computer online and without calling a real provider: the provider APIs are
replaced by a stand-in that speaks the OpenAI streaming protocol (httpx MockTransport), so the gateway's own code runs end to end.

    python3 scripts/test_providers.py
"""
import hashlib
import hmac
import json
import os
import sqlite3
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
tmp = tempfile.mkdtemp(prefix="ecprov_")
os.environ["HOME"] = tmp
for k in ("RAZORPAY_KEY_ID", "RAZORPAY_KEY_SECRET", "RAZORPAY_WEBHOOK_SECRET", "EXPERTCLOUD_ADMIN_TOKEN", "EXPERTCLOUD_SECRET"): os.environ.pop(k, None)

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from cloud.controller import Controller  # noqa: E402
from gateway import billing, omni_router as gw, providers as prov  # noqa: E402

SEEN = []   # (host, path, auth header, body) of every request that reached a "provider"


def fake_provider(request):
    body = json.loads(request.content or b"{}")
    auth = request.headers.get("authorization") or request.headers.get("api-key") or ""
    SEEN.append((request.url.host, request.url.path, auth, body))
    if not auth.endswith("good-key-1234"): return httpx.Response(401, json={"error": {"message": "invalid api key"}})
    if request.url.path.endswith("/models"): return httpx.Response(200, json={"data": [{"id": "x"}]})
    words = ["Hello", " from", " " + request.url.host]
    lines = ["data: " + json.dumps({"choices": [{"delta": {"content": w}}]}) for w in words]
    lines.append("data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 3000}}))
    lines.append("data: [DONE]")
    return httpx.Response(200, content=("\n\n".join(lines) + "\n\n").encode(), headers={"content-type": "text/event-stream"})


class MockClient(httpx.AsyncClient):
    def __init__(self, *a, **kw):
        kw["transport"] = httpx.MockTransport(fake_provider); super().__init__(*a, **kw)


prov.httpx.AsyncClient = MockClient
gw.init(os.path.join(tmp, "expertcloud.db"))
c = TestClient(gw.app)
A = {"Authorization": "Bearer " + gw.ADMIN}
bad = 0


def check(name, ok, info=""):
    global bad
    bad += not ok
    print("%s %-60s %s" % ("PASS" if ok else "FAIL", name, info))


def user(name):
    r = c.post("/auth/signup", json={"username": name, "password": "password123"}).json()
    return {"Authorization": "Bearer " + r["token"]}


def chat(headers, model, stream=False):
    gw.RATE.clear()   # the Free plan's 5 a minute is not what these checks test
    return c.post("/v1/chat/completions", headers=headers, json={"model": model, "stream": stream, "messages": [{"role": "user", "content": "hi"}]})


# ---- catalog
cat = c.get("/providers").json()
free = sum(1 for p in cat["providers"] for m in p["models"] if m["free"])
check("1 catalog: 50+ providers, free models marked, no secrets", cat["count"] >= 50 and free >= 40 and cat["keys_ready"]
      and all(p["models"] and "base_url" not in p for p in cat["providers"]), "%d providers, %d models, %d free" % (cat["count"], cat["models"], free))
check("1b every provider address is https", all(gw.CAT[p]["base_url"].startswith("https://") for p in gw.CAT))

# ---- a user's own key
U = user("asha")
r = c.put("/account/providers/groq", headers=U, json={"key": "gsk-bad", "check": True})
check("2 a wrong key is refused when checked", r.status_code == 400 and "refused" in r.json()["error"]["message"], r.json()["error"]["message"])
r = c.put("/account/providers/groq", headers=U, json={"key": "  gsk-good-key-1234 ", "check": True, "budget_usd": 1})
mine = c.get("/account/providers", headers=U).json()
check("3 a good key is saved; only a hint is shown", r.status_code == 200 and mine["keys"][0]["hint"] == "gsk…1234" and "good-key" not in json.dumps(mine), r.json().get("hint"))
row = sqlite3.connect(os.path.join(tmp, "expertcloud.db")).execute("SELECT sealed FROM provider_keys WHERE provider='groq'").fetchone()[0]
check("4 the key is encrypted in the database", "good-key" not in row and gw.VAULT.open(row) == "gsk-good-key-1234")
models = {m["id"] for m in c.get("/v1/models", headers=U).json()["data"]}
check("5 /v1/models lists the provider's models", "groq/llama-3.3-70b-versatile" in models and "openai/gpt-4.1" not in models)
r = chat(U, "groq/llama-3.3-70b-versatile").json()
host, path, auth, body = SEEN[-1]
check("6 chat goes to the provider with the user's key, streaming upstream", r["choices"][0]["message"]["content"] == "Hello from api.groq.com"
      and host == "api.groq.com" and path == "/openai/v1/chat/completions" and auth == "Bearer gsk-good-key-1234" and body["model"] == "llama-3.3-70b-versatile"
      and body["stream"] is True and body["max_tokens"] == 256, "plan cap max_tokens=%s" % body.get("max_tokens"))
cost = r["x_expertcloud"]["cost_usd"]
check("7 usage and cost: 1,000 in x $0.59 + 3,000 out x $0.79 per 1M", abs(cost - (1000 * 0.59 + 3000 * 0.79) / 1e6) < 1e-9 and r["usage"]["completion_tokens"] == 3000, "$%s" % cost)
s = chat(U, "groq/llama-3.3-70b-versatile", stream=True)
text = "".join(json.loads(l[6:])["choices"][0]["delta"].get("content", "") for l in s.text.split("\n") if l.startswith("data: {") and json.loads(l[6:])["choices"])
check("8 streaming answers pass through", text == "Hello from api.groq.com" and s.text.rstrip().endswith("[DONE]"))
use = c.get("/account/providers", headers=U).json()
check("9 usage page: requests, tokens and month spend per provider", use["usage"][0]["requests"] == 2 and use["usage"][0]["completion_tokens"] == 6000
      and use["keys"][0]["month_usd"] > 0, str(use["usage"][0]))

# ---- monthly budget
c.patch("/account/providers/groq", headers=U, json={"budget_usd": 0.004})
r = chat(U, "groq/llama-3.3-70b-versatile")
check("10 the monthly budget stops spending", r.status_code == 402 and "budget" in r.json()["error"]["message"], r.json()["error"]["message"])
c.patch("/account/providers/groq", headers=U, json={"budget_usd": None})
check("10b no budget = no limit", chat(U, "groq/llama-3.3-70b-versatile").status_code == 200)

# ---- providers that need more than a key, and parameters a provider refuses
r = c.put("/account/providers/cloudflare", headers=U, json={"key": "cf-good-key-1234", "fields": {"account_id": "evil.com/x"}})
check("11 fields cannot change the provider address", r.status_code == 400, r.json()["error"]["message"])
c.put("/account/providers/cloudflare", headers=U, json={"key": "cf-good-key-1234", "fields": {"account_id": "abc123"}})
chat(U, "cloudflare/@cf/meta/llama-3.1-8b-instruct")
check("12 Cloudflare: account id goes into the address", SEEN[-1][0] == "api.cloudflare.com" and SEEN[-1][1] == "/client/v4/accounts/abc123/ai/v1/chat/completions")
c.put("/account/providers/azure", headers=U, json={"key": "az-good-key-1234", "fields": {"resource": "myres"}})
chat(U, "azure/gpt-4o-mini")
check("13 Azure: resource sub-domain and api-key header", SEEN[-1][0] == "myres.openai.azure.com" and SEEN[-1][2] == "az-good-key-1234")
c.put("/account/providers/anthropic", headers=U, json={"key": "sk-ant-good-key-1234"})
gw.RATE.clear()
c.post("/v1/chat/completions", headers=U, json={"model": "anthropic/claude-haiku-5-5", "temperature": 0.7, "messages": [{"role": "user", "content": "hi"}]})
check("14 Anthropic: OpenAI-compatible endpoint, temperature not sent", SEEN[-1][0] == "api.anthropic.com" and "temperature" not in SEEN[-1][3])

# ---- plan limit on connected providers (Free: 5)
for pid in ("openai",): c.put("/account/providers/" + pid, headers=U, json={"key": "k-good-key-1234"})
r = c.put("/account/providers/xai", headers=U, json={"key": "k-good-key-1234"})
check("15 Free plan connects 5 providers, then 402", r.status_code == 402, r.json()["error"]["message"])
check("15b replacing a key you have is allowed", c.put("/account/providers/openai", headers=U, json={"key": "k2-good-key-1234"}).status_code == 200)
check("16 removing a key", c.delete("/account/providers/anthropic", headers=U).json()["deleted"] == 1)

# ---- the owner's keys: free models for everyone
V = user("ravi")
r = chat(V, "google/gemini-2.5-flash")
check("17 no key: says where to get one", r.status_code == 403 and "aistudio.google.com" in r.json()["error"]["message"])
check("17b the owner saves a key (admin token only)", c.put("/admin/providers/google", headers=V, json={"key": "AIza-good-key-1234"}).status_code == 401
      and c.put("/admin/providers/google", headers=A, json={"key": "AIza-good-key-1234"}).status_code == 200)
r = chat(V, "google/gemini-2.5-flash").json()
check("18 every user gets the owner's free models", r["choices"][0]["message"]["content"].endswith("googleapis.com") and r["x_expertcloud"]["key"] == "cloud"
      and SEEN[-1][2] == "Bearer AIza-good-key-1234")
r = chat(V, "google/gemini-2.5-pro")
check("19 paid models need the user's own key", r.status_code == 403 and "not free" in r.json()["error"]["message"])
ids = {m["id"]: m for m in c.get("/v1/models", headers=V).json()["data"]}
check("20 /v1/models shows the cloud's free models", ids.get("google/gemini-2.5-flash", {}).get("via") == "cloud" and "google/gemini-2.5-pro" not in ids)
check("21 the catalog says which providers the cloud offers", next(p for p in c.get("/providers").json()["providers"] if p["id"] == "google")["cloud"])
adm = c.get("/admin/providers", headers=A).json()
check("22 owner sees provider usage of everyone, never key text", len(adm["usage"]) >= 4 and "good-key" not in json.dumps(adm))
check("23 a daily request limit still applies to provider models", gw.STORE.requests_today(gw.STORE.user_id("ravi")) == 1)

# ---- compute tiers
c.post("/admin/billing", headers=A, json={"key_id": "rzp_test_abc", "key_secret": "sekret"})


async def fake_order(keys, amount, receipt, notes):
    return {"id": "order_c1", "amount": amount, "currency": "INR"}
billing.create_order = fake_order
co = c.post("/billing/checkout", headers=V, json={"plan": "compute_ultra"}).json()
sig = hmac.new(b"sekret", b"order_c1|pay_c1", hashlib.sha256).hexdigest()
ok = c.post("/billing/verify", headers=V, json={"order_id": "order_c1", "payment_id": "pay_c1", "signature": sig}).json()
acc = c.get("/account", headers=V).json()
check("24 buy Compute Ultra for Rs 2,499: +200 GB storage", co["amount"] == 249900 and ok["plan"]["id"] == "compute_ultra"
      and acc["plans"]["compute"]["id"] == "compute_ultra" and acc["storage"]["quota_bytes"] == (1024 << 20) + (200 << 30), "%.0f GB" % (acc["storage"]["quota_bytes"] / 2 ** 30))


class N:   # a stand-in node: only what the router reads
    def __init__(self, name, ram_gb, tps):
        self.id = self.name = name; self.caps = {"ram_mb": ram_gb * 1024}; self.models = {"m": {"warm": True}}; self.capacity = 1
        self.active = 0; self.load = {"ram_free_mb": 99999}; self.ttft = 0.5; self.tps = tps; self.failures = 0; self.last_seen = gw.time.time()


ctl = Controller(gw.STORE)
for n in (N("phone", 8, 30), N("pc", 32, 10), N("server", 64, 18)): ctl.nodes[n.id] = n
plain = [n.name for n in ctl.ranked("m", 500)]
ultra = [n.name for n in ctl.ranked("m", 500, min_ram_mb=32 * 1024, fast=True)]
check("25 compute tier: only 32 GB+ computers, fastest first", plain[0] == "phone" and ultra == ["server", "pc"], "any: %s, Ultra: %s" % (plain, ultra))
print("ALL PASSED" if not bad else "%d FAILED" % bad)
sys.exit(1 if bad else 0)
