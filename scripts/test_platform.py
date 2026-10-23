#!/usr/bin/env python3
"""ExpertCloud platform checks without any computer online: accounts, plans in INR, plan limits, the database service, Razorpay
payments (the Razorpay API call is replaced by a stand-in; signatures are computed exactly as Razorpay does), owner tools, and the
link with the ExpertStream site (one account, single sign-on, Expert One unlocking Cloud tiers, Training inside the Cloud).

    python3 scripts/test_platform.py
"""
import asyncio
import hashlib
import hmac
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
tmp = tempfile.mkdtemp(prefix="ecplat_")
os.environ["HOME"] = tmp
for k in ("RAZORPAY_KEY_ID", "RAZORPAY_KEY_SECRET", "RAZORPAY_WEBHOOK_SECRET", "EXPERTCLOUD_ADMIN_TOKEN", "DATABASE_URL"): os.environ.pop(k, None)
os.makedirs(os.path.join(tmp, "models")); open(os.path.join(tmp, "models", "tiny.gguf"), "w").close()   # the Training model library
os.environ["ES_MODEL_DIR"] = os.path.join(tmp, "models")

from fastapi.testclient import TestClient  # noqa: E402
from gateway import billing, omni_router as gw  # noqa: E402

# a 5 MB Free plan, so the space limit can be tested without writing a gigabyte (the real default is checked below)
with open(os.path.join(tmp, "plans.json"), "w") as f:
    json.dump({"cloud": {"plans": [{"id": "cloud_free", "limits": dict(gw.plans_mod.PLANS["cloud"]["plans"][0]["limits"], db_mb=5)}]}}, f)
gw.init(os.path.join(tmp, "expertcloud.db"))
c = TestClient(gw.app)
A = {"Authorization": "Bearer " + gw.ADMIN}
bad = 0


def check(name, ok, info=""):
    global bad
    bad += not ok
    print("%s %-52s %s" % ("PASS" if ok else "FAIL", name, info))


def user(name):
    r = c.post("/auth/signup", json={"email": name + "@example.com", "password": "password123"}).json()
    return {"Authorization": "Bearer " + r["token"]}


# ---- plans
p = c.get("/plans").json()
prices = {x["id"]: x["price_inr"] for prod in p["products"].values() for x in prod["plans"]}
check("1 plans in INR: Cloud 0/1700/4000, Expert One 0/2000/5000/10000", p["currency"] == "INR" and prices == {
    "cloud_free": 0, "cloud_pro": 1700, "cloud_max": 4000, "es-starter": 0, "one-starter": 2000, "one-mid": 5000, "one-top": 10000}, str(prices))

space = {p["id"]: p["limits"]["db_mb"] for p in gw.plans_mod.PLANS["cloud"]["plans"]}
check("1b database space: Free 1 GB, Pro 50 GB, Max 1 TB; extra Rs 30 per GB", space == {"cloud_free": 1024, "cloud_pro": 51200, "cloud_max": 1048576}
      and p["storage"]["price_inr_per_gb"] == 30, str(space))

U = user("asha")
acc = c.get("/account", headers=U).json()
check("2 a new account starts on Cloud Free and ExpertStream Starter", acc["plan"]["id"] == "cloud_free" and acc["plans"]["expertone"]["id"] == "es-starter",
      "%s / %s" % (acc["plan"]["name"], acc["plans"]["expertone"]["name"]))

# ---- free plan limits
k1 = c.post("/account/keys", headers=U, json={"label": "one"})
k2 = c.post("/account/keys", headers=U, json={"label": "two"})
check("3 Free plan: one API key", k1.status_code == 200 and k2.status_code == 402, k2.json().get("error", {}).get("message", ""))
uid = gw.STORE.user_id("asha@example.com")
for _ in range(25): gw.STORE.count_request(uid)
r = c.post("/v1/chat/completions", headers=U, json={"messages": [{"role": "user", "content": "hi"}]})
check("4 Free plan: 25 requests a day, then 429", r.status_code == 429, r.json()["error"]["message"][:70])

# ---- database
K = {"Authorization": "Bearer " + k1.json()["key"]}
check("5 create databases up to the plan's count (Free: 3)", all(c.post("/db", headers=U, json={"name": n}).status_code == 200 for n in ("shop", "b", "c"))
      and c.post("/db", headers=U, json={"name": "d"}).status_code == 402)
one = c.post("/db/shop/orders", headers=K, json={"item": "tea", "status": "paid", "qty": 2}).json()
rows = [{"id": "o-%d" % i, "item": "item %d" % i, "status": "paid" if i % 2 else "open", "total": i * 10, "tags": ["new"] if i % 10 == 0 else ["old"],
         "address": {"city": "Pune" if i % 3 == 0 else "Delhi"}} for i in range(100)]
bulk = c.post("/db/shop/orders", headers=U, json=rows).json()
found = c.get("/db/shop/orders", headers=U, params={"where": json.dumps({"status": "paid"}), "limit": 500}).json()
check("6 insert with a key, import 100 documents, filter by field", len(one["ids"]) == 1 and len(bulk["ids"]) == 100 and found["total"] == 51,
      "%d paid of %d" % (found["total"], c.get("/db/shop/orders", headers=U).json()["total"]))
q = lambda body: c.post("/db/shop/orders/_query", headers=U, json=body).json()
r1 = q({"where": {"total": {"$gte": 500, "$lt": 700}, "address.city": "Pune"}, "sort": ["-total"], "fields": ["total", "address.city"]})
r2 = q({"where": {"tags": {"$contains": "new"}}})
r3 = q({"where": {"$or": [{"total": {"$in": [10, 20]}}, {"item": {"$startsWith": "item 9"}}]}, "sort": ["total"]})
r4 = q({"search": "ITEM 42"})
check("7 queries: ranges, nested fields, sort, field choice, list contains, $or/$in/$startsWith, text search",
      [d["data"]["total"] for d in r1["docs"]] == [690, 660, 630, 600, 570, 540, 510] and r1["docs"][0]["data"] == {"total": 690, "address": {"city": "Pune"}}
      and r2["total"] == 10 and [d["data"]["total"] for d in r3["docs"]] == [10, 20] + [90] + list(range(900, 1000, 10)) and r4["total"] == 1,
      "%d, %d, %d, %d found" % (r1["total"], r2["total"], r3["total"], r4["total"]))
pt = c.patch("/db/shop/orders/o-3", headers=U, json={"status": "shipped", "address.pin": "411001", "$inc": {"total": 5}, "$push": {"tags": "vip"}, "$unset": ["item"]}).json()
d3 = c.get("/db/shop/orders/o-3", headers=U).json()["data"]
check("8 partial update: set, nested set, $inc, $push, $unset", d3 == {"status": "shipped", "total": 35, "tags": ["old", "vip"], "address": {"city": "Pune", "pin": "411001"}},
      json.dumps(d3))
before = q({"where": {"total": 690}, "explain": True})["plan"]
c.post("/db/shop/orders/_indexes", headers=U, json={"field": "total"})
after = q({"where": {"total": 690}, "explain": True})["plan"]
idx = c.get("/db/shop/orders/_indexes", headers=U).json()["indexes"]
check("9 an index on a field is used by queries", not any("ix_" in x for x in before) and any("ix_" in x for x in after) and idx[0]["field"] == "total",
      "plan: %s" % after[0][:60])
bad_batch = c.post("/db/shop/_batch", headers=U, json={"ops": [{"op": "put", "collection": "orders", "doc": {"id": "new-1", "x": 1}},
                                                               {"op": "delete", "collection": "orders", "id": "does-not-exist"}]})
ok_batch = c.post("/db/shop/_batch", headers=U, json={"ops": [{"op": "put", "collection": "stock", "doc": {"id": "tea", "n": 10}},
                                                              {"op": "patch", "collection": "stock", "id": "tea", "changes": {"$inc": {"n": -3}}},
                                                              {"op": "delete", "collection": "orders", "id": "o-4"}]}).json()
check("10 transactions: all or nothing", bad_batch.status_code == 404 and c.get("/db/shop/orders/new-1", headers=U).status_code == 404
      and c.get("/db/shop/stock/tea", headers=U).json()["data"]["n"] == 7 and c.get("/db/shop/orders/o-4", headers=U).status_code == 404, str(ok_batch)[:60])
snap = c.post("/db/shop/_snapshots", headers=U, json={"label": "before cleanup"}).json()
second = c.post("/db/shop/_snapshots", headers=U, json={}).status_code
c.delete("/db/shop/orders", headers=U)
gone = c.get("/db/shop/orders", headers=U).json()["total"]
c.post("/db/shop/_snapshots/%s/restore" % snap["id"], headers=U)
back = c.get("/db/shop/orders", headers=U).json()["total"]
exp = c.get("/db/shop/_export", headers=U)
check("11 snapshot, delete, restore; Free keeps 1 snapshot; download the database", gone == 0 and back == 100 and second == 402
      and exp.status_code == 200 and exp.content[:15] == b"SQLite format 3", "%d -> %d -> %d documents, export %d KB" % (100, gone, back, len(exp.content) // 1024))
big = c.post("/db/shop/blobs", headers=U, json=[{"pad": "x" * 900_000} for _ in range(6)])
check("12 the plan's space is enforced (test plan: 5 MB)", big.status_code == 402, big.json()["error"]["message"][:80])
other = user("ravi")
check("13 users cannot see each other's data", c.get("/db/shop/orders", headers=other).status_code == 404 and c.get("/db", headers=A).status_code == 403)

# ---- payments
check("14 checkout says when payments are not switched on", c.post("/billing/checkout", headers=U, json={"plan": "cloud_pro"}).status_code == 503)
c.post("/admin/billing", headers=A, json={"key_id": "rzp_test_abc", "key_secret": "sekret", "webhook_secret": "whsec", "expertstream_url": "https://expertstream.example"})
orders = []


async def fake_order(keys, amount, receipt, notes):   # stands in for POST https://api.razorpay.com/v1/orders
    orders.append((amount, notes)); return {"id": "order_%d" % len(orders), "amount": amount, "currency": "INR"}
billing.create_order = fake_order
co = c.post("/billing/checkout", headers=U, json={"plan": "cloud_pro"}).json()
check("15 checkout makes an order for Rs 1,700 (170000 paise)", co.get("amount") == 170000 and co.get("key_id") == "rzp_test_abc" and co["currency"] == "INR")
sig = lambda o, pid, key="sekret": hmac.new(key.encode(), ("%s|%s" % (o, pid)).encode(), hashlib.sha256).hexdigest()
wrong = c.post("/billing/verify", headers=U, json={"order_id": co["order_id"], "payment_id": "pay_1", "signature": sig(co["order_id"], "pay_1", "nope")})
right = c.post("/billing/verify", headers=U, json={"order_id": co["order_id"], "payment_id": "pay_1", "signature": sig(co["order_id"], "pay_1")}).json()
check("16 a forged payment is refused, a real one starts Pro", wrong.status_code == 400 and right.get("plan", {}).get("id") == "cloud_pro")
exp1 = c.get("/account/plans", headers=U).json()["cloud"]["expires"]
c.post("/billing/verify", headers=U, json={"order_id": co["order_id"], "payment_id": "pay_1", "signature": sig(co["order_id"], "pay_1")})
exp2 = c.get("/account/plans", headers=U).json()["cloud"]["expires"]
days = (exp1 - gw.time.time()) / 86400
check("17 Pro lasts 30 days; the same payment never counts twice", 29.9 < days <= 30 and exp1 == exp2, "%.2f days" % days)
check("18 Pro lifts the limits (keys, daily requests)", c.post("/account/keys", headers=U, json={"label": "two"}).status_code == 200
      and c.post("/v1/chat/completions", headers=U, json={"messages": [{"role": "user", "content": "hi"}]}).status_code != 429)
eo = c.post("/billing/checkout", headers=other, json={"plan": "cloud_max"}).json()
body = json.dumps({"event": "payment.captured", "payload": {"payment": {"entity": {"id": "pay_2", "order_id": eo["order_id"]}}}}).encode()
whsig = hmac.new(b"whsec", body, hashlib.sha256).hexdigest()
forged = c.post("/billing/webhook", content=body, headers={"x-razorpay-signature": "0" * 64}).status_code
c.post("/billing/webhook", content=body, headers={"x-razorpay-signature": whsig})
mx = c.get("/account/plans", headers=other).json()["cloud"]
one = c.post("/billing/checkout", headers=U, json={"plan": "one-mid"})
check("19 webhook starts Cloud Max (Rs 4,000); forged webhooks refused; Expert One is bought on the site", eo["amount"] == 400000 and mx["id"] == "cloud_max"
      and forged == 400 and one.status_code == 400 and "ExpertStream" in one.json()["error"]["message"])

so = c.post("/billing/checkout", headers=U, json={"storage_gb": 3}).json()
c.post("/billing/verify", headers=U, json={"order_id": so["order_id"], "payment_id": "pay_3", "signature": sig(so["order_id"], "pay_3")})
st = c.get("/account", headers=U).json()["storage"]
dbq = c.get("/db", headers=U).json()
check("20 buy 3 GB extra space for Rs 90; the quota grows by 3 GB", so["amount"] == 9000 and st["extra_gb"] == 3
      and dbq["quota_bytes"] == (51200 << 20) + (3 << 30), "quota %.1f GB (Pro 50 GB + 3 GB)" % (dbq["quota_bytes"] / 2 ** 30))
c.post("/admin/users/ravi@example.com/storage", headers=A, json={"gb": 2, "days": 7})
check("21 the owner can give extra space", c.get("/db", headers=other).json()["extra_gb"] == 2)

# ---- owner tools
c.post("/admin/users/ravi@example.com/plan", headers=A, json={"plan": "one-mid"})
users = {u["username"]: u for u in c.get("/admin/users", headers=A).json()["users"]}
pays = c.get("/admin/billing", headers=A).json()
check("24 owner sees users, plans and payments; can grant Expert One", users["ravi@example.com"]["plans"]["expertone"]["id"] == "one-mid"
      and users["asha@example.com"]["plans"]["cloud"]["id"] == "cloud_pro" and len([x for x in pays["payments"] if x["status"] == "paid"]) == 3
      and "rzp_key_secret" not in json.dumps(pays) and "sekret" not in json.dumps(pays))
login = c.post("/auth/login", json={"email": "ravi@example.com", "password": "password123"}).json()
viaapp = c.get("/account/plans", headers={"Authorization": "Bearer " + login["token"], "Origin": "https://expertstream.example"})
check("25 apps sign in and read the plans (CORS open)", viaapp.json()["expertone"]["name"] == "Mid"
      and viaapp.headers.get("access-control-allow-origin") == "*" and c.get("/plans").json()["apps"]["expertstream"]["url"] == "https://expertstream.example")

# ---- one account on both websites
import es_common as ESC  # noqa: E402  (the ExpertStream site's library, on the path the gateway set)
check("26 a Cloud sign-up is an ExpertStream account (the site's login takes the same email and password)",
      bool(ESC.login("asha@example.com", "password123")))
site_uid = ESC.signup("meera@example.com", "sitepass123")   # signs up on the ExpertStream site
b = TestClient(gw.app); b.cookies.set(ESC.COOKIE, ESC.make_token(site_uid))
s1 = b.post("/auth/sso")
me = b.get("/account", headers={"Authorization": "Bearer " + s1.json().get("token", "")}).json() if s1.status_code == 200 else {}
cookie_back = ESC.verify_token(TestClient(gw.app).post("/auth/login", json={"email": "meera@example.com", "password": "sitepass123"}).cookies.get(ESC.COOKIE) or "")
check("27 signed in on the site = signed in on the Cloud (and back)", me.get("username") == "meera@example.com" and cookie_back == site_uid,
      "sso %d, the Cloud's login gives the site's cookie: %s" % (s1.status_code, bool(cookie_back)))
ESC.set_subscription(site_uid, "one-starter", "order_x", "pay_x")
pro = b.get("/account/plans", headers={"Authorization": "Bearer " + s1.json()["token"]}).json()["cloud"]
ESC.set_subscription(site_uid, "one-top", "order_y", "pay_y")
top = b.get("/account/plans", headers={"Authorization": "Bearer " + s1.json()["token"]}).json()["cloud"]
check("28 Expert One includes Cloud: Starter -> Cloud Pro, Top -> Cloud Max", pro["id"] == "cloud_pro" and pro["source"] == "Expert One" and top["id"] == "cloud_max",
      "%s, %s" % (pro["name"], top["name"]))

# ---- Training inside the Cloud
anon = TestClient(gw.app)
t1 = TestClient(gw.app); t1.cookies.set(gw.COOKIE, c.post("/auth/login", json={"email": "asha@example.com", "password": "password123"}).json()["token"])
t2 = TestClient(gw.app); t2.cookies.set(gw.COOKIE, s1.json()["token"])
st1 = t1.get("/training/api/status")
check("29 Training needs a Cloud sign-in; the plan sets its limits", anon.get("/training/api/status").status_code == 401 and st1.status_code == 200
      and st1.json()["limits"]["max_steps"] == 3000 and t2.get("/training/api/status").json()["limits"]["max_steps"] == 20000,
      "Pro: %s steps, Max (Expert One Top): 20000" % st1.json()["limits"]["max_steps"])
model = os.path.join(tmp, "models", "tiny.gguf"); sample = os.path.join(ROOT, "apps", "es-trainer", "data", "qa_sample.jsonl")
r_model = t1.post("/training/api/train", json={"model": "/etc/passwd", "data": sample, "steps": 10}).status_code
r_data = t1.post("/training/api/train", json={"model": model, "data": "/etc/passwd", "steps": 10}).status_code
r_steps = t1.post("/training/api/train", json={"model": model, "data": sample, "steps": 999999}).status_code
check("30 Training only takes library models and your own data, within the plan", (r_model, r_data, r_steps) == (400, 400, 402), str((r_model, r_data, r_steps)))
free = TestClient(gw.app); free.cookies.set(gw.COOKIE, c.post("/auth/signup", json={"email": "free@example.com", "password": "password123"}).json()["token"])
check("30b Free plan cannot fill the shared model library from Hugging Face", free.post("/training/api/hf", json={"repo": "x/y"}).status_code == 402)
up = t1.post("/training/api/upload", files={"file": ("notes.jsonl", b'{"prompt": "hi", "response": "hello"}\n', "application/json")}).json()
d1, d2 = t1.get("/training/api/datasets").json(), t2.get("/training/api/datasets").json()
check("31 each user's uploads are private", up.get("path") in d1 and up.get("path") not in d2 and sample in d2, "upload: %s" % os.path.basename(up.get("path", "?")))
print("ALL PASSED" if not bad else "%d FAILED" % bad)
sys.exit(1 if bad else 0)
