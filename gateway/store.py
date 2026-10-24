"""SQLite state of the OmniRoute gateway (expertcloud.db): virtual API keys, the contribute-to-use ledger and the response cache.

Keys are stored as SHA-256 hashes; the full key is shown once, when it is made. Every write is a short transaction, so the gateway
and the MCP server can share the file.
"""
import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS keys (
    hash TEXT PRIMARY KEY, prefix TEXT NOT NULL, label TEXT, created REAL NOT NULL, revoked INTEGER DEFAULT 0,
    requests INTEGER DEFAULT 0, prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
    tokens_saved INTEGER DEFAULT 0, cache_hits INTEGER DEFAULT 0, contributed_tokens INTEGER DEFAULT 0, last_used REAL);
CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY, model TEXT NOT NULL, prompt TEXT NOT NULL, response TEXT NOT NULL, created REAL NOT NULL, hits INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL COLLATE NOCASE, pw TEXT NOT NULL, created REAL NOT NULL,
    console_hash TEXT, disabled INTEGER DEFAULT 0, last_login REAL);
CREATE TABLE IF NOT EXISTS sessions (hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, created REAL NOT NULL, last_used REAL);
CREATE TABLE IF NOT EXISTS subscriptions (
    user_id INTEGER NOT NULL, product TEXT NOT NULL, plan TEXT NOT NULL, started REAL NOT NULL, expires REAL, source TEXT,
    PRIMARY KEY (user_id, product));
CREATE TABLE IF NOT EXISTS payments (
    order_id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, plan TEXT NOT NULL, amount_paise INTEGER NOT NULL, status TEXT NOT NULL,
    payment_id TEXT, created REAL NOT NULL, paid REAL);
CREATE TABLE IF NOT EXISTS storage_addons (user_id INTEGER NOT NULL, gb INTEGER NOT NULL, started REAL NOT NULL, expires REAL NOT NULL,
    order_id TEXT);
CREATE TABLE IF NOT EXISTS usage_daily (user_id INTEGER NOT NULL, day TEXT NOT NULL, requests INTEGER DEFAULT 0, tokens INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, day));
CREATE TABLE IF NOT EXISTS credits (user_id INTEGER PRIMARY KEY, paise INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS credit_log (user_id INTEGER NOT NULL, paise INTEGER NOT NULL, kind TEXT NOT NULL, ref TEXT, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS model_usage (user_id INTEGER NOT NULL, model TEXT NOT NULL, day TEXT NOT NULL, requests INTEGER DEFAULT 0,
    prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0, paise INTEGER DEFAULT 0, PRIMARY KEY (user_id, model, day));
CREATE TABLE IF NOT EXISTS nodes_seen (node_id TEXT PRIMARY KEY, name TEXT, owner_hash TEXT, first_seen REAL, last_seen REAL,
    tokens_generated INTEGER DEFAULT 0);
"""
FTS = "CREATE VIRTUAL TABLE IF NOT EXISTS cache_fts USING fts5(key UNINDEXED, prompt)"


def key_hash(key):
    return hashlib.sha256(key.encode()).hexdigest()


def words(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class Store:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        cols = [r[1] for r in self.db.execute("PRAGMA table_info(keys)")]
        if "user_id" not in cols: self.db.execute("ALTER TABLE keys ADD COLUMN user_id INTEGER")   # databases from before accounts
        try:
            self.db.execute(FTS)
            self.fts = True
        except sqlite3.OperationalError:   # SQLite without FTS5: exact cache only
            self.fts = False

    # ---- settings (admin token, join token) ----
    def setting(self, name, make=None):
        with self.lock:
            row = self.db.execute("SELECT value FROM settings WHERE name=?", (name,)).fetchone()
            if row: return row[0]
            if make is None: return None
            value = make()
            self.db.execute("INSERT INTO settings(name, value) VALUES(?, ?)", (name, value))
            return value

    def set_setting(self, name, value):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO settings(name, value) VALUES(?, ?)", (name, value))

    # ---- virtual API keys ----
    def new_key(self, label="", user_id=None):
        key = "sk-expert-" + secrets.token_urlsafe(32)
        with self.lock:
            self.db.execute("INSERT INTO keys(hash, prefix, label, created, user_id) VALUES(?, ?, ?, ?, ?)",
                            (key_hash(key), key[:14], label, time.time(), user_id))
        return key

    def check_key(self, key):
        if not key or not key.startswith("sk-expert-"): return None
        h = key_hash(key)
        row = self.db.execute("SELECT hash, revoked FROM keys WHERE hash=?", (h,)).fetchone()
        return h if row and not row[1] else None

    def revoke(self, prefix_or_hash):
        with self.lock:
            n = self.db.execute("UPDATE keys SET revoked=1 WHERE hash=? OR prefix=?", (prefix_or_hash, prefix_or_hash)).rowcount
        return n

    def account(self, h, prompt_tokens=0, completion_tokens=0, saved=0, cache_hit=False):
        with self.lock:
            self.db.execute("UPDATE keys SET requests=requests+1, prompt_tokens=prompt_tokens+?, completion_tokens=completion_tokens+?,"
                            " tokens_saved=tokens_saved+?, cache_hits=cache_hits+?, last_used=? WHERE hash=?",
                            (prompt_tokens, completion_tokens, saved, 1 if cache_hit else 0, time.time(), h))

    def key_stats(self, h):
        row = self.db.execute("SELECT prefix, label, created, requests, prompt_tokens, completion_tokens, tokens_saved, cache_hits,"
                              " contributed_tokens FROM keys WHERE hash=?", (h,)).fetchone()
        if not row: return None
        names = ["prefix", "label", "created", "requests", "prompt_tokens", "completion_tokens", "tokens_saved", "cache_hits", "contributed_tokens"]
        return dict(zip(names, row))

    def list_keys(self):
        rows = self.db.execute("SELECT prefix, label, created, revoked, requests, completion_tokens, tokens_saved, contributed_tokens FROM keys"
                               " ORDER BY created DESC LIMIT 200").fetchall()
        names = ["prefix", "label", "created", "revoked", "requests", "completion_tokens", "tokens_saved", "contributed_tokens"]
        return [dict(zip(names, r)) for r in rows]

    def totals(self):
        r = self.db.execute("SELECT COUNT(*), COALESCE(SUM(requests),0), COALESCE(SUM(completion_tokens),0), COALESCE(SUM(tokens_saved),0),"
                            " COALESCE(SUM(cache_hits),0) FROM keys").fetchone()
        return dict(keys=r[0], requests=r[1], completion_tokens=r[2], tokens_saved=r[3], cache_hits=r[4])

    # ---- user accounts (the public cloud console) ----
    @staticmethod
    def _pw(password, salt):
        return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32).hex()

    def create_user(self, username, password):
        """The new user's id, or None if the name is taken. Each user gets a 'console' key that accounts for console use."""
        salt = secrets.token_bytes(16)
        try:
            with self.lock:
                cur = self.db.execute("INSERT INTO users(username, pw, created) VALUES(?, ?, ?)", (username, salt.hex() + ":" + self._pw(password, salt), time.time()))
            uid = cur.lastrowid
        except sqlite3.IntegrityError:
            return None
        console = self.new_key("console", uid)
        with self.lock:
            self.db.execute("UPDATE users SET console_hash=? WHERE id=?", (key_hash(console), uid))
        return uid

    def check_password(self, username, password):
        row = self.db.execute("SELECT id, pw, disabled FROM users WHERE username=?", (username,)).fetchone()
        if not row:
            self._pw(password, b"0" * 16); return None   # same time as a real check, so names cannot be probed by timing
        salt, h = row[1].split(":", 1)
        if row[2] or not secrets.compare_digest(self._pw(password, bytes.fromhex(salt)), h): return None
        with self.lock:
            self.db.execute("UPDATE users SET last_login=? WHERE id=?", (time.time(), row[0]))
        return row[0]

    def new_session(self, uid):
        tok = "ses-" + secrets.token_urlsafe(32)
        with self.lock:
            self.db.execute("INSERT INTO sessions(hash, user_id, created, last_used) VALUES(?, ?, ?, ?)", (key_hash(tok), uid, time.time(), time.time()))
        return tok

    def session_user(self, tok, max_age=30 * 86400):
        """(user id, username, console key hash) of a live session, else None."""
        if not tok or not tok.startswith("ses-"): return None
        row = self.db.execute("SELECT u.id, u.username, u.console_hash, s.last_used FROM sessions s JOIN users u ON u.id=s.user_id"
                              " WHERE s.hash=? AND u.disabled=0", (key_hash(tok),)).fetchone()
        if not row or time.time() - row[3] > max_age: return None
        if time.time() - row[3] > 300:
            with self.lock:
                self.db.execute("UPDATE sessions SET last_used=? WHERE hash=?", (time.time(), key_hash(tok)))
        return row[:3]

    def end_session(self, tok):
        with self.lock:
            self.db.execute("DELETE FROM sessions WHERE hash=?", (key_hash(tok),))

    def user_keys(self, uid):
        rows = self.db.execute("SELECT prefix, label, created, revoked, requests, completion_tokens, tokens_saved, contributed_tokens, last_used FROM keys"
                               " WHERE user_id=? AND label != 'console' ORDER BY created DESC", (uid,)).fetchall()
        names = ["prefix", "label", "created", "revoked", "requests", "completion_tokens", "tokens_saved", "contributed_tokens", "last_used"]
        return [dict(zip(names, r)) for r in rows]

    def user_revoke(self, uid, prefix):
        with self.lock:
            return self.db.execute("UPDATE keys SET revoked=1 WHERE user_id=? AND prefix=? AND label != 'console'", (uid, prefix)).rowcount

    def user_usage(self, uid):
        r = self.db.execute("SELECT COALESCE(SUM(requests),0), COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0),"
                            " COALESCE(SUM(tokens_saved),0), COALESCE(SUM(cache_hits),0), COALESCE(SUM(contributed_tokens),0) FROM keys WHERE user_id=?",
                            (uid,)).fetchone()
        return dict(zip(["requests", "prompt_tokens", "completion_tokens", "tokens_saved", "cache_hits", "contributed_tokens"], r))

    def user_count(self):
        return self.db.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def user_key_hashes(self, uid):
        return [r[0] for r in self.db.execute("SELECT hash FROM keys WHERE user_id=? AND revoked=0", (uid,))]

    # ---- plans, payments, daily usage ----
    def subscription(self, uid, product):
        """(plan id, expires) of a live paid or granted plan, else None."""
        row = self.db.execute("SELECT plan, expires FROM subscriptions WHERE user_id=? AND product=?", (uid, product)).fetchone()
        if not row or (row[1] is not None and row[1] < time.time()): return None
        return row

    def set_subscription(self, uid, product, plan, days, source):
        """Starts a plan, or extends it when the same plan is still live. days=None: never expires (granted by the owner)."""
        now = time.time()
        cur = self.subscription(uid, product)
        base = cur[1] if cur and cur[0] == plan and cur[1] else now
        expires = None if days is None else base + days * 86400
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO subscriptions(user_id, product, plan, started, expires, source) VALUES(?,?,?,?,?,?)",
                            (uid, product, plan, now, expires, source))
        return expires

    def end_subscription(self, uid, product):
        with self.lock:
            self.db.execute("DELETE FROM subscriptions WHERE user_id=? AND product=?", (uid, product))

    def new_payment(self, order_id, uid, plan, amount_paise):
        with self.lock:
            self.db.execute("INSERT INTO payments(order_id, user_id, plan, amount_paise, status, created) VALUES(?,?,?,?,?,?)",
                            (order_id, uid, plan, amount_paise, "created", time.time()))

    def payment(self, order_id):
        row = self.db.execute("SELECT order_id, user_id, plan, amount_paise, status, payment_id FROM payments WHERE order_id=?", (order_id,)).fetchone()
        return dict(zip(["order_id", "user_id", "plan", "amount_paise", "status", "payment_id"], row)) if row else None

    def mark_paid(self, order_id, payment_id):
        """True the first time an order is marked paid (so a payment never starts a plan twice)."""
        with self.lock:
            return self.db.execute("UPDATE payments SET status='paid', payment_id=?, paid=? WHERE order_id=? AND status != 'paid'",
                                   (payment_id, time.time(), order_id)).rowcount == 1

    def payments(self, uid=None, limit=200):
        q = "SELECT p.order_id, u.username, p.plan, p.amount_paise, p.status, p.payment_id, p.created, p.paid FROM payments p JOIN users u ON u.id=p.user_id"
        rows = self.db.execute(q + (" WHERE p.user_id=?" if uid else "") + " ORDER BY p.created DESC LIMIT ?", ((uid, limit) if uid else (limit,))).fetchall()
        return [dict(zip(["order_id", "username", "plan", "amount_paise", "status", "payment_id", "created", "paid"], r)) for r in rows]

    def add_storage(self, uid, gb, days, order_id=None):
        now = time.time()
        with self.lock:
            self.db.execute("INSERT INTO storage_addons(user_id, gb, started, expires, order_id) VALUES(?,?,?,?,?)", (uid, int(gb), now, now + days * 86400, order_id))

    def extra_gb(self, uid):
        return self.db.execute("SELECT COALESCE(SUM(gb),0) FROM storage_addons WHERE user_id=? AND expires>?", (uid, time.time())).fetchone()[0]

    def storage_addons(self, uid):
        rows = self.db.execute("SELECT gb, started, expires FROM storage_addons WHERE user_id=? AND expires>? ORDER BY expires", (uid, time.time())).fetchall()
        return [dict(gb=g, started=s, expires=e) for g, s, e in rows]

    def count_request(self, uid, tokens=0):
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with self.lock:
            self.db.execute("INSERT INTO usage_daily(user_id, day, requests, tokens) VALUES(?,?,1,?) ON CONFLICT(user_id, day)"
                            " DO UPDATE SET requests=requests+1, tokens=tokens+excluded.tokens", (uid, day, tokens))

    def requests_today(self, uid):
        row = self.db.execute("SELECT requests FROM usage_daily WHERE user_id=? AND day=?", (uid, time.strftime("%Y-%m-%d", time.gmtime()))).fetchone()
        return row[0] if row else 0

    def key_owner(self, h):
        row = self.db.execute("SELECT user_id FROM keys WHERE hash=?", (h,)).fetchone()
        return row[0] if row else None

    def user_id(self, username):
        row = self.db.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        return row[0] if row else None

    # ---- prepaid credits (paise) and per-model usage of catalog models ----
    def credit_balance(self, uid):
        row = self.db.execute("SELECT paise FROM credits WHERE user_id=?", (uid,)).fetchone()
        return row[0] if row else 0

    def add_credits(self, uid, paise, kind, ref=None):
        """Adds (or, with a negative amount, takes) credits and logs it; returns the new balance."""
        with self.lock:
            self.db.execute("INSERT INTO credits(user_id, paise) VALUES(?, ?) ON CONFLICT(user_id) DO UPDATE SET paise=paise+excluded.paise", (uid, int(paise)))
            self.db.execute("INSERT INTO credit_log(user_id, paise, kind, ref, created) VALUES(?,?,?,?,?)", (uid, int(paise), kind, ref, time.time()))
        return self.credit_balance(uid)

    def credit_log(self, uid, limit=50):
        rows = self.db.execute("SELECT paise, kind, ref, created FROM credit_log WHERE user_id=? AND kind != 'usage' ORDER BY created DESC LIMIT ?",
                               (uid, limit)).fetchall()
        return [dict(zip(["paise", "kind", "ref", "created"], r)) for r in rows]

    def model_use(self, uid, model, prompt_tokens, completion_tokens, paise):
        """Counts a request to a catalog model; a paid one is charged to the credits in the same transaction."""
        day = time.strftime("%Y-%m-%d", time.gmtime())
        with self.lock:
            self.db.execute("BEGIN")
            try:
                self._model_use(uid, model, day, prompt_tokens, completion_tokens, paise)
            except Exception:
                self.db.execute("ROLLBACK"); raise
            self.db.execute("COMMIT")

    def _model_use(self, uid, model, day, prompt_tokens, completion_tokens, paise):
        self.db.execute("INSERT INTO model_usage(user_id, model, day, requests, prompt_tokens, completion_tokens, paise) VALUES(?,?,?,1,?,?,?)"
                        " ON CONFLICT(user_id, model, day) DO UPDATE SET requests=requests+1, prompt_tokens=prompt_tokens+excluded.prompt_tokens,"
                        " completion_tokens=completion_tokens+excluded.completion_tokens, paise=paise+excluded.paise",
                        (uid, model, day, prompt_tokens, completion_tokens, paise))
        if paise:
            self.db.execute("INSERT INTO credits(user_id, paise) VALUES(?, ?) ON CONFLICT(user_id) DO UPDATE SET paise=paise+excluded.paise", (uid, -paise))
            self.db.execute("INSERT INTO credit_log(user_id, paise, kind, ref, created) VALUES(?,?,?,?,?)", (uid, -paise, "usage", model, time.time()))

    def model_usage(self, uid=None, days=31):
        """Requests, tokens and rupees by model over the last days, for one user (or everyone)."""
        since = time.strftime("%Y-%m-%d", time.gmtime(time.time() - days * 86400))
        q = ("SELECT model, SUM(requests), SUM(prompt_tokens), SUM(completion_tokens), SUM(paise) FROM model_usage WHERE day>=?"
             + (" AND user_id=?" if uid is not None else "") + " GROUP BY model ORDER BY SUM(paise) DESC, SUM(requests) DESC")
        rows = self.db.execute(q, (since, uid) if uid is not None else (since,)).fetchall()
        return [dict(zip(["model", "requests", "prompt_tokens", "completion_tokens", "paise"], r)) for r in rows]

    # ---- contribute-to-use ledger ----
    def node_seen(self, node_id, name, owner_hash):
        now = time.time()
        with self.lock:
            self.db.execute("INSERT INTO nodes_seen(node_id, name, owner_hash, first_seen, last_seen) VALUES(?,?,?,?,?)"
                            " ON CONFLICT(node_id) DO UPDATE SET name=excluded.name, owner_hash=excluded.owner_hash, last_seen=excluded.last_seen",
                            (node_id, name, owner_hash, now, now))

    def credit_node(self, node_id, owner_hash, tokens):
        with self.lock:
            self.db.execute("UPDATE nodes_seen SET tokens_generated=tokens_generated+?, last_seen=? WHERE node_id=?", (tokens, time.time(), node_id))
            if owner_hash:
                self.db.execute("UPDATE keys SET contributed_tokens=contributed_tokens+? WHERE hash=?", (tokens, owner_hash))

    def contribution(self, h):
        row = self.db.execute("SELECT contributed_tokens FROM keys WHERE hash=?", (h,)).fetchone()
        return row[0] if row else 0

    # ---- response cache ----
    def cache_get(self, key):
        row = self.db.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        if not row: return None
        with self.lock:
            self.db.execute("UPDATE cache SET hits=hits+1 WHERE key=?", (key,))
        return json.loads(row[0])

    def cache_similar(self, model, prompt, threshold):
        """The cached answer of the most similar prompt for this model, if its word overlap (Jaccard) is at least threshold."""
        if not self.fts: return None
        q = " OR ".join(sorted(words(prompt)))[:2000]
        if not q: return None
        try:
            rows = self.db.execute("SELECT c.key, c.prompt, c.response FROM cache_fts f JOIN cache c ON c.key=f.key"
                                   " WHERE cache_fts MATCH ? AND c.model=? ORDER BY bm25(cache_fts) LIMIT 5", (q, model)).fetchall()
        except sqlite3.OperationalError:
            return None
        a = words(prompt)
        for key, p, resp in rows:
            b = words(p)
            if a and b and len(a & b) / len(a | b) >= threshold:
                with self.lock:
                    self.db.execute("UPDATE cache SET hits=hits+1 WHERE key=?", (key,))
                return json.loads(resp)
        return None

    def cache_put(self, key, model, prompt, response):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO cache(key, model, prompt, response, created) VALUES(?,?,?,?,?)",
                            (key, model, prompt, json.dumps(response), time.time()))
            if self.fts:
                self.db.execute("DELETE FROM cache_fts WHERE key=?", (key,))
                self.db.execute("INSERT INTO cache_fts(key, prompt) VALUES(?, ?)", (key, prompt))

    def cache_stats(self):
        r = self.db.execute("SELECT COUNT(*), COALESCE(SUM(hits),0) FROM cache").fetchone()
        return dict(entries=r[0], hits=r[1])


def default_db():
    return os.environ.get("EXPERTCLOUD_DB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "expertcloud.db"))
