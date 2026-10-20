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

    # ---- virtual API keys ----
    def new_key(self, label=""):
        key = "sk-expert-" + secrets.token_urlsafe(32)
        with self.lock:
            self.db.execute("INSERT INTO keys(hash, prefix, label, created) VALUES(?, ?, ?, ?)", (key_hash(key), key[:14], label, time.time()))
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
