"""ExpertCloud Database: per-user document databases (like Firestore or MongoDB) in a SQLite file of their own (clouddata.db).

A user owns databases; a database holds collections; a collection holds JSON documents with an id. Documents can be filtered by
top-level field equality, and imported in bulk (an export of another app's database: a JSON list of objects). Space is counted in
bytes of document JSON, so plans can cap it.
"""
import json
import re
import secrets
import sqlite3
import threading
import time

NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
FIELD = re.compile(r"^[A-Za-z0-9_]{1,64}$")
MAX_DOC = 1 << 20   # 1 MB per document

SCHEMA = """
CREATE TABLE IF NOT EXISTS dbs (user_id INTEGER NOT NULL, name TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY (user_id, name));
CREATE TABLE IF NOT EXISTS docs (
    user_id INTEGER NOT NULL, db TEXT NOT NULL, coll TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL, size INTEGER NOT NULL,
    created REAL NOT NULL, updated REAL NOT NULL, PRIMARY KEY (user_id, db, coll, id));
CREATE INDEX IF NOT EXISTS docs_by_time ON docs(user_id, db, coll, created);
"""


class DbError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class DocStore:
    def __init__(self, path):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    @staticmethod
    def check_name(kind, name):
        if not NAME.match(name or ""): raise DbError(400, "%s name: 1 to 64 letters, digits, - or _" % kind)

    # ---- databases ----
    def list_dbs(self, uid):
        out = []
        for (name, created) in self.db.execute("SELECT name, created FROM dbs WHERE user_id=? ORDER BY name", (uid,)).fetchall():
            colls = self.db.execute("SELECT coll, COUNT(*), COALESCE(SUM(size),0) FROM docs WHERE user_id=? AND db=? GROUP BY coll ORDER BY coll",
                                    (uid, name)).fetchall()
            out.append(dict(name=name, created=created, collections=[dict(name=c, docs=n, bytes=b) for c, n, b in colls],
                            docs=sum(c[1] for c in colls), bytes=sum(c[2] for c in colls)))
        return out

    def create_db(self, uid, name, max_dbs):
        self.check_name("database", name)
        if self.db.execute("SELECT 1 FROM dbs WHERE user_id=? AND name=?", (uid, name)).fetchone(): raise DbError(409, "you already have a database called " + name)
        if self.db.execute("SELECT COUNT(*) FROM dbs WHERE user_id=?", (uid,)).fetchone()[0] >= max_dbs:
            raise DbError(402, "your plan allows %d database%s: upgrade for more" % (max_dbs, "" if max_dbs == 1 else "s"))
        with self.lock:
            self.db.execute("INSERT INTO dbs(user_id, name, created) VALUES(?,?,?)", (uid, name, time.time()))

    def drop_db(self, uid, name):
        with self.lock:
            self.db.execute("DELETE FROM docs WHERE user_id=? AND db=?", (uid, name))
            return self.db.execute("DELETE FROM dbs WHERE user_id=? AND name=?", (uid, name)).rowcount

    def drop_coll(self, uid, db, coll):
        with self.lock:
            return self.db.execute("DELETE FROM docs WHERE user_id=? AND db=? AND coll=?", (uid, db, coll)).rowcount

    def used_bytes(self, uid):
        return self.db.execute("SELECT COALESCE(SUM(size),0) FROM docs WHERE user_id=?", (uid,)).fetchone()[0]

    def _need(self, uid, db):
        if not self.db.execute("SELECT 1 FROM dbs WHERE user_id=? AND name=?", (uid, db)).fetchone(): raise DbError(404, "no database called " + db)

    # ---- documents ----
    def find(self, uid, db, coll, where=None, limit=50, offset=0):
        self._need(uid, db)
        q, args = "SELECT id, data, created, updated FROM docs WHERE user_id=? AND db=? AND coll=?", [uid, db, coll]
        for k, v in (where or {}).items():
            if not FIELD.match(k): raise DbError(400, "filter fields: letters, digits and _")
            q += " AND json_extract(data, ?) IS ?"; args += ["$." + k, v if not isinstance(v, bool) else int(v)]
        total = self.db.execute(q.replace("SELECT id, data, created, updated", "SELECT COUNT(*)"), args).fetchone()[0]
        rows = self.db.execute(q + " ORDER BY created DESC LIMIT ? OFFSET ?", args + [max(1, min(int(limit), 500)), max(0, int(offset))]).fetchall()
        return dict(total=total, docs=[dict(id=r[0], data=json.loads(r[1]), created=r[2], updated=r[3]) for r in rows])

    def get(self, uid, db, coll, doc_id):
        row = self.db.execute("SELECT data, created, updated FROM docs WHERE user_id=? AND db=? AND coll=? AND id=?", (uid, db, coll, doc_id)).fetchone()
        if not row: raise DbError(404, "no such document")
        return dict(id=doc_id, data=json.loads(row[0]), created=row[1], updated=row[2])

    def put_many(self, uid, db, coll, docs, quota_bytes):
        """Inserts or replaces documents ({"id"?: ..., ...fields}) all-or-nothing; returns their ids."""
        self._need(uid, db); self.check_name("collection", coll)
        rows, now = [], time.time()
        for d in docs:
            if not isinstance(d, dict): raise DbError(400, "each document must be a JSON object")
            doc_id = str(d.get("id") or d.get("_id") or secrets.token_hex(8))[:128]
            body = {k: v for k, v in d.items() if k not in ("id", "_id")}
            text = json.dumps(body, separators=(",", ":"), ensure_ascii=False)
            if len(text.encode()) > MAX_DOC: raise DbError(413, "a document is larger than 1 MB")
            rows.append((doc_id, text, len(text.encode())))
        with self.lock:
            old = {r[0]: r[1] for r in self.db.execute("SELECT id, size FROM docs WHERE user_id=? AND db=? AND coll=? AND id IN (%s)" % ",".join("?" * len(rows)),
                                                         [uid, db, coll] + [r[0] for r in rows]).fetchall()} if rows else {}
            grow = sum(r[2] for r in rows) - sum(old.values())
            if self.db.execute("SELECT COALESCE(SUM(size),0) FROM docs WHERE user_id=?", (uid,)).fetchone()[0] + grow > quota_bytes:
                raise DbError(402, "not enough database space on your plan (%d MB): upgrade or delete data" % (quota_bytes >> 20))
            self.db.execute("BEGIN")
            try:
                for doc_id, text, size in rows:
                    self.db.execute("INSERT INTO docs(user_id, db, coll, id, data, size, created, updated) VALUES(?,?,?,?,?,?,?,?)"
                                    " ON CONFLICT(user_id, db, coll, id) DO UPDATE SET data=excluded.data, size=excluded.size, updated=excluded.updated",
                                    (uid, db, coll, doc_id, text, size, now, now))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK"); raise
        return [r[0] for r in rows]

    def delete(self, uid, db, coll, doc_id):
        with self.lock:
            n = self.db.execute("DELETE FROM docs WHERE user_id=? AND db=? AND coll=? AND id=?", (uid, db, coll, doc_id)).rowcount
        if not n: raise DbError(404, "no such document")
