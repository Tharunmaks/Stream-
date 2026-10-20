"""ExpertCloud Database: document databases for every user, one SQLite file per database.

Layout: ROOT/registry.db lists every database (owner, name, size), its indexes and snapshots; ROOT/u<user>/<name>.db holds the
documents of one database (table docs: collection, id, JSON data). One file per database keeps users apart, lets each database
have its own indexes, and makes backups a file copy.

Documents are JSON objects with an id, grouped in collections. Features:
  * queries: field filters with operators ($eq $ne $gt $gte $lt $lte $in $nin $exists $contains $startsWith) on nested fields
    ("address.city"), $or of filters, sort on any fields, the fields to return, free-text search, limit and offset, counts
  * partial updates: set fields, $inc numbers, $push to lists, $unset fields, in one atomic step
  * indexes on any field of a collection (used by filters and sorts on that field)
  * transactions: a batch of puts, patches and deletes over several collections, all or nothing
  * snapshots: a copy of a database at a moment, restorable; the whole database downloads as one SQLite file
Space is the bytes of document JSON plus the snapshots' files; the gateway caps it by plan and purchased extra space.
"""
import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import threading
import time

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
PATH = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}(\.[A-Za-z_][A-Za-z0-9_]{0,63}){0,7}$")
MAX_DOC = 1 << 20   # 1 MB per document
OPS = {"$eq": "=", "$ne": "!=", "$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}

REGISTRY = """
CREATE TABLE IF NOT EXISTS dbs (user_id INTEGER NOT NULL, name TEXT NOT NULL, created REAL NOT NULL, bytes INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, name));
CREATE TABLE IF NOT EXISTS indexes (user_id INTEGER NOT NULL, db TEXT NOT NULL, coll TEXT NOT NULL, path TEXT NOT NULL, created REAL NOT NULL,
    PRIMARY KEY (user_id, db, coll, path));
CREATE TABLE IF NOT EXISTS snapshots (user_id INTEGER NOT NULL, db TEXT NOT NULL, id TEXT NOT NULL, label TEXT, created REAL NOT NULL,
    bytes INTEGER NOT NULL, PRIMARY KEY (user_id, db, id));
"""
DOCS = """
CREATE TABLE IF NOT EXISTS docs (coll TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL, size INTEGER NOT NULL,
    created REAL NOT NULL, updated REAL NOT NULL, PRIMARY KEY (coll, id));
CREATE INDEX IF NOT EXISTS docs_by_time ON docs(coll, created);
"""


class DbError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def jpath(path):
    if not isinstance(path, str) or not PATH.match(path): raise DbError(400, "field %r: letters, digits and _, nested with dots" % (path,))
    return "json_extract(data, '$.%s')" % path   # validated above, so it is safe in the SQL text (indexes need the literal path)


def sqlval(v):
    if isinstance(v, bool): return int(v)
    if isinstance(v, (int, float, str)) or v is None: return v
    raise DbError(400, "filter values are numbers, strings, booleans or null")


def index_name(coll, path):
    return "ix_" + hashlib.sha1(("%s\0%s" % (coll, path)).encode()).hexdigest()[:16]


class DocStore:
    def __init__(self, root, legacy=None):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.lock = threading.RLock()
        self.reg = sqlite3.connect(os.path.join(root, "registry.db"), check_same_thread=False, isolation_level=None)
        self.reg.execute("PRAGMA journal_mode=WAL")
        self.reg.executescript(REGISTRY)
        self.open = {}   # (user, db) -> connection, most recent last
        if legacy and os.path.exists(legacy): self._migrate(legacy)

    # ---- files ----
    def _file(self, uid, db):
        return os.path.join(self.root, "u%d" % int(uid), db + ".db")

    def _snap_file(self, uid, db, sid):
        return os.path.join(self.root, "u%d" % int(uid), "snapshots", "%s-%s.db" % (db, sid))

    def _conn(self, uid, db):
        key = (uid, db)
        c = self.open.pop(key, None)
        if c is None:
            if not self.reg.execute("SELECT 1 FROM dbs WHERE user_id=? AND name=?", (uid, db)).fetchone(): raise DbError(404, "no database called " + db)
            os.makedirs(os.path.dirname(self._file(uid, db)), exist_ok=True)
            c = sqlite3.connect(self._file(uid, db), check_same_thread=False, isolation_level=None)
            c.execute("PRAGMA journal_mode=WAL"); c.executescript(DOCS)
            while len(self.open) >= 64: self.open.pop(next(iter(self.open))).close()   # keep at most 64 files open
        self.open[key] = c
        return c

    def _close(self, uid, db):
        c = self.open.pop((uid, db), None)
        if c: c.close()

    def _migrate(self, legacy):
        """Databases from the first version (all users in one file) move to one file each."""
        old = sqlite3.connect(legacy)
        try:
            dbs = old.execute("SELECT user_id, name, created FROM dbs").fetchall()
        except sqlite3.OperationalError:
            old.close(); return
        for uid, name, created in dbs:
            if self.reg.execute("SELECT 1 FROM dbs WHERE user_id=? AND name=?", (uid, name)).fetchone(): continue
            self.reg.execute("INSERT INTO dbs(user_id, name, created) VALUES(?,?,?)", (uid, name, created))
            c = self._conn(uid, name)
            rows = old.execute("SELECT coll, id, data, size, created, updated FROM docs WHERE user_id=? AND db=?", (uid, name)).fetchall()
            with self.lock:
                c.execute("BEGIN"); c.executemany("INSERT OR REPLACE INTO docs VALUES(?,?,?,?,?,?)", rows); c.execute("COMMIT")
                self.reg.execute("UPDATE dbs SET bytes=? WHERE user_id=? AND name=?", (sum(r[3] for r in rows), uid, name))
        old.close()
        os.rename(legacy, legacy + ".migrated")

    @staticmethod
    def check_name(kind, name):
        if not isinstance(name, str) or not NAME.match(name): raise DbError(400, "%s name: 1 to 64 letters, digits, - or _, starting with a letter or digit" % kind)

    # ---- databases ----
    def list_dbs(self, uid):
        out = []
        for name, created, size in self.reg.execute("SELECT name, created, bytes FROM dbs WHERE user_id=? ORDER BY name", (uid,)).fetchall():
            c = self._conn(uid, name)
            colls = c.execute("SELECT coll, COUNT(*), COALESCE(SUM(size),0) FROM docs GROUP BY coll ORDER BY coll").fetchall()
            out.append(dict(name=name, created=created, bytes=size, docs=sum(x[1] for x in colls),
                            collections=[dict(name=k, docs=n, bytes=b) for k, n, b in colls],
                            snapshots=self.reg.execute("SELECT COUNT(*) FROM snapshots WHERE user_id=? AND db=?", (uid, name)).fetchone()[0]))
        return out

    def create_db(self, uid, name, max_dbs):
        self.check_name("database", name)
        with self.lock:
            if self.reg.execute("SELECT 1 FROM dbs WHERE user_id=? AND name=?", (uid, name)).fetchone(): raise DbError(409, "you already have a database called " + name)
            if self.reg.execute("SELECT COUNT(*) FROM dbs WHERE user_id=?", (uid,)).fetchone()[0] >= max_dbs:
                raise DbError(402, "your plan allows %d database%s: upgrade for more" % (max_dbs, "" if max_dbs == 1 else "s"))
            self.reg.execute("INSERT INTO dbs(user_id, name, created) VALUES(?,?,?)", (uid, name, time.time()))
        self._conn(uid, name)

    def drop_db(self, uid, name):
        with self.lock:
            if not self.reg.execute("SELECT 1 FROM dbs WHERE user_id=? AND name=?", (uid, name)).fetchone(): return 0
            self._close(uid, name)
            for suffix in ("", "-wal", "-shm"):
                try: os.remove(self._file(uid, name) + suffix)
                except FileNotFoundError: pass
            for (sid,) in self.reg.execute("SELECT id FROM snapshots WHERE user_id=? AND db=?", (uid, name)).fetchall():
                try: os.remove(self._snap_file(uid, name, sid))
                except FileNotFoundError: pass
            for t in ("dbs", "indexes", "snapshots"):
                self.reg.execute("DELETE FROM %s WHERE user_id=? AND %s=?" % (t, "name" if t == "dbs" else "db"), (uid, name))
        return 1

    def drop_coll(self, uid, db, coll):
        c = self._conn(uid, db)
        with self.lock:
            gone = c.execute("SELECT COALESCE(SUM(size),0) FROM docs WHERE coll=?", (coll,)).fetchone()[0]
            n = c.execute("DELETE FROM docs WHERE coll=?", (coll,)).rowcount
            for (path,) in self.reg.execute("SELECT path FROM indexes WHERE user_id=? AND db=? AND coll=?", (uid, db, coll)).fetchall():
                c.execute('DROP INDEX IF EXISTS "%s"' % index_name(coll, path))
            self.reg.execute("DELETE FROM indexes WHERE user_id=? AND db=? AND coll=?", (uid, db, coll))
            self.reg.execute("UPDATE dbs SET bytes=bytes-? WHERE user_id=? AND name=?", (gone, uid, db))
        return n

    def used_bytes(self, uid):
        docs = self.reg.execute("SELECT COALESCE(SUM(bytes),0) FROM dbs WHERE user_id=?", (uid,)).fetchone()[0]
        snaps = self.reg.execute("SELECT COALESCE(SUM(bytes),0) FROM snapshots WHERE user_id=?", (uid,)).fetchone()[0]
        return docs + snaps

    def _room(self, uid, grow, quota):
        if grow > 0 and self.used_bytes(uid) + grow > quota:
            raise DbError(402, "not enough database space (%s of your %s used): buy more space, upgrade, or delete data"
                          % (human(self.used_bytes(uid)), human(quota)))

    # ---- queries ----
    def _where(self, flt, args):
        """SQL for a filter object: {"field": value | {"$op": value, ...}, "$or": [filter, ...]}"""
        if not isinstance(flt, dict): raise DbError(400, "a filter is a JSON object")
        parts = []
        for k, v in flt.items():
            if k == "$or":
                if not isinstance(v, list) or not v: raise DbError(400, "$or takes a list of filters")
                parts.append("(" + " OR ".join(self._where(f, args) or "1" for f in v) + ")")
                continue
            e = jpath(k)
            conds = v if isinstance(v, dict) and v and all(str(x).startswith("$") for x in v) else {"$eq": v}
            for op, x in conds.items():
                if op in OPS:
                    if x is None: parts.append("%s IS %sNULL" % (e, "NOT " if op == "$ne" else "")); continue
                    parts.append("%s %s ?" % (e, OPS[op])); args.append(sqlval(x))
                elif op in ("$in", "$nin"):
                    if not isinstance(x, list) or not x or len(x) > 1000: raise DbError(400, "%s takes a list of 1 to 1000 values" % op)
                    parts.append("%s %sIN (%s)" % (e, "NOT " if op == "$nin" else "", ",".join("?" * len(x)))); args.extend(sqlval(y) for y in x)
                elif op == "$exists":
                    parts.append("json_type(data, '$.%s') IS %sNULL" % (k, "NOT " if x else ""))
                elif op == "$contains":   # text containing x, or a list holding x
                    parts.append("(CASE json_type(data, '$.%s') WHEN 'array' THEN EXISTS (SELECT 1 FROM json_each(data, '$.%s') WHERE value = ?)"
                                 " ELSE instr(lower(%s), lower(?)) > 0 END)" % (k, k, e))
                    args.extend([sqlval(x), str(x)])
                elif op == "$startsWith":
                    if not isinstance(x, str): raise DbError(400, "$startsWith takes text")
                    parts.append("substr(%s, 1, ?) = ?" % e); args.extend([len(x), x])
                else:
                    raise DbError(400, "unknown operator %s" % op)
        return " AND ".join(parts)

    def find(self, uid, db, coll, where=None, limit=50, offset=0, sort=None, fields=None, search=None):
        c = self._conn(uid, db)
        args = [coll]
        q = " FROM docs WHERE coll=?"
        w = self._where(where or {}, args)
        if w: q += " AND " + w
        if search:
            q += " AND instr(lower(data), lower(?)) > 0"; args.append(str(search)[:200])
        total = c.execute("SELECT COUNT(*)" + q, args).fetchone()[0]
        order = []
        for s in (sort or []):
            desc = isinstance(s, str) and s.startswith("-")
            order.append("%s %s" % (jpath(s[1:] if desc else s), "DESC" if desc else "ASC"))
        order.append("created DESC" if not order else "id")
        rows = c.execute("SELECT id, data, created, updated" + q + " ORDER BY " + ", ".join(order) + " LIMIT ? OFFSET ?",
                         args + [max(1, min(int(limit), 1000)), max(0, int(offset))]).fetchall()
        docs = []
        for r in rows:
            d = json.loads(r[1])
            if fields: d = project(d, fields)
            docs.append(dict(id=r[0], data=d, created=r[2], updated=r[3]))
        return dict(total=total, docs=docs)

    def explain(self, uid, db, coll, where):
        """How SQLite runs a filter (to show that an index is used)."""
        c = self._conn(uid, db); args = [coll]
        w = self._where(where or {}, args)
        return [r[3] for r in c.execute("EXPLAIN QUERY PLAN SELECT id FROM docs WHERE coll=?" + (" AND " + w if w else ""), args).fetchall()]

    def get(self, uid, db, coll, doc_id):
        row = self._conn(uid, db).execute("SELECT data, created, updated FROM docs WHERE coll=? AND id=?", (coll, doc_id)).fetchone()
        if not row: raise DbError(404, "no such document")
        return dict(id=doc_id, data=json.loads(row[0]), created=row[1], updated=row[2])

    # ---- writes ----
    @staticmethod
    def _encode(d):
        if not isinstance(d, dict): raise DbError(400, "each document must be a JSON object")
        doc_id = str(d.get("id") or d.get("_id") or secrets.token_hex(8))[:128]
        text = json.dumps({k: v for k, v in d.items() if k not in ("id", "_id")}, separators=(",", ":"), ensure_ascii=False)
        size = len(text.encode())
        if size > MAX_DOC: raise DbError(413, "a document is larger than 1 MB")
        return doc_id, text, size

    def _apply(self, uid, db, c, ops, quota):
        """ops: [("put", coll, doc) | ("patch", coll, id, changes) | ("delete", coll, id)] in one transaction. Returns results."""
        now, grow, out = time.time(), 0, []
        c.execute("BEGIN IMMEDIATE")
        try:
            for op in ops:
                kind, coll = op[0], op[1]
                self.check_name("collection", coll)
                if kind == "put":
                    doc_id, text, size = self._encode(op[2])
                    old = c.execute("SELECT size FROM docs WHERE coll=? AND id=?", (coll, doc_id)).fetchone()
                    c.execute("INSERT INTO docs VALUES(?,?,?,?,?,?) ON CONFLICT(coll, id) DO UPDATE SET data=excluded.data, size=excluded.size,"
                              " updated=excluded.updated", (coll, doc_id, text, size, now, now))
                    grow += size - (old[0] if old else 0); out.append(doc_id)
                elif kind == "patch":
                    row = c.execute("SELECT data, size FROM docs WHERE coll=? AND id=?", (coll, op[2])).fetchone()
                    if not row: raise DbError(404, "no document %s in %s" % (op[2], coll))
                    d = patched(json.loads(row[0]), op[3])
                    _, text, size = self._encode(dict(d, id=op[2]))
                    c.execute("UPDATE docs SET data=?, size=?, updated=? WHERE coll=? AND id=?", (text, size, now, coll, op[2]))
                    grow += size - row[1]; out.append(dict(id=op[2], data=d))
                elif kind == "delete":
                    row = c.execute("SELECT size FROM docs WHERE coll=? AND id=?", (coll, op[2])).fetchone()
                    if not row: raise DbError(404, "no document %s in %s" % (op[2], coll))
                    c.execute("DELETE FROM docs WHERE coll=? AND id=?", (coll, op[2]))
                    grow -= row[0]; out.append(op[2])
                else:
                    raise DbError(400, "unknown operation %s" % kind)
            self._room(uid, grow, quota)
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK"); raise
        self.reg.execute("UPDATE dbs SET bytes=bytes+? WHERE user_id=? AND name=?", (grow, uid, db))
        return out

    def put_many(self, uid, db, coll, docs, quota):
        c = self._conn(uid, db)
        with self.lock:
            return self._apply(uid, db, c, [("put", coll, d) for d in docs], quota)

    def patch(self, uid, db, coll, doc_id, changes, quota):
        c = self._conn(uid, db)
        with self.lock:
            return self._apply(uid, db, c, [("patch", coll, doc_id, changes)], quota)[0]

    def delete(self, uid, db, coll, doc_id):
        c = self._conn(uid, db)
        with self.lock:
            self._apply(uid, db, c, [("delete", coll, doc_id)], 1 << 62)

    def batch(self, uid, db, ops, quota):
        """[{"op": "put", "collection": c, "doc": {...}} | {"op": "patch", "collection": c, "id": i, "changes": {...}}
            | {"op": "delete", "collection": c, "id": i}]: all applied, or none."""
        if not isinstance(ops, list) or not ops or len(ops) > 1000: raise DbError(400, "a batch is a list of 1 to 1000 operations")
        conv = []
        for o in ops:
            if not isinstance(o, dict): raise DbError(400, "each operation is an object")
            k, coll = o.get("op"), o.get("collection")
            if k == "put": conv.append(("put", coll, o.get("doc")))
            elif k == "patch": conv.append(("patch", coll, str(o.get("id")), o.get("changes") or {}))
            elif k == "delete": conv.append(("delete", coll, str(o.get("id"))))
            else: raise DbError(400, "op is put, patch or delete")
        c = self._conn(uid, db)
        with self.lock:
            return self._apply(uid, db, c, conv, quota)

    # ---- indexes ----
    def indexes(self, uid, db, coll):
        return [dict(field=p, created=t) for p, t in self.reg.execute("SELECT path, created FROM indexes WHERE user_id=? AND db=? AND coll=? ORDER BY path",
                                                                       (uid, db, coll)).fetchall()]

    def create_index(self, uid, db, coll, path, max_indexes):
        self.check_name("collection", coll); e = jpath(path)
        c = self._conn(uid, db)
        with self.lock:
            if self.reg.execute("SELECT 1 FROM indexes WHERE user_id=? AND db=? AND coll=? AND path=?", (uid, db, coll, path)).fetchone(): return
            if self.reg.execute("SELECT COUNT(*) FROM indexes WHERE user_id=? AND db=?", (uid, db)).fetchone()[0] >= max_indexes:
                raise DbError(402, "your plan allows %d indexes per database" % max_indexes)
            c.execute('CREATE INDEX IF NOT EXISTS "%s" ON docs(coll, %s)' % (index_name(coll, path), e))
            self.reg.execute("INSERT INTO indexes VALUES(?,?,?,?,?)", (uid, db, coll, path, time.time()))

    def drop_index(self, uid, db, coll, path):
        jpath(path)
        c = self._conn(uid, db)
        with self.lock:
            c.execute('DROP INDEX IF EXISTS "%s"' % index_name(coll, path))
            return self.reg.execute("DELETE FROM indexes WHERE user_id=? AND db=? AND coll=? AND path=?", (uid, db, coll, path)).rowcount

    # ---- snapshots and export ----
    def snapshots(self, uid, db):
        self._conn(uid, db)
        return [dict(id=i, label=l, created=t, bytes=b) for i, l, t, b in
                self.reg.execute("SELECT id, label, created, bytes FROM snapshots WHERE user_id=? AND db=? ORDER BY created DESC", (uid, db)).fetchall()]

    def snapshot(self, uid, db, label, quota, max_snaps):
        c = self._conn(uid, db)
        with self.lock:
            if self.reg.execute("SELECT COUNT(*) FROM snapshots WHERE user_id=? AND db=?", (uid, db)).fetchone()[0] >= max_snaps:
                raise DbError(402, "your plan keeps %d snapshot%s per database: delete one first" % (max_snaps, "" if max_snaps == 1 else "s"))
            sid = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + secrets.token_hex(3)
            path = self._snap_file(uid, db, sid)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            dst = sqlite3.connect(path)
            c.backup(dst); dst.execute("VACUUM"); dst.close()
            size = os.path.getsize(path)
            try:
                self._room(uid, size, quota)
            except DbError:
                os.remove(path); raise
            self.reg.execute("INSERT INTO snapshots VALUES(?,?,?,?,?,?)", (uid, db, sid, str(label or "")[:80], time.time(), size))
        return dict(id=sid, bytes=size)

    def restore(self, uid, db, sid):
        """The database goes back to the snapshot (a snapshot of the current state is not taken: make one first)."""
        path = self._snap_file(uid, db, sid)
        if not self.reg.execute("SELECT 1 FROM snapshots WHERE user_id=? AND db=? AND id=?", (uid, db, sid)).fetchone() or not os.path.exists(path):
            raise DbError(404, "no such snapshot")
        c = self._conn(uid, db)
        with self.lock:
            src = sqlite3.connect(path); src.backup(c); src.close()
            size = c.execute("SELECT COALESCE(SUM(size),0) FROM docs").fetchone()[0]
            self.reg.execute("UPDATE dbs SET bytes=? WHERE user_id=? AND name=?", (size, uid, db))
            # the indexes are the database's current ones (the registry), whatever the snapshot had
            want = {index_name(k, p): (k, p) for k, p in self.reg.execute("SELECT coll, path FROM indexes WHERE user_id=? AND db=?", (uid, db)).fetchall()}
            for (name,) in c.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'ix_%'").fetchall():
                if name not in want: c.execute('DROP INDEX IF EXISTS "%s"' % name)
            for name, (k, p) in want.items(): c.execute('CREATE INDEX IF NOT EXISTS "%s" ON docs(coll, %s)' % (name, jpath(p)))
            return dict(ok=True, bytes=size)

    def drop_snapshot(self, uid, db, sid):
        with self.lock:
            n = self.reg.execute("DELETE FROM snapshots WHERE user_id=? AND db=? AND id=?", (uid, db, sid)).rowcount
            try: os.remove(self._snap_file(uid, db, sid))
            except FileNotFoundError: pass
        return n

    def export_file(self, uid, db):
        """A consistent copy of the database as one SQLite file (for download); the caller deletes it."""
        c = self._conn(uid, db)
        path = os.path.join(self.root, "u%d" % int(uid), ".export-%s-%s.db" % (db, secrets.token_hex(4)))
        dst = sqlite3.connect(path); c.backup(dst); dst.close()
        return path


def project(d, fields):
    out = {}
    for f in fields:
        cur, ok = d, True
        for part in f.split("."):
            if isinstance(cur, dict) and part in cur: cur = cur[part]
            else: ok = False; break
        if not ok: continue
        node = out; parts = f.split(".")
        for part in parts[:-1]: node = node.setdefault(part, {})
        node[parts[-1]] = cur
    return out


def patched(d, changes):
    """A partial update: plain keys are set (dots reach into nested objects); $inc adds, $push appends, $unset removes."""
    if not isinstance(changes, dict) or not changes: raise DbError(400, "changes must be a non-empty JSON object")

    def walk(path, make=True):
        parts = path.split(".")
        if not all(PATH.match(p) for p in parts): raise DbError(400, "field %r: letters, digits and _, nested with dots" % path)
        node = d
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                if not make: return None, None
                node[p] = {}
            node = node[p]
        return node, parts[-1]

    for k, v in changes.items():
        if k == "$inc":
            for f, n in (v or {}).items():
                if not isinstance(n, (int, float)) or isinstance(n, bool): raise DbError(400, "$inc takes numbers")
                node, last = walk(f)
                cur = node.get(last, 0)
                if not isinstance(cur, (int, float)) or isinstance(cur, bool): raise DbError(400, "$inc: %s is not a number" % f)
                node[last] = cur + n
        elif k == "$push":
            for f, x in (v or {}).items():
                node, last = walk(f)
                cur = node.setdefault(last, [])
                if not isinstance(cur, list): raise DbError(400, "$push: %s is not a list" % f)
                cur.append(x)
        elif k == "$unset":
            for f in (v if isinstance(v, list) else [v]):
                node, last = walk(f, make=False)
                if node is not None: node.pop(last, None)
        elif k.startswith("$"):
            raise DbError(400, "unknown update %s ($inc, $push, $unset, or plain fields)" % k)
        elif k not in ("id", "_id"):
            node, last = walk(k)
            node[last] = v
    return d


def human(n):
    for unit, size in (("TB", 1 << 40), ("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= size: return "%.1f %s" % (n / size, unit)
    return "%d bytes" % n
