import base64, hashlib, hmac, json, os, sqlite3, time, uuid
__all__ = ["db", "migrate", "signup", "login", "make_token", "verify_token", "user_from_request", "set_subscription",
           "subscription", "PLAN_LIMITS", "register_model", "list_models", "get_model", "MODELS_DIR", "es_bin", "COOKIE"]
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///" + os.path.expanduser("~/.expertstream/es.db"))
MODELS_DIR = os.environ.get("ES_MODELS_DIR", os.path.expanduser("~/.expertstream/models"))
SECRET = os.environ.get("ES_JWT_SECRET", "dev-only-change-me")
COOKIE = "es_session"   # same cookie name on both apps; set ES_COOKIE_DOMAIN in prod (e.g. .expertstream.in)
PLAN_LIMITS = {None: dict(max_params=2.5e6, max_steps=300, jobs=1), "one-starter": dict(max_params=2e7, max_steps=3000, jobs=2),
               "one-mid": dict(max_params=1e8, max_steps=20000, jobs=4), "one-top": dict(max_params=1e9, max_steps=100000, jobs=8)}

MIGRATIONS = [  # append-only; both apps run migrate() on start
    """CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, pw TEXT NOT NULL, created REAL)""",
    """CREATE TABLE IF NOT EXISTS subscriptions(user_id TEXT PRIMARY KEY, plan TEXT, status TEXT, order_id TEXT, payment_id TEXT, updated REAL)""",
    """CREATE TABLE IF NOT EXISTS models(id TEXT PRIMARY KEY, user_id TEXT, name TEXT, path TEXT, total_params REAL, active_params REAL,
       source TEXT, meta TEXT, created REAL)""",
    """CREATE TABLE IF NOT EXISTS schema_version(v INTEGER)""",
]

class _DB:
    """Tiny adapter: sqlite3 locally, psycopg (v3) when DATABASE_URL is postgres://. Use '?' placeholders."""
    def __init__(self, url):
        self.pg = url.startswith(("postgres://", "postgresql://"))
        if self.pg:
            import psycopg; self.c = psycopg.connect(url, autocommit=True)
        else:
            p = url.removeprefix("sqlite:///"); os.makedirs(os.path.dirname(p), exist_ok=True)
            self.c = sqlite3.connect(p, check_same_thread=False, isolation_level=None); self.c.execute("PRAGMA journal_mode=WAL")
    def q(self, sql, args=()):
        if self.pg: sql = sql.replace("?", "%s")
        cur = self.c.execute(sql, args); 
        try: cols = [d[0] for d in cur.description] if cur.description else []
        except Exception: cols = []
        return [dict(zip(cols, r)) for r in cur.fetchall()] if cols else []
_db = None
def db():
    global _db
    if _db is None: _db = _DB(DATABASE_URL); migrate()
    return _db
def migrate():
    d = _db
    for m in MIGRATIONS: d.q(m)
    if not d.q("SELECT v FROM schema_version"): d.q("INSERT INTO schema_version VALUES (?)", (len(MIGRATIONS),))
    os.makedirs(MODELS_DIR, exist_ok=True)

def _hash(pw, salt=None):
    salt = salt or os.urandom(16).hex(); return salt + "$" + hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=2**14, r=8, p=1).hex()
def signup(email, pw):
    email = email.strip().lower()
    if "@" not in email or len(pw) < 8: raise ValueError("valid email and 8+ char password needed")
    if db().q("SELECT id FROM users WHERE email=?", (email,)): raise ValueError("email already registered")
    uid = uuid.uuid4().hex; db().q("INSERT INTO users VALUES (?,?,?,?)", (uid, email, _hash(pw), time.time())); return uid
def login(email, pw):
    r = db().q("SELECT id, pw FROM users WHERE email=?", (email.strip().lower(),))
    if not r or not hmac.compare_digest(_hash(pw, r[0]["pw"].split("$")[0]), r[0]["pw"]): raise ValueError("wrong email or password")
    return r[0]["id"]

def _b64(b): return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
def make_token(uid, ttl=7 * 86400):  # HS256 JWT, shared secret across apps
    h = _b64(b'{"alg":"HS256","typ":"JWT"}'); p = _b64(json.dumps(dict(sub=uid, exp=int(time.time()) + ttl)).encode())
    return f"{h}.{p}." + _b64(hmac.new(SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
def verify_token(t):
    try:
        h, p, s = t.split(".")
        if not hmac.compare_digest(_b64(hmac.new(SECRET.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()), s): return None
        c = json.loads(base64.urlsafe_b64decode(p + "==")); return c["sub"] if c["exp"] > time.time() else None
    except Exception: return None
def user_from_request(req):
    t = req.cookies.get(COOKIE) or req.headers.get("authorization", "").removeprefix("Bearer ")
    uid = verify_token(t) if t else None
    if not uid: return None
    r = db().q("SELECT id, email FROM users WHERE id=?", (uid,)); return r[0] if r else None

def set_subscription(uid, plan, order_id, payment_id):
    db().q("DELETE FROM subscriptions WHERE user_id=?", (uid,))
    db().q("INSERT INTO subscriptions VALUES (?,?,?,?,?,?)", (uid, plan, "active", order_id, payment_id, time.time()))
def subscription(uid):
    r = db().q("SELECT plan, status FROM subscriptions WHERE user_id=?", (uid,)) if uid else []
    return r[0]["plan"] if r and r[0]["status"] == "active" else None

def register_model(uid, name, path, cfg, source):
    mid = uuid.uuid4().hex[:12]
    db().q("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?)", (mid, uid, name, path, cfg.get("total_params"), cfg.get("active_params"),
           source, json.dumps({k: v for k, v in cfg.items() if k in ("layers", "experts", "dim", "ffn", "topk", "train", "materialized", "backend", "quant")}), time.time()))
    return mid
def list_models(uid=None):
    rows = db().q("SELECT * FROM models ORDER BY created DESC") if uid is None else db().q("SELECT * FROM models WHERE user_id=? ORDER BY created DESC", (uid,))
    for r in rows: r["meta"] = json.loads(r["meta"] or "{}")
    return rows
def get_model(mid):
    r = db().q("SELECT * FROM models WHERE id=? OR name=?", (mid, mid)); return r[0] if r else None

def es_bin(name):
    """Prebuilt engine binaries only (es_train, es_chat). Never built from source here."""
    p = os.path.join(os.environ.get("ES_BIN_DIR", os.path.expanduser("~/.expertstream/bin")), name)
    if not os.access(p, os.X_OK): raise FileNotFoundError(f"{p} missing: set ES_BIN_DIR to the prebuilt engine binaries")
    return p
