#!/usr/bin/env python3
"""Expert Stream MCP server, for AI clients only (no browser UI).

  python3 product/mcp_server.py --stdio
  ES_MCP_API_KEY=secret python3 product/mcp_server.py --http 8795     # streamable HTTP at /mcp, header X-API-Key required
Tools: run_moe, create_model, train_start, train_status, train_stop, list_models.
"""
import argparse, json, os, subprocess, sys, time, uuid
import es_common as C
from es_common import esmoe
from mcp.server.fastmcp import FastMCP

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = C.MODELS_DIR
def _blas():
    try:
        import glob, scipy_openblas64 as so  # es_moe needs the ILP64 scipy_cblas_sgemm64_ symbol
        return glob.glob(os.path.join(os.path.dirname(so.__file__), "lib", "*.so*"))[0]
    except Exception: return ""
BLAS = os.environ.get("ES_BLAS") if "scipy_openblas64" in os.environ.get("ES_BLAS", "") else _blas()  # needs scipy_cblas_sgemm64_
JOBS_DIR = os.path.join(MODELS, ".jobs")
mcp = FastMCP("expert-stream")
_models = {}

def _path(name):
    p = os.path.realpath(os.path.join(MODELS, name))
    if not p.startswith(os.path.realpath(MODELS) + os.sep): raise ValueError("bad model name")
    return p

@mcp.tool()
def list_models() -> list:
    """List models in the shared Expert Stream model library (trained in the Trainer, created here, etc.)."""
    return [dict(id=m["id"], name=m["name"], total_params=m["total_params"], active_params=m["active_params"], source=m["source"],
                 runnable=m["meta"].get("materialized", True), final_loss=(m["meta"].get("train") or {}).get("loss_history", [[0, None]])[-1][1])
            for m in C.list_models()]

@mcp.tool()
def create_model(name: str, total_params: float, experts: int = 0, topk: int = 2, train_steps: int = 0, train_text: str = "") -> dict:
    """Create an MoE model (byte-level, int8 experts on disk) sized 1M..70B total params.
    Sizes up to 2B are written in full; larger sizes write the config plan only (materialized=false).
    train_steps>0 (small sizes only, <=20M) actually trains it on train_text (or the repo docs) on CPU."""
    if total_params > 7e10: raise ValueError("create_model supports up to 70B total params")
    cfg = esmoe.plan(total_params, experts or None, topk); p = _path(name)
    if train_steps > 0:
        if cfg["total_params"] > 2e7: raise ValueError("training via create_model is limited to <=20M params; use train for bigger jobs")
        text = train_text or open(os.path.join(ROOT, "data/sample.txt"), errors="replace").read()
        if len(text) < 1000: text = text * (1000 // max(1, len(text)) + 1)
        logs = []; c = esmoe.train_torch(p, cfg, text, steps=min(train_steps, 2000), log=logs.append)
    else: c = esmoe.create(p, cfg)
    mid = C.register_model(os.environ.get("ES_MCP_USER_ID", "mcp"), name, p, c, "mcp"); return dict(id=mid, path=p, **{k: v for k, v in c.items() if k != "train"})

@mcp.tool()
def run_moe(name: str, prompt: str, max_new_tokens: int = 48, temperature: float = 0.7, cache_mb: int = 256) -> dict:
    """Run inference on an es-moe-v1 model. Only routed experts are read from disk through an LRU cache of cache_mb,
    so total size can exceed RAM. Returns text plus measured tok/s and expert read stats."""
    m = C.get_model(name)
    if not m: raise ValueError(f"no model {name!r}; see list_models")
    key = (m["id"], cache_mb)
    if key not in _models: _models[key] = esmoe.StreamedMoE(m["path"], cache_mb=cache_mb)
    return _models[key].generate(prompt, max_new=min(max_new_tokens, 512), temperature=temperature)

def _job(jid): return os.path.join(JOBS_DIR, jid)

@mcp.tool()
def train_start(mode: str = "moe", layers: int = 2, experts: int = 8, dim: int = 128, ffn: int = 256, steps: int = 20,
                topk: int = 2, rank: int = 8, cache_mb: int = 128, data_file: str = "", gguf: str = "", threads: int = 1) -> dict:
    """Start an es_train job in the background. mode=moe: from-scratch MoE (experts generated first);
    mode=finetune: LoRA on gguf (path on server). Returns a job id for train_status / train_stop."""
    jid = uuid.uuid4().hex[:10]; d = _job(jid); os.makedirs(d)
    ES_TRAIN = C.es_bin("es_train")
    data = data_file or os.path.join(ROOT, "data/sample.txt")
    env = dict(os.environ, ES_BLAS=BLAS, OPENBLAS_NUM_THREADS=str(threads))
    if mode == "moe":
        ef = os.path.join(d, "experts.bin"); a = ["--experts-file", ef, "--layers", str(layers), "--experts", str(experts), "--dim", str(dim), "--ffn", str(ffn)]
        cmd = f"{ES_TRAIN} moe gen {' '.join(a)} && exec {ES_TRAIN} moe train {' '.join(a)} --data {data} --topk {topk} --rank {rank} --steps {steps} --cache-mb {cache_mb}"
    elif mode == "finetune":
        if not gguf: raise ValueError("finetune needs gguf")
        cmd = f"exec {ES_TRAIN} finetune train -g {gguf} --data {data} -o {d}/out.eslora --steps {steps} --rank {rank} -j {threads}"
    else: raise ValueError("mode is moe or finetune")
    log = open(os.path.join(d, "log.txt"), "w")
    pr = subprocess.Popen(["nice", "-n", "10", "sh", "-c", cmd], stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    json.dump(dict(pid=pr.pid, cmd=cmd, started=time.time()), open(os.path.join(d, "job.json"), "w"))
    return dict(job_id=jid, pid=pr.pid, cmd=cmd)

@mcp.tool()
def train_status(job_id: str, tail: int = 20) -> dict:
    """Status and last log lines of a train job."""
    j = json.load(open(os.path.join(_job(job_id), "job.json")))
    try: os.kill(j["pid"], 0); st = open(f"/proc/{j['pid']}/stat").read().split()[2]; running = st != "Z"
    except OSError: running = False
    lines = open(os.path.join(_job(job_id), "log.txt"), errors="replace").read().splitlines()
    return dict(job_id=job_id, running=running, elapsed_s=round(time.time() - j["started"], 1), log_tail=lines[-tail:])

@mcp.tool()
def train_stop(job_id: str) -> dict:
    """Stop a running train job."""
    j = json.load(open(os.path.join(_job(job_id), "job.json")))
    try: os.killpg(j["pid"], 15); return dict(stopped=True)
    except OSError: return dict(stopped=False, note="not running")

def http_app(key):
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    app = mcp.streamable_http_app()
    class Auth(BaseHTTPMiddleware):
        async def dispatch(self, req, nxt):
            got = req.headers.get("x-api-key") or req.headers.get("authorization", "").removeprefix("Bearer ")
            if got != key: return JSONResponse({"error": "missing or bad X-API-Key"}, status_code=401)
            return await nxt(req)
    app.add_middleware(Auth); return app

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stdio", action="store_true"); ap.add_argument("--http", type=int); ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(); C.db(); os.makedirs(JOBS_DIR, exist_ok=True)
    if a.http:
        key = os.environ.get("ES_MCP_API_KEY")
        if not key: sys.exit("set ES_MCP_API_KEY for HTTP mode")
        import uvicorn; uvicorn.run(http_app(key), host=a.host, port=a.http, log_level="warning")
    else: mcp.run("stdio")
