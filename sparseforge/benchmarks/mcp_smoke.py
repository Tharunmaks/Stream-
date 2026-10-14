"""Start sparseforge-mcp over stdio with the official SDK client, list tools, call a few."""
import asyncio, json, os, sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

async def main():
    env = {**os.environ, "PYTHONPATH": ROOT, "SPARSEFORGE_MODELS": "/tmp/sf_mcp_models", "SPARSEFORGE_HOME": "/tmp/sf_mcp_jobs"}
    params = StdioServerParameters(command=sys.executable, args=["-m", "sparseforge.mcp_server"], env=env)
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            print("TOOLS", len(tools), [t.name for t in tools])
            async def call(_tool, **a):
                res = await s.call_tool(_tool, a)
                txt = " ".join(getattr(c, "text", "") for c in res.content)
                print("CALL", _tool, txt[:300]); return txt
            await call("detect_hardware")
            await call("create_model", name="demo", n_experts=4096, hidden=64)
            job = json.loads(await call("train_model", name="demo", corpus_path="/tmp/train.txt", steps=20, batch=64, threads=2))
            for _ in range(120):
                st = json.loads(await call("get_job_status", job_id=job["job_id"]))
                if st.get("state") not in ("running", "starting"): break
                await asyncio.sleep(5)
            await call("sample_model", name="demo", prompt="def ", n_tokens=40)
asyncio.run(main())
