import asyncio, json, os, sys, time
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
def j(r): return json.loads(r.content[0].text)
async def go(s):
    await s.initialize(); print("tools:", [t.name for t in (await s.list_tools()).tools])
    print("create 1M:", {k: v for k, v in j(await s.call_tool("create_model", {"name": "m1", "total_params": 1e6})).items() if k in ("total_params", "layers", "materialized")})
    print("create 70B plan:", {k: v for k, v in j(await s.call_tool("create_model", {"name": "m70b", "total_params": 7e10})).items() if k in ("total_params", "active_params", "materialized")})
    print("run m1 (random init):", j(await s.call_tool("run_moe", {"name": "m1", "prompt": "hi", "max_new_tokens": 16})))
    t = j(await s.call_tool("train_start", {"mode": "moe", "steps": 10})); print("train_start:", t["job_id"])
    for _ in range(40):
        st = j(await s.call_tool("train_status", {"job_id": t["job_id"]}))
        if not st["running"]: break
        time.sleep(1)
    print("train_status:", st["running"], st["log_tail"][-4:])
async def main():
    if sys.argv[1] == "stdio":
        async with stdio_client(StdioServerParameters(command="python3", args=["product/mcp_server.py", "--stdio"])) as (r, w):
            async with ClientSession(r, w) as s: await go(s)
    else:
        async with streamablehttp_client("http://127.0.0.1:8795/mcp", headers={"X-API-Key": os.environ["ES_MCP_API_KEY"]}) as (r, w, _):
            async with ClientSession(r, w) as s:
                await s.initialize(); print("http tools:", [t.name for t in (await s.list_tools()).tools])
                print("http run tiny-moe:", j(await s.call_tool("run_moe", {"name": "tiny-moe", "prompt": "Expert Stream ", "max_new_tokens": 40, "temperature": 0.4, "cache_mb": 1})))
asyncio.run(main())
