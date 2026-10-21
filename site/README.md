# Expert Stream site (goes into Tharunmaks/Stream- main)
Pages: Plans (Expert Stream free Starter + Expert One ₹2,000/₹5,000/₹10,000, Razorpay TEST mode, the only place subscriptions are written),
Create (MoE 1M–70B; ≤2B written to disk, larger = plan only; optional tiny training), Run (streamed-expert inference), MCP (install docs), accounts.
    pip install -e ../es-common   (from the repo root: pip install -e es-common) flask razorpay "mcp>=1.10,<2" uvicorn torch scipy-openblas64
    export ES_JWT_SECRET=... ES_BIN_DIR=/path/to/prebuilt/bin DATABASE_URL=sqlite:///... (or postgres://...)
    RAZORPAY_KEY_ID=rzp_test_... RAZORPAY_KEY_SECRET=...   # unset = local stub orders, signature still verified server-side
    python3 app.py                       # :8780
    python3 mcp_server.py --stdio        # or ES_MCP_API_KEY=k python3 mcp_server.py --http 8795  (header X-API-Key)
Client configs: see /mcp page (Claude: command+args --stdio; Cursor/Grok: url + headers X-API-Key).
