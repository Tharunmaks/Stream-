# ExpertStream Cloud & OmniRoute Gateway

Pool your own machines (PCs, laptops, Raspberry Pis, friends' computers) into one OpenAI-compatible endpoint with free API keys,
running the native ExpertStream engine on every node. No third-party API, no paid dependency.

```
Stream-/
├── gateway/
│   ├── omni_router.py      OpenAI API (/v1/chat/completions JSON + SSE, /v1/models), keys, Token Killer, cache, admin API, /cloud/ws
│   ├── store.py            SQLite (expertcloud.db): hashed keys, contribute-to-use ledger, response cache (+ FTS5)
│   └── compress.py         Token Killer: safe / aggressive prompt compaction, code blocks untouched
├── cloud/
│   ├── controller.py       node registry, Auto-Combo scoring router, queue (contributors first), failover
│   └── worker_agent.py     one command turns a machine into a node (outbound WebSocket: no port forwarding)
├── engine/
│   └── expert_streamer.py  drives the native engine (es_serve): processes per request slot, streaming, deploy via es_get
├── mcp/
│   └── expertstream_mcp.py MCP server (stdio + HTTP 8765): list_cloud_nodes, deploy_model_to_node, generate_free_api_key, route_inference
├── dashboard/
│   └── index.html          control panel: nodes, speed, savings, key generator, join command
├── scripts/test_cloud.py   end-to-end test (gateway + 2 nodes + MCP on one machine)
└── src/, tools/            the native engine (es_serve, es_get, es_chat ...)
```

## Quick start

```sh
make es_serve es_get                                   # the native engine, on every machine
pip install -r requirements-cloud.txt
python3 gateway/omni_router.py --host 0.0.0.0          # prints a sign-in link for the dashboard, the admin token and the join command
# on each machine that should serve (models in ~/models, or deploy them from the dashboard/MCP):
python3 cloud/worker_agent.py --master ws://HOST:8000 --token JOIN_TOKEN [--key sk-expert-...] [--slots 2]
# any OpenAI client:
curl http://HOST:8000/v1/chat/completions -H "Authorization: Bearer sk-expert-..." \
     -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}],"stream":true}'
python3 mcp/expertstream_mcp.py --http 8765 --stdio     # MCP for agents (reads ~/.expertcloud/admin_token)
```

## How it works

| Layer | What it does |
|---|---|
| ExpertCloud | Workers dial out to the gateway over a WebSocket, so home routers need no port forwarding. Every 5 s they report CPU, RAM, GPU (nvidia-smi), disk read speed, load and which models are warm. Tokens a node generates are credited to its owner's key (`--key`); keys that contributed are exempt from the rate limit and are served first when every node is busy. |
| ExpertStream engine | Native code (src/es_engine.c): the model file is memory-mapped, only the experts each token routes to are read, the next layer's experts are prefetched, an expert cache sized to the free RAM keeps the hot ones, Turbo checks guessed tokens in one batched pass. `engine/expert_streamer.py` runs one engine process per request slot (a mapped model's weights are shared between processes through the page cache). |
| OmniRoute | OpenAI-compatible API with SSE streaming; free `sk-expert-...` keys (stored as SHA-256 hashes); Token Killer; exact response cache at temperature 0 (opt-in similar-prompt cache); Auto-Combo routing (free slots, time to first token, free RAM for model + context, weights warm, measured speed, recent failures) with failover to the next node when one drops before its first token. |
| MCP + dashboard | Agents and people manage the same cloud through the admin API. |

Measured by `scripts/test_cloud.py` (gateway, two nodes and the MCP server on one 4-core PC, SmolLM2-135M): all 10 checks pass, a
cache hit is looked up in under 1 ms inside the gateway, a killed node's requests go to the other node, four simultaneous requests
are spread over two one-slot nodes.

## What it does not do (yet)

* **Not Groq speed.** Speed is what the engine gets on each node's CPU (for example Qwen3-30B-A3B Q2_K: 8 tokens/s on a 4-core PC
  with the model in RAM; see the README). More nodes serve more users at once; they do not make one answer faster.
* **No continuous batching inside one engine process**: each process answers one request at a time. Concurrency comes from slots
  (several processes per node) and from several nodes; extra requests queue.
* **No separate draft model** for speculative decoding: the engine's Turbo guesses from the text so far.
* **The engine is CPU-only.** GPU memory is reported for scheduling, not used.
* Token Killer savings depend on the prompt: tidy prompts save little; duplicated or padded ones save a lot.

## Your own public cloud

ExpertCloud is a cloud platform of its own, like the big clouds but built from computers people share:

* **Public page** at `/`: live numbers (computers online, CPU cores, memory, speed, models), plans, sign-up and sign-in.
* **Console** for every signed-in person: Home (their usage and the cloud's status), Models, Playground, API keys,
  Compute (share a computer with one command; its work is credited to them and lifts their rate limit) and Docs.
* **Owner dashboard** at `/dashboard` (admin token): all computers, all keys, deploys, and two switches: whether anyone can
  create an account, and whether users may share their computers (a shared computer sees the prompts it answers).

Accounts are a username and a password (scrypt-hashed); sessions last 30 days. Sign-ups are limited to 5 an hour per address
and sign-ins to 10 failures per 10 minutes.

## Plans, payments, database and ExpertStream

**Plans** (monthly, INR; `gateway/plans.py`, override with a `plans.json` next to the database):

| ExpertCloud | Price | Limits |
|---|---|---|
| Free | ₹0 | 25 AI requests/day, 5/min, answers ≤256 tokens, 1 API key, 5 MB database |
| Pro | ₹1,700 | 2,000/day, 60/min, ≤4,096 tokens, 10 keys, 1 GB database (10 databases), faster queue |
| Max | ₹4,000 | 10,000/day, 200/min, ≤8,192 tokens, 50 keys, 10 GB database (50 databases), front of the queue |

| ExpertStream | Price |
|---|---|
| Free | ₹0 |
| Plus | ₹800 |
| Pro | ₹2,000 |

ExpertStream plans are sold here; the ExpertStream app applies its own features from the plan id.

**Payments: Razorpay** (UPI, cards, net banking). Owner dashboard > Billing: key id and secret (or `RAZORPAY_KEY_ID`,
`RAZORPAY_KEY_SECRET`, `RAZORPAY_WEBHOOK_SECRET`), and the webhook URL `https://YOUR-HOST/billing/webhook` (events
`payment.captured`, `order.paid`). Every payment is checked by its Razorpay signature; a plan lasts 30 days, paying again extends
it, and one payment never counts twice. The owner can also give plans by hand (Users > Change plan).

**Database**: each user gets document databases (collections of JSON documents), in the console and over the API with their
session or API key: `GET/POST /db`, `GET/POST /db/{db}/{collection}` (POST takes one object or a list: an import of another
database), `GET/PUT/DELETE /db/{db}/{collection}/{id}`, filters with `?where={"field": value}`. Space and count follow the plan.

**Connecting the ExpertStream app** (one account for both, like Drive and Photos): the app signs users in with
`POST /auth/login {username, password}` (or sends them to sign up at `/`), reads `GET /account/plans` for their ExpertStream plan
(`es_free`, `es_plus`, `es_pro`), and can keep their data in `/db/...`. The API allows calls from any website (CORS) since it uses
bearer tokens. To move the app's current data over, export it as JSON and import it per collection. The console's app switcher
opens the ExpertStream address set in the owner's Billing page.

## Easiest way: the `cloud` command

```
scripts/expertcloud.sh shortcut     # once: makes "cloud" work in new terminals
cloud start                         # gateway + this device as a node, in the background, restarted if they crash
cloud status                        # live: nodes online, speed, requests served
cloud open                          # the dashboard link (signs you in)
cloud logs | stop
```
`cloud start --lan` lets other devices join; `--models DIR` picks the model folder (default `~/models`); `--rotate` makes new tokens.

**On a phone:** Android freezes background apps, which stops Termux. `cloud start` takes a Termux wake lock (install the
Termux:API app and run `pkg install termux-api`), and once: Android Settings > Apps > Termux > Battery > Unrestricted.

## Free cloud computers

Your phone does not have to be the cloud. Host the gateway and a node on a free **Hugging Face Space** (Docker, CPU basic:
2 CPUs, 16 GB RAM): create a Space with the Docker SDK, add `deploy/huggingface/Dockerfile` and `deploy/huggingface/README.md`,
and set the secret `EXPERTCLOUD_ADMIN_TOKEN`. It builds the engine, downloads Qwen2.5 1.5B Instruct and serves the dashboard and
API at `https://NAME-expertcloud.hf.space`. The dashboard's Connect page walks through it and generates the token.

Add more free machines as nodes with one line (Kaggle and Colab notebooks: put `!` in front):
```
curl -sL https://raw.githubusercontent.com/tharunmaks/stream-/claude/expertstream-moe-android-oegt59/deploy/join.sh | bash -s -- https://NAME-expertcloud.hf.space join-TOKEN
```
Free limits: Spaces sleep after 48 h without visits and forget downloads on restart (the model downloads again); Kaggle sessions
last up to 12 h; Colab stops when its tab closes; Oracle Cloud Always Free (4 ARM cores, 24 GB) stays on. Any server can host
the whole cloud with `deploy/cloud_start.sh`.

## Security

* Prompts and answers pass through the nodes. Join only machines you trust, and put the gateway behind TLS (`wss://`, `https://`)
  when it is on the internet.
* Nodes need the join token; the admin API needs the admin token (both in `~/.expertcloud/`, mode 600). Keys are made by the admin
  unless the gateway runs with `--open-signup`.
* Signing in: open the link the gateway prints (`http://127.0.0.1:8000/dashboard#token=adm-...`); the page removes the token from
  the address bar. Or paste the token from `cat ~/.expertcloud/admin_token`. The `join-` token is for nodes and `sk-expert-` keys
  are for the API; neither signs in, and the gateway says so. Tokens are accepted with or without `Bearer `, and with stray
  quotes or spaces from copying.
* Leaked a token? Restart with `--rotate-tokens`: the old admin and join tokens stop working (nodes must rejoin with the new one;
  API keys stay valid).
* A node only accepts two commands: answer a chat with a model already on it, and download a `.gguf` from Hugging Face. It runs no
  other commands. The engine listens on 127.0.0.1 only.
