# Security

## Reporting a problem
Please open a private security advisory on GitHub (Security tab, "Report a vulnerability") instead of a public issue.
Include steps to reproduce. We aim to answer within a few days.

## What the project does to protect you

**Website (docs/index.html, GitHub Pages)**
- Strict Content Security Policy: only the page's own hashed scripts run, no `eval`, no third-party scripts, fonts or
  trackers (fonts are embedded), network access only to huggingface.co and your own device.
- A small service worker adds extra headers (CSP, `nosniff`, no-referrer, permissions policy, `frame-ancestors 'none'`).
- Models and downloads stay in your browser's private storage; nothing is uploaded anywhere.
- Hugging Face sign-in uses OAuth 2 with PKCE and random `state`; the page never sees your password. Tokens are kept in
  this browser only. "Disconnect" removes them.
- All text from models and from Hugging Face is HTML-escaped before it is shown.

**MCP server (`mcp/expertstream_mcp.py`)**
- HTTP mode always requires a token (auto-generated, stored in `~/.expertstream/token`, mode 600); constant-time compare;
  lock-out after 10 wrong tries per address.
- Loopback servers accept only loopback `Host` and `Origin` headers (blocks other websites and DNS rebinding).
- Request bodies capped at 4 MB; inputs validated; model files must live in the models folder; packs in `~/packs`;
  downloads follow redirects only to Hugging Face hosts; no shell is ever invoked with user text.
- TLS with `--cert` and `--key`. Plain http on a network is flagged with a warning.

**Native app server (`es_serve`)**: binds 127.0.0.1 only and rejects foreign `Host` and `Origin` headers.

- `--public` links carry the secret in the URL path (apps only accept a URL). Treat the whole link as a password; delete `~/.expertstream/token` to rotate. Public mode refuses connections without the secret and locks out guessers.

## Things to know
- Anyone who has your MCP token can use your model. Keep it private; delete the token file to rotate it.
- Model files are data, but a malicious GGUF could try to crash the engine; load files you trust.
- Agent presets are prompts, not guarantees. Review what an AI coder does with the tools you give it.
