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

**Expert Craft (`craft/`)**
- It is a command-line tool, not a server: it opens no port and has no token.
- Project folders must live inside `EXPERTCRAFT_HOME` (default `~/expertcraft`); paths outside it are refused, and path-like
  flags passed to generated training scripts are checked the same way.
- Downloads (`--dataset`, `--url`) must be `https://`. No shell is ever invoked with user text.
- Flash Experts write only inside the project folder. The wear meter and `--max-wear-gb` limit how much flash a run may write.

**ExpertStream Instant (`es_serve`)**
- Binds 127.0.0.1 unless started with `--lan`. Local mode refuses foreign `Host` names (DNS rebinding).
- Browsers: requests carrying an `Origin` are answered only for local pages, `https://tharunmaks.github.io` and origins given with
  `--cors`; everything else gets 403. Allowed origins get CORS and Private Network Access headers.
- `--lan`: every request from another device needs `Authorization: Bearer <key>`; the key is created once in `~/.expertstream/key` (mode 600).
- Request bodies are capped at 16 MB; JSON is parsed by a bounded parser (depth 64); no shell is ever invoked with request data.
- Saved prompt caches (`~/.expertstream/kv`, mode 700) hold the model's internal state for prompts you sent: treat them like chat history.
  Delete the folder to remove them, or run with `ES_KV_DISK=0`.

## Things to know
- Model files are data, but a malicious GGUF could try to crash the engine; load files you trust.
- Agent presets are prompts, not guarantees. Review what any AI tool does before you run its commands.
