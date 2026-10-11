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

**Native app server (`es_serve`)**: binds 127.0.0.1 only and rejects foreign `Host` and `Origin` headers.

## Things to know
- Model files are data, but a malicious GGUF could try to crash the engine; load files you trust.
- Agent presets are prompts, not guarantees. Review what any AI tool does before you run its commands.
