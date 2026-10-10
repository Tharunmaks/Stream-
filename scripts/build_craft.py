#!/usr/bin/env python3
"""Build the Expert Craft page: web/craft/template.html + embedded fonts -> docs/craft.html and web/craft.html,
with a strict CSP that pins the inline script by hash (same approach as build_web.py)."""
import base64, hashlib, os, re

root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
rd = lambda p: open(os.path.join(root, p), encoding="utf-8").read()
page = rd("web/craft/template.html").replace("@@FONTS@@", rd("web/src/fonts.css"))
hashes = ["'sha256-" + base64.b64encode(hashlib.sha256(m.encode("utf-8")).digest()).decode() + "'"
          for m in re.findall(r"<script>(.*?)</script>", page, re.S)]
CSP = "; ".join(["default-src 'none'", "script-src " + " ".join(hashes), "style-src 'unsafe-inline'", "font-src data:",
                 "img-src 'self' data:", "base-uri 'none'", "form-action 'none'", "object-src 'none'", "frame-src 'none'"])
page = page.replace("@@CSP@@", CSP)
for out in ("docs/craft.html", "web/craft.html"):
    open(os.path.join(root, out), "w", encoding="utf-8").write(page)
print("wrote docs/craft.html and web/craft.html,", len(page) // 1024, "KB")
