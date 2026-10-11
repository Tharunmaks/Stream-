#!/usr/bin/env python3
"""Build the Expert Craft page: web/craft/template.html + embedded fonts -> docs/craft.html and web/craft.html,
with a strict CSP that pins the inline script by hash (same approach as build_web.py)."""
import base64, hashlib, json, os, re, sys

root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
rd = lambda p: open(os.path.join(root, p), encoding="utf-8").read()
def results_html():
    r = json.load(open(os.path.join(root, "web/craft/results.json"), encoding="utf-8"))
    rows = "".join("<tr><td>%s</td><td class=\"r mono\">%s</td><td class=\"r mono\">%s</td><td class=\"r mono\">%s</td></tr>" % (x["name"], x["val"], x["delta"], x["ppl"]) for x in r["rows"])
    return ("<div class=\"scroll\"><table class=\"tbl res\"><thead><tr><th>%s</th><th class=\"r\">Validation loss</th><th class=\"r\">vs no experts</th><th class=\"r\">Perplexity</th></tr></thead><tbody>%s</tbody></table></div>"
            "<p class=\"dim\" style=\"font-size:13px\">%s</p>") % (r["title"], rows, r["note"])


# the phone planner on the page must size things exactly like craft/phone does: take the backbone table from the Python code
sys.path.insert(0, os.path.join(root, "craft", "phone"))
import builder as B, layout as F
table = [[c["params"], c["dim"], c["layers"], F.hidden_per_page(c["dim"])] for c in B.ladder(2048, 128)]
page = rd("web/craft/template.html").replace("@@FONTS@@", rd("web/src/fonts.css")).replace("@@PHONEJS@@", rd("web/craft/phone.js").replace("@@BACKBONES@@", json.dumps(table))).replace("@@RESULTS@@", results_html())
hashes = ["'sha256-" + base64.b64encode(hashlib.sha256(m.encode("utf-8")).digest()).decode() + "'"
          for m in re.findall(r"<script>(.*?)</script>", page, re.S)]
CSP = "; ".join(["default-src 'none'", "script-src " + " ".join(hashes), "style-src 'unsafe-inline'", "font-src data:",
                 "img-src 'self' data:", "base-uri 'none'", "form-action 'none'", "object-src 'none'", "frame-src 'none'"])
page = page.replace("@@CSP@@", CSP)
for out in ("docs/craft.html", "web/craft.html"):
    open(os.path.join(root, out), "w", encoding="utf-8").write(page)
print("wrote docs/craft.html and web/craft.html,", len(page) // 1024, "KB")
