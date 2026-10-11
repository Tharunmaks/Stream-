#!/usr/bin/env python3
"""Build the legal and trust page: web/legal/template.html + web/legal/operator.json + the 20-item checklist (mcp/legal.py)
-> docs/legal.html and web/legal.html. The page has no scripts, so its CSP is script-src 'none'."""
import html, json, os, sys, time
root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(root, 'mcp'))
import legal
rd = lambda p: open(os.path.join(root, p), encoding='utf-8').read()
op = json.load(open(os.path.join(root, 'web/legal/operator.json'), encoding='utf-8'))
E = lambda s: html.escape(s or '')
nope = '<span class="tag todo">not published</span>'
UPDATED = '2026-10-11'
SITE = {  # check number -> (what this site does, status ok|na|todo)
 1: ('Privacy policy above; linked from the footer, the notice and the sign-in form.', 'ok'),
 2: ('Terms above; linked from the footer.', 'ok'),
 3: ('Free, nothing to refund; stated above.', 'ok'),
 4: ('No cookies; every storage key is listed in the table above.', 'ok'),
 5: ('No non-essential cookies or trackers exist, so no consent wall is needed. A plain privacy notice with OK is shown on first visit.', 'ok'),
 6: ('The only form is the optional Hugging Face sign-in/token box. It states what access is given, where the token is kept, and links the privacy policy. No marketing boxes.', 'ok'),
 7: ('Nothing is collected. Hugging Face sign-in asks for the smallest scopes that make downloads work.', 'ok'),
 8: ('Audited: zero third-party scripts, SDKs, fonts or trackers. Only huggingface.co is contacted, on your action. The strict Content Security Policy enforces this.', 'ok'),
 9: ('No pre-ticked boxes, no countdowns, no guilt wording. Decline and OK carry equal weight. Disconnect and Delete are one tap.', 'ok'),
 10: ('Free. No fees of any kind.', 'ok'),
 11: ('There are no reviews or testimonials on the site.', 'ok'),
 12: ('Speed claims state the device, model, file type and threads. Superlatives such as "fastest" were removed from the model list.', 'ok'),
 13: ('No photographs. The only <img> (the Hugging Face avatar) is decorative with empty alt; decorative graphics are hidden from screen readers.', 'ok'),
 14: ('Text colours checked at 4.5:1 or more against every panel colour; low-contrast grey and the lighter button colour were fixed.', 'ok'),
 15: ('Skip link, visible focus, real buttons, model cards and the file drop zone respond to Enter and Space, dialogs close with Escape and return focus.', 'ok'),
 16: ('Operator and contact are shown above. Legal name, postal address and e-mail are not published until the owner adds them.', 'todo' if not (op.get('email') and op.get('postal_address')) else 'ok'),
 17: ('Minimum age 13 stated; no data is collected from anyone, so no age gate or parental-consent flow is needed.', 'ok'),
 18: ('No e-mails are sent and there is no mailing list.', 'na'),
 19: ('Fonts are OFL and credited above; no photographs. The code licence is not chosen yet.' if not op.get('code_license') else 'Fonts are OFL and credited above; no photographs; code licence stated in the terms.', 'todo' if not op.get('code_license') else 'ok'),
 20: ('Settings, Privacy, "Delete all my data" removes everything held on the device in one step.', 'ok'),
}
LAB = {'ok': ('Done', 'ok'), 'na': ('Not applicable', 'na'), 'todo': ('Owner to complete', 'todo')}
rows = []
for n, t, g, w, h, v in legal.ITEMS:
    txt, st = SITE[n]
    rows.append('<tr><td>%d</td><td><b>%s</b><br><span style="color:var(--ink-2);font-size:13px">%s</span></td><td>%s</td><td><span class="tag %s">%s</span></td></tr>' % (n, E(t), E(w), E(txt), LAB[st][1], LAB[st][0]))
page = rd('web/legal/template.html').replace('@@FONTS@@', rd('web/src/fonts.css'))
rep = {'@@UPDATED@@': UPDATED, '@@ROWS@@': '\n'.join(rows), '@@DISCLAIMER@@': E(legal.DISCLAIMER), '@@CONTACT@@': E(op.get('contact') or ''),
       '@@OP_NAME@@': E(op.get('name') or ''), '@@OP_EMAIL@@': E(op.get('email')) or nope, '@@OP_ADDRESS@@': E(op.get('postal_address')) or nope, '@@OP_ENTITY@@': E(op.get('legal_entity_or_tax_id')) or nope + ' (none: individual project)',
       '@@CODE_LICENSE@@': E(op.get('code_license')) or 'not chosen yet; until a licence file is added the author keeps all rights except those GitHub\'s terms give you to view and fork the repository',
       '@@GOVERNING_LAW@@': E(op.get('governing_law')) or 'not specified; the mandatory consumer-protection and data-protection law of the place where you live always applies.'}
for k, v in rep.items(): page = page.replace(k, v)
CSP = "; ".join(["default-src 'none'", "script-src 'none'", "style-src 'unsafe-inline'", "font-src data:", "img-src 'self' data:", "base-uri 'none'", "form-action 'none'", "object-src 'none'", "frame-src 'none'"])
page = page.replace('@@CSP@@', CSP)
assert '@@' not in page, [l for l in page.split('\n') if '@@' in l][:2]
for out in ('docs/legal.html', 'web/legal.html'):
    open(os.path.join(root, out), 'w', encoding='utf-8').write(page)
print('wrote docs/legal.html and web/legal.html,', len(page) // 1024, 'KB')
