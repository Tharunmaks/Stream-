"""The 20 legal and trust must-haves for an app or website, as data plus a static auditor.

Used by the MCP tools legal_checklist and legal_audit, and by scripts/build_legal.py (the legal page of the website).
This is a practical engineering checklist, NOT legal advice: laws differ by country and by what your app does.
Python standard library only."""
import os, re

DISCLAIMER = ('Engineering checklist, not legal advice. Rules differ by country (GDPR/UK GDPR, CCPA/CPRA, India DPDP Act, '
              'COPPA, CAN-SPAM, ePrivacy, WCAG/ADA ...) and by what the app does. Ask a qualified lawyer before launch if you take '
              'payments, handle health or finance data, serve children, or operate in several countries.')

# id, title, group, why, how, verify
ITEMS = [
 (1, 'Privacy policy', 'Policies', 'Most laws require you to say what personal data you collect, why, who gets it, how long you keep it and how users exercise their rights. App stores and payment providers require it too.',
  'Publish /privacy (linked from the footer and from every sign-up form). Name the operator and a contact, list data types, purposes, legal basis, processors/SDKs, retention, user rights, transfers, children, and the date of the last change.',
  'A reachable page that matches what the code really collects.'),
 (2, 'Terms of service', 'Policies', 'Sets the rules of use, limits liability, and says what happens to accounts and content.',
  'Publish /terms: who may use it, acceptable use, accounts, content ownership and licence, disclaimers, liability limit, termination, governing law, how changes are announced.',
  'Linked next to the sign-up button; users can read it before agreeing.'),
 (3, 'Refund policy', 'Policies', 'Consumer law and payment providers expect clear refund and cancellation terms whenever money changes hands.',
  'Publish /refunds: what can be refunded, time window, how to ask, how long it takes, how to cancel a subscription. Free apps say so explicitly.',
  'Present if there is any payment, price, checkout or subscription; a one-line "free, nothing to refund" otherwise.'),
 (4, 'Cookie policy', 'Policies', 'If you use cookies or similar storage (localStorage, pixels, fingerprinting) you must list them with purpose and lifetime.',
  'Publish /cookies: a table of every cookie/storage key (name, purpose, first or third party, lifetime) and how to change choices. Say plainly when there are none.',
  'The table matches document.cookie, localStorage keys and every third-party request.'),
 (5, 'Cookie consent banner', 'Consent', 'Non-essential cookies and trackers need prior opt-in in the EU/UK and similar regimes.',
  'Block analytics/ads/embeds until the user opts in; give "Accept" and "Reject" equal weight; remember the choice; let users change it later. If you set no non-essential cookies, show a plain notice instead.',
  'No tracking request fires before consent; Reject is one click.'),
 (6, 'Form consents', 'Consent', 'Consent must be informed, specific and freely given, and you must be able to show it.',
  'Next to each form say what is collected and why, link the privacy policy, use an UNCHECKED checkbox for marketing, keep the proof (time, text version).',
  'Marketing boxes are not pre-ticked; each purpose has its own text.'),
 (7, 'Data minimization', 'Data', 'Collecting less means less to protect, less to disclose and less liability after a breach.',
  'Ask only for what the feature needs; make other fields optional; set a retention period and delete on schedule; do not log secrets or full IPs unless needed.',
  'Every field in every form can be justified in one sentence.'),
 (8, 'Audit third-party SDKs', 'Data', 'Each SDK, script, font or embed can collect data about your users, and you are responsible for it.',
  'List every third-party script, SDK, font host, map, video and analytics tool; remove what you do not need; load the rest after consent; name them in the privacy policy.',
  'Network tab shows only hosts you listed.'),
 (9, 'Remove dark patterns', 'Fair design', 'Tricks that push people into choices (hidden opt-outs, confirm-shaming, fake urgency) are banned or fined by many regulators.',
  'No pre-ticked boxes, no countdown or "only 2 left" unless true, no guilt-trip labels, cancelling is as easy as signing up, equal-weight Accept/Reject.',
  'A new user can decline and cancel without searching.'),
 (10, 'No hidden fees', 'Fair design', 'The price shown first must be the price paid; drip pricing is unlawful in many places.',
  'Show the total with taxes and fees before the final click; no surprise add-ons; trials say when they convert and for how much.',
  'Checkout total equals the advertised price plus visible tax.'),
 (11, 'Remove fake reviews', 'Truthful claims', 'Fake or incentivised reviews and invented testimonials are illegal in the EU, UK, US and India.',
  'Show only real reviews from real customers, disclose incentives, never write testimonials yourself or buy reviews, do not hide the negative ones.',
  'Every quote and star rating can be traced to a real source.'),
 (12, 'Remove unsupported claims', 'Truthful claims', 'Advertising claims need evidence ("fastest", "100% secure", "guaranteed", "#1", "AI-certified").',
  'Back each claim with a measurement, a source or the conditions it was measured under, or delete it. Keep the evidence.',
  'Each superlative has a number, a date and a method next to it.'),
 (13, 'Accessibility: alt text', 'Accessibility', 'Screen-reader users need text for every meaningful image; accessibility law (ADA, EAA, EN 301 549, India RPwD) applies to websites and apps.',
  'Every <img> has alt (empty alt="" for pure decoration), icons that act as buttons have aria-label, charts have a text summary.',
  'No <img> without alt attribute; run a screen reader once.'),
 (14, 'Accessibility: colour contrast', 'Accessibility', 'Low-contrast text is unreadable for many people and is the most common WCAG failure.',
  'Text contrast at least 4.5:1 (3:1 for large text and UI borders); never rely on colour alone to convey meaning; support zoom to 200 percent.',
  'Check every text/background pair with a contrast checker.'),
 (15, 'Accessibility: keyboard navigation', 'Accessibility', 'People who cannot use a mouse must reach and operate everything.',
  'Everything clickable is a real <button> or <a> (or has role, tabindex and Enter/Space handling); visible focus ring; logical tab order; a "skip to content" link; dialogs close with Escape and return focus.',
  'Complete the main task using only Tab, Shift+Tab, Enter, Space and Escape.'),
 (16, 'Business details', 'Transparency', 'Many countries require visible operator identity and contact details (EU e-commerce directive, India Consumer Protection Rules, GDPR controller).',
  'Show legal name or the individual operating the service, a postal address or registered office where required, an e-mail or contact form, and company or tax numbers if you have them. Link it from the footer.',
  'A visitor can tell who runs the service and how to reach them.'),
 (17, 'Age consent for kids\' data', 'Data', 'Children\'s data has strict rules (COPPA under 13 in the US, GDPR-K 13-16 in the EU, India DPDP under 18 needs a parent).',
  'Say the minimum age in the terms; add an age gate before collecting data if kids may use it; get verifiable parental consent where required; no targeted ads to children; do not collect more than needed.',
  'There is a stated minimum age and a plan for under-age users.'),
 (18, 'Unsubscribe link in e-mails', 'Consent', 'Marketing e-mail needs a working one-click unsubscribe and your postal address (CAN-SPAM, GDPR/ePrivacy, CASL).',
  'Every marketing mail has an unsubscribe link and the List-Unsubscribe header; honour the request within 10 days (sooner is better); keep a suppression list; separate transactional from marketing mail.',
  'Unsubscribe works without logging in.'),
 (19, 'Licensed fonts and images', 'Rights', 'Using fonts, photos, icons or code without a licence is copyright infringement and attracts claims.',
  'Use open-licence (OFL, MIT, CC0/CC-BY) or paid assets; keep a credits/licence list; honour attribution; check that model weights and datasets allow your use.',
  'Every asset has a licence recorded in the repo.'),
 (20, 'Data deletion requests', 'Data', 'Users have the right to erase their data (GDPR art. 17, CCPA, DPDP s.12).',
  'Give an in-app "Delete my account and data" action or a documented e-mail route; delete or anonymise within 30 days; also remove it from backups on their normal cycle and from processors; confirm to the user.',
  'You can demonstrate a full deletion end to end.'),
]

def checklist(kind=''):
    kind = (kind or '').lower()
    rows = [dict(n=n, title=t, group=g, why=w, how=h, verify=v) for n, t, g, w, h, v in ITEMS]
    prompt = ('Audit this project against the 20 items. For each item answer PASS, FAIL, NEEDS-REVIEW or N/A with the file and line as evidence, then fix the FAIL items '
              'and write the missing pages. Do not invent business details: leave a clearly marked TODO for anything only the owner knows.')
    return dict(items=rows, count=len(rows), prompt=prompt, disclaimer=DISCLAIMER,
                note=('Pick the items that apply to %s.' % kind) if kind else 'Which items apply depends on the app: payments (3, 10), e-mail (18), accounts and forms (6, 7, 17, 20), analytics (4, 5, 8).')

# ------------------------------------------------------------------ static auditor
MAX_FILES, MAX_BYTES = 400, 1_500_000
TEXT_EXT = ('.html', '.htm', '.css', '.js', '.jsx', '.ts', '.tsx', '.vue', '.svelte', '.md', '.txt', '.php', '.erb', '.ejs', '.mjml', '.json', '.xml')
SKIP_DIR = {'.git', 'node_modules', '.venv', 'venv', '__pycache__', 'dist', 'build', '.next', 'vendor'}
TRACKERS = r'google-analytics|googletagmanager|gtag\(|fbq\(|facebook\.net|hotjar|segment\.(io|com)|mixpanel|amplitude|clarity\.ms|doubleclick|adsbygoogle|sentry\.io|fullstory|intercom|hubspot|tiktok|snap(chat)?\.com/|twitter\.com/i/|connect\.facebook|matomo|plausible'

def _read(path):
    try:
        if os.path.getsize(path) > MAX_BYTES: return ''
        return open(path, encoding='utf-8', errors='replace').read()
    except Exception: return ''

def _walk(root):
    n = 0
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIR and not x.startswith('.')]
        for f in sorted(files):
            n += 1
            if n > MAX_FILES: return
            yield os.path.join(d, f)

def _lum(h):
    h = h.lstrip('#')
    if len(h) == 3: h = ''.join(c * 2 for c in h)
    r, g, b = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return .2126 * f(r) + .7152 * f(g) + .0722 * f(b)

def contrast(a, b):
    la, lb = sorted([_lum(a), _lum(b)], reverse=True)
    return (la + .05) / (lb + .05)

def audit(root, kind=''):
    root = os.path.realpath(root)
    if not os.path.isdir(root): raise ValueError('not a folder: %s' % root)
    files = [p for p in _walk(root) if p.lower().endswith(TEXT_EXT) or os.path.basename(p).upper().startswith(('LICENSE', 'COPYING', 'NOTICE'))]
    texts = {p: _read(p) for p in files}
    rel = lambda p: os.path.relpath(p, root)
    html = {p: t for p, t in texts.items() if p.lower().endswith(('.html', '.htm', '.php', '.erb', '.ejs', '.vue', '.svelte', '.jsx', '.tsx'))}
    allt = '\n'.join(texts.values()); low = allt.lower()
    names = ' '.join(rel(p).lower() for p in files)
    res = {}
    def put(n, status, evidence, fix=''): res[n] = dict(n=n, title=ITEMS[n - 1][1], status=status, evidence=evidence, fix=fix)
    def has_page(word): return bool(re.search(word, names)) or bool(re.search(r'href=["\'][^"\']*(%s)' % word, low))

    pays = bool(re.search(r'stripe|paypal|razorpay|checkout|add to cart|subscription|price|pricing|\$\s?\d|₹\s?\d|€\s?\d', low))
    cookies = bool(re.search(r'document\.cookie|set-cookie|cookie\(|' + TRACKERS, low))
    storage = sorted(set(re.findall(r'(?:localStorage|sessionStorage)\.(?:set|get)Item\(\s*[\'"]([\w.\-:]+)', allt)))
    forms = [(p, m) for p, t in html.items() for m in re.findall(r'<form\b.*?</form>', t, re.S | re.I)]
    sdk_hosts = sorted(set(re.findall(r'(?:src|href)=["\']https?://([^/"\']+)', '\n'.join(t for p, t in html.items()))))
    tracker_hits = sorted(set(m if isinstance(m, str) else m[0] for m in re.findall('(' + TRACKERS + ')', low)))

    put(1, 'PASS' if has_page('privacy') else 'FAIL', 'privacy page/link found' if has_page('privacy') else 'no privacy policy page or link', 'Write /privacy and link it from the footer and every form.')
    put(2, 'PASS' if has_page('terms|tos\\b') else 'FAIL', 'terms page/link found' if has_page('terms|tos\\b') else 'no terms page or link', 'Write /terms and link it near sign-up.')
    if pays: put(3, 'PASS' if has_page('refund|cancel') else 'FAIL', 'payment-related words found; ' + ('refund page/link found' if has_page('refund|cancel') else 'no refund/cancellation page'), 'Publish /refunds with window, method and cancellation steps.')
    else: put(3, 'N/A', 'no payment or price terms found', 'If the app is free say so in the terms.')
    if cookies or storage: put(4, 'PASS' if has_page('cookie') else 'FAIL', ('cookies/trackers: ' + ', '.join(tracker_hits) if cookies else '') + (' storage keys: ' + ', '.join(storage[:8]) if storage else ''), 'List every cookie/storage key with purpose and lifetime on /cookies.')
    else: put(4, 'N/A', 'no cookie or storage use found', 'Still say "no cookies" in the privacy policy.')
    if tracker_hits or re.search(r'document\.cookie', low): put(5, 'PASS' if re.search(r'consent|cookie[- ]?banner|accept all|reject all|cookiebot|onetrust|osano', low) else 'FAIL', 'trackers/cookies present: ' + ', '.join(tracker_hits[:6]), 'Block them until opt-in; Accept and Reject must look equally important.')
    else: put(5, 'N/A' if not storage else 'NEEDS-REVIEW', 'no cookies or trackers detected' + ('; localStorage is used (strictly necessary storage needs a notice, not a consent)' if storage else ''), 'A short privacy notice is enough when nothing non-essential is stored.')
    if forms:
        bad = [rel(p) for p, f in forms if re.search(r'type=["\']?(email|password|tel)|name=["\']?(email|phone|name)', f, re.I) and not re.search(r'consent|privacy|agree|checkbox|terms', f, re.I)]
        pre = [rel(p) for p, f in forms if re.search(r'type=["\']?checkbox["\']?[^>]*\bchecked\b', f, re.I)]
        put(6, 'FAIL' if bad else 'PASS', ('forms collecting personal data without consent text: ' + ', '.join(sorted(set(bad))) if bad else '%d form(s), each mentions consent/privacy/terms' % len(forms)) + ('; PRE-TICKED checkbox in ' + ', '.join(sorted(set(pre))) if pre else ''), 'Add purpose text + privacy link; unchecked boxes only.')
        sens = [m for m in re.findall(r'name=["\']?(dob|birth\w*|ssn|aadhaar|aadhar|pan|passport|address|salary|religion|caste|health\w*)', '\n'.join(f for p, f in forms), re.I)]
        put(7, 'NEEDS-REVIEW' if sens else 'PASS', ('sensitive-looking fields: ' + ', '.join(sorted(set(s.lower() for s in sens)))) if sens else 'only ordinary fields found', 'Remove any field you cannot justify; make the rest optional.')
    else:
        put(6, 'N/A', 'no <form> found'); put(7, 'N/A', 'no form fields found')
    ext = [h for h in sdk_hosts if not re.search(r'^(localhost|127\.)', h)]
    put(8, 'NEEDS-REVIEW' if ext or tracker_hits else 'PASS', ('external hosts: ' + ', '.join(ext[:12]) if ext else 'no external scripts, styles or links to other hosts') + (' ; tracker patterns: ' + ', '.join(tracker_hits[:6]) if tracker_hits else ''), 'Name each one in the privacy policy; remove what you do not need.')
    dark = re.findall(r'(only \d+ (left|remaining)|hurry|limited time|offer ends|countdown|no thanks,? i (don\'t|do not)|i don\'t want to (save|win)|expires in|\d+ people are (viewing|watching))', low)
    pre_all = re.findall(r'type=["\']?checkbox["\']?[^>]*\bchecked\b', allt, re.I)
    put(9, 'NEEDS-REVIEW' if dark or pre_all else 'PASS', ('urgency/confirm-shaming phrases: ' + ', '.join(sorted(set(d[0] for d in dark))[:6]) if dark else 'no known dark-pattern phrases') + ('; pre-ticked checkbox x%d' % len(pre_all) if pre_all else ''), 'Make urgency real or remove it; make Decline neutral; untick boxes.')
    fees = re.findall(r'(service fee|processing fee|convenience fee|handling fee|booking fee|platform fee|\+ taxes|plus fees)', low)
    put(10, 'NEEDS-REVIEW' if fees else ('N/A' if not pays else 'PASS'), ('fee words: ' + ', '.join(sorted(set(fees))) if fees else 'no extra-fee wording found'), 'Show the all-in price before the last click.')
    rev = re.findall(r'(testimonial|★{3,}|⭐{3,}|5/5|4\.\d/5|rated \d|reviews?\b|trusted by|\d[\d,]*\+? (happy )?(customers|users|clients))', low)
    put(11, 'NEEDS-REVIEW' if rev else 'PASS', ('review/testimonial/social-proof text: ' + ', '.join(sorted(set(r[0] for r in rev))[:6]) if rev else 'no reviews or testimonials found'), 'Keep only real, traceable ones; disclose incentives.')
    claims = re.findall(r'(world\'?s (best|fastest|first)|#1\b|number one|100% (secure|safe|accurate|guaranteed|free|private)|guaranteed|risk-free|clinically proven|best in class|fastest|unbeatable|lightning[- ]fast|military[- ]grade|unhackable|\d+x faster)', low)
    put(12, 'NEEDS-REVIEW' if claims else 'PASS', ('claims needing evidence: ' + ', '.join(sorted(set(c[0] for c in claims))[:8]) if claims else 'no superlatives found'), 'State the measurement, date and conditions beside each claim, or remove it.')
    noalt = [(rel(p), len(re.findall(r'<img\b(?![^>]*\balt\s*=)[^>]*>', t, re.I))) for p, t in html.items()]
    noalt = [(f, n) for f, n in noalt if n]
    imgs = sum(len(re.findall(r'<img\b', t, re.I)) for t in html.values())
    put(13, 'FAIL' if noalt else 'PASS', ('<img> without alt: ' + ', '.join('%s (%d)' % x for x in noalt[:8]) if noalt else '%d <img> tag(s), all with alt' % imgs), 'Add alt text, or alt="" for decoration.')
    pairs = []; low_c = []
    for p, t in texts.items():
        if not p.lower().endswith(('.css', '.html', '.htm')): continue
        for body in re.findall(r'\{([^{}]*)\}', t):
            c = re.search(r'(?<![-\w])color\s*:\s*(#[0-9a-fA-F]{3,6})\b', body); b = re.search(r'background(?:-color)?\s*:\s*(#[0-9a-fA-F]{3,6})\b', body)
            if c and b and len(c.group(1)) in (4, 7) and len(b.group(1)) in (4, 7):
                r = contrast(c.group(1), b.group(1)); pairs.append(r)
                if r < 4.5: low_c.append('%s on %s = %.2f (%s)' % (c.group(1), b.group(1), r, rel(p)))
    put(14, 'FAIL' if low_c else ('PASS' if pairs else 'NEEDS-REVIEW'), ('below 4.5:1: ' + '; '.join(low_c[:6]) if low_c else ('%d same-rule colour pair(s) all >= 4.5:1' % len(pairs) if pairs else 'no same-rule colour pairs found; test the rendered page (CSS variables are not resolved here)')), 'Darken text or lighten the background until it reaches 4.5:1.')
    clicky = sum(len(re.findall(r'<(?:div|span|li|td|img)\b[^>]*\bonclick\s*=(?![^>]*(?:role=|tabindex))', t, re.I)) for t in html.values())
    clicky += sum(len(re.findall(r'\bh\(\s*[\'"](?:div|span)[\'"]\s*,\s*\{(?![^}]*(?:tabindex|role))[^}]*onclick', t)) for t in texts.values())
    nofocus = len(re.findall(r'outline\s*:\s*(?:none|0)\b', allt)); fv = len(re.findall(r':focus-visible|:focus\b', allt)); pos = len(re.findall(r'tabindex\s*=\s*["\']?[1-9]', allt))
    skip = bool(re.search(r'skip[- ]?(to|link|nav)|class=["\']skip', low))
    bad15 = []
    if clicky: bad15.append('%d clickable non-button element(s) without role/tabindex' % clicky)
    if nofocus and not fv: bad15.append('outline removed and no :focus-visible style')
    if pos: bad15.append('%d positive tabindex' % pos)
    put(15, 'FAIL' if bad15 else ('PASS' if skip else 'NEEDS-REVIEW'), '; '.join(bad15) if bad15 else ('no obvious keyboard traps; skip link present' if skip else 'no obvious keyboard traps but no "skip to content" link; test with Tab'), 'Use <button>/<a>, add focus ring, skip link, Escape to close dialogs.')
    biz = re.search(r'(contact|about)\b', names) or re.search(r'mailto:|contact us|registered (office|address)|©|copyright|all rights reserved|gstin|cin:|vat', low)
    put(16, 'PASS' if biz else 'FAIL', 'contact/operator wording found' if biz else 'no contact, operator or address found', 'Add the operator name, contact e-mail and (where required) postal address to the footer.')
    kids = re.search(r'\b(kids?|children|child|under 1[38]|minor|school)\b', low); agefield = re.search(r'name=["\']?(dob|age|birth)', allt, re.I); agetext = re.search(r'(at least|over|minimum age|aged?) (13|16|18)|13\+|16\+|18\+|parental consent|age gate', low)
    put(17, 'PASS' if agetext else ('NEEDS-REVIEW' if (kids or agefield or forms) else 'N/A'), 'age rule found' if agetext else ('children or age-related wording / data collection without an age rule' if (kids or agefield or forms) else 'no kids or data collection signals'), 'State a minimum age in the terms and gate under-age sign-ups.')
    mail = [rel(p) for p, t in texts.items() if re.search(r'newsletter|mjml|\.eml|email[-_ ]?template|sendgrid|mailchimp|nodemailer|smtp|ses\.send', rel(p).lower() + ' ' + t.lower()[:6000])]
    if mail: put(18, 'PASS' if re.search(r'unsubscribe|list-unsubscribe', low) else 'FAIL', 'e-mail sending/templates found in ' + ', '.join(mail[:4]), 'Add a one-click unsubscribe link and the List-Unsubscribe header to marketing mail.')
    else: put(18, 'N/A', 'no e-mail sending or templates found', 'If you add a newsletter later, add unsubscribe first.')
    assets = [rel(p) for p in _walk(root) if p.lower().endswith(('.woff', '.woff2', '.ttf', '.otf', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.mp3', '.mp4'))]
    fontcss = re.findall(r'@font-face|fonts\.googleapis|fonts\.gstatic|use\.typekit|font-family\s*:\s*["\']([^"\',;]+)', allt)
    lic = re.search(r'(licen[cs]e|credits?|attribution|ofl\b|open font|creative commons|cc-by|cc0|unsplash|pexels|pixabay|mit license|apache|third[- ]party)', low) or any(os.path.basename(p).upper().startswith(('LICENSE', 'NOTICE', 'CREDITS', 'THIRD')) for p in files)
    put(19, ('PASS' if lic else 'FAIL') if (assets or fontcss) else 'N/A', ('%d asset file(s), %d font reference(s); ' % (len(assets), len(fontcss))) + ('licence/credits wording found' if lic else 'no licence or credits found') if (assets or fontcss) else 'no fonts or media found', 'Add a credits/licence list with the licence and source of every font, image, icon and dataset.')
    dele = re.search(r'delete (my|your) (account|data)|erase (my|your)|data deletion|right to (be forgotten|erasure)|remove (my|your) data|clear all data|delete everything|delete all (my|your)', low)
    put(20, 'PASS' if dele else ('NEEDS-REVIEW' if (forms or storage) else 'N/A'), 'deletion route found' if dele else ('data is collected or stored but no deletion route is documented' if (forms or storage) else 'no personal data collection found'), 'Add an in-app "Delete my data" action or a documented e-mail route (30 days).')
    out = [res[i] for i in range(1, 21)]
    tally = {}
    for r in out: tally[r['status']] = tally.get(r['status'], 0) + 1
    return dict(root=root, files_scanned=len(files), summary=tally, results=out, disclaimer=DISCLAIMER,
                note='Static heuristics only: they read source files, not the running site, and cannot judge legal sufficiency. NEEDS-REVIEW means a human must look.')
