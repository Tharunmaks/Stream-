/* ===== Hugging Face account connector + import by link ===== */
/* ----- one-click "Sign in with Hugging Face" (OAuth 2 + PKCE). The page registers itself with
   Hugging Face (dynamic client registration), so no setup or secrets are needed. ----- */
const APP_URL = 'https://tharunmaks.github.io/Stream-/';
const b64url = a => btoa(String.fromCharCode(...a)).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
const OAuth = {
  scope: 'openid profile read-repos',
  redirect() { return location.origin + location.pathname; },
  usable() { return /^https?:$/.test(location.protocol) && window.isSecureContext && !!(window.crypto && crypto.subtle); },
  async clientId() {
    const key = 'oauth:' + this.redirect(); let c = store.get(key); if (c) return c;
    let r; try { r = await fetch(HF.base + '/oauth/register', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ client_name: 'ExpertStream', client_uri: this.redirect(), redirect_uris: [this.redirect()], grant_types: ['authorization_code', 'refresh_token'], response_types: ['code'], token_endpoint_auth_method: 'none', scope: this.scope }) }); } catch (e) { throw new Error(NET_MSG); }
    if (!r.ok) throw new Error('Hugging Face could not set up sign-in for this address (HTTP ' + r.status + ').');
    c = (await r.json()).client_id; store.set(key, c); return c;
  },
  async start() {
    if (!this.usable()) throw new Error('Sign-in needs the site opened over https in a normal browser tab.');
    const cid = await this.clientId(), verifier = b64url(crypto.getRandomValues(new Uint8Array(32))), state = b64url(crypto.getRandomValues(new Uint8Array(16)));
    const challenge = b64url(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(verifier))));
    store.set('oauthPending', { verifier, state, cid, redirect: this.redirect(), hash: location.hash, t: Date.now() });
    const u = HF.base + '/oauth/authorize?' + new URLSearchParams({ client_id: cid, redirect_uri: this.redirect(), response_type: 'code', scope: this.scope, state, code_challenge: challenge, code_challenge_method: 'S256' });
    if (window.top !== window) window.open(u, '_blank'); else location.assign(u);
  },
  async token(body) {
    let r; try { r = await fetch(HF.base + '/oauth/token', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams(body) }); } catch (e) { throw new Error(NET_MSG); }
    const j = await r.json().catch(() => ({}));
    if (!r.ok || !j.access_token) throw new Error('Hugging Face did not accept the sign-in (' + (j.error_description || j.error || 'HTTP ' + r.status) + ').');
    store.set('hfrefresh', j.refresh_token || store.get('hfrefresh', '')); store.set('hfexp', Date.now() + (j.expires_in || 28800) * 1000);
    return j.access_token;
  },
  /* called on every page load: finishes a sign-in that came back with ?code=… */
  async finish() {
    const q = new URLSearchParams(location.search), code = q.get('code'), state = q.get('state'), err = q.get('error');
    if (!code && !err) return this.refreshIfNeeded();
    const p = store.get('oauthPending', null); store.set('oauthPending', null);
    try { history.replaceState(null, '', location.pathname + ((p && p.hash) || '#settings')); } catch (e) { }
    if (err) return toast('Hugging Face sign-in was cancelled.', 'bad', 5000);
    if (!p || p.state !== state) return toast('That sign-in link expired. Press "Sign in with Hugging Face" again.', 'bad', 6000);
    try {
      const tok = await this.token({ grant_type: 'authorization_code', code, redirect_uri: p.redirect, client_id: p.cid, code_verifier: p.verifier });
      store.set('hfclient', p.cid);
      const u = await HFAcct.connect(tok, true); toast('Signed in to Hugging Face as @' + u.name, 'good', 5000);
    } catch (e) { toast(e.message, 'bad', 8000); }
  },
  async refreshIfNeeded() {
    const rt = store.get('hfrefresh', ''), exp = store.get('hfexp', 0), cid = store.get('hfclient', '');
    if (!rt || !cid || !HFAcct.token || Date.now() < exp - 3600e3) return;
    try { const tok = await this.token({ grant_type: 'refresh_token', refresh_token: rt, client_id: cid }); store.set('hftoken', tok); } catch (e) { }
  },
};
const HFAcct = {
  user: store.get('hfuser', null), subs: new Set(),
  on(f) { this.subs.add(f); }, emit() { this.subs.forEach(f => { try { f(); } catch (e) { } }); },
  get token() { return store.get('hftoken', ''); },
  async connect(token, oauth) {
    let r; try { r = await fetch(HF.base + '/api/whoami-v2', { headers: { Authorization: 'Bearer ' + token } }); } catch (e) { throw new Error(NET_MSG); }
    if (!r.ok && oauth) { try { const i = await (await fetch(HF.base + '/oauth/userinfo', { headers: { Authorization: 'Bearer ' + token } })).json(); store.set('hftoken', token); this.user = { name: i.preferred_username || i.name, fullname: i.name || i.preferred_username, avatar: i.picture, orgs: (i.orgs || []).map(o => o.preferred_username || o.name), role: 'signed in' }; store.set('hfuser', this.user); this.emit(); return this.user; } catch (e) { } }
    if (r.status === 401) throw new Error('Hugging Face rejected that token. Make a new one at huggingface.co/settings/tokens (type "Read") and paste it again.');
    if (!r.ok) throw new Error('Hugging Face answered HTTP ' + r.status);
    const j = await r.json();
    store.set('hftoken', token);
    this.user = { name: j.name, fullname: j.fullname || j.name, avatar: j.avatarUrl, orgs: (j.orgs || []).map(o => o.name), role: oauth ? 'signed in with Hugging Face' : (j.auth && j.auth.accessToken && j.auth.accessToken.role) };
    store.set('hfuser', this.user); this.emit(); return this.user;
  },
  disconnect() { store.set('hftoken', ''); store.set('hfuser', null); store.set('hfrefresh', ''); store.set('hfexp', 0); this.user = null; this.emit(); },
  async myRepos() {
    if (!this.user) return [];
    const seen = new Set(), out = [];
    for (const a of [this.user.name, ...this.user.orgs]) {
      try { const l = await HF.json(`${HF.base}/api/models?author=${encodeURIComponent(a)}&filter=gguf&limit=40&expand%5B%5D=gguf&expand%5B%5D=downloads&expand%5B%5D=private`); for (const m of l) if (!seen.has(m.id)) { seen.add(m.id); out.push({ id: m.id, arch: m.gguf && m.gguf.architecture, total: m.gguf && m.gguf.total, downloads: m.downloads || 0, private: !!m.private }); } } catch (e) { }
    }
    return out;
  },
};
function accountCard(onChange) {
  const box = h('div', { class: 'card' });
  const draw = () => {
    const u = HFAcct.user;
    if (u) {
      box.replaceChildren(h('div', { class: 'row', style: 'flex-wrap:nowrap' }, u.avatar ? h('img', { class: 'avatar', src: u.avatar, alt: '', referrerpolicy: 'no-referrer' }) : h('div', { class: 'avatar' }, glyphOf(u.name)), h('div', { class: 'grow' }, h('b', {}, u.fullname), h('div', { class: 'dim', style: 'font-size:13px' }, '@' + u.name + (u.orgs.length ? ' · ' + u.orgs.length + ' orgs' : '') + (u.role ? ' · token: ' + u.role : '')), ), chip('good', 'connected')),
        h('p', { class: 'muted', style: 'font-size:14px' }, 'Private and gated models in your account can now be searched and downloaded here. For gated models, accept the license once on huggingface.co.'),
        h('div', { class: 'row' }, h('button', { class: 'btn sm', onclick: () => { go('models'); showModelsTab('hf'); } }, 'Browse my models'), h('button', { class: 'btn sm danger', onclick: () => { HFAcct.disconnect(); toast('Disconnected. The token was removed from this browser.'); } }, 'Disconnect')));
    } else {
      const inp = h('input', { type: 'password', 'aria-label': 'Hugging Face access token', placeholder: 'hf_xxxxxxxxxxxxxxxx', autocomplete: 'off', id: 'hf-token-' + Math.random().toString(36).slice(2, 6) }), msg = h('div', { class: 'note bad', hidden: true });
      const go1 = async () => { const t = inp.value.trim(); if (!t) return; btn.disabled = true; btn.textContent = 'Checking…'; msg.hidden = true; try { const u = await HFAcct.connect(t); toast('Connected as @' + u.name, 'good'); } catch (e) { msg.hidden = false; msg.textContent = e.message; } btn.disabled = false; btn.textContent = 'Connect'; };
      const btn = h('button', { class: 'btn', onclick: go1 }, 'Connect');
      const sign = h('button', { class: 'btn hf-btn', onclick: async () => { sign.disabled = true; sign.lastChild.textContent = 'Opening Hugging Face…'; msg.hidden = true; try { await OAuth.start(); } catch (e) { msg.hidden = false; msg.replaceChildren(e.message, ' ', h('a', { href: APP_URL, target: '_blank', rel: 'noopener' }, 'Open the full app')); } sign.disabled = false; sign.lastChild.textContent = 'Sign in with Hugging Face'; } }, h('span', { class: 'hf-logo', 'aria-hidden': 'true' }), h('span', {}, 'Sign in with Hugging Face'));
      box.replaceChildren(h('div', { class: 'row', style: 'justify-content:space-between' }, h('h3', {}, 'Link your Hugging Face account'), chip('', 'not linked')),
        h('p', { class: 'muted', style: 'font-size:14px' }, 'One click: Hugging Face asks you to allow ExpertStream to read your models, then brings you back here. Your private and gated models become available.'),
        sign, msg,
        h('p', { class: 'dim', style: 'font-size:12.5px' }, 'By continuing you let this page read your Hugging Face profile name and your repositories, and download models for you. Your token is stored only in this browser (never sent to us; we run no server) and "Disconnect" deletes it. ', h('a', { href: 'legal.html#privacy' }, 'Privacy policy')),
        h('details', { class: 'think' }, h('summary', {}, 'Other way: paste an access token'), h('div', { class: 'stack', style: 'margin-top:8px' }, h('p', { class: 'dim', style: 'font-size:13px' }, 'Create a "Read" token at huggingface.co/settings/tokens and paste it. It stays in this browser.'), h('div', { class: 'row', style: 'flex-wrap:nowrap' }, inp, btn))));
      inp.addEventListener('keydown', e => { if (e.key === 'Enter') go1(); });
    }
  };
  draw(); HFAcct.on(() => { draw(); onChange && onChange(); });
  return box;
}
/* "paste a link or owner/name" */
function parseHF(text) {
  text = (text || '').trim(); if (!text) return null;
  let m = text.match(/huggingface\.co\/([^\/\s]+\/[^\/\s?#]+)(?:\/(?:blob|resolve)\/[^\/]+\/([^?#\s]+))?/i);
  if (m) return { repo: m[1], file: m[2] ? decodeURIComponent(m[2]) : null };
  m = text.match(/^([\w.\-]+\/[\w.\-]+?)(?:\s+(\S+\.gguf))?$/); if (m) return { repo: m[1], file: m[2] || null };
  return null;
}
async function importFromText(text) {
  const t = parseHF(text); if (!t) return toast('Paste a Hugging Face link or owner/name, like Qwen/Qwen3-0.6B-GGUF', 'bad', 5000);
  if (t.file && /\.gguf$/i.test(t.file)) { showModelsTab('hf'); return checkRemote(t.repo, t.file, 0, null); }
  openRepo(t.repo);
}
