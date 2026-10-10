/* ===== Hugging Face account connector + import by link ===== */
const HFAcct = {
  user: store.get('hfuser', null), subs: new Set(),
  on(f) { this.subs.add(f); }, emit() { this.subs.forEach(f => { try { f(); } catch (e) { } }); },
  get token() { return store.get('hftoken', ''); },
  async connect(token) {
    let r; try { r = await fetch(HF.base + '/api/whoami-v2', { headers: { Authorization: 'Bearer ' + token } }); } catch (e) { throw new Error(NET_MSG); }
    if (r.status === 401) throw new Error('Hugging Face rejected that token. Make a new one at huggingface.co/settings/tokens (type "Read") and paste it again.');
    if (!r.ok) throw new Error('Hugging Face answered HTTP ' + r.status);
    const j = await r.json();
    store.set('hftoken', token);
    this.user = { name: j.name, fullname: j.fullname || j.name, avatar: j.avatarUrl, orgs: (j.orgs || []).map(o => o.name), role: j.auth && j.auth.accessToken && j.auth.accessToken.role };
    store.set('hfuser', this.user); this.emit(); return this.user;
  },
  disconnect() { store.set('hftoken', ''); store.set('hfuser', null); this.user = null; this.emit(); },
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
      const inp = h('input', { type: 'password', placeholder: 'hf_xxxxxxxxxxxxxxxx', autocomplete: 'off', id: 'hf-token-' + Math.random().toString(36).slice(2, 6) }), msg = h('div', { class: 'note bad', hidden: true });
      const go1 = async () => { const t = inp.value.trim(); if (!t) return; btn.disabled = true; btn.textContent = 'Checking…'; msg.hidden = true; try { const u = await HFAcct.connect(t); toast('Connected as @' + u.name, 'good'); } catch (e) { msg.hidden = false; msg.textContent = e.message; } btn.disabled = false; btn.textContent = 'Connect'; };
      const btn = h('button', { class: 'btn primary', onclick: go1 }, 'Connect');
      box.replaceChildren(h('div', { class: 'row', style: 'justify-content:space-between' }, h('h3', {}, 'Connect your Hugging Face account'), chip('', 'not connected')),
        h('div', { class: 'step' }, h('b', {}, '1'), h('div', {}, 'Open ', h('a', { href: 'https://huggingface.co/settings/tokens/new?tokenType=read', target: '_blank', rel: 'noopener' }, 'huggingface.co/settings/tokens'), ' and create a token of type ', h('b', {}, 'Read'), '.')),
        h('div', { class: 'step' }, h('b', {}, '2'), h('div', {}, 'Paste it here. It is checked with Hugging Face and stored only in this browser.')),
        h('div', { class: 'row', style: 'flex-wrap:nowrap' }, inp, btn), msg);
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
