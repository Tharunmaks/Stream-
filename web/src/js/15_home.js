/* ===== Home status strip + command palette ===== */
function stripDraw() {
  const s = $('#strip'); if (!s) return;
  const card = (k, v, on, to) => h('button', { class: 'sc', onclick: () => go(to) }, h('span', { class: 'k' }, k), h('span', { class: 'v' }, h('i', { class: on ? 'on' : '' }), v));
  s.replaceChildren(card('Model', Brain.loading ? 'loading…' : Brain.ready ? Brain.name : 'none loaded', Brain.ready, Brain.ready ? 'chat' : 'models'), card('Hugging Face', HFAcct.user ? '@' + HFAcct.user.name : 'not connected', !!HFAcct.user, 'settings'),
    card('AI coder (MCP)', store.get('mcp') ? 'seen on this device' : 'set up', !!store.get('mcp'), 'settings'), card('Engine', Brain.mode === 'native' ? 'native (fast)' : 'in this page', Brain.mode === 'native', 'testing'));
}
Brain.on(stripDraw); HFAcct.on(stripDraw); pageHooks.home = stripDraw;
/* command palette */
async function openPalette() {
  const items = [['Home', () => go('home')], ['Models', () => go('models')], ['Chat', () => go('chat')], ['ExpertStream CLI', () => go('terminal')], ['Testing', () => go('testing')], ['History', () => go('history')], ['Settings', () => go('settings')],
    ['Run all tests', () => { go('testing'); labAll(); }], ['Connect Hugging Face account', () => go('settings')], ['Set up Claude Code (MCP server)', () => go('settings')], ['Search Hugging Face', () => { go('models'); showModelsTab('hf'); }], ['Capacity planner', () => { go('models'); showModelsTab('plan'); }], ['New chat', async () => { await Brain.reset(); go('chat'); chatWelcome(); }], ['Unload model', () => { Brain.close(); toast('Model unloaded'); }]];
  if (Demo.man) items.push(['Run built-in demo model', () => Demo.load(false)]);
  try { (await Lib.list()).filter(i => i.complete).forEach(i => items.push(['Load ' + (i.file || i.name), () => loadModel({ opfs: i.name, label: i.file || i.name }).catch(() => { })])); } catch (e) { }
  const inp = h('input', { class: 'pal-in', type: 'text', placeholder: 'Type a page or action…', autocomplete: 'off' }), list = h('div', { class: 'pal-list' }); let sel = 0, cur = items;
  const sh = sheet([inp, list]);
  const draw = () => { list.replaceChildren(...cur.map(([n, f], i) => h('button', { class: 'pal-item' + (i === sel ? ' sel' : ''), onclick: () => { sh.close(); f(); } }, n))); };
  inp.addEventListener('input', () => { const q = inp.value.toLowerCase(); cur = items.filter(([n]) => n.toLowerCase().includes(q)); sel = 0; draw(); });
  inp.addEventListener('keydown', e => { if (e.key === 'ArrowDown') { sel = Math.min(cur.length - 1, sel + 1); draw(); e.preventDefault(); } else if (e.key === 'ArrowUp') { sel = Math.max(0, sel - 1); draw(); e.preventDefault(); } else if (e.key === 'Enter' && cur[sel]) { sh.close(); cur[sel][1](); } else if (e.key === 'Escape') sh.close(); });
  draw(); setTimeout(() => inp.focus(), 50);
}
$('#palette-btn').addEventListener('click', openPalette);
addEventListener('keydown', e => { const typing = /INPUT|TEXTAREA|SELECT/.test((document.activeElement || {}).tagName || ''); if ((e.key === '/' && !typing) || (e.key.toLowerCase() === 'k' && (e.ctrlKey || e.metaKey))) { e.preventDefault(); openPalette(); } });
