/* ===== Settings: account, engine defaults, storage ===== */
const REPO = 'https://github.com/Tharunmaks/Stream-.git', BRANCH = 'claude/expertstream-moe-android-oegt59';
function cmdBlock(text) { return h('div', { style: 'position:relative' }, h('pre', { class: 'cmd' }, text), h('button', { class: 'btn sm', style: 'position:absolute;top:6px;right:6px', onclick: () => copyText(text) }, 'Copy')); }
function slider(key, label, min, max, step, def, fmt, note) { const v = h('b', { class: 'mono' }, fmt(store.get(key, def))); return h('label', { class: 'field' }, h('span', { class: 'row', style: 'justify-content:space-between' }, label, v), h('input', { type: 'range', min, max, step, value: store.get(key, def), oninput: e => { store.set(key, +e.target.value); v.textContent = fmt(+e.target.value); } }), note ? h('span', { class: 'dim', style: 'font-size:12px' }, note) : null); }
/* ----- Expert Craft ----- */
function craftCard() {
  return h('div', { class: 'card hot' }, h('span', { class: 'eyebrow' }, 'Expert Craft · free · on this phone'), h('h3', {}, 'Train a 70-billion-parameter model on your phone'),
    h('p', { class: 'muted', style: 'font-size:14.5px' }, 'Expert Craft keeps a small model in RAM and puts up to 70 billion parameters in millions of page-sized experts in a file on flash. Each word reads, trains and rewrites only a handful of them, so the work per word is that of a small model. It runs in Termux with Python and NumPy (plus small C kernels it compiles itself), no GPU and no account.'),
    h('div', { class: 'note' }, h('b', {}, 'Honest limit. '), 'The parameters are really trained here, but the quality of the text follows how many words your phone can process and the size of the small model, not the file size. See the plan for your own device before you start.'),
    h('p', { class: 'muted', style: 'font-size:14px' }, 'Give it to your AI as a skill (no MCP). Claude Code, on a PC or in Termux:'),
    cmdBlock('/plugin install expert-craft --marketplace Tharunmaks/Stream-'),
    h('p', { class: 'muted', style: 'font-size:14px' }, 'Any other agent that reads a skills folder, or by hand:'),
    cmdBlock('curl -fsSL https://raw.githubusercontent.com/Tharunmaks/Stream-/claude/expertstream-moe-android-oegt59/scripts/install-skill.sh | sh\n# by hand:\npkg install -y python python-numpy clang\npython3 ~/.claude/skills/expert-craft/scripts/expertcraft.py phone-plan 70e9 --minutes 600'),
    h('div', { class: 'row' }, h('a', { class: 'btn', href: 'craft.html' }, 'Open Expert Craft')));
}
/* ----- agents browser ----- */
function agentsCard() {
  const q = h('input', { type: 'search', placeholder: `Search ${AGENTS_DATA.count} agents: rust, docker, translate, resume…`, id: 'agent-q' }), cat = h('select', { id: 'agent-cat' }, h('option', { value: '' }, 'All categories'), Object.entries(AGENTS_DATA.categories).map(([k, v]) => h('option', { value: k }, v)));
  const list = h('div', { class: 'stack' }), more = h('button', { class: 'btn sm' }, 'Show more'); let shown = 12;
  const draw = () => {
    const w = q.value.toLowerCase().split(/\s+/).filter(Boolean);
    const rows = AGENTS_DATA.agents.filter(a => (!cat.value || a.category === cat.value) && w.every(x => (a.id + ' ' + a.name + ' ' + a.description + ' ' + (a.tags || '')).toLowerCase().includes(x)));
    list.replaceChildren(...rows.slice(0, shown).map(a => h('div', { class: 'mcard', style: 'grid-template-columns:minmax(0,1fr) auto' }, h('div', {}, h('div', { class: 't' }, a.name), h('div', { class: 'm' }, chip('', a.category_name), h('span', { class: 'dim mono', style: 'font-size:12px' }, a.id)), h('p', { class: 'dim', style: 'font-size:13px;margin-top:4px' }, a.description)), h('div', { class: 'acts' }, h('button', { class: 'btn sm primary', onclick: () => { useAgent(a.id); go('chat'); } }, 'Chat'), h('button', { class: 'btn sm', onclick: () => copyText(a.id) }, 'Copy id')))));
    if (!rows.length) list.append(h('div', { class: 'note' }, 'No agent matches. Try a shorter search.'));
    more.hidden = rows.length <= shown;
  };
  q.addEventListener('input', () => { shown = 12; draw(); }); cat.addEventListener('change', () => { shown = 12; draw(); }); more.onclick = () => { shown += 24; draw(); };
  draw();
  return h('div', { class: 'card' }, h('span', { class: 'eyebrow' }, AGENTS_DATA.count + ' agents'), h('h3', {}, 'Agent library'), h('p', { class: 'muted', style: 'font-size:14px' }, 'Each agent is an expert persona with its own instructions, run on whichever model you loaded. Chat with one here.'),
    h('div', { class: 'grid2' }, q, cat), list, h('div', { class: 'row' }, more));
}
async function renderSettings() {
  const box = $('#set-body'); box.replaceChildren();
  box.append(accountCard(() => { if (!$('#settings').hidden) renderSettings(); }));
  // engine defaults
  box.append(h('div', { class: 'card' }, h('h3', {}, 'Engine defaults'), h('div', { class: 'grid2' }, slider('temp', 'Creativity (temperature)', 0, 1.5, .05, .7, v => v.toFixed(2), '0 = always the most likely word'), slider('max', 'Longest answer (tokens)', 32, 1024, 32, 256, v => v), slider('ctx', 'Chat memory (tokens)', 256, 4096, 256, 1024, v => v, 'applies when a model loads'), slider('cache', 'Expert cache (MB)', 64, 1024, 32, 384, v => v + ' MB', 'applies when a model loads'), slider('threads', 'CPU cores for the engine', 0, 8, 1, 0, v => v ? String(v) : 'auto', window.crossOriginIsolated ? 'multi-core is available here · applies when a model loads' : 'this page runs on 1 core (multi-core needs the GitHub Pages site)'))));
  box.append(craftCard());
  box.append(agentsCard());
  // storage
  const usage = await Lib.usage(); let items = []; try { items = await Lib.list(); } catch (e) { }
  box.append(h('div', { class: 'card' }, h('h3', {}, 'Storage and data'), h('div', { class: 'table-box' }, trow('Models saved', String(items.filter(i => i.complete).length)), trow('Used by this site', usage ? fmtB(usage.used) : '?'), trow('Available', usage ? fmtB(usage.quota - usage.used) : '?'), trow('Prompts in history', String(hist.list.length))),
    h('div', { class: 'row' }, h('button', { class: 'btn sm', onclick: () => { hist.list = []; store.set('hist', []); toast('History cleared'); renderSettings(); } }, 'Clear history'), h('button', { class: 'btn sm danger', onclick: async () => { Brain.close(); for (const i of items) await Lib.remove(i.name); toast('All models deleted'); renderSettings(); } }, 'Delete all models'), h('button', { class: 'btn sm danger', onclick: () => { try { Object.keys(localStorage).filter(k => k.startsWith('es2.')).forEach(k => localStorage.removeItem(k)); } catch (e) { } location.reload(); } }, 'Reset everything'))));
  box.append(privacyCard());
  box.append(h('p', { class: 'dim', style: 'font-size:12.5px;text-align:center' }, 'ExpertStream web 2.2 · engine: ' + (Brain.mode || 'idle') + ' · ' + (Brain.native ? 'native server connected' : 'WebAssembly')));
}
pageHooks.settings = renderSettings;
