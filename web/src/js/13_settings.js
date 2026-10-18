/* ===== Settings: account, engine defaults, storage ===== */
const REPO = 'https://github.com/Tharunmaks/Stream-.git', BRANCH = 'claude/expertstream-moe-android-oegt59';
function cmdBlock(text) { return h('div', { style: 'position:relative' }, h('pre', { class: 'cmd' }, text), h('button', { class: 'btn sm', style: 'position:absolute;top:6px;right:6px', onclick: () => copyText(text) }, 'Copy')); }
function slider(key, label, min, max, step, def, fmt, note) { const v = h('b', { class: 'mono' }, fmt(store.get(key, def))); return h('label', { class: 'field' }, h('span', { class: 'row', style: 'justify-content:space-between' }, label, v), h('input', { type: 'range', min, max, step, value: store.get(key, def), oninput: e => { store.set(key, +e.target.value); v.textContent = fmt(+e.target.value); } }), note ? h('span', { class: 'dim', style: 'font-size:12px' }, note) : null); }
/* ----- ExpertStream Instant: the native engine for this site and for any app ----- */
function instantCard() {
  const url = h('input', { type: 'text', value: store.get('nativeUrl', 'http://127.0.0.1:8080'), id: 'instant-url', autocomplete: 'off', 'aria-label': 'Address of the local engine' });
  const st = h('div', { class: 'note' + (Brain.mode === 'native' ? ' ok' : ''), role: 'status' }, Brain.mode === 'native' ? 'Connected: ' + Brain.name + (Brain.base ? ' at ' + Brain.base : ' (this page is served by it)') : 'Not connected. Start the engine (below), then press Connect.');
  const go_ = h('button', { class: 'btn', onclick: async () => {
    const u = url.value.trim().replace(/\/$/, ''); store.set('nativeUrl', u); go_.textContent = 'Connecting…';
    const j = await Brain.detectNative(u); go_.textContent = 'Connect';
    st.className = 'note' + (j ? ' ok' : ''); st.textContent = j ? 'Connected: ' + j.model + ' at ' + u + '. Chat now uses the native engine.' : 'No engine answered at ' + u + '. Is es_serve running? (Chrome, Edge and Firefox allow this page to reach 127.0.0.1; Safari may not.)';
    if (j) toast('Connected to the native engine: ' + j.model, 'good');
  } }, 'Connect');
  const auto = h('label', { class: 'row', style: 'gap:8px;font-size:14px' }, h('input', { type: 'checkbox', checked: store.get('nativeAuto', 1) ? '' : null, onchange: e => store.set('nativeAuto', e.target.checked ? 1 : 0) }), 'Connect automatically when this site opens');
  const rows = [['OpenAI-compatible apps and SDKs', 'Base URL http://127.0.0.1:8080/v1 · API key: any text · model: the name in /v1/models'],
    ['Ollama apps', 'Server address http://127.0.0.1:8080'],
    ['Other devices (phone ⇄ PC)', 'Start with --lan; it prints the address and an API key (Authorization: Bearer KEY)'],
    ['Another website', 'Start with --cors https://that.site (this site is allowed already)']];
  return h('div', { class: 'card hot' }, h('span', { class: 'eyebrow' }, 'ExpertStream Instant · any browser, any app'), h('h3', {}, 'Fast answers in every app'),
    h('p', { class: 'muted', style: 'font-size:14.5px' }, 'Run the native engine once. This website and any app that speaks the OpenAI or Ollama API can then use it. Answers start quickly because the engine keeps what it has already read: the next turn reads only your new message, side requests do not throw the chat away, and long system prompts are saved to disk so they are instant even after a restart.'),
    st, h('div', { class: 'grid2' }, url, h('div', { class: 'row' }, go_)), auto,
    cmdBlock('cd ~/Stream- && make -s\n./es_serve -g ~/models/qwen2.5-0.5b-instruct-q4_k_m.gguf\n# OpenAI: http://127.0.0.1:8080/v1   Ollama: http://127.0.0.1:8080'),
    h('div', { class: 'table-box' }, ...rows.map(([k, v]) => trow(k, v))),
    h('details', { class: 'think' }, h('summary', {}, 'Code: curl, Python, JavaScript'),
      cmdBlock('curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \\\n  -d \'{"model":"local","messages":[{"role":"user","content":"Hi"}],"stream":true}\''),
      cmdBlock('from openai import OpenAI\nc = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")\nfor ch in c.chat.completions.create(model="local", messages=[{"role": "user", "content": "Hi"}], stream=True):\n    print(ch.choices[0].delta.content or "", end="")'),
      cmdBlock('const r = await fetch("http://127.0.0.1:8080/v1/chat/completions", {method: "POST", headers: {"Content-Type": "application/json"},\n  body: JSON.stringify({model: "local", messages: [{role: "user", content: "Hi"}]})});\nconsole.log((await r.json()).choices[0].message.content);')),
    h('div', { class: 'note' }, h('b', {}, 'Measured (4-core PC). '), 'Qwen2.5 0.5B, 750-token system prompt, a chat app that resends the conversation: the first word appears after 0.29 s from the 2nd turn on (15.6 s with the previous engine, 8.1 s without reuse), and after 0.12 s on the first turn after a restart (loaded from disk). A prompt seen for the first time is still read at normal speed (7.6 s here).'));
}
/* ----- Expert Craft ----- */
function craftCard() {
  return h('div', { class: 'card hot' }, h('span', { class: 'eyebrow' }, 'Expert Craft · free · on this phone'), h('h3', {}, 'Teach a 200B model your data, on your phone'),
    h('p', { class: 'muted', style: 'font-size:14.5px' }, 'Graft: a big open model (Qwen3 235B, DeepSeek R1, gpt-oss 120B, or any model this engine runs) stays frozen, and a small graft on top learns your facts, names, wording and style. The model reads your text once; then the graft trains without it, at thousands of tokens per second on CPU cores (measured on a PC; phones not yet). Use it in Chat, or in any app through the local OpenAI/Ollama server.'),
    h('div', { class: 'note' }, h('b', {}, 'Honest limit. '), 'Reading your text through a 200B model on a phone takes hours (es_graft plan measures it first), and a graft teaches facts and style, not new abilities. Measured on a PC: 0 to 119 of 120 questions about made-up facts answered right, general answers unchanged.'),
    h('p', { class: 'muted', style: 'font-size:14px' }, 'Also: train a model with up to 70 billion parameters from scratch (Flash Experts). Give it to your AI as a skill (no MCP). Claude Code, on a PC or in Termux:'),
    cmdBlock('/plugin install expert-craft --marketplace Tharunmaks/Stream-'),
    h('p', { class: 'muted', style: 'font-size:14px' }, 'Or by hand in Termux:'),
    cmdBlock('git clone https://github.com/Tharunmaks/Stream-.git ~/Stream- && cd ~/Stream- && make es_graft es_chat es_serve es_get\n./es_graft plan -g MODEL.gguf --data notes.txt\n./es_graft read -g MODEL.gguf --data notes.txt -o my.esfeat\n./es_graft train my.esfeat -o my.esgraft\n./es_chat -g MODEL.gguf -A my.esgraft'),
    h('div', { class: 'row' }, h('a', { class: 'btn', href: 'craft.html#graft' }, 'Open Expert Craft')));
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
  box.append(instantCard());
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
