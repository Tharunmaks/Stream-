/* ===== Settings: account, engine defaults, MCP server for AI coders, storage ===== */
const MCP_TOOLS = [['expertstream_status', 'RAM, disk, CPU, engine, loaded model'], ['recommend_models', 'rank models that fit this device (up to 250B) and give the setup steps'], ['plan_capacity', 'will a model of this size fit, and how fast would it run'], ['list_local_models', 'models on the device with a can-it-run verdict'], ['search_huggingface', 'search GGUF models, flags supported architectures'], ['list_repo_files', 'list the GGUF files of a repository'], ['inspect_model', 'read a model header without downloading it'], ['download_model', 'resumable download from Hugging Face'], ['download_status', 'progress of a download'], ['load_model', 'load a model into the engine'], ['generate', 'ask the loaded model'], ['chat_reset', 'start a new conversation'], ['benchmark', 'measure tokens per second'], ['unload_model', 'free memory'], ['import_pack', 'convert a split GGUF MoE for very large models'], ['open_web_ui', 'start this website on the device'], ['connect_huggingface', 'link a Hugging Face account: the AI shows a code, you click Allow'], ['connect_huggingface_finish', 'finish the Hugging Face sign-in'], ['list_agents', 'browse ' + 251 + ' agent presets'], ['get_agent', 'read an agent\'s instructions'], ['recommend_agents', 'which agents fit this task'], ['run_agent', 'run one agent on a task'], ['run_pipeline', 'chain agents: plan → code → test → review'], ['agent_council', 'several agents answer, one merges']];
const REPO = 'https://github.com/Tharunmaks/Stream-.git', BRANCH = 'claude/expertstream-moe-android-oegt59';
function cmdBlock(text) { return h('div', { style: 'position:relative' }, h('pre', { class: 'cmd' }, text), h('button', { class: 'btn sm', style: 'position:absolute;top:6px;right:6px', onclick: () => copyText(text) }, 'Copy')); }
function slider(key, label, min, max, step, def, fmt, note) { const v = h('b', { class: 'mono' }, fmt(store.get(key, def))); return h('label', { class: 'field' }, h('span', { class: 'row', style: 'justify-content:space-between' }, label, v), h('input', { type: 'range', min, max, step, value: store.get(key, def), oninput: e => { store.set(key, +e.target.value); v.textContent = fmt(+e.target.value); } }), note ? h('span', { class: 'dim', style: 'font-size:12px' }, note) : null); }
const MCP_PATH = '~/Stream-/mcp/expertstream_mcp.py';
const CLIENTS = [
  ['Claude Code', 'cc'], ['Cursor', 'cursor'], ['VS Code (Copilot agent)', 'vscode'], ['Windsurf / Cline / Roo / Continue', 'mcpjson'], ['Claude Desktop', 'desktop'],
  ['Gemini CLI', 'gemini'], ['OpenAI Codex CLI', 'codex'], ['Zed', 'zed'], ['Any app with an OpenAI-compatible URL', 'openai'], ['Any other MCP client', 'mcpjson'],
];
function mcpSnippet(kind, remote, host, token) {
  token = token || 'YOUR-TOKEN';
  const url = `http://${host || 'DEVICE-IP'}:8765/mcp`, hdr = { Authorization: 'Bearer ' + token };
  const local = { command: 'python3', args: [MCP_PATH] };
  const J = o => JSON.stringify(o, null, 2);
  if (kind === 'openai') return `Base URL : http://${host || 'DEVICE-IP'}:8765/v1\nAPI key  : ${token}\nModel    : expertstream      (or agent:python, agent:code-reviewer … any agent id)`;
  if (remote) {
    if (kind === 'cc') return `claude mcp add --transport http expertstream ${url} --header "Authorization: Bearer ${token}"`;
    if (kind === 'codex') return `[mcp_servers.expertstream]\nurl = "${url}"\nhttp_headers = { Authorization = "Bearer ${token}" }`;
    if (kind === 'vscode') return J({ servers: { expertstream: Object.assign({ type: 'http', url }, hdr ? { headers: hdr } : {}) } });
    if (kind === 'zed') return J({ context_servers: { expertstream: Object.assign({ url }, hdr ? { headers: hdr } : {}) } });
    return J({ mcpServers: { expertstream: Object.assign({ url }, hdr ? { headers: hdr } : {}) } });
  }
  if (kind === 'cc') return `claude mcp add expertstream -- python3 ${MCP_PATH}`;
  if (kind === 'codex') return `[mcp_servers.expertstream]\ncommand = "python3"\nargs = ["${MCP_PATH}"]`;
  if (kind === 'vscode') return J({ servers: { expertstream: Object.assign({ type: 'stdio' }, local) } });
  if (kind === 'zed') return J({ context_servers: { expertstream: { command: { path: 'python3', args: [MCP_PATH] } } } });
  return J({ mcpServers: { expertstream: local } });
}
const CLIENT_FILE = { cc: 'Run in a terminal (PC or Termux)', cursor: 'Save as .cursor/mcp.json (or ~/.cursor/mcp.json)', vscode: 'Save as .vscode/mcp.json', mcpjson: 'Add to the client\'s MCP settings JSON', desktop: 'Add to claude_desktop_config.json', gemini: 'Add to ~/.gemini/settings.json', codex: 'Add to ~/.codex/config.toml', zed: 'Add to Zed settings.json', openai: 'Enter in the app\'s custom / OpenAI-compatible provider settings' };
function mcpCard() {
  const st = { client: store.get('mcpclient', 0), remote: store.get('mcpremote', false), host: store.get('mcphost', ''), token: store.get('mcptoken', '') };
  const out = h('div', { class: 'stack' }), stat = h('div', { class: 'note' }, 'Not checked yet.');
  const draw = () => {
    const [label, kind] = CLIENTS[st.client];
    const serverCmd = st.remote ? `python3 ${MCP_PATH} --http 8765 --host 0.0.0.0 --model YOUR-MODEL.gguf` : `# nothing to start: ${kind === 'openai' ? 'run this once' : 'the AI coder starts the server itself'}\n${kind === 'openai' ? `python3 ${MCP_PATH} --http 8765 --model YOUR-MODEL.gguf` : ''}`.trim();
    out.replaceChildren(
      h('div', { class: 'step' }, h('b', {}, '1'), h('div', { class: 'stack' }, h('div', {}, h('b', {}, 'Install the engine once'), h('span', { class: 'dim' }, '  on the device that will hold the models (Termux, Linux or macOS)')), cmdBlock(`git clone -b ${BRANCH} ${REPO}\ncd Stream- && make`))),
      st.remote || kind === 'openai' ? h('div', { class: 'step' }, h('b', {}, '2'), h('div', { class: 'stack' }, h('div', {}, h('b', {}, 'Start the server'), h('span', { class: 'dim' }, st.remote ? '  it prints the address and a secret token' : '')), cmdBlock(serverCmd), h('p', { class: 'dim', style: 'font-size:12.5px' }, 'The access token is created automatically. Show it with: cat ~/.expertstream/token'))) : null,
      h('div', { class: 'step' }, h('b', {}, st.remote || kind === 'openai' ? '3' : '2'), h('div', { class: 'stack' }, h('div', {}, h('b', {}, 'Connect ' + label), h('span', { class: 'dim' }, '  ' + CLIENT_FILE[kind])), cmdBlock(mcpSnippet(kind, st.remote, st.host, st.token)))),
      h('div', { class: 'step' }, h('b', {}, st.remote || kind === 'openai' ? '4' : '3'), h('div', { class: 'stack' }, h('div', {}, h('b', {}, 'Ask it')), cmdBlock('Use expertstream: recommend the best model for this device, download it, load it, then use the code-reviewer agent on my last commit.'))));
  };
  const sel = h('select', { id: 'mcp-client', onchange: e => { st.client = +e.target.value; store.set('mcpclient', st.client); draw(); } }, CLIENTS.map(([n], i) => h('option', { value: i, selected: i === st.client }, n)));
  const seg = h('div', { class: 'seg' }, ['Same device', 'Another device (PC ⇄ phone)'].map((t, i) => h('button', { 'aria-selected': (i === 1) === st.remote, onclick: e => { st.remote = i === 1; store.set('mcpremote', st.remote); seg.querySelectorAll('button').forEach((b, j) => b.setAttribute('aria-selected', (j === 1) === st.remote)); netRow.hidden = !st.remote; draw(); } }, t)));
  const hostIn = h('input', { type: 'text', placeholder: 'device IP, e.g. 192.168.1.20', value: st.host, id: 'mcp-host', oninput: e => { st.host = e.target.value.trim(); store.set('mcphost', st.host); draw(); } });
  const tokIn = h('input', { type: 'password', placeholder: 'paste the token: cat ~/.expertstream/token', value: st.token, id: 'mcp-token', oninput: e => { st.token = e.target.value.trim(); store.set('mcptoken', st.token); draw(); } });
  const netRow = h('div', { class: 'grid2', hidden: !st.remote }, h('label', { class: 'field' }, 'IP address of the device running the server', hostIn), h('label', { class: 'field' }, 'Access token', tokIn));
  const chk = h('button', { class: 'btn', onclick: async () => { chk.textContent = 'Checking…'; stat.className = 'note'; const base = st.remote && st.host ? `http://${st.host}:8765` : 'http://127.0.0.1:8765'; try { const r = await fetch(base + '/health', { cache: 'no-store' }); const j = await r.json(); stat.className = 'note ok'; stat.replaceChildren(h('b', {}, 'Connected. '), `${j.tools} tools · ${j.agents} agents${j.device ? ` · RAM free ${j.device.ram_free_mb} MB · disk free ${j.device.disk_free_gb} GB · engine ${j.device.engine ? 'built' : 'NOT built (run make)'}` : ''}${j.loaded ? ' · model loaded' : ' · no model loaded yet'}`); store.set('mcp', Date.now()); } catch (e) { stat.className = 'note bad'; stat.replaceChildren(h('b', {}, 'Not reachable. '), 'Start the server with --http 8765 (see above). Browsers block plain-http addresses from secure pages on other devices; this check works for the same device, and for LAN addresses in Chrome.'); } chk.textContent = 'Check connection'; stripDraw(); } }, 'Check connection');
  draw();
  return h('div', { class: 'card hot' }, h('span', { class: 'eyebrow' }, 'For AI coders · PC and mobile'), h('h3', {}, 'MCP server: 250+ agents and local models for your AI coder'),
    h('p', { class: 'muted', style: 'font-size:14.5px' }, 'Works with Claude Code, Cursor, VS Code, Windsurf, Cline, Codex, Gemini CLI, Zed and any MCP client on a PC or a phone, and with any app that accepts an OpenAI-compatible URL. It searches Hugging Face, downloads, loads and runs models on the device, and gives your AI ' + AGENTS_DATA.count + ' ready-made agents. Models up to 250B parameters.'),
    h('label', { class: 'field' }, 'Which AI coder do you use?', sel), seg, netRow, out, h('div', { class: 'row' }, chk), stat,
    h('details', { class: 'think' }, h('summary', {}, MCP_TOOLS.length + ' tools the AI gets'), h('div', { class: 'toolgrid', style: 'margin-top:8px' }, MCP_TOOLS.map(([n, d]) => h('div', { class: 'tool' }, h('b', {}, n), h('p', {}, d))))),
    h('p', { class: 'dim', style: 'font-size:12.5px' }, 'Secure by default: the HTTP server always needs a token (created once in ~/.expertstream/token, readable only by you), rejects other websites and look-alike hostnames, locks out guessers after 10 wrong tries, caps request size, keeps model files inside one folder, and only downloads from Hugging Face. Over a network use --cert/--key (TLS), a VPN or an SSH tunnel; plain http lets others on the Wi-Fi read the token. One model is loaded at a time.'));
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
  return h('div', { class: 'card' }, h('span', { class: 'eyebrow' }, AGENTS_DATA.count + ' agents'), h('h3', {}, 'Agent library'), h('p', { class: 'muted', style: 'font-size:14px' }, 'Each agent is an expert persona with its own instructions, run on whichever model you loaded. Chat with one here, or let your AI coder call it through MCP (run_agent, run_pipeline, agent_council).'),
    h('div', { class: 'grid2' }, q, cat), list, h('div', { class: 'row' }, more));
}
async function renderSettings() {
  const box = $('#set-body'); box.replaceChildren();
  box.append(accountCard(() => { if (!$('#settings').hidden) renderSettings(); }));
  // engine defaults
  box.append(h('div', { class: 'card' }, h('h3', {}, 'Engine defaults'), h('div', { class: 'grid2' }, slider('temp', 'Creativity (temperature)', 0, 1.5, .05, .7, v => v.toFixed(2), '0 = always the most likely word'), slider('max', 'Longest answer (tokens)', 32, 1024, 32, 256, v => v), slider('ctx', 'Chat memory (tokens)', 256, 4096, 256, 1024, v => v, 'applies when a model loads'), slider('cache', 'Expert cache (MB)', 64, 1024, 32, 384, v => v + ' MB', 'applies when a model loads'), slider('threads', 'CPU cores for the engine', 0, 8, 1, 0, v => v ? String(v) : 'auto', window.crossOriginIsolated ? 'multi-core is available here · applies when a model loads' : 'this page runs on 1 core (multi-core needs the GitHub Pages site)'))));
  // MCP
  box.append(mcpCard());
  box.append(agentsCard());
  // storage
  const usage = await Lib.usage(); let items = []; try { items = await Lib.list(); } catch (e) { }
  box.append(h('div', { class: 'card' }, h('h3', {}, 'Storage and data'), h('div', { class: 'table-box' }, trow('Models saved', String(items.filter(i => i.complete).length)), trow('Used by this site', usage ? fmtB(usage.used) : '?'), trow('Available', usage ? fmtB(usage.quota - usage.used) : '?'), trow('Prompts in history', String(hist.list.length))),
    h('div', { class: 'row' }, h('button', { class: 'btn sm', onclick: () => { hist.list = []; store.set('hist', []); toast('History cleared'); renderSettings(); } }, 'Clear history'), h('button', { class: 'btn sm danger', onclick: async () => { Brain.close(); for (const i of items) await Lib.remove(i.name); toast('All models deleted'); renderSettings(); } }, 'Delete all models'), h('button', { class: 'btn sm danger', onclick: () => { try { Object.keys(localStorage).filter(k => k.startsWith('es2.')).forEach(k => localStorage.removeItem(k)); } catch (e) { } location.reload(); } }, 'Reset everything'))));
  box.append(h('p', { class: 'dim', style: 'font-size:12.5px;text-align:center' }, 'ExpertStream web 2.2 · engine: ' + (Brain.mode || 'idle') + ' · ' + (Brain.native ? 'native server connected' : 'WebAssembly')));
}
pageHooks.settings = renderSettings;
