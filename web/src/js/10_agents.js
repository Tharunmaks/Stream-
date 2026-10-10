/* ===== agents: 250+ expert presets; "Chat with an agent" prepends the agent's instructions ===== */
const AGENTS_DATA = @@AGENTS@@;
const AGENT_BY = Object.fromEntries(AGENTS_DATA.agents.map(a => [a.id, a]));
let agentPending = true;   // send the agent's instructions with the next message
function currentAgent() { return AGENT_BY[store.get('agent', '')] || null; }
function useAgent(id) {
  store.set('agent', id || ''); agentPending = true;
  Brain.reset(); const box = $('#msgs'); if (box) box.replaceChildren(); chatWelcome(); agentChipDraw();
  if (id) toast('Chatting with ' + AGENT_BY[id].name, 'good');
}
function agentChipDraw() { const b = $('#chat-agent'); if (b) b.textContent = currentAgent() ? currentAgent().name : 'General'; }
function agentPicker() {
  const q = h('input', { type: 'search', placeholder: 'Find an agent…', class: 'pal-in', id: 'agent-pick-q' }), list = h('div', { class: 'pal-list' });
  const sh = sheet([h('h2', { style: 'font-size:20px' }, 'Choose an agent'), q, list]);
  const draw = () => { const w = q.value.toLowerCase().split(/\s+/).filter(Boolean); const rows = [{ id: '', name: 'General (no agent)', description: 'Plain chat with the model' }, ...AGENTS_DATA.agents].filter(a => w.every(x => (a.id + ' ' + a.name + ' ' + a.description + ' ' + (a.tags || '')).toLowerCase().includes(x))).slice(0, 40); list.replaceChildren(...rows.map(a => h('button', { class: 'pal-item', onclick: () => { sh.close(); useAgent(a.id); } }, h('span', {}, a.name), h('span', { class: 'dim', style: 'font-size:12px;text-align:right' }, a.description.slice(0, 40))))); };
  q.addEventListener('input', draw); draw(); setTimeout(() => q.focus(), 50);
}
