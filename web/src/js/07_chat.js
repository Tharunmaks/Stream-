/* ===== Chat ===== */
const hist = { list: store.get('hist', []), add(e) { this.list.unshift(e); this.list = this.list.slice(0, 80); store.set('hist', this.list); } };
const settings = { get temp() { return store.get('temp', 0.7); }, get max() { return store.get('max', 256); } };
const STAGES = ['Words → numbers', 'Connect brain', 'Read prompt', 'Fetch experts', 'Answer'];
let chatBusy = false;
function answerHTML(text, live) {
  let think = '', rest = text;
  const m = text.match(/<think>([\s\S]*?)(<\/think>|$)/);
  if (m) { think = m[1].trim(); rest = (text.slice(0, m.index) + text.slice(m.index + m[0].length)).trim(); }
  return (think ? `<details class="think" ${m[2] ? '' : 'open'}><summary>${m[2] ? 'Thought process' : 'Thinking…'}</summary>${esc(think).replace(/\n/g, '<br>')}</details>` : '') + `<div class="md">${md(rest)}</div>` + (live ? '<span class="caret"></span>' : '');
}
function chatChipUpdate() {
  const c = $('#chat-chip'); const on = Brain.ready;
  c.className = 'chip ' + (Brain.loading ? 'warn' : on ? 'good' : '');
  c.replaceChildren(h('i', { style: on ? '' : 'background:var(--ink-3)' }), Brain.loading ? 'loading…' : on ? Brain.name + (Brain.mode === 'native' ? ' · native' : ' · in page') : 'no model');
  const p = $('#pill'); p.className = 'pill ' + (Brain.loading ? 'busy' : on ? 'on' : '');
  $('#pill-t').textContent = Brain.loading ? 'Loading…' : on ? Brain.name : 'No model';
  $('#term-state').textContent = on ? Brain.name : 'no model';
}
function chatWelcome() {
  const box = $('#msgs'); box.replaceChildren();
  if (!Brain.ready) {
    Lib.list().catch(() => []).then(items => {
      const have = items.filter(i => i.complete);
      box.append(h('div', { class: 'empty' }, h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.4, html: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>' }), h('h3', {}, 'No model is loaded'),
        h('p', {}, have.length ? 'Load one of your saved models, or get a new one.' : 'Download a small model first (138 MB), then come back. It takes a minute.'),
        h('div', { class: 'row', style: 'justify-content:center' }, have.slice(0, 2).map(it => h('button', { class: 'btn primary', onclick: () => loadModel({ opfs: it.name, label: it.file || it.name }).catch(() => { }) }, 'Load ' + (it.file || it.name).slice(0, 26))), h('button', { class: 'btn' + (have.length ? '' : ' primary'), onclick: () => go('models') }, 'Get a model'))));
    });
    return;
  }
  box.append(h('div', { class: 'msg bot' }, h('div', { class: 'bot-body' }, h('div', { class: 'md' }, h('p', {}, 'Hi. I\'m running ' + Brain.name + (Brain.mode === 'web' ? ' inside this page' : ' on your phone') + '. Ask me anything.')))),
    h('div', { class: 'suggest' }, SUGGEST.map(s => h('button', { onclick: () => sendChat(s) }, s))));
}
async function sendChat(text) {
  text = (text || '').trim(); if (!text || chatBusy) return;
  if (!Brain.ready) { toast('Load a model first', 'bad'); go('models'); return; }
  const box = $('#msgs');
  if (box.querySelector('.suggest')) chatWelcomeClear();
  chatBusy = true; $('#send').textContent = 'Stop'; $('#send').type = 'button';
  box.append(h('div', { class: 'msg user' }, text));
  const stages = STAGES.map((s, i) => h('span', { class: 'stage' }, h('span', { class: 'dot' }), s, h('b', { class: 'v mono', style: 'font-weight:400' })));
  const ids = h('div', { class: 'ids' }), ans = h('div', {}), stats = h('div', { class: 'stats' });
  const ramv = h('span', { class: 'mono dim' }), ramfill = h('i', { style: 'width:0' });
  const body = h('div', { class: 'bot-body' }, h('div', { class: 'stages' }, stages), ids, h('div', { class: 'row', style: 'justify-content:space-between;font-size:12px' }, h('span', { class: 'dim' }, 'Memory used by the engine'), ramv), h('div', { class: 'bar' }, ramfill), ans, stats);
  box.append(h('div', { class: 'msg bot' }, body)); box.scrollTop = box.scrollHeight;
  const st = (i, cls, v) => { stages[i].className = 'stage' + (cls ? ' ' + cls : ''); if (v != null) stages[i].querySelector('.v').textContent = ' ' + v; };
  const setRam = mb => { if (!mb) return; ramv.textContent = mb + ' MB'; ramfill.style.width = Math.min(100, mb / 29) + '%'; };
  st(0, 'on'); let answer = '', t0 = 0, n = 0, last = 0;
  try {
    await Brain.ask(text, { temp: settings.temp, max: settings.max }, e => {
      if (e.ev === 'tokens') { e.ids.slice(0, 48).forEach(id => ids.append(h('span', {}, id))); st(0, 'ok', e.n + ' tokens'); st(1, 'ok', Math.round(Brain.info.core_mb) + ' MB'); st(2, 'on'); st(3, 'on'); }
      else if (e.ev === 'prompt') { if (stages[0].className.includes('on')) { st(0, 'ok'); st(1, 'ok'); st(2, 'on'); st(3, 'on'); } st(2, 'on', e.done + '/' + e.n); if (e.flash_mb) st(3, 'on', Math.round(e.flash_mb) + ' MB'); setRam(e.ram_mb); if (e.done >= e.n) { st(2, 'ok'); st(4, 'on'); t0 = performance.now(); } }
      else if (e.ev === 'tok') { if (!t0) { t0 = performance.now(); st(2, 'ok'); st(4, 'on'); } answer += e.t; n++; const now = performance.now(); if (now - last > 60) { last = now; ans.innerHTML = answerHTML(answer, true); st(4, 'on', (n / ((now - t0) / 1000)).toFixed(2) + ' tok/s'); box.scrollTop = box.scrollHeight; } }
      else if (e.ev === 'info') body.insertBefore(h('div', { class: 'note' }, e.msg), ans);
      else if (e.ev === 'done') {
        stages.forEach((_, i) => st(i, 'ok')); setRam(e.ram_mb); ans.innerHTML = answerHTML(answer, false);
        if (e.flash_mb) st(3, 'ok', Math.round(e.flash_mb) + ' MB'); st(4, 'ok', e.tps ? (+e.tps).toFixed(2) + ' tok/s' : '');
        stats.textContent = `${e.tokens} tokens · ${(+e.secs).toFixed(1)} s · ${e.tps ? (+e.tps).toFixed(2) : '?'} tok/s · prompt ${(+e.prompt_secs || 0).toFixed(1)} s${e.hit ? ' · experts ' + Math.round(e.hit) + '% from cache' : ''}${e.stopped ? ' · stopped' : ''}`;
        hist.add({ t: Date.now(), q: text, a: answer, model: Brain.name, tps: e.tps, tokens: e.tokens }); if (answer) body.append(h('div', { class: 'row' }, h('button', { class: 'btn sm', onclick: () => copyText(answer) }, 'Copy answer')));
      }
    });
  } catch (err) {
    if (!(err && err.name === 'AbortError')) body.append(h('div', { class: 'note bad' }, 'The engine stopped: ' + err.message));
  }
  chatBusy = false; const sb = $('#send'); sb.textContent = 'Send'; sb.type = 'submit'; box.scrollTop = box.scrollHeight;
}
function chatWelcomeClear() { $$('#msgs .suggest').forEach(e => e.remove()); }
$('#compose').addEventListener('submit', e => { e.preventDefault(); const p = $('#prompt'); const t = p.value; p.value = ''; p.style.height = 'auto'; sendChat(t); });
$('#send').addEventListener('click', () => { if (chatBusy) Brain.stop(); });
$('#prompt').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey && !('ontouchstart' in window)) { e.preventDefault(); $('#compose').requestSubmit(); } });
$('#prompt').addEventListener('input', e => { e.target.style.height = 'auto'; e.target.style.height = Math.min(160, e.target.scrollHeight) + 'px'; });
$('#chat-new').addEventListener('click', async () => { if (chatBusy) return; await Brain.reset(); chatWelcome(); });
$('#chat-set').addEventListener('click', () => {
  const sl = (key, label, min, max, step, def, fmt, note) => { const v = h('b', { class: 'mono' }, fmt(store.get(key, def))); return h('label', { class: 'field' }, h('span', { class: 'row', style: 'justify-content:space-between' }, label, v), h('input', { type: 'range', min, max, step, value: store.get(key, def), oninput: e => { store.set(key, +e.target.value); v.textContent = fmt(+e.target.value); } }), note ? h('span', { class: 'dim', style: 'font-size:12px' }, note) : null); };
  const s = sheet([h('h2', { style: 'font-size:20px' }, 'Chat settings'), sl('temp', 'Creativity (temperature)', 0, 1.5, .05, .7, v => v.toFixed(2), '0 = always the most likely word'), sl('max', 'Longest answer (tokens)', 32, 1024, 32, 256, v => v), sl('ctx', 'Memory of the chat (tokens)', 256, 4096, 256, 1024, v => v, 'Applies the next time a model loads'), sl('cache', 'Expert cache (MB)', 64, 1024, 32, 384, v => v + ' MB', 'Applies the next time a model loads'), h('button', { class: 'btn', onclick: () => s.close() }, 'Done')]);
});
$('#pill').addEventListener('click', () => {
  if (!Brain.ready) return go('models');
  const i = Brain.info, s = sheet([h('span', { class: 'eyebrow' }, Brain.mode === 'native' ? 'Running natively' : 'Running in this page'), h('h2', { style: 'font-size:20px' }, Brain.name),
    h('div', { class: 'tiles', style: 'grid-template-columns:repeat(2,1fr)' }, tile(i.arch || '?', 'architecture'), tile(Math.round(i.core_mb) + ' MB', 'core in memory'), tile(i.experts ? i.experts + ' / ' + i.used : 'dense', 'experts / used'), tile(i.layers + ' layers', 'depth')),
    h('div', { class: 'row' }, h('button', { class: 'btn primary', onclick: () => { s.close(); go('chat'); } }, 'Chat'), h('button', { class: 'btn', onclick: () => { s.close(); go('models'); } }, 'Switch model'), Brain.mode === 'web' ? h('button', { class: 'btn danger', onclick: () => { Brain.close(); s.close(); toast('Model unloaded'); } }, 'Unload') : null)]);
});
Brain.on(() => { chatChipUpdate(); if (!chatBusy && !$('#msgs').querySelector('.msg.user')) chatWelcome(); });
pageHooks.chat = () => { chatChipUpdate(); if (!chatBusy && !$('#msgs').querySelector('.msg.user')) chatWelcome(); };
