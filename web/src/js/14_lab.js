/* ===== Test lab: environment, network, storage and model behaviour tests ===== */
async function labAsk(text, o = {}) {
  let out = '', done = null, ids = 0;
  await Brain.ask(text, Object.assign({ temp: 0, max: 48 }, o), e => { if (e.ev === 'tok') out += e.t; else if (e.ev === 'done') done = e; else if (e.ev === 'tokens') ids = e.n; if (o.on) o.on(e); });
  return { text: out.trim(), done, ids };
}
const need = () => { if (!Brain.ready) return { status: 'skip', detail: 'load a model first' }; };
const LAB = [
  { group: 'This browser', id: 'wasm', name: 'WebAssembly + SIMD', run: async () => { const r = await runDoctor(); const w = r.find(x => x.name === 'WebAssembly'), s = r.find(x => x.name === 'WASM SIMD'); return { status: w.ok && s.ok ? 'ok' : 'bad', detail: s.msg }; } },
  { group: 'This browser', id: 'worker', name: 'Background worker', run: async () => { const r = (await runDoctor()).find(x => x.name === 'Background worker'); return { status: r.ok ? 'ok' : 'bad', detail: r.msg }; } },
  { group: 'This browser', id: 'mem', name: 'Can allocate 1 GB of engine memory', run: async () => { try { const m = new WebAssembly.Memory({ initial: 16384 }); const n = m.buffer.byteLength; return { status: 'ok', detail: fmtB(n) + ' allocated' }; } catch (e) { try { const m = new WebAssembly.Memory({ initial: 8192 }); return { status: 'warn', detail: '1 GB failed, 512 MB works: use smaller models' }; } catch (x) { return { status: 'bad', detail: String(x.message) }; } } } },
  { group: 'Storage', id: 'opfs', name: 'Model storage read/write speed', run: async () => {
    if (!Lib.supported) return { status: 'bad', detail: 'no private file storage in this browser' };
    const root = await navigator.storage.getDirectory(), fh = await root.getFileHandle('speedtest.tmp', { create: true });
    const w = await fh.createWritable(); const chunk = new Uint8Array(4 << 20); const t0 = performance.now(); for (let i = 0; i < 8; i++) await w.write(chunk); await w.close();
    const t1 = performance.now(), f = await fh.getFile(); await f.arrayBuffer(); const t2 = performance.now(); await root.removeEntry('speedtest.tmp');
    const wr = 32 / ((t1 - t0) / 1000), rd = 32 / ((t2 - t1) / 1000); return { status: rd > 100 ? 'ok' : 'warn', detail: `write ${wr.toFixed(0)} MB/s · read ${rd.toFixed(0)} MB/s` }; } },
  { group: 'Storage', id: 'quota', name: 'Free space for models', run: async () => { const u = await Lib.usage(); if (!u) return { status: 'warn', detail: 'browser does not report it' }; const free = u.quota - u.used; return { status: free > 3e9 ? 'ok' : free > 5e8 ? 'warn' : 'bad', detail: fmtB(free) + ' available' }; } },
  { group: 'Network', id: 'hfapi', name: 'Reach Hugging Face', run: async () => { const t0 = performance.now(); try { const r = await fetch(HF.base + '/api/models?limit=1'); return { status: r.ok ? 'ok' : 'bad', detail: `HTTP ${r.status} in ${Math.round(performance.now() - t0)} ms` }; } catch (e) { return { status: 'bad', detail: NET_MSG }; } } },
  { group: 'Network', id: 'hfspeed', name: 'Download speed from Hugging Face', run: async () => { try { const t0 = performance.now(); const r = await fetch(HF.resolve('bartowski/SmolLM2-135M-Instruct-GGUF', 'SmolLM2-135M-Instruct-Q8_0.gguf'), { headers: { Range: 'bytes=0-8388607' } }); const b = await r.arrayBuffer(); const mbps = b.byteLength / 1e6 / ((performance.now() - t0) / 1000); return { status: mbps > 3 ? 'ok' : 'warn', detail: mbps.toFixed(1) + ' MB/s  (' + fmtT(2.5e9 / 1e6 / mbps) + ' for a 2.5 GB model)' }; } catch (e) { return { status: 'bad', detail: NET_MSG }; } } },
  { group: 'Network', id: 'hfauth', name: 'Hugging Face account', run: async () => { if (!HFAcct.token) return { status: 'skip', detail: 'not connected (optional)' }; try { const u = await HFAcct.connect(HFAcct.token); return { status: 'ok', detail: '@' + u.name }; } catch (e) { return { status: 'bad', detail: e.message }; } } },
  { group: 'Network', id: 'mcp', name: 'MCP server on this device', run: async () => { try { const j = await (await fetch('http://127.0.0.1:8765/health', { cache: 'no-store' })).json(); return { status: 'ok', detail: j.tools + ' tools' }; } catch (e) { return { status: 'skip', detail: 'not running (optional, see Settings)' }; } } },
  { group: 'Model', id: 'tok', name: 'Words become token numbers', run: async () => need() || (async () => { const r = await labAsk('Hello there, how are you today?', { max: 1 }); return { status: r.ids > 3 ? 'ok' : 'bad', detail: r.ids + ' tokens for 6 words' }; })() },
  { group: 'Model', id: 'speed', name: 'Speed (48 tokens)', run: async () => need() || (async () => { await Brain.reset(); const r = await labAsk('Write a short story about a robot who learns to paint.', { max: 48 }); const t = r.done ? +r.done.tps : 0; return { status: t > 1 ? 'ok' : t > .15 ? 'warn' : 'bad', detail: t.toFixed(2) + ' tok/s · prompt ' + (+r.done.prompt_secs).toFixed(1) + ' s' }; })() },
  { group: 'Model', id: 'greedy', name: 'Same question, same answer (temperature 0)', run: async () => need() || (async () => { await Brain.reset(); const a = await labAsk('Name three fruits.', { max: 24 }); await Brain.reset(); const b = await labAsk('Name three fruits.', { max: 24 }); return { status: a.text === b.text ? 'ok' : 'bad', detail: a.text === b.text ? 'identical' : 'answers differ' }; })() },
  { group: 'Model', id: 'qa', name: 'Knows the capital of France', run: async () => need() || (async () => { await Brain.reset(); const r = await labAsk('What is the capital of France? Answer in one sentence.', { max: 32 }); return { status: /paris/i.test(r.text) ? 'ok' : 'warn', detail: r.text.slice(0, 70) }; })() },
  { group: 'Model', id: 'math', name: 'Simple arithmetic', run: async () => need() || (async () => { await Brain.reset(); const r = await labAsk('What is 7 times 8? Reply with just the number.', { max: 12 }); return { status: /56/.test(r.text) ? 'ok' : 'warn', detail: r.text.slice(0, 50) || '(empty)' }; })() },
  { group: 'Model', id: 'json', name: 'Follows a format (JSON)', run: async () => need() || (async () => { await Brain.reset(); const r = await labAsk('Reply with only JSON: {"city": "Paris", "country": "France"}', { max: 40 }); const m = r.text.match(/\{[\s\S]*\}/); let ok = false; try { ok = !!(m && JSON.parse(m[0])); } catch (e) { } return { status: ok ? 'ok' : 'warn', detail: r.text.slice(0, 60) }; })() },
  { group: 'Model', id: 'memory', name: 'Remembers the conversation', run: async () => need() || (async () => { await Brain.reset(); await labAsk('My name is Zephyr. Remember it.', { max: 16 }); const r = await labAsk('What is my name?', { max: 16 }); await Brain.reset(); return { status: /zephyr/i.test(r.text) ? 'ok' : 'warn', detail: r.text.slice(0, 60) }; })() },
  { group: 'Model', id: 'long', name: 'Long prompt (about 250 words)', run: async () => need() || (async () => { await Brain.reset(); const para = 'The quick brown fox jumps over the lazy dog near the river bank. '.repeat(20); const r = await labAsk(para + '\nSummarize the text above in five words.', { max: 16 }); return { status: r.done ? 'ok' : 'bad', detail: `prompt took ${(+r.done.prompt_secs).toFixed(1)} s` }; })() },
  { group: 'Model', id: 'stop', name: 'Stop button works', run: async () => need() || (async () => { await Brain.reset(); let n = 0; const t0 = performance.now(); await Brain.ask('Count from 1 to 200 slowly.', { temp: 0, max: 200 }, e => { if (e.ev === 'tok' && ++n === 3) Brain.stop(); }).catch(() => { }); const ms = performance.now() - t0; return { status: n < 40 ? 'ok' : 'bad', detail: `stopped after ${n} tokens` }; })() },
  { group: 'Model', id: 'reset', name: 'New chat clears memory', run: async () => need() || (async () => { await Brain.reset(); await labAsk('The secret word is banana.', { max: 8 }); await Brain.reset(); const r = await labAsk('What is the secret word?', { max: 20 }); return { status: /banana/i.test(r.text) ? 'warn' : 'ok', detail: r.text.slice(0, 60) }; })() },
];
const labState = {};
function labDraw() {
  const list = $('#lab-list'); list.replaceChildren(); let g = '';
  for (const t of LAB) {
    if (t.group !== g) { g = t.group; list.append(h('div', { class: 'eyebrow', style: 'margin:14px 0 2px' }, g)); }
    const s = labState[t.id] || {}, cls = { ok: 'ok', warn: 'warn', bad: 'no', run: 'run', skip: 'wait' }[s.status] || 'wait', ic = { ok: '✓', warn: '!', bad: '✗', run: '…', skip: '–' }[s.status] || '·';
    list.append(h('div', { class: 'tl-row' }, h('span', { class: 'tick ' + cls }, ic), h('div', {}, h('div', {}, t.name), s.detail ? h('div', { class: 'd' }, s.detail) : null), h('span', { class: 'd mono' }, s.ms ? (s.ms / 1000).toFixed(1) + ' s' : ''), h('button', { class: 'btn sm', disabled: labBusy, onclick: () => labRun(t) }, 'Run')));
  }
}
let labBusy = false;
async function labRun(t) {
  if (labBusy) return; labBusy = true; labState[t.id] = { status: 'run', detail: 'running…' }; labDraw(); const t0 = performance.now();
  try { const r = await t.run(); labState[t.id] = Object.assign({ ms: performance.now() - t0 }, r); } catch (e) { labState[t.id] = { status: 'bad', detail: String(e.message || e), ms: performance.now() - t0 }; }
  labBusy = false; labDraw();
}
async function labAll() { for (const t of LAB) { await labRun(t); } const v = Object.values(labState); toast(`${v.filter(x => x.status === 'ok').length} passed · ${v.filter(x => x.status === 'warn').length} warnings · ${v.filter(x => x.status === 'bad').length} failed`, v.some(x => x.status === 'bad') ? 'bad' : 'good', 5000); }
$('#lab-all').addEventListener('click', labAll);
$('#lab-copy').addEventListener('click', () => copyText('ExpertStream test report · ' + new Date().toISOString() + '\nModel: ' + (Brain.ready ? Brain.name + ' (' + Brain.mode + ')' : 'none') + '\n' + LAB.map(t => { const s = labState[t.id] || {}; return `[${(s.status || 'not run').toUpperCase()}] ${t.name}${s.detail ? ' — ' + s.detail : ''}`; }).join('\n')));
const _th = pageHooks.testing; pageHooks.testing = () => { _th(); labDraw(); };
