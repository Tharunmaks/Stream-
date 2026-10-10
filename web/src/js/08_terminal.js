/* ===== ExpertStream CLI (terminal inside the page) ===== */
const T = { out: $('#tout'), inp: $('#tin'), hist: store.get('thist', []), hi: 0, busy: false };
function tline(text, cls, html) { const d = h('div', { class: 'ln ' + (cls || '') }); if (html) d.innerHTML = html; else d.textContent = text; T.out.append(d); T.out.scrollTop = T.out.scrollHeight; return d; }
function tprompt() { $('#tps').textContent = 'es' + (Brain.ready ? '(' + (Brain.info.arch || '?') + ')' : '') + ' ~ $'; }
function tbanner() {
  tline('', '', '<span class="warm">ExpertStream CLI</span> <span class="dim">· runs in this page, no Termux needed</span>');
  tline('Type "help". Plain text is sent to the model as a question.', 'dim');
  Brain.ready ? tline('Model: ' + Brain.name, 'ok') : tline('No model yet. Try:  models   then   get smol135   then   load 1', 'warn');
}
const CMDS = ['help', 'models', 'ls', 'search', 'get', 'load', 'unload', 'ask', 'raw', 'set', 'reset', 'status', 'doctor', 'plan', 'inspect', 'clear', 'rm'];
const HELP = [['models', 'list the catalog (id, size, status)'], ['get <id>', 'download a catalog model, e.g.  get olmoe'], ['get <repo> <file>', 'download any GGUF from Hugging Face'], ['search <text>', 'search Hugging Face for GGUF models'], ['ls', 'list models saved on this device'], ['load <n|name>', 'load a saved model into the engine'], ['unload', 'free the memory'], ['ask <text>', 'ask (same as typing plain text)'], ['raw <text>', 'continue text without the chat template'], ['set temp 0.7 | max 200', 'sampling settings'], ['reset', 'start a new conversation'], ['status', 'what is loaded, memory, speed'], ['inspect <n>', 'compatibility report for a saved model'], ['rm <n>', 'delete a saved model'], ['plan <B params> <bits> [active B]', 'will a huge model fit?  e.g.  plan 2000 2 40'], ['doctor', 'check what this browser can do'], ['clear', 'clear the screen']];
let lsCache = [];
async function refreshLs() { try { lsCache = (await Lib.list()).filter(i => i.complete); } catch (e) { lsCache = []; } return lsCache; }
function pickItem(a) { if (/^\d+$/.test(a)) return lsCache[+a - 1]; return lsCache.find(i => (i.file || i.name).toLowerCase().includes((a || '').toLowerCase())); }
async function runCmd(line) {
  const [cmd, ...rest] = line.trim().split(/\s+/), arg = rest.join(' ');
  switch (cmd) {
    case 'help': HELP.forEach(([c, d]) => tline('', '', `  <span class="code">${esc(c.padEnd(32))}</span> <span class="dim">${esc(d)}</span>`)); break;
    case 'clear': T.out.replaceChildren(); break;
    case 'models': CATALOG.forEach(m => { const s = STATUS[m.status]; tline('', '', `  <span class="code">${esc(m.id.padEnd(9))}</span> ${esc(m.name.padEnd(24))} <span class="dim">${esc(m.params.padEnd(18))}</span> <span class="${s[0] === 'good' ? 'ok' : s[0] === 'warn' ? 'warm' : 'err'}">${esc(s[1])}</span>`); }); tline('get <id> to download', 'dim'); break;
    case 'ls': { const l = await refreshLs(); if (!l.length) tline('Nothing saved yet. Try:  get smol135', 'warn'); l.forEach((i, n) => tline('', '', `  <span class="code">${n + 1}</span>  ${esc(i.file || i.name)}  <span class="dim">${fmtB(i.have)}${i.sum ? ' · ' + esc(i.sum.arch) : ''}</span>`)); break; }
    case 'get': return cmdGet(rest);
    case 'search': {
      tline('searching Hugging Face…', 'dim');
      try { const r = await HF.search(arg); r.slice(0, 12).forEach(m => tline('', '', `  <span class="code">${esc(m.id)}</span> <span class="dim">${esc(m.arch || '?')} · ${m.total ? fmtN(m.total) : '?'} · ↓${fmtN(m.downloads)}</span> ${ENGINE_ARCH[m.arch] ? '<span class="ok">supported</span>' : '<span class="err">unsupported</span>'}`)); tline('then:  get <repo> <file>   (files are listed with: files <repo>)', 'dim'); } catch (e) { tline(String(e.message || e), 'err'); }
      break;
    }
    case 'files': { try { const f = await HF.files(rest[0]); f.forEach(x => tline('', '', `  ${esc(x.path)}  <span class="dim">${fmtB(x.size)}</span>`)); } catch (e) { tline(String(e.message || e), 'err'); } break; }
    case 'load': {
      const l = await refreshLs(); const it = pickItem(arg || '1'); if (!it) { tline('No such model. Use  ls', 'err'); break; }
      tline('connecting to the brain: ' + (it.file || it.name) + ' …', 'blue');
      try { const m = await loadModel({ opfs: it.name, label: it.file || it.name }); tline(`brain connected in ${(m.ms / 1000).toFixed(1)} s · core ${Math.round(m.info.core_mb)} MB · ${m.info.experts ? m.info.experts + ' experts, ' + m.info.used + ' per word' : 'dense model'}`, 'ok'); } catch (e) { tline(String(e.message || e), 'err'); }
      break;
    }
    case 'unload': Brain.close(); tline('unloaded', 'dim'); break;
    case 'rm': { await refreshLs(); const it = pickItem(arg); if (!it) { tline('No such model', 'err'); break; } if (Brain.source && Brain.source.opfs === it.name) Brain.close(); await Lib.remove(it.name); tline('removed ' + (it.file || it.name), 'ok'); break; }
    case 'inspect': { await refreshLs(); const it = pickItem(arg || '1'); if (!it) { tline('No such model', 'err'); break; } try { const f = await (await (await Lib.dir()).getFileHandle(it.name)).getFile(); const s = await inspectFile(f), a = assess(s, { fileName: it.file }); tline(`${s.arch} · ${fmtB(s.size)} · ${s.moe ? s.nExp + ' experts' : 'dense'} · core ${fmtB(a.coreMB * 1048576)} · needs ${fmtB(a.needMB * 1048576)}`, a.ok ? 'ok' : 'err'); [...a.blockers].forEach(x => tline('✗ ' + x, 'err')); [...a.notes, ...a.warns].forEach(x => tline('! ' + x, 'warm')); } catch (e) { tline(String(e.message || e), 'err'); } break; }
    case 'set': { const [k, v] = rest; if (k === 'temp' && v != null) { store.set('temp', +v); tline('temperature ' + v, 'ok'); } else if (k === 'max' && v != null) { store.set('max', +v); tline('max tokens ' + v, 'ok'); } else tline('usage: set temp 0.7 | set max 200', 'err'); break; }
    case 'reset': await Brain.reset(); tline('new conversation', 'ok'); break;
    case 'status': tline(Brain.ready ? `${Brain.name}\n  mode ${Brain.mode} · arch ${Brain.info.arch} · core ${Math.round(Brain.info.core_mb)} MB · layers ${Brain.info.layers}${Brain.info.experts ? ' · experts ' + Brain.info.experts + '/' + Brain.info.used : ''}${Brain.heap ? '\n  wasm memory ' + Math.round(Brain.heap) + ' MB' : ''}` : 'no model loaded', Brain.ready ? 'ok' : 'warm'); break;
    case 'plan': { const [p, b, a] = rest.map(Number); if (!p || !b) { tline('usage: plan <billions> <bits> [active billions]   e.g. plan 2000 2 40', 'err'); break; } const d = p * b / 8, act = (a || p * .1) * b / 8, sec = act * .5 / ENV.flashGBs + act / 9.8; tline(`${p}B @ ${b} bits = ${d.toFixed(0)} GB on flash (you have ${ENV.disk} GB) → ${d <= ENV.disk ? 'FITS' : 'DOES NOT FIT'}\n≈ ${fmtT(sec)} per word (rough)`, d <= ENV.disk ? 'ok' : 'err'); break; }
    case 'doctor': { const r = await runDoctor(); r.forEach(x => tline((x.ok ? '✓ ' : x.warn ? '! ' : '✗ ') + x.name + ' — ' + x.msg, x.ok ? 'ok' : x.warn ? 'warm' : 'err')); break; }
    case 'raw': return cmdAsk(arg, true);
    case 'ask': return cmdAsk(arg, false);
    default: return cmdAsk(line, false);
  }
}
async function cmdGet(rest) {
  let repo, file, size;
  if (rest.length === 1) { const m = CATALOG.find(c => c.id === rest[0]); if (!m || !m.file) { tline('Unknown id. Use  models  (only "runs here" entries have a direct file) or  get <repo> <file>', 'err'); return; } if (m.status !== 'ready') { tline(m.name + ': ' + m.blurb, 'warm'); return; } ({ repo, file, size } = m); }
  else if (rest.length >= 2) { repo = rest[0]; file = rest[1]; if (!file.endsWith('.gguf')) { try { const f = await HF.files(repo); const hit = f.find(x => x.path.toLowerCase().includes(file.toLowerCase())); if (!hit) { tline('No file matching "' + file + '". Try:  files ' + repo, 'err'); return; } file = hit.path; size = hit.size; } catch (e) { tline(String(e.message || e), 'err'); return; } } }
  else { tline('usage: get <id>  or  get <repo> <file or part of its name>', 'err'); return; }
  const id = Dl.start(repo, file, size); if (!id) return;
  const line = tline('', 'blue'); tline('(you can run other commands while it downloads)', 'dim');
  await new Promise(res => { const off = Dl.on(() => { const j = Dl.jobs[id]; if (!j) return; const f = j.total ? j.done / j.total : 0; line.textContent = `  ${'█'.repeat(Math.round(f * 24)).padEnd(24, '░')} ${Math.round(f * 100)}%  ${fmtB(j.done)}/${fmtB(j.total)}${j.bps ? '  ' + fmtB(j.bps) + '/s' : ''}  ${j.status}`; if (j.status === 'done' || j.status === 'error' || j.status === 'paused') { off(); tline(j.status === 'done' ? 'saved. Now:  ls   then   load 1' : (j.note || j.status), j.status === 'done' ? 'ok' : 'err'); res(); } }); });
}
async function cmdAsk(text, raw) {
  if (!text) return;
  if (!Brain.ready) { tline('No model loaded. Use  ls  and  load 1  (or  get smol135  first).', 'warm'); return; }
  T.busy = true;
  const out = tline('', 'ans'); let ans = '', first = true, t0 = 0, n = 0;
  const status = tline('', 'dim');
  try {
    await Brain.ask(text, { raw, temp: settings.temp, max: settings.max }, e => {
      if (e.ev === 'tokens') { status.textContent = `[1/4] words → numbers: ${e.ids.slice(0, 16).join(' ')}${e.n > 16 ? ' …' : ''}  (${e.n} tokens)`; tline(`[2/4] brain: ${Math.round(Brain.info.core_mb)} MB core in memory`, 'dim'); }
      else if (e.ev === 'prompt') status.textContent = `[3/4] reading the prompt ${e.done}/${e.n}` + (e.flash_mb ? ` · ${Math.round(e.flash_mb)} MB from storage` : '');
      else if (e.ev === 'tok') { if (first) { first = false; t0 = performance.now(); } ans += e.t; n++; out.textContent = ans; T.out.scrollTop = T.out.scrollHeight; }
      else if (e.ev === 'info') tline(e.msg, 'warm');
      else if (e.ev === 'done') { tline(`[4/4] ${e.tokens} tokens · ${(+e.secs).toFixed(1)} s · ${e.tps ? (+e.tps).toFixed(2) : '?'} tok/s${e.hit ? ' · experts ' + Math.round(e.hit) + '% from cache' : ''}${e.flash_mb ? ' · ' + Math.round(e.flash_mb) + ' MB from storage' : ''}${e.stopped ? ' · stopped' : ''}`, 'dim'); hist.add({ t: Date.now(), q: text, a: ans, model: Brain.name, tps: e.tps, tokens: e.tokens }); }
    });
  } catch (e) { if (!(e && e.name === 'AbortError')) tline('The engine stopped: ' + e.message, 'err'); }
  T.busy = false;
}
$('#tform').addEventListener('submit', async e => {
  e.preventDefault(); const v = T.inp.value.trim(); if (!v) return; T.inp.value = '';
  if (T.busy) { tline('busy: press CTRL+C to stop first', 'warm'); return; }
  T.hist.push(v); T.hist = T.hist.slice(-50); T.hi = T.hist.length; store.set('thist', T.hist);
  tline('', '', `<span class="ps">$</span> <span class="cmd">${esc(v)}</span>`);
  T.busy = true; try { await runCmd(v); } catch (err) { tline(String(err.message || err), 'err'); } T.busy = false; tprompt();
});
T.inp.addEventListener('keydown', e => {
  if (e.key === 'ArrowUp') { e.preventDefault(); T.hi = Math.max(0, T.hi - 1); T.inp.value = T.hist[T.hi] || ''; }
  else if (e.key === 'ArrowDown') { e.preventDefault(); T.hi = Math.min(T.hist.length, T.hi + 1); T.inp.value = T.hist[T.hi] || ''; }
  else if (e.key === 'Tab') { e.preventDefault(); tab(); }
  else if (e.key === 'c' && e.ctrlKey) { e.preventDefault(); Brain.stop(); }
});
function tab() { const v = T.inp.value; if (!v || v.includes(' ')) return; const m = CMDS.filter(c => c.startsWith(v)); if (m.length === 1) T.inp.value = m[0] + ' '; else if (m.length) tline(m.join('  '), 'dim'); }
$('.keys').addEventListener('click', e => {
  const b = e.target.closest('button'); if (!b) return; const k = b.dataset.key;
  if (k === 'up') { T.hi = Math.max(0, T.hi - 1); T.inp.value = T.hist[T.hi] || ''; T.inp.focus(); }
  else if (k === 'tab') tab(); else if (k === 'stop') { Brain.stop(); tline('^C', 'dim'); }
  else { T.inp.value = k; $('#tform').requestSubmit(); }
});
$('#terminal .term').addEventListener('click', e => { if (!getSelection().toString() && !e.target.closest('button')) T.inp.focus(); });
pageHooks.terminal = () => { if (!T.out.children.length) tbanner(); tprompt(); if (matchMedia('(pointer:fine)').matches) T.inp.focus(); };
Brain.on(tprompt);
