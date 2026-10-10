/* ===== Testing: doctor, speed test, history ===== */
async function runDoctor() {
  const r = [], add = (name, ok, msg, warn) => r.push({ name, ok, msg, warn });
  add('Secure page', window.isSecureContext, window.isSecureContext ? location.protocol + '//' + location.host : 'needs https:// or 127.0.0.1');
  add('WebAssembly', typeof WebAssembly === 'object', typeof WebAssembly === 'object' ? 'available' : 'blocked by this browser or app');
  let simd = false; try { simd = WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11])); } catch (e) { }
  add('WASM SIMD', simd, simd ? 'fast math available' : 'missing: engine needs a recent Chrome/Edge/Safari');
  let worker = false; try { const w = new Worker(URL.createObjectURL(new Blob(['postMessage(1)'], { type: 'text/javascript' }))); worker = await new Promise(res => { w.onmessage = () => { w.terminate(); res(true); }; w.onerror = () => res(false); setTimeout(() => res(false), 2500); }); } catch (e) { }
  add('Background worker', worker, worker ? 'can start workers' : 'blocked: the page is sandboxed');
  let opfs = false, msg = 'not available'; try { if (navigator.storage && navigator.storage.getDirectory) { const d = await navigator.storage.getDirectory(); const fh = await d.getFileHandle('probe.tmp', { create: true }); await d.removeEntry('probe.tmp'); opfs = true; msg = 'private file storage works'; } } catch (e) { msg = 'blocked: ' + e.message; }
  add('Model storage (OPFS)', opfs, msg);
  add('Multi-core engine', !!window.crossOriginIsolated, window.crossOriginIsolated ? 'on: the model runs on several CPU cores' : 'off: 1 core (reload once on the GitHub Pages site to turn it on)', !window.crossOriginIsolated);
  const u = await Lib.usage(); if (u) add('Free storage', u.quota - u.used > 3e9, fmtB(u.quota - u.used) + ' available to this site', u.quota - u.used < 3e9);
  let hf = false, hfmsg = ''; try { const c = new AbortController(); setTimeout(() => c.abort(), 6000); const x = await fetch('https://huggingface.co/api/models?limit=1', { signal: c.signal }); hf = x.ok; hfmsg = hf ? 'reachable' : 'HTTP ' + x.status; } catch (e) { hfmsg = 'blocked: this page cannot reach the internet. Open the site in Chrome (not inside another app).'; }
  add('Hugging Face', hf, hfmsg);
  const dm = navigator.deviceMemory; if (dm) add('Device memory', true, '≥ ' + dm + ' GB reported (browsers cap the number)');
  add('CPU threads', true, (navigator.hardwareConcurrency || '?') + ' reported; the in-page engine uses 1, the native engine uses the big cores');
  add('Native engine (Termux)', !!Brain.native, Brain.native ? 'connected: ' + Brain.native.model : 'not connected (optional): run es_serve for full speed');
  return r;
}
async function renderDoctor() {
  const box = $('#doctor'); box.replaceChildren(h('div', { class: 'skel', style: 'height:140px' }));
  const r = await runDoctor();
  box.replaceChildren(...r.map(x => h('div', { class: 'trow' }, h('span', { class: 'tick ' + (x.ok ? 'ok' : x.warn ? 'wait' : 'no') }, x.ok ? '✓' : x.warn ? '!' : '✗'), h('div', {}, h('b', {}, x.name), h('div', { class: 'dim', style: 'font-size:12.5px' }, x.msg)), h('span'))));
  const bad = r.filter(x => !x.ok && !x.warn && x.name !== 'Native engine (Termux)');
  const hb = $('#home-banner'); hb.replaceChildren();
  if (bad.length) hb.append(h('div', { class: 'note bad' }, h('b', {}, 'This page is restricted here: '), bad.map(x => x.name).join(', '), '. This preview window blocks downloads and Hugging Face sign-in.', h('div', { class: 'row', style: 'margin-top:6px' }, h('a', { class: 'btn sm primary', href: APP_URL, target: '_blank', rel: 'noopener' }, 'Open the full app'), Demo.man ? h('button', { class: 'btn sm', onclick: () => Demo.load(false) }, 'Run the built-in demo here') : null, h('button', { class: 'btn sm', onclick: () => go('testing') }, 'Details'))));
}
$('#doc-run').addEventListener('click', renderDoctor);
$('#bench-run').addEventListener('click', async () => {
  const box = $('#bench'); if (!Brain.ready) { box.replaceChildren(h('div', { class: 'note' }, 'Load a model first (Models page).')); return; }
  if (Brain.busy) return; box.replaceChildren(h('div', { class: 'bar indet' }, h('i')));
  await Brain.reset(); let toks = 0;
  try { await Brain.ask('Write a short story about a robot who learns to paint.', { temp: 0, max: 48 }, e => { if (e.ev === 'done') { box.replaceChildren(h('div', { class: 'tiles', style: 'grid-template-columns:repeat(2,1fr)' }, tile((+e.tps).toFixed(2) + ' tok/s', 'generation speed'), tile((+e.prompt_secs).toFixed(1) + ' s', 'reading the prompt'), tile(Math.round(e.ram_mb) + ' MB', 'memory'), tile(e.hit ? Math.round(e.hit) + '%' : '—', 'experts from cache'))); store.set('bench', { model: Brain.name, tps: e.tps, t: Date.now() }); } }); } catch (e) { box.replaceChildren(h('div', { class: 'note bad' }, String(e.message))); }
  await Brain.reset();
});
function trow(name, val, kind) { return h('div', { class: 'trow', style: 'grid-template-columns:minmax(0,1fr) auto' }, h('span', {}, name), kind ? chip(kind, val) : h('span', { class: 'mono' }, val)); }
function renderStatic() {
  $('#t-native').replaceChildren(trow('Flash, 4 readers, cold', '1.71 GB/s'), trow('8 experts from one layer', '28.9 ms'), trow('2-bit kernel, 4× A78', '9.81 GB/s'), trow('Same, plus 4× A55', '2.86 GB/s'), trow('Memory read', '13.8 GB/s'), trow('OLMoE-1B-7B, answers', '≈ 3.6 tok/s'), trow('Import OLMoE (1,024 experts)', '17 s'));
  $('#t-correct').replaceChildren(trow('Weight decoding, 7 formats', 'exact', 'good'), trow('Import vs original file', 'byte-identical', 'good'), trow('Tokenizer vs llama.cpp (OLMoE)', '11/11', 'good'), trow('Tokenizer Qwen / Llama 3 / SmolLM rules', 'matches llama-tokenize', 'good'), trow('Logits vs float64 reference', '~1e-6', 'good'), trow('Top token vs llama.cpp (OLMoE)', '14/15', 'warn'), trow('Qwen3-30B-A3B answers', 'coherent', 'good'), trow('Qwen3-30B logits vs llama.cpp', VERIFY_Q3, VERIFY_Q3.startsWith('not') ? 'warn' : 'warn'), trow('Browser engine vs native', 'same tokens', 'good'));
}
const VERIFY_Q3 = '@@Q3VERIFY@@';
pageHooks.testing = () => { renderStatic(); renderDoctor(); };
/* history */
function renderHistory() {
  const box = $('#hist'); box.replaceChildren();
  if (!hist.list.length) box.append(h('div', { class: 'empty' }, h('p', {}, 'Nothing yet. Questions you ask show up here.'), h('button', { class: 'btn primary', onclick: () => go('chat') }, 'Open chat')));
  hist.list.forEach(e => box.append(h('details', { class: 'hitem' }, h('summary', { style: 'cursor:pointer;list-style:none' }, h('div', { class: 'p' }, e.q), h('div', { class: 'dim', style: 'font-size:12px' }, new Date(e.t).toLocaleString() + ' · ' + (e.model || '') + (e.tps ? ' · ' + (+e.tps).toFixed(2) + ' tok/s' : ''))), h('div', { class: 'md', style: 'margin-top:8px', html: answerHTML(e.a || '', false) }), h('div', { class: 'row', style: 'margin-top:8px' }, h('button', { class: 'btn sm', onclick: () => { go('chat'); $('#prompt').value = e.q; $('#prompt').focus(); } }, 'Ask again'), h('button', { class: 'btn sm', onclick: () => copyText(e.a || '') }, 'Copy')))));
  const tot = hist.list.reduce((a, e) => a + (e.tokens || 0), 0);
  $('#hist-stats').replaceChildren(trow('Prompts saved', String(hist.list.length)), trow('Tokens generated', String(tot)), trow('Last speed test', store.get('bench') ? store.get('bench').tps.toFixed(2) + ' tok/s' : '—'));
  const tl = [['Flash loader', '4 parallel readers, 1.75 GB/s'], ['2-bit kernel', 'NEON dot product, bit-exact with llama.cpp'], ['Cache and prefetch', 'full decode loop'], ['Hugging Face import', 'any GGUF MoE, split files too'], ['Real text', 'OLMoE-1B-7B with streamed experts'], ['In-page engine', 'WebAssembly, runs in your browser'], ['Qwen / Llama / dense models', 'Qwen3-30B-A3B generates'], ['Library and downloads', 'Hugging Face to on-device storage, resumable'], ['Faster engine', 'integer SIMD kernels, multi-core, NEON dot-product on phones'], ['More formats', 'BF16, IQ4_NL, IQ4_XS and split GGUF models'], ['Model builder', 'create models from 1 thousand to 37 billion parameters through MCP']];
  $('#hist-time').replaceChildren(...tl.map(([a, b], i) => h('div', { class: 'trow', style: 'grid-template-columns:22px minmax(0,1fr)' }, h('span', { class: 'tick ok' }, '✓'), h('div', {}, h('b', {}, a), h('div', { class: 'dim', style: 'font-size:12.5px' }, b)))), h('div', { class: 'trow', style: 'grid-template-columns:22px minmax(0,1fr)' }, h('span', { class: 'tick wait' }, '…'), h('div', {}, h('b', {}, 'Next'), h('div', { class: 'dim', style: 'font-size:12.5px' }, 'MLA attention (DeepSeek/Kimi), MXFP4 (gpt-oss), split GGUF in the page'))));
}
$('#hist-clear').addEventListener('click', () => { hist.list = []; store.set('hist', []); renderHistory(); });
pageHooks.history = renderHistory;
