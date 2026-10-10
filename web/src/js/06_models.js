/* ===== Models page ===== */
const glyphOf = n => (n.match(/[A-Za-z0-9]/g) || ['?']).slice(0, 2).join('').toUpperCase();
function ring(frac, label) {
  const C = 2 * Math.PI * 19, d = C * (1 - Math.min(1, Math.max(0, frac)));
  return h('div', { class: 'ring', html: `<svg viewBox="0 0 46 46"><circle class="bg" cx="23" cy="23" r="19"/><circle class="fg" cx="23" cy="23" r="19" stroke-dasharray="${C}" stroke-dashoffset="${d}"/></svg><b>${label ?? Math.round(frac * 100) + '%'}</b>` });
}
const chip = (kind, text) => h('span', { class: 'chip ' + (kind || '') }, kind ? h('i') : null, text);
const SUGGEST = ['Explain what a Mixture-of-Experts model is in simple words.', 'Write a Python function that checks if a number is prime.', 'Give me three ideas for a weekend project.', 'Translate "good morning, how are you?" to French and Spanish.'];

/* ----- loading a model into the engine ----- */
async function loadModel(src, opts = {}) {
  if (Brain.loading) return toast('Already loading a model', 'warn');
  if (!Lib.supported && src.opfs) return toast('This browser cannot read stored models', 'bad');
  const t0 = performance.now();
  const bar = h('div', { class: 'bar indet' }, h('i'));
  const stageEl = h('div', { class: 'muted' }, 'Starting the engine (WebAssembly)…');
  const tick = h('div', { class: 'mono dim' }, '0.0 s');
  const sh = sheet([h('span', { class: 'eyebrow' }, 'Connecting to the brain'), h('h2', { style: 'font-size:22px' }, src.label || 'model'), bar, stageEl, tick, h('div', { class: 'note info' }, 'The first load reads the model\'s core (attention, embeddings, routers) into memory. Experts stay in storage and are read only when a word needs them.')]);
  const iv = setInterval(() => { const s = (performance.now() - t0) / 1000; tick.textContent = s.toFixed(1) + ' s'; if (s > 2) stageEl.textContent = 'Reading the core of the model into memory…'; }, 200);
  try {
    const m = await Brain.open(src, opts);
    clearInterval(iv);
    const i = m.info;
    sh.el.replaceChildren(h('span', { class: 'eyebrow' }, 'Brain connected'), h('h2', { style: 'font-size:22px' }, i.name || src.label),
      h('div', { class: 'tiles', style: 'grid-template-columns:repeat(2,1fr)' },
        tile(Math.round(i.core_mb) + ' MB', 'core in memory'), tile((m.ms / 1000).toFixed(1) + ' s', 'load time'),
        tile(i.experts ? i.experts + ' experts' : 'dense', i.experts ? i.used + ' used per word' : 'every weight each word'), tile(i.cache_slots ? i.cache_slots + ' slots' : '—', i.cache_slots ? 'expert cache (' + Math.round(i.cache_mb) + ' MB)' : 'no expert cache')),
      h('div', { class: 'row' }, h('button', { class: 'btn primary', onclick: () => { sh.close(); go('chat'); } }, 'Chat now'), h('button', { class: 'btn', onclick: () => { sh.close(); go('terminal'); } }, 'Open CLI')));
    toast('Brain connected: ' + (i.name || src.label), 'good');
    return m;
  } catch (e) {
    clearInterval(iv);
    sh.el.replaceChildren(h('span', { class: 'eyebrow', style: 'color:var(--bad)' }, 'Could not load'), h('h2', { style: 'font-size:22px' }, src.label || 'model'),
      h('div', { class: 'note bad' }, String(e.message || e)), h('div', { class: 'row' }, h('button', { class: 'btn', onclick: sh.close }, 'Close')));
    throw e;
  }
}
const tile = (v, k) => h('div', { class: 'tile' }, h('span', { class: 'v', style: 'font-size:20px' }, v), h('span', { class: 'k' }, k));

/* ----- verdict sheet ----- */
function verdictBody(sum, a, label) {
  const lvl = a.level, title = lvl === 'good' ? 'Runs in this page' : lvl === 'warn' ? (a.ok && !a.web ? 'Too big for the page, fine in Termux' : 'Should run, with caveats') : 'Cannot run yet';
  const rows = [
    ['Architecture', sum.arch + (sum.moe ? ` · MoE ${sum.nExp} experts, ${sum.nUsed} per word` : ' · dense')],
    ['Size on disk', fmtB(sum.size)], ['Core kept in memory', fmtB(a.coreMB * 1048576)], ['Context memory (1024 tokens)', fmtB(a.kvMB * 1048576)],
    ...(sum.moe ? [['Experts read from storage', fmtB(sum.expB) + ' (' + fmtB(a.perTokenMB * 1048576) + ' touched per word)']] : []),
    ['Quantisation', typeMix(sum).slice(0, 4).map(([t]) => t).join(', ') || '?'],
    ['Needs in the page', fmtB(a.needMB * 1048576) + ' of ' + fmtB(ENV.webMemMB * 1048576)],
  ];
  if (a.ok) rows.push(['Speed estimate', `page ≈ ${a.tpsWeb.toFixed(a.tpsWeb < 1 ? 2 : 1)} tok/s · Termux ≈ ${a.tpsNative.toFixed(a.tpsNative < 1 ? 2 : 1)} tok/s (rough)`]);
  return [h('span', { class: 'eyebrow' }, label || 'Compatibility'), h('h2', { style: 'font-size:22px' }, sum.name || 'Model'),
    h('div', { class: 'note ' + (lvl === 'good' ? 'ok' : lvl === 'bad' ? 'bad' : '') }, h('b', {}, title)),
    ...a.blockers.map(t => h('div', { class: 'note bad' }, t)), ...a.notes.map(t => h('div', { class: 'note' }, t)), ...a.warns.map(t => h('div', { class: 'note info' }, t)),
    h('div', { class: 'table-box' }, rows.map(([k, v]) => h('div', { class: 'trow', style: 'grid-template-columns:minmax(0,1fr) auto' }, h('span', { class: 'dim' }, k), h('span', { class: 'mono', style: 'text-align:right' }, v))))];
}
async function checkRemote(repo, file, size, btn) {
  const orig = btn && btn.textContent; if (btn) { btn.disabled = true; btn.textContent = 'Checking…'; }
  try {
    const sum = await inspectUrl(HF.resolve(repo, file), size);
    const a = assess(sum, { fileName: file });
    const acts = [];
    if (a.ok) acts.push(h('button', { class: 'btn primary', onclick: () => { s.close(); Dl.start(repo, file, size, { sum }); showModelsTab('lib'); } }, 'Download'));
    acts.push(h('button', { class: 'btn', onclick: () => s.close() }, 'Close'));
    const s = sheet([...verdictBody(sum, a), h('div', { class: 'row' }, acts)]);
    return { sum, a };
  } catch (e) { toast('Could not read the model header: ' + friendlyNet(e.message), 'bad', 9000); }
  finally { if (btn) { btn.disabled = false; btn.textContent = orig; } }
}

/* ----- tabs ----- */
let mtab = store.get('mtab', 'lib');
function showModelsTab(t) { mtab = t; store.set('mtab', t); renderModels(); }
$('#mseg').addEventListener('click', e => { const b = e.target.closest('button[data-m]'); if (b) showModelsTab(b.dataset.m); });
function renderModels() {
  $$('#mseg button').forEach(b => b.setAttribute('aria-selected', b.dataset.m === mtab));
  const body = $('#mbody'); body.replaceChildren();
  ({ lib: renderLib, cat: renderCat, hf: renderHF, dev: renderDev, plan: renderPlan }[mtab])(body);
  if (mtab === 'lib' || mtab === 'cat') { const d = demoCard(); if (d) body.prepend(d); }
}
pageHooks.models = renderModels;

/* ----- My models ----- */
function jobCard(j) {
  const frac = j.total ? j.done / j.total : 0, active = j.status === 'downloading' || j.status === 'starting' || j.status === 'retrying';
  const rem = j.bps > 0 && j.total ? (j.total - j.done) / j.bps : 0;
  const rg = ring(frac), txt = h('span', { class: 'dim mono', style: 'font-size:12px' });
  j._ui = { fg: rg.querySelector('.fg'), pct: rg.querySelector('b'), txt };
  txt.textContent = `${fmtB(j.done)} / ${fmtB(j.total)}${j.bps ? ' · ' + fmtB(j.bps) + '/s' : ''}${rem ? ' · ' + fmtT(rem) + ' left' : ''}`;
  return h('div', { class: 'mcard active' }, rg, h('div', {}, h('div', { class: 't' }, j.file),
    h('div', { class: 'm' }, chip(j.status === 'error' ? 'bad' : j.status === 'done' ? 'good' : 'warn', j.status === 'done' ? 'saved' : j.status === 'error' ? 'failed' : j.status === 'paused' ? 'paused' : 'downloading'), txt),
    j.note ? h('div', { class: 'dim', style: 'font-size:12.5px;margin-top:4px' }, j.note) : null),
    h('div', { class: 'acts' }, active ? h('button', { class: 'btn sm', onclick: () => Dl.cancel(j.id) }, 'Pause') : (j.status === 'paused' || j.status === 'error') ? h('button', { class: 'btn sm primary', onclick: () => { Dl.start(j.repo, j.file, j.total); } }, 'Resume') : null));
}
async function renderLib(body) {
  const jobs = Object.values(Dl.jobs).filter(j => j.status !== 'done');
  if (Brain.mode === 'native') body.append(h('div', { class: 'mcard active' }, h('div', { class: 'glyph' }, glyphOf(Brain.name)), h('div', {}, h('div', { class: 't' }, Brain.name), h('div', { class: 'm' }, chip('good', 'running natively'), chip('', Brain.info.arch || ''), chip('', Math.round(Brain.info.core_mb) + ' MB core'))), h('div', { class: 'acts' }, h('button', { class: 'btn sm primary', onclick: () => go('chat') }, 'Chat'))));
  jobs.forEach(j => body.append(jobCard(j)));
  if (!Lib.supported) body.append(h('div', { class: 'note bad' }, h('b', {}, 'This browser cannot save big files.'), 'It has no private file storage (OPFS). Use Chrome or Edge on Android, or "From device" to run a file without saving it.'));
  let items = []; try { items = await Lib.list(); } catch (e) { }
  const done = items.filter(i => i.complete && !jobs.some(j => j.name === i.name));
  $('#home-lib').textContent = done.length;
  const usage = await Lib.usage();
  done.forEach(it => {
    const on = Brain.source && Brain.source.opfs === it.name && Brain.ready;
    const sum = it.sum;
    body.append(h('div', { class: 'mcard' + (on ? ' active' : '') }, h('div', { class: 'glyph' }, glyphOf(it.file || it.name)),
      h('div', {}, h('div', { class: 't' }, it.file || it.name), h('div', { class: 'm' }, chip('', fmtB(it.have)), sum ? chip('', sum.arch) : null, sum && sum.moe ? chip('', sum.nExp + ' experts') : null, it.repo ? h('span', { class: 'dim', style: 'font-size:12px' }, it.repo) : null)),
      h('div', { class: 'acts' }, on ? h('button', { class: 'btn sm good', onclick: () => go('chat') }, 'Loaded · Chat') : h('button', { class: 'btn sm primary', onclick: () => loadModel({ opfs: it.name, label: it.file || it.name }).catch(() => { }) }, 'Load'),
        h('button', { class: 'btn sm', onclick: async () => { const f = await (await (await Lib.dir()).getFileHandle(it.name)).getFile(); try { const s = await inspectFile(f); it.sum = s; Lib.meta[it.name] = Object.assign(Lib.meta[it.name] || {}, { sum: s }); Lib.saveMeta(); const a = assess(s, { fileName: it.file }); const sh = sheet([...verdictBody(s, a), h('div', { class: 'row' }, h('button', { class: 'btn', onclick: () => sh.close() }, 'Close'))]); } catch (e) { toast(e.message, 'bad'); } } }, 'Info'),
        h('button', { class: 'btn sm danger', onclick: async () => { if (on) Brain.close(); await Lib.remove(it.name); toast('Removed ' + (it.file || it.name)); renderModels(); } }, 'Delete'))));
  });
  if (!done.length && !jobs.length && Brain.mode !== 'native') body.append(h('div', { class: 'card empty' }, h('svg', { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 1.5, html: '<path d="M12 3l9 5-9 5-9-5z"/><path d="M3 13l9 5 9-5"/>' }), h('h3', {}, 'No models yet'), h('p', {}, 'Start with the 138 MB SmolLM2 from the catalog. It downloads in seconds and proves everything works.'), h('div', { class: 'row' }, h('button', { class: 'btn primary', onclick: () => showModelsTab('cat') }, 'Open catalog'), h('button', { class: 'btn', onclick: () => showModelsTab('hf') }, 'Search Hugging Face'))));
  if (usage) body.append(h('p', { class: 'dim', style: 'font-size:12.5px' }, `Storage used by this site: ${fmtB(usage.used)} of ${fmtB(usage.quota)} available.`));
}

/* ----- Catalog ----- */
function renderCat(body) {
  body.append(h('div', { class: 'note info' }, 'Statuses are honest: "runs here" means the engine was tested on that architecture. Everything else says why not.'));
  CATALOG.forEach(m => {
    const [kind, label] = STATUS[m.status];
    const job = Object.values(Dl.jobs).find(j => j.file === m.file && j.status !== 'done');
    const acts = [];
    if (m.file && (m.status === 'ready')) {
      acts.push(h('button', { class: 'btn sm primary', onclick: () => { Dl.start(m.repo, m.file, m.size); showModelsTab('lib'); } }, 'Download ' + fmtB(m.size)));
      acts.push(h('button', { class: 'btn sm', onclick: e => checkRemote(m.repo, m.file, m.size, e.currentTarget) }, 'Check'));
    } else if (m.repo && m.status !== 'no') {
      acts.push(h('button', { class: 'btn sm', onclick: () => { hfState.openRepo = m.repo; showModelsTab('hf'); } }, 'Browse files'));
    } else if (m.status === 'no') acts.push(h('button', { class: 'btn sm', onclick: () => { showModelsTab('plan'); setPlan(2000, 2, 40); } }, 'Open planner'));
    body.append(h('div', { class: 'mcard' }, h('div', { class: 'glyph' }, glyphOf(m.name)), h('div', {}, h('div', { class: 't' }, m.name), h('div', { class: 'm' }, chip(kind, label), chip('', m.kind), chip('', m.params), m.status !== 'ready' || !m.file ? null : chip('', fmtB(m.size))), h('p', { class: 'dim', style: 'font-size:13px;margin-top:6px' }, m.blurb)), h('div', { class: 'acts' }, acts)));
  });
}

/* ----- Hugging Face search ----- */
const hfState = { q: '', results: null, openRepo: null, loading: false, error: '' };
async function hfSearch(q) {
  hfState.q = q; hfState.loading = true; hfState.error = ''; renderModels();
  try { hfState.results = await HF.search(q); } catch (e) { hfState.error = String(e.message || e); hfState.results = []; }
  hfState.loading = false; if (mtab === 'hf') renderModels();
}
async function openRepo(repo) {
  const list = h('div', { class: 'files' }, h('div', { class: 'skel', style: 'height:44px' }), h('div', { class: 'skel', style: 'height:44px' }));
  const sh = sheet([h('span', { class: 'eyebrow' }, 'Hugging Face'), h('h2', { style: 'font-size:20px;word-break:break-all' }, repo), h('p', { class: 'dim', style: 'font-size:13px' }, 'Smaller quantisations (Q2_K–Q4_K_M) are faster and fit; IQ… and BF16 files are not supported yet.'), list, h('div', { class: 'row' }, h('button', { class: 'btn', onclick: () => sh.close() }, 'Close'))]);
  try {
    const files = await HF.files(repo);
    if (!files.length) { list.replaceChildren(h('div', { class: 'note bad' }, 'No .gguf files in this repository.')); return; }
    list.replaceChildren(...files.map(f => {
      const bad = /(^|[-_.])(IQ\d|BF16)|\bbf16\b/i.test(f.path), split = /-\d{5}-of-\d{5}/.test(f.path);
      return h('div', { class: 'file' }, h('div', { style: 'min-width:0' }, h('div', { class: 'n' }, f.path), h('div', { class: 'row', style: 'gap:6px;margin-top:3px' }, chip('', fmtB(f.size)), bad ? chip('bad', 'unsupported quant') : null, split ? chip('warn', 'split · Termux') : null)),
        h('div', { class: 'row', style: 'gap:6px;flex-wrap:nowrap' }, h('button', { class: 'btn sm', onclick: e => checkRemote(repo, f.path, f.size, e.currentTarget) }, 'Check'), h('button', { class: 'btn sm primary', disabled: split || bad, onclick: () => { sh.close(); Dl.start(repo, f.path, f.size); showModelsTab('lib'); } }, 'Get')));
    }));
  } catch (e) { list.replaceChildren(h('div', { class: 'note bad' }, String(e.message || e), h('span', { class: 'dim' }, ' If this page cannot reach huggingface.co, open it in a normal browser tab (see Testing → Doctor).'))); }
}
function renderHF(body) {
  const inp = h('input', { type: 'search', placeholder: 'Search models: qwen, olmoe, llama, smol…', value: hfState.q, enterkeyhint: 'search' });
  const form = h('form', { class: 'row', style: 'flex-wrap:nowrap', onsubmit: e => { e.preventDefault(); hfSearch(inp.value.trim()); } }, inp, h('button', { class: 'btn primary', type: 'submit' }, 'Search'));
  body.append(form, h('div', { class: 'row' }, ['qwen3', 'olmoe', 'smollm2', 'llama 3.2', 'qwen2.5', 'mixtral'].map(q => h('button', { class: 'chip', style: 'cursor:pointer', onclick: () => { inp.value = q; hfSearch(q); } }, q))));
  if (hfState.openRepo) { const r = hfState.openRepo; hfState.openRepo = null; setTimeout(() => openRepo(r), 50); }
  if (hfState.loading) { body.append(...[1, 2, 3, 4].map(() => h('div', { class: 'skel', style: 'height:64px' }))); return; }
  if (hfState.error) body.append(h('div', { class: 'note bad' }, hfState.error));
  if (hfState.results == null) { hfSearch(''); return; }
  hfState.results.forEach(m => {
    const sup = ENGINE_ARCH[m.arch];
    body.append(h('div', { class: 'mcard', style: 'cursor:pointer', onclick: () => openRepo(m.id) }, h('div', { class: 'glyph' }, glyphOf(m.id.split('/')[1] || m.id)),
      h('div', {}, h('div', { class: 't' }, m.id), h('div', { class: 'm' }, m.arch ? chip(sup ? 'good' : 'bad', m.arch + (sup ? '' : ' · unsupported')) : null, m.total ? chip('', fmtN(m.total) + ' params') : null, chip('', '↓ ' + fmtN(m.downloads)))),
      h('div', { class: 'acts' }, h('button', { class: 'btn sm' }, 'Files'))));
  });
  if (hfState.results && !hfState.results.length && !hfState.error) body.append(h('div', { class: 'card empty' }, h('p', {}, 'Nothing found. Try a shorter search.')));
  body.append(h('details', { class: 'think' }, h('summary', {}, 'Private or gated models'), h('p', { class: 'dim' }, 'Paste a Hugging Face read token to access gated repos. It is stored only in this browser and sent only to huggingface.co.'), h('input', { type: 'password', placeholder: 'hf_…', value: store.get('hftoken', ''), onchange: e => { store.set('hftoken', e.target.value.trim()); toast('Token saved on this device'); } })));
}

/* ----- From device ----- */
function renderDev(body) {
  const inp = h('input', { type: 'file', hidden: true });   // no accept filter: Android pickers hide .gguf otherwise
  const out = h('div', { class: 'stack' });
  const drop = h('div', { class: 'drop', tabindex: 0, role: 'button', onclick: () => inp.click() },
    h('svg', { width: 40, height: 40, viewBox: '0 0 24 24', fill: 'none', stroke: 'var(--warm)', 'stroke-width': 1.6, html: '<path d="M12 16V4m0 0l-4 4m4-4l4 4M4 20h16"/>' }),
    h('b', {}, 'Choose a .gguf file from your phone'), h('span', { class: 'dim' }, 'or drop it here · any GGUF, nothing is uploaded'));
  async function pick(f) {
    out.replaceChildren(h('div', { class: 'skel', style: 'height:80px' }));
    try {
      const sum = await inspectFile(f), a = assess(sum, { fileName: f.name });
      const acts = [];
      if (a.ok && a.web) acts.push(h('button', { class: 'btn primary', onclick: () => loadModel({ file: f, label: f.name }).catch(() => { }) }, 'Run now (no copy)'));
      if (a.ok && Lib.supported) acts.push(h('button', { class: 'btn', onclick: () => { Dl.copy(f); showModelsTab('lib'); } }, 'Save to library'));
      const card = h('div', { class: 'card' }, ...verdictBody(sum, a, f.name), h('div', { class: 'row' }, acts));
      out.replaceChildren(card);
    } catch (e) { out.replaceChildren(h('div', { class: 'note bad' }, 'Could not read this file: ' + e.message)); }
  }
  inp.addEventListener('change', () => { if (inp.files[0]) pick(inp.files[0]); });
  ['dragover', 'dragenter'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); if (ev === 'drop' && e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]); }));
  body.append(drop, inp, out, h('p', { class: 'dim', style: 'font-size:12.5px' }, 'On Android pick the file from "Files / Downloads". If you downloaded it with Termux, put it in ~/storage/downloads first (termux-setup-storage).'));
}

/* ----- Capacity planner ----- */
const plan = { params: 2000, bits: 2, active: 40, cache: 2 };
function setPlan(p, b, a) { plan.params = p; plan.bits = b; plan.active = a; renderModels(); }
function renderPlan(body) {
  const out = h('div', { class: 'stack' });
  const sl = (k, label, min, max, step, fmt) => { const v = h('b', { class: 'mono' }, fmt(plan[k])); const inp = h('input', { type: 'range', min, max, step, value: plan[k], oninput: e => { plan[k] = +e.target.value; v.textContent = fmt(plan[k]); calc(); } }); return h('label', { class: 'field' }, h('span', { class: 'row', style: 'justify-content:space-between' }, label, v), inp); };
  function calc() {
    const diskGB = plan.params * plan.bits / 8, actGB = plan.active * plan.bits / 8, fits = diskGB <= ENV.disk;
    const hit = Math.min(.9, plan.cache / Math.max(1, diskGB) * 12), sec = actGB * (1 - hit) / ENV.flashGBs + actGB / 9.8 * 1.0;
    const rows = [['File size on flash', gb(diskGB), fits ? 'good' : 'bad'], ['Your free flash', ENV.disk + ' GB', ''], ['Read per word (active weights)', gb(actGB), ''], ['Expert cache hit rate (rough)', Math.round(hit * 100) + '%', ''], ['Time per word', fmtT(sec), sec > 30 ? 'bad' : sec > 3 ? 'warn' : 'good'], ['A 200-word answer', fmtT(sec * 200 * 1.3), sec > 30 ? 'bad' : 'warn']];
    const verdict = !fits ? `Does not fit: ${gb(diskGB)} needs ${(diskGB / ENV.disk).toFixed(1)}× your flash. Options: fewer bits, a smaller model, or an external SSD over USB.` : sec > 30 ? 'It fits, but every word would take longer than half a minute.' : sec > 3 ? 'It fits and runs, at seconds per word. Useful for patient jobs only.' : 'Fits and reaches readable speed.';
    out.replaceChildren(h('div', { class: 'note ' + (!fits ? 'bad' : sec > 3 ? '' : 'ok') }, h('b', {}, verdict)), h('div', { class: 'table-box' }, rows.map(([k, v, c]) => h('div', { class: 'trow', style: 'grid-template-columns:minmax(0,1fr) auto' }, h('span', { class: 'dim' }, k), c ? chip(c, v) : h('span', { class: 'mono' }, v)))),
      h('p', { class: 'dim', style: 'font-size:12.5px' }, 'Reality check: no open 2-trillion-parameter model exists. If one did, at 2 bits it would be about 500 GB. The architecture here is built so size is limited by storage, not RAM; speed is limited by how many weights each word touches.'));
  }
  const presets = [['7B dense', 7, 4.5, 7], ['Qwen3 30B-A3B', 30, 2.7, 3], ['Qwen3 235B', 235, 2.7, 22], ['Kimi K2 1T', 1000, 2, 32], ['2T hypothetical', 2000, 2, 40]];
  const gb = g => g >= 1000 ? (g / 1000).toFixed(2) + ' TB' : g >= 100 ? Math.round(g) + ' GB' : g.toFixed(1) + ' GB';
  body.append(h('div', { class: 'row' }, presets.map(([n, p, b, a]) => h('button', { class: 'chip', style: 'cursor:pointer', onclick: () => setPlan(p, b, a) }, n))),
    h('div', { class: 'card' }, sl('params', 'Total parameters (billions)', 1, 2500, 1, v => v + 'B'), sl('bits', 'Bits per weight', 1, 8, .1, v => (+v).toFixed(1)), sl('active', 'Parameters used per word (billions)', 1, 400, 1, v => v + 'B'), sl('cache', 'Expert cache in RAM (GB)', .2, 4, .1, v => (+v).toFixed(1) + ' GB')), out);
  calc();
  body.append(simPanel());
}
let lastSig = '';
Dl.on(() => {
  if ($('#models').hidden || mtab !== 'lib') return;
  const jobs = Object.values(Dl.jobs), sig = jobs.map(j => j.id + j.status).join();
  if (sig !== lastSig) { lastSig = sig; renderModels(); return; }
  jobs.forEach(j => { if (!j._ui) return; const f = j.total ? j.done / j.total : 0, C = 2 * Math.PI * 19; j._ui.fg.setAttribute('stroke-dashoffset', C * (1 - Math.min(1, f))); j._ui.pct.textContent = Math.round(f * 100) + '%'; const rem = j.bps > 0 && j.total ? (j.total - j.done) / j.bps : 0; j._ui.txt.textContent = `${fmtB(j.done)} / ${fmtB(j.total)}${j.bps ? ' · ' + fmtB(j.bps) + '/s' : ''}${rem ? ' · ' + fmtT(rem) + ' left' : ''}`; });
});

/* ----- built-in demo model (published next to the page, no internet needed) ----- */
const Demo = { man: null, async probe() { try { const r = await fetch('demo/manifest.json'); if (r.ok) { this.man = await r.json(); renderModels(); if (typeof chatWelcome === 'function' && !$('#msgs').querySelector('.msg.user')) chatWelcome(); } } catch (e) { } },
  async load(save) {
    const man = this.man, prog = h('div', { class: 'bar' }, h('i', { style: 'width:0' })), txt = h('div', { class: 'mono dim' }, '0%');
    const sh = sheet([h('span', { class: 'eyebrow' }, 'Built-in demo'), h('h2', { style: 'font-size:20px' }, man.name), prog, txt]);
    try {
      const parts = []; let got = 0;
      for (const p of man.parts) { const b = await (await fetch('demo/' + p)).blob(); parts.push(b); got += b.size; prog.firstChild.style.width = (100 * got / man.size) + '%'; txt.textContent = fmtB(got) + ' / ' + fmtB(man.size); }
      const file = new File(parts, man.name); sh.close();
      if (save && Lib.supported) { Dl.copy(file); showModelsTab('lib'); } else await loadModel({ file, label: man.name + ' (demo)' });
    } catch (e) { sh.close(); toast('Could not load the demo: ' + e.message, 'bad', 6000); }
  } };
function demoCard() {
  if (!Demo.man) return null;
  return h('div', { class: 'mcard active' }, h('div', { class: 'glyph' }, 'SM'), h('div', {}, h('div', { class: 't' }, 'Built-in demo: SmolLM2 135M'), h('div', { class: 'm' }, chip('good', 'no internet needed'), chip('', fmtB(Demo.man.size))), h('p', { class: 'dim', style: 'font-size:13px;margin-top:6px' }, 'Ships with this page. Small and simple, but it proves the engine runs here.')),
    h('div', { class: 'acts' }, h('button', { class: 'btn sm primary', onclick: () => Demo.load(false) }, 'Run demo'), Lib.supported ? h('button', { class: 'btn sm', onclick: () => Demo.load(true) }, 'Save') : null));
}

/* ----- 2T simulation: clearly labelled, no real weights ----- */
function simPanel() {
  const cv = h('canvas', { style: 'width:100%;height:190px;border-radius:12px;background:#050810;display:block' }), stat = h('div', { class: 'mono', style: 'font-size:13px' }), btn = h('button', { class: 'btn' }, 'Run simulation');
  const card = h('div', { class: 'card' }, h('span', { class: 'eyebrow' }, 'Simulation, not a real model'), h('h3', {}, 'Watch a 2-trillion-parameter model stream'),
    h('p', { class: 'muted', style: 'font-size:14px' }, 'No such model exists, so nothing real is computed. This replays the engine\'s method with the planner numbers above: each word picks experts, cached ones are free, the rest are read from flash at 1.75 GB/s. It shows why speed, not memory, is the limit.'), cv, stat, h('div', { class: 'row' }, btn));
  let run = false;
  btn.onclick = () => {
    if (run) { run = false; btn.textContent = 'Run simulation'; return; }
    run = true; btn.textContent = 'Stop';
    const L = 60, E = 64, g = cv.getContext('2d'), W = cv.width = cv.clientWidth * 2, H = cv.height = 380, cached = new Uint8Array(L * E), heat = new Float32Array(L * E);
    const diskGB = plan.params * plan.bits / 8, actGB = plan.active * plan.bits / 8, perExpGB = actGB / (L * 8), cap = Math.max(8, Math.round(plan.cache / Math.max(perExpGB, 1e-6)));
    let tok = 0, simT = 0, readGB = 0, hits = 0, picks = 0, cc = 0; const fifo = [];
    const zipf = () => Math.min(E - 1, Math.floor(E * Math.pow(Math.random(), 2.2)));
    (function step() {
      if (!run) return;
      for (let l = 0; l < L; l++) for (let k = 0; k < 8; k++) { const i = l * E + zipf(); picks++; heat[i] = 1; if (cached[i]) hits++; else { readGB += perExpGB; simT += perExpGB / ENV.flashGBs; cached[i] = 1; fifo.push(i); if (fifo.length > cap) cached[fifo.shift()] = 0; } }
      simT += actGB / 9.8; tok++;
      g.clearRect(0, 0, W, H); const cw = W / E, ch = H / L;
      for (let l = 0; l < L; l++) for (let e = 0; e < E; e++) { const i = l * E + e; heat[i] *= .9; g.fillStyle = heat[i] > .15 ? `rgba(242,201,138,${.3 + heat[i] * .7})` : cached[i] ? 'rgba(57,135,229,.55)' : 'rgba(80,100,140,.18)'; g.fillRect(e * cw + .5, l * ch + .5, cw - 1, ch - 1); }
      stat.textContent = `word ${tok} · simulated time ${fmtT(simT)} · read from flash ${readGB.toFixed(0)} GB · cache hits ${Math.round(100 * hits / picks)}% · ${(tok / simT).toFixed(3)} words/s`;
      setTimeout(step, 90);
    })();
  };
  return card;
}
