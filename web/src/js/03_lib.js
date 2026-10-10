/* ===== model library (OPFS) + resumable downloads ===== */
const Lib = {
  supported: !!(navigator.storage && navigator.storage.getDirectory) && typeof Worker !== 'undefined',
  meta: store.get('lib', {}),
  saveMeta() { store.set('lib', this.meta); },
  async dir() { const r = await navigator.storage.getDirectory(); return r.getDirectoryHandle('models', { create: true }); },
  fname(repo, file) { return (repo ? repo.replace('/', '__') + '__' : 'local__') + file.replace(/[^\w.\-]+/g, '_'); },
  async list() {
    if (!this.supported) return [];
    const d = await this.dir(), out = [];
    for await (const [name, fh] of d.entries()) {
      if (fh.kind !== 'file') continue;
      const f = await fh.getFile(), m = this.meta[name] || {};
      out.push(Object.assign({ name, label: name }, m, { have: f.size, complete: m.done === true || (m.done === undefined && f.size > 0) }));
    }
    return out.sort((a, b) => (b.added || 0) - (a.added || 0));
  },
  async remove(name) {
    const d = await this.dir(); await d.removeEntry(name).catch(() => {});
    delete this.meta[name]; this.saveMeta();
  },
  async usage() { try { const e = await navigator.storage.estimate(); return { used: e.usage, quota: e.quota }; } catch (e) { return null; } },
};
const Dl = {
  jobs: {}, worker: null, subs: new Set(), seq: 0,
  on(fn) { this.subs.add(fn); return () => this.subs.delete(fn); },
  emit() { this.subs.forEach(f => { try { f(); } catch (e) { } }); },
  boot() {
    if (this.worker) return this.worker;
    const src = document.getElementById('dl-src').textContent;
    this.worker = new Worker(URL.createObjectURL(new Blob([src], { type: 'text/javascript' })));
    this.worker.onmessage = e => {
      const m = e.data, j = this.jobs[m.id]; if (!j) return;
      if (m.type === 'progress') { if (m.total && j.total !== m.total) { j.total = m.total; } j.done = m.done; j.total = m.total; j.bps = m.bps; j.status = 'downloading'; }
      else if (m.type === 'retry') { j.status = 'retrying'; j.note = 'connection dropped, retry ' + m.n; }
      else if (m.type === 'done') { j.status = 'done'; j.done = j.total = m.size; Lib.meta[j.name] = Object.assign(Lib.meta[j.name] || {}, { size: m.size, done: true }); Lib.saveMeta(); toast(j.file + ' is ready', 'good'); afterSave(j.name); }
      else if (m.type === 'error') { j.status = 'error'; j.note = m.text; toast('Download failed: ' + m.text, 'bad', 6000); }
      else if (m.type === 'cancelled') { j.status = 'paused'; j.done = m.done; }
      else if (m.type === 'copied') { j.status = 'done'; j.done = j.total = m.size; Lib.meta[j.name] = Object.assign(Lib.meta[j.name] || {}, { size: m.size, done: true }); Lib.saveMeta(); toast(j.file + ' added to your library', 'good'); afterSave(j.name); }
      this.emit();
    };
    this.worker.onerror = e => { toast('Download worker failed: ' + (e.message || 'unknown'), 'bad', 6000); };
    return this.worker;
  },
  start(repo, file, size, extra) {
    if (!Lib.supported) { toast('This browser cannot store big files (no OPFS).', 'bad'); return null; }
    const name = Lib.fname(repo, file);
    const old = Object.values(this.jobs).find(j => j.name === name && (j.status === 'downloading' || j.status === 'starting')); if (old) return old.id;
    const id = ++this.seq;
    this.jobs[id] = { id, repo, file, name, done: 0, total: size || 0, bps: 0, status: 'starting' };
    Lib.meta[name] = Object.assign(Lib.meta[name] || {}, { repo, file, size: size || 0, added: Date.now(), done: false }, extra || {}); Lib.saveMeta();
    try { navigator.storage.persist && navigator.storage.persist(); } catch (e) { }
    const hdr = {}; const tok = store.get('hftoken', ''); if (tok) hdr.Authorization = 'Bearer ' + tok;
    this.boot().postMessage({ type: 'start', id, name, total: size || 0, url: HF.resolve(repo, file), headers: hdr });
    this.emit(); return id;
  },
  copy(file) {   // copy a local File into OPFS so it persists
    const name = Lib.fname('', file.name), id = ++this.seq;
    this.jobs[id] = { id, repo: '', file: file.name, name, done: 0, total: file.size, bps: 0, status: 'starting' };
    Lib.meta[name] = Object.assign(Lib.meta[name] || {}, { file: file.name, size: file.size, added: Date.now(), local: true, done: false }); Lib.saveMeta();
    this.boot().postMessage({ type: 'copy', id, name, file });
    this.emit(); return id;
  },
  cancel(id) { this.worker && this.worker.postMessage({ type: 'cancel', id }); },
};

/* after a model is saved: read its header once so the library can show architecture and verdicts */
async function afterSave(name) {
  try { const f = await (await (await Lib.dir()).getFileHandle(name)).getFile(); const sum = await inspectFile(f); Lib.meta[name].sum = sum; Lib.saveMeta(); } catch (e) { }
  if (typeof renderModels === 'function' && !$('#models').hidden) renderModels();
}
