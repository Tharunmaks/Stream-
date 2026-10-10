/* ===== Brain: the running model. Either WebAssembly in a worker (this page) or the native engine behind es_serve ===== */
const Brain = {
  mode: null,          // 'web' | 'native' | null
  info: null, name: '', source: null, ready: false, busy: false, loading: false,
  native: null,        // /api/status result when the page is served by es_serve
  worker: null, subs: new Set(), stage: '',
  on(fn) { this.subs.add(fn); return () => this.subs.delete(fn); },
  emit() { this.subs.forEach(f => { try { f(); } catch (e) { } }); },
  async detectNative() {
    if (!/^https?:$/.test(location.protocol)) return null;
    try {
      const c = new AbortController(), t = setTimeout(() => c.abort(), 1500);
      const r = await fetch('/api/status', { signal: c.signal }); clearTimeout(t);
      if (!r.ok) return null;
      const j = await r.json(); if (!j || !j.model) return null;
      this.native = j; this.mode = 'native'; this.info = j; this.name = j.model; this.ready = true; this.emit();
      return j;
    } catch (e) { return null; }
  },
  /* open a model in this page. src = {file: File, label} or {opfs: name, label} */
  async open(src, opt = {}) {
    if (this.loading) return;
    this.close();
    this.loading = true; this.stage = 'starting'; this.emit();
    const mtSrc = document.getElementById('engine-mt-src');
    const mt = window.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined' && mtSrc && mtSrc.textContent.length > 1000 && store.get('threads', 0) !== 1;
    const threads = mt ? Math.max(1, Math.min(8, store.get('threads', 0) || (navigator.hardwareConcurrency >= 8 ? 4 : Math.max(1, (navigator.hardwareConcurrency || 2) - 2)))) : 1;
    const code = (mt ? mtSrc : document.getElementById('engine-src')).textContent;
    let w;
    try { w = new Worker(URL.createObjectURL(new Blob([code], { type: 'text/javascript' }))); }
    catch (e) { this.loading = false; this.emit(); throw new Error('This page cannot start the engine worker (' + e.message + '). Open the site in a normal browser tab, not inside another app.'); }
    this.worker = w;
    return new Promise((resolve, reject) => {
      const to = setTimeout(() => { if (this.loading) { fail('The engine did not answer in 60 s. The browser may have blocked WebAssembly.'); } }, 60000);
      const fail = (text) => { clearTimeout(to); this.loading = false; this.ready = false; try { w.terminate(); } catch (e) { } this.worker = null; this.emit(); reject(new Error(text)); };
      w.onerror = e => fail(e.message || 'engine worker crashed');
      w.onmessage = e => {
        const m = e.data;
        if (m.type === 'ready') {
          clearTimeout(to); this.loading = false; this.ready = true; this.mode = 'web'; this.info = m.info; this.heap = m.heap_mb;
          this.name = (src.label || m.info.name || 'model'); this.source = src; this.loadMs = m.ms; this.threads = m.threads || 1; this.emit(); resolve(m);
        } else if (m.type === 'error') fail(m.text);
        else if (m.type === 'log') this.lastLog = m.text;
        else this.handler && this.handler(m);
      };
      w.postMessage(Object.assign({ type: 'open', threads, ctx: opt.ctx || store.get('ctx', 1024), cache: opt.cache || store.get('cache', 384) }, src.opfs ? { opfs: src.opfs } : { file: src.file }));
    });
  },
  close() {
    if (this.worker) { try { this.worker.terminate(); } catch (e) { } this.worker = null; }
    if (this.mode === 'web') { this.mode = null; this.ready = false; this.info = null; this.name = ''; this.source = null; }
    this.busy = false; this.emit();
  },
  /* ask(text, {raw, temp, max}, onEvent). Events are normalised: tokens{n,ids} prompt{done,n} tok{t} info{msg} done{...stats} error{text} */
  async ask(text, o, on) {
    if (!this.ready || this.busy) throw new Error('no model is ready');
    this.busy = true; this.emit(); this.stopped = false;
    const t0 = performance.now(); let ntok = 0;
    try {
      if (this.mode === 'web') await new Promise((resolve, reject) => {
        const w = this.worker;
        this.handler = m => {
          if (m.type === 'tokens') on({ ev: 'tokens', n: m.n, ids: m.ids });
          else if (m.type === 'progress') on({ ev: 'prompt', done: m.done, n: m.total, ram_mb: Math.round(this.heap || 0) });
          else if (m.type === 'next') w.postMessage({ type: 'next' });
          else if (m.type === 'tok') { ntok++; on({ ev: 'tok', t: m.t }); w.postMessage({ type: this.stopped ? 'stop' : 'next' }); }
          else if (m.type === 'done') {
            const s = m.stats || {};
            on({ ev: 'done', tokens: s.tokens ?? ntok, secs: s.secs ?? (performance.now() - t0) / 1000, tps: s.tps || 0, prompt_secs: s.prompt_s || 0, hit: s.hit || 0, flash_mb: s.read_mb || 0, ram_mb: Math.round(m.heap_mb || this.heap || 0), stopped: !!m.stopped, text: m.text });
            this.handler = null; resolve();
          } else if (m.type === 'error') { this.handler = null; reject(new Error(m.text)); }
        };
        w.postMessage({ type: 'ask', text, raw: !!o.raw, temp: o.temp, max: o.max, topk: o.topk, topp: o.topp });
      });
      else {
        this.ctl = new AbortController();
        const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: text, signal: this.ctl.signal });
        if (!r.ok) throw new Error('server said HTTP ' + r.status);
        const rd = r.body.getReader(), dec = new TextDecoder(); let buf = '';
        for (;;) {
          const { done, value } = await rd.read(); if (done) break;
          buf += dec.decode(value, { stream: true });
          let i; while ((i = buf.indexOf('\n')) >= 0) {
            const line = buf.slice(0, i); buf = buf.slice(i + 1); if (!line.trim()) continue;
            let e; try { e = JSON.parse(line); } catch (x) { continue; }
            if (e.ev === 'done' && e.tokens !== undefined) e = Object.assign(e, { flash_mb: e.flash_mb ?? e.read_mb, ram_mb: e.ram_mb ?? e.mem_mb, prompt_secs: e.prompt_secs ?? e.prompt_s });
            if (e.ev === 'done' && e.tokens === undefined) continue;
            on(e);
          }
        }
      }
    } finally { this.busy = false; this.handler = null; this.emit(); }
  },
  stop() { this.stopped = true; if (this.ctl) { try { this.ctl.abort(); } catch (e) { } } },
  async reset() {
    if (this.mode === 'web' && this.worker) this.worker.postMessage({ type: 'reset' });
    else if (this.mode === 'native') { try { await fetch('/api/reset', { method: 'POST' }); } catch (e) { } }
  },
};
