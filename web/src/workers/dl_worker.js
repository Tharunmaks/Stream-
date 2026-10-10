// ---- download worker: resumable HTTP Range download straight into OPFS ----
// messages in:  {type:'start', id, url, name, total?, headers?}  {type:'cancel', id}
// messages out: {type:'progress', id, done, total, bps} {type:'done', id, size} {type:'error', id, text}
const jobs = {};
const post = (o) => postMessage(o);
async function run(m) {
  const job = jobs[m.id] = { cancel: false };
  let handle = null;
  try {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle('models', { create: true });
    const fh = await dir.getFileHandle(m.name, { create: true });
    handle = await fh.createSyncAccessHandle();
    let have = handle.getSize();
    let total = 0;
    {   // ask the server for the exact size (catalog sizes are only estimates)
      const r = await fetch(m.url, { method: 'GET', headers: Object.assign({ Range: 'bytes=0-0' }, m.headers || {}) });
      const cr = r.headers.get('content-range');
      total = cr ? +cr.split('/')[1] : +r.headers.get('content-length') || 0;
      try { await r.body.cancel(); } catch (e) {}
    }
    if (total && have > total) { handle.truncate(0); have = 0; }
    if (total && have === total) { post({ type: 'done', id: m.id, size: total }); return; }
    let t0 = performance.now(), b0 = have, bps = 0, lastPost = 0, retries = 0;
    while (!job.cancel && (!total || have < total)) {
      let r;
      try {
        r = await fetch(m.url, { headers: Object.assign({ Range: 'bytes=' + have + '-' }, m.headers || {}) });
        if (r.status === 200 && have > 0) { handle.truncate(0); have = 0; }   // server ignored Range
        else if (r.status !== 200 && r.status !== 206) throw new Error('HTTP ' + r.status + (r.status === 401 || r.status === 403 ? ' (this model needs a Hugging Face login/token)' : ''));
        if (!total) total = (+r.headers.get('content-length') || 0) + have;
        const rd = r.body.getReader();
        for (;;) {
          const { done, value } = await rd.read();
          if (done) break;
          handle.write(value, { at: have }); have += value.length;
          const now = performance.now();
          if (now - lastPost > 250) {
            if (now - t0 > 1500) { bps = (have - b0) / ((now - t0) / 1000); t0 = now; b0 = have; }
            lastPost = now; post({ type: 'progress', id: m.id, done: have, total, bps });
          }
          if (job.cancel) { try { await rd.cancel(); } catch (e) {} break; }
        }
        retries = 0;
      } catch (err) {
        if (job.cancel) break;
        if (/HTTP 4/.test(String(err.message))) throw err;
        if (++retries > 8) throw err;
        post({ type: 'retry', id: m.id, n: retries, text: String(err.message || err) });
        await new Promise(res => setTimeout(res, Math.min(8000, 800 * retries)));
      }
      if (!total && !job.cancel) break;
    }
    handle.flush();
    if (job.cancel) post({ type: 'cancelled', id: m.id, done: have });
    else if (total && have !== total) throw new Error('download ended early at ' + have + ' of ' + total + ' bytes');
    else post({ type: 'done', id: m.id, size: have });
  } catch (err) {
    post({ type: 'error', id: m.id, text: String(err && err.message || err) });
  } finally {
    try { handle && handle.close(); } catch (e) {}
    delete jobs[m.id];
  }
}
async function copyFile(m) {
  let handle = null;
  try {
    const root = await navigator.storage.getDirectory();
    const dir = await root.getDirectoryHandle('models', { create: true });
    handle = await (await dir.getFileHandle(m.name, { create: true })).createSyncAccessHandle();
    handle.truncate(0);
    const rd = m.file.stream().getReader(); let at = 0, last = 0;
    for (;;) {
      const { done, value } = await rd.read(); if (done) break;
      handle.write(value, { at }); at += value.length;
      const now = performance.now(); if (now - last > 250) { last = now; post({ type: 'progress', id: m.id, done: at, total: m.file.size, bps: 0 }); }
    }
    handle.flush(); post({ type: 'copied', id: m.id, size: at });
  } catch (err) { post({ type: 'error', id: m.id, text: String(err && err.message || err) }); }
  finally { try { handle && handle.close(); } catch (e) {} }
}
self.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'copy') copyFile(m);
  else if (m.type === 'start') run(m);
  else if (m.type === 'cancel' && jobs[m.id]) jobs[m.id].cancel = true;
};
