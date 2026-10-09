// ---- ExpertStream engine worker: runs the WebAssembly engine on the user's GGUF file ----
let M = null, file = null, busy = false;
const reader = new FileReaderSync();
const dec = new TextDecoder();
const post = (type, data) => postMessage(Object.assign({ type }, data || {}));
function readAt(ptr, n, off) {
  const ab = reader.readAsArrayBuffer(file.slice(off, off + n));
  M.HEAPU8.set(new Uint8Array(ab), ptr);
  return ab.byteLength;
}
function onProgress(d, t) {
  if (d === 0) {   // token ids are ready before the prompt is read
    const base = M._esw_ids() >> 2;
    return post('tokens', { n: t, ids: Array.from(M.HEAP32.subarray(base, base + Math.min(t, 64))) });
  }
  post('progress', { done: d, total: t });
}
function piece(ptr) {
  const heap = M.HEAPU8; let end = ptr; while (heap[end]) end++;
  return dec.decode(heap.subarray(ptr, end), { stream: true });
}
self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'open') {
      file = m.file;
      const t0 = performance.now();
      M = await ESEngine({ readAt, onProgress,  print: s => post('log', { text: s }), printErr: s => post('log', { text: s }) });
      const rc = M._esw_init(file.size, m.ctx || 1024, m.cache || 384);
      if (rc) return post('error', { text: M.UTF8ToString(M._esw_error()) });
      post('ready', { info: JSON.parse(M.UTF8ToString(M._esw_info())), ms: performance.now() - t0, heap_mb: M.HEAPU8.length / 1048576 });
    } else if (m.type === 'ask') {
      M._esw_sampling(m.temp ?? 0.7, 40, 0.9, m.max ?? 256);
      const p = M.stringToNewUTF8(m.text);
      const n = M._esw_begin(p, m.raw ? 1 : 0);
      M._free(p);
      if (n < 0) return post('done', { stats: null, text: 'prompt too long' });
      post('next');
    } else if (m.type === 'next') {
      const ptr = M._esw_next();
      if (!ptr) return post('done', { stats: JSON.parse(M.UTF8ToString(M._esw_stats())), heap_mb: M.HEAPU8.length / 1048576 });
      post('tok', { t: piece(ptr) });
    } else if (m.type === 'stop') {
      post('done', { stats: JSON.parse(M.UTF8ToString(M._esw_stats())), stopped: true, heap_mb: M.HEAPU8.length / 1048576 });
    } else if (m.type === 'reset') {
      M._esw_reset(); post('reset');
    }
  } catch (err) {
    post('error', { text: String(err && err.message || err) });
  }
};
