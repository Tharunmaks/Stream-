// ---- ExpertStream engine worker: runs the WebAssembly engine on a GGUF from a File or from OPFS ----
let M = null, src = null, busy = false, stage = 'idle';
// views over the (possibly shared, possibly grown) wasm memory
const H8 = () => (M.wasmMemory && M.HEAPU8.buffer !== M.wasmMemory.buffer) ? new Uint8Array(M.wasmMemory.buffer) : M.HEAPU8;
const dec = new TextDecoder();
const post = (type, data) => postMessage(Object.assign({ type }, data || {}));
let reader = null, handle = null;
function readAt(ptr, n, off) {
  if (handle) return handle.read(H8().subarray(ptr, ptr + n), { at: off });
  const ab = reader.readAsArrayBuffer(src.slice(off, off + n));
  H8().set(new Uint8Array(ab), ptr);
  return ab.byteLength;
}
function onProgress(d, t) {
  if (d === 0) {   // token ids are ready before the prompt is read
    const base = M._esw_ids() >> 2;
    return post('tokens', { n: t, ids: Array.from(new Int32Array(H8().buffer).subarray(base, base + Math.min(t, 64))) });
  }
  post('progress', { done: d, total: t });
}
function piece(ptr) {
  const heap = H8(); let end = ptr; while (heap[end]) end++;
  return dec.decode(heap.slice(ptr, end), { stream: true });   // slice: shared memory can't be decoded in place
}
if (!/^em-pthread/.test(self.name || '')) self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'open') {
      const t0 = performance.now();
      stage = 'finding the model file';
      let size;
      if (m.opfs) {
        const root = await navigator.storage.getDirectory();
        const dir = await root.getDirectoryHandle('models');
        const fh = await dir.getFileHandle(m.opfs);
        handle = await fh.createSyncAccessHandle();
        size = handle.getSize();
      } else { src = m.file; reader = new FileReaderSync(); size = src.size; }
      stage = 'compiling the engine (WebAssembly)';
      M = await ESEngine({ readAt, onProgress, mainScriptUrlOrBlob: self.location.href, print: s => post('log', { text: s }), printErr: s => post('log', { text: s }) });
      stage = 'reading the model file';
      const rc = M._esw_init(size, m.ctx || 1024, m.cache || 384, m.threads || 1);
      if (rc) return post('error', { text: M.UTF8ToString(M._esw_error()) });
      post('ready', { threads: m.threads || 1, info: JSON.parse(M.UTF8ToString(M._esw_info())), ms: performance.now() - t0, heap_mb: M.HEAPU8.length / 1048576 });
    } else if (m.type === 'ask') {
      M._esw_sampling(m.temp ?? 0.7, m.topk ?? 40, m.topp ?? 0.9, m.max ?? 256);
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
    } else if (m.type === 'close') {
      try { handle && handle.close(); } catch (e) {}
      post('closed');
    }
  } catch (err) {
    post('error', { text: '[' + stage + '] ' + String(err && err.message || err) });
  }
};
