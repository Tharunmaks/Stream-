/* ===== GGUF header inspector + "can this run?" verdict ===== */
const GT = { 0: ['F32', 1, 4], 1: ['F16', 1, 2], 2: ['Q4_0', 32, 18], 3: ['Q4_1', 32, 20], 6: ['Q5_0', 32, 22], 7: ['Q5_1', 32, 24], 8: ['Q8_0', 32, 34], 9: ['Q8_1', 32, 36], 10: ['Q2_K', 256, 84], 11: ['Q3_K', 256, 110], 12: ['Q4_K', 256, 144], 13: ['Q5_K', 256, 176], 14: ['Q6_K', 256, 210], 15: ['Q8_K', 256, 292], 16: ['IQ2_XXS', 256, 66], 17: ['IQ2_XS', 256, 74], 18: ['IQ3_XXS', 256, 98], 19: ['IQ1_S', 256, 50], 20: ['IQ4_NL', 32, 18], 21: ['IQ3_S', 256, 110], 22: ['IQ2_S', 256, 82], 23: ['IQ4_XS', 256, 136], 24: ['I8', 1, 1], 25: ['I16', 1, 2], 26: ['I32', 1, 4], 27: ['I64', 1, 8], 28: ['F64', 1, 8], 29: ['IQ1_M', 256, 56], 30: ['BF16', 1, 2], 34: ['TQ1_0', 256, 54], 35: ['TQ2_0', 256, 66], 39: ['MXFP4', 32, 17], 41: ['Q1_0', 128, 18], 42: ['Q2_0', 64, 18], 64: ['IQ1_XS', 256, 46], 65: ['IQ1_XXS', 256, 42], 66: ['IQ1_XXXS', 256, 38] };
const ENGINE_TYPES = new Set([0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30, 34, 35, 39, 41, 42, 64, 65, 66]);
/* verified = logits checked against PyTorch or llama.cpp (scripts/test_archs.py) */
const ENGINE_ARCH = {
  olmoe: 'verified', qwen3moe: 'verified', qwen3: 'verified', qwen2: 'verified', qwen2moe: 'verified', llama: 'verified', mixtral: 'verified',
  'gpt-oss': 'verified', deepseek2: 'verified', qwen35: 'verified', qwen35moe: 'verified',
};
const PRE_OK = new Set('default a.x-4.0 bailingmoe bailingmoe2 chatglm-bpe codeshell command-r dbrx deepseek-coder deepseek-llm deepseek-r1-qwen deepseek-v3 exaone exaone-moe exaone4 f2llmv2 falcon falcon-h1 falcon3 gigachat glm4 glm5 gpt-2 gpt-4o granite-docling grok-2 hunyuan hunyuan-dense hy_v4 jais jais-2 jina-v1-en jina-v2-code jina-v5-nano joyai-llm k2-horizon kanana2 kimi-k2 kormo laguna lfm2 llada-moe llama-bpe llama-v3 llama3 llama4 megrez mellum mellum2 midm-2.0 minerva-7b minicpm5 minimax-m2 modern-bert mpt olmo phi-2 pixtral qwen2 qwen35 refact roberta-bpe seed-coder smaug-bpe smollm solar-open stablelm2 starcoder talkie tekken trillion ufakzeka'.split(' '));
class NeedMore extends Error { }
function parseGGUF(buf) {
  const dv = new DataView(buf), u8 = new Uint8Array(buf), td = new TextDecoder();
  let p = 0;
  const need = n => { if (p + n > buf.byteLength) throw new NeedMore(); };
  const u32 = () => { need(4); const v = dv.getUint32(p, true); p += 4; return v; };
  const u64 = () => { need(8); const v = Number(dv.getBigUint64(p, true)); p += 8; return v; };
  const str = () => { const n = u64(); need(n); const s = td.decode(u8.subarray(p, p + n)); p += n; return s; };
  const skipStr = () => { const n = u64(); need(n); p += n; };
  const SZ = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
  const scalar = t => { need(SZ[t]); let v; switch (t) { case 0: v = dv.getUint8(p); break; case 1: v = dv.getInt8(p); break; case 2: v = dv.getUint16(p, true); break; case 3: v = dv.getInt16(p, true); break; case 4: v = dv.getUint32(p, true); break; case 5: v = dv.getInt32(p, true); break; case 6: v = dv.getFloat32(p, true); break; case 7: v = !!dv.getUint8(p); break; case 10: v = Number(dv.getBigUint64(p, true)); break; case 11: v = Number(dv.getBigInt64(p, true)); break; case 12: v = dv.getFloat64(p, true); break; } p += SZ[t]; return v; };
  need(24);
  if (dv.getUint32(0, true) !== 0x46554747) throw new Error('not a GGUF file (bad magic)');
  p = 4; const version = u32(); if (version < 2 || version > 3) throw new Error('GGUF version ' + version + ' is not supported');
  const nT = u64(), nKV = u64();
  const kv = {}, arrLen = {};
  for (let i = 0; i < nKV; i++) {
    const key = str(), t = u32();
    if (t === 8) kv[key] = str();
    else if (t === 9) {
      const et = u32(), n = u64(); arrLen[key] = n;
      if (et === 8) { for (let j = 0; j < n; j++) skipStr(); }
      else if (SZ[et]) { need(SZ[et] * n); if (n <= 16) { kv[key] = []; for (let j = 0; j < n; j++) kv[key].push(scalar(et)); } else p += SZ[et] * n; }
      else throw new Error('unsupported metadata array type ' + et);
    } else if (SZ[t]) kv[key] = scalar(t);
    else throw new Error('unsupported metadata type ' + t);
  }
  const tensors = [];
  for (let i = 0; i < nT; i++) {
    const name = str(), nd = u32(), dims = []; for (let d = 0; d < nd; d++) dims.push(u64());
    const type = u32(), off = u64();
    const g = GT[type]; let n = 1; for (const d of dims) n *= d;
    const bytes = g ? Math.ceil(n / g[1]) * g[2] : 0;
    tensors.push({ name, dims, type, n, bytes, off });
  }
  return { version, kv, arrLen, tensors, headerEnd: p };
}
/* Summarise a parsed header: sizes, quant mix, MoE shape. */
function summarize(g, fileSize) {
  const kv = g.kv, arch = kv['general.architecture'] || '?';
  const k = s => kv[arch + '.' + s];
  const L = k('block_count') || 0, emb = k('embedding_length') || 0, nh = k('attention.head_count') || 0;
  const nkv = Array.isArray(k('attention.head_count_kv')) ? k('attention.head_count_kv')[0] : (k('attention.head_count_kv') || nh);
  const hd = k('attention.key_length') || (nh ? emb / nh : 0);
  const nExp = k('expert_count') || 0, nUsed = k('expert_used_count') || 0;
  let expB = 0, coreB = 0, params = 0, expParams = 0; const types = {};
  for (const t of g.tensors) {
    params += t.n; types[t.type] = (types[t.type] || 0) + t.bytes;
    if (/_exps\./.test(t.name)) { expB += t.bytes; expParams += t.n; } else coreB += t.bytes;
  }
  const rope = k('rope.dimension_count');
  return {
    arch, name: kv['general.name'] || kv['general.basename'] || '', size: fileSize, L, emb, nh, nkv, hd, nExp, nUsed,
    ctxTrain: k('context_length') || 0, vocab: g.arrLen['tokenizer.ggml.tokens'] || 0,
    tokModel: kv['tokenizer.ggml.model'] || '', tokPre: kv['tokenizer.ggml.pre'] || '', rope,
    expB, coreB, params, expParams, types, moe: nExp > 1 && expB > 0,
  };
}
function typeMix(sum) {
  return Object.entries(sum.types).sort((a, b) => b[1] - a[1]).map(([t, b]) => [GT[t] ? GT[t][0] : 'type' + t, b]);
}
/* Device facts: what this browser/phone can do. */
const ENV = { webMemMB: 1792, nativeFreeMB: 0, flashGBs: 1.75, disk: 151 };
/* Verdict. mode: 'web' (WebAssembly in this page) or 'native' (es_serve / Termux) */
function assess(sum, opts = {}) {
  const r = { sum, blockers: [], warns: [], notes: [], web: false, native: false };
  const base = ENGINE_ARCH[sum.arch];
  if (!base) r.blockers.push(`Architecture "${sum.arch}" is not supported by the engine yet. Supported: Llama 3, Mixtral, Qwen2/2.5/3/3.5/3.6/3.8 (dense and MoE), OLMoE, gpt-oss, DeepSeek V2/V3/R1, Kimi K2 and Mistral Large 3.`);
  if (sum.tokModel && sum.tokModel !== 'gpt2' && sum.tokModel !== 'llama') r.blockers.push(`Tokenizer "${sum.tokModel}" is not supported yet (byte-level BPE and SentencePiece are).`);
  else if (sum.tokModel === 'gpt2' && sum.tokPre && !PRE_OK.has(sum.tokPre)) r.warns.push(`Tokenizer pre-splitter "${sum.tokPre}" is unknown to this engine; words may be split differently from the original.`);
  const bad = Object.keys(sum.types).filter(t => !ENGINE_TYPES.has(+t)).map(t => GT[t] ? GT[t][0] : 'type' + t);
  if (bad.length) r.blockers.push(`Uses a weight format the engine can't decode yet: ${bad.join(', ')}. Pick another quantization of the same model.`);
  if (/-0000\d-of-\d{5}/.test(opts.fileName || '')) r.warns.push('Split GGUF: run it with the native engine (Termux or PC), naming the first part (-00001-of-…). es_get downloads all parts, or only their core.');
  const ctx = opts.ctx || 1024, cacheMB = opts.cacheMB || 384;
  const kvB = sum.L * 2 * sum.nkv * sum.hd * ctx * 4;
  const coreMB = (sum.moe ? sum.coreB : sum.size) / 1048576, kvMB = kvB / 1048576;
  r.coreMB = coreMB; r.kvMB = kvMB; r.cacheMB = sum.moe ? cacheMB : 0;
  r.needMB = coreMB + kvMB + r.cacheMB + 120;
  r.expertMB = sum.expB / 1048576;
  r.perTokenMB = sum.moe ? (sum.nUsed / sum.nExp) * r.expertMB / 1 : 0;
  if (!r.blockers.length) {
    r.web = r.needMB <= ENV.webMemMB;
    if (!r.web) r.notes.push(`Needs about ${fmtB(r.needMB * 1048576)} of memory; a web page is limited to about ${fmtB(ENV.webMemMB * 1048576)}. Run it natively (Termux) instead.`);
    r.native = true;
  }
  if (sum.size > ENV.disk * 1e9) {
    if (sum.moe && sum.coreB < ENV.disk * 1e9 * 0.8) { r.stream = true; r.notes.push(`File (${fmtB(sum.size)}) is larger than your free flash (${ENV.disk} GB), but its core (${fmtB(sum.coreB)}) fits: es_get downloads the core and the engine downloads each expert the first time it is used, keeping as many as fit.`); }
    else r.blockers.push(`File (${fmtB(sum.size)}) is larger than your free flash (${ENV.disk} GB), and so is its core.`);
  }
  // speed estimate (rough). Native: compute+flash bound. Browser: single thread, ~6x slower.
  if (!r.blockers.length) {
    const active = sum.moe ? sum.coreB + (sum.nUsed / sum.nExp) * sum.expB : sum.size;
    const hit = sum.moe ? Math.min(.9, r.cacheMB / Math.max(1, r.expertMB) * 3) : 1;
    const flash = sum.moe ? active * (sum.expB ? (sum.nUsed / sum.nExp) * sum.expB / active : 0) * (1 - hit) : 0;
    const secNative = active / 1e9 / 5.5 + flash / 1e9 / ENV.flashGBs;
    r.tpsNative = 1 / secNative; r.tpsWeb = r.tpsNative / 6;
  }
  r.ok = !r.blockers.length;
  r.level = !r.ok ? 'bad' : (r.warns.length || !r.web) ? 'warn' : 'good';
  return r;
}
/* Read the header of a local File (grows the window until it parses). */
async function inspectFile(file) {
  for (const mb of [4, 16, 48]) {
    const buf = await file.slice(0, Math.min(file.size, mb << 20)).arrayBuffer();
    try { return summarize(parseGGUF(buf), file.size); } catch (e) { if (!(e instanceof NeedMore) || mb === 48 || buf.byteLength >= file.size) throw e; }
  }
}
async function inspectUrl(url, size) {
  for (const mb of [4, 16, 48]) {
    let r; try { r = await fetch(url, { headers: Object.assign({ Range: `bytes=0-${(mb << 20) - 1}` }, HF.headers()) }); } catch (e) { throw new Error(NET_MSG); }
    if (!r.ok && r.status !== 206) throw new Error(r.status === 401 || r.status === 403 ? 'This model is private or gated. Connect your Hugging Face account (Settings) and accept its license on huggingface.co.' : 'HTTP ' + r.status);
    const buf = await r.arrayBuffer(), cr = r.headers.get('content-range'), total = size || (cr ? +cr.split('/')[1] : buf.byteLength);
    try { return summarize(parseGGUF(buf), total); } catch (e) { if (!(e instanceof NeedMore) || mb === 48) throw e; }
  }
}
