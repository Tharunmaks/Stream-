const NET_MSG = 'This page cannot reach huggingface.co ("Failed to fetch"). That happens inside sandboxed viewers such as the claude.ai preview, or when offline. Open the site from GitHub Pages in Chrome, or use the built-in demo model, which needs no internet.';
const friendlyNet = t => /Failed to fetch|NetworkError|Load failed|network/i.test(t) ? NET_MSG : t;
/* ===== Hugging Face client + model catalog ===== */
const HF = {
  base: 'https://huggingface.co',
  headers() { const t = store.get('hftoken', ''); return t ? { Authorization: 'Bearer ' + t } : {}; },
  resolve(repo, file) { return `${this.base}/${repo}/resolve/main/${file.split('/').map(encodeURIComponent).join('/')}?download=true`; },
  async json(url) {
    let r; try { r = await fetch(url, { headers: this.headers() }); } catch (e) { throw new Error(NET_MSG); }
    if (!r.ok) throw new Error(r.status === 401 || r.status === 403 ? 'This repo needs a Hugging Face token (Settings in Models).' : 'Hugging Face answered HTTP ' + r.status);
    return r.json();
  },
  async search(q, sort = 'downloads') {
    const u = `${this.base}/api/models?filter=gguf&limit=24&sort=${sort}&direction=-1&expand%5B%5D=gguf&expand%5B%5D=downloads&expand%5B%5D=likes` + (q ? '&search=' + encodeURIComponent(q) : '');
    return (await this.json(u)).map(m => ({ id: m.id, downloads: m.downloads || 0, likes: m.likes || 0, arch: m.gguf && m.gguf.architecture, total: m.gguf && m.gguf.total, ctx: m.gguf && m.gguf.context_length }));
  },
  /* .gguf files of a repo (one level of sub folders for big repos) */
  async files(repo) {
    const out = [];
    const walk = async (path, depth) => {
      const list = await this.json(`${this.base}/api/models/${repo}/tree/main${path ? '/' + path : ''}`);
      for (const f of list) {
        if (f.type === 'file' && /\.gguf$/i.test(f.path)) out.push({ path: f.path, size: (f.lfs && f.lfs.size) || f.size || 0 });
        else if (f.type === 'directory' && depth < 1) await walk(f.path, depth + 1);
      }
    };
    await walk('', 0);
    return out.sort((a, b) => a.size - b.size);
  },
};
/* Catalog. status: ready = verified in-page; native = needs Termux (too big for a page) ; planned = architecture work needed; no = not possible on a phone */
const CATALOG = [
  { id: 'smol135', name: 'SmolLM2 135M', kind: 'Instant demo', repo: 'bartowski/SmolLM2-135M-Instruct-GGUF', file: 'SmolLM2-135M-Instruct-Q8_0.gguf', size: 138e6, params: '135M', status: 'ready', blurb: 'Tiny and fast. Proves the whole pipeline works in seconds. Not smart.' },
  { id: 'smol360', name: 'SmolLM2 360M', kind: 'Small', repo: 'bartowski/SmolLM2-360M-Instruct-GGUF', file: 'SmolLM2-360M-Instruct-Q4_K_M.gguf', size: 258e6, params: '360M', status: 'ready', blurb: 'Small chat model with basic answers.' },
  { id: 'qw05', name: 'Qwen2.5 0.5B Instruct', kind: 'Small', repo: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF', file: 'qwen2.5-0.5b-instruct-q4_k_m.gguf', size: 468e6, params: '0.5B', status: 'ready', blurb: 'Good little assistant, multilingual.' },
  { id: 'qw3-06', name: 'Qwen3 0.6B', kind: 'Small', repo: 'Qwen/Qwen3-0.6B-GGUF', file: 'Qwen3-0.6B-Q8_0.gguf', size: 609e6, params: '0.6B', status: 'ready', blurb: 'Newer Qwen with a "thinking" mode.' },
  { id: 'll32-1', name: 'Llama 3.2 1B Instruct', kind: 'Small', repo: 'bartowski/Llama-3.2-1B-Instruct-GGUF', file: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf', size: 770e6, params: '1.2B', status: 'ready', blurb: 'Meta\'s small model.' },
  { id: 'qw3-17', name: 'Qwen3 1.7B', kind: 'Medium', repo: 'Qwen/Qwen3-1.7B-GGUF', file: 'Qwen3-1.7B-Q8_0.gguf', size: 1749e6, params: '1.7B', status: 'ready', blurb: 'Best quality that still fits a page. Slower in the browser; fast in Termux.' },
  { id: 'olmoe', name: 'OLMoE 1B-7B Instruct', kind: 'MoE', repo: 'bartowski/OLMoE-1B-7B-0924-Instruct-GGUF', file: 'OLMoE-1B-7B-0924-Instruct-Q2_K.gguf', size: 2444e6, params: '6.9B (1.3B active)', status: 'ready', blurb: 'The MoE showcase: 64 experts, 8 per word, streamed from storage. Verified against a float64 reference.' },
  { id: 'qw3-30', name: 'Qwen3 30B-A3B', kind: 'MoE', repo: 'unsloth/Qwen3-30B-A3B-GGUF', file: 'Qwen3-30B-A3B-Q2_K.gguf', size: 10737e6, params: '30.5B (3B active)', status: 'ready', blurb: 'A real big MoE: 128 experts. Streams from flash. Slow but it answers; best in Termux.' },
  { id: 'mixtral', name: 'Mixtral 8x7B', kind: 'MoE', repo: 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF', status: 'planned', size: 15.6e9, params: '46.7B', blurb: 'Uses a SentencePiece tokenizer in this build, which the engine does not support yet.' },
  { id: 'qw3-235', name: 'Qwen3 235B-A22B', kind: 'Huge MoE', repo: 'unsloth/Qwen3-235B-A22B-GGUF', status: 'native', size: 80e9, params: '235B (22B active)', blurb: 'About 80 GB at 2-bit: fits your 151 GB flash. Files are split, so it needs the Termux importer. Expect seconds per word.' },
  { id: 'dsv3', name: 'DeepSeek V3 / R1', kind: 'Huge MoE', repo: 'unsloth/DeepSeek-R1-GGUF', status: 'planned', size: 200e9, params: '671B', blurb: 'Needs MLA attention, which is not implemented yet.' },
  { id: 'k2', name: 'Kimi K2', kind: 'Huge MoE', repo: 'unsloth/Kimi-K2-Instruct-GGUF', status: 'planned', size: 340e9, params: '1T', blurb: 'MLA attention again, and 340 GB is more than your flash.' },
  { id: 'gptoss', name: 'gpt-oss 120B', kind: 'Huge MoE', repo: 'ggml-org/gpt-oss-120b-GGUF', status: 'planned', size: 63e9, params: '117B', blurb: 'Needs the MXFP4 format and attention sinks, not implemented yet.' },
  { id: '2t', name: '2 trillion parameters', kind: 'Hypothetical', status: 'no', size: 500e9, params: '2T', blurb: 'No such open model exists today, and at 2 bits it would be about 500 GB, over three times your flash. See the capacity planner below.' },
];
const STATUS = {
  ready: ['good', 'runs here'], try: ['warn', 'may not load'], native: ['warn', 'Termux only'], planned: ['bad', 'engine work needed'], no: ['bad', 'does not exist / does not fit'],
};
