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
/* Catalog. status: ready = verified in-page; stream = native engine with experts downloaded on demand (core = what es_get downloads up front, measured with es_get --plan); native = needs Termux; planned = architecture work needed */
const CATALOG = [
  { id: 'smol135q4', name: 'SmolLM2 135M Q4_0', kind: 'Fast demo', repo: 'bartowski/SmolLM2-135M-Instruct-GGUF', file: 'SmolLM2-135M-Instruct-Q4_0.gguf', size: 92e6, params: '135M', status: 'ready', blurb: 'Smallest 4-bit file here: 92 MB. Over 100 tok/s with the native engine on a 4-core PC (measured), about 25 tok/s in the browser.' },
  { id: 'smol135', name: 'SmolLM2 135M', kind: 'Instant demo', repo: 'bartowski/SmolLM2-135M-Instruct-GGUF', file: 'SmolLM2-135M-Instruct-Q8_0.gguf', size: 138e6, params: '135M', status: 'ready', blurb: 'Tiny and fast. Proves the whole pipeline works in seconds. Not smart.' },
  { id: 'smol360', name: 'SmolLM2 360M', kind: 'Small', repo: 'bartowski/SmolLM2-360M-Instruct-GGUF', file: 'SmolLM2-360M-Instruct-Q4_K_M.gguf', size: 258e6, params: '360M', status: 'ready', blurb: 'Small chat model with basic answers.' },
  { id: 'qw05', name: 'Qwen2.5 0.5B Instruct', kind: 'Small', repo: 'Qwen/Qwen2.5-0.5B-Instruct-GGUF', file: 'qwen2.5-0.5b-instruct-q4_k_m.gguf', size: 468e6, params: '0.5B', status: 'ready', blurb: 'Good little assistant, multilingual.' },
  { id: 'qw3-06', name: 'Qwen3 0.6B', kind: 'Small', repo: 'Qwen/Qwen3-0.6B-GGUF', file: 'Qwen3-0.6B-Q8_0.gguf', size: 609e6, params: '0.6B', status: 'ready', blurb: 'Newer Qwen with a "thinking" mode.' },
  { id: 'll32-1', name: 'Llama 3.2 1B Instruct', kind: 'Small', repo: 'bartowski/Llama-3.2-1B-Instruct-GGUF', file: 'Llama-3.2-1B-Instruct-Q4_K_M.gguf', size: 770e6, params: '1.2B', status: 'ready', blurb: 'Meta\'s small model.' },
  { id: 'qw3-17', name: 'Qwen3 1.7B', kind: 'Medium', repo: 'Qwen/Qwen3-1.7B-GGUF', file: 'Qwen3-1.7B-Q8_0.gguf', size: 1749e6, params: '1.7B', status: 'ready', blurb: 'Best quality that still fits a page. Slower in the browser; fast in Termux.' },
  { id: 'olmoe', name: 'OLMoE 1B-7B Instruct', kind: 'MoE', repo: 'bartowski/OLMoE-1B-7B-0924-Instruct-GGUF', file: 'OLMoE-1B-7B-0924-Instruct-Q2_K.gguf', size: 2444e6, params: '6.9B (1.3B active)', status: 'ready', blurb: 'The MoE showcase: 64 experts, 8 per word, streamed from storage. Verified against a float64 reference.' },
  { id: 'qw3-30', name: 'Qwen3 30B-A3B', kind: 'MoE', repo: 'unsloth/Qwen3-30B-A3B-GGUF', file: 'Qwen3-30B-A3B-Q2_K.gguf', size: 10737e6, params: '30.5B (3B active)', status: 'ready', blurb: 'A real big MoE: 128 experts. Streams from flash. Slow but it answers; best in Termux.' },
  { id: 'gptoss20', name: 'gpt-oss 20B', kind: 'MoE', repo: 'ggml-org/gpt-oss-20b-GGUF', file: 'gpt-oss-20b-MXFP4.gguf', size: 12.1e9, core: 1.8e9, params: '21B (3.6B active)', status: 'stream', blurb: 'OpenAI\'s open model. Tested end to end: started from a 1.8 GB download and answered correctly while fetching its experts.' },
  { id: 'mixtral', name: 'Mixtral 8x7B', kind: 'MoE', repo: 'TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF', file: 'mixtral-8x7b-instruct-v0.1.Q2_K.gguf', size: 15.6e9, core: 0.8e9, params: '46.7B (13B active)', status: 'stream', blurb: 'SentencePiece tokenizer and routing verified against PyTorch. 0.8 GB to start.' },
  { id: 'gptoss', name: 'gpt-oss 120B', kind: 'Huge MoE', repo: 'ggml-org/gpt-oss-120b-GGUF', file: 'gpt-oss-120b-MXFP4.gguf', size: 63.4e9, core: 2.3e9, params: '117B (5.1B active)', status: 'stream', blurb: 'The bigger gpt-oss: same format and code path as the 20B, which was tested end to end.' },
  { id: 'qw3-235', name: 'Qwen3 235B-A22B 2507', kind: 'Huge MoE', repo: 'unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF', file: 'Q2_K/Qwen3-235B-A22B-Instruct-2507-Q2_K-00001-of-00002.gguf', size: 85.7e9, core: 3.2e9, params: '235B (22B active)', status: 'stream', blurb: '3.2 GB to start. Expect seconds per word on a phone.' },
  { id: 'q35-397', name: 'Qwen3.5 397B-A17B', kind: 'Huge MoE', repo: 'unsloth/Qwen3.5-397B-A17B-GGUF', file: 'UD-IQ1_M/Qwen3.5-397B-A17B-UD-IQ1_M-00001-of-00004.gguf', size: 106.8e9, core: 6.7e9, params: '397B (17B active)', status: 'stream', blurb: 'Gated DeltaNet hybrid, verified against PyTorch. 6.7 GB to start.' },
  { id: 'dsr1', name: 'DeepSeek R1 0528', kind: 'Huge MoE', repo: 'unsloth/DeepSeek-R1-0528-GGUF', file: 'DeepSeek-R1-0528-UD-TQ1_0.gguf', size: 161.6e9, core: 8.9e9, params: '671B (37B active)', status: 'stream', blurb: 'MLA attention verified against PyTorch. 8.9 GB to start; bigger than your flash, so the least used experts are dropped to stay within it.' },
  { id: 'ml3', name: 'Mistral Large 3', kind: 'Huge MoE', repo: 'unsloth/Mistral-Large-3-675B-Instruct-2512-GGUF', file: 'Mistral-Large-3-675B-Instruct-2512-UD-TQ1_0.gguf', size: 168.6e9, core: 11.2e9, params: '675B (41B active)', status: 'stream', blurb: 'Mistral\'s biggest open model, built like DeepSeek V3 (MLA attention). Verified against PyTorch on a small model of the same shape.' },
  { id: 'k2', name: 'Kimi K2 0905', kind: 'Huge MoE', repo: 'unsloth/Kimi-K2-Instruct-0905-GGUF', file: 'UD-TQ1_0/Kimi-K2-Instruct-0905-UD-TQ1_0-00001-of-00006.gguf', size: 248.6e9, core: 7.1e9, params: '1T (32B active)', status: 'stream', blurb: 'One trillion parameters from a 7.1 GB download. The 249 GB of experts never all live on the phone.' },
  { id: 'q38', name: 'Qwen3.8 2.4T-A95B', kind: 'Huge MoE', repo: 'unsloth/Qwen3.8-2.4T-A95B-GGUF', file: 'UD-Q1_0/Qwen3.8-2.4T-A95B-UD-Q1_0-00001-of-00010.gguf', size: 397.3e9, core: 34.3e9, params: '2.4T (95B active)', status: 'stream', blurb: 'Over 2 trillion parameters. Its formats, architecture and chat template are supported, but the core alone is 34 GB and is read for every word: expect a minute or more per word on a phone. Not run end to end here.' },
  { id: 'k3', name: 'Kimi K3', kind: 'Huge MoE', repo: 'unsloth/Kimi-K3-GGUF', size: 466e9, params: '2.8T (104B active)', status: 'planned', blurb: 'The largest open model. New architecture (KDA linear attention, latent MoE, cross-layer residual attention): engine work needed.' },
];
const STATUS = {
  ready: ['good', 'runs here'], stream: ['good', 'experts on demand'], try: ['warn', 'may not load'], native: ['warn', 'Termux only'], planned: ['bad', 'engine work needed'], no: ['bad', 'does not exist / does not fit'],
};
