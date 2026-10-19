#!/usr/bin/env python3
"""200 models through the engine: tiny copies of every supported design, real small models, real giant models.

    make es_chat es_get && python3 scripts/test_200.py --llama-cpp ./llama.cpp [--tier mini|small|big|all] [--work DIR] [--resume]

Three tiers, each checked against something independent:
  mini   91 small random models, 7 per design (Llama, Mistral, Qwen2, Qwen3, Qwen2-MoE, Qwen3-MoE, Mixtral, OLMoE, gpt-oss,
         DeepSeek V2, DeepSeek V3, Qwen3.5 dense and MoE), with shapes, attention layouts, expert counts, routing and rope
         settings drawn from what real models use. Built with Hugging Face transformers, converted with llama.cpp's
         convert_hf_to_gguf.py, run by es_chat: every logit must match PyTorch (float32, exact activations).
  small  real GGUF files (scripts/models_200.json), many quantizations, downloaded whole: the engine's logits on a real text
         are compared with llama.cpp's (the same file, the same token ids). Also the engine's perplexity on that text.
  big    real giant GGUF files, 14B to 2.4T parameters: only the headers are downloaded (es_get --skeleton, every weight reads
         as zero). The engine must load the real structure (all parts, tensors, formats, hyperparameters) and run a forward
         pass. This checks support, not answer quality: the weights are not there.
Needs: pip install torch transformers safetensors gguf sentencepiece numpy; a llama.cpp checkout built with cmake (for
convert_hf_to_gguf.py, llama-tokenize and libllama). Results: WORK/results.jsonl and WORK/report.md.
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--llama-cpp', required=True)
ap.add_argument('--work', default='/tmp/es_200')
ap.add_argument('--tier', default='all', choices=['mini', 'small', 'big', 'all'])
ap.add_argument('--resume', action='store_true', help='skip models already in WORK/results.jsonl')
ap.add_argument('--only', default='', help='run only names containing this text')
args = ap.parse_args()
import numpy as np

W = os.path.abspath(args.work)
LC = os.path.abspath(args.llama_cpp)
os.makedirs(W, exist_ok=True)
RES = os.path.join(W, 'results.jsonl')
done = set()
if args.resume and os.path.exists(RES):
    done = {json.loads(l)['name'] for l in open(RES) if l.strip()}
CHAT, GET = os.path.join(ROOT, 'es_chat'), os.path.join(ROOT, 'es_get')


def record(tier, name, ok, info, **kv):
    r = dict(tier=tier, name=name, ok=bool(ok), info=info, **kv)
    with open(RES, 'a') as f: f.write(json.dumps(r) + '\n')
    print('%s %-66s %s' % ('PASS' if ok else 'FAIL', name[:66], info), flush=True)


def want(name): return (not args.only or args.only.lower() in name.lower()) and name not in done


def engine_logits(gguf, ids, extra_env=None, opts=()):
    out = os.path.join(W, 'es_out.bin')
    if os.path.exists(out): os.remove(out)
    env = dict(os.environ, **(extra_env or {}))
    t0 = time.time()
    r = subprocess.run([CHAT, '-g', gguf, '-I', ','.join(map(str, ids)), '-D', out, *opts], env=env, capture_output=True, text=True)
    if r.returncode or not os.path.exists(out): return None, r.stderr[-600:], time.time() - t0
    a = np.fromfile(out, dtype=np.float32)
    return a.reshape(len(ids), -1), r.stderr, time.time() - t0


# ------------------------------------------------------------------------------------------------ mini: copies of each design
TOK = {'llama': ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', ['tokenizer.model', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'], 32000),
       'qwen': ('Qwen/Qwen2.5-0.5B-Instruct', ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'], 151936),
       'olmoe': ('allenai/OLMoE-1B-7B-0924', ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'], 50304),
       'gptoss': ('openai/gpt-oss-20b', ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja'], 201088),
       'deepseek': ('deepseek-ai/DeepSeek-V3', ['tokenizer.json', 'tokenizer_config.json'], 129280),
       'qwen35': ('Qwen/Qwen3.6-35B-A3B', ['tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt'], 248320)}


def tokdir(kind):
    repo, files, _ = TOK[kind]
    d = os.path.join(W, 'tok_' + kind)
    os.makedirs(d, exist_ok=True)
    for f in files:
        p = os.path.join(d, f)
        if not os.path.exists(p): urllib.request.urlretrieve('https://huggingface.co/%s/resolve/main/%s' % (repo, f), p)
    return d


def heads(r, dims=(16, 32, 64), hs=(2, 4, 8)):
    h = r.choice(hs); hd = r.choice(dims)
    kv = r.choice([k for k in (h, h // 2, h // 4, 1) if k >= 1 and h % k == 0])
    return h, kv, hd


def rope_llama(r):
    return r.choice([None, {'rope_type': 'llama3', 'factor': 8.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192},
                     {'rope_type': 'linear', 'factor': 2.0}])


def fam_llama(r):
    h, kv, hd = heads(r)
    return 'LlamaForCausalLM', 'LlamaConfig', 'llama', dict(hidden_size=h * hd, intermediate_size=r.choice([64, 128, 256, 384]), num_hidden_layers=r.randint(1, 4),
        num_attention_heads=h, num_key_value_heads=kv, rope_theta=r.choice([1e4, 5e5]), rope_scaling=rope_llama(r), max_position_embeddings=131072,
        rms_norm_eps=r.choice([1e-5, 1e-6]), tie_word_embeddings=r.random() < 0.4), {}


def fam_mistral(r):
    h, kv, hd = heads(r, dims=(32, 64))
    return 'MistralForCausalLM', 'MistralConfig', 'llama', dict(hidden_size=r.choice([64, 128, 192]), head_dim=hd, intermediate_size=r.choice([128, 256]),
        num_hidden_layers=r.randint(1, 4), num_attention_heads=h, num_key_value_heads=kv, rope_theta=r.choice([1e6, 1e8]), sliding_window=None,
        tie_word_embeddings=False), {}


def fam_qwen2(r):
    h, kv, hd = heads(r)
    return 'Qwen2ForCausalLM', 'Qwen2Config', 'qwen', dict(hidden_size=h * hd, intermediate_size=r.choice([96, 192, 256]), num_hidden_layers=r.randint(1, 4),
        num_attention_heads=h, num_key_value_heads=kv, rope_theta=r.choice([1e4, 1e6]), max_position_embeddings=32768, tie_word_embeddings=r.random() < 0.5,
        use_sliding_window=False), {}


def fam_qwen3(r):
    h, kv, hd = heads(r, dims=(32, 64, 128))
    return 'Qwen3ForCausalLM', 'Qwen3Config', 'qwen', dict(hidden_size=r.choice([64, 128, 256]), head_dim=hd, intermediate_size=r.choice([128, 256, 384]),
        num_hidden_layers=r.randint(1, 4), num_attention_heads=h, num_key_value_heads=kv, rope_theta=r.choice([1e6, 5e6]), max_position_embeddings=40960,
        tie_word_embeddings=r.random() < 0.5, use_sliding_window=False), {}


def fam_qwen2moe(r):
    h, kv, hd = heads(r)
    ne = r.choice([4, 8, 16])
    return 'Qwen2MoeForCausalLM', 'Qwen2MoeConfig', 'qwen', dict(hidden_size=h * hd, intermediate_size=192, moe_intermediate_size=r.choice([32, 64]),
        shared_expert_intermediate_size=r.choice([64, 128]), num_experts=ne, num_experts_per_tok=r.choice([1, 2, 4]), norm_topk_prob=False,
        num_hidden_layers=r.randint(1, 3), num_attention_heads=h, num_key_value_heads=kv, rope_theta=1e6, decoder_sparse_step=1, mlp_only_layers=[],
        tie_word_embeddings=False, use_sliding_window=False), {}


def fam_qwen3moe(r):
    h, kv, hd = heads(r, dims=(32, 64))
    ne = r.choice([8, 16, 32])
    return 'Qwen3MoeForCausalLM', 'Qwen3MoeConfig', 'qwen', dict(hidden_size=r.choice([64, 128]), head_dim=hd, moe_intermediate_size=r.choice([32, 64]),
        num_experts=ne, num_experts_per_tok=r.choice([2, 4, 8]), norm_topk_prob=True, num_hidden_layers=r.randint(1, 3), num_attention_heads=h,
        num_key_value_heads=kv, rope_theta=1e6, decoder_sparse_step=1, mlp_only_layers=[], tie_word_embeddings=False, max_position_embeddings=40960), {}


def fam_mixtral(r):
    h, kv, hd = heads(r)
    return 'MixtralForCausalLM', 'MixtralConfig', 'llama', dict(hidden_size=h * hd, intermediate_size=r.choice([64, 128]), num_local_experts=r.choice([4, 8]),
        num_experts_per_tok=r.choice([1, 2]), num_hidden_layers=r.randint(1, 3), num_attention_heads=h, num_key_value_heads=kv,
        rope_theta=r.choice([1e6, 1e4]), tie_word_embeddings=False), {}


def fam_olmoe(r):
    h, kv, hd = heads(r, hs=(2, 4))
    return 'OlmoeForCausalLM', 'OlmoeConfig', 'olmoe', dict(hidden_size=h * hd, intermediate_size=r.choice([32, 64]), num_experts=r.choice([8, 16]),
        num_experts_per_tok=r.choice([2, 4, 8]), norm_topk_prob=False, num_hidden_layers=r.randint(1, 3), num_attention_heads=h, num_key_value_heads=h,
        rope_theta=1e4, tie_word_embeddings=False, clip_qkv=None), {}


def fam_gptoss(r):
    h, kv, hd = heads(r, dims=(32, 64), hs=(4, 8))
    nl = r.choice([2, 4])
    return 'GptOssForCausalLM', 'GptOssConfig', 'gptoss', dict(hidden_size=r.choice([64, 96, 128]), head_dim=hd, intermediate_size=r.choice([32, 64]),
        num_local_experts=r.choice([4, 8, 16]), num_experts_per_tok=r.choice([1, 2, 4]), num_hidden_layers=nl, num_attention_heads=h, num_key_value_heads=kv,
        sliding_window=r.choice([2, 4, 8]), layer_types=['sliding_attention', 'full_attention'] * (nl // 2), max_position_embeddings=131072, rope_theta=150000.0,
        rope_scaling={'rope_type': 'yarn', 'factor': 32.0, 'beta_fast': 32.0, 'beta_slow': 1.0, 'truncate': False, 'original_max_position_embeddings': 4096},
        tie_word_embeddings=False), {}


def yarn_ds(r):
    return r.choice([None, {'type': 'yarn', 'rope_type': 'yarn', 'factor': 40.0, 'beta_fast': 32, 'beta_slow': 1, 'mscale': 1.0, 'mscale_all_dim': 1.0,
                            'original_max_position_embeddings': 4096}])


def fam_dsv3(r):
    ne = r.choice([8, 16, 32]); h = r.choice([2, 4]); ng = r.choice([g for g in (1, 2, 4, 8) if ne // g >= 2]); tg = r.randint(1, ng)
    k = r.choice([x for x in (1, 2, 4, 8) if x <= tg * (ne // ng)])
    return 'DeepseekV3ForCausalLM', 'DeepseekV3Config', 'deepseek', dict(hidden_size=r.choice([64, 128]), intermediate_size=192, moe_intermediate_size=r.choice([32, 64]),
        num_hidden_layers=r.randint(2, 3), num_attention_heads=h, num_key_value_heads=h,   # MLA: one latent, as many k/v heads as q heads
        n_shared_experts=r.choice([1, 2]), n_routed_experts=ne,
        num_experts_per_tok=k, n_group=ng, topk_group=tg, routed_scaling_factor=r.choice([1.0, 2.5]), first_k_dense_replace=r.choice([0, 1]),
        norm_topk_prob=True, q_lora_rank=r.choice([32, 64]), kv_lora_rank=r.choice([16, 32]), qk_nope_head_dim=r.choice([16, 32]), qk_rope_head_dim=16,
        v_head_dim=r.choice([16, 32]), max_position_embeddings=163840, rope_theta=10000.0, rope_scaling=yarn_ds(r), tie_word_embeddings=False,
        bos_token_id=0, eos_token_id=1), {'scoring_func': 'sigmoid', 'topk_method': 'noaux_tc'}


def fam_dsv2(r):
    ne = r.choice([8, 16]); h = r.choice([2, 4])
    return 'DeepseekV2ForCausalLM', 'DeepseekV2Config', 'deepseek', dict(hidden_size=r.choice([64, 128]), intermediate_size=192, moe_intermediate_size=r.choice([32, 64]),
        num_hidden_layers=r.randint(2, 3), num_attention_heads=h, num_key_value_heads=h,   # MLA: one latent, as many k/v heads as q heads
        n_shared_experts=r.choice([1, 2]), n_routed_experts=ne,
        num_experts_per_tok=r.choice([2, 4, 6]), n_group=1, topk_group=1, topk_method='greedy', routed_scaling_factor=1.0, first_k_dense_replace=1,
        norm_topk_prob=False, q_lora_rank=r.choice([None, 32]), kv_lora_rank=r.choice([16, 32]), qk_nope_head_dim=r.choice([16, 32]), qk_rope_head_dim=16,
        v_head_dim=r.choice([16, 32]), max_position_embeddings=163840, rope_theta=10000.0, rope_scaling=yarn_ds(r), tie_word_embeddings=False,
        bos_token_id=0, eos_token_id=1), {'scoring_func': 'softmax'}


def fam_qwen35moe(r):
    nl = r.choice([4, 8])
    return 'Qwen3_5MoeForCausalLM', 'Qwen3_5MoeTextConfig', 'qwen35', dict(hidden_size=r.choice([64, 128]), num_hidden_layers=nl, num_attention_heads=r.choice([2, 4]),
        num_key_value_heads=r.choice([1, 2]), head_dim=r.choice([32, 64]), linear_conv_kernel_dim=4, linear_key_head_dim=r.choice([16, 32]),
        linear_value_head_dim=r.choice([16, 32]), linear_num_key_heads=2, linear_num_value_heads=r.choice([2, 4]), moe_intermediate_size=r.choice([32, 64]),
        shared_expert_intermediate_size=r.choice([32, 64]), num_experts_per_tok=r.choice([2, 4]), num_experts=r.choice([8, 16]), max_position_embeddings=32768,
        rope_parameters={'rope_theta': 10000000.0, 'partial_rotary_factor': 0.25, 'rope_type': 'default'}, tie_word_embeddings=False), {}


def fam_qwen35(r):
    nl = r.choice([4, 8])
    return 'Qwen3_5ForCausalLM', 'Qwen3_5TextConfig', 'qwen35', dict(hidden_size=r.choice([64, 128]), intermediate_size=r.choice([128, 256]), num_hidden_layers=nl,
        num_attention_heads=r.choice([2, 4]), num_key_value_heads=r.choice([1, 2]), head_dim=r.choice([32, 64]), linear_conv_kernel_dim=4,
        linear_key_head_dim=r.choice([16, 32]), linear_value_head_dim=r.choice([16, 32]), linear_num_key_heads=2, linear_num_value_heads=r.choice([2, 4]),
        max_position_embeddings=32768, rope_parameters={'rope_theta': 10000000.0, 'partial_rotary_factor': 0.25, 'rope_type': 'default'},
        tie_word_embeddings=r.random() < 0.5), {}


FAMS = [('llama', fam_llama), ('mistral', fam_mistral), ('qwen2', fam_qwen2), ('qwen3', fam_qwen3), ('qwen2moe', fam_qwen2moe), ('qwen3moe', fam_qwen3moe),
        ('mixtral', fam_mixtral), ('olmoe', fam_olmoe), ('gptoss', fam_gptoss), ('dsv2', fam_dsv2), ('dsv3', fam_dsv3), ('qwen35moe', fam_qwen35moe),
        ('qwen35', fam_qwen35)]


def short(cfg):
    keys = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'num_key_value_heads', 'head_dim', 'num_experts', 'num_local_experts', 'n_routed_experts',
            'num_experts_per_tok', 'n_group', 'topk_group', 'q_lora_rank', 'sliding_window', 'tie_word_embeddings']
    ab = {'num_hidden_layers': 'L', 'hidden_size': 'd', 'num_attention_heads': 'h', 'num_key_value_heads': 'kv', 'head_dim': 'hd', 'num_experts': 'E',
          'num_local_experts': 'E', 'n_routed_experts': 'E', 'num_experts_per_tok': 'k', 'n_group': 'grp', 'topk_group': 'tg', 'q_lora_rank': 'qlora',
          'sliding_window': 'swa', 'tie_word_embeddings': 'tie'}
    s = ' '.join('%s%s' % (ab[k], cfg[k]) for k in keys if k in cfg and cfg[k] not in (None, False))
    rs = cfg.get('rope_scaling')
    if rs: s += ' rope-' + rs.get('rope_type', rs.get('type'))
    return s


def run_mini():
    import torch
    import transformers as T
    for fi, (fam, fn) in enumerate(FAMS):
        for v in range(7):
            r = random.Random(1000 * fi + v)
            cls, cfgcls, tok, cfg, extra = fn(r)
            name = 'mini %-9s #%d  %s' % (fam, v + 1, short(cfg))
            if not want(name): continue
            out = os.path.join(W, 'mini')
            shutil.rmtree(out, ignore_errors=True)
            try:
                vocab = TOK[tok][2]
                torch.manual_seed(fi * 100 + v)
                model = getattr(T, cls)(getattr(T, cfgcls)(vocab_size=vocab, **cfg)).float()
                with torch.no_grad():   # larger weights than the default init, so logits differ clearly between tokens
                    for n, p in model.named_parameters():
                        if p.dim() >= 2: p.normal_(0, 0.08)
                        elif 'norm' in n: p.fill_(1.0).add_(torch.randn_like(p) * 0.1)
                        elif 'A_log' in n: p.uniform_(0, 2)
                        else: p.normal_(0, 0.3)
                    if getattr(model.config, 'tie_word_embeddings', False): model.tie_weights()
                model.save_pretrained(out)
                if extra:
                    p = os.path.join(out, 'config.json'); c = json.load(open(p)); c.update(extra); json.dump(c, open(p, 'w'))
                td = tokdir(tok)
                for f in os.listdir(td): shutil.copy(os.path.join(td, f), out)
                gguf = out + '-f32.gguf'
                cargs = ['--no-mtp'] if fam.startswith('qwen35') else []
                rc = subprocess.run([sys.executable, os.path.join(LC, 'convert_hf_to_gguf.py'), out, '--outtype', 'f32', '--outfile', gguf, *cargs], capture_output=True, text=True)
                if rc.returncode:
                    record('mini', name, False, 'llama.cpp converter failed: ' + rc.stderr.strip().splitlines()[-1][:150]); continue
                ids = r.sample(range(10, 30000), 24)
                a, err, _ = engine_logits(gguf, ids, {'ES_FLOAT_ACT': '1'}, ('-q', '-j', '2'))
                c = T.AutoConfig.from_pretrained(out)
                if getattr(c, 'rope_scaling', None) is None and getattr(c, 'rope_parameters', None) and c.model_type in ('deepseek_v3', 'deepseek_v2'):
                    c.rope_scaling = dict(c.rope_parameters)
                m = T.AutoModelForCausalLM.from_pretrained(out, config=c, dtype=torch.float32).eval()
                with torch.no_grad(): b = m(torch.tensor([ids])).logits[0].float().numpy()
                if a is None:
                    record('mini', name, False, 'engine failed: ' + err.strip().splitlines()[-1][:150] if err.strip() else 'engine failed'); continue
                b = b[:, :a.shape[1]] if b.shape[1] >= a.shape[1] else b
                d = float(np.abs(a - b).max()); scale = float(np.abs(b).max())
                ok = a.shape == b.shape and d <= 1e-3 * max(1.0, scale) and (a.argmax(1) == b.argmax(1)).all()
                record('mini', name, ok, 'max |logit difference| %.2e (logits up to %.1f), %d positions' % (d, scale, len(ids)), diff=d, scale=scale)
            except Exception as e:
                record('mini', name, False, 'error: %s' % str(e).splitlines()[0][:160])
            finally:
                shutil.rmtree(out, ignore_errors=True)
                for f in (out + '-f32.gguf',):
                    if os.path.exists(f): os.remove(f)


# ------------------------------------------------------------------------------------------------ small: real models vs llama.cpp
TEXT = ('The old lighthouse keeper climbed the spiral stairs every evening at sunset. He lit the great lamp, wrote the weather in '
        'his logbook, and watched the ships pass safely around the rocky point. In 1912 the light was replaced by an electric one, '
        'but he kept climbing the stairs anyway, because the sea had become his oldest friend.')


def ref_tool():
    exe = os.path.join(W, 'ref_llama_logits')
    if not os.path.exists(exe):
        b = os.path.join(LC, 'build', 'bin')
        r = subprocess.run(['g++', '-O2', '-std=c++17', os.path.join(ROOT, 'tools', 'ref_llama_logits.cpp'), '-I', os.path.join(LC, 'include'),
                            '-I', os.path.join(LC, 'ggml', 'include'), '-L', b, '-lllama', '-lggml', '-lggml-base', '-Wl,-rpath,' + b, '-o', exe],
                           capture_output=True, text=True)
        if r.returncode: sys.exit('cannot build the llama.cpp reference: ' + r.stderr[-800:])
    return exe


def run_small(models):
    ref = ref_tool()
    tokz = os.path.join(LC, 'build', 'bin', 'llama-tokenize')
    dl = os.path.join(W, 'dl')
    for m in models:
        name = 'small %s (%s, %.2f GB)' % (os.path.basename(m['file']), m['repo'].split('/')[0], m['gb'])
        if not want(name): continue
        shutil.rmtree(dl, ignore_errors=True); os.makedirs(dl)
        try:
            t0 = time.time()
            r = subprocess.run([GET, m['repo'], m['file'], '-d', dl, '--all'], capture_output=True, text=True)
            if r.returncode: record('small', name, False, 'download failed: ' + r.stderr.strip().splitlines()[-1][:150]); continue
            path = os.path.join(dl, os.path.basename(m['file']))
            dl_s = time.time() - t0
            r = subprocess.run([tokz, '-m', path, '-p', TEXT, '--ids', '--log-disable'], capture_output=True, text=True)
            line = [l for l in r.stdout.splitlines() if l.startswith('[')]
            if not line:
                record('small', name, False, 'llama.cpp could not tokenize: ' + (r.stderr.strip().splitlines() or ['?'])[-1][:150]); continue
            ids = json.loads(line[-1])[:64]
            a, err, es_s = engine_logits(path, ids, None, ('-q',))
            if a is None: record('small', name, False, 'engine failed: ' + (err.strip().splitlines() or ['?'])[-1][:150]); continue
            nll = [float(np.log(np.exp(a[i] - a[i].max()).sum()) + a[i].max() - a[i][ids[i + 1]]) for i in range(len(ids) - 1)]
            ppl = math.exp(sum(nll[8:]) / len(nll[8:]))
            out = os.path.join(W, 'ref.bin')
            if os.path.exists(out): os.remove(out)
            rr = subprocess.run([ref, path, out, *map(str, ids)], capture_output=True, text=True)
            if rr.returncode or not os.path.exists(out):
                record('small', name, ppl < 60 and np.isfinite(a).all(), 'no llama.cpp reference (%s); engine perplexity on the text %.1f' %
                       ((rr.stderr.strip().splitlines() or ['?'])[-1][:80], ppl), ppl=ppl); continue
            b = np.fromfile(out, dtype=np.float32).reshape(len(ids), -1)[:, :a.shape[1]]
            def sm(x): e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
            pb, pa = sm(b.astype(np.float64)), sm(a.astype(np.float64))
            kl = float((pb * (np.log(pb + 1e-30) - np.log(pa + 1e-30))).sum(1).mean())
            top = float((a.argmax(1) == b.argmax(1)).mean())
            ok = kl < 0.01 and top >= 0.9 and np.isfinite(a).all()
            record('small', name, ok, 'vs llama.cpp: same top token %d%%, KL %.1e | perplexity %.1f | %d tokens, engine %.0f s, download %.0f s' %
                   (round(100 * top), kl, ppl, len(ids), es_s, dl_s), kl=kl, top1=top, ppl=ppl)
        except Exception as e:
            record('small', name, False, 'error: %s' % str(e).splitlines()[0][:160])
        finally:
            shutil.rmtree(dl, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ big: real giants, structure only
def run_big(models):
    sk = os.path.join(W, 'sk')
    for m in models:
        name = 'big %s (%.0f GB)' % (os.path.basename(m['file']).replace('-00001-of-', ' 1/'), m['gb'])
        if not want(name): continue
        shutil.rmtree(sk, ignore_errors=True); os.makedirs(sk)
        try:
            r = subprocess.run([GET, m['repo'], m['file'], '-d', sk, '--skeleton'], capture_output=True, text=True)
            if r.returncode: record('big', name, False, 'header: ' + (r.stderr.strip().splitlines() or ['?'])[-1][:150]); continue
            path = os.path.join(sk, os.path.basename(m['file']))
            a, err, es_s = engine_logits(path, [1, 2, 3], None, ('-C', '256', '-c', '64'))
            if a is None: record('big', name, False, 'engine: ' + (err.strip().splitlines() or ['?'])[-1][:150]); continue
            info = [l for l in err.splitlines() if 'layers' in l and 'experts' in l or 'layers' in l]
            desc = info[0].split(']', 1)[-1].strip() if info else ''
            desc = desc.replace('\x1b[0m', '').strip()
            ok = np.isfinite(a).all()
            record('big', name, ok, 'loaded and ran: %s | %.0f s' % (desc[:110], es_s), seconds=es_s)
        except Exception as e:
            record('big', name, False, 'error: %s' % str(e).splitlines()[0][:160])
        finally:
            shutil.rmtree(sk, ignore_errors=True)


lists = json.load(open(os.path.join(ROOT, 'scripts', 'models_200.json')))
if args.tier in ('mini', 'all'): run_mini()
if args.tier in ('small', 'all'): run_small(lists['small'])
if args.tier in ('big', 'all'): run_big(lists['big'])

rows = [json.loads(l) for l in open(RES)] if os.path.exists(RES) else []
last = {}
for x in rows: last[x['name']] = x
rows = list(last.values())
with open(os.path.join(W, 'report.md'), 'w') as f:
    for tier in ('mini', 'small', 'big'):
        t = [x for x in rows if x['tier'] == tier]
        if not t: continue
        f.write('\n## %s: %d of %d passed\n\n| | Model | Result |\n|---|---|---|\n' % (tier, sum(x['ok'] for x in t), len(t)))
        for x in t: f.write('| %s | %s | %s |\n' % ('PASS' if x['ok'] else 'FAIL', x['name'].split(' ', 1)[1], x['info'].replace('|', '/')))
n_ok = sum(x['ok'] for x in rows)
print('%d of %d passed (mini %d/%d, small %d/%d, big %d/%d); report: %s' % (n_ok, len(rows),
      *(v for t in ('mini', 'small', 'big') for v in (sum(x['ok'] for x in rows if x['tier'] == t), sum(1 for x in rows if x['tier'] == t))), os.path.join(W, 'report.md')))
sys.exit(0 if n_ok == len(rows) else 1)
