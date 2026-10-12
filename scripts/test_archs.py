#!/usr/bin/env python3
"""Check the engine's forward pass against the original PyTorch models, one small random model per family.

    pip install torch transformers safetensors gguf sentencepiece
    git clone --depth 1 https://github.com/ggml-org/llama.cpp     (only its convert_hf_to_gguf.py is used)
    make es_chat && python3 scripts/test_archs.py --llama-cpp ./llama.cpp [--work /tmp/es_archs]

For each family a small model with random weights is built with Hugging Face transformers, converted to GGUF with
llama.cpp's converter, and run by es_chat (-D: logits of every position). The logits must match PyTorch's
(float32 weights, exact float activations).
Families: gpt-oss (attention sinks, sliding window, YaRN, clamped SwiGLU, expert biases), DeepSeek-V3 (MLA with the absorbed
projections, sigmoid routing with expert groups and a selection bias, shared expert, dense first layer, YaRN), Mixtral
(both the merged and the old one-tensor-per-expert layouts) and Llama 3 (rope_freqs).
Tokenizers are downloaded from Hugging Face (openai/gpt-oss-20b, deepseek-ai/DeepSeek-V3, TinyLlama/TinyLlama-1.1B-Chat-v1.0).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument('--llama-cpp', required=True, help='llama.cpp checkout (for convert_hf_to_gguf.py)')
ap.add_argument('--work', default='/tmp/es_archs')
args = ap.parse_args()
try:
    import numpy as np
    import torch
    from transformers import (AutoConfig, AutoModelForCausalLM, DeepseekV3Config, DeepseekV3ForCausalLM, GptOssConfig,
                              GptOssForCausalLM, LlamaConfig, LlamaForCausalLM, MixtralConfig, MixtralForCausalLM)
except ImportError as e:
    sys.exit('SKIP: needs torch and transformers (%s)' % e)

W = args.work
os.makedirs(W, exist_ok=True)
TOK = {'gptoss': ('openai/gpt-oss-20b', ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'chat_template.jinja']),
       'deepseek': ('deepseek-ai/DeepSeek-V3', ['tokenizer.json', 'tokenizer_config.json']),
       'llama': ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', ['tokenizer.model', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'])}


def tokdir(kind):
    repo, files = TOK[kind]
    d = os.path.join(W, 'tok_' + kind)
    os.makedirs(d, exist_ok=True)
    for f in files:
        p = os.path.join(d, f)
        if not os.path.exists(p):
            urllib.request.urlretrieve('https://huggingface.co/%s/resolve/main/%s' % (repo, f), p)
    return d


def build(name, model, kind, extra_cfg=None):
    out = os.path.join(W, name)
    shutil.rmtree(out, ignore_errors=True)
    torch.manual_seed(0)
    with torch.no_grad():   # larger weights than the default init, so logits differ clearly between tokens
        for n, p in model.named_parameters():
            if p.dim() >= 2: p.normal_(0, 0.08)
            elif 'norm' in n: p.fill_(1.0).add_(torch.randn_like(p) * 0.1)
            else: p.normal_(0, 0.3)
    model.save_pretrained(out)
    if extra_cfg:
        p = os.path.join(out, 'config.json')
        c = json.load(open(p)); c.update(extra_cfg); json.dump(c, open(p, 'w'))
    td = tokdir(kind)
    for f in os.listdir(td): shutil.copy(os.path.join(td, f), out)
    gguf = out + '-f32.gguf'
    r = subprocess.run([sys.executable, os.path.join(args.llama_cpp, 'convert_hf_to_gguf.py'), out, '--outtype', 'f32', '--outfile', gguf], capture_output=True, text=True)
    if r.returncode: sys.exit('convert failed for %s:\n%s' % (name, r.stderr[-2000:]))
    return out, gguf


def torch_logits(d, ids):
    c = AutoConfig.from_pretrained(d)
    if getattr(c, 'rope_scaling', None) is None and getattr(c, 'rope_parameters', None) and c.model_type == 'deepseek_v3':
        c.rope_scaling = dict(c.rope_parameters)   # this modeling code reads rope_scaling
    m = AutoModelForCausalLM.from_pretrained(d, config=c, dtype=torch.float32).eval()
    with torch.no_grad():
        return m(torch.tensor([ids])).logits[0].float().numpy()


def engine_logits(gguf, ids, vocab):
    out = gguf + '.es.bin'
    env = dict(os.environ, ES_FLOAT_ACT='1')
    r = subprocess.run([os.path.join(ROOT, 'es_chat'), '-g', gguf, '-q', '-j', '2', '-I', ','.join(map(str, ids)), '-D', out], env=env, capture_output=True, text=True)
    if r.returncode: sys.exit('es_chat failed on %s: %s' % (gguf, r.stderr[-1000:]))
    return np.fromfile(out, dtype=np.float32).reshape(-1, vocab)


def split_experts(src, dst):   # the old Mixtral layout: blk.L.ffn_gate.E.weight
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    r = GGUFReader(src)
    w = GGUFWriter(dst, r.fields['general.architecture'].contents())
    for k, f in r.fields.items():
        if k.startswith('GGUF.') or k == 'general.architecture': continue
        t = f.types[0]
        if t == GGUFValueType.ARRAY: w.add_key_value(k, f.contents(), t, sub_type=f.types[-1])
        else: w.add_key_value(k, f.contents(), t)
    for t in r.tensors:
        a = np.array(t.data)
        if '_exps.weight' in t.name:
            for e in range(a.shape[0]): w.add_tensor('%s.%d.weight' % (t.name.replace('_exps.weight', ''), e), np.ascontiguousarray(a[e]), raw_dtype=t.tensor_type)
        else: w.add_tensor(t.name, a, raw_dtype=t.tensor_type)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


ids = list(range(1, 21))
cases = []
cases.append(('gpt-oss', *build('gptoss', GptOssForCausalLM(GptOssConfig(
    vocab_size=201088, hidden_size=96, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
    num_local_experts=8, num_experts_per_tok=2, sliding_window=4, max_position_embeddings=131072, rope_theta=150000.0,
    rope_scaling={'rope_type': 'yarn', 'factor': 32.0, 'beta_fast': 32.0, 'beta_slow': 1.0, 'truncate': False, 'original_max_position_embeddings': 4096},
    layer_types=['sliding_attention', 'full_attention'] * 2, tie_word_embeddings=False)).float(), 'gptoss'), 201088))
cases.append(('DeepSeek-V3 (YaRN)', *build('dsv3', DeepseekV3ForCausalLM(DeepseekV3Config(
    vocab_size=129280, hidden_size=128, intermediate_size=192, moe_intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
    num_key_value_heads=4, n_shared_experts=1, n_routed_experts=16, num_experts_per_tok=4, n_group=4, topk_group=2, routed_scaling_factor=2.5,
    first_k_dense_replace=1, norm_topk_prob=True, q_lora_rank=64, kv_lora_rank=32, qk_nope_head_dim=32, qk_rope_head_dim=16, v_head_dim=32,
    max_position_embeddings=163840, rope_theta=10000.0,
    rope_scaling={'type': 'yarn', 'rope_type': 'yarn', 'factor': 40.0, 'beta_fast': 32, 'beta_slow': 1, 'mscale': 1.0, 'mscale_all_dim': 1.0, 'original_max_position_embeddings': 4096},
    tie_word_embeddings=False, bos_token_id=0, eos_token_id=1)).float(), 'deepseek', {'scoring_func': 'sigmoid', 'topk_method': 'noaux_tc'}), 129280))
mix = build('mixtral', MixtralForCausalLM(MixtralConfig(vocab_size=32000, hidden_size=256, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4,
    num_key_value_heads=2, num_local_experts=8, num_experts_per_tok=2, rope_theta=1e6, tie_word_embeddings=False)).float(), 'llama')
cases.append(('Mixtral', *mix, 32000))
old = os.path.join(W, 'mixtral_old-f32.gguf')
split_experts(mix[1], old)
cases.append(('Mixtral (old layout)', mix[0], old, 32000))
cases.append(('Llama 3 (rope_freqs)', *build('llama3', LlamaForCausalLM(LlamaConfig(vocab_size=32000, hidden_size=256, intermediate_size=384, num_hidden_layers=3,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=131072, rope_theta=500000.0,
    rope_scaling={'rope_type': 'llama3', 'factor': 32.0, 'low_freq_factor': 1.0, 'high_freq_factor': 4.0, 'original_max_position_embeddings': 8192},
    tie_word_embeddings=False)).float(), 'llama'), 32000))

bad = 0
for name, hf_dir, gguf, vocab in cases:
    a, b = engine_logits(gguf, ids, vocab), torch_logits(hf_dir, ids)
    d = float(np.abs(a - b).max()); scale = float(np.abs(b).max())
    ok = a.shape == b.shape and d <= 1e-3 * max(1.0, scale) and (a.argmax(1) == b.argmax(1)).all()
    bad += not ok
    print('%s %-22s %d positions  max |logit difference| %.2e  (logits up to %.1f)' % ('PASS' if ok else 'FAIL', name, len(ids), d, scale))
print('ALL PASSED' if not bad else '%d FAILED' % bad)
sys.exit(1 if bad else 0)
