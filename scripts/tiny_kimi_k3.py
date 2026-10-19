#!/usr/bin/env python3
"""A tiny random Kimi K3 (arch kimi-k3) GGUF, for checking the engine against llama.cpp (no PyTorch reference exists here).

    python3 scripts/tiny_kimi_k3.py OUT.gguf TOKDIR [--seed N]      (TOKDIR: Qwen2.5's tokenizer.json, vocab.json, merges.txt)

Every part of the design is in it: Kimi delta-attention layers and latent-attention layers (pattern from attention.head_count_kv),
a leading dense layer, experts in a latent space with a selection bias, shared experts, the situ activation, the MLA output gate,
and cross-layer residuals banked every attn_res.block_size layers. Weights are float32, so the two engines can agree to ~1e-5.
"""
import json
import os
import sys

import numpy as np
from gguf import GGUFWriter, GGMLQuantizationType, TokenType

out, tokdir = sys.argv[1], sys.argv[2]
seed = int(sys.argv[sys.argv.index('--seed') + 1]) if '--seed' in sys.argv else 0
r = np.random.default_rng(seed)
pick = lambda xs: xs[int(r.integers(len(xs)))]

A = 'kimi-k3'
n, H = pick([64, 96]), pick([2, 4])
d = pick([16, 32])                     # delta-attention head size
nL = pick([5, 6, 7])
kv = [0] * nL
for l in range(nL):
    if (l + 1) % pick([2, 3]) == 0: kv[l] = 1   # latent-attention layers among the delta-attention ones
kv[-1] = 1
qlora, kvl, nope, rd, vh = pick([None, 32]), 32, 16, 8, 16
E, k, lat, ffe, ff = pick([8, 16]), pick([2, 4]), pick([32, 48]), 16, 48
bs = pick([2, 3])
lb = pick([-5.0, None])
opt = lambda f: f in sys.argv   # switches for narrowing a mismatch down: --all-kda --all-mla --no-res --no-latent --no-gate-bound
if opt('--all-kda'): kv = [0] * (nL - 1) + [1]
if opt('--all-mla'): kv = [1] * nL
if opt('--no-res'): bs = 0
if opt('--kv'): kv = [int(x) for x in sys.argv[sys.argv.index('--kv') + 1].split(',')]; nL = len(kv)   # e.g. --kv 0,1
dense_all = opt('--dense-all')   # every layer a dense FFN (no experts)
if opt('--no-gate-bound'): lb = None
cfg = dict(n_layer=nL, n_embd=n, heads=H, kda_head=d, kv=kv, q_lora=qlora, experts=E, used=k, latent=lat, res_block=bs, gate_bound=lb)

# tokenizer: Qwen2.5's byte-level BPE
vocab = json.load(open(os.path.join(tokdir, 'vocab.json')))
added = {t['content']: t['id'] for t in json.load(open(os.path.join(tokdir, 'tokenizer.json')))['added_tokens']}
V = 151936
toks, types = [''] * V, [TokenType.UNUSED] * V
for t, i in vocab.items(): toks[i], types[i] = t, TokenType.NORMAL
for t, i in added.items(): toks[i], types[i] = t, TokenType.CONTROL
for i in range(V):
    if not toks[i]: toks[i] = '[PAD%d]' % i
merges = [l.rstrip('\n') for l in open(os.path.join(tokdir, 'merges.txt'), encoding='utf-8') if l.strip() and not l.startswith('#version')]

w = GGUFWriter(out, A)
w.add_name('tiny-kimi-k3')
w.add_uint32(A + '.block_count', nL); w.add_uint32(A + '.context_length', 4096); w.add_uint32(A + '.embedding_length', n)
w.add_uint32(A + '.feed_forward_length', ff); w.add_uint32(A + '.attention.head_count', H)
w.add_array(A + '.attention.head_count_kv', kv)
w.add_float32(A + '.rope.freq_base', 10000.0); w.add_float32(A + '.attention.layer_norm_rms_epsilon', 1e-5)
w.add_uint32(A + '.expert_count', E); w.add_uint32(A + '.expert_used_count', k); w.add_uint32(A + '.expert_group_used_count', 1)
w.add_uint32(A + '.expert_gating_func', 2)
w.add_uint32(A + '.attention.key_length', kvl + rd); w.add_uint32(A + '.attention.value_length', kvl)
w.add_uint32(A + '.vocab_size', V); w.add_uint32(A + '.ssm.conv_kernel', 4); w.add_uint32(A + '.kda.head_dim', d)
if lb is not None: w.add_float32(A + '.kda.gate_lower_bound', lb)
if qlora: w.add_uint32(A + '.attention.q_lora_rank', qlora)
w.add_uint32(A + '.attention.kv_lora_rank', kvl); w.add_uint32(A + '.rope.dimension_count', rd)
w.add_uint32(A + '.attention.key_length_mla', nope + rd); w.add_uint32(A + '.attention.value_length_mla', vh)
w.add_uint32(A + '.expert_feed_forward_length', ffe); w.add_uint32(A + '.expert_shared_count', 1)
w.add_uint32(A + '.leading_dense_block_count', nL if dense_all else 1); w.add_float32(A + '.expert_weights_scale', 1.5)
w.add_bool(A + '.expert_weights_norm', True)
if not opt('--no-latent'): w.add_uint32(A + '.expert_latent_length', lat)
w.add_float32(A + '.activation.situ_beta', 1.0); w.add_float32(A + '.activation.situ_linear_beta', 2.0)
w.add_uint32(A + '.attn_res.block_size', bs)
w.add_tokenizer_model('gpt2'); w.add_tokenizer_pre('qwen2')
w.add_token_list(toks); w.add_token_types(types); w.add_token_merges(merges)
w.add_bos_token_id(added['<|endoftext|>']); w.add_eos_token_id(added['<|im_end|>'])


def T(name, *shape, scale=0.15, kind='w'):   # numpy shape: (rows, ..., cols), i.e. GGUF ne reversed
    if kind == 'norm': a = 1.0 + 0.1 * r.standard_normal(shape)
    elif kind == 'neg': a = -np.exp(r.uniform(-1, 1, shape))
    else: a = scale * r.standard_normal(shape)
    w.add_tensor(name, a.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)


T('token_embd.weight', V, n, scale=1.0); T('output.weight', V, n, scale=0.3)
T('output_norm.weight', n, kind='norm')
if bs: T('output_res_score.weight', n, scale=0.5)
di = H * d
for l in range(nL):
    b = 'blk.%d.' % l
    T(b + 'attn_norm.weight', n, kind='norm'); T(b + 'ffn_norm.weight', n, kind='norm')
    if bs: T(b + 'attn_res_score.weight', n, scale=0.5); T(b + 'ffn_res_score.weight', n, scale=0.5)
    if kv[l] == 0:
        for x in 'qkv': T(b + 'attn_%s.weight' % x, di, n); T(b + 'ssm_conv1d_%s.weight' % x, di, 1, 4, scale=0.5)
        T(b + 'ssm_f_a.weight', d, n); T(b + 'ssm_f_b.weight', di, d, scale=0.5); T(b + 'ssm_beta.weight', H, n, scale=0.5)
        T(b + 'ssm_a', H, kind='neg'); T(b + 'ssm_dt.bias', di, scale=0.5); T(b + 'ssm_g.weight', di, n)
        T(b + 'ssm_norm.weight', d, kind='norm'); T(b + 'attn_output.weight', n, di)
    else:
        if qlora: T(b + 'attn_q_a.weight', qlora, n); T(b + 'attn_q_a_norm.weight', qlora, kind='norm'); T(b + 'attn_q_b.weight', H * (nope + rd), qlora)
        else: T(b + 'attn_q.weight', H * (nope + rd), n)
        T(b + 'attn_kv_a_mqa.weight', kvl + rd, n); T(b + 'attn_kv_a_norm.weight', kvl, kind='norm')
        T(b + 'attn_k_b.weight', H, kvl, nope, scale=0.3); T(b + 'attn_v_b.weight', H, vh, kvl, scale=0.3)
        T(b + 'attn_gate.weight', H * vh, n); T(b + 'attn_output.weight', n, H * vh)
    if l == 0 or dense_all:
        T(b + 'ffn_gate.weight', ff, n); T(b + 'ffn_up.weight', ff, n); T(b + 'ffn_down.weight', n, ff)
    else:
        T(b + 'ffn_gate_inp.weight', E, n, scale=0.5); T(b + 'exp_probs_b.bias', E, scale=0.2)
        el = n if opt('--no-latent') else lat
        T(b + 'ffn_gate_exps.weight', E, ffe, el); T(b + 'ffn_up_exps.weight', E, ffe, el); T(b + 'ffn_down_exps.weight', E, el, ffe)
        if not opt('--no-latent'):
            T(b + 'ffn_routed_down.weight', lat, n); T(b + 'ffn_routed_up.weight', n, lat); T(b + 'ffn_routed_norm.weight', lat, kind='norm')
        T(b + 'ffn_gate_shexp.weight', ffe, n); T(b + 'ffn_up_shexp.weight', ffe, n); T(b + 'ffn_down_shexp.weight', n, ffe)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(json.dumps(cfg))
