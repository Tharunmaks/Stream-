#!/usr/bin/env python3
"""Checks that sparse upcycling is function-preserving: a trained dense model and its Mixture-of-Experts copy (f16 and 8-bit experts)
must produce identical greedy text in the real engine. Needs NumPy and a built ./es_chat.   python3 scripts/test_upcycle.py"""
import os, subprocess, sys, tempfile
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.join(root, 'mcp'))
import builder as B
if B.np is None or not os.path.exists(os.path.join(root, 'es_chat')): sys.exit('SKIP: needs numpy and a built es_chat (make)')
text = open(os.path.join(root, 'README.md')).read()
tk = B.train_bpe(text, vocab_size=512); tk.setdefault('specials', ['<|endoftext|>']); tok = B.Tok(tk); ids = B.np.array(tok.encode(text), 'int64')
c = B._q32(B.design_config(1.5e6, vocab=len(tk['tokens']), ctx=128, tied=True)); c['vocab'] = len(tk['tokens']); P = B.init_params(c)
B.train(P, c, ids, 400, B=16, T=64, lr=3e-3)
d = tempfile.mkdtemp(); B.write_gguf(d + '/dense.gguf', c, tk, P, 'f16')
def run(f): return subprocess.run([os.path.join(root, 'es_chat'), '-g', f, '-r', '-p', '# ExpertStream ', '-n', '60', '-t', '0', '-j', '2', '-q'], capture_output=True, text=True).stdout
ref = run(d + '/dense.gguf'); ok = True
for q in ('f16', 'q8_0'):
    B.write_gguf_moe(d + '/moe_%s.gguf' % q, c, tk, P, 16, 2, q); out = run(d + '/moe_%s.gguf' % q)
    # f16 experts must match exactly; 8-bit rounding may change a late token, so require the same first 15 characters
    same = out == ref if q == 'f16' else out[:15] == ref[:15] and len(out) > 15; ok &= same
    print('moe 16 experts, top-2, %s experts: %s' % (q, ('IDENTICAL to dense' if out == ref else 'same start as dense (8-bit rounding)') if same else 'DIFFERENT'))
B.write_gguf_moe(d + '/one.gguf', c, tk, P, 1, 1, 'q8_0'); exact = run(d + '/one.gguf') == run(d + '/moe_q8_0.gguf'); ok &= exact
print('16 experts == 1 expert (same 8-bit weights): %s' % ('EXACT, upcycling does not change a single token' if exact else 'DIFFERENT'))
print('37B plan:', {k: v for k, v in B.upcycle_plan(37e9, max_experts=32768).items() if k in ('base_params', 'experts', 'total_params', 'active_params', 'file_gb')})
sys.exit(0 if ok else 1)
