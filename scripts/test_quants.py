#!/usr/bin/env python3
"""Check every weight format's decoder (src/es_quant.c) against the gguf Python package on random blocks.

    make es_qtest && python3 scripts/test_quants.py        (needs: pip install numpy gguf)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
try:
    from gguf import GGMLQuantizationType as T, GGML_QUANT_SIZES
    from gguf.quants import dequantize
except ImportError:
    sys.exit('SKIP: needs the gguf package (pip install gguf)')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QT = os.path.join(ROOT, 'es_qtest')
# (type, byte offsets of fp16 scales inside a block); scales are set to sane values so no Inf/NaN appears
TYPES = {
    'Q4_0': [0], 'Q4_1': [0, 2], 'Q5_0': [0], 'Q5_1': [0, 2], 'Q8_0': [0],
    'Q2_K': [80, 82], 'Q3_K': [108], 'Q4_K': [0, 2], 'Q5_K': [0, 2], 'Q6_K': [208],
    'IQ4_NL': [0], 'IQ4_XS': [0], 'IQ2_XXS': [0], 'IQ2_XS': [0], 'IQ2_S': [0], 'IQ3_XXS': [0], 'IQ3_S': [0],
    'IQ1_S': [0], 'IQ1_M': [], 'TQ1_0': [52], 'TQ2_0': [64], 'MXFP4': [], 'BF16': [], 'F16': [],
}
rng = np.random.default_rng(1)
bad = 0
for name, scales in TYPES.items():
    if not hasattr(T, name):
        print('skip', name, '(gguf package too old)'); continue
    qt = getattr(T, name)
    blck, size = GGML_QUANT_SIZES[qt]
    nblk = max(1, 4096 // blck)
    raw = rng.integers(0, 256, size=(nblk, size), dtype=np.uint8)
    for off in scales:
        h = (rng.random(nblk) * 0.02 + 0.001).astype(np.float16).view(np.uint8).reshape(nblk, 2)
        raw[:, off:off + 2] = h
    if name == 'MXFP4':
        raw[:, 0] = rng.integers(110, 135, size=nblk)   # sane exponents
    if name == 'IQ1_M':   # the fp16 scale is spread over the top nibbles of the 4 scale words
        sc = raw[:, 48:56].copy().view(np.uint16)
        h = (rng.random(nblk) * 0.02 + 0.001).astype(np.float16).view(np.uint16)
        for k in range(4):
            sc[:, k] = (sc[:, k] & 0x0fff) | (((h >> (4 * k)) & 0xf) << 12)
        raw[:, 48:56] = sc.view(np.uint8)
    if name in ('BF16', 'F16'):
        vals = (rng.standard_normal(nblk) * 0.1).astype(np.float32)
        raw = (vals.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(nblk, 2) if name == 'BF16' else vals.astype(np.float16).view(np.uint8).reshape(nblk, 2)
    want = dequantize(raw.reshape(-1), qt).reshape(-1).astype(np.float32)
    with tempfile.TemporaryDirectory() as d:
        bi, bo = os.path.join(d, 'b.bin'), os.path.join(d, 'o.f32')
        raw.tofile(bi)
        if name in ('BF16', 'F16'):
            # es_qtest --raw counts 256-value rows; pad to whole rows
            pass
        r = subprocess.run([QT, '--raw', str(int(qt)), bi, bo], capture_output=True, text=True)
        if r.returncode:
            print('FAIL', name, r.stderr.strip()); bad += 1; continue
        got = np.fromfile(bo, dtype=np.float32)
    n = min(len(got), len(want))
    err = float(np.max(np.abs(got[:n] - want[:n]))) if n else 1.0
    scale = float(np.max(np.abs(want))) or 1.0
    ok = n == len(want) and err <= 1e-6 * scale
    bad += not ok
    print('%-4s %-8s %6d values  max diff %.2e (of %.2e)' % ('PASS' if ok else 'FAIL', name, n, err, scale))
print('ALL PASSED' if not bad else '%d FAILED' % bad)
sys.exit(1 if bad else 0)
