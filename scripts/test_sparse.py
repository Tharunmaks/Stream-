#!/usr/bin/env python3
"""Checks for es_train moe --sparse (moe/es_sparse.c) and libesd.so (moe/esd.c), the Trainer's Train MoE engine.

    make es_train libesd.so esd_ktest && python3 scripts/test_sparse.py [--data TEXT] [--work DIR]

1. gradcheck, plain and --p6 (finite differences on every parameter group)
2. tiny preset (the Trainer's: L2 E4 d512 f256 batch 8): held-out loss drops; plain and --p6 (4 threads) write identical checkpoints
3. the Trainer's log contract: "step N train_ce X ... train_tok/s Y", "heldout_loss X", RESULT keys
4. libesd.so reproduces the trainer's final held-out loss token by token, and es_common SparseRun generates through it
5. SIGTERM stops after the current step, saves the checkpoint and prints RESULT
6. ternary experts + low-rank router (the 70B phone format): gradcheck, training, libesd matches the trainer
7. esd_ktest: every SIMD kernel equals the scalar one bit for bit
"""
import argparse, ctypes, math, os, re, signal, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--data", help="training text (default: this repo's .md files)")
ap.add_argument("--work", default=tempfile.mkdtemp(prefix="es_sparse_"))
ap.add_argument("--bin", default=ROOT)
a = ap.parse_args()
TRAIN, LIB = os.path.join(a.bin, "es_train"), os.path.join(a.bin, "libesd.so")
fails = []
def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what); (None if ok else fails.append(what))

data = a.data
if not data:
    data = os.path.join(a.work, "corpus.txt")
    with open(data, "wb") as out:
        for dp, dn, fn in os.walk(ROOT):
            dn[:] = [d for d in dn if not d.startswith(".")]
            for f in sorted(fn):
                if f.endswith(".md"): out.write(open(os.path.join(dp, f), "rb").read())
X = os.path.join(a.work, "tiny-2m.bin")
TINY = ["--layers", "2", "--experts", "4", "--dim", "512", "--ffn", "256", "--batch", "8", "--seq", "128", "--experts-file", X]

def run(args, env=None):
    r = subprocess.run([TRAIN, "moe", "--sparse"] + args, capture_output=True, text=True, env=dict(os.environ, **(env or {})))
    return r.returncode, r.stdout + r.stderr

for mode in ([], ["--p6", "--threads", "3"], ["--bits", "2", "--router-rank", "8"]):
    rc, out = run(["--layers", "2", "--experts", "4", "--dim", "64", "--ffn", "32", "--batch", "2", "--seq", "16",
                   "--experts-file", os.path.join(a.work, "gc.bin"), "--gradcheck"] + mode)
    check(rc == 0 and "gradcheck PASS" in out, "gradcheck " + (" ".join(mode) if mode else "plain"))

common = TINY + ["--steps", "40", "--eval-every", "20", "--data", data]
ck1, ck2 = os.path.join(a.work, "plain.ckpt"), os.path.join(a.work, "p6.ckpt")
rc1, o1 = run(common + ["--save", ck1])
rc2, o2 = run(common + ["--p6", "--save", ck2], {"OMP_NUM_THREADS": "4"})
check(rc1 == 0 and rc2 == 0, "tiny preset trains (plain and p6)")
res = dict(kv.split("=", 1) for kv in [l for l in o1.splitlines() if l.startswith("RESULT")][0].split()[1:])
check({"steps", "train_tok/s", "heldout_before", "heldout_after"} <= set(res), "RESULT has the keys the Trainer reads")
check(float(res["heldout_after"]) < float(res["heldout_before"]) - 1.0, f"held-out loss drops: {res['heldout_before']} -> {res['heldout_after']}")
steps = [re.match(r"^step (\d+) train_ce ([\d.]+).*?train_tok\/s ([\d.]+)", l) for l in o1.splitlines()]
check(sum(1 for m in steps if m) == 40, "40 step lines match the Train MoE page's regex")
check(len(re.findall(r"heldout_loss ([\d.]+)", o1)) >= 3, "held-out lines (before, every 20, after)")
check(open(ck1, "rb").read() == open(ck2, "rb").read(), "plain and --p6 checkpoints are identical")
print("  tok/s plain %s, p6 x4 %s" % (res["train_tok/s"], re.search(r"train_tok/s=([\d.]+)", o2).group(1)))

# libesd: replay the trainer's held-out batches (2 batches x 8 sequences at fixed offsets in the last 10%)
L = ctypes.CDLL(LIB); L.esd_open.restype = ctypes.c_void_p; L.esd_open.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_long]
L.esd_step.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]; L.esd_reset.argtypes = [ctypes.c_void_p]
L.esd_set.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_float]; L.esd_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]
L.esd_close.argtypes = [ctypes.c_void_p]
h = L.esd_open(ck1.encode(), X.encode(), 1 << 20)
check(bool(h), "libesd opens the checkpoint")
buf = open(data, "rb").read(); dn = len(buf); split = dn - dn // 10; hn = dn - split; T, n = 128, 16
lg = (ctypes.c_float * 256)(); tot = 0.0
for i in range(n):
    o = split + int(i * (hn - T - 1) / (n - 1)); seq = buf[o:o + T + 1]; L.esd_reset(h)
    for t in range(T):
        L.esd_step(h, seq[t], lg); z = list(lg); m = max(z)
        tot += math.log(sum(math.exp(v - m) for v in z)) + m - z[seq[t + 1]]
def esd_heldout(ck, xf):
    h = L.esd_open(ck.encode(), xf.encode(), 1 << 22); tot = 0.0
    for i in range(n):
        o = split + int(i * (hn - T - 1) / (n - 1)); seq = buf[o:o + T + 1]; L.esd_reset(h)
        for t in range(T):
            L.esd_step(h, seq[t], lg); z = list(lg); m = max(z)
            tot += math.log(sum(math.exp(v - m) for v in z)) + m - z[seq[t + 1]]
    return h, tot / (n * T)
mine, theirs = tot / (n * T), float(res["heldout_after"])
check(abs(mine - theirs) < 2e-3, f"libesd held-out {mine:.4f} = trainer {theirs:.4f}")
st = (ctypes.c_long * 3)(); L.esd_stats(h, st); check(st[0] + st[1] == 2 * n * T, f"one expert per layer per token ({st[0]} hits, {st[1]} reads)")
L.esd_close(h)
XT, ckt = os.path.join(a.work, "tiny-tern.bin"), os.path.join(a.work, "tern.ckpt")
rc, ot = run(["--layers", "2", "--experts", "4", "--dim", "512", "--ffn", "256", "--batch", "8", "--seq", "128", "--experts-file", XT,
              "--bits", "2", "--router-rank", "32", "--steps", "40", "--eval-every", "20", "--data", data, "--save", ckt])
rt = dict(kv.split("=", 1) for kv in [l for l in ot.splitlines() if l.startswith("RESULT")][0].split()[1:]) if rc == 0 else {}
check(rc == 0 and float(rt["heldout_after"]) < float(rt["heldout_before"]) - 1.0, f"ternary + low-rank router trains: {rt.get('heldout_before')} -> {rt.get('heldout_after')}")
if rc == 0:
    h, mine = esd_heldout(ckt, XT); L.esd_close(h)
    check(abs(mine - float(rt["heldout_after"])) < 5e-3, f"libesd ternary held-out {mine:.4f} = trainer {rt['heldout_after']}")
kt = os.path.join(a.bin, "esd_ktest")
if os.path.exists(kt):
    r = subprocess.run([kt], capture_output=True, text=True); print(r.stdout.rstrip()); check(r.returncode == 0, "esd_ktest: SIMD kernels = scalar")
else: print("skip esd_ktest (make esd_ktest)")

sys.path.insert(0, os.path.join(ROOT, "es-common"))
try:
    import json
    from es_common import esmoe
    md = os.path.join(a.work, "model"); os.makedirs(md, exist_ok=True); os.replace(ck1, os.path.join(md, "m.ckpt"))
    json.dump(dict(format="es-sparse-v1", ckpt="m.ckpt", experts_file=X, layers=2, experts=4, dim=512, ffn=256, topk=1), open(os.path.join(md, "config.json"), "w"))
    os.environ["ES_ESD_LIB"] = LIB
    g = esmoe.load_run(md).generate("The engine ", max_new=48, temperature=0)
    check(g["new_tokens"] == 48 and g["prefetch_checks"] > 0, f"SparseRun generates: {g['text']!r} ({g['tok_per_s']} tok/s, prefetch recall {g['prefetch_recall']})")
except ImportError as e:
    print("skip SparseRun (es_common needs numpy):", e)

ck3 = os.path.join(a.work, "stop.ckpt")
p = subprocess.Popen([TRAIN, "moe", "--sparse"] + TINY + ["--batch", "32", "--steps", "100000", "--data", data, "--save", ck3], stdout=subprocess.PIPE, text=True)
lines = []
for l in p.stdout:
    lines.append(l)
    if l.startswith("step 3 "): p.send_signal(signal.SIGTERM)
p.wait()
check(p.returncode == 0 and os.path.exists(ck3) and any(l.startswith("RESULT") for l in lines), "SIGTERM: stops, saves and prints RESULT")

print("\n%d failed" % len(fails) if fails else "\nall passed"); sys.exit(1 if fails else 0)
