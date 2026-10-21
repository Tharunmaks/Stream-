import torch, json
from sparseforge.layers import QuantLoRALinear, HashMoEFFN, sampled_softmax_loss
from sparseforge.model import TinyGPT, apply_lora_quant
from sparseforge import estimators, hardware, notebook
from sparseforge.train import run
import tempfile, os
_C = os.path.join(tempfile.mkdtemp(), "c.txt"); open(_C, "w").write("tiny corpus for tests. " * 200)
torch.set_num_threads(2)

def test_quant_roundtrip():
    w = torch.randn(32, 64)
    for bits, tol in [(8, 0.03), (4, 0.6)]:
        l = QuantLoRALinear(w, None, bits, r=0, compute_dtype=torch.float32)
        assert (l.dequant() - w).abs().max() < tol * w.abs().max()

def test_lora_starts_identical_and_trains_only_adapters():
    torch.manual_seed(0); m = TinyGPT(d=64, n_layers=2, n_heads=2, ctx=16)
    x = torch.randint(0, 256, (2, 16)); ref = m(x)
    apply_lora_quant(m, bits=8, compute_dtype=torch.float32)
    assert (m(x) - ref).abs().max() < 0.1
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert names and all("lora_" in n for n in names)

def test_moe_and_sparse_out_and_ckpt():
    m = TinyGPT(d=64, n_layers=2, n_heads=2, ctx=16, moe=4); m.grad_ckpt = True
    x = torch.randint(0, 256, (2, 16)); loss = m(x, x); loss.backward()
    assert torch.isfinite(loss)
    h = torch.randn(4, 8, 32); W = torch.randn(5000, 32)
    assert torch.isfinite(sampled_softmax_loss(h, W, torch.randint(0, 5000, (4, 8)), 256))

def test_estimators_hw_notebook():
    assert estimators.estimate_memory(70e9, "qlora_int4")["base_gb"] < 50
    assert estimators.estimate_throughput(1e9, 1.0)["tokens_per_s"] > 0
    assert "logical_cores" in hardware.detect_hardware()
    assert notebook.make_notebook({"steps": 1})["nbformat"] == 4

def test_tiny_run():
    r = run(dict(d=32, n_layers=1, n_heads=2, ctx=16, batch=2, steps=3, warmup=1, eval_batches=1, mode="lora", bits=4, bf16=True, threads=2, corpus=_C))
    assert r["tokens_per_s"] > 0
