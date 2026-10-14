import os, tempfile, numpy as np
from sparseforge.flash.engine import create_model, train, sample, evaluate, FlashModel

def _corpus(d):
    p = os.path.join(d, "c.txt"); open(p, "w").write("hello world, the quick brown fox. " * 400); return p

def test_flash_train_reduces_loss_and_samples():
    d = tempfile.mkdtemp(); c = _corpus(d); m = os.path.join(d, "m")
    create_model(m, ctx=4, d=8, hidden=32, n_experts=1024)
    before = evaluate(m, c, n=256)
    r = train(m, c, steps=40, batch=64, lr=5e-3, log_every=5)
    assert evaluate(m, c, n=256) < before and r["tokens_per_s"] > 0
    assert isinstance(sample(m, "hello", 20), str)

def test_sparse_update_touches_few_experts_and_grad_check():
    d = tempfile.mkdtemp(); m = os.path.join(d, "m")
    create_model(m, ctx=3, d=4, hidden=8, n_experts=4096)
    fm = FlashModel(m); x = np.random.randint(0, 256, (5, 3)); y = np.random.randint(0, 256, 5)
    loss, gr, dh, ids = fm.loss_and_grads(x, y)
    # finite-difference check on W1
    eps = 1e-3; i, j = 2, 3; fm.p["W1"][i, j] += eps
    l2 = fm.loss_and_grads(x, y)[0]; fm.p["W1"][i, j] -= eps
    assert abs((l2 - loss) / eps - gr["W1"][i, j]) < 1e-2
    assert fm.update(gr, dh, ids, 1e-3, 1e-3) <= 15
