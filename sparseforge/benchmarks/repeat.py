import json, os
from sparseforge.flash.engine import create_model, train, evaluate
for rep in (1, 2):
    p = '/tmp/sf_bench/rep%d' % rep; os.system('rm -rf ' + p); create_model(p, n_experts=262144)
    r = train(p, '/tmp/train.txt', steps=40, batch=256, lr=3e-3, save_every=0); r['val_loss'] = evaluate(p, '/tmp/val.txt')
    print(json.dumps({k: r[k] for k in ('tokens_per_s', 'final_loss', 'val_loss')}), flush=True)
