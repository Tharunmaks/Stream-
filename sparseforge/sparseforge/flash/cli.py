import argparse, json, os
def main():
    from .engine import create_model, train, sample, evaluate
    ap = argparse.ArgumentParser(prog="sparseforge-flash"); sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("create"); a.add_argument("path"); a.add_argument("--ctx", type=int, default=8); a.add_argument("--d", type=int, default=32)
    a.add_argument("--hidden", type=int, default=256); a.add_argument("--experts", type=int, default=262144); a.add_argument("--k-hash", type=int, default=3)
    a.add_argument("--ternary", action="store_true")
    t = sp.add_parser("train"); t.add_argument("path"); t.add_argument("corpus"); t.add_argument("--steps", type=int, default=200)
    t.add_argument("--batch", type=int, default=256); t.add_argument("--lr", type=float, default=2e-3); t.add_argument("--expert-lr", type=float, default=2e-3)
    t.add_argument("--status-file"); t.add_argument("--eval-corpus"); t.add_argument("--stop-file")
    s = sp.add_parser("sample"); s.add_argument("path"); s.add_argument("--prompt", default=""); s.add_argument("-n", type=int, default=200)
    s.add_argument("--temperature", type=float, default=0.8)
    e = sp.add_parser("eval"); e.add_argument("path"); e.add_argument("corpus")
    x = ap.parse_args()
    if x.cmd == "create": r = create_model(x.path, x.ctx, x.d, x.hidden, x.experts, x.k_hash, x.ternary)
    elif x.cmd == "train": r = train(x.path, x.corpus, x.steps, x.batch, x.lr, x.expert_lr, status_file=x.status_file, eval_corpus=x.eval_corpus, stop_file=x.stop_file)
    elif x.cmd == "sample": print(sample(x.path, x.prompt, x.n, x.temperature)); return
    else: r = {"eval_loss": evaluate(x.path, x.corpus)}
    print(json.dumps(r, indent=1))
if __name__ == "__main__":
    main()
