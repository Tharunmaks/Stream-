"""Talk to your model.
  python sample.py --out run --prompt "Once upon a time"
  python sample.py --out run --chat        (type lines, empty line quits)"""
import argparse, json, os, torch
from model import GPT

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="run")
ap.add_argument("--prompt", default="Once upon a time")
ap.add_argument("--tokens", type=int, default=200)
ap.add_argument("--temperature", type=float, default=0.8)
ap.add_argument("--top_k", type=int, default=50)
ap.add_argument("--chat", action="store_true")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
path = os.path.join(a.out, "model.pt")
if not os.path.exists(path):
    path = os.path.join(a.out, "ckpt.pt")
ck = torch.load(path, map_location="cpu", weights_only=False)
cfg = ck.get("config") or json.load(open("config.json"))
cfg["grad_ckpt"] = False
model = GPT(cfg)
missing, _ = model.load_state_dict(ck["model"], strict=False)
assert set(missing) <= {"head.weight"}, missing
model = model.float().to(dev).eval()

if os.path.exists("tokenizer.json"):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file("tokenizer.json")
    enc, dec, eot = (lambda s: tok.encode(s).ids), (lambda ids: tok.decode(ids)), tok.token_to_id("<|endoftext|>")
else:
    enc = lambda s: list(s.encode("utf-8"))
    dec = lambda ids: bytes(i for i in ids if i < 256).decode("utf-8", "replace")
    eot = 256


def run(prompt):
    ids = enc(prompt) or [eot]
    out = model.generate(torch.tensor([ids], device=dev), a.tokens, a.temperature, a.top_k, stop_id=eot)
    return dec(out[0, len(ids):].tolist())


print(f"[{sum(p.numel() for p in model.parameters())/1e6:.1f}M params, step {ck.get('step')}]")
if a.chat:
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            break
        print(run(line))
else:
    print(a.prompt + run(a.prompt))
