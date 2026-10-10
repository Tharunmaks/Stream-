"""Get free training text, train a BPE tokenizer on it, and write token files.

  python data.py --dataset tinystories --vocab 8192 --max_tokens 50e6
  python data.py --dataset text --file my_notes.txt --vocab 4096

Free datasets (streamed from Hugging Face, nothing to pay):
  tinystories  roneneldan/TinyStories          simple English stories, best for small models
  fineweb-edu  HuggingFaceFW/fineweb-edu       high-quality web text (sample-10BT)
  wikitext     Salesforce/wikitext             Wikipedia articles (wikitext-103)
  text         --file your own .txt            works offline, also on a phone

Needs: pip install tokenizers datasets   (without 'tokenizers' it falls back to bytes, vocab 257;
'datasets' is only needed for the Hugging Face sets).
Writes data/train.bin, data/val.bin, data/meta.json, tokenizer.json and updates config.json vocab_size.
"""
import argparse, json, os, sys
import numpy as np

DATASETS = {
    "tinystories": ("roneneldan/TinyStories", None, "text"),
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "text"),
    "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1", "text"),
}
EOT = "<|endoftext|>"

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="tinystories", choices=list(DATASETS) + ["text"])
ap.add_argument("--file", help="your own text file (with --dataset text)")
ap.add_argument("--vocab", type=int, default=8192)
ap.add_argument("--max_tokens", type=float, default=20e6)
ap.add_argument("--tok_docs", type=int, default=50000, help="documents used to train the tokenizer")
ap.add_argument("--val_every", type=int, default=200, help="every Nth document goes to validation")
ap.add_argument("--out", default="data")
ap.add_argument("--config", default="config.json")
a = ap.parse_args()


def docs():
    if a.dataset == "text":
        if not a.file:
            sys.exit("--dataset text needs --file")
        txt = open(a.file, encoding="utf-8", errors="replace").read()
        parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
        # glue short paragraphs into ~2k-char documents
        buf = ""
        for p in parts:
            buf += p + "\n\n"
            if len(buf) > 2000:
                yield buf; buf = ""
        if buf:
            yield buf
        return
    try:
        from datasets import load_dataset
    except ImportError:
        sys.exit("pip install datasets   (or use --dataset text --file yourfile.txt)")
    name, cfg, field = DATASETS[a.dataset]
    ds = load_dataset(name, cfg, split="train", streaming=True)
    for row in ds:
        t = (row.get(field) or "").strip()
        if t:
            yield t


os.makedirs(a.out, exist_ok=True)
it = docs()
first = []
for d in it:
    first.append(d)
    if len(first) >= a.tok_docs:
        break
print(f"collected {len(first)} documents for the tokenizer")

try:
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=[EOT], min_frequency=2,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(first, trainer=tr)
    tok.save("tokenizer.json")
    eot = tok.token_to_id(EOT)
    vocab = tok.get_vocab_size()
    encode = lambda batch: [e.ids for e in tok.encode_batch(batch)]
    kind = "bpe"
except ImportError:
    print("tokenizers not installed: using byte-level tokens (vocab 257). pip install tokenizers for BPE.")
    eot, vocab, kind = 256, 257, "bytes"
    encode = lambda batch: [list(t.encode("utf-8")) for t in batch]

dtype = np.uint16 if vocab <= 65535 else np.uint32
ftr, fva = open(os.path.join(a.out, "train.bin"), "wb"), open(os.path.join(a.out, "val.bin"), "wb")
n_tr = n_va = n_doc = 0
limit = int(a.max_tokens)


def batches():
    b = []
    for d in first:
        b.append(d)
        if len(b) == 256:
            yield b; b = []
    for d in it:
        b.append(d)
        if len(b) == 256:
            yield b; b = []
    if b:
        yield b


for b in batches():
    for ids in encode(b):
        arr = np.array(ids + [eot], dtype=dtype)
        n_doc += 1
        if n_doc % a.val_every == 0:
            arr.tofile(fva); n_va += len(arr)
        else:
            arr.tofile(ftr); n_tr += len(arr)
    print(f"\r{n_tr/1e6:.1f}M train tokens, {n_va/1e6:.2f}M val", end="", flush=True)
    if n_tr >= limit:
        break
ftr.close(); fva.close()
print()
meta = {"dtype": np.dtype(dtype).name, "vocab_size": vocab, "eot_id": eot, "tokenizer": kind,
        "dataset": a.dataset, "train_tokens": n_tr, "val_tokens": n_va}
json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)

if os.path.exists(a.config):
    cfg = json.load(open(a.config))
    if cfg.get("vocab_size") != vocab:
        cfg["vocab_size"] = vocab
        json.dump(cfg, open(a.config, "w"), indent=2)
        print(f"config.json vocab_size set to {vocab}")
print(json.dumps(meta, indent=2))
