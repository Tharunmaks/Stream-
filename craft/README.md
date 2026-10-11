# Expert Craft

Build language models from scratch with free compute, including **a model with up to 70 billion parameters trained on a phone**.
A plain command-line tool (`craft/expertcraft.py`, Python 3 + NumPy). Website with a planner: https://tharunmaks.github.io/Stream-/craft.html

```sh
pkg install -y python python-numpy git                    # Termux; on a PC: pip install numpy
git clone https://github.com/Tharunmaks/Stream-.git ~/Stream- && cd ~/Stream-
python3 craft/expertcraft.py phone-plan 70e9 --minutes 600                    # measure this device, explain what 70B means here
python3 craft/expertcraft.py phone-train mymodel --dataset tinystories --params 70e9 --minutes 600 \
        --min-battery 25 --max-temp 42 --max-wear-gb 2000                     # Ctrl-C saves; run again to continue
python3 craft/expertcraft.py phone-sample mymodel --prompt "Once upon a time"
python3 craft/expertcraft.py phone-status mymodel
```

## The technology: Flash Experts

Training a dense 70B model costs 6 x 70e9 x tokens FLOPs: about 4e20 per billion tokens, hundreds of years on a phone CPU. So the model is
**sparse**: each word touches a tiny part of the parameters, and only that part is read, trained and written back.

* **Dense backbone in RAM**: a small Llama-style transformer (0.6M to 39M parameters, chosen from the minutes you give and the speed measured on
  your device), trained normally with hand-written, gradient-checked backpropagation.
* **Experts on flash**: one extra residual layer holds millions of experts. An expert is a tiny MLP (e.g. 10 hidden units at width 192 = 3,840
  parameters) whose 8-bit weights, per-row scales and optimizer state fill **exactly one 4096-byte flash page**. 18.2 million of them are 70 billion
  parameters. They live in a **sparse file**: a never-used expert costs no disk, is created on first use (up-projection from a shared random pool,
  down-projection zero, so a new layer is an exact no-op) and the file grows as training proceeds.
* **Routing by hashed n-grams** (default): the last 1, 2 and 3 tokens each pick one expert by hash, so the same context always reaches the same page
  and every occurrence of it trains that page. A context-aware sigmoid gate (from the current hidden state) decides how much each one contributes.
  This follows hashed n-gram memories such as Engram (DeepSeek-AI, arXiv 2601.07372) and the hash n-gram embeddings of the Byte Latent Transformer
  (arXiv 2412.09871); here the slot is a trainable expert in flash instead of a read-only embedding. `--route pk` selects a learned
  product-key router instead (Lample et al. 2019; He 2024, "Mixture of A Million Experts").
* **Train in place**: per step, the pages of the experts used are read, the gradients computed, a row-wise Adagrad step applied (the first step of a
  row scales with its gradient), the weights re-quantised to 8 bit with stochastic rounding and the pages written back, a chunk of 1,024 experts at
  a time so RAM stays small. Nothing else on the device is touched.

## What is measured (4-core Linux PC, TinyStories, same data, same steps)

| model | parameters trained | validation loss | perplexity |
|---|---|---|---|
| dense backbone only (3.0M, width 192) | 3.0M | 2.611 | 13.6 |
| + Flash Experts, 262,144 experts, orders 1+2+3 | 1.0B | **2.367** | **10.7** |

2,000 steps of 16 x 64 tokens (2.05M tokens). That is **21% lower perplexity** from experts that cost 3 flash pages per token. More experts help until the
table is larger than the number of distinct contexts in the data (small backbone, 800 steps: 1K experts 2.880, 4K 2.807, 262K 2.724, 1M 2.715, no
experts 3.187). Contexts: orders 1+2+3 beat 2+3 (2.676 vs 2.724), 3 alone is worst (2.848). Expert learning rate 0.003 beat 0.001 and 0.01.
The learned product-key router gained only about 0.01 to 0.04 here and its advantage shrank with training.

A real **69.98 billion parameter** run (`phone-train big70 --data tinystories.txt --params 70e9 --oversubscribe --width 192 --minutes 25`) on this PC: 1.83M tokens in 23 minutes (1,340 tokens/s), validation loss 2.345 on its own split, a 74.6 GB sparse file of which 2.15 GB was on disk (520,407 experts created), 15.5 GB of flash written. It generates coherent TinyStories text.

## Honest limits

* All 70B parameters are really trained on the device, for free, with no GPU. But each word uses about 0.004% of them. **Quality follows the number of
  tokens your phone processes and the size of the dense backbone, not the file size.** A day of phone training is a few tens of millions of tokens.
  At 70B, most experts will be used few times in that time (`phone-plan` prints it); the benefit comes from frequent contexts first.
* **Flash wear**: each token reads and rewrites up to 3 pages (12 KB). Nonstop training writes terabytes per day; the plan prints it and `--max-wear-gb`
  caps it. Use a charger and keep the phone cool (`--max-temp`, `--min-battery`).
* **Disk**: the file can grow to 75 GB. `phone-plan` shrinks the model to the free space unless you pass `--oversubscribe`; training stops safely when
  free space drops below a reserve (a full disk would kill a memory-mapped writer). Use internal storage, not an SD card (sparse files).
* Tested on a PC, not yet measured on a phone: phone speed and storage speed differ, which is why `phone-plan` measures them.
* A trained model is used with `phone-sample` (NumPy, no KV cache). It is not exported to the fast C engine.

## Tests
`python3 scripts/test_flash_experts.py`: gradients of both routers and of the full model against finite differences, the 8-bit store round trip,
no-op-at-start, learning, resume, and a 70B sparse file.

## Free GPUs (Colab / Kaggle) for dense models
`python3 craft/expertcraft.py plan 125e6` says whether free hardware can train a size and how long; `scaffold 125e6 NAME` writes a PyTorch project
(Llama-style, FSDP, resumable, `colab.ipynb`, `kaggle.ipynb`); `run NAME data|train|export|sample [args] [--background]`, `status NAME`, `stop NAME`.
A dense 70B model needs about 1.1 TB of training memory and cannot be trained on free GPUs. Projects live in `~/expertcraft` (`EXPERTCRAFT_HOME`);
paths outside it are refused.

## Layout
`expertcraft.py` CLI · `phone/flashexperts.py` layer + flash store · `phone/builder.py` trainer core (tokenizer, forward/backward, Adam) ·
`phone/phonetrain.py` plan, create, train, resume, sample · `phone/layout.py` page maths · `template/` PyTorch project for the GPU route.
