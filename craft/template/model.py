"""Expert Craft model: a Llama-style decoder (RMSNorm, RoPE, GQA attention, SwiGLU).
Same code from 1M to 70B parameters; only config.json changes."""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_ROPE = {}


def rope_tables(T, D, device, base):
    key = (T, D, str(device), base)
    if key not in _ROPE:
        inv = 1.0 / (base ** (torch.arange(0, D, 2, device=device, dtype=torch.float32) / D))
        f = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv)
        _ROPE[key] = (f.cos()[None, None], f.sin()[None, None])
    return _ROPE[key]


def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2).type_as(x)


class RMSNorm(nn.Module):
    def __init__(s, d, eps=1e-5):
        super().__init__()
        s.weight = nn.Parameter(torch.ones(d))
        s.eps = eps

    def forward(s, x):
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + s.eps)).type_as(x) * s.weight


class Attention(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.h, s.kv = c["n_heads"], c["n_kv_heads"]
        s.hd = c["d_model"] // s.h
        s.base = c.get("rope_base", 10000.0)
        s.q = nn.Linear(c["d_model"], s.h * s.hd, bias=False)
        s.k = nn.Linear(c["d_model"], s.kv * s.hd, bias=False)
        s.v = nn.Linear(c["d_model"], s.kv * s.hd, bias=False)
        s.o = nn.Linear(s.h * s.hd, c["d_model"], bias=False)

    def forward(s, x):
        B, T, _ = x.shape
        q = s.q(x).view(B, T, s.h, s.hd).transpose(1, 2)
        k = s.k(x).view(B, T, s.kv, s.hd).transpose(1, 2)
        v = s.v(x).view(B, T, s.kv, s.hd).transpose(1, 2)
        cos, sin = rope_tables(T, s.hd, x.device, s.base)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if s.kv != s.h:
            r = s.h // s.kv
            k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return s.o(y.transpose(1, 2).reshape(B, T, -1))


class MLP(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.gate = nn.Linear(c["d_model"], c["d_ff"], bias=False)
        s.up = nn.Linear(c["d_model"], c["d_ff"], bias=False)
        s.down = nn.Linear(c["d_ff"], c["d_model"], bias=False)

    def forward(s, x):
        return s.down(F.silu(s.gate(x)) * s.up(x))


class Block(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.n1, s.n2 = RMSNorm(c["d_model"]), RMSNorm(c["d_model"])
        s.attn, s.mlp = Attention(c), MLP(c)

    def forward(s, x):
        x = x + s.attn(s.n1(x))
        return x + s.mlp(s.n2(x))


class GPT(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.c = c
        s.emb = nn.Embedding(c["vocab_size"], c["d_model"])
        s.blocks = nn.ModuleList(Block(c) for _ in range(c["n_layers"]))
        s.norm = RMSNorm(c["d_model"])
        s.head = nn.Linear(c["d_model"], c["vocab_size"], bias=False)
        if c.get("tie_embeddings"):
            s.head.weight = s.emb.weight
        s.grad_ckpt = bool(c.get("grad_ckpt"))
        s.init_weights()

    @torch.no_grad()
    def init_weights(s):
        """Also called after building on the 'meta' device (huge models), so it must set every tensor."""
        std = 0.02
        out_std = std / (2 * s.c["n_layers"]) ** 0.5  # scaled residual projections (GPT-2 style)
        for name, p in s.named_parameters():
            if name.endswith("n1.weight") or name.endswith("n2.weight") or name == "norm.weight":
                p.fill_(1.0)
            elif name.endswith("attn.o.weight") or name.endswith("mlp.down.weight"):
                p.normal_(0.0, out_std)
            else:
                p.normal_(0.0, std)

    def forward(s, idx, targets=None):
        x = s.emb(idx)
        for b in s.blocks:
            x = checkpoint(b, x, use_reentrant=False) if (s.grad_ckpt and s.training) else b(x)
        logits = s.head(s.norm(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))

    @torch.no_grad()
    def generate(s, idx, n=200, temperature=0.8, top_k=50, stop_id=None):
        for _ in range(n):
            logits, _ = s(idx[:, -s.c["max_seq_len"]:])
            logits = logits[:, -1].float() / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if stop_id is not None and int(nxt[0, 0]) == stop_id:
                break
        return idx
