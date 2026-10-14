import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .layers import QuantLoRALinear, HashMoEFFN, sampled_softmax_loss

class Block(nn.Module):
    def __init__(self, d, h, d_ff, moe=0):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3*d), nn.Linear(d, d)
        self.moe = moe
        if moe:
            self.ffn = HashMoEFFN(d, d_ff, moe)
        else:
            self.fc1, self.fc2 = nn.Linear(d, d_ff), nn.Linear(d_ff, d)

    def forward(self, x, ids):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, -1)
        q, k, v = [t.reshape(B, T, self.h, -1).transpose(1, 2) for t in (q, k, v)]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D)).to(x.dtype)
        y = self.ln2(x)
        y = self.ffn(y, ids) if self.moe else self.fc2(F.gelu(self.fc1(y)))
        return x + y.to(x.dtype)

class TinyGPT(nn.Module):
    def __init__(self, vocab=256, d=384, n_layers=6, n_heads=6, d_ff=None, ctx=256, moe=0):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff or 4*d, ctx=ctx, moe=moe)
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, d_ff or 4*d, moe) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d)
        self.grad_ckpt = False
        self.sparse_out_neg = 0

    def forward(self, ids, targets=None):
        x = self.tok(ids) + self.pos(torch.arange(ids.shape[1]))
        for b in self.blocks:
            x = checkpoint(b, x, ids, use_reentrant=False) if (self.grad_ckpt and self.training) else b(x, ids)
        x = self.lnf(x)
        if targets is None:
            return x @ self.tok.weight.t()
        if self.sparse_out_neg and self.training:
            return sampled_softmax_loss(x, self.tok.weight, targets, self.sparse_out_neg)
        return F.cross_entropy((x @ self.tok.weight.t().to(x.dtype)).float().reshape(-1, self.cfg["vocab"]), targets.reshape(-1))

def apply_lora_quant(model, bits=8, r=8, alpha=16, compute_dtype=torch.bfloat16):
    """Freeze all params, replace attention/FFN Linear layers with quantized base + LoRA."""
    for p in model.parameters():
        p.requires_grad_(False)
    for b in model.blocks:
        for name in ["qkv", "proj", "fc1", "fc2"]:
            if hasattr(b, name) and isinstance(getattr(b, name), nn.Linear):
                l = getattr(b, name)
                setattr(b, name, QuantLoRALinear(l.weight, l.bias, bits, r, alpha, compute_dtype))
    return model

def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buf = sum(b.numel() * b.element_size() for b in model.buffers())
    return {"total_params": tot, "trainable_params": tr, "buffer_bytes": buf}
