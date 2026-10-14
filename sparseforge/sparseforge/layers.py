import math, torch, torch.nn as nn, torch.nn.functional as F

class QuantLoRALinear(nn.Module):
    """Frozen base weight stored as int8 or packed int4 (per-row absmax scale) + trainable LoRA."""
    def __init__(self, weight, bias=None, bits=8, r=8, alpha=16, compute_dtype=torch.bfloat16):
        super().__init__()
        self.bits, self.out_f, self.in_f = bits, weight.shape[0], weight.shape[1]
        self.compute_dtype = compute_dtype
        w = weight.detach().float()
        qmax = 127 if bits == 8 else 7
        scale = (w.abs().amax(1, keepdim=True) / qmax).clamp(min=1e-8)
        q = torch.clamp(torch.round(w / scale), -qmax, qmax).to(torch.int8)
        if bits == 4:
            assert self.in_f % 2 == 0
            u = (q + 8).to(torch.uint8)
            q = (u[:, 0::2] | (u[:, 1::2] << 4))
        elif bits != 8:
            raise ValueError("bits must be 8 or 4")
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale.to(compute_dtype))
        self.register_buffer("bias", None if bias is None else bias.detach().clone())
        self.r = r
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(r, self.in_f) / math.sqrt(self.in_f))
            self.lora_B = nn.Parameter(torch.zeros(self.out_f, r))
            self.scaling = alpha / r

    def dequant(self):
        q = self.qweight
        if self.bits == 4:
            lo = (q & 0xF).to(torch.int8) - 8; hi = (q >> 4).to(torch.int8) - 8
            q = torch.stack([lo, hi], -1).reshape(self.out_f, self.in_f)
        return q.to(self.compute_dtype) * self.scale

    def forward(self, x):
        y = F.linear(x.to(self.compute_dtype), self.dequant(),
                     None if self.bias is None else self.bias.to(self.compute_dtype))
        if self.r > 0:
            y = y + (x.to(self.lora_A.dtype) @ self.lora_A.t() @ self.lora_B.t() * self.scaling).to(y.dtype)
        return y

class HashMoEFFN(nn.Module):
    """Hash-routed MoE FFN (Hash Layers, Roller et al. 2021): token id -> fixed expert, top-1.
    Each expert has d_ff/E hidden units by default => ~1/E of dense FFN compute per token."""
    def __init__(self, d_model, d_ff, n_experts=4, expert_ff=None, seed=1234):
        super().__init__()
        self.E = n_experts
        h = expert_ff or max(1, d_ff // n_experts)
        self.w1 = nn.Parameter(torch.randn(n_experts, d_model, h) * 0.02)
        self.w2 = nn.Parameter(torch.randn(n_experts, h, d_model) * 0.02)
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("perm", torch.randperm(1 << 20, generator=g) % n_experts)

    def forward(self, x, token_ids):
        B, T, D = x.shape
        xf = x.reshape(-1, D); ids = token_ids.reshape(-1)
        expert = self.perm[ids % self.perm.numel()]
        out = torch.zeros_like(xf)
        for e in range(self.E):
            idx = (expert == e).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            h = F.gelu(xf[idx] @ self.w1[e].to(xf.dtype))
            out.index_copy_(0, idx, h @ self.w2[e].to(xf.dtype))
        return out.reshape(B, T, D)

def sampled_softmax_loss(hidden, weight, targets, n_neg=1024, generator=None):
    """Sparse output layer for training: logits only for targets + shared random negatives
    (sampled softmax with uniform proposal, logQ correction). Use full softmax for eval."""
    V = weight.shape[0]
    h = hidden.reshape(-1, hidden.shape[-1]); t = targets.reshape(-1)
    if n_neg >= V:
        return F.cross_entropy(h @ weight.t(), t)
    neg = torch.randint(0, V, (n_neg,), generator=generator, device=h.device)
    cand = torch.cat([t.unique(), neg]).unique()
    logits = h @ weight[cand].t().to(h.dtype)
    # map targets into candidate index
    pos = torch.searchsorted(cand, t)
    logq = math.log(cand.numel() / V)
    return F.cross_entropy(logits.float() - logq, pos)
