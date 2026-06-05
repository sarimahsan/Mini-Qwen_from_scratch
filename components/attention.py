import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from components.rope import apply_rope, build_rope_cache
except ModuleNotFoundError:
    from rope import apply_rope, build_rope_cache

# ---------------------------
# RMSNorm for QK Normalization
# ---------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

# ---------------------------
# KV repetition (GQA)
# ---------------------------
def repeat_kv(x, n_rep):
    """
    x: (batch, kv_heads, seq, head_dim)
    returns: (batch, kv_heads * n_rep, seq, head_dim)
    """

    if n_rep == 1:
        return x

    b, h, s, d = x.shape

    x = x[:, :, None, :, :]            # (b, h, 1, s, d)
    x = x.expand(b, h, n_rep, s, d)    # repeat
    x = x.reshape(b, h * n_rep, s, d)

    return x


# ---------------------------
# Qwen-GQA Attention
# ---------------------------
class Qwen3Attention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads):
        super().__init__()

        assert num_heads % num_kv_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        self.head_dim = hidden_dim // num_heads
        self.n_rep = num_heads // num_kv_heads

        # projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)

        # QK norm using learnable RMSNorm per head
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape

        # QKV projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape
        q = q.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK norm (RMSNorm along head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA repeat
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Causal Attention using PyTorch's native optimized SDPA
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True
        )

        # merge heads
        out = out.transpose(1, 2).contiguous().view(b, s, self.hidden_dim)

        return self.out_proj(out)


# ---------------------------
# Quick test
# ---------------------------
if __name__ == "__main__":

    batch = 2
    seq_len = 16
    hidden_dim = 128
    num_heads = 8
    num_kv_heads = 2

    x = torch.randn(batch, seq_len, hidden_dim)

    model = Qwen3Attention(hidden_dim, num_heads, num_kv_heads)

    cos, sin = build_rope_cache(seq_len, model.head_dim)

    out = model(x, cos, sin)

    print("Output shape:", out.shape)
    print("OK")