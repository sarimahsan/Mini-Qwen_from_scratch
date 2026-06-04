import torch
import torch.nn as nn
import math
try:
    from components.rope import apply_rope, build_rope_cache
except ModuleNotFoundError:
    from rope import apply_rope, build_rope_cache
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
# QK normalization (Qwen trick)
# ---------------------------
def qk_norm(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

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

        # QK norm
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-6)
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # RoPE
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # attention
        scale = self.head_dim ** -0.5

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)

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