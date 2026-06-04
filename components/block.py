import torch
import torch.nn as nn

# from norm import RMSProp
# from rope import apply_rope
try:
    from components.attention import Qwen3Attention
    from components.feedforward import SwiGLUFeedForward
except ModuleNotFoundError:
    from attention import Qwen3Attention
    from feedforward import SwiGLUFeedForward

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return self.weight * x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads):
        super().__init__()

        # attention
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = Qwen3Attention(hidden_dim, num_heads, num_kv_heads)

        # feedforward
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = SwiGLUFeedForward(hidden_dim)

    def forward(self, x, cos, sin):
        # ----------------------
        # Attention block
        # ----------------------
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, cos, sin)
        x = x + residual

        # ----------------------
        # FFN block
        # ----------------------
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + residual

        return x
    


if __name__ == "__main__":

    batch = 2
    seq_len = 16
    hidden_dim = 128

    x = torch.randn(batch, seq_len, hidden_dim)

    num_heads = 8
    num_kv_heads = 2

    # fake RoPE cache
    cos = torch.randn(seq_len, hidden_dim // num_heads)
    sin = torch.randn(seq_len, hidden_dim // num_heads)

    block = TransformerBlock(hidden_dim, num_heads, num_kv_heads)

    out = block(x, cos, sin)

    print("Output shape:", out.shape)
    print("OK")