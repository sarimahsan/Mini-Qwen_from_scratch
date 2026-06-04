import torch
import math

def build_rope_cache(seq_len, head_dim, base=10000, device="cpu"):
    """
    Returns cos and sin matrices for RoPE
    shape: (seq_len, head_dim)
    """

    assert head_dim % 2 == 0, "head_dim must be even"

    # Step 1: compute frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    # Step 2: positions
    positions = torch.arange(seq_len, device=device).float()

    # Step 3: outer product → (seq_len, head_dim/2)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)

    # Step 4: expand to full dim
    cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
    sin = torch.sin(freqs).repeat_interleave(2, dim=-1)

    return cos, sin



def apply_rope(x, cos, sin):
    """
    x: (batch, heads, seq, head_dim)
    cos/sin: (seq, head_dim)
    """

    cos = cos[None, None, :, :]  # broadcast
    sin = sin[None, None, :, :]

    x1 = x[..., ::2]   # even dims
    x2 = x[..., 1::2]  # odd dims

    out = torch.empty_like(x)

    out[..., ::2] = x1 * cos[..., ::2] - x2 * sin[..., ::2]
    out[..., 1::2] = x1 * sin[..., ::2] + x2 * cos[..., ::2]

    return out

# batch = 2
# seq_len = 16
# num_heads = 4
# head_dim = 64

# Q = torch.randn(batch, num_heads, seq_len, head_dim)
# K = torch.randn(batch, num_heads, seq_len, head_dim)
# V = torch.randn(batch, num_heads, seq_len, head_dim)

# cos, sin = build_rope_cache(seq_len, head_dim)

# Q_rope = apply_rope(Q, cos, sin)
# K_rope = apply_rope(K, cos, sin)


# print("Q shape:", Q.shape)
# print("Q_rope shape:", Q_rope.shape)

# print("\nOriginal Q[0,0,0,:5]:")
# print(Q[0, 0, 0, :5])

# print("\nRoPE Q[0,0,0,:5]:")
# print(Q_rope[0, 0, 0, :5])

# print("\nRoPE Q[0,0,1,:5] (next position):")
# print(Q_rope[0, 0, 1, :5])