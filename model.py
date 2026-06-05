import torch
import torch.nn as nn

from components.block import TransformerBlock
from components.rope import build_rope_cache
from components.block import RMSNorm

class MiniQwen(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_dim=128,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        max_seq_len=256,
        tie_word_embeddings=True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # token embedding
        self.embed = nn.Embedding(vocab_size, hidden_dim)

        # transformer stack
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, num_kv_heads)
            for _ in range(num_layers)
        ])

        # final norm
        self.norm = RMSNorm(hidden_dim)

        # output head
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # weight tying
        if tie_word_embeddings:
            self.lm_head.weight = self.embed.weight

    def forward(self, x):
        """
        x: (batch, seq)
        """

        b, s = x.shape

        # embeddings
        x = self.embed(x)

        # RoPE cache is recomputed each forward for simplicity.
        cos, sin = build_rope_cache(s, self.blocks[0].attn.head_dim, device=x.device)

        # transformer blocks
        for block in self.blocks:
            x = block(x, cos, sin)

        # final norm
        x = self.norm(x)

        # logits
        logits = self.lm_head(x)

        return logits
    

if __name__ == "__main__":

    vocab_size = 1000
    model = MiniQwen(
        vocab_size=vocab_size,
        hidden_dim=128,
        num_layers=2
    )

    x = torch.randint(0, vocab_size, (2, 16))

    logits = model(x)

    print("Input shape:", x.shape)
    print("Logits shape:", logits.shape)
    print("OK")
