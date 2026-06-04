import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFeedForward(nn.Module):
    def __init__(self, hidden_dim, ffn_dim=None):
        super().__init__()

        if ffn_dim is None:
            ffn_dim = int(hidden_dim * 4)  # standard LLM expansion

        # two projections (gate + up)
        self.w_gate = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.w_up = nn.Linear(hidden_dim, ffn_dim, bias=False)

        # down projection
        self.w_down = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        """
        x: (batch, seq, hidden_dim)
        """

        gate = self.w_gate(x)
        up = self.w_up(x)

        # SwiGLU activation
        x = F.silu(gate) * up

        return self.w_down(x)
    