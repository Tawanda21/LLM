import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network.

    Computes:
        FFN(x) = W2( SiLU(W1(x)) ⊙ W3(x) )

    where ⊙ is element-wise multiplication.  W1 is the "gate" projection,
    W3 is the "up" projection, and W2 is the "down" projection.

    Hidden dimension follows the LLaMA formula:
        hidden = round_up( (8/3) × dim,  multiple_of )

    No bias terms anywhere — consistent with modern LLM design.

    References:
        SwiGLU: Noam Shazeer, "GLU Variants Improve Transformers" (2020)
                https://arxiv.org/abs/2002.05202
        LLaMA sizing: https://arxiv.org/abs/2302.13971
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        # ── Hidden dimension ─────────────────────────────────────────────────
        hidden_dim = int(2 * 4 * config.dim / 3)  # ≈ (8/3) × dim
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * config.dim)

        # Round up to the nearest multiple_of for memory alignment
        m = config.multiple_of
        hidden_dim = m * ((hidden_dim + m - 1) // m)

        # ── Layers ───────────────────────────────────────────────────────────
        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)  # gate
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)  # down
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)  # up
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
        Returns:
            (B, T, dim)
        """
        # SiLU(gate) ⊙ up  →  down
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
