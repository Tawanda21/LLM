import torch
import torch.nn as nn

from .attention import Attention
from .config import ModelConfig
from .feedforward import FeedForward
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    """A single decoder-only transformer block with pre-norm residual connections.

    Layout (pre-LN, as in LLaMA):
        x ← x + Attention( RMSNorm(x) )
        x ← x + FFN(       RMSNorm(x) )

    Pre-norm (normalise before the sub-layer, not after) is more training-stable
    than the original post-norm formulation because gradients flow cleanly
    through the residual pathway without first passing through a normaliser.
    """

    def __init__(self, layer_id: int, config: ModelConfig) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (B, T, dim)   — residual stream input
            freqs_cis:  (T, head_dim // 2) complex — RoPE frequencies

        Returns:
            (B, T, dim)
        """
        # Attention sub-layer with pre-norm + residual
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        # FFN sub-layer with pre-norm + residual
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
