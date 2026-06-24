import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation.

    Unlike LayerNorm, RMSNorm skips mean subtraction (re-centering) and only
    performs re-scaling by the RMS of the activations:

        RMSNorm(x) = x / RMS(x) * weight
        RMS(x)     = sqrt( mean(x²) + eps )

    This makes it ~15 % faster than LayerNorm while achieving comparable
    training stability.  Used in LLaMA, Mistral, Gemma, and PaLM.

    Reference: Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
               https://arxiv.org/abs/1910.07467
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # learnable per-channel scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim)
        Returns:
            Normalised tensor of the same shape.
        """
        # Compute 1/RMS via rsqrt for numerical efficiency
        # rsqrt(a) = 1 / sqrt(a)
        rrms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rrms) * self.weight
