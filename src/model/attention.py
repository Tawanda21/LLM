from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

# ── RoPE helpers ──────────────────────────────────────────────────────────────


def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10_000.0,
) -> torch.Tensor:
    """Precompute complex-valued rotary frequency tensor.

    Each position gets a unique rotation angle per head-dimension pair:
        θ_i = 1 / (theta ^ (2i / head_dim))   for i in [0, head_dim/2)
        freq[pos, i] = pos * θ_i
        freqs_cis[pos, i] = e^(j * freq[pos, i])   (complex polar form)

    Args:
        head_dim:    per-head dimension (must be even)
        max_seq_len: maximum sequence length to precompute
        theta:       base for the geometric frequency sequence (default 10 000)

    Returns:
        Complex tensor of shape (max_seq_len, head_dim // 2).
    """
    # θ_i for i = 0, 2, 4, ..., head_dim-2
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # Position indices: 0, 1, ..., max_seq_len-1
    positions = torch.arange(max_seq_len)
    # Outer product → (max_seq_len, head_dim // 2)
    freqs = torch.outer(positions, freqs)
    # Convert to unit-magnitude complex numbers e^(j*freq)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rotate query and key vectors by their position-dependent angles.

    Multiplication in complex space implements a 2-D rotation of each
    consecutive pair of real dimensions — this is what makes RoPE encode
    relative position in the dot-product attention scores.

    Args:
        xq:        (B, T, n_heads,    head_dim)
        xk:        (B, T, n_kv_heads, head_dim)
        freqs_cis: (T, head_dim // 2) complex

    Returns:
        Rotated xq and xk — same shapes and dtype as inputs.
    """
    # Reshape last dim from head_dim real → head_dim//2 complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast freqs over batch and head dims: (T, D) → (1, T, 1, D)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # Complex multiply = rotation; convert back to real and flatten
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand K/V heads to match the number of Q heads for GQA.

    Each KV head is repeated n_rep times so the standard attention
    matmul works without any special-casing.

    Args:
        x:     (B, T, n_kv_heads, head_dim)
        n_rep: n_heads // n_kv_heads

    Returns:
        (B, T, n_kv_heads * n_rep, head_dim)  ==  (B, T, n_heads, head_dim)
    """
    if n_rep == 1:
        return x  # already MHA — nothing to do
    B, T, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]  # (B, T, n_kv, 1, D)
        .expand(B, T, n_kv_heads, n_rep, head_dim)  # (B, T, n_kv, n_rep, D)
        .reshape(B, T, n_kv_heads * n_rep, head_dim)  # (B, T, n_heads, D)
    )


# ── Attention module ──────────────────────────────────────────────────────────


class Attention(nn.Module):
    """Causal Multi-Head Attention with Grouped Query Attention (GQA) and RoPE.

    GQA groups Q heads so that multiple Q heads share one K/V head, reducing
    KV-cache size at inference.  Special cases:
        n_kv_heads == n_heads  →  standard Multi-Head Attention (MHA)
        n_kv_heads == 1        →  Multi-Query Attention (MQA)

    Uses torch.nn.functional.scaled_dot_product_attention which dispatches to
    Flash Attention on CUDA when available.

    Reference:
        GQA: https://arxiv.org/abs/2305.13245
        RoPE: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_rep
        self.head_dim = config.head_dim

        # Projections — no bias (standard in modern LLMs)
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (B, T, dim)
            freqs_cis:  (T, head_dim // 2) complex — precomputed RoPE frequencies

        Returns:
            (B, T, dim)
        """
        B, T, _ = x.shape

        # ── Project ───────────────────────────────────────────────────────────
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)

        # ── RoPE ──────────────────────────────────────────────────────────────
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # ── Expand KV heads to match Q heads (GQA) ────────────────────────────
        xk = repeat_kv(xk, self.n_rep)  # (B, T, n_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)

        # ── Attention ─────────────────────────────────────────────────────────
        # Transpose to (B, n_heads, T, head_dim) — shape expected by F.sdpa
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Flash Attention on CUDA; falls back to manual on CPU.
        # is_causal=True handles the autoregressive mask automatically.
        out = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )

        # ── Merge heads and project ───────────────────────────────────────────
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.resid_dropout(self.wo(out))
