from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Hyperparameters for the decoder-only transformer.

    All architectural choices default to the LLaMA-style design:
    RoPE positional encoding, RMSNorm, SwiGLU FFN, Grouped Query Attention.
    """

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab_size: int = 32_000

    # ── Model dimensions ─────────────────────────────────────────────────────
    dim: int = 512  # embedding / residual stream dimension
    n_layers: int = 8  # number of transformer blocks
    n_heads: int = 8  # number of query heads
    n_kv_heads: int = 4  # number of key/value heads (GQA; must divide n_heads)

    # ── Sequence length ───────────────────────────────────────────────────────
    max_seq_len: int = 2048

    # ── Feed-forward ─────────────────────────────────────────────────────────
    multiple_of: int = 256  # round FFN hidden dim up to this multiple
    ffn_dim_multiplier: Optional[float] = None  # optional explicit multiplier

    # ── Regularisation ───────────────────────────────────────────────────────
    dropout: float = 0.0

    # ── Normalisation ────────────────────────────────────────────────────────
    norm_eps: float = 1e-6

    # ── RoPE ─────────────────────────────────────────────────────────────────
    rope_theta: float = 10_000.0

    # ── Derived (read-only) ──────────────────────────────────────────────────

    @property
    def head_dim(self) -> int:
        assert self.dim % self.n_heads == 0, (
            f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        )
        return self.dim // self.n_heads

    @property
    def n_rep(self) -> int:
        """Number of Q heads that share each KV head."""
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        return self.n_heads // self.n_kv_heads


# ── Preset configs ────────────────────────────────────────────────────────────


def small_config() -> ModelConfig:
    """~50 M parameters — fast iteration / single GPU."""
    return ModelConfig(
        vocab_size=32_000,
        dim=512,
        n_layers=8,
        n_heads=8,
        n_kv_heads=4,
        max_seq_len=2048,
    )


def medium_config() -> ModelConfig:
    """~350 M parameters — GPT-2 medium scale."""
    return ModelConfig(
        vocab_size=32_000,
        dim=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=8,
        max_seq_len=2048,
    )
