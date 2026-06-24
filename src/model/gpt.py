import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import precompute_freqs_cis
from .config import ModelConfig
from .normalization import RMSNorm
from .transformer_block import TransformerBlock


class GPT(nn.Module):
    """Decoder-only transformer language model.

    Full architecture:
        Token Embedding (vocab_size → dim)
        ↓
        N × TransformerBlock
            RMSNorm → GQA Attention (RoPE) → residual
            RMSNorm → SwiGLU FFN           → residual
        ↓
        RMSNorm
        ↓
        Linear head (dim → vocab_size)   [weight-tied to token embedding]

    Design follows LLaMA: RoPE, RMSNorm, SwiGLU, GQA, no bias terms,
    weight-tied embeddings, GPT-2-style residual scaling.

    References:
        LLaMA: https://arxiv.org/abs/2302.13971
        GPT-2 weight init: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # ── Layers ────────────────────────────────────────────────────────────
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(i, config) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # ── Weight tying ──────────────────────────────────────────────────────
        # Sharing the embedding and output weight matrix reduces parameter count
        # and consistently improves perplexity in practice.
        self.tok_embeddings.weight = self.output.weight

        # ── RoPE frequencies (non-trainable buffer) ───────────────────────────
        # Precompute 2× max_seq_len so inference can extend context cheaply.
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.head_dim,
                config.max_seq_len * 2,
                config.rope_theta,
            ),
            persistent=False,
        )

        # ── Weight initialisation ─────────────────────────────────────────────
        self.apply(self._init_weights)

        # Scale the output projections of attention and FFN by 1/√(2·n_layers).
        # This keeps the residual stream variance stable at initialisation as
        # depth increases (GPT-2 paper, section 2.3).
        scale = 0.02 / math.sqrt(2 * config.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=scale)

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        tokens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens:  (B, T) — input token ids
            targets: (B, T) — target token ids for loss computation, or None

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalar cross-entropy loss, or None if targets is None
        """
        B, T = tokens.shape
        assert T <= self.config.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"
        )

        x = self.dropout(self.tok_embeddings(tokens))  # (B, T, dim)
        freqs_cis = self.freqs_cis[:T]

        for layer in self.layers:
            x = layer(x, freqs_cis)

        x = self.norm(x)
        logits = self.output(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten to (B·T, vocab_size) for F.cross_entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # positions masked with -1 are skipped
            )

        return logits, loss

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Auto-regressively sample tokens from the model.

        Args:
            idx:            (B, T) seed token ids
            max_new_tokens: number of new tokens to generate
            temperature:    softmax temperature (>1 = more random, <1 = sharper)
            top_k:          if set, restrict sampling to the top-k logits

        Returns:
            (B, T + max_new_tokens) token ids
        """
        for _ in range(max_new_tokens):
            # Crop context to max_seq_len (sliding window)
            idx_cond = idx[:, -self.config.max_seq_len :]
            logits, _ = self(idx_cond)
            # Take logits at the last position and apply temperature
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Zero out logits below the k-th largest value
                logits[logits < topk_vals[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    # ── Utilities ─────────────────────────────────────────────────────────────

    def num_params(self, exclude_embedding: bool = True) -> int:
        """Total trainable parameter count.

        Args:
            exclude_embedding: exclude the tied embedding/output weight
                               (conventional to report non-embedding params for LLMs)
        """
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.tok_embeddings.weight.numel()
        return n

    @classmethod
    def from_config(cls, config: ModelConfig) -> "GPT":
        return cls(config)
