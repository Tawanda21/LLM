"""Reward Model for RLHF.

A GPT backbone with the vocabulary head replaced by a scalar regression
head. Trained on human preference pairs using the Bradley-Terry model:

    P(y_w > y_l | x) = σ( r(x, y_w) - r(x, y_l) )
    Loss: L = -log σ(r_chosen - r_rejected)

Reference: InstructGPT — https://arxiv.org/abs/2203.02155
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import precompute_freqs_cis
from src.model.config import ModelConfig
from src.model.normalization import RMSNorm
from src.model.transformer_block import TransformerBlock


class RewardModel(nn.Module):
    """GPT backbone with a scalar reward head.

    Architecture is identical to GPT except:
    - No vocabulary output head
    - Scalar linear head outputs a single reward value per sequence
    - Reward is read from the LAST non-padding token position

    Args:
        config: ModelConfig — same hyperparameters as the policy GPT
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [TransformerBlock(i, config) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.reward_head = nn.Linear(config.dim, 1, bias=False)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.head_dim,
                config.max_seq_len * 2,
                config.rope_theta,
            ),
            persistent=False,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens:  (B, T) token ids
            lengths: (B,) actual sequence lengths (before padding). If None,
                     the reward is read from the last token (index T-1).

        Returns:
            rewards: (B,) scalar reward per sequence
        """
        B, T = tokens.shape
        assert T <= self.config.max_seq_len

        x = self.dropout(self.tok_embeddings(tokens))
        freqs_cis = self.freqs_cis[:T]

        for layer in self.layers:
            x = layer(x, freqs_cis)

        x = self.norm(x)  # (B, T, dim)

        # Extract the representation from the last meaningful token
        if lengths is not None:
            # Gather the hidden state at position (length - 1) for each example
            idx = (lengths - 1).clamp(0, T - 1)  # (B,)
            idx = idx.view(B, 1, 1).expand(B, 1, self.config.dim)
            last = x.gather(dim=1, index=idx).squeeze(1)  # (B, dim)
        else:
            last = x[:, -1, :]  # (B, dim)

        return self.reward_head(last).squeeze(-1)  # (B,)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, gpt_model: nn.Module) -> "RewardModel":
        """Initialise a RewardModel from a trained GPT, copying shared weights.

        The embedding, transformer blocks, and final norm are copied.
        The scalar reward_head starts from random initialisation.

        Args:
            gpt_model: a trained GPT instance

        Returns:
            RewardModel with pre-trained backbone weights loaded.
        """
        rm = cls(gpt_model.config)
        rm.tok_embeddings.load_state_dict(gpt_model.tok_embeddings.state_dict())
        rm.layers.load_state_dict(gpt_model.layers.state_dict())
        rm.norm.load_state_dict(gpt_model.norm.state_dict())
        # reward_head stays randomly initialised — it learns from scratch
        return rm


# ── Loss ─────────────────────────────────────────────────────────────────────


def preference_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bradley-Terry preference loss.

    Maximises the probability that the chosen response is rated higher:
        L = -E[ log σ(r_chosen - r_rejected) ]

    Args:
        reward_chosen:   (B,) scalar rewards for preferred responses
        reward_rejected: (B,) scalar rewards for dispreferred responses

    Returns:
        loss:    scalar cross-entropy loss
        acc:     (B,) 1.0 where model correctly ranks chosen > rejected
    """
    loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()
    acc = (reward_chosen > reward_rejected).float()
    return loss, acc
