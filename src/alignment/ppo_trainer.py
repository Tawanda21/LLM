"""Proximal Policy Optimisation (PPO) for RLHF.

PPO adapts reinforcement learning to fine-tune language models using
a trained reward model as the environment signal.

Algorithm (one iteration):

  Rollout phase:
    1. Sample prompts from dataset
    2. Generate responses using current policy (autoregressive)
    3. Score each response with the reward model → R_RM
    4. Compute per-token KL penalty: KL_t = log π(a_t|s_t) - log π_ref(a_t|s_t)
    5. Reward = R_RM - β * Σ KL_t  (KL prevents reward hacking)

  Update phase:
    6. Compute advantages A = whiten(Reward)
    7. For each minibatch:
       a. Compute log prob ratio: r = exp(log π_θ - log π_old)
       b. Clipped surrogate objective: min(r*A, clip(r, 1-ε, 1+ε)*A)
       c. Maximise surrogate (minimise negative)
       d. Optional: entropy bonus to prevent mode collapse

Note on practicality:
    PPO is the most complex component of this project. The implementation
    here is correct and educational, but getting stable reward improvement
    requires tuning KL coefficient, clip range, and generation parameters.
    DPO is recommended as the first alignment method to run end-to-end.

Reference: InstructGPT — https://arxiv.org/abs/2203.02155
           PPO — https://arxiv.org/abs/1707.06347
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.checkpointing import save_checkpoint
from src.training.optimizer import build_adamw
from src.training.scheduler import cosine_with_warmup

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class PPOConfig:
    """Hyperparameters for a PPO training run."""

    # Generation
    max_new_tokens: int = 128  # tokens to generate per prompt
    temperature: float = 1.0
    top_k: int = 50

    # Rewards
    kl_coef: float = 0.1  # β: KL penalty coefficient
    reward_clip: float = 5.0  # clip reward to [-clip, +clip] for stability

    # PPO update
    clip_range: float = 0.2  # ε: PPO clipping range
    entropy_coef: float = 0.01  # entropy bonus coefficient
    ppo_epochs: int = 1  # update passes per rollout batch
    minibatch_size: int = 2

    # Training
    max_steps: int = 500
    batch_size: int = 4  # prompts per rollout batch
    lr: float = 1e-5  # very low LR for PPO
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 20

    # Logging / saving
    log_every: int = 5
    save_every: int = 100
    checkpoint_dir: str = "checkpoints/ppo"


# ── Per-token log prob helper ─────────────────────────────────────────────────


def compute_token_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token log probs and entropy for response positions.

    Args:
        model:     language model
        input_ids: (B, T)
        labels:    (B, T), -1 for masked positions

    Returns:
        log_probs: (B, T) per-token log-probs (0 at masked positions)
        entropy:   (B, T) per-token entropy (0 at masked positions)
    """
    logits, _ = model(input_ids)  # (B, T, V)
    log_probs_all = F.log_softmax(logits, dim=-1)  # (B, T, V)
    probs_all = log_probs_all.exp()

    mask = labels != -1  # (B, T)
    clamped = labels.clone()
    clamped[~mask] = 0

    gathered = log_probs_all.gather(2, clamped.unsqueeze(2)).squeeze(2)  # (B, T)
    gathered = gathered * mask.float()

    # Per-token entropy H = -Σ p log p
    entropy = -(probs_all * log_probs_all).sum(dim=-1) * mask.float()  # (B, T)

    return gathered, entropy


# ── PPO Trainer ───────────────────────────────────────────────────────────────


class PPOTrainer:
    """RLHF training loop using Proximal Policy Optimisation.

    The training loop alternates between:
      - Rollout: generate responses and score them
      - Update:  improve the policy using the PPO objective

    The KL penalty between the policy and the frozen reference (SFT model)
    prevents reward hacking — the policy cannot drift too far from the
    original fine-tuned behaviour.

    Args:
        policy:       language model to train
        reference:    frozen SFT reference model
        reward_model: trained RewardModel that scores (prompt + response) sequences
        prompts:      list of prompt token tensors to sample from during rollout
        config:       PPOConfig hyperparameters
    """

    def __init__(
        self,
        policy: nn.Module,
        reference: nn.Module,
        reward_model: nn.Module,
        prompts: List[torch.Tensor],
        config: PPOConfig,
    ) -> None:
        self.policy = policy
        self.reference = reference
        self.reward_model = reward_model
        self.prompts = prompts
        self.config = config

        # Freeze reference and reward model
        for model in (reference, reward_model):
            for p in model.parameters():
                p.requires_grad = False
            model.eval()

        self.optimizer = build_adamw(
            policy,
            lr=config.lr,
            weight_decay=config.weight_decay,
            beta1=config.beta1,
            beta2=config.beta2,
            eps=config.eps,
        )
        self.scheduler = cosine_with_warmup(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            min_lr_ratio=0.0,
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def train(self) -> None:
        cfg = self.config
        step = 0

        print(
            f"PPO Training  |  KL coef={cfg.kl_coef}  |  "
            f"clip_range={cfg.clip_range}  |  steps={cfg.max_steps}"
        )

        prompt_idx = 0

        while step < cfg.max_steps:
            # ── Rollout ───────────────────────────────────────────────────────
            prompt_batch = self._sample_prompts(prompt_idx, cfg.batch_size)
            prompt_idx = (prompt_idx + cfg.batch_size) % len(self.prompts)

            rollouts = self._generate_rollouts(prompt_batch)
            rewards = self._compute_rewards(rollouts)

            # ── PPO update ────────────────────────────────────────────────────
            for _ in range(cfg.ppo_epochs):
                loss = self._ppo_update(rollouts, rewards)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % cfg.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                mean_reward = rewards.mean().item()
                print(
                    f"step {step:>5} | reward {mean_reward:+.4f} | "
                    f"loss {loss.item():.4f} | lr {lr:.2e}"
                )

            if step % cfg.save_every == 0:
                path = Path(cfg.checkpoint_dir) / f"ppo_step_{step:05d}.pt"
                save_checkpoint(
                    str(path),
                    self.policy,
                    self.optimizer,
                    self.scheduler,
                    step=step,
                    loss=loss.item(),
                )
                print(f"  Saved → {path}")

        print("PPO training complete.")

    # ── Rollout ───────────────────────────────────────────────────────────────

    def _sample_prompts(self, start: int, n: int) -> List[torch.Tensor]:
        indices = [(start + i) % len(self.prompts) for i in range(n)]
        return [self.prompts[i] for i in indices]

    @torch.no_grad()
    def _generate_rollouts(
        self, prompt_batch: List[torch.Tensor]
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate one response per prompt and record the log probs."""
        cfg = self.config
        rollouts = []

        for prompt in prompt_batch:
            # prompt shape: (1, prompt_len) or (prompt_len,) — normalise to (1, T)
            if prompt.dim() == 1:
                prompt = prompt.unsqueeze(0)

            # Generate response tokens
            full_ids = self.policy.generate(
                prompt,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_k=cfg.top_k,
            )  # (1, prompt_len + response_len)

            prompt_len = prompt.shape[1]
            response_len = full_ids.shape[1] - prompt_len

            # Labels: -1 for prompt, actual ids for response
            labels = full_ids.clone()
            labels[:, :prompt_len] = -1

            # Old log probs under current policy (will be frozen for PPO ratio)
            old_log_probs, _ = compute_token_log_probs(self.policy, full_ids, labels)

            # Reference log probs for KL computation
            ref_log_probs, _ = compute_token_log_probs(self.reference, full_ids, labels)

            rollouts.append(
                {
                    "full_ids": full_ids,
                    "labels": labels,
                    "old_log_probs": old_log_probs.detach(),
                    "ref_log_probs": ref_log_probs.detach(),
                    "prompt_len": prompt_len,
                }
            )

        return rollouts

    @torch.no_grad()
    def _compute_rewards(self, rollouts: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Score each rollout: R = RM_score - β * KL_penalty.

        Returns:
            rewards: (B,) combined scalar reward per rollout
        """
        cfg = self.config
        rewards = []

        for r in rollouts:
            # Reward model score (scalar per sequence)
            rm_score = self.reward_model(r["full_ids"]).squeeze()  # scalar

            # Per-sequence KL: Σ_t [log π(a_t) - log π_ref(a_t)] over response
            mask = r["labels"] != -1
            kl = ((r["old_log_probs"] - r["ref_log_probs"]) * mask.float()).sum()

            # Combined reward
            reward = rm_score - cfg.kl_coef * kl

            # Clip to prevent extreme values destabilising training
            reward = reward.clamp(-cfg.reward_clip, cfg.reward_clip)
            rewards.append(reward)

        rewards = torch.stack(rewards)  # (B,)

        # Whiten rewards: zero mean, unit variance
        if rewards.numel() > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        return rewards

    # ── PPO update ────────────────────────────────────────────────────────────

    def _ppo_update(
        self,
        rollouts: List[Dict[str, torch.Tensor]],
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """One PPO update pass over all rollouts.

        Returns the mean policy loss.
        """
        cfg = self.config
        losses = []

        for i, r in enumerate(rollouts):
            full_ids = r["full_ids"]
            labels = r["labels"]
            old_log_probs = r["old_log_probs"]  # (1, T) per-token, detached
            advantage = rewards[i]  # scalar

            # Current policy log probs (with gradients)
            self.policy.train()
            cur_log_probs, entropy = compute_token_log_probs(
                self.policy, full_ids, labels
            )  # (1, T)

            mask = (labels != -1).float()  # (1, T)

            # Per-token probability ratio r_t = π_θ(a_t) / π_old(a_t)
            log_ratio = (cur_log_probs - old_log_probs) * mask
            ratio = log_ratio.exp()  # (1, T)

            # Clipped PPO surrogate objective (per token)
            unclipped = ratio * advantage
            clipped = ratio.clamp(1 - cfg.clip_range, 1 + cfg.clip_range) * advantage
            policy_loss = -torch.min(unclipped, clipped) * mask  # (1, T)

            # Entropy bonus (encourages exploration)
            entropy_loss = -entropy * mask

            # Total loss (sum over response tokens, mean over batch)
            n_tokens = mask.sum().clamp(min=1)
            loss = (policy_loss + cfg.entropy_coef * entropy_loss).sum() / n_tokens

            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
        return total_loss.detach()
