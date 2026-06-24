"""Direct Preference Optimisation (DPO).

DPO is an elegant simplification of RLHF: it directly optimises the
policy on preference pairs without training a separate reward model.

The key insight is that the optimal RLHF policy can be expressed as a
closed-form function of the reference policy, which leads to a loss
that only needs the policy and reference model — no RL loop required.

DPO loss:
    L = -E[ log σ( β * (log π_θ/π_ref for chosen - log π_θ/π_ref for rejected) ) ]

Where:
    π_θ  = policy model (being trained)
    π_ref = reference model (frozen SFT model)
    β    = temperature controlling deviation from reference (typically 0.1–0.5)

Reference: Rafailov et al., 2023 — https://arxiv.org/abs/2305.18290
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.training.checkpointing import save_checkpoint
from src.training.optimizer import build_adamw
from src.training.scheduler import cosine_with_warmup

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class DPOConfig:
    """Hyperparameters for a DPO training run."""

    # Core DPO
    beta: float = 0.1  # KL penalty coefficient (lower = more deviation allowed)

    # Training
    max_steps: int = 1_000
    batch_size: int = 2
    lr: float = 5e-5  # much lower than SFT — DPO is sensitive to LR
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 50

    # Sequence lengths
    max_seq_len: int = 512

    # Logging / saving
    log_every: int = 10
    save_every: int = 200
    checkpoint_dir: str = "checkpoints/dpo"
    mixed_precision: str = "bf16"

    # W&B
    use_wandb: bool = False
    wandb_project: str = "llm-from-scratch"


# ── Dataset ───────────────────────────────────────────────────────────────────


class DPODataset(Dataset):
    """Preference dataset for DPO training.

    Each example is a (prompt, chosen, rejected) triplet. Both chosen and
    rejected responses are tokenised as full sequences and stored with their
    per-token labels.

    Crucially, the DPO labels unmask ONE MORE position than SFT labels:
    we include the prediction of the first response token r0 (position
    len(prompt_ids) in labels), because we need log P(r0 | prompt) as
    part of the full response log-probability.

    Args:
        tokenizer:   BPETokenizer instance
        examples:    list of dicts with "prompt", "chosen", "rejected" keys
        max_seq_len: maximum total sequence length
        format_fn:   optional callable(example) → (prompt_str, chosen_str, rejected_str)
    """

    def __init__(
        self,
        tokenizer,
        examples: List[Dict],
        max_seq_len: int = 512,
        format_fn: Optional[Callable] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.format_fn = format_fn or _default_format
        self._data = self._preprocess(examples)

    def _preprocess(self, examples: List[Dict]) -> List[Dict]:
        data, skipped = [], 0
        for ex in examples:
            try:
                item = self._tokenize(ex)
                if item is not None:
                    data.append(item)
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        if skipped:
            print(f"  DPODataset: skipped {skipped} examples (too long / malformed)")
        print(f"  DPODataset: {len(data)} preference pairs ready.")
        return data

    def _tokenize(self, example: Dict) -> Optional[Dict]:
        prompt_str, chosen_str, rejected_str = self.format_fn(example)

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        chosen_ids = self.tokenizer.encode(chosen_str, add_special_tokens=False)
        rejected_ids = self.tokenizer.encode(rejected_str, add_special_tokens=False)

        def _pack(response_ids):
            ids = (
                [self.tokenizer.bos_id]
                + prompt_ids
                + response_ids
                + [self.tokenizer.eos_id]
            )
            return ids

        chosen_full = _pack(chosen_ids)
        rejected_full = _pack(rejected_ids)

        if (
            len(chosen_full) > self.max_seq_len + 1
            or len(rejected_full) > self.max_seq_len + 1
        ):
            return None

        if not chosen_ids or not rejected_ids:
            return None

        def _make_tensors(full_ids):
            pad_needed = (self.max_seq_len + 1) - len(full_ids)
            full_ids = full_ids + [self.tokenizer.pad_id] * pad_needed

            input_ids = torch.tensor(full_ids[:-1], dtype=torch.long)
            labels = torch.tensor(full_ids[1:], dtype=torch.long)

            # DPO masking: mask labels before len(prompt_ids)
            # This preserves P(r0 | prompt), unlike SFT which masks it.
            labels[: len(prompt_ids)] = -1
            labels[input_ids == self.tokenizer.pad_id] = -1

            return input_ids, labels

        chosen_input, chosen_labels = _make_tensors(chosen_full)
        rejected_input, rejected_labels = _make_tensors(rejected_full)

        return {
            "chosen_input_ids": chosen_input,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input,
            "rejected_labels": rejected_labels,
        }

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict:
        return self._data[idx]

    @classmethod
    def from_jsonl(cls, tokenizer, path: Union[str, Path], **kwargs) -> "DPODataset":
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return cls(tokenizer, examples, **kwargs)


def _default_format(example: Dict) -> Tuple[str, str, str]:
    """Default: expect keys 'prompt', 'chosen', 'rejected'."""
    return example["prompt"], example["chosen"], example["rejected"]


# ── Core DPO functions ────────────────────────────────────────────────────────


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the sum of per-token log-probabilities for unmasked positions.

    For each sequence, returns Σ log π(a_t | s_t) over response tokens.

    Args:
        model:     the language model (policy or reference)
        input_ids: (B, T) token ids
        labels:    (B, T) target token ids; -1 positions are ignored

    Returns:
        (B,) total log-probability per sequence over unmasked positions
    """
    logits, _ = model(input_ids)  # (B, T, V)
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

    response_mask = labels != -1  # (B, T)
    clamped_labels = labels.clone()
    clamped_labels[~response_mask] = 0  # avoid gather error on -1

    # Gather log prob for each actual target token
    gathered = log_probs.gather(dim=2, index=clamped_labels.unsqueeze(2)).squeeze(
        2
    )  # (B, T)

    # Zero out masked positions and sum over sequence
    return (gathered * response_mask.float()).sum(dim=1)  # (B,)


def dpo_loss(
    policy_log_probs_chosen: torch.Tensor,
    policy_log_probs_rejected: torch.Tensor,
    ref_log_probs_chosen: torch.Tensor,
    ref_log_probs_rejected: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DPO loss and implicit reward margins.

    The loss is:
        L = -log σ( β * (log π/π_ref for chosen - log π/π_ref for rejected) )

    Intuitively: the policy should assign higher implicit reward to the
    chosen response than to the rejected response, where implicit reward
    is defined as β * log(π/π_ref).

    Args:
        policy_log_probs_chosen:   (B,) log π_θ(y_w | x)
        policy_log_probs_rejected: (B,) log π_θ(y_l | x)
        ref_log_probs_chosen:      (B,) log π_ref(y_w | x)
        ref_log_probs_rejected:    (B,) log π_ref(y_l | x)
        beta:                      KL penalty coefficient

    Returns:
        loss:             scalar DPO loss
        chosen_rewards:   (B,) implicit reward for chosen responses
        rejected_rewards: (B,) implicit reward for rejected responses
    """
    # Implicit reward = β * log(π_θ / π_ref)
    chosen_rewards = beta * (policy_log_probs_chosen - ref_log_probs_chosen).detach()
    rejected_rewards = (
        beta * (policy_log_probs_rejected - ref_log_probs_rejected).detach()
    )

    # Log ratio difference (the DPO signal)
    pi_log_ratios = policy_log_probs_chosen - policy_log_probs_rejected
    ref_log_ratios = ref_log_probs_chosen - ref_log_probs_rejected

    losses = -F.logsigmoid(beta * (pi_log_ratios - ref_log_ratios))
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


# ── Trainer ───────────────────────────────────────────────────────────────────


class DPOTrainer:
    """DPO training loop.

    Trains the policy model to prefer chosen responses over rejected ones
    without requiring a separate reward model.

    Args:
        policy:    the model being fine-tuned (SFT model with trainable params)
        reference: the frozen SFT reference model
        dataset:   DPODataset of preference pairs
        config:    DPOConfig hyperparameters
    """

    def __init__(
        self,
        policy: nn.Module,
        reference: nn.Module,
        dataset: DPODataset,
        config: DPOConfig,
    ) -> None:
        self.policy = policy
        self.reference = reference
        self.config = config

        # Freeze the reference model
        for p in self.reference.parameters():
            p.requires_grad = False
        self.reference.eval()

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
        self.loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
        )

    def train(self) -> None:
        cfg = self.config
        step = 0
        data_iter = self._cycle(self.loader)

        print(
            f"DPO Training  |  β={cfg.beta}  |  "
            f"steps={cfg.max_steps}  |  lr={cfg.lr:.1e}"
        )

        while step < cfg.max_steps:
            batch = next(data_iter)
            self.policy.train()

            chosen_ids = batch["chosen_input_ids"]
            chosen_lbl = batch["chosen_labels"]
            rejected_ids = batch["rejected_input_ids"]
            rejected_lbl = batch["rejected_labels"]

            # Policy forward (with gradients)
            pol_chosen_lp = compute_log_probs(self.policy, chosen_ids, chosen_lbl)
            pol_rejected_lp = compute_log_probs(self.policy, rejected_ids, rejected_lbl)

            # Reference forward (no gradients)
            with torch.no_grad():
                ref_chosen_lp = compute_log_probs(
                    self.reference, chosen_ids, chosen_lbl
                )
                ref_rejected_lp = compute_log_probs(
                    self.reference, rejected_ids, rejected_lbl
                )

            loss, chosen_rew, rejected_rew = dpo_loss(
                pol_chosen_lp,
                pol_rejected_lp,
                ref_chosen_lp,
                ref_rejected_lp,
                beta=cfg.beta,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % cfg.log_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                margin = (chosen_rew - rejected_rew).item()
                print(
                    f"step {step:>5} | loss {loss.item():.4f} | "
                    f"margin {margin:+.3f} | lr {lr:.2e}"
                )

            if step % cfg.save_every == 0:
                path = Path(cfg.checkpoint_dir) / f"dpo_step_{step:06d}.pt"
                save_checkpoint(
                    str(path),
                    self.policy,
                    self.optimizer,
                    self.scheduler,
                    step=step,
                    loss=loss.item(),
                )
                print(f"  Saved → {path}")

        print("DPO training complete.")

    @staticmethod
    def _cycle(loader: DataLoader):
        while True:
            yield from loader


# ── Synthetic preference data ─────────────────────────────────────────────────


def make_preference_examples(n: int = 200) -> List[Dict]:
    """Synthetic (prompt, chosen, rejected) pairs for testing DPO.

    chosen  = a well-formed story response
    rejected = a truncated / off-topic response
    """
    prompts = [
        "Write a short story about a brave rabbit.",
        "Tell a story about a child who learns to share.",
        "Write about a dragon who becomes kind.",
        "Tell a story about a lost puppy finding its way home.",
        "Write about a little fish who dreams of the ocean.",
    ]
    chosen_responses = [
        "Once upon a time, a brave rabbit named Leo hopped into the dark forest. "
        "Though scared, he remembered his mother's words: courage isn't the absence of fear. "
        "He rescued his friends and returned home a hero. The End.",
        "Mia had the only toy in the playroom and refused to share. "
        "When she saw her friend crying, she finally offered half. "
        "They played together and had twice the fun. The End.",
        "Draco the dragon breathed fire and everyone ran from him. "
        "One day a small bird taught him a song, and his heart softened. "
        "From that day he used his fire to warm cold children. The End.",
        "Biscuit the puppy got lost in the big park. "
        "A kind girl noticed his sad eyes and followed his nose back home. "
        "His family cheered when he trotted through the door. The End.",
        "Finn was a tiny fish in a small pond. "
        "He swam to the river, then to the sea, and found the whole wide ocean. "
        "He wrote home: the world is bigger than our pond. The End.",
    ]
    rejected_responses = [
        "The rabbit went outside.",
        "She had a toy.",
        "Dragon.",
        "The puppy was lost and then found.",
        "Fish swim in water.",
    ]
    examples = []
    for i in range(n):
        j = i % len(prompts)
        examples.append(
            {
                "prompt": prompts[j],
                "chosen": chosen_responses[j],
                "rejected": rejected_responses[j],
            }
        )
    return examples
