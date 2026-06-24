"""Learning rate schedules."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def cosine_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Cosine decay schedule with linear warmup.

    Three phases:
        1. Linear warmup  [0, warmup_steps):       LR rises from 0 → base_lr
        2. Cosine decay   [warmup_steps, max_steps]: LR falls from base_lr → min_lr
        3. Flat           [max_steps, ...):          LR stays at min_lr

    Args:
        optimizer:     the optimizer to schedule
        warmup_steps:  number of linear warmup steps
        max_steps:     total number of training steps
        min_lr_ratio:  floor expressed as a fraction of base_lr (e.g. 0.1 = 10 %)

    Returns:
        A LambdaLR scheduler that multiplies the base_lr by the computed factor.

    Example (base_lr=3e-4, warmup=200, max=10_000, min_ratio=0.1):
        step=0    → 0.0
        step=100  → 1.5e-4  (halfway through warmup)
        step=200  → 3e-4    (peak)
        step=5100 → ~1.65e-4 (cosine midpoint)
        step=10000→ 3e-5    (floor = 10 % of peak)
    """

    def lr_lambda(step: int) -> float:
        # Phase 1 — linear warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Phase 3 — flat floor (past max_steps)
        if step >= max_steps:
            return min_lr_ratio
        # Phase 2 — cosine decay
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Interpolate: min_lr_ratio → 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
