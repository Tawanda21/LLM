"""Optimizer construction with weight-decay parameter grouping."""

from typing import List

import torch
import torch.nn as nn


def _split_param_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    """Separate parameters into weight-decay and no-weight-decay groups.

    Standard practice: apply weight decay only to weight matrices.
    Exclude:
        - Bias terms             (very few params, regularising hurts more than helps)
        - Normalisation weights  (RMSNorm / LayerNorm gain vectors)
        - Embedding weights      (large but usually left unregularised)

    Weight-tied parameters (tok_embeddings.weight == output.weight) are only
    counted once using a seen-id set.
    """
    no_decay_keywords = {"bias", "norm", "embedding"}

    decay_params: List[torch.Tensor] = []
    no_decay_params: List[torch.Tensor] = []
    seen: set = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue  # skip weight-tied duplicate
        seen.add(id(param))

        if any(kw in name.lower() for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def build_adamw(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    use_fused: bool = True,
) -> torch.optim.AdamW:
    """Construct AdamW with proper weight-decay grouping.

    Uses PyTorch's fused CUDA kernel when available (requires CUDA device and
    torch >= 2.0). The fused kernel is significantly faster on GPU.

    Args:
        model:        the model to optimise
        lr:           peak learning rate (will be scaled by the scheduler)
        weight_decay: L2 regularisation coefficient for weight matrices
        beta1:        first-moment exponential decay (momentum)
        beta2:        second-moment exponential decay (RMS)
        eps:          denominator ε for numerical stability
        use_fused:    attempt to use the fused CUDA AdamW kernel
    """
    param_groups = _split_param_groups(model, weight_decay)

    # Check if fused kernel is available (torch >= 2.0, CUDA device present)
    fused_ok = (
        use_fused
        and torch.cuda.is_available()
        and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    )

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
        **({"fused": True} if fused_ok else {}),
    )
