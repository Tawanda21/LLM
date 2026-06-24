"""LoRA (Low-Rank Adaptation) — manual implementation from scratch.

Reference: Hu et al., 2021 — https://arxiv.org/abs/2106.09685
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Core LoRA layer ───────────────────────────────────────────────────────────


class LoRALinear(nn.Module):
    """A frozen Linear layer augmented with a trainable low-rank adapter.

    Forward pass:
        output = W·x  +  (B·A)·x · (α/r)

    Where:
        W   — frozen pre-trained weight  (d_out × d_in)
        A   — trainable adapter matrix   (r × d_in),   init: Kaiming uniform
        B   — trainable adapter matrix   (d_out × r),  init: zeros
        α/r — scaling factor

    B is initialised to zero so ΔW = B·A = 0 at the start of fine-tuning.
    The model therefore begins from exactly the pre-trained weights.

    Args:
        linear:  the nn.Linear layer to wrap and freeze
        r:       adapter rank  (common choices: 4, 8, 16, 64)
        alpha:   scaling numerator (set to r for no re-scaling, 2r to double)
        dropout: dropout applied to the input before the adapter path
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert r > 0, f"LoRA rank must be > 0, got {r}"

        self.linear = linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        d_out, d_in = linear.weight.shape

        # Freeze the base weight (and bias if present)
        linear.weight.requires_grad = False
        if linear.bias is not None:
            linear.bias.requires_grad = False

        # Trainable adapter matrices
        self.lora_A = nn.Parameter(torch.empty(r, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r))  # zero init
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Same init as nn.Linear default (Kaiming uniform)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen base path
        base = self.linear(x)
        # Adapter path: x → dropout → A → B, scaled by α/r
        # F.linear(x, W) computes x @ W.T
        lora = (
            F.linear(
                F.linear(self.dropout(x), self.lora_A),  # (*, r)
                self.lora_B,  # (*, d_out)
            )
            * self.scaling
        )
        return base + lora

    def merge(self) -> nn.Linear:
        """Absorb the adapter into the base weight and return a plain nn.Linear.

        After merging the model is mathematically identical to the LoRA model
        but has zero inference overhead — no extra matmuls.
        Call this before deployment.
        """
        with torch.no_grad():
            # ΔW = B @ A  (d_out × d_in)
            delta_W = (self.lora_B @ self.lora_A) * self.scaling
            self.linear.weight.data += delta_W
            self.linear.weight.requires_grad = True
            if self.linear.bias is not None:
                self.linear.bias.requires_grad = True
        return self.linear

    def extra_repr(self) -> str:
        d_out, d_in = self.linear.weight.shape
        return (
            f"in={d_in}, out={d_out}, r={self.r}, "
            f"alpha={self.alpha}, scaling={self.scaling:.3f}"
        )


# ── Injection API ─────────────────────────────────────────────────────────────


def inject_lora(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> nn.Module:
    """Replace target nn.Linear layers with LoRALinear wrappers (in-place).

    Args:
        model:          model to modify
        target_modules: list of substrings matched against each module's full
                        dotted name.  Default: ["wq", "wv"]  (Q and V in our GPT)
        r:              LoRA rank
        alpha:          LoRA scaling factor
        dropout:        dropout on the adapter input path

    Returns:
        The model with LoRA injected (same object, modified in-place).

    Raises:
        ValueError if no layers matched.
    """
    if target_modules is None:
        target_modules = ["wq", "wv"]

    replaced = 0
    for name, module in list(model.named_modules()):
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                parent, child_name = _get_parent(model, name)
                setattr(
                    parent,
                    child_name,
                    LoRALinear(module, r=r, alpha=alpha, dropout=dropout),
                )
                replaced += 1
                break

    if replaced == 0:
        linears = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
        raise ValueError(
            f"No layers matched target_modules={target_modules}.\n"
            f"Available linear layers: {linears}"
        )

    print(
        f"Injected LoRA into {replaced} layers  (r={r}, alpha={alpha}, dropout={dropout})"
    )
    return model


def freeze_non_lora(model: nn.Module) -> None:
    """Freeze every parameter except LoRA adapter matrices (lora_A, lora_B).

    After this call only ~0.1 % of parameters require gradients.
    """
    for name, param in model.named_parameters():
        param.requires_grad = "lora_A" in name or "lora_B" in name


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Return only the trainable LoRA adapter parameters."""
    return [p for n, p in model.named_parameters() if "lora_" in n]


def lora_param_count(model: nn.Module) -> Tuple[int, int]:
    """Return (lora_params, total_params)."""
    total = sum(p.numel() for p in model.parameters())
    lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    return lora, total


# ── Merge for inference ───────────────────────────────────────────────────────


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge all LoRA adapters into their base weights (in-place).

    Replaces every LoRALinear with its merged nn.Linear.
    After this call there is no LoRA overhead at inference time.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            parent, child_name = _get_parent(model, name)
            setattr(parent, child_name, module.merge())
    return model


# ── Persistence ───────────────────────────────────────────────────────────────


def save_lora(model: nn.Module, path: Union[str, Path]) -> None:
    """Save only the LoRA adapter weights (a few MB, not the full model).

    The pre-trained base weights are frozen and unchanged — no need to save them.
    Load them back with load_lora() applied to a model with matching LoRA structure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lora_state: Dict[str, torch.Tensor] = {
        n: p.detach().cpu() for n, p in model.named_parameters() if "lora_" in n
    }
    torch.save(lora_state, path)
    size_mb = path.stat().st_size / 1e6
    print(
        f"LoRA adapters saved → {path}  ({size_mb:.2f} MB, {len(lora_state)} tensors)"
    )


def load_lora(model: nn.Module, path: Union[str, Path]) -> None:
    """Load LoRA weights saved by save_lora() into a model with LoRA already injected."""
    lora_state = torch.load(str(path), map_location="cpu", weights_only=True)
    model_params = dict(model.named_parameters())

    missing, unexpected = [], []
    for name, tensor in lora_state.items():
        if name in model_params:
            model_params[name].data.copy_(tensor)
        else:
            unexpected.append(name)
    for name in model_params:
        if "lora_" in name and name not in lora_state:
            missing.append(name)

    if missing:
        print(f"  [load_lora] missing keys:    {missing}")
    if unexpected:
        print(f"  [load_lora] unexpected keys: {unexpected}")


# ── Private utility ───────────────────────────────────────────────────────────


def _get_parent(model: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    """Navigate the module tree and return (parent_module, child_attribute_name).

    Works with both regular attributes and nn.ModuleList indices (stored as
    string keys in _modules).
    """
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
