"""QLoRA — 4-bit quantised base model with LoRA adapters.

QLoRA (Dettmers et al., 2023) combines two techniques:
  1. NF4 quantisation   — compress the frozen base model to 4 bits/weight
  2. LoRA adapters       — add small trainable low-rank matrices on top

This allows fine-tuning models that would otherwise not fit in GPU memory.
A 7 B model that normally requires ~14 GB in fp16 fits in ~5 GB with QLoRA.

Reference: https://arxiv.org/abs/2305.14314

Requirements:
    bitsandbytes >= 0.43.0
    CUDA device (4-bit kernels are CUDA-only)
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .lora import (
    LoRALinear,
    _get_parent,
    freeze_non_lora,
    inject_lora,
    lora_param_count,
)


def quantize_model_4bit(model: nn.Module) -> nn.Module:
    """Replace all non-LoRA nn.Linear layers with 4-bit NF4 quantised equivalents.

    Uses bitsandbytes' Linear4bit which:
    - Stores weights in 4-bit Normal Float (NF4) format
    - Dequantises on-the-fly during the forward pass to bfloat16
    - Uses double quantisation to further reduce memory

    Call this AFTER inject_lora() so that LoRA-wrapped layers are skipped.
    Move the model to CUDA after this call to trigger actual quantisation.

    Args:
        model: the GPT model (with LoRA already injected)

    Returns:
        The model with non-LoRA Linear layers replaced by Linear4bit (in-place).

    Raises:
        ImportError if bitsandbytes is not installed.
        RuntimeError if no CUDA device is available.
    """
    try:
        from bitsandbytes.nn import Linear4bit
    except ImportError:
        raise ImportError(
            "bitsandbytes is required for QLoRA.\n"
            "Install with:  pip install bitsandbytes>=0.43.0\n"
            "Note: bitsandbytes requires a CUDA device."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA requires a CUDA GPU.  bitsandbytes 4-bit kernels are CUDA-only."
        )

    # Find names of all nn.Linear layers that are INSIDE a LoRALinear wrapper.
    # These must be skipped — the LoRA adapter lives in full precision.
    lora_prefixes = {
        name for name, mod in model.named_modules() if isinstance(mod, LoRALinear)
    }

    def is_inside_lora(name: str) -> bool:
        return any(name.startswith(prefix + ".") for prefix in lora_prefixes)

    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not is_inside_lora(name):
            parent, child_name = _get_parent(model, name)
            q_linear = Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                quant_type="nf4",
                compute_dtype=torch.bfloat16,
            )
            if module.bias is not None:
                q_linear.bias = nn.Parameter(module.bias.data.clone())
            # Weights are quantised lazily when .to("cuda") is called
            setattr(parent, child_name, q_linear)
            replaced += 1

    print(
        f"Quantised {replaced} Linear layers to 4-bit NF4 (weights quantised on .to('cuda'))"
    )
    return model


def prepare_model_for_qlora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """Full QLoRA preparation pipeline.

    Steps:
        1. Inject LoRA adapters into target layers (wq, wv by default)
        2. Quantise all remaining Linear layers to 4-bit NF4
        3. Freeze everything except LoRA adapter matrices
        4. Print a memory and parameter summary

    After calling this function, move the model to CUDA to trigger quantisation:
        model = prepare_model_for_qlora(model)
        model = model.to("cuda")   ← quantisation happens here

    Then train only the LoRA adapter parameters.

    Args:
        model:          pre-trained GPT model (on CPU)
        r:              LoRA rank
        alpha:          LoRA scaling factor
        dropout:        dropout on adapter input path
        target_modules: layer name substrings to inject LoRA into

    Returns:
        The prepared model (still on CPU; move to CUDA after this call).
    """
    # Step 1 — LoRA injection (while layers are still full-precision nn.Linear)
    model = inject_lora(
        model,
        target_modules=target_modules or ["wq", "wv"],
        r=r,
        alpha=alpha,
        dropout=dropout,
    )

    # Step 2 — Quantise everything else to 4-bit
    model = quantize_model_4bit(model)

    # Step 3 — Freeze non-LoRA parameters
    freeze_non_lora(model)

    # Summary
    lora, total = lora_param_count(model)
    print(
        f"QLoRA ready: {lora:,} trainable LoRA params  "
        f"({100 * lora / total:.2f} % of {total:,} total)"
    )
    return model
