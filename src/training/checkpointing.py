"""Save and resume training checkpoints."""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    loss: float,
    config: Any = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically save training state to disk.

    Writes to `path + ".tmp"` first, then renames — guaranteeing the file on
    disk is never partially written even if the process is interrupted.

    Args:
        path:      destination file path (.pt)
        model:     model to save (unwrap from Accelerate before passing)
        optimizer: current optimizer state
        scheduler: current LR scheduler state (pass None if no scheduler)
        step:      current effective training step
        loss:      most recent tracked loss
        config:    ModelConfig or TrainConfig dataclass (stored for reference)
        extra:     any additional key-value pairs
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "step": step,
        "loss": loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
    }
    if extra:
        payload.update(extra)

    tmp = path + ".tmp"
    torch.save(payload, tmp)
    Path(tmp).replace(path)  # atomic rename


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Any = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a checkpoint saved by `save_checkpoint`.

    Args:
        path:      path to the .pt file
        model:     model to load weights into
        optimizer: if provided, restore optimizer state
        scheduler: if provided, restore scheduler state
        device:    map tensors to this device during load

    Returns:
        The full checkpoint dict (contains 'step', 'loss', 'config', etc.)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    return ckpt


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return the path of the most recent `step_XXXXXXX.pt` checkpoint, or None."""
    d = Path(checkpoint_dir)
    if not d.exists():
        return None
    checkpoints = sorted(d.glob("step_*.pt"))
    return str(checkpoints[-1]) if checkpoints else None
