"""Main pre-training loop."""


def detect_precision() -> str:
    """Return the best mixed-precision mode for the current hardware.

    Rules:
      - No CUDA available            →  'no'   (plain float32)
      - CUDA, compute capability < 8  →  'fp16'  (pre-Ampere GPUs)
      - CUDA, compute capability >= 8 →  'bf16'  (Ampere / RTX 30xx+)

    Hint: use torch.cuda.is_available() to check for a GPU.
    Then use torch.cuda.get_device_capability(0) which returns a
    (major, minor) tuple, e.g. (8, 6) for an RTX 3080.
    """
    if not torch.cuda.is_available():
        return "no"
    major, _ = torch.cuda.get_device_capability(0)
    return "bf16" if major >= 8 else "fp16"


import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.utils.data import DataLoader

from .checkpointing import get_latest_checkpoint, load_checkpoint, save_checkpoint
from .optimizer import build_adamw
from .scheduler import cosine_with_warmup


@dataclass
class TrainConfig:
    """All hyperparameters for a pre-training run."""

    # ── Compute ───────────────────────────────────────────────────────────────
    max_steps: int = 10_000
    batch_size: int = 8  # per-device micro-batch
    gradient_accumulation_steps: int = 4  # effective batch = batch_size × grad_accum

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # ── Schedule ──────────────────────────────────────────────────────────────
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1  # LR floor = min_lr_ratio × peak_lr

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1_000

    # ── Paths ─────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    run_name: str = "pretrain"

    # ── Hardware ──────────────────────────────────────────────────────────────
    mixed_precision: str = field(default_factory=detect_precision)

    # ── W&B ───────────────────────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "llm-from-scratch"


class Trainer:
    """Pre-training loop with Accelerate, mixed precision, and checkpointing.

    Design principles:
    - One class, one responsibility: run the training loop.
    - Accelerate handles device placement, mixed precision, and multi-GPU
      distribution with zero changes to the loop logic.
    - Gradient accumulation is done manually (transparent, easy to debug).
    - Atomic checkpoint writes protect against corruption on interrupt.

    Usage::

        trainer = Trainer(model, train_loader, config, val_loader)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: TrainConfig,
        val_loader: Optional[DataLoader] = None,
        resume: bool = True,
    ) -> None:
        self.config = config
        self.step = 0
        self.best_val_loss = float("inf")

        # Accelerate: wraps the model, dataloaders, and optimizer for the
        # current device and precision without changing any logic below.
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            log_with="wandb" if config.use_wandb else None,
        )

        self.optimizer = build_adamw(
            model,
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
            min_lr_ratio=config.min_lr_ratio,
        )

        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.scheduler,
        ) = self.accelerator.prepare(
            model, self.optimizer, train_loader, self.scheduler
        )
        self.val_loader = (
            self.accelerator.prepare(val_loader) if val_loader is not None else None
        )

        # Resume from the latest checkpoint if one exists
        if resume:
            ckpt_path = get_latest_checkpoint(config.checkpoint_dir)
            if ckpt_path:
                self._print(f"Resuming from {ckpt_path}")
                meta = load_checkpoint(
                    ckpt_path,
                    self.accelerator.unwrap_model(self.model),
                    self.optimizer,
                    self.scheduler,
                    device=str(self.accelerator.device),
                )
                self.step = meta.get("step", 0)

    # ── Public entry point ────────────────────────────────────────────────────

    def train(self) -> None:
        cfg = self.config

        if cfg.use_wandb and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                cfg.wandb_project,
                config=vars(cfg),
                init_kwargs={"wandb": {"name": cfg.run_name}},
            )

        n_params = sum(p.numel() for p in self.model.parameters())
        self._print(
            f"Training  |  params: {n_params:,}  "
            f"|  device: {self.accelerator.device}  "
            f"|  precision: {cfg.mixed_precision}"
        )

        data_iter = self._cycle(self.train_loader)
        running_loss = 0.0
        t0 = time.perf_counter()

        while self.step < cfg.max_steps:
            self.model.train()

            # ── Gradient accumulation ─────────────────────────────────────────
            # Accumulate `gradient_accumulation_steps` micro-batches before
            # updating parameters. On all-but-last micro-step we skip the
            # cross-device gradient sync (no_sync) to save communication.
            batch_loss = 0.0
            for micro in range(cfg.gradient_accumulation_steps):
                batch = next(data_iter)
                is_last = micro == cfg.gradient_accumulation_steps - 1
                sync_ctx = (
                    _null_ctx() if is_last else self.accelerator.no_sync(self.model)
                )

                with sync_ctx:
                    _, loss = self.model(batch["input_ids"], batch["labels"])
                    loss = loss / cfg.gradient_accumulation_steps
                    self.accelerator.backward(loss)
                    batch_loss += loss.item()

            grad_norm = self.accelerator.clip_grad_norm_(
                self.model.parameters(), cfg.grad_clip
            )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

            self.step += 1
            running_loss += batch_loss

            # ── Logging ───────────────────────────────────────────────────────
            if self.step % cfg.log_every == 0 and self.accelerator.is_main_process:
                dt = time.perf_counter() - t0
                avg_loss = running_loss / cfg.log_every
                lr = self.scheduler.get_last_lr()[0]
                seq_len = batch["input_ids"].shape[1]
                tok_per_sec = (
                    cfg.log_every
                    * cfg.batch_size
                    * cfg.gradient_accumulation_steps
                    * self.accelerator.num_processes
                    * seq_len
                    / dt
                )
                self._print(
                    f"step {self.step:>6} | "
                    f"loss {avg_loss:.4f} | "
                    f"lr {lr:.2e} | "
                    f"gnorm {float(grad_norm):.3f} | "
                    f"{tok_per_sec / 1e3:.1f}k tok/s"
                )
                if cfg.use_wandb:
                    self.accelerator.log(
                        {
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/grad_norm": float(grad_norm),
                            "train/tok_per_sec": tok_per_sec,
                        },
                        step=self.step,
                    )
                running_loss = 0.0
                t0 = time.perf_counter()

            # ── Evaluation ────────────────────────────────────────────────────
            if self.step % cfg.eval_every == 0 and self.val_loader is not None:
                val_loss = self._evaluate()
                if self.accelerator.is_main_process:
                    self._print(f"  val_loss: {val_loss:.4f}")
                    if cfg.use_wandb:
                        self.accelerator.log({"val/loss": val_loss}, step=self.step)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save("best")

            # ── Checkpoint ────────────────────────────────────────────────────
            if self.step % cfg.save_every == 0 and self.accelerator.is_main_process:
                self._save(f"step_{self.step:07d}")

        self._print("Training complete.")
        if cfg.use_wandb:
            self.accelerator.end_training()

    # ── Private helpers ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self, max_batches: int = 50) -> float:
        self.model.eval()
        total, n = 0.0, 0
        for i, batch in enumerate(self.val_loader):
            if i >= max_batches:
                break
            _, loss = self.model(batch["input_ids"], batch["labels"])
            total += loss.item()
            n += 1
        return total / max(n, 1)

    def _save(self, tag: str) -> None:
        path = Path(self.config.checkpoint_dir) / f"{tag}.pt"
        save_checkpoint(
            str(path),
            self.accelerator.unwrap_model(self.model),
            self.optimizer,
            self.scheduler,
            step=self.step,
            loss=self.best_val_loss,
        )
        self._print(f"  Saved → {path}")

    def _print(self, msg: str) -> None:
        if self.accelerator.is_main_process:
            print(msg, flush=True)

    @staticmethod
    def _cycle(loader: DataLoader) -> Iterator:
        """Cycle through a DataLoader indefinitely."""
        while True:
            yield from loader


@contextmanager
def _null_ctx():
    """No-op context manager (replaces no_sync on the last micro-step)."""
    yield
