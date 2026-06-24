"""Pre-training launch script.

Usage:
    # Step 1 — train tokenizer (only needed once)
    python scripts/train_tokenizer.py

    # Step 2 — pre-train
    python scripts/pretrain.py
    python scripts/pretrain.py --max_steps 20000 --batch_size 16 --use_wandb
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from src.data.collator import build_dataloader
from src.data.dataset import PackedDataset
from src.model import GPT
from src.model.config import ModelConfig
from src.tokenizer import BPETokenizer
from src.training.trainer import TrainConfig, Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-train the GPT model on TinyStories")
    p.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    p.add_argument("--max_steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--run_name", type=str, default="pretrain-small")
    p.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"]
    )
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--use_wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        print(f"[error] Tokenizer not found at {tok_path}")
        print("        Run: python scripts/train_tokenizer.py")
        sys.exit(1)

    tokenizer = BPETokenizer.load(tok_path)
    print(f"Loaded tokenizer: {tokenizer}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = ModelConfig(vocab_size=len(tokenizer))
    model = GPT(model_cfg)
    n_params = model.num_params(exclude_embedding=False)
    print(f"Model: {n_params:,} parameters  (excl. embedding: {model.num_params():,})")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("Loading TinyStories (streaming)...")
    train_ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    val_ds = load_dataset("roneneldan/TinyStories", split="validation", streaming=True)

    train_dataset = PackedDataset(
        tokenizer, train_ds, max_seq_len=model_cfg.max_seq_len
    )
    val_dataset = PackedDataset(tokenizer, val_ds, max_seq_len=model_cfg.max_seq_len)

    # num_workers=0 on Windows (multiprocessing + IterableDataset has quirks)
    train_loader = build_dataloader(
        train_dataset, batch_size=args.batch_size, num_workers=0
    )
    val_loader = build_dataloader(
        val_dataset, batch_size=args.batch_size, num_workers=0
    )

    # ── Train config ──────────────────────────────────────────────────────────
    train_cfg = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        mixed_precision=args.mixed_precision,
        use_wandb=args.use_wandb,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        resume=not args.no_resume,
    )
    trainer.train()


if __name__ == "__main__":
    main()
