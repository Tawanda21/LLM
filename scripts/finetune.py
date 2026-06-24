"""Fine-tuning launch script — SFT with LoRA.

Usage:
    # Fine-tune with built-in synthetic demo data (no external files needed)
    python scripts/finetune.py --pretrain_ckpt checkpoints/best.pt --demo

    # Fine-tune on your own JSONL instruction dataset
    python scripts/finetune.py \\
        --pretrain_ckpt checkpoints/best.pt \\
        --data_path data/instructions.jsonl

    # QLoRA (requires CUDA + bitsandbytes)
    python scripts/finetune.py --pretrain_ckpt checkpoints/best.pt --qlora --demo

Data format (JSONL, one example per line):
    {"instruction": "...", "input": "", "output": "..."}
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader, random_split

from src.finetuning.lora import (
    freeze_non_lora,
    inject_lora,
    lora_param_count,
    save_lora,
)
from src.finetuning.sft import SFTDataset, make_tinystories_sft_examples
from src.model import GPT
from src.model.config import ModelConfig
from src.tokenizer import BPETokenizer
from src.training.checkpointing import load_checkpoint
from src.training.trainer import TrainConfig, Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SFT + LoRA fine-tuning")
    p.add_argument(
        "--pretrain_ckpt",
        type=str,
        default=None,
        help="Path to pre-trained checkpoint (.pt). If omitted, starts from random weights.",
    )
    p.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    p.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="JSONL file with instruction-response pairs.",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Use built-in synthetic TinyStories SFT data (no data_path needed).",
    )
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--max_steps", type=int, default=2_000)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/finetune")
    p.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"]
    )
    p.add_argument(
        "--qlora",
        action="store_true",
        help="Use QLoRA (4-bit quantisation). Requires CUDA + bitsandbytes.",
    )
    p.add_argument("--use_wandb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        print(f"[error] Tokenizer not found at {tok_path}")
        print("        Run: python scripts/train_tokenizer.py first.")
        sys.exit(1)
    tokenizer = BPETokenizer.load(tok_path)
    print(f"Tokenizer: {tokenizer}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = ModelConfig(vocab_size=len(tokenizer))
    model = GPT(model_cfg)

    if args.pretrain_ckpt and Path(args.pretrain_ckpt).exists():
        print(f"Loading pre-trained weights from {args.pretrain_ckpt}")
        load_checkpoint(args.pretrain_ckpt, model, device="cpu")
    else:
        print("No pre-trained checkpoint found — starting from random weights.")

    # ── LoRA / QLoRA injection ────────────────────────────────────────────────
    if args.qlora:
        from src.finetuning.qlora import prepare_model_for_qlora

        model = prepare_model_for_qlora(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
    else:
        model = inject_lora(
            model,
            target_modules=["wq", "wv"],
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        freeze_non_lora(model)

    lora, total = lora_param_count(model)
    print(f"Trainable LoRA params: {lora:,} / {total:,}  ({100 * lora / total:.2f} %)")

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.demo or args.data_path is None:
        print("Using synthetic TinyStories SFT demo data...")
        examples = make_tinystories_sft_examples(n=500)
    else:
        print(f"Loading data from {args.data_path}")
        import json

        examples = []
        with open(args.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        print(f"  Loaded {len(examples):,} examples.")

    dataset = SFTDataset(tokenizer, examples, max_seq_len=args.max_seq_len)

    # 90 / 10 train / val split
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ── Training ──────────────────────────────────────────────────────────────
    train_cfg = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        lr=args.lr,
        weight_decay=0.0,
        beta2=0.999,
        warmup_steps=50,
        min_lr_ratio=0.0,
        checkpoint_dir=args.checkpoint_dir,
        run_name="sft-lora",
        mixed_precision=args.mixed_precision,
        use_wandb=args.use_wandb,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        resume=False,  # SFT always starts fresh
    )
    trainer.train()

    # Save adapter weights separately
    save_lora(model, Path(args.checkpoint_dir) / "lora_adapters.pt")
    print("Done.")


if __name__ == "__main__":
    main()
