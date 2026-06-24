"""Alignment launch script — DPO or PPO.

Usage:

  # DPO (recommended — simpler, no reward model needed)
  python scripts/align.py --method dpo \\
      --sft_ckpt checkpoints/finetune/best.pt \\
      --demo

  # PPO (trains a reward model first, then runs PPO)
  python scripts/align.py --method ppo \\
      --sft_ckpt checkpoints/finetune/best.pt \\
      --demo

  # DPO with your own preference data (JSONL)
  python scripts/align.py --method dpo \\
      --sft_ckpt checkpoints/finetune/best.pt \\
      --data_path data/preferences.jsonl

Preference data format (JSONL):
  {"prompt": "...", "chosen": "...", "rejected": "..."}
"""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.alignment.dpo_trainer import (
    DPOConfig,
    DPODataset,
    DPOTrainer,
    make_preference_examples,
)
from src.alignment.ppo_trainer import PPOConfig, PPOTrainer
from src.alignment.reward_model import RewardModel, preference_loss
from src.model import GPT
from src.model.config import ModelConfig
from src.tokenizer import BPETokenizer
from src.training.checkpointing import load_checkpoint
from src.training.optimizer import build_adamw
from src.training.scheduler import cosine_with_warmup


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Alignment: DPO or PPO")
    p.add_argument(
        "--method",
        type=str,
        default="dpo",
        choices=["dpo", "ppo"],
        help="Alignment algorithm (dpo recommended).",
    )
    p.add_argument(
        "--sft_ckpt",
        type=str,
        default=None,
        help="Path to SFT fine-tuned checkpoint (.pt).",
    )
    p.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    p.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="JSONL preference data. Uses synthetic demo data if omitted.",
    )
    p.add_argument(
        "--demo", action="store_true", help="Use built-in synthetic preference data."
    )
    p.add_argument("--max_steps", type=int, default=1_000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument(
        "--beta", type=float, default=0.1, help="DPO/PPO KL penalty coefficient."
    )
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--use_wandb", action="store_true")
    return p.parse_args()


def load_model(ckpt_path: str, tokenizer, device: str = "cpu") -> GPT:
    """Load a GPT model from checkpoint, or create a fresh one."""
    cfg = ModelConfig(vocab_size=len(tokenizer))
    model = GPT(cfg)
    if ckpt_path and Path(ckpt_path).exists():
        print(f"Loading model from {ckpt_path}")
        load_checkpoint(ckpt_path, model, device=device)
    else:
        print("No checkpoint found — using random weights (results will be poor).")
    return model


def run_dpo(args, tokenizer, device: str) -> None:
    print("\n=== DPO Alignment ===")

    # Policy and reference are both the SFT model
    policy = load_model(args.sft_ckpt, tokenizer).to(device)
    reference = copy.deepcopy(policy)  # identical frozen copy

    # Preference dataset
    examples = _load_or_synthetic(args)
    dataset = DPODataset(tokenizer, examples, max_seq_len=args.max_seq_len)

    if len(dataset) == 0:
        print("[error] No valid preference pairs. Check data format.")
        return

    cfg = DPOConfig(
        beta=args.beta,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir or "checkpoints/dpo",
        use_wandb=args.use_wandb,
    )

    trainer = DPOTrainer(policy, reference, dataset, cfg)
    trainer.train()

    # Save the aligned policy
    out = Path(cfg.checkpoint_dir) / "dpo_final.pt"
    torch.save(policy.state_dict(), out)
    print(f"Aligned model saved → {out}")


def run_ppo(args, tokenizer, device: str) -> None:
    print("\n=== PPO Alignment ===")
    print("Step 1/2: Training reward model on preference data...")

    # Load SFT model as policy, reference, and RM backbone
    sft_model = load_model(args.sft_ckpt, tokenizer).to(device)
    reference = copy.deepcopy(sft_model)
    policy = copy.deepcopy(sft_model)

    # Train reward model
    examples = _load_or_synthetic(args)
    rm = RewardModel.from_pretrained(sft_model).to(device)

    rm_optimizer = build_adamw(rm, lr=1e-4)
    rm_scheduler = cosine_with_warmup(rm_optimizer, warmup_steps=20, max_steps=200)

    from src.alignment.dpo_trainer import DPODataset

    rm_dataset = DPODataset(tokenizer, examples, max_seq_len=args.max_seq_len)
    rm_loader = torch.utils.data.DataLoader(
        rm_dataset, batch_size=args.batch_size, shuffle=True
    )

    rm.train()
    rm_step = 0
    for batch in _cycle(rm_loader):
        if rm_step >= 200:
            break
        chosen_ids = batch["chosen_input_ids"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)

        r_chosen = rm(chosen_ids)
        r_rejected = rm(rejected_ids)
        loss, _ = preference_loss(r_chosen, r_rejected)

        loss.backward()
        rm_optimizer.step()
        rm_scheduler.step()
        rm_optimizer.zero_grad(set_to_none=True)
        rm_step += 1

        if rm_step % 20 == 0:
            print(f"  RM step {rm_step:>4} | loss {loss.item():.4f}")

    rm.eval()
    print("Reward model trained.")

    # Build prompt list from examples
    print("\nStep 2/2: Running PPO...")
    prompts = []
    for ex in examples[:50]:
        prompt_ids = tokenizer.encode(ex["prompt"], add_special_tokens=False)
        prompts.append(
            torch.tensor([[tokenizer.bos_id] + prompt_ids], dtype=torch.long).to(device)
        )

    cfg = PPOConfig(
        max_steps=args.max_steps,
        batch_size=min(args.batch_size, len(prompts)),
        lr=args.lr / 5,  # PPO uses lower LR than DPO
        kl_coef=args.beta,
        checkpoint_dir=args.checkpoint_dir or "checkpoints/ppo",
    )

    trainer = PPOTrainer(policy, reference, rm, prompts, cfg)
    trainer.train()

    out = Path(cfg.checkpoint_dir) / "ppo_final.pt"
    torch.save(policy.state_dict(), out)
    print(f"Aligned model saved → {out}")


def _load_or_synthetic(args) -> list:
    if args.demo or args.data_path is None:
        print("Using synthetic preference data...")
        from src.alignment.dpo_trainer import make_preference_examples

        return make_preference_examples(n=200)
    else:
        import json

        examples = []
        with open(args.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        print(f"Loaded {len(examples):,} preference pairs from {args.data_path}")
        return examples


def _cycle(loader):
    while True:
        yield from loader


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load tokenizer
    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        print(f"[error] Tokenizer not found at {tok_path}")
        print("        Run: python scripts/train_tokenizer.py first.")
        sys.exit(1)
    tokenizer = BPETokenizer.load(tok_path)
    print(f"Tokenizer: {tokenizer}")

    if args.method == "dpo":
        run_dpo(args, tokenizer, device)
    elif args.method == "ppo":
        run_ppo(args, tokenizer, device)


if __name__ == "__main__":
    main()
