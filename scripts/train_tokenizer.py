"""Train a BPE tokenizer on TinyStories.

Usage:
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --vocab_size 8000 --num_examples 50000
"""

import argparse
import sys
from pathlib import Path

# Make src importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from src.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BPE tokenizer on TinyStories")
    p.add_argument(
        "--vocab_size",
        type=int,
        default=8_000,
        help="Vocabulary size. 8k is plenty for TinyStories; use 32k for larger corpora.",
    )
    p.add_argument(
        "--num_examples",
        type=int,
        default=50_000,
        help="Number of training documents to use (subset of TinyStories train split).",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default="checkpoints/tokenizer.json",
        help="Output path for the tokenizer JSON file.",
    )
    p.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum merge-pair frequency.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading TinyStories (streaming)...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    print(f"Collecting {args.num_examples:,} examples for tokenizer training...")
    texts = []
    for i, example in enumerate(ds):
        if i >= args.num_examples:
            break
        texts.append(example["text"])
    print(f"  Collected {len(texts):,} documents.")

    print(f"Training BPE (vocab_size={args.vocab_size:,})...")
    tokenizer = BPETokenizer()
    tokenizer.train_from_iterator(
        iter(texts),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    tokenizer.save(args.save_path)
    print(f"\nTokenizer saved → {args.save_path}")
    print(f"  vocab_size : {tokenizer.vocab_size:,}")
    print(f"  pad_id     : {tokenizer.pad_id}")
    print(f"  bos_id     : {tokenizer.bos_id}")
    print(f"  eos_id     : {tokenizer.eos_id}")

    # Quick smoke-test
    sample = "Once upon a time, there was a little cat."
    ids = tokenizer.encode(sample)
    print(f"\nSample encode: {sample!r}")
    print(f"  → {ids}")
    print(f"  → {tokenizer.decode(ids)!r}")


if __name__ == "__main__":
    main()
