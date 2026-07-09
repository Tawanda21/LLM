"""
LLM from Scratch — unified entry point.

Usage:
    python main.py tokenize --vocab_size 8000
    python main.py pretrain --max_steps 10000
    python main.py finetune --demo
    python main.py align --method dpo --demo

Each subcommand delegates to the matching script in scripts/.
"""

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="LLM from Scratch — train, fine-tune, and align a language model.",
    )

    # TODO ① Create a subparsers object.
    #   Hint: call parser.add_subparsers() with dest="command"
    #   so that args.command will hold the chosen subcommand name.
    subparsers = parser.add_subparsers(dest="command")

    # ── tokenize subcommand ───────────────────────────────────────────────────
    tok = subparsers.add_parser("tokenize", help="Train the BPE tokenizer")

    # TODO ② Add three arguments to `tok`:
    #   --vocab_size   int,   default=8_000
    #   --num_examples int,   default=50_000
    #   --save_path    str,   default="checkpoints/tokenizer.json"
    tok.add_argument("--vocab_size", type=int, default=8_000)
    tok.add_argument("--num_examples", type=int, default=50_000)
    tok.add_argument("--save_path", type=str, default="checkpoints/tokenizer.json")

    # ── pretrain subcommand ───────────────────────────────────────────────────
    pre = subparsers.add_parser("pretrain", help="Pre-train the language model")

    # TODO ③ Add four arguments to `pre`:
    #   --tokenizer       str,   default="checkpoints/tokenizer.json"
    #   --max_steps       int,   default=10_000
    #   --batch_size      int,   default=8
    #   --mixed_precision str,   default="bf16"
    pre.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    pre.add_argument("--max_steps", type=int, default=10_000)
    pre.add_argument("--batch_size", type=int, default=8)
    pre.add_argument("--mixed_precision", type=str, default="bf16")

    # ── finetune subcommand ───────────────────────────────────────────────────
    ft = subparsers.add_parser("finetune", help="Fine-tune with SFT + LoRA")
    ft.add_argument("--pretrain_ckpt", type=str, default=None)
    ft.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    ft.add_argument("--demo", action="store_true")
    ft.add_argument("--max_steps", type=int, default=2_000)
    ft.add_argument("--mixed_precision", type=str, default="bf16")

    # ── align subcommand ──────────────────────────────────────────────────────
    al = subparsers.add_parser("align", help="Align with DPO or PPO")
    al.add_argument("--method", type=str, default="dpo", choices=["dpo", "ppo"])
    al.add_argument("--sft_ckpt", type=str, default=None)
    al.add_argument("--tokenizer", type=str, default="checkpoints/tokenizer.json")
    al.add_argument("--demo", action="store_true")
    al.add_argument("--max_steps", type=int, default=1_000)
    al.add_argument("--beta", type=float, default=0.1)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # TODO ④ If no subcommand was given, print help and exit cleanly.
    #   Hint: check if args.command is None, then call parser.print_help()
    #   and sys.exit(0).
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # ── dispatch ──────────────────────────────────────────────────────────────
    if args.command == "tokenize":
        # Rebuild sys.argv for the tokenizer script using parsed args,
        # then delegate to its main() function.
        sys.argv = [
            "train_tokenizer.py",
            "--vocab_size",
            str(args.vocab_size),
            "--num_examples",
            str(args.num_examples),
            "--save_path",
            args.save_path,
        ]
        from scripts.train_tokenizer import main as run

        run()

    elif args.command == "pretrain":
        # Rebuild sys.argv for the pretrain script using parsed args,
        # then delegate to its main() function.
        sys.argv = [
            "pretrain.py",
            "--tokenizer",
            args.tokenizer,
            "--max_steps",
            str(args.max_steps),
            "--batch_size",
            str(args.batch_size),
            "--mixed_precision",
            args.mixed_precision,
        ]
        from scripts.pretrain import main as run

        run()

    elif args.command == "finetune":
        # Rebuild sys.argv for the finetune script using parsed args,
        # then delegate to its main() function.
        sys.argv = [
            "finetune.py",
            "--pretrain_ckpt",
            args.pretrain_ckpt,
            "--tokenizer",
            args.tokenizer,
            "--max_steps",
            str(args.max_steps),
            "--mixed_precision",
            args.mixed_precision,
        ]
        # Append --demo flag only if it was set by the user.
        if args.demo:
            sys.argv.append("--demo")
        from scripts.finetune import main as run

        run()

    elif args.command == "align":
        # Rebuild sys.argv for the align script using parsed args,
        # then delegate to its main() function.
        sys.argv = [
            "align.py",
            "--method",
            args.method,
            "--sft_ckpt",
            args.sft_ckpt,
            "--tokenizer",
            args.tokenizer,
            "--max_steps",
            str(args.max_steps),
            "--beta",
            str(args.beta),
        ]
        # Append --demo flag only if it was set by the user.
        if args.demo:
            sys.argv.append("--demo")
        from scripts.align import main as run

        run()


if __name__ == "__main__":
    main()
