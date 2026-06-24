"""Supervised Fine-Tuning dataset and utilities.

SFT teaches a pre-trained model to follow instructions by training on
(instruction, response) pairs. The key difference from pre-training is
prompt masking: we compute the loss ONLY on the response tokens, not on
the instruction tokens. This prevents the model from "memorising" the
instruction wording and forces it to learn how to respond.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

# ── Prompt templates ──────────────────────────────────────────────────────────

ALPACA_TEMPLATE = (
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)

ALPACA_TEMPLATE_NO_INPUT = "### Instruction:\n{instruction}\n\n### Response:\n"

# Simpler template for story/creative writing fine-tuning
STORY_TEMPLATE = "Write a story about {topic}:\n\n"


def format_alpaca(example: Dict[str, str]) -> Tuple[str, str]:
    """Format an Alpaca-style dict into (prompt, response) string pair.

    Expected keys: "instruction", "input" (optional), "output".

    Returns:
        prompt:   the instruction part (NOT included in loss)
        response: the answer part (loss IS computed here)
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "").strip()
    response = example.get("output", example.get("response", ""))

    if input_text:
        prompt = ALPACA_TEMPLATE.format(instruction=instruction, input=input_text)
    else:
        prompt = ALPACA_TEMPLATE_NO_INPUT.format(instruction=instruction)

    return prompt, response


def format_story(example: Dict[str, str]) -> Tuple[str, str]:
    """Format a story-writing example.  Expected keys: "topic", "story"."""
    prompt = STORY_TEMPLATE.format(topic=example.get("topic", "a little adventure"))
    response = example.get("story", example.get("output", ""))
    return prompt, response


# ── Dataset ───────────────────────────────────────────────────────────────────


class SFTDataset(Dataset):
    """Instruction-response dataset for Supervised Fine-Tuning.

    Tokenises each example as:

        [<bos>] [prompt tokens] [response tokens] [<eos>] [<pad> ...]

    Labels are then set to -1 (ignored) for all prompt positions so that
    the cross-entropy loss is computed only over response tokens.

    This teaches the model to generate the response given the instruction,
    without overfitting to the instruction wording.

    Args:
        tokenizer:   BPETokenizer instance (or any object with encode/decode
                     and bos_id, eos_id, pad_id attributes)
        examples:    list of raw example dicts
        max_seq_len: maximum total sequence length (prompt + response + specials)
        format_fn:   callable(example) → (prompt_str, response_str)
                     defaults to Alpaca format
    """

    def __init__(
        self,
        tokenizer,
        examples: List[Dict[str, str]],
        max_seq_len: int = 512,
        format_fn: Optional[Callable] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.format_fn = format_fn or format_alpaca
        self._data = self._preprocess(examples)

    # ── Construction ──────────────────────────────────────────────────────────

    def _preprocess(self, examples: List[Dict]) -> List[Dict[str, torch.Tensor]]:
        data, skipped = [], 0
        for ex in examples:
            try:
                item = self._tokenize(ex)
                if item is not None:
                    data.append(item)
                else:
                    skipped += 1
            except Exception:
                skipped += 1
        if skipped:
            print(f"  SFTDataset: skipped {skipped} examples (too long or malformed)")
        print(f"  SFTDataset: {len(data)} examples ready.")
        return data

    def _tokenize(self, example: Dict) -> Optional[Dict[str, torch.Tensor]]:
        prompt, response = self.format_fn(example)

        # Encode without special tokens — we add bos/eos manually
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        # Full sequence: <bos> + prompt + response + <eos>
        ids = (
            [self.tokenizer.bos_id]
            + prompt_ids
            + response_ids
            + [self.tokenizer.eos_id]
        )

        if len(ids) > self.max_seq_len + 1:
            return None  # too long; skip rather than truncate mid-response

        if len(response_ids) == 0:
            return None  # no response to learn from

        # Pad to max_seq_len + 1 so all examples have the same length
        pad_needed = (self.max_seq_len + 1) - len(ids)
        ids = ids + [self.tokenizer.pad_id] * pad_needed

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)

        # ── Prompt masking ────────────────────────────────────────────────────
        # Mask labels for every position whose input is a prompt token.
        # prompt_len = number of positions in input_ids that are prompt
        #            = 1 (<bos>) + len(prompt_ids)
        # After masking, the first unmasked label is labels[prompt_len]
        # which is the second response token predicted from position prompt_len.
        # The cross-entropy loss therefore covers:
        #   response_ids[1], response_ids[2], ..., response_ids[-1], <eos>
        prompt_len = 1 + len(prompt_ids)
        labels[:prompt_len] = -1

        # Also mask padding
        labels[input_ids == self.tokenizer.pad_id] = -1

        return {"input_ids": input_ids, "labels": labels}

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._data[idx]

    # ── Factory classmethods ──────────────────────────────────────────────────

    @classmethod
    def from_jsonl(
        cls,
        tokenizer,
        path: Union[str, Path],
        **kwargs,
    ) -> "SFTDataset":
        """Load examples from a JSONL file (one JSON object per line)."""
        examples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return cls(tokenizer, examples, **kwargs)

    @classmethod
    def from_json(
        cls,
        tokenizer,
        path: Union[str, Path],
        **kwargs,
    ) -> "SFTDataset":
        """Load examples from a JSON file containing a list."""
        with open(path, "r", encoding="utf-8") as f:
            examples = json.load(f)
        return cls(tokenizer, examples, **kwargs)


# ── Synthetic demo data ───────────────────────────────────────────────────────


def make_tinystories_sft_examples(n: int = 100) -> List[Dict[str, str]]:
    """Generate synthetic instruction-response pairs in TinyStories style.

    Useful for quickly testing the SFT pipeline without downloading a real
    instruction dataset. The examples follow the story-writing format that
    aligns with TinyStories pre-training data.
    """
    topics = [
        "a brave rabbit who found a magic carrot",
        "a little girl who could talk to clouds",
        "a dragon who was afraid of fire",
        "a boy who made friends with a lost star",
        "a turtle who wanted to run faster than anyone",
        "a cat who discovered a hidden door in the library",
        "a shy bear who learned to sing",
        "a fish who dreamed of flying",
        "a robot who wanted to feel emotions",
        "a tree who could walk and explore the forest",
    ]
    openings = [
        "Once upon a time,",
        "In a faraway land,",
        "Long ago,",
        "There was once",
        "One sunny morning,",
    ]
    examples = []
    for i in range(n):
        topic = topics[i % len(topics)]
        opening = openings[i % len(openings)]
        examples.append(
            {
                "instruction": f"Write a short children's story about {topic}.",
                "input": "",
                "output": (
                    f"{opening} there was {topic}. "
                    "Every day they would go on new adventures and learn something wonderful. "
                    "Their friends would always be there to help them. "
                    "And at the end of each day, they knew that kindness and courage "
                    "were the greatest gifts of all. The End."
                ),
            }
        )
    return examples
