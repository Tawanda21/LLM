"""Tokenizer utility functions."""

from typing import Iterable, Iterator


def text_file_iterator(path: str) -> Iterator[str]:
    """Yield non-empty lines from a plain-text file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def dataset_text_iterator(dataset: Iterable, field: str = "text") -> Iterator[str]:
    """Yield text strings from a HuggingFace Dataset or any iterable of dicts."""
    for example in dataset:
        text = example.get(field, "")
        if text:
            yield text


def count_tokens(tokenizer, texts: Iterable[str]) -> int:
    """Count total tokens across a list of strings (no special tokens)."""
    return sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts)
