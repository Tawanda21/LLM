"""Streaming pre-training dataset with sequence packing."""

from typing import Any, Dict, Iterable, Iterator, Optional

import torch
from torch.utils.data import IterableDataset

from .preprocessing import clean


class PackedDataset(IterableDataset):
    """Streaming dataset that packs tokenized documents into fixed-length chunks.

    Instead of padding short sequences (wasting GPU cycles on <pad> tokens),
    we concatenate multiple documents end-to-end with an <eos> separator and
    slice the resulting stream into `max_seq_len + 1` chunks.

    For each chunk the input is `chunk[:-1]` and the target is `chunk[1:]`
    (standard next-token-prediction / causal LM objective).

    This approach:
    - Achieves close to 100 % token utilization
    - Works with streaming HuggingFace datasets (no full download required)
    - Handles variable-length documents transparently

    Args:
        tokenizer:   any object with `.encode(text, add_special_tokens=False)`
                     and `.eos_id` / `.pad_id` attributes
        dataset:     HuggingFace dataset (streaming or in-memory) or any
                     iterable of dicts with a text field
        max_seq_len: number of tokens per training example
        text_field:  key to extract text from each dataset example
        do_clean:    apply the preprocessing pipeline before tokenising
    """

    def __init__(
        self,
        tokenizer,
        dataset: Iterable[Dict[str, Any]],
        max_seq_len: int = 2048,
        text_field: str = "text",
        do_clean: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.text_field = text_field
        self.do_clean = do_clean

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer = []

        for example in self.dataset:
            text = example.get(self.text_field, "")
            if not text:
                continue
            if self.do_clean:
                text = clean(text)
                if text is None:
                    continue

            # Encode without special tokens; manually append <eos> as doc boundary
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            ids.append(self.tokenizer.eos_id)
            buffer.extend(ids)

            # Yield every complete max_seq_len+1 chunk immediately
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len + 1 :]
                yield self._make_example(chunk)

        # Yield the last partial chunk with <pad> fill
        if len(buffer) > 1:
            pad_needed = (self.max_seq_len + 1) - len(buffer)
            chunk = buffer + [self.tokenizer.pad_id] * pad_needed
            yield self._make_example(chunk, has_padding=True)

    # ── Private ───────────────────────────────────────────────────────────────

    def _make_example(
        self,
        chunk: list,
        has_padding: bool = False,
    ) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        if has_padding:
            # Mask pad positions so they don't contribute to the cross-entropy loss
            labels[input_ids == self.tokenizer.pad_id] = -1
        return {"input_ids": input_ids, "labels": labels}
