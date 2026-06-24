"""DataLoader collation and builder utilities."""

from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset


def collate_packed(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack pre-packed fixed-length tensors into a batch.

    PackedDataset always yields tensors of exactly `max_seq_len` elements,
    so this is a simple stack — no padding arithmetic required.
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = 2,
    pin_memory: bool = True,
    shuffle: bool = False,
) -> DataLoader:
    """Build a DataLoader for a PackedDataset.

    Args:
        dataset:         a PackedDataset (or compatible IterableDataset)
        batch_size:      examples per batch
        num_workers:     worker processes for data loading (0 = main process only)
        prefetch_factor: batches to prefetch per worker (ignored if num_workers=0)
        pin_memory:      pin memory for faster CPU→GPU transfer (auto-disabled on CPU)
        shuffle:         not supported for IterableDatasets; ignored silently
    """
    kwargs: dict = {}
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_packed,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        **kwargs,
    )
