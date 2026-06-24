"""BPE tokenizer wrapper around HuggingFace `tokenizers`."""

from pathlib import Path
from typing import Iterable, List, Optional, Union

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

# ── Special token constants ───────────────────────────────────────────────────
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]


class BPETokenizer:
    """Byte-Pair Encoding tokenizer built on HuggingFace `tokenizers`.

    API:
        train(files)               — learn a vocab from text files
        train_from_iterator(iter)  — learn from a Python iterator of strings
        encode(text)               — str → list[int]
        decode(ids)                — list[int] → str
        save(path)                 — persist to a single JSON file
        BPETokenizer.load(path)    — restore from JSON

    Special tokens added automatically:
        <pad>  — padding / ignored positions
        <bos>  — prepended to every encoded sequence
        <eos>  — appended to every encoded sequence
        <unk>  — out-of-vocabulary fallback
    """

    def __init__(self) -> None:
        self._tok: Optional[Tokenizer] = None

    # ── Special-token IDs ─────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tok.token_to_id(PAD)

    @property
    def bos_id(self) -> int:
        return self._tok.token_to_id(BOS)

    @property
    def eos_id(self) -> int:
        return self._tok.token_to_id(EOS)

    @property
    def unk_id(self) -> int:
        return self._tok.token_to_id(UNK)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        files: List[str],
        vocab_size: int = 32_000,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> None:
        """Train BPE on a list of plain-text file paths.

        Args:
            files:         paths to UTF-8 text files
            vocab_size:    target vocabulary size (includes special tokens)
            min_frequency: minimum merge-pair frequency to keep
            show_progress: show tqdm progress bar during training
        """
        tok = self._make_tokenizer()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=show_progress,
        )
        tok.train(files, trainer)
        self._finalize(tok)

    def train_from_iterator(
        self,
        iterator: Iterable[str],
        vocab_size: int = 32_000,
        min_frequency: int = 2,
        show_progress: bool = True,
    ) -> None:
        """Train from any Python iterator of strings (e.g. a HuggingFace dataset).

        Args:
            iterator:      iterable of raw text strings
            vocab_size:    target vocabulary size
            min_frequency: minimum merge-pair frequency to keep
        """
        tok = self._make_tokenizer()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=show_progress,
        )
        tok.train_from_iterator(iterator, trainer=trainer)
        self._finalize(tok)

    # ── Encoding / decoding ───────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> List[int]:
        """Encode a string to a list of token ids.

        By default, wraps the sequence with <bos> and <eos>. Set
        add_special_tokens=False to get raw subword ids only.
        """
        ids = self._tok.encode(text).ids
        if not add_special_tokens:
            if ids and ids[0] == self.bos_id:
                ids = ids[1:]
            if ids and ids[-1] == self.eos_id:
                ids = ids[:-1]
        return ids

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """Batch-encode a list of strings."""
        encs = self._tok.encode_batch(texts)
        if add_special_tokens:
            return [e.ids for e in encs]
        result = []
        for e in encs:
            ids = e.ids
            if ids and ids[0] == self.bos_id:
                ids = ids[1:]
            if ids and ids[-1] == self.eos_id:
                ids = ids[:-1]
            result.append(ids)
        return result

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a list of token ids to a string."""
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self,
        batch: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Batch-decode a list of id sequences."""
        return self._tok.decode_batch(batch, skip_special_tokens=skip_special_tokens)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Save the tokenizer to a single JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        """Load a tokenizer saved with `save()`."""
        obj = cls()
        obj._tok = Tokenizer.from_file(str(path))
        return obj

    # ── Dunder helpers ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        vs = self.vocab_size if self._tok is not None else "untrained"
        return f"BPETokenizer(vocab_size={vs})"

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_tokenizer() -> Tokenizer:
        """Create a bare tokenizer with byte-level BPE pre-tokenizer."""
        tok = Tokenizer(BPE(unk_token=UNK))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()
        return tok

    def _finalize(self, tok: Tokenizer) -> None:
        """Attach the post-processor and store the finished tokenizer."""
        bos_id = tok.token_to_id(BOS)
        eos_id = tok.token_to_id(EOS)
        tok.post_processor = TemplateProcessing(
            single=f"{BOS} $A {EOS}",
            special_tokens=[(BOS, bos_id), (EOS, eos_id)],
        )
        self._tok = tok
