"""Dedicated unit tests for BPETokenizer.

Run with:
    python -m pytest tests/test_tokenizer.py -v
"""

import pytest

from src.tokenizer import BPETokenizer

# ── Training corpus ───────────────────────────────────────────────────────────
CORPUS = [
    "Once upon a time there was a little rabbit who loved to explore.",
    "She hopped through the meadow and sang songs to the butterflies.",
    "The old oak tree whispered secrets to anyone who would listen.",
    "A brave knight rode across the golden hills under a bright sun.",
    "Stars twinkled softly above the quiet village by the river.",
] * 40


# ── Shared fixture ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    tok = BPETokenizer()
    tok.train_from_iterator(iter(CORPUS), vocab_size=300, min_frequency=1)
    return tok


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestBPETokenizer:
    # ── Vocabulary ────────────────────────────────────────────────────────────

    def test_vocab_size_within_limit(self, tokenizer):
        assert tokenizer.vocab_size <= 300

    def test_len_equals_vocab_size(self, tokenizer):
        assert len(tokenizer) == tokenizer.vocab_size

    # ── Special tokens ────────────────────────────────────────────────────────

    def test_special_token_ids_are_non_negative(self, tokenizer):
        assert tokenizer.pad_id >= 0
        assert tokenizer.bos_id >= 0
        assert tokenizer.eos_id >= 0
        assert tokenizer.unk_id >= 0

    def test_special_token_ids_are_distinct(self, tokenizer):
        ids = [tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id]
        assert len(set(ids)) == len(ids)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def test_encode_with_special_tokens_wraps_with_bos_eos(self, tokenizer):
        ids = tokenizer.encode("Once upon a time")
        assert ids[0] == tokenizer.bos_id
        assert ids[-1] == tokenizer.eos_id

    def test_encode_without_special_tokens_has_no_bos_eos(self, tokenizer):
        ids = tokenizer.encode("Once upon a time", add_special_tokens=False)
        assert ids[0] != tokenizer.bos_id
        assert ids[-1] != tokenizer.eos_id

    def test_empty_string_with_special_tokens(self, tokenizer):
        ids = tokenizer.encode("", add_special_tokens=True)
        assert ids == [tokenizer.bos_id, tokenizer.eos_id]

    def test_longer_text_produces_more_tokens(self, tokenizer):
        short_ids = tokenizer.encode("hello", add_special_tokens=False)
        long_ids = tokenizer.encode(
            "hello world once upon a time", add_special_tokens=False
        )
        assert len(long_ids) >= len(short_ids)

    # ── Decoding ─────────────────────────────────────────────────────────────

    def test_decode_roundtrip(self, tokenizer):
        text = "Once upon a time"
        ids = tokenizer.encode(text)
        recovered = tokenizer.decode(ids)
        assert text in recovered

    # ── Batch operations ──────────────────────────────────────────────────────

    def test_batch_encode_returns_correct_number_of_sequences(self, tokenizer):
        texts = ["hello", "world", "once upon a time"]
        batch = tokenizer.encode_batch(texts)
        assert len(batch) == len(texts)

    def test_batch_encode_each_sequence_has_bos_and_eos(self, tokenizer):
        texts = ["hello", "world"]
        batch = tokenizer.encode_batch(texts)
        for ids in batch:
            assert ids[0] == tokenizer.bos_id
            assert ids[-1] == tokenizer.eos_id

    # ── Persistence ───────────────────────────────────────────────────────────

    def test_save_and_load_preserves_vocab_size(self, tokenizer, tmp_path):
        path = str(tmp_path / "tok.json")
        tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        assert loaded.vocab_size == tokenizer.vocab_size

    def test_save_and_load_preserves_encoding(self, tokenizer, tmp_path):
        path = str(tmp_path / "tok2.json")
        tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        assert tokenizer.encode("hello") == loaded.encode("hello")

    # ── Robustness ───────────────────────────────────────────────────────────

    def test_handles_unicode(self, tokenizer):
        unicode_text = "こんにちは 🌟 café naïve résumé"
        try:
            ids = tokenizer.encode(unicode_text)
            assert len(ids) > 0
        except Exception as e:
            pytest.fail(f"Unicode encoding raised an unexpected exception: {e}")

    def test_handles_repeated_whitespace(self, tokenizer):
        ids = tokenizer.encode("   hello   world   ")
        assert len(ids) > 2
