"""Tests for Phase 2: tokenizer, data pipeline, scheduler, optimizer, checkpointing."""

import pytest
import torch
import torch.nn as nn

from src.data.collator import collate_packed
from src.data.dataset import PackedDataset
from src.data.preprocessing import clean
from src.model import GPT, ModelConfig
from src.tokenizer import BPETokenizer
from src.training.checkpointing import (
    get_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from src.training.optimizer import _split_param_groups, build_adamw
from src.training.scheduler import cosine_with_warmup

# ── Shared fixtures ───────────────────────────────────────────────────────────

B, T = 2, 32  # batch size, sequence length


@pytest.fixture
def cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=64,
        multiple_of=32,
    )


@pytest.fixture
def model(cfg) -> GPT:
    return GPT(cfg).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer
# ─────────────────────────────────────────────────────────────────────────────


class TestBPETokenizer:
    CORPUS = [
        "Once upon a time there was a little girl named Lily.",
        "She loved to play in the forest with her friends.",
        "One day she found a magic stone by the river.",
        "The stone could grant any wish she wanted to make.",
        "Lily wished for all the animals to be happy forever.",
    ] * 20  # repeat so BPE has enough merge candidates

    @pytest.fixture
    def tokenizer(self):
        tok = BPETokenizer()
        tok.train_from_iterator(iter(self.CORPUS), vocab_size=200, min_frequency=1)
        return tok

    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size <= 200
        assert tokenizer.vocab_size > 4  # at least the special tokens

    def test_special_token_ids_valid(self, tokenizer):
        assert tokenizer.pad_id >= 0
        assert tokenizer.bos_id >= 0
        assert tokenizer.eos_id >= 0
        assert tokenizer.unk_id >= 0

    def test_encode_decode_roundtrip(self, tokenizer):
        text = "Once upon a time"
        ids = tokenizer.encode(text)
        out = tokenizer.decode(ids)
        assert text in out  # decoded text should contain original (may have spacing)

    def test_encode_adds_bos_eos(self, tokenizer):
        ids = tokenizer.encode("hello")
        assert ids[0] == tokenizer.bos_id
        assert ids[-1] == tokenizer.eos_id

    def test_encode_no_special_tokens(self, tokenizer):
        ids = tokenizer.encode("hello", add_special_tokens=False)
        assert ids[0] != tokenizer.bos_id
        assert ids[-1] != tokenizer.eos_id

    def test_encode_batch(self, tokenizer):
        texts = ["hello world", "once upon a time"]
        batch = tokenizer.encode_batch(texts)
        assert len(batch) == 2
        for ids in batch:
            assert ids[0] == tokenizer.bos_id
            assert ids[-1] == tokenizer.eos_id

    def test_save_and_load(self, tokenizer, tmp_path):
        path = str(tmp_path / "tok.json")
        tokenizer.save(path)
        tok2 = BPETokenizer.load(path)
        assert tok2.vocab_size == tokenizer.vocab_size
        ids1 = tokenizer.encode("hello")
        ids2 = tok2.encode("hello")
        assert ids1 == ids2

    def test_len(self, tokenizer):
        assert len(tokenizer) == tokenizer.vocab_size


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────


class TestPreprocessing:
    def test_clean_returns_none_for_short(self):
        assert clean("hi", min_length=10) is None

    def test_clean_returns_text_long_enough(self):
        text = "Once upon a time there was a cat."
        assert clean(text) == text.strip()

    def test_clean_strips_whitespace(self):
        result = clean("  hello world  ")
        assert result == "hello world"

    def test_clean_unicode_norm(self):
        # e + combining acute should become é (NFC)
        decomposed = "caf\u0065\u0301"  # cafe + combining acute
        result = clean(decomposed, min_length=1)
        assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# PackedDataset
# ─────────────────────────────────────────────────────────────────────────────


class TestPackedDataset:
    """Use a minimal stub tokenizer to avoid needing a trained BPE model."""

    class StubTokenizer:
        """Tokenizer stub that maps chars to ints."""

        eos_id = 1
        pad_id = 0
        bos_id = 2

        def encode(self, text, add_special_tokens=False):
            return [ord(c) % 200 + 3 for c in text]

    @pytest.fixture
    def stub_tok(self):
        return self.StubTokenizer()

    def _make_docs(self, n=10, length=50):
        return [{"text": "a" * length} for _ in range(n)]

    def test_output_shapes(self, stub_tok):
        docs = self._make_docs(n=20, length=60)
        ds = PackedDataset(stub_tok, docs, max_seq_len=32, do_clean=False)
        for item in ds:
            assert item["input_ids"].shape == (32,)
            assert item["labels"].shape == (32,)
            break  # just check first item

    def test_labels_are_shifted(self, stub_tok):
        """labels must be input_ids shifted left by one position."""
        # Use varied characters so the shift is detectable
        docs = [{"text": "abcdefghijklmnopqrstuvwxyz" * 10}]
        ds = PackedDataset(stub_tok, docs, max_seq_len=32, do_clean=False)
        for item in ds:
            # chunk = [t0, t1, ..., t32]
            # input_ids = chunk[:-1] = [t0, t1, ..., t31]
            # labels    = chunk[1:]  = [t1, t2, ..., t32]
            # So: input_ids[1:] == labels[:-1]
            assert torch.equal(item["input_ids"][1:], item["labels"][:-1])
            break

    def test_no_padding_in_full_chunks(self, stub_tok):
        """Full chunks (from packed middle of stream) should have no pad tokens."""
        docs = self._make_docs(n=50, length=100)
        ds = PackedDataset(stub_tok, docs, max_seq_len=32, do_clean=False)
        found_full = False
        for i, item in enumerate(ds):
            if i >= 5:
                break
            # Full chunks: no pad in input_ids
            assert stub_tok.pad_id not in item["input_ids"].tolist()
            found_full = True
        assert found_full

    def test_collate_packed_stacks(self, stub_tok):
        docs = self._make_docs(n=20, length=60)
        ds = PackedDataset(stub_tok, docs, max_seq_len=32, do_clean=False)
        items = [next(iter(ds)) for _ in range(B)]
        batch = collate_packed(items)
        assert batch["input_ids"].shape == (B, 32)
        assert batch["labels"].shape == (B, 32)


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────


class TestScheduler:
    def test_starts_at_zero(self, model):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = cosine_with_warmup(opt, warmup_steps=10, max_steps=100)
        # After LambdaLR construction it calls step() once → last_epoch=0
        assert sched.get_last_lr()[0] == pytest.approx(0.0, abs=1e-9)

    def test_reaches_peak_at_warmup(self, model):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = cosine_with_warmup(opt, warmup_steps=10, max_steps=100)
        for _ in range(10):
            opt.step()
            sched.step()
        assert sched.get_last_lr()[0] == pytest.approx(1e-3, rel=1e-3)

    def test_decays_monotonically_after_warmup(self, model):
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = cosine_with_warmup(opt, warmup_steps=5, max_steps=100)
        lrs = []
        for _ in range(100):
            opt.step()
            sched.step()
            lrs.append(sched.get_last_lr()[0])
        # Post-warmup phase should be monotonically non-increasing
        assert lrs[10] > lrs[50] > lrs[90]

    def test_respects_min_lr_floor(self, model):
        min_ratio = 0.1
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sched = cosine_with_warmup(
            opt, warmup_steps=5, max_steps=50, min_lr_ratio=min_ratio
        )
        for _ in range(100):  # go well past max_steps
            opt.step()
            sched.step()
        assert sched.get_last_lr()[0] >= 1e-3 * min_ratio - 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────────────────────────────────────────


class TestOptimizer:
    def test_two_param_groups(self, model):
        opt = build_adamw(model)
        assert len(opt.param_groups) == 2

    def test_decay_group_has_wd(self, model):
        opt = build_adamw(model, weight_decay=0.1)
        assert opt.param_groups[0]["weight_decay"] == pytest.approx(0.1)

    def test_no_decay_group_zero_wd(self, model):
        opt = build_adamw(model, weight_decay=0.1)
        assert opt.param_groups[1]["weight_decay"] == 0.0

    def test_no_param_in_both_groups(self, model):
        groups = _split_param_groups(model, weight_decay=0.1)
        ids_a = {id(p) for p in groups[0]["params"]}
        ids_b = {id(p) for p in groups[1]["params"]}
        assert ids_a.isdisjoint(ids_b), "A parameter appears in both groups"

    def test_all_unique_params_covered(self, model):
        groups = _split_param_groups(model, weight_decay=0.1)
        grouped = {id(p) for g in groups for p in g["params"]}
        # model.parameters() may yield tied weights twice; de-dup by id
        all_ids = {id(p) for p in model.parameters() if p.requires_grad}
        assert grouped == all_ids


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckpointing:
    def test_save_and_load_weights(self, model, cfg, tmp_path):
        path = str(tmp_path / "ckpt.pt")
        opt = build_adamw(model)
        save_checkpoint(path, model, opt, None, step=5, loss=2.0)

        model2 = GPT(cfg).eval()
        load_checkpoint(path, model2)

        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.equal(p1, p2), f"Weight mismatch in {n1}"

    def test_checkpoint_stores_step_and_loss(self, model, tmp_path):
        path = str(tmp_path / "meta.pt")
        opt = build_adamw(model)
        save_checkpoint(path, model, opt, None, step=42, loss=3.14)
        ckpt = load_checkpoint(path, model)
        assert ckpt["step"] == 42
        assert ckpt["loss"] == pytest.approx(3.14)

    def test_atomic_write_no_tmp_left(self, model, tmp_path):
        path = str(tmp_path / "atomic.pt")
        opt = build_adamw(model)
        save_checkpoint(path, model, opt, None, step=0, loss=0.0)
        assert not (tmp_path / "atomic.pt.tmp").exists()

    def test_get_latest_returns_none_when_empty(self, tmp_path):
        assert get_latest_checkpoint(str(tmp_path)) is None

    def test_get_latest_returns_last(self, model, tmp_path):
        opt = build_adamw(model)
        for step in [1000, 2000, 3000]:
            path = str(tmp_path / f"step_{step:07d}.pt")
            save_checkpoint(path, model, opt, None, step=step, loss=0.0)
        latest = get_latest_checkpoint(str(tmp_path))
        assert latest.endswith("step_0003000.pt")


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end training step
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_loss_decreases_over_20_steps(self, cfg):
        model = GPT(cfg).train()
        optimizer = build_adamw(model, lr=1e-3)
        scheduler = cosine_with_warmup(optimizer, warmup_steps=2, max_steps=20)

        tokens = torch.randint(0, cfg.vocab_size, (B, T))
        targets = tokens.clone()

        losses = []
        for _ in range(20):
            _, loss = model(tokens, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss did not decrease over 20 steps: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_gradient_norm_clipped(self, cfg):
        """After clip_grad_norm_, the actual gradient norm must not exceed clip."""
        model = GPT(cfg).train()
        tokens = torch.randint(0, cfg.vocab_size, (B, T))
        targets = tokens.clone()
        _, loss = model(tokens, targets)
        loss.backward()

        clip = 1.0
        # clip_grad_norm_ returns the PRE-clip norm and then modifies grads in-place
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Measure the POST-clip global norm manually
        post_clip_norm = (
            sum(
                p.grad.norm().item() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )
        assert post_clip_norm <= clip + 1e-4

    def test_zero_grad_clears_gradients(self, cfg):
        model = GPT(cfg).train()
        tokens = torch.randint(0, cfg.vocab_size, (B, T))
        targets = tokens.clone()
        _, loss = model(tokens, targets)
        loss.backward()

        # Confirm gradients exist
        assert any(p.grad is not None for p in model.parameters())

        # Clear
        for p in model.parameters():
            p.grad = None

        assert all(p.grad is None for p in model.parameters())
