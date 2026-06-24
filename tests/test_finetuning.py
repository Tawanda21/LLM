"""Tests for Phase 3: LoRA, SFT dataset, and end-to-end fine-tuning."""

import pytest
import torch
import torch.nn as nn

from src.finetuning.lora import (
    LoRALinear,
    _get_parent,
    freeze_non_lora,
    get_lora_params,
    inject_lora,
    load_lora,
    lora_param_count,
    merge_lora_weights,
    save_lora,
)
from src.finetuning.sft import (
    SFTDataset,
    format_alpaca,
    make_tinystories_sft_examples,
)
from src.model import GPT, ModelConfig
from src.training.optimizer import build_adamw
from src.training.scheduler import cosine_with_warmup

# ── Shared fixtures ───────────────────────────────────────────────────────────

B, T = 2, 16


@pytest.fixture
def cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=128,
        multiple_of=32,
    )


@pytest.fixture
def model(cfg) -> GPT:
    return GPT(cfg).eval()


# ── Stub tokenizer (no trained BPE needed) ────────────────────────────────────


class StubTokenizer:
    bos_id = 1
    eos_id = 2
    pad_id = 0
    unk_id = 3
    vocab_size = 256

    def encode(self, text, add_special_tokens=True):
        ids = [ord(c) % 200 + 4 for c in text[:30]]  # cap length
        if add_special_tokens:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(
            chr((i - 4) % 128 + 32) for i in ids if i >= 4 or not skip_special_tokens
        )

    def __len__(self):
        return self.vocab_size


@pytest.fixture
def stub_tok():
    return StubTokenizer()


# ─────────────────────────────────────────────────────────────────────────────
# LoRALinear
# ─────────────────────────────────────────────────────────────────────────────


class TestLoRALinear:
    @pytest.fixture
    def linear(self):
        return nn.Linear(64, 32, bias=False)

    @pytest.fixture
    def lora_layer(self, linear):
        return LoRALinear(linear, r=4, alpha=8.0)

    def test_output_shape(self, lora_layer):
        x = torch.randn(B, T, 64)
        out = lora_layer(x)
        assert out.shape == (B, T, 32)

    def test_zero_delta_at_init(self, linear):
        """B is initialised to zero → ΔW = B@A = 0 → LoRA output == base output at init."""
        lora = LoRALinear(linear, r=4, alpha=8.0)
        x = torch.randn(B, T, 64)
        with torch.no_grad():
            base_out = linear(x)
            lora_out = lora(x)
        assert torch.allclose(base_out, lora_out, atol=1e-6), (
            "LoRA output should equal base output at initialisation (B=0)"
        )

    def test_base_weight_frozen(self, lora_layer):
        assert not lora_layer.linear.weight.requires_grad

    def test_adapter_params_trainable(self, lora_layer):
        assert lora_layer.lora_A.requires_grad
        assert lora_layer.lora_B.requires_grad

    def test_adapter_shapes(self, lora_layer):
        r = lora_layer.r
        assert lora_layer.lora_A.shape == (r, 64)  # (r, d_in)
        assert lora_layer.lora_B.shape == (32, r)  # (d_out, r)

    def test_scaling(self, linear):
        lora = LoRALinear(linear, r=4, alpha=8.0)
        assert lora.scaling == pytest.approx(8.0 / 4)

    def test_merge_output_equivalent(self, linear):
        """Merged model must produce identical output to unmerged LoRA model."""
        lora = LoRALinear(linear, r=4, alpha=8.0)
        # Give the adapter non-zero weights so merge actually changes something
        with torch.no_grad():
            lora.lora_B.fill_(0.01)
            lora.lora_A.fill_(0.01)

        x = torch.randn(B, T, 64)
        with torch.no_grad():
            out_lora = lora(x)

        merged = lora.merge()  # returns nn.Linear with absorbed weights
        with torch.no_grad():
            out_merged = merged(x)

        assert torch.allclose(out_lora, out_merged, atol=1e-5), (
            "Merged model output must match LoRA model output"
        )

    def test_merge_returns_linear(self, lora_layer):
        merged = lora_layer.merge()
        assert isinstance(merged, nn.Linear)

    def test_merge_unfreezes_weight(self, lora_layer):
        merged = lora_layer.merge()
        assert merged.weight.requires_grad


# ─────────────────────────────────────────────────────────────────────────────
# inject_lora
# ─────────────────────────────────────────────────────────────────────────────


class TestInjectLoRA:
    def test_replaces_target_layers(self, model):
        inject_lora(model, target_modules=["wq", "wv"], r=4, alpha=8.0)
        for name, module in model.named_modules():
            # Only check the direct injected layer, not its children (dropout, linear, etc.)
            if name.endswith(".wq") or name.endswith(".wv"):
                assert isinstance(module, LoRALinear), (
                    f"{name} should be LoRALinear after injection"
                )

    def test_correct_layer_count(self, model, cfg):
        inject_lora(model, target_modules=["wq", "wv"], r=4, alpha=8.0)
        lora_count = sum(
            1 for _, m in model.named_modules() if isinstance(m, LoRALinear)
        )
        # 2 LoRA layers per block × n_layers blocks
        assert lora_count == 2 * cfg.n_layers

    def test_non_target_layers_untouched(self, model):
        inject_lora(model, target_modules=["wq"], r=4, alpha=8.0)
        for name, module in model.named_modules():
            if "wv" in name and "." not in name.split("wv")[-1].lstrip("."):
                assert not isinstance(module, LoRALinear), (
                    f"{name} should NOT be LoRALinear (not in target_modules)"
                )

    def test_raises_on_bad_target(self, model):
        with pytest.raises(ValueError, match="No layers matched"):
            inject_lora(model, target_modules=["nonexistent_layer"], r=4, alpha=8.0)

    def test_returns_same_model_object(self, model):
        result = inject_lora(model, r=4, alpha=8.0)
        assert result is model  # in-place modification


# ─────────────────────────────────────────────────────────────────────────────
# freeze_non_lora / param counts
# ─────────────────────────────────────────────────────────────────────────────


class TestFreezeAndCount:
    def test_only_lora_params_trainable(self, model):
        inject_lora(model, r=4, alpha=8.0)
        freeze_non_lora(model)
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_lora_params_much_smaller_than_total(self, model):
        inject_lora(model, r=4, alpha=8.0)
        freeze_non_lora(model)
        lora, total = lora_param_count(model)
        assert lora < total
        assert lora / total < 0.05, (
            f"LoRA params should be < 5 % of total, got {100 * lora / total:.2f} %"
        )

    def test_get_lora_params(self, model):
        inject_lora(model, r=4, alpha=8.0)
        params = get_lora_params(model)
        assert len(params) > 0
        for p in params:
            assert isinstance(p, torch.nn.Parameter)


# ─────────────────────────────────────────────────────────────────────────────
# merge_lora_weights
# ─────────────────────────────────────────────────────────────────────────────


class TestMergeLoRA:
    def test_merge_removes_lora_layers(self, model):
        inject_lora(model, r=4, alpha=8.0)
        merge_lora_weights(model)
        lora_count = sum(
            1 for _, m in model.named_modules() if isinstance(m, LoRALinear)
        )
        assert lora_count == 0, "All LoRALinear layers should be replaced after merge"

    def test_merge_preserves_output(self, model, cfg):
        """Merged model output must be numerically identical to LoRA model output."""
        inject_lora(model, r=4, alpha=8.0)

        # Give adapters non-zero values
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.005)

        tokens = torch.randint(0, cfg.vocab_size, (1, T))
        model.eval()
        with torch.no_grad():
            logits_lora, _ = model(tokens)

        merge_lora_weights(model)
        with torch.no_grad():
            logits_merged, _ = model(tokens)

        assert torch.allclose(logits_lora, logits_merged, atol=1e-5), (
            "Merged model logits must match LoRA model logits"
        )


# ─────────────────────────────────────────────────────────────────────────────
# save_lora / load_lora
# ─────────────────────────────────────────────────────────────────────────────


class TestSaveLoadLoRA:
    def test_save_and_load_preserves_weights(self, model, tmp_path):
        inject_lora(model, r=4, alpha=8.0)
        # Give adapters distinct values
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n:
                    p.fill_(0.123)
                if "lora_A" in n:
                    p.fill_(0.456)

        path = str(tmp_path / "lora.pt")
        save_lora(model, path)

        # New model with same LoRA structure
        from src.model import GPT

        model2 = GPT(model.config).eval()
        inject_lora(model2, r=4, alpha=8.0)
        load_lora(model2, path)

        for (n1, p1), (n2, p2) in zip(
            [(n, p) for n, p in model.named_parameters() if "lora_" in n],
            [(n, p) for n, p in model2.named_parameters() if "lora_" in n],
        ):
            assert torch.equal(p1, p2), f"LoRA weight mismatch in {n1}"

    def test_save_file_is_small(self, model, tmp_path, cfg):
        """LoRA adapter file should be much smaller than the full model."""
        inject_lora(model, r=4, alpha=8.0)
        path = str(tmp_path / "lora.pt")
        save_lora(model, path)

        import os

        lora_size = os.path.getsize(path)

        # Full model size
        full_path = str(tmp_path / "full.pt")
        torch.save(model.state_dict(), full_path)
        full_size = os.path.getsize(full_path)

        assert lora_size < full_size, "LoRA file should be smaller than full model"


# ─────────────────────────────────────────────────────────────────────────────
# format_alpaca
# ─────────────────────────────────────────────────────────────────────────────


class TestFormatAlpaca:
    def test_prompt_contains_instruction(self):
        ex = {"instruction": "Write a poem", "input": "", "output": "Roses are red"}
        prompt, response = format_alpaca(ex)
        assert "Write a poem" in prompt
        assert "Response" in prompt
        assert response == "Roses are red"

    def test_prompt_includes_input_when_given(self):
        ex = {"instruction": "Translate", "input": "Hello", "output": "Hola"}
        prompt, response = format_alpaca(ex)
        assert "Hello" in prompt
        assert "Input" in prompt

    def test_prompt_omits_input_section_when_empty(self):
        ex = {"instruction": "Tell a joke", "input": "", "output": "Why..."}
        prompt, response = format_alpaca(ex)
        # No "### Input:" section when input is empty
        assert "### Input:" not in prompt

    def test_response_is_output(self):
        ex = {"instruction": "Say hi", "input": "", "output": "Hello!"}
        _, response = format_alpaca(ex)
        assert response == "Hello!"


# ─────────────────────────────────────────────────────────────────────────────
# SFTDataset
# ─────────────────────────────────────────────────────────────────────────────


class TestSFTDataset:
    @pytest.fixture
    def examples(self):
        return make_tinystories_sft_examples(n=20)

    @pytest.fixture
    def dataset(self, stub_tok, examples):
        return SFTDataset(stub_tok, examples, max_seq_len=128)

    def test_output_shapes(self, dataset):
        assert len(dataset) > 0
        item = dataset[0]
        assert item["input_ids"].shape == (128,)
        assert item["labels"].shape == (128,)

    def test_prompt_region_is_masked(self, stub_tok):
        """All label positions corresponding to prompt tokens must be -1."""
        prompt = "### Instruction:\nWrite a story.\n\n### Response:\n"
        response = "Once there was a cat. The End."
        example = {"instruction": "Write a story.", "input": "", "output": response}

        ds = SFTDataset(stub_tok, [example], max_seq_len=128)
        item = ds[0]

        prompt_ids = stub_tok.encode(prompt, add_special_tokens=False)
        prompt_len = 1 + len(prompt_ids)  # <bos> + prompt

        # All labels before prompt_len must be -1
        assert (item["labels"][:prompt_len] == -1).all(), (
            "Prompt labels should all be -1 (masked)"
        )

    def test_response_region_not_all_masked(self, dataset):
        """At least some labels must be valid (not -1)."""
        for item in dataset:
            if (item["labels"] != -1).any():
                return  # found at least one valid label
        pytest.fail("All labels are masked — no response tokens to learn from")

    def test_padding_is_masked(self, stub_tok):
        """Pad positions in labels must be -1."""
        example = {"instruction": "Hi", "input": "", "output": "Hello."}
        ds = SFTDataset(stub_tok, [example], max_seq_len=128)
        item = ds[0]
        pad_mask = item["input_ids"] == stub_tok.pad_id
        if pad_mask.any():
            assert (item["labels"][pad_mask] == -1).all(), (
                "Pad positions should be masked in labels"
            )

    def test_too_long_examples_skipped(self, stub_tok):
        """Examples exceeding max_seq_len should be silently dropped."""
        long_response = "x" * 200
        example = {
            "instruction": "Say something long",
            "input": "",
            "output": long_response,
        }
        ds = SFTDataset(stub_tok, [example], max_seq_len=32)
        assert len(ds) == 0  # skipped

    def test_from_jsonl(self, stub_tok, tmp_path):
        import json

        path = tmp_path / "data.jsonl"
        examples = [
            {
                "instruction": "Tell me a story",
                "input": "",
                "output": "Once upon a time.",
            },
            {"instruction": "Say hello", "input": "", "output": "Hello world!"},
        ]
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")

        ds = SFTDataset.from_jsonl(stub_tok, path, max_seq_len=128)
        assert len(ds) == len(examples)

    def test_make_synthetic_examples(self):
        examples = make_tinystories_sft_examples(n=10)
        assert len(examples) == 10
        for ex in examples:
            assert "instruction" in ex
            assert "output" in ex
            assert len(ex["output"]) > 0


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: SFT fine-tuning step
# ─────────────────────────────────────────────────────────────────────────────


class TestSFTEndToEnd:
    def test_sft_loss_decreases(self, cfg, stub_tok):
        """20 SFT gradient steps on a fixed batch should reduce the loss."""
        model = GPT(cfg).train()
        inject_lora(model, r=4, alpha=8.0)
        freeze_non_lora(model)

        optimizer = build_adamw(model, lr=1e-3)
        scheduler = cosine_with_warmup(optimizer, warmup_steps=2, max_steps=20)

        examples = make_tinystories_sft_examples(n=10)
        dataset = SFTDataset(stub_tok, examples, max_seq_len=64)

        # Use the first example as a fixed batch
        item = dataset[0]
        tokens = item["input_ids"].unsqueeze(0)
        targets = item["labels"].unsqueeze(0)

        losses = []
        for _ in range(20):
            model.train()
            _, loss = model(tokens, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"SFT loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_only_lora_params_updated(self, cfg, stub_tok):
        """Non-LoRA parameters must not change during a LoRA fine-tuning step."""
        model = GPT(cfg).train()
        inject_lora(model, r=4, alpha=8.0)
        freeze_non_lora(model)

        # Snapshot base (non-LoRA) weights before training
        base_before = {
            n: p.data.clone() for n, p in model.named_parameters() if "lora_" not in n
        }

        examples = make_tinystories_sft_examples(n=5)
        dataset = SFTDataset(stub_tok, examples, max_seq_len=64)
        item = dataset[0]
        tokens = item["input_ids"].unsqueeze(0)
        targets = item["labels"].unsqueeze(0)

        optimizer = build_adamw(model, lr=1e-3)
        _, loss = model(tokens, targets)
        loss.backward()
        optimizer.step()

        # Base weights must be unchanged
        for name, param in model.named_parameters():
            if "lora_" not in name:
                assert torch.equal(param.data, base_before[name]), (
                    f"Non-LoRA parameter {name} was modified during LoRA training!"
                )
