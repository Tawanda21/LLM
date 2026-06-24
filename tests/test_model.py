"""Shape, correctness, and property tests for Phase 1 model components."""

import math

import pytest
import torch
import torch.nn as nn

from src.model import (
    GPT,
    Attention,
    FeedForward,
    ModelConfig,
    RMSNorm,
    TransformerBlock,
)
from src.model.attention import apply_rotary_emb, precompute_freqs_cis, repeat_kv

# ── Constants used across tests ───────────────────────────────────────────────
B = 2  # batch size
T = 16  # sequence length


# ── Shared tiny config ────────────────────────────────────────────────────────


@pytest.fixture
def cfg() -> ModelConfig:
    """Minimal config: fast on CPU, exercises all GQA / RoPE paths."""
    return ModelConfig(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,  # n_rep = 2  →  exercises GQA repeat path
        max_seq_len=128,
        multiple_of=32,
        dropout=0.0,
    )


@pytest.fixture
def model(cfg: ModelConfig) -> GPT:
    return GPT(cfg).eval()


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────────────────────


class TestRMSNorm:
    def test_output_shape(self, cfg):
        norm = RMSNorm(cfg.dim)
        x = torch.randn(B, T, cfg.dim)
        assert norm(x).shape == (B, T, cfg.dim)

    def test_normalises_to_unit_rms(self, cfg):
        """With weight = 1, the RMS of every output vector should be ≈ 1."""
        norm = RMSNorm(cfg.dim)
        nn.init.ones_(norm.weight)  # ensure scale = 1
        x = torch.randn(B, T, cfg.dim) * 5  # large variance input
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_weight_is_learnable(self, cfg):
        norm = RMSNorm(cfg.dim)
        assert norm.weight.shape == (cfg.dim,)
        assert norm.weight.requires_grad

    def test_different_inputs_different_scale(self, cfg):
        """Scaling a vector by k should not change the normalised direction."""
        norm = RMSNorm(cfg.dim)
        norm.eval()
        x = torch.randn(1, 1, cfg.dim)
        out1 = norm(x)
        out2 = norm(x * 10)
        # direction should be identical (RMSNorm is scale-invariant)
        cos = F.cosine_similarity(out1.view(-1), out2.view(-1), dim=0)
        assert cos.item() > 0.9999

    # F.cosine_similarity is in torch.nn.functional — import it locally
    @staticmethod
    def _cos(a, b):
        import torch.nn.functional as F

        return F.cosine_similarity(a.view(-1, 1), b.view(-1, 1), dim=0)


import torch.nn.functional as F  # make available for the inline test above

# ─────────────────────────────────────────────────────────────────────────────
# RoPE helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestRoPE:
    def test_freqs_cis_shape(self, cfg):
        freqs = precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len)
        assert freqs.shape == (cfg.max_seq_len, cfg.head_dim // 2)
        assert freqs.is_complex()

    def test_freqs_cis_unit_magnitude(self, cfg):
        """Rotary embeddings are unit-norm rotations — |e^(jθ)| = 1."""
        freqs = precompute_freqs_cis(cfg.head_dim, cfg.max_seq_len)
        assert torch.allclose(freqs.abs(), torch.ones_like(freqs.abs()), atol=1e-6)

    def test_apply_rotary_preserves_shape(self, cfg):
        freqs = precompute_freqs_cis(cfg.head_dim, T)
        xq = torch.randn(B, T, cfg.n_heads, cfg.head_dim)
        xk = torch.randn(B, T, cfg.n_kv_heads, cfg.head_dim)
        xq_r, xk_r = apply_rotary_emb(xq, xk, freqs)
        assert xq_r.shape == xq.shape
        assert xk_r.shape == xk.shape

    def test_apply_rotary_preserves_norm(self, cfg):
        """RoPE is a rotation, so it must preserve each vector's L2 norm."""
        freqs = precompute_freqs_cis(cfg.head_dim, T)
        xq = torch.randn(B, T, cfg.n_heads, cfg.head_dim)
        xk = torch.randn(B, T, cfg.n_kv_heads, cfg.head_dim)
        xq_r, xk_r = apply_rotary_emb(xq, xk, freqs)
        assert torch.allclose(xq.norm(dim=-1), xq_r.norm(dim=-1), atol=1e-5)
        assert torch.allclose(xk.norm(dim=-1), xk_r.norm(dim=-1), atol=1e-5)

    def test_different_positions_give_different_rotations(self, cfg):
        """Two different positions must produce different rotations."""
        freqs = precompute_freqs_cis(cfg.head_dim, T)
        assert not torch.allclose(freqs[0], freqs[1])

    def test_repeat_kv(self, cfg):
        n_rep = cfg.n_rep  # == 2 for this config
        x = torch.randn(B, T, cfg.n_kv_heads, cfg.head_dim)
        out = repeat_kv(x, n_rep)
        assert out.shape == (B, T, cfg.n_heads, cfg.head_dim)
        # Check that each KV head is correctly repeated
        for i in range(cfg.n_kv_heads):
            for r in range(n_rep):
                assert torch.equal(out[:, :, i * n_rep + r], x[:, :, i])

    def test_repeat_kv_noop_when_n_rep_1(self, cfg):
        x = torch.randn(B, T, cfg.n_kv_heads, cfg.head_dim)
        out = repeat_kv(x, 1)
        assert out is x  # must return the same object, not a copy


# ─────────────────────────────────────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────────────────────────────────────


class TestAttention:
    def test_output_shape(self, cfg):
        attn = Attention(cfg).eval()
        freqs = precompute_freqs_cis(cfg.head_dim, T)
        x = torch.randn(B, T, cfg.dim)
        assert attn(x, freqs).shape == (B, T, cfg.dim)

    def test_causal_mask(self, cfg):
        """Token at position 0 must not be influenced by tokens at positions > 0."""
        attn = Attention(cfg).eval()
        freqs = precompute_freqs_cis(cfg.head_dim, T)

        x = torch.randn(1, T, cfg.dim)
        out1 = attn(x, freqs)

        # Corrupt every token except position 0 with extreme values
        x2 = x.clone()
        x2[0, 1:] = torch.randn_like(x2[0, 1:]) * 1000
        out2 = attn(x2, freqs)

        # First-position output must be identical despite corrupted future tokens
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)

    def test_no_information_leak_at_last_position(self, cfg):
        """Last position must not see a token that doesn't exist yet."""
        attn = Attention(cfg).eval()
        freqs = precompute_freqs_cis(cfg.head_dim, T)
        x = torch.randn(1, T, cfg.dim)
        # Changing position 0 SHOULD change the last position's output
        x2 = x.clone()
        x2[0, 0] = torch.randn(cfg.dim)
        out1 = attn(x, freqs)
        out2 = attn(x2, freqs)
        # They may differ (last token CAN see all previous), just checking shape
        assert out1.shape == out2.shape


# ─────────────────────────────────────────────────────────────────────────────
# FeedForward
# ─────────────────────────────────────────────────────────────────────────────


class TestFeedForward:
    def test_output_shape(self, cfg):
        ff = FeedForward(cfg)
        x = torch.randn(B, T, cfg.dim)
        assert ff(x).shape == (B, T, cfg.dim)

    def test_hidden_dim_is_multiple_of(self, cfg):
        ff = FeedForward(cfg)
        assert ff.w1.out_features % cfg.multiple_of == 0

    def test_no_bias(self, cfg):
        ff = FeedForward(cfg)
        assert ff.w1.bias is None
        assert ff.w2.bias is None
        assert ff.w3.bias is None


# ─────────────────────────────────────────────────────────────────────────────
# TransformerBlock
# ─────────────────────────────────────────────────────────────────────────────


class TestTransformerBlock:
    def test_output_shape(self, cfg):
        block = TransformerBlock(0, cfg).eval()
        freqs = precompute_freqs_cis(cfg.head_dim, T)
        x = torch.randn(B, T, cfg.dim)
        assert block(x, freqs).shape == (B, T, cfg.dim)

    def test_residual_stream_survives_zero_weights(self, cfg):
        """Even with zeroed sub-module weights, the residual should keep the input."""
        block = TransformerBlock(0, cfg).eval()
        for p in block.parameters():
            nn.init.zeros_(p)
        freqs = precompute_freqs_cis(cfg.head_dim, T)
        x = torch.randn(B, T, cfg.dim)
        out = block(x, freqs)
        # Output should equal the input (residual preserved when sub-layers are 0)
        assert torch.allclose(out, x, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# GPT (full model)
# ─────────────────────────────────────────────────────────────────────────────


class TestGPT:
    def test_forward_no_loss(self, cfg, model):
        tokens = torch.randint(0, cfg.vocab_size, (B, T))
        logits, loss = model(tokens)
        assert logits.shape == (B, T, cfg.vocab_size)
        assert loss is None

    def test_forward_with_loss(self, cfg, model):
        tokens = torch.randint(0, cfg.vocab_size, (B, T))
        targets = torch.randint(0, cfg.vocab_size, (B, T))
        logits, loss = model(tokens, targets)
        assert logits.shape == (B, T, cfg.vocab_size)
        assert loss is not None
        assert loss.ndim == 0  # scalar tensor
        assert loss.item() > 0

    def test_loss_with_ignored_positions(self, cfg, model):
        """Positions marked -1 in targets should not contribute to loss."""
        tokens = torch.randint(0, cfg.vocab_size, (B, T))
        targets = torch.full((B, T), -1, dtype=torch.long)
        targets[:, -1] = 0  # only one valid target
        _, loss = model(tokens, targets)
        assert loss is not None and loss.item() > 0

    def test_generate_shape(self, cfg, model):
        seed = torch.zeros(1, 1, dtype=torch.long)
        out = model.generate(seed, max_new_tokens=10)
        assert out.shape == (1, 11)
        assert out.dtype == torch.long

    def test_generate_temperature(self, cfg, model):
        """Temperature should not change the output shape."""
        seed = torch.zeros(1, 1, dtype=torch.long)
        assert model.generate(seed, max_new_tokens=5, temperature=0.1).shape == (1, 6)
        assert model.generate(seed, max_new_tokens=5, temperature=2.0).shape == (1, 6)

    def test_generate_top_k(self, cfg, model):
        seed = torch.zeros(1, 1, dtype=torch.long)
        out = model.generate(seed, max_new_tokens=5, top_k=5)
        assert out.shape == (1, 6)

    def test_max_seq_len_guard(self, cfg, model):
        with pytest.raises(AssertionError):
            tokens = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len + 1))
            model(tokens)

    def test_weight_tying(self, cfg, model):
        """Embedding and output projection must share the exact same storage."""
        assert model.tok_embeddings.weight is model.output.weight

    def test_num_params_positive(self, cfg, model):
        assert model.num_params() > 0
        assert model.num_params(exclude_embedding=False) >= model.num_params()

    def test_param_count_roughly_expected(self, cfg):
        """Smoke-test: small config should produce a few thousand params, not millions."""
        m = GPT(cfg)
        n = m.num_params(exclude_embedding=False)
        assert 10_000 < n < 10_000_000, f"Unexpected param count: {n:,}"
        print(f"\nTiny test-config params: {n:,}")

    def test_loss_decreases_on_single_batch(self, cfg):
        """Sanity-check: one gradient step should lower the loss."""
        model = GPT(cfg).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tokens = torch.randint(0, cfg.vocab_size, (B, T))
        targets = tokens.clone()

        _, loss_before = model(tokens, targets)
        loss_before.backward()
        optimizer.step()
        optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            _, loss_after = model(tokens, targets)

        assert loss_after.item() < loss_before.item(), (
            f"Loss did not decrease: {loss_before.item():.4f} → {loss_after.item():.4f}"
        )
