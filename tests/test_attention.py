"""Deep unit tests for attention internals.

Run with:
    python -m pytest tests/test_attention.py -v
"""

import pytest
import torch

from src.model.attention import (
    Attention,
    apply_rotary_emb,
    precompute_freqs_cis,
    repeat_kv,
)
from src.model.config import ModelConfig

B = 2
T = 16
HEAD_DIM = 16


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


# ─────────────────────────────────────────────────────────────────────────────
# precompute_freqs_cis
# ─────────────────────────────────────────────────────────────────────────────


class TestPrecomputeFreqsCis:
    def test_output_shape(self):
        freqs = precompute_freqs_cis(HEAD_DIM, max_seq_len=32)
        assert freqs.shape == (32, HEAD_DIM // 2)

    def test_output_is_complex(self):
        freqs = precompute_freqs_cis(HEAD_DIM, max_seq_len=32)
        assert freqs.is_complex()

    def test_unit_magnitude(self):
        """e^(j*theta) always has magnitude exactly 1."""
        freqs = precompute_freqs_cis(HEAD_DIM, max_seq_len=32)
        magnitudes = freqs.abs()
        assert torch.allclose(magnitudes, torch.ones_like(magnitudes), atol=1e-6)

    def test_frequencies_decrease_with_dimension(self):
        """Higher dimension index → lower rotation frequency."""
        freqs = precompute_freqs_cis(HEAD_DIM, max_seq_len=32)
        angles = freqs[1].angle()
        for i in range(len(angles) - 1):
            assert angles[i] > angles[i + 1]

    def test_position_zero_is_identity(self):
        """At position 0, RoPE must not change the vector."""
        freqs = precompute_freqs_cis(HEAD_DIM, max_seq_len=10)
        q = torch.randn(1, 1, 1, HEAD_DIM)
        k = torch.randn(1, 1, 1, HEAD_DIM)
        freqs_pos0 = freqs[:1]
        q_rot, _ = apply_rotary_emb(q, k, freqs_pos0)
        assert torch.allclose(q_rot, q, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# repeat_kv
# ─────────────────────────────────────────────────────────────────────────────


class TestRepeatKV:
    def test_noop_when_n_rep_is_1(self):
        x = torch.randn(B, T, 4, HEAD_DIM)
        out = repeat_kv(x, n_rep=1)
        assert out is x

    def test_output_shape_with_n_rep_4(self):
        n_kv, n_rep = 2, 4
        x = torch.randn(B, T, n_kv, HEAD_DIM)
        out = repeat_kv(x, n_rep=n_rep)
        assert out.shape == (B, T, n_kv * n_rep, HEAD_DIM)

    def test_each_kv_head_is_repeated_consecutively(self):
        n_kv, n_rep = 3, 2
        x = torch.randn(B, T, n_kv, HEAD_DIM)
        out = repeat_kv(x, n_rep=n_rep)
        for i in range(n_kv):
            for r in range(n_rep):
                assert torch.equal(out[:, :, i * n_rep + r], x[:, :, i])


# ─────────────────────────────────────────────────────────────────────────────
# apply_rotary_emb
# ─────────────────────────────────────────────────────────────────────────────


class TestApplyRotaryEmb:
    def test_preserves_query_shape(self):
        freqs = precompute_freqs_cis(HEAD_DIM, T)
        xq = torch.randn(B, T, 2, HEAD_DIM)
        xk = torch.randn(B, T, 2, HEAD_DIM)
        xq_r, _ = apply_rotary_emb(xq, xk, freqs)
        assert xq_r.shape == xq.shape

    def test_preserves_key_shape(self):
        freqs = precompute_freqs_cis(HEAD_DIM, T)
        xq = torch.randn(B, T, 2, HEAD_DIM)
        xk = torch.randn(B, T, 2, HEAD_DIM)
        _, xk_r = apply_rotary_emb(xq, xk, freqs)
        assert xk_r.shape == xk.shape

    def test_preserves_query_norm(self):
        """RoPE is a rotation — it must not change the L2 norm."""
        freqs = precompute_freqs_cis(HEAD_DIM, T)
        xq = torch.randn(B, T, 2, HEAD_DIM)
        xk = torch.randn(B, T, 2, HEAD_DIM)
        xq_r, _ = apply_rotary_emb(xq, xk, freqs)
        assert torch.allclose(xq.norm(dim=-1), xq_r.norm(dim=-1), atol=1e-5)

    def test_relative_position_invariance(self):
        """dot(rotate(q,3), rotate(k,1)) == dot(rotate(q,2), rotate(k,0))"""
        freqs = precompute_freqs_cis(HEAD_DIM, max_seq_len=10)
        q = torch.randn(HEAD_DIM)
        k = torch.randn(HEAD_DIM)

        def rotate_at(v: torch.Tensor, pos: int) -> torch.Tensor:
            v4d = v.view(1, 1, 1, HEAD_DIM)
            f = freqs[pos : pos + 1]
            rotated, _ = apply_rotary_emb(v4d, v4d, f)
            return rotated.view(HEAD_DIM)

        dot_31 = (rotate_at(q, 3) * rotate_at(k, 1)).sum()
        dot_20 = (rotate_at(q, 2) * rotate_at(k, 0)).sum()
        assert torch.allclose(dot_31, dot_20, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Attention module
# ─────────────────────────────────────────────────────────────────────────────


class TestAttentionModule:
    def test_gqa_output_shape(self, cfg):
        attn = Attention(cfg).eval()
        freqs = precompute_freqs_cis(cfg.head_dim, T)
        x = torch.randn(B, T, cfg.dim)
        assert attn(x, freqs).shape == (B, T, cfg.dim)

    def test_mha_output_shape(self):
        mha_cfg = ModelConfig(
            vocab_size=256,
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=4,
            max_seq_len=128,
            multiple_of=32,
        )
        attn = Attention(mha_cfg).eval()
        freqs = precompute_freqs_cis(mha_cfg.head_dim, T)
        x = torch.randn(B, T, mha_cfg.dim)
        assert attn(x, freqs).shape == (B, T, mha_cfg.dim)

    def test_mqa_output_shape(self):
        mqa_cfg = ModelConfig(
            vocab_size=256,
            dim=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=1,
            max_seq_len=128,
            multiple_of=32,
        )
        attn = Attention(mqa_cfg).eval()
        freqs = precompute_freqs_cis(mqa_cfg.head_dim, T)
        x = torch.randn(B, T, mqa_cfg.dim)
        assert attn(x, freqs).shape == (B, T, mqa_cfg.dim)

    def test_no_nan_in_output(self, cfg):
        attn = Attention(cfg).eval()
        freqs = precompute_freqs_cis(cfg.head_dim, T)
        x = torch.randn(B, T, cfg.dim)
        out = attn(x, freqs)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_causal_consistency_across_sequence_lengths(self, cfg):
        """Outputs at positions 0..7 must be identical for T=8 and T=16."""
        attn = Attention(cfg).eval()
        x8 = torch.randn(1, 8, cfg.dim)
        x16 = torch.cat([x8, torch.randn(1, 8, cfg.dim)], dim=1)
        f8 = precompute_freqs_cis(cfg.head_dim, 8)
        f16 = precompute_freqs_cis(cfg.head_dim, 16)
        out8 = attn(x8, f8)
        out16 = attn(x16, f16)
        assert torch.allclose(out16[:, :8, :], out8, atol=1e-5)
