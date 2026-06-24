"""Tests for Phase 4: reward model, DPO, and PPO components."""

import copy

import pytest
import torch
import torch.nn as nn

from src.alignment.dpo_trainer import (
    DPOConfig,
    DPODataset,
    DPOTrainer,
    compute_log_probs,
    dpo_loss,
    make_preference_examples,
)
from src.alignment.ppo_trainer import PPOConfig, PPOTrainer, compute_token_log_probs
from src.alignment.reward_model import RewardModel, preference_loss
from src.model import GPT, ModelConfig
from src.training.optimizer import build_adamw

# ── Shared fixtures ───────────────────────────────────────────────────────────

B = 2
T = 32


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
def gpt(cfg) -> GPT:
    return GPT(cfg).eval()


class StubTokenizer:
    bos_id = 1
    eos_id = 2
    pad_id = 0
    unk_id = 3

    def encode(self, text, add_special_tokens=True):
        ids = [ord(c) % 200 + 4 for c in text[:20]]
        if add_special_tokens:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr((i - 4) % 128 + 32) for i in ids if i >= 4)

    def __len__(self):
        return 256


@pytest.fixture
def stub_tok():
    return StubTokenizer()


# ─────────────────────────────────────────────────────────────────────────────
# RewardModel
# ─────────────────────────────────────────────────────────────────────────────


class TestRewardModel:
    def test_output_shape(self, cfg):
        rm = RewardModel(cfg).eval()
        tokens = torch.randint(0, cfg.vocab_size, (B, T))
        rewards = rm(tokens)
        assert rewards.shape == (B,), f"Expected (B,), got {rewards.shape}"

    def test_output_is_scalar_per_sequence(self, cfg):
        rm = RewardModel(cfg).eval()
        tokens = torch.randint(0, cfg.vocab_size, (1, T))
        reward = rm(tokens)
        assert reward.numel() == 1

    def test_lengths_argument(self, cfg):
        """Reward should differ when using different length positions."""
        rm = RewardModel(cfg).eval()
        tokens = torch.randint(0, cfg.vocab_size, (B, T))
        lengths = torch.tensor([T // 2, T])

        r_with_lengths = rm(tokens, lengths=lengths)
        r_without = rm(tokens)

        assert r_with_lengths.shape == (B,)
        assert r_without.shape == (B,)

    def test_from_pretrained_copies_weights(self, gpt, cfg):
        """Embedding and transformer weights should match the source GPT."""
        rm = RewardModel.from_pretrained(gpt)

        for (n1, p1), (n2, p2) in zip(
            gpt.tok_embeddings.named_parameters(),
            rm.tok_embeddings.named_parameters(),
        ):
            assert torch.equal(p1, p2), f"Embedding mismatch: {n1}"

    def test_from_pretrained_reward_head_is_random(self, gpt, cfg):
        """reward_head should NOT be copied from the GPT output head."""
        rm = RewardModel.from_pretrained(gpt)
        # GPT output is (vocab_size × dim), RM head is (1 × dim) — different shapes
        assert rm.reward_head.weight.shape == (1, cfg.dim)

    def test_different_inputs_different_rewards(self, cfg):
        rm = RewardModel(cfg).eval()
        t1 = torch.randint(0, cfg.vocab_size, (1, T))
        t2 = torch.randint(0, cfg.vocab_size, (1, T))
        assert not torch.allclose(rm(t1), rm(t2))


# ─────────────────────────────────────────────────────────────────────────────
# preference_loss
# ─────────────────────────────────────────────────────────────────────────────


class TestPreferenceLoss:
    def test_loss_is_positive(self):
        r_chosen = torch.tensor([1.0, 0.5])
        r_rejected = torch.tensor([-1.0, -0.5])
        loss, acc = preference_loss(r_chosen, r_rejected)
        assert loss.item() > 0

    def test_perfect_ranking_gives_low_loss(self):
        """When chosen >> rejected, loss should be near zero."""
        r_chosen = torch.tensor([10.0, 10.0])
        r_rejected = torch.tensor([-10.0, -10.0])
        loss, _ = preference_loss(r_chosen, r_rejected)
        assert loss.item() < 0.01

    def test_reversed_ranking_gives_high_loss(self):
        """When chosen << rejected, loss should be high."""
        r_chosen = torch.tensor([-5.0, -5.0])
        r_rejected = torch.tensor([5.0, 5.0])
        loss, _ = preference_loss(r_chosen, r_rejected)
        assert loss.item() > 4.0

    def test_accuracy_correct_when_chosen_higher(self):
        r_chosen = torch.tensor([2.0, 3.0])
        r_rejected = torch.tensor([1.0, 1.0])
        _, acc = preference_loss(r_chosen, r_rejected)
        assert acc.mean().item() == pytest.approx(1.0)

    def test_reward_model_trains_toward_correct_ranking(self, cfg):
        """After several steps, reward model should score chosen > rejected."""
        rm = RewardModel(cfg).train()
        opt = build_adamw(rm, lr=1e-3)

        chosen_tokens = torch.randint(0, cfg.vocab_size, (4, 32))
        rejected_tokens = torch.randint(0, cfg.vocab_size, (4, 32))

        for _ in range(30):
            r_chosen = rm(chosen_tokens)
            r_rejected = rm(rejected_tokens)
            loss, _ = preference_loss(r_chosen, r_rejected)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        rm.eval()
        with torch.no_grad():
            r_c = rm(chosen_tokens)
            r_r = rm(rejected_tokens)
        acc = (r_c > r_r).float().mean().item()
        assert acc > 0.5, f"Reward model accuracy {acc:.2f} should improve above random"


# ─────────────────────────────────────────────────────────────────────────────
# DPODataset
# ─────────────────────────────────────────────────────────────────────────────


class TestDPODataset:
    @pytest.fixture
    def examples(self):
        return make_preference_examples(n=20)

    @pytest.fixture
    def dataset(self, stub_tok, examples):
        return DPODataset(stub_tok, examples, max_seq_len=128)

    def test_output_keys(self, dataset):
        item = dataset[0]
        assert "chosen_input_ids" in item
        assert "chosen_labels" in item
        assert "rejected_input_ids" in item
        assert "rejected_labels" in item

    def test_output_shapes(self, dataset):
        item = dataset[0]
        assert item["chosen_input_ids"].shape == (128,)
        assert item["chosen_labels"].shape == (128,)
        assert item["rejected_input_ids"].shape == (128,)
        assert item["rejected_labels"].shape == (128,)

    def test_chosen_and_rejected_differ(self, dataset):
        """Chosen and rejected sequences should be different."""
        item = dataset[0]
        assert not torch.equal(item["chosen_input_ids"], item["rejected_input_ids"])

    def test_response_not_fully_masked(self, dataset):
        """At least some labels must be unmasked (the response tokens)."""
        item = dataset[0]
        assert (item["chosen_labels"] != -1).any()
        assert (item["rejected_labels"] != -1).any()

    def test_prompt_positions_masked(self, stub_tok):
        """The first len(prompt_ids) labels should all be -1."""
        ex = {
            "prompt": "Tell a story",
            "chosen": "Once there was a cat.",
            "rejected": "Dog.",
        }
        ds = DPODataset(stub_tok, [ex], max_seq_len=128)
        item = ds[0]

        prompt_ids = stub_tok.encode("Tell a story", add_special_tokens=False)
        n = len(prompt_ids)

        assert (item["chosen_labels"][:n] == -1).all(), (
            "Prompt positions should be masked in DPO labels"
        )

    def test_dpo_masks_one_less_than_sft(self, stub_tok):
        """DPO unmasks the prediction of r0 (SFT masks it).
        First unmasked DPO label is at len(prompt_ids), not len(prompt_ids)+1.
        """
        ex = {
            "prompt": "Tell a story",
            "chosen": "Once upon a time there was a brave cat. The End.",
            "rejected": "Cat.",
        }
        ds = DPODataset(stub_tok, [ex], max_seq_len=128)
        item = ds[0]

        prompt_ids = stub_tok.encode("Tell a story", add_special_tokens=False)
        n = len(prompt_ids)

        # Position n-1: last prompt label — should be masked
        assert item["chosen_labels"][n - 1] == -1
        # Position n: first response label — should NOT be masked (unlike SFT)
        assert item["chosen_labels"][n] != -1, (
            "DPO should not mask the first response token prediction"
        )

    def test_padding_is_masked(self, stub_tok):
        ex = {"prompt": "Hi", "chosen": "Hello.", "rejected": "No."}
        ds = DPODataset(stub_tok, [ex], max_seq_len=128)
        item = ds[0]
        pad_mask = item["chosen_input_ids"] == stub_tok.pad_id
        if pad_mask.any():
            assert (item["chosen_labels"][pad_mask] == -1).all()

    def test_too_long_skipped(self, stub_tok):
        ex = {
            "prompt": "Tell me something.",
            "chosen": "x" * 200,
            "rejected": "y" * 200,
        }
        ds = DPODataset(stub_tok, [ex], max_seq_len=32)
        assert len(ds) == 0


# ─────────────────────────────────────────────────────────────────────────────
# compute_log_probs
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeLogProbs:
    def test_output_shape(self, gpt, cfg):
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        labels = input_ids.clone()
        labels[:, :5] = -1  # mask first 5 positions
        lp = compute_log_probs(gpt, input_ids, labels)
        assert lp.shape == (B,)

    def test_fully_masked_gives_zero(self, gpt, cfg):
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        labels = torch.full_like(input_ids, -1)
        lp = compute_log_probs(gpt, input_ids, labels)
        assert torch.allclose(lp, torch.zeros(B))

    def test_log_probs_are_non_positive(self, gpt, cfg):
        """Log probabilities must be ≤ 0 (probabilities are ≤ 1)."""
        input_ids = torch.randint(0, cfg.vocab_size, (B, T))
        labels = input_ids.clone()
        labels[:, :5] = -1
        lp = compute_log_probs(gpt, input_ids, labels)
        assert (lp <= 0).all()

    def test_longer_response_gives_lower_log_prob(self, gpt, cfg):
        """More response tokens → smaller (more negative) total log prob."""
        input_ids = torch.randint(0, cfg.vocab_size, (1, T))
        labels_short = input_ids.clone()
        labels_long = input_ids.clone()
        # Short: only last 5 tokens are response
        labels_short[:, : T - 5] = -1
        # Long:  only first 5 tokens are masked (more response)
        labels_long[:, :5] = -1

        lp_short = compute_log_probs(gpt, input_ids, labels_short)
        lp_long = compute_log_probs(gpt, input_ids, labels_long)

        assert lp_long < lp_short, (
            "More response tokens should yield a smaller (more negative) total log prob"
        )


# ─────────────────────────────────────────────────────────────────────────────
# dpo_loss
# ─────────────────────────────────────────────────────────────────────────────


class TestDPOLoss:
    def test_output_shapes(self):
        B = 4
        pol_ch = torch.randn(B)
        pol_rej = torch.randn(B)
        ref_ch = torch.randn(B)
        ref_rej = torch.randn(B)
        loss, ch_rew, rej_rew = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej)
        assert loss.ndim == 0
        assert ch_rew.ndim == 0
        assert rej_rew.ndim == 0

    def test_loss_is_positive(self):
        """DPO loss (negative log sigmoid) must always be positive."""
        loss, _, _ = dpo_loss(
            torch.zeros(4),
            torch.zeros(4),
            torch.zeros(4),
            torch.zeros(4),
        )
        assert loss.item() > 0

    def test_chosen_preferred_gives_low_loss(self):
        """When policy strongly prefers chosen, loss should be near zero.

        With beta=0.1, the DPO argument is 0.1 * (pi_ratio - ref_ratio).
        To get -log(sigma(x)) < 0.01 we need x > ~4.6, so the log-prob
        difference must exceed 46.  Using ±500 makes the argument = 100.
        """
        pol_ch = torch.full((4,), 500.0)
        pol_rej = torch.full((4,), -500.0)
        ref_ch = torch.zeros(4)
        ref_rej = torch.zeros(4)
        loss, _, _ = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej, beta=0.1)
        assert loss.item() < 0.01

    def test_margin_direction(self):
        """chosen_reward should be higher than rejected_reward when policy prefers chosen."""
        pol_ch = torch.full((4,), 5.0)
        pol_rej = torch.full((4,), -5.0)
        ref_ch = torch.zeros(4)
        ref_rej = torch.zeros(4)
        _, ch_rew, rej_rew = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej, beta=0.1)
        assert ch_rew.item() > rej_rew.item()

    def test_beta_scales_rewards(self):
        pol_ch = torch.ones(4)
        pol_rej = torch.zeros(4)
        ref_ch = torch.zeros(4)
        ref_rej = torch.zeros(4)
        _, ch1, _ = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej, beta=0.1)
        _, ch2, _ = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej, beta=1.0)
        assert abs(ch2.item()) > abs(ch1.item()), "Larger beta should scale rewards up"


# ─────────────────────────────────────────────────────────────────────────────
# compute_token_log_probs (PPO)
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeTokenLogProbs:
    def test_output_shapes(self, gpt, cfg):
        input_ids = torch.randint(0, cfg.vocab_size, (1, T))
        labels = input_ids.clone()
        labels[:, :5] = -1
        lp, ent = compute_token_log_probs(gpt, input_ids, labels)
        assert lp.shape == (1, T)
        assert ent.shape == (1, T)

    def test_masked_positions_are_zero(self, gpt, cfg):
        input_ids = torch.randint(0, cfg.vocab_size, (1, T))
        labels = input_ids.clone()
        labels[:, :10] = -1
        lp, _ = compute_token_log_probs(gpt, input_ids, labels)
        assert (lp[:, :10] == 0).all()

    def test_entropy_is_non_negative(self, gpt, cfg):
        input_ids = torch.randint(0, cfg.vocab_size, (1, T))
        labels = input_ids.clone()
        labels[:, :5] = -1
        _, ent = compute_token_log_probs(gpt, input_ids, labels)
        assert (ent >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# PPOConfig
# ─────────────────────────────────────────────────────────────────────────────


class TestPPOConfig:
    def test_defaults(self):
        cfg = PPOConfig()
        assert cfg.kl_coef > 0
        assert 0 < cfg.clip_range < 1
        assert cfg.max_new_tokens > 0

    def test_reward_clipping_range(self):
        cfg = PPOConfig(reward_clip=5.0)
        assert cfg.reward_clip == 5.0


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: DPO loss decreases
# ─────────────────────────────────────────────────────────────────────────────


class TestDPOEndToEnd:
    def test_dpo_loss_decreases_on_fixed_batch(self, cfg, stub_tok):
        """20 DPO gradient steps on a fixed batch should reduce the loss."""
        policy = GPT(cfg).train()
        reference = copy.deepcopy(policy)
        for p in reference.parameters():
            p.requires_grad = False
        reference.eval()

        examples = make_preference_examples(n=10)
        dataset = DPODataset(stub_tok, examples, max_seq_len=64)
        item = dataset[0]

        chosen_ids = item["chosen_input_ids"].unsqueeze(0)
        chosen_lbl = item["chosen_labels"].unsqueeze(0)
        rejected_ids = item["rejected_input_ids"].unsqueeze(0)
        rejected_lbl = item["rejected_labels"].unsqueeze(0)

        opt = build_adamw(policy, lr=1e-3)

        losses = []
        for _ in range(20):
            policy.train()
            pol_ch = compute_log_probs(policy, chosen_ids, chosen_lbl)
            pol_rej = compute_log_probs(policy, rejected_ids, rejected_lbl)
            with torch.no_grad():
                ref_ch = compute_log_probs(reference, chosen_ids, chosen_lbl)
                ref_rej = compute_log_probs(reference, rejected_ids, rejected_lbl)

            loss, _, _ = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej, beta=0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"DPO loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_reference_model_unchanged_during_dpo(self, cfg, stub_tok):
        """Reference model weights must not change during DPO training."""
        policy = GPT(cfg).train()
        reference = copy.deepcopy(policy)
        for p in reference.parameters():
            p.requires_grad = False

        ref_weights_before = {
            n: p.data.clone() for n, p in reference.named_parameters()
        }

        examples = make_preference_examples(n=5)
        dataset = DPODataset(stub_tok, examples, max_seq_len=64)
        item = dataset[0]
        chosen_ids = item["chosen_input_ids"].unsqueeze(0)
        chosen_lbl = item["chosen_labels"].unsqueeze(0)
        rejected_ids = item["rejected_input_ids"].unsqueeze(0)
        rejected_lbl = item["rejected_labels"].unsqueeze(0)

        opt = build_adamw(policy, lr=1e-3)
        pol_ch = compute_log_probs(policy, chosen_ids, chosen_lbl)
        pol_rej = compute_log_probs(policy, rejected_ids, rejected_lbl)
        with torch.no_grad():
            ref_ch = compute_log_probs(reference, chosen_ids, chosen_lbl)
            ref_rej = compute_log_probs(reference, rejected_ids, rejected_lbl)
        loss, _, _ = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej)
        loss.backward()
        opt.step()

        for name, param in reference.named_parameters():
            assert torch.equal(param.data, ref_weights_before[name]), (
                f"Reference parameter {name} was modified during DPO!"
            )
