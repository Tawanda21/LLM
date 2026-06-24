# Phase 4 — Alignment with Feedback

A bottom-up walkthrough of everything built in Phase 4: why alignment
exists, how we teach a model which responses humans prefer, and the
mathematics behind both DPO and PPO.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Why Alignment?](#2-why-alignment)
3. [The Alignment Problem Formally](#3-the-alignment-problem-formally)
4. [Reward Model](#4-reward-model)
5. [The Bradley-Terry Model](#5-the-bradley-terry-model)
6. [Training the Reward Model](#6-training-the-reward-model)
7. [The Log-Probability Computation](#7-the-log-probability-computation)
8. [DPO — Direct Preference Optimisation](#8-dpo--direct-preference-optimisation)
9. [The DPO Loss Derived](#9-the-dpo-loss-derived)
10. [DPO Masking vs. SFT Masking](#10-dpo-masking-vs-sft-masking)
11. [PPO — Proximal Policy Optimisation](#11-ppo--proximal-policy-optimisation)
12. [The PPO Loop](#12-the-ppo-loop)
13. [The KL Penalty — Preventing Reward Hacking](#13-the-kl-penalty--preventing-reward-hacking)
14. [Reward Whitening](#14-reward-whitening)
15. [The PPO Clipped Objective](#15-the-ppo-clipped-objective)
16. [DPO vs. PPO — When to Use Each](#16-dpo-vs-ppo--when-to-use-each)
17. [Running Alignment](#17-running-alignment)
18. [Test Suite](#18-test-suite)
19. [Design Decisions at a Glance](#19-design-decisions-at-a-glance)
20. [The Complete Pipeline](#20-the-complete-pipeline)

---

## 1. The Big Picture

Phase 4 answers the question: **how do you teach a model not just to
follow instructions, but to follow them *well*?**

After SFT, the model produces grammatically correct, instruction-following
text. But "following an instruction" and "producing a response a human
actually prefers" are different. A model might:

- Give technically correct but overly verbose answers
- Add excessive caveats
- Produce coherent but off-topic responses
- Follow the letter of the instruction but miss the spirit

**Alignment teaches the model which responses humans prefer** by learning
from explicit (chosen > rejected) comparisons.

```
Phase 1: Pre-training    →  learns what language looks like
Phase 2: Pre-training    →  learns to predict text fluently
Phase 3: SFT             →  learns to follow instructions
Phase 4: Alignment       →  learns which responses are actually good
```

The Phase 4 stack:

```
(prompt, chosen_response, rejected_response) pairs
        │
        ├── Method A: DPO ──────────────────────────────────────────────┐
        │    Policy + frozen reference → DPO loss → update policy       │
        │                                                                │
        └── Method B: PPO ──────────────────────────────────────────────┘
             Step 1: Train reward model on preference pairs
             Step 2: Policy generates rollouts → RM scores
                     → KL penalty → PPO update → repeat
```

---

## 2. Why Alignment?

Consider training a model to maximise the number of likes on a platform.
A sufficiently capable model might discover that extreme content gets
more engagement than helpful content, and optimise for that — perfectly
maximising the metric while completely violating the intent.

This is **misalignment**: the model is optimising the wrong objective
because we gave it a proxy measure (likes) instead of what we actually
want (helpfulness).

The RLHF approach (Reinforcement Learning from Human Feedback) addresses
this by:

1. Collecting human preferences directly (not proxy metrics)
2. Training a reward model to predict what humans prefer
3. Fine-tuning the policy to maximise the reward model's score

---

## 3. The Alignment Problem Formally

We want a policy π that produces responses y to prompts x such that:

```
π* = argmax_π  E_{x~D, y~π(·|x)} [ R(x, y) ]

subject to:    KL(π || π_ref) ≤ δ
```

Where:
- `R(x, y)` is human preference (which we don't have directly)
- `π_ref` is the SFT reference model
- The KL constraint prevents the policy from diverging too far

The KL constraint exists because without it, the policy would quickly
learn to game the reward model — producing responses that score highly
on the RM but are actually nonsensical (reward hacking).

---

## 4. Reward Model

**File:** `src/alignment/reward_model.py`

The reward model is a GPT backbone with the vocabulary prediction head
replaced by a scalar regression head:

```
RewardModel Architecture:
  Token Embeddings  (same as GPT)
  ↓
  N × TransformerBlock  (same as GPT)
  ↓
  RMSNorm  (same as GPT)
  ↓
  Linear(dim → 1)     ← scalar reward head (NEW)
  ↓
  r(x, y)  ∈ ℝ       ← one number per sequence
```

The reward is extracted from the **last token position** — the final
hidden state contains information about the entire sequence through
the causal attention mechanism, so it serves as a natural summary
representation of the full (prompt + response).

### Initialising from the SFT model

`RewardModel.from_pretrained(gpt)` copies:
- Token embedding weights
- All transformer block weights
- Final RMSNorm weights

The scalar `reward_head` starts randomly — it must learn from scratch
what "quality" means. Starting from the SFT backbone means the RM
benefits from all the language understanding already learned during
pre-training and fine-tuning.

---

## 5. The Bradley-Terry Model

The reward model is trained using the **Bradley-Terry** probabilistic
model of pairwise preferences.

Given a pair of responses (y_w = chosen, y_l = rejected) to prompt x:

```
P(y_w > y_l | x) = σ( r(x, y_w) - r(x, y_l) )
```

Where σ is the sigmoid function. This says: the probability that the
human prefers y_w over y_l is a sigmoid of how much higher the reward
is for y_w.

The training loss is the negative log-likelihood of this model:

```
L_RM = -E_{(x, y_w, y_l)} [ log σ( r(x, y_w) - r(x, y_l) ) ]
```

**Intuition:** if r(chosen) >> r(rejected), the loss is near zero (the
model already ranks them correctly). If r(chosen) ≈ r(rejected) or
reversed, the loss is large.

This never gives the RM an absolute notion of "good" or "bad" — only
relative rankings. The absolute scale of the rewards is arbitrary; only
the differences matter.

---

## 6. Training the Reward Model

```python
# For each preference pair in the batch:
r_chosen   = reward_model(chosen_sequence)    # (B,) scalar
r_rejected = reward_model(rejected_sequence)  # (B,) scalar

loss = -log_sigmoid(r_chosen - r_rejected).mean()
loss.backward()
```

A trained reward model should satisfy:
- `r(chosen) > r(rejected)` for most preference pairs
- Accuracy (fraction where r_chosen > r_rejected) approaches 1.0

**In practice:** A good reward model needs thousands of high-quality
human-labeled preference pairs. Our `make_preference_examples()` generates
a synthetic toy dataset — sufficient for testing the pipeline, but a
real alignment run needs real human data (e.g., Anthropic HH-RLHF).

---

## 7. The Log-Probability Computation

**File:** `src/alignment/dpo_trainer.py` — `compute_log_probs()`

Both DPO and PPO need to compute the log-probability of a response
given a prompt under a language model:

```
log π(y | x) = Σ_t log π(y_t | x, y_1, ..., y_{t-1})
```

In our implementation:

```python
def compute_log_probs(model, input_ids, labels):
    logits, _ = model(input_ids)          # (B, T, V)
    log_probs = log_softmax(logits, -1)   # (B, T, V)

    response_mask  = (labels != -1)       # True where we want log probs
    clamped_labels = labels.clone()
    clamped_labels[~response_mask] = 0   # prevent gather error on -1

    gathered = log_probs.gather(2, clamped_labels.unsqueeze(2)).squeeze(2)
    return (gathered * response_mask.float()).sum(dim=1)  # (B,)
```

**How it works:**
- `logits[:, t, :]` predicts the token at position `t+1`
- `labels[:, t]` is already the shifted target (`ids[t+1]`)
- So `gathered[:, t]` = log P(next_token at t | context up to t)
- Summing over unmasked positions gives log P(full response | prompt)

---

## 8. DPO — Direct Preference Optimisation

**File:** `src/alignment/dpo_trainer.py`  
**Paper:** [Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)

### The PPO problem

PPO requires:
1. A separate reward model (expensive to train)
2. An RL loop with rollout generation (slow)
3. Careful KL penalty tuning (brittle)
4. Value function estimation (another network)

DPO sidesteps all of this with a single mathematical insight.

### The key insight

Given the KL-constrained RLHF objective, the optimal policy has a
closed form:

```
π*(y | x) = π_ref(y | x) · exp( r(x,y) / β ) / Z(x)
```

Where `Z(x)` is a normalisation constant (partition function).

Rearranging to express `r(x,y)` as a function of the policy:

```
r(x, y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
```

Plugging this into the Bradley-Terry loss:

```
L_DPO = -E [ log σ( r(x,y_w) - r(x,y_l) ) ]
       = -E [ log σ( β·log(π/π_ref) for y_w  -  β·log(π/π_ref) for y_l ) ]
```

The Z(x) terms cancel! We don't need the reward model at all — the
policy itself implicitly defines the reward.

---

## 9. The DPO Loss Derived

```
L_DPO(π_θ; π_ref) =
    -E_{(x,y_w,y_l)~D} [
        log σ(
            β · log π_θ(y_w|x)/π_ref(y_w|x)
          - β · log π_θ(y_l|x)/π_ref(y_l|x)
        )
    ]
```

In code:

```python
def dpo_loss(pol_lp_chosen, pol_lp_rejected, ref_lp_chosen, ref_lp_rejected, beta):
    pi_ratio  = pol_lp_chosen  - pol_lp_rejected   # log(π/π) difference for chosen vs rejected
    ref_ratio = ref_lp_chosen  - ref_lp_rejected   # same for reference
    losses    = -log_sigmoid(beta * (pi_ratio - ref_ratio))
    return losses.mean()
```

### What the loss is actually optimising

The term `β·log(π_θ/π_ref)` is the **implicit reward** — how much more
(or less) the policy assigns probability to a response compared to the
reference. The loss pushes the policy to assign higher implicit reward
to chosen responses than to rejected ones.

| Scenario | Policy behaviour | Loss |
|---|---|---|
| π assigns much more prob to y_w | pol_lp_chosen >> pol_lp_rejected | Near 0 ✓ |
| π assigns equal prob to both | pol_lp_chosen ≈ pol_lp_rejected | ~0.69 (log 2) |
| π assigns more prob to y_l | pol_lp_chosen << pol_lp_rejected | High ✗ |

### The β parameter

β controls how much the policy can deviate from the reference:

- **Small β (0.1):** the policy can assign very different probabilities to
  chosen vs rejected — allows large adaptation
- **Large β (1.0):** keeps the policy close to the reference — conservative
  adaptation

---

## 10. DPO Masking vs. SFT Masking

One subtle but critical implementation difference between SFT and DPO.

**SFT masking** (Phase 3 — `SFTDataset`):
```
labels[:prompt_len] = -1    # where prompt_len = 1 + len(prompt_ids)
```
The prediction of r0 (first response token from last prompt position)
is **masked**. We don't train the model to start responses.

**DPO masking** (`DPODataset`):
```
labels[:len(prompt_ids)] = -1    # one position less masked
```
The prediction of r0 is **unmasked**. This is intentional — we need
`log P(r0 | prompt)` as part of the full response log probability.

```
Position:    0      1    ...   n-1      n      n+1   ...
Input:      bos    p0   ...  p_{n-1}  r0     r1    ...
SFT label:  -1    -1   ...    -1    MASKED   r2    ...
DPO label:  -1    -1   ...    -1     r1      r2    ...
                                      ↑
                              DPO includes this!
```

If you used SFT-style labels in DPO, you'd be computing
`log P(response | context)` starting from r1, missing r0 entirely.
The log probability would be systematically different between chosen and
rejected responses in a way that doesn't reflect their true likelihood.

**Test:** `test_dpo_masks_one_less_than_sft` verifies this property
explicitly — `labels[n-1] == -1` (last prompt masked) but
`labels[n] != -1` (first response token unmasked).

---

## 11. PPO — Proximal Policy Optimisation

**File:** `src/alignment/ppo_trainer.py`  
**Paper:** [InstructGPT](https://arxiv.org/abs/2203.02155), [PPO](https://arxiv.org/abs/1707.06347)

PPO is the older, more complex approach that InstructGPT uses.
It requires a trained reward model and runs an RL loop.

The core idea: generate responses, score them, update the policy to make
high-scoring responses more likely.

---

## 12. The PPO Loop

```
for step in range(max_steps):

    # ── ROLLOUT PHASE ────────────────────────────────────────
    for prompt in prompt_batch:
        # Generate a response autoregressively
        response = policy.generate(prompt, max_new_tokens=128)

        # Score it
        rm_score = reward_model(prompt + response)  # scalar

        # Compute per-token KL from reference
        log_probs_policy = log_prob_per_token(policy, response)
        log_probs_ref    = log_prob_per_token(reference, response)
        kl = (log_probs_policy - log_probs_ref).sum()

        # Combined reward: RM score minus KL penalty
        reward = rm_score - β * kl

        # Record old log probs (frozen snapshot for PPO ratio)
        old_log_probs = log_probs_policy.detach()

    # ── UPDATE PHASE ─────────────────────────────────────────
    advantages = whiten(rewards)  # zero-mean, unit-variance

    for response in rollouts:
        cur_log_probs = log_prob_per_token(policy, response)  # trainable

        # PPO probability ratio: how much the policy changed
        ratio = exp(cur_log_probs - old_log_probs)

        # Clipped surrogate objective
        loss = -min(
            ratio * advantage,
            clip(ratio, 1-ε, 1+ε) * advantage
        ).mean()

        loss.backward()

    optimizer.step()
```

---

## 13. The KL Penalty — Preventing Reward Hacking

The KL divergence penalty is the most important part of PPO for LLMs:

```
reward = R_RM(x, y)  -  β · KL(π || π_ref)
       = R_RM(x, y)  -  β · Σ_t log(π(a_t|s_t) / π_ref(a_t|s_t))
```

**Why is this essential?**

Without the KL penalty, PPO would rapidly learn to produce responses
that score highly on the reward model while being completely incoherent:

```
No KL:  "The answer is !!!! GREAT !!!! AMAZING !!!! YES !!!!"
        → RM score: 9.8  (triggers RM's "enthusiasm" heuristic)
        → Actually awful

With KL: The policy can't deviate far from the SFT reference,
         so responses remain fluent and coherent
```

This is **reward hacking**: optimising the proxy measure (RM score) at
the expense of what we actually want (good responses).

The β parameter trades off exploration vs. constraint:
- Large β: policy stays close to SFT reference (safe, conservative)
- Small β: policy can deviate more (more adaptation, more hacking risk)

---

## 14. Reward Whitening

Before the PPO update, we normalise the rewards:

```python
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

**Why whitening?**

The reward model outputs arbitrary absolute values — the scale is
meaningless (only relative rankings matter). If the RM outputs values
like [0.02, 0.018, 0.021, 0.019], the differences are tiny and the
PPO update would be negligible. If it outputs [1000, 998, 1001, 999],
the advantages would be enormous and training would be unstable.

Whitening makes the advantage values approximately standard normal
regardless of the RM's absolute scale.

---

## 15. The PPO Clipped Objective

The PPO clip prevents the policy from making too-large updates in a
single step:

```
L_PPO = -min(
    r_t · A_t,
    clip(r_t, 1-ε, 1+ε) · A_t
)

where r_t = π_θ(a_t|s_t) / π_old(a_t|s_t)   (probability ratio)
      A_t = advantage (whitened reward)
      ε   = clip range (typically 0.2)
```

**Intuition for positive advantage (A > 0, we want more of this):**

```
r_t < 1-ε: policy moved away too much → gradient pushes r_t toward 1
r_t ∈ [1-ε, 1+ε]: normal update
r_t > 1+ε: policy already moved enough → gradient is zero (clipped)
```

The clipping prevents any single update from moving the policy by more
than a factor of (1+ε) in either direction. This avoids catastrophic
updates that destroy the language model's prior.

---

## 16. DPO vs. PPO — When to Use Each

| Aspect | DPO | PPO |
|---|---|---|
| Needs reward model? | No | Yes |
| Training stability | High | Medium (finicky) |
| Compute cost | Low (one forward pass per pair) | High (generation + RM + policy update) |
| Memory | Moderate (policy + reference) | High (policy + reference + RM) |
| Research maturity | Growing fast | Well-established |
| When to use | First choice for most tasks | When you need precise reward shaping |
| Implementation complexity | Simple | Complex |

**The verdict:** DPO is now the default choice for most LLM alignment
research. PPO is still used in production systems (like GPT-4 and
Claude) where you have the compute budget and need fine-grained reward
control.

For our project:
- **Use DPO** for the primary alignment run
- **Use PPO** to understand how RLHF works mechanistically

---

## 17. Running Alignment

### DPO (recommended)

```bash
# With synthetic demo data
python scripts/align.py \
    --method dpo \
    --sft_ckpt checkpoints/finetune/best.pt \
    --demo

# With real preference data (JSONL format)
python scripts/align.py \
    --method dpo \
    --sft_ckpt checkpoints/finetune/best.pt \
    --data_path data/preferences.jsonl \
    --max_steps 1000 \
    --beta 0.1
```

Expected output:
```
=== DPO Alignment ===
  DPODataset: 200 preference pairs ready.
DPO Training  |  β=0.1  |  steps=1000  |  lr=5.0e-05
step    10 | loss 0.6821 | margin +0.032 | lr 5.0e-05
step    20 | loss 0.6734 | margin +0.071 | lr 5.0e-05
step   100 | loss 0.5912 | margin +0.218 | lr 4.9e-05
...
```

The key metric to watch: **margin** (chosen_reward - rejected_reward).
It should increase over training. If it stays at zero, the model isn't
learning to distinguish the two response types.

### PPO

```bash
python scripts/align.py \
    --method ppo \
    --sft_ckpt checkpoints/finetune/best.pt \
    --demo \
    --max_steps 500
```

### Preference data format

```json
{"prompt": "Write a story about a brave rabbit.", "chosen": "Once upon a time...", "rejected": "Rabbit."}
{"prompt": "Explain gravity to a child.", "chosen": "Imagine you throw a ball...", "rejected": "F=mg."}
```

---

## 18. Test Suite

**File:** `tests/test_alignment.py` — 35 tests.

| Test class | What is verified |
|---|---|
| `TestRewardModel` | Shape (B,), lengths argument, from_pretrained copies weights, reward_head is new |
| `TestPreferenceLoss` | Positive loss, perfect ranking → low loss, reversed → high loss, accuracy, trains toward correct ranking |
| `TestDPODataset` | Keys, shapes, chosen≠rejected, response unmasked, prompt masked, DPO one-less masking, padding masked, too-long skipped |
| `TestComputeLogProbs` | Shape, fully masked → 0, log probs ≤ 0, longer response → lower log prob |
| `TestDPOLoss` | Shape, always positive, strong preference → low loss, margin direction, beta scaling |
| `TestComputeTokenLogProbs` | Shape, masked → 0, entropy non-negative |
| `TestPPOConfig` | Defaults valid, attributes accessible |
| `TestDPOEndToEnd` | Loss decreases over 20 steps, reference model unchanged |

The two most important tests:

**`test_dpo_masks_one_less_than_sft`** — catches the subtle DPO masking
difference. If this is wrong, the log-probability computation includes
the wrong set of tokens and the DPO loss is computing something incorrect.

**`test_reference_model_unchanged_during_dpo`** — confirms the frozen
reference model is truly frozen. If the reference updates, DPO degenerates
(both π_θ and π_ref move together, the ratio stays near zero, and the
model learns nothing).

---

## 19. Design Decisions at a Glance

| Choice | What we did | Alternative | Reason |
|---|---|---|---|
| Reward extraction | Last token position | Average pooling, CLS token | Last token sees full context via causal attention |
| RM initialisation | Copy from SFT backbone | Train from scratch | Transfer learning accelerates RM training |
| RM loss | Bradley-Terry (-log σ(r_w - r_l)) | MSE on absolute scores | Only relative rankings are observable |
| Default method | DPO | PPO | Simpler, no RM needed, more stable |
| DPO masking | mask[:len(prompt_ids)] | mask[:1+len(prompt_ids)] | Need log P(r0|prompt) for correct sequence log prob |
| β (KL coefficient) | 0.1 | 0.01 (loose) to 0.5 (tight) | 0.1 is a safe starting point |
| PPO advantages | Whitened reward | GAE with value function | Simpler; good enough for our scale |
| PPO clip range | 0.2 | 0.1 (tight) to 0.3 (loose) | Standard value from original PPO paper |
| PPO entropy bonus | 0.01 × entropy | None | Mild encouragement to explore; prevents collapse |
| Reward clipping | ±5.0 | No clipping | Prevents outlier rewards destabilising training |

---

## 20. The Complete Pipeline

With all four phases complete, the full workflow is:

```bash
# Phase 2 — Tokenizer + Pre-training
python scripts/train_tokenizer.py                    # ~5 min
python scripts/pretrain.py                           # ~9h (10k steps, RTX 4090)

# Phase 3 — Supervised Fine-Tuning
python scripts/finetune.py \
    --pretrain_ckpt checkpoints/best.pt \
    --demo                                           # ~30 min

# Phase 4 — Alignment
python scripts/align.py \
    --method dpo \
    --sft_ckpt checkpoints/finetune/best.pt \
    --demo                                           # ~15 min
```

What each phase adds to the model:

```
After pre-training:   "Once upon a time there was a cat and a dog..."
After SFT:            "### Response:\nOnce upon a time, a brave rabbit..."
After DPO/PPO:        "Once upon a time, there was a brave rabbit named Leo.
                       He learned that courage means facing your fears, not
                       the absence of them. The End."
```

The difference is subtle but real: the aligned model produces responses
that are better structured, more appropriate in tone, and more likely
to satisfy what the human instruction was actually asking for.
