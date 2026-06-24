# Phase 1 — Transformer Architecture

A bottom-up walkthrough of every component we built, what the math means,
and why each design decision was made.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [ModelConfig — the single source of truth](#2-modelconfig)
3. [RMSNorm — normalisation without the mean](#3-rmsnorm)
4. [RoPE — how the model knows word order](#4-rope)
5. [Attention — what every token looks at](#5-attention)
6. [Grouped Query Attention (GQA)](#6-grouped-query-attention-gqa)
7. [FeedForward — SwiGLU](#7-feedforward--swiglu)
8. [TransformerBlock — one layer](#8-transformerblock--one-layer)
9. [GPT — the full model](#9-gpt--the-full-model)
10. [Weight Initialisation](#10-weight-initialisation)
11. [Generation — how we produce text](#11-generation)
12. [Test Suite — what we verified](#12-test-suite)
13. [Design Decisions at a Glance](#13-design-decisions-at-a-glance)
14. [Parameter Count](#14-parameter-count)

---

## 1. The Big Picture

A language model does one thing: **given a sequence of tokens, predict the
next token**.  We do this with a *decoder-only transformer* — the same
family as GPT-2, LLaMA, Mistral, and Gemma.

The data flows like this:

```
Token IDs  (B, T)
    │
    ▼
Token Embedding          maps each integer id → a learned vector of size `dim`
    │
    ▼
× N Transformer Blocks   each block refines every token's representation
    │  ┌─────────────────────────────────────────────┐
    │  │  RMSNorm → Attention (RoPE, GQA) → residual │
    │  │  RMSNorm → FeedForward (SwiGLU)  → residual │
    │  └─────────────────────────────────────────────┘
    │
    ▼
Final RMSNorm
    │
    ▼
Linear Head              projects `dim` → `vocab_size` logits
    │
    ▼
Softmax → probability over every token in the vocabulary
```

> **Decoder-only** means there is no encoder and no cross-attention.
> The model only sees its own past — it cannot peek at future tokens.
> This is enforced by the *causal mask* inside attention.

---

## 2. ModelConfig

**File:** `src/model/config.py`

Every number that controls the model lives in one `@dataclass`:

```python
@dataclass
class ModelConfig:
    vocab_size: int = 32_000   # number of unique tokens
    dim:        int = 512      # width of the residual stream
    n_layers:   int = 8        # depth (number of stacked blocks)
    n_heads:    int = 8        # query attention heads
    n_kv_heads: int = 4        # key/value heads (GQA)
    max_seq_len: int = 2048    # longest sequence the model can process
    multiple_of: int = 256     # FFN hidden dim alignment
    dropout:    float = 0.0
    norm_eps:   float = 1e-6
    rope_theta: float = 10_000.0
```

Two derived properties are computed from the above:

| Property | Formula | Meaning |
|---|---|---|
| `head_dim` | `dim // n_heads` | size of each attention head's vector |
| `n_rep` | `n_heads // n_kv_heads` | how many Q heads share each KV head |

**Why a dataclass?**  
It is plain Python with no magic — you can print it, copy it, serialise it
to YAML, and pass it around with zero overhead. Every sub-module receives the
same `config` object, so there is no risk of mismatched values.

**Presets:**

| Config | `dim` | `n_layers` | `n_heads` | Approx params |
|---|---|---|---|---|
| `small_config()` | 512 | 8 | 8 | ~50 M |
| `medium_config()` | 1024 | 24 | 16 | ~350 M |

---

## 3. RMSNorm

**File:** `src/model/normalization.py`  
**Paper:** [Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)

### Why normalise at all?

Without normalisation, activations grow or shrink exponentially as they pass
through many layers, making training unstable. Normalisation keeps them in a
consistent range.

### LayerNorm vs. RMSNorm

The original transformer used **LayerNorm**:

```
LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + ε) * weight + bias
```

It subtracts the mean (re-centering) and then divides by the standard
deviation (re-scaling).

**RMSNorm** skips the mean subtraction entirely:

```
RMSNorm(x) = x / RMS(x) * weight
RMS(x)     = sqrt( mean(x²) + ε )
```

> **Intuition:** The re-centering step (subtracting the mean) is expensive
> and turns out to be largely unnecessary. The re-scaling (dividing by RMS)
> is what actually stabilises training. Removing the mean gives ~15% speed
> improvement with equivalent results.

### Implementation

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # learnable scale

    def forward(self, x):
        # rsqrt(a) = 1/sqrt(a)  — one fused op instead of sqrt then divide
        rrms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rrms) * self.weight
```

- `x.pow(2).mean(-1)` → variance without mean subtraction
- `.add(eps)` → numerical stability (prevents divide-by-zero)
- `.rsqrt()` → `1 / sqrt(...)` in a single GPU op
- `* self.weight` → learnable per-channel rescaling (no bias)

### Key property — scale invariance

If you multiply an input by a constant `k`, the output is identical:

```
RMSNorm(k·x) = (k·x) / RMS(k·x) * w
             = (k·x) / (k · RMS(x)) * w
             = x / RMS(x) * w
             = RMSNorm(x)
```

This is verified by `test_different_inputs_different_scale`.

---

## 4. RoPE

**File:** `src/model/attention.py`  
**Paper:** [Su et al., 2021 — RoFormer](https://arxiv.org/abs/2104.09864)

### Why position encodings?

Self-attention is *permutation-equivariant* by default: if you shuffle
the tokens, the attention scores just shuffle too. The model has no idea
which token came first. Position encodings break this symmetry by injecting
position information into the token representations.

### The original approach: sinusoidal embeddings

The original transformer added a fixed sinusoidal vector to each token
embedding *before* any processing. This works but has a key flaw: **once
positions are baked into the embedding, the model cannot easily distinguish
relative distances** — it only knows absolute positions.

### RoPE: rotations in 2D

RoPE encodes position by *rotating* the query and key vectors before the
dot-product. The key insight is that if you rotate `q` by angle `θ_pos_q`
and `k` by angle `θ_pos_k`, their dot product becomes:

```
q · k  =  |q| |k| cos( θ_pos_q - θ_pos_k )
```

The score now depends on the **relative position** `(pos_q - pos_k)`, not
the absolute positions. This is exactly what you want for generalisation.

### How the rotation works

The head dimension is split into pairs: `(d₀, d₁), (d₂, d₃), ...`.  
Each pair is treated as a 2D complex number and multiplied by a unit-magnitude
rotator `e^(j·m·θᵢ)`, where `m` is the position and `θᵢ` is a frequency:

```
θᵢ = 1 / (10000 ^ (2i / head_dim))     for i = 0, 1, ..., head_dim/2 - 1
```

Low `i` → high frequency (changes fast with position).  
High `i` → low frequency (changes slowly — encodes coarse position).

This geometric sequence of frequencies is the same idea as sinusoidal
encodings, but applied as a rotation at every layer rather than once at
the input.

### Precomputing frequencies

```python
def precompute_freqs_cis(head_dim, max_seq_len, theta=10_000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len)
    freqs = torch.outer(positions, freqs)    # (max_seq_len, head_dim//2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex: e^(j·freq)
```

`torch.polar(r, θ)` produces `r · e^(jθ)`.  With `r=1` everywhere, these
are unit-magnitude complex rotators — they rotate without scaling.

**Shape:** `(max_seq_len, head_dim // 2)` complex.

### Applying the rotation

```python
def apply_rotary_emb(xq, xk, freqs_cis):
    # Reshape: (..., head_dim) real → (..., head_dim//2) complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # (1, T, 1, D//2)

    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

Each consecutive pair of real dimensions `(x₀, x₁)` is treated as the
real and imaginary parts of a complex number.  Multiplying by `e^(jθ)`
applies a 2D rotation to that pair.

**Key test:** `test_apply_rotary_preserves_norm` confirms RoPE is a
pure rotation (norm-preserving). If it changed norms, it would be
scaling attention scores unpredictably.

---

## 5. Attention

**File:** `src/model/attention.py`  
**Paper:** [Vaswani et al., 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Intuition

Each token gets to ask a question ("what am I looking for?") and broadcast
an answer ("what do I contain?"). The *query* encodes the question, the
*key* encodes the answer label, and the *value* encodes the answer content.

```
Attention(Q, K, V) = softmax( Q·Kᵀ / sqrt(head_dim) ) · V
```

- `Q·Kᵀ` — how similar is each query to each key? (a score matrix)
- `/ sqrt(head_dim)` — scale down to prevent softmax saturation
- `softmax(...)` — turn scores into a probability distribution (weights)
- `· V` — weighted average of all value vectors

### The causal mask

In a language model, token at position `t` must **not** see tokens at
positions `> t` (that would be cheating — those tokens don't exist yet
at generation time). We enforce this with a *causal mask*: set all scores
above the diagonal to `-∞` before the softmax, so they get probability 0.

```
Scores (T=4):          After masking:
 q₀ → [s₀₀  s₀₁  s₀₂  s₀₃]     [s₀₀  -∞   -∞   -∞ ]
 q₁ → [s₁₀  s₁₁  s₁₂  s₁₃]     [s₁₀  s₁₁  -∞   -∞ ]
 q₂ → [s₂₀  s₂₁  s₂₂  s₂₃]     [s₂₀  s₂₁  s₂₂  -∞ ]
 q₃ → [s₃₀  s₃₁  s₃₂  s₃₃]     [s₃₀  s₃₁  s₃₂  s₃₃]
```

We use `F.scaled_dot_product_attention(..., is_causal=True)` which handles
the mask automatically and dispatches to the Flash Attention CUDA kernel
when running on GPU (significantly faster and more memory-efficient).

### Multi-head attention

Instead of one big attention operation, we run `n_heads` smaller ones in
parallel — each head can specialise in different relationships (syntax,
coreference, proximity, etc.).

The `dim`-dimensional stream is split into `n_heads` chunks of size
`head_dim = dim // n_heads`. Each chunk gets its own Q, K, V projections,
runs attention independently, and the outputs are concatenated and projected
back to `dim`.

---

## 6. Grouped Query Attention (GQA)

**Paper:** [Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)

### The KV-cache problem

At inference time, an autoregressive model generates one token at a time.
To avoid recomputing K and V for all previous tokens on every step, we
cache them. With standard MHA and `n_heads=32`, this cache grows as:

```
KV cache size = 2 × n_layers × n_heads × head_dim × seq_len × dtype_bytes
```

For a large model at long context, this becomes gigabytes.

### GQA: share K/V heads across Q heads

GQA uses fewer K/V heads than Q heads. `n_kv_heads` K/V projections serve
`n_heads` Q projections, grouped `n_rep = n_heads // n_kv_heads` at a time.

```
n_heads    = 8   (8 separate Q projections)
n_kv_heads = 4   (4 shared KV projections)
n_rep      = 2   (each KV head serves 2 Q heads)
```

At inference, the KV cache shrinks by `n_rep×` with minimal quality loss.

```
Standard MHA:   cache ∝ n_heads
GQA (ours):     cache ∝ n_kv_heads  =  n_heads / n_rep
MQA (extreme):  cache ∝ 1           (all Q heads share one KV head)
```

### Implementation: `repeat_kv`

Since PyTorch's matmul expects matching head counts, we expand K and V by
repeating each KV head `n_rep` times:

```python
def repeat_kv(x, n_rep):
    # x: (B, T, n_kv_heads, head_dim)
    B, T, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
         .expand(B, T, n_kv_heads, n_rep, head_dim)
         .reshape(B, T, n_kv_heads * n_rep, head_dim)
    )
```

After `repeat_kv`, K and V have shape `(B, T, n_heads, head_dim)` and the
standard attention matmul works unchanged.

### Full Attention forward pass

```
x (B, T, dim)
│
├── wq → Q (B, T, n_heads × head_dim)    Q has n_heads heads
├── wk → K (B, T, n_kv_heads × head_dim) K has fewer heads
└── wv → V (B, T, n_kv_heads × head_dim)

Apply RoPE to Q and K
Repeat K and V → (B, T, n_heads, head_dim)

Transpose → (B, n_heads, T, head_dim)

scaled_dot_product_attention(Q, K, V, is_causal=True)
→ out (B, n_heads, T, head_dim)

Transpose + reshape → (B, T, dim)
└── wo → (B, T, dim)
```

---

## 7. FeedForward — SwiGLU

**File:** `src/model/feedforward.py`  
**Paper:** [Shazeer, 2020](https://arxiv.org/abs/2002.05202)

### What FFN does

After attention lets tokens communicate with each other, the FFN processes
each token *independently* — it is a position-wise transformation. Think of
attention as the "communication" step and FFN as the "thinking" step.

### From ReLU to SwiGLU

The original FFN used:
```
FFN(x) = ReLU(x·W1) · W2
```

SwiGLU replaces this with a *gated* version:
```
FFN(x) = ( SiLU(x·W1) ⊙ x·W3 ) · W2
```

where `⊙` is element-wise multiplication.

**Three weight matrices instead of two:**
- `W1` (gate): passes through `SiLU` activation
- `W3` (up): the "content" projection — no activation
- `W2` (down): projects back to `dim`

The gate `SiLU(x·W1)` acts as a soft selector: it decides which features
in the up-projection `x·W3` to keep. This gating mechanism lets the network
be more selective than a standard ReLU, which is why SwiGLU outperforms it.

**SiLU** (Sigmoid Linear Unit): `SiLU(x) = x · σ(x)`  
Unlike ReLU, SiLU is smooth and non-monotonic — it has a small negative
lobe which gives gradient information even for negative inputs.

### Hidden dimension sizing

Standard FFN: `hidden = 4 × dim`

LLaMA uses `(8/3) × dim`, rounded up to the nearest `multiple_of`:

```python
hidden_dim = int(2 * 4 * config.dim / 3)          # ≈ (8/3) × dim
hidden_dim = multiple_of * ceil(hidden_dim / multiple_of)
```

This gives slightly fewer parameters than `4×` while performing better,
because the third weight matrix `W3` is additional capacity that replaces
some of the raw width.

---

## 8. TransformerBlock — one layer

**File:** `src/model/transformer_block.py`

### Pre-norm vs. post-norm

**Original transformer (post-norm):**
```
x → SubLayer → x + SubLayer(x) → LayerNorm
```

**Modern (pre-norm, what we use):**
```
x → LayerNorm → SubLayer → x + SubLayer(LayerNorm(x))
```

In pre-norm, the residual path `x` flows through the block **unchanged**
(no normalisation in the skip connection). Gradients from the loss can flow
directly back to early layers without passing through any normaliser, which
prevents vanishing gradients and makes training stable even without a
learning rate warmup.

### Our block

```python
def forward(self, x, freqs_cis):
    x = x + self.attention(self.attention_norm(x), freqs_cis)
    x = x + self.feed_forward(self.ffn_norm(x))
    return x
```

Two sub-layers, each with:
1. Pre-norm (RMSNorm applied to a copy of `x`)
2. Sub-layer computation (Attention or FFN)
3. Residual addition back to the original `x`

**Test insight:** `test_residual_stream_survives_zero_weights` zeros all
parameters and confirms the output equals the input. This proves the
residual connections are wired correctly — even a completely untrained
model passes the input through unchanged.

---

## 9. GPT — the full model

**File:** `src/model/gpt.py`

### Assembly

```python
self.tok_embeddings = nn.Embedding(vocab_size, dim)
self.layers         = nn.ModuleList([TransformerBlock(i, config) for i in range(n_layers)])
self.norm           = RMSNorm(dim)
self.output         = nn.Linear(dim, vocab_size, bias=False)
```

### Weight tying

The input embedding and the output linear head share the **exact same
weight matrix**:

```python
self.tok_embeddings.weight = self.output.weight
```

This means token embeddings and output logit projections are learned
jointly. For a `vocab_size=32000, dim=512` model, this saves
32000 × 512 = **16.4 M parameters** with no quality loss — often
with better perplexity, because the embedding space is forced to be
consistent with what the model predicts.

### Loss function

```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),
    targets.view(-1),
    ignore_index=-1,
)
```

**Cross-entropy loss** is the negative log-likelihood of the correct
next token:

```
loss = -log( p(correct_token) )
```

If the model perfectly predicts every token, `loss = 0`.  
A random model over 32k vocab produces `loss ≈ log(32000) ≈ 10.4`.

`ignore_index=-1` lets us pad sequences without polluting the loss —
we'll use this in the data pipeline to handle variable-length sequences.

---

## 10. Weight Initialisation

Good initialisation is critical — bad starting weights can cause gradients
to explode or vanish before training even begins.

### Standard weights

All `nn.Linear` and `nn.Embedding` weights are initialised with:
```
N(μ=0, σ=0.02)
```

This is the GPT-2 scheme: a narrow normal distribution that keeps
activations near zero at the start.

### Residual scaling

The output projections of each sub-layer (`wo` in attention, `w2` in FFN)
are scaled down by an extra factor:

```python
scale = 0.02 / sqrt(2 × n_layers)
```

**Why?** Each transformer block adds two residual contributions (attention
+ FFN). With `n_layers` blocks, the total residual variance at the final
layer is proportional to `n_layers`. Scaling the output projections by
`1/sqrt(2 × n_layers)` cancels this growth, keeping the initial output
variance the same regardless of model depth.

Without this, a 24-layer model would have 5× larger activations than an
8-layer model at initialisation — causing very different effective learning
rates.

---

## 11. Generation

### Autoregressive decoding

The model generates one token at a time:

```
[BOS]  →  predicts token 1
[BOS, t1]  →  predicts token 2
[BOS, t1, t2]  →  predicts token 3
...
```

At each step we:
1. Run the full forward pass on the current sequence
2. Take only the **last** position's logits (the prediction for the next token)
3. Sample from the resulting probability distribution
4. Append the sampled token and repeat

### Temperature

```python
logits = logits / temperature
```

- `temperature < 1` → sharper distribution, more deterministic, less creative
- `temperature > 1` → flatter distribution, more random, more diverse
- `temperature = 1` → unmodified distribution

### Top-k sampling

```python
topk_vals, _ = torch.topk(logits, k)
logits[logits < topk_vals[:, [-1]]] = float("-inf")
```

Before sampling, restrict to only the `k` highest-logit tokens. This
prevents the model from ever picking a very unlikely token (which can
produce incoherent output) while still allowing diversity among the
plausible choices.

---

## 12. Test Suite

**File:** `tests/test_model.py` — 30 tests, all passing.

| Test class | What is verified |
|---|---|
| `TestRMSNorm` | Output shape, unit-RMS property, learnable weight, scale invariance |
| `TestRoPE` | Frequency shape & unit magnitude, norm preservation, unique positions, `repeat_kv` correctness |
| `TestAttention` | Output shape, causal mask (position 0 blind to future tokens) |
| `TestFeedForward` | Output shape, `multiple_of` alignment, no bias |
| `TestTransformerBlock` | Output shape, residual stream with zeroed weights |
| `TestGPT` | Forward shapes, loss computation, masked positions, generate shapes, temperature/top-k, seq-len guard, weight tying, param count, **loss decreases on one gradient step** |

The most important test is `test_loss_decreases_on_single_batch` — it
proves the entire stack (forward pass, loss, backward pass, parameter
update) is wired correctly end-to-end.

---

## 13. Design Decisions at a Glance

| Component | Our choice | Alternative | Reason for our choice |
|---|---|---|---|
| Positional encoding | RoPE | Sinusoidal, ALiBi, learned | Better length generalisation; encodes relative position in dot product |
| Normalisation | RMSNorm (pre-norm) | LayerNorm (post-norm) | Faster; more stable gradients; no re-centering needed |
| FFN activation | SwiGLU | ReLU, GeLU, GEGLU | Better empirical performance; used in LLaMA, PaLM |
| Attention variant | GQA | MHA, MQA | Reduces KV cache at inference; minimal quality loss |
| Output projection | Tied to embedding | Separate linear | Saves ~16 M params; improves perplexity |
| Bias terms | None | Bias in all linears | Consistent with LLaMA; slight efficiency gain |
| Attention kernel | `F.scaled_dot_product_attention` | Manual matmul | Auto-dispatches to Flash Attention on GPU |
| Init | GPT-2 (σ=0.02) + residual scale | Glorot, orthogonal | Proven stable; prevents depth-related variance growth |

---

## 14. Parameter Count

For the **small config** (`dim=512, n_layers=8, n_heads=8, n_kv_heads=4`):

| Component | Parameters |
|---|---|
| Token embedding (tied) | 32,000 × 512 = 16.4 M |
| Per-layer attention (wq + wk + wv + wo) | ~786 K |
| Per-layer FFN (w1 + w2 + w3, hidden=1536) | ~2.36 M |
| Per-layer RMSNorm × 2 | ~1 K |
| **× 8 layers** | ~25.2 M |
| Final RMSNorm | ~0.5 K |
| **Total (excl. tied embedding)** | **~25 M** |
| **Total (incl. tied embedding)** | **~41 M** |

> The embedding is counted once even though it appears in two places
> (`tok_embeddings` and `output`), because they share the same tensor.

---

## What's Next — Phase 2

With the model architecture complete, Phase 2 builds everything needed to
actually train it:

```
src/tokenizer/   ← BPE tokenizer trained on our corpus
src/data/        ← streaming dataset, sequence packing, DataLoader
src/training/    ← training loop, AdamW, cosine LR schedule, checkpointing
```

We will train on the **TinyStories** dataset — a corpus of short stories
written for children, small enough to train on a single GPU in hours but
rich enough to produce coherent English text.
