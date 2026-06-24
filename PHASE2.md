# Phase 2 — Pre-Training

A bottom-up walkthrough of everything built in Phase 2: how raw text becomes
token ids, how those ids are fed to the model efficiently, and how the training
loop works from gradient accumulation all the way to checkpointing.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Tokenization — BPE](#2-tokenization--bpe)
3. [Special Tokens](#3-special-tokens)
4. [Data Pipeline — PackedDataset](#4-data-pipeline--packeddataset)
5. [Sequence Packing vs. Padding](#5-sequence-packing-vs-padding)
6. [Text Preprocessing](#6-text-preprocessing)
7. [The Training Objective](#7-the-training-objective)
8. [Optimizer — AdamW with Weight Decay Grouping](#8-optimizer--adamw-with-weight-decay-grouping)
9. [LR Schedule — Cosine with Warmup](#9-lr-schedule--cosine-with-warmup)
10. [Gradient Accumulation](#10-gradient-accumulation)
11. [Mixed Precision — bf16](#11-mixed-precision--bf16)
12. [Gradient Clipping](#12-gradient-clipping)
13. [Checkpointing](#13-checkpointing)
14. [The Trainer](#14-the-trainer)
15. [Running Training](#15-running-training)
16. [What to Watch During Training](#16-what-to-watch-during-training)
17. [Design Decisions at a Glance](#17-design-decisions-at-a-glance)
18. [What's Next — Phase 3](#18-whats-next--phase-3)

---

## 1. The Big Picture

Phase 2 answers the question: **how do you go from a pile of text files to a
trained language model?**

```
Raw text corpus (TinyStories)
        │
        ▼
  BPETokenizer.train()         ← learn a vocabulary of subword units
        │
        ▼
  PackedDataset                ← stream, tokenise, pack into fixed-length chunks
        │
        ▼
  DataLoader (batch_size=8)    ← collate chunks into GPU batches
        │
        ▼
  Trainer.train()
    ├── GPT.forward()          ← compute logits + cross-entropy loss
    ├── accelerator.backward() ← backprop (mixed precision aware)
    ├── clip_grad_norm_()      ← prevent gradient explosion
    ├── AdamW.step()           ← update weights
    ├── cosine_scheduler.step()← decay the learning rate
    └── save_checkpoint()      ← persist state to disk periodically
```

The corpus we use is **TinyStories** — a dataset of short English stories
written for children. It is:

- Small enough to download and iterate in minutes
- Linguistically rich enough that a small model can learn coherent English
- Available on HuggingFace Hub with a streaming API (no full download needed)

---

## 2. Tokenization — BPE

**File:** `src/tokenizer/bpe_tokenizer.py`

### Why tokenize at all?

Neural networks work with numbers, not strings. We need a way to map text to
integers (and back). The mapping must:
1. Cover the entire vocabulary of the language
2. Handle unseen words gracefully
3. Be compact (not one int per character, not one int per word)

### What is BPE?

**Byte-Pair Encoding** is a compression algorithm adapted for tokenisation.
It starts with individual bytes (so it can handle any Unicode character) and
iteratively merges the most frequent adjacent pair into a new token.

Example of the merge process:

```
Start:    [ h, e, l, l, o,  , w, o, r, l, d ]
Step 1:   merge (l, o) → lo      [ h, e, lo, lo,  , w, o, r, lo, d ]
          (wait, that's wrong — BPE is greedy on frequency)
```

More realistic:

```
Corpus: "low lower lowest"

Byte level:  l o w   l o w e r   l o w e s t

Frequent pairs:
  (l, o) → 3 times   → merge → lo
  (lo, w) → 3 times  → merge → low
  (l, o, w, e) ...   → merge → lowe   ...

Final vocabulary might include:
  l, o, w, e, r, s, t, lo, low, lowe, lower, lowest
```

Each merge reduces the average sequence length. After `vocab_size` merges,
the vocabulary is fixed and all future text is encoded by greedily applying
learned merges.

### Byte-level BPE

We use **byte-level** BPE: the alphabet starts as all 256 possible byte values.
This means the tokenizer can represent any Unicode text without `<unk>`, because
any character can be decomposed into bytes, and all bytes are in the vocabulary.

### Our implementation

```python
tokenizer = BPETokenizer()
tokenizer.train_from_iterator(texts, vocab_size=8_000)
tokenizer.save("checkpoints/tokenizer.json")

# Encode
ids = tokenizer.encode("Once upon a time")
# → [bos_id, 512, 891, 4, 203, eos_id]

# Decode
text = tokenizer.decode(ids)
# → "Once upon a time"
```

The tokenizer is backed by HuggingFace `tokenizers` — a Rust implementation
that is ~100× faster than a pure Python BPE at both training and inference.

---

## 3. Special Tokens

Four special tokens are always part of the vocabulary:

| Token | ID | Purpose |
|---|---|---|
| `<pad>` | 0 | Padding — positions ignored by loss (`ignore_index=-1` in our case) |
| `<bos>` | 1 | Beginning-of-sequence — prepended to every encoded document |
| `<eos>` | 2 | End-of-sequence — appended after every document, used as document boundary |
| `<unk>` | 3 | Unknown token — fallback for byte-level BPE (rarely used) |

The `TemplateProcessing` post-processor automatically wraps every `encode()`
call with `<bos> ... <eos>`. Set `add_special_tokens=False` to suppress this.

---

## 4. Data Pipeline — PackedDataset

**File:** `src/data/dataset.py`

### The challenge

Documents in TinyStories have variable lengths — some are 50 tokens, some
are 500. A naive DataLoader would pad short sequences to the max length in
the batch, wasting GPU time on `<pad>` tokens the model ignores.

### Sequence packing

`PackedDataset` solves this with a streaming buffer:

```
Document stream (tokenised):

  [d1_tok1, d1_tok2, ..., d1_tokN, <eos>,
   d2_tok1, d2_tok2, ...,          <eos>,
   d3_tok1, d3_tok2, ...,          <eos>,  ...]

Buffer:    ↑ continuously fills from stream

Chunk 1:   buffer[ 0 : max_seq_len+1 ]    → input = chunk[:-1], label = chunk[1:]
Chunk 2:   buffer[ max_seq_len+1 : 2*(max_seq_len+1) ]
...
```

Documents flow into the buffer continuously. As soon as there are
`max_seq_len + 1` tokens in the buffer, we slice off a chunk and yield it.
The `+1` is because we need one extra token so we can produce a shifted
target: input is `chunk[:-1]`, target is `chunk[1:]`.

### Why `max_seq_len + 1` in the buffer?

```
chunk = [t0, t1, t2, ..., t_max]     ← length: max_seq_len + 1

input_ids = [t0, t1, ..., t_{max-1}] ← length: max_seq_len
labels    = [t1, t2, ..., t_max]     ← length: max_seq_len, shifted by 1
```

The model sees `input_ids` and predicts `labels`. This is the standard
**next-token prediction** (causal language modelling) objective.

### EOS as document separator

We explicitly append `<eos>` after each document's tokens:

```python
ids = tokenizer.encode(text, add_special_tokens=False)
ids.append(tokenizer.eos_id)   # ← document boundary marker
buffer.extend(ids)
```

This means the model sees `<eos>` tokens at document boundaries and learns
to associate them with "the story ended — what comes next is a new story".
Without this, the model would see arbitrary cut-points mid-document.

### Streaming

We iterate the HuggingFace dataset with `streaming=True`, so the data is
downloaded in chunks as needed — no need to fit the entire corpus in RAM.

---

## 5. Sequence Packing vs. Padding

This choice matters a lot for GPU efficiency.

**Padding approach:**

```
Batch of 4 sequences (max_len = 10):

  [t1, t2, t3, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>]
  [t1, t2, t3, t4,    t5,    t6,    <pad>, <pad>, <pad>, <pad>]
  [t1, t2, t3, t4,    t5,    t6,    t7,    t8,    <pad>, <pad>]
  [t1, t2, t3, t4,    t5,    t6,    t7,    t8,    t9,    t10  ]

Token utilisation: (3+6+8+10) / 40 = 67.5 %
```

**Packing approach (what we use):**

```
Continuous token stream → slice into fixed chunks

  [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]
  [t11, t12, <eos>, t1, t2, t3, t4, t5, t6, t7]  ← doc boundary mid-chunk is fine
  ...

Token utilisation: ~100 % (only the final partial chunk is padded)
```

For a `max_seq_len=2048` model training on TinyStories (average ~200 tokens/doc),
packing saves roughly 90 % of the compute that padding would waste.

---

## 6. Text Preprocessing

**File:** `src/data/preprocessing.py`

Before tokenisation, each document passes through `clean()`:

```python
def clean(text, do_unicode_norm=True, do_remove_control=True, min_length=10):
    text = normalize_unicode(text)       # NFC: é vs e+combining accent
    text = remove_control_characters(text)  # strip \x00, \x01, etc.
    text = text.strip()
    return text if len(text) >= min_length else None
```

**NFC normalisation** collapses different Unicode representations of the same
character into one canonical form. Without it, the tokenizer sees two different
byte sequences for the same visual character → larger vocabulary, worse
compression.

**Control character removal** strips invisible non-printable characters that
occasionally appear in scraped web text (null bytes, escape sequences, etc.).
These would waste vocabulary slots on rare garbage tokens.

---

## 7. The Training Objective

The model is trained with **causal language modelling (CLM)**:

```
Given: [Once, upon, a, time, there]
Predict: [upon, a, time, there, was]
```

At every position `t`, the model predicts the token at position `t+1`.
This is supervised learning — the correct answer is always the next real
token in the sequence.

The loss is **cross-entropy** over the vocabulary:

```
loss = -( 1/T ) * Σ log P(correct_token_t | tokens_{0..t-1})
```

A well-initialised untrained model with `vocab_size=8000` produces
`loss ≈ log(8000) ≈ 9.0`. After training to convergence on TinyStories,
a small model reaches roughly `loss ≈ 1.5–2.0` (perplexity ≈ 4–7).

---

## 8. Optimizer — AdamW with Weight Decay Grouping

**File:** `src/training/optimizer.py`

### AdamW

We use **AdamW** — Adam with decoupled weight decay. The update rule:

```
m_t = β1 * m_{t-1} + (1 - β1) * g_t        ← first moment (momentum)
v_t = β2 * v_{t-1} + (1 - β2) * g_t²       ← second moment (RMS)

θ_t = θ_{t-1} - lr * ( m_t / (√v_t + ε) )  ← Adam step
              - lr * λ * θ_{t-1}             ← weight decay (decoupled)
```

The key insight of AdamW over Adam-with-L2: weight decay should shrink the
weights directly, not through the gradient. In Adam, adding L2 to the loss
interacts with the adaptive scaling `1/√v_t`, making the effective decay
depend on the gradient history. AdamW separates these concerns.

**Our hyperparameters** (Chinchilla/LLaMA-style):

| Parameter | Value | What it controls |
|---|---|---|
| `lr` | 3e-4 | Peak learning rate |
| `weight_decay` | 0.1 | L2 regularisation strength |
| `beta1` | 0.9 | Momentum decay (how much past gradients matter) |
| `beta2` | 0.95 | RMS decay (how quickly second moment adapts) |
| `eps` | 1e-8 | Numerical stability in denominator |

Note: GPT-2 used `beta2=0.999`. Modern LLMs (PaLM, LLaMA) prefer `beta2=0.95`
because it adapts faster to gradient changes, which is better when the loss
landscape shifts during training.

### Weight decay grouping

We **do not** apply weight decay to every parameter. The rule:

```
Apply weight decay:     weight matrices (Linear.weight)
Do NOT apply:           bias terms, RMSNorm weights, embedding weights
```

Why exclude these?

- **Biases**: very few parameters, decaying them doesn't help and can slightly hurt
- **Norm weights**: these are scale factors; pulling them toward 0 would kill normalisation
- **Embeddings**: these are discrete lookups; decaying rare tokens would unfairly shrink them

```python
def _split_param_groups(model, weight_decay):
    no_decay_keywords = {"bias", "norm", "embedding"}
    decay_params, no_decay_params = [], []
    seen = set()  # handles weight-tied parameters (tok_emb == output)

    for name, param in model.named_parameters():
        if id(param) in seen: continue
        seen.add(id(param))
        if any(kw in name.lower() for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
```

The `seen` set handles the **weight-tied** embedding/output matrix — without it,
the same tensor would appear twice and receive double the gradient update.

---

## 9. LR Schedule — Cosine with Warmup

**File:** `src/training/scheduler.py`

A fixed learning rate is suboptimal: too high at the start causes instability,
too low throughout wastes capacity. We use three phases:

```
LR
 ▲
 │         ╭──────╮
 │        ╱        ╲
 │       ╱          ╲
 │      ╱            ╲____________
 │     ╱
 │    ╱
 │   ╱
 │──╱
 └──────────────────────────────────── step
    ↑         ↑              ↑
  start    warmup         max_steps
              ↑
           peak LR = base_lr
```

**Phase 1 — Linear warmup** `[0, warmup_steps)`:

```
lr = step / warmup_steps * base_lr
```

Gradients are large and noisy at initialisation. Starting with a tiny LR
and ramping up gives the optimiser time to collect momentum statistics
before taking large steps. Skipping warmup often causes the loss to spike
in the first few hundred steps.

**Phase 2 — Cosine decay** `[warmup_steps, max_steps]`:

```
progress = (step - warmup_steps) / (max_steps - warmup_steps)
cosine   = 0.5 * (1 + cos(π * progress))           ← goes from 1.0 → 0.0
lr       = min_lr + (base_lr - min_lr) * cosine
```

Cosine decay is smoother than linear decay — the LR drops slowly at first
(when the model is still making large progress) and faster near the end
(when it has mostly converged).

**Phase 3 — Floor** `[max_steps, ∞)`:

```
lr = min_lr_ratio * base_lr    (typically 10% of peak)
```

We never go to zero — a small residual LR allows slight continued adaptation.

---

## 10. Gradient Accumulation

**File:** `src/training/trainer.py`

**The problem:** the effective batch size for stable LLM training is large
(32–512 sequences). A single consumer GPU might only fit 4–8 sequences at a time.

**The solution:** accumulate gradients across multiple forward passes before
doing one optimizer step.

```
gradient_accumulation_steps = 4

Step 1:  forward(micro_batch_1) → loss/4  → backward  [no optimizer step]
Step 2:  forward(micro_batch_2) → loss/4  → backward  [no optimizer step]
Step 3:  forward(micro_batch_3) → loss/4  → backward  [no optimizer step]
Step 4:  forward(micro_batch_4) → loss/4  → backward  ← optimizer.step()
                                                         scheduler.step()
                                                         zero_grad()
```

Dividing the loss by `gradient_accumulation_steps` before each backward ensures
the gradients sum (not average) to the correct scale — equivalent to training
on a batch 4× larger.

**Effective batch size:**
```
effective_batch = batch_size × grad_accum × n_gpus
                = 8 × 4 × 1
                = 32 sequences × 2048 tokens
                = 65,536 tokens per optimizer step
```

### `no_sync` optimisation

In distributed training (multiple GPUs), PyTorch normally synchronises
gradients after every backward pass. With gradient accumulation, we only
need to sync on the last micro-step:

```python
for micro in range(gradient_accumulation_steps):
    is_last = (micro == gradient_accumulation_steps - 1)
    ctx     = _null_ctx() if is_last else accelerator.no_sync(model)
    with ctx:
        loss = model(batch) / gradient_accumulation_steps
        accelerator.backward(loss)
```

`no_sync` suppresses the all-reduce communication on intermediate steps,
giving significant speedups on multi-GPU runs.

---

## 11. Mixed Precision — bf16

**File:** `src/training/trainer.py` (via `Accelerator`)

Modern GPUs have specialised hardware for 16-bit floating-point arithmetic
that is 2–4× faster than 32-bit. We use **bfloat16 (bf16)**:

| Format | Bits | Range | Precision |
|---|---|---|---|
| float32 | 32 | ±3.4×10³⁸ | 7 decimal digits |
| float16 | 16 | ±65,504 | 3 decimal digits |
| bfloat16 | 16 | ±3.4×10³⁸ | 2 decimal digits |

**Why bf16 over fp16?**

fp16 has a very limited range (max value 65,504). Large activation values or
gradient norms can overflow to `inf`, causing NaN propagation and training
collapse. bf16 has the same exponent range as float32, making it much more
stable.

The model weights and activations are stored in bf16; the master copy of
weights used for the optimizer update is kept in float32 for precision.
This is handled automatically by `Accelerate`.

bf16 is supported on Ampere GPUs (RTX 30xx) and newer. On older GPUs, use
`mixed_precision="fp16"` or `"no"`.

---

## 12. Gradient Clipping

Before each optimizer step:

```python
grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

This computes the global gradient norm:

```
global_norm = sqrt( Σ ||grad_p||² )   for all parameters p
```

If `global_norm > max_norm`, all gradients are scaled down by
`max_norm / global_norm` so the global norm equals exactly `max_norm`.

**Why clip?**

Early in training (or when the loss landscape is rough), a single bad batch
can produce very large gradients that would cause the parameters to jump far
from a good region. Clipping to 1.0 bounds the worst-case step size.

Note: `clip_grad_norm_` returns the **pre-clip** norm. This is logged as
`gnorm` in the training output — high values (>> 1.0) indicate instability.

---

## 13. Checkpointing

**File:** `src/training/checkpointing.py`

### What is saved

Every `save_every` steps, we write a checkpoint containing:

```python
{
    "step":      int,          # current effective training step
    "loss":      float,        # best validation loss seen so far
    "model":     state_dict,   # all model weights
    "optimizer": state_dict,   # AdamW momentum / RMS buffers
    "scheduler": state_dict,   # LR scheduler position
    "config":    ModelConfig,  # hyperparameters
}
```

Saving the optimizer state is critical for resuming correctly. Without it,
Adam's first and second moment estimates (`m_t`, `v_t`) would be reset to
zero, effectively restarting the optimiser — this causes a loss spike after
resumption.

### Atomic writes

A partially-written checkpoint is worse than no checkpoint (it can corrupt
the training state on resume). We use an atomic write pattern:

```python
# Write to a temporary file first
torch.save(payload, path + ".tmp")
# Atomically rename — on POSIX this is guaranteed to be atomic
Path(path + ".tmp").replace(path)
```

If the process is killed mid-write, only the `.tmp` file is corrupted.
The last good checkpoint on disk is untouched.

### Checkpoint naming

```
checkpoints/
├── step_0001000.pt    ← periodic checkpoint at step 1000
├── step_0002000.pt    ← periodic checkpoint at step 2000
└── best.pt            ← lowest validation loss seen so far
```

On resume, `get_latest_checkpoint()` sorts by name and picks the last
`step_XXXXXXX.pt`.

---

## 14. The Trainer

**File:** `src/training/trainer.py`

The `Trainer` class ties everything together. Here is the full inner loop
in pseudocode:

```
while step < max_steps:

    model.train()

    batch_loss = 0

    for micro in range(gradient_accumulation_steps):
        batch = next(data_iter)

        with no_sync (if not last micro-step):
            logits, loss = model(batch["input_ids"], batch["labels"])
            loss /= gradient_accumulation_steps
            backward(loss)
            batch_loss += loss.item()

    grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    step += 1

    if step % log_every == 0:
        print(step, loss, lr, grad_norm, tok/s)

    if step % eval_every == 0:
        val_loss = evaluate()
        if val_loss < best:  save("best.pt")

    if step % save_every == 0:
        save(f"step_{step}.pt")
```

### `set_to_none=True`

`optimizer.zero_grad(set_to_none=True)` sets gradients to `None` rather than
zero. This saves one memory write per parameter per step — the gradient
tensor is released entirely until the next backward pass allocates it fresh.

### `_cycle(loader)`

`PackedDataset` is a streaming `IterableDataset`. When the dataset runs out
(after one pass through TinyStories), `_cycle` restarts it:

```python
@staticmethod
def _cycle(loader):
    while True:
        yield from loader
```

This ensures training can continue for any number of steps without hitting
a `StopIteration`.

### Accelerate

`Accelerate` is a thin wrapper that handles:

- **Device placement**: moves model and batches to GPU automatically
- **Mixed precision**: wraps forward/backward in autocast
- **Distributed training**: handles DDP gradient sync when using multiple GPUs
- **W&B logging**: optional integration via `log_with="wandb"`

No code changes are needed to go from 1 GPU to 4 GPUs — just run:

```bash
accelerate launch --num_processes 4 scripts/pretrain.py
```

---

## 15. Running Training

### Step 1 — Train the tokenizer (once)

```bash
python scripts/train_tokenizer.py
```

This downloads ~50,000 TinyStories documents (streaming), trains a BPE
tokenizer with vocab_size=8,000, and saves it to `checkpoints/tokenizer.json`.

Takes about 2–5 minutes. Only needs to be done once.

Expected output:
```
Loading TinyStories (streaming)...
Collecting 50,000 examples for tokenizer training...
  Collected 50,000 documents.
Training BPE (vocab_size=8,000)...
Tokenizer saved → checkpoints/tokenizer.json
  vocab_size : 8,000
  bos_id     : 1
  eos_id     : 2

Sample encode: 'Once upon a time, there was a little cat.'
  → [1, 512, 891, 4, 203, 6, 122, 4, 88, 1341, 22, 2]
  → 'Once upon a time, there was a little cat.'
```

### Step 2 — Pre-train

```bash
python scripts/pretrain.py
```

Streams TinyStories, packs into 2048-token chunks, and trains the model.

Expected output (every 10 steps):
```
Training  |  params: 41,156,608  |  device: cuda  |  precision: bf16
step      10 | loss 9.0241 | lr 1.50e-04 | gnorm 1.000 | 12.3k tok/s
step      20 | loss 8.7392 | lr 3.00e-04 | gnorm 0.987 | 14.1k tok/s
step      50 | loss 7.1234 | lr 2.98e-04 | gnorm 0.763 | 14.8k tok/s
step     100 | loss 5.8821 | lr 2.95e-04 | gnorm 0.612 | 15.2k tok/s
...
```

### Optional flags

```bash
# Enable W&B logging
python scripts/pretrain.py --use_wandb

# Train longer with a larger batch
python scripts/pretrain.py --max_steps 50000 --batch_size 16 --grad_accum 8

# CPU-only (slow, but works)
python scripts/pretrain.py --mixed_precision no

# Start fresh (ignore existing checkpoints)
python scripts/pretrain.py --no_resume
```

---

## 16. What to Watch During Training

### Loss curve

| Phase | Expected loss | What it means |
|---|---|---|
| Step 0 | ~9.0 (`log(8000)`) | Random model — uniform over vocab |
| Steps 1–200 | 9.0 → 6.0 | Warmup: model learns common tokens quickly |
| Steps 200–2000 | 6.0 → 3.0 | Rapid improvement: syntax, common phrases |
| Steps 2000–10000 | 3.0 → 1.8 | Slower gains: coherence, story structure |
| Converged | ~1.5–2.0 | Good TinyStories model |

### Gradient norm (`gnorm`)

- **Near 1.0**: normal — gradient clipping is engaged occasionally
- **Consistently << 1.0**: gradients are very small, LR might be too low
- **Consistently 1.0 exactly**: all steps are being clipped, LR may be too high or model is unstable

### Token throughput (`tok/s`)

On a modern GPU (RTX 3090 / 4090) with bf16:
- `batch_size=8, grad_accum=4`: ~15–25k tok/s
- `batch_size=16, grad_accum=8`: ~30–50k tok/s

At 20k tok/s and 10,000 steps with 65,536 tokens/step:
```
Total tokens = 10,000 × 65,536 = 655 M tokens
Training time ≈ 655M / 20,000 / 3600 ≈ 9 hours
```

---

## 17. Design Decisions at a Glance

| Choice | What we did | Alternative | Reason |
|---|---|---|---|
| Tokenizer | BPE (byte-level, 8k vocab) | WordPiece, Unigram, char-level | BPE is fast, language-agnostic, no `<unk>` with byte-level |
| Corpus | TinyStories (streaming) | OpenWebText, C4, The Pile | Small, clean, fast to stream; good for iteration |
| Data strategy | Sequence packing | Padding | ~100 % token utilisation vs. ~70 % |
| Optimiser | AdamW (β2=0.95) | Adam (β2=0.999), SGD | Modern LLMs use 0.95; faster adaptation |
| Weight decay | Applied to weights only | All params | Biases and norms are small and don't benefit |
| LR schedule | Cosine + linear warmup | Constant, linear decay | Smooth decay, widely used in LLM training |
| Precision | bf16 | fp32, fp16 | Same range as fp32, fewer NaN issues than fp16 |
| Grad accumulation | Manual loop | `accelerator.accumulate` | More transparent, easier to debug |
| Checkpointing | Atomic (write + rename) | Direct write | Crash-safe — disk state is never partially written |
| Resume | Auto-detect latest step | Manual path | Zero-friction restarts after interruption |

---

## 18. What's Next — Phase 3

With a pre-trained model, Phase 3 teaches it to **follow instructions**.

```
Phase 3 — Fine-Tuning & Scaling

  Supervised Fine-Tuning (SFT)
      Train on (instruction, response) pairs
      Same cross-entropy loss, different data distribution

  LoRA — Low-Rank Adaptation
      Freeze the pre-trained weights
      Inject small trainable rank-r matrices into Q and V projections
      Fine-tune only ~0.1 % of the parameters — 10× less memory

  QLoRA — Quantised LoRA
      Quantise the frozen base model to 4-bit
      Add LoRA adapters in bf16
      Fits a 7B model on a single 24 GB GPU
```

The key insight of Phase 3: pre-training teaches the model **what language
looks like**. Fine-tuning teaches it **how to be helpful**. These are
separate skills that require separate data and separate training recipes.
