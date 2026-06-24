# Phase 3 — Fine-Tuning & Scaling

A bottom-up walkthrough of everything built in Phase 3: why fine-tuning
exists, how LoRA works at the mathematical level, how SFT prompt masking
forces the model to learn the right thing, and how QLoRA makes all of this
fit on a single consumer GPU.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Pre-training vs. Fine-tuning](#2-pre-training-vs-fine-tuning)
3. [Supervised Fine-Tuning (SFT)](#3-supervised-fine-tuning-sft)
4. [Prompt Masking — Why We Only Learn on the Response](#4-prompt-masking)
5. [The Alpaca Prompt Format](#5-the-alpaca-prompt-format)
6. [LoRA — Low-Rank Adaptation](#6-lora--low-rank-adaptation)
7. [The Math Behind LoRA](#7-the-math-behind-lora)
8. [Why Initialize B = 0?](#8-why-initialize-b--0)
9. [The Scaling Factor α/r](#9-the-scaling-factor-αr)
10. [Which Layers to Inject?](#10-which-layers-to-inject)
11. [Merging Adapters for Inference](#11-merging-adapters-for-inference)
12. [Saving Only the Adapter Weights](#12-saving-only-the-adapter-weights)
13. [QLoRA — 4-bit Quantisation + LoRA](#13-qlora--4-bit-quantisation--lora)
14. [NF4 — Normal Float 4](#14-nf4--normal-float-4)
15. [The QLoRA Preparation Pipeline](#15-the-qlora-preparation-pipeline)
16. [Fine-tuning Hyperparameters vs. Pre-training](#16-fine-tuning-hyperparameters-vs-pre-training)
17. [Running Fine-tuning](#17-running-fine-tuning)
18. [Test Suite](#18-test-suite)
19. [Design Decisions at a Glance](#19-design-decisions-at-a-glance)
20. [What's Next — Phase 4](#20-whats-next--phase-4)

---

## 1. The Big Picture

Phase 3 answers the question: **how do you turn a text predictor into a
model that follows instructions?**

Pre-training on TinyStories teaches the model to produce fluent text.
But "fluent text" and "helpful responses" are different skills.
A pre-trained model given `"Write a story about a cat"` might just
continue with `"and a dog and a mouse..."` — treating the instruction
as text to continue, not as a command to obey.

Fine-tuning recalibrates the model on examples of the correct input-output
behaviour:

```
Before fine-tuning:
  Input:  "Write a story about a cat."
  Output: "and a dog played in the yard one afternoon..."

After fine-tuning:
  Input:  "Write a story about a cat."
  Output: "Once there was a little cat named Milo who loved exploring..."
```

The full Phase 3 stack:

```
Pre-trained GPT checkpoint
        │
        ▼
inject_lora()              ← wrap Q and V projections with low-rank adapters
        │
        ▼
freeze_non_lora()          ← freeze 99+ % of parameters
        │
        ▼
SFTDataset                 ← (instruction, response) pairs, prompt masked
        │
        ▼
Trainer.train()            ← same loop as Phase 2, only LoRA params update
        │
        ▼
save_lora()                ← save only the adapter weights (~few MB)
        │
        ▼
merge_lora_weights()       ← (optional) absorb adapters into base for inference
```

---

## 2. Pre-training vs. Fine-tuning

| Aspect | Pre-training (Phase 2) | Fine-tuning (Phase 3) |
|---|---|---|
| Data | Raw text (TinyStories) | Instruction-response pairs |
| Objective | Predict every next token | Predict only response tokens |
| Trainable params | All ~41 M | Only LoRA: ~200 K (0.5 %) |
| Learning rate | 3e-4 | 2e-4 (lower) |
| Steps | 10,000 | 2,000 |
| Goal | Learn language | Learn to follow instructions |

Fine-tuning starts from the pre-trained weights — all the language
knowledge accumulated during pre-training is preserved. We only nudge
the model's behaviour, not re-learn from scratch.

---

## 3. Supervised Fine-Tuning (SFT)

**File:** `src/finetuning/sft.py`

SFT is conceptually identical to pre-training: compute cross-entropy
loss, backprop, update weights. The only differences are:

1. **The data** — (instruction, response) pairs instead of raw text
2. **The loss mask** — only response tokens contribute to the loss
3. **The trainable parameters** — only LoRA adapters update (not the full model)

The training loop, optimizer, and scheduler are completely reused from
Phase 2. SFT is not a new algorithm — it's the same next-token prediction
objective applied to a different data distribution.

---

## 4. Prompt Masking

**Why we don't compute loss on the prompt**

Consider this training example:

```
Prompt:   "### Instruction:\nWrite a children's story.\n\n### Response:\n"
Response: "Once upon a time there was a brave rabbit."
```

If we naively compute loss on the entire sequence:

```
Full sequence: [<bos>, ###, Inst, ruction, :, \n, Write, ..., \n, Once, upon, ...]
Loss on:        every single token in the sequence
```

The model would be rewarded for predicting `"Instruction"` given `"###"`,
for predicting `"\n"` given `"Response:"`, and so on. This is wasted
signal — we don't want the model to memorise the instruction format,
we want it to learn the correct **response**.

**The fix: mask the prompt**

```
Input IDs: [<bos>, ###, Inst, ruction, :, ..., \n, Once, upon, ...]
Labels:    [  -1,  -1,  -1,     -1,   -1, ..., -1, upon,  a, ...]
                  ← masked ─────────────────────→  ← loss here →
```

Labels set to `-1` are ignored by `F.cross_entropy(..., ignore_index=-1)`.

In code:

```python
prompt_len = 1 + len(prompt_ids)   # <bos> + instruction tokens
labels[:prompt_len] = -1           # mask prompt region
labels[input_ids == pad_id] = -1   # mask padding
```

**What position does the loss start from?**

```
Position:   0     1     2    ...   prompt_len-1   prompt_len   prompt_len+1
Input:     <bos>  p0    p1   ...      p_last         r0            r1
Label:      -1    -1    -1   ...        -1            r1            r2
                  ← masked →             ← unmasked from here →
```

The first label that contributes to loss is at position `prompt_len`,
which predicts `r1` (the second response token) given `r0`.
The transition from `p_last → r0` (first response token) is masked —
during inference this is handled naturally because we feed the full
prompt and let the model generate from there.

---

## 5. The Alpaca Prompt Format

**File:** `src/finetuning/sft.py`

We use the Stanford Alpaca prompt format, which was designed to be
clear and unambiguous for the model:

**With additional input context:**
```
### Instruction:
Translate the following sentence to French.

### Input:
The cat sat on the mat.

### Response:
Le chat était assis sur le tapis.
```

**Without input context (most common):**
```
### Instruction:
Write a short children's story about a brave rabbit.

### Response:
Once upon a time, there lived a brave little rabbit named Pip...
```

The section headers (`### Instruction:`, `### Response:`) act as
reliable delimiters the model learns to recognise. After fine-tuning,
generating `"### Response:\n"` at the end of the prompt reliably
triggers response-mode behaviour.

---

## 6. LoRA — Low-Rank Adaptation

**File:** `src/finetuning/lora.py`  
**Paper:** [Hu et al., 2021](https://arxiv.org/abs/2106.09685)

### The full fine-tuning problem

Standard fine-tuning updates every parameter in the model. For a
50M parameter model that's 50M gradient-tracking tensors, 50M
momentum buffers, 50M RMS buffers — easily 3–4× the model size in
optimizer state alone.

For a 7B model that means:
- Model weights: ~14 GB (fp16)
- Gradient copy: ~14 GB
- Optimizer state: ~28 GB
- **Total: ~56 GB** — requires at least 3 × A100 80GB cards

That's impractical for most researchers.

### The key observation

Aghajanyan et al. (2020) showed that the **intrinsic dimensionality**
of fine-tuning is very low. That is: the actual change in weights
needed to solve a new task lives in a much lower-dimensional subspace
than the full parameter space.

LoRA exploits this: instead of updating the full weight matrix W,
we only learn a low-rank decomposition of the change ΔW.

### The LoRA decomposition

For a weight matrix W ∈ ℝ^(d_out × d_in):

```
Full fine-tuning:  W' = W + ΔW
LoRA:              W' = W + B·A

where:
  A ∈ ℝ^(r × d_in)    r << min(d_out, d_in)
  B ∈ ℝ^(d_out × r)
  ΔW ≈ B·A             (rank-r approximation of the true ΔW)
```

W is frozen. Only A and B are trained.

**Parameter savings example** for one Q projection (d=512, r=8):

```
Full fine-tuning:  512 × 512 = 262,144 params
LoRA (r=8):        (8 × 512) + (512 × 8) = 8,192 params
Savings:           32× fewer parameters
```

---

## 7. The Math Behind LoRA

### Forward pass

The output of a LoRALinear layer:

```
h = W·x  +  (B·A)·x  ·  (α/r)
    └──┘     └─────┘
    base     adapter
  (frozen)  (trainable)
```

Step by step:

```
x shape:       (batch, seq, d_in)
A·x:           F.linear(x, A)   → (batch, seq, r)        [down-project]
B·(A·x):       F.linear(..., B) → (batch, seq, d_out)    [up-project]
Scaled:        × (α/r)
Output:        W·x + B·A·x·(α/r)
```

At initialisation, B = 0, so the adapter contributes exactly zero to
the output. The model starts from the pre-trained weights and gradually
adapts.

### Gradient flow

The gradients for A and B during backprop:

```
∂L/∂B = ∂L/∂h · (A·x)ᵀ · (α/r)      [gradient for B]
∂L/∂A = Bᵀ · ∂L/∂h · xᵀ · (α/r)     [gradient for A]
```

W receives no gradient (frozen). The optimizer only maintains momentum
and RMS estimates for A and B.

### Memory savings

For our small model with LoRA on wq and wv, r=8:

```
Standard fine-tuning memory (optimizer state for all weights):
  ~41M params × 3 (fp32 weights + Adam m + Adam v) × 4 bytes = ~492 MB

LoRA fine-tuning memory (optimizer state only for adapters):
  ~200K params × 3 × 4 bytes = ~2.4 MB

Reduction: ~200× in optimizer state
```

---

## 8. Why Initialize B = 0?

This is one of the most important implementation details in LoRA.

At the start of fine-tuning, the model should behave identically to
the pre-trained checkpoint. If the adapter started with random weights,
it would immediately add noise to every forward pass — disrupting the
language model's carefully learned representations before any gradient
signal has been processed.

**With B = 0:**
```
output = W·x + B·A·x·(α/r)
       = W·x + 0·A·x·(α/r)
       = W·x                   ← exactly the pre-trained output
```

The adapter starts as a no-op and learns from zero. Only A is
initialised randomly (Kaiming uniform, same as `nn.Linear`), because
A's gradients flow through B, which starts at zero:

```
∂L/∂A = Bᵀ · ∂L/∂h · xᵀ · (α/r)
```

At step 0: B = 0 → ∂L/∂A = 0 → A doesn't move initially.
At step 1: B has been updated slightly → ∂L/∂A is small but non-zero.
The adapter wakes up gradually, starting from the pre-trained state.

**Test:** `test_zero_delta_at_init` verifies:
```python
base_out = linear(x)           # frozen base
lora_out = lora_layer(x)       # base + zero adapter
assert torch.allclose(base_out, lora_out)   # must be identical
```

---

## 9. The Scaling Factor α/r

The LoRA output is scaled by `α/r`:

```
ΔW_effective = B·A · (α/r)
```

**Why scale at all?**

Without scaling, changing the rank `r` would change the effective
magnitude of the adapter. If you double r from 8 to 16 (doubling the
number of parameters), the adapter's contribution would also roughly
double — making it hard to compare runs with different ranks.

**The α trick:**

By defining the scale as `α/r`, you can set `α` once (e.g. α=16)
and then freely sweep `r` (4, 8, 16, 32) without changing the
effective learning rate. The adapter magnitude stays constant as
long as you keep α fixed.

Common choices:

| Setting | Meaning |
|---|---|
| α = r | Scale factor = 1.0 (no scaling) |
| α = 2r | Scale factor = 2.0 (amplify adapter) |
| α = r/2 | Scale factor = 0.5 (dampen adapter) |

Our default: `r=8, α=16 → scale=2.0`. This is a mild amplification
that helps the adapter learn faster in the initial steps.

---

## 10. Which Layers to Inject?

The original LoRA paper tried different combinations:

| Injected layers | Downstream accuracy | Params |
|---|---|---|
| W_q only | 79.8 % | baseline |
| W_q + W_v | 80.7 % | 2× |
| W_q + W_k + W_v + W_o | 80.9 % | 4× |
| All linears | 81.2 % | 8× |

The result: **Q and V** give almost all the benefit at half the cost
of doing all four attention projections. This is our default:
`target_modules=["wq", "wv"]`.

**Why Q and V but not K?**

- Q (query) shapes what each token is "looking for"
- V (value) shapes what information is retrieved
- K (key) mainly determines which tokens are attended to — the
  patterns learned here transfer well across tasks without adaptation

The FFN matrices (w1, w2, w3) can also be targeted, which gives
more capacity at the cost of more parameters. For most fine-tuning
tasks, attention adapters alone are sufficient.

---

## 11. Merging Adapters for Inference

After fine-tuning, the LoRA adapter adds two extra matmuls to every
forward pass. For deployment you can **merge** the adapter into the
base weight:

```
W_merged = W + B·A·(α/r)
```

The merged weight is mathematically identical to the LoRA model:

```python
# Before merge:
output = W·x + B·A·x·(α/r)

# After merge (W_merged = W + B·A·(α/r)):
output = W_merged·x
       = (W + B·A·(α/r))·x
       = W·x + B·A·x·(α/r)    ← same result, zero extra cost
```

In our implementation:

```python
def merge(self) -> nn.Linear:
    with torch.no_grad():
        delta_W = (self.lora_B @ self.lora_A) * self.scaling
        self.linear.weight.data += delta_W
    return self.linear   # a plain nn.Linear, no LoRA overhead
```

`merge_lora_weights(model)` traverses the entire model and replaces
every `LoRALinear` with its merged `nn.Linear`.

**Test:** `test_merge_preserves_output` gives adapters non-zero values,
records logits from the LoRA model, then merges and records logits again.
The assertion `torch.allclose(logits_lora, logits_merged, atol=1e-5)`
confirms numerical identity.

---

## 12. Saving Only the Adapter Weights

Fine-tuning produces a complete model checkpoint — but the base weights
haven't changed at all. There's no need to save them again.

`save_lora()` extracts only the LoRA matrices:

```python
lora_state = {
    n: p.detach().cpu()
    for n, p in model.named_parameters()
    if "lora_" in n         # only lora_A and lora_B tensors
}
torch.save(lora_state, path)
```

**Size comparison** for our small model with r=8:

```
Full model checkpoint: ~160 MB
LoRA adapter file:     ~0.1 MB   (1,600× smaller)
```

For a 7B model with r=64:
```
Full model:  ~14 GB
LoRA file:   ~100 MB  (140× smaller)
```

To share a fine-tuned model, you only need to distribute:
1. The original pre-trained checkpoint (or a link to it)
2. The tiny adapter file

`load_lora()` restores the adapters into a model that already has
LoRA injected with matching structure.

---

## 13. QLoRA — 4-bit Quantisation + LoRA

**File:** `src/finetuning/qlora.py`  
**Paper:** [Dettmers et al., 2023](https://arxiv.org/abs/2305.14314)

### The problem LoRA doesn't fully solve

LoRA reduces **optimizer state** and **gradient memory** dramatically.
But the base model weights still need to be loaded into GPU memory in
their original precision to compute the forward pass.

For a 7B model in bf16: 7B × 2 bytes ≈ **14 GB**.
A consumer GPU (RTX 4090) has 24 GB — that leaves only 10 GB for
activations, gradients, and the adapter.

### QLoRA's answer: quantise the frozen weights

Since the base weights are **frozen** (they receive no gradient updates),
their exact values don't need to be maintained in high precision during
fine-tuning. We can compress them aggressively without affecting the
quality of the trained adapter.

QLoRA pipeline:

```
Base model (bf16, frozen)
    │
    ├─ Quantise to 4-bit NF4  →  compressed frozen backbone
    │
    └─ Inject LoRA in bf16    →  small trainable adapters
```

Memory breakdown for a 7B model:

```
Full fine-tuning:  14 GB (weights) + 28 GB (optimizer) + 14 GB (gradients) = 56 GB
LoRA only:         14 GB (weights) +  1 GB (optimizer) +  0.5 GB (gradients) = 15.5 GB
QLoRA:              3.5 GB (4-bit) +  1 GB (optimizer) +  0.5 GB (gradients) = 5 GB  ✓
```

QLoRA makes 7B models fine-tunable on a single 8 GB GPU.

---

## 14. NF4 — Normal Float 4

Standard 4-bit integers store values as uniform steps between 0 and 15.
But neural network weights are **not** uniformly distributed — they
follow an approximately normal (Gaussian) distribution centred at zero.

**NF4 (Normal Float 4)** is a custom 4-bit data type that places its
16 quantisation levels at the optimal positions for a standard normal
distribution, minimising the average quantisation error.

```
Uniform int4:  [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
               (evenly spaced — wastes resolution in tails)

NF4:           [-1.0, -0.69, -0.52, -0.40, -0.28, -0.18, -0.09, 0,
                 0.07,  0.14,  0.23,  0.33,  0.44,  0.56,  0.72,  1.0]
               (densely packed near zero where most weights live)
```

NF4 achieves lower mean squared quantisation error than any
other 4-bit format for normally distributed data. The weights are
dequantised back to bf16 on-the-fly during the forward pass.

### Double quantisation

QLoRA also applies a second round of quantisation to the
**quantisation constants themselves** (the per-block scale factors
used to convert between NF4 and bf16). This extra step saves roughly
0.5 bits per parameter.

---

## 15. The QLoRA Preparation Pipeline

**File:** `src/finetuning/qlora.py`

```python
def prepare_model_for_qlora(model, r=8, alpha=16, dropout=0.05, target_modules=None):

    # Step 1: Inject LoRA into Q and V projections first
    #         (while they are still full-precision nn.Linear)
    model = inject_lora(model, target_modules=["wq", "wv"], ...)

    # Step 2: Quantise all remaining nn.Linear layers to 4-bit NF4
    #         (skipping the ones inside LoRALinear wrappers)
    model = quantize_model_4bit(model)

    # Step 3: Freeze everything except lora_A and lora_B
    freeze_non_lora(model)

    return model

# After calling:
model = prepare_model_for_qlora(model)
model = model.to("cuda")   ← actual quantisation happens here
```

**Why inject LoRA before quantising?**

The LoRA adapters must stay in bf16 — they need full precision to
compute accurate gradients. If we quantised first, the LoRA layers
would be inside quantised 4-bit modules and their weight updates would
be imprecise.

By injecting LoRA first, the `wq` and `wv` modules become `LoRALinear`
objects (containing a frozen bf16 `nn.Linear` + bf16 `lora_A`, `lora_B`).
The quantisation step then skips any `nn.Linear` that is inside a
`LoRALinear` wrapper:

```python
lora_prefixes = {name for name, mod in model.named_modules()
                 if isinstance(mod, LoRALinear)}

def is_inside_lora(name):
    return any(name.startswith(prefix + ".") for prefix in lora_prefixes)

# Quantise only non-LoRA linears
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and not is_inside_lora(name):
        # replace with Linear4bit
```

**Result:** all non-target layers (wk, wo, w1, w2, w3, output) are
quantised to 4-bit, while wq and wv remain in bf16 with LoRA adapters.

---

## 16. Fine-tuning Hyperparameters vs. Pre-training

Fine-tuning uses a different set of hyperparameters:

| Hyperparameter | Pre-training | Fine-tuning | Reason |
|---|---|---|---|
| Learning rate | 3e-4 | 2e-4 | Lower to avoid disturbing pre-trained knowledge |
| Weight decay | 0.1 | 0.0 | LoRA params are tiny; regularisation hurts more than helps |
| β2 (Adam) | 0.95 | 0.999 | Less gradient noise during fine-tuning → can use larger β2 |
| Warmup steps | 200 | 50 | Much shorter run — warmup fraction stays similar |
| Max steps | 10,000 | 2,000 | Much less data; overfitting risk is higher |
| min_lr_ratio | 0.1 | 0.0 | Fine-tuning benefits from decaying to zero |
| Batch size | 8 | 4 | Fine-tuning data is often scarcer |
| Grad accum | 4 | 8 | Compensate for smaller batch size |

### Why is the learning rate lower?

During pre-training, the model is learning from scratch — large
learning rate steps are needed to make fast progress from random
initialisation.

During fine-tuning, the model already has well-tuned representations.
Too large a learning rate would cause **catastrophic forgetting** —
the model would overwrite its pre-trained language skills while learning
the new task.

LoRA makes this even more forgiving: since only the tiny adapter
matrices are updated (not the base weights), the risk of catastrophic
forgetting is essentially eliminated. The base model is permanently
frozen.

---

## 17. Running Fine-tuning

### With synthetic demo data (no files needed)

```bash
python scripts/finetune.py \
    --pretrain_ckpt checkpoints/best.pt \
    --demo
```

Uses the built-in `make_tinystories_sft_examples()` which generates 500
story-writing instruction pairs aligned with TinyStories pre-training.

### With your own instruction dataset

Your JSONL file should have one example per line:

```json
{"instruction": "Write a story about a brave rabbit.", "input": "", "output": "Once there was..."}
{"instruction": "Tell me about photosynthesis.", "input": "", "output": "Photosynthesis is..."}
```

```bash
python scripts/finetune.py \
    --pretrain_ckpt checkpoints/best.pt \
    --data_path data/my_instructions.jsonl \
    --max_steps 2000
```

### With QLoRA (CUDA + bitsandbytes required)

```bash
python scripts/finetune.py \
    --pretrain_ckpt checkpoints/best.pt \
    --demo \
    --qlora
```

### Expected output

```
Tokenizer: BPETokenizer(vocab_size=8000)
Loading pre-trained weights from checkpoints/best.pt
Injected LoRA into 4 layers  (r=8, alpha=16.0, dropout=0.05)
Trainable LoRA params: 196,608 / 41,156,608  (0.48 %)
Using synthetic TinyStories SFT demo data...
  SFTDataset: 450 examples ready.
Training  |  params: 41,156,608  |  device: cuda  |  precision: bf16
step      5 | loss 3.1204 | lr 1.00e-04 | gnorm 0.743 | 8.2k tok/s
step     10 | loss 2.7341 | lr 2.00e-04 | gnorm 0.612 | 9.1k tok/s
step     50 | loss 1.9823 | lr 1.98e-04 | gnorm 0.481 | 9.8k tok/s
...
LoRA adapters saved → checkpoints/finetune/lora_adapters.pt  (0.38 MB, 8 tensors)
```

---

## 18. Test Suite

**File:** `tests/test_finetuning.py` — 34 tests.

| Test class | What is verified |
|---|---|
| `TestLoRALinear` | Output shape, zero delta at init (B=0), base frozen, adapter shapes, scaling, merge identity, merge unfreezes |
| `TestInjectLoRA` | Correct layers replaced, correct count, non-targets untouched, bad target raises, in-place modification |
| `TestFreezeAndCount` | Only lora_A/B trainable, <5% trainable, get_lora_params returns params |
| `TestMergeLoRA` | All LoRALinear replaced after merge, logits numerically identical |
| `TestSaveLoadLoRA` | Weights preserved after save/load, adapter file smaller than full model |
| `TestFormatAlpaca` | Instruction in prompt, input included when given, input omitted when empty |
| `TestSFTDataset` | Shapes, prompt masked, response not masked, padding masked, too-long skipped, from_jsonl |
| `TestSFTEndToEnd` | Loss decreases over 20 steps, non-LoRA params unchanged after step |

The two most important tests:

**`test_zero_delta_at_init`** — proves the model starts from exactly
the pre-trained state. Without this guarantee, LoRA injection would
immediately degrade model quality before training even begins.

**`test_only_lora_params_updated`** — snapshots all non-LoRA weights
before a training step and asserts they are bit-for-bit unchanged
after. This is the key correctness property of the entire Phase 3
pipeline: we are truly fine-tuning with LoRA, not accidentally
computing gradients on frozen layers.

---

## 19. Design Decisions at a Glance

| Choice | What we did | Alternative | Reason |
|---|---|---|---|
| Fine-tuning method | LoRA | Full fine-tuning, prefix tuning, adapters | Orders of magnitude fewer params; no catastrophic forgetting |
| LoRA targets | wq, wv | All linears, wq+wk+wv+wo | Best accuracy/cost tradeoff from original paper |
| LoRA rank | r=8 | r=4 (less), r=64 (more) | 8 is a well-validated default for instruction tuning |
| B initialisation | zeros | Random | Ensures exact pre-trained baseline at step 0 |
| Loss masking | Prompt tokens masked | Loss on full sequence | Model learns response generation, not instruction memorisation |
| Prompt format | Alpaca (### headers) | ChatML, Llama-3 format | Simple, widely understood, easy to parse |
| Adapter persistence | Save only A, B tensors | Save full model | LoRA file is 1,000× smaller; base weights unchanged |
| 4-bit format | NF4 | Int4, FP4, Int8 | Optimal for normally distributed weights (proven in QLoRA paper) |
| LoRA-then-quantise | Inject LoRA first | Quantise then inject | Keeps adapter matrices in bf16; avoids quantising trainable params |
| Weight decay | 0.0 | 0.1 (pre-train) | Adapter matrices are tiny; regularisation is not needed |

---

## 20. What's Next — Phase 4

With SFT complete, the model follows instructions. Phase 4 teaches it
to follow instructions **well** — to produce responses that humans
actually prefer.

```
Phase 4 — Alignment with Feedback

  Reward Model
      Train a model to score responses on a 0–1 scale
      Input: (instruction, response) pair
      Output: scalar reward
      Data: preference pairs (chosen > rejected)

  PPO — Proximal Policy Optimisation
      Use the reward model as a training signal
      Fine-tune the SFT model to maximise expected reward
      KL penalty against the SFT model prevents reward hacking

  DPO — Direct Preference Optimisation
      Skip the reward model entirely
      Directly optimise the policy on preference pairs
      Simpler, more stable, increasingly preferred over PPO
```

The central insight of Phase 4:

- **Pre-training** teaches the model what language looks like
- **SFT** teaches the model to follow instructions
- **Alignment** teaches the model which responses are actually *good*

A model that scores high on all three is what we typically call an
"aligned language model" — the kind that powers ChatGPT, Claude, and
Gemini.
