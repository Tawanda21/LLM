# Cleanup Guide

Eight tasks to make the project fully end-to-end runnable.
They are ordered from most to least impactful. Each task includes
what to do, why it matters, and hints — but **not** the solution.
The goal is for you to implement each one yourself.

Run `python -m pytest tests/` after each task to confirm nothing broke.

---

## Task 1 — Fill `main.py`

**File:** `main.py`  
**Current state:** empty  
**Why it matters:** The README says `main.py` is the unified project entry
point, but right now it does nothing. Without it the project has no single
place to run everything from.

### What the end result should look like

Running `python main.py --help` should print something like:

```
usage: main.py [-h] {tokenize,pretrain,finetune,align} ...

LLM from Scratch

subcommands:
  tokenize    Train the BPE tokenizer
  pretrain    Pre-train the language model
  finetune    Fine-tune with SFT + LoRA
  align       Align with DPO or PPO
```

Each subcommand should delegate to the existing script logic.

### Hints

- Use `argparse` with `add_subparsers()`
- Each subparser (`tokenize`, `pretrain`, etc.) maps to an existing
  function in `scripts/`. You can either `import` and call those
  functions directly, or use `subprocess.run` to invoke the scripts.
- The simplest approach: import `main` from each script and call it.
- Add `if __name__ == "__main__": main()` at the bottom.

### How to verify

```bash
python main.py --help                  # shows subcommands
python main.py tokenize --help         # shows tokenizer args
python main.py pretrain --help         # shows pretrain args
```

---

## Task 2 — Remove unused imports from `scripts/finetune.py`

**File:** `scripts/finetune.py`  
**Current state:** imports `build_adamw` and `cosine_with_warmup` at the
top but never uses them (the `Trainer` class handles these internally).  
**Why it matters:** unused imports are noise that confuse readers and
linters. `ruff` will flag these.

### What to do

Open `scripts/finetune.py`, find the two unused imports, and remove them.
Then run the linter to confirm:

```bash
ruff check scripts/finetune.py
```

### How to verify

```bash
python -m pytest tests/ -q       # still passes
ruff check scripts/finetune.py   # no errors
```

---

## Task 3 — Auto-detect `mixed_precision` in `TrainConfig`

**File:** `src/training/trainer.py`  
**Current state:** `TrainConfig` defaults to `mixed_precision="bf16"`.
This is correct for a CUDA GPU but confusing for CPU-only runs (Accelerate
will warn and auto-downgrade, but it should be cleaner).  
**Why it matters:** if someone runs without a GPU, or on an older GPU that
doesn't support bf16, the default should adapt automatically.

### What to do

Add a small helper function (near the top of `trainer.py` or in a
`utils.py`) that detects the best available precision:

```python
def detect_precision() -> str:
    """Return 'bf16', 'fp16', or 'no' based on available hardware."""
    ...
```

Rules:
- If no CUDA → `"no"`
- If CUDA but GPU compute capability < 8.0 (pre-Ampere) → `"fp16"`
- If CUDA and compute capability ≥ 8.0 → `"bf16"`

Then change the `TrainConfig` default from `mixed_precision: str = "bf16"`
to `mixed_precision: str = field(default_factory=detect_precision)`.

### Hints

- `torch.cuda.is_available()` — checks for CUDA
- `torch.cuda.get_device_capability(0)` — returns `(major, minor)` tuple
- Ampere is compute capability 8.x (RTX 30xx and newer)
- `dataclasses.field(default_factory=...)` is how you use a function as
  a default value in a dataclass

### How to verify

```python
from src.training.trainer import TrainConfig, detect_precision
cfg = TrainConfig()
print(cfg.mixed_precision)   # should print the right value for your GPU
```

---

## Task 4 — Fill `tests/test_tokenizer.py`

**File:** `tests/test_tokenizer.py`  
**Current state:** `# placeholder`  
**Why it matters:** we already have tokenizer tests buried inside
`test_training_step.py`, but there should be a dedicated file for
tokenizer unit tests — that's what the scaffold promised.

### What to write

Write at least 10 tests covering:

1. **Training** — train a small BPE tokenizer on a few strings and confirm it
   doesn't crash and has a vocab size ≤ the requested size
2. **Special tokens** — confirm `pad_id`, `bos_id`, `eos_id`, `unk_id` are
   all valid non-negative integers and distinct from each other
3. **Encode with special tokens** — first id should be `bos_id`, last
   should be `eos_id`
4. **Encode without special tokens** — first id should NOT be `bos_id`
5. **Decode roundtrip** — `decode(encode(text))` should contain the original text
6. **Batch encode** — `encode_batch(texts)` returns a list of the same
   length as the input
7. **Save and load** — save to a temp file, load, confirm `vocab_size` matches
   and `encode("hello")` gives the same result before and after
8. **`len(tokenizer)`** — should return `vocab_size`
9. **Handles unicode** — encode a non-ASCII string without raising an exception
   (byte-level BPE can handle anything)
10. **Empty string** — encoding `""` with special tokens should return
    `[bos_id, eos_id]`

### Hints

- Use `pytest.fixture` to train the tokenizer once and reuse it across tests
- Use `tmp_path` (a built-in pytest fixture) for the save/load test
- Import `BPETokenizer` from `src.tokenizer`
- A small corpus of 5–10 repeated sentences is enough to train a tiny vocab

### How to verify

```bash
python -m pytest tests/test_tokenizer.py -v
```

---

## Task 5 — Fill `tests/test_attention.py`

**File:** `tests/test_attention.py`  
**Current state:** `# placeholder`  
**Why it matters:** attention is the most important and complex component
in the model. The existing `test_model.py` tests it at a high level, but
deeper unit tests belong in a dedicated file.

### What to write

Focus on the attention internals that `test_model.py` doesn't cover deeply:

1. **`precompute_freqs_cis` shape** — `(max_seq_len, head_dim // 2)` complex
2. **Frequencies decrease monotonically** — θ_0 > θ_1 > ... > θ_{head_dim/2-1}
   (lower frequency for higher-index dimensions)
3. **RoPE at position 0** — rotation at position 0 should be a no-op
   (multiplying by e^(j·0) = 1 leaves the vector unchanged)
4. **Relative position encoding** — the dot product of a rotated query and
   key should depend only on their relative position, not absolute positions.
   Verify: `dot(rotate(q, pos=3), rotate(k, pos=1)) == dot(rotate(q, pos=2), rotate(k, pos=0))`
5. **`repeat_kv` with n_rep=1`** — must return the exact same tensor object
6. **`repeat_kv` with n_rep=4`** — shape `(B, T, n_kv*4, head_dim)`; each
   kv head appears 4 consecutive times
7. **Attention output shape with MHA** — when `n_kv_heads == n_heads`, shape
   is `(B, T, dim)`
8. **Attention output shape with MQA** — when `n_kv_heads == 1`, shape is
   still `(B, T, dim)`
9. **Causal: varying sequence length** — run attention on T=8 and T=16;
   the first 8 position outputs should be identical (consistent with causality)
10. **No NaN in output** — with random inputs, attention output should not
    contain NaN or Inf

### Hints

- Import from `src.model.attention`
- For test 4 (relative positions), you need to apply RoPE manually — look
  at `apply_rotary_emb` and construct a single-head case
- For MQA (test 8), create a `ModelConfig` with `n_kv_heads=1`

### How to verify

```bash
python -m pytest tests/test_attention.py -v
```

---

## Task 6 — Clean up `src/__init__.py`

**File:** `src/__init__.py`  
**Current state:** `# placeholder`  
**Why it matters:** this file is the top-level package init for all source
code. Right now importing `from src.model import GPT` works, but `from src
import GPT` doesn't. A proper `__init__.py` makes common objects
conveniently importable.

### What to do

Add a short `__init__.py` that re-exports the most commonly used symbols:

```python
# Things you'd typically import at the top of a training script
from src.model import GPT, ModelConfig, small_config, medium_config
from src.tokenizer import BPETokenizer
```

Keep it minimal — only export things a user would actually import at
the top level. Internal helpers stay in their submodules.

### How to verify

```python
# This should work after your change:
from src import GPT, ModelConfig, BPETokenizer
print(GPT.__module__)       # src.model.gpt
print(BPETokenizer.__module__)  # src.tokenizer.bpe_tokenizer
```

---

## Task 7 — Create `data/` and `checkpoints/` directories with `.gitkeep`

**Files:** `data/.gitkeep`, `checkpoints/.gitkeep`  
**Current state:** neither directory exists in the repo  
**Why it matters:** when someone clones the project and runs
`python scripts/train_tokenizer.py`, the first thing it tries to save to
is `checkpoints/tokenizer.json` — but `checkpoints/` doesn't exist yet
and the script will crash.

### What to do

1. Create the two directories
2. Add a `.gitkeep` file inside each (an empty file that makes git track
   the otherwise-empty directory)
3. Create a `.gitignore` in the project root (or add to it if it exists)
   that ignores the **contents** of these directories but keeps the
   directories themselves:

```
# In .gitignore:
checkpoints/*
!checkpoints/.gitkeep
data/*
!data/.gitkeep
```

### How to verify

```bash
# These should both work without errors:
python scripts/train_tokenizer.py --num_examples 100 --vocab_size 500
ls checkpoints/tokenizer.json
```

---

## Task 8 — Run the full pipeline end-to-end

**Files:** all scripts  
**Why it matters:** 132 unit tests pass, but unit tests don't prove the
scripts compose correctly into a real training run. This task is about
actually running the four phases and confirming the outputs look right.

### What to do

Run each step in sequence. For speed, use small settings:

```bash
# 1. Train tokenizer (fast — ~2 min)
python scripts/train_tokenizer.py --vocab_size 2000 --num_examples 5000

# 2. Pretrain for a handful of steps to confirm the loop runs
python scripts/pretrain.py --max_steps 50 --batch_size 2 --mixed_precision no

# 3. Fine-tune with demo data
python scripts/finetune.py \
    --pretrain_ckpt checkpoints/best.pt \
    --demo \
    --max_steps 20 \
    --mixed_precision no

# 4. Align with DPO
python scripts/align.py \
    --method dpo \
    --sft_ckpt checkpoints/finetune/best.pt \
    --demo \
    --max_steps 20
```

### What to check at each step

| Step | Success looks like |
|---|---|
| Tokenizer | `checkpoints/tokenizer.json` created, sample encode/decode printed |
| Pretrain | Loss printed every 10 steps, starts near `log(vocab_size)`, decreasing |
| Finetune | Loss printed, LoRA param count shown (~0.5 % of total) |
| Align | DPO margin increasing over steps |

### Common issues and fixes

| Issue | Fix |
|---|---|
| `FileNotFoundError: checkpoints/` | Complete Task 7 first |
| `RuntimeError: bf16 not supported` | Add `--mixed_precision no` |
| `CUDA out of memory` | Reduce `--batch_size` to 1 |
| `ImportError: bitsandbytes` | Only affects `--qlora` flag; regular finetune doesn't need it |

---

## Completion Checklist

```
Task 1 — main.py entry point              [ ]
Task 2 — Remove unused imports            [ ]
Task 3 — Auto-detect mixed_precision      [ ]
Task 4 — tests/test_tokenizer.py          [ ]
Task 5 — tests/test_attention.py          [ ]
Task 6 — src/__init__.py                  [ ]
Task 7 — data/ and checkpoints/ dirs      [ ]
Task 8 — Full pipeline end-to-end run     [ ]
```

When all eight are done, run the full test suite one final time:

```bash
python -m pytest tests/ -v
```

And then the complete pipeline:

```bash
python main.py tokenize
python main.py pretrain
python main.py finetune --demo
python main.py align --method dpo --demo
```
