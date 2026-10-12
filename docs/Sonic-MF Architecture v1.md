# Sonic + ManifoldFlow Architecture v1

Describes the system as actually implemented and currently training in
`src/manifoldflow/experiments/manifoldflow_production_train.py`, as of the
70B-additional-token production run launched from the 30B-token Sonic
checkpoint. This is a "what's actually built and running" doc, not a
design proposal — see `Sonic-MF Training Design v0.1.md` for the earlier
pre-implementation design this superseded/refined during build.

## 1. What this is

Sonic is a pretrained recurrent SSM language model. ManifoldFlow is a
continuous external memory (`BasinMemory`) wired into Sonic's forward pass
through a small set of additional trainable modules (the "adapter"). The
combined system is trained to: (a) keep doing ordinary next-token
prediction on plain text, and (b) write facts encountered in a session into
an external memory bank and, when later asked about them, retrieve the
right one and use it to answer correctly — without Sonic's own recurrent
state having to carry that information across the whole session itself.

## 2. Components

### 2.1 Sonic SSM backbone

Vendored byte-identical from `GTR-sonic-ssm/gator/models/sonic_ssm.py` into
`src/manifoldflow/sonic_ssm.py` — never hand-edited, only re-copied
upstream. The currently-loaded checkpoint's config:

| field | value |
|---|---|
| `vocab_size` | 50,000 |
| `d_model` | 1,504 |
| `lexical_dim` | 256 |
| `n_layers` | 29 |
| `state_dim` | 64 |
| `n_heads` | 1 |
| `num_modes` | 120 (sparse routing) |
| `sparsity_target` | 0.78 |
| `dense_a` | `false` |
| `activation_checkpointing` | `true` (see §6.1 — important caveat) |
| `memory_num_slots` | 0 (Sonic's own internal memory bank is **disabled**) |

Each of the 29 layers is an independent recurrent SSM block. Two scan
implementations exist for the (non-dense) diagonal-transition recurrence:

- **`doubling`** (default): a parallel prefix-doubling scan, plain PyTorch
  ops.
- **`hgrn`**: a `torch.compile`-friendly Triton kernel port of
  flash-linear-attention's `chunk_hgrn`, vendored at `src/gator_linear/`.
  Mathematically equivalent (verified: matching retrieval accuracy,
  ~0.3-0.6% loss deltas consistent with bf16 accumulation-order noise, not
  a bug) and measurably faster + lower VRAM in this project's own
  architecture (~3-4% throughput, ~1.8GB less VRAM at lanes=6 unfrozen) —
  now the default for the production run. Requires `flash-linear-attention`
  + `triton`; getting it importable required removing a broken
  `torchvision` install that an *unrelated, optional* `fla.models` import
  path was transitively pulling in (see git history).

`dense_a=false` means the O(seq_len) per-token Python-loop transition path
in `SonicSSMBlock.forward` is **not** taken — the diagonal, parallelizable
scan (`doubling` or `hgrn`) is what actually runs.

Sonic's *own* internal read/write memory (`SonicReadWriteMemory`, gated by
`memory_num_slots > 0`) is a **separate, unrelated mechanism** to
ManifoldFlow's `BasinMemory` — it's disabled in this checkpoint
(`memory_num_slots: 0`) and plays no role here.

### 2.2 ManifoldFlow adapter

Three small trainable modules, none part of the vendored Sonic file:

- **`SonicKeyProjection`** (`src/manifoldflow/sonic_key_projection.py`) —
  one identity-initialized `Linear(d_model, d_model, bias=False)` +
  `F.normalize`. Two independent instances are used: `key_projection` (for
  retrieval keys) and `value_projection` (for stored values) — same class,
  different weights, trained independently.
- **`SonicMemoryPort`** (`src/manifoldflow/sonic_memory_port.py`) —
  injects a retrieved memory vector into Sonic's hidden states:
  `down: Linear(memory_dim, rank)` → `ReLU` → `up: Linear(rank, d_model)`
  (zero-initialized, so injection starts as a no-op) → gated by
  `sigmoid(Linear(d_model, 1))` (bias initialized negative, so the gate
  starts mostly closed) → `hidden + gate * injected`. One retrieved vector
  per batch row, broadcast across the whole answer span (not re-queried
  per token).
- **`BasinMemory`** (`src/manifoldflow/continuous_memory.py`) — a
  basin-keyed payload store. `write()` routes a new key to the nearest
  existing basin if within `novelty_threshold`, else creates a new basin.
  **This training run sets `novelty_threshold=0.0`**, which makes that
  nearest-basin check provably unreachable (a squared distance is never
  exactly ≤ 0), so every write always creates a new basin — no basin
  merging happens during training. `write_always_new()` exploits this to
  skip the (otherwise pointless) novelty-check's CPU-GPU syncs.
  `read()`/`read_raw()` do an approximate nearest-neighbor lookup over the
  bank's basins.

Each lane (see §3) also keeps its own plain Python lists,
`statement_keys`/`statement_values` — the *exact*, differentiable
per-statement keys/values used for the retrieval loss target set, distinct
from `BasinMemory`'s own approximate basin index (used for the actual
retrieval mechanism and accuracy tracking). Two retrieval computations
coexist deliberately: one exact and differentiable (for the loss), one
approximate (matching what a real bank lookup would return at inference).

## 3. Data & session structure

Sessions are pretokenized offline into shards
(`scripts/pregenerate_manifoldflow_sessions.py`) — interleaved chat, plain
text, and "distractor" episodes (statements to be written, plus queries
that ask about a previously-written statement). Each shard holds many
sessions as JSONL rows (`token_ids`, `episodes`, `total_tokens`).

Training slices each session sequentially into **physical windows of 8,192
tokens**. A **lane** is one persistent worker: it holds one session,
advances through that session's windows one at a time across training
steps, and resets to a fresh session once exhausted. A lane's `BasinMemory`
bank and `statement_keys`/`statement_values` persist across windows within
a session — that's the actual long-context mechanism.

Sonic's *own* recurrent state does **not** carry across windows or steps:
`run_step()` calls `model.forward_hidden(input_ids)` with no `states`
argument, so every window's forward pass starts from a fresh zero state per
layer. This was a deliberate simplification and turned out to matter for
`scan_backend="hgrn"`, which cannot differentiate through a carried initial
state — a non-issue here since no state is ever carried.

## 4. Per-step routing

One micro-step of `run_step()`, run once per lane-batch:

1. **Advance windows.** Each lane pulls its next 8,192-token slice
   (prefetched in the background by a `ThreadPoolExecutor` pool, ahead of
   need).
2. **Sonic forward.** Batched `model.forward_hidden(input_ids)` over all
   lanes at once → `hidden` of shape `[num_lanes, 8191, d_model]` (8,191 =
   window_tokens − 1; the last input position has no next-token target).
   Runs under `torch.no_grad()` while Sonic is frozen, or with full
   autograd once unfrozen (see §5.1) — the forward code path is otherwise
   identical either way, so there's no train/inference architecture
   mismatch.
3. **Plain-LM loss.** Chunked cross-entropy (`_ChunkedLinearCrossEntropy`)
   over the whole batch, next-token prediction — runs on *every* window
   regardless of episode content.
4. **Write phase — unconditional, every window with a pending statement.**
   For every (lane, span) with a statement due to be written this window:
   - Gather all such spans' hidden states in one `index_select`-based
     vectorized gather (`_gather_padded_spans`/`_batched_pool`), mean-pool
     each span.
   - `key_projection(pooled)` → key; `value_projection(pooled or
     quantity-pooled)` → value (a `quantity`-mode subset pools a narrower
     span for `value_source="quantity"`, the mode this run uses).
   - `lane.bank.write_always_new(key, ...)` — appends a new basin.
     `lane.statement_keys`/`statement_values` also get the differentiable
     key/value appended.
5. **Query phase — conditional, only lanes whose current window has a due
   query.** For each such (lane, query):
   - Pool the query span the same vectorized way → `key_projection` →
     `query_key`.
   - **Retrieval loss**: `info_nce_retrieval_loss(query_key,
     lane.statement_keys, target_index, temperature)` — an
     `num_statements`-way classification (dot-product logits over that
     lane's own accumulated statement keys, cross-entropy against the true
     target). Exact and differentiable.
   - **Actual retrieval**: `lane.bank.read_raw(query_key, top_k)` — the
     approximate basin lookup. With `recency_weight > 0`, breaks ties among
     the top-k hits by score plus a recency bonus
     (`recency_decay ** age`). Determines `retrieval_correct` (the
     accuracy metric) and which stored value gets retrieved.
   - **Injection**: the retrieved value is broadcast via
     `port(answer_hidden_span, retrieved_value)` — gated injection into
     Sonic's hidden states over the answer span only (not the whole
     window). If no lane has a due query this step, `port()` is **never
     called** — nothing is retrieved, nothing is injected, Sonic's hidden
     states pass through the plain-LM path unmodified.
   - **Decode loss**: injected hidden states → `model.logits_from_hidden`
     → cross-entropy against the true answer tokens (target shift already
     baked into the span construction).
6. **Combine and backward.**
   `micro_loss = 0.6 * plain_lm_loss + 0.4 * (retrieval_loss + decode_loss
   averaged over due queries)`; `(micro_loss / grad_accum_steps).backward()`.
7. **Repeat for `grad_accum_steps` micro-steps** (8 in the current run),
   accumulating loss/gradients on-GPU with no host sync between them, then
   one `optimizer.step()`.
8. **Periodic**: eval (every 200 updates), checkpoint save (every 200
   updates, plus any new best), accuracy-gated unfreeze check.

## 5. Training recipe

### 5.1 Cross-optimizer staged unfreeze

One `torch.optim.AdamW` (fused) with two param groups:

- **Adapter** (`port` + `key_projection` + `value_projection`) — trainable
  from step 1, LR `1e-3`.
- **Sonic's own params** — join the optimizer from step 1 (so AdamW's
  moment buffers exist when unfreeze happens) at LR `1e-5`, but
  `requires_grad_(False)` until `retrieval_acc >= 0.95` holds for
  `unfreeze_patience=3` consecutive evals. AdamW safely skips
  no-grad params in the meantime.

### 5.2 WSD learning-rate schedule

Warmup-Stable-Decay, chosen over cosine specifically because the stable
phase's length is accuracy-gated, not known in advance — a cosine curve
needs its full shape upfront:

```
update <= warmup_steps:        lr_mult = update / warmup_steps
update > total_steps - decay:  lr_mult = 1 - progress * (1 - min_lr_ratio)
otherwise:                     lr_mult = 1.0
```

Current run: `warmup_steps=2000`, `decay_steps=26700`, `min_lr_ratio=0.1`
(kept as absolute update counts from the checkpoint's original recipe when
`--steps` was recalculated for the new per-update token count — see §7).

### 5.3 Gradient accumulation

`--num-lanes 6 --grad-accum-steps 8` → effective batch 48 sessions,
393,216 tokens per optimizer update — matching vanilla Sonic's own
recipe exactly (physical batch 6, accumulation 8, context 8192). Reaches
that effective batch size without needing 48 lanes' worth of simultaneous
VRAM.

### 5.4 Compile & precision

`torch.compile(mode="max-autotune-no-cudagraphs")`, applied **per-layer**
(`model.compile_layers()`) rather than whole-model — confirmed this
matches vanilla Sonic's own training script, which explicitly disables
full-model compile for this architecture (`"full-model --compile is
intentionally disabled for Sonic"`) in favor of the same scoped per-layer
path. `mixed_precision=bf16`. The adapter modules (`port`,
`key_projection`, `value_projection`) are **not** compiled — tried once,
measured a real regression (~13.1s/update vs ~12.1-12.4s/update eager),
reverted (see git history for the removed code + reasoning).

### 5.5 Checkpoint format

`CHECKPOINT_FORMAT` payload includes `model` / `port` / `key_projection` /
`value_projection` state dicts, `optimizer` state, `update`/`tokens_seen`/
`sonic_unfrozen`/`consecutive_high_evals`/`best_eval_retrieval_accuracy`,
RNG state, tokenizer identity, and the full `recipe` dict (hyperparameters,
for the next resume to read back). Best (`*_best.pt`) and latest
(`*.pt`) are saved separately; best only updates on eval improvement.

## 6. Known gaps / open questions as of this doc

### 6.1 Activation checkpointing status is unverified

The loaded checkpoint's own `model_config` has `activation_checkpointing:
true, checkpoint_every: 1` baked in from upstream Sonic pretraining, and
`SonicSSMLanguageModel.__init__` sets `self.gradient_checkpointing = True`
from that config unconditionally — **before** this script's own
`--activation-checkpointing` CLI flag ever runs. That flag only ever calls
`gradient_checkpointing_enable()` (turns it on); there is no code path that
calls `gradient_checkpointing_disable()` when the flag is absent. Net
effect: **activation checkpointing has likely been on for every throughput
measurement taken during this project's optimization work**, including
runs that were treated as an "uncompiled/unmodified baseline." An earlier
session conclusion that explicitly enabling activation checkpointing was a
"net negative" compared runs at different lane counts, not a clean
on/off toggle at the same lane count — that comparison was confounded and
should be treated as unverified, not settled. A `--no-activation-checkpointing`
flag (calling `gradient_checkpointing_disable()`) would be needed for a
genuine controlled test.

### 6.2 Throughput vs. vanilla Sonic

Vanilla Sonic (same architecture, no adapter) trains at ~57k tok/s with
full backward + AdamW, batch 6 / accum 8 / context 8192 — the same
effective-batch shape this run uses. Measured for this project: ~53k tok/s
frozen/adapter-only, ~32k tok/s once Sonic is unfrozen (up from an initial
~20.6k tok/s before this session's fixes — see git history for the
sync-elimination and vectorized-gather changes that closed most of that
gap). The remaining unfrozen-phase gap to vanilla's 57k has not been fully
attributed; compile-granularity and adapter-compile were both tested and
ruled out as explanations (see §5.4 and git history).

## 7. Current live run (concrete config)

Resumed from update 600 of the original 30B-checkpoint continuation
(recipe recalculated for the new per-update token count once lanes/accum
changed from the original lanes=32/no-accumulation shape):

```
--num-lanes 6 --grad-accum-steps 8 --window-tokens 8192
--scan-backend hgrn --compile layers --compile-mode max-autotune-no-cudagraphs
--mixed-precision bf16
--learning-rate 1e-3 --sonic-learning-rate 1e-5
--warmup-steps 2000 --decay-steps 26700 --min-lr-ratio 0.1
--unfreeze-accuracy-threshold 0.95 --unfreeze-patience 3
--plain-lm-loss-weight 0.6 --distractor-loss-weight 0.4 --value-source quantity
--steps 178220   (600 + ceil((70e9 - 157,286,400) / 393,216))
```

Checkpoints and logs write to `/path/to/manifoldflow_production/` (moved off
the D: USB drive, which measured ~6.3 MB/s writes vs. C:'s ~245 MB/s
through the same WSL 9P bridge — a drive-specific bottleneck, not a WSL
overhead one). Shard reads still come from `/path/to/manifoldflow_sessions`
— read bandwidth was never the constraint (a 50k tok/s target needs ~313
KB/s of shard-read throughput; even D:'s slow side had ~30 MB/s reads,
~100x headroom).
