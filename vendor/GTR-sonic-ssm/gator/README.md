# Gator Python API

The `gator` package is the reusable library boundary. Top-level experiment scripts should be thin clients that assemble models, tasks, and callbacks through this API.

See [`ARCHITECTURE.md`](../ARCHITECTURE.md) for the canonical Arrow cell,
relay equations, cell-coupling and Gator-S stewardship transports, and
parameter accounting.
See [`PERFORMANCE_TODO.md`](../PERFORMANCE_TODO.md) for the pinned six-item
GPU scaling program and rollback ledger.

## Optimization API

```python
from gator import (
    OptimizationConfig,
    OptimizationEngine,
    SchedulerConfig,
)

config = OptimizationConfig(
    name="adamw",
    learning_rate=8e-3,
    weight_decay=1e-4,
    gradient_clip=1.0,
    scheduler=SchedulerConfig(
        name="milestone",
        total_steps=20_000,
        milestones=(12_000, 16_000),
        gamma=0.25,
    ),
)

engine = OptimizationEngine(model, config)

for batch in batches:
    loss = compute_loss(model, batch)
    result = engine.step(loss)
    print(result.step, result.loss, result.learning_rate, result.gradient_norm)
```

The engine owns optimizer and scheduler state and provides a serializable `state_dict()` suitable for checkpoints.

### Experimental active-row AdamW

Row-indexed private node tables have an explicit experimental optimizer API:

```python
from gator import (
    ActiveRowAdamW,
    active_rows_from_routes,
    estimate_active_row_state,
    sparse_row_gather,
)

optimizer = ActiveRowAdamW(
    [node_weight],
    lr=1e-3,
    weight_decay=1e-2,
    mode="lazy",
)

selected = sparse_row_gather(node_weight, active_node_ids)
loss = use_selected_rows(selected)
loss.backward()
optimizer.step()
```

`sparse_row_gather` supports row-indexed tensors such as `[N,D,D]` and emits
a COO gradient with one sparse row dimension. `active_rows_from_routes`
returns the sorted union selected by node IDs, validity, and optional minimum
mass. Dense-gradient users can call `optimizer.mark_active_rows(...)`; route
activity accumulates across relays until the next step.

`ActiveRowAdamW` has two intentionally distinct modes:

- `exact` matches dense AdamW. Old momentum and decoupled weight decay move
  inactive rows even when their current gradient is zero, so this mode
  densifies gradients and is a correctness oracle rather than a speedup.
- `lazy` advances parameters, moments, decay, and row-local clocks only for
  active rows. It can reduce arithmetic, but it changes optimizer semantics
  and is therefore a training ablation.

Both modes retain dense first- and second-moment tensors. Lazy mode does not
reduce asymptotic Adam state memory and adds one int64 clock per row. A real
state-memory saving requires a future paged/fused moment store. Use
`estimate_active_row_state` and its `ActiveRowStateEstimate` result to report
the current footprint. `ActiveRowMode` names the `"exact" | "lazy"` type.
Checkpoint metadata records the semantic mode and rejects cross-mode loading.

This optimizer is deliberately standalone and is not used by the default
`OptimizationEngine` or language trainer. Measure batch-level active-node
union before integrating it: a batch may be sparse per packet while touching
nearly every private node overall.

## Compute policy API

Hardware execution policy is separate from model architecture and learned
parameters:

```python
from gator import (
    CompileConfig,
    ComputeConfig,
    Float32RuntimeConfig,
    compile_module,
    configure_float32_runtime,
)

compute = ComputeConfig(
    float32=Float32RuntimeConfig(
        matmul_precision="high",
        cudnn_tf32=True,
    ),
    compile=CompileConfig(
        enabled=True,
        mode="default",
        shape_padding=True,
    ),
)

runtime = configure_float32_runtime(compute.float32, device)
train_model = compile_module(model, compute.compile)
print(runtime.summary())
```

`Float32RuntimeConfig`, `CompileConfig`, and `ComputeConfig` validate and
serialize the parameter-free execution policy. The language trainer records
both the requested policy and resolved device capability as checkpoint runtime
metadata. A checkpoint can therefore move between CPU and different CUDA
generations without treating a hardware policy as learned architecture.

Equivalent language-training flags are:

- `--float32-matmul-precision {highest,high,medium}`
- `--cudnn-tf32` / `--no-cudnn-tf32`
- `--compile-mode {default,reduce-overhead,max-autotune,max-autotune-no-cudagraphs}`
- `--compile-shape-padding` / `--no-compile-shape-padding`

Custom compile shape padding is available only with `compile-mode=default`.
The designated Blackwell performance policy is matmul precision `high` with
cuDNN TF32 enabled, but it still requires an on-device A/B; no GPU speed result
is recorded yet. Automatic mixed precision is intentionally deferred. The
current routing graph uses a `-1e9` masked logit, which overflows FP16, and a
`1e-9` entropy clamp, which underflows FP16. A future AMP path must keep
routing, softmax, entropy, and language/alignment losses in an FP32 island.

## Packed Convergent routing API

The reusable packed primitives are exported from `gator.models`:

```python
from gator.models import (
    HierarchicalPackedIngress,
    PackedNodeFrontier,
    SharedTrunkNodeAdapters,
    adaptive_uncertainty_top_k,
    packed_convolution_node_context,
    packed_convergent_node_context,
    packed_node_occupancy,
    transport_packed_frontier,
)
```

They are grouped by contract below.

| API | Contract |
|---|---|
| `PackedNodeFrontier` | Immutable fixed-capacity `(nodes, mass, valid, n_nodes)` passport. Invalid slots are structural padding; differentiable probability mass is kept separate from discrete node IDs. |
| `dense_to_packed_frontier` / `packed_frontier_to_dense` | Pack a dense node distribution for oracle comparisons or reconstruct it only for diagnostics. |
| `merge_packed_destinations` | Merge duplicate destination IDs, apply optional mass thresholding, normalize, and preserve fixed output capacity without allocating a node-wide passport. |
| `transport_packed_frontier` | Gather local connection destinations, multiply route mass by transition probability, merge collisions, and carry the bounded frontier into the next relay. |
| `packed_node_occupancy` | Compute per-packet destination occupancy from packed routes. It uses a static dense node-only reduction when that is cheaper than packed sorting, but never reconstructs a lane passport. |

`packed_convergent_writer_features` and
`packed_convergent_node_context` implement the exact receiver-memory matrix
oracle. `PackedConvergentMemoryDiagnostics` exposes its dense node-indexed
writer features and memory for equivalence tests. This backend proves packed
route correctness; it is not the intended long-context production kernel.

`packed_convolution_node_context` implements the same normalized
relative-position equation through direct grouped convolution or FFT. It
processes hidden features in bounded chunks, avoids `[T,T,N]`, and gathers
only memory requested by packed readers. `ConvolutionBackend` names
`"direct" | "fft"`, while `PackedConvolutionMemoryDiagnostics` exposes the
active node union, denominator, and kernel for profiling. Direct convolution
can win at short context despite FFT's better asymptotic scaling. Optional
active-node compaction is currently dynamic/eager; target-GPU benchmarking is
required before selecting a default.

`opaque_packed_fft_node_context` wraps the complete FFT receiver and an
explicit chunk-local adjoint in real-valued PyTorch custom operators. This is
the compile-safe `fft_opaque` boundary: cuFFT still performs the transforms
while Inductor never sees complex spectra. Backward calls the selected packed
reader's explicit adjoint, applies conjugated-spectrum FFT correlations, and
immediately folds each feature chunk through the writer scatter; it does not
invoke nested autograd or replay the whole receiver with `torch.func.vjp`.
`packed_reader_context` independently selects the portable Torch
packed-reader oracle or a CUDA float32 Triton kernel. The Triton
forward/backward fuse denominator normalization, node gather, route weighting,
and K reduction without materializing the expanded `[B,T,L,K,C]` tensors.
Explicit Triton selection raises when unsupported and never silently changes
the A/B.

`adaptive_uncertainty_top_k` returns an `AdaptiveTopKResult` containing a
fixed-maximum-width frontier plus packet-local selected K, normalized entropy,
top-one margin, and uncertainty. `AdaptiveTopKMode` is
`"entropy" | "margin" | "hybrid"`. The live-count decision is discrete;
gradients pass through retained masses and diagnostics, not through the
integer K. This changes the routing estimator and must remain a separate A/B
from packed execution itself.

`SharedTrunkNodeAdapters` evaluates one shared `D x D` packet trunk and
private node embeddings, biases, and rank-`R` down/up adapters only at packed
node IDs. The helpers `shared_trunk_adapter_parameter_count`,
`shared_trunk_adapter_macs`, `full_private_parameter_count`, and
`full_private_sparse_macs` report comparable conventions. For `P` packets,
`N` nodes, width `D`, retained width `K`, and adapter rank `R`:

```text
parameters = D^2 + D + N * (2D + 2DR)
active MACs = P * D^2 + P * K * 2DR
```

This factorization changes learned capacity; the full-private node bank remains
the capability oracle.

`HierarchicalIngressLayout` validates a fixed padded partition in which every
node occurs exactly once. `hierarchical_packed_ingress` and the module wrapper
`HierarchicalPackedIngress` score coarse groups first and then nodes within a
fixed number of selected groups. Every node remains reachable whenever its
group wins. `HierarchicalIngressCost` and
`estimate_hierarchical_ingress_cost` report the candidate/scoring trade-off.
Learned group keys and biases add `groups * (router_rank + 1)` parameters;
derived groups add none. Hierarchical selection is an architecture A/B versus
flat ingress, although selecting every group exactly matches the corresponding
dense hierarchical logit.

The integrated Convergent model keeps dense routing, matrix memory,
full-private nodes, and flat ingress as compatibility defaults. Fixed packed
frontiers and convolutional memory are execution backends; adaptive K, shared
adapters, and hierarchical ingress alter model or estimator behavior. Dynamic
node-union memory compaction and zero-row dispatch are eager-only until a
fixed-cap fused implementation exists. See
[`GATOR_LANGUAGE_TRAINER_ARGUMENTS.md`](../GATOR_LANGUAGE_TRAINER_ARGUMENTS.md)
for the corresponding trainer flags and legal combinations.

## Data pipeline API

Fixed-window sampling, producer overlap, and device transfer are reusable
library policies rather than trainer-specific code:

```python
from gator import BatchPipelineConfig, LanguageBatchPipeline
from gator.tasks import LanguageBatcher

provider = LanguageBatcher(encoded_examples, task_config, seed=0)
config = BatchPipelineConfig(
    batch_size=32,
    device="cuda",
    mode="prefetch",
    prefetch_batches=2,
    pin_memory=True,
    non_blocking=True,
    dedicated_cuda_stream=True,
)

with LanguageBatchPipeline(provider, config) as batches:
    batch = batches.next_batch()
    checkpoint_data_state = batches.state_dict()
```

`LanguageBatcher` pre-collates a finite corpus and samples through a vectorized
row gather. `PackedLanguageSource` performs the same direct gather over
fixed-width int32 storage. `MixtureLanguageBatcher` preserves the historical
Python RNG selection order while keeping packed rows tensorized.

`LanguageBatchPipeline` uses one bounded producer, keeps queued batches on the
CPU, optionally pins them, and can stage transfer on a dedicated CUDA stream.
Its checkpoint contract stores queued batches in consumption order followed by
the provider state after those batches were sampled. Exact continuation is
advertised only by providers that can restore all sampling state. Generic
streaming JSONL iterators intentionally report `exact_state_supported=False`.
Finite and packed providers bind their state to a SHA-256 fingerprint of the
actual fixed token windows, including masks and completion flags where those
are independent state, not only row count and shape. Packed-source constructor
inputs are cloned; provider corpus tensors are immutable after construction.

The current consumer waits for the staged copy before entering forward. CPU
preparation can overlap model work; device-batch lookahead does not yet. The
pipeline's transfer statistic measures host enqueue latency rather than
CUDA-event H2D duration.

Ordered startup preparation lives in `gator.preparation`:

```python
from gator.preparation import ParallelMapConfig, tokenize_assistant_examples

encoded = tokenize_assistant_examples(
    examples,
    tokenizer,
    config=ParallelMapConfig(
        backend="processes",
        workers=4,
        chunk_size=256,
        max_pending_chunks=8,
    ),
)
```

Thread and spawned-process backends preserve source order and bound pending
work. The coordinator remains responsible for shuffle, deduplication, filter
statistics, and all other stateful stream decisions.

The corresponding language-trainer flags are:

- `--execution-preset {manual,dense_gpu_safe,dense_gpu,optimized_gpu_safe,optimized_gpu}`
- `--batch-pipeline {serial,prefetch}`
- `--prefetch-batches`
- `--pin-memory` / `--no-pin-memory`
- `--non-blocking-transfer` / `--no-non-blocking-transfer`
- `--dedicated-copy-stream` / `--no-dedicated-copy-stream`
- `--resume-data-state {auto,exact,portable_exact,reset}`
- `--tokenizer-workers`
- `--tokenizer-worker-backend {threads,processes}`
- `--tokenizer-chunk-size`
- `--tokenizer-pending-chunks`

`benchmark_language_pipeline.py` is deliberately forward-free. It measures
input preparation and consumer latency, verifies a deterministic checksum, and
does not claim end-to-end model or GPU speed.

The GPU presets provide a staged trainer-level A/B. The `*_safe` variants use
strict FP32 without compilation; the unsuffixed variants additionally enable
compilation and FP32 Tensor Core/TF32 policy. Dense variants remain dense,
while optimized variants enable `packed_auto` with the persistent bucketed
backend. Explicit low-level flags override preset defaults. The selected
preset and resolved controls are recorded in trainer logs and checkpoint data
metadata. Long compiled Blackwell runs remain experimental because they have
eventually produced non-finite backward gradients; use
`optimized_gpu_safe` without compilation as the preserved Arrow fallback.

## Sparse stewardship API

Persistent packed execution is configured independently of stewardship
semantics:

```python
from gator.models import GatorLanguageConfig

config = GatorLanguageConfig(
    # ...language and Arrow dimensions...
    cell_transport_mode="stewardship",
    stewardship_execution="packed_auto",
    packed_stewardship_backend="persistent_bucketed",
)
```

`persistent_bucketed` is the default packed backend. It packs once at the
dense-to-sparse boundary and keeps branch state, node passport, mass, and
indices packed through uncertainty, pruning, support-only handoff transport,
and final readout. `repack` preserves the original globally padded,
pack/unpack-per-relay path as the forward numerical oracle. Gradients match
when pruning is disabled; sparse pruning deliberately removes the surrogate
gradient of routes that no longer execute.

Reusable public model-level primitives are exported from `gator.models`,
including `PackedStewardship`, `PackedGroupPlan`,
`build_packed_group_plan`, `bucketed_group_linear`, packed route diagnostics,
pruning/transport, receiver-memory, and readout helpers. This keeps experiment
scripts from duplicating sparse execution logic.

## Design rules

- Core modules must not depend on `argparse`, Colab, or a specific experiment.
- Configuration objects must be serializable and validated.
- CLI logic belongs in `gator.cli`.
- Training entrypoints consume the API rather than duplicating optimizer logic.
- General training/compute objects are exported from `gator`; reusable
  model-level objects are exported from `gator.models`.
- Checkpoints should store model state, training-engine state, architecture configuration, task configuration, and evaluation metadata separately.

## SmolTalk Everyday data API

Conversation parsing is kept separate from tokenization so language experiments
can compare byte, character, and learned compact vocabularies without changing
the train/validation split.

```python
from gator.data import assistant_examples, load_smoltalk, split_conversations

dataset = load_smoltalk(
    "/content/drive/MyDrive/GTR/training_data/smoltalk_everyday.jsonl"
)
train_conversations, validation_conversations = split_conversations(
    dataset,
    validation_fraction=0.1,
    seed=0,
)
train_examples = list(assistant_examples(train_conversations))
```

Splitting occurs at the conversation level. Assistant turns from one dialogue
can therefore never appear in both training and validation data.

`CompactTokenizer` learns frequent pieces from the training conversations only
and uses UTF-8 byte fallback for unseen text. It is lossless without requiring
an external tokenizer package.

## Simultaneous language formation

`GatorLanguageModel` is a vocabulary-agnostic, vectorized GTR-ARW cell ring.
The model receives a fixed context window and creates every masked response slot
in one forward pass. Response predictions are never fed back as inputs.

The first wide-cell SmolTalk configuration used a 2,048-token vocabulary, 64
context tokens, 32 simultaneous response tokens, width 32, ten cells, four
nodes, four relays, and four neighboring cells. It contained exactly 130,000
parameters. The stitched configuration separates the 32-wide language embedding
from width-8 graph cells. `private_hash` fits 62 literal 980-parameter cells in
129,304 parameters. `communal` removes the modulo-16 private lexical overlay,
uses the shared vocabulary plus cell identities, and has 121,864 parameters at
the same 62-cell topology.

```bash
python -u gator_language_train.py \
  --data training_data/smoltalk_everyday.jsonl \
  --lmsys-data training_data/lmsys_chat_1m.jsonl \
  --lmsys-weight 0.9 --lmsys-load-mode stream --lmsys-shuffle-buffer 16384 \
  --lmsys-max-characters 512 --lmsys-min-ascii-letter-ratio 0.9 \
  --lmsys-min-response-characters 8 --lmsys-require-user-prompt \
  --require-complete-context --trim-context-turns --deduplicate-examples \
  --steps 25000 --batch-size 32 --eval-batch-size 64 --log-every 1000 \
  --vocab-size 2048 --context-tokens 64 --response-tokens 32 \
  --whitespace-omission-penalty 0.10 --adjacent-whitespace-penalty 0.05 \
  --embedding-dim 32 --d-model 8 --cells 62 --lexical-mode communal \
  --cell-input-mode feature_gated --cell-output-mode grouped_concat \
  --nodes 4 --node-topology hybrid_hemispheric --relays 4 --neighbors 4 \
  --skip-first-assistant --scheduler cosine --minimum-lr 0.0001 \
  --learning-rate 0.003 --compile
```

The clean LMSYS path streams JSONL through a bounded shuffle buffer. It removes
non-English-heavy and tiny responses, requires a user prompt, drops only whole
old turns until the context fits, rejects a prompt if it still cannot fit, and
deduplicates normalized context/response pairs within each traversal. It does
not require the complete response to fit: the first 32 target tokens remain a
valid simultaneous formation window. Add `--require-complete-response` only for
an explicitly labeled short-response experiment.

`--skip-first-assistant` applies to SmolTalk's repeated greeting. LMSYS keeps its
first assistant answer unless `--lmsys-skip-first-assistant` is explicitly set;
most audited LMSYS conversations contain only one assistant reply. The first
512 LMSYS conversations are reserved for validation. Logs and checkpoints store
filter rejection counts, mixture sample counts, and streaming restart counts.
The real-data smoke test retained 5,320/5,726 SmolTalk training examples (92.9%)
and 130/210 sampled LMSYS training candidates (61.9%).

### Multi-corpus continuation

The trainer accepts repeated `--chat-data` and `--chat-source-weight` pairs for
additional `text`/`source` transcript JSONL corpora. `--streaming-weight`
controls the total probability of all streamed sources, while source weights
are normalized inside that streamed share. Additional corpora are training-only
by design, so the original SmolTalk/LMSYS validation benchmark remains fixed.
Every stream has its own shuffle buffer, restart counter, filter counters, and
sample count. Additional chat streams reuse the LMSYS user-prompt, ASCII-ratio,
minimum-response, complete-window, turn-trimming, and deduplication controls;
`--chat-max-characters` and `--chat-shuffle-buffer` override their two
source-specific streaming settings.

`--resume-checkpoint` restores the model and its exact tokenizer. With
`--resume-optimizer`, Adam moments are also restored, but a new scheduler begins
at step zero and `--learning-rate` explicitly sets the new phase's initial LR.
`--steps` means additional steps, logged checkpoint step numbers remain global,
and completed alignment/stewardship warmups are not replayed. This is the
recommended way to extend a learned model onto a broader corpus without
renumbering vocabulary IDs or reheating its routing policy.

The overnight reference mixture is 10% finite SmolTalk Everyday and 90%
streamed chat. The streamed portion uses relative weights 4:3:2 for LMSYS,
Tulu3 SFT, and SmolTalk Preferences. See the final two cells of
`Gator_Language_Colab.ipynb`.

The optional whitespace formation terms target under-formation without
penalizing padding after EOS. `--whitespace-omission-penalty` applies when an
active non-whitespace target slot receives whitespace probability;
`--adjacent-whitespace-penalty` discourages consecutive whitespace probability
inside the scored response. The initial A/B uses weights 0.10 and 0.05. Logs
report content-only PPL/top-1/top-5 accuracy, whitespace omission and mass,
adjacent whitespace, and formation coverage. Aggregate validation PPL remains
ordinary unmodified cross-entropy for comparison with earlier runs.

### Output symbols and alignment objectives

`gator-compact-v2` adds a true scored, non-rendering BLANK while preserving the
V1 special/byte ID layout. PAD remains storage only, MASK remains a control
marker, EOS terminates a complete response, and whitespace remains visible
text. Its requested vocabulary size is exact; tokenizer training asks for a
smaller size if the corpus cannot fill every learned-piece row. Gator uses
BLANK to initialize output queries. Both Gator and the matched
Transformer mask PAD, BOS, MASK, USER, and ASSISTANT out of response logits.
Ordinary decoding hides BLANK and stops at EOS; diagnostic decoding exposes
the complete slot vector.

The trainer now implements four objective modes:

| `--formation-objective` | Behavior |
|---|---|
| `slot_ce` | Historical rigid CE with the post-reference PAD tail ignored |
| `blank_ce` | Rigid addresses; known complete tails score BLANK, censored tails remain unobserved |
| `viterbi` | Detached best monotonic placement followed by all-slot CE |
| `monotonic` | Differentiable forward-sum over every legal emit-or-BLANK placement |

Aligned modes can warm from `blank_ce` through
`--alignment-warmup-fraction`; `--alignment-max-shift` limits rightward token
movement, with `-1` enabling all monotonic placements. The optional
`--sequence-quality-penalty` adds a global semantic-composition,
expected-length, and EOS-count auxiliary. Whitespace penalties are rejected
for aligned objectives because their old rigid-slot interpretation would be
incorrect.

Use `--reference-tokens` below `--response-tokens` to reserve real physical
alignment slack. For example, `--reference-tokens 32 --response-tokens 40`
asks Gator to place a 32-piece reference prefix across 40 simultaneous output
slots. With equal values, a full-width reference has only the rigid path.

Evaluation logs historical CE/PPL plus BLANK CE, Viterbi and forward-sum loss
normalized both per output and per required token, balanced best-aligned
accuracy, required/content accuracy, content precision/recall/F1, documented
edit accuracy and counts, rendered exactness, displacement, BLANK rate,
EOS/tail errors, and forbidden-special rate. Version-1 tokenizers/checkpoints
remain readable but do not silently opt into BLANK alignment. See the
[formation-alignment roadmap](../Papers/GATOR_FORMATION_ALIGNMENT_ROADMAP.md).

Node topology is checkpointed and selectable through `--node-topology`:

- `asymmetric`: validated directed Arrow mask
- `symmetric`: equal self/left/right cyclic connectivity
- `hybrid_alternating`: symmetric and asymmetric cells alternate around the ring
- `hybrid_hemispheric`: one contiguous half-ring of each cell type

Lexical organization is checkpointed and selectable through `--lexical-mode`:

- `private_hash`: add a private 16-entry `token_id % 16` table to every cell
- `communal`: share full token meaning and add only a learned cell identity

Cell input views are selectable through `--cell-input-mode`:

- `shared_projection`: one identical projected token summary for every cell
- `feature_gated`: learned per-cell gates before the shared projection

Cell population readout is selectable through `--cell-output-mode`:

- `shared_pool`: compress all cells to one width-8 state
- `grouped_concat`: concatenate four independently pooled width-8 cell groups

## Gator-S token stewardship

`GatorLanguageModel` now exposes two population transports:

- `broadcast_mix` is the legacy path. Every cell processes every slot, neighbor
  states are mixed by static learned gates, and final cell participation is a
  static vote.
- `stewardship` is Gator-S. Every token carries explicit ownership over
  cells. A source cell applies its Arrow transform, chooses `stay` or a legal
  ring neighbor from current token/context/occupancy state, and transports
  ownership, hidden state, and node passport together. Final ownership replaces
  static population voting.

Gator-S begins with uniform ownership over the complete cell population. After
a configurable dense warmup, an adaptive nucleus retains the smallest cell set
covering a probability-mass target. The target is elastic: normalized handoff
entropy raises retained mass toward one, so an uncertain route can widen again
instead of being trapped in a permanently shrinking fixed `top-k`. This is
currently an uncertainty-based *proxy* for novelty; semantic novelty detection
has not been established.

One or more low-mass scout routes are retained outside the exploitation
nucleus. The scout makes pruning non-absorbing: a familiar path can stay sharp
while an alternate steward continues probing. Every steward has its own
elastic handoff beam. When it is uncertain, it retains `X` of its five legal
exits; every receiver then performs its own Arrow computation and can hand the
branch onward at the next relay. Maximum live-steward entropy controls global
reopening, while mean entropy supplies the smoother training diagnostic.
The selected formation objective supplies the end-task learning signal, so
paths that help the final simultaneous response gain routing probability. The
aligned objectives remain language-shell experiments rather than changes to
stewardship semantics.

Add the following arguments to the clean language command for the first
capacity-matched Gator-S experiment:

```bash
  --cell-transport-mode stewardship \
  --stewardship-keep-mass 0.9 \
  --stewardship-handoff-keep-mass 0.9 \
  --stewardship-min-active 1 --stewardship-max-active 0 \
  --stewardship-exploration 0.01 --stewardship-scouts 1 \
  --stewardship-execution packed_auto \
  --packed-stewardship-backend persistent_bucketed \
  --stewardship-packed-threshold 0.5 \
  --stewardship-execution-check-every 100 \
  --receiver-memory-backend convolution \
  --stewardship-warmup-fraction 0.05 \
  --stewardship-temperature-start 1.5 \
  --stewardship-temperature-end 0.35 \
  --stewardship-familiar-entropy-penalty 0.001 \
  --stewardship-balance-penalty 0.01
```

| Argument | Meaning |
|---|---|
| `--cell-transport-mode` | Select legacy broadcast or Gator-S semantic transport |
| `--stewardship-keep-mass` | Base nucleus mass for confident/familiar routes |
| `--stewardship-handoff-keep-mass` | Base local action-nucleus mass for each steward |
| `--stewardship-min-active` | Minimum retained cells per token |
| `--stewardship-max-active` | Maximum retained cells; CLI value `0` permits all cells |
| `--stewardship-exploration` | Uniform probability floor across each source cell's legal handoffs |
| `--stewardship-scouts` | Alternate branches retained outside the exploitation nucleus |
| `--stewardship-execution` | Dense numerical reference or automatic live-route packing |
| `--packed-stewardship-backend` | `persistent_bucketed` (default) or legacy `repack` forward oracle |
| `--stewardship-packed-threshold` | Maximum live branch fraction at which packed relays activate |
| `--stewardship-execution-check-every` | Steps between scheduled route-density observations; no relay-loop host checks |
| `--receiver-memory-backend` | Quadratic `attention` oracle or exact normalized `convolution` for dense receiver memory |
| `--stewardship-warmup-fraction` | Fraction of the run before hard-forward nucleus pruning begins |
| `--stewardship-temperature-start/end` | Geometric handoff-temperature schedule across the run |
| `--stewardship-familiar-entropy-penalty` | Late sharpening pressure weighted by one minus route uncertainty |
| `--stewardship-balance-penalty` | Early batch-marginal load-balance pressure; decays to zero |

Stewardship logs report support and entropy-effective cells for the initial
state and every relay, then uncertainty, widening, narrowing, stay rate, load
coefficient of variation, local handoff-beam width, dead-cell fraction, and packed-kernel use for each handoff. Training
also prints the familiar-route entropy and load-balance auxiliary losses. These
diagnostics describe semantic routing. The packed flag reports dispatcher
selection, not measured FLOPs.

The packed dispatcher carries only active `(batch, token, cell, state,
passport, mass)` records and executes Arrow transforms only for live branches.
The default persistent backend packs once at the selected boundary and keeps
the ragged representation across the complete sparse suffix. Uncertainty,
adaptive ownership pruning, handoff support, transport, collision merging, and
final readout all operate directly on packed rows. Cell-local grouped
projections gather only live cell parameter slices, while support-only
transport expands retained exits instead of all legal actions.

`packed_auto` uses a host-side controller at the configured step interval to
select the earliest relay whose complete suffix remains below the live-support
threshold. Before this plan is latched, complete dense exploration calls can
remain compiled. After latching, a packed model call is eager in full,
including any dense relays before the packed boundary; the trainer does not yet
stage a compiled prefix and eager suffix within one call. There are no
route-support decisions inside the relay loop, and the plan is checkpointed.

Dense and packed forward results have an explicit numerical-equivalence test.
Straight-through pruning makes their backward estimators different: packed
execution removes zero-forward branches that can still carry surrogate
gradients in the dense graph. Treat a sparse-training transition as an
execution-policy choice and compare learning curves as well as forward values.
Receiver groups now use fixed capacity buckets. For `M` total live rows and
group lengths `L_g`, the power-of-two bucket plan has row capacity below `2M`
and pairwise work below `4 * sum_g(L_g^2)`. The legacy `repack` backend remains
the globally padded numerical oracle. These are structural bounds, not a
performance claim: dynamic `nonzero`, sorting, and `unique` remain eager and
may synchronize. Benchmark before claiming a GPU speed or memory advantage.
Checkpoint
selection always uses fixed final-temperature, pruned routing. Left-padded
context slots carry no stewardship mass. `PackedLanguageSource` is a data-window
storage class and is unrelated to packed route execution.

Initial ownership is uniform across all cells and pruning is intentionally
disabled during warmup, so packed savings begin only after the router learns
sparse support. The node passport is also still soft: an active cell evaluates
all four node transforms. `benchmark_sparse_stewardship.py` compares
forward-only execution of the two eager sparse backends over forced supports;
its tiny CPU run is a harness and correctness smoke, not GPU or end-to-end
evidence. The current local validation
passes 145 tests plus scratch and exact data-pipeline resume trainer smokes.

Ordinary `route_loss` diagnostics reduce entropy, per-cell load, uncertainty,
and active fraction without retaining a dense ownership history. Evaluation
and full trace modes still materialize ownership when explicitly requested.

The public `StewardshipExecutionController` and
`StewardshipExecutionPlan` APIs live in `gator.execution`. Model calls accept a
static `stewardship_packed_from_relay` value. Training requests only
`diagnostics="route_loss"`; evaluation uses `route_metrics`; the legacy
`return_diagnostics=True` contract still requests the complete trace.

Dense Arrow reads default to `--receiver-memory-backend convolution`. Relative
node bias depends only on receiver/writer displacement, so the normalized read
is exactly a grouped one-dimensional correlation:

```text
memory = convolution(relative_kernel, writer_mass * node_state)
       / convolution(relative_kernel, writer_mass)
```

This removes the dense `[batch, cells, receiver, writer, nodes]` score and
softmax tensors without changing trainable parameters or the learned equation.
`attention` remains available as the quadratic semantic oracle. Use
`benchmark_receiver_memory.py` to compare forward time, peak CUDA allocation,
and numerical difference on the target GPU. Truly sparse packed groups retain
their segmented attention path until convolution's sparse break-even point is
measured.

Dense Arrow node computation defaults to
`--dense-node-backend folded_bmm`. For each cell and node, the exact identity

```text
W(base + node_embedding) + bias
    = W(base) + (W(node_embedding) + bias)
```

folds four width-8-to-width-8 transforms into one width-8-to-width-32
projection. The node-dependent offset is computed once per forward with live
parameters, so gradients and parameters are unchanged, while the dense
`[B,C,T,N,D]` node-input allocation disappears. At
`B=32,C=62,T=104,N=4,D=8`, that avoided tensor is 25.2 MiB of FP32 values per
relay.

Use these independent execution controls:

| Flag | Purpose and default |
|---|---|
| `--dense-node-backend` | Exact expanded `einsum` oracle or `folded_bmm`; default `folded_bmm` |
| `--dense-router-backend` | Router `einsum` or grouped BMM; default `einsum` |
| `--dense-kernel-padding` | Parameter-free inner-dimension multiple; default `1` (no padding) |

`benchmark_dense_kernels.py` checks numerical agreement and measures router,
node, and shared handoff kernels. On the local CPU at
`B=4,C=62,T=104,N=4,D=8`, folded node execution measured 2.6216 ms versus
9.9242 ms expanded (3.79x). Grouped router BMM was only 1.13x faster. Padding
the narrow handoff to eight measured 3.0146 ms versus 1.1743 ms at padding
one, so router BMM remains opt-in and padding remains disabled by default.
These are isolated forward-kernel CPU measurements, not end-to-end or GPU
speed claims.

For a strict memorization diagnostic, restrict SmolTalk training to one fixed,
seeded subset:

```bash
python -u gator_language_train.py \
  --data training_data/smoltalk_everyday.jsonl \
  --repeat-training-examples 500 \
  --checkpoint-dir experimental_checkpoints/language_repeat500 \
  --steps 10000 --batch-size 32 --eval-batch-size 64 --log-every 500 \
  --vocab-size 2048 --context-tokens 64 --response-tokens 32 \
  --embedding-dim 32 --d-model 8 --cells 62 --lexical-mode communal \
  --cell-input-mode feature_gated --cell-output-mode grouped_concat \
  --nodes 4 --node-topology hybrid_hemispheric --relays 4 --neighbors 4 \
  --skip-first-assistant --scheduler cosine --minimum-lr 0.0001 \
  --learning-rate 0.003 --compile
```

The subset is selected deterministically from the SmolTalk training split and
sampled repeatedly with replacement. LMSYS mixing is rejected in this mode so
the pool remains exact. Every log evaluates all selected examples and prints a
`train_subset` line; `best_train.pt` preserves the lowest subset-loss state.
The ordinary validation set remains unchanged.

## Matched one-shot Transformer baseline

The same entrypoint can replace Gator with a non-autoregressive Transformer
without changing tokenization, filtered corpora, batches, simultaneous target
window, loss, optimizer, or evaluation:

```bash
python -u gator_language_train.py \
  --architecture transformer \
  --data training_data/smoltalk_everyday.jsonl \
  --lmsys-data training_data/lmsys_chat_1m.jsonl \
  --lmsys-weight 0.9 --lmsys-load-mode stream --lmsys-shuffle-buffer 16384 \
  --lmsys-max-characters 512 --lmsys-min-ascii-letter-ratio 0.9 \
  --lmsys-min-response-characters 8 --lmsys-require-user-prompt \
  --require-complete-context --trim-context-turns --deduplicate-examples \
  --steps 100000 --batch-size 32 --eval-batch-size 64 --log-every 250 \
  --vocab-size 2048 --context-tokens 64 --response-tokens 32 \
  --d-model 38 --transformer-heads 2 --transformer-ff-width 67 \
  --transformer-layers 4 --transformer-dropout 0 \
  --skip-first-assistant --scheduler cosine --minimum-lr 0.0001 \
  --learning-rate 0.003 --seed 0
```

This configuration has 130,072 parameters. It concatenates 64 context states
and 32 learned empty output queries, processes all 96 positions with a fully
noncausal encoder, and reads all 32 output logits simultaneously. Correct
response tokens are never supplied or fed back. The token embedding/output
matrix is tied, matching Gator's parameter-saving language interface.

## Module status

- Implemented: `gator.models`, `gator.tasks`, `gator.alignment`, and the generic
  optimizer/training engine.
- Implemented Convergent primitives: fixed-cap packed frontiers, packed
  occupancy and transport, exact matrix/direct-convolution/FFT receiver
  memory, uncertainty-adaptive route width, shared-trunk private node
  adapters, and hierarchical learned ingress.
- Experimental only: active-row AdamW. Exact mode is a dense correctness
  oracle; lazy mode changes update semantics and neither mode removes dense
  moment storage.
- Integrated in the language runner: versioned checkpoints, deterministic
  evaluation, aligned formation metrics, sequence-quality objectives, and route
  diagnostics, plus opt-in packed Convergent routing and memory backends.
- Benchmark-only evidence: `benchmark_convergent_routing.py` separates
  forward, backward, optimizer, route-state, and memory measurements. Local
  CPU screens are harness checks, not target-GPU speed claims.
- Remaining extraction work: reusable standalone evaluation, checkpointing, and
  diagnostics modules shared by every future task/API.
