# ImageNet-1K scale readiness

This note distinguishes an **ImageNet-shaped scale attempt** from a credible
ManifoldFlow MVP result. The first proves that the pipeline can ingest,
initialize, query, and update a large representation stream. The second must
also compare fairly against simpler memories and demonstrate a useful quality,
adaptation, or systems advantage.

## Current position

The repository now has:

- frozen small-image encoders, normalization, probes, and paired decoders;
- exact basin initialization with class-value sufficient statistics;
- stable hard and soft analytic flow;
- matched centroid, VQ, prototype, and RBF baselines;
- held-out operating-point selection across three seeds;
- local Welford updates that touch only assigned basin rows;
- persistence-based novelty staging with bounded pending state;
- controlled basin creation from staged sufficient statistics;
- an override-aware immutable-main plus exact-delta index with rebuild support;
- a frozen 224px backbone/projection adapter, ImageFolder ingestion, and
  sharded/restartable latent extraction (`backbone.py`, `data.py`,
  `sharded_latents.py`), smoke-validated end to end on a synthetic
  ImageFolder via `phase_a_imagenet.py`;
- that same path validated on real ImageNet-1K data, now at 500,000 real
  train and 50,000 real validation latents, class-balanced (283-567
  examples/class), with matched-baseline accuracy approaching the frozen
  backbone's own native top-1 (`imagenet-500k-real-data.md`);
- real, well-populated (not degenerate) basin banks built and routed from
  those latents at 10k and 100k, clearing 95% recall for both scorers
  (`imagenet-500k-real-data.md`, superseding the earlier degenerate-ratio
  bank in `first-imagenet-bank.md`/`imagenet-100k-scale-and-persistence.md`);
- a robust, non-streaming path for large real train pulls
  (`imagenet_local_train_extract.py`: download whole shard files, extract
  locally, delete the raw files) after row-streaming proved rate-limit-prone
  on repeated resume against this gated dataset; and
- versioned bank/value/delta persistence with atomic publish and rollback
  (`checkpoint.py`), restart-verified at both small synthetic scale and real
  ~100k-basin scale, before and after the bank was rebuilt
  (`imagenet-500k-real-data.md`); and
- a consolidated top-1/top-5, VQ/RBF-matched, routing-recall,
  latency-and-memory evaluation at real 100k scale
  (`imagenet-scale-evaluation.md`): linear probe wins outright; ManifoldFlow's
  bank is weakest on accuracy and largest by memory, but its ANN-routed
  lookup is 37x faster than its own exact lookup at 99.95% recall.

The new streaming core is correctness-first. It has unit and synthetic non-IID
coverage, but it has not yet passed a real Fashion-MNIST/CIFAR-10 continual
stream experiment.

## Remaining gates

| Area | Current state | Required before a credible ImageNet attempt |
|---|---|---|
| Continual adaptation | Core updater and novelty lifecycle implemented | Ordered/non-IID small-image runs with adaptation, forgetting, growth, and delta-size metrics |
| Retrieval | ANN main index plus exact delta implemented; Euclidean/RBF clears 95% recall (M=32) with no exceptions on every real/synthetic bank tested, 10k through 1M, including well-populated real ImageNet banks at 10k and 100k (99.9% each). Energy clears too, but its required candidate width varies with each bank's genuine rate heterogeneity (real, non-degenerate ImageNet 10k needs M=512 for 98%; 100k needs M=256 for 97.65%; the real Fashion-MNIST 10k bank needed ~8x the synthetic study's width) - not predictable from real-vs-synthetic, scale, or basin count alone, only by measuring the bank (`first-imagenet-bank.md`, `imagenet-500k-real-data.md`) | None outstanding for Euclidean/RBF at any scale tested; energy at real 1M is untested (only synthetic 1M exists) |
| Persistence | Versioned bank/value/delta serialization (`checkpoint.py`) with atomic publish and rollback, validated by restart verification at both small synthetic scale (`test_checkpoint.py`) and real 100k-basin scale (99,967 basins, 1.2GB checkpoint, bit-identical restart, `imagenet-100k-scale-and-persistence.md`) | None for the persistence substrate; full merge/split/prune consolidation (§5) remains explicitly Phase D and out of scope for a static attempt |
| Representation | Frozen 224px ResNet-18 backbone plus a fixed (fit-once, then frozen) PCA projection to `d=256`, validated end to end on real ImageNet-1K: 100,000 train and 50,000 validation latents, matched baselines in the healthy range for a frozen/dimension-reduced backbone (`first-imagenet-representation.md`) | None for a first static pass; a real (not projected/reduced) top-5 number would need the full 512-d features or a larger projection |
| Data | ImageFolder ingestion (`data.py`) plus sharded/restartable latent extraction with verified resume at the file level, including a real crash-recovery test (`sharded_latents.py`). Real ImageNet-1K validation split (6.3GB) downloaded and fully encoded locally. Row-streaming (`imagenet_stream_extract.py`) works but proved rate-limit-exposed on repeated `skip()` calls against this gated dataset's CDN (`HTTP 429` on the first shard, every resume attempt, after sustained use); superseded for large train pulls by `imagenet_local_train_extract.py`, which downloads whole shard files (the same robust path already used for validation) and reads them locally with no `skip()` exposure - used to reach 500,000 real, well-balanced (283-567 examples/class) train rows in ~81 minutes with zero failures (`imagenet-500k-real-data.md`) | A full-train (1.28M-row) pass remains untried; not required for a first static attempt |
| Bank scale | Real banks built and routed at 1k/10k (Fashion-MNIST) and 10k/100k (ImageNet-1K); the 100k ImageNet bank has been rebuilt from a well-populated 500,000-row train set (50:1 and 5:1 basin:example ratios at 10k/100k, real occupancy up to 21/2 median) superseding the earlier degenerate 1:1-ratio bank (`imagenet-500k-real-data.md`); synthetic 10k/100k/1M systems runs beyond that (`ann-routing-scale.md`, `real-10k-ann-routing.md`) | Real 1M for either scorer, on either a real or synthetic bank at that scale |
| Evaluation | Top-1/top-5 classification, matched-memory VQ/RBF baselines, ANN routing recall, latency, and memory all validated together on real ImageNet-1K data at 100k bank scale (`imagenet-scale-evaluation.md`): linear probe wins outright (69.05%/87.10%, cheapest and smallest of every method tried); ManifoldFlow's own bank is the weakest classifier and largest by memory, but its ANN-routed lookup is 37x faster than its own exact lookup at 99.95% recall - the systems mechanism holds even though the learning result does not | Reconstruction/noise robustness only exercised on the small-image MVP - not required for a classification-only ImageNet entry |
| Consolidation | Not implemented | Not required for the first static scale attempt; required before claiming a long-running continual MVP |

## Recommended first configuration

The first run should be classification-only. Training a paired generative
decoder on ImageNet would add cost without testing the sparse-memory scaling
hypothesis.

- Frozen, standard small image-classification backbone at 224px.
- Fixed projection or normalization to `d=256`.
- 10,000 basins with exact routing on a bounded subset as the correctness gate.
- 100,000 basins with an ANN main index plus the exact delta index as the scale
  gate.
- Class-distribution basin values; no reconstruction decoder.
- Hard/no-flow, RBF, nearest-prototype, VQ, and centroid controls at matched
  vector-value budgets.
- Static validation first, then an ordered class/domain stream only after the
  static routing and memory numbers pass.

One million replicated or synthetic basins remains a systems benchmark, not an
ImageNet learning target.

## Distance estimate

Assuming the ImageNet data and suitable compute are already available:

- **ImageNet-shaped smoke attempt:** about four focused milestones remain:
  real non-IID small-image validation, backbone/data adapters, sharded latent
  extraction, and ANN-backed 10k/100k bank runs. This is roughly one to two
  focused engineering weeks, not a research-quality result.
- **Credible classification MVP:** about six to eight milestones remain once
  ANN recall/latency, restartable state, matched baselines, hardware accounting,
  and repeatability are included. A realistic range is three to five focused
  weeks plus experiment runtime.
- **Full continual ImageNet claim:** farther away. It additionally needs safe
  consolidation, rollback, rare-class retention tests, and evidence that local
  updates beat online prototype or mixture baselines.

In readiness terms, the project is approximately halfway to an ImageNet-shaped
smoke test and roughly one-third of the way to a credible ImageNet-scale MVP.
Those fractions describe engineering coverage, not confidence that analytic
flow will beat the baselines. Static experiments so far still favor the simpler
RBF/prototype transforms.

## Immediate next gate

Every row in the table above except Evaluation and Consolidation is now
validated on real ImageNet-1K data, at real 10k and 100k bank scale, on a
well-populated (not degenerate) train set, and combined end to end: a basin
bank built from 500,000 real, class-balanced train latents, routed with both
scorers, checkpointed and restart-verified (`imagenet-500k-real-data.md`).

Euclidean/RBF clears 95% recall (M=32) with no exceptions on every bank
tested, real or synthetic, 10k through 1M. Anisotropic energy also clears
95% on every real ImageNet bank tested (10k needs M=512, 100k needs M=256)
and on synthetic 100k, but needed ~8x the candidate width on the real
Fashion-MNIST 10k bank specifically. An earlier pass through this same
question found the (since-superseded) degenerate 100k ImageNet bank hit
100% recall trivially at M=32 and read that as further evidence energy
routing is easy on real data; rebuilding that bank with genuine occupancy
reversed the reading - the degenerate bank's near-uniform basin rates were
themselves the artifact, not evidence about ImageNet geometry. The
standing conclusion is unchanged, just now on cleaner evidence: energy's
required candidate width tracks each bank's actual rate heterogeneity and
is not predictable from real-vs-synthetic, scale, or basin count alone -
only by measuring the bank.

Persistence closed the same way and was re-verified against the corrected
bank: `checkpoint.py` provides versioned bank/value/delta serialization with
atomic publish and rollback, restart-verified at real ~100k-basin scale
(99,968 basins, 1.2GB checkpoint, bit-identical restart in ~5-6s to publish
and load) both before and after the bank was rebuilt.

The streaming ingestion path (`imagenet_stream_extract.py`) also had a real
reliability problem beyond the process-exit hang: repeated `skip()` calls
against this gated dataset's CDN eventually draw `HTTP 429` rate-limiting on
the first shard, on every resume attempt, since `skip()` re-reads every
already-consumed shard from scratch each time. The 500,000-row pull above
used a different script instead - `imagenet_local_train_extract.py`
downloads whole shard files (no `skip()`, no streaming) the same way the
validation split was already downloaded, and completed with zero failures
in ~81 minutes. That script is now the recommended path for any future
large real train pull; `imagenet_stream_extract.py` remains correct and
useful for smaller, bounded pulls where its one-shot streaming (not
resumed) has not shown this failure mode.

Evaluation at scale is also now closed: top-1/top-5, matched-memory VQ and
RBF baselines, ANN routing recall, latency, and memory were all reported
together for the real 100k bank (`imagenet-scale-evaluation.md`). Building
it surfaced one more real fix - the ImageNet banks had never been whitened
before initialization, unlike every Fashion-MNIST bank in this project;
both were rebuilt on whitened latents first. The result itself lands where
`matched-online-controls.md` already pointed: a linear probe beats every
non-parametric memory tried (69.05%/87.10% top-1/top-5, and cheapest by
three orders of magnitude), and ManifoldFlow's own bank is the weakest
classifier and largest by memory of the group. What the evaluation adds
beyond confirming that: routing through ANN instead of exact search takes
ManifoldFlow's own lookup from 37.5ms to 1.0ms at 99.95% recall - the
sparse, bounded-compute systems mechanism works exactly as designed, it
just does not change which method gives the better answer.

What is left before a credible (not just scale-shaped) attempt:

- **Real bank scale at 1M** - only checked on synthetic banks so far
  (`ann-routing-scale.md`); a real 1M-basin bank would need extending the
  train pull well past 500,000 rows.

Nothing about routing, persistence, representation, or evaluation itself
blocks starting the recommended first configuration below at real 100k
scale, routed on Euclidean/RBF - the only open item is real bank scale
beyond 100k, and the learning-quality question is answered (negatively for
the attractor formulation), not open.
