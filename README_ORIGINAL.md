# ManifoldFlow

ManifoldFlow is a geometry-first sparse attractor memory research prototype. A
compact encoder maps an input into a normalized latent, sparse retrieval selects
a bounded candidate set, and an exact local flow moves the latent through a bank
of stable quadratic attractors. Only retrieved basin records participate in a
query or online update.

The central hypothesis is deliberately falsifiable:

> A fixed compact backbone plus a growing sparse attractor bank can increase
> memory capacity while keeping per-query bank compute approximately constant,
> while local basin updates reduce interference during continual learning.

This repository starts with the correctness-first core from design v0.2:

- diagonal positive contraction rates;
- exact hard and frozen-gate soft top-K flows;
- exact candidate retrieval before ANN integration;
- CPU-resident basin state with Welford-style local updates;
- explicit experiment settings for 64- and 128-dimensional latents; and
- tests for stability, analytic correctness, routing, and local statistics.

## Status

Research prototype. The current implementation covers the correctness-first
mathematical kernel, Phase A small-image representation training, populated bank
initialization, normalized-latent decoder calibration, and matched clean/noisy
hard-flow and soft-flow ablations. Neutral decoder calibration, held-out
operating-point selection, and confidence-plus-distance conditional flow are
now implemented. The `d=128` results cover three independently trained models
and banks. Foreground statistical updates, bounded novelty staging, controlled
basin creation, and an exact main-plus-delta index are now implemented at the
correctness-core level. The first three-seed non-IID latent stream shows a
repeatable adaptation gain with bounded creation. The completed matched online
gate is negative for the current attractor formulation: at equal core-vector
storage, online top-4 RBF reaches 85.97% shifted accuracy versus 82.97% for
local novelty. ANN routing is now a systems-focused pivot supporting both
scores. The first HNSW integration clears 95% exact-winner recall at 1k/2k
rows, while 10k→1M scaling, versioned consolidation, and CIFAR-10 results remain
staged work. A
matched-memory comparison found that a
simple RBF contraction still outperforms analytic flow on the primary noisy
metrics; the first conditional gate reduces exposure but damages more
RBF-correct examples than it rescues. A follow-up semantic-boundary gate now
reverses that imbalance on corrupted inputs and passes the clean-quality gates,
although RBF remains the stronger overall transform.
An RBF-default hybrid preserves the semantic gate's small corrupted-input
classification gain, but increases reconstruction error and does not beat the
nearest-prototype accuracy; it remains an ablation rather than the default.

Design v0.3 now makes the frozen Qwen3.5-4B continuous-learning testbed the
governing implementation track. It reuses the proven RBF, local-update, and ANN
primitives but treats persistent language memory as unproven. Phase 0—stable
hidden-state capture, deterministic D128 keys, version manifests, and replay—is
the next implementation milestone.

## Quick start

Requires Python 3.10+ and PyTorch 2.2+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

Phase A image experiments also require torchvision:

```bash
python -m pip install -e ".[dev,vision]"
manifoldflow-phase-a \
  --dataset FashionMNIST \
  --latent-width 64 \
  --download
```

Dataset downloads are never implicit. Use `--smoke` for a two-batch pipeline
check, or omit it for the configured training run. Artifacts are written below
`artifacts/phase_a/<dataset>/d<width>/seed<seed>/` and include metrics, frozen
model/normalizer state, and normalized train/test latents.

Initialize a small exact-routing basin bank from a completed latent artifact:

```bash
manifoldflow-init-bank \
  --latents artifacts/phase_a/FashionMNIST/d64/seed17/latents.pt \
  --basins 1000 \
  --output artifacts/phase_b/FashionMNIST/d64/k1000/seed17
```

The initializer uses deterministic mini-batch K-means, bounded exact assignment,
short Lloyd refinement, and a final exact pass for counts, diagonal second
moments, calibrated inverse-variance rates, occupancy biases, and class-value
statistics. Empty basins are removed and reported rather than published; rate
ceiling saturation is an explicit diagnostic.

Calibrate the paired decoder for the bank's normalized latent space, then run a
matched `t=0`, hard-flow, and soft top-K ablation:

```bash
manifoldflow-calibrate-decoder \
  --dataset FashionMNIST \
  --checkpoint artifacts/phase_a/FashionMNIST/d64/seed17/checkpoint.pt \
  --bank artifacts/phase_b/FashionMNIST/d64/k1000/seed17/bank.pt \
  --output artifacts/flow/FashionMNIST/d64/k1000/seed17

manifoldflow-flow-ablation \
  --dataset FashionMNIST \
  --checkpoint artifacts/flow/FashionMNIST/d64/k1000/seed17/checkpoint.pt \
  --bank artifacts/phase_b/FashionMNIST/d64/k1000/seed17/bank.pt \
  --output artifacts/flow/FashionMNIST/d64/k1000/seed17/ablation
```

The calibration freezes the encoder and trains only the decoder on normalized
and gently hard-flowed latents. The ablation uses one shared calibrated decoder
and probe across every route and evaluates reconstruction and classification on
both clean and deterministically noised inputs.

Every experiment command accepts `--seed`; flow ablations also accept
`--image-noise-std` and a space-separated `--times` grid. Aggregate compatible
repeats with:

```bash
manifoldflow-aggregate-flow \
  --metrics path/to/seed17/metrics.json path/to/seed18/metrics.json \
            path/to/seed19/metrics.json \
  --output path/to/three_seed_summary.json
```

The aggregator validates the experimental protocol before calculating means,
sample standard deviations, and per-seed values.

Compare flow with 2,000-center VQ, isotropic-prototype, RBF, and class-centroid
transforms at the same core-vector storage budget:

```bash
manifoldflow-baseline-ablation \
  --dataset FashionMNIST \
  --checkpoint path/to/calibrated/checkpoint.pt \
  --flow-bank path/to/1000_basin/bank.pt \
  --prototype-bank path/to/2000_center/bank.pt \
  --latents path/to/latents.pt \
  --output artifacts/baseline_ablation \
  --seed 17
```

The command refuses unmatched center/rate vector counts.

For an unbiased flow/prototype comparison, calibrate a shared decoder across
all gentle transform families, split validation from final evaluation, and
select operating points using only validation metrics:

```bash
manifoldflow-calibrate-neutral \
  --dataset FashionMNIST \
  --checkpoint path/to/phase_a/checkpoint.pt \
  --flow-bank path/to/1000_basin/bank.pt \
  --prototype-bank path/to/2000_center/bank.pt \
  --output artifacts/neutral/FashionMNIST/d128/seed17

manifoldflow-baseline-ablation \
  --dataset FashionMNIST \
  --checkpoint artifacts/neutral/FashionMNIST/d128/seed17/checkpoint.pt \
  --flow-bank path/to/1000_basin/bank.pt \
  --prototype-bank path/to/2000_center/bank.pt \
  --latents path/to/latents.pt \
  --output artifacts/neutral_baselines/seed17/validation \
  --skip-samples 0 --max-samples 2000
```

The baseline ablation calibrates the conditional distance threshold from clean
training latents and reports activation, RBF-correct-to-wrong, and
RBF-wrong-to-correct counts. `--diagnostic-rbf-strength` selects the RBF
operating point used as the transition reference.

Add `--semantic-grid` on the validation slice to evaluate the configured
confidence, distance-quantile, and class-margin grid with one shared routing
pass. After selection, run one frozen gate with `--semantic-confidence`,
`--semantic-distance-quantile`, and `--semantic-margin-threshold`. Semantic
routing uses the top-K mixture of per-basin class distributions and refuses to
flow low-margin boundary cases.

Add `--semantic-hybrid` with a semantic grid or frozen semantic gate to retain
the diagnostic RBF transform except where that gate activates analytic flow.

Minimal use:

```python
import torch

from manifoldflow import analytic_topk_flow

z0 = torch.randn(2, 64)
mu = torch.randn(2, 4, 64)
raw_rate = torch.zeros_like(mu)
bias = torch.zeros(2, 4)

zt, energy, weight = analytic_topk_flow(z0, mu, raw_rate, bias, t=0.5, temperature=0.1)
```

## MVP order

1. Establish Fashion-MNIST and CIFAR-10 representation baselines at `d=64` and
   `d=128`.
2. Validate exact routing and hard one-basin flow.
3. Add frozen-gate soft top-K flow.
4. Validate the implemented statistical updates, novelty staging, and exact
   delta index on real ordered/non-IID streams.
5. Match exact routing with HNSW or IVF-PQ before enabling approximate routing.
6. Introduce snapshot-based merge, split, prune, quantize, validate, and atomic
   publication operations one at a time.

See [docs/design-v0.3.md](docs/design-v0.3.md) for the governing Qwen testbed
design. [Design v0.2](docs/design-v0.2.md) remains the image-research foundation,
with [configs/mvp.toml](configs/mvp.toml) retaining the initial image
experimental envelope. The
[design progress ledger](docs/design-progress.md) maps every active and pending
milestone to its design section, evidence, and next exit condition. The
[first substantive training record](docs/results/first-training-runs.md)
captures the initial Fashion-MNIST width comparison. The
[first analytic-flow ablation](docs/results/first-flow-ablation.md) records the
preliminary denoising result. The
[three-seed repeatability record](docs/results/three-seed-flow-repeatability.md)
captures the initial robustness evidence. The
[matched prototype comparison](docs/results/matched-prototype-baselines.md)
records the subsequent negative result for the current flow formulation and the
resulting research pivot. The
[neutral conditional-flow experiment](docs/results/neutral-conditional-flow.md)
removes the decoder-calibration confound, uses disjoint validation and final
slices, and records the first conditional gate's negative rescue/damage result.
The [semantic-boundary gate experiment](docs/results/semantic-boundary-gate.md)
records the first narrow positive conditional-flow result under corruption.
The [RBF-default hybrid experiment](docs/results/hybrid-flow-policy.md) shows
that the active-set classification signal repeats but does not produce a
quality-dominating inference policy, motivating the move to online updates.
The [ImageNet-1K scale-readiness note](docs/imagenet-scale-readiness.md) defines
the classification-only entry configuration and the remaining engineering and
research gates.
The [first continual-stream experiment](docs/results/first-continual-stream.md)
records the selected novelty policy, adaptation/forgetting metrics, and the
original matched-control requirement. The
[matched online control result](docs/results/matched-online-controls.md)
records the decisive three-seed RBF win and resulting ImageNet scope change.
The [first ANN routing benchmark](docs/results/first-ann-routing.md) records the
small-bank HNSW recall and latency pass and the still-open scale gate.
