"""Measure whether terminal streams form a complementary coalition.

The reference architecture does not settle on a single winning trace. On a
4,096-example probe its mastered checkpoint had three material lanes at
40.6/29.9/29.1% fusion weight, yet the best lane alone scored 33.25%, the
best pair 82.65%, and all three 99.27%
(``Papers/GATOR_PATH_DESIGN_SPEC.md``). No lane was a complete solver; the
answer came from refinement across complementary traces.

"Retains primary/corrector fusion synergy" is an acceptance criterion for
GTR-L, so it needs measuring rather than assuming. Vote concentration alone
does not establish it -- a stream can carry large weight and contribute
little -- so this ablates subsets and reports what each actually buys.
"""

from __future__ import annotations

from itertools import combinations

import torch
from torch import Tensor


def effective_voters(weights: Tensor) -> tuple[float, float]:
    """Return entropy-effective and concentration-effective voter counts.

    ``K_H = exp(-sum w log w)`` and ``K_2 = 1 / sum w^2``. These describe how
    concentrated the vote is, not how much each voter contributes.
    """

    mean = weights.reshape(-1, weights.shape[-1]).mean(0)
    mean = mean / mean.sum().clamp_min(1e-9)
    entropy = -(mean.clamp_min(1e-12).log() * mean).sum()
    return float(entropy.exp()), float(1.0 / mean.pow(2).sum())


def coalition_report(
    per_stream_logits: Tensor,
    weights: Tensor,
    targets: Tensor,
    *,
    max_subset: int = 3,
) -> dict[str, object]:
    """Ablate stream subsets and report what each combination achieves.

    Args:
        per_stream_logits: ``[B, T, S, V]`` logits from each stream alone.
        weights: ``[B, L, S]`` fusion weights.
        targets: ``[B, T]`` target token ids.
        max_subset: largest subset size to enumerate.

    Returns:
        Full-coalition accuracy, the best subset at each size, per-stream
        weights, and effective voter counts.
    """

    if per_stream_logits.ndim != 4:
        raise ValueError("per_stream_logits must be [batch, slots, streams, vocab]")
    streams = per_stream_logits.shape[2]
    slots = per_stream_logits.shape[1]
    # Align fusion weights to the scored slots.
    slot_weights = weights[:, -slots:]

    def accuracy(indices: tuple[int, ...]) -> float:
        selected = per_stream_logits[:, :, indices, :]
        subset = slot_weights[:, :, indices]
        subset = subset / subset.sum(-1, keepdim=True).clamp_min(1e-9)
        fused = (selected * subset.unsqueeze(-1)).sum(2)
        return float(fused.argmax(-1).eq(targets).float().mean())

    full = accuracy(tuple(range(streams)))
    mean_weights = slot_weights.reshape(-1, streams).mean(0)
    order = torch.argsort(mean_weights, descending=True).tolist()

    # Per-stream solo accuracy. The diagnostic is the spread: streams scoring
    # identically are one voter duplicated, whatever the shared level. The
    # reference's ~33% is where its solution landed, not a requirement.
    solo = torch.stack([
        per_stream_logits[:, :, index, :].argmax(-1).eq(targets).float().mean()
        for index in range(streams)
    ])

    best_by_size: dict[int, dict[str, object]] = {}
    for size in range(1, min(max_subset, streams) + 1):
        # Enumerating all subsets is exponential; restrict to the highest
        # weighted streams, which is where the reference found its coalition.
        pool = order[: min(streams, max(4, size + 2))]
        best = max(
            (( accuracy(combo), combo) for combo in combinations(pool, size)),
            key=lambda item: item[0],
        )
        best_by_size[size] = {"accuracy": best[0], "streams": list(best[1])}

    entropy_voters, concentration_voters = effective_voters(slot_weights)
    return {
        "full_accuracy": full,
        "best_by_size": best_by_size,
        "mean_weights": [float(w) for w in mean_weights],
        "effective_voters_entropy": entropy_voters,
        "effective_voters_concentration": concentration_voters,
        # Complementary if the whole materially exceeds its best single part.
        "complementarity_gain": full - best_by_size[1]["accuracy"],
        "solo_accuracies": [float(value) for value in solo],
        # Zero spread means the streams are interchangeable -- the collapse
        # signature, independent of what level they all sit at.
        "solo_spread": float(solo.std()),
        "solo_range": (float(solo.min()), float(solo.max())),
    }


def format_report(report: dict[str, object]) -> str:
    """Render a coalition report as a compact block."""

    lines = [
        f"full coalition accuracy : {report['full_accuracy']:.4f}",
    ]
    for size, entry in sorted(report["best_by_size"].items()):
        streams = ",".join(str(s) for s in entry["streams"])
        lines.append(
            f"  best {size}-stream subset : {entry['accuracy']:.4f}  "
            f"(streams {streams})"
        )
    lines.append(
        f"complementarity gain    : {report['complementarity_gain']:+.4f} "
        f"(full minus best single stream)"
    )
    lines.append(
        f"effective voters        : K_H={report['effective_voters_entropy']:.2f} "
        f"K_2={report['effective_voters_concentration']:.2f}"
    )
    low, high = report["solo_range"]
    lines.append(
        f"solo accuracy spread    : {low:.4f} .. {high:.4f}  "
        f"std={report['solo_spread']:.4f}  "
        f"({'differentiated' if report['solo_spread'] > 0.01 else 'COLLAPSED: streams interchangeable'})"
    )
    weights = " ".join(f"{w:.3f}" for w in report["mean_weights"])
    lines.append(f"mean fusion weights     : {weights}")
    return "\n".join(lines)
