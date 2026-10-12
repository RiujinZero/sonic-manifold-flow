"""Adaptive compatibility gate (design-v0.3.md SS6.2-SS6.3, SS9 Phase 4).

`PersistentBasinContext` (used through Phase 3) takes one fixed `alpha` for
an entire run - a single, hand-picked tradeoff between trusting the current
observation and trusting recent context. SS6.3's adaptive gate is meant to
choose that tradeoff per observation instead, from a documented table:

| Situation | Required behavior |
|---|---|
| Observation and context agree | Permit strong context bias |
| Observation is ambiguous, context confident | Permit moderate context bias |
| Observation is confident and conflicts | Raise `alpha` toward 1 |
| Novel observation or topic switch | Use `alpha=1`; weaken or fork context |
| Explicit reset/forget instruction | Clear targeted context state |

`AdaptiveBasinContext` implements this with one continuous rule rather than
the doc's two separate parameters (`alpha_t` and a multiplicative gate
`g_t` on the context term, SS6.2): binary agreement between the raw
observation's and the context's top candidate directly sets how much
`alpha` is allowed to relax toward `alpha_base`, and raw confidence scales
how strongly a *disagreement* pushes `alpha` back toward 1. This is a
documented simplification for a first implementation (SS10: "starting
points, not claimed optima"), not the literal dual-parameter formula.

## Three real evaluations, one surprising conclusion

SS6.3 says plainly: "The gate should compare raw and contextual candidate
distributions, not merely cosine similarity between two embeddings" - a
direct argument against the binary top-1 check this class actually uses.
Two attempts were made to replace it with the distributional comparison
the doc calls for, and both made results measurably *worse*, not better,
on the same real Qwen3.5-4B evaluation
(`docs/results/qwen-phase4-adaptive-gate.md`,
`qwen-phase4b-fixed-agreement.md`, `qwen-phase4c-narrowed-agreement.md`):

| Variant | Carryover (fixed -> adaptive) | Retention at N=20 (fixed -> adaptive) |
|---|---|---|
| Binary top-1 (this implementation) | 13.3% -> **83.3%** | 98.3% -> 86.7% |
| Overlap coefficient, full `candidate_k`-wide union | 13.3% -> 3.3% | 98.3% -> 38.3% |
| Overlap coefficient, narrowed to top-2 | 13.3% -> 16.7% | 98.3% -> 40.0% |

The distributional versions were not obviously wrong in theory - at real
bank scale, RBF scores are graded rather than sharply bimodal, so a wide
candidate union carries enough shared low/moderate-probability mass on
mutually irrelevant candidates that overlap stayed high even during
genuine topic switches, degrading exactly the case the gate exists to
handle; narrowing the comparison window did not fix it either. The
binary check, despite being the version SS6.3 explicitly warns is too
brittle, is empirically the best of the three tried here. This module
keeps it for that reason, not because the doc's concern was wrong in
principle - only because two attempts at the doc's own recommended fix
underperformed it on the one real evaluation available, and a third guess
without a clearer theory of why did not seem like a good use of further
real-model evaluation runs. A genuine, open, better-motivated fix (e.g. an
actually-learned or held-out-calibrated compatibility function, not a
hand-picked formula) remains future work.

Both raw and context candidates are drawn via two independent ANN/exact
queries (SS6.2's "two-pass candidate retrieval": a blended query must not
be allowed to hide candidates the current observation alone would have
found), then reranked together with an RBF score
`S_i(x) = exp(-||x-mu_i||^2 / (2*sigma^2))` - the same scoring family used
throughout this project's RBF baselines, just evaluated over the union of
two candidate sets instead of one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from manifoldflow.index import ExactIndex, HnswIndex


@dataclass(frozen=True)
class RoutingDecision:
    routed_id: int
    alpha_used: float
    raw_top_id: int
    raw_confidence: float
    context_top_id: int | None
    agreement: bool
    novel: bool


class AdaptiveBasinContext:
    """Two-pass retrieval with an adaptive alpha, replacing a fixed one."""

    def __init__(
        self,
        centers: Tensor,
        index: ExactIndex | HnswIndex,
        *,
        alpha_base: float,
        decay: float,
        rbf_bandwidth: float,
        novelty_threshold: float,
        candidate_k: int = 8,
    ) -> None:
        if not 0.0 <= alpha_base <= 1.0:
            raise ValueError("alpha_base must be in [0, 1]")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if rbf_bandwidth <= 0:
            raise ValueError("rbf_bandwidth must be positive")
        if not 0.0 <= novelty_threshold <= 1.0:
            raise ValueError("novelty_threshold must be in [0, 1]")
        if centers.ndim != 2 or centers.shape[0] == 0:
            raise ValueError("centers must have shape [basins, d] with basins > 0")
        self.centers = centers
        self.index = index
        self.alpha_base = alpha_base
        self.decay = decay
        self.rbf_bandwidth = rbf_bandwidth
        self.novelty_threshold = novelty_threshold
        self.candidate_k = min(candidate_k, centers.shape[0])
        self.context = torch.zeros(centers.shape[1], dtype=centers.dtype)
        self._has_context = False

    def reset(self) -> None:
        """Clear accumulated context (SS6.3's explicit reset/forget row)."""
        self.context = torch.zeros_like(self.context)
        self._has_context = False

    def _rbf_scores(self, query: Tensor, candidate_ids: Tensor) -> Tensor:
        candidates = self.centers[candidate_ids].to(dtype=query.dtype)
        squared_distance = (candidates - query).square().sum(dim=-1)
        return torch.exp(-squared_distance / (2 * self.rbf_bandwidth**2))

    @torch.no_grad()
    def route(self, z0: Tensor) -> RoutingDecision:
        if z0.shape != self.context.shape:
            raise ValueError("z0 must have shape [dimension]")

        raw_candidate_ids = self.index.query(z0[None, :], self.candidate_k).squeeze(0)
        raw_scores = self._rbf_scores(z0, raw_candidate_ids)
        raw_top_local = int(raw_scores.argmax().item())
        raw_top_id = int(raw_candidate_ids[raw_top_local].item())
        raw_confidence = float(raw_scores[raw_top_local].item())
        novel = raw_confidence < self.novelty_threshold

        if not self._has_context or novel:
            # SS6.3: novel observation or topic switch -> alpha=1, and the
            # context that follows is a fresh anchor, not a blend with
            # whatever was stale before ("weaken or fork").
            routed_id = raw_top_id
            alpha_used = 1.0
            context_top_id = None
            agreement = False
        else:
            context_candidate_ids = self.index.query(
                self.context[None, :], self.candidate_k
            ).squeeze(0)
            context_scores = self._rbf_scores(self.context, context_candidate_ids)
            context_top_local = int(context_scores.argmax().item())
            context_top_id = int(context_candidate_ids[context_top_local].item())
            agreement = raw_top_id == context_top_id

            disagreement = 0.0 if agreement else 1.0
            alpha_used = self.alpha_base + (
                1.0 - self.alpha_base
            ) * raw_confidence * disagreement
            alpha_used = min(1.0, max(self.alpha_base, alpha_used))

            union_ids = torch.unique(
                torch.cat([raw_candidate_ids, context_candidate_ids])
            )
            raw_union_scores = self._rbf_scores(z0, union_ids)
            context_union_scores = self._rbf_scores(self.context, union_ids)
            combined = (
                alpha_used * raw_union_scores + (1 - alpha_used) * context_union_scores
            )
            best_local = int(combined.argmax().item())
            routed_id = int(union_ids[best_local].item())

        center = self.centers[routed_id].to(dtype=self.context.dtype)
        if novel or not self._has_context:
            self.context = center
            self._has_context = True
        else:
            self.context = self.decay * self.context + (1 - self.decay) * center

        return RoutingDecision(
            routed_id=routed_id,
            alpha_used=alpha_used,
            raw_top_id=raw_top_id,
            raw_confidence=raw_confidence,
            context_top_id=context_top_id,
            agreement=agreement,
            novel=novel,
        )
