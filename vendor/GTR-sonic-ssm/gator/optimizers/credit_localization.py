"""Credit localization: map surprise features to eligible parameter groups
(Procoim-S design spec section 8.1-8.2).

Now that SGBM (Phases A-G) provides real per-lane surprise features —
``predictive_surprise``, ``stream_disagreement``, ``route_surprise``,
``semantic_novelty`` (``gator/memory/surprise.py``) — this module maps
those directly onto Procoim's six parameter groups (``routing``,
``receiver_memory``, ``stream_mix``, ``node_compute``, ``readout``,
``embedding``) per the spec's qualitative table, rather than guessing at
signals that don't exist yet.

The mapping below is a v1 simplification of section 8.1's table: it does
not yet distinguish "receiver mismatch" or "correct concepts but bad final
placement" as separate diagnostics (those need deeper semantic probes this
repo does not build), and folds them into the closest available SGBM
signal instead. Documented here rather than silently overclaiming fidelity
to the full table.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "SurpriseFeatures",
    "localize_groups",
    "GroupEligibilityTrace",
]


@dataclass(frozen=True)
class SurpriseFeatures:
    """The subset of SGBM's surprise decomposition credit localization uses."""

    route_surprise: float
    stream_disagreement: float
    predictive_surprise: float
    semantic_novelty: float


#: Section 8.1's mapping, approximated with the SGBM signals actually
#: available: route surprise -> routing/stream_mix; stream disagreement
#: (the closest available proxy for "receiver mismatch") -> receiver_memory;
#: predictive surprise (semantic/expectation error) -> node_compute/readout;
#: semantic novelty (lexical/vocabulary mismatch) -> embedding.
_GROUP_WEIGHTS: dict[str, dict[str, float]] = {
    "routing": {"route_surprise": 1.0},
    "stream_mix": {"route_surprise": 0.5, "stream_disagreement": 0.5},
    "receiver_memory": {"stream_disagreement": 1.0},
    "node_compute": {"predictive_surprise": 1.0},
    "readout": {"predictive_surprise": 0.5},
    "embedding": {"semantic_novelty": 1.0},
}


def localize_groups(
    features: SurpriseFeatures,
    *,
    top_k: int = 2,
    threshold: float = 0.0,
) -> tuple[str, ...]:
    """Rank Procoim parameter groups by how likely they are responsible for
    the observed surprise, returning the top ``top_k`` above ``threshold``.

    Always returns at least one group (the single highest-scoring one) even
    if every score is at or below ``threshold``, so a ticket always has
    somewhere to search rather than an empty eligible-groups tuple.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    feature_values = {
        "route_surprise": features.route_surprise,
        "stream_disagreement": features.stream_disagreement,
        "predictive_surprise": features.predictive_surprise,
        "semantic_novelty": features.semantic_novelty,
    }
    scores = {
        group: sum(weight * feature_values[name] for name, weight in weights.items())
        for group, weights in _GROUP_WEIGHTS.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    selected = tuple(group for group, score in ranked[:top_k] if score > threshold)
    return selected if selected else (ranked[0][0],)


class GroupEligibilityTrace:
    """Per-group activity EMA feeding the ticket-priority formula (section 8.2:
    ``P_g = E_weight * eligibility_g * plasticity_g``).

    Callers feed in whatever cheap per-group activity signal is available
    (route participation, mailbox write/read magnitude, node occupancy,
    candidate sensitivity history, ...) — this class only tracks the
    running EMA and combines it into a priority score; it does not itself
    compute activity from a GTR-L trace.
    """

    def __init__(self, *, decay: float = 0.9) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = decay
        self._activity: dict[str, float] = {}

    def update(self, group: str, activity: float) -> None:
        previous = self._activity.get(group, activity)
        self._activity[group] = self.decay * previous + (1.0 - self.decay) * activity

    def eligibility(self, group: str) -> float:
        return self._activity.get(group, 0.0)

    def priority(
        self,
        group: str,
        weight_eligibility: float,
        *,
        plasticity: float = 1.0,
    ) -> float:
        """``P_g = E_weight * eligibility_g * plasticity_g`` (section 8.2)."""

        return weight_eligibility * self.eligibility(group) * plasticity

    def rank_groups(
        self,
        groups: tuple[str, ...],
        weight_eligibility: float,
        *,
        plasticity: dict[str, float] | None = None,
    ) -> tuple[str, ...]:
        """Groups sorted by priority, highest first — "only the top few
        groups receive candidate banks" (section 8.2)."""

        plasticity = plasticity or {}
        scored = [
            (group, self.priority(group, weight_eligibility, plasticity=plasticity.get(group, 1.0)))
            for group in groups
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return tuple(group for group, _ in scored)
