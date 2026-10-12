"""Temporal evidence accumulators for stable visual entity attributes."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class TemporalLabelEvidence:
    """Maintain a hysteretic label posterior instead of frame-wise argmax labels."""

    labels: tuple[str, ...]
    decay: float = 0.9
    minimum_evidence: float = 0.25
    establish_margin: float = 0.08
    switch_margin: float = 0.2
    scores: dict[str, float] = field(default_factory=dict)
    label: str = ""
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.labels or len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be non-empty and unique")
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError("decay must be between zero and one")
        if min(self.minimum_evidence, self.establish_margin, self.switch_margin) < 0:
            raise ValueError("evidence thresholds must be non-negative")
        if not self.scores:
            self.scores = dict.fromkeys(self.labels, 0.0)

    def update(self, probabilities: Tensor, weight: float) -> tuple[str, float]:
        if probabilities.shape != (len(self.labels),):
            raise ValueError("probabilities must have one value per label")
        if weight < 0:
            raise ValueError("weight must be non-negative")
        values = probabilities.detach().float().cpu()
        if bool((values < 0).any()) or float(values.sum()) <= 0:
            raise ValueError("probabilities must be non-negative with positive mass")
        values /= values.sum()
        for label in self.labels:
            self.scores[label] *= self.decay
        for label, probability in zip(self.labels, values.tolist(), strict=True):
            self.scores[label] += weight * probability

        ranked = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)
        total = sum(self.scores.values())
        top_label, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        self.confidence = top_score / max(total, 1e-8)
        margin = (top_score - second_score) / max(total, 1e-8)
        if not self.label:
            if total >= self.minimum_evidence and margin >= self.establish_margin:
                self.label = top_label
        elif top_label != self.label:
            current_share = self.scores[self.label] / max(total, 1e-8)
            if self.confidence - current_share >= self.switch_margin:
                self.label = top_label
        return self.label, self.confidence
