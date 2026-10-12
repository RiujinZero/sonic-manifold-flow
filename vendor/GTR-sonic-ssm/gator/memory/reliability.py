"""Reliability estimation for SGBM (design spec section 10).

Surprise answers "was this unexpected?"; reliability answers "should we
trust what was formed?" Separating the two is mandatory (spec section 10.1)
— without it, memory would preferentially store confused, malformed, or
hallucinated blocks just because they were surprising.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "consensus_reliability",
    "entropy_reliability",
    "quality_reliability",
    "combined_reliability",
]


def consensus_reliability(stream_disagreement_score: Tensor) -> Tensor:
    """``R_consensus = exp(-S_streams)`` (design spec section 10)."""

    return torch.exp(-stream_disagreement_score)


def entropy_reliability(probabilities: Tensor, vocab_size: int, *, eps: float = 1e-8) -> Tensor:
    """``R_entropy = 1 - H(p) / log(vocab_size)``.

    ``probabilities``: ``[B, ..., V]`` (any number of middle dimensions,
    e.g. output-slot count — entropy is computed per slot then averaged).
    Low entropy (confident predictions) yields reliability near 1; uniform
    predictions yield reliability near 0.
    """

    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than 1")
    entropy = -(probabilities.clamp_min(eps) * probabilities.clamp_min(eps).log()).sum(dim=-1)
    normalized_entropy = entropy / math.log(vocab_size)
    reliability = 1.0 - normalized_entropy
    while reliability.ndim > 1:
        reliability = reliability.mean(dim=-1)
    return reliability


def quality_reliability(
    blank_rate: Tensor,
    repeat_rate: Tensor,
    forbidden_rate: Tensor,
) -> Tensor:
    """``R_quality = (1 - blank_rate)(1 - repeat_rate)(1 - forbidden_rate)`` (spec section 10)."""

    return (1.0 - blank_rate) * (1.0 - repeat_rate) * (1.0 - forbidden_rate)


def combined_reliability(
    consensus: Tensor,
    entropy: Tensor,
    quality: Tensor,
) -> Tensor:
    """``R_t = clip(R_consensus * R_entropy * R_quality, 0, 1)`` (spec section 10)."""

    return (consensus * entropy * quality).clamp(0.0, 1.0)
