"""Surprise feature computation for SGBM (design spec sections 7-9, 11 combination).

Every function here takes already-computed summary/state tensors (block
summaries, stream states, route distributions, candidate keys) and produces
a scalar-per-batch-lane surprise feature. Nothing in this module knows how
those summaries were produced — that is Phase D's ``block_predictor.py``/
``block_summarizer.py``, and eventually the real GTR-L trace in Phase F.
Every feature stays independently loggable, per the spec's explicit
instruction to "keep them separate for logging even if they are later
combined by a learned controller."
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "predictive_surprise",
    "stream_disagreement",
    "route_surprise",
    "update_route_baseline",
    "semantic_novelty",
    "context_continuity",
    "SurpriseNormalizer",
    "surprise_momentum_update",
    "SurpriseCombinationWeights",
    "combine_surprise",
]


def predictive_surprise(
    predicted_summary: Tensor,
    actual_summary: Tensor,
    *,
    cosine_weight: float = 0.25,
    huber_delta: float = 1.0,
) -> Tensor:
    """``S_prediction = Huber(z_hat, z) + cosine_weight * (1 - cos(z_hat, z))``.

    Design spec section 7.1. ``actual_summary`` should already be detached
    by the caller (spec section 6: "the actual summary is detached when
    used as the target for the predictor").
    """

    huber = F.huber_loss(predicted_summary, actual_summary, delta=huber_delta, reduction="none")
    huber_per_example = huber.mean(dim=-1)
    cosine_term = 1.0 - F.cosine_similarity(predicted_summary, actual_summary, dim=-1, eps=1e-8)
    return huber_per_example + cosine_weight * cosine_term


def stream_disagreement(stream_states: Tensor) -> Tensor:
    """``S_streams = mean_s ||h_s - h_bar||^2 / (S * Ds)``.

    ``stream_states``: ``[B, S, Ds]``. High disagreement means the parallel
    work streams interpreted the block differently — design spec section
    7.2. This deliberately does not project through the vocabulary head.
    """

    mean_state = stream_states.mean(dim=1, keepdim=True)
    squared_deviation = (stream_states - mean_state).pow(2)
    return squared_deviation.mean(dim=(1, 2))


def route_surprise(route_distribution: Tensor, route_baseline: Tensor, *, eps: float = 1e-8) -> Tensor:
    """``S_route = mean_r KL(q_t^(r) || qbar^(r))``.

    ``route_distribution``/``route_baseline``: ``[B, R, N]`` per-relay node
    occupancy distributions (design spec section 7.3). Both are expected to
    already sum to 1 along the last dimension; a small floor keeps the KL
    term finite when either distribution has near-zero mass on a node.
    """

    p = route_distribution.clamp_min(eps)
    q = route_baseline.clamp_min(eps)
    kl_per_relay = (p * (p.log() - q.log())).sum(dim=-1)
    return kl_per_relay.mean(dim=-1)


def update_route_baseline(route_baseline: Tensor, route_distribution: Tensor, *, decay: float) -> Tensor:
    """EMA update for the running route baseline used by :func:`route_surprise`."""

    if not 0.0 < decay < 1.0:
        raise ValueError("decay must be in (0, 1)")
    return decay * route_baseline + (1.0 - decay) * route_distribution


def semantic_novelty(candidate_key: Tensor, memory_keys: Tensor, valid_mask: Tensor) -> Tensor:
    """``S_novel = 1 - max_j cos(k_t, K_j)`` over valid memory slots.

    Novelty is exactly 1 when no memory slot is valid yet (design spec
    section 7.4 — "If no memory slots are valid, novelty is 1").
    ``candidate_key``: ``[B, K]``; ``memory_keys``: ``[B, M, K]``;
    ``valid_mask``: ``[B, M]``.
    """

    normalized_candidate = F.normalize(candidate_key, dim=-1, eps=1e-8)
    normalized_keys = F.normalize(memory_keys, dim=-1, eps=1e-8)
    similarity = torch.einsum("bk,bmk->bm", normalized_candidate, normalized_keys)
    similarity = similarity.masked_fill(~valid_mask, -2.0)
    has_valid = valid_mask.any(dim=1)
    best_similarity = similarity.max(dim=1).values
    novelty = 1.0 - best_similarity
    return torch.where(has_valid, novelty, torch.ones_like(novelty))


def context_continuity(previous_summary: Tensor, current_summary: Tensor) -> Tensor:
    """``c_t = cos(z_{t-1}, z_t)`` (design spec section 7.6)."""

    return F.cosine_similarity(previous_summary, current_summary, dim=-1, eps=1e-8)


class SurpriseNormalizer:
    """Per-feature EMA mean/variance normalization (design spec section 9).

    One instance tracks one named feature (or a whole feature vector) across
    calls to :meth:`normalize`. Statistics are updated in-place under
    ``torch.no_grad()``; callers control when *not* to update (e.g. for
    padding-only or invalid blocks) simply by not calling
    :meth:`normalize` with ``update=True`` for those examples.
    """

    def __init__(self, *, decay: float = 0.99, eps: float = 1e-6, clamp: float = 5.0) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        self.decay = decay
        self.eps = eps
        self.clamp = clamp
        self._mean: Tensor | None = None
        self._var: Tensor | None = None

    @property
    def initialized(self) -> bool:
        return self._mean is not None

    @torch.no_grad()
    def normalize(self, x: Tensor, *, update: bool = True) -> Tensor:
        if self._mean is None:
            self._mean = x.detach().mean(dim=0, keepdim=True).expand_as(x[:1]).clone()
            self._var = torch.ones_like(self._mean)
        if update:
            batch_mean = x.detach().mean(dim=0, keepdim=True)
            self._mean = self.decay * self._mean + (1.0 - self.decay) * batch_mean
            batch_var = (x.detach() - self._mean).pow(2).mean(dim=0, keepdim=True)
            self._var = self.decay * self._var + (1.0 - self.decay) * batch_var
        z = (x - self._mean) / (self._var + self.eps).sqrt()
        return z.clamp(-self.clamp, self.clamp)

    def state_dict(self) -> dict[str, Tensor | None]:
        return {
            "mean": None if self._mean is None else self._mean.clone(),
            "var": None if self._var is None else self._var.clone(),
        }

    def load_state_dict(self, state: dict[str, Tensor | None]) -> None:
        mean = state["mean"]
        var = state["var"]
        self._mean = None if mean is None else mean.clone()
        self._var = None if var is None else var.clone()


def surprise_momentum_update(
    momentum: Tensor,
    momentary_surprise: Tensor,
    continuity: Tensor,
    *,
    beta: float = 0.9,
) -> Tensor:
    """``m_t = continuity * beta * m_{t-1} + (1 - beta) * s_t``.

    The design spec's "simpler first implementation" (section 8) of surprise
    momentum: high context continuity keeps momentum alive across blocks; a
    topic boundary (low continuity) lets it decay faster.
    """

    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    return continuity * beta * momentum + (1.0 - beta) * momentary_surprise


@dataclass(frozen=True)
class SurpriseCombinationWeights:
    """Default fixed combination weights, design spec section 11."""

    prediction: float = 0.45
    streams: float = 0.15
    route: float = 0.10
    novelty: float = 0.30
    momentum: float = 0.5


def combine_surprise(
    *,
    prediction: Tensor,
    streams: Tensor,
    route: Tensor,
    novelty: Tensor,
    momentum: Tensor,
    weights: SurpriseCombinationWeights | None = None,
) -> tuple[Tensor, Tensor]:
    """Fixed-weight combination into raw ``S_t`` and momentum-blended ``S~_t``.

    ``S_t = w_pred*prediction + w_streams*streams + w_route*route +
    w_novelty*novelty``; ``S~_t = w_momentum*S_t + (1-w_momentum)*m_t``
    (design spec section 11). Returns ``(S_t, S~_t)`` — both are logged
    separately per the spec.
    """

    weights = weights or SurpriseCombinationWeights()
    raw = (
        weights.prediction * prediction
        + weights.streams * streams
        + weights.route * route
        + weights.novelty * novelty
    )
    blended = weights.momentum * raw + (1.0 - weights.momentum) * momentum
    return raw, blended
