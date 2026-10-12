"""Training losses for Surprise-Gated Block Memory (design spec section 19).

Note on placement: mirrors ``episodic_stream.py`` — the spec's file layout
puts this at ``gator/training/memory_losses.py``, unavailable here since
``gator/training.py`` is already a module. Lives under ``gator/continual/``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "predictor_loss",
    "retrieval_utility",
    "write_calibration_loss",
    "contrastive_key_loss",
]


def predictor_loss(
    predicted_summary: Tensor,
    actual_summary: Tensor,
    *,
    huber_delta: float = 1.0,
    cosine_weight: float = 0.25,
) -> Tensor:
    """Train the block-expectation predictor toward the real block summary.

    Design spec section 19.2: Huber + cosine loss between the predicted and
    the *detached* actual summary. ``actual_summary`` must already be
    detached by the caller (section 6.2's "no target leakage" — the
    predictor should chase the summary, the summary must never chase the
    predictor).
    """

    if actual_summary.requires_grad:
        raise ValueError(
            "actual_summary must be detached before computing predictor_loss "
            "(design spec section 6.2)"
        )
    huber = F.huber_loss(
        predicted_summary, actual_summary, delta=huber_delta, reduction="none"
    ).mean(dim=-1)
    cosine_term = 1.0 - F.cosine_similarity(predicted_summary, actual_summary, dim=-1, eps=1e-8)
    return (huber + cosine_weight * cosine_term).mean()


def retrieval_utility(loss_without_memory: Tensor, loss_with_memory: Tensor) -> Tensor:
    """``U_t = L_without_memory - L_with_memory`` (design spec section 19.3).

    Positive means memory helped this example; negative means it hurt.
    Callers are expected to sample this rarely (the spec suggests 1-5% of
    blocks), since it requires a second forward pass.
    """

    return loss_without_memory - loss_with_memory


def write_calibration_loss(write_gate: Tensor, observed_utility: Tensor) -> Tensor:
    """Calibrate write strength toward observed future utility (section 19.5).

    Target is ``sigmoid(observed_utility)``: positive utility pushes the
    gate toward 1 (write more confidently next time this pattern recurs),
    negative utility pushes it toward 0. ``observed_utility`` is treated as
    a fixed target (detached) — this loss trains the gate, not the utility
    measurement.
    """

    target = torch.sigmoid(observed_utility.detach())
    return F.binary_cross_entropy(write_gate.clamp(1e-6, 1.0 - 1e-6), target)


def contrastive_key_loss(
    query: Tensor,
    positive_key: Tensor,
    negative_keys: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    """InfoNCE-style loss: a query should score its matching key above
    irrelevant ones (design spec section 19.4).

    ``query``/``positive_key``: ``[B, K]``; ``negative_keys``: ``[B, N, K]``.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    positive_score = (query * positive_key).sum(dim=-1) / temperature
    negative_scores = torch.einsum("bk,bnk->bn", query, negative_keys) / temperature
    logits = torch.cat([positive_score.unsqueeze(-1), negative_scores], dim=-1)
    targets = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    return F.cross_entropy(logits, targets)
