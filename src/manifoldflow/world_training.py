"""Losses and stage gates for training the world-memory adapters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class WorldLoss:
    total: Tensor
    answer: Tensor
    retrieval: Tensor
    memory_contrast: Tensor


def answer_mismatched_indices(answers: tuple[str, ...]) -> Tensor:
    """Choose a deterministic hard-negative row with a different answer."""
    if not answers:
        raise ValueError("answers cannot be empty")
    if len(answers) == 1:
        return torch.zeros(1, dtype=torch.long)
    indices: list[int] = []
    for row, answer in enumerate(answers):
        candidates = [
            (row + offset) % len(answers) for offset in range(1, len(answers))
        ]
        indices.append(
            next(
                (item for item in candidates if answers[item] != answer), candidates[0]
            )
        )
    return torch.tensor(indices, dtype=torch.long)


def retrieval_info_nce(
    query_keys: Tensor, positive_keys: Tensor, temperature: float = 0.07
) -> Tensor:
    if query_keys.shape != positive_keys.shape or query_keys.ndim != 2:
        raise ValueError("query_keys and positive_keys must share shape [batch, d]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    query = torch.nn.functional.normalize(query_keys, dim=-1)
    positive = torch.nn.functional.normalize(positive_keys, dim=-1)
    logits = query @ positive.transpose(0, 1) / temperature
    target = torch.arange(query.shape[0], device=query.device)
    return torch.nn.functional.cross_entropy(logits, target)


def masked_candidate_retrieval_loss(
    query_keys: Tensor,
    candidate_keys: Tensor,
    candidate_mask: Tensor,
    evidence_mask: Tensor,
    *,
    temperature: float = 0.07,
) -> Tensor:
    """Multi-positive retrieval loss over each example's entity candidates."""
    if query_keys.ndim != 2 or candidate_keys.ndim != 3:
        raise ValueError(
            "queries and candidates must have shapes [batch, d] and [batch, n, d]"
        )
    if (
        candidate_keys.shape[0] != query_keys.shape[0]
        or candidate_keys.shape[2] != query_keys.shape[1]
    ):
        raise ValueError("query and candidate dimensions must match")
    if (
        candidate_mask.shape != candidate_keys.shape[:2]
        or evidence_mask.shape != candidate_mask.shape
    ):
        raise ValueError("candidate and evidence masks must have shape [batch, n]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid_rows = evidence_mask.any(dim=1)
    if not valid_rows.any():
        return query_keys.sum() * 0.0
    queries = torch.nn.functional.normalize(query_keys[valid_rows], dim=-1)
    candidates = torch.nn.functional.normalize(candidate_keys[valid_rows], dim=-1)
    valid_candidates = candidate_mask[valid_rows]
    positives = evidence_mask[valid_rows] & valid_candidates
    logits = torch.einsum("bd,bnd->bn", queries, candidates) / temperature
    logits = logits.masked_fill(~valid_candidates, float("-inf"))
    positive_logits = logits.masked_fill(~positives, float("-inf"))
    return (
        torch.logsumexp(logits, dim=1) - torch.logsumexp(positive_logits, dim=1)
    ).mean()


def memory_use_contrast(
    correct_loss: Tensor, shuffled_loss: Tensor, margin: float = 0.4
) -> Tensor:
    """Penalize a port when shuffled memory is almost as useful as correct memory."""
    if correct_loss.ndim != 0 or shuffled_loss.ndim != 0:
        raise ValueError("language-model losses must be scalar")
    return torch.relu(correct_loss - shuffled_loss + margin)


def combine_world_losses(
    answer_loss: Tensor,
    retrieval_loss: Tensor,
    contrast_loss: Tensor,
    *,
    retrieval_weight: float = 0.25,
    contrast_weight: float = 0.10,
) -> WorldLoss:
    total = (
        answer_loss
        + retrieval_weight * retrieval_loss
        + contrast_weight * contrast_loss
    )
    return WorldLoss(total, answer_loss, retrieval_loss, contrast_loss)


def should_enable_lora(
    retrieval_recall_at_4: float,
    heldout_answer_accuracy: float,
    *,
    recall_gate: float = 0.95,
    answer_gate: float = 0.85,
) -> bool:
    return (
        retrieval_recall_at_4 >= recall_gate and heldout_answer_accuracy < answer_gate
    )
