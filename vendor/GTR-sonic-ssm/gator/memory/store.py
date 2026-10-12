"""Stable-memory merge, allocation, decay, and eviction (SGBM design spec section 14, 16).

Every function here mutates a :class:`SurpriseMemoryState` in place under
``torch.no_grad()`` — this is the online, backward-free write path the
design spec requires. All operations are fully vectorized over the batch
dimension (no per-lane Python loop): each batch lane independently decides
whether to merge into an existing slot or allocate a new one, so a single
call handles an entire batch of candidate writes in one pass.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from gator.memory.config import SurpriseMemoryConfig
from gator.memory.state import SurpriseMemoryState

__all__ = [
    "compute_slot_utility",
    "write_stable_memory",
    "decay_stable_memory",
]


def compute_slot_utility(state: SurpriseMemoryState, *, eviction_gamma: float) -> Tensor:
    """``utility_j = strength_j * reliability_j * (1 + log(usage_j + 1)) * exp(-gamma * age_j)``.

    Design spec section 14.2. Used to pick an eviction target when a slot
    must be freed and there is no already-invalid slot available.
    """

    return (
        state.strength
        * state.reliability
        * torch.log1p(state.usage).add(1.0)
        * torch.exp(-eviction_gamma * state.age)
    )


@torch.no_grad()
def write_stable_memory(
    state: SurpriseMemoryState,
    *,
    candidate_key: Tensor,
    candidate_value: Tensor,
    commit_strength: Tensor,
    reliability: Tensor,
    surprise: Tensor,
    config: SurpriseMemoryConfig,
    active_mask: Tensor | None = None,
    eviction_gamma: float = 1e-3,
    reliability_ema_decay: float = 0.9,
) -> Tensor:
    """Merge or allocate one candidate write per batch lane (spec section 14).

    ``candidate_key``/``candidate_value`` are ``[B, K]``/``[B, V]``;
    ``commit_strength``/``reliability``/``surprise`` are ``[B]``.
    ``active_mask`` (``[B]``, default all-True) lets a caller skip specific
    lanes entirely — those lanes' memory is left byte-for-byte unchanged,
    which is what lets this same function serve both "every lane writes
    every block" callers and callers (e.g. quarantine promotion) that only
    act on a subset of lanes in a given call.

    Returns the per-lane slot index written or merged into (``[B]``,
    meaningless for inactive lanes).
    """

    batch_size, memory_slots, _ = state.keys.shape
    device = state.device
    if active_mask is None:
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    batch_arange = torch.arange(batch_size, device=device)

    normalized_key = F.normalize(candidate_key, dim=-1, eps=1e-8)

    similarity = torch.einsum("bk,bmk->bm", normalized_key, state.keys)
    similarity = similarity.masked_fill(~state.valid, -2.0)
    best_similarity, best_index = similarity.max(dim=1)
    has_valid = state.valid.any(dim=1)

    merge_mask = has_valid & (best_similarity > config.merge_threshold)

    free_mask = ~state.valid
    has_free = free_mask.any(dim=1)
    first_free_index = free_mask.float().argmax(dim=1)

    utility = compute_slot_utility(state, eviction_gamma=eviction_gamma)
    evict_utility = utility.masked_fill(~state.valid, float("inf"))
    evict_index = evict_utility.argmin(dim=1)

    allocate_index = torch.where(has_free, first_free_index, evict_index)
    target_index = torch.where(merge_mask, best_index, allocate_index)

    old_key = state.keys[batch_arange, target_index]
    old_value = state.values[batch_arange, target_index]
    old_strength = state.strength[batch_arange, target_index]
    old_reliability = state.reliability[batch_arange, target_index]
    old_usage = state.usage[batch_arange, target_index]
    old_surprise = state.surprise[batch_arange, target_index]

    similarity_at_target = best_similarity.clamp(-1.0, 1.0)
    alpha = (commit_strength * (1.0 - 0.5 * similarity_at_target)).clamp(0.0, 1.0)

    merged_key = F.normalize(
        (1.0 - alpha).unsqueeze(-1) * old_key + alpha.unsqueeze(-1) * normalized_key,
        dim=-1,
        eps=1e-8,
    )
    merged_value = old_value + alpha.unsqueeze(-1) * (candidate_value - old_value)
    merged_strength = (old_strength + commit_strength).clamp(0.0, 1.0)
    merged_reliability = (
        reliability_ema_decay * old_reliability + (1.0 - reliability_ema_decay) * reliability
    )
    merged_usage = old_usage + 1.0

    new_key = torch.where(merge_mask.unsqueeze(-1), merged_key, normalized_key)
    new_value = torch.where(merge_mask.unsqueeze(-1), merged_value, candidate_value)
    new_strength = torch.where(merge_mask, merged_strength, commit_strength.clamp(0.0, 1.0))
    new_reliability = torch.where(merge_mask, merged_reliability, reliability)
    new_usage = torch.where(merge_mask, merged_usage, torch.ones_like(old_usage))
    new_surprise = surprise

    write_mask = active_mask
    final_key = torch.where(write_mask.unsqueeze(-1), new_key, old_key)
    final_value = torch.where(write_mask.unsqueeze(-1), new_value, old_value)
    final_strength = torch.where(write_mask, new_strength, old_strength)
    final_reliability = torch.where(write_mask, new_reliability, old_reliability)
    final_usage = torch.where(write_mask, new_usage, old_usage)
    final_surprise = torch.where(write_mask, new_surprise, old_surprise)
    final_age = torch.where(write_mask, torch.zeros_like(old_usage), state.age[batch_arange, target_index])
    final_valid = state.valid[batch_arange, target_index] | write_mask

    state.keys[batch_arange, target_index] = final_key
    state.values[batch_arange, target_index] = final_value
    state.strength[batch_arange, target_index] = final_strength
    state.reliability[batch_arange, target_index] = final_reliability
    state.usage[batch_arange, target_index] = final_usage
    state.surprise[batch_arange, target_index] = final_surprise
    state.age[batch_arange, target_index] = final_age
    state.valid[batch_arange, target_index] = final_valid

    return target_index


@torch.no_grad()
def decay_stable_memory(
    state: SurpriseMemoryState,
    config: SurpriseMemoryConfig,
    *,
    time_constant: float = 10.0,
    forget_threshold: float = 1e-4,
) -> Tensor:
    """Advance age and decay strength for every valid slot (spec section 16).

    ``decay_j = base_decay * (1 - reliability_j) * age_j / (age_j + time_constant)``

    Slots whose strength decays below ``forget_threshold`` are freed
    (``valid`` cleared, ``strength`` zeroed) so they become eligible for
    immediate reuse rather than lingering as a near-zero-utility resident.
    Returns the boolean mask of newly-forgotten slots (``[B, M]``).
    """

    valid_float = state.valid.to(state.strength.dtype)
    state.age = state.age + valid_float
    decay_rate = (
        config.base_decay * (1.0 - state.reliability) * (state.age / (state.age + time_constant))
    )
    state.strength = state.strength * torch.exp(-decay_rate)

    forgotten = state.valid & (state.strength < forget_threshold)
    if bool(forgotten.any()):
        state.valid = state.valid & ~forgotten
        state.strength = torch.where(forgotten, torch.zeros_like(state.strength), state.strength)
    return forgotten
