"""Quarantine buffer: temporary storage and confirmation logic (SGBM design spec section 15).

High-surprise, low-reliability candidates land here instead of stable
memory. An entry is promoted into stable memory only after repeated
confirmation; unconfirmed entries expire quickly. This is the mechanism
that keeps a confused-but-confident model from polluting long-lived memory
(spec sections 10.1 and 15).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from gator.memory.config import SurpriseMemoryConfig
from gator.memory.state import SurpriseMemoryState
from gator.memory.store import write_stable_memory

__all__ = [
    "write_or_confirm_quarantine",
    "promotion_ready_mask",
    "promote_confirmed",
    "expire_quarantine",
]


@torch.no_grad()
def write_or_confirm_quarantine(
    state: SurpriseMemoryState,
    *,
    candidate_key: Tensor,
    candidate_value: Tensor,
    write_strength: Tensor,
    reliability: Tensor,
    config: SurpriseMemoryConfig,
    active_mask: Tensor | None = None,
    eviction_gamma: float = 1e-3,
    reliability_ema_decay: float = 0.5,
) -> tuple[Tensor, Tensor]:
    """Add a new quarantine entry, or confirm a matching existing one.

    A quarantine slot whose key already cosine-matches the candidate above
    ``config.merge_threshold`` is treated as re-encountering the same
    candidate: its confirmation counter increments and its strength/
    reliability EMA toward the new observation (spec section 15's
    "confirmation update"), rather than being blended like stable-memory
    merge. A non-matching candidate allocates a free slot, or evicts the
    lowest-utility resident when the quarantine buffer is full.

    Returns ``(target_index [B], was_confirmation [B])``.
    """

    batch_size, quarantine_slots, _ = state.quarantine_keys.shape
    device = state.device
    if active_mask is None:
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    batch_arange = torch.arange(batch_size, device=device)

    normalized_key = F.normalize(candidate_key, dim=-1, eps=1e-8)

    similarity = torch.einsum("bk,bqk->bq", normalized_key, state.quarantine_keys)
    similarity = similarity.masked_fill(~state.quarantine_valid, -2.0)
    best_similarity, best_index = similarity.max(dim=1)
    has_valid = state.quarantine_valid.any(dim=1)
    is_confirmation = has_valid & (best_similarity > config.merge_threshold)

    free_mask = ~state.quarantine_valid
    has_free = free_mask.any(dim=1)
    first_free_index = free_mask.float().argmax(dim=1)

    # Quarantine eviction utility: unlike stable memory there is no
    # reliability field per quarantine slot (that is the whole point of
    # quarantine), so eviction favors slots with more confirmations and
    # higher strength, decaying with age.
    utility = (
        (1.0 + state.quarantine_confirmations.to(state.quarantine_strength.dtype))
        * state.quarantine_strength
        * torch.exp(-eviction_gamma * state.quarantine_age)
    )
    evict_utility = utility.masked_fill(~state.quarantine_valid, float("inf"))
    evict_index = evict_utility.argmin(dim=1)

    allocate_index = torch.where(has_free, first_free_index, evict_index)
    target_index = torch.where(is_confirmation, best_index, allocate_index)

    old_key = state.quarantine_keys[batch_arange, target_index]
    old_strength = state.quarantine_strength[batch_arange, target_index]
    old_confirmations = state.quarantine_confirmations[batch_arange, target_index]
    old_age = state.quarantine_age[batch_arange, target_index]

    confirmed_key = normalized_key  # refresh key toward the latest observation
    confirmed_strength = (
        reliability_ema_decay * old_strength + (1.0 - reliability_ema_decay) * write_strength
    )
    confirmed_confirmations = old_confirmations + 1

    new_key = torch.where(is_confirmation.unsqueeze(-1), confirmed_key, normalized_key)
    new_strength = torch.where(is_confirmation, confirmed_strength, write_strength)
    new_confirmations = torch.where(
        is_confirmation, confirmed_confirmations, torch.zeros_like(old_confirmations)
    )

    write_mask = active_mask
    final_key = torch.where(write_mask.unsqueeze(-1), new_key, old_key)
    final_strength = torch.where(write_mask, new_strength, old_strength)
    final_confirmations = torch.where(write_mask, new_confirmations, old_confirmations)
    final_age = torch.where(write_mask, torch.zeros_like(old_age), old_age)
    final_valid = state.quarantine_valid[batch_arange, target_index] | write_mask
    final_value = torch.where(
        write_mask.unsqueeze(-1),
        candidate_value,
        state.quarantine_values[batch_arange, target_index],
    )

    state.quarantine_keys[batch_arange, target_index] = final_key
    state.quarantine_values[batch_arange, target_index] = final_value
    state.quarantine_strength[batch_arange, target_index] = final_strength
    state.quarantine_confirmations[batch_arange, target_index] = final_confirmations
    state.quarantine_age[batch_arange, target_index] = final_age
    state.quarantine_valid[batch_arange, target_index] = final_valid

    return target_index, is_confirmation & write_mask


def _as_stable_view(state: SurpriseMemoryState) -> SurpriseMemoryState:
    """Adapt the quarantine block into a view ``compute_slot_utility`` can
    read (that helper reads ``strength``/``reliability``/``usage``/``age``
    off whatever object it is given)."""

    class _QuarantineView:
        strength = state.quarantine_strength
        reliability = state.quarantine_strength.new_zeros(state.quarantine_strength.shape) + 0.5
        usage = state.quarantine_confirmations.to(state.quarantine_strength.dtype)
        age = state.quarantine_age

    return _QuarantineView()  # type: ignore[return-value]


def promotion_ready_mask(state: SurpriseMemoryState, config: SurpriseMemoryConfig) -> Tensor:
    """Which quarantine slots (``[B, Q]``) meet the promotion bar (spec section 15)."""

    reliability_ready = state.quarantine_strength >= config.write_threshold
    confirmations_ready = state.quarantine_confirmations >= config.promotion_confirmations
    return state.quarantine_valid & reliability_ready & confirmations_ready


@torch.no_grad()
def promote_confirmed(
    state: SurpriseMemoryState,
    config: SurpriseMemoryConfig,
    *,
    eviction_gamma: float = 1e-3,
) -> Tensor:
    """Move one ready quarantine entry per lane into stable memory.

    When a lane has more than one eligible entry, the highest-strength one
    is promoted first; the rest remain queued for a later call. Returns
    which lanes promoted something this call (``[B]``).
    """

    ready = promotion_ready_mask(state, config)
    any_ready = ready.any(dim=1)
    if not bool(any_ready.any()):
        return any_ready

    strength_for_selection = state.quarantine_strength.masked_fill(~ready, float("-inf"))
    _, chosen_index = strength_for_selection.max(dim=1)

    batch_arange = torch.arange(state.batch_size, device=state.device)
    chosen_key = state.quarantine_keys[batch_arange, chosen_index]
    chosen_value = state.quarantine_values[batch_arange, chosen_index]
    chosen_strength = state.quarantine_strength[batch_arange, chosen_index]

    write_stable_memory(
        state,
        candidate_key=chosen_key,
        candidate_value=chosen_value,
        commit_strength=chosen_strength,
        reliability=chosen_strength,
        surprise=torch.zeros_like(chosen_strength),
        config=config,
        active_mask=any_ready,
        eviction_gamma=eviction_gamma,
    )

    clear_mask = any_ready
    state.quarantine_valid[batch_arange[clear_mask], chosen_index[clear_mask]] = False
    state.quarantine_strength[batch_arange[clear_mask], chosen_index[clear_mask]] = 0.0
    state.quarantine_confirmations[batch_arange[clear_mask], chosen_index[clear_mask]] = 0
    state.quarantine_age[batch_arange[clear_mask], chosen_index[clear_mask]] = 0.0

    return any_ready


@torch.no_grad()
def expire_quarantine(
    state: SurpriseMemoryState,
    *,
    max_age_unconfirmed: float,
    max_age_confirmed: float | None = None,
) -> Tensor:
    """Invalidate stale quarantine entries (spec section 15: "expire quickly if unconfirmed").

    Unconfirmed entries (``confirmations == 0``) expire past
    ``max_age_unconfirmed``. Entries that have at least one confirmation
    are given more patience, expiring only past ``max_age_confirmed`` (if
    given at all — ``None`` means confirmed entries never age out here,
    left instead to eventually promote or to be evicted by utility when
    the buffer fills).
    """

    state.quarantine_age = state.quarantine_age + state.quarantine_valid.to(state.quarantine_age.dtype)

    unconfirmed = state.quarantine_confirmations == 0
    expire = state.quarantine_valid & unconfirmed & (state.quarantine_age > max_age_unconfirmed)
    if max_age_confirmed is not None:
        expire = expire | (
            state.quarantine_valid
            & ~unconfirmed
            & (state.quarantine_age > max_age_confirmed)
        )

    if bool(expire.any()):
        state.quarantine_valid = state.quarantine_valid & ~expire
        state.quarantine_strength = torch.where(
            expire, torch.zeros_like(state.quarantine_strength), state.quarantine_strength
        )
        state.quarantine_confirmations = torch.where(
            expire, torch.zeros_like(state.quarantine_confirmations), state.quarantine_confirmations
        )
    return expire
