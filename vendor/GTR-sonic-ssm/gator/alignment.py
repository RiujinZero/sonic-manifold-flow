"""Monotonic latent alignment utilities for simultaneous sequence formation.

The alignment lattice has one deliberately small transition system.  At each
physical output slot a path either emits the next required reference token or
emits ``BLANK``.  Required tokens therefore retain their order, while their
physical slots are latent.  ``EOS`` is an ordinary required token in complete
references; ``BLANK`` is removed during rendering and never terminates it.

The differentiable forward-sum uses a loop over output slots and vectorizes
over the batch and reference states.  Viterbi traceback and edit diagnostics
are intended for evaluation and deliberately operate on detached tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F


Reduction = Literal["none", "mean", "sum"]


@dataclass(frozen=True)
class ViterbiAlignment:
    """A detached best path expanded back to physical output slots."""

    expanded_ids: torch.Tensor
    emission_mask: torch.Tensor
    score_mask: torch.Tensor
    positions: torch.Tensor
    required_lengths: torch.Tensor
    path_score: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class AlignedCrossEntropyResult:
    """Viterbi-aligned cross entropy and the labels that produced it."""

    loss: torch.Tensor
    per_example: torch.Tensor
    required_token_loss: torch.Tensor
    per_required_token: torch.Tensor
    alignment: ViterbiAlignment


@dataclass(frozen=True)
class CollapsedSequences:
    """Ragged rendered token sequences plus EOS termination metadata."""

    tokens: tuple[tuple[int, ...], ...]
    terminated: torch.Tensor
    eos_positions: torch.Tensor


@dataclass(frozen=True)
class AlignmentDiagnostics:
    """Per-example best-aligned and rendered-sequence diagnostics."""

    alignment: ViterbiAlignment
    formation_accuracy: torch.Tensor
    required_token_accuracy: torch.Tensor
    content_accuracy: torch.Tensor
    content_precision: torch.Tensor
    content_recall: torch.Tensor
    content_f1: torch.Tensor
    mean_displacement: torch.Tensor
    max_displacement: torch.Tensor
    insertions: torch.Tensor
    deletions: torch.Tensor
    substitutions: torch.Tensor
    edit_distance: torch.Tensor
    edit_accuracy: torch.Tensor
    rendered_exact: torch.Tensor
    rendered_exact_mask: torch.Tensor
    predicted_lengths: torch.Tensor
    reference_lengths: torch.Tensor
    predicted_terminated: torch.Tensor


@dataclass(frozen=True)
class SequenceQualityResult:
    """A modest global formation loss and its independently logged parts."""

    loss: torch.Tensor
    semantic_composition: torch.Tensor
    expected_length_huber: torch.Tensor
    complete_eos_count: torch.Tensor
    expected_lengths: torch.Tensor
    expected_eos_counts: torch.Tensor


def _check_reduction(reduction: Reduction) -> None:
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _check_inputs(
    logits: torch.Tensor,
    reference_ids: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    blank_id: int,
    eos_id: int,
    complete: torch.Tensor | None,
) -> None:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, output_slots, vocabulary]")
    if reference_ids.ndim != 2 or reference_mask.shape != reference_ids.shape:
        raise ValueError("reference IDs and mask must have matching [batch, reference] shapes")
    if logits.shape[0] != reference_ids.shape[0]:
        raise ValueError("logits and references must have the same batch size")
    if logits.shape[1] <= 0:
        raise ValueError("at least one output slot is required")
    if not 0 <= blank_id < logits.shape[-1] or not 0 <= eos_id < logits.shape[-1]:
        raise ValueError("blank_id and eos_id must be inside the vocabulary")
    if blank_id == eos_id:
        raise ValueError("BLANK and EOS must be distinct tokens")
    if complete is not None and (complete.ndim != 1 or complete.shape[0] != logits.shape[0]):
        raise ValueError("complete must have shape [batch]")


def _compact_references(
    reference_ids: torch.Tensor, reference_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stable-pack arbitrary masked references without a Python batch loop."""

    mask = reference_mask.to(dtype=torch.bool)
    batch, width = reference_ids.shape
    if width == 0:
        compact = reference_ids
        lengths = torch.zeros(batch, dtype=torch.long, device=reference_ids.device)
        return compact, mask, lengths
    position = torch.arange(width, device=reference_ids.device)[None].expand(batch, -1)
    # Valid keys precede invalid keys, and original order is retained in both groups.
    keys = torch.where(mask, position, width + position)
    order = keys.argsort(dim=1)
    compact = reference_ids.gather(1, order)
    lengths = mask.sum(dim=1, dtype=torch.long)
    compact_mask = position < lengths[:, None]
    return compact, compact_mask, lengths


def _resolve_complete(
    complete: torch.Tensor | None,
    compact: torch.Tensor,
    lengths: torch.Tensor,
    *,
    eos_id: int,
) -> torch.Tensor:
    if complete is not None:
        return complete.to(device=compact.device, dtype=torch.bool)
    if compact.shape[1] == 0:
        return torch.zeros_like(lengths, dtype=torch.bool)
    last = compact.gather(1, (lengths - 1).clamp_min(0)[:, None]).squeeze(1)
    return lengths.gt(0) & last.eq(eos_id)


def _masked_log_probs(
    logits: torch.Tensor, valid_token_mask: torch.Tensor | None
) -> torch.Tensor:
    if valid_token_mask is None:
        return F.log_softmax(logits, dim=-1)
    if valid_token_mask.ndim != 1 or valid_token_mask.shape[0] != logits.shape[-1]:
        raise ValueError("valid_token_mask must have one boolean entry per vocabulary item")
    valid = valid_token_mask.to(device=logits.device, dtype=torch.bool)
    return F.log_softmax(logits.masked_fill(~valid[None, None], -torch.inf), dim=-1)


def _resolve_log_probs(
    logits: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    precomputed_log_probs: torch.Tensor | None,
) -> torch.Tensor:
    if precomputed_log_probs is None:
        return _masked_log_probs(logits, valid_token_mask)
    if precomputed_log_probs.shape != logits.shape:
        raise ValueError("precomputed_log_probs must match logits")
    if precomputed_log_probs.device != logits.device:
        raise ValueError("precomputed_log_probs must share the logits device")
    return precomputed_log_probs


def _blank_weights(
    blank_weight: float | torch.Tensor | None,
    lengths: torch.Tensor,
    output_slots: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if blank_weight is None:
        blanks = output_slots - lengths
        ratio = lengths.to(dtype) / blanks.clamp_min(1).to(dtype)
        # Equalize large blank tails without ever magnifying a small tail.
        return torch.where(
            blanks.gt(0),
            torch.where(lengths.gt(0), ratio.clamp_max(1.0), torch.ones_like(ratio)),
            torch.ones_like(ratio),
        )
    if isinstance(blank_weight, torch.Tensor):
        value = blank_weight.to(device=lengths.device, dtype=dtype)
        if value.ndim == 0:
            value = value.expand_as(lengths)
        if value.ndim != 1 or value.shape[0] != lengths.shape[0]:
            raise ValueError("tensor blank_weight must be scalar or have shape [batch]")
        return value
    if blank_weight < 0:
        raise ValueError("blank_weight cannot be negative")
    return torch.full_like(lengths, float(blank_weight), dtype=dtype)


def _reduce(values: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    _check_reduction(reduction)
    if reduction == "none":
        return values
    if reduction == "sum":
        return values.sum()
    return values.mean()


def _normalizer(
    lengths: torch.Tensor,
    output_slots: int,
    blank_weights: torch.Tensor,
    *,
    normalize: bool,
) -> torch.Tensor:
    if not normalize:
        return torch.ones_like(blank_weights)
    return (
        lengths.to(blank_weights.dtype)
        + blank_weights * (output_slots - lengths).to(blank_weights.dtype)
    ).clamp_min(1.0)


def _forward_lattice(
    log_probs: torch.Tensor,
    compact: torch.Tensor,
    lengths: torch.Tensor,
    *,
    blank_id: int,
    blank_weights: torch.Tensor,
    complete: torch.Tensor,
    max_right_shift: int | None,
) -> torch.Tensor:
    batch, output_slots, _ = log_probs.shape
    width = compact.shape[1]
    # ``logaddexp(-inf, -inf)`` has undefined derivatives in PyTorch.  A very
    # negative finite sentinel keeps unreachable lattice states numerically
    # inert while preserving finite gradients for every legal example.
    neg_large = log_probs.new_full((), -torch.finfo(log_probs.dtype).max / 4)
    alpha = neg_large.expand(batch, width + 1).clone()
    alpha[:, 0] = 0.0
    state_index = torch.arange(width + 1, device=log_probs.device)[None]
    target_index = torch.arange(width, device=log_probs.device)[None]

    for slot in range(output_slots):
        blank_increment = (
            blank_weights[:, None] * log_probs[:, slot, blank_id, None]
        )
        final_state = state_index.eq(lengths[:, None])
        # Once an incomplete prefix has been consumed, the remaining physical
        # slots are unobserved continuation. Their total probability is one,
        # represented by a zero-cost absorbing transition.
        blank_increment = torch.where(
            final_state & ~complete[:, None],
            torch.zeros_like(blank_increment),
            blank_increment,
        )
        blank = alpha + blank_increment
        blank = torch.where(state_index <= lengths[:, None], blank, neg_large)

        if width:
            token_log_probs = log_probs[:, slot].gather(1, compact)
            emit = alpha[:, :-1] + token_log_probs
            allowed = target_index < lengths[:, None]
            if max_right_shift is not None:
                allowed = allowed & ((slot - target_index) <= max_right_shift)
            emit = torch.where(allowed, emit, neg_large)
            emit_to = torch.cat([neg_large.expand(batch, 1), emit], dim=1)
            alpha = torch.logaddexp(blank, emit_to)
        else:
            alpha = blank
    result = alpha.gather(1, lengths[:, None]).squeeze(1)
    return torch.where(
        lengths <= output_slots,
        result,
        result.new_full((), -torch.inf),
    )


def monotonic_forward_sum_nll(
    logits: torch.Tensor,
    reference_ids: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    blank_id: int,
    eos_id: int,
    complete: torch.Tensor | None = None,
    valid_token_mask: torch.Tensor | None = None,
    precomputed_log_probs: torch.Tensor | None = None,
    blank_weight: float | torch.Tensor | None = 1.0,
    max_right_shift: int | None = None,
    normalize: bool = True,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Marginalize every legal emit-next-token-or-BLANK alignment.

    ``complete`` is explicit sequence metadata. It does not add EOS: complete
    references are expected to contain EOS and require BLANK afterward, while a
    censored reference enters a zero-cost free-continuation state after its last
    observed token. With ``blank_weight`` one this is a proper forward-sum NLL.
    Other values implement class-weighted aligned CE. Impossible examples (more
    required tokens than output slots) return positive infinity.
    """

    _check_inputs(
        logits,
        reference_ids,
        reference_mask,
        blank_id=blank_id,
        eos_id=eos_id,
        complete=complete,
    )
    if max_right_shift is not None and max_right_shift < 0:
        raise ValueError("max_right_shift cannot be negative")
    compact, _, lengths = _compact_references(reference_ids, reference_mask)
    complete_mask = _resolve_complete(complete, compact, lengths, eos_id=eos_id)
    log_probs = _resolve_log_probs(
        logits, valid_token_mask, precomputed_log_probs
    )
    blank_weights = _blank_weights(
        blank_weight, lengths, logits.shape[1], dtype=log_probs.dtype
    )
    path_log_probability = _forward_lattice(
        log_probs,
        compact,
        lengths,
        blank_id=blank_id,
        blank_weights=blank_weights,
        complete=complete_mask,
        max_right_shift=max_right_shift,
    )
    per_example = -path_log_probability / _normalizer(
        lengths, logits.shape[1], blank_weights, normalize=normalize
    )
    return _reduce(per_example, reduction)


def _viterbi_from_scores(
    blank_scores: torch.Tensor,
    emit_scores: torch.Tensor,
    compact: torch.Tensor,
    lengths: torch.Tensor,
    *,
    blank_id: int,
    complete: torch.Tensor,
    max_right_shift: int | None,
) -> ViterbiAlignment:
    """Run max-sum DP from precomputed blank and required-token scores."""

    batch, output_slots = blank_scores.shape
    width = compact.shape[1]
    neg_inf = blank_scores.new_full((), -torch.inf)
    score = blank_scores.new_full((batch, width + 1), -torch.inf)
    score[:, 0] = 0.0
    state_index = torch.arange(width + 1, device=blank_scores.device)[None]
    target_index = torch.arange(width, device=blank_scores.device)[None]
    decisions: list[torch.Tensor] = []

    for slot in range(output_slots):
        blank_increment = blank_scores[:, slot, None]
        final_state = state_index.eq(lengths[:, None])
        blank_increment = torch.where(
            final_state & ~complete[:, None],
            torch.zeros_like(blank_increment),
            blank_increment,
        )
        blank = score + blank_increment
        blank = torch.where(state_index <= lengths[:, None], blank, neg_inf)
        if width:
            emit = score[:, :-1] + emit_scores[:, slot]
            allowed = target_index < lengths[:, None]
            if max_right_shift is not None:
                allowed = allowed & ((slot - target_index) <= max_right_shift)
            emit = torch.where(allowed, emit, neg_inf)
            emit_to = F.pad(emit, (1, 0), value=-torch.inf)
            # Prefer an early emission on an exact tie.
            choose_emit = emit_to >= blank
            score = torch.where(choose_emit, emit_to, blank)
        else:
            choose_emit = torch.zeros_like(score, dtype=torch.bool)
            score = blank
        decisions.append(choose_emit)

    final_score = score.gather(1, lengths[:, None]).squeeze(1)
    expanded = torch.full(
        (batch, output_slots), blank_id, dtype=compact.dtype, device=compact.device
    )
    positions = torch.full_like(compact, -1)
    state = lengths.clone()
    batch_index = torch.arange(batch, device=compact.device)
    for slot in range(output_slots - 1, -1, -1):
        decision = decisions[slot].gather(1, state[:, None]).squeeze(1)
        emitted = decision & state.gt(0)
        target = (state - 1).clamp_min(0)
        if width:
            token = compact.gather(1, target[:, None]).squeeze(1)
            expanded[:, slot] = torch.where(emitted, token, expanded[:, slot])
            live_batch = batch_index[emitted]
            live_target = target[emitted]
            positions[live_batch, live_target] = slot
        state = state - emitted.to(state.dtype)

    valid = state.eq(0) & torch.isfinite(final_score)
    if width:
        last_target = (lengths - 1).clamp_min(0)
        last_emission = positions.gather(1, last_target[:, None]).squeeze(1)
        last_emission = torch.where(
            lengths.gt(0), last_emission, torch.full_like(last_emission, -1)
        )
    else:
        last_emission = torch.full_like(lengths, -1)
    slot_index = torch.arange(output_slots, device=compact.device)[None]
    score_mask = complete[:, None] | slot_index.le(last_emission[:, None])
    score_mask = score_mask & valid[:, None]
    return ViterbiAlignment(
        expanded_ids=expanded.detach(),
        emission_mask=expanded.ne(blank_id).detach(),
        score_mask=score_mask.detach(),
        positions=positions.detach(),
        required_lengths=lengths.detach(),
        path_score=final_score.detach(),
        valid=valid.detach(),
    )


def viterbi_expand_targets(
    logits: torch.Tensor,
    reference_ids: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    blank_id: int,
    eos_id: int,
    complete: torch.Tensor | None = None,
    valid_token_mask: torch.Tensor | None = None,
    precomputed_log_probs: torch.Tensor | None = None,
    blank_weight: float | torch.Tensor | None = 1.0,
    max_right_shift: int | None = None,
) -> ViterbiAlignment:
    """Return detached expanded labels from the maximum-likelihood path."""

    _check_inputs(
        logits,
        reference_ids,
        reference_mask,
        blank_id=blank_id,
        eos_id=eos_id,
        complete=complete,
    )
    if max_right_shift is not None and max_right_shift < 0:
        raise ValueError("max_right_shift cannot be negative")
    compact, _, lengths = _compact_references(reference_ids, reference_mask)
    complete_mask = _resolve_complete(complete, compact, lengths, eos_id=eos_id)
    log_probs = _resolve_log_probs(
        logits, valid_token_mask, precomputed_log_probs
    ).detach()
    blank_weights = _blank_weights(
        blank_weight, lengths, logits.shape[1], dtype=log_probs.dtype
    )
    blank_scores = blank_weights[:, None] * log_probs[:, :, blank_id]
    emit_scores = log_probs.new_empty(
        logits.shape[0], logits.shape[1], compact.shape[1]
    )
    for slot in range(logits.shape[1]):
        if compact.shape[1]:
            emit_scores[:, slot] = log_probs[:, slot].gather(1, compact)
    return _viterbi_from_scores(
        blank_scores,
        emit_scores,
        compact,
        lengths,
        blank_id=blank_id,
        complete=complete_mask,
        max_right_shift=max_right_shift,
    )


def viterbi_aligned_cross_entropy(
    logits: torch.Tensor,
    reference_ids: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    blank_id: int,
    eos_id: int,
    complete: torch.Tensor | None = None,
    valid_token_mask: torch.Tensor | None = None,
    precomputed_log_probs: torch.Tensor | None = None,
    blank_weight: float | torch.Tensor | None = 1.0,
    max_right_shift: int | None = None,
    normalize: bool = True,
    reduction: Reduction = "mean",
) -> AlignedCrossEntropyResult:
    """Choose a detached Viterbi path, then train its expanded labels with CE."""

    log_probs = _resolve_log_probs(
        logits, valid_token_mask, precomputed_log_probs
    )
    alignment = viterbi_expand_targets(
        logits,
        reference_ids,
        reference_mask,
        blank_id=blank_id,
        eos_id=eos_id,
        complete=complete,
        valid_token_mask=valid_token_mask,
        precomputed_log_probs=log_probs.detach(),
        blank_weight=blank_weight,
        max_right_shift=max_right_shift,
    )
    token_nll = -log_probs.gather(2, alignment.expanded_ids[:, :, None]).squeeze(2)
    blank_weights = _blank_weights(
        blank_weight,
        alignment.required_lengths,
        logits.shape[1],
        dtype=log_probs.dtype,
    )
    weights = torch.where(
        alignment.emission_mask,
        torch.ones_like(token_nll),
        blank_weights[:, None],
    )
    weights = weights * alignment.score_mask.to(weights.dtype)
    weighted_nll = (token_nll * weights).sum(1)
    per_example = weighted_nll / _normalizer(
        alignment.required_lengths,
        logits.shape[1],
        blank_weights,
        normalize=normalize,
    )
    per_required_token = weighted_nll / alignment.required_lengths.clamp_min(1).to(
        weighted_nll.dtype
    )
    per_example = torch.where(
        alignment.valid,
        per_example,
        per_example.new_full((), torch.inf),
    )
    return AlignedCrossEntropyResult(
        loss=_reduce(per_example, reduction),
        per_example=per_example,
        required_token_loss=_reduce(per_required_token, reduction),
        per_required_token=per_required_token,
        alignment=alignment,
    )


# Concise spelling for training call sites.
viterbi_aligned_ce = viterbi_aligned_cross_entropy


def collapse_blank_sequences(
    token_ids: torch.Tensor,
    *,
    blank_id: int,
    eos_id: int,
    token_mask: torch.Tensor | None = None,
) -> CollapsedSequences:
    """Remove BLANK tokens and stop at EOS without coalescing any other IDs."""

    if token_ids.ndim == 1:
        token_ids = token_ids[None]
        if token_mask is not None:
            token_mask = token_mask[None]
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [slots] or [batch, slots]")
    if token_mask is not None and token_mask.shape != token_ids.shape:
        raise ValueError("token_mask must match token_ids")
    mask_rows = (
        token_mask.to(dtype=torch.bool).detach().cpu().tolist()
        if token_mask is not None
        else [[True] * token_ids.shape[1] for _ in range(token_ids.shape[0])]
    )
    rows = token_ids.detach().cpu().tolist()
    collapsed: list[tuple[int, ...]] = []
    terminated: list[bool] = []
    eos_positions: list[int] = []
    for row, mask in zip(rows, mask_rows):
        result: list[int] = []
        found_eos = False
        eos_position = -1
        for position, (token, valid) in enumerate(zip(row, mask)):
            if not valid:
                continue
            if token == eos_id:
                found_eos = True
                eos_position = position
                break
            if token != blank_id:
                result.append(int(token))
        collapsed.append(tuple(result))
        terminated.append(found_eos)
        eos_positions.append(eos_position)
    return CollapsedSequences(
        tokens=tuple(collapsed),
        terminated=torch.tensor(terminated, dtype=torch.bool, device=token_ids.device),
        eos_positions=torch.tensor(eos_positions, dtype=torch.long, device=token_ids.device),
    )


def _edit_counts(reference: tuple[int, ...], prediction: tuple[int, ...]) -> tuple[int, int, int]:
    """Return insertion, deletion, and substitution counts for one pair."""

    rows, columns = len(reference) + 1, len(prediction) + 1
    distance = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        distance[row][0] = row
    for column in range(columns):
        distance[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            substitution = distance[row - 1][column - 1] + (
                reference[row - 1] != prediction[column - 1]
            )
            distance[row][column] = min(
                substitution,
                distance[row - 1][column] + 1,
                distance[row][column - 1] + 1,
            )
    insertion = deletion = substitution_count = 0
    row, column = len(reference), len(prediction)
    while row or column:
        if (
            row
            and column
            and reference[row - 1] == prediction[column - 1]
            and distance[row][column] == distance[row - 1][column - 1]
        ):
            row -= 1
            column -= 1
        elif row and column and distance[row][column] == distance[row - 1][column - 1] + 1:
            substitution_count += 1
            row -= 1
            column -= 1
        elif column and distance[row][column] == distance[row][column - 1] + 1:
            insertion += 1
            column -= 1
        else:
            deletion += 1
            row -= 1
    return insertion, deletion, substitution_count


def best_aligned_diagnostics(
    logits: torch.Tensor,
    reference_ids: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    blank_id: int,
    eos_id: int,
    complete: torch.Tensor | None = None,
    valid_token_mask: torch.Tensor | None = None,
    precomputed_log_probs: torch.Tensor | None = None,
    content_token_mask: torch.Tensor | None = None,
    max_right_shift: int | None = None,
) -> AlignmentDiagnostics:
    """Find the hard alignment with the best balanced formation accuracy.

    Blank correctness receives a per-example weight no greater than one, chosen
    so a large unused tail cannot outweigh all required tokens.  Edit metrics
    are computed independently after BLANK collapse and EOS termination.
    """

    _check_inputs(
        logits,
        reference_ids,
        reference_mask,
        blank_id=blank_id,
        eos_id=eos_id,
        complete=complete,
    )
    if max_right_shift is not None and max_right_shift < 0:
        raise ValueError("max_right_shift cannot be negative")
    compact, compact_mask, lengths = _compact_references(reference_ids, reference_mask)
    complete_mask = _resolve_complete(complete, compact, lengths, eos_id=eos_id)
    log_probs = _resolve_log_probs(
        logits, valid_token_mask, precomputed_log_probs
    ).detach()
    prediction = log_probs.argmax(dim=-1)
    blank_weights = _blank_weights(None, lengths, logits.shape[1], dtype=log_probs.dtype)
    blank_scores = blank_weights[:, None] * prediction.eq(blank_id).to(log_probs.dtype)
    emit_scores = log_probs.new_empty(logits.shape[0], logits.shape[1], compact.shape[1])
    for slot in range(logits.shape[1]):
        if compact.shape[1]:
            emit_scores[:, slot] = prediction[:, slot, None].eq(compact).to(log_probs.dtype)
    alignment = _viterbi_from_scores(
        blank_scores,
        emit_scores,
        compact,
        lengths,
        blank_id=blank_id,
        complete=complete_mask,
        max_right_shift=max_right_shift,
    )

    slot_match = prediction.eq(alignment.expanded_ids)
    formation_weights = torch.where(
        alignment.emission_mask,
        torch.ones_like(log_probs[:, :, 0]),
        blank_weights[:, None],
    )
    formation_weights = formation_weights * alignment.score_mask.to(
        formation_weights.dtype
    )
    formation_accuracy = (
        slot_match.to(log_probs.dtype) * formation_weights
    ).sum(1) / formation_weights.sum(1).clamp_min(1.0)

    if compact.shape[1]:
        aligned_prediction = prediction.gather(1, alignment.positions.clamp_min(0))
        required_match = aligned_prediction.eq(compact) & compact_mask
    else:
        required_match = compact_mask
    required_token_accuracy = required_match.sum(1).to(log_probs.dtype) / lengths.clamp_min(1)
    required_token_accuracy = torch.where(
        lengths.gt(0), required_token_accuracy, torch.ones_like(required_token_accuracy)
    )

    if content_token_mask is None:
        content_positions = compact_mask & compact.ne(eos_id) & compact.ne(blank_id)
    else:
        if content_token_mask.ndim != 1 or content_token_mask.shape[0] != logits.shape[-1]:
            raise ValueError("content_token_mask must have one entry per vocabulary item")
        content_positions = (
            compact_mask
            & compact.ne(eos_id)
            & compact.ne(blank_id)
            & content_token_mask.to(device=compact.device, dtype=torch.bool)[compact]
        )
    content_count = content_positions.sum(1)
    content_accuracy = (
        (required_match & content_positions).sum(1).to(log_probs.dtype)
        / content_count.clamp_min(1)
    )
    content_accuracy = torch.where(
        content_count.gt(0), content_accuracy, torch.ones_like(content_accuracy)
    )

    target_index = torch.arange(compact.shape[1], device=compact.device)[None]
    displacement = (alignment.positions - target_index).clamp_min(0)
    displacement = displacement * compact_mask.to(displacement.dtype)
    mean_displacement = displacement.sum(1).to(log_probs.dtype) / lengths.clamp_min(1)
    max_displacement = displacement.masked_fill(~compact_mask, 0).max(1).values.to(log_probs.dtype) \
        if compact.shape[1] else torch.zeros_like(mean_displacement)

    collapsed_prediction = collapse_blank_sequences(
        prediction,
        blank_id=blank_id,
        eos_id=eos_id,
        token_mask=alignment.score_mask,
    )
    collapsed_reference = collapse_blank_sequences(
        compact,
        blank_id=blank_id,
        eos_id=eos_id,
        token_mask=compact_mask,
    )
    insertions: list[int] = []
    deletions: list[int] = []
    substitutions: list[int] = []
    edit_accuracy: list[float] = []
    content_precision: list[float] = []
    content_recall: list[float] = []
    content_f1: list[float] = []
    rendered_exact: list[bool] = []
    allowed_content = (
        content_token_mask.detach().to(device="cpu", dtype=torch.bool).tolist()
        if content_token_mask is not None
        else None
    )
    for index, (target, predicted) in enumerate(
        zip(collapsed_reference.tokens, collapsed_prediction.tokens)
    ):
        insertion, deletion, substitution = _edit_counts(target, predicted)
        matches = max(len(target) - deletion - substitution, 0)
        insertions.append(insertion)
        deletions.append(deletion)
        substitutions.append(substitution)
        edit_accuracy.append(
            matches / max(matches + substitution + insertion + deletion, 1)
        )
        target_content = tuple(
            token
            for token in target
            if token not in {blank_id, eos_id}
            and (allowed_content is None or allowed_content[token])
        )
        predicted_content = tuple(
            token
            for token in predicted
            if token not in {blank_id, eos_id}
            and (allowed_content is None or allowed_content[token])
        )
        c_insert, c_delete, c_substitute = _edit_counts(
            target_content, predicted_content
        )
        c_matches = max(len(target_content) - c_delete - c_substitute, 0)
        precision_denominator = c_matches + c_substitute + c_insert
        recall_denominator = c_matches + c_substitute + c_delete
        precision = (
            c_matches / precision_denominator
            if precision_denominator
            else float(recall_denominator == 0)
        )
        recall = (
            c_matches / recall_denominator
            if recall_denominator
            else float(precision_denominator == 0)
        )
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        content_precision.append(precision)
        content_recall.append(recall)
        content_f1.append(f1)
        rendered_exact.append(
            bool(complete_mask[index])
            and bool(collapsed_prediction.terminated[index])
            and target == predicted
        )
    insertions_tensor = torch.tensor(insertions, dtype=torch.long, device=logits.device)
    deletions_tensor = torch.tensor(deletions, dtype=torch.long, device=logits.device)
    substitutions_tensor = torch.tensor(substitutions, dtype=torch.long, device=logits.device)
    edit_distance = insertions_tensor + deletions_tensor + substitutions_tensor
    return AlignmentDiagnostics(
        alignment=alignment,
        formation_accuracy=formation_accuracy,
        required_token_accuracy=required_token_accuracy,
        content_accuracy=content_accuracy,
        content_precision=torch.tensor(
            content_precision, dtype=log_probs.dtype, device=logits.device
        ),
        content_recall=torch.tensor(
            content_recall, dtype=log_probs.dtype, device=logits.device
        ),
        content_f1=torch.tensor(
            content_f1, dtype=log_probs.dtype, device=logits.device
        ),
        mean_displacement=mean_displacement,
        max_displacement=max_displacement,
        insertions=insertions_tensor,
        deletions=deletions_tensor,
        substitutions=substitutions_tensor,
        edit_distance=edit_distance,
        edit_accuracy=torch.tensor(edit_accuracy, dtype=log_probs.dtype, device=logits.device),
        rendered_exact=torch.tensor(rendered_exact, dtype=torch.bool, device=logits.device),
        rendered_exact_mask=complete_mask,
        predicted_lengths=torch.tensor(
            [len(row) for row in collapsed_prediction.tokens],
            dtype=torch.long,
            device=logits.device,
        ),
        reference_lengths=torch.tensor(
            [len(row) for row in collapsed_reference.tokens],
            dtype=torch.long,
            device=logits.device,
        ),
        predicted_terminated=collapsed_prediction.terminated,
    )


def sequence_quality_loss(
    logits: torch.Tensor,
    reference_ids: torch.Tensor,
    reference_mask: torch.Tensor,
    *,
    blank_id: int,
    eos_id: int,
    complete: torch.Tensor | None = None,
    valid_token_mask: torch.Tensor | None = None,
    precomputed_log_probs: torch.Tensor | None = None,
    semantic_token_mask: torch.Tensor | None = None,
    composition_weight: float = 1.0,
    length_weight: float = 0.25,
    eos_weight: float = 0.1,
    epsilon: float = 1e-8,
    reduction: Reduction = "mean",
) -> SequenceQualityResult:
    """Compare global semantic composition, emitted length, and EOS count.

    Composition is the Hellinger distance between normalized expected token
    counts and reference counts.  Callers may exclude whitespace and other
    nonsemantic pieces through ``semantic_token_mask``.  EOS count pressure is
    applied only to examples marked complete.  Complete targets receive a
    two-sided length loss; an incomplete target is right-censored and receives
    only an under-length penalty.
    """

    _check_inputs(
        logits,
        reference_ids,
        reference_mask,
        blank_id=blank_id,
        eos_id=eos_id,
        complete=complete,
    )
    if min(composition_weight, length_weight, eos_weight) < 0:
        raise ValueError("sequence-quality component weights cannot be negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    compact, compact_mask, lengths = _compact_references(reference_ids, reference_mask)
    complete_mask = _resolve_complete(complete, compact, lengths, eos_id=eos_id)
    log_probs = _resolve_log_probs(
        logits, valid_token_mask, precomputed_log_probs
    )
    probabilities = log_probs.exp()
    vocabulary = logits.shape[-1]

    if semantic_token_mask is None:
        semantic = torch.ones(vocabulary, dtype=torch.bool, device=logits.device)
    else:
        if semantic_token_mask.ndim != 1 or semantic_token_mask.shape[0] != vocabulary:
            raise ValueError("semantic_token_mask must have one entry per vocabulary item")
        semantic = semantic_token_mask.to(device=logits.device, dtype=torch.bool)
    if valid_token_mask is not None:
        semantic = semantic & valid_token_mask.to(device=logits.device, dtype=torch.bool)
    semantic = semantic.clone()
    semantic[blank_id] = False
    semantic[eos_id] = False

    target_counts = probabilities.new_zeros(logits.shape[0], vocabulary)
    target_counts.scatter_add_(
        1,
        compact,
        compact_mask.to(probabilities.dtype),
    )
    target_counts = target_counts * semantic[None]
    predicted_counts = probabilities.sum(1) * semantic[None]
    target_mass = target_counts.sum(1)
    predicted_mass = predicted_counts.sum(1)
    target_distribution = target_counts / target_mass[:, None].clamp_min(epsilon)
    predicted_distribution = predicted_counts / predicted_mass[:, None].clamp_min(epsilon)
    composition = 0.5 * (
        torch.sqrt(target_distribution + epsilon)
        - torch.sqrt(predicted_distribution + epsilon)
    ).square().sum(1)
    composition = torch.where(target_mass.gt(0), composition, torch.zeros_like(composition))
    # A clipped prefix does not reveal the semantic composition of its unknown
    # continuation. Applying a whole-window composition target there would
    # quietly turn right-censoring back into a stop/imitate-the-prefix signal.
    composition = composition * complete_mask.to(composition.dtype)

    expected_lengths = (1.0 - probabilities[:, :, blank_id]).sum(1)
    length_error = (expected_lengths - lengths.to(probabilities.dtype)) / logits.shape[1]
    # An incomplete reference is right-censored: its visible token count is a
    # lower bound, not a known endpoint.  Extra formation is therefore free,
    # while under-forming the observed prefix is still penalized.
    length_error = torch.where(
        complete_mask,
        length_error,
        length_error.clamp_max(0.0),
    )
    length_huber = F.smooth_l1_loss(
        length_error,
        torch.zeros_like(length_error),
        reduction="none",
    )
    expected_eos_counts = probabilities[:, :, eos_id].sum(1)
    eos_count = (expected_eos_counts - 1.0).square() * complete_mask.to(probabilities.dtype)

    per_example = (
        composition_weight * composition
        + length_weight * length_huber
        + eos_weight * eos_count
    )
    return SequenceQualityResult(
        loss=_reduce(per_example, reduction),
        semantic_composition=_reduce(composition, reduction),
        expected_length_huber=_reduce(length_huber, reduction),
        complete_eos_count=_reduce(eos_count, reduction),
        expected_lengths=expected_lengths,
        expected_eos_counts=expected_eos_counts,
    )


__all__ = [
    "AlignedCrossEntropyResult",
    "AlignmentDiagnostics",
    "CollapsedSequences",
    "SequenceQualityResult",
    "ViterbiAlignment",
    "best_aligned_diagnostics",
    "collapse_blank_sequences",
    "monotonic_forward_sum_nll",
    "sequence_quality_loss",
    "viterbi_aligned_ce",
    "viterbi_aligned_cross_entropy",
    "viterbi_expand_targets",
]
