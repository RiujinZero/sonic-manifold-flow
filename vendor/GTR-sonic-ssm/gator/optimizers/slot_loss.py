"""Task-independent slotwise loss results for forward-only optimization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class SlotLossResult:
    """Detached loss for every output slot plus its valid aggregate.

    ``losses`` and ``valid_mask`` always have shape ``[batch, output_slots]``.
    Invalid losses are stored as exact zeros so padding IDs never leak into
    diagnostics or a forward-only candidate score.
    """

    losses: Tensor
    valid_mask: Tensor
    aggregate: Tensor
    guard_aggregate: Tensor | None = None

    def validate(self) -> "SlotLossResult":
        """Validate the evaluator contract and return ``self``."""

        if self.losses.ndim != 2 or not self.losses.is_floating_point():
            raise TypeError("losses must be floating point [batch, output_slots]")
        if self.valid_mask.shape != self.losses.shape:
            raise ValueError("valid_mask must match losses")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")
        if self.aggregate.ndim != 0 or not self.aggregate.is_floating_point():
            raise TypeError("aggregate must be a floating-point scalar")
        if self.losses.requires_grad or self.losses.grad_fn is not None:
            raise RuntimeError("slot losses must be detached from autograd")
        if self.aggregate.requires_grad or self.aggregate.grad_fn is not None:
            raise RuntimeError("aggregate loss must be detached from autograd")
        if self.losses.device != self.valid_mask.device:
            raise ValueError("losses and valid_mask must share a device")
        if self.aggregate.device != self.losses.device:
            raise ValueError("aggregate and slot losses must share a device")
        if not bool(self.valid_mask.any()):
            raise ValueError("at least one slot must be valid")
        valid_losses = self.losses[self.valid_mask]
        invalid_losses = self.losses[~self.valid_mask]
        if invalid_losses.numel() and (
            not bool(torch.isfinite(invalid_losses).all())
            or not bool(invalid_losses.eq(0).all())
        ):
            raise ValueError("invalid slot losses must be finite exact zeros")
        if not bool(torch.isfinite(valid_losses).all()):
            raise FloatingPointError("valid slot losses must be finite")
        if bool((valid_losses < 0).any()):
            raise ValueError("slot losses cannot be negative")
        if not bool(torch.isfinite(self.aggregate)):
            raise FloatingPointError("aggregate loss must be finite")
        if self.guard_aggregate is not None:
            if (
                self.guard_aggregate.ndim != 0
                or not self.guard_aggregate.is_floating_point()
                or self.guard_aggregate.device != self.losses.device
            ):
                raise TypeError(
                    "guard_aggregate must be a floating-point scalar on "
                    "the losses device"
                )
            if not bool(torch.isfinite(self.guard_aggregate)):
                raise FloatingPointError("guard aggregate must be finite")
            if (
                self.guard_aggregate.requires_grad
                or self.guard_aggregate.grad_fn is not None
            ):
                raise RuntimeError(
                    "guard aggregate must be detached from autograd"
                )
        expected = valid_losses.mean()
        if not torch.allclose(
            self.aggregate,
            expected,
            atol=1e-6,
            rtol=1e-5,
        ):
            raise ValueError("aggregate must equal the mean valid slot loss")
        return self


def cross_entropy_slot_loss(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor | None = None,
) -> SlotLossResult:
    """Return ordinary cross-entropy independently for every valid slot."""

    if logits.ndim != 3 or not logits.is_floating_point():
        raise TypeError("logits must be floating point [batch, slots, vocabulary]")
    batch, slots, vocabulary = logits.shape
    if (
        targets.shape != (batch, slots)
        or targets.dtype != torch.long
        or targets.device != logits.device
    ):
        raise TypeError("targets must be long [batch, slots] on the logits device")
    if valid_mask is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        if (
            valid_mask.shape != targets.shape
            or valid_mask.dtype != torch.bool
            or valid_mask.device != logits.device
        ):
            raise TypeError(
                "valid_mask must be boolean, match targets, and share its device"
            )
        valid = valid_mask
    if not bool(valid.any()):
        raise ValueError("at least one slot must be valid")

    # Padding targets are allowed to contain sentinels outside the vocabulary.
    # Replace them before CE and then zero their losses exactly.
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    valid_targets = safe_targets[valid]
    if bool(((valid_targets < 0) | (valid_targets >= vocabulary)).any()):
        raise ValueError("valid targets must lie inside the vocabulary")
    ce_logits = (
        logits.float()
        if logits.dtype in (torch.float16, torch.bfloat16)
        else logits
    )
    losses = F.cross_entropy(
        ce_logits.reshape(batch * slots, vocabulary),
        safe_targets.reshape(batch * slots),
        reduction="none",
    ).reshape(batch, slots)
    losses = torch.where(valid, losses, torch.zeros_like(losses)).detach()
    aggregate = losses[valid].mean().detach()
    return SlotLossResult(losses, valid.detach().clone(), aggregate).validate()


def batch_centered_cross_entropy_slot_loss(
    logits: Tensor,
    targets: Tensor,
    valid_mask: Tensor | None = None,
) -> SlotLossResult:
    """Score only input-dependent logits within a balanced microbatch.

    For every physical slot and vocabulary item, subtract the mean logit
    across valid examples before computing CE. A batch-constant token bias
    can therefore neither help nor hurt candidate selection. On an exactly
    target-balanced batch, improvements must come from variation correlated
    with each example rather than from fitting the target marginal.
    """

    if logits.ndim != 3 or not logits.is_floating_point():
        raise TypeError("logits must be floating point [batch, slots, vocabulary]")
    batch, slots, _ = logits.shape
    if (
        targets.shape != (batch, slots)
        or targets.dtype != torch.long
        or targets.device != logits.device
    ):
        raise TypeError("targets must be long [batch, slots] on the logits device")
    if valid_mask is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        if (
            valid_mask.shape != targets.shape
            or valid_mask.dtype != torch.bool
            or valid_mask.device != logits.device
        ):
            raise TypeError(
                "valid_mask must be boolean, match targets, and share its device"
            )
        valid = valid_mask
    counts = valid.sum(dim=0)
    populated = counts.gt(0)
    if not bool(populated.any()):
        raise ValueError("at least one slot must be valid")
    if bool(counts[populated].lt(2).any()):
        raise ValueError(
            "batch-centered slot loss requires at least two valid examples "
            "for every populated slot"
        )
    weights = valid.unsqueeze(-1).to(dtype=logits.dtype)
    mean_logits = (logits * weights).sum(dim=0) / counts.clamp_min(1).view(
        slots, 1
    )
    centered_logits = logits - mean_logits.unsqueeze(0)
    centered = cross_entropy_slot_loss(centered_logits, targets, valid)
    ordinary = cross_entropy_slot_loss(logits, targets, valid)
    return SlotLossResult(
        centered.losses,
        centered.valid_mask,
        centered.aggregate,
        ordinary.aggregate,
    ).validate()


def batch_mean_slot_loss(result: SlotLossResult) -> SlotLossResult:
    """Average examples while preserving independent physical-slot losses.

    Procoim protects objective rows independently. Treating every
    ``(example, slot)`` as a separate objective makes larger microbatches
    multiply the number of states that must freeze and remain protected.
    This reduction instead produces one objective per physical output slot:
    valid examples are averaged independently for each slot and slots with
    no valid examples remain exact-zero padding.
    """

    result.validate()
    valid = result.valid_mask
    counts = valid.sum(dim=0)
    reduced_valid = counts.gt(0)
    safe_counts = counts.clamp_min(1).to(dtype=result.losses.dtype)
    summed = torch.where(
        valid,
        result.losses,
        torch.zeros_like(result.losses),
    ).sum(dim=0)
    reduced_losses = torch.where(
        reduced_valid,
        summed / safe_counts,
        torch.zeros_like(summed),
    ).unsqueeze(0).detach()
    reduced_mask = reduced_valid.unsqueeze(0).detach()
    aggregate = reduced_losses[reduced_mask].mean().detach()
    return SlotLossResult(
        reduced_losses,
        reduced_mask,
        aggregate,
        result.guard_aggregate,
    ).validate()
