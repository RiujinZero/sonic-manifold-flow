"""Generic optimizer and learning-rate scheduling utilities for Gator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Iterable, Literal

import torch
from torch import nn


OptimizerName = Literal["adamw", "adam", "sgd"]
SchedulerName = Literal["constant", "cosine", "milestone"]


def _nonfinite_gradient_summary(
    model: nn.Module,
    *,
    limit: int = 8,
) -> str:
    """Describe invalid gradients after the aggregate finite check fails."""

    failures: list[str] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient._values() if gradient.is_sparse else gradient
        finite = torch.isfinite(values)
        if bool(finite.all()):
            continue
        invalid = ~finite
        failures.append(
            f"{name} shape={tuple(gradient.shape)} "
            f"invalid={int(invalid.sum())}/{values.numel()} "
            f"nan={int(torch.isnan(values).sum())} "
            f"+inf={int(torch.isposinf(values).sum())} "
            f"-inf={int(torch.isneginf(values).sum())}"
        )
        if len(failures) >= limit:
            break
    return (
        "; ".join(failures)
        if failures
        else "all gradient elements are finite; aggregate norm overflow suspected"
    )


def _stable_gradient_norm(model: nn.Module) -> float:
    """Return the global L2 gradient norm with FP64 accumulation.

    PyTorch's default FP32 clipping path can report ``inf`` when every
    individual gradient element is finite but their squared reduction
    overflows. Per-tensor FP64 reductions avoid allocating FP64 gradient
    copies while preserving enough range for a reliable clipping decision.
    """

    squared_by_device: dict[torch.device, torch.Tensor] = {}
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient._values() if gradient.is_sparse else gradient
        norm = torch.linalg.vector_norm(
            values.detach(),
            ord=2,
            dtype=torch.float64,
        )
        squared = norm.square()
        previous = squared_by_device.get(squared.device)
        squared_by_device[squared.device] = (
            squared if previous is None else previous + squared
        )
    if not squared_by_device:
        return 0.0
    squared_total = math.fsum(
        float(value.detach().cpu())
        for value in squared_by_device.values()
    )
    return math.sqrt(squared_total)


@torch.no_grad()
def _scale_gradients(model: nn.Module, coefficient: float) -> None:
    """Scale dense gradients through foreach kernels and sparse values safely."""

    dense_groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    sparse_values: list[torch.Tensor] = []
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        if gradient.is_sparse:
            sparse_values.append(gradient._values())
        else:
            dense_groups.setdefault(
                (gradient.device, gradient.dtype),
                [],
            ).append(gradient)
    for gradients in dense_groups.values():
        torch._foreach_mul_(gradients, coefficient)
    for values in sparse_values:
        values.mul_(coefficient)


@dataclass(frozen=True)
class SchedulerConfig:
    """Learning-rate schedule independent of any particular model."""

    name: SchedulerName = "constant"
    total_steps: int = 20_000
    minimum_lr: float = 0.0
    milestones: tuple[int, ...] = ()
    gamma: float = 0.25

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.minimum_lr < 0:
            raise ValueError("minimum_lr cannot be negative")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if any(step <= 0 for step in self.milestones):
            raise ValueError("milestones must contain positive step numbers")
        if tuple(sorted(set(self.milestones))) != self.milestones:
            raise ValueError("milestones must be sorted and unique")


@dataclass(frozen=True)
class OptimizationConfig:
    """Serializable optimizer settings shared by training entrypoints."""

    name: OptimizerName = "adamw"
    learning_rate: float = 8e-3
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    momentum: float = 0.9
    gradient_clip: float | None = 1.0
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive or None")

    def to_dict(self) -> dict:
        return asdict(self)


def build_optimizer(
    parameters: Iterable[nn.Parameter], config: OptimizationConfig
) -> torch.optim.Optimizer:
    """Build an optimizer without coupling it to a model implementation."""

    parameters = list(parameters)
    if not parameters:
        raise ValueError("cannot build an optimizer with no parameters")
    if config.name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
        )
    if config.name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.epsilon,
            weight_decay=config.weight_decay,
        )
    if config.name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"unsupported optimizer: {config.name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: OptimizationConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Return None for a constant LR, otherwise a step-based scheduler."""

    schedule = config.scheduler
    if schedule.name == "constant":
        return None
    if schedule.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=schedule.total_steps,
            eta_min=schedule.minimum_lr,
        )
    if schedule.name == "milestone":
        if not schedule.milestones:
            raise ValueError("milestone scheduler requires at least one milestone")
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(schedule.milestones),
            gamma=schedule.gamma,
        )
    raise ValueError(f"unsupported scheduler: {schedule.name}")


def accumulate_gradients(
    loss: torch.Tensor,
    *,
    scale: float = 1.0,
) -> None:
    """Backpropagate one micro-batch's loss without stepping the optimizer.

    Callers doing gradient accumulation zero the optimizer once per
    accumulation window (not once per micro-batch) and call this once per
    micro-batch, then :func:`apply_optimizer_step` once the window is full.
    ``scale`` should normally be ``1 / accumulation_steps`` so the summed
    gradient over the window matches a true large-batch average rather than
    a sum that grows with the number of micro-batches.
    """

    if not bool(torch.isfinite(loss.detach()).all()):
        raise FloatingPointError("refusing to accumulate a non-finite loss")
    (loss * scale).backward()


def apply_optimizer_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: OptimizationConfig,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    """Clip already-accumulated gradients, step, and return the gradient norm.

    Assumes gradients for this update are already populated (via one or more
    prior :func:`accumulate_gradients` calls) and have not yet been zeroed.
    """

    if config.gradient_clip is None:
        gradient_norm = _stable_gradient_norm(model)
    else:
        try:
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip,
                    error_if_nonfinite=True,
                )
            )
        except RuntimeError as error:
            if "non-finite" not in str(error):
                raise
            # A finite FP32 gradient can still overflow while its L2 norm is
            # squared and reduced. Recompute only this exceptional path in
            # FP64, preserving the fast foreach clip during ordinary steps.
            gradient_norm = _stable_gradient_norm(model)
            if not math.isfinite(gradient_norm):
                summary = _nonfinite_gradient_summary(model)
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "refusing optimizer step with non-finite gradient norm; "
                    f"gradients: {summary}"
                ) from error
            if gradient_norm > config.gradient_clip:
                _scale_gradients(
                    model,
                    config.gradient_clip / (gradient_norm + 1e-6),
                )
    if not math.isfinite(gradient_norm):
        summary = _nonfinite_gradient_summary(model)
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError(
            "refusing optimizer step with non-finite gradient norm; "
            f"gradients: {summary}"
        )
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return gradient_norm


def optimization_step(
    *,
    loss: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: OptimizationConfig,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    """Backpropagate, clip, update, and return the pre-clipping gradient norm.

    Equivalent to one accumulation window of size 1: zero, accumulate this
    loss, then step. Kept for callers that do not use gradient accumulation.
    """

    optimizer.zero_grad(set_to_none=True)
    accumulate_gradients(loss, scale=1.0)
    return apply_optimizer_step(
        model=model,
        optimizer=optimizer,
        config=config,
        scheduler=scheduler,
    )
