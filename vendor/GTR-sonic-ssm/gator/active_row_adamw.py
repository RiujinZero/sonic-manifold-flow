"""Experimental AdamW semantics for row-indexed private node parameters.

This module is deliberately standalone.  It does not change Gator's default
optimizer or silently reinterpret existing checkpoints.  It exists to make
the optimizer trade-off around sparse node execution explicit:

``exact``
    Mathematically matches dense AdamW.  A parameter with a gradient updates
    every row, including rows whose current gradient is zero, because old
    Adam momentum and decoupled weight decay still move those rows.  Sparse
    gradients are densified.  This is a correctness oracle, not a speedup.

``lazy``
    Advances only active rows.  Inactive parameters, moments, weight decay,
    and row-local Adam clocks are frozen.  This can reduce optimizer compute,
    but it is intentionally a different optimizer and therefore an
    architecture/training ablation rather than a transparent optimization.

Both modes retain dense first- and second-moment tensors.  Consequently lazy
execution reduces optimizer arithmetic and gradient storage when paired with
``sparse_row_gather``; it does *not* reduce the asymptotic Adam state memory.
The lazy mode adds one int64 clock per parameter row.  A paged moment store
could reduce state memory for never-visited nodes, but would need a fused GPU
lookup/update kernel to avoid replacing tensor work with Python dispatch.

Checkpoint format
-----------------
The normal :class:`torch.optim.Optimizer` ``state`` and ``param_groups`` are
preserved.  A top-level ``active_row_adamw`` record identifies format version
1, the semantic mode, moment layout, and clock layout.  Route activity is
ephemeral and is never checkpointed.  Loading a checkpoint into an optimizer
with different semantics is rejected rather than silently changing training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import torch
from torch import nn
from torch.optim import Optimizer


ActiveRowMode = Literal["exact", "lazy"]


class _SparseRowGather(torch.autograd.Function):
    """Gather dense rows while returning a one-sparse-dimension gradient."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        weight: torch.Tensor,
        rows: torch.Tensor,
    ) -> torch.Tensor:
        if weight.ndim < 1:
            raise ValueError("weight must have a row dimension")
        if rows.dtype != torch.long:
            raise TypeError("rows must use torch.long indices")
        if rows.device != weight.device:
            raise ValueError("rows and weight must be on the same device")
        flat_rows = rows.reshape(-1)
        # ``index_select`` performs the device-side bounds check.  Avoid a
        # Python ``bool(tensor.any())`` here because it would synchronize every
        # CUDA relay merely to duplicate that validation.
        gathered = weight.index_select(0, flat_rows)
        ctx.save_for_backward(flat_rows)
        ctx.weight_shape = tuple(weight.shape)
        ctx.row_shape = tuple(rows.shape)
        return gathered.reshape(*rows.shape, *weight.shape[1:])

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        (flat_rows,) = ctx.saved_tensors
        values = gradient.reshape(-1, *ctx.weight_shape[1:])
        # Forward ``index_select`` already established that every saved row is
        # in range, while the shapes are built locally.  Explicitly disabling
        # duplicate invariant checks avoids a host synchronization and also
        # silences a misleading warning emitted by some PyTorch builds when
        # the constructor keyword is used directly.
        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            sparse_gradient = torch.sparse_coo_tensor(
                flat_rows.unsqueeze(0),
                values,
                size=ctx.weight_shape,
                dtype=values.dtype,
                device=values.device,
                is_coalesced=False,
            )
        return sparse_gradient, None


def sparse_row_gather(
    weight: torch.Tensor,
    rows: torch.Tensor,
) -> torch.Tensor:
    """Return ``weight[rows]`` with a row-sparse gradient for ``weight``.

    PyTorch's ordinary indexed lookup allocates a dense gradient shaped like
    the complete node table.  ``torch.nn.functional.embedding(...,
    sparse=True)`` avoids that allocation only for two-dimensional tables;
    reshaping a sparse embedding gradient back to ``[N,D,D]`` is unsupported.
    This custom gather directly emits a COO tensor with one sparse dimension
    and arbitrary dense row dimensions, so existing ``[N,D,D]`` node weights
    do not need a checkpoint-breaking flattening migration.

    The helper is first-order only in its intended use.  A production rollout
    must separately validate its ``torch.compile`` and target-GPU behavior.
    """

    return _SparseRowGather.apply(weight, rows)


def active_rows_from_routes(
    nodes: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    mass: torch.Tensor | None = None,
    minimum_mass: float = 0.0,
) -> torch.Tensor:
    """Return sorted unique node rows that actually participated in a route."""

    if nodes.dtype != torch.long:
        raise TypeError("nodes must use torch.long indices")
    if minimum_mass < 0.0:
        raise ValueError("minimum_mass cannot be negative")
    selected = torch.ones_like(nodes, dtype=torch.bool)
    if valid is not None:
        if valid.shape != nodes.shape or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean tensor shaped like nodes")
        if valid.device != nodes.device:
            raise ValueError("valid and nodes must be on the same device")
        selected &= valid
    if mass is not None:
        if mass.shape != nodes.shape or not mass.is_floating_point():
            raise ValueError("mass must be floating point and shaped like nodes")
        if mass.device != nodes.device:
            raise ValueError("mass and nodes must be on the same device")
        selected &= mass > minimum_mass
    return torch.unique(nodes[selected], sorted=True)


@dataclass(frozen=True)
class ActiveRowStateEstimate:
    """Asymptotic optimizer-state footprint, excluding model parameters."""

    parameter_elements: int
    moment_elements: int
    moment_bytes: int
    clock_elements: int
    clock_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.moment_bytes + self.clock_bytes


def estimate_active_row_state(
    parameters: Iterable[nn.Parameter],
    *,
    mode: ActiveRowMode,
) -> ActiveRowStateEstimate:
    """Estimate fully materialized state for :class:`ActiveRowAdamW`.

    Lazy state still has two dense moment tensors.  It differs only by storing
    one int64 step counter per row instead of one float32 scalar per parameter.
    """

    parameters = list(parameters)
    if mode not in ("exact", "lazy"):
        raise ValueError(f"unsupported active-row mode: {mode}")
    for parameter in parameters:
        if parameter.ndim < 1:
            raise ValueError("active-row parameters need a row dimension")
    parameter_elements = sum(parameter.numel() for parameter in parameters)
    moment_bytes = 2 * sum(
        parameter.numel() * parameter.element_size()
        for parameter in parameters
    )
    clock_elements = (
        len(parameters)
        if mode == "exact"
        else sum(parameter.shape[0] for parameter in parameters)
    )
    clock_bytes = clock_elements * (4 if mode == "exact" else 8)
    return ActiveRowStateEstimate(
        parameter_elements=parameter_elements,
        moment_elements=2 * parameter_elements,
        moment_bytes=moment_bytes,
        clock_elements=clock_elements,
        clock_bytes=clock_bytes,
    )


class ActiveRowAdamW(Optimizer):
    """AdamW for row-indexed tensors with explicit exact or lazy semantics.

    In lazy mode, a dense gradient requires an explicit call to
    :meth:`mark_active_rows` before :meth:`step`.  A gradient produced by
    :func:`sparse_row_gather` carries its row activity automatically.  Multiple
    calls accumulate a union, which supports several relays and gradient
    accumulation before one optimizer step.

    ``strict_activity=True`` scans a dense gradient and rejects nonzero rows
    outside the declared support.  That diagnostic is intentionally optional:
    scanning the full node table would erase part of the intended speedup.
    """

    FORMAT_VERSION = 1

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        mode: ActiveRowMode = "lazy",
        strict_activity: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError("learning rate cannot be negative")
        if eps <= 0.0:
            raise ValueError("epsilon must be positive")
        if weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative")
        if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if mode not in ("exact", "lazy"):
            raise ValueError(f"unsupported active-row mode: {mode}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "active_row_mode": mode,
        }
        super().__init__(params, defaults)
        self.mode: ActiveRowMode = mode
        self.strict_activity = strict_activity
        self._activity: dict[int, list[torch.Tensor]] = {}
        self._parameters_by_id: dict[int, nn.Parameter] = {}
        for group in self.param_groups:
            if group["active_row_mode"] != mode:
                raise ValueError("all parameter groups must use the optimizer mode")
            for parameter in group["params"]:
                if parameter.ndim < 1:
                    raise ValueError(
                        "ActiveRowAdamW parameters need a row dimension"
                    )
                self._parameters_by_id[id(parameter)] = parameter

    def mark_active_rows(
        self,
        parameter: nn.Parameter,
        rows: torch.Tensor,
        *,
        valid: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
        minimum_mass: float = 0.0,
    ) -> None:
        """Accumulate row activity for one parameter until the next step."""

        registered = self._parameters_by_id.get(id(parameter))
        if registered is not parameter:
            raise ValueError("parameter is not registered with this optimizer")
        if rows.device != parameter.device:
            raise ValueError("row activity and parameter must share a device")
        selected = active_rows_from_routes(
            rows,
            valid=valid,
            mass=mass,
            minimum_mass=minimum_mass,
        )
        self._activity.setdefault(id(parameter), []).append(selected)

    def _declared_rows(self, parameter: nn.Parameter) -> torch.Tensor | None:
        declarations = self._activity.get(id(parameter))
        if declarations is None:
            return None
        if len(declarations) == 1:
            return declarations[0]
        return torch.unique(torch.cat(declarations), sorted=True)

    @staticmethod
    def _initialize_exact_state(
        state: dict,
        parameter: nn.Parameter,
    ) -> None:
        if state:
            return
        state["step"] = torch.zeros(
            (), dtype=torch.float32, device=parameter.device
        )
        state["exp_avg"] = torch.zeros_like(
            parameter, memory_format=torch.preserve_format
        )
        state["exp_avg_sq"] = torch.zeros_like(
            parameter, memory_format=torch.preserve_format
        )

    @staticmethod
    def _initialize_lazy_state(
        state: dict,
        parameter: nn.Parameter,
    ) -> None:
        if state:
            return
        state["row_step"] = torch.zeros(
            parameter.shape[0], dtype=torch.int64, device=parameter.device
        )
        state["exp_avg"] = torch.zeros_like(
            parameter, memory_format=torch.preserve_format
        )
        state["exp_avg_sq"] = torch.zeros_like(
            parameter, memory_format=torch.preserve_format
        )

    @staticmethod
    def _exact_update(
        parameter: nn.Parameter,
        gradient: torch.Tensor,
        state: dict,
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        ActiveRowAdamW._initialize_exact_state(state, parameter)
        state["step"].add_(1)
        step = int(state["step"].item())
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        parameter.mul_(1.0 - lr * weight_decay)
        exp_avg.lerp_(gradient, 1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(
            gradient, gradient, value=1.0 - beta2
        )
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        denominator = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
        parameter.addcdiv_(
            exp_avg,
            denominator,
            value=-(lr / bias_correction1),
        )

    def _selected_gradient(
        self,
        parameter: nn.Parameter,
        gradient: torch.Tensor,
        rows: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if gradient.is_sparse:
            gradient = gradient.coalesce()
            if gradient.sparse_dim() != 1 or gradient.dense_dim() != (
                parameter.ndim - 1
            ):
                raise ValueError(
                    "sparse gradients need one sparse row dimension and "
                    "the remaining parameter dimensions dense"
                )
            gradient_rows = gradient.indices()[0]
            gradient_values = gradient.values()
            if rows is None:
                return gradient_rows, gradient_values
            selected_gradient = torch.zeros(
                (rows.numel(), *parameter.shape[1:]),
                dtype=gradient.dtype,
                device=gradient.device,
            )
            if gradient_rows.numel() == 0:
                return rows, selected_gradient
            positions = torch.searchsorted(rows, gradient_rows)
            in_range = positions < rows.numel()
            matches = torch.zeros_like(in_range)
            matches[in_range] = (
                rows[positions[in_range]] == gradient_rows[in_range]
            )
            if not bool(matches.all()):
                missing = gradient_rows[~matches]
                raise ValueError(
                    "sparse gradient contains undeclared active rows: "
                    f"{missing.detach().cpu().tolist()}"
                )
            selected_gradient.index_add_(0, positions, gradient_values)
            return rows, selected_gradient

        if rows is None:
            raise RuntimeError(
                "lazy ActiveRowAdamW requires mark_active_rows() for a "
                "dense gradient; use sparse_row_gather() to carry activity "
                "in the gradient instead"
            )
        if self.strict_activity:
            nonzero = gradient.detach().reshape(parameter.shape[0], -1).ne(0)
            nonzero = nonzero.any(dim=-1)
            declared = torch.zeros_like(nonzero)
            declared[rows] = True
            unexpected = torch.nonzero(nonzero & ~declared).flatten()
            if unexpected.numel() > 0:
                raise ValueError(
                    "dense gradient contains undeclared nonzero rows: "
                    f"{unexpected.detach().cpu().tolist()}"
                )
        return rows, gradient.index_select(0, rows)

    @staticmethod
    def _lazy_update(
        parameter: nn.Parameter,
        rows: torch.Tensor,
        gradient: torch.Tensor,
        state: dict,
        *,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
        weight_decay: float,
    ) -> None:
        if rows.numel() == 0:
            return
        ActiveRowAdamW._initialize_lazy_state(state, parameter)
        row_step = state["row_step"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        parameter_rows = parameter.index_select(0, rows)
        average_rows = exp_avg.index_select(0, rows)
        square_rows = exp_avg_sq.index_select(0, rows)
        steps = row_step.index_select(0, rows).add_(1)

        parameter_rows.mul_(1.0 - lr * weight_decay)
        average_rows.lerp_(gradient, 1.0 - beta1)
        square_rows.mul_(beta2).addcmul_(
            gradient, gradient, value=1.0 - beta2
        )

        correction_dtype = (
            torch.float64
            if parameter.dtype == torch.float64
            else torch.float32
        )
        floating_steps = steps.to(dtype=correction_dtype)
        correction1 = 1.0 - torch.pow(
            torch.tensor(beta1, dtype=correction_dtype, device=parameter.device),
            floating_steps,
        )
        correction2 = 1.0 - torch.pow(
            torch.tensor(beta2, dtype=correction_dtype, device=parameter.device),
            floating_steps,
        )
        broadcast_shape = (rows.numel(),) + (1,) * (parameter.ndim - 1)
        step_size = (lr / correction1).reshape(broadcast_shape)
        denominator = square_rows.sqrt().div_(
            correction2.sqrt().reshape(broadcast_shape)
        ).add_(eps)
        parameter_rows.add_(-step_size * average_rows / denominator)

        parameter.index_copy_(0, rows, parameter_rows)
        exp_avg.index_copy_(0, rows, average_rows)
        exp_avg_sq.index_copy_(0, rows, square_rows)
        row_step.index_copy_(0, rows, steps)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one exact-dense or explicit lazy-active AdamW step."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        try:
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                for parameter in group["params"]:
                    gradient = parameter.grad
                    if gradient is None:
                        continue
                    if gradient.is_complex() or parameter.is_complex():
                        raise RuntimeError(
                            "ActiveRowAdamW does not support complex parameters"
                        )
                    state = self.state[parameter]
                    if self.mode == "exact":
                        dense_gradient = (
                            gradient.to_dense() if gradient.is_sparse else gradient
                        )
                        self._exact_update(
                            parameter,
                            dense_gradient,
                            state,
                            lr=group["lr"],
                            beta1=beta1,
                            beta2=beta2,
                            eps=group["eps"],
                            weight_decay=group["weight_decay"],
                        )
                    else:
                        rows = self._declared_rows(parameter)
                        rows, selected_gradient = self._selected_gradient(
                            parameter,
                            gradient,
                            rows,
                        )
                        self._lazy_update(
                            parameter,
                            rows,
                            selected_gradient,
                            state,
                            lr=group["lr"],
                            beta1=beta1,
                            beta2=beta2,
                            eps=group["eps"],
                            weight_decay=group["weight_decay"],
                        )
        finally:
            # Route tensors describe exactly one optimizer step.  Retaining
            # them after success or failure risks stale-row updates and keeps
            # potentially large relay supports alive unnecessarily.
            self._activity.clear()
        return loss

    def state_dict(self) -> dict:
        payload = super().state_dict()
        payload["active_row_adamw"] = {
            "format_version": self.FORMAT_VERSION,
            "mode": self.mode,
            "moment_storage": "dense",
            "clock_storage": (
                "scalar_per_parameter"
                if self.mode == "exact"
                else "int64_per_row"
            ),
            "activity_persisted": False,
        }
        return payload

    def load_state_dict(self, state_dict: dict) -> None:
        # Do not deep-copy moment tensors while loading a large checkpoint.
        # The optimizer loader owns any value conversion it needs; this shallow
        # top-level copy is only so removing metadata does not mutate callers.
        payload = dict(state_dict)
        metadata = payload.pop("active_row_adamw", None)
        if metadata is None:
            raise ValueError("checkpoint is missing active_row_adamw metadata")
        if metadata.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("unsupported ActiveRowAdamW checkpoint version")
        if metadata.get("mode") != self.mode:
            raise ValueError(
                "optimizer semantic mismatch: checkpoint uses "
                f"{metadata.get('mode')!r}, optimizer uses {self.mode!r}"
            )
        super().load_state_dict(payload)
        # Optimizer's generic loader casts unknown state tensors to the
        # parameter dtype.  ``row_step`` is a semantic integer clock, not a
        # floating-point moment, so restore its declared checkpoint layout.
        if self.mode == "lazy":
            for state in self.state.values():
                if "row_step" in state:
                    state["row_step"] = state["row_step"].to(torch.int64)
        self._activity.clear()
