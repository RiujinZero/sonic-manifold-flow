"""Reversible additive fast-weight deltas layered on trusted slow GTR-L weights.

Procoim-S Phase 1 (see ``Papers/GTRL_PROCOIM_TRACE.md`` and the Procoim-S
design specification it links). The live model is conceptually::

    theta_live = theta_slow + delta_fast

``theta_slow`` is whatever the model's own parameters currently hold; this
module never mutates them outside of an explicit, narrow window (either the
temporary ``apply_fast_delta`` context, or an explicit ``commit_`` call that
is meant to represent consolidation). ``delta_fast`` is represented here as
one :class:`FastWeightDelta` per Procoim parameter group, each holding fully
materialized per-parameter delta tensors (not coefficients over a
regenerable basis) so that applying and removing a delta is exact and does
not depend on Procoim's own RNG state.

This module has no dependency on surprise memory, replay, probation, or
consolidation policy — it is purely the reversible container described in
Procoim-S design spec, Phase 1 (``fast_weights.py`` in the proposed file
layout, section 21).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

__all__ = [
    "FastWeightDelta",
    "FastWeightState",
    "apply_fast_delta",
]


@dataclass
class FastWeightDelta:
    """One parameter group's additive contribution to ``delta_fast``.

    ``values`` maps parameter name -> delta tensor, matching the shape,
    dtype, and device of the live parameter it will be added to.
    """

    group: str
    values: dict[str, Tensor] = field(default_factory=dict)

    def clone(self) -> "FastWeightDelta":
        return FastWeightDelta(
            group=self.group,
            values={name: value.clone() for name, value in self.values.items()},
        )

    def norm(self) -> float:
        if not self.values:
            return 0.0
        total = torch.zeros((), dtype=torch.float64)
        for value in self.values.values():
            total = total + value.detach().double().pow(2).sum()
        return float(total.sqrt())

    def scale_(self, factor: float) -> "FastWeightDelta":
        for value in self.values.values():
            value.mul_(factor)
        return self

    def add_(self, other: "FastWeightDelta", *, alpha: float = 1.0) -> "FastWeightDelta":
        """Merge ``other`` into this delta in place: ``self += alpha * other``."""

        if other.group != self.group:
            raise ValueError(
                f"cannot merge delta for group {other.group!r} into "
                f"{self.group!r} — groups must match"
            )
        for name, value in other.values.items():
            if name in self.values:
                if self.values[name].shape != value.shape:
                    raise ValueError(
                        f"shape mismatch merging delta for parameter {name!r}"
                    )
                self.values[name].add_(value, alpha=alpha)
            else:
                self.values[name] = value.detach().clone().mul_(alpha)
        return self

    def is_zero(self) -> bool:
        return not self.values or all(
            bool(torch.count_nonzero(value) == 0) for value in self.values.values()
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "group": self.group,
            "values": {name: value.detach().clone() for name, value in self.values.items()},
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "FastWeightDelta":
        group = state["group"]
        values = state["values"]
        if not isinstance(group, str) or not isinstance(values, Mapping):
            raise ValueError("malformed FastWeightDelta state_dict")
        return cls(
            group=group,
            values={name: value.clone() for name, value in values.items()},
        )


class FastWeightState:
    """Manages ``delta_fast`` for a model: ``theta_live = theta_slow + delta_fast``.

    Deltas are stored per parameter group so that decay, norm inspection,
    and consolidation can operate on one group at a time (matching Procoim's
    existing per-group bookkeeping) while :meth:`combined_deltas` gives the
    total additive change to any single named parameter, summed across
    every group that touches it.
    """

    def __init__(self) -> None:
        self._deltas: dict[str, FastWeightDelta] = {}

    def __contains__(self, group: str) -> bool:
        return group in self._deltas

    def __iter__(self) -> Iterator[str]:
        return iter(self._deltas)

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(self._deltas)

    @property
    def is_empty(self) -> bool:
        return all(delta.is_zero() for delta in self._deltas.values())

    def get_group(self, group: str) -> FastWeightDelta | None:
        return self._deltas.get(group)

    def add_group(self, delta: FastWeightDelta, *, alpha: float = 1.0) -> None:
        """Merge ``delta`` into this group's accumulated fast weight."""

        existing = self._deltas.get(delta.group)
        if existing is None:
            self._deltas[delta.group] = delta.clone().scale_(alpha)
        else:
            existing.add_(delta, alpha=alpha)

    def remove_group(self, group: str) -> None:
        self._deltas.pop(group, None)

    def clear(self) -> None:
        self._deltas.clear()

    def norm(self, group: str | None = None) -> float:
        if group is not None:
            delta = self._deltas.get(group)
            return delta.norm() if delta is not None else 0.0
        total = 0.0
        for delta in self._deltas.values():
            total += delta.norm() ** 2
        return total**0.5

    def decay_(self, factor: float | Mapping[str, float]) -> None:
        """Scale every group's delta toward zero (fast-weight decay)."""

        for group, delta in self._deltas.items():
            group_factor = factor if isinstance(factor, float) else factor.get(group, 1.0)
            delta.scale_(group_factor)

    def combined_deltas(self) -> dict[str, Tensor]:
        """Sum every group's contribution to each named parameter."""

        combined: dict[str, Tensor] = {}
        for delta in self._deltas.values():
            for name, value in delta.values.items():
                if name in combined:
                    combined[name] = combined[name] + value
                else:
                    combined[name] = value
        return combined

    def clone(self) -> "FastWeightState":
        clone = FastWeightState()
        for group, delta in self._deltas.items():
            clone._deltas[group] = delta.clone()
        return clone

    @torch.no_grad()
    def commit_(
        self,
        model: nn.Module,
        *,
        alpha: float = 1.0,
        group: str | None = None,
    ) -> None:
        """Consolidate a fraction of delta_fast permanently into the model.

        ``theta_slow += alpha * delta``; the committed fraction is then
        removed from delta_fast (``delta *= (1 - alpha)``), matching the
        Procoim-S consolidation commit rule::

            theta_slow  += alpha * safe_delta
            delta_fast  -= alpha * safe_delta

        Restricting ``group`` commits only that group's delta; the default
        commits every tracked group. ``alpha=1.0`` fully consolidates and
        zeroes the committed delta; ``alpha=0.0`` is a no-op.
        """

        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if alpha == 0.0:
            return
        parameters = dict(model.named_parameters())
        groups = (group,) if group is not None else tuple(self._deltas)
        for group_name in groups:
            delta = self._deltas.get(group_name)
            if delta is None or delta.is_zero():
                continue
            for name, value in delta.values.items():
                if name not in parameters:
                    raise KeyError(
                        f"committing fast weights: unknown parameter {name!r}"
                    )
                parameters[name].data.add_(value, alpha=alpha)
            delta.scale_(1.0 - alpha)

    def state_dict(self) -> dict[str, object]:
        return {group: delta.state_dict() for group, delta in self._deltas.items()}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self._deltas = {
            group: FastWeightDelta.from_state_dict(group_state)
            for group, group_state in state.items()
        }


@contextmanager
def apply_fast_delta(model: nn.Module, state: FastWeightState) -> Iterator[None]:
    """Temporarily add ``delta_fast`` to the live model's parameters.

    Inside the context, ``model``'s parameters equal ``theta_slow +
    delta_fast``. On exit — including via an exception — the original
    parameter values are restored bit-for-bit from a snapshot (not by
    subtracting the delta back out), so ``theta_slow`` is recovered exactly
    regardless of floating-point rounding in the addition.
    """

    combined = state.combined_deltas()
    if not combined:
        yield
        return
    parameters = dict(model.named_parameters())
    missing = set(combined) - set(parameters)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise KeyError(
            f"fast-weight delta references unknown parameters: {preview}"
        )
    with torch.no_grad():
        snapshot = {name: parameters[name].data.clone() for name in combined}
        for name, delta in combined.items():
            if delta.shape != parameters[name].shape:
                raise ValueError(f"fast-weight delta shape mismatch for {name!r}")
            parameters[name].data.add_(delta)
    try:
        yield
    finally:
        with torch.no_grad():
            for name, original in snapshot.items():
                parameters[name].data.copy_(original)
