"""Execution planning for dense-to-packed Gator stewardship.

The model forward pass consumes a static :class:`StewardshipExecutionPlan`.
Automatic policy decisions live here, outside the relay loop, so observing
route density never introduces a GPU-to-CPU synchronization between relays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal, Mapping, Sequence

import torch


StewardshipExecutionMode = Literal["dense", "packed_auto"]


@dataclass(frozen=True)
class StewardshipExecutionPlan:
    """A static execution decision for one model invocation.

    ``packed_from_relay=None`` means every relay uses the dense engine.
    Otherwise, relays at and after the specified zero-based index use the
    packed engine.
    """

    packed_from_relay: int | None = None

    def __post_init__(self) -> None:
        if self.packed_from_relay is not None and self.packed_from_relay < 0:
            raise ValueError("packed_from_relay must be nonnegative or None")

    @property
    def uses_packed_execution(self) -> bool:
        return self.packed_from_relay is not None

    def uses_packed_at(self, relay: int) -> bool:
        """Return whether the zero-based relay should use packed execution."""

        if relay < 0:
            raise ValueError("relay must be nonnegative")
        return (
            self.packed_from_relay is not None
            and relay >= self.packed_from_relay
        )


def stewardship_active_fraction(
    ownership: torch.Tensor,
    route_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the live cell-route fraction entering each relay.

    ``ownership`` has shape ``[B, R+1, C, T]``. Trace zero enters relay zero,
    while the final trace is the state after the last relay and therefore does
    not select a relay engine. The result consequently has shape ``[R]``.
    Invalid padded token positions are excluded when ``route_valid`` is given.

    The calculation remains on the input device. A caller may transfer the
    complete ``[R]`` result once at a scheduled policy-observation boundary.
    """

    if ownership.ndim != 4:
        raise ValueError("ownership must have [batch, relays+1, cells, tokens] shape")
    batch, trace_count, cells, tokens = ownership.shape
    if trace_count < 2:
        raise ValueError("ownership must contain an input and post-relay trace")
    if cells <= 0:
        raise ValueError("ownership must contain at least one cell")
    if route_valid is None:
        route_valid = torch.ones(
            batch,
            tokens,
            dtype=torch.bool,
            device=ownership.device,
        )
    elif route_valid.shape != (batch, tokens):
        raise ValueError("route_valid must have [batch, tokens] shape")
    else:
        route_valid = route_valid.to(device=ownership.device, dtype=torch.bool)

    # Trace r is the ownership entering relay r. The final post-relay trace
    # does not correspond to another execution decision.
    live_routes = ownership[:, :-1].gt(0).sum(dim=2).to(torch.float32)
    valid = route_valid[:, None].to(torch.float32)
    live_sum = (live_routes * valid).sum(dim=(0, 2))
    possible = (valid.sum() * cells).clamp_min(1.0)
    return live_sum / possible


@dataclass
class StewardshipExecutionController:
    """Latch a dense-to-packed execution plan from periodic observations.

    The controller is deliberately host-side. ``should_observe`` limits the
    synchronization cadence, and ``observe`` transfers the short per-relay
    density vector once before choosing the earliest relay whose complete
    suffix remains under the measured break-even threshold. A packed decision
    is one-way for the lifetime of the controller.
    """

    mode: StewardshipExecutionMode = "dense"
    threshold: float = 0.5
    relays: int = 4
    check_every: int = 100
    _packed_from_relay: int | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.mode not in {"dense", "packed_auto"}:
            raise ValueError("mode must be 'dense' or 'packed_auto'")
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if self.relays <= 0:
            raise ValueError("relays must be positive")
        if self.check_every <= 0:
            raise ValueError("check_every must be positive")

    @property
    def plan(self) -> StewardshipExecutionPlan:
        """Return the immutable plan to use for the next forward call."""

        return StewardshipExecutionPlan(self._packed_from_relay)

    @property
    def packed_from_relay(self) -> int | None:
        return self._packed_from_relay

    @property
    def latched(self) -> bool:
        return self._packed_from_relay is not None

    def should_observe(self, step: int, *, pruning: bool) -> bool:
        """Return whether a completed one-based step should update the policy."""

        if step <= 0:
            raise ValueError("step must be positive")
        return (
            self.mode == "packed_auto"
            and not self.latched
            and pruning
            and step % self.check_every == 0
        )

    def observe(
        self,
        active_fraction_by_relay: torch.Tensor | Sequence[float],
    ) -> StewardshipExecutionPlan:
        """Observe route density and latch the earliest qualifying suffix.

        Tensor inputs may live on a GPU. They are detached and transferred as
        one short vector here, outside the model relay loop.
        """

        if self.mode == "dense" or self.latched:
            return self.plan
        if isinstance(active_fraction_by_relay, torch.Tensor):
            fractions_tensor = active_fraction_by_relay
            if fractions_tensor.ndim != 1:
                raise ValueError("active fractions must be a one-dimensional vector")
            if fractions_tensor.numel() != self.relays:
                raise ValueError("active fractions must contain one value per relay")
            fractions = (
                fractions_tensor.detach()
                .to(device="cpu", dtype=torch.float64)
                .tolist()
            )
        else:
            fractions = [float(value) for value in active_fraction_by_relay]
            if len(fractions) != self.relays:
                raise ValueError("active fractions must contain one value per relay")

        for fraction in fractions:
            if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError("active fractions must be finite values in [0, 1]")
        # A plan is a packed *suffix*. Reversible novelty routing can widen a
        # later relay again, so select the earliest index whose complete suffix
        # remains below the measured break-even threshold.
        suffix_maximum = 0.0
        qualifying_relay = None
        for relay in range(self.relays - 1, -1, -1):
            suffix_maximum = max(suffix_maximum, fractions[relay])
            if suffix_maximum <= self.threshold:
                qualifying_relay = relay
        self._packed_from_relay = qualifying_relay
        return self.plan

    def state_dict(self) -> dict:
        """Return JSON-serializable policy configuration and latch state."""

        return {
            "version": 1,
            "mode": self.mode,
            "threshold": self.threshold,
            "relays": self.relays,
            "check_every": self.check_every,
            "packed_from_relay": self._packed_from_relay,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore a controller created with the same policy configuration."""

        expected = {
            "mode": self.mode,
            "threshold": self.threshold,
            "relays": self.relays,
            "check_every": self.check_every,
        }
        for key, expected_value in expected.items():
            actual = state.get(key, expected_value)
            if actual != expected_value:
                raise ValueError(
                    f"execution controller {key} differs from saved state: "
                    f"{expected_value!r} != {actual!r}"
                )
        packed_from_relay = state.get("packed_from_relay")
        if packed_from_relay is not None:
            if (
                isinstance(packed_from_relay, bool)
                or not isinstance(packed_from_relay, int)
                or not 0 <= packed_from_relay < self.relays
            ):
                raise ValueError(
                    "saved packed_from_relay must be a valid relay index or None"
                )
            if self.mode != "packed_auto":
                raise ValueError("dense execution cannot restore a packed latch")
        self._packed_from_relay = packed_from_relay
