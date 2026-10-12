"""Forgetting-surprise monitoring (Procoim-S design spec section 15).

Detects when a replay item's current loss has drifted meaningfully above
its expected retained loss — the signal that a previously-learned skill is
being forgotten — and classifies severity into the spec's three response
tiers (warning / repair / critical).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gator.optimizers.procoim_s import ProcoimSConfig

__all__ = [
    "ForgettingSeverity",
    "ForgettingEvent",
    "forgetting_surprise",
    "classify_forgetting",
    "RetentionProbe",
    "RetentionMonitor",
]


class ForgettingSeverity(str, Enum):
    NONE = "none"
    WARNING = "warning"
    REPAIR = "repair"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ForgettingEvent:
    item_id: str
    forgetting_surprise: float
    severity: ForgettingSeverity


def forgetting_surprise(
    current_loss: float,
    expected_loss: float,
    scale: float,
    *,
    eps: float = 1e-8,
) -> float:
    """``S_forget = (L_current - mu) / (sigma + eps)`` (design spec section 15)."""

    return (current_loss - expected_loss) / (scale + eps)


def classify_forgetting(
    surprise: float,
    config: ProcoimSConfig,
    *,
    critical_multiplier: float = 1.5,
) -> ForgettingSeverity:
    """Bucket a forgetting-surprise value into the spec's three response
    tiers. The spec names only ``forgetting_surprise_warning`` and
    ``forgetting_surprise_repair`` as config thresholds (both already
    defined in ``ProcoimSConfig``); "critical" has no separate named
    threshold in the spec, so it is derived as ``critical_multiplier`` times
    the repair threshold — configurable, not hardcoded, and documented here
    rather than silently invented.
    """

    if critical_multiplier <= 1.0:
        raise ValueError("critical_multiplier must be greater than 1.0")
    if surprise >= config.forgetting_surprise_repair * critical_multiplier:
        return ForgettingSeverity.CRITICAL
    if surprise >= config.forgetting_surprise_repair:
        return ForgettingSeverity.REPAIR
    if surprise >= config.forgetting_surprise_warning:
        return ForgettingSeverity.WARNING
    return ForgettingSeverity.NONE


@dataclass(frozen=True)
class RetentionProbe:
    """One replay item's current-vs-expected loss reading."""

    item_id: str
    current_loss: float
    expected_loss: float
    scale: float


class RetentionMonitor:
    """Evaluates a batch of :class:`RetentionProbe` and classifies each."""

    def __init__(self, config: ProcoimSConfig, *, critical_multiplier: float = 1.5) -> None:
        self.config = config
        self.critical_multiplier = critical_multiplier

    def evaluate(self, probes: list[RetentionProbe]) -> list[ForgettingEvent]:
        events = []
        for probe in probes:
            surprise = forgetting_surprise(probe.current_loss, probe.expected_loss, probe.scale)
            severity = classify_forgetting(
                surprise, self.config, critical_multiplier=self.critical_multiplier
            )
            events.append(
                ForgettingEvent(item_id=probe.item_id, forgetting_surprise=surprise, severity=severity)
            )
        return events

    def events_at_or_above(
        self, events: list[ForgettingEvent], severity: ForgettingSeverity
    ) -> list[ForgettingEvent]:
        order = list(ForgettingSeverity)
        threshold_index = order.index(severity)
        return [event for event in events if order.index(event.severity) >= threshold_index]
