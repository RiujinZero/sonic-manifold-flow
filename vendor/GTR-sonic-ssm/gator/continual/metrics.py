"""A-B-A continual-learning metrics (Procoim-S design spec section 24.2).

Pure arithmetic over accuracy measurements the caller supplies — this
module does not itself run any training or evaluation, so it stays testable
independent of a real training loop and reusable across whatever benchmark
harness (this repo's synthetic tasks, or a real one) produced the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ABAMetrics", "compute_aba_metrics"]


@dataclass(frozen=True)
class ABAMetrics:
    a_acquisition: float
    b_acquisition: float
    a_forgetting: float
    a_recovery: float
    backward_transfer: float
    forward_transfer: float


def compute_aba_metrics(
    *,
    accuracy_a_after_phase_a: float,
    accuracy_a_after_phase_b: float,
    accuracy_a_after_return: float,
    accuracy_b_before_phase_b: float,
    accuracy_b_after_phase_b: float,
    chance_accuracy: float,
) -> ABAMetrics:
    """Design spec section 24.2's named measurements:

    - ``a_acquisition``: how well A was learned in phase A.
    - ``b_acquisition``: how well B was learned in phase B.
    - ``a_forgetting``: how much A accuracy dropped after training on B
      (positive = forgetting occurred).
    - ``a_recovery``: how much A accuracy recovered during the phase-A
      return, relative to its post-B low point.
    - ``backward_transfer``: net change in A's accuracy from before B to
      after the A return, relative to A's original acquisition level
      (positive = the A/B/A cycle left A *better* than first learned).
    - ``forward_transfer``: how much better than chance the model already
      is on B *before* any B training — evidence A's learning transferred.
    """

    return ABAMetrics(
        a_acquisition=accuracy_a_after_phase_a,
        b_acquisition=accuracy_b_after_phase_b,
        a_forgetting=accuracy_a_after_phase_a - accuracy_a_after_phase_b,
        a_recovery=accuracy_a_after_return - accuracy_a_after_phase_b,
        backward_transfer=accuracy_a_after_return - accuracy_a_after_phase_a,
        forward_transfer=accuracy_b_before_phase_b - chance_accuracy,
    )
