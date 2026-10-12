"""Write-gate controller: combine surprise and reliability into a write decision.

SGBM design spec section 11 gives the write-strength formula
(``g_write = sigmoid(a*S~_t + b*S_novel - c)``, ``g_commit = g_write *
R_t``) and section 10.1 gives the three-outcome policy (low surprise -> no
write; high surprise + high reliability -> stable memory; high surprise +
low reliability -> quarantine). This module reconciles both into one
decision: since ``g_commit`` is already reliability-weighted (a low-
reliability event depresses it even when ``g_write`` is high), the two
``SurpriseMemoryConfig`` thresholds already defined in Phase A
(``quarantine_threshold``, ``write_threshold``) directly become the bands
from section 11's suggested states — ``g_commit`` below
``quarantine_threshold`` ignores the block, the middle band is uncertain
enough to only quarantine, and only the top band is trusted enough for
stable memory.

Section 11's numeric coefficients (``a``, ``b``, ``c``) for the fixed gate
formula are not given in the spec (unlike the 0.45/0.15/0.10/0.30 surprise
combination weights, which are) — the defaults below are an initial,
explicitly-unvalidated guess, matching the spec's own framing ("a small MLP
should eventually learn this mapping, but the fixed version is easier to
debug").
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
from torch import Tensor, nn

from gator.memory.config import SurpriseMemoryConfig

__all__ = [
    "WriteGateCoefficients",
    "write_gate_strength",
    "commit_strength_from_gate",
    "WriteDecision",
    "decide_write",
    "LearnedWriteGate",
]


@dataclass(frozen=True)
class WriteGateCoefficients:
    """Coefficients for the fixed write-gate formula (spec section 11)."""

    surprise_weight: float = 4.0  # 'a'
    novelty_weight: float = 2.0  # 'b'
    bias: float = 3.0  # 'c'


def write_gate_strength(
    blended_surprise: Tensor,
    novelty: Tensor,
    *,
    coefficients: WriteGateCoefficients | None = None,
) -> Tensor:
    """``g_write = sigmoid(a * S~_t + b * S_novel - c)`` (spec section 11)."""

    coefficients = coefficients or WriteGateCoefficients()
    return torch.sigmoid(
        coefficients.surprise_weight * blended_surprise
        + coefficients.novelty_weight * novelty
        - coefficients.bias
    )


def commit_strength_from_gate(write_gate: Tensor, reliability: Tensor) -> Tensor:
    """``g_commit = g_write * R_t`` (spec section 11)."""

    return write_gate * reliability


class WriteDecision(IntEnum):
    IGNORE = 0
    QUARANTINE = 1
    STABLE = 2


def decide_write(commit_strength: Tensor, config: SurpriseMemoryConfig) -> Tensor:
    """Bucket ``g_commit`` into ignore/quarantine/stable using the config's
    own ``quarantine_threshold``/``write_threshold`` (design spec section 11
    suggested states, reusing the section 10.1 reliability-gated bands —
    see module docstring)."""

    decision = torch.full(
        commit_strength.shape,
        int(WriteDecision.IGNORE),
        dtype=torch.int64,
        device=commit_strength.device,
    )
    decision[commit_strength >= config.quarantine_threshold] = int(WriteDecision.QUARANTINE)
    decision[commit_strength >= config.write_threshold] = int(WriteDecision.STABLE)
    return decision


class LearnedWriteGate(nn.Module):
    """Optional learned alternative to the fixed write-gate formula.

    Spec section 11: "A small MLP should eventually learn this mapping, but
    the fixed version is easier to debug." Takes the same raw feature
    vector callers would otherwise hand to :func:`write_gate_strength`
    directly, so it is a drop-in swap once training data exists to fit it.
    """

    def __init__(self, *, feature_count: int = 9, hidden_dim: int = 16) -> None:
        super().__init__()
        if feature_count <= 0:
            raise ValueError("feature_count must be positive")
        self.net = nn.Sequential(
            nn.Linear(feature_count, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        """``features``: ``[B, feature_count]`` -> write gate ``[B]``, in (0, 1)."""

        return torch.sigmoid(self.net(features).squeeze(-1))
