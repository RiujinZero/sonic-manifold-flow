"""Trainable low-rank memory port for Sonic (design-v0.4.md SS6.2):

    h'_t = h_t + g_t * U(phi(V(m_t)))

`m_t` is a retrieved `ManifoldFlow` basin center - a detached constant,
per SS6.3's "the retrieved basin values are detached constants." Only
`U`, `V`, and the gate are new trainable parameters this module owns;
Sonic's own weights train too in the full system (SS6.2: "Only U, V, the
memory gate, and Sonic are trainable") but that is the caller's concern,
not this module's.

`V` starts zero-initialized and the gate bias starts very negative
(`gate_init_bias=-4.0`, mirroring `SonicSSMConfig.memory_gate_init_bias`'s
own documented reasoning in `sonic_ssm.py`): the port starts as close to
a no-op as both mechanisms can make it, so grafting it onto a real
checkpoint (or training it from a fresh init) doesn't destabilize
ordinary language modeling before the gate has learned anything.

This deliberately does *not* touch `sonic_ssm.py` (kept vendored
byte-identical from upstream) or hook into a specific layer depth the
way design-v0.4's SS6.2 formula technically specifies ("one or two
selected depths"). It is applied once, to the final hidden state for the
event's answer span - a documented simplification for a first viability
test of whether the mechanism trains at all, not the literal multi-depth
injection point design-v0.4 eventually wants.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class SonicMemoryPortConfig:
    d_model: int
    memory_dim: int
    rank: int = 32
    gate_init_bias: float = -4.0

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.memory_dim <= 0 or self.rank <= 0:
            raise ValueError("d_model, memory_dim, and rank must be positive")


class SonicMemoryPort(nn.Module):
    """Injects a retrieved memory vector into Sonic's hidden state."""

    def __init__(self, config: SonicMemoryPortConfig) -> None:
        super().__init__()
        self.config = config
        self.down = nn.Linear(config.memory_dim, config.rank, bias=False)
        self.up = nn.Linear(config.rank, config.d_model, bias=False)
        self.gate = nn.Linear(config.d_model, 1, bias=True)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        with torch.no_grad():
            self.gate.bias.fill_(config.gate_init_bias)

    def forward(self, hidden: Tensor, memory_vector: Tensor) -> Tensor:
        """`hidden`: `[batch, seq, d_model]`. `memory_vector`: `[batch,
        memory_dim]`, one retrieved (detached) vector per batch row,
        broadcast across `seq` - matching SS6.2's "cached for the
        remainder of the event", not re-queried per token."""
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.d_model:
            raise ValueError("hidden must have shape [batch, seq, d_model]")
        if memory_vector.shape != (hidden.shape[0], self.config.memory_dim):
            raise ValueError("memory_vector must have shape [batch, memory_dim]")
        gate = torch.sigmoid(self.gate(hidden))
        injected = self.up(F.relu(self.down(memory_vector)))
        return hidden + gate * injected.unsqueeze(1)
