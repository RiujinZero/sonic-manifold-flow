"""Per-episode SGBM runtime state: memory slots plus the running summaries
that feed the block-expectation predictor and query projection.

``SurpriseMemoryState`` (state.py) only covers the memory slots/quarantine
themselves — the design spec's reference dataclass (section 21) does not
include the "context summary" and "exact cache summary" running features
section 6/13 read from. ``MemoryRuntime`` bundles those alongside the memory
state so a caller threads exactly one object through a generation episode,
the same way ``GatorLinearBlockCache`` is threaded through prefill/forward/
append.
"""

from __future__ import annotations

import torch
from torch import Tensor

from gator.memory.config import SurpriseMemoryConfig
from gator.memory.state import SurpriseMemoryState

__all__ = ["MemoryRuntime"]


class MemoryRuntime:
    """``state`` plus two-timescale running summaries of committed content.

    ``context_summary`` is a fast-decaying (short-term) pooled embedding of
    recently committed blocks; ``cache_summary`` is a slow-decaying
    (long-term) pooled embedding of everything committed so far in the
    episode. Both are ``[B, fused_dim]``, updated by the caller after each
    committed block — see ``gator_linear/prefix.py``'s ``forward_cached``
    integration for the concrete update rule.
    """

    def __init__(
        self,
        *,
        state: SurpriseMemoryState,
        context_summary: Tensor,
        cache_summary: Tensor,
    ) -> None:
        if context_summary.shape != cache_summary.shape:
            raise ValueError("context_summary and cache_summary must share shape")
        if context_summary.shape[0] != state.batch_size:
            raise ValueError("summary batch dimension must match memory state")
        self.state = state
        self.context_summary = context_summary
        self.cache_summary = cache_summary

    @classmethod
    def allocate(
        cls,
        batch_size: int,
        config: SurpriseMemoryConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "MemoryRuntime":
        return cls(
            state=SurpriseMemoryState.allocate(batch_size, config, device=device, dtype=dtype),
            context_summary=torch.zeros(batch_size, config.fused_dim, device=device, dtype=dtype),
            cache_summary=torch.zeros(batch_size, config.fused_dim, device=device, dtype=dtype),
        )

    @torch.no_grad()
    def reset_lanes(self, lane_mask: Tensor) -> None:
        """Clear memory slots and running summaries for the masked lanes only."""

        self.state.reset_lanes(lane_mask)
        mask = lane_mask.to(dtype=torch.bool, device=self.context_summary.device)
        if bool(mask.any()):
            self.context_summary[mask] = 0.0
            self.cache_summary[mask] = 0.0

    @torch.no_grad()
    def update_summaries(
        self,
        new_block_summary: Tensor,
        *,
        context_decay: float = 0.5,
        cache_decay: float = 0.99,
    ) -> None:
        """Fold a newly committed block's pooled embedding into both EMAs."""

        self.context_summary = (
            context_decay * self.context_summary + (1.0 - context_decay) * new_block_summary
        )
        self.cache_summary = (
            cache_decay * self.cache_summary + (1.0 - cache_decay) * new_block_summary
        )

    def detach_(self) -> "MemoryRuntime":
        self.state.detach_()
        self.context_summary = self.context_summary.detach()
        self.cache_summary = self.cache_summary.detach()
        return self
