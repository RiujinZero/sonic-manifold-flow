"""Block expectation predictor (SGBM design spec sections 6 and 21).

Before GTR-L forms block ``t``, ``BlockExpectationPredictor`` estimates what
the block's summary will look like from context alone: ``z_hat_t = P(c_t,
e_t, r_t)``. After the block is actually formed, ``predictive_surprise``
(``surprise.py``) compares this prediction against the real, detached
summary — a repeated, predictable block should be easy to predict and
therefore low-surprise; a genuine surprise is one the predictor could not
see coming.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["BlockExpectationPredictor"]


class BlockExpectationPredictor(nn.Module):
    """``P(context_summary, exact_cache_summary, memory_read) -> predicted_summary``."""

    def __init__(
        self,
        *,
        context_dim: int,
        cache_dim: int,
        memory_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if min(context_dim, cache_dim, memory_dim, output_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        hidden_dim = hidden_dim or output_dim
        input_dim = context_dim + cache_dim + memory_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        context_summary: Tensor,
        exact_cache_summary: Tensor,
        memory_read: Tensor,
    ) -> Tensor:
        combined = torch.cat([context_summary, exact_cache_summary, memory_read], dim=-1)
        return self.net(combined)
