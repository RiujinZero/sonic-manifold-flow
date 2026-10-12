"""Actual block summarizer (SGBM design spec sections 6 and 6.1).

After a block is formed and committed, ``BlockSummarizer`` computes ``z_t =
Z(H_t, E(y_t), R_t)`` — the real summary the expectation predictor's guess
gets compared against. This module only combines already-pooled/compact
features (mean/max token embeddings, fused hidden state, pooled stream
state, route statistics, quality indicators); deriving those features from
an actual GTR-L forward pass is Phase F's job, not this one.

Per spec section 6.2 ("No target leakage"): the caller must only invoke
this after the current block's logits have been produced and committed —
memory written from block ``t`` may influence block ``t+1``, never block
``t`` itself. That ordering constraint lives in the training-loop wiring
(Phase F), not in this module.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["BlockSummarizer"]


class BlockSummarizer(nn.Module):
    """Combine pooled token/hidden/stream/route/quality features into one summary vector."""

    def __init__(
        self,
        *,
        token_embedding_dim: int,
        hidden_dim: int,
        stream_width: int,
        route_stat_dim: int,
        indicator_dim: int,
        output_dim: int,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        dims = (
            token_embedding_dim,
            hidden_dim,
            stream_width,
            route_stat_dim,
            indicator_dim,
            output_dim,
        )
        if min(dims) <= 0:
            raise ValueError("all dimensions must be positive")
        # mean + max pooled token embeddings, so the token contribution to
        # the input is 2 * token_embedding_dim.
        input_dim = 2 * token_embedding_dim + hidden_dim + stream_width + route_stat_dim + indicator_dim
        hidden = hidden or output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(
        self,
        *,
        token_embeddings: Tensor,
        fused_hidden: Tensor,
        stream_states: Tensor,
        route_stats: Tensor,
        indicators: Tensor,
    ) -> Tensor:
        """``token_embeddings``: ``[B, O, E]``; ``stream_states``: ``[B, S, Ds]``;
        ``fused_hidden``/``route_stats``/``indicators``: ``[B, *]``."""

        mean_tokens = token_embeddings.mean(dim=1)
        max_tokens = token_embeddings.max(dim=1).values
        pooled_streams = stream_states.mean(dim=1)
        combined = torch.cat(
            [mean_tokens, max_tokens, fused_hidden, pooled_streams, route_stats, indicators],
            dim=-1,
        )
        return self.net(combined)
