"""Memory read: query scoring, top-k retrieval, and stream-gated injection.

SGBM design spec section 13. Unlike ``store.py``, retrieval is *not*
wrapped in ``torch.no_grad()`` at the function level: the query projection
and the stream injection gates are ordinary learned parameters that must
receive normal training gradients (spec section 21 — "the outer-loop
trainable projections receive normal training gradients"). Only the memory
*state* being read is already gradient-free, since it was written under
``torch.no_grad()`` by ``store.py``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gator.memory.state import SurpriseMemoryState

__all__ = [
    "score_memory_slots",
    "retrieve_top_k",
    "bump_retrieval_usage",
    "StreamMemoryInjector",
]


def score_memory_slots(
    query: Tensor,
    state: SurpriseMemoryState,
    *,
    strength_weight: float = 0.0,
    usage_weight: float = 0.0,
    age_weight: float = 0.0,
) -> Tensor:
    """``score_j = q^T K_j / sqrt(K) + lambda_s log(strength+eps) + lambda_u log(usage+1) - lambda_a age``.

    Invalid slots score ``-inf`` so they never win top-k selection or
    receive nonzero softmax weight. Design spec section 13.
    """

    key_dim = query.shape[-1]
    scale = key_dim**0.5
    raw = torch.einsum("bk,bmk->bm", query, state.keys) / scale
    if strength_weight:
        raw = raw + strength_weight * torch.log(state.strength.clamp(min=1e-8))
    if usage_weight:
        raw = raw + usage_weight * torch.log1p(state.usage)
    if age_weight:
        raw = raw - age_weight * state.age
    return raw.masked_fill(~state.valid, float("-inf"))


def retrieve_top_k(
    query: Tensor,
    state: SurpriseMemoryState,
    *,
    top_k: int,
    temperature: float = 1.0,
    strength_weight: float = 0.0,
    usage_weight: float = 0.0,
    age_weight: float = 0.0,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Score, select top-k, and softmax-blend into one retrieved vector (spec section 13).

    Returns ``(retrieved [B, V], metrics)``. A batch lane with zero valid
    memory slots gets an all-zero retrieval and zero weights rather than a
    NaN from softmaxing an all ``-inf`` row.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    scores = score_memory_slots(
        query,
        state,
        strength_weight=strength_weight,
        usage_weight=usage_weight,
        age_weight=age_weight,
    )
    k = min(top_k, state.memory_slots)
    top_scores, top_indices = scores.topk(k, dim=1)

    finite_mask = torch.isfinite(top_scores)
    masked_scores = top_scores.masked_fill(~finite_mask, float("-inf"))
    weights = torch.softmax(masked_scores / temperature, dim=1)
    weights = torch.nan_to_num(weights, nan=0.0)
    weights = weights * finite_mask.to(weights.dtype)

    batch_arange = torch.arange(state.batch_size, device=state.device).unsqueeze(1)
    top_values = state.values[batch_arange, top_indices]
    retrieved = torch.einsum("bk,bkv->bv", weights, top_values)

    entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=1)
    top_similarity = torch.where(
        finite_mask.any(dim=1),
        top_scores.masked_fill(~finite_mask, float("-inf")).max(dim=1).values,
        torch.zeros(state.batch_size, device=state.device),
    )

    metrics = {
        "retrieval_weights": weights,
        "retrieval_indices": top_indices,
        "retrieval_entropy": entropy,
        "retrieval_top_similarity": top_similarity,
    }
    return retrieved, metrics


@torch.no_grad()
def bump_retrieval_usage(
    state: SurpriseMemoryState,
    top_indices: Tensor,
    weights: Tensor,
    *,
    threshold: float = 0.05,
) -> None:
    """Increment ``usage`` for retrieved slots whose weight cleared ``threshold``.

    Kept separate from :func:`retrieve_top_k` so scoring/blending stays a
    pure, gradient-carrying function while this bookkeeping mutation stays
    an explicit, no-grad, opt-in step.
    """

    batch_arange = torch.arange(state.batch_size, device=state.device).unsqueeze(1)
    used = weights > threshold
    increment = used.to(state.usage.dtype)
    state.usage[batch_arange, top_indices] = state.usage[batch_arange, top_indices] + increment


class StreamMemoryInjector(nn.Module):
    """Stream-specific gated residual injection of retrieved memory (spec section 13.1).

    For each private work stream ``s``: ``g_s = sigmoid(W_g,s[h_s; r_t])``,
    ``h_s' = h_s + g_s * (W_r,s r_t)``. A separate gate and value projection
    per stream lets each stream decide independently how much of the shared
    retrieved memory to absorb. Memory dropout (spec section 13.2) zeroes
    the whole injection for a fraction of batch lanes during training so
    the core model does not become dependent on memory always being
    present.
    """

    def __init__(
        self,
        *,
        stream_count: int,
        stream_width: int,
        value_dim: int,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if stream_count <= 0:
            raise ValueError("stream_count must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        self.stream_count = stream_count
        self.dropout = dropout
        self.gate_projections = nn.ModuleList(
            nn.Linear(stream_width + value_dim, 1) for _ in range(stream_count)
        )
        self.value_projections = nn.ModuleList(
            nn.Linear(value_dim, stream_width) for _ in range(stream_count)
        )

    def forward(self, stream_states: Tensor, retrieved: Tensor) -> Tensor:
        """``stream_states``: ``[B, S, Ds]``; ``retrieved``: ``[B, V]``."""

        batch_size, stream_count, _ = stream_states.shape
        if stream_count != self.stream_count:
            raise ValueError(
                f"expected {self.stream_count} streams, got {stream_count}"
            )
        if self.training and self.dropout > 0.0:
            keep = (
                torch.rand(batch_size, 1, device=stream_states.device) >= self.dropout
            ).to(stream_states.dtype)
        else:
            keep = torch.ones(batch_size, 1, device=stream_states.device, dtype=stream_states.dtype)

        updated_streams = []
        for stream_index in range(stream_count):
            h_s = stream_states[:, stream_index, :]
            gate_input = torch.cat([h_s, retrieved], dim=-1)
            gate = torch.sigmoid(self.gate_projections[stream_index](gate_input))
            injected = gate * self.value_projections[stream_index](retrieved) * keep
            updated_streams.append(h_s + injected)
        return torch.stack(updated_streams, dim=1)
