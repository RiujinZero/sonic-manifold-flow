"""Modality-independent representation boundary (design-v0.3.md SS4, SS15).

`RepresentationAdapter` is the interface a frozen language model implements
so ManifoldFlow's memory code carries no model-specific assumptions beyond
the recorded representation manifest - Qwen now, Sonic later, without
touching memory code (SS15: "Qwen and Sonic should implement the same
`RepresentationAdapter`"). `pool_hidden_states` and `FixedSeededProjection`
are the pure, model-independent pieces of the event key
`z_t = normalize(R * pool(H_t^(l)))` (SS4.1) - deliberately separated from
the actual Qwen forward pass so they are unit-testable without downloading
or running the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class RepresentationAdapter(ABC):
    """Read/write boundary between a frozen language model and ManifoldFlow.

    Any implementation must be able to explain, via `manifest()`, exactly
    which model revision, tokenizer, chat template, layer, pooling rule,
    projection seed, and normalization produced its keys (SS4.1, SS11) -
    without that, stored keys cannot be trusted to reproduce after restart.
    """

    @abstractmethod
    def encode_event(self, text: str) -> Tensor:
        """Return a normalized `[dimension]` key for one event's text."""

    @abstractmethod
    def generate(self, prompt: str, memories: list[str]) -> str:
        """Generate a response, with retrieved memory payloads injected."""

    @abstractmethod
    def clear_runtime_context(self) -> None:
        """Reset any per-session runtime state (e.g. KV cache), if held."""

    @abstractmethod
    def manifest(self) -> dict[str, Any]:
        """Return the versioning record required to reproduce this adapter's keys."""


def pool_hidden_states(
    hidden_states: Tensor, attention_mask: Tensor, *, pooling: str
) -> Tensor:
    """Pool per-token hidden states into one vector per sequence.

    `hidden_states` has shape `[batch, seq, hidden]`; `attention_mask` has
    shape `[batch, seq]` with 1 for content tokens and 0 for padding.
    `pooling="last_token"` assumes right-padding (content tokens form a
    contiguous prefix), matching standard causal-LM tokenizer behavior.
    """
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, seq, hidden]")
    if attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask must have shape [batch, seq]")
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        return summed / counts
    if pooling == "last_token":
        lengths = attention_mask.sum(dim=1).clamp_min(1).long() - 1
        batch_indices = torch.arange(
            hidden_states.shape[0], device=hidden_states.device
        )
        return hidden_states[batch_indices, lengths]
    raise ValueError(f"unknown pooling rule: {pooling!r}")


def layer_index_for_fraction(num_layers: int, fraction: float) -> int:
    """Resolve a 0-indexed decoder-layer index from a depth fraction.

    Lets the representation sweep (design-v0.3.md SS4.2:
    "final hidden layer; ~75% depth; ~50% depth") be specified in a way
    that is independent of any one model's exact layer count, so the same
    `fraction` transfers when a later `RepresentationAdapter` (e.g. Sonic)
    has a different depth.
    """
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    return min(num_layers - 1, max(0, round(num_layers * fraction) - 1))


class FixedSeededProjection(nn.Module):
    """Deterministic random projection, L2-normalized at the output.

    Regenerates the identical projection matrix from `seed` on every
    construction (design-v0.3.md SS4.1: "a fixed seeded projection") - the
    manifest only needs to record the seed, not the matrix, for exact
    reproducibility after restart.
    """

    def __init__(self, input_dim: int, output_dim: int, *, seed: int) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(input_dim, output_dim, generator=generator)
        self.register_buffer("projection", projection)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape [batch, {self.input_dim}]")
        projected = features.to(self.projection.dtype) @ self.projection
        return F.normalize(projected, p=2, dim=-1)


def retrieval_topk_accuracy(
    query: Tensor, bank: Tensor, true_index: Tensor, *, k: int
) -> float:
    """Fraction of queries whose true bank match lands in the top-k neighbors.

    `query` and `bank` must already be L2-normalized (SS4.1's `z_t`), so
    similarity is the plain dot product - the same scoring a stateless RBF
    lookup would use. Used to grade design-v0.3.md Phase 1's representation
    sweep: which (layer, pooling) choice retrieves a paraphrased query's
    source fact against a bank of distractor facts.
    """
    if query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]:
        raise ValueError("query and bank must have shape [n, d] with matching d")
    if true_index.shape != (query.shape[0],):
        raise ValueError("true_index must have shape [query_count]")
    if not 0 < k <= bank.shape[0]:
        raise ValueError("k must be in [1, bank size]")
    scores = query @ bank.transpose(0, 1)
    top_k = scores.topk(k, dim=1).indices
    hits = (top_k == true_index[:, None]).any(dim=1)
    return float(hits.float().mean().item())
