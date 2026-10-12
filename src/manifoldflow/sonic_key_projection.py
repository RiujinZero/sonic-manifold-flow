"""Trainable retrieval-key projection - the fix the discrimination probe's
finding pointed at directly (`sonic-pretrained-memory-discrimination-probe.md`):
with a frozen Sonic and only `SonicMemoryPort` trainable, retrieval
accuracy sat at a flat ~83.6% ceiling across 2000 training steps because
nothing in that setup could touch it - `BasinMemory.read()`'s routing
depends entirely on the frozen checkpoint's own mean-pooled hidden
states, and `SonicMemoryPort` only ever sees retrieval's *output*, never
influences what gets retrieved in the first place.

`SonicKeyProjection` sits between "mean-pooled frozen hidden state" and
"key handed to `BasinMemory.write`/`.read`" - a single trainable linear
layer, applied identically to both fact and query representations (a
shared, symmetric projection, matching nearest-neighbor search's own
assumption that both sides of a comparison live in the same space).
Initialized to the identity matrix, so a freshly constructed projection
reproduces the untrained ~83.6% starting point exactly rather than
regressing it before training has done anything.

It is trained by a separate contrastive (InfoNCE-style) loss over the
*undetached* projected keys - not by backpropagating through
`BasinMemory.read()`'s nearest-neighbor selection, which is a discrete,
non-differentiable operation with no gradient to give. This is the
standard fix for training a retrieval key space (dense-retrieval-style
contrastive training): shape the embedding geometry so that a fixed,
untrained nearest-neighbor search naturally becomes more accurate,
rather than trying to differentiate through the search itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class SonicKeyProjectionConfig:
    dim: int

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")


class SonicKeyProjection(nn.Module):
    """A trainable, identity-initialized linear map from raw pooled hidden
    states to retrieval keys."""

    def __init__(self, config: SonicKeyProjectionConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.dim, config.dim, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(config.dim))

    def forward(self, pooled: Tensor) -> Tensor:
        """`pooled`: `[batch, dim]`, mean-pooled (not yet normalized)
        hidden states. Returns an L2-normalized key of the same shape."""
        if pooled.ndim != 2 or pooled.shape[-1] != self.config.dim:
            raise ValueError("pooled must have shape [batch, dim]")
        return F.normalize(self.projection(pooled), dim=-1)


def info_nce_retrieval_loss(
    query_key: Tensor, candidate_keys: Tensor, correct_index: int, *, temperature: float
) -> Tensor:
    """Per-episode InfoNCE loss: `query_key` (L2-normalized, `[dim]`)
    should score higher (dot product) against
    `candidate_keys[correct_index]` than against every other row of
    `candidate_keys` (`[num_candidates, dim]`, also L2-normalized) - the
    other facts written into the *same* episode's bank, which is exactly
    the negative set `BasinMemory.read()` actually has to discriminate
    against at inference. One episode, one query, `num_candidates`-way
    classification - not a cross-episode in-batch-negatives scheme, since
    each episode's candidate set is unrelated to every other episode's."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 <= correct_index < candidate_keys.shape[0]:
        raise ValueError("correct_index must index an existing candidate")
    logits = (candidate_keys @ query_key) / temperature
    target = torch.tensor(correct_index, device=logits.device)
    return F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))
