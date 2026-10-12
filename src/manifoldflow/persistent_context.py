"""Persistent cross-query basin context for traversal-as-thought experiments.

Every mechanism in this project up to now processes each query
independently: encode, route, answer, with no memory of what was recently
active. This adds the piece the design's basin model never implemented - a
lightweight state that persists across a *sequence* of queries and
measurably biases where the next one routes, so a later, related
observation can re-associate with an earlier one despite unrelated
observations in between.

Deliberately minimal: no new index, no bank changes - the bank stays
"data agnostic, it just stores what the model encounters"; what this class
adds is a routing bias layered on top, not a change to what's stored. It
must stay ANN-routing-cheap (~1ms/query, per `imagenet-scale-evaluation.md`)
since it is meant to sit underneath a larger reasoning layer that may query
it often, not replace the routing path that was already validated.
"""

from __future__ import annotations

import torch
from torch import Tensor

from manifoldflow.index import ExactIndex, HnswIndex


class PersistentBasinContext:
    """Blend a decaying trace of recently visited basin centers into routing.

    `query_t = alpha * z0_t + (1 - alpha) * context_{t-1}`, routed through
    the wrapped index, then `context_t = decay * context_{t-1} + (1 - decay)
    * mu[routed_id]`. `alpha=1` makes every query stateless (identical to
    routing `z0` directly, today's baseline everywhere else in this
    project) - the natural control condition for comparison.
    """

    def __init__(
        self,
        centers: Tensor,
        index: ExactIndex | HnswIndex,
        *,
        alpha: float,
        decay: float,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if centers.ndim != 2 or centers.shape[0] == 0:
            raise ValueError("centers must have shape [basins, d] with basins > 0")
        self.centers = centers
        self.index = index
        self.alpha = alpha
        self.decay = decay
        self.context = torch.zeros(centers.shape[1], dtype=centers.dtype)
        self._has_context = False

    def reset(self) -> None:
        """Clear accumulated context, e.g. at the start of a new stream."""
        self.context = torch.zeros_like(self.context)
        self._has_context = False

    @torch.no_grad()
    def route(self, z0: Tensor) -> int:
        """Route one observation, blending in and then updating context.

        `z0` must have shape `[dimension]` - context is inherently
        sequential, one observation at a time.
        """
        if z0.shape != self.context.shape:
            raise ValueError("z0 must have shape [dimension]")
        if self.alpha < 1.0 and self._has_context:
            query = self.alpha * z0 + (1 - self.alpha) * self.context
        else:
            query = z0
        routed_id = int(self.index.query(query[None, :], 1).item())
        center = self.centers[routed_id].to(dtype=self.context.dtype)
        if self._has_context:
            self.context = self.decay * self.context + (1 - self.decay) * center
        else:
            self.context = center
            self._has_context = True
        return routed_id
