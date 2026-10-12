"""Memory-on/off evaluation utilities (design spec section 24: "loss with
memory; loss without memory on sampled probes").

Note on placement: mirrors ``episodic_stream.py``/``memory_losses.py`` —
lives under ``gator/continual/`` since ``gator/training.py`` already exists
as a module.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from gator.memory.runtime import MemoryRuntime

__all__ = ["compare_with_without_memory"]


@torch.no_grad()
def compare_with_without_memory(
    model,
    cache,
    *,
    targets: Tensor,
    memory: MemoryRuntime | None,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
) -> dict[str, Tensor | None]:
    """Run ``forward_cached`` with and without memory against the same
    cache/targets, directly operationalizing design spec section 19.3's
    counterfactual comparison. ``forward_cached`` never mutates ``cache``,
    so calling it twice against the same cache is safe.

    This is an evaluation utility (no gradients) — training the predictor
    or write gate from the resulting utility is a separate step, using
    ``memory_losses.predictor_loss``/``write_calibration_loss`` in a
    grad-enabled context.
    """

    logits_without = model.forward_cached(cache)
    loss_without = loss_fn(logits_without, targets)

    if memory is None:
        return {
            "loss_without_memory": loss_without,
            "loss_with_memory": None,
            "utility": None,
        }

    logits_with = model.forward_cached(cache, memory=memory)
    loss_with = loss_fn(logits_with, targets)
    return {
        "loss_without_memory": loss_without,
        "loss_with_memory": loss_with,
        "utility": loss_without - loss_with,
    }
