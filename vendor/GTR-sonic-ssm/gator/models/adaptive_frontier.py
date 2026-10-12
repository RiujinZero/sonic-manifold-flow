"""Adaptive uncertainty-based capacity for packed node frontiers.

The packed routing backend needs a static storage shape for compilation, but
not every packet needs to *use* every stored route.  This module chooses a
per-packet live route count from the packet's current uncertainty and returns
a ``max_k``-wide :class:`PackedNodeFrontier` whose ``valid`` mask carries that
decision.  Familiar, sharply routed packets therefore use the minimum route
count while ambiguous packets retain more alternatives.

The decision itself is discrete.  Gradients continue through retained route
mass (and through the uncertainty diagnostics), but do not pass through the
integer live-count choice.  This is the same piecewise-differentiable behavior
as ordinary top-k routing and keeps the implementation friendly to
``torch.compile``: no data-dependent tensor shapes or Python loops are used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch

from .packed_frontier import PackedNodeFrontier


AdaptiveTopKMode = Literal["entropy", "margin", "hybrid"]


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _mass_threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("mass_threshold must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class AdaptiveTopKResult:
    """Fixed-shape frontier plus the packet-local routing decision.

    Diagnostic tensors have ``frontier.packet_shape`` shape. ``selected_k``
    is the requested live-route budget and is always in ``[min_k, max_k]``.
    The actual number of valid entries may be lower when the input is padded,
    has fewer positive destinations, or ``mass_threshold`` removes entries.
    """

    frontier: PackedNodeFrontier
    selected_k: torch.Tensor
    normalized_entropy: torch.Tensor
    top1_margin: torch.Tensor
    uncertainty: torch.Tensor


def _merge_duplicate_nodes(
    frontier: PackedNodeFrontier,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge duplicate node IDs in candidate-width storage.

    Returned tensors retain the input capacity. Unique entries occupy group
    starts and all other positions are padding. No ``n_nodes``-wide tensor is
    allocated.
    """

    active = frontier.valid & frontier.mass.gt(0)
    mass = torch.where(active, frontier.mass, torch.zeros_like(frontier.mass))
    sentinel = torch.full_like(frontier.nodes, frontier.n_nodes)
    sortable_nodes = torch.where(active, frontier.nodes, sentinel)
    sorted_nodes, node_order = sortable_nodes.sort(dim=-1, stable=True)
    sorted_mass = mass.gather(-1, node_order)
    sorted_valid = active.gather(-1, node_order)

    group_start = sorted_valid.clone()
    group_start[..., 1:] &= (
        sorted_nodes[..., 1:] != sorted_nodes[..., :-1]
    )
    group_index = group_start.to(torch.long).cumsum(dim=-1) - 1
    group_index = group_index.clamp_min(0)
    merged_by_group = torch.zeros_like(sorted_mass).scatter_add(
        -1, group_index, sorted_mass
    )
    merged_mass = torch.where(
        group_start,
        merged_by_group.gather(-1, group_index),
        torch.zeros_like(sorted_mass),
    )
    merged_nodes = torch.where(
        group_start, sorted_nodes, torch.zeros_like(sorted_nodes)
    )
    return merged_nodes, merged_mass, group_start


def adaptive_uncertainty_top_k(
    frontier: PackedNodeFrontier,
    *,
    min_k: int,
    max_k: int,
    mode: AdaptiveTopKMode = "hybrid",
    entropy_mix: float = 0.5,
    mass_threshold: float = 0.0,
) -> AdaptiveTopKResult:
    """Choose a packet-local top-k from route uncertainty.

    Uncertainty can use normalized Shannon entropy, inverse top-1 margin, or a
    convex combination of both. The score is mapped linearly to the inclusive
    integer range ``[min_k, max_k]`` using deterministic nearest-integer
    rounding. Equal-mass destinations are ordered by node ID.

    The returned frontier always has capacity ``max_k``. Invalid/padded input
    entries and duplicate node IDs are handled directly. Thresholding is
    applied to normalized mass after the adaptive budget is chosen; the
    strongest route always survives. Retained probability is renormalized to
    one, so pruning never leaks packet mass.

    Args:
        frontier: Candidate packed node distribution.
        min_k: Live-route budget for completely certain packets.
        max_k: Live-route budget for maximally uncertain packets and fixed
            output storage capacity.
        mode: ``"entropy"``, ``"margin"``, or ``"hybrid"``.
        entropy_mix: Entropy weight used only in hybrid mode. The complementary
            weight multiplies inverse top-1 margin.
        mass_threshold: Remove retained destinations below this normalized
            probability, with a strongest-route fallback.
    """

    min_k = _positive_integer(min_k, "min_k")
    max_k = _positive_integer(max_k, "max_k")
    if min_k > max_k:
        raise ValueError("min_k cannot exceed max_k")
    if max_k > min(frontier.capacity, frontier.n_nodes):
        raise ValueError("max_k exceeds packed or graph capacity")
    if mode not in ("entropy", "margin", "hybrid"):
        raise ValueError("mode must be 'entropy', 'margin', or 'hybrid'")
    entropy_mix = _unit_interval(entropy_mix, "entropy_mix")
    mass_threshold = _mass_threshold(mass_threshold)

    merged_nodes, merged_mass, merged_valid = _merge_duplicate_nodes(frontier)
    total_mass = merged_mass.sum(dim=-1, keepdim=True)
    torch._assert_async(
        total_mass.gt(0).all(),
        "each packet must contain positive valid route mass",
    )
    probabilities = merged_mass / total_mass.clamp_min(
        torch.finfo(merged_mass.dtype).tiny
    )

    support = merged_valid.sum(dim=-1)
    entropy_terms = torch.where(
        merged_valid,
        -probabilities * probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log(),
        torch.zeros_like(probabilities),
    )
    entropy = entropy_terms.sum(dim=-1)
    entropy_denominator = support.clamp_min(2).to(probabilities.dtype).log()
    normalized_entropy = torch.where(
        support.gt(1),
        entropy / entropy_denominator,
        torch.zeros_like(entropy),
    ).clamp(0.0, 1.0)

    # Sorting by node first and then stably by mass makes equal-mass choices
    # deterministic by node identity rather than candidate-list order.
    mass_order = probabilities.argsort(
        dim=-1, descending=True, stable=True
    )
    ranked_mass = probabilities.gather(-1, mass_order)
    ranked_nodes = merged_nodes.gather(-1, mass_order)
    ranked_valid = merged_valid.gather(-1, mass_order)
    top1 = ranked_mass[..., 0]
    if frontier.capacity == 1:
        top2 = torch.zeros_like(top1)
    else:
        top2 = ranked_mass[..., 1]
    top1_margin = (top1 - top2).clamp(0.0, 1.0)
    margin_uncertainty = 1.0 - top1_margin

    if mode == "entropy":
        uncertainty = normalized_entropy
    elif mode == "margin":
        uncertainty = margin_uncertainty
    else:
        uncertainty = (
            entropy_mix * normalized_entropy
            + (1.0 - entropy_mix) * margin_uncertainty
        )
    uncertainty = uncertainty.clamp(0.0, 1.0)

    span = max_k - min_k
    selected_k = min_k + torch.floor(
        uncertainty * span + 0.5
    ).to(torch.long)
    selected_k = selected_k.clamp(min=min_k, max=max_k)

    ranked_mass = ranked_mass[..., :max_k]
    ranked_nodes = ranked_nodes[..., :max_k]
    ranked_valid = ranked_valid[..., :max_k]
    rank = torch.arange(max_k, device=ranked_mass.device)
    within_budget = rank < selected_k.unsqueeze(-1)
    retained = ranked_valid & within_budget & ranked_mass.gt(0)
    if mass_threshold > 0.0:
        retained &= ranked_mass.ge(mass_threshold)

    # Thresholds and input padding are not allowed to erase a packet.
    strongest = torch.zeros_like(retained)
    strongest[..., 0] = True
    retained |= strongest & ranked_valid[..., :max_k]
    retained_mass = torch.where(
        retained, ranked_mass, torch.zeros_like(ranked_mass)
    )
    retained_total = retained_mass.sum(dim=-1, keepdim=True)
    torch._assert_async(
        retained_total.gt(0).all(),
        "adaptive top-k must retain positive route mass",
    )
    retained_mass = retained_mass / retained_total.clamp_min(
        torch.finfo(retained_mass.dtype).tiny
    )
    retained_nodes = torch.where(
        retained, ranked_nodes, torch.zeros_like(ranked_nodes)
    )
    selected_frontier = PackedNodeFrontier(
        nodes=retained_nodes,
        mass=retained_mass,
        valid=retained,
        n_nodes=frontier.n_nodes,
    )
    return AdaptiveTopKResult(
        frontier=selected_frontier,
        selected_k=selected_k,
        normalized_entropy=normalized_entropy,
        top1_margin=top1_margin,
        uncertainty=uncertainty,
    )


__all__ = [
    "AdaptiveTopKMode",
    "AdaptiveTopKResult",
    "adaptive_uncertainty_top_k",
]
