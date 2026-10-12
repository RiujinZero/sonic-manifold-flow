"""Fixed-capacity sparse node passports for Convergent Gator.

The dense Convergent implementation represents a packet's location as a full
``[..., nodes]`` probability vector.  This module keeps only a fixed number of
live node identifiers and their probability mass.  All public transforms keep
their output capacity independent of how many collisions occur, which makes
the representation suitable for ``torch.compile`` and, later, fused kernels.

Node identifiers are discrete routing decisions.  Probability mass remains a
normal differentiable tensor: dense packing, collision merging, transport,
diagnostic unpacking, and occupancy queries all preserve its gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _mass_threshold(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("mass_threshold must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class PackedNodeFrontier:
    """A fixed-width sparse node distribution.

    ``nodes``, ``mass``, and ``valid`` have identical ``[..., capacity]``
    shapes. Invalid entries are structural padding and never contribute even
    if their stored mass is nonzero. Public constructors canonicalize padding
    to node zero with zero mass.
    """

    nodes: torch.Tensor
    mass: torch.Tensor
    valid: torch.Tensor
    n_nodes: int

    def __post_init__(self) -> None:
        _positive_integer(self.n_nodes, "n_nodes")
        if self.nodes.shape != self.mass.shape or self.nodes.shape != self.valid.shape:
            raise ValueError("nodes, mass, and valid must have identical shapes")
        if self.nodes.ndim < 1 or self.nodes.shape[-1] < 1:
            raise ValueError("a packed frontier needs a nonempty capacity axis")
        if self.nodes.dtype != torch.long:
            raise ValueError("nodes must be an int64 tensor")
        if not self.mass.is_floating_point():
            raise ValueError("mass must be a floating-point tensor")
        if self.valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean tensor")
        if not (
            self.nodes.device == self.mass.device == self.valid.device
        ):
            raise ValueError("nodes, mass, and valid must share a device")
        torch._assert_async(
            self.nodes.ge(0).logical_and(self.nodes.lt(self.n_nodes)).all(),
            "packed node identifier is outside the graph",
        )
        active_mass = torch.where(
            self.valid,
            self.mass,
            torch.zeros((), dtype=self.mass.dtype, device=self.mass.device),
        )
        torch._assert_async(
            torch.isfinite(active_mass).all(),
            "packed node mass must be finite",
        )
        torch._assert_async(
            active_mass.ge(0).all(),
            "packed node mass must be nonnegative",
        )

    @property
    def capacity(self) -> int:
        """Maximum active nodes stored for each packet."""

        return self.nodes.shape[-1]

    @property
    def packet_shape(self) -> torch.Size:
        """Leading packet axes, excluding packed node capacity."""

        return self.nodes.shape[:-1]

    @property
    def masked_mass(self) -> torch.Tensor:
        """Mass with invalid structural entries forced to zero."""

        return torch.where(
            self.valid,
            self.mass,
            torch.zeros((), dtype=self.mass.dtype, device=self.mass.device),
        )

    @classmethod
    def from_dense(
        cls,
        dense: torch.Tensor,
        *,
        top_k: int,
        mass_threshold: float = 0.0,
    ) -> "PackedNodeFrontier":
        """Pack and renormalize the strongest entries of a dense passport."""

        return dense_to_packed_frontier(
            dense,
            top_k=top_k,
            mass_threshold=mass_threshold,
        )

    def to_dense(self) -> torch.Tensor:
        """Materialize the dense diagnostic passport."""

        return packed_frontier_to_dense(self)


def _normalize_selected(
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    *,
    n_nodes: int,
    mass_threshold: float,
) -> PackedNodeFrontier:
    """Apply a per-entry threshold, retain one winner, and normalize."""

    threshold = _mass_threshold(mass_threshold)
    positive = valid & mass.gt(0)
    if threshold > 0.0:
        retained = positive & mass.ge(threshold)
    else:
        retained = positive

    strongest = mass.masked_fill(~positive, -1).argmax(
        dim=-1, keepdim=True
    )
    has_mass = positive.any(dim=-1, keepdim=True)
    strongest_mask = torch.zeros_like(retained).scatter(
        -1, strongest, has_mass
    )
    retained = retained | strongest_mask
    retained_mass = torch.where(retained, mass, torch.zeros_like(mass))
    total = retained_mass.sum(dim=-1, keepdim=True)
    torch._assert_async(
        total.gt(0).all(),
        "each packet must retain positive node mass",
    )
    retained_mass = retained_mass / total.clamp_min(
        torch.finfo(retained_mass.dtype).tiny
    )
    retained_nodes = torch.where(retained, nodes, torch.zeros_like(nodes))
    return PackedNodeFrontier(
        nodes=retained_nodes,
        mass=retained_mass,
        valid=retained,
        n_nodes=n_nodes,
    )


def dense_to_packed_frontier(
    dense: torch.Tensor,
    *,
    top_k: int,
    mass_threshold: float = 0.0,
) -> PackedNodeFrontier:
    """Convert ``[..., N]`` node mass to a fixed-capacity top-k frontier.

    Thresholding happens after top-k selection. At least the strongest entry
    survives, and retained mass is normalized to one. This intentionally
    matches Convergent's existing dense sparsification semantics.
    """

    if dense.ndim < 1 or dense.shape[-1] < 1:
        raise ValueError("dense must have a nonempty node axis")
    if not dense.is_floating_point():
        raise ValueError("dense must be a floating-point tensor")
    top_k = _positive_integer(top_k, "top_k")
    n_nodes = dense.shape[-1]
    if top_k > n_nodes:
        raise ValueError("top_k cannot exceed the dense node count")
    torch._assert_async(
        torch.isfinite(dense).all(),
        "dense node mass must be finite",
    )
    torch._assert_async(
        dense.ge(0).all(),
        "dense node mass must be nonnegative",
    )
    retained_mass, retained_nodes = dense.topk(top_k, dim=-1)
    return _normalize_selected(
        retained_nodes,
        retained_mass,
        torch.ones_like(retained_nodes, dtype=torch.bool),
        n_nodes=n_nodes,
        mass_threshold=mass_threshold,
    )


def packed_frontier_to_dense(frontier: PackedNodeFrontier) -> torch.Tensor:
    """Materialize a dense diagnostic view, summing accidental duplicates."""

    dense = frontier.mass.new_zeros(
        frontier.packet_shape + (frontier.n_nodes,)
    )
    return dense.scatter_add(-1, frontier.nodes, frontier.masked_mass)


def merge_packed_destinations(
    candidate_nodes: torch.Tensor,
    candidate_mass: torch.Tensor,
    candidate_valid: torch.Tensor,
    *,
    n_nodes: int,
    top_k: int,
    mass_threshold: float = 0.0,
) -> PackedNodeFrontier:
    """Merge repeated destination IDs before selecting a normalized top-k.

    Candidate tensors have a fixed ``[..., candidates]`` shape. Sorting and a
    candidate-width ``scatter_add`` merge collisions without allocating an
    ``N``-wide passport. The result always has exactly ``top_k`` storage slots.
    """

    n_nodes = _positive_integer(n_nodes, "n_nodes")
    top_k = _positive_integer(top_k, "top_k")
    if (
        candidate_nodes.shape != candidate_mass.shape
        or candidate_nodes.shape != candidate_valid.shape
    ):
        raise ValueError(
            "candidate_nodes, candidate_mass, and candidate_valid must match"
        )
    if candidate_nodes.ndim < 1 or candidate_nodes.shape[-1] < 1:
        raise ValueError("candidate tensors need a nonempty candidate axis")
    if candidate_nodes.dtype != torch.long:
        raise ValueError("candidate_nodes must be int64")
    if not candidate_mass.is_floating_point():
        raise ValueError("candidate_mass must be floating point")
    if candidate_valid.dtype != torch.bool:
        raise ValueError("candidate_valid must be boolean")
    if not (
        candidate_nodes.device
        == candidate_mass.device
        == candidate_valid.device
    ):
        raise ValueError("candidate tensors must share a device")
    if top_k > min(candidate_nodes.shape[-1], n_nodes):
        raise ValueError("top_k exceeds candidate or graph capacity")
    torch._assert_async(
        candidate_nodes.ge(0).logical_and(candidate_nodes.lt(n_nodes)).all(),
        "candidate destination is outside the graph",
    )
    active_mass = torch.where(
        candidate_valid,
        candidate_mass,
        torch.zeros_like(candidate_mass),
    )
    torch._assert_async(
        torch.isfinite(active_mass).all(),
        "candidate mass must be finite",
    )
    torch._assert_async(
        active_mass.ge(0).all(),
        "candidate mass must be nonnegative",
    )

    sentinel = torch.full_like(candidate_nodes, n_nodes)
    sortable_nodes = torch.where(candidate_valid, candidate_nodes, sentinel)
    sorted_nodes, order = sortable_nodes.sort(dim=-1, stable=True)
    sorted_mass = active_mass.gather(-1, order)
    sorted_valid = candidate_valid.gather(-1, order)

    group_start = sorted_valid.clone()
    group_start[..., 1:] &= (
        sorted_nodes[..., 1:] != sorted_nodes[..., :-1]
    )
    group_index = group_start.to(torch.long).cumsum(dim=-1) - 1
    group_index = group_index.clamp_min(0)
    merged_by_group = torch.zeros_like(sorted_mass).scatter_add(
        -1, group_index, sorted_mass
    )
    merged_at_start = torch.where(
        group_start,
        merged_by_group.gather(-1, group_index),
        torch.zeros_like(sorted_mass),
    )

    retained_mass, retained_positions = merged_at_start.topk(
        top_k, dim=-1
    )
    retained_nodes = sorted_nodes.gather(-1, retained_positions)
    retained_valid = group_start.gather(-1, retained_positions)
    retained_nodes = torch.where(
        retained_valid, retained_nodes, torch.zeros_like(retained_nodes)
    )
    return _normalize_selected(
        retained_nodes,
        retained_mass,
        retained_valid,
        n_nodes=n_nodes,
        mass_threshold=mass_threshold,
    )


def transport_packed_frontier(
    frontier: PackedNodeFrontier,
    destinations: torch.Tensor,
    transition: torch.Tensor,
    *,
    top_k: int | None = None,
    mass_threshold: float = 0.0,
) -> PackedNodeFrontier:
    """Transport packed source mass over graph edges and merge collisions.

    ``destinations`` may be a shared ``[N, degree]`` graph table or already
    gathered ``[..., capacity, degree]`` destination IDs. ``transition`` must
    match the gathered shape and supplies nonnegative edge weights. Final
    normalization makes the primitive robust to masked, not-yet-normalized
    edge weights while preserving all mass gradients.
    """

    if destinations.dtype != torch.long:
        raise ValueError("destinations must be an int64 tensor")
    if destinations.device != frontier.nodes.device:
        raise ValueError("destinations and frontier must share a device")
    if destinations.ndim == 2 and destinations.shape[0] == frontier.n_nodes:
        gathered_destinations = destinations[frontier.nodes]
    else:
        gathered_destinations = destinations
    expected_prefix = frontier.nodes.shape
    if (
        gathered_destinations.ndim != frontier.nodes.ndim + 1
        or gathered_destinations.shape[:-1] != expected_prefix
        or gathered_destinations.shape[-1] < 1
    ):
        raise ValueError(
            "destinations must be [nodes,degree] or match the packed frontier"
        )
    if transition.shape != gathered_destinations.shape:
        raise ValueError("transition must match gathered destinations")
    if not transition.is_floating_point():
        raise ValueError("transition must be floating point")
    if transition.device != frontier.mass.device:
        raise ValueError("transition and frontier must share a device")
    torch._assert_async(
        torch.isfinite(transition).all(),
        "transition weights must be finite",
    )
    torch._assert_async(
        transition.ge(0).all(),
        "transition weights must be nonnegative",
    )

    candidate_valid = frontier.valid.unsqueeze(-1) & transition.gt(0)
    candidate_mass = frontier.masked_mass.unsqueeze(-1) * transition
    candidate_nodes = gathered_destinations
    candidates = frontier.capacity * gathered_destinations.shape[-1]
    packet_shape = frontier.packet_shape
    candidate_nodes = candidate_nodes.reshape(packet_shape + (candidates,))
    candidate_mass = candidate_mass.reshape(packet_shape + (candidates,))
    candidate_valid = candidate_valid.reshape(packet_shape + (candidates,))
    if top_k is None:
        top_k = frontier.capacity
    return merge_packed_destinations(
        candidate_nodes,
        candidate_mass,
        candidate_valid,
        n_nodes=frontier.n_nodes,
        top_k=top_k,
        mass_threshold=mass_threshold,
    )


def packed_node_occupancy(
    frontier: PackedNodeFrontier,
    *,
    position_valid: torch.Tensor | None = None,
    causal: bool,
    query_nodes: torch.Tensor | None = None,
    query_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Query causal or noncausal node occupancy without an ``N``-wide tensor.

    The frontier must have ``[batch, token, lane, capacity]`` shape. Query
    nodes must start with ``[batch, token, ...]`` and the returned tensor has
    the same shape. Occupancy matches the dense Convergent definition: writer
    mass is summed across lanes and divided by ``valid_tokens * lanes``.

    Internally, fixed-count writer keys are sorted by ``(batch,node,token)``.
    A differentiable segmented prefix sum and ``searchsorted`` then answer all
    packed queries. Runtime storage scales with packed writers and queries,
    not with the total graph population.
    """

    if frontier.nodes.ndim != 4:
        raise ValueError(
            "occupancy requires [batch,token,lane,capacity] frontiers"
        )
    batch, tokens, lanes, capacity = frontier.nodes.shape
    if position_valid is None:
        position_valid = torch.ones(
            batch, tokens, dtype=torch.bool, device=frontier.nodes.device
        )
    if position_valid.shape != (batch, tokens) or position_valid.dtype != torch.bool:
        raise ValueError("position_valid must be boolean [batch,token]")
    if position_valid.device != frontier.nodes.device:
        raise ValueError("position_valid and frontier must share a device")

    if query_nodes is None:
        query_nodes = frontier.nodes
        if query_valid is None:
            query_valid = frontier.valid
    if query_nodes.dtype != torch.long:
        raise ValueError("query_nodes must be int64")
    if (
        query_nodes.ndim < 3
        or query_nodes.shape[:2] != (batch, tokens)
    ):
        raise ValueError("query_nodes must begin with [batch,token] axes")
    if query_nodes.device != frontier.nodes.device:
        raise ValueError("query_nodes and frontier must share a device")
    torch._assert_async(
        query_nodes.ge(0).logical_and(query_nodes.lt(frontier.n_nodes)).all(),
        "occupancy query node is outside the graph",
    )
    if query_valid is None:
        query_valid = torch.ones_like(query_nodes, dtype=torch.bool)
    if query_valid.shape != query_nodes.shape or query_valid.dtype != torch.bool:
        raise ValueError("query_valid must be boolean and match query_nodes")
    if query_valid.device != frontier.nodes.device:
        raise ValueError("query_valid and frontier must share a device")

    # A packed sort/search is the right representation while each packet
    # touches far fewer nodes than the graph owns. Once L*K reaches N,
    # however, sorting more packed writer rows than there are graph bins is
    # pure overhead. In that regime a node-only scatter is both smaller than
    # the old dense passport (there is no lane axis) and substantially more
    # GPU friendly. The branch depends only on configured tensor shapes, so it
    # remains stable under torch.compile.
    if frontier.n_nodes <= lanes * capacity:
        writer_valid = (
            frontier.valid & position_valid[:, :, None, None]
        )
        writer_mass = torch.where(
            writer_valid,
            frontier.mass,
            torch.zeros_like(frontier.mass),
        )
        position_mass = frontier.mass.new_zeros(
            batch,
            tokens,
            frontier.n_nodes,
        ).scatter_add(
            2,
            frontier.nodes.reshape(batch, tokens, lanes * capacity),
            writer_mass.reshape(batch, tokens, lanes * capacity),
        )
        if causal:
            node_occupancy = position_mass.cumsum(dim=1)
            denominator = position_valid.cumsum(dim=1).to(
                frontier.mass.dtype
            )
        else:
            node_occupancy = position_mass.sum(
                dim=1,
                keepdim=True,
            ).expand(-1, tokens, -1)
            denominator = position_valid.sum(
                dim=1,
                keepdim=True,
            ).to(frontier.mass.dtype).expand(-1, tokens)
        query_shape = query_nodes.shape
        occupancy = node_occupancy.gather(
            2,
            query_nodes.reshape(batch, tokens, -1),
        ).reshape(query_shape)
        denominator = (denominator * lanes).clamp_min(1).view(
            (batch, tokens) + (1,) * (query_nodes.ndim - 2)
        )
        return torch.where(
            query_valid,
            occupancy / denominator,
            torch.zeros_like(occupancy),
        )

    batch_ids = torch.arange(
        batch, device=frontier.nodes.device, dtype=torch.long
    )[:, None, None, None]
    token_ids = torch.arange(
        tokens, device=frontier.nodes.device, dtype=torch.long
    )[None, :, None, None]
    writer_group = batch_ids * frontier.n_nodes + frontier.nodes
    writer_key = writer_group * tokens + token_ids
    writer_valid = frontier.valid & position_valid[:, :, None, None]
    writer_mass = torch.where(
        writer_valid, frontier.mass, torch.zeros_like(frontier.mass)
    )

    flat_key = writer_key.reshape(-1)
    flat_valid = writer_valid.reshape(-1)
    sentinel_key = batch * frontier.n_nodes * tokens
    flat_key = torch.where(
        flat_valid, flat_key, torch.full_like(flat_key, sentinel_key)
    )
    sorted_key, order = flat_key.sort(stable=True)
    sorted_valid = flat_valid[order]
    sorted_mass = writer_mass.reshape(-1)[order]
    sorted_group = torch.div(sorted_key, tokens, rounding_mode="floor")

    group_start = sorted_valid.clone()
    group_start[1:] &= sorted_group[1:] != sorted_group[:-1]
    global_prefix = sorted_mass.cumsum(0)
    prefix_before_group = torch.where(
        group_start,
        global_prefix - sorted_mass,
        torch.zeros_like(global_prefix),
    )
    positions = torch.arange(
        sorted_mass.numel(), device=frontier.nodes.device, dtype=torch.long
    )
    latest_group_start = torch.where(
        group_start, positions, torch.zeros_like(positions)
    ).cummax(0).values
    segmented_prefix = global_prefix - prefix_before_group[latest_group_start]

    query_tail = query_nodes.shape[2:]
    query_batch = torch.arange(
        batch, device=query_nodes.device, dtype=torch.long
    ).view((batch, 1) + (1,) * len(query_tail))
    query_group = query_batch * frontier.n_nodes + query_nodes
    if causal:
        query_time = torch.arange(
            tokens, device=query_nodes.device, dtype=torch.long
        ).view((1, tokens) + (1,) * len(query_tail))
        query_time = query_time.expand_as(query_nodes)
    else:
        query_time = torch.full_like(query_nodes, tokens - 1)
    query_key = (query_group * tokens + query_time).reshape(-1)
    found_index = torch.searchsorted(sorted_key, query_key, right=True) - 1
    safe_index = found_index.clamp(min=0, max=sorted_key.numel() - 1)
    found = (
        found_index.ge(0)
        & sorted_valid[safe_index]
        & sorted_group[safe_index].eq(query_group.reshape(-1))
    )
    occupancy = torch.where(
        found,
        segmented_prefix[safe_index],
        torch.zeros_like(segmented_prefix[safe_index]),
    ).reshape(query_nodes.shape)

    if causal:
        denominator = position_valid.cumsum(dim=1).to(frontier.mass.dtype)
    else:
        denominator = position_valid.sum(dim=1, keepdim=True).to(
            frontier.mass.dtype
        ).expand(-1, tokens)
    denominator = (denominator * lanes).clamp_min(1)
    denominator = denominator.view(
        (batch, tokens) + (1,) * len(query_tail)
    )
    # ``position_valid`` controls which positions contribute as writers and
    # the normalization count. It does not disable a receiver query: the
    # dense Convergent oracle still computes a state for padded/output-masked
    # receivers, even though their result is excluded from the objective.
    live_query = query_valid
    return torch.where(
        live_query,
        occupancy / denominator,
        torch.zeros_like(occupancy),
    )


__all__ = [
    "PackedNodeFrontier",
    "dense_to_packed_frontier",
    "merge_packed_destinations",
    "packed_frontier_to_dense",
    "packed_node_occupancy",
    "transport_packed_frontier",
]
