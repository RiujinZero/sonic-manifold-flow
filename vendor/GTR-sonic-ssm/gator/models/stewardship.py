"""Token-stewardship primitives.

Ownership, branch state, and the node passport always travel together through
learned handoffs.  The dense transport is the semantic reference.  The packed
helpers represent only live ``(batch, cell, token)`` branches so a sparse relay
kernel can avoid evaluating dormant stewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .sparse_kernels import build_packed_group_plan


SparseStewardshipBackend = Literal["repack", "persistent_bucketed"]

# A route carrying less than one tenth of one percent of token mass is
# numerically present but not trustworthy enough to receive an unbounded
# conditional-state gain. Keeping the floor below the 1% scout budget
# preserves deliberate exploration while bounding BF16 backward gain.
STEWARDSHIP_NORMALISATION_MASS_FLOOR = 1e-3


@dataclass(frozen=True)
class PackedStewardship:
    """Ragged live stewardship branches.

    Every row is one live ``(batch, cell, token)`` route.  ``mass`` is the
    token's ownership of that route; ``state`` and ``node`` are conditional on
    the route and therefore must be transported with it.
    """

    batch: torch.Tensor
    cell: torch.Tensor
    token: torch.Tensor
    mass: torch.Tensor
    state: torch.Tensor
    node: torch.Tensor

    def __post_init__(self) -> None:
        rows = self.mass.shape[0] if self.mass.ndim == 1 else -1
        if (
            self.batch.ndim != 1
            or self.cell.ndim != 1
            or self.token.ndim != 1
            or self.state.ndim != 2
            or self.node.ndim != 2
            or rows < 0
            or any(
                value.shape[0] != rows
                for value in (
                    self.batch,
                    self.cell,
                    self.token,
                    self.state,
                    self.node,
                )
            )
        ):
            raise ValueError(
                "packed indices/mass must be row vectors and state/node "
                "must have the same leading row count"
            )
        if any(
            index.dtype != torch.long
            for index in (self.batch, self.cell, self.token)
        ):
            raise ValueError("packed indices must use int64")
        if not self.mass.is_floating_point():
            raise ValueError("packed mass must be floating point")
        devices = {
            value.device
            for value in (
                self.batch,
                self.cell,
                self.token,
                self.mass,
                self.state,
                self.node,
            )
        }
        if len(devices) != 1:
            raise ValueError("every packed field must use the same device")

    @property
    def size(self) -> int:
        return int(self.mass.numel())


def ring_handoff_offsets(neighbors: int, *, device: torch.device | None = None) -> torch.Tensor:
    """Return self, then alternating left/right ring offsets."""

    if neighbors <= 0 or neighbors % 2:
        raise ValueError("neighbors must be a positive even number")
    offsets = [0]
    for distance in range(1, neighbors // 2 + 1):
        offsets.extend((-distance, distance))
    return torch.tensor(offsets, dtype=torch.long, device=device)


def ring_handoff_destinations(
    cells: int, neighbors: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """Map every source cell/action pair to its ring destination."""

    offsets = ring_handoff_offsets(neighbors, device=device)
    sources = torch.arange(cells, device=device)[:, None]
    return (sources + offsets[None]).remainder(cells)


def adaptive_nucleus_stewardship(
    ownership: torch.Tensor,
    *,
    keep_mass: float | torch.Tensor,
    min_active: int = 1,
    max_active: int | None = None,
    cell_groups: torch.Tensor | None = None,
    cell_group_count: int | None = None,
    scouts: int = 0,
    exploration_mass: float = 0.0,
    straight_through: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep an elastic nucleus of cells for each batch/token pair.

    ``ownership`` uses ``[B, C, T]`` layout. Familiar, peaked routes retain a
    small support, while diffuse routes retain enough cells to cover
    ``keep_mass``. Optional cell groups guarantee at least one live steward per
    language-output channel. Optional scouts reserve a small amount of mass
    outside the exploitation nucleus.  That makes narrowing reversible: a
    familiar route can stay sharp while a live scout continues testing an
    alternate path. The returned boolean mask is the actual forward support.
    """

    if ownership.ndim != 3:
        raise ValueError("ownership must have [batch, cells, tokens] shape")
    batch, cells, tokens = ownership.shape
    if not 1 <= min_active <= cells:
        raise ValueError("min_active must be between one and the cell count")
    maximum = cells if max_active is None else max_active
    if not min_active <= maximum <= cells:
        raise ValueError("max_active must be between min_active and the cell count")
    if not 0 <= scouts < cells:
        raise ValueError("scouts must be between zero and cells - 1")
    if not 0.0 <= exploration_mass < 1.0:
        raise ValueError("exploration_mass must be in [0, 1)")

    probabilities = ownership.transpose(1, 2)  # [B, T, C]
    sorted_probabilities, sorted_indices = probabilities.sort(dim=-1, descending=True)
    cumulative_before = sorted_probabilities.cumsum(-1) - sorted_probabilities
    if isinstance(keep_mass, torch.Tensor):
        threshold = keep_mass.to(probabilities).clamp(0.0, 1.0)
        if threshold.ndim == 0:
            threshold = threshold.expand(batch, tokens)
        if threshold.shape != (batch, tokens):
            raise ValueError("tensor keep_mass must have [batch, tokens] shape")
        threshold = threshold.unsqueeze(-1)
    else:
        if not 0.0 < keep_mass <= 1.0:
            raise ValueError("keep_mass must be in (0, 1]")
        threshold = torch.full(
            (batch, tokens, 1), keep_mass, dtype=probabilities.dtype, device=probabilities.device
        )

    ranks = torch.arange(cells, device=probabilities.device).view(1, 1, cells)
    sorted_keep = cumulative_before < threshold
    sorted_keep = sorted_keep | (ranks < min_active)
    sorted_keep = sorted_keep & sorted_probabilities.gt(0)
    mask = torch.zeros_like(sorted_keep).scatter(2, sorted_indices, sorted_keep)

    forced = torch.zeros_like(mask)
    group_count = 0
    if cell_groups is not None:
        groups = cell_groups.to(device=probabilities.device, dtype=torch.long)
        if groups.shape != (cells,):
            raise ValueError("cell_groups must have one entry per cell")
        group_count = (
            int(groups.max().detach().cpu()) + 1
            if cell_group_count is None
            else cell_group_count
        )
        if group_count <= 0:
            raise ValueError("cell_group_count must be positive")
        if maximum < group_count:
            raise ValueError("max_active must cover every required cell group")
        membership = groups[None].eq(
            torch.arange(group_count, device=groups.device)[:, None]
        )
        group_probabilities = probabilities[:, :, None].masked_fill(
            ~membership[None, None], -1.0
        )
        best = group_probabilities.argmax(-1)
        group_has_mass = (
            probabilities[:, :, None]
            * membership[None, None].to(probabilities.dtype)
        ).sum(-1).gt(0)
        forced = forced.scatter(2, best, group_has_mass)

    # First preserve the complete exploitation nucleus up to the configured
    # cap. Scouts use only genuinely spare capacity; when uncertainty already
    # makes the nucleus cover every action/cell, no candidate is demoted to a
    # tiny scout merely to satisfy the exploration setting.
    candidates = mask | forced
    priority = probabilities + forced.to(probabilities.dtype) * 2.0
    priority = priority.masked_fill(~candidates, -1.0)
    candidate_order = priority.argsort(dim=-1, descending=True)
    candidate_valid = priority.gather(2, candidate_order).ge(0.0)
    exploit_sorted = candidate_valid & (ranks < maximum)
    exploit_mask = torch.zeros_like(mask).scatter(2, candidate_order, exploit_sorted)

    scout_mask = torch.zeros_like(mask)
    if scouts and exploration_mass:
        scout_candidates = probabilities.masked_fill(
            exploit_mask | probabilities.le(0), -1.0
        )
        scout_order = scout_candidates.argsort(dim=-1, descending=True)
        scout_valid = scout_candidates.gather(2, scout_order).ge(0.0)
        available = (maximum - exploit_mask.sum(-1)).clamp_min(0)
        scout_sorted = (
            scout_valid
            & (ranks < scouts)
            & (ranks < available.unsqueeze(-1))
        )
        scout_mask = torch.zeros_like(mask).scatter(2, scout_order, scout_sorted)

    exploit = probabilities * exploit_mask.to(probabilities.dtype)
    exploit = exploit / exploit.sum(-1, keepdim=True).clamp_min(1e-9)
    scout = probabilities * scout_mask.to(probabilities.dtype)
    scout_total = scout.sum(-1, keepdim=True)
    scout = scout / scout_total.clamp_min(1e-9)
    has_scout = scout_total.gt(0).to(probabilities.dtype)
    reserved_mass = exploration_mass * has_scout
    hard = (1.0 - reserved_mass) * exploit + reserved_mass * scout
    final_mask = exploit_mask | scout_mask
    if straight_through:
        routed = probabilities + (hard - probabilities).detach()
    else:
        routed = hard
    return routed.transpose(1, 2), final_mask.transpose(1, 2)


def elastic_handoff_beam(
    handoffs: torch.Tensor,
    handoff_entropy: torch.Tensor,
    *,
    base_keep_mass: float,
    scouts: int,
    exploration_mass: float,
    straight_through: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn each steward's action distribution into an elastic local beam.

    A confident source keeps a small exploitation set plus optional scout
    actions. An uncertain source raises its retained mass toward one and can
    therefore send the token to several receivers. Leading dimensions are
    arbitrary; only the final action axis is interpreted.
    """

    if handoffs.ndim < 2 or handoff_entropy.shape != handoffs.shape[:-1]:
        raise ValueError("handoff entropy must match every non-action axis")
    if not 0.0 < base_keep_mass <= 1.0:
        raise ValueError("base_keep_mass must be in (0, 1]")
    actions = handoffs.shape[-1]
    if not 0 <= scouts < actions:
        raise ValueError("handoff scouts must be between zero and actions - 1")

    probabilities = handoffs
    if exploration_mass:
        probabilities = (
            (1.0 - exploration_mass) * probabilities
            + exploration_mass / actions
        )
    flat_probabilities = probabilities.reshape(-1, actions)
    flat_entropy = handoff_entropy.reshape(-1)
    keep_mass = base_keep_mass + (1.0 - base_keep_mass) * flat_entropy
    routed, mask = adaptive_nucleus_stewardship(
        flat_probabilities.unsqueeze(-1),
        keep_mass=keep_mass[:, None],
        min_active=1,
        scouts=scouts,
        exploration_mass=exploration_mass,
        straight_through=straight_through,
    )
    return routed.squeeze(-1).reshape_as(handoffs), mask.squeeze(-1).reshape_as(handoffs)


def pack_stewardship(
    ownership: torch.Tensor,
    states: torch.Tensor,
    node_locations: torch.Tensor,
) -> PackedStewardship:
    """Pack the exact nonzero stewardship support into ragged branch rows."""

    if ownership.ndim != 3:
        raise ValueError("ownership must have [batch, cells, tokens] shape")
    if states.shape[:3] != ownership.shape or node_locations.shape[:3] != ownership.shape:
        raise ValueError("state and node prefixes must match ownership")
    batch, cell, token = ownership.gt(0).nonzero(as_tuple=True)
    return PackedStewardship(
        batch=batch,
        cell=cell,
        token=token,
        mass=ownership[batch, cell, token],
        state=states[batch, cell, token],
        node=node_locations[batch, cell, token],
    )


def unpack_stewardship(
    packed: PackedStewardship,
    *,
    batch_size: int,
    cells: int,
    tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand packed branches for diagnostics and dense equivalence tests."""

    hidden = packed.state.shape[-1]
    nodes = packed.node.shape[-1]
    ownership = packed.mass.new_zeros(batch_size, cells, tokens)
    states = packed.state.new_zeros(batch_size, cells, tokens, hidden)
    locations = packed.node.new_zeros(batch_size, cells, tokens, nodes)
    index = (packed.batch, packed.cell, packed.token)
    ownership = ownership.index_put(index, packed.mass)
    states = states.index_put(index, packed.state)
    locations = locations.index_put(index, packed.node)
    return ownership, states, locations


def packed_ownership(
    packed: PackedStewardship,
    *,
    batch_size: int,
    cells: int,
    tokens: int,
) -> torch.Tensor:
    """Materialize only the scalar ownership grid from packed branches."""

    ownership = packed.mass.new_zeros(batch_size, cells, tokens)
    return ownership.index_put(
        (packed.batch, packed.cell, packed.token),
        packed.mass,
    )


def packed_route_uncertainty(
    packed: PackedStewardship,
    handoff_entropy: torch.Tensor,
    *,
    batch_size: int,
    tokens: int,
    route_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-branch uncertainty without a dense cell axis."""

    if handoff_entropy.shape != packed.mass.shape:
        raise ValueError("handoff_entropy must have one value per packed row")
    if route_valid.shape != (batch_size, tokens):
        raise ValueError("route_valid must have [batch,tokens] shape")
    key = packed.batch * tokens + packed.token
    maximum = handoff_entropy.new_full(
        (batch_size * tokens,), -torch.inf
    ).scatter_reduce(
        0,
        key,
        handoff_entropy,
        reduce="amax",
        include_self=True,
    )
    mean = handoff_entropy.new_zeros(batch_size * tokens).scatter_add(
        0,
        key,
        packed.mass * handoff_entropy,
    )
    maximum = maximum.view(batch_size, tokens)
    mean = mean.view(batch_size, tokens)
    valid = route_valid.to(device=maximum.device, dtype=torch.bool)
    maximum = torch.where(
        valid & maximum.isfinite(),
        maximum,
        torch.zeros_like(maximum),
    )
    mean = torch.where(valid, mean, torch.zeros_like(mean))
    return maximum, mean


def packed_weighted_token_sum(
    packed: PackedStewardship,
    values: torch.Tensor,
    *,
    batch_size: int,
    tokens: int,
) -> torch.Tensor:
    """Ownership-weighted sum of packed row values by batch/token."""

    if values.ndim < 1 or values.shape[0] != packed.size:
        raise ValueError("values must have one leading row per packed branch")
    key = packed.batch * tokens + packed.token
    tail = values.shape[1:]
    weighted = packed.mass.reshape(
        packed.size, *([1] * len(tail))
    ) * values
    result = values.new_zeros(batch_size * tokens, *tail)
    index = key.reshape(
        packed.size, *([1] * len(tail))
    ).expand_as(weighted)
    return result.scatter_add(0, index, weighted).view(
        batch_size, tokens, *tail
    )


def packed_handoffs_to_dense(
    packed: PackedStewardship,
    handoffs: torch.Tensor,
    *,
    batch_size: int,
    cells: int,
    tokens: int,
) -> torch.Tensor:
    """Expand packed handoffs only when route diagnostics request them."""

    if handoffs.ndim != 2 or handoffs.shape[0] != packed.size:
        raise ValueError("handoffs must have [packed rows,actions] shape")
    dense = handoffs.new_zeros(
        batch_size, cells, tokens, handoffs.shape[-1]
    )
    return dense.index_put(
        (packed.batch, packed.cell, packed.token),
        handoffs,
    )


def padded_packed_receiver_context(
    packed: PackedStewardship,
    *,
    write: torch.Tensor,
    relative_code: torch.Tensor,
    relative_node_bias: torch.Tensor,
    cells: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Legacy globally padded receiver memory for live rows.

    Rows are grouped by ``(batch, cell)`` and every occupied group is padded to
    the busiest live steward. The receiver/writer interaction therefore scales
    with ``G_active * L_max^2``. This implementation is retained as the
    forward numerical oracle for the bucketed path.
    """

    if packed.size == 0:
        raise ValueError("packed stewardship cannot be empty")
    batch_size, tokens = write.shape
    nodes = packed.node.shape[-1]
    hidden = packed.state.shape[-1]
    group_key = packed.batch * cells + packed.cell
    order = group_key.argsort(stable=True)
    sorted_key = group_key[order]
    unique_key, counts = sorted_key.unique_consecutive(return_counts=True)
    groups = int(unique_key.numel())
    maximum = int(counts.max().detach().cpu())
    starts = counts.cumsum(0) - counts
    group_index = torch.repeat_interleave(
        torch.arange(groups, device=order.device), counts
    )
    rank = torch.arange(packed.size, device=order.device) - torch.repeat_interleave(
        starts, counts
    )

    state = packed.state[order]
    node = packed.node[order]
    mass = packed.mass[order]
    token_index = packed.token[order]
    batch_index = packed.batch[order]

    padded_state = state.new_zeros(groups, maximum, hidden)
    padded_node = node.new_zeros(groups, maximum, nodes)
    padded_mass = mass.new_zeros(groups, maximum)
    padded_token = token_index.new_zeros(groups, maximum)
    valid = torch.zeros(groups, maximum, dtype=torch.bool, device=order.device)
    slot = (group_index, rank)
    padded_state = padded_state.index_put(slot, state)
    padded_node = padded_node.index_put(slot, node)
    padded_mass = padded_mass.index_put(slot, mass)
    padded_token = padded_token.index_put(slot, token_index)
    valid = valid.index_put(slot, torch.ones_like(mass, dtype=torch.bool))

    group_batch = unique_key.div(cells, rounding_mode="floor")
    group_cell = unique_key.remainder(cells)
    group_write = write[group_batch[:, None], padded_token]
    writer_mass = padded_mass * group_write * valid.to(mass.dtype)
    relative = relative_code[
        padded_token[:, :, None], padded_token[:, None, :]
    ]
    relative_bias = torch.einsum(
        "gijr,gnr->gijn", relative, relative_node_bias[group_cell]
    )
    scores = (
        padded_node[:, None].clamp_min(1e-9).log()
        + relative_bias
        + writer_mass[:, None, :, None].clamp_min(1e-9).log()
    )
    writer_valid = writer_mass.gt(0)[:, None, :, None]
    scores = scores.masked_fill(~writer_valid, -torch.inf)
    has_writer = writer_valid.any(2, keepdim=True)
    safe_scores = torch.where(has_writer, scores, torch.zeros_like(scores))
    attention = safe_scores.softmax(2) * writer_valid.to(scores.dtype)
    attention = attention * has_writer.to(scores.dtype)
    node_memory = torch.einsum("gijn,gjd->gind", attention, padded_state)
    context = (padded_node.unsqueeze(-1) * node_memory).sum(2)

    global_mass = write.sum(1).clamp_min(1e-6)
    traffic = (
        padded_node * writer_mass.unsqueeze(-1)
    ).sum(1) * cells / global_mass[group_batch, None]
    padded_traffic = traffic[:, None].expand(-1, maximum, -1)
    context_sorted = context[group_index, rank]
    traffic_sorted = padded_traffic[group_index, rank]
    context_rows = context_sorted.new_empty(context_sorted.shape).index_copy(
        0, order, context_sorted
    )
    traffic_rows = traffic_sorted.new_empty(traffic_sorted.shape).index_copy(
        0, order, traffic_sorted
    )
    return context_rows, traffic_rows


def bucketed_packed_receiver_context(
    packed: PackedStewardship,
    *,
    write: torch.Tensor,
    relative_code: torch.Tensor,
    relative_node_bias: torch.Tensor,
    cells: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact receiver memory in power-of-two live-token buckets."""

    if packed.size == 0:
        raise ValueError("packed stewardship cannot be empty")
    batch_size, tokens = write.shape
    nodes = packed.node.shape[-1]
    hidden = packed.state.shape[-1]
    group_key = packed.batch * cells + packed.cell
    plan = build_packed_group_plan(
        group_key,
        maximum_group_size=tokens,
    )
    context_rows = packed.state.new_zeros(packed.size, hidden)
    traffic_rows = packed.node.new_zeros(packed.size, nodes)
    global_mass = write.sum(1).clamp_min(1e-6)

    for bucket in plan.buckets:
        row = bucket.row_indices
        valid = bucket.valid
        valid_float = valid.to(packed.mass.dtype)
        state = packed.state[row] * valid[..., None].to(
            packed.state.dtype
        )
        node = packed.node[row] * valid[..., None].to(packed.node.dtype)
        mass = packed.mass[row] * valid_float
        token_index = packed.token[row]
        group_batch = bucket.group_keys.div(cells, rounding_mode="floor")
        group_cell = bucket.group_keys.remainder(cells)

        group_write = write[group_batch[:, None], token_index]
        writer_mass = mass * group_write * valid_float
        relative = relative_code[
            token_index[:, :, None], token_index[:, None, :]
        ]
        relative_bias = torch.einsum(
            "gijr,gnr->gijn",
            relative,
            relative_node_bias[group_cell],
        )
        scores = (
            node[:, None].clamp_min(1e-9).log()
            + relative_bias
            + writer_mass[:, None, :, None].clamp_min(1e-9).log()
        )
        writer_valid = writer_mass.gt(0)[:, None, :, None]
        scores = scores.masked_fill(~writer_valid, -torch.inf)
        has_writer = writer_valid.any(2, keepdim=True)
        safe_scores = torch.where(
            has_writer, scores, torch.zeros_like(scores)
        )
        attention = safe_scores.softmax(2) * writer_valid.to(scores.dtype)
        attention = attention * has_writer.to(scores.dtype)
        node_memory = torch.einsum("gijn,gjd->gind", attention, state)
        context = (node.unsqueeze(-1) * node_memory).sum(2)
        context = context * valid[..., None].to(context.dtype)

        traffic = (
            node * writer_mass.unsqueeze(-1)
        ).sum(1) * cells / global_mass[group_batch, None]
        traffic = traffic[:, None].expand(
            -1, bucket.capacity, -1
        )
        traffic = traffic * valid[..., None].to(traffic.dtype)

        context_rows = context_rows.index_add(
            0,
            row.reshape(-1),
            context.reshape(-1, hidden),
        )
        traffic_rows = traffic_rows.index_add(
            0,
            row.reshape(-1),
            traffic.reshape(-1, nodes),
        )
    return context_rows, traffic_rows


def packed_receiver_context(
    packed: PackedStewardship,
    *,
    write: torch.Tensor,
    relative_code: torch.Tensor,
    relative_node_bias: torch.Tensor,
    cells: int,
    backend: Literal["padded", "bucketed"] = "bucketed",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch exact packed receiver memory to its selected layout."""

    if backend == "padded":
        return padded_packed_receiver_context(
            packed,
            write=write,
            relative_code=relative_code,
            relative_node_bias=relative_node_bias,
            cells=cells,
        )
    if backend == "bucketed":
        return bucketed_packed_receiver_context(
            packed,
            write=write,
            relative_code=relative_code,
            relative_node_bias=relative_node_bias,
            cells=cells,
        )
    raise ValueError("packed receiver backend must be 'padded' or 'bucketed'")


def prune_packed_stewardship(
    packed: PackedStewardship,
    *,
    keep_mass: float | torch.Tensor,
    batch_size: int,
    cells: int,
    tokens: int,
    min_active: int = 1,
    max_active: int | None = None,
    cell_groups: torch.Tensor | None = None,
    cell_group_count: int | None = None,
    scouts: int = 0,
    exploration_mass: float = 0.0,
    straight_through: bool = True,
) -> tuple[PackedStewardship, torch.Tensor]:
    """Apply the global adaptive nucleus while remaining in packed form.

    Rows are grouped by ``(batch, token)``. The returned mask addresses rows in
    the input packed record, which makes support changes auditable without
    constructing dense hidden or node tensors.
    """

    if packed.size == 0:
        raise ValueError("packed stewardship cannot be empty")
    if not 1 <= min_active <= cells:
        raise ValueError("min_active must be between one and the cell count")
    maximum = cells if max_active is None else max_active
    if not min_active <= maximum <= cells:
        raise ValueError("max_active must be between min_active and the cell count")
    if not 0 <= scouts < cells:
        raise ValueError("scouts must be between zero and cells - 1")
    if not 0.0 <= exploration_mass < 1.0:
        raise ValueError("exploration_mass must be in [0, 1)")
    if isinstance(keep_mass, torch.Tensor):
        threshold = keep_mass.to(packed.mass)
        if threshold.ndim == 0:
            threshold = threshold.expand(batch_size, tokens)
        if threshold.shape != (batch_size, tokens):
            raise ValueError("tensor keep_mass must have [batch,tokens] shape")
        threshold = threshold.clamp(0.0, 1.0)
    else:
        if not 0.0 < keep_mass <= 1.0:
            raise ValueError("keep_mass must be in (0, 1]")
        threshold = packed.mass.new_full(
            (batch_size, tokens), keep_mass
        )

    groups = None
    group_count = 0
    if cell_groups is not None:
        groups = cell_groups.to(
            device=packed.mass.device, dtype=torch.long
        )
        if groups.shape != (cells,):
            raise ValueError("cell_groups must have one entry per cell")
        if cell_group_count is None:
            group_count = int(groups.max().detach().cpu()) + 1
        else:
            group_count = cell_group_count
        if group_count <= 0:
            raise ValueError("cell_group_count must be positive")
        if maximum < group_count:
            raise ValueError("max_active must cover every required cell group")

    token_key = packed.batch * tokens + packed.token
    plan = build_packed_group_plan(
        token_key,
        maximum_group_size=cells,
    )
    routed_rows = packed.mass.new_zeros(packed.size)
    keep_count = torch.zeros(
        packed.size, dtype=torch.long, device=packed.mass.device
    )

    for bucket in plan.buckets:
        row = bucket.row_indices
        valid = bucket.valid
        probabilities = packed.mass[row] * valid.to(packed.mass.dtype)
        sorted_probabilities, sorted_indices = probabilities.sort(
            dim=-1, descending=True
        )
        cumulative_before = (
            sorted_probabilities.cumsum(-1) - sorted_probabilities
        )
        group_batch = bucket.group_keys.div(tokens, rounding_mode="floor")
        group_token = bucket.group_keys.remainder(tokens)
        group_threshold = threshold[group_batch, group_token, None]
        ranks = torch.arange(
            bucket.capacity,
            device=packed.mass.device,
        )[None]
        sorted_keep = cumulative_before < group_threshold
        sorted_keep = sorted_keep | (ranks < min_active)
        sorted_keep = sorted_keep & sorted_probabilities.gt(0)
        exploitation = torch.zeros_like(sorted_keep).scatter(
            1, sorted_indices, sorted_keep
        )

        forced = torch.zeros_like(exploitation)
        if groups is not None:
            row_groups = groups[packed.cell[row]]
            membership = (
                row_groups[:, None]
                == torch.arange(
                    group_count, device=packed.mass.device
                )[None, :, None]
            )
            membership = membership & valid[:, None]
            group_probabilities = probabilities[:, None].masked_fill(
                ~membership, -1.0
            )
            best = group_probabilities.argmax(-1)
            group_has_mass = (
                probabilities[:, None]
                * membership.to(probabilities.dtype)
            ).sum(-1).gt(0)
            forced = forced.scatter(1, best, group_has_mass)

        candidates = exploitation | forced
        priority = probabilities + forced.to(probabilities.dtype) * 2.0
        priority = priority.masked_fill(~candidates, -1.0)
        candidate_order = priority.argsort(dim=-1, descending=True)
        candidate_valid = priority.gather(1, candidate_order).ge(0.0)
        exploit_sorted = candidate_valid & (ranks < maximum)
        exploit_mask = torch.zeros_like(exploitation).scatter(
            1, candidate_order, exploit_sorted
        )

        scout_mask = torch.zeros_like(exploitation)
        if scouts and exploration_mass:
            scout_candidates = probabilities.masked_fill(
                exploit_mask | probabilities.le(0), -1.0
            )
            scout_order = scout_candidates.argsort(
                dim=-1, descending=True
            )
            scout_valid = scout_candidates.gather(
                1, scout_order
            ).ge(0.0)
            available = (
                maximum - exploit_mask.sum(-1)
            ).clamp_min(0)
            scout_sorted = (
                scout_valid
                & (ranks < scouts)
                & (ranks < available[:, None])
            )
            scout_mask = torch.zeros_like(exploitation).scatter(
                1, scout_order, scout_sorted
            )

        exploit = probabilities * exploit_mask.to(probabilities.dtype)
        exploit = exploit / exploit.sum(
            -1, keepdim=True
        ).clamp_min(1e-9)
        scout = probabilities * scout_mask.to(probabilities.dtype)
        scout_total = scout.sum(-1, keepdim=True)
        scout = scout / scout_total.clamp_min(1e-9)
        has_scout = scout_total.gt(0).to(probabilities.dtype)
        reserved_mass = exploration_mass * has_scout
        hard = (1.0 - reserved_mass) * exploit + reserved_mass * scout
        final_mask = (exploit_mask | scout_mask) & valid
        routed = (
            probabilities + (hard - probabilities).detach()
            if straight_through
            else hard
        )
        routed = routed * valid.to(routed.dtype)

        routed_rows = routed_rows.index_add(
            0, row.reshape(-1), routed.reshape(-1)
        )
        keep_count = keep_count.index_add(
            0,
            row.reshape(-1),
            final_mask.to(torch.long).reshape(-1),
        )

    keep = keep_count.gt(0)
    index = keep.nonzero(as_tuple=False).flatten()
    return (
        PackedStewardship(
            batch=packed.batch[index],
            cell=packed.cell[index],
            token=packed.token[index],
            mass=routed_rows[index],
            state=packed.state[index],
            node=packed.node[index],
        ),
        keep,
    )


def packed_readout_hidden(
    packed: PackedStewardship,
    address_state: torch.Tensor,
    *,
    batch_size: int,
    context_tokens: int,
    output_tokens: int,
    cell_groups: torch.Tensor | None = None,
    cell_group_count: int | None = None,
) -> torch.Tensor:
    """Ownership-weight packed branches directly into output-slot states."""

    if address_state.ndim != 3:
        raise ValueError("address_state must have [cells,tokens,hidden] shape")
    total_tokens = context_tokens + output_tokens
    if address_state.shape[1] != total_tokens:
        raise ValueError("address_state token dimension does not match the window")
    output_row = packed.token.ge(context_tokens) & packed.token.lt(
        total_tokens
    )
    row = output_row.nonzero(as_tuple=False).flatten()
    if not row.numel():
        raise ValueError("packed stewardship contains no output-token rows")
    batch = packed.batch[row]
    cell = packed.cell[row]
    token = packed.token[row]
    output_index = token - context_tokens
    hidden = packed.state[row] + address_state[cell, token]
    mass = packed.mass[row]
    width = hidden.shape[-1]

    if cell_groups is None:
        key = batch * output_tokens + output_index
        result = hidden.new_zeros(batch_size * output_tokens, width)
        result = result.scatter_add(
            0,
            key[:, None].expand(-1, width),
            mass[:, None] * hidden,
        )
        return result.view(batch_size, output_tokens, width)

    groups = cell_groups.to(device=cell.device, dtype=torch.long)
    if groups.shape != (address_state.shape[0],):
        raise ValueError("cell_groups must have one entry per cell")
    group_count = (
        int(groups.max().detach().cpu()) + 1
        if cell_group_count is None
        else cell_group_count
    )
    if group_count <= 0:
        raise ValueError("cell_group_count must be positive")
    group = groups[cell]
    key = (
        (batch * output_tokens + output_index) * group_count + group
    )
    numerator = hidden.new_zeros(
        batch_size * output_tokens * group_count, width
    ).scatter_add(
        0,
        key[:, None].expand(-1, width),
        mass[:, None] * hidden,
    )
    denominator = mass.new_zeros(
        batch_size * output_tokens * group_count
    ).scatter_add(0, key, mass)
    grouped = normalise_stewardship_messages(numerator, denominator)
    return grouped.view(
        batch_size, output_tokens, group_count, width
    ).reshape(batch_size, output_tokens, group_count * width)


def normalise_stewardship_messages(
    message_sum: torch.Tensor,
    mass: torch.Tensor,
    *,
    minimum_mass: float = STEWARDSHIP_NORMALISATION_MASS_FLOOR,
) -> torch.Tensor:
    """Normalize stewardship messages with bounded near-empty route gain.

    Straight-through pruning can force a private output group to remain live
    after its soft probability has fallen to the BF16 underflow boundary. A
    literal weighted mean then has a backward gain proportional to ``1/mass``;
    at roughly ``1e-40`` that produces NaNs even though the forward result is
    finite. Empty routes remain exactly zero. Near-empty routes are attenuated
    linearly by ``mass / minimum_mass`` while retaining a bounded gradient that
    can revive them. Routes above the floor retain the historical weighted
    mean exactly, including the configured one-percent scout paths.
    """

    if message_sum.shape[:-1] != mass.shape:
        raise ValueError("message sum prefix must match destination mass")
    if minimum_mass <= 0:
        raise ValueError("minimum mass must be positive")
    live = mass.gt(0)
    denominator = mass.clamp_min(minimum_mass).unsqueeze(-1)
    normalized = message_sum / denominator
    return torch.where(
        live.unsqueeze(-1),
        normalized,
        torch.zeros_like(normalized),
    )


def transport_stewardship(
    ownership: torch.Tensor,
    states: torch.Tensor,
    node_locations: torch.Tensor,
    handoffs: torch.Tensor,
    destinations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transport mass, branch state, and node passport through cell handoffs."""

    if ownership.ndim != 3:
        raise ValueError("ownership must have [batch, cells, tokens] shape")
    if states.ndim != 4 or node_locations.ndim != 4 or handoffs.ndim != 4:
        raise ValueError("states, node_locations, and handoffs must be rank four")
    batch, cells, tokens = ownership.shape
    hidden = states.shape[-1]
    nodes = node_locations.shape[-1]
    actions = handoffs.shape[-1]
    if states.shape[:3] != (batch, cells, tokens):
        raise ValueError("state prefix must match ownership")
    if node_locations.shape[:3] != (batch, cells, tokens):
        raise ValueError("node-location prefix must match ownership")
    if handoffs.shape[:3] != (batch, cells, tokens):
        raise ValueError("handoff prefix must match ownership")
    if destinations.shape != (cells, actions):
        raise ValueError("destinations must have [cells, actions] shape")

    # Autocast may leave softmax probabilities in FP32 while the branch
    # state and node passport are BF16. Transport is a state operation, so
    # keep its mass/messages in the state dtype before scatter accumulation.
    flow = (
        ownership.to(states.dtype).unsqueeze(-1)
        * handoffs.to(states.dtype)
    )
    destination_index = destinations.reshape(1, 1, cells * actions).expand(batch, tokens, -1)
    flow_flat = flow.permute(0, 2, 1, 3).reshape(batch, tokens, cells * actions)
    mass = torch.zeros(
        batch, tokens, cells, dtype=flow.dtype, device=flow.device
    ).scatter_add(2, destination_index, flow_flat)

    state_messages = (
        flow.unsqueeze(-1) * states.unsqueeze(3)
    ).permute(0, 2, 1, 3, 4).reshape(batch, tokens, cells * actions, hidden)
    state_sum = torch.zeros(
        batch, tokens, cells, hidden, dtype=states.dtype, device=states.device
    ).scatter_add(
        2,
        destination_index.unsqueeze(-1).expand(-1, -1, -1, hidden),
        state_messages,
    )

    node_messages = (
        flow.unsqueeze(-1) * node_locations.to(states.dtype).unsqueeze(3)
    ).permute(0, 2, 1, 3, 4).reshape(batch, tokens, cells * actions, nodes)
    node_sum = torch.zeros(
        batch,
        tokens,
        cells,
        nodes,
        dtype=states.dtype,
        device=node_locations.device,
    ).scatter_add(
        2,
        destination_index.unsqueeze(-1).expand(-1, -1, -1, nodes),
        node_messages,
    )

    next_states = normalise_stewardship_messages(
        state_sum, mass
    ).permute(0, 2, 1, 3)
    next_nodes = normalise_stewardship_messages(
        node_sum, mass
    ).permute(0, 2, 1, 3)
    return mass.permute(0, 2, 1), next_states, next_nodes


def transport_packed_stewardship(
    packed: PackedStewardship,
    handoffs: torch.Tensor,
    destinations: torch.Tensor,
    *,
    batch_size: int,
    cells: int,
    tokens: int,
    support: torch.Tensor | None = None,
) -> PackedStewardship:
    """Expand only live rows over legal actions and merge colliding arrivals."""

    if handoffs.ndim != 2 or handoffs.shape[0] != packed.size:
        raise ValueError("handoffs must have [live branches, actions] shape")
    actions = handoffs.shape[1]
    if destinations.shape != (cells, actions):
        raise ValueError("destinations must have [cells, actions] shape")
    if packed.size == 0:
        raise ValueError("packed stewardship cannot be empty")
    torch._assert_async(
        (
            packed.batch.ge(0)
            & packed.batch.lt(batch_size)
            & packed.cell.ge(0)
            & packed.cell.lt(cells)
            & packed.token.ge(0)
            & packed.token.lt(tokens)
        ).all(),
        "packed branch index is outside the declared batch/cell/token shape",
    )

    if support is not None:
        if support.shape != handoffs.shape or support.dtype != torch.bool:
            raise ValueError("support must be a boolean handoff-shaped mask")
        source_rows, action_rows = support.nonzero(as_tuple=True)
        candidate_mass = (
            packed.mass[source_rows]
            * handoffs[source_rows, action_rows]
        )
        positive = candidate_mass.gt(0)
        source_rows = source_rows[positive]
        action_rows = action_rows[positive]
        candidate_mass = candidate_mass[positive]
        candidate_cell = destinations[
            packed.cell[source_rows], action_rows
        ]
        candidate_key = (
            (
                packed.batch[source_rows] * tokens
                + packed.token[source_rows]
            )
            * cells
            + candidate_cell
        )
    else:
        candidate_mass = packed.mass[:, None] * handoffs
        candidate_cell = destinations[packed.cell]
        candidate_batch = packed.batch[:, None].expand(-1, actions)
        candidate_token = packed.token[:, None].expand(-1, actions)
        candidate_key = (
            (candidate_batch * tokens + candidate_token) * cells
            + candidate_cell
        ).reshape(-1)
        candidate_mass = candidate_mass.reshape(-1)
        live = candidate_mass.gt(0)
        candidate_key = candidate_key[live]
        candidate_mass = candidate_mass[live]
        source_rows = (
            torch.arange(packed.size, device=handoffs.device)[:, None]
            .expand(-1, actions)
            .reshape(-1)[live]
        )
    if not candidate_mass.numel():
        raise ValueError("packed handoff support cannot remove every branch")

    unique_key, inverse = candidate_key.unique(sorted=True, return_inverse=True)
    merged = int(unique_key.numel())
    merged_mass = candidate_mass.new_zeros(merged).scatter_add(
        0, inverse, candidate_mass
    )
    state_sum = packed.state.new_zeros(merged, packed.state.shape[-1]).scatter_add(
        0,
        inverse[:, None].expand(-1, packed.state.shape[-1]),
        candidate_mass[:, None] * packed.state[source_rows],
    )
    node_sum = packed.node.new_zeros(merged, packed.node.shape[-1]).scatter_add(
        0,
        inverse[:, None].expand(-1, packed.node.shape[-1]),
        candidate_mass[:, None] * packed.node[source_rows],
    )
    batch_token = unique_key.div(cells, rounding_mode="floor")
    return PackedStewardship(
        batch=batch_token.div(tokens, rounding_mode="floor"),
        cell=unique_key.remainder(cells),
        token=batch_token.remainder(tokens),
        mass=merged_mass,
        state=normalise_stewardship_messages(state_sum, merged_mass),
        node=normalise_stewardship_messages(node_sum, merged_mass),
    )
