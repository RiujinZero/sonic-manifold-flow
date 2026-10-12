"""Bucketed kernels for persistent sparse token stewardship.

Packed Gator rows are ragged.  These helpers group rows with the same logical
owner and then place similarly sized groups into power-of-two capacity
buckets.  Work therefore scales with bucketed live capacity instead of padding
every group to the single busiest owner.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedGroupBucket:
    """A set of ragged groups sharing one padded capacity."""

    capacity: int
    group_keys: torch.Tensor
    row_indices: torch.Tensor
    valid: torch.Tensor
    counts: torch.Tensor

    @property
    def groups(self) -> int:
        return self.group_keys.shape[0]


@dataclass(frozen=True)
class PackedGroupPlan:
    """Reusable row-gather plan for one vector of group keys."""

    rows: int
    maximum_group_size: int
    buckets: tuple[PackedGroupBucket, ...]

    def statistics(self) -> dict:
        """Return host-side diagnostics for benchmarks and logs."""

        histogram = {
            bucket.capacity: int(bucket.groups)
            for bucket in self.buckets
            if bucket.groups
        }
        padded_rows = sum(
            bucket.groups * bucket.capacity for bucket in self.buckets
        )
        return {
            "rows": self.rows,
            "padded_rows": padded_rows,
            "padding_efficiency": self.rows / max(padded_rows, 1),
            "bucket_histogram": histogram,
        }


def _bucket_capacities(maximum_group_size: int) -> tuple[int, ...]:
    if (
        isinstance(maximum_group_size, bool)
        or not isinstance(maximum_group_size, int)
        or maximum_group_size <= 0
    ):
        raise ValueError("maximum_group_size must be a positive integer")
    capacities = []
    capacity = 1
    while capacity < maximum_group_size:
        capacities.append(capacity)
        capacity *= 2
    capacities.append(maximum_group_size)
    return tuple(dict.fromkeys(capacities))


def build_packed_group_plan(
    keys: torch.Tensor,
    *,
    maximum_group_size: int,
) -> PackedGroupPlan:
    """Group row indices by key and bucket groups by padded row capacity.

    ``maximum_group_size`` is a structural bound known by the caller:
    sequence length for ``(batch, cell)`` memory groups, cell count for
    ``(batch, token)`` route groups, and ``batch * tokens`` for cell-local
    projections. Supplying the bound avoids an explicit GPU-to-CPU maximum
    synchronization inside the relay.
    """

    if keys.ndim != 1 or keys.dtype != torch.long:
        raise ValueError("keys must be a one-dimensional int64 tensor")
    if not keys.numel():
        raise ValueError("cannot bucket an empty row set")
    torch._assert_async(
        keys.ge(0).all(),
        "packed group keys must be nonnegative",
    )
    capacities = _bucket_capacities(maximum_group_size)
    order = keys.argsort(stable=True)
    sorted_keys = keys[order]
    unique_keys, counts = sorted_keys.unique_consecutive(return_counts=True)
    torch._assert_async(
        counts.le(maximum_group_size).all(),
        "maximum_group_size is smaller than an observed group",
    )
    starts = counts.cumsum(0) - counts

    buckets = []
    lower = 0
    for capacity in capacities:
        selected = (
            counts.gt(lower) & counts.le(capacity)
        ).nonzero(as_tuple=False).flatten()
        if selected.numel() == 0:
            lower = capacity
            continue
        selected_counts = counts[selected]
        selected_starts = starts[selected]
        ranks = torch.arange(
            capacity, device=keys.device, dtype=torch.long
        )[None].expand(selected.shape[0], -1)
        valid = ranks < selected_counts[:, None]
        safe_rank = torch.minimum(
            ranks,
            (selected_counts - 1).clamp_min(0)[:, None],
        )
        sorted_positions = selected_starts[:, None] + safe_rank
        row_indices = order[sorted_positions]
        buckets.append(
            PackedGroupBucket(
                capacity=capacity,
                group_keys=unique_keys[selected],
                row_indices=row_indices,
                valid=valid,
                counts=selected_counts,
            )
        )
        lower = capacity
    return PackedGroupPlan(
        rows=keys.shape[0],
        maximum_group_size=maximum_group_size,
        buckets=tuple(buckets),
    )


def bucketed_group_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    plan: PackedGroupPlan,
) -> torch.Tensor:
    """Apply one affine transform per group key to packed input rows.

    ``inputs`` is ``[M,I]`` and ``weight`` is ``[G,O,I]``. Rows with the same
    integer key in ``plan`` use the corresponding weight slice. Inactive
    groups never have their parameter slice gathered or evaluated.
    """

    if inputs.ndim != 2 or weight.ndim != 3:
        raise ValueError("inputs and weight must have [rows,input] and [groups,output,input]")
    if inputs.shape[0] != plan.rows or inputs.shape[1] != weight.shape[2]:
        raise ValueError("inputs must match the plan row count and weight width")
    if bias is not None and bias.shape != weight.shape[:2]:
        raise ValueError("bias must have [groups,output] shape")

    output = inputs.new_zeros(plan.rows, weight.shape[1])
    for bucket in plan.buckets:
        torch._assert_async(
            bucket.group_keys.lt(weight.shape[0]).all(),
            "packed group key exceeds the available parameter groups",
        )
        gathered = inputs[bucket.row_indices]
        gathered = gathered * bucket.valid[..., None].to(inputs.dtype)
        projected = torch.bmm(
            gathered,
            weight[bucket.group_keys].transpose(1, 2),
        )
        if bias is not None:
            projected = projected + bias[bucket.group_keys, None]
        projected = projected * bucket.valid[..., None].to(projected.dtype)
        output = output.index_add(
            0,
            bucket.row_indices.reshape(-1),
            projected.reshape(-1, projected.shape[-1]),
        )
    return output


def bucketed_cell_node_linear(
    inputs: torch.Tensor,
    node_embedding: torch.Tensor,
    node_weight: torch.Tensor,
    node_bias: torch.Tensor,
    plan: PackedGroupPlan,
) -> torch.Tensor:
    """Apply active cells' node transforms without folding dormant cells.

    ``inputs`` has ``[M,D]`` shape. Node parameters have
    ``[C,N,D]``, ``[C,N,D,D]``, and ``[C,N,D]`` shapes. Rows are grouped by
    cell in ``plan``. Only parameter slices for cells represented in a bucket
    are gathered and evaluated.
    """

    if inputs.ndim != 2:
        raise ValueError("inputs must have [rows,hidden] shape")
    if node_embedding.ndim != 3 or node_weight.ndim != 4:
        raise ValueError(
            "node parameters must have [cells,nodes,hidden] and "
            "[cells,nodes,output,hidden] shapes"
        )
    if node_bias.shape != node_embedding.shape:
        raise ValueError("node_bias must match node_embedding")
    cells, nodes, hidden = node_embedding.shape
    if (
        node_weight.shape != (cells, nodes, hidden, hidden)
        or inputs.shape != (plan.rows, hidden)
    ):
        raise ValueError("node parameter and packed input widths must match")

    output = inputs.new_zeros(plan.rows, nodes * hidden)
    for bucket in plan.buckets:
        torch._assert_async(
            bucket.group_keys.lt(cells).all(),
            "packed cell key exceeds the available node parameters",
        )
        cell_weight = node_weight[bucket.group_keys]
        cell_embedding = node_embedding[bucket.group_keys]
        cell_offset = (
            torch.einsum(
                "gnoi,gni->gno",
                cell_weight,
                cell_embedding,
            )
            + node_bias[bucket.group_keys]
        ).reshape(bucket.groups, nodes * hidden)
        gathered = inputs[bucket.row_indices]
        gathered = gathered * bucket.valid[..., None].to(inputs.dtype)
        projected = torch.bmm(
            gathered,
            cell_weight.reshape(
                bucket.groups, nodes * hidden, hidden
            ).transpose(1, 2),
        )
        projected = projected + cell_offset[:, None]
        projected = projected * bucket.valid[..., None].to(
            projected.dtype
        )
        output = output.index_add(
            0,
            bucket.row_indices.reshape(-1),
            projected.reshape(-1, nodes * hidden),
        )
    return output.view(plan.rows, nodes, hidden)
