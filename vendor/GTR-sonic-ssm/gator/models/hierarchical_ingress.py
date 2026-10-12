"""Compile-friendly hierarchical ingress for packed Convergent routes.

Dense learned ingress scores every packet against every node, costing
``P * N * R`` multiply-accumulates for ``P`` packets, ``N`` nodes, and router
rank ``R``.  Hierarchical ingress first scores ``G`` fixed groups, retains a
fixed ``top_groups`` groups, and scores only their padded node candidates. For
balanced groups with padded width ``M = ceil(N / G)``, its score work is
approximately ``P * R * (G + top_groups * M)``.  Persistent route state is a
:class:`~gator.models.packed_frontier.PackedNodeFrontier`; no dense packet-by-
node passport is materialized.

The group table is a strict partition: every node occurs in exactly one group.
Consequently every node remains reachable whenever its group wins the learned
packet-conditioned group gate.  The hierarchical node logit is::

    <packet, node_key[n]> + node_bias[n]
      + <packet, group_key[group(n)]> + group_bias[group(n)]

Adding the selected group logit to each candidate node gives the group scorer
a useful gradient after its discrete top-group decision.  When all groups are
selected, the packed result is exactly the top-k result of evaluating that
same hierarchical logit densely over every node, including parameter and
packet-feature gradients.

Learned group keys and biases are optional.  If group keys are omitted, the
scorer derives each group key as the masked mean of its private node keys. If
group bias is also omitted, it is similarly derived from node biases. This
parameter-free form is useful as an integration baseline; private learned
group parameters add only ``G * (R + 1)`` scalars.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .packed_frontier import PackedNodeFrontier


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class HierarchicalIngressLayout:
    """A fixed, padded partition of graph nodes into ingress groups.

    ``nodes`` and ``valid`` have shape ``[groups, group_capacity]``. Invalid
    padding is canonicalized to node zero. ``node_to_group`` has shape
    ``[n_nodes]`` and is convenient for dense diagnostic oracles.  Construct
    layouts with :meth:`from_padded` or :meth:`balanced` so the partition is
    validated once, outside the compiled training step.
    """

    nodes: torch.Tensor
    valid: torch.Tensor
    node_to_group: torch.Tensor
    n_nodes: int

    def __post_init__(self) -> None:
        _positive_integer(self.n_nodes, "n_nodes")
        if self.nodes.ndim != 2 or self.nodes.shape[0] < 1 or self.nodes.shape[1] < 1:
            raise ValueError("nodes must be a nonempty [groups, capacity] tensor")
        if self.nodes.dtype != torch.long:
            raise ValueError("nodes must be an int64 tensor")
        if self.valid.shape != self.nodes.shape or self.valid.dtype != torch.bool:
            raise ValueError("valid must be boolean and match nodes")
        if self.node_to_group.shape != (self.n_nodes,):
            raise ValueError("node_to_group must have shape [n_nodes]")
        if self.node_to_group.dtype != torch.long:
            raise ValueError("node_to_group must be an int64 tensor")
        if not (
            self.nodes.device == self.valid.device == self.node_to_group.device
        ):
            raise ValueError("layout tensors must share a device")

        torch._assert_async(
            self.nodes.ge(0).logical_and(self.nodes.lt(self.n_nodes)).all(),
            "layout node identifier is outside the graph",
        )
        torch._assert_async(
            self.valid.any(dim=-1).all(),
            "every ingress group must contain at least one node",
        )
        counts = torch.zeros(
            self.n_nodes,
            dtype=torch.long,
            device=self.nodes.device,
        ).scatter_add(
            0,
            self.nodes.reshape(-1),
            self.valid.reshape(-1).to(torch.long),
        )
        torch._assert_async(
            counts.eq(1).all(),
            "ingress groups must contain every node exactly once",
        )
        group_ids = torch.arange(
            self.n_groups,
            dtype=torch.long,
            device=self.nodes.device,
        )[:, None].expand_as(self.nodes)
        derived = torch.zeros_like(self.node_to_group).scatter(
            0,
            self.nodes[self.valid],
            group_ids[self.valid],
        )
        torch._assert_async(
            self.node_to_group.eq(derived).all(),
            "node_to_group is inconsistent with the padded group table",
        )

    @property
    def n_groups(self) -> int:
        return self.nodes.shape[0]

    @property
    def group_capacity(self) -> int:
        return self.nodes.shape[1]

    @classmethod
    def from_padded(
        cls,
        padded_nodes: torch.Tensor,
        *,
        n_nodes: int,
    ) -> "HierarchicalIngressLayout":
        """Build a validated layout from ``-1``-padded group node IDs."""

        _positive_integer(n_nodes, "n_nodes")
        if padded_nodes.ndim != 2 or padded_nodes.shape[0] < 1 or padded_nodes.shape[1] < 1:
            raise ValueError(
                "padded_nodes must be a nonempty [groups, capacity] tensor"
            )
        if padded_nodes.dtype != torch.long:
            raise ValueError("padded_nodes must be an int64 tensor")
        torch._assert_async(
            padded_nodes.ge(-1).all(),
            "padded node identifiers must use -1 for padding",
        )
        valid = padded_nodes.ge(0)
        torch._assert_async(
            padded_nodes.lt(n_nodes).all(),
            "padded node identifier is outside the graph",
        )
        nodes = torch.where(valid, padded_nodes, torch.zeros_like(padded_nodes))
        group_ids = torch.arange(
            padded_nodes.shape[0],
            dtype=torch.long,
            device=padded_nodes.device,
        )[:, None].expand_as(padded_nodes)
        node_to_group = torch.zeros(
            n_nodes,
            dtype=torch.long,
            device=padded_nodes.device,
        ).scatter(0, nodes[valid], group_ids[valid])
        return cls(
            nodes=nodes,
            valid=valid,
            node_to_group=node_to_group,
            n_nodes=n_nodes,
        )

    @classmethod
    def balanced(
        cls,
        *,
        n_nodes: int,
        n_groups: int,
        device: torch.device | str | None = None,
    ) -> "HierarchicalIngressLayout":
        """Create a deterministic contiguous balanced node partition.

        Group sizes differ by at most one. Earlier groups receive the extra
        nodes and unused cells in the rectangular table are padded with -1.
        """

        n_nodes = _positive_integer(n_nodes, "n_nodes")
        n_groups = _positive_integer(n_groups, "n_groups")
        if n_groups > n_nodes:
            raise ValueError("n_groups cannot exceed n_nodes")
        capacity = math.ceil(n_nodes / n_groups)
        padded = torch.full(
            (n_groups, capacity),
            -1,
            dtype=torch.long,
            device=device,
        )
        base, remainder = divmod(n_nodes, n_groups)
        cursor = 0
        for group in range(n_groups):
            size = base + int(group < remainder)
            padded[group, :size] = torch.arange(
                cursor,
                cursor + size,
                dtype=torch.long,
                device=padded.device,
            )
            cursor += size
        return cls.from_padded(padded, n_nodes=n_nodes)


@dataclass(frozen=True)
class HierarchicalIngressCost:
    """Leading-order parameter and scoring cost estimate."""

    dense_score_macs: int
    hierarchical_score_macs: int
    candidate_slots: int
    node_parameters: int
    learned_group_parameters: int

    @property
    def score_ratio(self) -> float:
        """Hierarchical score MACs divided by dense score MACs."""

        return self.hierarchical_score_macs / self.dense_score_macs


def estimate_hierarchical_ingress_cost(
    *,
    packets: int,
    n_nodes: int,
    rank: int,
    n_groups: int,
    group_capacity: int,
    top_groups: int,
    learned_groups: bool = True,
) -> HierarchicalIngressCost:
    """Estimate score MACs, candidate storage, and key/bias parameters.

    Top-k, gather, bias, and softmax work is intentionally excluded; the
    estimate isolates the dominant key dot products. ``candidate_slots`` is
    per packet and includes structural padding.
    """

    packets = _positive_integer(packets, "packets")
    n_nodes = _positive_integer(n_nodes, "n_nodes")
    rank = _positive_integer(rank, "rank")
    n_groups = _positive_integer(n_groups, "n_groups")
    group_capacity = _positive_integer(group_capacity, "group_capacity")
    top_groups = _positive_integer(top_groups, "top_groups")
    if top_groups > n_groups:
        raise ValueError("top_groups cannot exceed n_groups")
    candidates = top_groups * group_capacity
    return HierarchicalIngressCost(
        dense_score_macs=packets * n_nodes * rank,
        hierarchical_score_macs=(
            packets * (n_groups + candidates) * rank
        ),
        candidate_slots=candidates,
        node_parameters=n_nodes * (rank + 1),
        learned_group_parameters=(
            n_groups * (rank + 1) if learned_groups else 0
        ),
    )


def _derived_group_parameters(
    node_keys: torch.Tensor,
    node_bias: torch.Tensor,
    layout: HierarchicalIngressLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = layout.valid.to(node_keys.dtype)
    count = valid.sum(dim=-1, keepdim=True).clamp_min(1)
    gathered_keys = node_keys[layout.nodes]
    gathered_bias = node_bias[layout.nodes]
    keys = (gathered_keys * valid[..., None]).sum(dim=1) / count
    bias = (gathered_bias * valid).sum(dim=1) / count.squeeze(-1)
    return keys, bias


def hierarchical_packed_ingress(
    packet_features: torch.Tensor,
    node_keys: torch.Tensor,
    node_bias: torch.Tensor,
    *,
    layout: HierarchicalIngressLayout,
    top_groups: int,
    top_k: int,
    group_keys: torch.Tensor | None = None,
    group_bias: torch.Tensor | None = None,
    temperature: float | torch.Tensor = 1.0,
    mass_threshold: float = 0.0,
) -> PackedNodeFrontier:
    """Route packets through a fixed group hierarchy into packed node state.

    The output always has capacity ``top_k``. If padded selected groups expose
    fewer than ``top_k`` real nodes, the remaining entries are invalid with
    zero mass. Group and node top-k choices are discrete, while gradients flow
    through the retained group and node logits exactly as in ordinary sparse
    routing.
    """

    top_groups = _positive_integer(top_groups, "top_groups")
    top_k = _positive_integer(top_k, "top_k")
    mass_threshold = _nonnegative_finite(mass_threshold, "mass_threshold")
    if packet_features.ndim < 1 or packet_features.shape[-1] < 1:
        raise ValueError("packet_features must end with a nonempty rank axis")
    if not packet_features.is_floating_point():
        raise ValueError("packet_features must be floating point")
    rank = packet_features.shape[-1]
    if node_keys.shape != (layout.n_nodes, rank):
        raise ValueError("node_keys must have shape [n_nodes, rank]")
    if node_bias.shape != (layout.n_nodes,):
        raise ValueError("node_bias must have shape [n_nodes]")
    if not node_keys.is_floating_point() or not node_bias.is_floating_point():
        raise ValueError("node keys and bias must be floating point")
    if (
        node_keys.dtype != packet_features.dtype
        or node_bias.dtype != packet_features.dtype
    ):
        raise ValueError("node parameters must match packet feature dtype")
    if not (
        packet_features.device
        == node_keys.device
        == node_bias.device
        == layout.nodes.device
    ):
        raise ValueError("features, parameters, and layout must share a device")
    if top_groups > layout.n_groups:
        raise ValueError("top_groups cannot exceed the layout group count")
    candidate_capacity = top_groups * layout.group_capacity
    if top_k > min(layout.n_nodes, candidate_capacity):
        raise ValueError("top_k exceeds graph or selected candidate capacity")

    uses_derived_group_keys = group_keys is None
    derived_keys: torch.Tensor | None = None
    derived_bias: torch.Tensor | None = None
    if group_keys is None or group_bias is None:
        derived_keys, derived_bias = _derived_group_parameters(
            node_keys, node_bias, layout
        )
    if group_keys is None:
        assert derived_keys is not None
        group_keys = derived_keys
    elif group_keys.shape != (layout.n_groups, rank):
        raise ValueError("group_keys must have shape [groups, rank]")
    if group_bias is None:
        if derived_bias is not None and uses_derived_group_keys:
            group_bias = derived_bias
        else:
            group_bias = node_bias.new_zeros(layout.n_groups)
    elif group_bias.shape != (layout.n_groups,):
        raise ValueError("group_bias must have shape [groups]")
    if not group_keys.is_floating_point() or not group_bias.is_floating_point():
        raise ValueError("group keys and bias must be floating point")
    if (
        group_keys.dtype != packet_features.dtype
        or group_bias.dtype != packet_features.dtype
    ):
        raise ValueError("group parameters must match packet feature dtype")
    if not (
        group_keys.device == packet_features.device
        and group_bias.device == packet_features.device
    ):
        raise ValueError("group parameters and packet features must share a device")

    if isinstance(temperature, torch.Tensor):
        if temperature.numel() != 1:
            raise ValueError("temperature tensor must contain one value")
        temperature_tensor = temperature.to(
            device=packet_features.device,
            dtype=packet_features.dtype,
        )
        torch._assert_async(
            torch.isfinite(temperature_tensor).logical_and(
                temperature_tensor.gt(0)
            ).all(),
            "temperature must be finite and positive",
        )
    else:
        temperature_value = float(temperature)
        if not math.isfinite(temperature_value) or temperature_value <= 0:
            raise ValueError("temperature must be finite and positive")
        temperature_tensor = packet_features.new_tensor(temperature_value)

    group_logits = (
        torch.einsum("...r,gr->...g", packet_features, group_keys)
        + group_bias
    )
    selected_group_logits, selected_groups = group_logits.topk(
        top_groups, dim=-1
    )
    candidate_nodes = layout.nodes[selected_groups]
    candidate_valid = layout.valid[selected_groups]
    candidate_keys = node_keys[candidate_nodes]
    candidate_bias = node_bias[candidate_nodes]
    candidate_node_logits = (
        torch.einsum("...r,...gmr->...gm", packet_features, candidate_keys)
        + candidate_bias
    )
    hierarchical_logits = (
        candidate_node_logits + selected_group_logits.unsqueeze(-1)
    ) / temperature_tensor

    flat_nodes = candidate_nodes.flatten(-2)
    flat_valid = candidate_valid.flatten(-2)
    flat_logits = hierarchical_logits.flatten(-2)
    masked_logits = flat_logits.masked_fill(~flat_valid, -torch.inf)
    retained_logits, retained_position = masked_logits.topk(top_k, dim=-1)
    retained_nodes = flat_nodes.gather(-1, retained_position)
    retained_valid = flat_valid.gather(-1, retained_position)

    normalizer = torch.logsumexp(masked_logits, dim=-1, keepdim=True)
    retained_mass = torch.where(
        retained_valid,
        (retained_logits - normalizer).exp(),
        torch.zeros_like(retained_logits),
    )
    retained = retained_valid & retained_mass.gt(0)
    if mass_threshold > 0.0:
        retained &= retained_mass.ge(mass_threshold)
    # Each selected group is nonempty, so the strongest candidate is real.
    strongest = torch.zeros_like(retained)
    strongest[..., 0] = True
    retained |= strongest & retained_valid
    retained_mass = torch.where(
        retained, retained_mass, torch.zeros_like(retained_mass)
    )
    retained_mass = retained_mass / retained_mass.sum(
        dim=-1, keepdim=True
    ).clamp_min(torch.finfo(retained_mass.dtype).tiny)
    retained_nodes = torch.where(
        retained, retained_nodes, torch.zeros_like(retained_nodes)
    )
    return PackedNodeFrontier(
        nodes=retained_nodes,
        mass=retained_mass,
        valid=retained,
        n_nodes=layout.n_nodes,
    )


class HierarchicalPackedIngress(nn.Module):
    """Registered-layout wrapper for use inside a trainable graph model.

    The functional API keeps topology explicit for tests. This wrapper owns
    the fixed partition as buffers so ordinary ``module.to(device)`` and
    checkpoint round-trips keep the layout aligned with learned group keys.
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        rank: int,
        n_groups: int,
        top_groups: int,
        top_k: int,
        learned_groups: bool = True,
    ) -> None:
        super().__init__()
        rank = _positive_integer(rank, "rank")
        top_groups = _positive_integer(top_groups, "top_groups")
        top_k = _positive_integer(top_k, "top_k")
        layout = HierarchicalIngressLayout.balanced(
            n_nodes=n_nodes,
            n_groups=n_groups,
        )
        if top_groups > layout.n_groups:
            raise ValueError("top_groups cannot exceed n_groups")
        if top_k > min(
            n_nodes,
            top_groups * layout.group_capacity,
        ):
            raise ValueError("top_k exceeds selected hierarchical candidates")
        self.n_nodes = n_nodes
        self.rank = rank
        self.n_groups = n_groups
        self.top_groups = top_groups
        self.top_k = top_k
        self.learned_groups = learned_groups
        self.register_buffer("group_nodes", layout.nodes)
        self.register_buffer("group_valid", layout.valid)
        self.register_buffer("node_to_group", layout.node_to_group)
        if learned_groups:
            self.group_keys = nn.Parameter(torch.empty(n_groups, rank))
            self.group_bias = nn.Parameter(torch.zeros(n_groups))
            # The containing model initializes optional group keys after all
            # legacy tensors so seeded dense-vs-hierarchical A/Bs preserve
            # the historical parameter stream. Standalone use begins from an
            # unbiased partition and can call a preferred initializer.
            nn.init.zeros_(self.group_keys)
        else:
            self.register_parameter("group_keys", None)
            self.register_parameter("group_bias", None)

    @property
    def nodes(self) -> torch.Tensor:
        return self.group_nodes

    @property
    def valid(self) -> torch.Tensor:
        return self.group_valid

    @property
    def group_capacity(self) -> int:
        return self.group_nodes.shape[1]

    @property
    def extra_parameter_count(self) -> int:
        return self.n_groups * (self.rank + 1) if self.learned_groups else 0

    def forward(
        self,
        packet_features: torch.Tensor,
        node_keys: torch.Tensor,
        node_bias: torch.Tensor,
        *,
        temperature: float | torch.Tensor = 1.0,
        mass_threshold: float = 0.0,
    ) -> PackedNodeFrontier:
        return hierarchical_packed_ingress(
            packet_features,
            node_keys,
            node_bias,
            layout=self,
            top_groups=self.top_groups,
            top_k=self.top_k,
            group_keys=self.group_keys,
            group_bias=self.group_bias,
            temperature=temperature,
            mass_threshold=mass_threshold,
        )


__all__ = [
    "HierarchicalIngressCost",
    "HierarchicalIngressLayout",
    "HierarchicalPackedIngress",
    "estimate_hierarchical_ingress_cost",
    "hierarchical_packed_ingress",
]
