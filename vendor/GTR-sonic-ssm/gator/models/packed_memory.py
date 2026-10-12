"""Exact packed-route receiver memory for Convergent Gator.

The current Convergent reference stores every packet passport as a dense
``[batch, tokens, lanes, nodes]`` tensor.  This module is a numerical oracle
for the first packed execution boundary: callers keep only a bounded list of
node IDs and masses per packet, while receiver memory retains the exact
``q.clamp_min(1e-9)`` semantics of the dense implementation.

Duplicate-route merging and the epsilon floor are both applied on a single
``[batch,tokens,lanes,nodes]`` tensor built with one ``scatter_add`` over the
route axis (this is exactly ``ConvergentGator._node_memory``'s dense ``q``,
just derived from a packed route list instead of held densely everywhere).
State is then produced with one ``einsum`` against ``hidden``, never a
``[batch,tokens,lanes,route_width,hidden]`` tensor. An earlier version of
this module processed one retained route at a time in a Python loop with
out-of-place accumulation; that was correct but forced autograd to retain
one large intermediate per loop iteration and replay the backward
sequentially, so it cost far more in backward than in forward despite never
being the memory bottleneck the loop was written to avoid. The vectorized
form below removes the loop (and its backward-side cost) while keeping
peak memory to one ``[batch,tokens,lanes,nodes]`` tensor, never the fully
expanded ``[batch,tokens,lanes,route_width,hidden]`` one.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedConvergentMemoryDiagnostics:
    """Dense oracle intermediates useful for equivalence tests and profiling.

    ``writer_features`` uses the matrix-multiply layout ``[B,N,T,D+1]``;
    feature zero is node mass and the remaining features are weighted state.
    ``node_memory`` uses the public semantic layout ``[B,T,N,D]``.
    """

    writer_features: torch.Tensor
    node_memory: torch.Tensor


def _validate_packed_inputs(
    hidden: torch.Tensor,
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    write: torch.Tensor,
    *,
    n_nodes: int,
    epsilon: float,
) -> tuple[int, int, int, int, int]:
    if hidden.ndim != 4:
        raise ValueError("hidden must have [batch,tokens,lanes,width] shape")
    if nodes.ndim != 4 or mass.shape != nodes.shape or valid.shape != nodes.shape:
        raise ValueError(
            "nodes, mass, and valid must share [batch,tokens,lanes,K] shape"
        )
    batch, tokens, lanes, width = hidden.shape
    if nodes.shape[:3] != (batch, tokens, lanes):
        raise ValueError("packed route prefix must match hidden")
    if write.shape != (batch, tokens):
        raise ValueError("write must have [batch,tokens] shape")
    if nodes.dtype != torch.long:
        raise ValueError("nodes must use int64")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    if not hidden.is_floating_point() or not mass.is_floating_point():
        raise ValueError("hidden and mass must be floating point")
    if not write.is_floating_point():
        raise ValueError("write must be floating point")
    if hidden.dtype != mass.dtype or hidden.dtype != write.dtype:
        raise ValueError("hidden, mass, and write must share a dtype")
    devices = {
        value.device for value in (hidden, nodes, mass, valid, write)
    }
    if len(devices) != 1:
        raise ValueError("every packed memory input must share a device")
    if isinstance(n_nodes, bool) or not isinstance(n_nodes, int) or n_nodes <= 0:
        raise ValueError("n_nodes must be a positive integer")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    torch._assert_async(
        ((~valid) | (nodes.ge(0) & nodes.lt(n_nodes))).all(),
        "a valid packed node ID is outside the graph",
    )
    torch._assert_async(
        torch.where(valid, mass, torch.zeros_like(mass)).ge(0).all(),
        "valid packed route mass must be nonnegative",
    )
    return batch, tokens, lanes, width, nodes.shape[-1]


def _packed_dense_locations(
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    n_nodes: int,
) -> torch.Tensor:
    """Merge a packed route list into one ``[B,T,L,N]`` per-lane mass tensor.

    A single ``scatter_add`` over the route axis both merges duplicate node
    IDs within a lane (matching indices simply sum) and produces exactly the
    dense-per-lane ``q`` tensor ``ConvergentGator._node_memory`` operates on
    directly. This never materializes a ``[B,T,L,K,D]``-sized tensor.
    """

    safe_nodes = torch.where(valid, nodes, torch.zeros_like(nodes))
    valid_mass = mass * valid.to(mass.dtype)
    return mass.new_zeros(*mass.shape[:-1], n_nodes).scatter_add(
        -1,
        safe_nodes,
        valid_mass,
    )


def packed_convergent_writer_features(
    hidden: torch.Tensor,
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    write: torch.Tensor,
    *,
    n_nodes: int,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Build exact dense node-writer features from a packed route frontier.

    The returned tensor is ``[B,N,T,D+1]``.  This is not a dense passport: the
    lane axis has already been reduced, and route IDs are scattered directly
    into node writer channels.

    Duplicate node IDs inside one lane are merged (via ``scatter_add``)
    before the epsilon floor is applied, matching a dense passport
    reconstructed with ``scatter_add`` and then passed through
    ``clamp_min(epsilon)`` — the same ``q.clamp_min(1e-9)`` contract
    ``ConvergentGator._node_memory`` uses on a fully dense passport.
    """

    _validate_packed_inputs(
        hidden,
        nodes,
        mass,
        valid,
        write,
        n_nodes=n_nodes,
        epsilon=epsilon,
    )
    dense_locations = _packed_dense_locations(nodes, mass, valid, n_nodes)
    writer_mass_per_lane = (
        dense_locations.clamp_min(epsilon) * write[:, :, None, None]
    )
    writer_mass = writer_mass_per_lane.sum(dim=2)
    writer_state = torch.einsum(
        "btln,btld->btnd",
        writer_mass_per_lane,
        hidden,
    )
    return torch.cat(
        (writer_mass.unsqueeze(-1), writer_state),
        dim=-1,
    ).permute(0, 2, 1, 3)


def packed_convergent_node_context(
    hidden: torch.Tensor,
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    write: torch.Tensor,
    receiver_kernel: torch.Tensor,
    *,
    epsilon: float = 1e-9,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[
    torch.Tensor,
    PackedConvergentMemoryDiagnostics,
]:
    """Return exact receiver context from packed node IDs and masses.

    ``receiver_kernel`` must use the Convergent layout ``[T,T,N]``.  It may
    already include a causal mask.  Context is contracted with one ``einsum``
    against the same ``[B,T,L,N]`` merged-route tensor used for the writer
    features, instead of gathering one retained route at a time.
    """

    if receiver_kernel.ndim != 3:
        raise ValueError("receiver_kernel must have [tokens,tokens,nodes] shape")
    tokens, writer_tokens, n_nodes = receiver_kernel.shape
    if writer_tokens != tokens:
        raise ValueError("receiver_kernel token axes must be square")
    _, hidden_tokens, _, _, _ = _validate_packed_inputs(
        hidden,
        nodes,
        mass,
        valid,
        write,
        n_nodes=n_nodes,
        epsilon=epsilon,
    )
    if hidden_tokens != tokens:
        raise ValueError("receiver_kernel token length must match hidden")
    if receiver_kernel.device != hidden.device:
        raise ValueError("receiver_kernel must share the packed input device")
    if receiver_kernel.dtype != hidden.dtype:
        raise ValueError("receiver_kernel must share the packed input dtype")

    dense_locations = _packed_dense_locations(nodes, mass, valid, n_nodes)
    writer_mass_per_lane = (
        dense_locations.clamp_min(epsilon) * write[:, :, None, None]
    )
    writer_mass = writer_mass_per_lane.sum(dim=2)
    writer_state = torch.einsum(
        "btln,btld->btnd",
        writer_mass_per_lane,
        hidden,
    )
    writer_features = torch.cat(
        (writer_mass.unsqueeze(-1), writer_state),
        dim=-1,
    ).permute(0, 2, 1, 3)
    mixed = torch.matmul(
        receiver_kernel.permute(2, 0, 1).unsqueeze(0),
        writer_features,
    )
    denominator = mixed[..., 0].permute(0, 2, 1)
    numerator = mixed[..., 1:].permute(0, 2, 1, 3)
    node_memory = numerator / denominator.clamp_min(epsilon).unsqueeze(-1)

    context = torch.einsum(
        "btln,btnd->btld",
        dense_locations,
        node_memory,
    )

    if not return_diagnostics:
        return context
    return context, PackedConvergentMemoryDiagnostics(
        writer_features=writer_features,
        node_memory=node_memory,
    )
