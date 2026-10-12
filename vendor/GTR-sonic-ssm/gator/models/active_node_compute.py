"""Active-node computation with a shared trunk and private adapters.

Convergent Gator packets normally carry a small, packed set of destination
nodes.  Giving every node a private ``D x D`` transform makes both parameters
and node execution scale harshly with graph size.  This module factors that
operation into:

* one shared ``D x D`` trunk, evaluated once per packet; and
* private node embeddings, biases, and rank-``R`` down/up adapters, evaluated
  only for the ``K`` packed nodes carried by that packet.

For ``P = B * T * L`` packets, ``N`` graph nodes, width ``D``, adapter rank
``R``, and ``K`` retained nodes per packet, the exact formulas are::

    parameters = D^2 + D + N * (2D + 2DR)
    active MACs = P * D^2 + P * K * 2DR

The corresponding bank of full private transforms uses ``N * (D^2 + D)``
parameters and, after sparse dispatch, ``P * K * D^2`` MACs.  The helpers at
the bottom of this file expose the formulas so experiment logs can report the
trade-off without reconstructing it elsewhere.

The packed forward path never creates a node axis of length ``N``.  A dense
reference path is provided for numerical and gradient oracles; it should not
be used for normal training.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def shared_trunk_adapter_parameter_count(
    *,
    n_nodes: int,
    width: int,
    adapter_rank: int,
) -> int:
    """Return the exact trainable parameter count used by this module."""

    n_nodes = _positive_integer(n_nodes, "n_nodes")
    width = _positive_integer(width, "width")
    adapter_rank = _positive_integer(adapter_rank, "adapter_rank")
    return width * width + width + n_nodes * (
        2 * width + 2 * width * adapter_rank
    )


def shared_trunk_adapter_macs(
    *,
    packet_count: int,
    active_nodes_per_packet: int,
    width: int,
    adapter_rank: int,
) -> int:
    """Return multiply-accumulates for one packed node-update call.

    Elementwise embedding additions, activation, and mass reduction are not
    included.  This is deliberately the same GEMM-only convention used for
    comparing dense and sparse node transforms.
    """

    packet_count = _positive_integer(packet_count, "packet_count")
    active_nodes_per_packet = _positive_integer(
        active_nodes_per_packet,
        "active_nodes_per_packet",
    )
    width = _positive_integer(width, "width")
    adapter_rank = _positive_integer(adapter_rank, "adapter_rank")
    return packet_count * width * width + packet_count * (
        active_nodes_per_packet * 2 * width * adapter_rank
    )


def full_private_parameter_count(*, n_nodes: int, width: int) -> int:
    """Parameter count of an ``N``-element bank of affine ``D x D`` nodes."""

    n_nodes = _positive_integer(n_nodes, "n_nodes")
    width = _positive_integer(width, "width")
    return n_nodes * (width * width + width)


def full_private_sparse_macs(
    *,
    packet_count: int,
    active_nodes_per_packet: int,
    width: int,
) -> int:
    """GEMM MACs for full private transforms after packed dispatch."""

    packet_count = _positive_integer(packet_count, "packet_count")
    active_nodes_per_packet = _positive_integer(
        active_nodes_per_packet,
        "active_nodes_per_packet",
    )
    width = _positive_integer(width, "width")
    return packet_count * active_nodes_per_packet * width * width


class SharedTrunkNodeAdapters(nn.Module):
    """Compute a weighted node update from a packed active-node frontier.

    Args:
        n_nodes: Total number of addressable graph nodes.
        width: Packet payload width ``D``.
        adapter_rank: Private node adapter rank ``R``.
        adapter_scale: Multiplier applied to every private adapter residual.
            ``None`` selects ``1 / sqrt(R)``.
        activation: Final pointwise activation.  ``torch.tanh`` matches the
            current Convergent Gator node transform.

    Inputs:
        base: Mutable packet state with shape ``[B, T, L, D]``.
        active_nodes: Packed node identifiers ``[B, T, L, K]``.
        active_mass: Corresponding routing mass ``[B, T, L, K]``.
        active_valid: Optional structural mask with the same packed shape.
            Invalid entries contribute exactly zero; their identifier must
            still be in range (packed frontiers conventionally use node 0).

    Returns:
        The mass-weighted node output with shape ``[B, T, L, D]``.

    ``trunk(base)`` is computed before any ``K`` expansion.  Consequently the
    expensive shared ``D x D`` projection runs once per packet, while private
    parameters are gathered only for packed node identifiers.
    """

    def __init__(
        self,
        *,
        n_nodes: int,
        width: int,
        adapter_rank: int,
        adapter_scale: float | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ) -> None:
        super().__init__()
        self.n_nodes = _positive_integer(n_nodes, "n_nodes")
        self.width = _positive_integer(width, "width")
        self.adapter_rank = _positive_integer(
            adapter_rank,
            "adapter_rank",
        )
        if adapter_scale is None:
            adapter_scale = 1.0 / math.sqrt(self.adapter_rank)
        adapter_scale = float(adapter_scale)
        if not math.isfinite(adapter_scale):
            raise ValueError("adapter_scale must be finite")
        if not callable(activation):
            raise TypeError("activation must be callable")
        self.adapter_scale = adapter_scale
        self.activation = activation

        # Matrices use input-major storage so the equations remain simple:
        # ``x @ weight``.  That also matches the existing node-weight bank.
        self.trunk_weight = nn.Parameter(torch.empty(self.width, self.width))
        self.trunk_bias = nn.Parameter(torch.empty(self.width))
        self.node_embedding = nn.Parameter(
            torch.empty(self.n_nodes, self.width)
        )
        self.adapter_down = nn.Parameter(
            torch.empty(
                self.n_nodes,
                self.width,
                self.adapter_rank,
            )
        )
        self.adapter_up = nn.Parameter(
            torch.empty(
                self.n_nodes,
                self.adapter_rank,
                self.width,
            )
        )
        self.node_bias = nn.Parameter(torch.empty(self.n_nodes, self.width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize a useful shared function with initially gentle adapters."""

        nn.init.kaiming_uniform_(self.trunk_weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(self.width)
        nn.init.uniform_(self.trunk_bias, -bound, bound)
        nn.init.normal_(self.node_embedding, std=self.width**-0.5)
        nn.init.kaiming_uniform_(self.adapter_down, a=math.sqrt(5))
        # A small nonzero up projection lets both adapter factors receive a
        # gradient on the first optimizer step while keeping the shared trunk
        # dominant at initialization.
        nn.init.normal_(self.adapter_up, std=1.0e-3)
        nn.init.zeros_(self.node_bias)

    @property
    def exact_parameter_count(self) -> int:
        """Formula-based count, useful in architecture summaries."""

        return shared_trunk_adapter_parameter_count(
            n_nodes=self.n_nodes,
            width=self.width,
            adapter_rank=self.adapter_rank,
        )

    def _validate_packed_inputs(
        self,
        base: torch.Tensor,
        active_nodes: torch.Tensor,
        active_mass: torch.Tensor,
        active_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        if base.ndim != 4 or base.shape[-1] != self.width:
            raise ValueError(
                f"base must have shape [B, T, L, {self.width}]"
            )
        expected = base.shape[:-1]
        if active_nodes.ndim != 4 or active_nodes.shape[:-1] != expected:
            raise ValueError(
                "active_nodes must have shape [B, T, L, K] matching base"
            )
        if active_nodes.shape[-1] < 1:
            raise ValueError("active_nodes needs a nonempty packed axis")
        if active_mass.shape != active_nodes.shape:
            raise ValueError("active_mass must match active_nodes")
        if active_nodes.dtype != torch.long:
            raise ValueError("active_nodes must be an int64 tensor")
        if not active_mass.is_floating_point():
            raise ValueError("active_mass must be floating point")
        if base.device != active_nodes.device or base.device != active_mass.device:
            raise ValueError("base and packed routing tensors must share a device")
        if active_valid is None:
            active_valid = torch.ones_like(active_nodes, dtype=torch.bool)
        else:
            if active_valid.shape != active_nodes.shape:
                raise ValueError("active_valid must match active_nodes")
            if active_valid.dtype != torch.bool:
                raise ValueError("active_valid must be a boolean tensor")
            if active_valid.device != base.device:
                raise ValueError("active_valid must be on the base device")
        torch._assert_async(
            active_nodes.ge(0).logical_and(active_nodes.lt(self.n_nodes)).all(),
            "active node identifier is outside the graph",
        )
        return active_valid

    @staticmethod
    def _is_compiling() -> bool:
        compiler = getattr(torch, "compiler", None)
        predicate = getattr(compiler, "is_compiling", None)
        return bool(predicate is not None and predicate())

    @staticmethod
    def _padded_multiple(size: int, dtype: torch.dtype) -> int:
        """Return a 16-byte-stride-compatible feature width."""

        element_size = torch.empty((), dtype=dtype).element_size()
        multiple = max(1, 16 // element_size)
        return ((size + multiple - 1) // multiple) * multiple

    @staticmethod
    def _grouped_execution_dtype(master_dtype: torch.dtype) -> torch.dtype:
        """Return the explicit CUDA grouped-GEMM execution dtype.

        Blackwell's grouped-MM boundary used by the frozen full-private 140M
        model accepts BF16 operands.  Gator therefore keeps FP32 master
        parameters, residual state, activation, and accumulation, but casts
        only the adapter down/up grouped GEMMs to BF16.  Native FP16 and BF16
        models stay in their configured dtype.  Other dtypes use the portable
        path and never reach the grouped kernel.
        """

        if master_dtype == torch.float32:
            return torch.bfloat16
        return master_dtype

    def _can_use_grouped_mm(self, base: torch.Tensor) -> bool:
        """Whether the target device has the grouped GEMM used for dispatch.

        ``torch._grouped_mm`` is still an internal CUDA primitive.  Keep the
        portable path authoritative on CPU and pre-Ampere accelerators rather
        than silently changing dtype or discovering an unsupported kernel in
        the middle of a long training run.
        """

        if not hasattr(torch, "_grouped_mm") or base.device.type != "cuda":
            return False
        if base.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            return False
        return torch.cuda.get_device_capability(base.device) >= (8, 0)

    def _grouped_cuda_update(
        self,
        shared: torch.Tensor,
        active_nodes: torch.Tensor,
        effective_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch active routes without packet-wise adapter-bank gathers.

        One zero-mass sentinel row is appended for every graph node.  This
        makes the grouped-MM offset table fixed-width and guarantees that no
        group is empty, avoiding data-dependent unique-node shapes under
        ``torch.compile``.  Sentinel outputs are discarded by their zero mass.
        """

        packet_shape = shared.shape[:-1]
        packet_count = math.prod(packet_shape)
        route_width = active_nodes.shape[-1]
        route_count = packet_count * route_width
        flat_shared = shared.reshape(packet_count, self.width)
        flat_nodes = active_nodes.reshape(route_count)
        flat_mass = effective_mass.reshape(route_count).to(shared.dtype)
        packet_ids = torch.arange(
            packet_count,
            dtype=torch.long,
            device=shared.device,
        ).repeat_interleave(route_width)

        route_state = (
            flat_shared.index_select(0, packet_ids)
            + self.node_embedding.index_select(0, flat_nodes)
        )
        sentinel_nodes = torch.arange(
            self.n_nodes,
            dtype=torch.long,
            device=shared.device,
        )
        dispatch_nodes = torch.cat((flat_nodes, sentinel_nodes))
        dispatch_state = torch.cat(
            (
                route_state,
                route_state.new_zeros(self.n_nodes, self.width),
            )
        )
        dispatch_mass = torch.cat(
            (flat_mass, flat_mass.new_zeros(self.n_nodes))
        )
        dispatch_packets = torch.cat(
            (
                packet_ids,
                packet_ids.new_zeros(self.n_nodes),
            )
        )

        order = dispatch_nodes.argsort(stable=True)
        sorted_nodes = dispatch_nodes.index_select(0, order)
        sorted_state = dispatch_state.index_select(0, order)
        counts = torch.bincount(
            sorted_nodes,
            minlength=self.n_nodes,
        )
        offsets = counts.cumsum(0).to(torch.int32)

        execution_dtype = self._grouped_execution_dtype(shared.dtype)
        padded_width = self._padded_multiple(self.width, execution_dtype)
        padded_rank = self._padded_multiple(
            self.adapter_rank,
            execution_dtype,
        )
        state_padding = padded_width - self.width
        rank_padding = padded_rank - self.adapter_rank
        padded_state = (
            sorted_state
            if state_padding == 0
            else F.pad(sorted_state, (0, state_padding))
        )
        padded_down = (
            self.adapter_down
            if state_padding == 0 and rank_padding == 0
            else F.pad(
                self.adapter_down,
                (0, rank_padding, 0, state_padding),
            )
        )
        hidden = torch._grouped_mm(
            padded_state.to(execution_dtype),
            padded_down.to(execution_dtype),
            offs=offsets,
        )
        padded_up = (
            self.adapter_up
            if state_padding == 0 and rank_padding == 0
            else F.pad(
                self.adapter_up,
                (0, state_padding, 0, rank_padding),
            )
        )
        adapted = torch._grouped_mm(
            hidden,
            padded_up.to(execution_dtype),
            offs=offsets,
        )[:, : self.width].to(shared.dtype)

        destination = self.activation(
            sorted_state
            + self.adapter_scale * adapted
            + self.node_bias.index_select(0, sorted_nodes)
        )
        weighted = destination * dispatch_mass.index_select(0, order)[:, None]
        sorted_packets = dispatch_packets.index_select(0, order)
        packet_update = shared.new_zeros(packet_count, self.width).scatter_add(
            0,
            sorted_packets[:, None].expand(-1, self.width),
            weighted,
        )
        return packet_update.reshape(*packet_shape, self.width)

    def _portable_active_update(
        self,
        shared: torch.Tensor,
        active_nodes: torch.Tensor,
        effective_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Eager correctness path that dispatches one private node at a time."""

        packet_shape = shared.shape[:-1]
        packet_count = math.prod(packet_shape)
        route_width = active_nodes.shape[-1]
        flat_shared = shared.reshape(packet_count, self.width)
        flat_nodes = active_nodes.reshape(-1)
        flat_mass = effective_mass.reshape(-1).to(shared.dtype)
        packet_ids = torch.arange(
            packet_count,
            dtype=torch.long,
            device=shared.device,
        ).repeat_interleave(route_width)
        packet_update = torch.zeros_like(flat_shared)
        for node in range(self.n_nodes):
            selected = flat_nodes.eq(node) & flat_mass.gt(0)
            if not bool(selected.any()):
                continue
            selected_packets = packet_ids[selected]
            node_state = (
                flat_shared.index_select(0, selected_packets)
                + self.node_embedding[node]
            )
            hidden = torch.matmul(node_state, self.adapter_down[node])
            adapted = torch.matmul(hidden, self.adapter_up[node])
            destination = self.activation(
                node_state
                + self.adapter_scale * adapted
                + self.node_bias[node]
            ) * flat_mass[selected, None]
            packet_update = packet_update.scatter_add(
                0,
                selected_packets[:, None].expand(-1, self.width),
                destination,
            )
        return packet_update.reshape(*packet_shape, self.width)

    def _portable_static_update(
        self,
        shared: torch.Tensor,
        active_nodes: torch.Tensor,
        effective_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Static-shape compile oracle for devices without grouped GEMM.

        This path deliberately trades arithmetic for graph portability.  It
        never gathers ``[packet, route, D, R]`` adapter banks, and it is used
        only while compiling on a device lacking the target grouped kernel.
        """

        packet_update = torch.zeros_like(shared)
        for node in range(self.n_nodes):
            node_mass = torch.where(
                active_nodes.eq(node),
                effective_mass,
                torch.zeros_like(effective_mass),
            ).sum(dim=-1).to(shared.dtype)
            node_state = shared + self.node_embedding[node]
            hidden = torch.matmul(node_state, self.adapter_down[node])
            adapted = torch.matmul(hidden, self.adapter_up[node])
            destination = self.activation(
                node_state
                + self.adapter_scale * adapted
                + self.node_bias[node]
            )
            packet_update = packet_update + destination * node_mass[..., None]
        return packet_update

    def forward(
        self,
        base: torch.Tensor,
        active_nodes: torch.Tensor,
        active_mass: torch.Tensor,
        active_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the packed active-node path without materializing ``N``."""

        active_valid = self._validate_packed_inputs(
            base,
            active_nodes,
            active_mass,
            active_valid,
        )
        effective_mass = torch.where(
            active_valid,
            active_mass,
            torch.zeros((), dtype=active_mass.dtype, device=active_mass.device),
        )

        # [B, T, L, D], intentionally computed before introducing K.
        shared = torch.matmul(base, self.trunk_weight) + self.trunk_bias
        if self._can_use_grouped_mm(base):
            return self._grouped_cuda_update(
                shared,
                active_nodes,
                effective_mass,
            )
        if self._is_compiling():
            return self._portable_static_update(
                shared,
                active_nodes,
                effective_mass,
            )
        return self._portable_active_update(
            shared,
            active_nodes,
            effective_mass,
        )

    def dense_reference(
        self,
        base: torch.Tensor,
        node_mass: torch.Tensor,
    ) -> torch.Tensor:
        """Dense ``N``-axis oracle for values and gradients.

        ``node_mass`` must have shape ``[B, T, L, N]``.  This routine computes
        every private node and exists only for validation and small ablations.
        """

        if base.ndim != 4 or base.shape[-1] != self.width:
            raise ValueError(
                f"base must have shape [B, T, L, {self.width}]"
            )
        if node_mass.shape != base.shape[:-1] + (self.n_nodes,):
            raise ValueError(
                "node_mass must have shape [B, T, L, n_nodes]"
            )
        if not node_mass.is_floating_point():
            raise ValueError("node_mass must be floating point")
        if node_mass.device != base.device:
            raise ValueError("base and node_mass must share a device")

        shared = torch.matmul(base, self.trunk_weight) + self.trunk_bias
        node_state = shared.unsqueeze(-2) + self.node_embedding
        hidden = torch.einsum(
            "...nd,ndr->...nr",
            node_state,
            self.adapter_down,
        )
        adapted = torch.einsum(
            "...nr,nrd->...nd",
            hidden,
            self.adapter_up,
        )
        destination = self.activation(
            node_state
            + self.adapter_scale * adapted
            + self.node_bias
        )
        return (destination * node_mass.to(base.dtype).unsqueeze(-1)).sum(
            dim=-2
        )
