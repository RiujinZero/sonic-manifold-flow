"""Vocabulary-agnostic Gator for simultaneous response formation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal, NamedTuple

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_checkpoint

from ..cuda_graph_broadcast_mix import cuda_graph_broadcast_mix
from ..cuda_graph_receiver_memory import cuda_graph_linear_scan_receiver_context
from ..cuda_graph_stewardship_traffic import cuda_graph_stewardship_traffic
from ..mx_precision import (
    NODE_COMPUTE_PRECISIONS,
    READOUT_PRECISIONS,
    NodeComputePrecision,
    ReadoutPrecision,
    mx_readout_linear,
    node_compute_linear,
)
from .dense_kernels import (
    DenseKernelBackend,
    DenseNodeBackend,
    cell_linear,
    cell_node_linear,
    fold_cell_node_parameters,
    padded_linear,
)
from .memory import (
    LinearScanReceiverCache,
    ReceiverMemoryBackend,
    detach_linear_scan_receiver_cache,
    relative_position_code,
    relative_receiver_context,
)
from .sparse_kernels import (
    bucketed_cell_node_linear,
    bucketed_group_linear,
    build_packed_group_plan,
)
from .stewardship import (
    PackedStewardship,
    SparseStewardshipBackend,
    adaptive_nucleus_stewardship,
    elastic_handoff_beam,
    pack_stewardship,
    packed_handoffs_to_dense,
    packed_ownership,
    packed_readout_hidden,
    packed_receiver_context,
    packed_route_uncertainty,
    packed_weighted_token_sum,
    normalise_stewardship_messages,
    prune_packed_stewardship,
    ring_handoff_destinations,
    transport_packed_stewardship,
    transport_stewardship,
    unpack_stewardship,
)


DiagnosticMode = Literal["none", "route_loss", "route_metrics", "full"]


_TORCH_COMPILER_DISABLE = getattr(
    getattr(torch, "compiler", None),
    "disable",
    None,
)


def _outside_torch_compile(function):
    """Keep module importable when only no-CUDA-graph execution is used."""

    if _TORCH_COMPILER_DISABLE is None:
        return function
    return _TORCH_COMPILER_DISABLE(function)


class GatorARWRelayBlockCache(NamedTuple):
    """Receiver and traffic carries needed by one committed relay prefix."""

    receiver: LinearScanReceiverCache
    traffic_numerator: torch.Tensor
    write_sum: torch.Tensor


class GatorARWBlockCache(NamedTuple):
    """Immutable constant-size block cache for original GTR-ARW."""

    next_position: torch.Tensor
    block_size: int
    relays: tuple[GatorARWRelayBlockCache, ...]


def detach_gator_arw_block_cache(cache: GatorARWBlockCache) -> GatorARWBlockCache:
    return GatorARWBlockCache(
        cache.next_position.detach(),
        cache.block_size,
        tuple(
            GatorARWRelayBlockCache(
                detach_linear_scan_receiver_cache(relay.receiver),
                relay.traffic_numerator.detach(),
                relay.write_sum.detach(),
            )
            for relay in cache.relays
        ),
    )


def _fixed_code(
    length: int,
    device: torch.device,
    *,
    offset: int | torch.Tensor = 0,
    reference_length: int | None = None,
) -> torch.Tensor:
    """Return stable absolute slot codes under cached context growth."""

    if length <= 0:
        raise ValueError("position-code length must be positive")
    if reference_length is not None and reference_length <= 1:
        raise ValueError("position reference length must be greater than one")
    if isinstance(offset, torch.Tensor):
        if offset.ndim != 0 or offset.dtype not in (torch.int32, torch.int64):
            raise TypeError("position offset must be a scalar integer tensor")
        start = offset.to(device=device, dtype=torch.float32)
        torch._assert_async(offset >= 0, "position offset cannot be negative")
        if reference_length is not None:
            torch._assert_async(
                offset + length <= reference_length,
                "position range exceeds the fixed reference length",
            )
    else:
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise TypeError("position offset must be a nonnegative integer")
        if reference_length is not None and offset + length > reference_length:
            raise ValueError("position range exceeds the fixed reference length")
        start = torch.tensor(offset, device=device, dtype=torch.float32)
    positions = torch.arange(length, device=device, dtype=torch.float32) + start
    denominator = (
        max(length - 1, 1)
        if reference_length is None
        else reference_length - 1
    )
    angle = positions / denominator * math.pi
    return torch.stack(
        [
            torch.sin(angle),
            torch.cos(angle),
            torch.sin(2 * angle),
            torch.cos(2 * angle),
        ],
        -1,
    )


@dataclass(frozen=True)
class GatorLanguageConfig:
    vocab_size: int
    d_model: int = 32
    embedding_dim: int | None = None
    lexical_embedding_dim: int | None = None
    n_cells: int = 10
    n_nodes: int = 4
    relays: int = 4
    neighbors: int = 4
    mask_token_id: int = 3
    blank_token_id: int | None = None
    forbidden_output_token_ids: tuple[int, ...] = ()
    lexical_mode: Literal["private_hash", "communal"] = "private_hash"
    cell_input_mode: Literal["shared_projection", "feature_gated"] = "shared_projection"
    cell_output_mode: Literal["shared_pool", "grouped_concat"] = "shared_pool"
    cell_transport_mode: Literal[
        "broadcast_mix", "stewardship"
    ] = "broadcast_mix"
    stewardship_keep_mass: float = 0.9
    stewardship_handoff_keep_mass: float = 0.9
    stewardship_min_active: int = 1
    stewardship_max_active: int | None = None
    stewardship_exploration: float = 0.01
    stewardship_scouts: int = 1
    stewardship_execution: Literal["dense", "packed_auto"] = "dense"
    stewardship_packed_threshold: float = 0.5
    packed_stewardship_backend: SparseStewardshipBackend = (
        "persistent_bucketed"
    )
    receiver_memory_backend: ReceiverMemoryBackend = "convolution"
    receiver_memory_cuda_graph: bool = False
    receiver_memory_hgrn_scan: bool = False
    broadcast_mix_cuda_graph: bool = False
    stewardship_traffic_cuda_graph: bool = False
    block_causal: bool = False
    block_size: int | None = None
    position_reference_length: int | None = None
    dense_router_backend: DenseKernelBackend = "einsum"
    dense_node_backend: DenseNodeBackend = "folded_bmm"
    node_compute_precision: NodeComputePrecision = "bf16"
    readout_precision: ReadoutPrecision = "bf16"
    dense_kernel_padding: int = 1
    activation_checkpointing: bool = False
    node_topology: Literal[
        "asymmetric",
        "symmetric",
        "hybrid_alternating",
        "hybrid_hemispheric",
    ] = "asymmetric"

    def __post_init__(self) -> None:
        if self.vocab_size <= 0 or self.d_model <= 0:
            raise ValueError("vocab_size and d_model must be positive")
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive or None")
        if self.lexical_embedding_dim is not None and self.lexical_embedding_dim <= 0:
            raise ValueError("lexical_embedding_dim must be positive or None")
        if self.n_cells < 2 or self.n_nodes < 2 or self.relays <= 0:
            raise ValueError("Gator requires at least two cells, two nodes, and one relay")
        if self.neighbors <= 0 or self.neighbors % 2 or self.neighbors >= self.n_cells:
            raise ValueError("neighbors must be positive, even, and less than n_cells")
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("mask_token_id must be inside the vocabulary")
        if self.blank_token_id is not None:
            if not 0 <= self.blank_token_id < self.vocab_size:
                raise ValueError("blank_token_id must be inside the vocabulary")
            if self.blank_token_id in self.forbidden_output_token_ids:
                raise ValueError("blank_token_id cannot be forbidden at readout")
        if len(set(self.forbidden_output_token_ids)) != len(
            self.forbidden_output_token_ids
        ):
            raise ValueError("forbidden_output_token_ids must be unique")
        if any(
            token_id < 0 or token_id >= self.vocab_size
            for token_id in self.forbidden_output_token_ids
        ):
            raise ValueError("forbidden output token IDs must be inside the vocabulary")
        if self.lexical_mode not in {"private_hash", "communal"}:
            raise ValueError("lexical_mode must be 'private_hash' or 'communal'")
        if self.cell_input_mode not in {"shared_projection", "feature_gated"}:
            raise ValueError(
                "cell_input_mode must be 'shared_projection' or 'feature_gated'"
            )
        if self.cell_output_mode not in {"shared_pool", "grouped_concat"}:
            raise ValueError(
                "cell_output_mode must be 'shared_pool' or 'grouped_concat'"
            )
        if self.cell_output_mode == "grouped_concat":
            if self.language_width % self.d_model:
                raise ValueError(
                    "grouped_concat requires embedding_dim divisible by d_model"
                )
            if self.n_cells < self.language_width // self.d_model:
                raise ValueError("grouped_concat requires at least one cell per output group")
        if self.cell_transport_mode not in {"broadcast_mix", "stewardship"}:
            raise ValueError(
                "cell_transport_mode must be 'broadcast_mix' or 'stewardship'"
            )
        if not 0.0 < self.stewardship_keep_mass <= 1.0:
            raise ValueError("stewardship_keep_mass must be in (0, 1]")
        if not 0.0 < self.stewardship_handoff_keep_mass <= 1.0:
            raise ValueError("stewardship_handoff_keep_mass must be in (0, 1]")
        if not 1 <= self.stewardship_min_active <= self.n_cells:
            raise ValueError("stewardship_min_active must be inside the cell population")
        if self.stewardship_max_active is not None and not (
            self.stewardship_min_active
            <= self.stewardship_max_active
            <= self.n_cells
        ):
            raise ValueError(
                "stewardship_max_active must be between min_active and n_cells"
            )
        if not 0.0 <= self.stewardship_exploration < 1.0:
            raise ValueError("stewardship_exploration must be in [0, 1)")
        if not 0 <= self.stewardship_scouts < self.n_cells:
            raise ValueError("stewardship_scouts must be between zero and n_cells - 1")
        # Execution presets may select packed stewardship globally. Broadcast
        # transport has no stewardship controller, so this option is inert
        # there and must not make the naked block-cache path invalid.
        if self.stewardship_execution not in {"dense", "packed_auto"}:
            raise ValueError("stewardship_execution must be 'dense' or 'packed_auto'")
        if not 0.0 < self.stewardship_packed_threshold <= 1.0:
            raise ValueError("stewardship_packed_threshold must be in (0, 1]")
        if self.packed_stewardship_backend not in {
            "repack",
            "persistent_bucketed",
        }:
            raise ValueError(
                "packed_stewardship_backend must be 'repack' or "
                "'persistent_bucketed'"
            )
        if self.receiver_memory_backend not in {
            "attention",
            "convolution",
            "linear_scan",
        }:
            raise ValueError(
                "receiver_memory_backend must be 'attention', 'convolution', "
                "or 'linear_scan'"
            )
        if type(self.block_causal) is not bool:
            raise TypeError("block_causal must be a boolean")
        if self.block_size is not None and (
            isinstance(self.block_size, bool)
            or not isinstance(self.block_size, int)
            or self.block_size <= 0
        ):
            raise TypeError("block_size must be a positive integer or None")
        if self.position_reference_length is not None and (
            isinstance(self.position_reference_length, bool)
            or not isinstance(self.position_reference_length, int)
            or self.position_reference_length <= 1
        ):
            raise TypeError(
                "position_reference_length must be an integer greater than one"
            )
        if self.block_causal:
            if self.receiver_memory_backend != "linear_scan":
                raise ValueError("block_causal GTR-ARW requires linear_scan memory")
            if self.block_size is None:
                raise ValueError("block_causal GTR-ARW requires block_size")
            if self.position_reference_length is None:
                raise ValueError(
                    "block_causal GTR-ARW requires position_reference_length"
                )
        elif self.block_size is not None:
            raise ValueError("block_size requires block_causal=True")
        if self.dense_router_backend not in {"einsum", "grouped_bmm"}:
            raise ValueError(
                "dense_router_backend must be 'einsum' or 'grouped_bmm'"
            )
        if self.dense_node_backend not in {"einsum", "folded_bmm"}:
            raise ValueError(
                "dense_node_backend must be 'einsum' or 'folded_bmm'"
            )
        if self.node_compute_precision not in NODE_COMPUTE_PRECISIONS:
            raise ValueError(
                f"node_compute_precision must be one of {NODE_COMPUTE_PRECISIONS}"
            )
        if (
            self.node_compute_precision != "bf16"
            and self.dense_node_backend != "folded_bmm"
        ):
            raise ValueError(
                "node_compute_precision other than 'bf16' requires "
                "dense_node_backend='folded_bmm' -- the mx_precision "
                "grouped-node-linear path is shaped for the folded "
                "per-cell weight, not the einsum backend's raw "
                "[cells,nodes,out,in] weight"
            )
        if self.readout_precision not in READOUT_PRECISIONS:
            raise ValueError(
                f"readout_precision must be one of {READOUT_PRECISIONS}"
            )
        if self.receiver_memory_cuda_graph and (
            self.receiver_memory_backend != "linear_scan" or self.block_size is None
        ):
            raise ValueError(
                "receiver_memory_cuda_graph requires "
                "receiver_memory_backend='linear_scan' and a fixed "
                "block_size -- cuda_graph_linear_scan_receiver_context "
                "only implements the block-causal-cached path"
            )
        if self.receiver_memory_hgrn_scan and (
            self.receiver_memory_backend != "linear_scan" or self.block_size is None
        ):
            raise ValueError(
                "receiver_memory_hgrn_scan requires "
                "receiver_memory_backend='linear_scan' and a fixed "
                "block_size -- it only replaces the block-causal-cached "
                "forward scan"
            )
        if self.receiver_memory_hgrn_scan and self.receiver_memory_cuda_graph:
            raise ValueError(
                "receiver_memory_hgrn_scan and receiver_memory_cuda_graph "
                "are not wired together yet -- cuda_graph_linear_scan_"
                "receiver_context always calls the eager decayed_scan "
                "internally, not chunk_hgrn_decayed_scan"
            )
        if self.broadcast_mix_cuda_graph and self.cell_transport_mode != "broadcast_mix":
            raise ValueError(
                "broadcast_mix_cuda_graph requires "
                "cell_transport_mode='broadcast_mix' -- "
                "cuda_graph_broadcast_mix only implements that transport's "
                "roll/stack/gate/project/norm coupling block"
            )
        if self.stewardship_traffic_cuda_graph and (
            self.cell_transport_mode != "broadcast_mix" or self.block_size is None
        ):
            raise ValueError(
                "stewardship_traffic_cuda_graph requires "
                "cell_transport_mode='broadcast_mix' and a fixed "
                "block_size (block-causal mode) -- "
                "cuda_graph_stewardship_traffic only implements the "
                "traffic_scale=1.0 (ownership is None) block-causal path"
            )
        if (
            isinstance(self.dense_kernel_padding, bool)
            or not isinstance(self.dense_kernel_padding, int)
            or self.dense_kernel_padding <= 0
        ):
            raise ValueError("dense_kernel_padding must be a positive integer")
        if (
            self.cell_transport_mode == "stewardship"
            and self.cell_output_mode == "grouped_concat"
            and self.stewardship_max_active is not None
            and self.stewardship_max_active < self.language_width // self.d_model
        ):
            raise ValueError(
                "stewardship_max_active must cover every grouped output channel"
            )
        valid_topologies = {
            "asymmetric",
            "symmetric",
            "hybrid_alternating",
            "hybrid_hemispheric",
        }
        if self.node_topology not in valid_topologies:
            raise ValueError(f"node_topology must be one of {sorted(valid_topologies)}")
        if self.node_topology != "symmetric" and self.n_nodes != 4:
            raise ValueError("asymmetric and hybrid topology require four nodes")
        if self.node_topology.startswith("hybrid_") and self.n_cells % 2:
            raise ValueError("hybrid topology requires an even number of cells")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def language_width(self) -> int:
        return self.d_model if self.embedding_dim is None else self.embedding_dim

    @property
    def lexical_width(self) -> int:
        return self.language_width if self.lexical_embedding_dim is None else self.lexical_embedding_dim


class GatorLanguageModel(nn.Module):
    """Arrow cells with broadcast or elastic token-stewardship transport."""

    def __init__(self, config: GatorLanguageConfig) -> None:
        super().__init__()
        self.config = config
        c, n, d = config.n_cells, config.n_nodes, config.d_model
        embedding_dim = config.language_width
        lexical_dim = config.lexical_width
        edges = n * n
        router_inputs = 2 * d + n

        self.token_embedding = nn.Embedding(config.vocab_size, lexical_dim)
        if lexical_dim == embedding_dim:
            self.lexical_projection = nn.Identity()
        else:
            # This one matrix is used in both directions. The vocabulary
            # table and lexical/language bridge are therefore tied at readout.
            self.lexical_projection = nn.Linear(lexical_dim, embedding_dim, bias=False)
        self.register_buffer(
            "_forbidden_output_token_ids",
            torch.tensor(config.forbidden_output_token_ids, dtype=torch.long),
            persistent=False,
        )
        if embedding_dim == d:
            self.input_projection = nn.Identity()
        else:
            self.input_projection = nn.Linear(embedding_dim, d)
        if config.cell_output_mode == "grouped_concat":
            self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        elif embedding_dim == d:
            self.output_projection = nn.Identity()
        else:
            self.output_projection = nn.Linear(d, embedding_dim)

        if config.cell_input_mode == "feature_gated":
            # A parameter-efficient bank of cell-specific views into the
            # communal language embedding. Gates initialize to one through
            # 1 + tanh(0), exactly reproducing the shared projection at step 0.
            self.input_feature_gates = nn.Parameter(torch.zeros(c, embedding_dim))
        else:
            self.register_parameter("input_feature_gates", None)

        if config.lexical_mode == "communal":
            self.cell_embedding = nn.Parameter(torch.empty(c, d))
            self.register_parameter("lexical_prototypes", None)
        else:
            self.register_parameter("cell_embedding", None)
            # Preserve the validated math cell's private 16 x 8
            # lexical table. Full-vocabulary semantics come from the shared
            # embedding; token_id % 16 selects a private cell prototype.
            self.lexical_prototypes = nn.Parameter(torch.empty(c, 16, d))
        self.address_to_state = nn.Parameter(torch.empty(c, d, 4))
        self.address_to_edges = nn.Parameter(torch.empty(c, edges, 4))
        self.address_to_readout = nn.Parameter(torch.empty(c, d, 4))
        self.step_projection = nn.Parameter(torch.empty(c, d, 4))
        self.node_embedding = nn.Parameter(torch.empty(c, n, d))
        self.node_weight = nn.Parameter(torch.empty(c, n, d, d))
        self.node_bias = nn.Parameter(torch.zeros(c, n, d))
        self.router_weight = nn.Parameter(torch.empty(c, edges, router_inputs))
        self.router_bias = nn.Parameter(torch.zeros(c, edges))
        self.relative_node_bias = nn.Parameter(torch.zeros(c, n, 4))
        self.norm_weight = nn.Parameter(torch.ones(c, d))
        self.norm_bias = nn.Parameter(torch.zeros(c, d))
        self.start_logits = nn.Parameter(torch.full((c, n), -4.0))
        with torch.no_grad():
            self.start_logits[:, 0] = 4.0

        if config.cell_transport_mode == "broadcast_mix":
            self.left_projection = nn.Linear(d, d)
            self.right_projection = nn.Linear(d, d)
            self.coupling_norm = nn.LayerNorm(d)
            self.neighbor_gates = nn.Parameter(torch.zeros(c, 2))
            self.self_gates = nn.Parameter(torch.ones(c))
            self.output_votes = nn.Parameter(torch.zeros(c))
            self.register_parameter("handoff_weight", None)
            self.register_parameter("handoff_bias", None)
            self.register_buffer("handoff_destinations", None)
        else:
            # Replace the 408-parameter static mixing/voting shell with a
            # 410-parameter conditional handoff shell at C=62,D=8,N=4,K=4.
            # Private cell computation feeds a shared routing law while each
            # source cell retains its own learned action preference.
            self.left_projection = None
            self.right_projection = None
            self.coupling_norm = None
            self.register_parameter("neighbor_gates", None)
            self.register_parameter("self_gates", None)
            self.register_parameter("output_votes", None)
            actions = config.neighbors + 1
            self.handoff_weight = nn.Parameter(torch.empty(actions, router_inputs))
            self.handoff_bias = nn.Parameter(torch.zeros(c, actions))
            self.register_buffer(
                "handoff_destinations",
                ring_handoff_destinations(c, config.neighbors),
            )
        self.lm_bias = nn.Parameter(torch.zeros(config.vocab_size))

        asymmetric = torch.tensor(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
            ],
            dtype=torch.bool,
        ) if n == 4 else None
        symmetric = torch.eye(n, dtype=torch.bool)
        for source in range(n):
            symmetric[source, (source - 1) % n] = True
            symmetric[source, (source + 1) % n] = True
        if config.node_topology == "asymmetric":
            adjacency = asymmetric
        elif config.node_topology == "symmetric":
            adjacency = symmetric
        elif config.node_topology == "hybrid_alternating":
            # Alternate topology around the cell ring so every local region
            # contains both directed/staged and symmetric/flexible cells.
            adjacency = torch.stack(
                [symmetric if cell % 2 == 0 else asymmetric for cell in range(c)]
            )
        else:
            # Two contiguous half-rings create within-topology neighborhoods
            # and two communication boundaries between the cell populations.
            adjacency = torch.stack(
                [symmetric if cell < c // 2 else asymmetric for cell in range(c)]
            )
        self.register_buffer("adjacency", adjacency)
        self._cudagraph_accumulation_enabled = False
        self.reset_parameters()

    def configure_cudagraph_accumulation(self, enabled: bool) -> None:
        """Select the tied-ingress storage break for CUDA-graph training."""

        if type(enabled) is not bool:
            raise TypeError("CUDA-graph accumulation mode must be a boolean")
        if enabled and _TORCH_COMPILER_DISABLE is None:
            raise RuntimeError(
                "compiled CUDA execution requires torch.compiler.disable; "
                "upgrade PyTorch or use compile mode "
                "'max-autotune-no-cudagraphs'"
            )
        self._cudagraph_accumulation_enabled = enabled

    @_outside_torch_compile
    def _lexical_input_outside_compiled(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full ingress path (embedding through input projection)
        eagerly, returning the cell-input activation the compiled model
        body consumes next."""

        lexical_content = self.token_embedding(token_ids)
        language_content = self.lexical_projection(lexical_content)
        if self.input_feature_gates is None:
            content = self.input_projection(language_content)[:, None]
        else:
            feature_gates = 1.0 + torch.tanh(self.input_feature_gates)
            cell_language_content = (
                language_content[:, None] * feature_gates[None, :, None]
            )
            content = self.input_projection(cell_language_content)
        return content.clone()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if isinstance(self.lexical_projection, nn.Linear):
            nn.init.orthogonal_(self.lexical_projection.weight)
        if self.cell_embedding is not None:
            nn.init.normal_(self.cell_embedding, std=0.02)
        if self.lexical_prototypes is not None:
            nn.init.normal_(self.lexical_prototypes, std=0.02)
        for parameter in (
            self.address_to_state,
            self.address_to_edges,
            self.address_to_readout,
            self.step_projection,
            self.node_embedding,
            self.node_weight,
            self.router_weight,
        ):
            nn.init.xavier_uniform_(parameter.reshape(-1, parameter.shape[-1]))
        if self.handoff_weight is not None:
            nn.init.xavier_uniform_(self.handoff_weight)
        if self.left_projection is not None:
            self.left_projection.reset_parameters()
            self.right_projection.reset_parameters()
        if self.config.cell_output_mode == "grouped_concat":
            with torch.no_grad():
                self.output_projection.weight.copy_(
                    torch.eye(
                        self.config.language_width,
                        device=self.output_projection.weight.device,
                        dtype=self.output_projection.weight.dtype,
                    )
                )
                self.output_projection.bias.zero_()

    def _cell_norm(self, states: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(states, (self.config.d_model,))
        return normalized * self.norm_weight[None, :, None] + self.norm_bias[None, :, None]

    def _selected_cell_norm(
        self, states: torch.Tensor, cells: torch.Tensor
    ) -> torch.Tensor:
        normalized = F.layer_norm(states, (self.config.d_model,))
        return normalized * self.norm_weight[cells] + self.norm_bias[cells]

    def _route_handoffs(
        self,
        handoffs: torch.Tensor,
        handoff_entropy: torch.Tensor,
        *,
        prune: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prune:
            return elastic_handoff_beam(
                handoffs,
                handoff_entropy,
                base_keep_mass=self.config.stewardship_handoff_keep_mass,
                scouts=min(self.config.stewardship_scouts, handoffs.shape[-1] - 1),
                exploration_mass=self.config.stewardship_exploration,
                straight_through=self.training,
            )
        exploration = self.config.stewardship_exploration
        if exploration:
            handoffs = (
                (1.0 - exploration) * handoffs
                + exploration / handoffs.shape[-1]
            )
        return handoffs, handoffs.gt(0)

    @staticmethod
    def _route_uncertainty(
        ownership: torch.Tensor,
        handoff_entropy: torch.Tensor,
        route_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return any-steward uncertainty and ownership-weighted mean.

        The maximum drives widening: if even a low-mass live scout is unsure,
        its candidate fanout is allowed to survive. The mean is the smoother
        familiarity signal used by auxiliary route sharpening.
        """

        active = ownership.gt(0)
        # Entropy is in [0, 1], so -1 is an exact finite exclusion sentinel.
        # Avoiding -inf keeps the compiled pruning backward finite when an
        # invalid/padded token has no active steward.
        maximum = handoff_entropy.masked_fill(~active, -1.0).amax(1)
        maximum = torch.where(route_valid, maximum, torch.zeros_like(maximum))
        mean = (ownership * handoff_entropy).sum(1)
        mean = torch.where(route_valid, mean, torch.zeros_like(mean))
        return maximum, mean

    def _dense_relay_transform(
        self,
        h: torch.Tensor,
        context: torch.Tensor,
        occupancy: torch.Tensor,
        q: torch.Tensor,
        dense_address_edges: torch.Tensor,
        address_state: torch.Tensor,
        step_state: torch.Tensor,
        temperature: torch.Tensor | float,
        folded_node_weight: torch.Tensor | None,
        folded_node_offset: torch.Tensor | None,
        relay: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One dense relay's router/node computation: the peak-memory segment.

        Pure function of its arguments (no side effects, no trace-list
        appends), so it is safe to wrap in activation checkpointing --
        recomputing this segment during backward instead of keeping its
        intermediate `[batch, cells, tokens, nodes, nodes]`-shaped tensors
        resident trades compute time for the memory that otherwise scales
        directly with cell count and context length.
        """

        n = self.config.n_nodes
        router_input = torch.cat([h, context, occupancy], -1)
        edge_logits = cell_linear(
            router_input,
            self.router_weight,
            self.router_bias,
            backend=self.config.dense_router_backend,
            pad_to_multiple=self.config.dense_kernel_padding,
        )
        batch, c, total_length = h.shape[0], h.shape[1], h.shape[2]
        edge_logits = (edge_logits + dense_address_edges[None]).view(
            batch, c, total_length, n, n
        )
        adjacency = (
            self.adjacency[None, None, None]
            if self.adjacency.ndim == 2
            else self.adjacency[None, :, None]
        )
        transition = F.softmax(
            edge_logits.masked_fill(~adjacency, -1e9) / temperature,
            -1,
        )
        q_next = torch.einsum("bcts,bctsd->bctd", q, transition)
        base = (
            self._cell_norm(h + context)
            + address_state[None]
            + step_state[None, :, None]
        )
        if self.config.dense_node_backend == "einsum":
            node_affine = cell_node_linear(
                base.unsqueeze(3) + self.node_embedding[None, :, None],
                self.node_weight,
                self.node_bias,
                backend="einsum",
            )
            destination = torch.tanh(node_affine)
            node_result = (q_next.unsqueeze(-1) * destination).sum(3)
        else:
            assert folded_node_weight is not None
            assert folded_node_offset is not None
            # A single wide folded GEMM is ideal while its [B,C,T,N,D]
            # result remains modest.  At long CPU prefills that transient is
            # enormous (about 790 MiB for the production 4K checkpoint), and
            # both the affine result and tanh destination coexist.  Execute
            # the same folded weights one node slice at a time once the wide
            # result crosses 64 MiB.  This keeps BLAS-sized GEMMs, preserves
            # the checkpoint/state layout, and reduces immediately into the
            # routed node result instead of materializing every destination.
            wide_bytes = (
                batch
                * c
                * total_length
                * n
                * self.config.d_model
                * base.element_size()
            )
            memory_efficient_cpu = (
                base.device.type == "cpu"
                and not torch.is_grad_enabled()
                and not torch.compiler.is_compiling()
                and self.config.node_compute_precision == "bf16"
                and wide_bytes >= 64 * 1024 * 1024
            )
            if memory_efficient_cpu:
                node_result = None
                width = self.config.d_model
                for node in range(n):
                    node_affine = cell_linear(
                        base,
                        folded_node_weight[:, node * width : (node + 1) * width],
                        backend="grouped_bmm",
                        pad_to_multiple=self.config.dense_kernel_padding,
                    )
                    node_affine = (
                        node_affine
                        + folded_node_offset[None, :, None, node]
                    )
                    contribution = (
                        q_next[..., node, None] * torch.tanh(node_affine)
                    )
                    node_result = (
                        contribution
                        if node_result is None
                        else node_result + contribution
                    )
                assert node_result is not None
            elif self.config.node_compute_precision == "bf16":
                node_affine = cell_linear(
                    base,
                    folded_node_weight,
                    backend="grouped_bmm",
                    pad_to_multiple=self.config.dense_kernel_padding,
                ).view(batch, c, total_length, n, self.config.d_model)
                node_affine = (
                    node_affine + folded_node_offset[None, :, None]
                )
                destination = torch.tanh(node_affine)
                node_result = (q_next.unsqueeze(-1) * destination).sum(3)
            else:
                # Interior node compute at MXFP8/NVFP4-approx precision.
                # Routing remains BF16 by explicit scope decision.
                node_affine = node_compute_linear(
                    base,
                    folded_node_weight,
                    None,
                    precision=self.config.node_compute_precision,
                    graph_slot=relay,
                ).view(batch, c, total_length, n, self.config.d_model)
                node_affine = (
                    node_affine + folded_node_offset[None, :, None]
                )
                destination = torch.tanh(node_affine)
                node_result = (q_next.unsqueeze(-1) * destination).sum(3)
        local = self._cell_norm(h + node_result)
        return local, q_next

    def _dense_stewardship_context(
        self,
        states: torch.Tensor,
        node_locations: torch.Tensor,
        ownership: torch.Tensor | None,
        write: torch.Tensor,
        relative: torch.Tensor | None = None,
        *,
        block_size: int | None = None,
        cache: GatorARWRelayBlockCache | None = None,
        return_cache: bool = False,
        graph_slot: int = 0,
    ):
        """Receiver memory plus block-causal cumulative cell traffic.

        Stewardship scales each cell's writes by its ownership and restores
        population-normalized traffic with a ``cells`` multiplier. Naked
        broadcast routing lets every cell write at full mass, so it uses the
        same cache carries without that multiplier.
        """

        cells = self.config.n_cells
        if ownership is None:
            writer_mass = write[:, None].expand(-1, cells, -1)
            traffic_scale = 1.0
        else:
            writer_mass = ownership * write[:, None]
            traffic_scale = float(cells)
        if (
            self.config.receiver_memory_cuda_graph
            and self.config.receiver_memory_backend == "linear_scan"
            and block_size is not None
        ):
            batch, _, _, hidden = states.shape
            nodes = node_locations.shape[-1]
            if cache is None:
                cache_forward_state = states.new_zeros(
                    batch * (hidden + 1), cells * nodes
                )
            else:
                cache_forward_state = cache.receiver.forward_state
            context, next_forward_state = cuda_graph_linear_scan_receiver_context(
                states,
                node_locations,
                writer_mass,
                self.relative_node_bias.to(states.dtype),
                block_size=block_size,
                cache_forward_state=cache_forward_state,
                graph_slot=graph_slot,
            )
            next_receiver = (
                LinearScanReceiverCache(next_forward_state) if return_cache else None
            )
        else:
            receiver_result = relative_receiver_context(
                states,
                node_locations,
                writer_mass,
                self.relative_node_bias.to(states.dtype),
                backend=self.config.receiver_memory_backend,
                block_size=block_size,
                cache=None if cache is None else cache.receiver,
                return_cache=return_cache,
                hgrn_scan=self.config.receiver_memory_hgrn_scan,
            )
            if return_cache:
                context, next_receiver = receiver_result
            else:
                context = receiver_result
                next_receiver = None

        if block_size is None:
            if cache is not None or return_cache:
                raise ValueError("traffic caches require block_size")
            global_mass = write.sum(1, keepdim=True).clamp_min(1e-6)
            traffic = (
                node_locations * writer_mass.unsqueeze(-1)
            ).sum(2) * traffic_scale / global_mass[:, :, None]
            return context, traffic[:, :, None].expand(
                -1, -1, states.shape[2], -1
            )

        batch, _, tokens, nodes = node_locations.shape
        if tokens % block_size:
            raise ValueError("traffic token length must be divisible by block_size")
        accumulator_dtype = (
            torch.float32
            if states.dtype in (torch.float16, torch.bfloat16)
            else states.dtype
        )
        if cache is None:
            previous_numerator = torch.zeros(
                batch,
                cells,
                nodes,
                device=states.device,
                dtype=accumulator_dtype,
            )
            previous_write = torch.zeros(
                batch, device=states.device, dtype=accumulator_dtype
            )
        else:
            if cache.traffic_numerator.shape != (batch, cells, nodes):
                raise ValueError("traffic cache numerator has the wrong shape")
            if cache.write_sum.shape != (batch,):
                raise ValueError("traffic cache write_sum has the wrong shape")
            previous_numerator = cache.traffic_numerator
            previous_write = cache.write_sum

        if self.config.stewardship_traffic_cuda_graph and ownership is None:
            traffic, next_numerator, next_write = cuda_graph_stewardship_traffic(
                node_locations,
                writer_mass,
                write,
                previous_numerator,
                previous_write,
                block_size=block_size,
                states_dtype=states.dtype,
                graph_slot=graph_slot,
            )
        else:
            blocks = tokens // block_size
            token_numerator = node_locations * writer_mass.unsqueeze(-1)
            block_numerator = token_numerator.reshape(
                batch, cells, blocks, block_size, nodes
            ).sum(dim=3, dtype=accumulator_dtype).permute(0, 2, 1, 3)
            block_write = write.reshape(batch, blocks, block_size).sum(
                dim=2, dtype=accumulator_dtype
            )
            cumulative_numerator = (
                block_numerator.cumsum(dim=1) + previous_numerator.unsqueeze(1)
            )
            cumulative_write = (
                block_write.cumsum(dim=1) + previous_write.unsqueeze(1)
            ).clamp_min(1e-6)
            traffic_blocks = (
                cumulative_numerator
                * traffic_scale
                / cumulative_write[:, :, None, None]
            ).to(states.dtype)
            traffic = traffic_blocks[:, :, None].expand(
                -1, -1, block_size, -1, -1
            ).reshape(batch, tokens, cells, nodes).permute(0, 2, 1, 3)
            next_numerator = previous_numerator + block_numerator.sum(dim=1)
            next_write = previous_write + block_write.sum(dim=1)

        if return_cache:
            assert next_receiver is not None
            next_cache = GatorARWRelayBlockCache(
                next_receiver, next_numerator, next_write
            )
            return context, traffic, next_cache
        return context, traffic

    def _packed_stewardship_local(
        self,
        packed: PackedStewardship,
        *,
        write: torch.Tensor,
        relative: torch.Tensor,
        addresses: torch.Tensor,
        address_state: torch.Tensor,
        step_state: torch.Tensor,
        temperature: float,
        stewardship_temperature: float,
    ) -> tuple[PackedStewardship, torch.Tensor, torch.Tensor]:
        """Evaluate Arrow computation only for live packed routes."""

        n = self.config.n_nodes
        persistent_bucketed = (
            self.config.packed_stewardship_backend
            == "persistent_bucketed"
        )
        context, traffic = packed_receiver_context(
            packed,
            write=write,
            relative_code=relative,
            relative_node_bias=self.relative_node_bias.to(packed.state.dtype),
            cells=self.config.n_cells,
            backend="bucketed" if persistent_bucketed else "padded",
        )
        router_input = torch.cat([packed.state, context, traffic], -1)
        if persistent_bucketed:
            cell_plan = build_packed_group_plan(
                packed.cell,
                maximum_group_size=write.shape[0] * write.shape[1],
            )
            edge_logits = bucketed_group_linear(
                router_input,
                self.router_weight,
                self.router_bias,
                cell_plan,
            )
            address_edges = bucketed_group_linear(
                addresses[packed.token],
                self.address_to_edges,
                None,
                cell_plan,
            ).view(packed.size, n, n)
        else:
            edge_logits = torch.einsum(
                "mi,mei->me", router_input, self.router_weight[packed.cell]
            ) + self.router_bias[packed.cell]
            address_edges = torch.einsum(
                "ma,meoa->meo",
                addresses[packed.token],
                self.address_to_edges[packed.cell].view(
                    packed.size, n, n, addresses.shape[-1]
                ),
            )
        edge_logits = edge_logits.view(packed.size, n, n) + address_edges
        adjacency = (
            self.adjacency[None].expand(packed.size, -1, -1)
            if self.adjacency.ndim == 2
            else self.adjacency[packed.cell]
        )
        transition = F.softmax(
            edge_logits.masked_fill(~adjacency, -1e9) / temperature, -1
        )
        node_next = torch.einsum("ms,msd->md", packed.node, transition)

        base = (
            self._selected_cell_norm(packed.state + context, packed.cell)
            + address_state[packed.cell, packed.token]
            + step_state[packed.cell]
        )
        if persistent_bucketed:
            node_affine = bucketed_cell_node_linear(
                base,
                self.node_embedding,
                self.node_weight,
                self.node_bias,
                cell_plan,
            )
        else:
            node_input = base[:, None] + self.node_embedding[packed.cell]
            node_affine = (
                torch.einsum(
                    "mni,mnoi->mno",
                    node_input,
                    self.node_weight[packed.cell],
                )
                + self.node_bias[packed.cell]
            )
        destination = torch.tanh(node_affine)
        local = self._selected_cell_norm(
            packed.state + (node_next.unsqueeze(-1) * destination).sum(1),
            packed.cell,
        )
        handoff_input = torch.cat([local, context, traffic], -1)
        handoff_logits = padded_linear(
            handoff_input,
            self.handoff_weight,
            pad_to_multiple=self.config.dense_kernel_padding,
        ) + self.handoff_bias[packed.cell]
        handoffs = F.softmax(handoff_logits / stewardship_temperature, -1)
        handoff_entropy = -(
            handoffs.clamp_min(1e-9) * handoffs.clamp_min(1e-9).log()
        ).sum(-1) / math.log(handoffs.shape[-1])
        return PackedStewardship(
            batch=packed.batch,
            cell=packed.cell,
            token=packed.token,
            mass=packed.mass,
            state=local,
            node=node_next,
        ), handoffs, handoff_entropy

    def forward(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor,
        *,
        output_slots: int,
        temperature: float = 0.35,
        stewardship_temperature: float | None = None,
        stewardship_prune: bool = True,
        stewardship_keep_mass: float | None = None,
        stewardship_max_active: int | None = None,
        stewardship_packed_from_relay: int | None = None,
        diagnostics: DiagnosticMode | None = None,
        return_diagnostics: bool = False,
        cache: GatorARWBlockCache | None = None,
        return_cache: bool = False,
    ):
        if context_ids.ndim != 2 or context_mask.shape != context_ids.shape:
            raise ValueError("context_ids and context_mask must have matching [batch, length] shapes")
        if output_slots < 0 or (output_slots == 0 and not return_cache):
            raise ValueError("output_slots must be positive unless committing a cache")
        if stewardship_temperature is not None and stewardship_temperature <= 0:
            raise ValueError("stewardship_temperature must be positive")
        if self.config.block_causal:
            assert self.config.block_size is not None
            if cache is not None:
                if cache.block_size != self.config.block_size:
                    raise ValueError("cache block_size differs from model")
                if len(cache.relays) != self.config.relays:
                    raise ValueError("cache relay count differs from model")
                if cache.next_position.ndim != 0 or cache.next_position.dtype not in (
                    torch.int32,
                    torch.int64,
                ):
                    raise TypeError("cache next_position must be a scalar integer tensor")
            cache_length = (
                torch.zeros((), device=context_ids.device, dtype=torch.long)
                if cache is None
                else cache.next_position
            )
            if cache_length.device != context_ids.device:
                raise ValueError("cache and inputs must share a device")
            total_requested = context_ids.shape[1] + output_slots
            if total_requested % self.config.block_size:
                raise ValueError(
                    "block-causal forward length must be divisible by block_size"
                )
            if stewardship_packed_from_relay is not None:
                raise ValueError("packed stewardship is not cache-aware yet")
        else:
            if cache is not None or return_cache:
                raise ValueError("block caches require block_causal=True")
            cache_length = 0
        if stewardship_packed_from_relay is not None:
            if self.config.cell_transport_mode != "stewardship":
                raise ValueError(
                    "packed relay execution requires stewardship transport"
                )
            if not 0 <= stewardship_packed_from_relay < self.config.relays:
                raise ValueError(
                    "stewardship_packed_from_relay must identify an existing relay"
                )
        if diagnostics is None:
            diagnostic_mode: DiagnosticMode = (
                "full" if return_diagnostics else "none"
            )
        else:
            if diagnostics not in {
                "none",
                "route_loss",
                "route_metrics",
                "full",
            }:
                raise ValueError(f"unknown diagnostic mode: {diagnostics}")
            if return_diagnostics:
                raise ValueError(
                    "use diagnostics='full' instead of combining diagnostics "
                    "with return_diagnostics=True"
                )
            diagnostic_mode = diagnostics
        collect_route_loss = diagnostic_mode in {
            "route_loss",
            "route_metrics",
            "full",
        }
        collect_route_metrics = diagnostic_mode in {"route_metrics", "full"}
        collect_full = diagnostic_mode == "full"
        batch, context_length = context_ids.shape
        c, n, d = self.config.n_cells, self.config.n_nodes, self.config.d_model
        total_length = context_length + output_slots
        device = context_ids.device

        query_token_id = (
            self.config.blank_token_id
            if self.config.blank_token_id is not None
            else self.config.mask_token_id
        )
        target_ids = torch.full(
            (batch, output_slots), query_token_id, dtype=torch.long, device=device
        )
        all_token_ids = torch.cat([context_ids, target_ids], 1)
        # token_embedding and lexical_projection are reused, transposed, at
        # readout. input_projection isn't tied/reused elsewhere, but its
        # output still crosses the accumulation loop as a saved-for-backward
        # activation, so it's exposed to the same CUDA-graph overwrite
        # hazard as the tied-weight ingress path -- confirmed by
        # tests/test_gator_arw_cudagraph.py catching the identical error one
        # step downstream after only token_embedding/lexical_projection were
        # moved outside compile. The compiler-disabled clone is enabled only
        # for CUDA-graph modes; eager/default/no-cudagraph execution keeps
        # the original uninterrupted path. The expensive model body and
        # vocabulary readout stay compiled in either case.
        if (
            self._cudagraph_accumulation_enabled
            and self.training
            and torch.is_grad_enabled()
            and context_ids.is_cuda
        ):
            content = self._lexical_input_outside_compiled(all_token_ids)
        else:
            lexical_content = self.token_embedding(all_token_ids)
            language_content = self.lexical_projection(lexical_content)
            if self.input_feature_gates is None:
                content = self.input_projection(language_content)[:, None]
            else:
                feature_gates = 1.0 + torch.tanh(self.input_feature_gates)
                cell_language_content = (
                    language_content[:, None] * feature_gates[None, :, None]
                )
                content = self.input_projection(cell_language_content)
        addresses = _fixed_code(
            total_length,
            device,
            offset=cache_length,
            reference_length=self.config.position_reference_length,
        )
        address_state = padded_linear(
            addresses,
            self.address_to_state.reshape(c * d, addresses.shape[-1]),
            pad_to_multiple=self.config.dense_kernel_padding,
        ).view(total_length, c, d).permute(1, 0, 2)
        if self.lexical_prototypes is None:
            h = content + self.cell_embedding[None, :, None]
        else:
            lexical = self.lexical_prototypes[:, all_token_ids.remainder(16)]
            lexical = lexical.permute(1, 0, 2, 3)
            h = content + lexical
        h = self._cell_norm(h + address_state[None])

        # Autocast keeps the learned routing logits/softmax in FP32 while the
        # projected token state is BF16. Receiver memory combines ``q`` with
        # the BF16 state, writer mass, and relative bias, so normalize the
        # routing dtype at this boundary. The cast remains differentiable and
        # avoids a mixed-dtype convolution under BF16 + torch.compile.
        q = F.softmax(self.start_logits / temperature, -1).to(h.dtype)
        q = q[None, :, None]
        q = q.expand(batch, -1, total_length, -1)
        ownership = None
        ownership_traces = [] if collect_route_metrics else None
        route_entropy_traces = [] if collect_route_loss else None
        route_cell_load_traces = [] if collect_route_loss else None
        active_fraction_traces = [] if collect_route_loss else None
        mean_uncertainty_traces = [] if collect_route_loss else None
        handoff_traces = [] if collect_route_metrics else None
        uncertainty_traces = [] if collect_route_metrics else None
        handoff_support_traces = [] if collect_route_metrics else None
        execution_traces = [] if collect_route_metrics else None
        route_valid = torch.cat(
            [
                context_mask,
                torch.ones(
                    batch, output_slots, dtype=torch.bool, device=device
                ),
            ],
            1,
        )
        if self.config.cell_transport_mode == "stewardship":
            ownership = torch.full(
                (batch, c, total_length),
                1.0 / c,
                dtype=h.dtype,
                device=device,
            ) * route_valid[:, None].to(h.dtype)
            if ownership_traces is not None:
                ownership_traces.append(ownership)
        persistent_packed = None
        next_relay_caches: list[GatorARWRelayBlockCache] = []
        relative = (
            relative_position_code(
                total_length, device, dtype=h.dtype
            )
            if stewardship_packed_from_relay is not None
            else None
        )
        traces = [q] if collect_full else None
        relay_codes = _fixed_code(self.config.relays, device)
        step_states = padded_linear(
            relay_codes,
            self.step_projection.reshape(c * d, relay_codes.shape[-1]),
            pad_to_multiple=self.config.dense_kernel_padding,
        ).view(self.config.relays, c, d)
        dense_address_edges = (
            None
            if stewardship_packed_from_relay == 0
            else padded_linear(
                addresses,
                self.address_to_edges.reshape(
                    c * n * n, addresses.shape[-1]
                ),
                pad_to_multiple=self.config.dense_kernel_padding,
            ).view(total_length, c, n * n).permute(1, 0, 2)
        )
        output_groups = (
            self.config.language_width // d
            if self.config.cell_output_mode == "grouped_concat"
            else 1
        )
        cell_indices = torch.arange(c, device=device)
        cell_groups = (
            cell_indices.remainder(output_groups)
            if output_groups > 1
            else None
        )
        if self.config.dense_node_backend == "folded_bmm":
            folded_node_weight, folded_node_offset = (
                fold_cell_node_parameters(
                    self.node_embedding,
                    self.node_weight,
                    self.node_bias,
                )
            )
        else:
            folded_node_weight = folded_node_offset = None

        for relay in range(self.config.relays):
            # Routing transitions may use an autocast-eligible einsum and
            # change dtype between relays. Keep routing/node locations aligned
            # with the current hidden state before any memory or packed-route
            # operation consumes them.
            q = q.to(h.dtype)
            if active_fraction_traces is not None:
                possible_routes = (
                    route_valid.sum().to(torch.float32) * c
                ).clamp_min(1.0)
                if persistent_packed is None:
                    assert ownership is not None
                    live_routes = ownership.gt(0).sum().to(torch.float32)
                else:
                    live_routes = persistent_packed.mass.gt(0).sum().to(
                        torch.float32
                    )
                active_fraction_traces.append(
                    (live_routes / possible_routes).clamp(0.0, 1.0)
                )
            target_write = relay / max(self.config.relays - 1, 1)
            write = torch.cat(
                [
                    context_mask.to(h.dtype),
                    torch.full((batch, output_slots), target_write, dtype=h.dtype, device=device),
                ],
                1,
            )
            step_state = step_states[relay]
            cell_temperature = (
                temperature
                if stewardship_temperature is None
                else stewardship_temperature
            )
            use_packed = (
                stewardship_packed_from_relay is not None
                and relay >= stewardship_packed_from_relay
            )
            next_packed = None

            if use_packed:
                assert relative is not None
                persistent_backend = (
                    self.config.packed_stewardship_backend
                    == "persistent_bucketed"
                )
                packed = (
                    persistent_packed
                    if persistent_backend and persistent_packed is not None
                    else pack_stewardship(ownership, h, q)
                )
                processed, handoff_rows, entropy_rows = self._packed_stewardship_local(
                    packed,
                    write=write,
                    relative=relative,
                    addresses=addresses,
                    address_state=address_state,
                    step_state=step_state,
                    temperature=temperature,
                    stewardship_temperature=cell_temperature,
                )
                handoff_rows, handoff_row_mask = self._route_handoffs(
                    handoff_rows, entropy_rows, prune=stewardship_prune
                )
                if persistent_backend:
                    uncertainty, mean_uncertainty = (
                        packed_route_uncertainty(
                            processed,
                            entropy_rows,
                            batch_size=batch,
                            tokens=total_length,
                            route_valid=route_valid,
                        )
                    )
                    if collect_route_metrics:
                        handoff_support = packed_weighted_token_sum(
                            processed,
                            handoff_row_mask.sum(-1).to(
                                processed.mass.dtype
                            ),
                            batch_size=batch,
                            tokens=total_length,
                        )
                else:
                    entropy_grid = h.new_zeros(
                        batch, c, total_length
                    ).index_put(
                        (processed.batch, processed.cell, processed.token),
                        entropy_rows,
                    )
                    uncertainty, mean_uncertainty = self._route_uncertainty(
                        ownership, entropy_grid, route_valid
                    )
                    if collect_route_metrics:
                        handoff_support_grid = h.new_zeros(
                            batch, c, total_length
                        ).index_put(
                            (
                                processed.batch,
                                processed.cell,
                                processed.token,
                            ),
                            handoff_row_mask.sum(-1).to(h.dtype),
                        )
                        handoff_support = (
                            ownership * handoff_support_grid
                        ).sum(1)
                transported = transport_packed_stewardship(
                    processed,
                    handoff_rows,
                    self.handoff_destinations,
                    batch_size=batch,
                    cells=c,
                    tokens=total_length,
                    support=handoff_row_mask,
                )
                if persistent_backend:
                    next_packed = transported
                    if collect_route_metrics:
                        handoffs = packed_handoffs_to_dense(
                            processed,
                            handoff_rows,
                            batch_size=batch,
                            cells=c,
                            tokens=total_length,
                        )
                else:
                    next_ownership, h, q = unpack_stewardship(
                        transported,
                        batch_size=batch,
                        cells=c,
                        tokens=total_length,
                    )
                    if collect_route_metrics:
                        handoffs = h.new_zeros(
                            batch,
                            c,
                            total_length,
                            handoff_rows.shape[-1],
                        ).index_put(
                            (
                                processed.batch,
                                processed.cell,
                                processed.token,
                            ),
                            handoff_rows,
                        )
            else:
                if ownership is None:
                    if self.config.block_causal:
                        context_result = self._dense_stewardship_context(
                            h,
                            q,
                            None,
                            write,
                            block_size=self.config.block_size,
                            cache=(
                                None
                                if cache is None
                                else cache.relays[relay]
                            ),
                            return_cache=return_cache,
                            graph_slot=relay,
                        )
                        if return_cache:
                            context, occupancy, next_relay_cache = context_result
                            next_relay_caches.append(next_relay_cache)
                        else:
                            context, occupancy = context_result
                    else:
                        writer_mass = write[:, None].expand(-1, c, -1)
                        context = relative_receiver_context(
                            h,
                            q,
                            writer_mass,
                            self.relative_node_bias.to(h.dtype),
                            backend=self.config.receiver_memory_backend,
                        )
                        mass = write.sum(1, keepdim=True).clamp_min(1e-6)
                        occupancy = (
                            q * write[:, None, :, None]
                        ).sum(2) / mass[:, None]
                        occupancy = occupancy[:, :, None].expand(
                            -1, -1, total_length, -1
                        )
                else:
                    context_result = self._dense_stewardship_context(
                        h,
                        q,
                        ownership,
                        write,
                        block_size=(
                            self.config.block_size
                            if self.config.block_causal
                            else None
                        ),
                        cache=(
                            None
                            if cache is None
                            else cache.relays[relay]
                        ),
                        return_cache=return_cache,
                        graph_slot=relay,
                    )
                    if return_cache:
                        context, occupancy, next_relay_cache = context_result
                        next_relay_caches.append(next_relay_cache)
                    else:
                        context, occupancy = context_result
                assert dense_address_edges is not None
                relay_transform_args = (
                    h,
                    context,
                    occupancy,
                    q,
                    dense_address_edges,
                    address_state,
                    step_state,
                    temperature,
                    folded_node_weight,
                    folded_node_offset,
                    relay,
                )
                if (
                    self.config.activation_checkpointing
                    and self.training
                    and torch.is_grad_enabled()
                ):
                    local, q_next = torch_checkpoint.checkpoint(
                        self._dense_relay_transform,
                        *relay_transform_args,
                        use_reentrant=False,
                    )
                else:
                    local, q_next = self._dense_relay_transform(
                        *relay_transform_args
                    )
                if ownership is None:
                    q = q_next
                    radius = self.config.neighbors // 2
                    if self.config.broadcast_mix_cuda_graph:
                        h = cuda_graph_broadcast_mix(
                            local,
                            left_projection=self.left_projection,
                            right_projection=self.right_projection,
                            coupling_norm=self.coupling_norm,
                            neighbor_gates=self.neighbor_gates,
                            self_gates=self.self_gates,
                            radius=radius,
                            graph_slot=relay,
                        )
                    else:
                        left = torch.stack(
                            [
                                torch.roll(local, distance, 1)
                                for distance in range(1, radius + 1)
                            ]
                        ).mean(0)
                        right = torch.stack(
                            [
                                torch.roll(local, -distance, 1)
                                for distance in range(1, radius + 1)
                            ]
                        ).mean(0)
                        gates = torch.sigmoid(self.neighbor_gates)
                        mixed = self.self_gates[None, :, None, None] * local
                        mixed = mixed + gates[None, :, 0, None, None] * self.left_projection(left)
                        mixed = mixed + gates[None, :, 1, None, None] * self.right_projection(right)
                        h = self.coupling_norm(mixed)
                    if traces is not None:
                        traces.append(q)
                    continue

                handoff_input = torch.cat([local, context, occupancy], -1)
                handoff_logits = padded_linear(
                    handoff_input,
                    self.handoff_weight,
                    pad_to_multiple=self.config.dense_kernel_padding,
                ) + self.handoff_bias[None, :, None]
                handoffs = F.softmax(handoff_logits / cell_temperature, -1)
                handoff_entropy = -(
                    handoffs.clamp_min(1e-9) * handoffs.clamp_min(1e-9).log()
                ).sum(-1) / math.log(handoffs.shape[-1])
                handoffs, handoff_mask = self._route_handoffs(
                    handoffs, handoff_entropy, prune=stewardship_prune
                )
                uncertainty, mean_uncertainty = self._route_uncertainty(
                    ownership, handoff_entropy, route_valid
                )
                if collect_route_metrics:
                    handoff_support = (
                        ownership * handoff_mask.sum(-1).to(h.dtype)
                    ).sum(1)
                next_ownership, h, q = transport_stewardship(
                    ownership,
                    local,
                    q_next,
                    handoffs,
                    self.handoff_destinations,
                )

            if ownership is not None:
                base_keep_mass = (
                    self.config.stewardship_keep_mass
                    if stewardship_keep_mass is None
                    else stewardship_keep_mass
                )
                # Familiar/confident work uses the base nucleus. Uncertain
                # handoffs raise the retained probability mass toward one,
                # allowing the route beam to widen again on novel terrain.
                elastic_keep_mass = base_keep_mass + (
                    1.0 - base_keep_mass
                ) * uncertainty
                if (
                    use_packed
                    and self.config.packed_stewardship_backend
                    == "persistent_bucketed"
                ):
                    assert next_packed is not None
                    if stewardship_prune:
                        persistent_packed, _ = prune_packed_stewardship(
                            next_packed,
                            keep_mass=elastic_keep_mass,
                            batch_size=batch,
                            cells=c,
                            tokens=total_length,
                            min_active=self.config.stewardship_min_active,
                            max_active=(
                                self.config.stewardship_max_active
                                if stewardship_max_active is None
                                else stewardship_max_active
                            ),
                            cell_groups=cell_groups,
                            cell_group_count=(
                                output_groups
                                if cell_groups is not None
                                else None
                            ),
                            scouts=self.config.stewardship_scouts,
                            exploration_mass=(
                                self.config.stewardship_exploration
                            ),
                            straight_through=self.training,
                        )
                    else:
                        persistent_packed = next_packed
                    if ownership_traces is not None:
                        ownership = packed_ownership(
                            persistent_packed,
                            batch_size=batch,
                            cells=c,
                            tokens=total_length,
                        )
                else:
                    if stewardship_prune:
                        next_ownership, _ = adaptive_nucleus_stewardship(
                            next_ownership,
                            keep_mass=elastic_keep_mass,
                            min_active=self.config.stewardship_min_active,
                            max_active=(
                                self.config.stewardship_max_active
                                if stewardship_max_active is None
                                else stewardship_max_active
                            ),
                            cell_groups=cell_groups,
                            cell_group_count=(
                                output_groups
                                if cell_groups is not None
                                else None
                            ),
                            scouts=self.config.stewardship_scouts,
                            exploration_mass=(
                                self.config.stewardship_exploration
                            ),
                            straight_through=self.training,
                        )
                    ownership = next_ownership
                if ownership_traces is not None:
                    ownership_traces.append(ownership)
                if route_entropy_traces is not None:
                    if (
                        use_packed
                        and self.config.packed_stewardship_backend
                        == "persistent_bucketed"
                    ):
                        assert persistent_packed is not None
                        route_entropy = packed_weighted_token_sum(
                            persistent_packed,
                            -persistent_packed.mass.clamp_min(
                                1e-9
                            ).log(),
                            batch_size=batch,
                            tokens=total_length,
                        ) / math.log(c)
                        route_cell_load = (
                            persistent_packed.mass.new_zeros(c).scatter_add(
                                0,
                                persistent_packed.cell,
                                persistent_packed.mass,
                            )
                            / route_valid.sum().to(
                                persistent_packed.mass.dtype
                            ).clamp_min(1.0)
                        )
                    else:
                        valid_mass = route_valid[:, None].to(
                            ownership.dtype
                        )
                        route_entropy = -(
                            ownership
                            * ownership.clamp_min(1e-9).log()
                        ).sum(1) / math.log(c)
                        route_cell_load = (
                            (ownership * valid_mass).sum(dim=(0, 2))
                            / route_valid.sum().to(
                                ownership.dtype
                            ).clamp_min(1.0)
                        )
                    route_entropy_traces.append(route_entropy)
                    route_cell_load_traces.append(route_cell_load)
                if mean_uncertainty_traces is not None:
                    mean_uncertainty_traces.append(mean_uncertainty)
                if handoff_traces is not None:
                    handoff_traces.append(handoffs)
                    uncertainty_traces.append(uncertainty)
                    handoff_support_traces.append(handoff_support)
                    execution_traces.append(use_packed)
            if traces is not None:
                if (
                    use_packed
                    and self.config.packed_stewardship_backend
                    == "persistent_bucketed"
                ):
                    assert persistent_packed is not None
                    _, _, packed_locations = unpack_stewardship(
                        persistent_packed,
                        batch_size=batch,
                        cells=c,
                        tokens=total_length,
                    )
                    traces.append(packed_locations)
                else:
                    traces.append(q)

        address_out = padded_linear(
            addresses,
            self.address_to_readout.reshape(c * d, addresses.shape[-1]),
            pad_to_multiple=self.config.dense_kernel_padding,
        ).view(total_length, c, d).permute(1, 0, 2)
        target_hidden = None
        if persistent_packed is not None:
            combined_hidden = packed_readout_hidden(
                persistent_packed,
                address_out,
                batch_size=batch,
                context_tokens=context_length,
                output_tokens=output_slots,
                cell_groups=cell_groups,
                cell_group_count=(
                    output_groups if cell_groups is not None else None
                ),
            )
            if collect_full:
                _, final_states, _ = unpack_stewardship(
                    persistent_packed,
                    batch_size=batch,
                    cells=c,
                    tokens=total_length,
                )
                target_hidden = (
                    final_states[:, :, context_length:]
                    + address_out[None, :, context_length:]
                )
        else:
            target_hidden = (
                h[:, :, context_length:]
                + address_out[None, :, context_length:]
            )
            # The vocabulary embedding is shared, so voting in hidden space
            # is algebraically identical to projecting every cell and then
            # voting. This performs the large vocabulary projection once.
            if self.config.cell_output_mode == "shared_pool":
                if ownership is None:
                    combined_hidden = torch.einsum(
                        "c,bctd->btd",
                        F.softmax(self.output_votes, 0),
                        target_hidden,
                    )
                else:
                    target_ownership = ownership[:, :, context_length:]
                    combined_hidden = torch.einsum(
                        "bct,bctd->btd",
                        target_ownership,
                        target_hidden,
                    )
            else:
                grouped_hidden = []
                for group in range(output_groups):
                    selected = cell_indices.remainder(output_groups).eq(
                        group
                    )
                    if ownership is None:
                        group_votes = F.softmax(
                            self.output_votes[selected], 0
                        )
                        grouped_hidden.append(
                            torch.einsum(
                                "c,bctd->btd",
                                group_votes,
                                target_hidden[:, selected],
                            )
                        )
                    else:
                        group_ownership = ownership[
                            :, selected, context_length:
                        ]
                        group_mass = group_ownership.sum(1)
                        group_weights = normalise_stewardship_messages(
                            group_ownership.permute(0, 2, 1),
                            group_mass,
                        ).permute(0, 2, 1)
                        grouped_hidden.append(
                            torch.einsum(
                                "bct,bctd->btd",
                                group_weights,
                                target_hidden[:, selected],
                            )
                        )
                combined_hidden = torch.cat(grouped_hidden, -1)
        language_hidden = self.output_projection(combined_hidden)
        if isinstance(self.lexical_projection, nn.Linear):
            lexical_hidden = F.linear(
                language_hidden,
                self.lexical_projection.weight.t(),
            )
        else:
            lexical_hidden = language_hidden
        if self.config.readout_precision == "bf16":
            logits = lexical_hidden @ self.token_embedding.weight.t() + self.lm_bias
        else:
            logits = mx_readout_linear(
                lexical_hidden,
                self.token_embedding.weight,
                self.lm_bias,
                precision=self.config.readout_precision,
            )
        if self._forbidden_output_token_ids.numel():
            logits = logits.index_fill(
                -1, self._forbidden_output_token_ids, float("-inf")
            )
        next_block_cache = None
        if return_cache:
            if len(next_relay_caches) != self.config.relays:
                raise RuntimeError("not every relay produced a block cache")
            assert self.config.block_size is not None
            next_block_cache = GatorARWBlockCache(
                cache_length + total_length,
                self.config.block_size,
                tuple(next_relay_caches),
            )
        if diagnostic_mode != "none":
            diagnostic_values = {}
            if ownership is not None:
                assert route_entropy_traces is not None
                assert route_cell_load_traces is not None
                assert active_fraction_traces is not None
                assert mean_uncertainty_traces is not None
                diagnostic_values.update(
                    {
                        "route_entropy": torch.stack(
                            route_entropy_traces, 1
                        ),
                        "route_cell_load": torch.stack(
                            route_cell_load_traces, 0
                        ),
                        "route_active_fraction": torch.stack(
                            active_fraction_traces, 0
                        ),
                        "route_mean_uncertainty": torch.stack(
                            mean_uncertainty_traces, 1
                        ),
                        "route_valid": route_valid,
                    }
                )
                if ownership_traces is not None:
                    diagnostic_values["ownership"] = torch.stack(
                        ownership_traces, 1
                    )
                if collect_route_metrics:
                    assert handoff_traces is not None
                    assert uncertainty_traces is not None
                    assert handoff_support_traces is not None
                    assert execution_traces is not None
                    diagnostic_values.update(
                        {
                            "handoffs": torch.stack(handoff_traces, 1),
                            "route_uncertainty": torch.stack(
                                uncertainty_traces, 1
                            ),
                            "handoff_support": torch.stack(
                                handoff_support_traces, 1
                            ),
                            "packed_execution": torch.tensor(
                                execution_traces,
                                dtype=torch.bool,
                                device=device,
                            ),
                        }
                    )
            if collect_full:
                assert traces is not None
                assert target_hidden is not None
                diagnostic_values.update(
                    {
                        "locations": torch.stack(traces, 1),
                        "hidden": target_hidden,
                    }
                )
            if return_cache:
                assert next_block_cache is not None
                return logits, diagnostic_values, next_block_cache
            return logits, diagnostic_values
        if return_cache:
            assert next_block_cache is not None
            return logits, next_block_cache
        return logits


    @torch.no_grad()
    def prefill_cache(
        self,
        context_ids: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> GatorARWBlockCache:
        """Commit an observed block-aligned prefix into a detached cache."""

        if not self.config.block_causal:
            raise ValueError("prefill_cache requires block_causal=True")
        if context_mask is None:
            context_mask = torch.ones_like(context_ids, dtype=torch.bool)
        _, cache = self(
            context_ids,
            context_mask,
            output_slots=0,
            cache=None,
            return_cache=True,
        )
        return detach_gator_arw_block_cache(cache)

    def forward_cached(
        self,
        cache: GatorARWBlockCache,
        *,
        output_slots: int | None = None,
        temperature: float = 0.35,
    ) -> torch.Tensor:
        """Propose one parallel block without mutating committed state."""

        if not isinstance(cache, GatorARWBlockCache):
            raise TypeError("cache must be a GatorARWBlockCache")
        slots = cache.block_size if output_slots is None else output_slots
        if slots != cache.block_size:
            raise ValueError(
                "forward_cached proposes exactly one configured output block"
            )
        batch = cache.relays[0].traffic_numerator.shape[0]
        empty_ids = torch.empty(
            batch, 0, dtype=torch.long, device=cache.next_position.device
        )
        empty_mask = torch.empty(
            batch, 0, dtype=torch.bool, device=cache.next_position.device
        )
        return self(
            empty_ids,
            empty_mask,
            output_slots=slots,
            temperature=temperature,
            cache=cache,
        )

    @torch.no_grad()
    def append_cache(
        self,
        cache: GatorARWBlockCache,
        accepted_ids: torch.Tensor,
        accepted_mask: torch.Tensor | None = None,
    ) -> GatorARWBlockCache:
        """Recompute accepted tokens as observed context and commit them."""

        if accepted_mask is None:
            accepted_mask = torch.ones_like(accepted_ids, dtype=torch.bool)
        _, next_cache = self(
            accepted_ids,
            accepted_mask,
            output_slots=0,
            cache=cache,
            return_cache=True,
        )
        return detach_gator_arw_block_cache(next_cache)
