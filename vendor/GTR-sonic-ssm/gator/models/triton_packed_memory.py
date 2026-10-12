"""Optional fused packed-reader reduction for Convergent Gator.

The packed convolution and FFT receiver backends produce one normalized
memory numerator per active node.  A reader then selects ``K`` node memories,
normalizes them by the corresponding denominator, weights them by route mass,
and reduces them into one lane context.  Expressing that final operation with
``expand``/``gather`` materializes a large ``[B,T,L,K,C]`` tensor for every
feature chunk.

This module keeps that boundary explicit:

``torch``
    Portable numerical oracle implemented entirely with PyTorch operations.

``triton``
    CUDA-only fused forward and backward.  The implementation is registered
    through :func:`torch.library.triton_op`, so ``torch.compile`` can retain a
    structured custom-kernel boundary.  Triton is an optional dependency;
    importing this module never requires it.

The first production contract is deliberately float32.  Supporting lower
precision requires an explicit accumulation and atomic-gradient policy rather
than an implicit cast or a silent fallback.
"""

from __future__ import annotations

import math
from typing import Literal

import torch


PackedReaderBackend = Literal["torch", "triton"]


_TRITON_IMPORT_ERROR: BaseException | None = None
try:  # Triton is unavailable on the supported local Windows/CPU test host.
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
except (ImportError, AttributeError, OSError, RuntimeError) as error:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    triton_op = None  # type: ignore[assignment]
    wrap_triton = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR = error


_TRITON_AVAILABLE = all(
    value is not None for value in (triton, tl, triton_op, wrap_triton)
)


def triton_packed_reader_available() -> bool:
    """Return whether the optional Triton implementation can be launched."""

    return _TRITON_AVAILABLE


def _validate_inputs(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    reader_nodes: torch.Tensor,
    reader_mass: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[int, int, int, int, int, int]:
    """Validate the common packed-reader contract and return its dimensions."""

    if numerator.ndim != 4:
        raise ValueError(
            "numerator must have [batch,tokens,active_nodes,features] shape"
        )
    if denominator.ndim != 3:
        raise ValueError(
            "denominator must have [batch,tokens,active_nodes] shape"
        )
    if reader_nodes.ndim != 4 or reader_mass.shape != reader_nodes.shape:
        raise ValueError(
            "reader_nodes and reader_mass must share "
            "[batch,tokens,lanes,K] shape"
        )
    batch, tokens, active_nodes, features = numerator.shape
    if denominator.shape != (batch, tokens, active_nodes):
        raise ValueError("denominator prefix must match numerator")
    if reader_nodes.shape[:2] != (batch, tokens):
        raise ValueError("packed reader prefix must match numerator")
    lanes, route_width = reader_nodes.shape[2:]
    if min(batch, tokens, active_nodes, features, lanes, route_width) <= 0:
        raise ValueError("every packed-reader dimension must be positive")
    if reader_nodes.dtype != torch.long:
        raise TypeError("reader_nodes must use torch.long indices")
    if not numerator.is_floating_point():
        raise TypeError("numerator must be floating point")
    if denominator.dtype != numerator.dtype or reader_mass.dtype != numerator.dtype:
        raise TypeError(
            "numerator, denominator, and reader_mass must share a dtype"
        )
    devices = {
        tensor.device
        for tensor in (numerator, denominator, reader_nodes, reader_mass)
    }
    if len(devices) != 1:
        raise ValueError("every packed-reader input must share a device")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite positive number")

    # Invalid packed entries must already carry a safe node ID and zero mass.
    # Asynchronous assertions preserve that contract without a host sync on
    # CUDA and protect the custom kernel from an out-of-bounds load.
    torch._assert_async(
        reader_nodes.ge(0).logical_and(reader_nodes.lt(active_nodes)).all(),
        "a packed reader node ID is outside the active node table",
    )
    torch._assert_async(
        reader_mass.ge(0).all(),
        "packed reader mass must be nonnegative",
    )
    return batch, tokens, active_nodes, features, lanes, route_width


def torch_packed_reader_context(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    reader_nodes: torch.Tensor,
    reader_mass: torch.Tensor,
    *,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Portable packed-reader oracle using ordinary PyTorch operations.

    Layouts are ``numerator[B,T,U,C]``, ``denominator[B,T,U]``, and packed
    reader node IDs/masses ``[B,T,L,K]``.  The result is ``[B,T,L,C]``.
    """

    _validate_inputs(
        numerator,
        denominator,
        reader_nodes,
        reader_mass,
        epsilon=epsilon,
    )
    return _torch_packed_reader_context_unchecked(
        numerator,
        denominator,
        reader_nodes,
        reader_mass,
        epsilon=epsilon,
    )


def _torch_packed_reader_context_unchecked(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    reader_nodes: torch.Tensor,
    reader_mass: torch.Tensor,
    *,
    epsilon: float,
) -> torch.Tensor:
    """Evaluate the Torch equation after a trusted caller validates inputs."""

    batch, tokens, active_nodes, features = numerator.shape
    lanes = reader_nodes.shape[2]
    selected_numerator = numerator.unsqueeze(2).expand(
        batch,
        tokens,
        lanes,
        active_nodes,
        features,
    ).gather(
        3,
        reader_nodes[..., None].expand(-1, -1, -1, -1, features),
    )
    selected_denominator = denominator.unsqueeze(2).expand(
        batch,
        tokens,
        lanes,
        active_nodes,
    ).gather(3, reader_nodes)
    return (
        reader_mass[..., None]
        * selected_numerator
        / selected_denominator.clamp_min(epsilon)[..., None]
    ).sum(dim=3)


def _torch_packed_reader_backward_unchecked(
    grad_output: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    reader_nodes: torch.Tensor,
    reader_mass: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the exact packed-reader adjoint without an autograd replay.

    The route loop keeps the portable oracle bounded by one lane-sized
    feature block.  In particular it never expands the active-node table to
    ``[B,T,L,U,C]`` merely to gather ``K`` entries from it.
    """

    batch, tokens, _, features = numerator.shape
    lanes, routes = reader_nodes.shape[2:]
    if grad_output.shape != (batch, tokens, lanes, features):
        raise ValueError("grad_output must match the packed-reader output")
    grad_numerator = torch.zeros_like(
        numerator, memory_format=torch.contiguous_format
    )
    grad_denominator = torch.zeros_like(
        denominator, memory_format=torch.contiguous_format
    )
    grad_mass = torch.empty_like(
        reader_mass, memory_format=torch.contiguous_format
    )
    for route in range(routes):
        route_nodes = reader_nodes[..., route]
        route_mass = reader_mass[..., route]
        feature_index = route_nodes[..., None].expand(
            -1, -1, -1, features
        )
        selected_numerator = numerator.gather(2, feature_index)
        selected_denominator = denominator.gather(2, route_nodes)
        safe_denominator = selected_denominator.clamp_min(epsilon)
        scaled_output_gradient = (
            route_mass[..., None]
            * grad_output
            / safe_denominator[..., None]
        )
        grad_numerator.scatter_add_(
            2,
            feature_index,
            scaled_output_gradient,
        )
        dot = (grad_output * selected_numerator).sum(dim=-1)
        denominator_gradient = torch.where(
            selected_denominator.ge(epsilon),
            -route_mass * dot / safe_denominator.square(),
            torch.zeros_like(dot),
        )
        grad_denominator.scatter_add_(
            2,
            route_nodes,
            denominator_gradient,
        )
        grad_mass[..., route] = dot / safe_denominator
    return grad_numerator, grad_denominator, grad_mass


if _TRITON_AVAILABLE:

    @triton.jit
    def _packed_reader_forward_kernel(
        numerator_ptr,
        denominator_ptr,
        nodes_ptr,
        mass_ptr,
        output_ptr,
        stride_nb,
        stride_nt,
        stride_nu,
        stride_nc,
        stride_db,
        stride_dt,
        stride_du,
        stride_ib,
        stride_it,
        stride_il,
        stride_ik,
        stride_mb,
        stride_mt,
        stride_ml,
        stride_mk,
        stride_ob,
        stride_ot,
        stride_ol,
        stride_oc,
        TOKENS: tl.constexpr,
        LANES: tl.constexpr,
        ROUTES: tl.constexpr,
        FEATURES: tl.constexpr,
        EPSILON: tl.constexpr,
        BLOCK_FEATURES: tl.constexpr,
    ):
        packed_row = tl.program_id(0)
        feature_block = tl.program_id(1)
        lane = packed_row % LANES
        token = (packed_row // LANES) % TOKENS
        batch = packed_row // (TOKENS * LANES)
        feature = feature_block * BLOCK_FEATURES + tl.arange(
            0, BLOCK_FEATURES
        )
        feature_mask = feature < FEATURES
        accumulator = tl.zeros((BLOCK_FEATURES,), dtype=tl.float32)

        # K reaches 32 in the target model. Aggressively unrolling the full
        # gather body creates a very large Blackwell kernel and can inflate
        # both register pressure and first-compile latency. Keep a real loop;
        # the feature vector remains statically blocked and vectorized.
        for route in tl.range(0, ROUTES):
            node_offset = (
                batch * stride_ib
                + token * stride_it
                + lane * stride_il
                + route * stride_ik
            )
            node = tl.load(nodes_ptr + node_offset)
            mass = tl.load(
                mass_ptr
                + batch * stride_mb
                + token * stride_mt
                + lane * stride_ml
                + route * stride_mk
            )
            denominator = tl.load(
                denominator_ptr
                + batch * stride_db
                + token * stride_dt
                + node * stride_du
            )
            safe_denominator = tl.maximum(denominator, EPSILON)
            values = tl.load(
                numerator_ptr
                + batch * stride_nb
                + token * stride_nt
                + node * stride_nu
                + feature * stride_nc,
                mask=feature_mask,
                other=0.0,
            )
            accumulator += mass * values / safe_denominator

        tl.store(
            output_ptr
            + batch * stride_ob
            + token * stride_ot
            + lane * stride_ol
            + feature * stride_oc,
            accumulator,
            mask=feature_mask,
        )


    @triton.jit
    def _packed_reader_backward_kernel(
        grad_output_ptr,
        numerator_ptr,
        denominator_ptr,
        nodes_ptr,
        mass_ptr,
        grad_numerator_ptr,
        grad_denominator_ptr,
        grad_mass_ptr,
        stride_gob,
        stride_got,
        stride_gol,
        stride_goc,
        stride_nb,
        stride_nt,
        stride_nu,
        stride_nc,
        stride_db,
        stride_dt,
        stride_du,
        stride_ib,
        stride_it,
        stride_il,
        stride_ik,
        stride_mb,
        stride_mt,
        stride_ml,
        stride_mk,
        stride_gnb,
        stride_gnt,
        stride_gnu,
        stride_gnc,
        stride_gdb,
        stride_gdt,
        stride_gdu,
        stride_gmb,
        stride_gmt,
        stride_gml,
        stride_gmk,
        TOKENS: tl.constexpr,
        LANES: tl.constexpr,
        ROUTES: tl.constexpr,
        FEATURES: tl.constexpr,
        EPSILON: tl.constexpr,
        BLOCK_FEATURES: tl.constexpr,
    ):
        packed_row = tl.program_id(0)
        feature_block = tl.program_id(1)
        lane = packed_row % LANES
        token = (packed_row // LANES) % TOKENS
        batch = packed_row // (TOKENS * LANES)
        feature = feature_block * BLOCK_FEATURES + tl.arange(
            0, BLOCK_FEATURES
        )
        feature_mask = feature < FEATURES
        grad_output = tl.load(
            grad_output_ptr
            + batch * stride_gob
            + token * stride_got
            + lane * stride_gol
            + feature * stride_goc,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)

        for route in tl.range(0, ROUTES):
            packed_offset = (
                batch * stride_ib
                + token * stride_it
                + lane * stride_il
                + route * stride_ik
            )
            node = tl.load(nodes_ptr + packed_offset)
            mass = tl.load(
                mass_ptr
                + batch * stride_mb
                + token * stride_mt
                + lane * stride_ml
                + route * stride_mk
            )
            denominator = tl.load(
                denominator_ptr
                + batch * stride_db
                + token * stride_dt
                + node * stride_du
            )
            safe_denominator = tl.maximum(denominator, EPSILON)
            numerator = tl.load(
                numerator_ptr
                + batch * stride_nb
                + token * stride_nt
                + node * stride_nu
                + feature * stride_nc,
                mask=feature_mask,
                other=0.0,
            ).to(tl.float32)

            tl.atomic_add(
                grad_numerator_ptr
                + batch * stride_gnb
                + token * stride_gnt
                + node * stride_gnu
                + feature * stride_gnc,
                mass * grad_output / safe_denominator,
                mask=feature_mask,
            )
            dot = tl.sum(grad_output * numerator, axis=0)
            # ``clamp_min`` blocks the denominator gradient below its floor.
            denominator_gradient = tl.where(
                denominator >= EPSILON,
                -mass * dot / (safe_denominator * safe_denominator),
                0.0,
            )
            tl.atomic_add(
                grad_denominator_ptr
                + batch * stride_gdb
                + token * stride_gdt
                + node * stride_gdu,
                denominator_gradient,
            )
            tl.atomic_add(
                grad_mass_ptr
                + batch * stride_gmb
                + token * stride_gmt
                + lane * stride_gml
                + route * stride_gmk,
                dot / safe_denominator,
            )


    def _feature_block_size(features: int) -> int:
        # Width chunks are normally 32--64.  Bound larger chunks so register
        # pressure cannot grow without limit; the backward atomically combines
        # partial feature-block dot products when more than one block is used.
        return min(128, triton.next_power_of_2(features))


    @triton_op("gator::packed_reader_forward", mutates_args={})
    def _triton_packed_reader_forward(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        reader_nodes: torch.Tensor,
        reader_mass: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        batch, tokens, _, features = numerator.shape
        lanes, routes = reader_nodes.shape[2:]
        output = torch.empty(
            (batch, tokens, lanes, features),
            device=numerator.device,
            dtype=numerator.dtype,
        )
        block_features = _feature_block_size(features)
        grid = (
            batch * tokens * lanes,
            triton.cdiv(features, block_features),
        )
        wrap_triton(_packed_reader_forward_kernel)[grid](
            numerator,
            denominator,
            reader_nodes,
            reader_mass,
            output,
            *numerator.stride(),
            *denominator.stride(),
            *reader_nodes.stride(),
            *reader_mass.stride(),
            *output.stride(),
            TOKENS=tokens,
            LANES=lanes,
            ROUTES=routes,
            FEATURES=features,
            EPSILON=epsilon,
            BLOCK_FEATURES=block_features,
            num_warps=4,
        )
        return output


    @triton_op("gator::packed_reader_backward", mutates_args={})
    def _triton_packed_reader_backward(
        grad_output: torch.Tensor,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        reader_nodes: torch.Tensor,
        reader_mass: torch.Tensor,
        epsilon: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, tokens, _, features = numerator.shape
        lanes, routes = reader_nodes.shape[2:]
        grad_numerator = torch.zeros_like(
            numerator, memory_format=torch.contiguous_format
        )
        grad_denominator = torch.zeros_like(
            denominator, memory_format=torch.contiguous_format
        )
        grad_mass = torch.zeros_like(
            reader_mass, memory_format=torch.contiguous_format
        )
        block_features = _feature_block_size(features)
        grid = (
            batch * tokens * lanes,
            triton.cdiv(features, block_features),
        )
        wrap_triton(_packed_reader_backward_kernel)[grid](
            grad_output,
            numerator,
            denominator,
            reader_nodes,
            reader_mass,
            grad_numerator,
            grad_denominator,
            grad_mass,
            *grad_output.stride(),
            *numerator.stride(),
            *denominator.stride(),
            *reader_nodes.stride(),
            *reader_mass.stride(),
            *grad_numerator.stride(),
            *grad_denominator.stride(),
            *grad_mass.stride(),
            TOKENS=tokens,
            LANES=lanes,
            ROUTES=routes,
            FEATURES=features,
            EPSILON=epsilon,
            BLOCK_FEATURES=block_features,
            num_warps=4,
        )
        return grad_numerator, grad_denominator, grad_mass


    def _setup_triton_backward_context(ctx, inputs, output) -> None:
        del output
        numerator, denominator, reader_nodes, reader_mass, epsilon = inputs
        ctx.save_for_backward(
            numerator,
            denominator,
            reader_nodes,
            reader_mass,
        )
        ctx.epsilon = epsilon


    def _triton_backward(ctx, grad_output):
        numerator, denominator, reader_nodes, reader_mass = ctx.saved_tensors
        grad_numerator, grad_denominator, grad_mass = (
            _triton_packed_reader_backward(
                grad_output,
                numerator,
                denominator,
                reader_nodes,
                reader_mass,
                ctx.epsilon,
            )
        )
        return grad_numerator, grad_denominator, None, grad_mass, None


    _triton_packed_reader_forward.register_autograd(
        _triton_backward,
        setup_context=_setup_triton_backward_context,
    )

else:
    _triton_packed_reader_forward = None
    _triton_packed_reader_backward = None


def _triton_unavailable_message() -> str:
    detail = (
        f" ({type(_TRITON_IMPORT_ERROR).__name__}: {_TRITON_IMPORT_ERROR})"
        if _TRITON_IMPORT_ERROR is not None
        else ""
    )
    return (
        "packed reader backend='triton' requires a CUDA PyTorch build with "
        f"Triton and torch.library.triton_op support{detail}"
    )


def packed_reader_context(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    reader_nodes: torch.Tensor,
    reader_mass: torch.Tensor,
    *,
    epsilon: float = 1e-9,
    backend: PackedReaderBackend = "torch",
) -> torch.Tensor:
    """Reduce packed node memories through an explicit execution backend.

    ``backend='torch'`` is portable and differentiable on every PyTorch
    device. ``backend='triton'`` is intentionally strict: it requires CUDA,
    float32, and the optional Triton integration.  It never falls back to the
    Torch implementation silently.
    """

    _validate_inputs(
        numerator,
        denominator,
        reader_nodes,
        reader_mass,
        epsilon=epsilon,
    )
    return _packed_reader_context_unchecked(
        numerator,
        denominator,
        reader_nodes,
        reader_mass,
        epsilon=epsilon,
        backend=backend,
    )


def _packed_reader_context_unchecked(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    reader_nodes: torch.Tensor,
    reader_mass: torch.Tensor,
    *,
    epsilon: float,
    backend: PackedReaderBackend,
) -> torch.Tensor:
    """Dispatch one prevalidated chunk without tensor-wide safety scans.

    This private entry point exists for ``packed_convolution_node_context``.
    That receiver validates the original frontier once, constructs bounded
    compact IDs itself, and then emits multiple hidden-feature chunks. Calling
    the public function for every chunk would repeat two full-device assertion
    reductions despite every chunk sharing the same trusted IDs and masses.

    External callers must use :func:`packed_reader_context` instead.
    """

    if backend == "torch":
        return _torch_packed_reader_context_unchecked(
            numerator,
            denominator,
            reader_nodes,
            reader_mass,
            epsilon=epsilon,
        )
    if backend != "triton":
        raise ValueError("packed reader backend must be 'torch' or 'triton'")
    if numerator.device.type != "cuda":
        raise ValueError("packed reader backend='triton' requires CUDA tensors")
    if numerator.dtype != torch.float32:
        raise TypeError(
            "packed reader backend='triton' currently supports float32 only"
        )
    if not _TRITON_AVAILABLE or _triton_packed_reader_forward is None:
        raise RuntimeError(_triton_unavailable_message())
    return _triton_packed_reader_forward(
        numerator,
        denominator,
        reader_nodes,
        reader_mass,
        float(epsilon),
    )


def _packed_reader_backward_unchecked(
    grad_output: torch.Tensor,
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    reader_nodes: torch.Tensor,
    reader_mass: torch.Tensor,
    *,
    epsilon: float,
    backend: PackedReaderBackend,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch the local reader adjoint after trusted forward validation."""

    if backend == "torch":
        return _torch_packed_reader_backward_unchecked(
            grad_output,
            numerator,
            denominator,
            reader_nodes,
            reader_mass,
            epsilon=epsilon,
        )
    if backend != "triton":
        raise ValueError("packed reader backend must be 'torch' or 'triton'")
    if numerator.device.type != "cuda":
        raise ValueError("packed reader backend='triton' requires CUDA tensors")
    if numerator.dtype != torch.float32:
        raise TypeError(
            "packed reader backend='triton' currently supports float32 only"
        )
    if not _TRITON_AVAILABLE or _triton_packed_reader_backward is None:
        raise RuntimeError(_triton_unavailable_message())
    return _triton_packed_reader_backward(
        grad_output,
        numerator,
        denominator,
        reader_nodes,
        reader_mass,
        float(epsilon),
    )


__all__ = [
    "PackedReaderBackend",
    "packed_reader_context",
    "torch_packed_reader_context",
    "triton_packed_reader_available",
]
