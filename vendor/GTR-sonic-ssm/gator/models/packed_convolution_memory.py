"""Chunked convolutional receiver memory for packed Convergent routes.

This module is an exact-equation alternative to the dense Convergent memory
path.  It uses the translation-invariant relative-position kernel as a grouped
one-dimensional convolution, so it never materializes ``[T,T,N]``.  Hidden
features are processed in bounded chunks and the result is gathered directly
at packed reader nodes, so it also avoids a persistent ``[B,T,N,D]`` node
memory tensor.

The implementation intentionally lives behind a standalone API.  It can be
used as a numerical oracle before wiring the backend into the compiled model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from gator.models.memory import _relative_convolution_kernel
from gator.models.triton_packed_memory import (
    PackedReaderBackend,
    _packed_reader_context_unchecked,
)
from gator.models.triton_packed_writer import (
    PackedWriterBackend,
    _packed_writer_features_unchecked,
    triton_packed_writer_available,
)


@dataclass(frozen=True)
class PackedConvolutionMemoryDiagnostics:
    """Small intermediates useful for profiling and equivalence tests."""

    active_nodes: torch.Tensor
    denominator: torch.Tensor
    kernel: torch.Tensor


ConvolutionBackend = Literal["direct", "fft"]


def _validate_inputs(
    hidden: torch.Tensor,
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    write: torch.Tensor,
    relative_node_bias: torch.Tensor,
    *,
    epsilon: float,
    feature_chunk_size: int,
) -> tuple[int, int, int, int, int, int]:
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
    if relative_node_bias.ndim != 2 or relative_node_bias.shape[-1] != 4:
        raise ValueError("relative_node_bias must have [nodes,4] shape")
    n_nodes = relative_node_bias.shape[0]
    if n_nodes <= 0:
        raise ValueError("relative_node_bias must contain at least one node")
    if nodes.dtype != torch.long:
        raise ValueError("nodes must use int64")
    if valid.dtype != torch.bool:
        raise ValueError("valid must be boolean")
    if not hidden.is_floating_point() or not mass.is_floating_point():
        raise ValueError("hidden and mass must be floating point")
    if not write.is_floating_point() or not relative_node_bias.is_floating_point():
        raise ValueError("write and relative_node_bias must be floating point")
    if not (
        hidden.dtype == mass.dtype == write.dtype == relative_node_bias.dtype
    ):
        raise ValueError(
            "hidden, mass, write, and relative_node_bias must share a dtype"
        )
    if len(
        {
            value.device
            for value in (hidden, nodes, mass, valid, write, relative_node_bias)
        }
    ) != 1:
        raise ValueError("every packed memory input must share a device")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if (
        isinstance(feature_chunk_size, bool)
        or not isinstance(feature_chunk_size, int)
        or feature_chunk_size <= 0
    ):
        raise ValueError("feature_chunk_size must be a positive integer")
    torch._assert_async(
        ((~valid) | (nodes.ge(0) & nodes.lt(n_nodes))).all(),
        "a valid packed node ID is outside the graph",
    )
    torch._assert_async(
        torch.where(valid, mass, torch.zeros_like(mass)).ge(0).all(),
        "valid packed route mass must be nonnegative",
    )
    return batch, tokens, lanes, width, nodes.shape[-1], n_nodes


def _route_deltas(
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    compact_nodes: torch.Tensor,
    *,
    epsilon: float,
    assume_unique_routes: bool,
    identity_compact_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map routes to compact IDs and return exact post-floor mass deltas."""

    safe_nodes = torch.where(valid, nodes, torch.zeros_like(nodes))
    if identity_compact_ids:
        # The compiled production path keeps the complete node table, so
        # ``compact_nodes == arange(N)``. Searching every route for this
        # identity mapping only adds a large CUDA launch, then repeats in the
        # explicit backward recomputation.
        compact_ids = safe_nodes
    else:
        compact_ids = torch.searchsorted(compact_nodes, safe_nodes)
        compact_ids = compact_ids.clamp_max(compact_nodes.numel() - 1)
    valid_mass = mass * valid.to(mass.dtype)
    if assume_unique_routes:
        # Integrated PackedNodeFrontier transport already collision-merges
        # every packet. Avoid rebuilding a K-by-K equality matrix merely to
        # rediscover that invariant in every memory call.
        route_delta = torch.where(
            valid,
            valid_mass.clamp_min(epsilon) - epsilon,
            torch.zeros_like(valid_mass),
        )
        return safe_nodes, compact_ids, route_delta
    deltas: list[torch.Tensor] = []
    for route in range(nodes.shape[-1]):
        route_node = safe_nodes[..., route]
        route_valid = valid[..., route]
        same_node = safe_nodes.eq(route_node.unsqueeze(-1)) & valid
        merged_lane_mass = (valid_mass * same_node.to(mass.dtype)).sum(-1)
        if route:
            already_seen = (
                safe_nodes[..., :route].eq(route_node.unsqueeze(-1))
                & valid[..., :route]
            ).any(-1)
        else:
            already_seen = torch.zeros_like(route_valid)
        representative = route_valid & ~already_seen
        deltas.append(
            (merged_lane_mass.clamp_min(epsilon) - epsilon)
            * representative.to(mass.dtype)
        )
    return safe_nodes, compact_ids, torch.stack(deltas, dim=-1)


def _writer_mass(
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    *,
    lanes: int,
    active_count: int,
    epsilon: float,
) -> torch.Tensor:
    batch, tokens = write.shape
    result = (
        write * (lanes * epsilon)
    )[:, :, None].expand(batch, tokens, active_count).clone()
    flat_ids = compact_ids.flatten(2, 3)
    written_delta = (
        route_delta * write[:, :, None, None]
    ).flatten(2, 3)
    result = result.scatter_add(2, flat_ids, written_delta)
    return result.permute(0, 2, 1)


def _writer_state_chunk(
    hidden: torch.Tensor,
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    start: int,
    stop: int,
    *,
    active_count: int,
    epsilon: float,
) -> torch.Tensor:
    batch, tokens, _, _ = hidden.shape
    chunk = hidden[..., start:stop]
    chunk_width = stop - start
    baseline = (
        write[..., None] * epsilon * chunk.sum(dim=2)
    )[:, :, None, :].expand(batch, tokens, active_count, chunk_width)
    result = baseline.clone()
    # Route expansion is bounded by packed K and the feature chunk, never N.
    # Flattening lanes and routes turns K scatter launches into one operation.
    written_delta = route_delta * write[:, :, None, None]
    route_state = (
        written_delta[..., None] * chunk.unsqueeze(3)
    ).flatten(2, 3)
    flat_ids = compact_ids.flatten(2, 3)
    result = result.scatter_add(
        2,
        flat_ids[..., None].expand(-1, -1, -1, chunk_width),
        route_state,
    )
    return result.permute(0, 2, 1, 3)


def _reader_context_chunk(
    numerator_by_token: torch.Tensor,
    denominator_by_token: torch.Tensor,
    compact_ids: torch.Tensor,
    route_mass: torch.Tensor,
    *,
    epsilon: float = 1e-9,
    backend: PackedReaderBackend = "torch",
) -> torch.Tensor:
    """Gather every packed route through the selected reader backend."""

    # The complete receiver validates the original frontier once before it
    # constructs compact IDs and route mass. Repeating public packed-reader
    # validation here would launch two tensor-wide assertion reductions for
    # every hidden-feature chunk.
    return _packed_reader_context_unchecked(
        numerator_by_token,
        denominator_by_token,
        compact_ids,
        route_mass,
        epsilon=epsilon,
        backend=backend,
    )


def _grouped_convolution(
    features: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """Convolve ``[B,U,T,C]`` with one translation kernel per node."""

    batch, active_count, tokens, feature_width = features.shape
    packed = features.permute(0, 3, 1, 2).reshape(
        batch * feature_width,
        active_count,
        tokens,
    )
    mixed = F.conv1d(
        packed,
        kernel[:, None],
        padding=tokens - 1,
        groups=active_count,
    )
    return mixed.reshape(
        batch,
        feature_width,
        active_count,
        tokens,
    ).permute(0, 2, 3, 1)


def _fft_kernel_spectrum(
    kernel: torch.Tensor,
    *,
    tokens: int,
    feature_dtype: torch.dtype,
) -> tuple[int, torch.Tensor]:
    """Prepare one reusable kernel spectrum for an FFT memory call."""

    linear_length = 3 * tokens - 2
    transform_length = 1 << max(linear_length - 1, 0).bit_length()
    # CUDA FFT does not support every low-precision dtype.  Accumulating the
    # receiver normalization in float32 is also safer than a long BF16 sum.
    compute_dtype = (
        torch.float32
        if feature_dtype in (torch.float16, torch.bfloat16)
        else feature_dtype
    )
    # conv1d is cross-correlation.  Flipping its kernel turns the operation
    # into an ordinary linear convolution; receiver i lives at full output
    # index ``tokens - 1 + i``.
    kernel_spectrum = torch.fft.rfft(
        kernel.to(compute_dtype).flip(-1),
        n=transform_length,
        dim=-1,
    )
    return transform_length, kernel_spectrum


def _fft_grouped_convolution(
    features: torch.Tensor,
    kernel_spectrum: torch.Tensor,
    *,
    transform_length: int,
) -> torch.Tensor:
    """Evaluate one correlation using a precomputed kernel spectrum."""

    result, _ = _fft_grouped_convolution_with_source_spectrum(
        features,
        kernel_spectrum,
        transform_length=transform_length,
    )
    return result


def _fft_grouped_convolution_with_source_spectrum(
    features: torch.Tensor,
    kernel_spectrum: torch.Tensor,
    *,
    transform_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate one correlation and retain its reusable source spectrum.

    The selected receiver window is materialized in feature-contiguous
    ``[B,U,T,C]`` layout.  Returning a permuted view here would both retain
    the much larger length-``transform_length`` inverse-FFT allocation and
    make adjacent reader features separated by ``U * transform_length``
    elements.
    """

    native_features = features.to(
        torch.float32
        if features.dtype in (torch.float16, torch.bfloat16)
        else features.dtype
    ).permute(0, 3, 1, 2)
    return _fft_native_grouped_convolution_with_source_spectrum(
        native_features,
        kernel_spectrum,
        transform_length=transform_length,
        original_dtype=features.dtype,
    )


def _fft_native_grouped_convolution(
    features: torch.Tensor,
    kernel_spectrum: torch.Tensor,
    *,
    transform_length: int,
    original_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convolve FFT-native contiguous ``[B,C,U,T]`` writer features."""

    result, _ = _fft_native_grouped_convolution_with_source_spectrum(
        features,
        kernel_spectrum,
        transform_length=transform_length,
        original_dtype=original_dtype,
    )
    return result


def _fft_native_grouped_convolution_with_source_spectrum(
    features: torch.Tensor,
    kernel_spectrum: torch.Tensor,
    *,
    transform_length: int,
    original_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convolve ``[B,C,U,T]`` features without a writer-layout transpose."""

    if features.ndim != 4:
        raise ValueError("FFT-native features must have [B,C,U,T] shape")
    _, _, active_count, tokens = features.shape
    if kernel_spectrum.shape[0] != active_count:
        raise ValueError("kernel spectrum must contain one row per active node")
    if original_dtype is None:
        original_dtype = features.dtype
    compute_dtype = (
        torch.float32
        if original_dtype in (torch.float16, torch.bfloat16)
        else original_dtype
    )
    feature_fft = torch.fft.rfft(
        features.to(compute_dtype),
        n=transform_length,
        dim=-1,
    )
    full = torch.fft.irfft(
        feature_fft * kernel_spectrum[None, None],
        n=transform_length,
        dim=-1,
    )
    selected = full[..., tokens - 1 : 2 * tokens - 1]
    result = (
        selected.permute(0, 2, 3, 1)
        .to(original_dtype)
        .contiguous()
    )
    return result, feature_fft


def packed_convolution_node_context(
    hidden: torch.Tensor,
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    write: torch.Tensor,
    relative_node_bias: torch.Tensor,
    *,
    causal: bool = False,
    epsilon: float = 1e-9,
    feature_chunk_size: int = 32,
    compact_active_nodes: bool = False,
    convolution_backend: ConvolutionBackend = "direct",
    reader_backend: PackedReaderBackend = "torch",
    writer_backend: PackedWriterBackend = "torch",
    assume_unique_routes: bool = False,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[
    torch.Tensor,
    PackedConvolutionMemoryDiagnostics,
]:
    """Return exact-equation receiver context without dense quadratic memory.

    ``compact_active_nodes`` computes only the union of nodes carrying positive
    packed reader mass.  It is useful for eager sparse execution; because the
    union has data-dependent shape, compiled callers should normally leave it
    disabled or supply a future fixed-cap active-node plan.

    ``assume_unique_routes`` skips duplicate merging when the caller already
    carries a canonical collision-merged frontier. It must remain false for
    arbitrary external packed inputs.
    """

    if convolution_backend not in ("direct", "fft"):
        raise ValueError("convolution_backend must be 'direct' or 'fft'")
    if reader_backend not in ("torch", "triton"):
        raise ValueError("reader_backend must be 'torch' or 'triton'")
    if writer_backend not in ("torch", "triton"):
        raise ValueError("writer_backend must be 'torch' or 'triton'")
    if writer_backend == "triton":
        if convolution_backend != "fft":
            raise ValueError(
                "writer_backend='triton' requires convolution_backend='fft'"
            )
        if hidden.device.type != "cuda":
            raise ValueError("writer_backend='triton' requires CUDA tensors")
        if hidden.dtype != torch.float32:
            raise TypeError(
                "writer_backend='triton' currently supports float32 only"
            )
        if feature_chunk_size > 128:
            raise ValueError(
                "writer_backend='triton' currently supports feature chunks "
                "up to 128"
            )
        if not triton_packed_writer_available():
            raise RuntimeError(
                "writer_backend='triton' requires Triton plus "
                "torch.library.triton_op/wrap_triton support"
            )
    batch, tokens, lanes, width, _, n_nodes = _validate_inputs(
        hidden,
        nodes,
        mass,
        valid,
        write,
        relative_node_bias,
        epsilon=epsilon,
        feature_chunk_size=feature_chunk_size,
    )
    positive = valid & mass.gt(0)
    if compact_active_nodes:
        active_nodes = torch.unique(nodes[positive], sorted=True)
    else:
        active_nodes = torch.arange(n_nodes, device=nodes.device)
    if active_nodes.numel() == 0:
        # Keep every differentiable input attached to the zero result.
        zero = (
            hidden.sum()
            + mass.sum()
            + write.sum()
            + relative_node_bias.sum()
        ) * 0
        context = hidden * zero
        if not return_diagnostics:
            return context
        empty_denominator = hidden.new_empty(batch, 0, tokens)
        empty_kernel = hidden.new_empty(0, 2 * tokens - 1)
        return context, PackedConvolutionMemoryDiagnostics(
            active_nodes=active_nodes,
            denominator=empty_denominator,
            kernel=empty_kernel,
        )

    _, compact_ids, route_delta = _route_deltas(
        nodes,
        mass,
        valid,
        active_nodes,
        epsilon=epsilon,
        assume_unique_routes=assume_unique_routes,
        identity_compact_ids=not compact_active_nodes,
    )
    active_count = active_nodes.numel()
    kernel = _relative_convolution_kernel(
        tokens,
        relative_node_bias[active_nodes][None],
    ).squeeze(0)
    if causal:
        # Kernel index ``tokens - 1`` is displacement zero; later indices read
        # future writers and therefore vanish in a causal memory.
        causal_mask = torch.arange(
            2 * tokens - 1,
            device=kernel.device,
        ).le(tokens - 1)
        kernel = kernel * causal_mask.to(kernel.dtype)

    first_stop = min(feature_chunk_size, width)
    fft_transform_length = None
    fft_kernel_spectrum = None
    if writer_backend == "triton":
        fft_transform_length, fft_kernel_spectrum = _fft_kernel_spectrum(
            kernel,
            tokens=tokens,
            feature_dtype=hidden.dtype,
        )
        first_features_native = _packed_writer_features_unchecked(
            hidden,
            write,
            compact_ids,
            route_delta,
            start=0,
            stop=first_stop,
            active_count=active_count,
            epsilon=epsilon,
            include_mass=True,
            backend=writer_backend,
        )
        first_convolved = _fft_native_grouped_convolution(
            first_features_native,
            fft_kernel_spectrum,
            transform_length=fft_transform_length,
            original_dtype=hidden.dtype,
        )
    else:
        writer_mass = _writer_mass(
            write,
            compact_ids,
            route_delta,
            lanes=lanes,
            active_count=active_count,
            epsilon=epsilon,
        )
        first_writer_state = _writer_state_chunk(
            hidden,
            write,
            compact_ids,
            route_delta,
            0,
            first_stop,
            active_count=active_count,
            epsilon=epsilon,
        )
        first_features = torch.cat(
            (writer_mass[..., None], first_writer_state),
            dim=-1,
        )
        if convolution_backend == "direct":
            first_convolved = _grouped_convolution(first_features, kernel)
        else:
            fft_transform_length, fft_kernel_spectrum = _fft_kernel_spectrum(
                kernel,
                tokens=tokens,
                feature_dtype=hidden.dtype,
            )
            first_convolved = _fft_grouped_convolution(
                first_features,
                fft_kernel_spectrum,
                transform_length=fft_transform_length,
            )
    denominator = first_convolved[..., 0]
    first_numerator = first_convolved[..., 1:]
    safe_denominator = denominator.clamp_min(epsilon)
    denominator_by_token = safe_denominator.permute(0, 2, 1)
    route_mass = mass * valid.to(mass.dtype)

    context_chunks: list[torch.Tensor] = []
    for start in range(0, width, feature_chunk_size):
        stop = min(start + feature_chunk_size, width)
        if start == 0:
            numerator = first_numerator
        elif writer_backend == "triton":
            if fft_transform_length is None or fft_kernel_spectrum is None:
                raise RuntimeError("FFT convolution plan was not initialized")
            writer_features_native = _packed_writer_features_unchecked(
                hidden,
                write,
                compact_ids,
                route_delta,
                start=start,
                stop=stop,
                active_count=active_count,
                epsilon=epsilon,
                include_mass=False,
                backend=writer_backend,
            )
            numerator = _fft_native_grouped_convolution(
                writer_features_native,
                fft_kernel_spectrum,
                transform_length=fft_transform_length,
                original_dtype=hidden.dtype,
            )
        else:
            writer_state = _writer_state_chunk(
                hidden,
                write,
                compact_ids,
                route_delta,
                start,
                stop,
                active_count=active_count,
                epsilon=epsilon,
            )
            if convolution_backend == "direct":
                numerator = _grouped_convolution(writer_state, kernel)
            else:
                if fft_transform_length is None or fft_kernel_spectrum is None:
                    raise RuntimeError("FFT convolution plan was not initialized")
                numerator = _fft_grouped_convolution(
                    writer_state,
                    fft_kernel_spectrum,
                    transform_length=fft_transform_length,
                )
        numerator_by_token = numerator.permute(0, 2, 1, 3)
        context_chunks.append(
            _reader_context_chunk(
                numerator_by_token,
                denominator_by_token,
                compact_ids,
                route_mass,
                epsilon=epsilon,
                backend=reader_backend,
            )
        )
    context = torch.cat(context_chunks, dim=-1)

    if not return_diagnostics:
        return context
    return context, PackedConvolutionMemoryDiagnostics(
        active_nodes=active_nodes,
        denominator=denominator,
        kernel=kernel,
    )
