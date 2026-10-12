"""Compile-safe opaque boundary for packed FFT receiver memory.

The packed FFT implementation uses real inputs and returns a real receiver
context, but its implementation necessarily creates complex FFT spectra.
TorchInductor currently attempts to lower those internal ``complex64``
tensors when the implementation is inlined into a compiled model.  This
module keeps the *complete* receiver call behind a custom-operator boundary so
only real tensors cross between the compiled graph and eager cuFFT execution.

The backward pass is a second custom operator with an explicit, chunk-local
adjoint.  It invokes the fused packed-reader adjoint directly, applies the
FFT correlation adjoint with conjugated spectra, and folds writer gradients
back into the packed routes without constructing an autograd replay.  This
keeps both complex spectra and all backward implementation details outside
the AOTAutograd/Inductor graph.

Integration belongs at the Convergent relay dispatch site: select
``opaque_packed_fft_node_context`` instead of
``packed_convolution_node_context(..., convolution_backend="fft")``.  This
module imports the existing implementation one way and therefore introduces
no circular dependency.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from gator.models.memory import _relative_convolution_kernel
from gator.models.packed_convolution_memory import (
    _fft_grouped_convolution,
    _fft_grouped_convolution_with_source_spectrum,
    _fft_kernel_spectrum,
    _fft_native_grouped_convolution,
    _fft_native_grouped_convolution_with_source_spectrum,
    _route_deltas,
    _validate_inputs,
    _writer_mass,
    _writer_state_chunk,
    packed_convolution_node_context,
)
from gator.models.triton_packed_writer import (
    PackedWriterBackend,
    _packed_writer_adjoint_unchecked,
    _packed_writer_features_unchecked,
    triton_packed_writer_available,
)
from gator.models.triton_packed_memory import (
    PackedReaderBackend,
    _packed_reader_backward_unchecked,
    triton_packed_reader_available,
)


def _eager_fft_receiver(
    hidden: torch.Tensor,
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    write: torch.Tensor,
    relative_node_bias: torch.Tensor,
    causal: bool,
    epsilon: float,
    feature_chunk_size: int,
    compact_active_nodes: bool,
    assume_unique_routes: bool,
    use_triton_reader: bool,
    use_triton_writer: bool,
) -> torch.Tensor:
    """Call the complete eager FFT receiver implementation."""

    result = packed_convolution_node_context(
        hidden,
        nodes,
        mass,
        valid,
        write,
        relative_node_bias,
        causal=causal,
        epsilon=epsilon,
        feature_chunk_size=feature_chunk_size,
        compact_active_nodes=compact_active_nodes,
        convolution_backend="fft",
        reader_backend=("triton" if use_triton_reader else "torch"),
        writer_backend=("triton" if use_triton_writer else "torch"),
        assume_unique_routes=assume_unique_routes,
        return_diagnostics=False,
    )
    if not isinstance(result, torch.Tensor):
        raise RuntimeError("packed FFT receiver unexpectedly returned diagnostics")
    return result


@torch.library.custom_op(
    "gator::opaque_packed_fft_receiver",
    mutates_args=(),
)
def _opaque_packed_fft_receiver(
    hidden: torch.Tensor,
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    write: torch.Tensor,
    relative_node_bias: torch.Tensor,
    causal: bool,
    epsilon: float,
    feature_chunk_size: int,
    compact_active_nodes: bool,
    assume_unique_routes: bool,
    use_triton_reader: bool,
    use_triton_writer: bool,
) -> torch.Tensor:
    return _eager_fft_receiver(
        hidden,
        nodes,
        mass,
        valid,
        write,
        relative_node_bias,
        causal,
        epsilon,
        feature_chunk_size,
        compact_active_nodes,
        assume_unique_routes,
        use_triton_reader,
        use_triton_writer,
    )


@_opaque_packed_fft_receiver.register_fake
def _opaque_packed_fft_receiver_fake(
    hidden: torch.Tensor,
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    write: torch.Tensor,
    relative_node_bias: torch.Tensor,
    causal: bool,
    epsilon: float,
    feature_chunk_size: int,
    compact_active_nodes: bool,
    assume_unique_routes: bool,
    use_triton_reader: bool,
    use_triton_writer: bool,
) -> torch.Tensor:
    del (
        nodes,
        mass,
        valid,
        write,
        relative_node_bias,
        causal,
        epsilon,
        feature_chunk_size,
        compact_active_nodes,
        assume_unique_routes,
        use_triton_reader,
        use_triton_writer,
    )
    # The real receiver concatenates feature chunks and therefore returns a
    # fresh contiguous tensor even when ``hidden`` is a noncontiguous view.
    # FakeTensor stride metadata must describe that actual layout or Inductor
    # may compile downstream indexing against the wrong contract.
    return torch.empty_like(hidden, memory_format=torch.contiguous_format)


def _fft_grouped_convolution_adjoint(
    features: torch.Tensor,
    kernel_spectrum: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    transform_length: int,
    source_spectrum: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the feature adjoint and an additive kernel-spectrum adjoint.

    ``_fft_grouped_convolution`` is an ordinary real linear convolution with
    a bounded output crop.  Its adjoint is therefore a circular correlation:
    embed the output cotangent at the crop, multiply by conjugated spectra,
    and transform back.  The second result remains in frequency space so all
    feature chunks can contribute before one final inverse FFT.
    """

    native_features = features.to(
        torch.float32
        if features.dtype in (torch.float16, torch.bfloat16)
        else features.dtype
    ).permute(0, 3, 1, 2)
    native_gradient, kernel_spectrum_adjoint = (
        _fft_native_grouped_convolution_adjoint(
            native_features,
            kernel_spectrum,
            grad_output,
            transform_length=transform_length,
            original_dtype=features.dtype,
            source_spectrum=source_spectrum,
        )
    )
    return (
        native_gradient.permute(0, 2, 3, 1).contiguous(),
        kernel_spectrum_adjoint,
    )


def _fft_native_grouped_convolution_adjoint(
    features: torch.Tensor,
    kernel_spectrum: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    transform_length: int,
    original_dtype: torch.dtype | None = None,
    source_spectrum: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the FFT adjoint to native ``[B,C,U,T]`` writer features."""

    if features.ndim != 4:
        raise ValueError("FFT-native features must have [B,C,U,T] shape")
    batch, feature_width, active_count, tokens = features.shape
    if grad_output.shape != (batch, active_count, tokens, feature_width):
        raise ValueError("FFT convolution gradient must match its output")
    if original_dtype is None:
        original_dtype = features.dtype
    compute_dtype = (
        torch.float32
        if original_dtype in (torch.float16, torch.bfloat16)
        else original_dtype
    )
    if source_spectrum is None:
        source = features.to(compute_dtype)
        source_spectrum = torch.fft.rfft(
            source,
            n=transform_length,
            dim=-1,
        )
    else:
        expected_spectrum_shape = (
            batch,
            feature_width,
            active_count,
            transform_length // 2 + 1,
        )
        if source_spectrum.shape != expected_spectrum_shape:
            raise ValueError("source spectrum does not match FFT features")
        if source_spectrum.device != features.device:
            raise ValueError("source spectrum must share the feature device")
    selected_gradient = grad_output.to(compute_dtype).permute(0, 3, 1, 2)
    crop_start = tokens - 1
    crop_stop = crop_start + tokens
    full_gradient = F.pad(
        selected_gradient,
        (crop_start, transform_length - crop_stop),
    )
    gradient_spectrum = torch.fft.rfft(
        full_gradient,
        n=transform_length,
        dim=-1,
    )
    feature_gradient = torch.fft.irfft(
        gradient_spectrum * kernel_spectrum.conj()[None, None],
        n=transform_length,
        dim=-1,
    )[..., :tokens]
    kernel_spectrum_adjoint = (
        source_spectrum.conj() * gradient_spectrum
    ).sum(dim=(0, 1))
    return feature_gradient.to(original_dtype).contiguous(), kernel_spectrum_adjoint


def _gather_compact_routes(
    values: torch.Tensor,
    compact_ids: torch.Tensor,
) -> torch.Tensor:
    """Gather ``[B,T,U,C]`` values into bounded ``[B,T,L,K,C]`` routes."""

    batch, tokens, _, features = values.shape
    lanes, routes = compact_ids.shape[2:]
    flat_ids = compact_ids.flatten(2, 3)
    selected = values.gather(
        2,
        flat_ids[..., None].expand(-1, -1, -1, features),
    )
    return selected.reshape(batch, tokens, lanes, routes, features)


def _writer_state_adjoint(
    hidden_chunk: torch.Tensor,
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    grad_writer_state: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the bounded scatter-writer adjoint for one feature chunk."""

    _, _, lanes, _ = hidden_chunk.shape
    gradient_by_token = grad_writer_state.permute(0, 2, 1, 3)
    gradient_sum = gradient_by_token.sum(dim=2)
    selected_gradient = _gather_compact_routes(
        gradient_by_token,
        compact_ids,
    )
    route_weight = route_delta * write[:, :, None, None]
    grad_hidden = (
        epsilon
        * write[:, :, None, None]
        * gradient_sum[:, :, None, :]
    ).expand(-1, -1, lanes, -1)
    grad_hidden = grad_hidden + (
        route_weight[..., None] * selected_gradient
    ).sum(dim=3)
    route_score = (
        selected_gradient * hidden_chunk.unsqueeze(3)
    ).sum(dim=-1)
    grad_route_delta = write[:, :, None, None] * route_score
    grad_write = epsilon * (
        gradient_sum * hidden_chunk.sum(dim=2)
    ).sum(dim=-1)
    grad_write = grad_write + (route_delta * route_score).sum(dim=(2, 3))
    return grad_hidden, grad_route_delta, grad_write


def _writer_mass_adjoint(
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    grad_writer_mass: torch.Tensor,
    *,
    lanes: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the scalar writer-mass scatter adjoint."""

    gradient_by_token = grad_writer_mass.permute(0, 2, 1)
    flat_ids = compact_ids.flatten(2, 3)
    selected_gradient = gradient_by_token.gather(2, flat_ids).reshape_as(
        compact_ids
    )
    grad_route_delta = write[:, :, None, None] * selected_gradient
    grad_write = lanes * epsilon * gradient_by_token.sum(dim=2)
    grad_write = grad_write + (
        route_delta * selected_gradient
    ).sum(dim=(2, 3))
    return grad_route_delta, grad_write


def _route_delta_adjoint(
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    grad_route_delta: torch.Tensor,
    *,
    epsilon: float,
    assume_unique_routes: bool,
) -> torch.Tensor:
    """Map post-floor writer-delta gradients back to packed route mass."""

    valid_mass = mass * valid.to(mass.dtype)
    if assume_unique_routes:
        return (
            grad_route_delta
            * valid.to(mass.dtype)
            * mass.ge(epsilon).to(mass.dtype)
        )
    safe_nodes = torch.where(valid, nodes, torch.zeros_like(nodes))
    grad_mass = torch.zeros_like(mass, memory_format=torch.contiguous_format)
    for route in range(nodes.shape[-1]):
        route_node = safe_nodes[..., route]
        route_valid = valid[..., route]
        same_node = safe_nodes.eq(route_node.unsqueeze(-1)) & valid
        merged_lane_mass = (
            valid_mass * same_node.to(mass.dtype)
        ).sum(dim=-1)
        if route:
            already_seen = (
                safe_nodes[..., :route].eq(route_node.unsqueeze(-1))
                & valid[..., :route]
            ).any(dim=-1)
        else:
            already_seen = torch.zeros_like(route_valid)
        representative = route_valid & ~already_seen
        representative_gradient = (
            grad_route_delta[..., route]
            * representative.to(mass.dtype)
            * merged_lane_mass.ge(epsilon).to(mass.dtype)
        )
        grad_mass = grad_mass + (
            representative_gradient.unsqueeze(-1)
            * same_node.to(mass.dtype)
        )
    return grad_mass


def _relative_kernel_bias_adjoint(
    kernel: torch.Tensor,
    grad_kernel: torch.Tensor,
) -> torch.Tensor:
    """Differentiate the detached-max relative convolution kernel."""

    tokens = (kernel.shape[-1] + 1) // 2
    radius = tokens - 1
    displacement = torch.arange(
        radius,
        -radius - 1,
        -1,
        device=kernel.device,
        dtype=kernel.dtype,
    )
    normalized = displacement / max(radius, 1)
    code = torch.stack(
        (
            normalized,
            normalized.abs(),
            torch.sin(math.pi * normalized),
            torch.cos(math.pi * normalized),
        ),
        dim=-1,
    )
    return torch.einsum("up,pr->ur", grad_kernel * kernel, code)


@torch.library.custom_op(
    "gator::opaque_packed_fft_receiver_backward",
    mutates_args=(),
)
def _opaque_packed_fft_receiver_backward(
    hidden: torch.Tensor,
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    write: torch.Tensor,
    relative_node_bias: torch.Tensor,
    grad_output: torch.Tensor,
    causal: bool,
    epsilon: float,
    feature_chunk_size: int,
    compact_active_nodes: bool,
    assume_unique_routes: bool,
    use_triton_reader: bool,
    use_triton_writer: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, tokens, lanes, width, _, n_nodes = _validate_inputs(
        hidden,
        nodes,
        mass,
        valid,
        write,
        relative_node_bias,
        epsilon=epsilon,
        feature_chunk_size=feature_chunk_size,
    )
    if grad_output.shape != hidden.shape:
        raise ValueError("grad_output must match the receiver context")
    if grad_output.dtype != hidden.dtype or grad_output.device != hidden.device:
        raise ValueError("grad_output must share hidden dtype and device")

    positive = valid & mass.gt(0)
    if compact_active_nodes:
        active_nodes = torch.unique(nodes[positive], sorted=True)
    else:
        active_nodes = torch.arange(n_nodes, device=nodes.device)
    if active_nodes.numel() == 0:
        return (
            torch.zeros_like(hidden, memory_format=torch.contiguous_format),
            torch.zeros_like(mass, memory_format=torch.contiguous_format),
            torch.zeros_like(write, memory_format=torch.contiguous_format),
            torch.zeros_like(
                relative_node_bias,
                memory_format=torch.contiguous_format,
            ),
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
        causal_mask = torch.arange(
            2 * tokens - 1,
            device=kernel.device,
        ).le(tokens - 1)
        kernel = kernel * causal_mask.to(kernel.dtype)
    transform_length, kernel_spectrum = _fft_kernel_spectrum(
        kernel,
        tokens=tokens,
        feature_dtype=hidden.dtype,
    )

    first_stop = min(feature_chunk_size, width)
    writer_backend: PackedWriterBackend = (
        "triton" if use_triton_writer else "torch"
    )
    first_writer_state: torch.Tensor | None = None
    first_features: torch.Tensor | None = None
    first_features_native: torch.Tensor | None = None
    if use_triton_writer:
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
            kernel_spectrum,
            transform_length=transform_length,
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
        denominator_features = writer_mass[..., None]
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
            (denominator_features, first_writer_state),
            dim=-1,
        )
        first_convolved = _fft_grouped_convolution(
            first_features,
            kernel_spectrum,
            transform_length=transform_length,
        )
    denominator = first_convolved[..., 0]
    first_numerator = first_convolved[..., 1:]
    safe_denominator_by_token = denominator.clamp_min(epsilon).permute(0, 2, 1)
    route_mass = mass * valid.to(mass.dtype)

    grad_hidden = torch.zeros_like(
        hidden, memory_format=torch.contiguous_format
    )
    grad_write = torch.zeros_like(
        write, memory_format=torch.contiguous_format
    )
    grad_route_delta = torch.zeros_like(
        route_delta, memory_format=torch.contiguous_format
    )
    grad_route_mass = torch.zeros_like(
        route_mass, memory_format=torch.contiguous_format
    )
    grad_safe_denominator = torch.zeros_like(
        safe_denominator_by_token,
        memory_format=torch.contiguous_format,
    )
    kernel_spectrum_adjoint = torch.zeros_like(
        kernel_spectrum,
        memory_format=torch.contiguous_format,
    )
    reader_backend: PackedReaderBackend = (
        "triton" if use_triton_reader else "torch"
    )
    first_grad_numerator: torch.Tensor | None = None

    # Recompute, consume, and release one feature chunk at a time.  The native
    # reader adjoint never expands active nodes, and no autograd graph spans
    # chunks, so peak memory is independent of total hidden width.
    for start in range(0, width, feature_chunk_size):
        stop = min(start + feature_chunk_size, width)
        writer_state: torch.Tensor | None = None
        writer_features_native: torch.Tensor | None = None
        writer_state_spectrum: torch.Tensor | None = None
        if start == 0:
            numerator = first_numerator
        elif use_triton_writer:
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
            (
                numerator,
                writer_state_spectrum,
            ) = _fft_native_grouped_convolution_with_source_spectrum(
                writer_features_native,
                kernel_spectrum,
                transform_length=transform_length,
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
            (
                numerator,
                writer_state_spectrum,
            ) = _fft_grouped_convolution_with_source_spectrum(
                writer_state,
                kernel_spectrum,
                transform_length=transform_length,
            )
        numerator_by_token = numerator.permute(0, 2, 1, 3)
        (
            grad_numerator_by_token,
            chunk_grad_denominator,
            chunk_grad_route_mass,
        ) = _packed_reader_backward_unchecked(
            grad_output[..., start:stop],
            numerator_by_token,
            safe_denominator_by_token,
            compact_ids,
            route_mass,
            epsilon=epsilon,
            backend=reader_backend,
        )
        grad_numerator = grad_numerator_by_token.permute(0, 2, 1, 3)
        grad_safe_denominator = (
            grad_safe_denominator + chunk_grad_denominator
        )
        grad_route_mass = grad_route_mass + chunk_grad_route_mass
        if start == 0:
            # Forward fuses denominator and first-state convolution into one
            # FFT batch. Defer this chunk until every denominator contribution
            # is known so backward can mirror that exact batching contract.
            first_grad_numerator = grad_numerator
            continue
        if use_triton_writer:
            if writer_features_native is None:
                raise RuntimeError("native writer features were not initialized")
            (
                grad_writer_features_native,
                chunk_kernel_adjoint,
            ) = _fft_native_grouped_convolution_adjoint(
                writer_features_native,
                kernel_spectrum,
                grad_numerator,
                transform_length=transform_length,
                source_spectrum=writer_state_spectrum,
                original_dtype=hidden.dtype,
            )
            (
                chunk_grad_hidden,
                chunk_grad_write,
                chunk_grad_route_delta,
            ) = _packed_writer_adjoint_unchecked(
                hidden,
                write,
                compact_ids,
                route_delta,
                grad_writer_features_native,
                start=start,
                stop=stop,
                active_count=active_count,
                epsilon=epsilon,
                include_mass=False,
                backend=writer_backend,
            )
        else:
            if writer_state is None:
                raise RuntimeError("Torch writer state was not initialized")
            grad_writer_state, chunk_kernel_adjoint = (
                _fft_grouped_convolution_adjoint(
                    writer_state,
                    kernel_spectrum,
                    grad_numerator,
                    transform_length=transform_length,
                    source_spectrum=writer_state_spectrum,
                )
            )
            (
                chunk_grad_hidden,
                chunk_grad_route_delta,
                chunk_grad_write,
            ) = _writer_state_adjoint(
                hidden[..., start:stop],
                write,
                compact_ids,
                route_delta,
                grad_writer_state,
                epsilon=epsilon,
            )
        grad_hidden[..., start:stop] = chunk_grad_hidden
        grad_route_delta = grad_route_delta + chunk_grad_route_delta
        grad_write = grad_write + chunk_grad_write
        kernel_spectrum_adjoint = (
            kernel_spectrum_adjoint + chunk_kernel_adjoint
        )

    if first_grad_numerator is None:
        raise RuntimeError("first FFT feature chunk was not processed")
    grad_denominator = (
        grad_safe_denominator.permute(0, 2, 1)
        * denominator.ge(epsilon).to(denominator.dtype)
    )
    first_grad_convolved = torch.cat(
        (grad_denominator[..., None], first_grad_numerator),
        dim=-1,
    )
    if use_triton_writer:
        if first_features_native is None:
            raise RuntimeError("native first writer features were not initialized")
        grad_first_features_native, first_kernel_adjoint = (
            _fft_native_grouped_convolution_adjoint(
                first_features_native,
                kernel_spectrum,
                first_grad_convolved,
                transform_length=transform_length,
                original_dtype=hidden.dtype,
            )
        )
        (
            first_grad_hidden,
            first_grad_write,
            first_grad_route_delta,
        ) = _packed_writer_adjoint_unchecked(
            hidden,
            write,
            compact_ids,
            route_delta,
            grad_first_features_native,
            start=0,
            stop=first_stop,
            active_count=active_count,
            epsilon=epsilon,
            include_mass=True,
            backend=writer_backend,
        )
    else:
        if first_features is None or first_writer_state is None:
            raise RuntimeError("Torch first writer features were not initialized")
        grad_first_features, first_kernel_adjoint = (
            _fft_grouped_convolution_adjoint(
                first_features,
                kernel_spectrum,
                first_grad_convolved,
                transform_length=transform_length,
            )
        )
        grad_writer_mass = grad_first_features[..., 0]
        grad_first_writer_state = grad_first_features[..., 1:]
        (
            first_grad_hidden,
            first_grad_route_delta,
            first_grad_write,
        ) = _writer_state_adjoint(
            hidden[..., :first_stop],
            write,
            compact_ids,
            route_delta,
            grad_first_writer_state,
            epsilon=epsilon,
        )
        denominator_grad_route_delta, denominator_grad_write = (
            _writer_mass_adjoint(
                write,
                compact_ids,
                route_delta,
                grad_writer_mass,
                lanes=lanes,
                epsilon=epsilon,
            )
        )
        first_grad_route_delta = (
            first_grad_route_delta + denominator_grad_route_delta
        )
        first_grad_write = first_grad_write + denominator_grad_write
    grad_hidden[..., :first_stop] = first_grad_hidden
    grad_route_delta = grad_route_delta + first_grad_route_delta
    grad_write = grad_write + first_grad_write
    kernel_spectrum_adjoint = kernel_spectrum_adjoint + first_kernel_adjoint

    grad_mass = grad_route_mass * valid.to(mass.dtype)
    grad_mass = grad_mass + _route_delta_adjoint(
        nodes,
        mass,
        valid,
        grad_route_delta,
        epsilon=epsilon,
        assume_unique_routes=assume_unique_routes,
    )

    kernel_width = 2 * tokens - 1
    grad_flipped_kernel = torch.fft.irfft(
        kernel_spectrum_adjoint,
        n=transform_length,
        dim=-1,
    )[..., :kernel_width]
    grad_kernel = grad_flipped_kernel.flip(-1).to(kernel.dtype)
    grad_active_bias = _relative_kernel_bias_adjoint(kernel, grad_kernel)
    grad_bias = torch.zeros_like(
        relative_node_bias,
        memory_format=torch.contiguous_format,
    )
    grad_bias.index_add_(0, active_nodes, grad_active_bias)
    return (
        grad_hidden.contiguous(),
        grad_mass.contiguous(),
        grad_write.contiguous(),
        grad_bias.contiguous(),
    )


@_opaque_packed_fft_receiver_backward.register_fake
def _opaque_packed_fft_receiver_backward_fake(
    hidden: torch.Tensor,
    nodes: torch.Tensor,
    mass: torch.Tensor,
    valid: torch.Tensor,
    write: torch.Tensor,
    relative_node_bias: torch.Tensor,
    grad_output: torch.Tensor,
    causal: bool,
    epsilon: float,
    feature_chunk_size: int,
    compact_active_nodes: bool,
    assume_unique_routes: bool,
    use_triton_reader: bool,
    use_triton_writer: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        nodes,
        valid,
        grad_output,
        causal,
        epsilon,
        feature_chunk_size,
        compact_active_nodes,
        assume_unique_routes,
        use_triton_reader,
        use_triton_writer,
    )
    return (
        torch.empty_like(hidden, memory_format=torch.contiguous_format),
        torch.empty_like(mass, memory_format=torch.contiguous_format),
        torch.empty_like(write, memory_format=torch.contiguous_format),
        torch.empty_like(
            relative_node_bias,
            memory_format=torch.contiguous_format,
        ),
    )


def _setup_opaque_fft_context(
    ctx: object,
    inputs: tuple[object, ...],
    output: torch.Tensor,
) -> None:
    del output
    (
        hidden,
        nodes,
        mass,
        valid,
        write,
        relative_node_bias,
        causal,
        epsilon,
        feature_chunk_size,
        compact_active_nodes,
        assume_unique_routes,
        use_triton_reader,
        use_triton_writer,
    ) = inputs
    ctx.save_for_backward(
        hidden,
        nodes,
        mass,
        valid,
        write,
        relative_node_bias,
    )
    ctx.causal = causal
    ctx.epsilon = epsilon
    ctx.feature_chunk_size = feature_chunk_size
    ctx.compact_active_nodes = compact_active_nodes
    ctx.assume_unique_routes = assume_unique_routes
    ctx.use_triton_reader = use_triton_reader
    ctx.use_triton_writer = use_triton_writer


def _opaque_fft_autograd_backward(
    context: object,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    (
        hidden,
        nodes,
        mass,
        valid,
        write,
        relative_node_bias,
    ) = context.saved_tensors
    grad_hidden, grad_mass, grad_write, grad_bias = (
        _opaque_packed_fft_receiver_backward(
            hidden,
            nodes,
            mass,
            valid,
            write,
            relative_node_bias,
            grad_output,
            context.causal,
            context.epsilon,
            context.feature_chunk_size,
            context.compact_active_nodes,
            context.assume_unique_routes,
            context.use_triton_reader,
            context.use_triton_writer,
        )
    )
    return (
        grad_hidden,
        None,
        grad_mass,
        None,
        grad_write,
        grad_bias,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_opaque_packed_fft_receiver.register_autograd(
    _opaque_fft_autograd_backward,
    setup_context=_setup_opaque_fft_context,
)


def opaque_packed_fft_node_context(
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
    assume_unique_routes: bool = False,
    reader_backend: PackedReaderBackend = "torch",
    writer_backend: PackedWriterBackend = "torch",
) -> torch.Tensor:
    """Return packed FFT receiver context through an opaque custom op.

    Every tensor crossing the custom-op boundary is real.  The public shape,
    value, and first-order gradient contract matches
    :func:`packed_convolution_node_context` with
    ``convolution_backend="fft"``.
    """

    if reader_backend not in ("torch", "triton"):
        raise ValueError("reader_backend must be 'torch' or 'triton'")
    if reader_backend == "triton":
        # Reject unsupported launches before entering the opaque custom-op
        # boundary. Otherwise the user sees a nested dispatcher failure and
        # cannot tell whether execution silently changed backends.
        if hidden.device.type != "cuda":
            raise ValueError("reader_backend='triton' requires CUDA tensors")
        if hidden.dtype != torch.float32:
            raise TypeError(
                "reader_backend='triton' currently supports float32 only"
            )
        if not triton_packed_reader_available():
            raise RuntimeError(
                "reader_backend='triton' requires Triton plus "
                "torch.library.triton_op/wrap_triton support"
            )
    if writer_backend not in ("torch", "triton"):
        raise ValueError("writer_backend must be 'torch' or 'triton'")
    if writer_backend == "triton":
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
    return _opaque_packed_fft_receiver(
        hidden,
        nodes,
        mass,
        valid,
        write,
        relative_node_bias,
        causal,
        float(epsilon),
        feature_chunk_size,
        compact_active_nodes,
        assume_unique_routes,
        reader_backend == "triton",
        writer_backend == "triton",
    )


__all__ = ["opaque_packed_fft_node_context"]
