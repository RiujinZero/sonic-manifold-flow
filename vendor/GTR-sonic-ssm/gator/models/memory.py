"""Receiver-specific graph-memory kernels.

The Arrow relative read is translation invariant: its learned receiver/writer
bias depends on ``receiver_position - writer_position``.  The dense reference
expresses that read as an explicit ``tokens x tokens`` softmax.  The
convolution backend evaluates the same normalized kernel without materializing
that quadratic attention tensor.
"""

from __future__ import annotations

import math
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F

from .stewardship import normalise_stewardship_messages


ReceiverMemoryBackend = Literal["attention", "convolution", "linear_scan"]

# Receiver denominators include per-node occupancy as well as route mass,
# so legitimate channels can be smaller than the stewardship/group floor.
# This still sits 34 orders of magnitude above the BF16 failing case.
RECEIVER_NORMALISATION_MASS_FLOOR = 1e-6

# A 128-token slab keeps the largest production receiver temporary near
# 24 MiB in FP32 (B=1, C*N=32, D=1504), small enough for this CPU's cache/RAM
# hierarchy while still giving the folded tensor kernels useful work.
LINEAR_SCAN_INFERENCE_SLAB_TOKENS = 128


class LinearScanReceiverCache(NamedTuple):
    """Constant-size causal carry for one GTR-ARW relay receiver."""

    forward_state: torch.Tensor


def detach_linear_scan_receiver_cache(
    cache: LinearScanReceiverCache,
) -> LinearScanReceiverCache:
    return LinearScanReceiverCache(cache.forward_state.detach())


def relative_position_code(
    length: int,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return ``[receiver, writer, feature]`` relative-position codes."""

    if length <= 0:
        raise ValueError("length must be positive")
    positions = torch.arange(length, device=device, dtype=dtype)
    delta = positions[:, None] - positions[None]
    delta = delta / max(length - 1, 1)
    return torch.stack(
        [
            delta,
            delta.abs(),
            torch.sin(math.pi * delta),
            torch.cos(math.pi * delta),
        ],
        -1,
    )


def _validate_receiver_inputs(
    states: torch.Tensor,
    node_locations: torch.Tensor,
    writer_mass: torch.Tensor,
    relative_node_bias: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if states.ndim != 4:
        raise ValueError("states must have [batch, cells, tokens, hidden] shape")
    if node_locations.ndim != 4:
        raise ValueError(
            "node_locations must have [batch, cells, tokens, nodes] shape"
        )
    if writer_mass.ndim != 3:
        raise ValueError("writer_mass must have [batch, cells, tokens] shape")
    batch, cells, tokens, hidden = states.shape
    if node_locations.shape[:3] != (batch, cells, tokens):
        raise ValueError("node-location prefix must match states")
    if writer_mass.shape != (batch, cells, tokens):
        raise ValueError("writer_mass must match the state prefix")
    nodes = node_locations.shape[-1]
    if relative_node_bias.shape != (cells, nodes, 4):
        raise ValueError(
            "relative_node_bias must have [cells, nodes, 4] shape"
        )
    if states.device != node_locations.device or states.device != writer_mass.device:
        raise ValueError("states, node locations, and writer mass must share a device")
    if relative_node_bias.device != states.device:
        raise ValueError("relative_node_bias must share the state device")
    if (
        states.dtype != node_locations.dtype
        or states.dtype != writer_mass.dtype
        or states.dtype != relative_node_bias.dtype
    ):
        raise ValueError(
            "states, node locations, writer mass, and relative bias must share a dtype"
        )
    return batch, cells, tokens, hidden, nodes


def attention_receiver_context(
    states: torch.Tensor,
    node_locations: torch.Tensor,
    writer_mass: torch.Tensor,
    relative_node_bias: torch.Tensor,
) -> torch.Tensor:
    """Quadratic semantic reference for receiver-specific node memory."""

    _, _, tokens, _, _ = _validate_receiver_inputs(
        states, node_locations, writer_mass, relative_node_bias
    )
    relative = relative_position_code(
        tokens, states.device, dtype=states.dtype
    )
    relative_bias = torch.einsum(
        "ijr,cnr->cijn", relative, relative_node_bias
    )
    scores = (
        node_locations[:, :, None].clamp_min(1e-9).log()
        + relative_bias[None]
        + writer_mass[:, :, None, :, None].clamp_min(1e-9).log()
    )
    writer_valid = writer_mass.gt(0)[:, :, None, :, None]
    scores = scores.masked_fill(~writer_valid, -torch.inf)
    has_writer = writer_valid.any(3, keepdim=True)
    safe_scores = torch.where(has_writer, scores, torch.zeros_like(scores))
    attention = safe_scores.softmax(3) * writer_valid.to(scores.dtype)
    attention = attention * has_writer.to(scores.dtype)
    node_memory = torch.einsum("bcijn,bcjd->bcind", attention, states)
    return (node_locations.unsqueeze(-1) * node_memory).sum(3)


def _relative_convolution_kernel(
    length: int, relative_node_bias: torch.Tensor
) -> torch.Tensor:
    """Return grouped-correlation kernels ordered for ``conv1d``."""

    radius = length - 1
    # ``conv1d`` cross-correlation with padding ``radius`` reads writer
    # ``j = i + kernel_index - radius``.  Therefore the required displacement
    # ``i - j`` runs from ``+radius`` to ``-radius``.
    displacement = torch.arange(
        radius,
        -radius - 1,
        -1,
        device=relative_node_bias.device,
        dtype=relative_node_bias.dtype,
    )
    normalized = displacement / max(radius, 1)
    code = torch.stack(
        [
            normalized,
            normalized.abs(),
            torch.sin(math.pi * normalized),
            torch.cos(math.pi * normalized),
        ],
        -1,
    )
    logits = torch.einsum("kr,cnr->cnk", code, relative_node_bias)
    # A constant scale per cell/node cancels in numerator / denominator.  This
    # keeps exponentiation stable without changing the normalized memory.
    logits = logits - logits.amax(-1, keepdim=True).detach()
    return logits.exp()


def convolution_receiver_context(
    states: torch.Tensor,
    node_locations: torch.Tensor,
    writer_mass: torch.Tensor,
    relative_node_bias: torch.Tensor,
) -> torch.Tensor:
    """Exact normalized relative read using grouped one-dimensional convolution.

    The operation has the same learned equation as
    :func:`attention_receiver_context`, but it never creates a
    ``[batch, cells, receiver, writer, nodes]`` score tensor.
    """

    batch, cells, tokens, hidden, nodes = _validate_receiver_inputs(
        states, node_locations, writer_mass, relative_node_bias
    )
    # Match the reference mask's value *and gradient* contract: an inactive
    # writer contributes neither probability mass nor a gradient through its
    # stored zero.
    effective_writer_mass = torch.where(
        writer_mass.gt(0), writer_mass, torch.zeros_like(writer_mass)
    )
    writer_node_mass = (
        node_locations.clamp_min(1e-9)
        * effective_writer_mass.unsqueeze(-1)
    )
    denominator_input = writer_node_mass.permute(0, 1, 3, 2)
    numerator_input = (
        writer_node_mass.unsqueeze(-1) * states.unsqueeze(3)
    ).permute(0, 4, 1, 3, 2)
    features = torch.cat(
        [denominator_input[:, None], numerator_input],
        1,
    ).reshape(batch * (hidden + 1), cells * nodes, tokens)

    kernel = _relative_convolution_kernel(
        tokens, relative_node_bias
    ).reshape(cells * nodes, 1, 2 * tokens - 1)
    convolved = F.conv1d(
        features,
        kernel,
        padding=tokens - 1,
        groups=cells * nodes,
    ).reshape(batch, hidden + 1, cells, nodes, tokens)

    denominator = convolved[:, 0].permute(0, 1, 3, 2)
    numerator = convolved[:, 1:].permute(0, 2, 4, 3, 1)
    node_memory = normalise_stewardship_messages(
        numerator,
        denominator,
        minimum_mass=RECEIVER_NORMALISATION_MASS_FLOOR,
    )
    return (node_locations.unsqueeze(-1) * node_memory).sum(3)


def _streaming_block_receiver_context(
    states: torch.Tensor,
    node_locations: torch.Tensor,
    writer_mass: torch.Tensor,
    relative_node_bias: torch.Tensor,
    *,
    chunk_size: int,
    block_size: int,
    cache: LinearScanReceiverCache | None,
    return_cache: bool,
) -> torch.Tensor | tuple[torch.Tensor, LinearScanReceiverCache]:
    """Memory-bounded inference path for block-causal receiver memory.

    The historical implementation constructs and retains complete
    ``[batch, hidden + 1, tokens, cells * nodes]`` value, forward, backward,
    and mixed-scan tensors.  At the production 4K shape each is roughly
    790 MB (753 MiB) in FP32.  This path evaluates aligned token slabs and
    immediately normalises/reduces each one, leaving only the output and
    constant-size forward carry live across slabs.

    Slab boundaries are aligned to both ``chunk_size`` and ``block_size``.
    Consequently every :func:`decayed_scan` sees the same semantic scan and
    block boundaries as the materialised implementation; this is a scheduling
    and lifetime change, not a model approximation.
    """

    from gator_linear.core import decayed_scan

    batch, cells, tokens, hidden, nodes = _validate_receiver_inputs(
        states, node_locations, writer_mass, relative_node_bias
    )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if tokens % block_size:
        raise ValueError("receiver token length must be divisible by block_size")

    channels = cells * nodes
    rows = batch * (hidden + 1)
    if cache is None:
        carry = states.new_zeros(rows, channels)
    else:
        if cache.forward_state.shape != (rows, channels):
            raise ValueError("receiver cache forward_state has the wrong shape")
        if cache.forward_state.device != states.device:
            raise ValueError("receiver cache and values must share a device")
        if cache.forward_state.dtype != states.dtype:
            raise ValueError("receiver cache and values must share a dtype")
        carry = cache.forward_state

    decay = (
        torch.sigmoid(relative_node_bias[..., 0])
        .clamp(0.01, 0.99)
        .reshape(channels)
    )
    mix = torch.sigmoid(relative_node_bias[..., 1]).reshape(channels)

    # The least common multiple makes a new slab begin at the same scan-chunk
    # and semantic-block boundary it occupied in the full materialisation.
    alignment = math.lcm(chunk_size, block_size)
    slab_tokens = max(
        alignment,
        (LINEAR_SCAN_INFERENCE_SLAB_TOKENS // alignment) * alignment,
    )
    context = torch.empty_like(states)
    for start in range(0, tokens, slab_tokens):
        stop = min(start + slab_tokens, tokens)
        span = stop - start
        slab_states = states[:, :, start:stop]
        slab_locations = node_locations[:, :, start:stop]
        slab_writer_mass = writer_mass[:, :, start:stop]
        effective_writer_mass = torch.where(
            slab_writer_mass.gt(0),
            slab_writer_mass,
            torch.zeros_like(slab_writer_mass),
        )
        writer_node_mass = (
            slab_locations.clamp_min(1e-9)
            * effective_writer_mass.unsqueeze(-1)
        )
        numerator_input = (
            writer_node_mass.unsqueeze(-1) * slab_states.unsqueeze(3)
        )
        combined = torch.cat(
            [writer_node_mass.unsqueeze(-1), numerator_input], dim=-1
        )
        values = combined.permute(0, 4, 2, 1, 3).reshape(
            rows, span, channels
        )

        forward, carry = decayed_scan(
            values,
            decay,
            chunk_size=chunk_size,
            initial_state=carry,
            return_final_state=True,
        )
        blocks = span // block_size
        block_values = values.reshape(rows, blocks, block_size, channels)
        reversed_blocks = block_values.flip(2).reshape(
            rows * blocks, block_size, channels
        )
        backward = decayed_scan(
            reversed_blocks,
            decay,
            chunk_size=chunk_size,
        ).reshape(rows, blocks, block_size, channels).flip(2).reshape(
            rows, span, channels
        )

        scanned = (
            mix[None, None, :] * forward
            + (1.0 - mix[None, None, :]) * backward
        ).reshape(batch, hidden + 1, span, cells, nodes).permute(
            0, 3, 2, 4, 1
        )
        denominator = scanned[..., 0]
        numerator = scanned[..., 1:]
        node_memory = normalise_stewardship_messages(
            numerator,
            denominator,
            minimum_mass=RECEIVER_NORMALISATION_MASS_FLOOR,
        )
        context[:, :, start:stop] = (
            slab_locations.unsqueeze(-1) * node_memory
        ).sum(3)
    if return_cache:
        return context, LinearScanReceiverCache(carry.contiguous())
    return context


def linear_scan_receiver_context(
    states: torch.Tensor,
    node_locations: torch.Tensor,
    writer_mass: torch.Tensor,
    relative_node_bias: torch.Tensor,
    *,
    chunk_size: int = 64,
    block_size: int | None = None,
    cache: LinearScanReceiverCache | None = None,
    return_cache: bool = False,
    hgrn_scan: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, LinearScanReceiverCache]:
    """Linear-time normalized receiver memory with optional block cache.

    Without ``block_size`` this preserves the historical bidirectional scan.
    Block-causal execution carries only the forward scan across blocks while
    resetting the reverse scan at every block boundary. Positions may
    therefore revise one another inside the current parallel block, but no
    later block can alter an already committed prefix.

    ``hgrn_scan`` -- swaps the block-causal FORWARD scan (the expensive,
    full-sequence, cache-carrying direction) to
    ``gator_linear.core.chunk_hgrn_decayed_scan`` (FLA's fused Triton
    ``chunk_hgrn`` kernel) instead of the chunked pure-PyTorch
    ``decayed_scan``. Measured ~5.5x faster at GTR-ARW's real production
    receiver-memory shape (see ``chunk_hgrn_decayed_scan``'s docstring
    for the full verification). Only affects the block-causal path
    (``block_size`` not ``None``) -- the backward/per-block scan and the
    non-block-causal bidirectional path both stay on ``decayed_scan``
    regardless, since neither is the expensive full-sequence direction
    this targets.
    """

    from gator_linear.core import chunk_hgrn_decayed_scan, decayed_scan

    batch, cells, tokens, hidden, nodes = _validate_receiver_inputs(
        states, node_locations, writer_mass, relative_node_bias
    )
    if block_size is not None:
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size <= 0
        ):
            raise TypeError("block_size must be a positive integer")
        if tokens % block_size:
            raise ValueError("receiver token length must be divisible by block_size")
        # Prefill/append are no-grad operations.  Keep training on the original
        # fully vectorised graph so this inference lifetime optimization cannot
        # alter autograd topology or compiled training graphs.
        if (
            states.device.type == "cpu"
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
        ):
            return _streaming_block_receiver_context(
                states,
                node_locations,
                writer_mass,
                relative_node_bias,
                chunk_size=chunk_size,
                block_size=block_size,
                cache=cache,
                return_cache=return_cache,
            )
    effective_writer_mass = torch.where(
        writer_mass.gt(0), writer_mass, torch.zeros_like(writer_mass)
    )
    writer_node_mass = (
        node_locations.clamp_min(1e-9) * effective_writer_mass.unsqueeze(-1)
    )
    numerator_input = writer_node_mass.unsqueeze(-1) * states.unsqueeze(3)
    combined = torch.cat(
        [writer_node_mass.unsqueeze(-1), numerator_input], dim=-1
    )

    channels = cells * nodes
    rows = batch * (hidden + 1)
    values = combined.permute(0, 4, 2, 1, 3).reshape(
        rows, tokens, channels
    )
    decay = (
        torch.sigmoid(relative_node_bias[..., 0])
        .clamp(0.01, 0.99)
        .reshape(channels)
    )
    mix = torch.sigmoid(relative_node_bias[..., 1]).reshape(channels)

    if block_size is None:
        if cache is not None or return_cache:
            raise ValueError("receiver caches require block_size")
        forward = decayed_scan(values, decay, chunk_size=chunk_size)
        backward = decayed_scan(
            values.flip(1), decay, chunk_size=chunk_size
        ).flip(1)
        next_cache = None
    else:
        if (
            isinstance(block_size, bool)
            or not isinstance(block_size, int)
            or block_size <= 0
        ):
            raise TypeError("block_size must be a positive integer")
        if tokens % block_size:
            raise ValueError("receiver token length must be divisible by block_size")
        if cache is None:
            initial_state = values.new_zeros(rows, channels)
        else:
            if cache.forward_state.shape != (rows, channels):
                raise ValueError("receiver cache forward_state has the wrong shape")
            if cache.forward_state.device != values.device:
                raise ValueError("receiver cache and values must share a device")
            if cache.forward_state.dtype != values.dtype:
                raise ValueError("receiver cache and values must share a dtype")
            initial_state = cache.forward_state
        forward_scan = chunk_hgrn_decayed_scan if hgrn_scan else decayed_scan
        forward_kwargs = {} if hgrn_scan else {"chunk_size": chunk_size}
        forward, final_state = forward_scan(
            values,
            decay,
            initial_state=initial_state,
            return_final_state=True,
            **forward_kwargs,
        )
        blocks = tokens // block_size
        block_values = values.reshape(rows, blocks, block_size, channels)
        reversed_blocks = block_values.flip(2).reshape(
            rows * blocks, block_size, channels
        )
        backward = decayed_scan(
            reversed_blocks, decay, chunk_size=chunk_size
        ).reshape(rows, blocks, block_size, channels).flip(2).reshape(
            rows, tokens, channels
        )
        next_cache = LinearScanReceiverCache(final_state.contiguous())

    scanned = (
        mix[None, None, :] * forward
        + (1.0 - mix[None, None, :]) * backward
    )
    scanned = scanned.reshape(
        batch, hidden + 1, tokens, cells, nodes
    ).permute(0, 3, 2, 4, 1)
    denominator = scanned[..., 0]
    numerator = scanned[..., 1:]
    node_memory = normalise_stewardship_messages(
        numerator,
        denominator,
        minimum_mass=RECEIVER_NORMALISATION_MASS_FLOOR,
    )
    context = (node_locations.unsqueeze(-1) * node_memory).sum(3)
    if return_cache:
        assert next_cache is not None
        return context, next_cache
    return context

def relative_receiver_context(
    states: torch.Tensor,
    node_locations: torch.Tensor,
    writer_mass: torch.Tensor,
    relative_node_bias: torch.Tensor,
    *,
    backend: ReceiverMemoryBackend = "convolution",
    block_size: int | None = None,
    cache: LinearScanReceiverCache | None = None,
    return_cache: bool = False,
    hgrn_scan: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, LinearScanReceiverCache]:
    """Evaluate receiver-specific node memory through a selectable backend.

    ``hgrn_scan`` -- see ``linear_scan_receiver_context``'s docstring;
    only meaningful for ``backend='linear_scan'`` with ``block_size``
    set (block-causal mode).
    """

    if backend != "linear_scan" and (
        block_size is not None or cache is not None or return_cache
    ):
        raise ValueError("block receiver caches require the linear_scan backend")
    if backend != "linear_scan" and hgrn_scan:
        raise ValueError("hgrn_scan requires the linear_scan backend")
    if backend == "attention":
        return attention_receiver_context(
            states, node_locations, writer_mass, relative_node_bias
        )
    if backend == "convolution":
        return convolution_receiver_context(
            states, node_locations, writer_mass, relative_node_bias
        )
    if backend == "linear_scan":
        return linear_scan_receiver_context(
            states,
            node_locations,
            writer_mass,
            relative_node_bias,
            block_size=block_size,
            cache=cache,
            return_cache=return_cache,
            hgrn_scan=hgrn_scan,
        )
    raise ValueError(
        "receiver-memory backend must be 'attention', 'convolution', or "
        "'linear_scan'"
    )
