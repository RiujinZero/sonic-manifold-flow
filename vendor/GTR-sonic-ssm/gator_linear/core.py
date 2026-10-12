"""GTR-L linear-scaling Gator core.

The design principle is "dense hypotheses, linear communication": every work
stream keeps a private hidden state and broad soft occupancy over every node,
but no operator in the relay ever materialises an ``L x L``, ``N x N``, or
``S x S`` tensor.  The quadratic machinery is replaced piece by piece:

* receiver-by-writer memory becomes a linear sequence mixer (grouped
  convolution + prefix/suffix scans + global summaries), ``O(L N)``;
* the full source-to-destination transition matrix becomes a fixed
  constant-degree graph plus a low-rank global bus, ``O(N d + N M)``;
* one full transform per node becomes a shared rank-``r`` program basis
  selected by the route distribution, ``O(r D^2 + N r)``;
* stream-to-stream all-pairs mixing becomes shared node mailboxes,
  ``O(S N)``;
* per-stream vocabulary projections become one projection over a fused
  hidden state, ``O(L S + L V)``.

What deliberately survives, because the cell ablations showed it is the
load-bearing part: separate hidden states, separate route distributions,
broad participation at initialisation, communication through residual
mailboxes rather than hidden-state averaging, and late learned fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from gator_convergent import connection_destinations


@dataclass(frozen=True)
class GatorLinearConfig:
    """Architecture settings for the linear-scaling Gator core."""

    vocab_size: int = 16
    streams: int = 10
    nodes: int = 40
    width: int = 8
    relays: int = 4
    degree: int = 6
    buses: int = 4
    mailbox_channels: int = 4
    mailbox_width: int = 8
    basis_rank: int = 4
    address_width: int = 4
    router_rank: int = 8
    conv_kernel: int = 3
    scan_chunk: int = 64
    scan_normalise: bool = True
    context_length: int = 1024
    graph_layout: str = "small_world"
    graph_seed: int = 0
    graph_rewire: float = 0.5
    causal: bool = False
    block_causal: bool = False
    ingress_temperature: float = 1.0
    private_stream_io: bool = True
    private_stream_rank: int = 4
    heterogeneous_streams: bool = False
    stream_width_min_fraction: float = 0.25
    update_gate_bias: float = 0.0
    readout_bias: bool = True
    tie_token_embeddings: bool = False
    hard_routing: bool = False
    router_occupancy: bool = False
    controlled_writing: bool = True
    linear_attention: bool = False
    linear_attention_key_dim: int = 8

    def __post_init__(self) -> None:
        positive = (
            "vocab_size",
            "streams",
            "nodes",
            "width",
            "relays",
            "degree",
            "buses",
            "mailbox_channels",
            "mailbox_width",
            "basis_rank",
            "address_width",
            "router_rank",
            "conv_kernel",
            "scan_chunk",
            "context_length",
            "private_stream_rank",
            "linear_attention_key_dim",
        )
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.degree > self.nodes:
            raise ValueError("degree cannot exceed nodes")
        if self.address_width < 2:
            raise ValueError("address_width must be at least two")
        if type(self.causal) is not bool:
            raise TypeError("causal must be a boolean")
        if type(self.block_causal) is not bool:
            raise TypeError("block_causal must be a boolean")
        if self.causal and self.block_causal:
            raise ValueError("causal and block_causal are mutually exclusive")
        # Only the symmetric (non-causal) padding needs a centre tap. Causal
        # mode pads entirely on the left, so an even width is well defined --
        # and production language configs use even widths (k=32 at 100M).
        if not self.causal and self.conv_kernel % 2 == 0:
            raise ValueError("conv_kernel must be odd when causal is False")
        if type(self.scan_normalise) is not bool:
            raise TypeError("scan_normalise must be a boolean")
        if type(self.private_stream_io) is not bool:
            raise TypeError("private_stream_io must be a boolean")
        if type(self.readout_bias) is not bool:
            raise TypeError("readout_bias must be a boolean")
        if type(self.tie_token_embeddings) is not bool:
            raise TypeError("tie_token_embeddings must be a boolean")
        if type(self.hard_routing) is not bool:
            raise TypeError("hard_routing must be a boolean")
        if not math.isfinite(self.update_gate_bias):
            raise ValueError("update_gate_bias must be finite")
        if type(self.heterogeneous_streams) is not bool:
            raise TypeError("heterogeneous_streams must be a boolean")
        if not 0.0 < self.stream_width_min_fraction <= 1.0:
            raise ValueError("stream_width_min_fraction must be in (0, 1]")
        if self.heterogeneous_streams:
            narrowest = int(
                round(self.width * self.stream_width_min_fraction)
            )
            if narrowest < 2:
                raise ValueError(
                    "heterogeneous_streams needs the narrowest stream to keep "
                    "at least two dimensions; raise width or "
                    "stream_width_min_fraction"
                )
        if not math.isfinite(self.ingress_temperature):
            raise ValueError("ingress_temperature must be finite")
        if self.ingress_temperature <= 0:
            raise ValueError("ingress_temperature must be positive")
        if self.graph_layout not in {"ring", "small_world", "dense"}:
            raise ValueError("graph_layout must be ring, small_world, or dense")
        if not 0.0 <= self.graph_rewire <= 1.0:
            raise ValueError("graph_rewire must be in [0, 1]")


class LinearSequenceBlockCache(NamedTuple):
    """Constant-size recurrent state for one relay's sequence mixer."""

    local_tail: Tensor
    forward_state: Tensor
    node_sum: Tensor


class LinearAttentionBlockCache(NamedTuple):
    """Raw recurrent states for one relay's linear-attention scan."""

    kv_state: Tensor
    key_state: Tensor


class LinearRelayBlockCache(NamedTuple):
    """All temporal state needed to run one relay on the next block."""

    mixer: LinearSequenceBlockCache
    attention: LinearAttentionBlockCache | None
    occupancy_sum: Tensor | None


class GatorLinearBlockCache(NamedTuple):
    """Immutable committed-prefix cache shared by every GTR-L relay.

    ``next_position`` deliberately remains a scalar tensor rather than a
    Python integer. Compiled proposal graphs can therefore be reused at every
    block position instead of specializing once per growing context length.
    """

    next_position: Tensor
    block_size: int
    relays: tuple[LinearRelayBlockCache, ...]


def detach_block_cache(cache: GatorLinearBlockCache) -> GatorLinearBlockCache:
    """Return a graph-free cache without mutating the supplied cache."""

    relay_caches: list[LinearRelayBlockCache] = []
    for relay in cache.relays:
        attention = relay.attention
        if attention is not None:
            attention = LinearAttentionBlockCache(
                attention.kv_state.detach(), attention.key_state.detach()
            )
        occupancy_sum = (
            None if relay.occupancy_sum is None else relay.occupancy_sum.detach()
        )
        relay_caches.append(
            LinearRelayBlockCache(
                LinearSequenceBlockCache(
                    relay.mixer.local_tail.detach(),
                    relay.mixer.forward_state.detach(),
                    relay.mixer.node_sum.detach(),
                ),
                attention,
                occupancy_sum,
            )
        )
    return GatorLinearBlockCache(
        cache.next_position.detach(), cache.block_size, tuple(relay_caches)
    )


def fixed_address_code(
    length: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    offset: int | Tensor = 0,
    reference_length: int | None = None,
) -> Tensor:
    """Return an immutable sinusoidal slot address code of shape [L, width].

    Legacy callers normalize positions by the requested ``length``. A
    block-causal caller supplies the model's fixed ``reference_length`` and a
    scalar-tensor ``offset`` so an address is identical whether it was built
    during vectorized training, prefill, or a later cached append.
    """

    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    if isinstance(width, bool) or not isinstance(width, int) or width < 2:
        raise ValueError("width must be an integer of at least two")
    if reference_length is not None and (
        isinstance(reference_length, bool)
        or not isinstance(reference_length, int)
        or reference_length <= 0
    ):
        raise ValueError("reference_length must be a positive integer")
    # FP16/BF16 cannot represent every integer near a 2k context boundary.
    # Building positions in the model dtype would merge adjacent slots before
    # sine/cosine is even evaluated (e.g. 2040..2047 in BF16). Do positional
    # arithmetic in FP32 and cast only the finished address code.
    position_dtype = (
        torch.float32
        if dtype in (torch.float16, torch.bfloat16)
        else dtype
    )
    if isinstance(offset, Tensor):
        if offset.ndim != 0 or offset.dtype not in (torch.int32, torch.int64):
            raise TypeError("tensor offset must be a scalar integer")
        torch._assert_async(offset >= 0, "offset must be nonnegative")
        if reference_length is not None:
            torch._assert_async(
                offset + length <= reference_length,
                "address range exceeds reference_length/context_length",
            )
        start = offset.to(device=device, dtype=position_dtype)
    elif isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a nonnegative integer or scalar tensor")
    else:
        if reference_length is not None and offset + length > reference_length:
            raise ValueError(
                "address range exceeds reference_length/context_length"
            )
        start = torch.tensor(offset, device=device, dtype=position_dtype)
    position = torch.arange(
        length, device=device, dtype=position_dtype
    ) + start
    denominator = (
        max(length - 1, 1)
        if reference_length is None
        else max(reference_length - 1, 1)
    )
    angle = position / denominator * math.pi
    channels: list[Tensor] = []
    frequency = 1
    while len(channels) < width:
        channels.append(torch.sin(frequency * angle))
        if len(channels) < width:
            channels.append(torch.cos(frequency * angle))
        frequency += 1
    return torch.stack(channels, dim=-1).to(dtype=dtype)


def _decayed_scan_recurrent_cpu(
    values: Tensor,
    decay: Tensor,
    *,
    initial_state: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Evaluate the scan recurrence directly for CPU inference.

    The general scan below forms a dense triangular kernel inside every
    chunk.  That is friendly to accelerators and autograd, but on CPU it does
    ``chunk_size`` times more arithmetic than the recurrence requires.  A
    prefill does not need an autograd graph, so walk the sequence once and use
    fused ``addcmul`` for ``value + decay * carry``.  The carry is private to
    this function because the ``out=`` form is intentionally in-place.

    This formulation never raises ``decay`` to a sequence-length exponent or
    divides by a small power, so channels near either end of the supported
    decay range remain numerically stable at long context lengths.
    """

    rows, length, channels = values.shape
    output = values.new_empty(rows, length, channels)
    if initial_state is None:
        carry = values.new_zeros(rows, channels)
    else:
        # Do not mutate a caller-owned cache while updating the recurrent
        # state in place.
        carry = initial_state.clone()
    for position in range(length):
        torch.addcmul(values[:, position], carry, decay, out=carry)
        output[:, position].copy_(carry)
    return output, carry.contiguous()


def decayed_scan(
    values: Tensor,
    decay: Tensor,
    *,
    chunk_size: int = 64,
    initial_state: Tensor | None = None,
    return_final_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Return ``h[l] = sum_{l' <= l} decay^(l - l') * values[l']``.

    This is the linear-time stand-in for a distance-weighted receiver kernel.
    A plain cumulative sum weights every earlier slot equally, so a receiver
    cannot tell slot ``l-1`` from slot ``l-30`` and the only remaining way to
    address anything is through the node dimension. A per-channel decay
    restores distance selectivity: channels with small ``decay`` read locally,
    channels near one read globally, and the mixture spans the scales that an
    ``[L, L]`` kernel would have covered explicitly.

    Evaluated in fixed-size chunks so the transient is ``[chunk, chunk, C]``
    rather than ``[L, L, C]`` -- the cost stays ``O(L * chunk * C)``, i.e.
    linear in sequence length, which is the entire point of the design.

    Args:
        values: ``[N, L, C]`` input.
        decay: ``[C]`` per-channel decay in ``(0, 1)``.
        chunk_size: block length; a constant, not a function of ``L``.

    Returns:
        Tensor of shape ``[N, L, C]``.
    """

    if values.ndim != 3:
        raise ValueError("values must have shape [rows, length, channels]")
    if decay.ndim != 1 or decay.shape[0] != values.shape[-1]:
        raise ValueError("decay must have shape [channels]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    rows, length, channels = values.shape
    output = values.new_empty(rows, length, channels)
    if initial_state is None:
        carry = values.new_zeros(rows, channels)
    else:
        if initial_state.shape != (rows, channels):
            raise ValueError("initial_state must have shape [rows, channels]")
        carry = initial_state

    # CPU prefill is dominated by the chunk-local triangular einsum.  In
    # inference there is no backward graph to preserve, so use the exact
    # first-order recurrence and reduce the work from O(L * chunk * C) to
    # O(L * C).  Keep the vectorized implementation for grad-enabled calls:
    # its ordinary PyTorch graph supports training and higher-order autograd.
    if (
        values.device.type == "cpu"
        and not torch.is_grad_enabled()
        and not torch.compiler.is_compiling()
    ):
        output, carry = _decayed_scan_recurrent_cpu(
            values,
            decay,
            initial_state=initial_state,
        )
        if return_final_state:
            return output, carry
        return output

    positions = torch.arange(chunk_size, device=values.device)
    offsets = positions[:, None] - positions[None, :]
    causal = offsets >= 0
    decay_row = decay.view(1, 1, channels)

    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        span = stop - start
        exponent = offsets[:span, :span].clamp_min(0).to(values.dtype)
        weight = decay_row ** exponent.unsqueeze(-1)
        weight = weight * causal[:span, :span].unsqueeze(-1)
        local = torch.einsum("ijc,njc->nic", weight, values[:, start:stop])
        carry_weight = decay_row ** (
            positions[:span].view(1, span, 1).to(values.dtype) + 1.0
        )
        block = local + carry_weight * carry.unsqueeze(1)
        output[:, start:stop] = block
        # Canonical layout matters for compiled recurrent use: a view inherits
        # a stride tied to the just-processed span, which otherwise changes
        # between vectorized prefill and one-block append and forces Dynamo to
        # compile a second proposal graph despite identical cache shapes.
        carry = block[:, -1].contiguous()
    if return_final_state:
        return output, carry
    return output


try:
    from gator_linear._hgrn_fixed_decay import fixed_decay_chunk_hgrn as _chunk_hgrn

    _FLA_IMPORT_ERROR: Exception | None = None
except Exception as error:  # pragma: no cover - exercised only without fla
    _chunk_hgrn = None  # type: ignore[assignment]
    _FLA_IMPORT_ERROR = error


def chunk_hgrn_decayed_scan(
    values: Tensor,
    decay: Tensor,
    *,
    initial_state: Tensor | None = None,
    return_final_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Fused-Triton equivalent of ``decayed_scan``'s forward (causal)
    direction: ``h[l] = sum_{l' <= l} decay^(l - l') * values[l']``.

    HGRN's (Hierarchically Gated Recurrent Network) recurrence is
    ``h_t = exp(g_t) * h_{t-1} + x_t`` -- identical to this function's
    decayed-sum recurrence with ``g_t = log(decay)``, broadcast
    constant across every position since ``decay`` here is learned but
    time-invariant. Only replaces the FORWARD direction --
    ``decayed_scan`` itself remains the implementation for the (much
    shorter, already cheap) backward/per-block scan in block-causal
    receiver memory.

    Phase 1 (stock ``flash-linear-attention`` ``chunk_hgrn``, general
    kernel with a materialized ``[rows, length, channels]`` gate):
    ~5.5x faster than ``decayed_scan`` eager-vs-eager, but stock
    ``chunk_hgrn`` is ``@torch.compiler.disable``'d, forcing a Dynamo
    graph break on every call -- ate the whole win in a real production
    A/B (up to ~32 calls/update in this model).

    Phase 1.5 (``gator_linear._hgrn_compiled.compiled_chunk_hgrn``):
    the same general kernel's forward/backward reimplemented as
    ``torch.library.triton_op``s with ``register_autograd``, so
    ``torch.compile`` captures it as one opaque node -- zero graph
    breaks, verified bit-for-bit matching gradients. But an opaque
    custom op can't be fused by Inductor, and ``torch.compile`` already
    closes most of the eager-vs-eager gap on ``decayed_scan`` for free
    (102ms -> 22.7ms at production shape) -- so the *compiled-vs-
    compiled* edge was only ~1.2x, not 5.5x. Kept around
    (``gator_linear._hgrn_compiled``) as the correctness/performance
    reference for Phase 2.

    Phase 2 (``gator_linear._hgrn_fixed_decay.fixed_decay_chunk_hgrn``,
    used here): the general kernel materializes a full ``[rows,
    length, channels]`` gate and a same-shape cumulative-decay
    intermediate purely to satisfy an API built for a per-position
    gate -- wasted memory traffic for a gate that's actually constant
    per channel. The specialized kernel takes ``log(decay)`` directly
    as a ``[channels]``-shaped input, drops the cumulative-decay
    intermediate entirely (closed-form recomputed instead of
    materialized), and accumulates ``decay``'s gradient directly via
    ``[channels]``-shaped ``tl.atomic_add`` instead of materializing a
    full-size gradient tensor for autograd to reduce afterward.
    Verified to match the general kernel exactly (forward, final
    state, gradients w.r.t. both ``values`` and ``decay``, with and
    without a real ``initial_state`` carry, under bf16 and
    ``torch.compile(fullgraph=True)``) and **~1.8x faster than
    compiled ``decayed_scan``, ~1.6x faster than Phase 1.5's compiled
    general kernel** at GTR-ARW's real production receiver-memory
    shape (rows=2626, length=2032, channels=32; all three under
    ``torch.compile(mode="max-autotune-no-cudagraphs")``, the mode
    production actually uses).

    Args:
        values: ``[rows, length, channels]`` (same convention as
            ``decayed_scan``).
        decay: ``[channels]`` in ``(0, 1)``.
        initial_state: optional ``[rows, channels]`` carry from a prior
            call (block-causal cache carry).

    Returns:
        ``[rows, length, channels]``, or ``(output, final_state)`` if
        ``return_final_state``.
    """

    if _chunk_hgrn is None:
        raise RuntimeError(
            "chunk_hgrn_decayed_scan requires flash-linear-attention "
            f"(`pip install flash-linear-attention`); import failed with: "
            f"{_FLA_IMPORT_ERROR!r}"
        )
    if values.ndim != 3:
        raise ValueError("values must have shape [rows, length, channels]")
    if decay.ndim != 1 or decay.shape[0] != values.shape[-1]:
        raise ValueError("decay must have shape [channels]")
    output, final_state = _chunk_hgrn(
        values, decay.log(), initial_state=initial_state, output_final_state=True,
    )
    if return_final_state:
        return output, final_state
    return output


class LinearSequenceMixer(nn.Module):
    """Mix node mailboxes along the sequence in ``O(L N)`` work.

    Four context views are produced per node channel -- a local grouped
    convolution window, a prefix scan, a suffix scan, and a global summary --
    and combined by learned per-channel matrices.  No receiver-by-writer
    tensor is ever built, which is the whole point: this replaces the
    ``[L, L, N]`` relation kernel that made context length quadratic.

    In causal mode the convolution is left-padded and the suffix scan is
    dropped, so a slot can only ever read itself and earlier slots.  The
    global summary becomes the prefix mean for the same reason.
    """

    def __init__(self, config: GatorLinearConfig) -> None:
        super().__init__()
        self.config = config
        channels = config.mailbox_channels
        mailbox_width = config.mailbox_width
        # Depthwise over (channel, feature) and shared across nodes: nodes stay
        # symmetric under the mixer, and their differentiation comes from the
        # route distribution and the program basis instead.
        self.local_conv = nn.Conv1d(
            channels * mailbox_width,
            channels * mailbox_width,
            kernel_size=config.conv_kernel,
            groups=channels * mailbox_width,
            bias=False,
        )
        # One [Dm, Dm] matrix per (channel, view). Views are ordered
        # local/prefix/suffix/global.
        self.view_weight = nn.Parameter(
            torch.empty(channels, 4, mailbox_width, mailbox_width)
        )
        self.view_bias = nn.Parameter(torch.zeros(channels, mailbox_width))
        # One decay rate per (channel, feature), spread across scales at
        # initialisation so some channels start local and others start global.
        self.forward_decay = nn.Parameter(
            torch.empty(channels * mailbox_width)
        )
        self.reverse_decay = nn.Parameter(
            torch.empty(channels * mailbox_width)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.local_conv.weight,
            std=1.0 / math.sqrt(self.config.conv_kernel),
        )
        # 1/sqrt(4 * Dm) keeps the summed four-view output unit-scaled.
        bound = 1.0 / math.sqrt(4 * self.config.mailbox_width)
        nn.init.uniform_(self.view_weight, -bound, bound)
        nn.init.zeros_(self.view_bias)
        # Receptive fields are set by *timescale*, derived from the context
        # the model is built for, rather than by a fixed logit range.
        #
        # A channel with decay d retains a token n slots back as d^n, so its
        # characteristic timescale is tau = -1/ln(d), i.e. d = exp(-1/tau).
        # Initialising tau geometrically from 1 to the context length gives
        # channels covering every scale from adjacent-slot structure up to
        # whole-window accumulation, and it moves with the context instead of
        # needing to be retuned. This is the standard SSM initialisation.
        #
        # A fixed range silently fails when the context outgrows it: a
        # ceiling of tau ~ 55 retains only 4e-6 of a token 620 slots back, so
        # the decayed scans -- which exist precisely to supply distance
        # structure -- contribute nothing beyond their ceiling and only the
        # positionless global mean carries long-range information.
        channels = self.forward_decay.numel()
        timescales = torch.logspace(
            0.0,
            math.log10(max(float(self.config.context_length), 2.0)),
            channels,
        )
        decay = torch.exp(-1.0 / timescales)
        # Stored as logits; sigmoid() is applied at use.
        spread = torch.log(decay / (1.0 - decay))
        with torch.no_grad():
            self.forward_decay.copy_(spread)
            self.reverse_decay.copy_(spread)

    @staticmethod
    def _restore(
        rows: Tensor,
        batch: int,
        length: int,
        nodes: int,
        channels: int,
        mailbox_width: int,
    ) -> Tensor:
        """Undo the [B*N, L, C*Dm] scan layout back to [B, L, N, C, Dm]."""

        return (
            rows.transpose(1, 2)
            .reshape(batch, nodes, channels, mailbox_width, length)
            .permute(0, 4, 1, 2, 3)
        )

    @staticmethod
    def _scan_normaliser(
        decay: Tensor,
        length: int,
        reference: Tensor,
        *,
        offset: int | Tensor = 0,
    ) -> Tensor:
        """Return sum_{j<=l} decay^j, broadcast to [1, L, 1, C, Dm].

        Closed form rather than a second scan. Keeps every channel on the same
        scale whatever its decay rate, so a long-range channel does not simply
        shout over a short-range one.

        **This converts a sum into a mean, which some tasks cannot afford.**
        The normaliser is ``(1 - decay^(l+1)) / (1 - decay)``. Whenever
        ``1/(1 - decay)`` exceeds the sequence length -- true for the
        long-range channels, where ``decay ~ 0.98`` gives ``~55`` against a
        prefix task's ``L = 15`` -- it grows essentially as ``l + 1`` across
        the whole sequence, so the "prefix scan" returns a prefix *mean*. A
        model needing a running total must then undo a position-dependent
        ``1/(l+1)``. Short-range channels stay sum-like but only reach
        locally, so with normalisation on there is no channel offering a
        long-range sum at all. Disable via ``scan_normalise=False`` for tasks
        built on accumulation; the downstream LayerNorm handles the scale.
        """

        channels, mailbox_width = reference.shape[3], reference.shape[4]
        accumulator_dtype = (
            torch.float32
            if reference.dtype in (torch.float16, torch.bfloat16)
            else reference.dtype
        )
        steps = torch.arange(
            1, length + 1, device=reference.device, dtype=accumulator_dtype
        ).view(length, 1)
        if isinstance(offset, Tensor):
            steps = steps + offset.to(
                device=reference.device, dtype=accumulator_dtype
            )
        else:
            steps = steps + offset
        decay_row = decay.view(1, -1).to(accumulator_dtype)
        total = (1.0 - decay_row**steps) / (1.0 - decay_row).clamp_min(1e-6)
        return total.to(reference.dtype).view(
            1, length, 1, channels, mailbox_width
        )

    @staticmethod
    def _asymptotic_scale(decay: Tensor, reference: Tensor) -> Tensor:
        """Return the position-independent steady-state scale ``1/(1-decay)``.

        Used with ``scan_normalise=False`` instead of leaving the scan fully
        raw. A raw decayed sum grows with the true accumulated count, which
        is exactly the point (see ``decayed_scan``), but with no ceiling that
        growth was found to dominate the per-token LayerNorm applied
        downstream: at long range the raw value became a strong,
        near-monotonic function of position alone, which is trivial for a
        gate to learn to key off and was measured driving ``bus_gate`` to
        1.0 -- fully open -- within 500 steps and never leaving, versus the
        gradual 0.12 -> 0.58 opening seen when the scans were bounded.

        Dividing by the constant asymptote ``1/(1-decay)`` (the sum's value
        as position -> infinity, not its running partial sum) bounds the
        output to the same order of magnitude at every position while still
        growing with the true count -- unlike the per-position normaliser
        above, which forces every position to the *same* final value and
        destroys the count entirely.
        """

        channels, mailbox_width = reference.shape[3], reference.shape[4]
        accumulator_dtype = (
            torch.float32
            if reference.dtype in (torch.float16, torch.bfloat16)
            else reference.dtype
        )
        scale = 1.0 / (
            1.0 - decay.to(accumulator_dtype)
        ).clamp_min(1e-6)
        return scale.to(reference.dtype).view(
            1, 1, 1, channels, mailbox_width
        )

    def _block_forward(
        self,
        node_input: Tensor,
        *,
        block_size: int,
        cache: LinearSequenceBlockCache | None,
        cache_length: Tensor,
    ) -> tuple[Tensor, LinearSequenceBlockCache]:
        """Mix fixed-size causal blocks and return the advanced raw state.

        Every block may read all committed blocks and every position inside
        itself. It cannot flow backward into an earlier block. The vectorized
        path is also the training reference for the recurrent one-block cache.
        """

        batch, length, nodes, channels, mailbox_width = node_input.shape
        if block_size <= 0 or length % block_size != 0:
            raise ValueError("length must be divisible by positive block_size")
        if cache_length.ndim != 0 or cache_length.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise TypeError("cache_length must be a scalar integer tensor")
        flat_channels = channels * mailbox_width
        blocks = length // block_size
        rows_count = batch * nodes
        moved = node_input.permute(0, 2, 3, 4, 1).reshape(
            rows_count, flat_channels, length
        )
        side = (self.config.conv_kernel - 1) // 2

        if cache is None:
            local_tail = moved.new_zeros(rows_count, flat_channels, side)
            forward_state = moved.new_zeros(rows_count, flat_channels)
            accumulator_dtype = (
                torch.float32
                if node_input.dtype in (torch.float16, torch.bfloat16)
                else node_input.dtype
            )
            node_sum = torch.zeros(
                batch,
                nodes,
                channels,
                mailbox_width,
                device=node_input.device,
                dtype=accumulator_dtype,
            )
        else:
            expected_tail = (rows_count, flat_channels, side)
            expected_state = (rows_count, flat_channels)
            expected_sum = (batch, nodes, channels, mailbox_width)
            if cache.local_tail.shape != expected_tail:
                raise ValueError("mixer cache local_tail has the wrong shape")
            if cache.forward_state.shape != expected_state:
                raise ValueError("mixer cache forward_state has the wrong shape")
            if cache.node_sum.shape != expected_sum:
                raise ValueError("mixer cache node_sum has the wrong shape")
            local_tail = cache.local_tail
            forward_state = cache.forward_state
            node_sum = cache.node_sum

        # Each convolution window receives the real left prefix tail, the
        # complete current block, and zeros on its right. ``unfold`` produces
        # all blocks together and prevents a block's final slots from reading
        # the next block through symmetric padding.
        history = torch.cat((local_tail, moved), dim=-1)
        windows = history.unfold(-1, side + block_size, block_size)
        windows = windows.permute(0, 2, 1, 3).reshape(
            rows_count * blocks, flat_channels, side + block_size
        )
        windows = F.pad(windows, (0, side))
        local_rows = self.local_conv(windows)
        local_rows = (
            local_rows.reshape(rows_count, blocks, flat_channels, block_size)
            .permute(0, 2, 1, 3)
            .reshape(rows_count, flat_channels, length)
        )
        local = local_rows.reshape(
            batch, nodes, channels, mailbox_width, length
        ).permute(0, 4, 1, 2, 3)

        rows = moved.transpose(1, 2)
        forward_decay = torch.sigmoid(self.forward_decay)
        prefix_result = decayed_scan(
            rows,
            forward_decay,
            chunk_size=self.config.scan_chunk,
            initial_state=forward_state,
            return_final_state=True,
        )
        prefix_rows, next_forward_state = prefix_result
        prefix = self._restore(
            prefix_rows, batch, length, nodes, channels, mailbox_width
        )
        if self.config.scan_normalise:
            prefix = prefix / self._scan_normaliser(
                forward_decay,
                length,
                node_input,
                offset=cache_length,
            )
        else:
            prefix = prefix / self._asymptotic_scale(
                forward_decay, node_input
            )

        # Reverse state resets at every block boundary: positions may revise
        # one another inside their block, but no suffix state reaches backward
        # from a later block.
        reverse_decay = torch.sigmoid(self.reverse_decay)
        block_rows = rows.reshape(rows_count, blocks, block_size, flat_channels)
        reversed_rows = block_rows.flip(dims=(2,)).reshape(
            rows_count * blocks, block_size, flat_channels
        )
        suffix_rows = decayed_scan(
            reversed_rows,
            reverse_decay,
            chunk_size=self.config.scan_chunk,
        )
        suffix_rows = (
            suffix_rows.reshape(rows_count, blocks, block_size, flat_channels)
            .flip(dims=(2,))
            .reshape(rows_count, length, flat_channels)
        )
        suffix = self._restore(
            suffix_rows, batch, length, nodes, channels, mailbox_width
        )
        if self.config.scan_normalise:
            block_normaliser = self._scan_normaliser(
                reverse_decay, block_size, node_input[:, :block_size]
            ).flip(dims=(1,))
            suffix = suffix / block_normaliser.repeat(1, blocks, 1, 1, 1)
        else:
            suffix = suffix / self._asymptotic_scale(
                reverse_decay, node_input
            )

        # A block-global view includes the complete current block plus all
        # committed blocks. It is constant inside each block, exactly matching
        # the recurrent proposal calculation.
        block_values = node_input.reshape(
            batch, blocks, block_size, nodes, channels, mailbox_width
        )
        accumulator_dtype = node_sum.dtype
        block_sums = block_values.sum(dim=2, dtype=accumulator_dtype)
        cumulative = block_sums.cumsum(dim=1) + node_sum.unsqueeze(1)
        completed = (
            cache_length.to(device=node_input.device, dtype=node_input.dtype)
            + torch.arange(
                1,
                blocks + 1,
                device=node_input.device,
                dtype=accumulator_dtype,
            )
            * block_size
        ).view(1, blocks, 1, 1, 1)
        summary = (cumulative / completed).to(node_input.dtype).unsqueeze(2).expand(
            -1, -1, block_size, -1, -1, -1
        ).reshape(batch, length, nodes, channels, mailbox_width)

        mixed = torch.einsum(
            "blncd,cde->blnce", local, self.view_weight[:, 0]
        )
        mixed = mixed + torch.einsum(
            "blncd,cde->blnce", prefix, self.view_weight[:, 1]
        )
        mixed = mixed + torch.einsum(
            "blncd,cde->blnce", suffix, self.view_weight[:, 2]
        )
        mixed = mixed + torch.einsum(
            "blncd,cde->blnce", summary, self.view_weight[:, 3]
        )
        next_tail = (
            history[..., -side:] if side else history[..., :0]
        ).contiguous()
        next_cache = LinearSequenceBlockCache(
            next_tail,
            next_forward_state,
            node_sum + block_sums.sum(dim=1),
        )
        return mixed + self.view_bias, next_cache

    def forward(
        self,
        node_input: Tensor,
        *,
        block_size: int | None = None,
        cache: LinearSequenceBlockCache | None = None,
        cache_length: Tensor | None = None,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, LinearSequenceBlockCache]:
        """Map node mailboxes [B, L, N, C, Dm] to mixed memory of that shape."""

        if block_size is not None:
            if cache_length is None:
                cache_length = torch.zeros(
                    (), device=node_input.device, dtype=torch.long
                )
            mixed, next_cache = self._block_forward(
                node_input,
                block_size=block_size,
                cache=cache,
                cache_length=cache_length,
            )
            if return_cache:
                return mixed, next_cache
            return mixed
        if cache is not None or cache_length is not None or return_cache:
            raise ValueError("sequence caches require block_size")

        batch, length, nodes, channels, mailbox_width = node_input.shape
        flat_channels = channels * mailbox_width
        # [B, L, N, C, Dm] -> [B * N, C * Dm, L] for the depthwise convolution.
        moved = node_input.permute(0, 2, 3, 4, 1).reshape(
            batch * nodes, flat_channels, length
        )
        if self.config.causal:
            padded = F.pad(moved, (self.config.conv_kernel - 1, 0))
        else:
            side = (self.config.conv_kernel - 1) // 2
            padded = F.pad(moved, (side, side))
        local = self.local_conv(padded)
        local = local.reshape(
            batch, nodes, channels, mailbox_width, length
        ).permute(0, 4, 1, 2, 3)

        # Prefix/suffix scans are parallel cumulative operations over slots,
        # not autoregressive generation. They carry a learned per-channel
        # decay so a receiver can distinguish near from far -- a plain
        # cumulative mean weights every earlier slot identically and destroys
        # exactly the distance information an [L, L] kernel would supply.
        rows = node_input.permute(0, 2, 3, 4, 1).reshape(
            batch * nodes, flat_channels, length
        ).transpose(1, 2)
        forward_decay = torch.sigmoid(self.forward_decay)
        prefix = decayed_scan(
            rows, forward_decay, chunk_size=self.config.scan_chunk
        )
        prefix = self._restore(prefix, batch, length, nodes, channels,
                               mailbox_width)
        if self.config.scan_normalise:
            prefix = prefix / self._scan_normaliser(
                forward_decay, length, node_input
            )
        else:
            prefix = prefix / self._asymptotic_scale(
                forward_decay, node_input
            )

        if self.config.causal:
            # No reverse scan: a causal slot may not read future writers. The
            # view is skipped entirely rather than multiplied by zeros, which
            # would allocate a full-size tensor for no arithmetic.
            suffix = None
            # The "global" view degrades to the running mean for the same
            # reason, so it is length-varying here and cannot be broadcast.
            counts = torch.arange(
                1, length + 1, device=node_input.device,
                dtype=node_input.dtype,
            ).view(1, length, 1, 1, 1)
            summary = node_input.cumsum(dim=1) / counts
        else:
            reverse_decay = torch.sigmoid(self.reverse_decay)
            reversed_rows = rows.flip(dims=(1,))
            suffix = decayed_scan(
                reversed_rows, reverse_decay,
                chunk_size=self.config.scan_chunk,
            ).flip(dims=(1,))
            suffix = self._restore(suffix, batch, length, nodes, channels,
                                   mailbox_width)
            if self.config.scan_normalise:
                suffix = suffix / self._scan_normaliser(
                    reverse_decay, length, node_input
                ).flip(dims=(1,))
            else:
                suffix = suffix / self._asymptotic_scale(
                    reverse_decay, node_input
                )
            summary = node_input.mean(dim=1, keepdim=True)

        # Contract each view separately and accumulate.
        #
        # Stacking the four views into [B, L, N, C, 4, Dm] and running one
        # einsum reads better but materialises the single largest tensor in
        # the model -- 39 MB per relay at batch 128, ~21% of training peak
        # across four relays, purely so the contraction could be written on
        # one line. Summing four [B, L, N, C, Dm] contractions is identical
        # arithmetic without that allocation.
        #
        # `summary` is deliberately left unexpanded: it is constant along the
        # sequence, so contracting it at length 1 and letting the sum
        # broadcast avoids materialising L copies of the same value.
        mixed = torch.einsum("blncd,cde->blnce", local, self.view_weight[:, 0])
        mixed = mixed + torch.einsum(
            "blncd,cde->blnce", prefix, self.view_weight[:, 1]
        )
        if suffix is not None:
            mixed = mixed + torch.einsum(
                "blncd,cde->blnce", suffix, self.view_weight[:, 2]
            )
        mixed = mixed + torch.einsum(
            "blncd,cde->blnce", summary, self.view_weight[:, 3]
        )
        return mixed + self.view_bias


def straight_through_commit(distribution: Tensor) -> Tensor:
    """Commit to the argmax in the forward pass; keep the soft gradient.

    This is the mechanism the reference architecture uses at every relay
    (``gator_path/relay.py``: ``hard_delta + soft_delta - soft_delta.detach()``)
    and GTR-L was missing entirely. The reference's forward pass is a
    genuine discrete route through the graph -- "the forward pass is one
    actual stewardship decision... no route beam survives this relay" --
    while its backward pass still reaches every alternative through the
    undetached soft mixture, so unselected paths keep learning.

    GTR-L instead blended everywhere: soft occupancy over every node, soft
    bus redistribution, soft basis-coefficient averaging. Every one of those
    soft averages is a value the forward pass can *actually output*, which
    makes "blend toward the input-independent marginal" a real, cheap point
    in the function's range for gradient descent to settle into -- observed
    directly as the bus opening fully, then (with the bus closed) occupancy
    collapsing onto a single node instead, in both cases while the loss sat
    exactly on the target marginal's entropy. A hard commit removes that
    point from the range entirely: the only way to produce a constant
    output is if the argmax itself is constant across every input, not
    merely if the soft distribution is.

    Applied to the routing transition between relays, not to ingress -- the
    non-negotiable is broad initial participation, which this preserves via
    the same mechanism the reference relies on: the surrogate gradient still
    reaches every node, hard-committed or not.
    """

    hard = torch.zeros_like(distribution).scatter_(
        -1, distribution.argmax(dim=-1, keepdim=True), 1.0
    )
    return hard + distribution - distribution.detach()


def route_entropy(occupancy: Tensor) -> Tensor:
    """Mean entropy of the per-(slot, stream) node distribution."""

    return -(
        occupancy.clamp_min(1e-9).log() * occupancy
    ).sum(-1).mean()


def load_balance(loads: Tensor) -> Tensor:
    """Squared deviation of mean node load from uniform.

    Mirrors ``stewardship_route_losses_from_statistics`` in
    ``gator/tasks/language.py``: with ``N`` nodes, a perfectly balanced
    marginal load is ``1/N``, so ``load * N - 1`` is zero. Pairing this with
    the entropy term is what makes sharpening safe -- entropy alone drives
    every packet onto the same few nodes and starves the rest, which is the
    dead-node failure the design forbids. Sharp per slot, balanced in
    aggregate.
    """

    if loads.ndim != 2:
        raise ValueError("loads must have shape [relays, nodes]")
    nodes = loads.shape[-1]
    return ((loads.mean(dim=0) * nodes - 1.0) ** 2).mean()


class LinearRelay(nn.Module):
    """One linear relay: write, mix, read, route, transport, transform."""

    def __init__(self, config: GatorLinearConfig) -> None:
        super().__init__()
        self.config = config
        width = config.width
        channels = config.mailbox_channels
        mailbox_width = config.mailbox_width

        self.write_projection = nn.Linear(width, channels * mailbox_width)
        self.write_gate = nn.Linear(width, channels)
        # Private per-stream computation, not merely private state.
        #
        # With a fully shared relay every stream applies the *same* function
        # and differs only by an additive embedding and a few gates. Streams
        # that compute the same function from near-identical inputs converge
        # under any amount of mixing, so the coalition degenerates into ten
        # copies of one voter -- observed directly as fusion weights
        # collapsing to 0.0998..0.1004.
        #
        # The reference avoids this with private_cell_io: each cell owns its
        # token tables, so traces are genuinely different functions and fail
        # differently, which is what makes their combination exceed any one
        # of them. Low-rank adapters give the same property cheaply: cost is
        # O(S * rank * width), and S is a fixed architectural constant, so
        # the linear-scaling claims in N and L are untouched.
        if config.private_stream_io:
            rank = config.private_stream_rank
            self.write_down = nn.Parameter(
                torch.empty(config.streams, width, rank)
            )
            self.write_up = nn.Parameter(
                torch.empty(config.streams, rank, channels * mailbox_width)
            )
            self.read_down = nn.Parameter(
                torch.empty(config.streams, channels * mailbox_width, rank)
            )
            self.read_up = nn.Parameter(
                torch.empty(config.streams, rank, width)
            )
        else:
            for name in ("write_down", "write_up", "read_down", "read_up"):
                self.register_parameter(name, None)
        self.mixer = LinearSequenceMixer(config)
        # Read gates are per stream so a stream can weight mailbox channels
        # differently -- this is what lets channel roles (content, position,
        # correction, formation) differentiate without being hand-assigned.
        self.read_projection = nn.Linear(channels * mailbox_width, width)
        self.read_gate = nn.Parameter(torch.zeros(config.streams, channels))
        # Keep the mailbox context on the same scale as the private state, so
        # evidence is not drowned out by the residual stream.
        self.context_norm = nn.LayerNorm(width)

        # Linear (retention-style) attention: a content-addressed mailbox
        # read at O(L) cost, approximating the reference's relative-position
        # attention without ever materialising an [L, L] or [N, N] tensor.
        #
        # The existing decayed scans in LinearSequenceMixer already give
        # every node an O(L) memory of "what was written here, weighted by
        # recency" -- but the combination weights are fixed per channel, not
        # a function of content. A reading position cannot ask "which past
        # packet is actually relevant to *me* specifically", only "how far
        # back". This closes that gap the same way linear attention /
        # RetNet do: replace the plain decayed running value with a decayed
        # running key(x)value outer product, then read it out with a
        # content query instead of a fixed weight. The scan itself is
        # unchanged (same decayed_scan helper, same O(L) chunked cost) --
        # only the thing being accumulated (a [Dk, Dm] matrix instead of a
        # Dm vector) and the read (a query dot product instead of a fixed
        # per-channel combination) are new.
        if config.linear_attention:
            key_dim = config.linear_attention_key_dim
            self.attn_key = nn.Linear(width, key_dim)
            self.attn_query = nn.Linear(width, key_dim)
            self.attn_value = nn.Linear(width, mailbox_width)
            self.attn_out = nn.Linear(mailbox_width, width)
            # A single retention decay, initialised mid-range (~0.9) the same
            # way the mixer's scan channels are: stored as a logit, sigmoid
            # applied at use.
            self.attn_decay = nn.Parameter(torch.zeros(key_dim * mailbox_width))
            self.attn_key_decay = nn.Parameter(torch.zeros(key_dim))

        router_input = 2 * width + config.address_width
        if config.router_occupancy:
            router_input += config.nodes
        self.router_query = nn.Linear(router_input, config.router_rank)
        self.edge_key = nn.Parameter(
            torch.empty(config.nodes, config.degree, config.router_rank)
        )
        self.edge_bias = nn.Parameter(torch.zeros(config.nodes, config.degree))
        self.stream_edge_bias = nn.Parameter(
            torch.zeros(config.streams, config.degree)
        )

        # Buses are per stream so they stay plural and specialised. A single
        # shared bus pair collapses into one homogenising pipe: every stream
        # then receives the same near-uniform redistribution, occupancy
        # flattens, and the graph loses the ability to address anything.
        self.bus_in = nn.Parameter(
            torch.empty(config.streams, config.nodes, config.buses)
        )
        self.bus_out = nn.Parameter(
            torch.empty(config.streams, config.buses, config.nodes)
        )
        self.bus_gate = nn.Linear(router_input, 1)

        self.basis_weight = nn.Parameter(
            torch.empty(config.basis_rank, width, width)
        )
        self.basis_bias = nn.Parameter(torch.zeros(config.basis_rank, width))
        self.node_coefficients = nn.Parameter(
            torch.empty(config.nodes, config.basis_rank)
        )
        self.update_gate = nn.Linear(width, width)
        # The expected program coefficient is E_{n~Q}[a_n], whose magnitude
        # falls as ~1/sqrt(N_effective) when occupancy is broad. Normalising
        # the resulting update keeps relay strength independent of node count,
        # which both makes early training viable and keeps behaviour
        # comparable when scaling 40 -> 160 nodes.
        self.delta_norm = nn.LayerNorm(width)
        self.norm = nn.LayerNorm(width)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        config = self.config
        nn.init.uniform_(
            self.edge_key,
            -1.0 / math.sqrt(config.router_rank),
            1.0 / math.sqrt(config.router_rank),
        )
        # Edge 0 is the self-loop in every supported layout. Favouring it at
        # initialisation stops route mass from diffusing to the graph's
        # stationary distribution before the router has learned anything --
        # soft repeated random-walk steps otherwise wash out addressing within
        # a couple of relays.
        # Kept mild: the routing softmax divides by an annealed temperature
        # (down to ~0.35), so a bias of 2.0 becomes an effective ~5.7 and
        # freezes every packet on its self-loop -- observed as route entropy
        # collapsing to 0.22 against a ln(6)=1.79 maximum. 0.5 is a
        # preference, not a lock.
        with torch.no_grad():
            self.edge_bias.zero_()
            self.edge_bias[:, 0] = 0.5
        nn.init.zeros_(self.stream_edge_bias)
        # Selective (not flat) bus profiles, so each bus moves mass somewhere
        # specific rather than smearing it everywhere.
        nn.init.normal_(self.bus_in, std=1.0)
        nn.init.normal_(self.bus_out, std=1.0)
        # Start the bus mostly closed; local transport should dominate until
        # long-distance movement is actually needed.
        nn.init.zeros_(self.bus_gate.weight)
        nn.init.constant_(self.bus_gate.bias, -2.0)
        nn.init.uniform_(
            self.basis_weight,
            -1.0 / math.sqrt(config.width),
            1.0 / math.sqrt(config.width),
        )
        nn.init.zeros_(self.basis_bias)
        # Small non-zero spread so nodes start with slightly different program
        # mixtures but none is born dead.
        nn.init.normal_(self.node_coefficients, std=0.5)
        # Open the residual gate at initialisation.
        #
        # A zero bias gives sigmoid(0) = 0.5, so each relay admits only half
        # of the context-derived update and LayerNorm renormalises straight
        # after -- input influence is diluted once per relay, four times over.
        # Measured consequence: every stream emits a single constant token
        # (share=1.00, distinct=1) while the readout encodes the marginal for
        # free, and loss sits at exactly ln(12). The input path has to be
        # able to outweigh that constant for the model to leave the basin.
        nn.init.constant_(self.update_gate.bias, config.update_gate_bias)
        if self.config.private_stream_io:
            # Down projections are seeded independently per stream so the
            # adapters start genuinely different; up projections start small
            # so the shared trunk still dominates early and the private paths
            # differentiate under gradient rather than injecting noise.
            for tensor in (self.write_down, self.read_down):
                nn.init.normal_(tensor, std=1.0 / math.sqrt(tensor.shape[1]))
            for tensor in (self.write_up, self.read_up):
                nn.init.normal_(tensor, std=0.05)
        if self.config.linear_attention:
            # Mid-range decay (sigmoid(2.2) ~= 0.9), the same starting point
            # the mixer's own scan channels use before they spread out.
            nn.init.constant_(self.attn_decay, 2.2)
            nn.init.constant_(self.attn_key_decay, 2.2)
            # Small output projection: the content-addressed term starts as
            # a minor correction to the existing decay-only context rather
            # than overwhelming it before it has learned anything useful.
            nn.init.normal_(self.attn_out.weight, std=0.02)
            nn.init.zeros_(self.attn_out.bias)

    def _linear_attention(
        self,
        hidden: Tensor,
        write_occupancy: Tensor,
        occupancy: Tensor,
        *,
        cache: LinearAttentionBlockCache | None = None,
        return_cache: bool = False,
    ) -> Tensor | tuple[Tensor, LinearAttentionBlockCache]:
        """Content-addressed mailbox read via a decayed key(x)value scan.

        See the comment beside ``attn_key`` in ``__init__`` for the
        mechanism. Every contraction here is per (position, node) or per
        (position, stream, node) with small constant feature dimensions
        (``Dk``, ``Dm``) -- the same order of cost as the occupancy-weighted
        read a few lines below in ``forward``, not a new asymptotic term.
        Only the scan itself (``decayed_scan``, already ``O(L)``) grows with
        length, exactly as it does for the existing mixer.
        """

        config = self.config
        batch, length, streams, _ = hidden.shape
        key_dim = config.linear_attention_key_dim
        mailbox_width = config.mailbox_width
        nodes = config.nodes

        key = self.attn_key(hidden)
        query = self.attn_query(hidden)
        value = self.attn_value(hidden)
        # elu(x)+1: the standard linear-attention feature map -- keeps
        # features non-negative (so the running normaliser cannot cancel
        # itself out) without exp()'s overflow risk.
        phi_key = F.elu(key) + 1.0
        phi_query = F.elu(query) + 1.0

        # Aggregate per node the same way node_input is built in step 1:
        # occupancy-weighted over streams, so what a node "remembers
        # writing" matches what it actually received in the mailbox.
        node_key = torch.einsum("blsn,blsk->blnk", write_occupancy, phi_key)
        node_value = torch.einsum("blsn,blsm->blnm", write_occupancy, value)

        kv_outer = torch.einsum("blnk,blnm->blnkm", node_key, node_value)
        kv_rows = kv_outer.permute(0, 2, 3, 4, 1).reshape(
            batch * nodes, key_dim * mailbox_width, length
        ).transpose(1, 2)
        key_rows = node_key.permute(0, 2, 3, 1).reshape(
            batch * nodes, key_dim, length
        ).transpose(1, 2)

        decay = torch.sigmoid(self.attn_decay)
        key_decay = torch.sigmoid(self.attn_key_decay)
        if cache is None:
            initial_kv = None
            initial_key = None
        else:
            if cache.kv_state.shape != (batch * nodes, key_dim * mailbox_width):
                raise ValueError("attention cache kv_state has the wrong shape")
            if cache.key_state.shape != (batch * nodes, key_dim):
                raise ValueError("attention cache key_state has the wrong shape")
            initial_kv = cache.kv_state
            initial_key = cache.key_state
        kv_result = decayed_scan(
            kv_rows,
            decay,
            chunk_size=config.scan_chunk,
            initial_state=initial_kv,
            return_final_state=True,
        )
        key_result = decayed_scan(
            key_rows,
            key_decay,
            chunk_size=config.scan_chunk,
            initial_state=initial_key,
            return_final_state=True,
        )
        kv_state_raw, next_kv = kv_result
        key_state_raw, next_key = key_result

        kv_state = LinearSequenceMixer._restore(
            kv_state_raw, batch, length, nodes, key_dim, mailbox_width
        )
        key_state = LinearSequenceMixer._restore(
            key_state_raw, batch, length, nodes, key_dim, 1
        ).squeeze(-1)

        numerator = torch.einsum("blsk,blnkm->blsnm", phi_query, kv_state)
        denominator = torch.einsum(
            "blsk,blnk->blsn", phi_query, key_state
        )
        # Hard routing leaves most nodes exactly empty. Normalising an empty
        # numerator by a tiny epsilon gives the correct zero forward value,
        # but its surrogate derivative scales as 1 / epsilon. With
        # straight-through routing that sent otherwise ordinary CE gradients
        # into the millions and left the optimiser permanently clipped.
        #
        # An empty mailbox has no content to read and therefore no useful
        # content gradient. Mask that branch explicitly. Occupied but
        # extremely low-mass nodes retain a bounded normalised read so soft
        # routing can still learn to use them.
        occupied = denominator > 0
        safe_denominator = denominator.clamp_min(1e-3)
        content_per_node = numerator / safe_denominator.unsqueeze(-1)
        content_per_node = torch.where(
            occupied.unsqueeze(-1),
            content_per_node,
            torch.zeros_like(content_per_node),
        )

        content = torch.einsum("blsn,blsnm->blsm", occupancy, content_per_node)
        output = self.attn_out(content)
        if return_cache:
            return output, LinearAttentionBlockCache(next_kv, next_key)
        return output

    def forward(
        self,
        hidden: Tensor,
        occupancy: Tensor,
        addresses: Tensor,
        edge_destinations: Tensor,
        *,
        temperature: float = 1.0,
        write_gate: Tensor | None = None,
        block_size: int | None = None,
        cache: LinearRelayBlockCache | None = None,
        cache_length: Tensor | None = None,
        return_cache: bool = False,
    ) -> (
        tuple[Tensor, Tensor, dict[str, Tensor]]
        | tuple[Tensor, Tensor, dict[str, Tensor], LinearRelayBlockCache]
    ):
        """Advance one relay.

        Args:
            hidden: private stream states ``[B, L, S, D]``.
            occupancy: soft node occupancy ``[B, L, S, N]``, rows sum to one.
            addresses: immutable slot addresses ``[L, Da]``.
            edge_destinations: legal outgoing edges ``[N, degree]``.
            temperature: routing softmax temperature.
            write_gate: optional ``[B, L, 1, 1]`` controlled-writing schedule
                -- silences a slot's contribution to the shared mailbox write
                (e.g. blank output slots early on) without touching what that
                slot reads, how it routes, or its own residual update.

        Returns:
            Next hidden state, next occupancy, and diagnostics.
        """

        config = self.config
        batch, length, streams, width = hidden.shape
        channels = config.mailbox_channels
        mailbox_width = config.mailbox_width
        if cache is not None:
            if config.linear_attention and cache.attention is None:
                raise ValueError(
                    "linear-attention relay cache is missing attention state"
                )
            if config.router_occupancy and cache.occupancy_sum is None:
                raise ValueError(
                    "router-occupancy relay cache is missing occupancy state"
                )

        # 1. Mailbox write. Streams deposit gated messages; private states are
        #    never averaged, only their projected messages are combined.
        message = self.write_projection(hidden)
        if self.write_down is not None:
            message = message + torch.einsum(
                "blsd,sdr,sre->blse", hidden, self.write_down, self.write_up
            )
        message = message.reshape(
            batch, length, streams, channels, mailbox_width
        )
        gate = torch.sigmoid(self.write_gate(hidden)).unsqueeze(-1)
        message = message * gate
        # Controlled writing: a slot's contribution to the shared mailbox can
        # be silenced by an external schedule (e.g. blank output slots at
        # relay 0, ramped to full by the final relay -- gator_path/prefix.py's
        # write_gate_schedule). Only the *write* is gated; the same slot
        # still reads the mailbox and updates its own state normally, so a
        # silenced slot is invisible to everyone else without being frozen
        # itself.
        write_occupancy = occupancy if write_gate is None else occupancy * write_gate
        node_input = torch.einsum("blsn,blscm->blncm", write_occupancy, message)
        # Fix the overall scale without destroying node selectivity.
        #
        # Unnormalised, the mailbox carries O(S/N) magnitude -- at S=10, N=40
        # that is ~0.08 of the private residual stream, and after four relays
        # the evidence reaching an output slot is ~1e-4: the model cannot see
        # its own input at all.
        #
        # Dividing by each node's *own* mass (the numerator/denominator form
        # in gator_convergent._node_memory) fixes the scale but breaks
        # addressing: a node nobody routed to would return a full-magnitude
        # average of near-zero mass instead of ~0, so "which nodes did this
        # slot write to" stops being readable. Normalising by the mean mass
        # across nodes fixes the scale while preserving the relative
        # differences between nodes, which is what carries the address.
        # Computed from the same (gated) write mass as node_input, so a
        # silenced writer shrinks the scale along with its contribution
        # instead of being divided by a scale that assumes it wrote in full.
        node_mass = write_occupancy.sum(dim=2)
        scale = node_mass.mean(dim=-1, keepdim=True).clamp_min(1e-6)
        node_input = node_input / scale.unsqueeze(-1).unsqueeze(-1)

        # 2. Linear sequence mixing -- O(L N), no [L, L] tensor.
        if block_size is None:
            if cache is not None or cache_length is not None or return_cache:
                raise ValueError("relay caches require block_size")
            node_memory = self.mixer(node_input)
            next_mixer_cache = None
        else:
            if length % block_size != 0:
                raise ValueError("relay length must be divisible by block_size")
            if cache_length is None:
                cache_length = torch.zeros(
                    (), device=hidden.device, dtype=torch.long
                )
            mixer_result = self.mixer(
                node_input,
                block_size=block_size,
                cache=None if cache is None else cache.mixer,
                cache_length=cache_length,
                return_cache=True,
            )
            node_memory, next_mixer_cache = mixer_result

        # 3. Mailbox read through each stream's own occupancy and gates.
        read = torch.einsum("blsn,blncm->blscm", occupancy, node_memory)
        read = read * torch.sigmoid(self.read_gate).view(1, 1, streams, channels, 1)
        flat_read = read.reshape(
            batch, length, streams, channels * mailbox_width
        )
        context = self.read_projection(flat_read)
        if self.read_down is not None:
            context = context + torch.einsum(
                "blse,ser,srd->blsd", flat_read, self.read_down, self.read_up
            )
        if config.linear_attention:
            attention_result = self._linear_attention(
                hidden,
                write_occupancy,
                occupancy,
                cache=None if cache is None else cache.attention,
                return_cache=block_size is not None,
            )
            if block_size is None:
                context = context + attention_result
                next_attention_cache = None
            else:
                attention_context, next_attention_cache = attention_result
                context = context + attention_context
        else:
            if cache is not None and cache.attention is not None:
                raise ValueError("cache has attention state but relay does not")
            next_attention_cache = None
        context = self.context_norm(context)

        # 4. Constant-degree routing: score only legal edges, never N x N.
        address = addresses.view(1, length, 1, -1).expand(
            batch, length, streams, -1
        )
        router_terms = [hidden, context, address]
        if config.router_occupancy:
            # Current per-node load, averaged across the sequence and
            # broadcast back to every slot -- the router's missing "how
            # loaded is each node right now" signal (gtr_a_experiments.py's
            # GTRFixedSlots: occupancy = q.mean(1, keepdim=True).expand_as(q)).
            if block_size is None:
                occupancy_summary = occupancy.mean(
                    dim=1, keepdim=True
                ).expand_as(occupancy)
                next_occupancy_sum = None
            else:
                blocks = length // block_size
                if cache is None or cache.occupancy_sum is None:
                    accumulator_dtype = (
                        torch.float32
                        if occupancy.dtype in (torch.float16, torch.bfloat16)
                        else occupancy.dtype
                    )
                    previous_occupancy = torch.zeros(
                        batch,
                        streams,
                        config.nodes,
                        device=occupancy.device,
                        dtype=accumulator_dtype,
                    )
                else:
                    expected = (batch, streams, config.nodes)
                    if cache.occupancy_sum.shape != expected:
                        raise ValueError(
                            "relay cache occupancy_sum has the wrong shape"
                        )
                    previous_occupancy = cache.occupancy_sum
                block_occupancy = occupancy.reshape(
                    batch, blocks, block_size, streams, config.nodes
                )
                accumulator_dtype = previous_occupancy.dtype
                occupancy_sums = block_occupancy.sum(
                    dim=2, dtype=accumulator_dtype
                )
                cumulative_occupancy = (
                    occupancy_sums.cumsum(dim=1)
                    + previous_occupancy.unsqueeze(1)
                )
                completed = (
                    cache_length.to(device=hidden.device, dtype=hidden.dtype)
                    + torch.arange(
                        1,
                        blocks + 1,
                        device=hidden.device,
                        dtype=accumulator_dtype,
                    )
                    * block_size
                ).view(1, blocks, 1, 1)
                occupancy_summary = (
                    (cumulative_occupancy / completed).to(occupancy.dtype)
                    .unsqueeze(2)
                    .expand(-1, -1, block_size, -1, -1)
                    .reshape(batch, length, streams, config.nodes)
                )
                next_occupancy_sum = (
                    previous_occupancy + occupancy_sums.sum(dim=1)
                )
            router_terms.append(occupancy_summary)
        else:
            if cache is not None and cache.occupancy_sum is not None:
                raise ValueError("cache has occupancy state but relay does not")
            next_occupancy_sum = None
        router_input = torch.cat(router_terms, dim=-1)
        query = self.router_query(router_input)
        edge_logits = torch.einsum("blsr,ndr->blsnd", query, self.edge_key)
        edge_logits = edge_logits + self.edge_bias
        edge_logits = edge_logits + self.stream_edge_bias.view(
            1, 1, streams, 1, config.degree
        )
        edge_probability = torch.softmax(edge_logits / temperature, dim=-1)

        # 5. Transport dense route mass through sparse edges. Mass is conserved
        #    because the edge distribution sums to one per source node.
        moved = occupancy.unsqueeze(-1) * edge_probability
        destinations = edge_destinations.reshape(-1).view(1, 1, 1, -1).expand(
            batch, length, streams, -1
        )
        local_occupancy = torch.zeros_like(occupancy).scatter_add_(
            -1,
            destinations,
            moved.reshape(batch, length, streams, -1),
        )

        # 6. Low-rank global bus transport for long-distance movement. Both
        #    factors are normalised so the result is still a distribution.
        bus_in = torch.softmax(self.bus_in, dim=-1)
        bus_out = torch.softmax(self.bus_out, dim=-1)
        bus_mass = torch.einsum("blsn,snm->blsm", occupancy, bus_in)
        global_occupancy = torch.einsum("blsm,smn->blsn", bus_mass, bus_out)
        bus_gate = torch.sigmoid(self.bus_gate(router_input))
        next_occupancy = (
            1.0 - bus_gate
        ) * local_occupancy + bus_gate * global_occupancy
        next_occupancy = next_occupancy / next_occupancy.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)
        if self.config.hard_routing:
            next_occupancy = straight_through_commit(next_occupancy)

        # 7. Shared node-program basis. The route distribution selects an
        #    expected coefficient vector, so only `basis_rank` transforms run
        #    instead of one transform per node.
        signal = hidden + context
        basis = torch.einsum("blsd,rde->blsre", signal, self.basis_weight)
        basis = F.gelu(basis + self.basis_bias)
        coefficients = torch.einsum(
            "blsn,nr->blsr", next_occupancy, self.node_coefficients
        )
        delta = self.delta_norm(
            torch.einsum("blsr,blsrd->blsd", coefficients, basis)
        )

        # 8. Private residual update -- states stay separate per stream.
        next_hidden = self.norm(
            hidden + torch.sigmoid(self.update_gate(hidden)) * delta
        )

        diagnostics = {
            "edge_probability": edge_probability.detach(),
            "occupancy": next_occupancy.detach(),
            "bus_gate": bus_gate.detach(),
        }
        if return_cache:
            if block_size is None or next_mixer_cache is None:
                raise ValueError("return_cache requires block_size")
            return (
                next_hidden,
                next_occupancy,
                diagnostics,
                LinearRelayBlockCache(
                    next_mixer_cache,
                    next_attention_cache,
                    next_occupancy_sum,
                ),
            )
        return next_hidden, next_occupancy, diagnostics


class GatorLinearCore(nn.Module):
    """Stack of linear relays over private streams and shared nodes."""

    def __init__(self, config: GatorLinearConfig) -> None:
        super().__init__()
        if not isinstance(config, GatorLinearConfig):
            raise TypeError("config must be a GatorLinearConfig")
        self.config = config
        self.relays = nn.ModuleList(
            LinearRelay(config) for _ in range(config.relays)
        )
        # Heterogeneous stream capacity.
        #
        # Per-stream adapters still give every stream the *same shape*, so
        # nothing stops them converging to the same function -- measured
        # directly: stream agreement rose from 0.13 to 0.86 during training,
        # and rose fastest with private adapters enabled.
        #
        # Giving streams genuinely different widths removes the shared
        # function space they were converging into. A stream restricted to
        # d_s < D dimensions cannot represent what a wider stream can, so
        # they must specialise rather than duplicate. Widths are spread
        # linearly from stream_width_min_fraction * D up to D.
        #
        # The mask is applied after each relay's normalisation, so inactive
        # dimensions are exactly zero rather than picking up the mean shift
        # that LayerNorm would otherwise reintroduce.
        widths = torch.linspace(
            max(2.0, config.width * config.stream_width_min_fraction),
            float(config.width),
            config.streams,
        ).round().long()
        mask = torch.zeros(config.streams, config.width)
        for index, active in enumerate(widths.tolist()):
            mask[index, :active] = 1.0
        self.register_buffer(
            "stream_width_mask",
            mask if config.heterogeneous_streams else torch.ones_like(mask),
            persistent=False,
        )
        self.stream_widths = (
            widths.tolist()
            if config.heterogeneous_streams
            else [config.width] * config.streams
        )
        self.register_buffer(
            "edge_destinations",
            connection_destinations(
                n_nodes=config.nodes,
                degree=config.degree,
                layout=config.graph_layout,
                seed=config.graph_seed,
                rewire_probability=config.graph_rewire,
            ),
            persistent=False,
        )
        # Ingress must be *broad*, not *uniform*. Every node has to receive
        # mass (a starved node gets no gradient, the Gator-P failure), but a
        # degenerately flat occupancy is just as useless: every node then
        # holds identical content and the graph cannot address anything. The
        # key scale below gives real per-slot/per-stream structure while the
        # softmax still leaves every node well above zero.
        self.ingress_key = nn.Parameter(
            torch.empty(config.nodes, config.width)
        )
        self.ingress_bias = nn.Parameter(torch.zeros(config.nodes))
        self.stream_ingress_bias = nn.Parameter(
            torch.zeros(config.streams, config.nodes)
        )
        nn.init.normal_(self.ingress_key, std=1.0 / math.sqrt(config.width))
        nn.init.normal_(self.stream_ingress_bias, std=0.5)

    def initial_occupancy(
        self,
        hidden: Tensor,
        *,
        temperature: float | None = None,
    ) -> Tensor:
        """Return broad soft occupancy ``[B, L, S, N]`` over every node."""

        config = self.config
        scale = (
            config.ingress_temperature if temperature is None else temperature
        )
        logits = torch.einsum("blsd,nd->blsn", hidden, self.ingress_key)
        logits = logits + self.ingress_bias
        logits = logits + self.stream_ingress_bias.view(
            1, 1, config.streams, config.nodes
        )
        return torch.softmax(logits / scale, dim=-1)

    def forward(
        self,
        hidden: Tensor,
        addresses: Tensor,
        *,
        temperature: float = 1.0,
        write_gate_schedule: Tensor | None = None,
        return_diagnostics: bool = False,
        block_size: int | None = None,
        cache: GatorLinearBlockCache | None = None,
        return_cache: bool = False,
    ) -> (
        tuple[Tensor, list[dict[str, Tensor]], dict[str, Tensor]]
        | tuple[
            Tensor,
            list[dict[str, Tensor]],
            dict[str, Tensor],
            GatorLinearBlockCache,
        ]
    ):
        """Run every relay over the initial stream states.

        Args:
            write_gate_schedule: optional ``[relays, B, L]`` controlled-
                writing schedule (see ``LinearRelay.forward``); ``schedule[i]``
                gates the mailbox write for relay ``i``.

        Returns the final hidden state, optional detached diagnostics, and
        differentiable route statistics for the sharpening/balance losses.
        """

        if self.config.block_causal:
            if block_size is None:
                raise ValueError("block_causal core execution requires block_size")
            if (
                isinstance(block_size, bool)
                or not isinstance(block_size, int)
                or block_size <= 0
            ):
                raise TypeError("block_size must be a positive integer")
            if hidden.shape[1] % block_size != 0:
                raise ValueError(
                    "block_causal sequence length must be divisible by block_size"
                )
            if cache is not None:
                if (
                    isinstance(cache.block_size, bool)
                    or not isinstance(cache.block_size, int)
                    or cache.block_size <= 0
                ):
                    raise TypeError("cache block_size must be a positive integer")
                if cache.block_size != block_size:
                    raise ValueError("cache block_size does not match execution")
                if len(cache.relays) != len(self.relays):
                    raise ValueError("cache relay count does not match model")
                cache_length = cache.next_position
                if (
                    cache_length.ndim != 0
                    or cache_length.dtype not in (torch.int32, torch.int64)
                ):
                    raise TypeError(
                        "cache next_position must be a scalar integer tensor"
                    )
                if cache_length.device != hidden.device:
                    raise ValueError(
                        "cache next_position and hidden must share a device"
                    )
            else:
                cache_length = torch.zeros(
                    (), device=hidden.device, dtype=torch.long
                )
            torch._assert_async(
                cache_length >= 0,
                "cache next_position cannot be negative",
            )
            torch._assert_async(
                cache_length.remainder(block_size) == 0,
                "cache next_position must align to block_size",
            )
            torch._assert_async(
                cache_length + hidden.shape[1] <= self.config.context_length,
                "block cache would exceed context_length; sliding eviction "
                "is not supported",
            )
        else:
            if block_size is not None or cache is not None or return_cache:
                raise ValueError("block cache execution requires block_causal=True")
            cache_length = torch.zeros(
                (), device=hidden.device, dtype=torch.long
            )

        mask = self.stream_width_mask.view(1, 1, *self.stream_width_mask.shape)
        hidden = hidden * mask
        occupancy = self.initial_occupancy(hidden)
        records: list[dict[str, Tensor]] = []
        entropies: list[Tensor] = []
        loads: list[Tensor] = []
        next_relay_caches: list[LinearRelayBlockCache] = []
        for index, relay in enumerate(self.relays):
            # Statistics are taken on the occupancy entering each relay, so
            # the ingress distribution is included and the final transport is
            # accounted for by the trailing append below.
            entropies.append(route_entropy(occupancy))
            loads.append(occupancy.mean(dim=(0, 1, 2)))
            write_gate = (
                None
                if write_gate_schedule is None
                else write_gate_schedule[index][:, :, None, None]
            )
            if self.config.block_causal:
                relay_result = relay(
                    hidden,
                    occupancy,
                    addresses,
                    self.edge_destinations,
                    temperature=temperature,
                    write_gate=write_gate,
                    block_size=block_size,
                    cache=None if cache is None else cache.relays[index],
                    cache_length=cache_length,
                    return_cache=True,
                )
                hidden, occupancy, diagnostics, next_relay_cache = relay_result
                next_relay_caches.append(next_relay_cache)
            else:
                hidden, occupancy, diagnostics = relay(
                    hidden,
                    occupancy,
                    addresses,
                    self.edge_destinations,
                    temperature=temperature,
                    write_gate=write_gate,
                )
            # Re-apply after the relay's LayerNorm, which would otherwise
            # push a mean shift back into the inactive dimensions.
            hidden = hidden * mask
            if return_diagnostics:
                records.append(diagnostics)
        entropies.append(route_entropy(occupancy))
        loads.append(occupancy.mean(dim=(0, 1, 2)))
        statistics = {
            "route_entropy": torch.stack(entropies).mean(),
            "route_balance": load_balance(torch.stack(loads)),
        }
        if return_cache:
            next_cache = GatorLinearBlockCache(
                cache_length + hidden.shape[1],
                block_size,
                tuple(next_relay_caches),
            )
            return hidden, records, statistics, next_cache
        return hidden, records, statistics
