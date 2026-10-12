"""Optional fused packed-writer construction for FFT receiver memory.

The packed receiver writes one hidden vector through ``K`` routes per lane.
The portable Torch equation naturally forms a temporary
``[B,T,L,K,C]`` route-state tensor before scattering it into active nodes.
That tensor is much larger than the final ``[B,C,U,T]`` writer table at the
production route width.

This module keeps the writer boundary explicit:

``torch``
    Portable numerical oracle.  It intentionally favors a direct, readable
    equation over production memory use.

``triton``
    CUDA/float32-only fused construction and explicit adjoint.  Route state is
    multiplied and scattered inside the kernel and is never materialized.

The first hidden-feature chunk may include writer mass in feature zero.  This
lets the caller convolve denominator mass and the first numerator chunk in one
FFT batch without a separate writer-mass tensor or concatenation.
"""

from __future__ import annotations

import math
from typing import Literal

import torch


PackedWriterBackend = Literal["torch", "triton"]


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
_MAX_TRITON_FEATURES = 128


def triton_packed_writer_available() -> bool:
    """Return whether the optional Triton writer can be launched."""

    return _TRITON_AVAILABLE


def _validate_inputs(
    hidden: torch.Tensor,
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    *,
    start: int,
    stop: int,
    active_count: int,
    epsilon: float,
) -> tuple[int, int, int, int, int, int]:
    if hidden.ndim != 4:
        raise ValueError("hidden must have [batch,tokens,lanes,width] shape")
    batch, tokens, lanes, width = hidden.shape
    if write.shape != (batch, tokens):
        raise ValueError("write must have [batch,tokens] shape")
    if compact_ids.ndim != 4 or route_delta.shape != compact_ids.shape:
        raise ValueError(
            "compact_ids and route_delta must share "
            "[batch,tokens,lanes,K] shape"
        )
    if compact_ids.shape[:3] != (batch, tokens, lanes):
        raise ValueError("packed routes must match the hidden prefix")
    routes = compact_ids.shape[-1]
    if min(batch, tokens, lanes, width, routes) <= 0:
        raise ValueError("every packed-writer dimension must be positive")
    if compact_ids.dtype != torch.long:
        raise TypeError("compact_ids must use torch.long")
    if not hidden.is_floating_point():
        raise TypeError("hidden must be floating point")
    if write.dtype != hidden.dtype or route_delta.dtype != hidden.dtype:
        raise TypeError("hidden, write, and route_delta must share a dtype")
    devices = {value.device for value in (hidden, write, compact_ids, route_delta)}
    if len(devices) != 1:
        raise ValueError("every packed-writer input must share a device")
    if isinstance(start, bool) or not isinstance(start, int):
        raise TypeError("start must be an integer")
    if isinstance(stop, bool) or not isinstance(stop, int):
        raise TypeError("stop must be an integer")
    if not 0 <= start < stop <= width:
        raise ValueError("start and stop must select a nonempty hidden chunk")
    if (
        isinstance(active_count, bool)
        or not isinstance(active_count, int)
        or active_count <= 0
    ):
        raise ValueError("active_count must be a positive integer")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a finite positive number")
    torch._assert_async(
        compact_ids.ge(0).logical_and(compact_ids.lt(active_count)).all(),
        "a compact writer node ID is outside the active node table",
    )
    torch._assert_async(
        route_delta.ge(0).all(),
        "packed writer route deltas must be nonnegative",
    )
    return batch, tokens, lanes, width, routes, stop - start


def _expected_output_shape(
    hidden: torch.Tensor,
    *,
    start: int,
    stop: int,
    active_count: int,
    include_mass: bool,
) -> tuple[int, int, int, int]:
    batch, tokens = hidden.shape[:2]
    return (
        batch,
        stop - start + int(include_mass),
        active_count,
        tokens,
    )


def _torch_packed_writer_features_unchecked(
    hidden: torch.Tensor,
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    *,
    start: int,
    stop: int,
    active_count: int,
    epsilon: float,
    include_mass: bool,
) -> torch.Tensor:
    """Evaluate the direct Torch writer equation after validation."""

    batch, tokens, lanes, _ = hidden.shape
    chunk = hidden[..., start:stop]
    chunk_width = stop - start
    baseline_state = (
        epsilon * write[..., None] * chunk.sum(dim=2)
    )[:, :, None, :].expand(batch, tokens, active_count, chunk_width)
    writer_state = baseline_state.clone()
    flat_ids = compact_ids.flatten(2, 3)
    written_delta = route_delta * write[:, :, None, None]
    route_state = (
        written_delta[..., None] * chunk.unsqueeze(3)
    ).flatten(2, 3)
    writer_state.scatter_add_(
        2,
        flat_ids[..., None].expand(-1, -1, -1, chunk_width),
        route_state,
    )
    writer_state = writer_state.permute(0, 3, 2, 1).contiguous()
    if not include_mass:
        return writer_state

    writer_mass = (
        write * (lanes * epsilon)
    )[:, :, None].expand(batch, tokens, active_count).clone()
    writer_mass.scatter_add_(2, flat_ids, written_delta.flatten(2, 3))
    return torch.cat(
        (writer_mass.permute(0, 2, 1)[:, None], writer_state),
        dim=1,
    )


def torch_packed_writer_features(
    hidden: torch.Tensor,
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    *,
    start: int,
    stop: int,
    active_count: int,
    epsilon: float = 1e-9,
    include_mass: bool = False,
) -> torch.Tensor:
    """Return contiguous FFT-native ``[B,C(+1),U,T]`` writer features."""

    _validate_inputs(
        hidden,
        write,
        compact_ids,
        route_delta,
        start=start,
        stop=stop,
        active_count=active_count,
        epsilon=epsilon,
    )
    return _torch_packed_writer_features_unchecked(
        hidden,
        write,
        compact_ids,
        route_delta,
        start=start,
        stop=stop,
        active_count=active_count,
        epsilon=epsilon,
        include_mass=include_mass,
    )


def _torch_packed_writer_adjoint_unchecked(
    hidden: torch.Tensor,
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    grad_features: torch.Tensor,
    *,
    start: int,
    stop: int,
    active_count: int,
    epsilon: float,
    include_mass: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the explicit direct-equation writer adjoint."""

    del active_count  # Encoded by the validated grad_features shape.
    batch, tokens, lanes, _ = hidden.shape
    routes = compact_ids.shape[-1]
    chunk = hidden[..., start:stop]
    state_offset = int(include_mass)
    gradient_by_token = grad_features[:, state_offset:].permute(0, 3, 2, 1)
    gradient_sum = gradient_by_token.sum(dim=2)
    flat_ids = compact_ids.flatten(2, 3)
    selected_gradient = gradient_by_token.gather(
        2,
        flat_ids[..., None].expand(
            -1,
            -1,
            -1,
            stop - start,
        ),
    ).reshape(batch, tokens, lanes, routes, stop - start)
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
        selected_gradient * chunk.unsqueeze(3)
    ).sum(dim=-1)
    grad_route_delta = write[:, :, None, None] * route_score
    grad_write = epsilon * (
        gradient_sum * chunk.sum(dim=2)
    ).sum(dim=-1)
    grad_write = grad_write + (route_delta * route_score).sum(dim=(2, 3))

    if include_mass:
        mass_gradient_by_token = grad_features[:, 0].permute(0, 2, 1)
        selected_mass_gradient = mass_gradient_by_token.gather(
            2,
            flat_ids,
        ).reshape_as(compact_ids)
        grad_route_delta = grad_route_delta + (
            write[:, :, None, None] * selected_mass_gradient
        )
        grad_write = grad_write + (
            lanes * epsilon * mass_gradient_by_token.sum(dim=2)
        )
        grad_write = grad_write + (
            route_delta * selected_mass_gradient
        ).sum(dim=(2, 3))
    return (
        grad_hidden.contiguous(),
        grad_write.contiguous(),
        grad_route_delta.contiguous(),
    )


def _validate_adjoint(
    hidden: torch.Tensor,
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    grad_features: torch.Tensor,
    *,
    start: int,
    stop: int,
    active_count: int,
    epsilon: float,
    include_mass: bool,
) -> None:
    _validate_inputs(
        hidden,
        write,
        compact_ids,
        route_delta,
        start=start,
        stop=stop,
        active_count=active_count,
        epsilon=epsilon,
    )
    expected = _expected_output_shape(
        hidden,
        start=start,
        stop=stop,
        active_count=active_count,
        include_mass=include_mass,
    )
    if grad_features.shape != expected:
        raise ValueError(
            f"grad_features must have writer output shape {expected}"
        )
    if grad_features.dtype != hidden.dtype:
        raise TypeError("grad_features must share the hidden dtype")
    if grad_features.device != hidden.device:
        raise ValueError("grad_features must share the hidden device")


def torch_packed_writer_adjoint(
    hidden: torch.Tensor,
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    grad_features: torch.Tensor,
    *,
    start: int,
    stop: int,
    active_count: int,
    epsilon: float = 1e-9,
    include_mass: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gradients for hidden chunk, write, and route delta."""

    _validate_adjoint(
        hidden,
        write,
        compact_ids,
        route_delta,
        grad_features,
        start=start,
        stop=stop,
        active_count=active_count,
        epsilon=epsilon,
        include_mass=include_mass,
    )
    return _torch_packed_writer_adjoint_unchecked(
        hidden,
        write,
        compact_ids,
        route_delta,
        grad_features,
        start=start,
        stop=stop,
        active_count=active_count,
        epsilon=epsilon,
        include_mass=include_mass,
    )


if _TRITON_AVAILABLE:

    @triton.jit
    def _packed_writer_lane_sum_kernel(
        hidden_ptr,
        lane_sum_ptr,
        stride_hb,
        stride_ht,
        stride_hl,
        stride_hc,
        stride_sb,
        stride_st,
        stride_sc,
        TOKENS: tl.constexpr,
        LANES: tl.constexpr,
        START: tl.constexpr,
        FEATURES: tl.constexpr,
        BLOCK_FEATURES: tl.constexpr,
    ):
        packed_row = tl.program_id(0)
        batch = packed_row // TOKENS
        token = packed_row % TOKENS
        feature = tl.arange(0, BLOCK_FEATURES)
        feature_mask = feature < FEATURES
        accumulator = tl.zeros((BLOCK_FEATURES,), dtype=tl.float32)
        for lane in tl.range(0, LANES):
            accumulator += tl.load(
                hidden_ptr
                + batch * stride_hb
                + token * stride_ht
                + lane * stride_hl
                + (START + feature) * stride_hc,
                mask=feature_mask,
                other=0.0,
            ).to(tl.float32)
        tl.store(
            lane_sum_ptr
            + batch * stride_sb
            + token * stride_st
            + feature * stride_sc,
            accumulator,
            mask=feature_mask,
        )


    @triton.jit
    def _packed_writer_init_kernel(
        lane_sum_ptr,
        write_ptr,
        output_ptr,
        stride_sb,
        stride_st,
        stride_sc,
        stride_wb,
        stride_wt,
        stride_ob,
        stride_oc,
        stride_ou,
        stride_ot,
        TOKENS: tl.constexpr,
        ACTIVE: tl.constexpr,
        LANES: tl.constexpr,
        FEATURES: tl.constexpr,
        EPSILON: tl.constexpr,
        INCLUDE_MASS: tl.constexpr,
        BLOCK_FEATURES: tl.constexpr,
    ):
        packed_row = tl.program_id(0)
        token = packed_row % TOKENS
        active = (packed_row // TOKENS) % ACTIVE
        batch = packed_row // (TOKENS * ACTIVE)
        feature = tl.arange(0, BLOCK_FEATURES)
        feature_mask = feature < FEATURES
        write_value = tl.load(write_ptr + batch * stride_wb + token * stride_wt)
        lane_sum = tl.load(
            lane_sum_ptr
            + batch * stride_sb
            + token * stride_st
            + feature * stride_sc,
            mask=feature_mask,
            other=0.0,
        )
        state_feature = feature + INCLUDE_MASS
        tl.store(
            output_ptr
            + batch * stride_ob
            + state_feature * stride_oc
            + active * stride_ou
            + token * stride_ot,
            write_value * EPSILON * lane_sum,
            mask=feature_mask,
        )
        if INCLUDE_MASS:
            tl.store(
                output_ptr
                + batch * stride_ob
                + active * stride_ou
                + token * stride_ot,
                write_value * LANES * EPSILON,
            )


    @triton.jit
    def _packed_writer_scatter_kernel(
        hidden_ptr,
        write_ptr,
        ids_ptr,
        delta_ptr,
        output_ptr,
        stride_hb,
        stride_ht,
        stride_hl,
        stride_hc,
        stride_wb,
        stride_wt,
        stride_ib,
        stride_it,
        stride_il,
        stride_ik,
        stride_db,
        stride_dt,
        stride_dl,
        stride_dk,
        stride_ob,
        stride_oc,
        stride_ou,
        stride_ot,
        TOKENS: tl.constexpr,
        LANES: tl.constexpr,
        ROUTES: tl.constexpr,
        START: tl.constexpr,
        FEATURES: tl.constexpr,
        INCLUDE_MASS: tl.constexpr,
        BLOCK_FEATURES: tl.constexpr,
    ):
        packed_row = tl.program_id(0)
        lane = packed_row % LANES
        token = (packed_row // LANES) % TOKENS
        batch = packed_row // (TOKENS * LANES)
        feature = tl.arange(0, BLOCK_FEATURES)
        feature_mask = feature < FEATURES
        hidden = tl.load(
            hidden_ptr
            + batch * stride_hb
            + token * stride_ht
            + lane * stride_hl
            + (START + feature) * stride_hc,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        write_value = tl.load(write_ptr + batch * stride_wb + token * stride_wt)
        for route in tl.range(0, ROUTES):
            packed_offset = (
                batch * stride_ib
                + token * stride_it
                + lane * stride_il
                + route * stride_ik
            )
            active = tl.load(ids_ptr + packed_offset)
            delta = tl.load(
                delta_ptr
                + batch * stride_db
                + token * stride_dt
                + lane * stride_dl
                + route * stride_dk
            )
            scale = write_value * delta
            tl.atomic_add(
                output_ptr
                + batch * stride_ob
                + (feature + INCLUDE_MASS) * stride_oc
                + active * stride_ou
                + token * stride_ot,
                scale * hidden,
                mask=feature_mask,
            )
            if INCLUDE_MASS:
                tl.atomic_add(
                    output_ptr
                    + batch * stride_ob
                    + active * stride_ou
                    + token * stride_ot,
                    scale,
                )


    @triton.jit
    def _packed_writer_gradient_sum_kernel(
        grad_features_ptr,
        gradient_sum_ptr,
        mass_sum_ptr,
        stride_gb,
        stride_gc,
        stride_gu,
        stride_gt,
        stride_sb,
        stride_st,
        stride_sc,
        stride_mb,
        stride_mt,
        TOKENS: tl.constexpr,
        ACTIVE: tl.constexpr,
        FEATURES: tl.constexpr,
        INCLUDE_MASS: tl.constexpr,
        BLOCK_FEATURES: tl.constexpr,
    ):
        packed_row = tl.program_id(0)
        batch = packed_row // TOKENS
        token = packed_row % TOKENS
        feature = tl.arange(0, BLOCK_FEATURES)
        feature_mask = feature < FEATURES
        state_sum = tl.zeros((BLOCK_FEATURES,), dtype=tl.float32)
        mass_sum = 0.0
        for active in tl.range(0, ACTIVE):
            state_sum += tl.load(
                grad_features_ptr
                + batch * stride_gb
                + (feature + INCLUDE_MASS) * stride_gc
                + active * stride_gu
                + token * stride_gt,
                mask=feature_mask,
                other=0.0,
            ).to(tl.float32)
            if INCLUDE_MASS:
                mass_sum += tl.load(
                    grad_features_ptr
                    + batch * stride_gb
                    + active * stride_gu
                    + token * stride_gt
                ).to(tl.float32)
        tl.store(
            gradient_sum_ptr
            + batch * stride_sb
            + token * stride_st
            + feature * stride_sc,
            state_sum,
            mask=feature_mask,
        )
        tl.store(
            mass_sum_ptr + batch * stride_mb + token * stride_mt,
            mass_sum,
        )


    @triton.jit
    def _packed_writer_adjoint_kernel(
        hidden_ptr,
        write_ptr,
        ids_ptr,
        delta_ptr,
        grad_features_ptr,
        gradient_sum_ptr,
        mass_sum_ptr,
        grad_hidden_ptr,
        grad_write_lane_ptr,
        grad_delta_ptr,
        stride_hb,
        stride_ht,
        stride_hl,
        stride_hc,
        stride_wb,
        stride_wt,
        stride_ib,
        stride_it,
        stride_il,
        stride_ik,
        stride_db,
        stride_dt,
        stride_dl,
        stride_dk,
        stride_gb,
        stride_gc,
        stride_gu,
        stride_gt,
        stride_sb,
        stride_st,
        stride_sc,
        stride_mb,
        stride_mt,
        stride_ghb,
        stride_ght,
        stride_ghl,
        stride_ghc,
        stride_gwb,
        stride_gwt,
        stride_gwl,
        stride_gdb,
        stride_gdt,
        stride_gdl,
        stride_gdk,
        TOKENS: tl.constexpr,
        LANES: tl.constexpr,
        ROUTES: tl.constexpr,
        START: tl.constexpr,
        FEATURES: tl.constexpr,
        EPSILON: tl.constexpr,
        INCLUDE_MASS: tl.constexpr,
        BLOCK_FEATURES: tl.constexpr,
    ):
        packed_row = tl.program_id(0)
        lane = packed_row % LANES
        token = (packed_row // LANES) % TOKENS
        batch = packed_row // (TOKENS * LANES)
        feature = tl.arange(0, BLOCK_FEATURES)
        feature_mask = feature < FEATURES
        hidden = tl.load(
            hidden_ptr
            + batch * stride_hb
            + token * stride_ht
            + lane * stride_hl
            + (START + feature) * stride_hc,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        gradient_sum = tl.load(
            gradient_sum_ptr
            + batch * stride_sb
            + token * stride_st
            + feature * stride_sc,
            mask=feature_mask,
            other=0.0,
        ).to(tl.float32)
        write_value = tl.load(write_ptr + batch * stride_wb + token * stride_wt)
        grad_hidden = EPSILON * write_value * gradient_sum
        grad_write_lane = EPSILON * tl.sum(hidden * gradient_sum, axis=0)
        if INCLUDE_MASS:
            grad_write_lane += EPSILON * tl.load(
                mass_sum_ptr + batch * stride_mb + token * stride_mt
            )

        for route in tl.range(0, ROUTES):
            packed_offset = (
                batch * stride_ib
                + token * stride_it
                + lane * stride_il
                + route * stride_ik
            )
            active = tl.load(ids_ptr + packed_offset)
            delta = tl.load(
                delta_ptr
                + batch * stride_db
                + token * stride_dt
                + lane * stride_dl
                + route * stride_dk
            )
            selected_gradient = tl.load(
                grad_features_ptr
                + batch * stride_gb
                + (feature + INCLUDE_MASS) * stride_gc
                + active * stride_gu
                + token * stride_gt,
                mask=feature_mask,
                other=0.0,
            ).to(tl.float32)
            score = tl.sum(hidden * selected_gradient, axis=0)
            grad_hidden += write_value * delta * selected_gradient
            mass_gradient = 0.0
            if INCLUDE_MASS:
                mass_gradient = tl.load(
                    grad_features_ptr
                    + batch * stride_gb
                    + active * stride_gu
                    + token * stride_gt
                ).to(tl.float32)
            total_score = score + mass_gradient
            grad_write_lane += delta * total_score
            tl.store(
                grad_delta_ptr
                + batch * stride_gdb
                + token * stride_gdt
                + lane * stride_gdl
                + route * stride_gdk,
                write_value * total_score,
            )

        tl.store(
            grad_hidden_ptr
            + batch * stride_ghb
            + token * stride_ght
            + lane * stride_ghl
            + feature * stride_ghc,
            grad_hidden,
            mask=feature_mask,
        )
        tl.store(
            grad_write_lane_ptr
            + batch * stride_gwb
            + token * stride_gwt
            + lane * stride_gwl,
            grad_write_lane,
        )


    @triton.jit
    def _packed_writer_write_reduce_kernel(
        grad_write_lane_ptr,
        grad_write_ptr,
        stride_lb,
        stride_lt,
        stride_ll,
        stride_wb,
        stride_wt,
        TOKENS: tl.constexpr,
        LANES: tl.constexpr,
    ):
        packed_row = tl.program_id(0)
        batch = packed_row // TOKENS
        token = packed_row % TOKENS
        total = 0.0
        for lane in tl.range(0, LANES):
            total += tl.load(
                grad_write_lane_ptr
                + batch * stride_lb
                + token * stride_lt
                + lane * stride_ll
            ).to(tl.float32)
        tl.store(
            grad_write_ptr + batch * stride_wb + token * stride_wt,
            total,
        )


    def _feature_block_size(features: int) -> int:
        return triton.next_power_of_2(features)


    @triton_op("gator::packed_writer_features", mutates_args={})
    def _triton_packed_writer_features(
        hidden: torch.Tensor,
        write: torch.Tensor,
        compact_ids: torch.Tensor,
        route_delta: torch.Tensor,
        start: int,
        stop: int,
        active_count: int,
        epsilon: float,
        include_mass: bool,
    ) -> torch.Tensor:
        batch, tokens, lanes, _ = hidden.shape
        routes = compact_ids.shape[-1]
        features = stop - start
        output = torch.empty(
            (batch, features + int(include_mass), active_count, tokens),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        lane_sum = torch.empty(
            (batch, tokens, features),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        block_features = _feature_block_size(features)
        wrap_triton(_packed_writer_lane_sum_kernel)[(batch * tokens,)](
            hidden,
            lane_sum,
            *hidden.stride(),
            *lane_sum.stride(),
            TOKENS=tokens,
            LANES=lanes,
            START=start,
            FEATURES=features,
            BLOCK_FEATURES=block_features,
            num_warps=4,
        )
        wrap_triton(_packed_writer_init_kernel)[
            (batch * active_count * tokens,)
        ](
            lane_sum,
            write,
            output,
            *lane_sum.stride(),
            *write.stride(),
            *output.stride(),
            TOKENS=tokens,
            ACTIVE=active_count,
            LANES=lanes,
            FEATURES=features,
            EPSILON=epsilon,
            INCLUDE_MASS=include_mass,
            BLOCK_FEATURES=block_features,
            num_warps=4,
        )
        wrap_triton(_packed_writer_scatter_kernel)[
            (batch * tokens * lanes,)
        ](
            hidden,
            write,
            compact_ids,
            route_delta,
            output,
            *hidden.stride(),
            *write.stride(),
            *compact_ids.stride(),
            *route_delta.stride(),
            *output.stride(),
            TOKENS=tokens,
            LANES=lanes,
            ROUTES=routes,
            START=start,
            FEATURES=features,
            INCLUDE_MASS=include_mass,
            BLOCK_FEATURES=block_features,
            num_warps=4,
        )
        return output


    @triton_op("gator::packed_writer_adjoint", mutates_args={})
    def _triton_packed_writer_adjoint(
        hidden: torch.Tensor,
        write: torch.Tensor,
        compact_ids: torch.Tensor,
        route_delta: torch.Tensor,
        grad_features: torch.Tensor,
        start: int,
        stop: int,
        active_count: int,
        epsilon: float,
        include_mass: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, tokens, lanes, _ = hidden.shape
        routes = compact_ids.shape[-1]
        features = stop - start
        block_features = _feature_block_size(features)
        gradient_sum = torch.empty(
            (batch, tokens, features),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        mass_sum = torch.empty(
            (batch, tokens),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        grad_hidden = torch.empty(
            (batch, tokens, lanes, features),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        grad_write_lane = torch.empty(
            (batch, tokens, lanes),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        grad_write = torch.empty_like(write)
        grad_route_delta = torch.empty_like(route_delta)
        wrap_triton(_packed_writer_gradient_sum_kernel)[(batch * tokens,)](
            grad_features,
            gradient_sum,
            mass_sum,
            *grad_features.stride(),
            *gradient_sum.stride(),
            *mass_sum.stride(),
            TOKENS=tokens,
            ACTIVE=active_count,
            FEATURES=features,
            INCLUDE_MASS=include_mass,
            BLOCK_FEATURES=block_features,
            num_warps=4,
        )
        wrap_triton(_packed_writer_adjoint_kernel)[
            (batch * tokens * lanes,)
        ](
            hidden,
            write,
            compact_ids,
            route_delta,
            grad_features,
            gradient_sum,
            mass_sum,
            grad_hidden,
            grad_write_lane,
            grad_route_delta,
            *hidden.stride(),
            *write.stride(),
            *compact_ids.stride(),
            *route_delta.stride(),
            *grad_features.stride(),
            *gradient_sum.stride(),
            *mass_sum.stride(),
            *grad_hidden.stride(),
            *grad_write_lane.stride(),
            *grad_route_delta.stride(),
            TOKENS=tokens,
            LANES=lanes,
            ROUTES=routes,
            START=start,
            FEATURES=features,
            EPSILON=epsilon,
            INCLUDE_MASS=include_mass,
            BLOCK_FEATURES=block_features,
            num_warps=4,
        )
        wrap_triton(_packed_writer_write_reduce_kernel)[(batch * tokens,)](
            grad_write_lane,
            grad_write,
            *grad_write_lane.stride(),
            *grad_write.stride(),
            TOKENS=tokens,
            LANES=lanes,
            num_warps=1,
        )
        return grad_hidden, grad_write, grad_route_delta


    def _setup_triton_writer_context(ctx, inputs, output) -> None:
        del output
        (
            hidden,
            write,
            compact_ids,
            route_delta,
            start,
            stop,
            active_count,
            epsilon,
            include_mass,
        ) = inputs
        ctx.save_for_backward(hidden, write, compact_ids, route_delta)
        ctx.start = start
        ctx.stop = stop
        ctx.active_count = active_count
        ctx.epsilon = epsilon
        ctx.include_mass = include_mass


    def _triton_writer_backward(ctx, grad_features):
        hidden, write, compact_ids, route_delta = ctx.saved_tensors
        grad_hidden_chunk, grad_write, grad_route_delta = (
            _triton_packed_writer_adjoint(
                hidden,
                write,
                compact_ids,
                route_delta,
                grad_features,
                ctx.start,
                ctx.stop,
                ctx.active_count,
                ctx.epsilon,
                ctx.include_mass,
            )
        )
        grad_hidden = torch.zeros_like(
            hidden,
            memory_format=torch.contiguous_format,
        )
        grad_hidden[..., ctx.start : ctx.stop] = grad_hidden_chunk
        return (
            grad_hidden,
            grad_write,
            None,
            grad_route_delta,
            None,
            None,
            None,
            None,
            None,
        )


    _triton_packed_writer_features.register_autograd(
        _triton_writer_backward,
        setup_context=_setup_triton_writer_context,
    )

else:
    _triton_packed_writer_features = None
    _triton_packed_writer_adjoint = None


def _triton_unavailable_message() -> str:
    detail = ""
    if _TRITON_IMPORT_ERROR is not None:
        detail = f": {_TRITON_IMPORT_ERROR}"
    return (
        "packed writer backend='triton' requires Triton plus "
        f"torch.library.triton_op/wrap_triton support{detail}"
    )


def _require_triton_contract(hidden: torch.Tensor, features: int) -> None:
    if hidden.device.type != "cuda":
        raise ValueError("packed writer backend='triton' requires CUDA tensors")
    if hidden.dtype != torch.float32:
        raise TypeError(
            "packed writer backend='triton' currently supports float32 only"
        )
    if features > _MAX_TRITON_FEATURES:
        raise ValueError(
            "packed writer backend='triton' currently supports feature "
            f"chunks up to {_MAX_TRITON_FEATURES}"
        )
    if not _TRITON_AVAILABLE or _triton_packed_writer_features is None:
        raise RuntimeError(_triton_unavailable_message())


def _packed_writer_features_unchecked(
    hidden: torch.Tensor,
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    *,
    start: int,
    stop: int,
    active_count: int,
    epsilon: float,
    include_mass: bool,
    backend: PackedWriterBackend,
) -> torch.Tensor:
    if backend == "torch":
        return _torch_packed_writer_features_unchecked(
            hidden,
            write,
            compact_ids,
            route_delta,
            start=start,
            stop=stop,
            active_count=active_count,
            epsilon=epsilon,
            include_mass=include_mass,
        )
    if backend != "triton":
        raise ValueError("packed writer backend must be 'torch' or 'triton'")
    _require_triton_contract(hidden, stop - start)
    if _triton_packed_writer_features is None:
        raise RuntimeError(_triton_unavailable_message())
    return _triton_packed_writer_features(
        hidden,
        write,
        compact_ids,
        route_delta,
        start,
        stop,
        active_count,
        float(epsilon),
        include_mass,
    )


def packed_writer_features(
    hidden: torch.Tensor,
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    *,
    start: int,
    stop: int,
    active_count: int,
    epsilon: float = 1e-9,
    include_mass: bool = False,
    backend: PackedWriterBackend = "torch",
) -> torch.Tensor:
    """Construct packed writer features through the selected backend."""

    _validate_inputs(
        hidden,
        write,
        compact_ids,
        route_delta,
        start=start,
        stop=stop,
        active_count=active_count,
        epsilon=epsilon,
    )
    return _packed_writer_features_unchecked(
        hidden,
        write,
        compact_ids,
        route_delta,
        start=start,
        stop=stop,
        active_count=active_count,
        epsilon=epsilon,
        include_mass=include_mass,
        backend=backend,
    )


def _packed_writer_adjoint_unchecked(
    hidden: torch.Tensor,
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    grad_features: torch.Tensor,
    *,
    start: int,
    stop: int,
    active_count: int,
    epsilon: float,
    include_mass: bool,
    backend: PackedWriterBackend,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if backend == "torch":
        return _torch_packed_writer_adjoint_unchecked(
            hidden,
            write,
            compact_ids,
            route_delta,
            grad_features,
            start=start,
            stop=stop,
            active_count=active_count,
            epsilon=epsilon,
            include_mass=include_mass,
        )
    if backend != "triton":
        raise ValueError("packed writer backend must be 'torch' or 'triton'")
    _require_triton_contract(hidden, stop - start)
    if _triton_packed_writer_adjoint is None:
        raise RuntimeError(_triton_unavailable_message())
    return _triton_packed_writer_adjoint(
        hidden,
        write,
        compact_ids,
        route_delta,
        grad_features,
        start,
        stop,
        active_count,
        float(epsilon),
        include_mass,
    )


def packed_writer_adjoint(
    hidden: torch.Tensor,
    write: torch.Tensor,
    compact_ids: torch.Tensor,
    route_delta: torch.Tensor,
    grad_features: torch.Tensor,
    *,
    start: int,
    stop: int,
    active_count: int,
    epsilon: float = 1e-9,
    include_mass: bool = False,
    backend: PackedWriterBackend = "torch",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the explicit writer adjoint through the selected backend.

    Returns ``(grad_hidden_chunk, grad_write, grad_route_delta)``.  Hidden
    features outside ``[start:stop]`` do not participate in this invocation.
    """

    _validate_adjoint(
        hidden,
        write,
        compact_ids,
        route_delta,
        grad_features,
        start=start,
        stop=stop,
        active_count=active_count,
        epsilon=epsilon,
        include_mass=include_mass,
    )
    return _packed_writer_adjoint_unchecked(
        hidden,
        write,
        compact_ids,
        route_delta,
        grad_features,
        start=start,
        stop=stop,
        active_count=active_count,
        epsilon=epsilon,
        include_mass=include_mass,
        backend=backend,
    )


__all__ = [
    "PackedWriterBackend",
    "packed_writer_adjoint",
    "packed_writer_features",
    "torch_packed_writer_adjoint",
    "torch_packed_writer_features",
    "triton_packed_writer_available",
]
