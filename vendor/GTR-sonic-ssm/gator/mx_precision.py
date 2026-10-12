"""Optional MXFP8/MXFP4/NVFP4 low-precision compute for GTR-ARW's most
expensive matmuls: the interior per-cell node compute and the vocabulary
readout. Routing (the router/edge-logit computation in
gator/models/language.py's ``_dense_relay`` and matching stewardship
paths) never routes through this module and always stays bf16 -- that
was an explicit scope decision, not an oversight.

Block size and scale format follow the OCP Microscaling spec as
documented by NVIDIA's Transformer Engine MXFP8 guide: 32-element 1D
blocks, an E8M0 (8 exponent bits, 0 mantissa, power-of-two-only) shared
scale per block, computed from each block's max-abs value
(https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html).

Requires torchao (not a hard dependency of this repo -- only imported
when a non-bf16 precision is actually requested) and a CUDA device.
Despite the module being developed and first verified in a dedicated
Python 3.10 + torch 2.13.0+cu130 + torchao 0.18.0 venv (needed at the
time to chase torchao's compiled hardware-kernel extensions, whose
prebuilt wheel is cpython-310-specific and requires CUDA 13's
libcudart -- see Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md for
that full story), the working code path in this file (straight-through
fake-quantization via ``MXTensor.to_mx()``/``.dequantize()``) does
*not* actually need those compiled kernels and has been confirmed
working directly in this repo's regular main venv too (Python 3.14).
Both import failures and non-functional-but-importable installs (the
compiled-kernel case) are caught and reported clearly at the point of
use via ``_require_torchao()``, not at module import time, so the rest
of the trainer keeps working when these precisions aren't requested.

Node compute (interior, grouped over cells) and readout (a single
non-grouped matmul) both currently use straight-through
fake-quantization for every precision (MXFP8, MXFP4, NVFP4-approximated):
quantize inputs/weight to the target format, dequantize back to bf16,
run the real matmul in bf16 through the existing dense kernels, with an
explicit straight-through-estimator gradient
(``x + (quantize(x) - x).detach()``). This is standard
quantization-aware-training practice -- correct gradients, a real read
on each format's precision/outlier behavior -- but **no raw hardware
speedup** for any of the three, which matters when reading throughput
numbers from a run using this module.

Two genuinely hardware-accelerated paths were attempted and are kept in
this file (prefixed ``_..._hardware``/``_MXReadoutMatmulHardware``) for
future revisiting, but neither works yet:
- Node compute MXFP8 via torchao's real grouped GEMM
  (``torchao.prototype.moe_training.mxfp8_grouped_mm``, built for MoE
  expert-grouped matmul -- structurally identical to our per-cell
  grouped transform) hits a ``to_mx()`` contiguity assertion inside
  torchao 0.18.0's own EMULATED fallback that a `.contiguous()` call on
  our side doesn't resolve, in both AUTO/SM100-kernel and EMULATED
  modes, with or without ``pad_token_groups_for_grouped_mm``.
- Readout via direct ``torch._scaled_mm``/``F.scaled_mm`` calls (the
  exact calling convention verified against torchao's own
  ``test/prototype/mx_formats/test_mx_mm.py``) raises
  ``CUDA error: CUBLAS_STATUS_NOT_SUPPORTED``.
- NVFP4 additionally has no grouped hardware kernel in torchao at all
  yet (checked directly: only an "emulated" dequantize-then-dense-matmul
  reference implementation exists, no autograd.Function, no CUDA/Triton
  kernel) -- node compute for it approximates NVFP4's 16-element/FP8-scale
  block using MXFP4's element format at NVFP4's block size, which is
  *not* a true NVFP4 implementation, only an approximation of its block
  granularity.
"""

from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn.functional as F

try:
    from torch.nn.functional import ScalingType, SwizzleType
except ImportError:  # pragma: no cover - exercised only on older torch
    ScalingType = None  # type: ignore[assignment]
    SwizzleType = None  # type: ignore[assignment]

NodeComputePrecision = Literal["bf16", "mxfp8", "nvfp4", "fp8_te"]
ReadoutPrecision = Literal["bf16", "mxfp8", "mxfp4", "fp8_te"]

NODE_COMPUTE_PRECISIONS: tuple[str, ...] = ("bf16", "mxfp8", "nvfp4", "fp8_te")
READOUT_PRECISIONS: tuple[str, ...] = ("bf16", "mxfp8", "mxfp4", "fp8_te")

MX_BLOCK_SIZE = 32

try:
    from torchao.prototype.mx_formats.mx_tensor import MXTensor
    from torchao.prototype.mx_formats.utils import to_blocked
    from torchao.prototype.moe_training.mxfp8_grouped_mm import (
        _to_mxfp8_then_scaled_grouped_mm,
    )
    from torchao.quantization.quantize_.common import KernelPreference

    _TORCHAO_IMPORT_ERROR: Exception | None = None
except Exception as error:  # pragma: no cover - exercised only without torchao
    MXTensor = None  # type: ignore[assignment]
    to_blocked = None  # type: ignore[assignment]
    _to_mxfp8_then_scaled_grouped_mm = None  # type: ignore[assignment]
    KernelPreference = None  # type: ignore[assignment]
    _TORCHAO_IMPORT_ERROR = error

try:
    import transformer_engine.pytorch as _te
    import transformer_engine.common.recipe as _te_recipe

    _TRANSFORMER_ENGINE_IMPORT_ERROR: Exception | None = None
except Exception as error:  # pragma: no cover - exercised only without transformer_engine
    _te = None  # type: ignore[assignment]
    _te_recipe = None  # type: ignore[assignment]
    _TRANSFORMER_ENGINE_IMPORT_ERROR = error


_TORCHAO_FUNCTIONAL_ERROR: Exception | None | Literal["unchecked"] = "unchecked"


def _torchao_functional_error() -> Exception | None:
    """Cache and return whether torchao's compiled kernels actually load.

    A plain Python import can succeed (as it does in this repo's main
    venv, where torchao happens to be pip-installed as a leftover from
    initial setup) while the compiled CUDA extensions underneath still
    fail to load -- e.g. built for a different cpython ABI than the
    running interpreter. That failure prints a warning to stderr but
    does not raise ImportError, so it would otherwise surface as a
    confusing failure deep inside MXTensor internals instead of a clear
    message here. Runs one tiny real quantize call to check for real,
    cached after the first call since it's not free.
    """

    global _TORCHAO_FUNCTIONAL_ERROR
    if _TORCHAO_FUNCTIONAL_ERROR != "unchecked":
        return _TORCHAO_FUNCTIONAL_ERROR
    try:
        probe = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
        MXTensor.to_mx(probe, torch.float8_e4m3fn, MX_BLOCK_SIZE)
        _TORCHAO_FUNCTIONAL_ERROR = None
    except Exception as error:  # noqa: BLE001 - deliberately broad, see docstring
        _TORCHAO_FUNCTIONAL_ERROR = error
    return _TORCHAO_FUNCTIONAL_ERROR


def _require_torchao(precision: str) -> None:
    if _TORCHAO_IMPORT_ERROR is not None:
        raise RuntimeError(
            f"precision {precision!r} requires torchao and an SM100+ "
            "(Blackwell) GPU; import failed in this environment with: "
            f"{_TORCHAO_IMPORT_ERROR!r}. See "
            "Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md for the "
            "known-working torchao environment (Python 3.10, "
            "torch 2.13.0+cu130)."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"precision {precision!r} requires a CUDA device; none available"
        )
    functional_error = _torchao_functional_error()
    if functional_error is not None:
        raise RuntimeError(
            f"precision {precision!r}: torchao imported but its compiled "
            "kernels failed a functional probe (likely built for a "
            f"different cpython ABI than this interpreter): {functional_error!r}. "
            "See Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md for "
            "the known-working torchao environment (Python 3.10, "
            "torch 2.13.0+cu130)."
        )


def _require_transformer_engine() -> None:
    if _TRANSFORMER_ENGINE_IMPORT_ERROR is not None:
        raise RuntimeError(
            "precision 'fp8_te' requires NVIDIA Transformer Engine and a "
            f"CUDA device; import failed with: {_TRANSFORMER_ENGINE_IMPORT_ERROR!r}. "
            "See Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md for the "
            "known-working environment (Python 3.10 venv, CUDA Toolkit "
            "12.8 + cuDNN + libnccl-dev installed in WSL)."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("precision 'fp8_te' requires a CUDA device; none available")


class _FP8CellBlock(torch.nn.Module):
    """``cells`` independent ``te.Linear`` layers, one per cell.

    Mirrors ``cell_linear``'s per-cell weight convention ([cells,
    out_features, in_features]). Wrapped by
    ``torch.cuda.make_graphed_callables`` at construction time in
    ``te_fp8_cudagraph_node_linear`` below -- its parameters get
    value-copied from the model's real (freshly folded) node weight on
    every call via ``_BridgeFP8CellWeights``, since ``fold_cell_node_
    parameters`` produces a fresh tensor each forward pass rather than a
    persistent ``nn.Parameter`` CUDA graphs could reference directly.
    """

    def __init__(
        self,
        cells: int,
        in_features: int,
        out_features: int,
        has_bias: bool,
        recipe: "_te_recipe.DelayedScaling",
    ) -> None:
        super().__init__()
        self.cells = cells
        self.has_bias = has_bias
        self.recipe = recipe
        self.linears = torch.nn.ModuleList(
            [
                _te.Linear(in_features, out_features, bias=has_bias)
                for _ in range(cells)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _te.autocast(enabled=True, recipe=self.recipe):
            outputs = [self.linears[c](x[c]) for c in range(self.cells)]
        return torch.stack(outputs, dim=0)


class _BridgeFP8CellWeights(torch.autograd.Function):
    """Route the live folded node weight's value/gradient through ``_FP8CellBlock``.

    ``_FP8CellBlock``'s per-cell ``te.Linear.weight``/``.bias`` are the
    only tensors CUDA-graph replay can read updated values from (fixed
    memory addresses); the model's actual node weight
    (``fold_cell_node_parameters``'s output) is a fresh tensor each
    forward call and can't be used directly. This bridges the two:
    forward copies ``weight``/``bias`` values into the block's
    parameters (a plain value copy, invisible to autograd) and passes
    ``inputs`` through unchanged so the real dependency chain (and
    therefore correct backward ordering relative to
    ``graphed_module``'s own backward, which populates
    ``self.linears[c].weight.grad``) is preserved; backward reads those
    accumulated ``.grad`` tensors back out as this function's gradient
    w.r.t. the original ``weight``/``bias`` inputs. Verified correct via
    a real upstream-op -> graphed FP8 layer -> downstream-op gradient
    chain, not just an isolated forward/backward check (see
    ``Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md``).
    """

    @staticmethod
    def forward(ctx, inputs, weight, bias, raw_module, cells, has_bias):  # type: ignore[no-untyped-def]
        with torch.no_grad():
            for c in range(cells):
                raw_module.linears[c].weight.copy_(weight[c])
                raw_module.linears[c].weight.grad = None
                if has_bias:
                    raw_module.linears[c].bias.copy_(bias[c])
                    raw_module.linears[c].bias.grad = None
        ctx.raw_module = raw_module
        ctx.cells = cells
        ctx.has_bias = has_bias
        return inputs

    @staticmethod
    def backward(ctx, grad_inputs):  # type: ignore[no-untyped-def]
        raw_module = ctx.raw_module
        grad_weight = torch.stack(
            [
                raw_module.linears[c].weight.grad.clone()
                for c in range(ctx.cells)
            ]
        )
        grad_bias = None
        if ctx.has_bias:
            grad_bias = torch.stack(
                [raw_module.linears[c].bias.grad.clone() for c in range(ctx.cells)]
            )
        return grad_inputs, grad_weight, grad_bias, None, None, None


_FP8_CUDAGRAPH_CACHE: dict[tuple, tuple[_FP8CellBlock, Callable]] = {}


def te_fp8_cudagraph_node_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    graph_slot: int = 0,
) -> torch.Tensor:
    """Real hardware-accelerated FP8 grouped node compute via NVIDIA
    Transformer Engine, with CUDA-graph capture to eliminate the per-call
    dispatch overhead that otherwise makes plain eager FP8 ~2x SLOWER
    than bf16 at this model's node-compute shape.

    ``graph_slot`` -- **required to be distinct per call-site when this
    function is called more than once before a shared backward() fires**
    (e.g. once per relay, this model's real usage pattern with
    ``relays>1``). Found via direct test: a single graphed instance
    shared across two forward calls silently corrupts backward's
    gradient for a *different, more severe* reason than the output-
    buffer-aliasing bug this file already fixes with ``.clone()`` --
    even with that fix applied, a second call's forward before the
    first's backward runs leaves the wrapped module's parameter
    ``.grad`` as ``None`` when the bridge Function's backward tries to
    read it (crashes) or, in variants without the crash, silently
    produces wrong gradients. Verified fix: give each call-site its own
    dedicated graphed instance rather than sharing one -- ``graph_slot``
    is folded into the shape-cache key for exactly this. The caller is
    responsible for passing a stable, distinct slot per call-site (e.g.
    the relay loop index) -- this function has no way to detect a
    missing or colliding slot itself.

    **Only wins at large-enough shapes.** Measured directly at this
    model's exact production node-compute shape (rows = batch*tokens =
    4,128, d_model = 1,312): plain eager FP8 is genuinely ~2x slower
    than bf16 (per-call amax/scale-update bookkeeping dominates at this
    size) -- graphing an *isolated* FP8 layer at that same shape
    recovers a real 1.4-2x speedup over bf16 by collapsing the many
    small per-call kernel launches into one CUDA-graph replay. **But**
    that gain is a CPU-dispatch-overhead saving specifically, and when
    this layer runs embedded among plenty of *other* un-graphed GPU work
    (i.e. inside a real training step, surrounded by every other layer
    in the model), the GPU is already kept busy by that surrounding
    work, so the isolated layer's own launch-count reduction stops
    being the bottleneck and the measured end-to-end gain shrinks toward
    zero (confirmed directly: ~1.0x in an upstream-matmul ->
    this-layer -> downstream-matmul benchmark, versus ~1.4-2x for the
    same layer measured alone). A real end-to-end production win likely
    requires graphing a much larger contiguous span of the model (up to
    and including the whole forward+backward, which is Item 1 in
    ``Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md`` and has its
    own separate, unresolved blocker), not just this one layer -- use
    this function accordingly, and re-measure end-to-end rather than
    assuming the isolated-layer number transfers.

    Uses TE's ``DelayedScaling`` recipe (plain scalar E4M3/E5M2 FP8,
    real hardware tensor-core path on this GPU -- MXFP8/NVFP4 block-
    scaled formats are separately gated on Blackwell consumer silicon,
    see the same runbook). Correctness (finite forward/backward,
    gradients flow to both ``inputs`` and ``weight`` through a real
    surrounding upstream/downstream computation, not just in isolation)
    verified via ``tests/test_mx_precision.py``.

    Shape-cached per ``(cells, rows, in_features, out_features, has_bias,
    device)`` -- the first call for a given shape pays graph-capture
    warmup cost (a handful of real forward+backward iterations); later
    calls with the same shape reuse the captured graph. Batch size and
    context length must stay fixed across calls sharing a cache entry,
    same requirement as any CUDA-graph-captured code.

    **Cold-start accuracy warning, measured directly:** ``DelayedScaling``
    computes its FP8 scale from an amax history that starts empty, so the
    very first real call for a given shape produces a badly wrong scale
    (measured: output magnitude ~10x too small vs. the bf16 reference).
    This is inherent to the recipe -- an empty history has nothing to
    compute a real scale from yet -- not specific to this CUDA-graph
    wrapper, and self-corrects from the second call onward (steady-state
    mean absolute error measured at ~0.6% of the reference's own scale,
    ordinary FP8 (E4M3) precision loss, not a bug). Concretely: the very
    first training step after this layer's graph is (re)built will have
    a measurably wrong node-compute output; every step after that is
    fine. Not addressed here since the "fix" (running throwaway
    calibration passes on real data before considering the layer ready)
    would need real training data at construction time, not just a
    zero-tensor shape probe -- left for the caller to handle explicitly
    if a clean first step matters (e.g. running one dummy real-shaped
    batch through before the real first optimizer step).
    """

    _require_transformer_engine()
    if inputs.ndim != 4:
        raise ValueError("fp8_te node-linear inputs must be [batch,cells,tokens,in]")
    if weight.ndim != 3:
        raise ValueError("fp8_te node-linear weight must be [cells,out,in]")
    batch, cells, tokens, in_features = inputs.shape
    if weight.shape[0] != cells or weight.shape[2] != in_features:
        raise ValueError("fp8_te node-linear weight must match input cells/features")
    out_features = weight.shape[1]
    has_bias = bias is not None
    rows = batch * tokens

    key = (
        cells, rows, in_features, out_features, has_bias, inputs.device, graph_slot,
    )
    cached = _FP8_CUDAGRAPH_CACHE.get(key)
    if cached is None:
        raw_module = _FP8CellBlock(
            cells, in_features, out_features, has_bias, _te_recipe.DelayedScaling()
        ).to(device=inputs.device, dtype=torch.bfloat16)
        sample = torch.zeros(
            cells, rows, in_features,
            device=inputs.device, dtype=torch.bfloat16, requires_grad=True,
        )
        # make_graphed_callables explicitly rejects being called from
        # inside an active autocast context with caching enabled
        # ("does not support the autocast caching") -- real production
        # training always wraps the whole forward pass in
        # torch.autocast(bf16), so capture would otherwise crash on the
        # very first real training step. Suppress autocast for just this
        # one-time capture; replay doesn't go through autocast-affected
        # op dispatch (it's a raw kernel replay), so it doesn't need this.
        with torch.autocast(device_type="cuda", enabled=False):
            graphed_module = torch.cuda.make_graphed_callables(
                raw_module, (sample,), num_warmup_iters=5
            )
        cached = (raw_module, graphed_module)
        _FP8_CUDAGRAPH_CACHE[key] = cached
    raw_module, graphed_module = cached

    grouped_inputs = (
        inputs.permute(1, 0, 2, 3).reshape(cells, rows, in_features).to(torch.bfloat16)
    )
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16) if has_bias else None

    bridged_inputs = _BridgeFP8CellWeights.apply(
        grouped_inputs, weight_bf16, bias_bf16, raw_module, cells, has_bias
    )
    # .clone() is required, not optional: make_graphed_callables' output is a
    # live view into the CUDA graph's static output buffer, silently
    # overwritten by the NEXT call to this same graphed callable (e.g. the
    # next relay's node-compute call in the same forward pass, before this
    # relay's activation has been consumed by backward()) -- without this,
    # multi-relay training would silently corrupt earlier relays' gradients
    # rather than crash, which is a worse failure mode than an error.
    grouped_output = graphed_module(bridged_inputs).clone()
    output = grouped_output.reshape(cells, batch, tokens, out_features).permute(
        1, 0, 2, 3
    )
    return output.to(inputs.dtype)


def _fake_quant_round_trip(
    x: torch.Tensor,
    elem_dtype: torch.dtype,
    *,
    block_size: int = MX_BLOCK_SIZE,
) -> torch.Tensor:
    """Quantize to an MX-family format and immediately dequantize.

    Explicit straight-through estimator: ``output = x + (quantize(x) -
    x).detach()``. Forward value numerically equals the quantized
    result (real precision loss included); backward gradient is
    identity, since the quantization delta is detached. This is applied
    explicitly rather than relying on ``MXTensor``'s ops being
    autograd-transparent -- confirmed by direct test that they are NOT
    for the FP4 element dtype (``float4_e2m1fn_x2``): without this
    explicit STE, ``.backward()`` silently left ``.grad`` as ``None``
    for FP4 inputs (presumably because 2-values-per-byte packing uses a
    non-differentiable bitwise op internally), while FP8 happened to
    work by luck. Applying STE unconditionally for both makes the
    gradient path an explicit guarantee of this function rather than an
    implementation detail of whatever torchao happens to do internally.
    """

    _require_torchao(str(elem_dtype))
    original_dtype = x.dtype
    original_shape = x.shape
    flat = x.reshape(-1, original_shape[-1])
    pad = (-flat.shape[-1]) % block_size
    padded = F.pad(flat, (0, pad)) if pad else flat
    padded_bf16 = padded.to(torch.bfloat16)
    with torch.no_grad():
        quantized = MXTensor.to_mx(padded_bf16.detach(), elem_dtype, block_size)
        dequantized = quantized.dequantize(torch.bfloat16)
    straight_through = padded_bf16 + (dequantized - padded_bf16).detach()
    if pad:
        straight_through = straight_through[..., : original_shape[-1]]
    return straight_through.reshape(original_shape).to(original_dtype)


def nvfp4_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """Straight-through NVFP4-precision round trip for node-compute inputs.

    torchao's NVFP4 uses a 16-element block with an FP8 (not E8M0) shared
    scale, distinct from MXFP4's 32-element/E8M0 block -- but no
    differentiable NVFP4 primitive exists in the installed torchao
    version (only an emulated dequantize helper, see module docstring),
    so this reuses the MXFP4 element dtype (``float4_e2m1fn_x2``) at
    NVFP4's 16-element block size as the closest available substitute,
    per the user's explicit call to try NVFP4 as a stand-in for node
    compute. This is *not* a true NVFP4 implementation (wrong scale
    format), only an approximation of its block granularity -- flagged
    here so it's never mistaken for the real thing.
    """

    return _fake_quant_round_trip(x, torch.float4_e2m1fn_x2, block_size=16)


def mxfp8_fake_quant(x: torch.Tensor) -> torch.Tensor:
    """Straight-through MXFP8-precision round trip (32-element block)."""

    return _fake_quant_round_trip(x, torch.float8_e4m3fn, block_size=MX_BLOCK_SIZE)


def mxfp8_fake_quant_node_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Straight-through MXFP8-precision-approximated grouped node compute.

    The real hardware-accelerated path (``_mxfp8_grouped_node_linear_hardware``
    below) hits a genuine bug in torchao 0.18.0's grouped-mm EMULATED
    fallback (an internal ``to_mx()`` contiguity assertion fails even
    when the input tensor is explicitly made contiguous first, in both
    the AUTO/SM100-kernel path and the EMULATED path, regardless of
    ``pad_token_groups_for_grouped_mm``) -- confirmed not fixable from
    this side without a torchao patch, so this is the working default
    for now. Same structure as ``nvfp4_grouped_node_linear``: quantizes
    both operands, runs the real matmul at bf16 through the existing
    ``cell_linear`` grouped-bmm kernel. Correct gradients, real
    precision-loss signal, no hardware speedup.
    """

    from gator.models.dense_kernels import cell_linear

    quantized_inputs = mxfp8_fake_quant(inputs)
    quantized_weight = mxfp8_fake_quant(weight)
    return cell_linear(
        quantized_inputs,
        quantized_weight,
        bias,
        backend="grouped_bmm",
        pad_to_multiple=32,
    )


def _mxfp8_grouped_node_linear_hardware(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    kernel_preference: "KernelPreference | None" = None,
) -> torch.Tensor:
    """Real hardware-accelerated MXFP8 grouped GEMM for folded node compute.

    NOT CURRENTLY USED -- hits a torchao 0.18.0 bug, see
    ``mxfp8_fake_quant_node_linear``'s docstring. Kept here for when
    torchao fixes the underlying issue; revisit by swapping the
    dispatch in ``node_compute_linear`` below.

    ``inputs``: ``[batch, cells, tokens, in_features]`` (matches
    ``cell_linear``'s convention). ``weight``: ``[cells, out_features,
    in_features]`` (the folded per-cell node transform). Differentiable
    via torchao's ``_to_mxfp8_then_scaled_grouped_mm``, built for MoE
    expert-grouped matmul with uniform group sizes here (every cell gets
    exactly ``batch * tokens`` rows, unlike MoE's variable per-expert
    routing) -- ``offs`` is therefore just evenly spaced.
    """

    _require_torchao("mxfp8")
    if inputs.ndim != 4:
        raise ValueError("mxfp8 node-linear inputs must be [batch,cells,tokens,in]")
    if weight.ndim != 3:
        raise ValueError("mxfp8 node-linear weight must be [cells,out,in]")
    batch, cells, tokens, in_features = inputs.shape
    if weight.shape[0] != cells or weight.shape[2] != in_features:
        raise ValueError("mxfp8 node-linear weight must match input cells/features")
    out_features = weight.shape[1]

    rows_per_group = batch * tokens
    flat_inputs = inputs.permute(1, 0, 2, 3).reshape(
        cells * rows_per_group, in_features
    )
    offs = torch.arange(
        1, cells + 1, device=inputs.device, dtype=torch.int32
    ) * rows_per_group
    weight_t = weight.transpose(-1, -2).contiguous()  # [cells, in, out]

    grouped_output = _to_mxfp8_then_scaled_grouped_mm(
        flat_inputs,
        weight_t,
        offs=offs,
        out_dtype=torch.bfloat16,
        kernel_preference=kernel_preference or KernelPreference.AUTO,
        pad_token_groups_for_grouped_mm=True,
    )
    output = grouped_output.reshape(cells, batch, tokens, out_features).permute(
        1, 0, 2, 3
    )
    if bias is not None:
        output = output + bias[None, :, None]
    return output.to(inputs.dtype)


def nvfp4_grouped_node_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Straight-through NVFP4-precision-approximated grouped node compute.

    No real grouped hardware kernel exists for this in torchao (see
    module docstring) -- quantizes/dequantizes both operands, then runs
    the actual matmul at bf16 through the existing ``cell_linear``
    grouped-bmm kernel. Correct gradients, real precision-loss signal,
    no hardware speedup.
    """

    from gator.models.dense_kernels import cell_linear

    quantized_inputs = nvfp4_fake_quant(inputs)
    quantized_weight = nvfp4_fake_quant(weight)
    return cell_linear(
        quantized_inputs,
        quantized_weight,
        bias,
        backend="grouped_bmm",
        pad_to_multiple=32,
    )


def node_compute_linear(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    precision: NodeComputePrecision,
    graph_slot: int = 0,
) -> torch.Tensor:
    """Dispatch the folded per-cell node transform to the requested precision.

    ``bf16`` is a plain passthrough to the existing ``cell_linear``
    kernel (byte-identical to not calling this module at all). ``mxfp8``
    and ``nvfp4`` are both straight-through fake-quantization (see
    ``mxfp8_fake_quant_node_linear``'s docstring for why the real
    hardware-accelerated MXFP8 path isn't wired in yet) -- correct
    gradients and real precision-loss signal, no hardware speedup either
    way. ``fp8_te`` is different: real hardware FP8 via NVIDIA
    Transformer Engine with CUDA-graph capture -- see
    ``te_fp8_cudagraph_node_linear``'s docstring for the important
    caveat that its measured speedup is CPU-dispatch-overhead-bound and
    shrinks when this layer runs embedded in the rest of the model
    rather than in isolation. ``graph_slot`` (``fp8_te`` only) must be
    distinct per call-site called before a shared backward() -- see
    ``te_fp8_cudagraph_node_linear``'s docstring; this model passes the
    relay loop index, since node compute runs once per relay before one
    combined backward.
    """

    if precision == "bf16":
        from gator.models.dense_kernels import cell_linear

        return cell_linear(
            inputs, weight, bias, backend="grouped_bmm", pad_to_multiple=32
        )
    if precision == "fp8_te":
        return te_fp8_cudagraph_node_linear(
            inputs, weight, bias, graph_slot=graph_slot
        )
    _require_torchao(precision)
    if precision == "mxfp8":
        return mxfp8_fake_quant_node_linear(inputs, weight, bias)
    if precision == "nvfp4":
        return nvfp4_grouped_node_linear(inputs, weight, bias)
    raise ValueError(f"unknown node-compute precision: {precision!r}")


class _MXReadoutMatmulHardware(torch.autograd.Function):
    """Differentiable MX-format readout matmul: ``x @ weight.T``.

    NOT CURRENTLY USED -- ``torch._scaled_mm`` raises
    ``CUDA error: CUBLAS_STATUS_NOT_SUPPORTED`` when called directly
    with this layout, despite matching the shapes/dtypes shown in
    torchao's own correctness test
    (``test/prototype/mx_formats/test_mx_mm.py``). Not resolved --
    ``mx_readout_linear`` uses the fake-quant STE path instead (see
    ``_fake_quant_round_trip``). Kept here for future revisiting; the
    likely next step is comparing against exactly what that test does
    (including its ``to_blocked``/swizzle handling) line by line rather
    than the adapted version here, since something in the adaptation is
    producing a layout cuBLASLt rejects.

    Backward recomputes both matmuls (``grad_x = grad_out @ weight``,
    ``grad_weight = grad_out.T @ x``) at the same quantized precision as
    forward, not a full-precision straight-through shortcut, so the
    reported gradient behavior would reflect the real format if the
    forward pass worked.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, elem_dtype: torch.dtype):
        ctx.save_for_backward(x, weight)
        ctx.elem_dtype = elem_dtype
        return _mx_matmul_transposed(x, weight, elem_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        elem_dtype = ctx.elem_dtype
        grad_output = grad_output.contiguous()
        grad_x = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_x = _mx_matmul(grad_output, weight, elem_dtype)
        if ctx.needs_input_grad[1]:
            grad_weight = _mx_matmul_transposed(
                grad_output.transpose(-1, -2).contiguous(),
                x.transpose(-1, -2).contiguous(),
                elem_dtype,
            ).transpose(-1, -2)
        return grad_x, grad_weight, None


def _mx_quantize_2d(x: torch.Tensor, elem_dtype: torch.dtype):
    x_mx = MXTensor.to_mx(x.to(torch.bfloat16).contiguous(), elem_dtype, MX_BLOCK_SIZE)
    scale = x_mx.scale.view(x.shape[0], x.shape[1] // MX_BLOCK_SIZE)
    return x_mx.qdata, to_blocked(scale)


def _mx_matmul_transposed(
    x: torch.Tensor, weight: torch.Tensor, elem_dtype: torch.dtype
) -> torch.Tensor:
    """``x @ weight.T`` at MX precision. ``x``: [M,K], ``weight``: [N,K]."""

    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1])
    x_data, x_scale = _mx_quantize_2d(x_2d, elem_dtype)
    w_data, w_scale = _mx_quantize_2d(weight, elem_dtype)
    w_data_t = w_data.transpose(-1, -2).contiguous()
    if elem_dtype == torch.float8_e4m3fn:
        out = torch._scaled_mm(
            x_data, w_data_t, x_scale, w_scale, out_dtype=torch.bfloat16
        )
    else:
        out = F.scaled_mm(
            x_data.view(torch.float4_e2m1fn_x2),
            w_data_t.view(torch.float4_e2m1fn_x2),
            scale_a=x_scale,
            scale_recipe_a=ScalingType.BlockWise1x32,
            scale_b=w_scale,
            scale_recipe_b=ScalingType.BlockWise1x32,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4,
            swizzle_b=SwizzleType.SWIZZLE_32_4_4,
            output_dtype=torch.bfloat16,
        )
    return out.reshape(*original_shape[:-1], weight.shape[0])


def _mx_matmul(
    x: torch.Tensor, weight: torch.Tensor, elem_dtype: torch.dtype
) -> torch.Tensor:
    """``x @ weight`` at MX precision. ``x``: [M,K], ``weight``: [K,N]."""

    return _mx_matmul_transposed(x, weight.transpose(-1, -2).contiguous(), elem_dtype)


class _FP8ReadoutBlock(torch.nn.Module):
    """A single ``te.Linear`` wrapping the vocab readout matmul.

    Simpler than ``_FP8CellBlock`` -- readout is one dense matmul, not a
    grouped per-cell one, so no cell loop is needed. Wrapped by
    ``torch.cuda.make_graphed_callables`` at construction time in
    ``te_fp8_cudagraph_readout_linear`` below; its parameters get
    value-copied from the live (tied) ``token_embedding.weight``/
    ``lm_bias`` on every call via ``_BridgeFP8ReadoutWeight``, same
    reasoning as ``_BridgeFP8CellWeights`` for node compute -- CUDA-graph
    replay needs a fixed memory address to read updated values from, and
    the live weight tensor (even though it's a genuine persistent
    ``nn.Parameter`` here, unlike node compute's freshly-folded weight)
    still isn't the SAME memory address as this block's own internal
    ``te.Linear.weight``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool,
        recipe: "_te_recipe.DelayedScaling",
    ) -> None:
        super().__init__()
        self.has_bias = has_bias
        self.recipe = recipe
        self.linear = _te.Linear(in_features, out_features, bias=has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with _te.autocast(enabled=True, recipe=self.recipe):
            return self.linear(x)


class _BridgeFP8ReadoutWeight(torch.autograd.Function):
    """Route the live readout weight/bias's value/gradient through ``_FP8ReadoutBlock``.

    Same mechanism as ``_BridgeFP8CellWeights`` (see its docstring for
    the full reasoning on why a value-copy-in/gradient-copy-out bridge
    is needed and why passing ``inputs`` through unchanged, rather than
    merging it as a sibling branch, is required for autograd to
    guarantee correct backward ordering), just without the per-cell loop.
    """

    @staticmethod
    def forward(ctx, inputs, weight, bias, raw_module, has_bias):  # type: ignore[no-untyped-def]
        with torch.no_grad():
            raw_module.linear.weight.copy_(weight)
            raw_module.linear.weight.grad = None
            if has_bias:
                raw_module.linear.bias.copy_(bias)
                raw_module.linear.bias.grad = None
        ctx.raw_module = raw_module
        ctx.has_bias = has_bias
        return inputs

    @staticmethod
    def backward(ctx, grad_inputs):  # type: ignore[no-untyped-def]
        raw_module = ctx.raw_module
        grad_weight = raw_module.linear.weight.grad.clone()
        grad_bias = raw_module.linear.bias.grad.clone() if ctx.has_bias else None
        return grad_inputs, grad_weight, grad_bias, None, None


_FP8_READOUT_CUDAGRAPH_CACHE: dict[tuple, tuple[_FP8ReadoutBlock, Callable]] = {}


def te_fp8_cudagraph_readout_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    graph_slot: int = 0,
) -> torch.Tensor:
    """Real hardware-accelerated FP8 vocabulary readout via NVIDIA
    Transformer Engine, with CUDA-graph capture -- the readout
    counterpart to ``te_fp8_cudagraph_node_linear``, same mechanism and
    same caveats (see that function's docstring for the full detail on
    why plain eager FP8 loses to bf16 at small-to-medium shapes, why the
    CUDA-graph-embedded-in-a-real-model speedup is smaller than an
    isolated-layer measurement suggests, and the ``DelayedScaling``
    cold-start accuracy caveat -- all apply identically here).

    ``weight``: ``[vocab_size, embedding_dim]`` (nn.Linear convention,
    the tied ``token_embedding.weight`` this is called with). Readout's
    matmul is much larger than node compute's per-cell one (production:
    vocab_size=128,256 vs. node compute's ~5,248 folded output width),
    already well past the small-shape overhead crossover documented for
    node compute -- re-measure rather than assume the same modest
    ~1.09x applies here.

    ``graph_slot`` -- see ``te_fp8_cudagraph_node_linear``'s docstring
    for why this matters (a shared graphed instance called more than
    once before a shared backward() silently breaks gradients). Default
    0 is safe for readout's current real usage (called exactly once per
    forward pass, confirmed by reading its call site in
    ``gator/models/language.py`` -- see
    ``Runbooks/GTR_ARW_100M_PRODUCTION_ABLATION_QUEUE.md``) but is
    exposed here defensively rather than assumed permanently true.

    Shape-cached per ``(rows, in_features, out_features, has_bias,
    device, graph_slot)``, same requirement as
    ``te_fp8_cudagraph_node_linear``: input row count must stay fixed
    across calls sharing a cache entry.
    """

    _require_transformer_engine()
    original_shape = x.shape
    in_features = original_shape[-1]
    if weight.ndim != 2 or weight.shape[1] != in_features:
        raise ValueError("fp8_te readout weight must be [vocab_size, in_features]")
    out_features = weight.shape[0]
    has_bias = bias is not None
    rows = x.numel() // in_features

    key = (rows, in_features, out_features, has_bias, x.device, graph_slot)
    cached = _FP8_READOUT_CUDAGRAPH_CACHE.get(key)
    if cached is None:
        raw_module = _FP8ReadoutBlock(
            in_features, out_features, has_bias, _te_recipe.DelayedScaling()
        ).to(device=x.device, dtype=torch.bfloat16)
        sample = torch.zeros(
            rows, in_features, device=x.device, dtype=torch.bfloat16, requires_grad=True
        )
        # See the identical comment in te_fp8_cudagraph_node_linear above
        # -- make_graphed_callables rejects capture from inside an
        # active autocast context, which real training always has.
        with torch.autocast(device_type="cuda", enabled=False):
            graphed_module = torch.cuda.make_graphed_callables(
                raw_module, (sample,), num_warmup_iters=5
            )
        cached = (raw_module, graphed_module)
        _FP8_READOUT_CUDAGRAPH_CACHE[key] = cached
    raw_module, graphed_module = cached

    flat_x = x.reshape(rows, in_features).to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    bias_bf16 = bias.to(torch.bfloat16) if has_bias else None

    bridged_x = _BridgeFP8ReadoutWeight.apply(
        flat_x, weight_bf16, bias_bf16, raw_module, has_bias
    )
    # .clone() required -- see the identical comment in
    # te_fp8_cudagraph_node_linear above for why (make_graphed_callables'
    # output is a live view into the graph's static buffer, overwritten by
    # the next call to this same graphed callable).
    flat_output = graphed_module(bridged_x).clone()
    output = flat_output.reshape(*original_shape[:-1], out_features)
    return output.to(x.dtype)


def mx_readout_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    precision: ReadoutPrecision,
) -> torch.Tensor:
    """MXFP8/MXFP4/fp8_te vocabulary readout: ``x @ weight.T + bias``.

    ``weight``: ``[vocab_size, embedding_dim]`` (nn.Linear convention,
    matches the tied ``token_embedding.weight`` this is called with).
    ``mxfp8``/``mxfp4`` are straight-through fake-quantization (see
    ``_fake_quant_round_trip``'s docstring) -- the real hardware-
    accelerated path attempted for those (``_MXReadoutMatmulHardware``
    above) hits a ``CUBLAS_STATUS_NOT_SUPPORTED`` error not yet
    resolved. ``fp8_te`` is different: real hardware FP8 via NVIDIA
    Transformer Engine with CUDA-graph capture, see
    ``te_fp8_cudagraph_readout_linear``'s docstring.
    """

    if precision == "bf16":
        return F.linear(x, weight, bias)
    if precision == "fp8_te":
        return te_fp8_cudagraph_readout_linear(x, weight, bias)
    _require_torchao(precision)
    elem_dtype = (
        torch.float8_e4m3fn if precision == "mxfp8" else torch.float4_e2m1fn_x2
    )
    quantized_x = _fake_quant_round_trip(x, elem_dtype)
    quantized_weight = _fake_quant_round_trip(weight, elem_dtype)
    return F.linear(quantized_x, quantized_weight, bias)
