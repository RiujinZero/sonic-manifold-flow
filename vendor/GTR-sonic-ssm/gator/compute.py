"""Reusable execution policy for Gator training and inference.

The model architecture and the hardware execution policy are intentionally
separate. A checkpoint can therefore move between CPUs, pre-TF32 GPUs, and
newer Tensor-Core GPUs without changing learned parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import torch
from torch import nn


Float32MatmulPrecision = Literal["highest", "high", "medium"]
CompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
CUDA_GRAPH_COMPILE_MODES = frozenset({"reduce-overhead", "max-autotune"})


@dataclass(frozen=True)
class Float32RuntimeConfig:
    """Policy for FP32 operations that may use lower-precision internals.

    ``cudnn_tf32=None`` preserves the PyTorch process default. This matters
    because cuDNN convolutions have a separate TF32 switch from matmul/BMM.
    """

    matmul_precision: Float32MatmulPrecision = "highest"
    cudnn_tf32: bool | None = None

    def __post_init__(self) -> None:
        if self.matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError(
                "matmul_precision must be 'highest', 'high', or 'medium'"
            )
        if self.cudnn_tf32 is not None and not isinstance(
            self.cudnn_tf32, bool
        ):
            raise ValueError("cudnn_tf32 must be bool or None")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CompileConfig:
    """Static graph compilation policy.

    PyTorch treats a named compile mode and custom backend options as
    alternative configuration mechanisms. Shape padding is consequently
    available with ``mode='default'`` only.
    """

    enabled: bool = False
    mode: CompileMode = "default"
    shape_padding: bool | None = None

    def __post_init__(self) -> None:
        if self.mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError("unsupported torch.compile mode")
        if self.shape_padding is not None and not isinstance(
            self.shape_padding, bool
        ):
            raise ValueError("shape_padding must be bool or None")
        if self.mode != "default" and self.shape_padding is not None:
            raise ValueError(
                "shape_padding cannot be combined with a named compile mode"
            )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ComputeConfig:
    """Complete parameter-free execution policy."""

    float32: Float32RuntimeConfig = field(
        default_factory=Float32RuntimeConfig
    )
    compile: CompileConfig = field(default_factory=CompileConfig)

    def to_dict(self) -> dict:
        return {
            "float32": self.float32.to_dict(),
            "compile": self.compile.to_dict(),
        }


@dataclass(frozen=True)
class ResolvedFloat32Runtime:
    """Requested policy plus the CUDA capability that can realize it."""

    device_type: str
    device_index: int | None
    gpu_name: str | None
    cuda_capability: tuple[int, int] | None
    tf32_supported: bool
    matmul_precision: Float32MatmulPrecision
    matmul_tensor_core_policy_effective: bool
    cudnn_tf32_requested: bool | None
    cudnn_tf32_backend_enabled: bool
    cudnn_tf32_hardware_effective: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        capability = (
            "none"
            if self.cuda_capability is None
            else f"{self.cuda_capability[0]}.{self.cuda_capability[1]}"
        )
        return (
            f"float32_matmul_precision={self.matmul_precision} "
            f"cuda_capability={capability} "
            f"tf32_supported={self.tf32_supported} "
            f"matmul_tensor_core_policy_effective="
            f"{self.matmul_tensor_core_policy_effective} "
            f"cudnn_tf32_backend={self.cudnn_tf32_backend_enabled} "
            f"cudnn_tf32_effective={self.cudnn_tf32_hardware_effective}"
        )


def resolve_float32_runtime(
    config: Float32RuntimeConfig,
    device: torch.device | str,
) -> ResolvedFloat32Runtime:
    """Inspect the effective global settings and target hardware."""

    device = torch.device(device)
    device_index = None
    gpu_name = None
    capability = None
    if device.type == "cuda":
        device_index = (
            torch.cuda.current_device()
            if device.index is None
            else device.index
        )
        capability = tuple(torch.cuda.get_device_capability(device_index))
        gpu_name = torch.cuda.get_device_name(device_index)
    tf32_supported = capability is not None and capability[0] >= 8
    matmul_precision = torch.get_float32_matmul_precision()
    cudnn_enabled = bool(torch.backends.cudnn.allow_tf32)
    return ResolvedFloat32Runtime(
        device_type=device.type,
        device_index=device_index,
        gpu_name=gpu_name,
        cuda_capability=capability,
        tf32_supported=tf32_supported,
        matmul_precision=matmul_precision,
        matmul_tensor_core_policy_effective=(
            tf32_supported and matmul_precision in {"high", "medium"}
        ),
        cudnn_tf32_requested=config.cudnn_tf32,
        cudnn_tf32_backend_enabled=cudnn_enabled,
        cudnn_tf32_hardware_effective=tf32_supported and cudnn_enabled,
    )


def configure_float32_runtime(
    config: Float32RuntimeConfig,
    device: torch.device | str,
) -> ResolvedFloat32Runtime:
    """Apply the process-wide FP32 policy before compilation/first forward."""

    torch.set_float32_matmul_precision(config.matmul_precision)
    if config.cudnn_tf32 is not None:
        torch.backends.cudnn.allow_tf32 = config.cudnn_tf32
    return resolve_float32_runtime(config, device)


def compile_module(module: nn.Module, config: CompileConfig) -> nn.Module:
    """Configure safeguards and compile under a serializable policy."""

    configure_cudagraph_accumulation = getattr(
        module,
        "configure_cudagraph_accumulation",
        None,
    )
    if callable(configure_cudagraph_accumulation):
        configure_cudagraph_accumulation(
            config.enabled
            and config.mode in CUDA_GRAPH_COMPILE_MODES
        )
    if not config.enabled:
        return module
    if config.shape_padding is not None:
        return torch.compile(
            module,
            options={"shape_padding": config.shape_padding},
        )
    if config.mode == "default":
        return torch.compile(module)
    return torch.compile(module, mode=config.mode)


def mark_compiled_iteration_begin(
    config: CompileConfig,
    device: torch.device | str,
) -> bool:
    """Start one CUDA-graph training/inference iteration when applicable.

    PyTorch normally infers this boundary from compiled calls and their
    backwards. Gradient accumulation can defeat that heuristic and let the
    next replay overwrite saved activations from the preceding micro-batch.
    The marker must run outside the compiled callable, immediately before its
    first invocation for the new logical iteration.
    """

    device = torch.device(device)
    if (
        not config.enabled
        or device.type != "cuda"
        or config.mode not in CUDA_GRAPH_COMPILE_MODES
    ):
        return False
    compiler = getattr(torch, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None)
    if marker is None:
        raise RuntimeError(
            "compiled CUDA execution requires "
            "torch.compiler.cudagraph_mark_step_begin; upgrade PyTorch or "
            "use compile mode 'max-autotune-no-cudagraphs'"
        )
    marker()
    return True
