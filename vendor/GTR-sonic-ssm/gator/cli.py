"""Command-line adapters for the programmatic Gator API."""

from __future__ import annotations

import argparse

from .compute import (
    CompileConfig,
    ComputeConfig,
    Float32RuntimeConfig,
)
from .optimization import OptimizationConfig, SchedulerConfig
from .pipeline import BatchPipelineConfig
from .preparation import ParallelMapConfig


def add_compute_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the reusable hardware execution policy through a CLI."""

    group = parser.add_argument_group("compute")
    group.add_argument("--compile", action="store_true")
    group.add_argument(
        "--compile-mode",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        default="default",
    )
    group.add_argument(
        "--compile-shape-padding",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "ask TorchInductor to pad matrix shapes; omit to preserve its "
            "backend default"
        ),
    )
    group.add_argument(
        "--float32-matmul-precision",
        choices=["highest", "high", "medium"],
        default="highest",
        help=(
            "FP32 matmul internal precision; 'high' permits TF32 Tensor Cores "
            "on Ampere-or-newer CUDA hardware"
        ),
    )
    group.add_argument(
        "--cudnn-tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "enable or disable TF32 for cuDNN receiver-memory convolutions; "
            "omit to preserve the PyTorch process default"
        ),
    )


def compute_config_from_args(args: argparse.Namespace) -> ComputeConfig:
    """Build a validated execution policy from parsed CLI arguments."""

    return ComputeConfig(
        float32=Float32RuntimeConfig(
            matmul_precision=args.float32_matmul_precision,
            cudnn_tf32=args.cudnn_tf32,
        ),
        compile=CompileConfig(
            enabled=args.compile,
            mode=args.compile_mode,
            shape_padding=args.compile_shape_padding,
        ),
    )


def add_optimization_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("optimization")
    group.add_argument("--optimizer", choices=["adamw", "adam", "sgd"], default="adamw")
    group.add_argument("--learning-rate", type=float, default=8e-3)
    group.add_argument("--weight-decay", type=float, default=1e-4)
    group.add_argument(
        "--optimizer-epsilon",
        type=float,
        default=1e-8,
        help=(
            "Adam/AdamW denominator epsilon; larger values can reduce "
            "amplification of tiny backend-dependent gradients"
        ),
    )
    group.add_argument("--gradient-clip", type=float, default=1.0)
    group.add_argument("--scheduler", choices=["constant", "cosine", "milestone"], default="constant")
    group.add_argument(
        "--scheduler-total-steps",
        type=int,
        default=0,
        help=(
            "schedule horizon independent of this invocation's update count; "
            "0 uses --steps"
        ),
    )
    group.add_argument("--minimum-lr", type=float, default=0.0)
    group.add_argument("--lr-milestones", type=int, nargs="*", default=[])
    group.add_argument("--lr-gamma", type=float, default=0.25)


def optimization_config_from_args(args: argparse.Namespace) -> OptimizationConfig:
    scheduler_total_steps = getattr(
        args,
        "scheduler_total_steps",
        0,
    )
    if scheduler_total_steps < 0:
        raise ValueError("scheduler_total_steps cannot be negative")
    schedule = SchedulerConfig(
        name=args.scheduler,
        total_steps=(
            args.steps
            if scheduler_total_steps == 0
            else scheduler_total_steps
        ),
        minimum_lr=args.minimum_lr,
        milestones=tuple(sorted(set(args.lr_milestones))),
        gamma=args.lr_gamma,
    )
    return OptimizationConfig(
        name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epsilon=getattr(args, "optimizer_epsilon", 1e-8),
        gradient_clip=args.gradient_clip,
        scheduler=schedule,
    )


def add_data_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose deterministic preparation, prefetch, and resume policy."""

    group = parser.add_argument_group("data pipeline")
    group.add_argument(
        "--batch-pipeline",
        choices=["serial", "prefetch"],
        default="prefetch",
        help="sample inline or overlap CPU batch preparation with model work",
    )
    group.add_argument(
        "--prefetch-batches",
        type=int,
        default=2,
        help="bounded number of prepared CPU batches",
    )
    group.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="page-lock queued CPU tensors; defaults on for CUDA",
    )
    group.add_argument(
        "--non-blocking-transfer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="permit asynchronous device copies from pinned batches",
    )
    group.add_argument(
        "--dedicated-copy-stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stage CUDA copies on a dedicated stream",
    )
    group.add_argument(
        "--resume-data-state",
        choices=["auto", "exact", "portable_exact", "reset"],
        default="auto",
        help=(
            "restore an exact sampler/prefetch queue when compatible, require "
            "strict file identity, require exact encoded-corpus portability, "
            "or deliberately begin a new data phase"
        ),
    )
    group.add_argument(
        "--tokenizer-workers",
        type=int,
        default=0,
        help="ordered tokenization workers; 0 keeps serial preparation",
    )
    group.add_argument(
        "--tokenizer-worker-backend",
        choices=["threads", "processes"],
        default="processes",
    )
    group.add_argument(
        "--tokenizer-chunk-size",
        type=int,
        default=256,
        help="examples per ordered tokenization work item",
    )
    group.add_argument(
        "--tokenizer-pending-chunks",
        type=int,
        default=0,
        help="maximum queued chunks; 0 uses twice the worker count",
    )


def data_pipeline_config_from_args(
    args: argparse.Namespace,
    *,
    batch_size: int,
    device,
) -> BatchPipelineConfig:
    """Build the reusable batch-pipeline configuration."""

    pin_memory = (
        str(device).startswith("cuda")
        if args.pin_memory is None
        else args.pin_memory
    )
    return BatchPipelineConfig(
        batch_size=batch_size,
        device=device,
        mode=args.batch_pipeline,
        prefetch_batches=args.prefetch_batches,
        pin_memory=pin_memory,
        non_blocking=args.non_blocking_transfer,
        dedicated_cuda_stream=args.dedicated_copy_stream,
    )


def preparation_config_from_args(
    args: argparse.Namespace,
) -> ParallelMapConfig:
    """Build ordered tokenization policy from parsed arguments."""

    if args.tokenizer_workers < 0:
        raise ValueError("tokenizer_workers cannot be negative")
    backend = (
        "serial"
        if args.tokenizer_workers == 0
        else args.tokenizer_worker_backend
    )
    pending = (
        None
        if args.tokenizer_pending_chunks == 0
        else args.tokenizer_pending_chunks
    )
    return ParallelMapConfig(
        backend=backend,
        workers=(
            None if args.tokenizer_workers == 0 else args.tokenizer_workers
        ),
        chunk_size=args.tokenizer_chunk_size,
        max_pending_chunks=pending,
    )
