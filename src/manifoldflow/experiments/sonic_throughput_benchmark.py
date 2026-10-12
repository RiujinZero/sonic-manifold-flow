"""Twelfth probe, and a different kind of measurement than every one
before it: not "does the mechanism work" but "is training fast enough to
be worth doing on this hardware at all."

Every probe in this project's whole correction/session/batch-lane line is
structurally overhead-bound by design - short synthetic statements (13
tokens), small per-call batches, individual Python-loop iterations - and
that was the right choice for isolating and debugging one mechanism at a
time. It is the wrong tool for this question. The genuine ~57k tok/s
production target this project needs to stay within reach of was measured
at production shape: a large batch of long, densely packed sequences,
under full compiled/bf16 execution. This benchmark measures exactly that
shape directly - synthetic random token IDs (no `distractor_reasoning`
generator, no `BasinMemory`, no correctness scoring at all - pure
forward+backward+optimizer-step throughput) at a batch/sequence-length the
caller chooses, defaulting to the production recipe's own 96x8192.

Two conditions, mirroring the user's own framing ("I expect some overhead
from the basins but it shouldn't be much"): `--with-memory` off measures
vanilla Sonic training alone; on, it adds one `SonicMemoryPort` injection
plus the `SonicKeyProjection` key/value cost on `--memory-fraction` of the
batch's rows (default 16/96, the production ratio) - not a real
`BasinMemory` read (a CPU-side, non-differentiable nearest-neighbor
lookup, and besides, deliberately excluded here: this benchmark measures
GPU compute cost, and a real bank read's own cost is negligible CPU-side
work already characterized separately in the correction/session/batch-lane
probes) but a same-shape, same-cost stand-in for what a retrieved vector
and its injection actually cost the forward/backward graph.

Unlike every prior probe, Sonic itself is **trainable** here
(`model.train()`, no `requires_grad_(False)`) - matching how production
actually trains (this project's probe line deliberately kept Sonic frozen
throughout to isolate the memory mechanism's own trainability, but a
frozen model with nothing else consuming its output builds no backward
graph at all, which would understate real training's backward-pass cost
substantially).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from manifoldflow.compute import (
    CompileConfig,
    Float32RuntimeConfig,
    configure_float32_runtime,
    mark_compiled_iteration_begin,
)
from manifoldflow.sonic_key_projection import (
    SonicKeyProjection,
    SonicKeyProjectionConfig,
)
from manifoldflow.sonic_memory_port import SonicMemoryPort, SonicMemoryPortConfig
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel

_DEFAULT_CHECKPOINT = Path(
    "checkpoints/sonic_ssm/sonic_ssm_291k_steps_colab_baseline.pt"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(_PROJECT_ROOT / "vendor" / "GTR-sonic-ssm"))


def _run_step(
    *,
    model: SonicSSMLanguageModel,
    optimizer: torch.optim.Optimizer,
    port: SonicMemoryPort | None,
    key_projection: SonicKeyProjection | None,
    value_projection: SonicKeyProjection | None,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    memory_rows: int,
    device: str,
    compile_config: CompileConfig,
    autocast_enabled: bool,
) -> None:
    mark_compiled_iteration_begin(compile_config, device)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    optimizer.zero_grad()
    with torch.autocast(
        device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled
    ):
        hidden, _ = model.forward_hidden(input_ids)
    hidden = hidden.float()

    if port is not None:
        assert key_projection is not None and value_projection is not None
        # A same-shape, same-cost stand-in for a real BasinMemory read's
        # retrieved vector - see the module docstring for why a real bank
        # read isn't exercised here.
        memory_slice = hidden[:memory_rows]
        pooled = memory_slice.mean(dim=1)
        key_projection(pooled)  # cost paid, output otherwise unused here
        retrieved = value_projection(pooled)
        injected = port(memory_slice, retrieved)
        hidden = torch.cat([injected, hidden[memory_rows:]], dim=0)

    # Not logits_from_hidden + F.cross_entropy: that materializes a full
    # [batch, seq, vocab_size] logits tensor - at production shape
    # (96x8192x50000, fp32) that alone is ~157GB, which is what actually
    # OOM'd this benchmark at even a modest 16x2048 shape before this fix
    # (already ~24GB allocated for logits+softmax intermediates+gradients
    # on a 16GB card). chunked_cross_entropy is the same machinery
    # sonic_ssm_train.py itself uses instead (model.chunked_cross_entropy,
    # or optionally the Liger fused kernel) - it never forms the full
    # logits tensor, chunking the LM head + softmax + backward together.
    loss = model.chunked_cross_entropy(hidden, target_ids)
    loss.backward()
    optimizer.step()


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=False
    )
    model_config = SonicSSMConfig(**checkpoint["model_config"])
    model = SonicSSMLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.train()  # trainable, unlike every prior probe - see module docstring

    configure_float32_runtime(
        Float32RuntimeConfig(matmul_precision=args.matmul_precision), device
    )
    model.configure_execution(
        scan_backend=args.scan_backend, packed_projections=args.packed_projections
    )
    compile_config = CompileConfig(
        enabled=args.compile != "none", mode=args.compile_mode
    )
    if args.compile == "layers":
        model.compile_layers(mode=args.compile_mode)
    elif args.compile == "blocks":
        model.compile_ssm_blocks(mode=args.compile_mode)

    trainable_parameters = list(model.parameters())
    port = key_projection = value_projection = None
    memory_rows = 0
    if args.with_memory:
        port = SonicMemoryPort(
            SonicMemoryPortConfig(
                d_model=model_config.d_model,
                memory_dim=model_config.d_model,
                rank=args.port_rank,
                gate_init_bias=0.0,
            )
        ).to(device)
        key_projection = SonicKeyProjection(
            SonicKeyProjectionConfig(dim=model_config.d_model)
        ).to(device)
        value_projection = SonicKeyProjection(
            SonicKeyProjectionConfig(dim=model_config.d_model)
        ).to(device)
        trainable_parameters += (
            list(port.parameters())
            + list(key_projection.parameters())
            + list(value_projection.parameters())
        )
        memory_rows = max(1, round(args.batch_size * args.memory_fraction))

    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=1e-4, fused=(device == "cuda")
    )
    autocast_enabled = args.mixed_precision == "bf16" and device in ("cuda", "cpu")

    step_kwargs = dict(
        model=model,
        optimizer=optimizer,
        port=port,
        key_projection=key_projection,
        value_projection=value_projection,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=model_config.vocab_size,
        memory_rows=memory_rows,
        device=device,
        compile_config=compile_config,
        autocast_enabled=autocast_enabled,
    )

    for _ in range(args.warmup_steps):
        _run_step(**step_kwargs)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(args.steps):
        _run_step(**step_kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    total_tokens = args.steps * args.batch_size * args.seq_len
    peak_memory_bytes = (
        torch.cuda.max_memory_allocated(device) if device == "cuda" else None
    )

    output = {
        "checkpoint_path": str(args.checkpoint_path),
        "device": device,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "with_memory": args.with_memory,
        "memory_fraction": args.memory_fraction if args.with_memory else None,
        "memory_rows": memory_rows if args.with_memory else None,
        "port_rank": args.port_rank if args.with_memory else None,
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        "mixed_precision": args.mixed_precision,
        "scan_backend": args.scan_backend,
        "packed_projections": args.packed_projections,
        "matmul_precision": args.matmul_precision,
        "elapsed_seconds": elapsed,
        "total_tokens": total_tokens,
        "tokens_per_second": total_tokens / elapsed,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_memory_gb": (
            peak_memory_bytes / (1024**3) if peak_memory_bytes is not None else None
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--with-memory", action="store_true")
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=16 / 96,
        help="Fraction of batch rows that get a memory injection - only "
        "used with --with-memory. Matches the production recipe's 16-of-96.",
    )
    parser.add_argument("--port-rank", type=int, default=64)
    parser.add_argument(
        "--compile", choices=("none", "layers", "blocks"), default="layers"
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="max-autotune-no-cudagraphs",
    )
    parser.add_argument(
        "--mixed-precision", choices=("fp32", "bf16"), default="bf16"
    )
    parser.add_argument(
        "--scan-backend", choices=("doubling", "hgrn"), default="doubling"
    )
    parser.add_argument("--packed-projections", action="store_true", default=True)
    parser.add_argument(
        "--no-packed-projections",
        dest="packed_projections",
        action="store_false",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        metrics = run(args)
    except BaseException:
        sys.stdout.flush()
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
