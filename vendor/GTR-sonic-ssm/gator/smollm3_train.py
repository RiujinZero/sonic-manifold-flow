"""SmolLM3-26B scaled training loop for original GTR-ARW."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

from .compute import (
    CompileConfig,
    compile_module,
    mark_compiled_iteration_begin,
)
from .execution import StewardshipExecutionController
from .models import GatorLanguageConfig, GatorLanguageModel
from .smollm3_recipe import (
    ACCUMULATION_STEPS,
    EFFECTIVE_BATCH_SEQUENCES,
    SEQUENCE_LENGTH,
    TOTAL_UPDATES,
    WARMUP_UPDATES,
    SmolLM3BatchProvider,
    adamw_parameter_groups,
    learning_rate_for_update,
    stage_for_update,
    total_updates_for_sequence_length,
)


def _tensor_is_finite(tensor: torch.Tensor) -> bool:
    if not (tensor.is_floating_point() or tensor.is_complex()):
        return True
    return bool(torch.isfinite(tensor.detach()).all().item())


def _first_nonfinite_optimizer_state(
    optimizer: torch.optim.Optimizer,
) -> str | None:
    for parameter_index, state in enumerate(optimizer.state.values()):
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and not _tensor_is_finite(value):
                return f"parameter[{parameter_index}].{name}"
    return None


def _assert_training_state_finite(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    context: str,
) -> None:
    for name, parameter in model.named_parameters():
        if not _tensor_is_finite(parameter):
            raise FloatingPointError(
                f"non-finite model parameter {name!r} detected {context}"
            )
    optimizer_state = _first_nonfinite_optimizer_state(optimizer)
    if optimizer_state is not None:
        raise FloatingPointError(
            f"non-finite optimizer state {optimizer_state!r} detected {context}"
        )


def _require_finite_loss(
    loss: torch.Tensor,
    *,
    update: int,
    micro_step: int,
    sources: tuple[str, ...],
) -> None:
    if _tensor_is_finite(loss):
        return
    raise FloatingPointError(
        "non-finite SmolLM3 loss before backward; "
        f"next_update={update + 1} micro_step={micro_step} "
        f"sources={sources!r}; optimizer step and checkpoint were skipped"
    )


def _clip_gradients_or_raise(
    model: torch.nn.Module,
    *,
    update: int,
    micro_step: int,
    sources: tuple[str, ...],
) -> float:
    try:
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
            error_if_nonfinite=True,
        )
    except RuntimeError as error:
        raise FloatingPointError(
            "non-finite SmolLM3 gradient before optimizer step; "
            f"next_update={update + 1} micro_step={micro_step} "
            f"sources={sources!r}; optimizer step and checkpoint were skipped"
        ) from error
    return float(total_norm)


def _gradient_failure_diagnostic(
    model: torch.nn.Module,
    *,
    detail_limit: int = 8,
) -> str:
    """Describe nonfinite gradient tensors before a failed window is cleared."""

    nonfinite_parameter_count = 0
    details: list[str] = []
    largest_finite_abs = -1.0
    largest_finite_name = "none"
    gradient_parameter_count = 0
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        values = gradient.detach()
        if values.is_sparse:
            values = values.coalesce().values()
        if not (values.is_floating_point() or values.is_complex()):
            continue
        gradient_parameter_count += 1
        finite_mask = torch.isfinite(values)
        all_finite = bool(finite_mask.all().item())
        if all_finite:
            finite_abs_max = (
                float(values.abs().max().item()) if values.numel() else 0.0
            )
        else:
            finite_values = values[finite_mask]
            finite_abs_max = (
                float(finite_values.abs().max().item())
                if finite_values.numel()
                else 0.0
            )
            nonfinite_parameter_count += 1
            if len(details) < detail_limit:
                nan_count = int(torch.isnan(values).sum().item())
                inf_count = int(torch.isinf(values).sum().item())
                nonfinite_count = values.numel() - int(finite_mask.sum().item())
                details.append(
                    f"name={name!r} shape={tuple(values.shape)!r} "
                    f"dtype={str(values.dtype)!r} nonfinite={nonfinite_count} "
                    f"nan={nan_count} inf={inf_count} "
                    f"finite_abs_max={finite_abs_max:.6g}"
                )
        if finite_abs_max > largest_finite_abs:
            largest_finite_abs = finite_abs_max
            largest_finite_name = name

    largest = max(largest_finite_abs, 0.0)
    if nonfinite_parameter_count:
        omitted = max(nonfinite_parameter_count - len(details), 0)
        return (
            f"gradient_parameters={gradient_parameter_count} "
            f"nonfinite_parameters={nonfinite_parameter_count} "
            f"largest_finite_name={largest_finite_name!r} "
            f"largest_finite_abs={largest:.6g} "
            f"details={details!r} omitted={omitted}"
        )
    return (
        f"gradient_parameters={gradient_parameter_count} "
        "all_gradient_elements_finite=true "
        f"largest_finite_name={largest_finite_name!r} "
        f"largest_finite_abs={largest:.6g}; aggregate_norm_overflow_suspected"
    )


def _discard_nonfinite_accumulation(
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    accumulation_steps: int,
    consecutive_failures: int,
    max_consecutive_recoveries: int,
    error: FloatingPointError,
) -> tuple[int, int]:
    """Clear a poisoned accumulation and return its clean microstep boundary."""

    optimizer.zero_grad(set_to_none=True)
    next_failures = consecutive_failures + 1
    if next_failures > max_consecutive_recoveries:
        raise FloatingPointError(
            f"{error}; recovery budget exhausted after "
            f"{max_consecutive_recoveries} consecutive discarded windows"
        ) from error
    return update * accumulation_steps, next_failures


def _remap_resume_micro_step(
    update: int,
    micro_step: int,
    *,
    checkpoint_batch_size: int,
    checkpoint_accumulation: int,
    requested_batch_size: int,
    requested_accumulation: int,
    effective_batch_sequences: int = EFFECTIVE_BATCH_SEQUENCES,
) -> int:
    """Map an update-boundary checkpoint to a new physical batch geometry."""

    values = {
        "checkpoint_batch_size": checkpoint_batch_size,
        "checkpoint_accumulation": checkpoint_accumulation,
        "requested_batch_size": requested_batch_size,
        "requested_accumulation": requested_accumulation,
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError("resume batch and accumulation values must be positive")
    checkpoint_effective = checkpoint_batch_size * checkpoint_accumulation
    requested_effective = requested_batch_size * requested_accumulation
    if (
        checkpoint_effective != effective_batch_sequences
        or requested_effective != effective_batch_sequences
    ):
        raise ValueError(
            "resume may change physical batch/accumulation only while both "
            f"effective batches remain {effective_batch_sequences}"
        )
    if update < 0 or micro_step != update * checkpoint_accumulation:
        raise ValueError(
            "resume checkpoint is not on a complete optimizer-update boundary"
        )
    return update * requested_accumulation


def _train(
    args: Any,
    parser: Any,
    *,
    device: torch.device,
    tokenizer: Any,
    cfg: GatorLanguageConfig,
    model: GatorLanguageModel,
    train_model: torch.nn.Module,
    compile_config: CompileConfig,
    optimizer: torch.optim.Optimizer,
    provider: SmolLM3BatchProvider,
    execution_controller: StewardshipExecutionController | None,
) -> None:
    out = Path(args.checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    update = micro = 0
    recipe_update = 0
    tokens_seen = 0
    nonfinite_windows = 0
    context_growth = False
    identity = {"name": args.hf_tokenizer, "revision": args.hf_tokenizer_revision}
    if args.resume_checkpoint:
        state = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        checkpoint_model_config = dict(state.get("model_config", {}))
        checkpoint_model_config.setdefault("lexical_embedding_dim", None)
        checkpoint_model_config.setdefault("block_causal", False)
        checkpoint_model_config.setdefault("block_size", None)
        checkpoint_model_config.setdefault("position_reference_length", None)
        checkpoint_recipe = dict(state.get("recipe", {}))
        checkpoint_peak_lr = float(checkpoint_recipe.get("peak_lr", 2e-4))
        checkpoint_sequence_length = int(
            checkpoint_recipe.get("sequence_length", SEQUENCE_LENGTH)
        )
        checkpoint_batch_size = int(
            checkpoint_recipe.get("physical_batch_size", 1)
        )
        checkpoint_accumulation = int(
            checkpoint_recipe.get("accumulation", ACCUMULATION_STEPS)
        )
        if (
            state.get("format") != "gator-smollm3-26b-v2"
            or checkpoint_model_config != cfg.to_dict()
            or state.get("tokenizer") != identity
        ):
            parser.error("resume checkpoint recipe/model/tokenizer identity mismatch")
        if checkpoint_peak_lr != args.smollm3_peak_lr:
            parser.error(
                "resume checkpoint peak learning rate differs: "
                f"checkpoint={checkpoint_peak_lr:.6g} requested={args.smollm3_peak_lr:.6g}"
            )
        if checkpoint_sequence_length != args.smollm3_sequence_length:
            context_growth = (
                args.smollm3_allow_context_growth
                and args.smollm3_sequence_length > checkpoint_sequence_length
            )
            if not context_growth:
                parser.error(
                    "resume checkpoint sequence length differs: "
                    f"checkpoint={checkpoint_sequence_length} "
                    f"requested={args.smollm3_sequence_length}; use "
                    "--smollm3-allow-context-growth only for a longer context"
                )
            if (
                cfg.position_reference_length is None
                or args.smollm3_sequence_length > cfg.position_reference_length
            ):
                parser.error(
                    "context growth requires position_reference_length at least "
                    "as large as the requested sequence"
                )
        batch_geometry_changed = (
            checkpoint_batch_size != args.batch_size
            or checkpoint_accumulation != args.gradient_accumulation_steps
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        provider.load_state_dict(
            state["data"], allow_window_resize=context_growth
        )
        if execution_controller is not None and state.get("execution") is not None:
            try:
                execution_controller.load_state_dict(state["execution"])
            except ValueError as error:
                parser.error(str(error))
        update = int(state["update"])
        try:
            micro = _remap_resume_micro_step(
                update,
                int(state["micro_step"]),
                checkpoint_batch_size=checkpoint_batch_size,
                checkpoint_accumulation=checkpoint_accumulation,
                requested_batch_size=args.batch_size,
                requested_accumulation=args.gradient_accumulation_steps,
            )
        except ValueError as error:
            parser.error(str(error))
        tokens_seen = int(
            state.get(
                "tokens_seen",
                update * checkpoint_sequence_length * EFFECTIVE_BATCH_SEQUENCES,
            )
        )
        nonfinite_windows = int(state.get("nonfinite_windows", 0))
        recipe_update = tokens_seen // (
            args.smollm3_sequence_length * EFFECTIVE_BATCH_SEQUENCES
        )
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        _assert_training_state_finite(
            model,
            optimizer,
            context=f"while loading checkpoint {args.resume_checkpoint!r}",
        )
        resume_mode = "context_growth" if context_growth else "exact"
        if batch_geometry_changed:
            print(
                "resume_batch_geometry="
                f"{checkpoint_batch_size}x{checkpoint_accumulation}->"
                f"{args.batch_size}x{args.gradient_accumulation_steps} "
                f"effective={EFFECTIVE_BATCH_SEQUENCES}",
                flush=True,
            )
        print(
            f"resume={resume_mode} update={update} recipe_update={recipe_update} "
            f"micro_step={micro} "
            f"stage={stage_for_update(recipe_update, sequence_length=args.smollm3_sequence_length) + 1} "
            f"tokens_seen={tokens_seen} checkpoint={args.resume_checkpoint}",
            flush=True,
        )
    if recipe_update > args.steps:
        parser.error(
            "--steps is an absolute token-equivalent recipe target and "
            "precedes the checkpoint"
        )
    final_update = update + (args.steps - recipe_update)
    starting_update = update

    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    last_file_log = started
    interval_loss = 0.0
    interval_updates = 0
    file_loss = 0.0
    file_updates = 0
    log_path = out / "train.log"
    accumulation_sources: list[str] = []

    def save(path: Path) -> None:
        _assert_training_state_finite(
            model,
            optimizer,
            context=f"before checkpoint update={update} path={str(path)!r}",
        )
        provider.set_progress(update=recipe_update, micro_step=micro)
        payload = {
            "format": "gator-smollm3-26b-v2",
            "update": update,
            "micro_step": micro,
            "recipe_update": recipe_update,
            "tokens_seen": tokens_seen,
            "nonfinite_windows": nonfinite_windows,
            "model": model.state_dict(),
            "model_config": cfg.to_dict(),
            "optimizer": optimizer.state_dict(),
            "data": provider.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "tokenizer": identity,
            "execution": (
                execution_controller.state_dict()
                if execution_controller is not None
                else None
            ),
            "recipe": {
                "sequence_length": args.smollm3_sequence_length,
                "physical_batch_size": args.batch_size,
                "accumulation": args.gradient_accumulation_steps,
                "effective_batch_sequences": EFFECTIVE_BATCH_SEQUENCES,
                "total_updates": total_updates_for_sequence_length(
                    args.smollm3_sequence_length
                ),
                "peak_lr": args.smollm3_peak_lr,
            },
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    pending_active_fractions: torch.Tensor | None = None
    accumulation_steps = args.gradient_accumulation_steps
    accumulation_loss = 0.0
    consecutive_nonfinite_windows = 0

    def recover_nonfinite_window(
        error: FloatingPointError,
        *,
        reason: str,
        diagnostic: str | None = None,
    ) -> None:
        nonlocal micro
        nonlocal accumulation_loss
        nonlocal pending_active_fractions
        nonlocal consecutive_nonfinite_windows
        nonlocal nonfinite_windows

        source_window = tuple(dict.fromkeys(accumulation_sources))
        micro, consecutive_nonfinite_windows = _discard_nonfinite_accumulation(
            optimizer,
            update=update,
            accumulation_steps=accumulation_steps,
            consecutive_failures=consecutive_nonfinite_windows,
            max_consecutive_recoveries=(
                args.smollm3_max_consecutive_nonfinite_windows
            ),
            error=error,
        )
        nonfinite_windows += 1
        accumulation_loss = 0.0
        accumulation_sources.clear()
        pending_active_fractions = None
        event = (
            f"[smollm3:train] event=nonfinite_window_discarded "
            f"reason={reason} update={update} micro_step={micro} "
            f"consecutive={consecutive_nonfinite_windows}/"
            f"{args.smollm3_max_consecutive_nonfinite_windows} "
            f"total={nonfinite_windows} sources={source_window!r}"
        )
        if diagnostic is not None:
            event += f" diagnostic=({diagnostic})"
        print(event, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"timestamp={datetime.now(timezone.utc).isoformat()} {event}"
            )
            handle.write(os.linesep)
            handle.flush()
            os.fsync(handle.fileno())
    while update < final_update:
        provider.set_progress(update=recipe_update, micro_step=micro)
        startup_diagnostic = update == 0
        micro_started = time.perf_counter()
        if startup_diagnostic:
            print(
                f"[smollm3:train] event=batch_start micro_step={micro + 1}/{accumulation_steps}",
                flush=True,
            )
        batch_started = time.perf_counter()
        batch = provider.sample_cpu(args.batch_size).to(device)
        accumulation_sources.extend(provider.last_source_names)
        if startup_diagnostic:
            print(
                f"[smollm3:train] event=batch_ready micro_step={micro + 1}/{accumulation_steps} "
                f"source={provider.last_source_names} seconds={time.perf_counter() - batch_started:.2f}",
                flush=True,
            )
        first_microstep = startup_diagnostic and micro == 0
        if first_microstep:
            print(f"[smollm3:train] event=first_forward_start compile={args.compile}", flush=True)
        forward_started = time.perf_counter()
        execution_plan = (
            execution_controller.plan
            if execution_controller is not None
            else None
        )
        observe_execution = (
            execution_controller is not None
            and (micro + 1) % accumulation_steps == 0
            and execution_controller.should_observe(update + 1, pruning=True)
        )
        active_train_model = (
            model
            if observe_execution
            or (execution_plan is not None and execution_plan.uses_packed_execution)
            else train_model
        )
        if active_train_model is train_model:
            mark_compiled_iteration_begin(compile_config, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=args.mixed_precision == "bf16" and device.type in {"cuda", "cpu"},
        ):
            forward_result = active_train_model(
                batch.context_ids,
                batch.context_mask,
                output_slots=args.response_tokens,
                temperature=args.route_temperature_start,
                stewardship_prune=True,
                stewardship_packed_from_relay=(
                    None
                    if execution_plan is None
                    else execution_plan.packed_from_relay
                ),
                diagnostics="route_loss" if observe_execution else None,
            )
            if observe_execution:
                logits, route_diagnostics = forward_result
                pending_active_fractions = route_diagnostics[
                    "route_active_fraction"
                ].detach()
            else:
                logits = forward_result
            loss = F.cross_entropy(
                logits.transpose(1, 2), batch.target_ids, reduction="none"
            )
            loss = (loss * batch.target_mask).sum() / batch.target_mask.sum().clamp_min(1)
        if first_microstep:
            print(
                f"[smollm3:train] event=first_forward_done "
                f"seconds={time.perf_counter() - forward_started:.2f}",
                flush=True,
            )
            print("[smollm3:train] event=first_backward_start", flush=True)
        backward_started = time.perf_counter()
        try:
            _require_finite_loss(
                loss,
                update=update,
                micro_step=micro + 1,
                sources=provider.last_source_names,
            )
        except FloatingPointError as error:
            recover_nonfinite_window(error, reason="loss")
            continue
        (loss / accumulation_steps).backward()
        if first_microstep:
            print(
                f"[smollm3:train] event=first_backward_done "
                f"seconds={time.perf_counter() - backward_started:.2f}",
                flush=True,
            )
        accumulation_loss += float(loss.detach())
        micro += 1
        if startup_diagnostic:
            print(
                f"[smollm3:train] event=micro_step_done micro_step={micro}/{accumulation_steps} "
                f"seconds={time.perf_counter() - micro_started:.2f}",
                flush=True,
            )
        if micro % accumulation_steps:
            continue

        next_update = update + 1
        next_recipe_update = recipe_update + 1
        lr = learning_rate_for_update(
            next_recipe_update,
            peak_lr=args.smollm3_peak_lr,
            sequence_length=args.smollm3_sequence_length,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        source_window = tuple(dict.fromkeys(accumulation_sources))
        try:
            grad = _clip_gradients_or_raise(
                model,
                update=update,
                micro_step=micro,
                sources=source_window,
            )
        except FloatingPointError as error:
            diagnostic = _gradient_failure_diagnostic(model)
            recover_nonfinite_window(
                error,
                reason="gradient",
                diagnostic=diagnostic,
            )
            continue
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        interval_loss += accumulation_loss
        file_loss += accumulation_loss
        accumulation_loss = 0.0
        accumulation_sources.clear()
        consecutive_nonfinite_windows = 0
        update = next_update
        tokens_seen += (
            args.smollm3_sequence_length * EFFECTIVE_BATCH_SEQUENCES
        )
        recipe_update = tokens_seen // (
            args.smollm3_sequence_length * EFFECTIVE_BATCH_SEQUENCES
        )
        interval_updates += 1
        file_updates += 1
        if pending_active_fractions is not None:
            assert execution_controller is not None
            previous_plan = execution_controller.plan
            observed_fractions = pending_active_fractions.to(
                device="cpu", dtype=torch.float64
            ).tolist()
            observed_plan = execution_controller.observe(
                pending_active_fractions
            )
            pending_active_fractions = None
            if observed_plan != previous_plan:
                print(
                    f"smollm3_execution_transition update={update} "
                    f"active_fraction_by_relay={observed_fractions} "
                    f"packed_from_relay={observed_plan.packed_from_relay} "
                    f"packed_calls_eager=True",
                    flush=True,
                )
        now = time.perf_counter()
        elapsed = now - started
        if update % args.log_every == 0 or update == final_update:
            print(
                f"update={update}/{final_update} recipe_update={recipe_update}/{args.steps} "
                f"micro_step={micro} "
                f"stage={stage_for_update(recipe_update, sequence_length=args.smollm3_sequence_length) + 1} "
                f"loss={interval_loss / max(interval_updates * accumulation_steps, 1):.4f} "
                f"lr={lr:.3e} grad={grad:.3f} "
                f"updates_per_second={(update - starting_update) / max(elapsed, 1e-9):.3f} "
                f"tokens={tokens_seen} nonfinite_windows={nonfinite_windows}",
                flush=True,
            )
            interval_loss = 0.0
            interval_updates = 0
        if update % args.checkpoint_every_updates == 0 or update == final_update:
            save(out / f"smollm3_update_{update:06d}.pt")
            save(out / "latest.pt")
        if (
            args.log_every_seconds > 0 and now - last_file_log >= args.log_every_seconds
        ) or update == final_update:
            line = (
                f"timestamp={datetime.now(timezone.utc).isoformat()} "
                f"update={update}/{final_update} recipe_update={recipe_update}/{args.steps} "
                f"micro_step={micro} "
                f"stage={stage_for_update(recipe_update, sequence_length=args.smollm3_sequence_length) + 1} "
                f"loss={file_loss / max(file_updates * accumulation_steps, 1):.4f} "
                f"lr={lr:.3e} grad={grad:.3f} "
                f"updates_per_second={(update - starting_update) / max(elapsed, 1e-9):.3f} "
                f"tokens={tokens_seen} nonfinite_windows={nonfinite_windows}\n"
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            last_file_log = now
            file_loss = 0.0
            file_updates = 0


def run(args: Any, parser: Any, *, device: torch.device) -> None:
    if args.architecture != "gator":
        parser.error("smollm3-26b requires --architecture gator (original GTR-ARW)")
    if (
        args.batch_size <= 0
        or args.gradient_accumulation_steps <= 0
        or args.batch_size * args.gradient_accumulation_steps
        != EFFECTIVE_BATCH_SEQUENCES
    ):
        parser.error(
            "smollm3-26b requires batch-size * gradient-accumulation-steps = "
            f"{EFFECTIVE_BATCH_SEQUENCES}"
        )
    if args.smollm3_sequence_length <= 1:
        parser.error("--smollm3-sequence-length must be at least 2")
    if args.smollm3_max_consecutive_nonfinite_windows < 0:
        parser.error(
            "--smollm3-max-consecutive-nonfinite-windows cannot be negative"
        )
    if args.smollm3_allow_context_growth and not args.resume_checkpoint:
        parser.error(
            "--smollm3-allow-context-growth requires --resume-checkpoint"
        )
    if args.position_norm_length is not None:
        if args.position_norm_length <= 1:
            parser.error("--position-norm-length must be greater than 1")
        if args.position_norm_length < args.smollm3_sequence_length:
            parser.error(
                "--position-norm-length cannot be shorter than the SmolLM3 sequence"
            )
    if args.block_size is not None and args.block_size <= 0:
        parser.error("--block-size must be positive")
    if args.block_causal:
        if args.receiver_memory_backend != "linear_scan":
            parser.error(
                "--block-causal requires --receiver-memory-backend linear_scan"
            )
        if args.position_norm_length is None:
            parser.error("--block-causal requires --position-norm-length")
        resolved_block_size = (
            args.response_tokens if args.block_size is None else args.block_size
        )
        if args.smollm3_sequence_length % resolved_block_size:
            parser.error(
                "--smollm3-sequence-length must be divisible by the block size"
            )
        if (
            args.cell_transport_mode == "stewardship"
            and args.stewardship_execution != "dense"
        ):
            parser.error(
                "--block-causal currently requires --stewardship-execution dense"
            )
    elif args.block_size is not None:
        parser.error("--block-size requires --block-causal")
    if not 0 < args.response_tokens < args.smollm3_sequence_length:
        parser.error(
            "--response-tokens must be positive and smaller than "
            "--smollm3-sequence-length"
        )
    if args.log_every_seconds < 0:
        parser.error("--log-every-seconds cannot be negative")
    if args.smollm3_peak_lr <= 0:
        parser.error("--smollm3-peak-lr must be positive")
    if args.smollm3_token_chunk_tokens < args.smollm3_sequence_length:
        parser.error(
            "--smollm3-token-chunk-tokens must be at least the sequence length"
        )
    if not 0 <= args.smollm3_prefetch_tokens < args.smollm3_token_chunk_tokens:
        parser.error("--smollm3-prefetch-tokens must be in [0, token chunk size)")
    if args.smollm3_prefetch_workers <= 0 or args.smollm3_max_document_chars <= 0:
        parser.error("SmolLM3 prefetch workers and max document chars must be positive")
    if args.smollm3_max_open_streams <= 0:
        parser.error("--smollm3-max-open-streams must be positive")
    if args.smollm3_max_host_ram_gb < 0 or args.smollm3_memory_log_every_seconds < 0:
        parser.error(
            "SmolLM3 max host RAM and memory log interval cannot be negative"
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error(
            f"install the recipe dependencies with `pip install transformers datasets`: {error}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_tokenizer,
        revision=args.hf_tokenizer_revision,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        parser.error("HF tokenizer must define eos_token_id")
    forbidden = tuple(
        sorted(
            token_id
            for token_id in {tokenizer.pad_token_id, tokenizer.bos_token_id}
            if token_id is not None and token_id != tokenizer.eos_token_id
        )
    )
    cfg = GatorLanguageConfig(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        embedding_dim=args.embedding_dim,
        lexical_embedding_dim=(
            None
            if args.lexical_embedding_dim in {None, 0}
            else args.lexical_embedding_dim
        ),
        lexical_mode=args.lexical_mode,
        cell_input_mode=args.cell_input_mode,
        cell_output_mode=args.cell_output_mode,
        cell_transport_mode=args.cell_transport_mode,
        stewardship_keep_mass=args.stewardship_keep_mass,
        stewardship_handoff_keep_mass=args.stewardship_handoff_keep_mass,
        stewardship_min_active=args.stewardship_min_active,
        stewardship_max_active=(
            None if args.stewardship_max_active == 0 else args.stewardship_max_active
        ),
        stewardship_exploration=args.stewardship_exploration,
        stewardship_scouts=args.stewardship_scouts,
        stewardship_execution=args.stewardship_execution,
        stewardship_packed_threshold=args.stewardship_packed_threshold,
        packed_stewardship_backend=args.packed_stewardship_backend,
        receiver_memory_backend=args.receiver_memory_backend,
        block_causal=args.block_causal,
        block_size=(
            args.response_tokens
            if args.block_causal and args.block_size is None
            else args.block_size
        ),
        position_reference_length=args.position_norm_length,
        dense_router_backend=args.dense_router_backend,
        dense_node_backend=args.dense_node_backend,
        dense_kernel_padding=args.dense_kernel_padding,
        activation_checkpointing=args.activation_checkpointing,
        n_cells=args.cells,
        n_nodes=args.nodes,
        node_topology=args.node_topology,
        relays=args.relays,
        neighbors=args.neighbors,
        mask_token_id=(
            tokenizer.mask_token_id
            if tokenizer.mask_token_id is not None
            else tokenizer.eos_token_id
        ),
        blank_token_id=None,
        forbidden_output_token_ids=forbidden,
    )
    model = GatorLanguageModel(cfg).to(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    lexical_table_parameters = model.token_embedding.weight.numel()
    lexical_projection_parameters = (
        model.lexical_projection.weight.numel()
        if isinstance(model.lexical_projection, torch.nn.Linear)
        else 0
    )
    lexical_path_parameters = (
        lexical_table_parameters + lexical_projection_parameters + model.lm_bias.numel()
    )
    print(
        f"smollm3_model_parameters total={total_parameters} "
        f"lexical_path={lexical_path_parameters} "
        f"core={total_parameters - lexical_path_parameters} "
        f"vocab={cfg.vocab_size} lexical_width={cfg.lexical_width} "
        f"language_width={cfg.language_width}",
        flush=True,
    )
    print(
        f"smollm3_schedule sequence_length={args.smollm3_sequence_length} "
        f"peak_lr={args.smollm3_peak_lr:.6g} "
        f"warmup_updates={math.ceil(WARMUP_UPDATES * SEQUENCE_LENGTH / args.smollm3_sequence_length)} "
        f"recipe_updates={total_updates_for_sequence_length(args.smollm3_sequence_length)}",
        flush=True,
    )
    compile_config = CompileConfig(
        enabled=args.compile,
        mode=args.compile_mode,
    )
    train_model = compile_module(model, compile_config)
    optimizer = torch.optim.AdamW(
        adamw_parameter_groups(model),
        lr=args.smollm3_peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    execution_controller = None
    if args.cell_transport_mode == "stewardship" and not cfg.block_causal:
        execution_controller = StewardshipExecutionController(
            mode=args.stewardship_execution,
            threshold=args.stewardship_packed_threshold,
            relays=args.relays,
            check_every=args.stewardship_execution_check_every,
        )
    print(
        f"smollm3_batch physical={args.batch_size} "
        f"accumulation={args.gradient_accumulation_steps} "
        f"effective={args.batch_size * args.gradient_accumulation_steps} "
        f"execution={'dense_block_causal' if cfg.block_causal else args.stewardship_execution}",
        flush=True,
    )
    provider = SmolLM3BatchProvider(
        tokenizer,
        context_tokens=args.smollm3_sequence_length - args.response_tokens,
        output_slots=args.response_tokens,
        seed=args.seed,
        tokenized_data_root=args.smollm3_tokenized_data_root,
        stream_cache_dir=args.smollm3_stream_cache_dir,
        chunk_tokens=args.smollm3_token_chunk_tokens,
        prefetch_tokens=args.smollm3_prefetch_tokens,
        prefetch_workers=args.smollm3_prefetch_workers,
        max_document_chars=args.smollm3_max_document_chars,
        max_open_streams=args.smollm3_max_open_streams,
        max_host_ram_gb=args.smollm3_max_host_ram_gb,
        memory_log_every_seconds=args.smollm3_memory_log_every_seconds,
    )
    mode = "datatrove_ds" if args.smollm3_tokenized_data_root else "hf_stream"
    print(
        f"smollm3_data_mode={mode} token_chunk={args.smollm3_token_chunk_tokens} "
        f"prefetch_at={args.smollm3_prefetch_tokens} "
        f"prefetch_workers={args.smollm3_prefetch_workers} "
        f"max_open_streams={args.smollm3_max_open_streams} "
        f"host_ram_limit_gb={args.smollm3_max_host_ram_gb:g} "
        f"cache={provider.cache_root}",
        flush=True,
    )
    try:
        _train(
            args,
            parser,
            device=device,
            tokenizer=tokenizer,
            cfg=cfg,
            model=model,
            train_model=train_model,
            compile_config=compile_config,
            optimizer=optimizer,
            provider=provider,
            execution_controller=execution_controller,
        )
    finally:
        provider.close()
