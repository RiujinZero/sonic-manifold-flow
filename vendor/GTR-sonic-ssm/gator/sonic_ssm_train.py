"""SmolLM3-26B autoregressive training loop for Sonic's sparse-mode SSM."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import time
from typing import Any

import torch

from .compute import CompileConfig, mark_compiled_iteration_begin
from .models import SonicSSMConfig, SonicSSMLanguageModel
from .smollm3_recipe import (
    ACCUMULATION_STEPS,
    EFFECTIVE_BATCH_SEQUENCES,
    SEQUENCE_LENGTH,
    STAGES,
    STAGES_DISTRACTOR_REASONING,
    WARMUP_UPDATES,
    SmolLM3BatchProvider,
    _scaled_update_count,
    adamw_parameter_groups,
    decay_boundaries_for_updates,
    learning_rate_for_update,
    total_updates_for_sequence_length,
)
from .smollm3_train import (
    _assert_training_state_finite,
    _clip_gradients_or_raise,
    _discard_nonfinite_accumulation,
    _gradient_failure_diagnostic,
    _remap_resume_micro_step,
    _require_finite_loss,
)
from .tasks import LanguageBatch


CHECKPOINT_FORMAT = "sonic-ssm-smollm3-26b-v2"

# Excluded from the strict model_config equality check on resume: both are
# pure recompute-vs-memory tradeoffs (what gets recomputed during backward,
# not what the model actually computes), so they never change saved weights
# or optimizer state -- unlike every other SonicSSMConfig field, which does
# describe real model structure. Same reasoning as the dedicated
# sequence_length/batch carve-outs just below, just without needing any
# remapping since nothing about the data pipeline or step counting depends
# on them.
RESUME_CONFIG_EXEMPT_FIELDS = frozenset({"activation_checkpointing", "checkpoint_every"})


def _model_config_for_resume_comparison(config_dict: dict) -> dict:
    return {k: v for k, v in config_dict.items() if k not in RESUME_CONFIG_EXEMPT_FIELDS}


# Same fixed prompt set as GTR-ARW's own eval harness (gator/smollm3_train.py,
# gator_monarch branch) so the two journals' completions are directly
# comparable at matched updates -- the first 11 only; the distractor
# probes below are Sonic-specific and diverge from that parity on
# purpose, added 2026-08-24 once distractor_reasoning became part of the
# training mixture. Deliberately plain templates matching the generator
# suite's own task shapes (binding-with-a-distractor-fact, state update
# through an event, ownership/refusal, turn-tracking across a short
# dialogue) rather than anything from distractor_reasoning's own vocab
# pools, so passing isn't just recognizing a memorized template.
FIXED_EVAL_PROMPTS = (
    "The capital of France is",
    "In machine learning, a neural network is",
    "def add(a, b):\n    return",
    "The three primary colors are",
    "Once upon a time, there was a",
    "Water boils at a temperature of",
    "The main causes of climate change include",
    "To bake a simple loaf of bread, you need",
    "The speed of light in a vacuum is approximately",
    "The best way to learn a new programming language is",
    "David's father has three sons: Snap, Crackle, and what is the third son's name?",
    # Binding with a distractor fact present for the wrong subject.
    "Kevin has 4 cats. Maria has 8 parrots. How many parrots does Kevin have?",
    # State update through an event.
    "Sarah had 3 books. She gave 1 book to Tom. How many books does Sarah have now?",
    # Ownership/refusal -- the fact needed to answer was never stated.
    "Kevin has 4 cats. What color are Kevin's cats?",
    # Turn-tracking across a short dialogue.
    "User: My dog's name is Max.\nAssistant: That's a great name!\nUser: What is my dog's name?\nAssistant:",
)


def _run_sonic_fixed_prompt_eval(
    model: SonicSSMLanguageModel,
    tokenizer: Any,
    *,
    device: torch.device,
    context_tokens: int,
    response_tokens: int,
    mixed_precision: str,
    update: int,
    eval_log_path: Path,
    compile_config: CompileConfig,
) -> None:
    """Greedy-decode a fixed prompt set for a human-readable progress log.

    Unlike GTR-ARW's parallel block-formation eval (one forward pass
    predicts all `output_slots` at once), Sonic is a plain causal LM, so
    this generates autoregressively: one forward pass over the prompt to
    seed the recurrent state, then one forward call per new token, greedy
    argmax each step, carrying `states` forward.
    """
    was_training = model.training
    model.eval()
    lines = [f"=== fixed_prompt_eval update={update} ==="]
    try:
        with torch.no_grad():
            for prompt in FIXED_EVAL_PROMPTS:
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                prompt_ids = prompt_ids[-context_tokens:]
                pad_len = context_tokens - len(prompt_ids)
                padded = [tokenizer.eos_token_id] * pad_len + prompt_ids
                input_ids = torch.tensor([padded], dtype=torch.long, device=device)
                # Eval is a separate logical iteration stream from training's
                # own microstep loop (same CUDA-graph-overwrite hazard the
                # training loop marks against, see the training loop's own
                # mark_compiled_iteration_begin call) -- mark once per prompt,
                # before its first compiled forward call.
                mark_compiled_iteration_begin(compile_config, device)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=mixed_precision == "bf16" and device.type in {"cuda", "cpu"},
                ):
                    logits, states = model(input_ids)
                    next_id = logits[:, -1, :].argmax(-1, keepdim=True)
                    generated = [int(next_id.item())]
                    for _ in range(response_tokens - 1):
                        logits, states = model(next_id, states)
                        next_id = logits[:, -1, :].argmax(-1, keepdim=True)
                        generated.append(int(next_id.item()))
                completion = tokenizer.decode(generated, skip_special_tokens=True)
                lines.append(f"prompt={prompt!r} completion={completion!r}")
    finally:
        if was_training:
            model.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    text = "\n".join(lines) + "\n"
    print(text, end="", flush=True)
    with eval_log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _autoregressive_batch(batch: LanguageBatch) -> tuple[torch.Tensor, torch.Tensor]:
    """Turn one contiguous recipe window into standard shifted LM tensors."""

    if batch.target_ids.shape[1] != 1:
        raise ValueError("Sonic SSM autoregressive batches require one trailing token")
    tokens = torch.cat((batch.context_ids, batch.target_ids), dim=1)
    return tokens[:, :-1], tokens[:, 1:]


def _tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    """Content identity that remains valid when a local tokenizer moves hosts."""

    serialized = tokenizer.backend_tokenizer.to_str().encode("utf-8")
    return {
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "vocab_size": len(tokenizer),
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def _append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + os.linesep)
        handle.flush()
        os.fsync(handle.fileno())


def _fused_linear_cross_entropy(
    model: SonicSSMLanguageModel,
    hidden: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Numerically conservative Liger linear-CE used by optimized Sonic."""

    try:
        from liger_kernel.transformers.functional import (
            liger_fused_linear_cross_entropy,
        )
    except ImportError as error:
        raise RuntimeError(
            "--ssm-fused-linear-ce requires `pip install liger-kernel`"
        ) from error
    lexical = model.lexical_hidden(hidden).reshape(-1, model.config.lexical_dim)
    # Sonic's existing tiled CE forms FP32 logits.  Disable the surrounding
    # BF16 autocast and preserve that behavior while still using Liger's fused
    # projection/softmax/backward kernel.
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return liger_fused_linear_cross_entropy(
            lexical.float(),
            model.token_embedding.weight,
            labels.reshape(-1).long(),
            bias=model.lm_bias,
        )


def _validate_args(args: Any, parser: Any, *, device: torch.device) -> None:
    if args.architecture != "sonic_ssm":
        parser.error("the Sonic SSM recipe requires --architecture sonic_ssm")
    if args.smollm3_effective_batch_sequences <= 0:
        parser.error("--smollm3-effective-batch-sequences must be positive")
    if (
        args.batch_size <= 0
        or args.gradient_accumulation_steps <= 0
        or args.batch_size * args.gradient_accumulation_steps
        != args.smollm3_effective_batch_sequences
    ):
        parser.error(
            "smollm3-26b requires batch-size * gradient-accumulation-steps = "
            f"{args.smollm3_effective_batch_sequences} "
            "(--smollm3-effective-batch-sequences)"
        )
    if args.smollm3_sequence_length <= 1:
        parser.error("--smollm3-sequence-length must be at least 2")
    if args.smollm3_peak_lr <= 0:
        parser.error("--smollm3-peak-lr must be positive")
    if args.ssm_ce_chunk_rows <= 0:
        parser.error("--ssm-ce-chunk-rows must be positive")
    if args.ssm_checkpoint_every <= 0:
        parser.error("--ssm-checkpoint-every must be positive")
    if args.ssm_num_modes <= 0:
        parser.error("--ssm-num-modes must be positive")
    if not 0 <= args.ssm_sparsity_target < 1:
        parser.error("--ssm-sparsity-target must be in [0, 1)")
    if args.smollm3_max_consecutive_nonfinite_windows < 0:
        parser.error(
            "--smollm3-max-consecutive-nonfinite-windows cannot be negative"
        )
    if args.smollm3_allow_context_growth and not args.resume_checkpoint:
        parser.error("--smollm3-allow-context-growth requires --resume-checkpoint")
    if args.compile:
        parser.error(
            "full-model --compile is intentionally disabled for Sonic; use "
            "--ssm-compile-layers for its optimized scoped compile path"
        )
    if args.ssm_compile_blocks and args.ssm_compile_layers:
        parser.error("choose only one of --ssm-compile-blocks and --ssm-compile-layers")
    if args.ssm_scan_backend == "hgrn" and device.type != "cuda":
        parser.error("--ssm-scan-backend hgrn requires CUDA")
    if args.ssm_fused_linear_ce and device.type != "cuda":
        parser.error("--ssm-fused-linear-ce requires CUDA")
    if args.ssm_fused_adamw and device.type != "cuda":
        parser.error("--ssm-fused-adamw requires CUDA")
    if args.log_every <= 0 or args.checkpoint_every_updates <= 0:
        parser.error("log and checkpoint update intervals must be positive")
    if args.log_every_seconds < 0 or args.max_runtime_seconds < 0:
        parser.error("wall-clock intervals cannot be negative")
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
        parser.error("SmolLM3 max host RAM and memory log interval cannot be negative")


def run(args: Any, parser: Any, *, device: torch.device) -> None:
    """Train Sonic's sparse-mode SSM under Gator's SmolLM3 data recipe."""

    _validate_args(args, parser, device=device)
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        parser.error(
            "install the recipe dependencies with "
            f"`pip install transformers datasets`: {error}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_tokenizer,
        revision=args.hf_tokenizer_revision,
        use_fast=True,
    )
    if len(tokenizer) != 50_000:
        parser.error(
            "the production Gator/Sonic comparison requires the 50,000-token "
            f"tokenizer, but {args.hf_tokenizer!r} contains {len(tokenizer)} tokens"
        )
    # A separate instance, never shared with the data provider's background
    # prefetch threads (gator/smollm3_recipe.py calls tokenizer.encode from
    # up to --smollm3-prefetch-workers threads concurrently). HF fast
    # tokenizers mutate internal Rust state on every .encode() call
    # (set_truncation_and_padding), which isn't safe to call concurrently
    # from multiple threads on the same object -- sharing one here crashed
    # the run with "RuntimeError: Already borrowed" the first time the eval
    # fired while a prefetch thread was mid-encode.
    eval_tokenizer = AutoTokenizer.from_pretrained(
        args.hf_tokenizer,
        revision=args.hf_tokenizer_revision,
        use_fast=True,
    )
    # Same reasoning as eval_tokenizer above: DistractorReasoningTextSource
    # calls .encode() synchronously on whatever thread calls provider.take()
    # (the main training thread), which can race the data provider's own
    # background prefetch threads if they share one tokenizer object --
    # give it a dedicated instance too, only actually used when
    # --smollm3-distractor-reasoning-stages is set.
    distractor_reasoning_tokenizer = AutoTokenizer.from_pretrained(
        args.hf_tokenizer,
        revision=args.hf_tokenizer_revision,
        use_fast=True,
    )
    if tokenizer.eos_token_id is None:
        parser.error("HF tokenizer must define eos_token_id")

    config = SonicSSMConfig(
        vocab_size=len(tokenizer),
        d_model=args.ssm_d_model,
        lexical_dim=args.ssm_lexical_dim,
        n_layers=args.ssm_layers,
        state_dim=args.ssm_state_dim,
        num_modes=args.ssm_num_modes,
        sparsity_target=args.ssm_sparsity_target,
        gate_residual=args.ssm_gate_residual,
        dense_a=args.ssm_dense_a,
        activation_checkpointing=args.ssm_activation_checkpointing,
        checkpoint_every=args.ssm_checkpoint_every,
    )
    model = SonicSSMLanguageModel(config).to(device)
    model.configure_execution(
        scan_backend=args.ssm_scan_backend,
        packed_projections=args.ssm_packed_projections,
    )
    if args.ssm_compile_layers:
        model.compile_layers(mode=args.compile_mode)
    elif args.ssm_compile_blocks:
        model.compile_ssm_blocks(mode=args.compile_mode)
    # Sonic compiles per-layer (model.compile_layers/compile_ssm_blocks) rather
    # than through gator.compute.compile_module's whole-model wrapper, so it
    # never picked up that path's CUDA-graph safety scaffolding. A CUDA-graph
    # compile mode (max-autotune, reduce-overhead) needs an explicit marker
    # before the first compiled call of each new logical training step, or
    # the graph replay overwrites a still-referenced tensor from the previous
    # step -- exactly mirrors gator/smollm3_train.py's own
    # mark_compiled_iteration_begin(compile_config, device) call, placed the
    # same way (once per micro-step, immediately before the compiled
    # forward). Harmless/no-op when compile is off or the mode doesn't use
    # CUDA graphs (mark_compiled_iteration_begin's own early-return).
    compile_config = CompileConfig(
        enabled=args.ssm_compile_layers or args.ssm_compile_blocks,
        mode=args.compile_mode,
    )
    optimizer = torch.optim.AdamW(
        adamw_parameter_groups(model),
        lr=args.smollm3_peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=args.ssm_fused_adamw,
    )
    identity = _tokenizer_identity(tokenizer)

    print(
        f"sonic_ssm_model_parameters total={model.parameter_count} "
        f"lexical_path={model.lexical_parameter_count} "
        f"core={model.parameter_count - model.lexical_parameter_count} "
        f"vocab={config.vocab_size} d_model={config.d_model} "
        f"state_dim={config.state_dim} layers={config.n_layers} "
        f"num_modes={config.num_modes} "
        f"active_modes={model.layers[0].sparse_gate.active_modes} "
        f"sparsity_target={config.sparsity_target:.3f} "
        f"gate_residual={config.gate_residual}",
        flush=True,
    )
    _stage3_update, _decay_updates = decay_boundaries_for_updates(
        args.smollm3_sequence_length, args.smollm3_effective_batch_sequences, args.steps,
    )
    print(
        f"sonic_ssm_schedule sequence_length={args.smollm3_sequence_length} "
        f"peak_lr={args.smollm3_peak_lr:.6g} "
        f"warmup_updates={_scaled_update_count(WARMUP_UPDATES, args.smollm3_sequence_length, args.smollm3_effective_batch_sequences)} "
        f"recipe_updates={total_updates_for_sequence_length(args.smollm3_sequence_length, args.smollm3_effective_batch_sequences)} "
        f"steps={args.steps} decay_starts_at={_stage3_update} decay_ends_at={_stage3_update + _decay_updates} "
        f"physical_batch={args.batch_size} "
        f"accumulation={args.gradient_accumulation_steps} "
        f"effective_batch={args.smollm3_effective_batch_sequences} "
        f"compile_blocks={args.ssm_compile_blocks} "
        f"compile_layers={args.ssm_compile_layers} "
        f"scan_backend={args.ssm_scan_backend} "
        f"packed_projections={args.ssm_packed_projections} "
        f"fused_linear_ce={args.ssm_fused_linear_ce} "
        f"fused_adamw={args.ssm_fused_adamw} "
        f"activation_checkpointing={args.ssm_activation_checkpointing}",
        flush=True,
    )

    provider = SmolLM3BatchProvider(
        tokenizer,
        context_tokens=args.smollm3_sequence_length - 1,
        output_slots=1,
        seed=args.seed,
        sources=STAGES_DISTRACTOR_REASONING if args.smollm3_distractor_reasoning_stages else STAGES,
        distractor_reasoning_chat_data=args.smollm3_distractor_reasoning_chat_data,
        distractor_reasoning_tokenizer=distractor_reasoning_tokenizer,
        force_stage=args.smollm3_force_stage,
        tokenized_data_root=args.smollm3_tokenized_data_root,
        stream_cache_dir=args.smollm3_stream_cache_dir,
        chunk_tokens=args.smollm3_token_chunk_tokens,
        prefetch_tokens=args.smollm3_prefetch_tokens,
        prefetch_workers=args.smollm3_prefetch_workers,
        max_document_chars=args.smollm3_max_document_chars,
        max_open_streams=args.smollm3_max_open_streams,
        max_host_ram_gb=args.smollm3_max_host_ram_gb,
        memory_log_every_seconds=args.smollm3_memory_log_every_seconds,
        effective_batch_sequences=args.smollm3_effective_batch_sequences,
    )
    mode = "datatrove_ds" if args.smollm3_tokenized_data_root else "hf_stream"
    print(
        f"sonic_ssm_data_mode={mode} cache={provider.cache_root} "
        f"distractor_reasoning_stages={args.smollm3_distractor_reasoning_stages} "
        f"force_stage={args.smollm3_force_stage} "
        "objective=causal_next_token_shifted_all_positions",
        flush=True,
    )

    out = Path(args.checkpoint_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Defaults to checkpoint_dir (unchanged behavior) but can be pointed at
    # fast local disk separately -- e.g. on Colab, checkpoint_dir is a
    # Drive-mounted path (fine for infrequent large checkpoint writes) while
    # logs get appended far more often and are better batched to Drive on
    # their own schedule rather than written there directly.
    log_dir = Path(args.log_dir) if getattr(args, "log_dir", None) else out
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    eval_log_path = log_dir / "eval_prompts.log"
    update = 0
    micro_step = 0
    recipe_update = 0
    tokens_seen = 0
    nonfinite_windows = 0
    best_loss = float("inf")
    context_growth = False

    try:
        if args.resume_checkpoint:
            state = torch.load(
                args.resume_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            checkpoint_recipe = dict(state.get("recipe", {}))
            checkpoint_sequence_length = int(
                checkpoint_recipe.get("sequence_length", SEQUENCE_LENGTH)
            )
            checkpoint_batch_size = int(
                checkpoint_recipe.get("physical_batch_size", 1)
            )
            checkpoint_accumulation = int(
                checkpoint_recipe.get("accumulation", ACCUMULATION_STEPS)
            )
            checkpoint_effective_batch = int(
                checkpoint_recipe.get("effective_batch_sequences", EFFECTIVE_BATCH_SEQUENCES)
            )
            saved_model_config = state.get("model_config")
            if (
                state.get("format") != CHECKPOINT_FORMAT
                or (
                    _model_config_for_resume_comparison(saved_model_config or {})
                    != _model_config_for_resume_comparison(config.to_dict())
                )
                or state.get("tokenizer") != identity
            ):
                parser.error("resume checkpoint recipe/model/tokenizer identity mismatch")
            if float(checkpoint_recipe.get("peak_lr", 2e-4)) != args.smollm3_peak_lr:
                parser.error("resume checkpoint peak learning rate differs")
            if checkpoint_effective_batch != args.smollm3_effective_batch_sequences:
                parser.error(
                    "resume checkpoint effective batch differs: "
                    f"checkpoint={checkpoint_effective_batch} "
                    f"requested={args.smollm3_effective_batch_sequences}"
                )
            if checkpoint_sequence_length != args.smollm3_sequence_length:
                context_growth = (
                    args.smollm3_allow_context_growth
                    and args.smollm3_sequence_length > checkpoint_sequence_length
                )
                if not context_growth:
                    parser.error(
                        "resume sequence length differs; context may only grow with "
                        "--smollm3-allow-context-growth"
                    )
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            provider.load_state_dict(
                state["data"], allow_window_resize=context_growth
            )
            update = int(state["update"])
            try:
                micro_step = _remap_resume_micro_step(
                    update,
                    int(state["micro_step"]),
                    checkpoint_batch_size=checkpoint_batch_size,
                    checkpoint_accumulation=checkpoint_accumulation,
                    requested_batch_size=args.batch_size,
                    requested_accumulation=args.gradient_accumulation_steps,
                    effective_batch_sequences=args.smollm3_effective_batch_sequences,
                )
            except ValueError as error:
                parser.error(str(error))
            tokens_seen = int(
                state.get(
                    "tokens_seen",
                    update
                    * checkpoint_sequence_length
                    * checkpoint_effective_batch,
                )
            )
            recipe_update = tokens_seen // (
                args.smollm3_sequence_length * args.smollm3_effective_batch_sequences
            )
            nonfinite_windows = int(state.get("nonfinite_windows", 0))
            best_loss = float(state.get("best_loss", float("inf")))
            torch.set_rng_state(state["torch_rng"])
            if torch.cuda.is_available() and state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            _assert_training_state_finite(
                model,
                optimizer,
                context=f"while loading checkpoint {args.resume_checkpoint!r}",
            )
            print(
                f"resume={'context_growth' if context_growth else 'exact'} "
                f"update={update} recipe_update={recipe_update} "
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
        accumulation_steps = args.gradient_accumulation_steps
        accumulation_loss = 0.0
        accumulation_sources: list[str] = []
        consecutive_nonfinite_windows = 0
        interval_loss = 0.0
        interval_updates = 0
        file_loss = 0.0
        file_updates = 0
        started = time.perf_counter()
        last_file_log = started
        optimizer.zero_grad(set_to_none=True)

        def save(path: Path) -> None:
            _assert_training_state_finite(
                model,
                optimizer,
                context=f"before checkpoint update={update}",
            )
            provider.set_progress(update=recipe_update, micro_step=micro_step)
            payload = {
                "format": CHECKPOINT_FORMAT,
                "update": update,
                "micro_step": micro_step,
                "recipe_update": recipe_update,
                "tokens_seen": tokens_seen,
                "nonfinite_windows": nonfinite_windows,
                "best_loss": best_loss,
                "model": model.state_dict(),
                "model_config": config.to_dict(),
                "optimizer": optimizer.state_dict(),
                "data": provider.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "tokenizer": identity,
                "recipe": {
                    "sequence_length": args.smollm3_sequence_length,
                    "physical_batch_size": args.batch_size,
                    "accumulation": accumulation_steps,
                    "effective_batch_sequences": args.smollm3_effective_batch_sequences,
                    "total_updates": total_updates_for_sequence_length(
                        args.smollm3_sequence_length,
                        args.smollm3_effective_batch_sequences,
                    ),
                    "peak_lr": args.smollm3_peak_lr,
                    "objective": "causal_next_token_shifted_all_positions",
                },
            }
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(path)

        while update < final_update:
            provider.set_progress(update=recipe_update, micro_step=micro_step)
            batch = provider.sample_cpu(args.batch_size).to(device)
            accumulation_sources.extend(provider.last_source_names)
            input_ids, labels = _autoregressive_batch(batch)
            mark_compiled_iteration_begin(compile_config, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=(
                    args.mixed_precision == "bf16"
                    and device.type in {"cuda", "cpu"}
                ),
            ):
                hidden, _ = model.forward_hidden(input_ids)
                if args.ssm_fused_linear_ce:
                    loss = _fused_linear_cross_entropy(model, hidden, labels)
                else:
                    loss = model.chunked_cross_entropy(
                        hidden,
                        labels,
                        chunk_rows=args.ssm_ce_chunk_rows,
                    )
            try:
                _require_finite_loss(
                    loss,
                    update=update,
                    micro_step=micro_step + 1,
                    sources=provider.last_source_names,
                )
            except FloatingPointError as error:
                micro_step, consecutive_nonfinite_windows = (
                    _discard_nonfinite_accumulation(
                        optimizer,
                        update=update,
                        accumulation_steps=accumulation_steps,
                        consecutive_failures=consecutive_nonfinite_windows,
                        max_consecutive_recoveries=(
                            args.smollm3_max_consecutive_nonfinite_windows
                        ),
                        error=error,
                    )
                )
                nonfinite_windows += 1
                accumulation_loss = 0.0
                accumulation_sources.clear()
                continue

            (loss / accumulation_steps).backward()
            accumulation_loss += float(loss.detach())
            micro_step += 1
            if micro_step % accumulation_steps:
                continue

            next_update = update + 1
            next_recipe_update = recipe_update + 1
            lr = learning_rate_for_update(
                next_recipe_update,
                peak_lr=args.smollm3_peak_lr,
                sequence_length=args.smollm3_sequence_length,
                effective_batch_sequences=args.smollm3_effective_batch_sequences,
                total_updates=args.steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            source_window = tuple(dict.fromkeys(accumulation_sources))
            try:
                grad = _clip_gradients_or_raise(
                    model,
                    update=update,
                    micro_step=micro_step,
                    sources=source_window,
                )
            except FloatingPointError as error:
                diagnostic = _gradient_failure_diagnostic(model)
                micro_step, consecutive_nonfinite_windows = (
                    _discard_nonfinite_accumulation(
                        optimizer,
                        update=update,
                        accumulation_steps=accumulation_steps,
                        consecutive_failures=consecutive_nonfinite_windows,
                        max_consecutive_recoveries=(
                            args.smollm3_max_consecutive_nonfinite_windows
                        ),
                        error=error,
                    )
                )
                nonfinite_windows += 1
                accumulation_loss = 0.0
                accumulation_sources.clear()
                event = (
                    "[sonic_ssm:train] event=nonfinite_window_discarded "
                    f"diagnostic=({diagnostic})"
                )
                print(event, flush=True)
                _append_log(log_path, event)
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            mean_update_loss = accumulation_loss / accumulation_steps
            interval_loss += mean_update_loss
            file_loss += mean_update_loss
            interval_updates += 1
            file_updates += 1
            accumulation_loss = 0.0
            accumulation_sources.clear()
            consecutive_nonfinite_windows = 0
            update = next_update
            recipe_update = next_recipe_update
            tokens_seen += (
                args.smollm3_sequence_length * args.smollm3_effective_batch_sequences
            )
            now = time.perf_counter()
            elapsed = now - started
            updates_completed = update - starting_update
            tokens_per_second = (
                updates_completed
                * args.smollm3_sequence_length
                * args.smollm3_effective_batch_sequences
                / max(elapsed, 1e-9)
            )

            if update % args.log_every == 0 or update == final_update:
                mean_loss = interval_loss / max(interval_updates, 1)
                print(
                    f"update={update}/{final_update} "
                    f"recipe_update={recipe_update}/{args.steps} "
                    f"stage={provider._stage_for_update(recipe_update) + 1} "
                    f"loss={mean_loss:.4f} ppl={math.exp(min(mean_loss, 20.0)):.3f} "
                    f"lr={lr:.3e} grad={grad:.3f} "
                    f"updates_per_second={updates_completed / max(elapsed, 1e-9):.3f} "
                    f"tokens_per_second={tokens_per_second:.1f} "
                    f"tokens={tokens_seen} nonfinite_windows={nonfinite_windows}",
                    flush=True,
                )
                interval_loss = 0.0
                interval_updates = 0

            if (
                update % args.checkpoint_every_updates == 0
                or update == final_update
            ):
                # Update best_loss (and save best.pt) before latest.pt, so
                # latest.pt's own stored best_loss is always current --
                # otherwise a resume reads back a stale value from before
                # this iteration's check.
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    save(out / "best.pt")
                save(out / "latest.pt")

            if (
                update % args.smollm3_fixed_eval_every_updates == 0
                or update == final_update
            ):
                _run_sonic_fixed_prompt_eval(
                    model,
                    eval_tokenizer,
                    device=device,
                    context_tokens=args.smollm3_sequence_length - args.response_tokens,
                    response_tokens=args.response_tokens,
                    mixed_precision=args.mixed_precision,
                    update=update,
                    eval_log_path=eval_log_path,
                    compile_config=compile_config,
                )

            if (
                args.log_every_seconds > 0
                and now - last_file_log >= args.log_every_seconds
            ) or update == final_update:
                mean_file_loss = file_loss / max(file_updates, 1)
                _append_log(
                    log_path,
                    f"timestamp={datetime.now(timezone.utc).isoformat()} "
                    f"update={update}/{final_update} "
                    f"recipe_update={recipe_update}/{args.steps} "
                    f"stage={provider._stage_for_update(recipe_update) + 1} "
                    f"loss={mean_file_loss:.4f} "
                    f"ppl={math.exp(min(mean_file_loss, 20.0)):.3f} "
                    f"lr={lr:.3e} grad={grad:.3f} "
                    f"tokens_per_second={tokens_per_second:.1f} "
                    f"tokens={tokens_seen} nonfinite_windows={nonfinite_windows}",
                )
                last_file_log = now
                file_loss = 0.0
                file_updates = 0

            if (
                args.max_runtime_seconds > 0
                and elapsed >= args.max_runtime_seconds
                and update < final_update
            ):
                save(out / "latest.pt")
                print(
                    f"sonic_ssm_runtime_limit update={update} "
                    f"seconds={elapsed:.1f} checkpoint={out / 'latest.pt'}",
                    flush=True,
                )
                break
    finally:
        provider.close()


__all__ = ["run"]
