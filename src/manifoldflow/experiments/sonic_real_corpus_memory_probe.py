"""Fifth training-viability probe: does the memory pathway help on real
text, not just a synthetic template - the first probe to train against
actual corpus data instead of the templated "Entity owns the object"
task every earlier probe used.

Source: the pretokenized `distractor-reasoning` corpus
(`Sonic-MF Training Design v0.1.md`'s SS9 "existing distractor generator"
output, 430 Nanotron/DataTrove `.ds` shards, ~27B tokens, tokenized with
the same `gtr_arw_english_50k` tokenizer this project already uses). A
manual sample confirmed it already interleaves distractor/binding tasks,
natural chat, topic switches, and relational composition in one
continuous stream - materially closer to design v0.1's session structure
than this project's own single-template synthetic generator.

This is deliberately the smallest useful next step, not an attempt at
design v0.1's full session/window/persistent-state machinery (SS4-5,
SS30-33 - multi-window persistent state surviving physical 8K
boundaries, batch-lane scheduling, causal state updates). Structurally
identical to the single-fact probe that started this project's
frozen-checkpoint line of work
(`sonic-pretrained-memory-probe.md`): one context segment, one
continuation segment, one write, one (trivial, single-basin) read, one
injection point. What's new is that both segments are now real,
unlabeled corpus text instead of a hand-built template, and the metric
is genuine language-modeling loss on the continuation (next-token
cross-entropy, standard LM training metric) rather than an exact-match
answer-accuracy score - there is no ground-truth "correct answer" to
grade against in raw corpus text, so this probe asks the more literal
version of the user's actual question: does conditioning on a memory of
the context measurably reduce loss on what comes after, on real data,
compared to a `no_memory` control that never sees the context at all.

`SonicKeyProjection` is intentionally not used here: with exactly one
segment written per example, retrieval is trivial by construction (there
is nothing to discriminate against), so a trainable key space has
nothing to learn yet - matching the reasoning that made the first
training probe skip it too, before discrimination became the question
worth asking. Sonic stays frozen (same reasoning as every
`sonic_pretrained_memory_*_probe.py`): only `SonicMemoryPort` trains.

Reads through `vendor/GTR-sonic-ssm/gator/smollm3_recipe.py`'s
`DataTroveTokenizedSource` - the real reader for this shard format,
already vendored, not reimplemented. Data source path is a WSL UNC path
(see `_DEFAULT_DATA_ROOT` below) since the corpus lives on the training
machine's local WSL filesystem, not something copied into this repo
(101GB).
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
import torch.nn.functional as F

from manifoldflow.continuous_memory import BasinMemory, Source
from manifoldflow.sonic_memory_port import SonicMemoryPort, SonicMemoryPortConfig
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel

_DEFAULT_CHECKPOINT = Path(
    "checkpoints/sonic_ssm/sonic_ssm_291k_steps_colab_baseline.pt"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TOKENIZER_DIR = (
    _PROJECT_ROOT / "resources" / "tokenizers" / "gtr_arw_english_50k"
)
# The corpus lives on the training machine's WSL filesystem either way;
# only the path spelling differs by which OS is running this process -
# the Windows .venv needs the UNC path, WSL's own Python (now the
# default way these probes run - see project_sonic_gpu_env.md) sees it
# as a plain local path.
_DEFAULT_DATA_ROOT = (
    "/home/user/sonic_data/tokenized"
    if os.name == "posix"
    else r"\\wsl.localhost\Ubuntu\home\user\sonic_data\tokenized"
)
_DATA_SOURCE_NAME = "distractor-reasoning"

sys.path.insert(0, str(_PROJECT_ROOT / "vendor" / "GTR-sonic-ssm"))


def _open_source(data_root: str):
    from gator.smollm3_recipe import DataTroveTokenizedSource, SourceSpec

    spec = SourceSpec(name=_DATA_SOURCE_NAME, repo="local", weight=1.0)
    return DataTroveTokenizedSource(spec, data_root)


def _read_window(
    source, context_length: int, continuation_length: int
) -> tuple[list[int], list[int]]:
    tokens = source.take(context_length + continuation_length)
    return tokens[:context_length], tokens[context_length:]


def _episode_step(
    *,
    model: SonicSSMLanguageModel,
    context_ids: list[int],
    continuation_ids: list[int],
    device: str,
    port: SonicMemoryPort | None,
    reduction: str = "mean",
    context_pooling: str = "mean",
) -> torch.Tensor:
    """Returns the continuation's LM loss - a scalar mean
    (`reduction="mean"`) or one value per predicted position
    (`reduction="none"`, length `len(continuation_ids)-1`; used for the
    per-token difficulty breakdown in `run()`, since a uniform mean over
    a highly templated corpus dilutes whatever signal exists on the
    small number of randomized, binding-relevant positions - see the
    module docstring's later note on this). `port=None` is the
    no-memory control - the continuation is processed with fresh state
    and never sees the context at all, matching every earlier probe's
    two-forward-pass separation (see the module docstring).

    `context_pooling`: 'last' pools the context's last-position hidden
    state instead of mean-pooling all context positions - the analogous
    write-side fix to sonic-correction-quantity-pooling-fix.md's dramatic
    result on the synthetic correction task, applied here since raw
    corpus text has no ground-truth "value span" to target directly the
    way that task's digit token could be. Tested and confirmed real but
    modest (sonic-real-corpus-context-pooling-last.md): this probe's
    original null result was already diagnosed, and independently
    cross-validated by the multi-query generator probe's large effect on
    a fix that guarantees the context-continuation dependency instead,
    as primarily *structural* - most random corpus windows simply aren't
    about their preceding context at all - not primarily a pooling
    problem. Pooling precision only helps once a real dependency exists,
    so 'last' measurably improves both the aggregate loss and the
    hard-token bucket specifically, but by nowhere near the order of
    magnitude the correction probe's fix produced."""
    if context_pooling not in ("mean", "last"):
        raise ValueError("context_pooling must be 'mean' or 'last'")
    if port is not None:
        context_tensor = torch.tensor([context_ids], device=device)
        with torch.no_grad():
            context_hidden, _ = model.forward_hidden(context_tensor)
        pooled = (
            context_hidden[0].mean(dim=0, keepdim=True)
            if context_pooling == "mean"
            else context_hidden[0, -1:, :]
        )
        context_repr = F.normalize(pooled, dim=-1)
        bank = BasinMemory(dimension=model.config.d_model, novelty_threshold=0.0)
        bank.write(context_repr.squeeze(0), "context", Source("user"))

    continuation_tensor = torch.tensor([continuation_ids], device=device)
    with torch.no_grad():
        hidden, _ = model.forward_hidden(continuation_tensor)

    if port is not None:
        query_repr = F.normalize(hidden[0].mean(dim=0, keepdim=True), dim=-1)
        hits = bank.read(query_repr.squeeze(0), top_k=1)
        retrieved = (
            bank.centers[hits[0].basin_id].unsqueeze(0)
            if hits
            else torch.zeros(1, model.config.d_model, device=device)
        )
        hidden = port(hidden, retrieved)

    logits = model.logits_from_hidden(hidden)
    target_ids = torch.tensor(continuation_ids, device=device)
    predicted_logits = logits[0, :-1, :]
    predicted_targets = target_ids[1:]
    return F.cross_entropy(predicted_logits, predicted_targets, reduction=reduction)


def _run_condition(
    *,
    condition: str,
    model: SonicSSMLanguageModel,
    train_source,
    eval_windows: list[tuple[list[int], list[int]]],
    steps: int,
    eval_every: int,
    learning_rate: float,
    device: str,
    port_rank: int,
    gate_init_bias: float,
    episodes_per_step: int,
    context_length: int,
    continuation_length: int,
    port_seed: int,
    context_pooling: str,
) -> dict[str, Any]:
    port = None
    optimizer = None
    if condition == "memory":
        torch.manual_seed(port_seed)
        port = SonicMemoryPort(
            SonicMemoryPortConfig(
                d_model=model.config.d_model,
                memory_dim=model.config.d_model,
                rank=port_rank,
                gate_init_bias=gate_init_bias,
            )
        ).to(device)
        optimizer = torch.optim.Adam(port.parameters(), lr=learning_rate)

    heartbeat_every = max(1, min(eval_every, 20))
    start_time = time.monotonic()
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        if step == 1 or step % heartbeat_every == 0:
            print(
                f"[{condition}] step {step}/{steps} "
                f"elapsed={time.monotonic() - start_time:.1f}s",
                flush=True,
            )
        if port is not None:
            optimizer.zero_grad()
            for _ in range(episodes_per_step):
                context_ids, continuation_ids = _read_window(
                    train_source, context_length, continuation_length
                )
                loss = _episode_step(
                    model=model,
                    context_ids=context_ids,
                    continuation_ids=continuation_ids,
                    device=device,
                    port=port,
                    context_pooling=context_pooling,
                )
                (loss / episodes_per_step).backward()
            optimizer.step()
        else:
            for _ in range(episodes_per_step):
                _read_window(train_source, context_length, continuation_length)

        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                losses = [
                    _episode_step(
                        model=model,
                        context_ids=context_ids,
                        continuation_ids=continuation_ids,
                        device=device,
                        port=port,
                        context_pooling=context_pooling,
                    ).item()
                    for context_ids, continuation_ids in eval_windows
                ]
            eval_lm_loss_mean = sum(losses) / len(losses)
            history.append({"step": step, "eval_lm_loss_mean": eval_lm_loss_mean})
            elapsed = time.monotonic() - start_time
            print(
                f"[{condition}] eval step {step}/{steps} elapsed={elapsed:.1f}s "
                f"lm_loss_mean={eval_lm_loss_mean:.4f}",
                flush=True,
            )

    return {"condition": condition, "history": history, "port": port}


def _difficulty_bucketed_breakdown(
    *,
    model: SonicSSMLanguageModel,
    eval_windows: list[tuple[list[int], list[int]]],
    device: str,
    port: SonicMemoryPort,
    hard_quantile: float,
    context_pooling: str,
) -> dict[str, Any]:
    """Per-token, not per-window, comparison: a uniform mean over a
    highly templated corpus (repeated scaffolding phrasing, only a few
    randomized name/number positions per window) dilutes whatever signal
    memory provides on the positions that actually require binding. This
    uses the `no_memory` condition's own per-position loss as a
    difficulty proxy - a token `no_memory` already predicts confidently
    is presumably scaffolding; one it finds hard is presumably where
    real content (a randomized name or number) had to be predicted - and
    compares the two conditions' mean loss restricted to the hardest
    `1 - hard_quantile` fraction of positions, pooled across all eval
    windows so the threshold is a single corpus-wide cutoff, not
    per-window."""
    with torch.no_grad():
        no_memory_per_token = [
            _episode_step(
                model=model,
                context_ids=context_ids,
                continuation_ids=continuation_ids,
                device=device,
                port=None,
                reduction="none",
            )
            for context_ids, continuation_ids in eval_windows
        ]
        memory_per_token = [
            _episode_step(
                model=model,
                context_ids=context_ids,
                continuation_ids=continuation_ids,
                device=device,
                port=port,
                reduction="none",
                context_pooling=context_pooling,
            )
            for context_ids, continuation_ids in eval_windows
        ]

    no_memory_flat = torch.cat(no_memory_per_token)
    memory_flat = torch.cat(memory_per_token)
    threshold = torch.quantile(no_memory_flat, hard_quantile).item()
    hard_mask = no_memory_flat >= threshold

    return {
        "hard_quantile": hard_quantile,
        "hard_token_loss_threshold": threshold,
        "hard_token_count": int(hard_mask.sum().item()),
        "total_token_count": int(no_memory_flat.numel()),
        "no_memory_hard_loss_mean": no_memory_flat[hard_mask].mean().item(),
        "memory_hard_loss_mean": memory_flat[hard_mask].mean().item(),
        "no_memory_easy_loss_mean": no_memory_flat[~hard_mask].mean().item(),
        "memory_easy_loss_mean": memory_flat[~hard_mask].mean().item(),
        "no_memory_all_loss_mean": no_memory_flat.mean().item(),
        "memory_all_loss_mean": memory_flat.mean().item(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", default=_DEFAULT_DATA_ROOT)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-windows", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--port-seed", type=int, default=17)
    parser.add_argument("--port-rank", type=int, default=64)
    parser.add_argument("--gate-init-bias", type=float, default=0.0)
    parser.add_argument("--episodes-per-step", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=192)
    parser.add_argument("--continuation-length", type=int, default=192)
    parser.add_argument("--eval-skip-tokens", type=int, default=50_000_000)
    parser.add_argument(
        "--context-pooling",
        choices=("mean", "last"),
        default="last",
        help=(
            "'last' (default): the write-side context representation is "
            "the context's own last-position hidden state, not a mean "
            "pool over all context positions - the analogous fix to "
            "sonic-correction-quantity-pooling-fix.md for raw corpus text, "
            "which has no known 'value span' to target directly. 'mean' "
            "reproduces this probe's original null-result behavior."
        ),
    )
    parser.add_argument(
        "--hard-quantile",
        type=float,
        default=0.75,
        help=(
            "Quantile of no_memory's own per-token loss used as the "
            "hard/easy cutoff for the difficulty-bucketed breakdown."
        ),
    )
    parser.add_argument(
        "--save-port-path",
        type=Path,
        default=None,
        help="Optional path to save the trained SonicMemoryPort's state_dict.",
    )
    parser.add_argument("--device", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=False
    )
    model_config = SonicSSMConfig(**checkpoint["model_config"])
    model = SonicSSMLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # Two independent readers positioned far apart in the same corpus, not
    # one reader split into train/eval - DataTroveTokenizedSource only
    # supports sequential reads, so a fixed, non-overlapping eval set is
    # built by skipping the eval reader far ahead of anywhere training
    # will reach (2000 steps * 8 episodes * 384 tokens ~= 6.1M tokens,
    # well under the default 50M-token eval skip).
    train_source = _open_source(args.data_root)
    eval_source = _open_source(args.data_root)
    eval_source.take(args.eval_skip_tokens)
    eval_windows = [
        _read_window(eval_source, args.context_length, args.continuation_length)
        for _ in range(args.eval_windows)
    ]
    eval_source.close()

    results = {}
    for condition in ("no_memory", "memory"):
        results[condition] = _run_condition(
            condition=condition,
            model=model,
            train_source=train_source,
            eval_windows=eval_windows,
            steps=args.steps,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
            device=device,
            port_rank=args.port_rank,
            gate_init_bias=args.gate_init_bias,
            episodes_per_step=args.episodes_per_step,
            context_length=args.context_length,
            continuation_length=args.continuation_length,
            port_seed=args.port_seed,
            context_pooling=args.context_pooling,
        )
    train_source.close()

    trained_port = results["memory"].pop("port")
    results["no_memory"].pop("port")
    if args.save_port_path is not None:
        args.save_port_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(trained_port.state_dict(), args.save_port_path)

    breakdown = _difficulty_bucketed_breakdown(
        model=model,
        eval_windows=eval_windows,
        device=device,
        port=trained_port,
        hard_quantile=args.hard_quantile,
        context_pooling=args.context_pooling,
    )

    final_no_memory = results["no_memory"]["history"][-1]["eval_lm_loss_mean"]
    final_memory = results["memory"]["history"][-1]["eval_lm_loss_mean"]
    output = {
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_update": checkpoint.get("update"),
        "checkpoint_tokens_seen": checkpoint.get("tokens_seen"),
        "model_config": model_config.to_dict(),
        "data_source": _DATA_SOURCE_NAME,
        "steps": args.steps,
        "episodes_per_step": args.episodes_per_step,
        "context_length": args.context_length,
        "continuation_length": args.continuation_length,
        "learning_rate": args.learning_rate,
        "gate_init_bias": args.gate_init_bias,
        "port_rank": args.port_rank,
        "eval_windows": args.eval_windows,
        "context_pooling": args.context_pooling,
        "results": results,
        "final_no_memory_lm_loss": final_no_memory,
        "final_memory_lm_loss": final_memory,
        "memory_reduced_loss": final_memory < final_no_memory,
        "difficulty_bucketed_breakdown": breakdown,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    args = _parser().parse_args()
    try:
        metrics = run(args)
    except BaseException:
        sys.stdout.flush()
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    summary = {k: v for k, v in metrics.items() if k != "results"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
