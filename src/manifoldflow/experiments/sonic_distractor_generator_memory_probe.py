"""Sixth training-viability probe: multiple recall questions per
generated world, instead of one random continuation window - a direct
answer to the density problem `sonic_real_corpus_memory_probe.py` ran
into.

That probe read raw windows from the pretokenized `distractor-reasoning`
corpus and scored uniform mean LM loss over the whole continuation. The
corpus is deliberately repetitive in scaffolding phrasing while
randomizing names/values to force real binding (not a bug - a design
choice, confirmed directly), but that means a random continuation window
is mostly predictable filler with only a few binding-relevant positions,
diluting whatever signal memory provides when averaged uniformly.

The fix implemented here: generate a `World` (a set of facts about
several entities) with the real, vendored `distractor_reasoning`
generator once, then ask it *several* different questions instead of
just the one question the generator naturally produces - reusing the
same context to extract more supervised signal per example, and giving
each question a token-tracked answer span so loss and exact-match
accuracy can be scored on the actual answer content specifically, not
diluted by question-phrasing tokens either. Two question kinds:

- **Direct** (what the generator itself produces): "{subject} - how many
  {noun}?" resolved by `distractor_reasoning.query.resolve_query`,
  reusing different (subject, noun) pairs already present in the same
  world's facts.
- **Reverse** ("who has N {noun}?"): the inverse lookup - given a
  (noun, quantity) pair taken from an existing fact, find every subject
  in the world with exactly that count. Not a query kind the vendored
  generator supports; built directly against `World.direct_having_facts()`
  in this probe, since it only needs the same structured fact list the
  generator already produces, no changes to vendored code.

Deliberately not attempted here: a "how many distinct animals were
named" category-count meta-question, and questions grounded in the
corpus's natural chat turns rather than the symbolic generator - the
former needs a bit more design thought, the latter needs real parsing of
freeform chat text (a different, harder problem than reading the
generator's own structured `World`) rather than reusing structured
facts. Both are natural follow-ups, not done in this pass.

Same frozen-Sonic, one-context/one-continuation, single-basin-trivial-retrieval
design as `sonic_real_corpus_memory_probe.py` (see that script and
`sonic-pretrained-memory-probe.md` for why each of those choices was
made) - only the content and the loss/accuracy scoring changed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import traceback
from dataclasses import dataclass
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

sys.path.insert(0, str(_PROJECT_ROOT / "vendor" / "GTR-sonic-ssm"))


@dataclass(frozen=True)
class Episode:
    context_ids: list[int]
    continuation_ids: list[int]
    answer_spans: list[tuple[int, int]]  # [start, end) indices into continuation_ids


def _build_episode(
    generator: random.Random,
    tokenizer: Any,
    *,
    level: int,
    num_direct_queries: int,
    num_reverse_queries: int,
) -> Episode:
    from distractor_reasoning.generate import generate_example
    from distractor_reasoning.query import Query, resolve_query
    from distractor_reasoning.render import render_answer, render_fact, render_question
    from distractor_reasoning.vocab import nouns_for

    split = "train"
    nouns = nouns_for(split)
    example = generate_example(generator, level=level, split=split)
    world = example.world
    facts = world.direct_having_facts()

    pair_to_quantity: dict[tuple[str, str], int] = {}
    for fact in facts:
        pair_to_quantity.setdefault((fact.subject, fact.noun), fact.quantity)

    qa_pairs: list[tuple[str, str]] = [
        (
            f"Q: {render_question(example.query, generator, split, nouns)}",
            render_answer(example.query, example.answer_value, generator, split, nouns),
        )
    ]
    used_pairs = {(example.query.subject, example.query.target)}

    direct_candidates = [pair for pair in pair_to_quantity if pair not in used_pairs]
    generator.shuffle(direct_candidates)
    for subject, noun in direct_candidates[:num_direct_queries]:
        query = Query(subject=subject, target=noun, is_category=False)
        answer = resolve_query(query, world, nouns)
        qa_pairs.append(
            (
                f"Q: {render_question(query, generator, split, nouns)}",
                render_answer(query, answer, generator, split, nouns),
            )
        )

    reverse_candidates = list({(fact.noun, fact.quantity) for fact in facts})
    generator.shuffle(reverse_candidates)
    for noun, quantity in reverse_candidates[:num_reverse_queries]:
        matching_subjects = sorted(
            {
                fact.subject
                for fact in facts
                if fact.noun == noun and fact.quantity == quantity
            }
        )
        noun_info = nouns[noun]
        noun_phrase = noun_info.singular if quantity == 1 else noun_info.plural
        question = f"Q: Who has {quantity} {noun_phrase}?"
        answer = f"A: {' and '.join(matching_subjects)}."
        qa_pairs.append((question, answer))

    ordered_facts = list(world.facts)
    generator.shuffle(ordered_facts)
    context_text = "\n".join(
        render_fact(fact, generator, split, nouns) for fact in ordered_facts
    )
    context_ids = tokenizer.encode(context_text, add_special_tokens=True)

    continuation_ids: list[int] = []
    answer_spans: list[tuple[int, int]] = []
    for question_text, answer_text in qa_pairs:
        question_ids = tokenizer.encode(
            f"\n{question_text}\n", add_special_tokens=False
        )
        continuation_ids.extend(question_ids)
        answer_start = len(continuation_ids)
        answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
        continuation_ids.extend(answer_ids)
        answer_spans.append((answer_start, len(continuation_ids)))

    return Episode(
        context_ids=context_ids,
        continuation_ids=continuation_ids,
        answer_spans=answer_spans,
    )


def _episode_step(
    *,
    model: SonicSSMLanguageModel,
    episode: Episode,
    device: str,
    port: SonicMemoryPort | None,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Returns (loss_on_answer_positions, loss_on_all_positions,
    correct_answer_count, total_answer_count). `port=None` is the
    no-memory control - fresh state, never sees the context."""
    if port is not None:
        context_tensor = torch.tensor([episode.context_ids], device=device)
        with torch.no_grad():
            context_hidden, _ = model.forward_hidden(context_tensor)
        context_repr = F.normalize(context_hidden[0].mean(dim=0, keepdim=True), dim=-1)
        bank = BasinMemory(dimension=model.config.d_model, novelty_threshold=0.0)
        bank.write(context_repr.squeeze(0), "context", Source("user"))

    continuation_tensor = torch.tensor([episode.continuation_ids], device=device)
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
    target_ids = torch.tensor(episode.continuation_ids, device=device)
    predicted_logits = logits[0, :-1, :]
    predicted_targets = target_ids[1:]
    per_token_loss = F.cross_entropy(
        predicted_logits, predicted_targets, reduction="none"
    )

    # Position i of predicted_logits/predicted_targets corresponds to
    # continuation index i+1 (it predicts token i+1 from the hidden state
    # after token i) - shift each answer span down by one accordingly,
    # and drop a span's first position if it starts at index 0 (its
    # predicting hidden state would be the never-injected pre-continuation
    # state, out of range here).
    answer_mask = torch.zeros_like(per_token_loss, dtype=torch.bool)
    correct = 0
    total = 0
    for start, end in episode.answer_spans:
        shifted_start = max(start - 1, 0)
        shifted_end = end - 1
        if shifted_end <= shifted_start:
            continue
        answer_mask[shifted_start:shifted_end] = True
        span_logits = predicted_logits[shifted_start:shifted_end]
        span_targets = predicted_targets[shifted_start:shifted_end]
        total += 1
        if bool((span_logits.argmax(dim=-1) == span_targets).all().item()):
            correct += 1

    answer_loss = (
        per_token_loss[answer_mask].mean()
        if answer_mask.any()
        else per_token_loss.mean()
    )
    return answer_loss, per_token_loss.mean(), correct, total


def _run_condition(
    *,
    condition: str,
    model: SonicSSMLanguageModel,
    tokenizer: Any,
    eval_episodes: list[Episode],
    steps: int,
    eval_every: int,
    learning_rate: float,
    device: str,
    port_rank: int,
    gate_init_bias: float,
    episodes_per_step: int,
    port_seed: int,
    data_seed: int,
    level: int,
    num_direct_queries: int,
    num_reverse_queries: int,
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

    train_generator = random.Random(data_seed)
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        if port is not None:
            optimizer.zero_grad()
            for _ in range(episodes_per_step):
                episode = _build_episode(
                    train_generator,
                    tokenizer,
                    level=level,
                    num_direct_queries=num_direct_queries,
                    num_reverse_queries=num_reverse_queries,
                )
                answer_loss, _all_loss, _correct, _total = _episode_step(
                    model=model, episode=episode, device=device, port=port
                )
                (answer_loss / episodes_per_step).backward()
            optimizer.step()
        else:
            for _ in range(episodes_per_step):
                _build_episode(
                    train_generator,
                    tokenizer,
                    level=level,
                    num_direct_queries=num_direct_queries,
                    num_reverse_queries=num_reverse_queries,
                )

        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                answer_losses = []
                all_losses = []
                correct_total = 0
                query_total = 0
                for episode in eval_episodes:
                    answer_loss, all_loss, correct, total = _episode_step(
                        model=model, episode=episode, device=device, port=port
                    )
                    answer_losses.append(answer_loss.item())
                    all_losses.append(all_loss.item())
                    correct_total += correct
                    query_total += total
            history.append(
                {
                    "step": step,
                    "eval_answer_loss_mean": sum(answer_losses) / len(answer_losses),
                    "eval_all_token_loss_mean": sum(all_losses) / len(all_losses),
                    "eval_answer_accuracy": correct_total / query_total,
                }
            )

    return {"condition": condition, "history": history}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-dir", type=Path, default=_DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=31)
    parser.add_argument("--port-seed", type=int, default=17)
    parser.add_argument("--port-rank", type=int, default=64)
    parser.add_argument("--gate-init-bias", type=float, default=0.0)
    parser.add_argument("--episodes-per-step", type=int, default=8)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--num-direct-queries", type=int, default=3)
    parser.add_argument("--num-reverse-queries", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    transformers = _require_transformers()
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    if len(tokenizer) != model_config.vocab_size:
        raise ValueError(
            f"tokenizer vocab size {len(tokenizer)} does not match "
            f"model vocab_size {model_config.vocab_size}"
        )

    eval_generator = random.Random(args.eval_seed)
    eval_episodes = [
        _build_episode(
            eval_generator,
            tokenizer,
            level=args.level,
            num_direct_queries=args.num_direct_queries,
            num_reverse_queries=args.num_reverse_queries,
        )
        for _ in range(args.eval_episodes)
    ]

    results = {}
    for condition in ("no_memory", "memory"):
        results[condition] = _run_condition(
            condition=condition,
            model=model,
            tokenizer=tokenizer,
            eval_episodes=eval_episodes,
            steps=args.steps,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
            device=device,
            port_rank=args.port_rank,
            gate_init_bias=args.gate_init_bias,
            episodes_per_step=args.episodes_per_step,
            port_seed=args.port_seed,
            data_seed=args.data_seed,
            level=args.level,
            num_direct_queries=args.num_direct_queries,
            num_reverse_queries=args.num_reverse_queries,
        )

    final_no_memory = results["no_memory"]["history"][-1]
    final_memory = results["memory"]["history"][-1]
    output = {
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_update": checkpoint.get("update"),
        "checkpoint_tokens_seen": checkpoint.get("tokens_seen"),
        "model_config": model_config.to_dict(),
        "steps": args.steps,
        "episodes_per_step": args.episodes_per_step,
        "level": args.level,
        "num_direct_queries": args.num_direct_queries,
        "num_reverse_queries": args.num_reverse_queries,
        "learning_rate": args.learning_rate,
        "gate_init_bias": args.gate_init_bias,
        "port_rank": args.port_rank,
        "eval_episodes": args.eval_episodes,
        "results": results,
        "final_no_memory_answer_accuracy": final_no_memory["eval_answer_accuracy"],
        "final_memory_answer_accuracy": final_memory["eval_answer_accuracy"],
        "final_no_memory_answer_loss": final_no_memory["eval_answer_loss_mean"],
        "final_memory_answer_loss": final_memory["eval_answer_loss_mean"],
        "memory_helped": (
            final_memory["eval_answer_accuracy"]
            > final_no_memory["eval_answer_accuracy"] + 0.1
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "This probe requires the qwen extra (shared tokenizer dependency): "
            "python -m pip install -e '.[qwen]'"
        ) from error
    return transformers


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
