"""Seventh training-viability probe: per-fact writes, fixing the plateau
`sonic_distractor_generator_memory_probe.py` found and diagnosed.

That probe asked 6 heterogeneous questions against **one pooled
representation of the whole generated world** - the write side still
pooled every fact into a single vector before writing, exactly like
every probe since the very first one. Answer loss dropped 10x but
exact-match accuracy plateaued around 17.6%, and the reason was
specific: one vector has to carry enough about *every* fact to answer
*any* of several different questions about it, not just the one
question a simpler design pairs it with. The discrimination and
learned-retrieval probes already showed the fix for exactly this shape
of problem - write facts *separately* into the bank, train a key
projection with a contrastive loss so retrieval can tell them apart -
just never applied to real generator content until now.

Structural change from the multi-query probe: retrieval and injection
now happen **once per query, not once per episode**. Every fact in the
generated `World` (not just the ones a query will ask about - including
distractors, filler facts, and reported/hearsay facts, so retrieval
faces genuine competition, not a curated subset) gets its own basin, via
its own short forward pass and its own `SonicKeyProjection`-projected
key. Each direct query then runs as its own separate forward pass
(fresh state, matching every probe's two-pass separation) against the
*same* shared per-episode bank, reads back whichever basin its own
query key finds, and is scored on retrieval correctness (did it find the
fact the query is actually about) and answer accuracy/loss separately -
exactly the split the discrimination probe introduced, now applied to
real generator text instead of a template.

**Reverse ("who has N noun?") queries are dropped for this probe**,
documented here rather than silently: a reverse query can have several
equally correct target facts (multiple people can share the same count),
which doesn't map cleanly onto the single-correct-index InfoNCE loss the
learned-retrieval probe's contrastive design assumes. Supporting
multi-positive retrieval targets is a real extension, not attempted
here - this probe keeps the generator's own primary query plus
additional direct (subject, noun) queries only, where "the one fact this
question is about" is always unambiguous.
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
from manifoldflow.sonic_key_projection import (
    SonicKeyProjection,
    SonicKeyProjectionConfig,
    info_nce_retrieval_loss,
)
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
class QueryItem:
    question_ids: list[int]
    answer_ids: list[int]
    target_fact_index: int


@dataclass(frozen=True)
class Episode:
    fact_ids: list[list[int]]
    queries: list[QueryItem]


def _build_episode(
    generator: random.Random,
    tokenizer: Any,
    *,
    level: int,
    num_direct_queries: int,
) -> Episode:
    from distractor_reasoning.generate import generate_example
    from distractor_reasoning.query import Query, resolve_query
    from distractor_reasoning.render import render_answer, render_fact, render_question
    from distractor_reasoning.vocab import nouns_for

    split = "train"
    nouns = nouns_for(split)
    example = generate_example(generator, level=level, split=split)
    world = example.world

    fact_ids = [
        tokenizer.encode(
            render_fact(fact, generator, split, nouns), add_special_tokens=True
        )
        for fact in world.facts
    ]

    pair_to_index: dict[tuple[str, str], int] = {}
    for index, fact in enumerate(world.facts):
        if fact.is_having and not fact.is_reported:
            pair_to_index.setdefault((fact.subject, fact.noun), index)

    queries: list[QueryItem] = []

    def _add_query(query_obj, answer_value, target_index: int) -> None:
        question_text = f"Q: {render_question(query_obj, generator, split, nouns)}"
        answer_text = render_answer(query_obj, answer_value, generator, split, nouns)
        queries.append(
            QueryItem(
                question_ids=tokenizer.encode(question_text, add_special_tokens=True),
                answer_ids=tokenizer.encode(
                    f"\n{answer_text}", add_special_tokens=False
                ),
                target_fact_index=target_index,
            )
        )

    primary_key = (example.query.subject, example.query.target)
    primary_index = pair_to_index.get(primary_key)
    if primary_index is not None:
        _add_query(example.query, example.answer_value, primary_index)

    direct_candidates = [pair for pair in pair_to_index if pair != primary_key]
    generator.shuffle(direct_candidates)
    for subject, noun in direct_candidates[:num_direct_queries]:
        query_obj = Query(subject=subject, target=noun, is_category=False)
        answer = resolve_query(query_obj, world, nouns)
        _add_query(query_obj, answer, pair_to_index[(subject, noun)])

    return Episode(fact_ids=fact_ids, queries=queries)


@dataclass(frozen=True)
class QueryResult:
    decode_loss: torch.Tensor
    retrieval_loss: torch.Tensor
    answer_correct: bool
    retrieval_correct: bool


def _run_episode(
    *,
    model: SonicSSMLanguageModel,
    episode: Episode,
    device: str,
    port: SonicMemoryPort,
    key_projection: SonicKeyProjection,
    temperature: float,
) -> list[QueryResult]:
    """Writes every fact into a fresh per-episode bank once, then runs
    each query as its own independent forward pass (fresh state) against
    that shared bank - see the module docstring for why retrieval and
    injection had to move from once-per-episode to once-per-query."""
    if not episode.queries:
        return []

    bank = BasinMemory(dimension=model.config.d_model, novelty_threshold=0.0)
    fact_keys: list[torch.Tensor] = []
    for fact_ids in episode.fact_ids:
        fact_tensor = torch.tensor([fact_ids], device=device)
        with torch.no_grad():
            fact_hidden, _ = model.forward_hidden(fact_tensor)
        pooled = fact_hidden[0].mean(dim=0, keepdim=True)
        fact_key = key_projection(pooled).squeeze(0)
        fact_keys.append(fact_key)
        bank.write(fact_key.detach(), "fact", Source("user"))
    stacked_fact_keys = torch.stack(fact_keys)

    results: list[QueryResult] = []
    for query in episode.queries:
        qa_ids = query.question_ids + query.answer_ids
        qa_tensor = torch.tensor([qa_ids], device=device)
        with torch.no_grad():
            hidden, _ = model.forward_hidden(qa_tensor)
        answer_start = len(query.question_ids)
        answer_hidden = hidden[:, answer_start:, :]

        query_pooled = hidden[0, :answer_start].mean(dim=0, keepdim=True)
        query_key = key_projection(query_pooled).squeeze(0)
        hits = bank.read(query_key.detach(), top_k=1)
        retrieved = (
            bank.centers[hits[0].basin_id].unsqueeze(0)
            if hits
            else torch.zeros(1, model.config.d_model, device=device)
        )
        retrieval_correct = bool(hits and hits[0].basin_id == query.target_fact_index)
        retrieval_loss = info_nce_retrieval_loss(
            query_key,
            stacked_fact_keys,
            query.target_fact_index,
            temperature=temperature,
        )
        answer_hidden = port(answer_hidden, retrieved)

        logits = model.logits_from_hidden(answer_hidden)
        target_ids = torch.tensor(query.answer_ids, device=device)
        predicted_logits = logits[0, :-1, :]
        predicted_targets = target_ids[1:]
        decode_loss = F.cross_entropy(predicted_logits, predicted_targets)
        answer_correct = bool(
            (predicted_logits.argmax(dim=-1) == predicted_targets).all().item()
        )

        results.append(
            QueryResult(
                decode_loss=decode_loss,
                retrieval_loss=retrieval_loss,
                answer_correct=answer_correct,
                retrieval_correct=retrieval_correct,
            )
        )

    return results


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
    retrieval_loss_weight: float,
    temperature: float,
) -> dict[str, Any]:
    port = None
    key_projection = None
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
        key_projection = SonicKeyProjection(
            SonicKeyProjectionConfig(dim=model.config.d_model)
        ).to(device)
        optimizer = torch.optim.Adam(
            list(port.parameters()) + list(key_projection.parameters()),
            lr=learning_rate,
        )

    train_generator = random.Random(data_seed)
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        if port is not None:
            optimizer.zero_grad()
            step_results: list[QueryResult] = []
            for _ in range(episodes_per_step):
                episode = _build_episode(
                    train_generator,
                    tokenizer,
                    level=level,
                    num_direct_queries=num_direct_queries,
                )
                step_results.extend(
                    _run_episode(
                        model=model,
                        episode=episode,
                        device=device,
                        port=port,
                        key_projection=key_projection,
                        temperature=temperature,
                    )
                )
            if step_results:
                total_loss = sum(
                    (result.decode_loss + retrieval_loss_weight * result.retrieval_loss)
                    for result in step_results
                ) / len(step_results)
                total_loss.backward()
                optimizer.step()
        else:
            for _ in range(episodes_per_step):
                _build_episode(
                    train_generator,
                    tokenizer,
                    level=level,
                    num_direct_queries=num_direct_queries,
                )

        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                eval_results: list[QueryResult] = []
                for episode in eval_episodes:
                    eval_results.extend(
                        _run_episode(
                            model=model,
                            episode=episode,
                            device=device,
                            port=port,
                            key_projection=key_projection,
                            temperature=temperature,
                        )
                        if port is not None
                        else _run_episode_no_memory(
                            model=model, episode=episode, device=device
                        )
                    )
            if eval_results:
                row = {
                    "step": step,
                    "eval_answer_accuracy": sum(
                        r.answer_correct for r in eval_results
                    )
                    / len(eval_results),
                    "eval_decode_loss_mean": sum(
                        r.decode_loss.item() for r in eval_results
                    )
                    / len(eval_results),
                }
                if port is not None:
                    row["eval_retrieval_accuracy"] = sum(
                        r.retrieval_correct for r in eval_results
                    ) / len(eval_results)
                history.append(row)

    return {"condition": condition, "history": history}


def _run_episode_no_memory(
    *, model: SonicSSMLanguageModel, episode: Episode, device: str
) -> list[QueryResult]:
    """The no-memory control: each query still runs its own forward pass
    with fresh state, but there is no bank, no port, no retrieval - the
    query is answered from nothing but its own text, structurally unable
    to see any fact."""
    results: list[QueryResult] = []
    for query in episode.queries:
        qa_ids = query.question_ids + query.answer_ids
        qa_tensor = torch.tensor([qa_ids], device=device)
        hidden, _ = model.forward_hidden(qa_tensor)
        answer_start = len(query.question_ids)
        answer_hidden = hidden[:, answer_start:, :]
        logits = model.logits_from_hidden(answer_hidden)
        target_ids = torch.tensor(query.answer_ids, device=device)
        predicted_logits = logits[0, :-1, :]
        predicted_targets = target_ids[1:]
        decode_loss = F.cross_entropy(predicted_logits, predicted_targets)
        answer_correct = bool(
            (predicted_logits.argmax(dim=-1) == predicted_targets).all().item()
        )
        results.append(
            QueryResult(
                decode_loss=decode_loss,
                retrieval_loss=torch.tensor(0.0),
                answer_correct=answer_correct,
                retrieval_correct=False,
            )
        )
    return results


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
    parser.add_argument("--retrieval-loss-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
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
            retrieval_loss_weight=args.retrieval_loss_weight,
            temperature=args.temperature,
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
        "learning_rate": args.learning_rate,
        "gate_init_bias": args.gate_init_bias,
        "port_rank": args.port_rank,
        "retrieval_loss_weight": args.retrieval_loss_weight,
        "temperature": args.temperature,
        "eval_episodes": args.eval_episodes,
        "results": results,
        "final_no_memory_answer_accuracy": final_no_memory["eval_answer_accuracy"],
        "final_memory_answer_accuracy": final_memory["eval_answer_accuracy"],
        "final_memory_retrieval_accuracy": final_memory.get(
            "eval_retrieval_accuracy"
        ),
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
