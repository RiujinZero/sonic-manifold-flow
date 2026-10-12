"""Ninth training-viability probe, and the first test of genuine
cross-episode persistence - every probe before this one reset its
`BasinMemory` bank fresh at the start of each example, the single
biggest, most-repeated gap flagged across this project's whole probe
lineage. Per explicit direction (chain related episodes into a session,
chain sessions into the training recipe), this is the first, smallest
slice of that: a **session** made of several independent correction
"episodes" (different subjects, each its own initial-value-plus-corrections
chain and its own query, exactly like `sonic_correction_memory_probe.py`'s
single episode) sharing **one bank that is never reset between them**.

Structure: write every sub-episode's statements first, in order,
accumulating into one growing shared bank (so a later sub-episode's
writes sit alongside, not instead of, an earlier one's) - then run every
sub-episode's own query *after* all writes are done, against the fully
populated, fully cluttered bank. That specifically tests whether an
early-written fact is still findable once several unrelated, later facts
have also been written - the direct test of "does this survive being a
session, not just an isolated example" - rather than the weaker version
where each query only ever faces whatever was written before it and
never sees later clutter at all.

Each query also records how many statements were written *after* its own
target fact and before the query itself - a distance measure, following
design-v0.1.md SS38's "retention curve" metric (accuracy vs. distance) -
reported per query in the output rather than only as an aggregate, so a
degradation-with-distance pattern (or its absence) is directly visible.

Reuses every mechanism validated so far without re-litigating them: the
key/value split and the write-order recency bonus
(`sonic-correction-key-value-split-and-recency-bonus.md`), and
`eval_value_accuracy` scored on just the quantity digits, not the whole
templated sentence (`sonic_correction_memory_probe.py`'s own fix for the
"is Sonic just dumb and exact unforgiving" question). What's new here is
purely the session structure - one shared, persistent bank instead of a
fresh one per example - not a new decode or retrieval mechanism.

**Deliberately not attempted here:** true multi-optimizer-step session
persistence with `detach_graph()`-style truncated backprop
(design-v0.1.md SS5 - state surviving *across separate training steps*,
not just across episodes within one forward/backward pass). This probe's
whole session is still processed as one combined gradient step, matching
this project's established pattern of testing the smallest slice of a
new capability before building the harder infrastructure around it.
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


def _render_statement_pieces(
    statement: Any, generator: random.Random, nouns: Any, relation: str
) -> tuple[str, str, str]:
    """Reimplements `render_statement`'s exact template logic (not a
    guess - `CORRECTION_TEMPLATES`/`INITIAL_TEMPLATES`/`OBSERVATION_TEMPLATES`
    each have exactly one `{quantity}` placeholder embedded in an
    otherwise-fixed string), returning (prefix, quantity_str, suffix)
    instead of one rendered string, so the quantity's own token span can
    be tracked for --value-source=quantity, mirroring
    sonic_correction_memory_probe.py's identical fix
    (sonic-correction-quantity-pooling-fix.md). Concatenating the three
    pieces reproduces `render_statement`'s output exactly."""
    from distractor_reasoning.correction import (
        CORRECTION_TEMPLATES,
        HAVING_RELATIONS_PLURAL,
        INITIAL_TEMPLATES,
        OBSERVATION_TEMPLATES,
    )

    noun_info = nouns[statement.noun]
    phrase = noun_info.singular if statement.quantity == 1 else noun_info.plural
    if statement.kind == "decoy_observation":
        template = generator.choice(OBSERVATION_TEMPLATES)
        before, after = template.split("{quantity}")
        return before, str(statement.quantity), after.format(
            noun_phrase=phrase, subject=statement.subject
        )

    subject_text = "They" if statement.pronoun else statement.subject
    rel = HAVING_RELATIONS_PLURAL[relation] if statement.pronoun else relation
    template_pool = (
        CORRECTION_TEMPLATES
        if statement.kind in ("correction", "decoy_correction")
        else INITIAL_TEMPLATES
    )
    template = generator.choice(template_pool)
    before, after = template.split("{quantity}")
    return (
        before.format(subject=subject_text, relation=rel),
        str(statement.quantity),
        after.format(noun_phrase=phrase),
    )


@dataclass(frozen=True)
class SessionQuery:
    question_ids: list[int]
    answer_ids: list[int]
    value_span: tuple[int, int]
    target_statement_index: int  # global index into Session.statement_ids
    # statements written after the target and before this query
    intervening_statement_count: int
    last_touching_statement_is_decoy_observation: bool


@dataclass(frozen=True)
class Session:
    statement_ids: list[list[int]]
    # [start, end) per statement, into that statement's own token ids - the
    # write-side analogue of each query's value_span, used only when
    # --value-source=quantity, mirroring sonic_correction_memory_probe.py.
    statement_value_spans: list[tuple[int, int]]
    queries: list[SessionQuery]


def _build_session(
    generator: random.Random, tokenizer: Any, *, level: int, episodes_per_session: int
) -> Session | None:
    from distractor_reasoning.correction import (
        HAVING_RELATIONS_PLURAL,
        generate_correction_example,
    )
    from distractor_reasoning.query import UNKNOWN
    from distractor_reasoning.vocab import HAVING_RELATIONS, nouns_for

    split = "train"
    nouns = nouns_for(split)

    all_statement_ids: list[list[int]] = []
    all_statement_value_spans: list[tuple[int, int]] = []
    all_statement_meta: list[tuple[str, str, str]] = []  # (subject, noun, kind)
    pending_queries: list[dict[str, Any]] = []

    for _ in range(episodes_per_session):
        example = generate_correction_example(generator, level=level, split=split)
        if example.answer_value == UNKNOWN:
            # Keep this first session probe simple - reject and retry at
            # the caller rather than handle UNKNOWN cases here too.
            return None

        relation = next(
            candidate
            for candidate in HAVING_RELATIONS
            if f" {candidate} " in example.answer_text
        )
        base_index = len(all_statement_ids)

        for statement in example.statements:
            prefix_text, quantity_text, suffix_text = _render_statement_pieces(
                statement, generator, nouns, relation
            )
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=True)
            quantity_ids = tokenizer.encode(quantity_text, add_special_tokens=False)
            suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
            all_statement_ids.append(prefix_ids + quantity_ids + suffix_ids)
            all_statement_value_spans.append(
                (len(prefix_ids), len(prefix_ids) + len(quantity_ids))
            )
            all_statement_meta.append(
                (statement.subject, statement.noun, statement.kind)
            )

        target_key = (example.query.subject, example.query.target)
        local_target_index = None
        for index, statement in enumerate(example.statements):
            if statement.kind == "decoy_observation":
                continue
            if (statement.subject, statement.noun) == target_key:
                local_target_index = index
        if local_target_index is None:
            return None
        global_target_index = base_index + local_target_index

        relation_base = HAVING_RELATIONS_PLURAL[relation]
        question_text = (
            f"Q: How many {nouns[example.query.target].plural} "
            f"does {example.query.subject} {relation_base}?"
        )
        quantity = example.answer_value
        noun_info = nouns[example.query.target]
        noun_phrase = noun_info.singular if quantity == 1 else noun_info.plural
        prefix_ids = tokenizer.encode(
            f"\nA: {example.query.subject} {relation} ", add_special_tokens=False
        )
        quantity_ids = tokenizer.encode(str(quantity), add_special_tokens=False)
        suffix_ids = tokenizer.encode(f" {noun_phrase}.", add_special_tokens=False)

        pending_queries.append(
            {
                "question_ids": tokenizer.encode(
                    question_text, add_special_tokens=True
                ),
                "answer_ids": prefix_ids + quantity_ids + suffix_ids,
                "value_span": (len(prefix_ids), len(prefix_ids) + len(quantity_ids)),
                "global_target_index": global_target_index,
                "last_statement_kind": example.statements[-1].kind,
            }
        )

    total_statements = len(all_statement_ids)
    queries = [
        SessionQuery(
            question_ids=q["question_ids"],
            answer_ids=q["answer_ids"],
            value_span=q["value_span"],
            target_statement_index=q["global_target_index"],
            intervening_statement_count=(
                total_statements - 1 - q["global_target_index"]
            ),
            last_touching_statement_is_decoy_observation=(
                q["last_statement_kind"] == "decoy_observation"
            ),
        )
        for q in pending_queries
    ]
    return Session(
        statement_ids=all_statement_ids,
        statement_value_spans=all_statement_value_spans,
        queries=queries,
    )


def _build_session_retrying(
    generator: random.Random, tokenizer: Any, *, level: int, episodes_per_session: int
) -> Session:
    while True:
        session = _build_session(
            generator, tokenizer, level=level, episodes_per_session=episodes_per_session
        )
        if session is not None:
            return session


@dataclass(frozen=True)
class QueryResult:
    decode_loss: torch.Tensor
    retrieval_loss: torch.Tensor
    answer_correct: bool
    value_correct: bool
    retrieval_correct: bool
    intervening_statement_count: int


def _run_session(
    *,
    model: SonicSSMLanguageModel,
    session: Session,
    device: str,
    port: SonicMemoryPort | None,
    key_projection: SonicKeyProjection | None,
    value_projection: SonicKeyProjection | None,
    temperature: float,
    recency_weight: float,
    recency_decay: float,
    value_source: str = "full",
) -> list[QueryResult]:
    if port is None:
        return [
            _run_query_no_memory(model=model, query=query, device=device)
            for query in session.queries
        ]

    assert key_projection is not None
    bank = BasinMemory(dimension=model.config.d_model, novelty_threshold=0.0)
    statement_keys: list[torch.Tensor] = []
    statement_values: list[torch.Tensor] = []
    for statement_ids, value_span in zip(
        session.statement_ids, session.statement_value_spans, strict=True
    ):
        statement_tensor = torch.tensor([statement_ids], device=device)
        with torch.no_grad():
            statement_hidden, _ = model.forward_hidden(statement_tensor)
        # The key always pools the whole statement - discrimination
        # benefits from the full sentence's context. The value, when
        # value_source="quantity", pools only the digit's own hidden state
        # instead (sonic-correction-quantity-pooling-fix.md).
        pooled = statement_hidden[0].mean(dim=0, keepdim=True)
        if value_source == "quantity":
            start, end = value_span
            value_pooled = statement_hidden[0, start:end].mean(dim=0, keepdim=True)
        else:
            value_pooled = pooled
        statement_key = key_projection(pooled).squeeze(0)
        statement_keys.append(statement_key)
        bank.write(statement_key.detach(), "statement", Source("user"))
        statement_values.append(
            value_projection(value_pooled).squeeze(0)
            if value_projection is not None
            else statement_key
        )
    stacked_keys = torch.stack(statement_keys)
    num_statements = len(session.statement_ids)

    results = []
    for query in session.queries:
        qa_ids = query.question_ids + query.answer_ids
        qa_tensor = torch.tensor([qa_ids], device=device)
        with torch.no_grad():
            hidden, _ = model.forward_hidden(qa_tensor)
        answer_start = len(query.question_ids)
        answer_hidden = hidden[:, answer_start:, :]

        query_pooled = hidden[0, :answer_start].mean(dim=0, keepdim=True)
        query_key = key_projection(query_pooled).squeeze(0)
        top_k = num_statements if recency_weight > 0 else 1
        hits = bank.read(query_key.detach(), top_k=top_k)
        if hits and recency_weight > 0:
            best_hit = max(
                hits,
                key=lambda hit: hit.score
                + recency_weight * recency_decay ** (num_statements - 1 - hit.basin_id),
            )
        elif hits:
            best_hit = hits[0]
        else:
            best_hit = None
        retrieved = (
            statement_values[best_hit.basin_id].unsqueeze(0)
            if best_hit is not None
            else torch.zeros(1, model.config.d_model, device=device)
        )
        retrieval_correct = bool(
            best_hit is not None and best_hit.basin_id == query.target_statement_index
        )
        retrieval_loss = info_nce_retrieval_loss(
            query_key,
            stacked_keys,
            query.target_statement_index,
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
        value_start = max(query.value_span[0] - 1, 0)
        value_end = query.value_span[1] - 1
        value_correct = bool(
            (
                predicted_logits[value_start:value_end].argmax(dim=-1)
                == predicted_targets[value_start:value_end]
            )
            .all()
            .item()
        )

        results.append(
            QueryResult(
                decode_loss=decode_loss,
                retrieval_loss=retrieval_loss,
                answer_correct=answer_correct,
                value_correct=value_correct,
                retrieval_correct=retrieval_correct,
                intervening_statement_count=query.intervening_statement_count,
            )
        )
    return results


def _run_query_no_memory(
    *, model: SonicSSMLanguageModel, query: SessionQuery, device: str
) -> QueryResult:
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
    value_start = max(query.value_span[0] - 1, 0)
    value_end = query.value_span[1] - 1
    value_correct = bool(
        (
            predicted_logits[value_start:value_end].argmax(dim=-1)
            == predicted_targets[value_start:value_end]
        )
        .all()
        .item()
    )
    return QueryResult(
        decode_loss=decode_loss,
        retrieval_loss=torch.tensor(0.0),
        answer_correct=answer_correct,
        value_correct=value_correct,
        retrieval_correct=False,
        intervening_statement_count=query.intervening_statement_count,
    )


def _run_condition(
    *,
    condition: str,
    model: SonicSSMLanguageModel,
    tokenizer: Any,
    eval_sessions: list[Session],
    steps: int,
    eval_every: int,
    learning_rate: float,
    device: str,
    port_rank: int,
    gate_init_bias: float,
    sessions_per_step: int,
    port_seed: int,
    data_seed: int,
    level: int,
    episodes_per_session: int,
    retrieval_loss_weight: float,
    temperature: float,
    recency_weight: float,
    recency_decay: float,
    value_source: str,
) -> dict[str, Any]:
    port = None
    key_projection = None
    value_projection = None
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
        value_projection = SonicKeyProjection(
            SonicKeyProjectionConfig(dim=model.config.d_model)
        ).to(device)
        optimizer = torch.optim.Adam(
            list(port.parameters())
            + list(key_projection.parameters())
            + list(value_projection.parameters()),
            lr=learning_rate,
        )

    train_generator = random.Random(data_seed)
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        if port is not None:
            optimizer.zero_grad()
            all_results: list[QueryResult] = []
            for _ in range(sessions_per_step):
                session = _build_session_retrying(
                    train_generator,
                    tokenizer,
                    level=level,
                    episodes_per_session=episodes_per_session,
                )
                all_results.extend(
                    _run_session(
                        model=model,
                        session=session,
                        device=device,
                        port=port,
                        key_projection=key_projection,
                        value_projection=value_projection,
                        temperature=temperature,
                        recency_weight=recency_weight,
                        recency_decay=recency_decay,
                        value_source=value_source,
                    )
                )
            total_loss = sum(
                (r.decode_loss + retrieval_loss_weight * r.retrieval_loss)
                for r in all_results
            ) / len(all_results)
            total_loss.backward()
            optimizer.step()
        else:
            for _ in range(sessions_per_step):
                _build_session_retrying(
                    train_generator,
                    tokenizer,
                    level=level,
                    episodes_per_session=episodes_per_session,
                )

        if step % eval_every == 0 or step == steps:
            with torch.no_grad():
                eval_results: list[QueryResult] = []
                for session in eval_sessions:
                    eval_results.extend(
                        _run_session(
                            model=model,
                            session=session,
                            device=device,
                            port=port,
                            key_projection=key_projection,
                            value_projection=value_projection,
                            temperature=temperature,
                            recency_weight=recency_weight,
                            recency_decay=recency_decay,
                            value_source=value_source,
                        )
                    )
            row: dict[str, Any] = {
                "step": step,
                "eval_answer_accuracy": sum(r.answer_correct for r in eval_results)
                / len(eval_results),
                "eval_value_accuracy": sum(r.value_correct for r in eval_results)
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
                # Retention curve (design-v0.1.md SS38): value accuracy
                # bucketed by how many statements intervened between the
                # target fact and its query - the direct test of whether
                # a session's later writes degrade recall of earlier ones.
                by_bucket: dict[str, list[bool]] = {}
                for r in eval_results:
                    if r.intervening_statement_count <= 2:
                        bucket = "0-2"
                    elif r.intervening_statement_count <= 5:
                        bucket = "3-5"
                    else:
                        bucket = "6+"
                    by_bucket.setdefault(bucket, []).append(r.value_correct)
                row["eval_value_accuracy_by_intervening_statements"] = {
                    bucket: sum(values) / len(values)
                    for bucket, values in sorted(by_bucket.items())
                }
            history.append(row)

    return {"condition": condition, "history": history}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-dir", type=Path, default=_DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-sessions", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=31)
    parser.add_argument("--port-seed", type=int, default=17)
    parser.add_argument("--port-rank", type=int, default=64)
    parser.add_argument("--gate-init-bias", type=float, default=0.0)
    parser.add_argument("--sessions-per-step", type=int, default=4)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--episodes-per-session", type=int, default=4)
    parser.add_argument("--retrieval-loss-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--recency-weight", type=float, default=0.3)
    parser.add_argument("--recency-decay", type=float, default=0.7)
    parser.add_argument(
        "--value-source",
        choices=("full", "quantity"),
        default="full",
        help=(
            "'full' (default, matches this probe's original canonical run): "
            "value_projection pools the whole rendered statement. "
            "'quantity': pools only the digit's own hidden state - the fix "
            "that resolved the same bottleneck in "
            "sonic_correction_memory_probe.py "
            "(sonic-correction-quantity-pooling-fix.md)."
        ),
    )
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
    eval_sessions = [
        _build_session_retrying(
            eval_generator,
            tokenizer,
            level=args.level,
            episodes_per_session=args.episodes_per_session,
        )
        for _ in range(args.eval_sessions)
    ]

    results = {}
    for condition in ("no_memory", "memory"):
        results[condition] = _run_condition(
            condition=condition,
            model=model,
            tokenizer=tokenizer,
            eval_sessions=eval_sessions,
            steps=args.steps,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
            device=device,
            port_rank=args.port_rank,
            gate_init_bias=args.gate_init_bias,
            sessions_per_step=args.sessions_per_step,
            port_seed=args.port_seed,
            data_seed=args.data_seed,
            level=args.level,
            episodes_per_session=args.episodes_per_session,
            retrieval_loss_weight=args.retrieval_loss_weight,
            temperature=args.temperature,
            recency_weight=args.recency_weight,
            recency_decay=args.recency_decay,
            value_source=args.value_source,
        )

    final_no_memory = results["no_memory"]["history"][-1]
    final_memory = results["memory"]["history"][-1]
    output = {
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_update": checkpoint.get("update"),
        "checkpoint_tokens_seen": checkpoint.get("tokens_seen"),
        "model_config": model_config.to_dict(),
        "steps": args.steps,
        "sessions_per_step": args.sessions_per_step,
        "level": args.level,
        "episodes_per_session": args.episodes_per_session,
        "learning_rate": args.learning_rate,
        "gate_init_bias": args.gate_init_bias,
        "port_rank": args.port_rank,
        "retrieval_loss_weight": args.retrieval_loss_weight,
        "temperature": args.temperature,
        "recency_weight": args.recency_weight,
        "recency_decay": args.recency_decay,
        "value_source": args.value_source,
        "eval_sessions": args.eval_sessions,
        "results": results,
        "final_no_memory_answer_accuracy": final_no_memory["eval_answer_accuracy"],
        "final_memory_answer_accuracy": final_memory["eval_answer_accuracy"],
        "final_no_memory_value_accuracy": final_no_memory["eval_value_accuracy"],
        "final_memory_value_accuracy": final_memory["eval_value_accuracy"],
        "final_memory_retrieval_accuracy": final_memory.get(
            "eval_retrieval_accuracy"
        ),
        "final_memory_value_accuracy_by_intervening_statements": final_memory.get(
            "eval_value_accuracy_by_intervening_statements"
        ),
        "memory_helped": (
            final_memory["eval_value_accuracy"]
            > final_no_memory["eval_value_accuracy"] + 0.1
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
