"""Tenth training-viability probe: genuine cross-optimizer-step
persistence, via batch lanes - the piece
`sonic_session_correction_probe.py` explicitly deferred (that probe's
whole session was still one combined forward/backward pass; nothing in
this project until now has tested a fact surviving an actual step
boundary, gradient graph torn down and restarted, contents kept).

Design, matching the production recipe directly: a fixed number of
persistent **lanes** (a scaled-down stand-in for the production run's 96
sequences-per-batch, not a literal match), each holding its own bank
that survives across training steps rather than resetting every step,
plus an append-only, never-consumed `history` of every query the lane
has ever asked (so the same fact can legitimately be recalled more than
once at different gaps - design-v0.1.md SS22: "some are queried
repeatedly", not a one-shot queue). At every step, every lane always
writes a new episode into its (already-populated, ever-growing) bank and
appends that episode's own query to `history` - the bank keeps growing
regardless of what gets asked, matching "the target data is already
randomly dispersed... packed prompts" in real training. Separately, the
query actually *scored* this step is chosen as either:

- **the one just written** - "current episode," the same signal every
  earlier probe has trained on, just now against a bank that keeps the
  previous steps' writes too; or
- **a uniformly sampled older entry from `history`** - "prior episode."
  Selected with probability `prior_query_probability` (default 16/96,
  the user's own production ratio) whenever an eligible older entry
  exists yet (falls back to "current" for a lane's first few steps,
  before any history has accumulated).

Both conditions score against the *same* forward/backward mechanism
(key/value split, recency bonus, value-only accuracy - nothing new
there); what's new is that a "prior" query's target fact was written in
a genuinely separate optimizer step, with that step's own graph already
torn down. After every step, each lane's stored keys/values are
`.detach()`-ed before the next step runs - `sonic_session_correction_probe.py`
and design-v0.1.md SS5 both describe this the same way: "contents
survive, gradient history does not."

Each lane's session is capped at `session_length_steps` steps, after
which it resets (fresh bank, fresh history) and starts a new one -
design-v0.1.md SS31's "when a session finishes, that lane receives a
new session." A held-out eval pass mirrors the same lane mechanics on
its own separate, fixed set of lanes, evaluated periodically without
disturbing training's own lane states.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from manifoldflow.compute import (
    CompileConfig,
    Float32RuntimeConfig,
    configure_float32_runtime,
    mark_compiled_iteration_begin,
)
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

# Fixed padding shapes, mirroring sonic_correction_memory_probe.py's
# identical fix (sonic-correction-quantity-pooling-fix.md's module docs):
# --compile's model.compile_layers uses fullgraph=True, dynamic=False, so
# every distinct call shape forces a recompile and hard-fails past 8
# distinct shapes. Each step writes exactly one new episode per lane (not
# a whole session's worth), so these bounds are the same measured
# ceilings as the correction probe's (max 7 statements, 13 tokens/
# statement, 25 QA tokens across 2000 sampled level-5 episodes). Padding
# every lane's per-step write/query to these fixed shapes, batched across
# all lanes in one call, is what makes --compile usable and is the same
# batching this probe needs to reach anywhere near 96 lanes in reasonable
# wall-clock time - one Python-loop iteration and one forward call per
# lane per step, at 96 lanes, would be 6x this probe's original 16-lane
# cost for no reason.
_MAX_STATEMENTS_PER_STEP = 8
_MAX_STATEMENT_TOKENS = 16
_MAX_QA_TOKENS = 32


def _pad_batch(
    sequences: list[list[int]], *, rows: int, width: int, pad_id: int, device: str
) -> torch.Tensor:
    """Right-pad every sequence to `width` tokens and the batch to `rows`
    sequences (with all-pad dummy rows), so the returned tensor always has
    exactly shape [rows, width] regardless of the real content."""
    for sequence in sequences:
        if len(sequence) > width:
            raise ValueError(
                f"sequence length {len(sequence)} exceeds fixed padding "
                f"width {width} - raise the module's padding constant"
            )
    if len(sequences) > rows:
        raise ValueError(
            f"batch of {len(sequences)} sequences exceeds fixed padding "
            f"row count {rows} - raise the module's padding constant"
        )
    padded_rows = [
        sequence + [pad_id] * (width - len(sequence)) for sequence in sequences
    ]
    padded_rows += [[pad_id] * width] * (rows - len(sequences))
    return torch.tensor(padded_rows, device=device)


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
class QueuedQuery:
    question_ids: list[int]
    answer_ids: list[int]
    value_span: tuple[int, int]
    target_statement_index: int
    originating_step: int


@dataclass
class Lane:
    bank: BasinMemory
    statement_keys: list[torch.Tensor] = field(default_factory=list)
    statement_values: list[torch.Tensor] = field(default_factory=list)
    # Every past "current" query this lane has asked, in write order -
    # sampled from (not consumed) for "prior" asks, so the same fact can
    # legitimately be recalled more than once at different gaps (design-v0.1.md
    # SS22: "some are queried repeatedly").
    history: list[QueuedQuery] = field(default_factory=list)
    steps_since_reset: int = 0


def _new_lane(d_model: int) -> Lane:
    return Lane(bank=BasinMemory(dimension=d_model, novelty_threshold=0.0))


def _generate_episode_statements_and_query(
    generator: random.Random, tokenizer: Any, *, level: int, step: int
) -> tuple[list[list[int]], list[tuple[int, int]], QueuedQuery] | None:
    from distractor_reasoning.correction import (
        HAVING_RELATIONS_PLURAL,
        generate_correction_example,
    )
    from distractor_reasoning.query import UNKNOWN
    from distractor_reasoning.vocab import HAVING_RELATIONS, nouns_for

    split = "train"
    nouns = nouns_for(split)
    example = generate_correction_example(generator, level=level, split=split)
    if example.answer_value == UNKNOWN:
        return None

    relation = next(
        candidate
        for candidate in HAVING_RELATIONS
        if f" {candidate} " in example.answer_text
    )
    statement_ids: list[list[int]] = []
    statement_value_spans: list[tuple[int, int]] = []
    for statement in example.statements:
        prefix_text, quantity_text, suffix_text = _render_statement_pieces(
            statement, generator, nouns, relation
        )
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=True)
        quantity_ids = tokenizer.encode(quantity_text, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)
        statement_ids.append(prefix_ids + quantity_ids + suffix_ids)
        statement_value_spans.append(
            (len(prefix_ids), len(prefix_ids) + len(quantity_ids))
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

    query = QueuedQuery(
        question_ids=tokenizer.encode(question_text, add_special_tokens=True),
        answer_ids=prefix_ids + quantity_ids + suffix_ids,
        value_span=(len(prefix_ids), len(prefix_ids) + len(quantity_ids)),
        target_statement_index=local_target_index,  # rebased below by the caller
        originating_step=step,
    )
    return statement_ids, statement_value_spans, query


@dataclass(frozen=True)
class StepResult:
    decode_loss: torch.Tensor
    retrieval_loss: torch.Tensor
    answer_correct: bool
    value_correct: bool
    retrieval_correct: bool
    is_prior_query: bool
    # 0 for a current-episode query, else steps since the target was written
    step_gap: int


def _advance_lanes(
    *,
    lanes: list[Lane],
    generator: random.Random,
    tokenizer: Any,
    model: SonicSSMLanguageModel,
    device: str,
    port: SonicMemoryPort | None,
    key_projection: SonicKeyProjection | None,
    value_projection: SonicKeyProjection | None,
    level: int,
    step: int,
    session_length_steps: int,
    prior_query_probability: float,
    temperature: float,
    recency_weight: float,
    recency_decay: float,
    value_source: str = "full",
    mixed_precision: str = "fp32",
    pad_token_id: int = 0,
    sonic_requires_grad: bool = False,
) -> list[StepResult]:
    """Batched across every lane for one step - two forward_hidden calls
    total (one for every lane's one new episode this step, one for every
    lane's one scored query), not two per lane. Each lane keeps its own
    independent, ever-growing bank/history exactly as before; batching
    only changes how many forward calls it costs to advance all of them
    one step, not what's being measured (see sonic_correction_memory_probe.py's
    _run_episodes for the identical design decision and its rationale).

    `sonic_requires_grad`: see _run_episodes's identical parameter - a
    tensor produced inside torch.no_grad() carries no grad_fn regardless
    of whether the parameters that produced it require grad, so this must
    be a nullcontext (not torch.no_grad()) for --unfreeze-sonic-at-step
    to have any effect once its threshold is reached."""
    autocast_enabled = mixed_precision == "bf16" and device in ("cuda", "cpu")
    grad_context = (
        contextlib.nullcontext() if sonic_requires_grad else torch.no_grad()
    )

    def _autocast() -> torch.autocast:
        return torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled
        )

    for lane in lanes:
        if lane.steps_since_reset >= session_length_steps:
            lane.bank = _new_lane(model.config.d_model).bank
            lane.statement_keys = []
            lane.statement_values = []
            lane.history = []
            lane.steps_since_reset = 0

    generated_per_lane = []
    for _lane in lanes:
        generated = None
        while generated is None:
            generated = _generate_episode_statements_and_query(
                generator, tokenizer, level=level, step=step
            )
        generated_per_lane.append(generated)

    if port is not None:
        assert key_projection is not None
        statement_batch = torch.cat(
            [
                _pad_batch(
                    new_statement_ids,
                    rows=_MAX_STATEMENTS_PER_STEP,
                    width=_MAX_STATEMENT_TOKENS,
                    pad_id=pad_token_id,
                    device=device,
                )
                for new_statement_ids, _, _ in generated_per_lane
            ],
            dim=0,
        )
        with grad_context, _autocast():
            statement_hidden_batch, _ = model.forward_hidden(statement_batch)
        statement_hidden_batch = statement_hidden_batch.float()

        for lane_index, (lane, generated) in enumerate(
            zip(lanes, generated_per_lane, strict=True)
        ):
            new_statement_ids, new_statement_value_spans, new_query = generated
            base_index = len(lane.statement_keys)
            row_offset = lane_index * _MAX_STATEMENTS_PER_STEP
            for statement_index, (statement_ids, value_span) in enumerate(
                zip(new_statement_ids, new_statement_value_spans, strict=True)
            ):
                # The key always pools the whole statement - discrimination
                # benefits from the full sentence's context. The value,
                # when value_source="quantity", pools only the digit's own
                # hidden state instead
                # (sonic-correction-quantity-pooling-fix.md).
                statement_hidden = statement_hidden_batch[
                    row_offset + statement_index, : len(statement_ids)
                ]
                pooled = statement_hidden.mean(dim=0, keepdim=True)
                if value_source == "quantity":
                    start, end = value_span
                    value_pooled = statement_hidden[start:end].mean(
                        dim=0, keepdim=True
                    )
                else:
                    value_pooled = pooled
                statement_key = key_projection(pooled).squeeze(0)
                lane.bank.write(statement_key.detach(), "statement", Source("user"))
                lane.statement_keys.append(statement_key)
                lane.statement_values.append(
                    value_projection(value_pooled).squeeze(0)
                    if value_projection is not None
                    else statement_key
                )
            rebased_query = QueuedQuery(
                question_ids=new_query.question_ids,
                answer_ids=new_query.answer_ids,
                value_span=new_query.value_span,
                target_statement_index=base_index + new_query.target_statement_index,
                originating_step=step,
            )
            # history is append-only and never consumed, so the same fact
            # can legitimately be sampled as a "prior" query more than
            # once at different gaps - a growing pool to sample from, not
            # a queue.
            lane.history.append(rebased_query)
            lane.steps_since_reset += 1
    else:
        for lane, generated in zip(lanes, generated_per_lane, strict=True):
            _new_statement_ids, _new_statement_value_spans, new_query = generated
            base_index = len(lane.statement_keys)  # always 0 - never populated here
            rebased_query = QueuedQuery(
                question_ids=new_query.question_ids,
                answer_ids=new_query.answer_ids,
                value_span=new_query.value_span,
                target_statement_index=base_index + new_query.target_statement_index,
                originating_step=step,
            )
            lane.history.append(rebased_query)
            lane.steps_since_reset += 1

    queries: list[QueuedQuery] = []
    is_prior_list: list[bool] = []
    for lane in lanes:
        # exclude the one just appended - it has no age yet
        older_entries = lane.history[:-1]
        use_prior = (
            bool(older_entries) and generator.random() < prior_query_probability
        )
        queries.append(
            generator.choice(older_entries) if use_prior else lane.history[-1]
        )
        is_prior_list.append(use_prior)

    qa_batch = torch.cat(
        [
            _pad_batch(
                [query.question_ids + query.answer_ids],
                rows=1,
                width=_MAX_QA_TOKENS,
                pad_id=pad_token_id,
                device=device,
            )
            for query in queries
        ],
        dim=0,
    )
    with grad_context, _autocast():
        qa_hidden_batch, _ = model.forward_hidden(qa_batch)
    qa_hidden_batch = qa_hidden_batch.float()

    results: list[StepResult] = []
    for lane_index, (lane, query, use_prior) in enumerate(
        zip(lanes, queries, is_prior_list, strict=True)
    ):
        qa_len = len(query.question_ids) + len(query.answer_ids)
        hidden = qa_hidden_batch[lane_index : lane_index + 1, :qa_len, :]
        answer_start = len(query.question_ids)
        answer_hidden = hidden[:, answer_start:, :]

        if port is not None:
            assert key_projection is not None
            stacked_keys = torch.stack(lane.statement_keys)
            num_statements = len(lane.statement_keys)
            query_pooled = hidden[0, :answer_start].mean(dim=0, keepdim=True)
            query_key = key_projection(query_pooled).squeeze(0)
            top_k = num_statements if recency_weight > 0 else 1
            hits = lane.bank.read(query_key.detach(), top_k=top_k)
            if hits and recency_weight > 0:
                best_hit = max(
                    hits,
                    key=lambda hit: hit.score
                    + recency_weight
                    * recency_decay ** (num_statements - 1 - hit.basin_id),
                )
            elif hits:
                best_hit = hits[0]
            else:
                best_hit = None
            retrieved = (
                lane.statement_values[best_hit.basin_id].unsqueeze(0)
                if best_hit is not None
                else torch.zeros(1, model.config.d_model, device=device)
            )
            retrieval_correct = bool(
                best_hit is not None
                and best_hit.basin_id == query.target_statement_index
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
                StepResult(
                    decode_loss=decode_loss,
                    retrieval_loss=retrieval_loss,
                    answer_correct=answer_correct,
                    value_correct=value_correct,
                    retrieval_correct=retrieval_correct,
                    is_prior_query=use_prior,
                    step_gap=(step - query.originating_step) if use_prior else 0,
                )
            )
        else:
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
                StepResult(
                    decode_loss=decode_loss,
                    retrieval_loss=torch.tensor(0.0),
                    answer_correct=answer_correct,
                    value_correct=value_correct,
                    retrieval_correct=False,
                    is_prior_query=use_prior,
                    step_gap=(step - query.originating_step) if use_prior else 0,
                )
            )

    # Contents survive, gradient history does not: detach every stored
    # key/value in place before the next step's forward pass runs, so
    # backprop never needs to span a step boundary.
    if port is not None:
        for lane in lanes:
            lane.statement_keys = [key.detach() for key in lane.statement_keys]
            lane.statement_values = [
                value.detach() for value in lane.statement_values
            ]

    return results


def _run_condition(
    *,
    condition: str,
    model: SonicSSMLanguageModel,
    tokenizer: Any,
    steps: int,
    eval_every: int,
    eval_steps: int,
    learning_rate: float,
    device: str,
    port_rank: int,
    gate_init_bias: float,
    num_lanes: int,
    eval_lanes: int,
    session_length_steps: int,
    prior_query_probability: float,
    port_seed: int,
    data_seed: int,
    eval_seed: int,
    level: int,
    retrieval_loss_weight: float,
    temperature: float,
    recency_weight: float,
    recency_decay: float,
    value_source: str,
    compile_config: CompileConfig,
    mixed_precision: str,
    pad_token_id: int,
    unfreeze_sonic_at_step: int | None = None,
    sonic_learning_rate: float = 1e-5,
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
        param_groups: list[dict[str, Any]] = [
            {
                "params": (
                    list(port.parameters())
                    + list(key_projection.parameters())
                    + list(value_projection.parameters())
                ),
                "lr": learning_rate,
            }
        ]
        if unfreeze_sonic_at_step is not None:
            # See sonic_correction_memory_probe.py's identical staged-
            # unfreeze design (sonic-correction-staged-unfreeze.md): only
            # the "memory" condition unfreezes, Sonic's params join the
            # optimizer from the start regardless of the threshold (Adam
            # skips params with requires_grad=False until they start
            # producing gradients), and requires_grad flips exactly once,
            # in the training loop below, when the threshold is reached.
            param_groups.append(
                {"params": list(model.parameters()), "lr": sonic_learning_rate}
            )
        optimizer = torch.optim.Adam(param_groups)
    sonic_unfrozen = False  # tracks whether the threshold below has fired yet

    train_generator = random.Random(data_seed)
    train_lanes = [_new_lane(model.config.d_model) for _ in range(num_lanes)]

    # Heartbeat cadence for the training-step print below: frequent enough
    # that a run stuck or just slow (e.g. CPU-bound episode generation at
    # a high lane count - see sonic-batch-lane-persistence-96-lanes.md)
    # is visible well before the next eval checkpoint, not so frequent it
    # floods stdout.
    heartbeat_every = max(1, min(eval_every, 20))
    start_time = time.monotonic()
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        # Once per logical training iteration, before its first compiled
        # forward call - see sonic_correction_memory_probe.py's identical
        # comment for the full rationale; a documented no-op for this
        # project's default compile mode (max-autotune-no-cudagraphs).
        mark_compiled_iteration_begin(compile_config, device)
        if step == 1 or step % heartbeat_every == 0:
            print(
                f"[{condition}] step {step}/{steps} "
                f"elapsed={time.monotonic() - start_time:.1f}s",
                flush=True,
            )
        if (
            condition == "memory"
            and unfreeze_sonic_at_step is not None
            and not sonic_unfrozen
            and step >= unfreeze_sonic_at_step
        ):
            for parameter in model.parameters():
                parameter.requires_grad_(True)
            sonic_unfrozen = True
            print(
                f"[{condition}] unfreezing Sonic at step {step} "
                f"(sonic_learning_rate={sonic_learning_rate})",
                flush=True,
            )
        sonic_requires_grad = condition == "memory" and sonic_unfrozen
        if port is not None:
            optimizer.zero_grad()
            step_results = _advance_lanes(
                lanes=train_lanes,
                generator=train_generator,
                tokenizer=tokenizer,
                model=model,
                device=device,
                port=port,
                key_projection=key_projection,
                value_projection=value_projection,
                level=level,
                step=step,
                session_length_steps=session_length_steps,
                prior_query_probability=prior_query_probability,
                temperature=temperature,
                recency_weight=recency_weight,
                recency_decay=recency_decay,
                value_source=value_source,
                mixed_precision=mixed_precision,
                pad_token_id=pad_token_id,
                sonic_requires_grad=sonic_requires_grad,
            )
            total_loss = sum(
                (r.decode_loss + retrieval_loss_weight * r.retrieval_loss)
                for r in step_results
            ) / len(step_results)
            total_loss.backward()
            optimizer.step()
        else:
            _advance_lanes(
                lanes=train_lanes,
                generator=train_generator,
                tokenizer=tokenizer,
                model=model,
                device=device,
                port=None,
                key_projection=None,
                value_projection=None,
                level=level,
                step=step,
                session_length_steps=session_length_steps,
                prior_query_probability=prior_query_probability,
                temperature=temperature,
                recency_weight=recency_weight,
                recency_decay=recency_decay,
                value_source=value_source,
                mixed_precision=mixed_precision,
                pad_token_id=pad_token_id,
            )

        if step % eval_every == 0 or step == steps:
            mark_compiled_iteration_begin(compile_config, device)
            eval_generator = random.Random(eval_seed + step)
            eval_lanes_list = [
                _new_lane(model.config.d_model) for _ in range(eval_lanes)
            ]
            eval_results: list[StepResult] = []
            with torch.no_grad():
                for eval_step in range(1, eval_steps + 1):
                    eval_results.extend(
                        _advance_lanes(
                            lanes=eval_lanes_list,
                            generator=eval_generator,
                            tokenizer=tokenizer,
                            model=model,
                            device=device,
                            port=port,
                            key_projection=key_projection,
                            value_projection=value_projection,
                            level=level,
                            step=eval_step,
                            session_length_steps=session_length_steps,
                            prior_query_probability=prior_query_probability,
                            temperature=temperature,
                            recency_weight=recency_weight,
                            recency_decay=recency_decay,
                            value_source=value_source,
                            mixed_precision=mixed_precision,
                            pad_token_id=pad_token_id,
                            sonic_requires_grad=sonic_requires_grad,
                        )
                    )

            current_results = [r for r in eval_results if not r.is_prior_query]
            prior_results = [r for r in eval_results if r.is_prior_query]
            row: dict[str, Any] = {
                "step": step,
                "eval_current_value_accuracy": (
                    sum(r.value_correct for r in current_results) / len(current_results)
                    if current_results
                    else None
                ),
                "eval_prior_value_accuracy": (
                    sum(r.value_correct for r in prior_results) / len(prior_results)
                    if prior_results
                    else None
                ),
                "eval_prior_query_count": len(prior_results),
                "eval_current_query_count": len(current_results),
            }
            if port is not None and prior_results:
                row["eval_prior_retrieval_accuracy"] = sum(
                    r.retrieval_correct for r in prior_results
                ) / len(prior_results)
                by_gap: dict[int, list[bool]] = {}
                for r in prior_results:
                    by_gap.setdefault(r.step_gap, []).append(r.value_correct)
                row["eval_prior_value_accuracy_by_step_gap"] = {
                    str(gap): sum(values) / len(values)
                    for gap, values in sorted(by_gap.items())
                }
            history.append(row)
            elapsed = time.monotonic() - start_time
            print(
                f"[{condition}] eval step {step}/{steps} elapsed={elapsed:.1f}s "
                f"current_value_acc={row['eval_current_value_accuracy']} "
                f"prior_value_acc={row['eval_prior_value_accuracy']} "
                f"n_prior={row['eval_prior_query_count']}",
                flush=True,
            )

    return {"condition": condition, "history": history}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-dir", type=Path, default=_DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-seed", type=int, default=29)
    parser.add_argument("--eval-seed", type=int, default=31)
    parser.add_argument("--port-seed", type=int, default=17)
    parser.add_argument("--port-rank", type=int, default=64)
    parser.add_argument("--gate-init-bias", type=float, default=0.0)
    parser.add_argument("--num-lanes", type=int, default=16)
    parser.add_argument("--eval-lanes", type=int, default=8)
    parser.add_argument("--session-length-steps", type=int, default=6)
    parser.add_argument(
        "--prior-query-probability",
        type=float,
        default=16 / 96,
        help="Matches the production recipe's 16-of-96 batch composition.",
    )
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--retrieval-loss-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--recency-weight", type=float, default=0.3)
    parser.add_argument("--recency-decay", type=float, default=0.7)
    parser.add_argument(
        "--value-source",
        choices=("full", "quantity"),
        default="quantity",
        help=(
            "'quantity' (default): value_projection pools only the digit's "
            "own hidden state, the fix that resolved the answer-accuracy "
            "bottleneck in sonic_correction_memory_probe.py "
            "(sonic-correction-quantity-pooling-fix.md) - this probe has no "
            "prior canonical run to stay comparable with, so it defaults to "
            "the known-working configuration rather than reproducing the "
            "known-broken one. 'full': pools the whole rendered statement."
        ),
    )
    parser.add_argument(
        "--compile",
        choices=("none", "layers", "blocks"),
        default="layers",
        help=(
            "Sonic's own scoped compile path - see "
            "sonic_correction_memory_probe.py's identical flag. Defaults to "
            "'layers' (production's default), not 'none' - this probe has "
            "no prior canonical eager run to stay comparable with."
        ),
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
    parser.add_argument(
        "--unfreeze-sonic-at-step",
        type=int,
        default=None,
        help=(
            "Train Sonic's own weights jointly with the port/key/value "
            "projections in the 'memory' condition only, starting at this "
            "step - staged (a value well past where the frozen curve "
            "plateaus) resolved sonic_correction_memory_probe.py's "
            "immediate-unfreeze collapse entirely and beat the frozen "
            "ceiling (sonic-correction-staged-unfreeze.md); '1' reproduces "
            "unfreezing from the start (found to collapse there). Omit "
            "(default) to keep Sonic frozen throughout."
        ),
    )
    parser.add_argument(
        "--sonic-learning-rate",
        type=float,
        default=1e-5,
        help="Separate, smaller LR than --learning-rate for Sonic's own "
        "already-pretrained parameters.",
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    if len(tokenizer) != model_config.vocab_size:
        raise ValueError(
            f"tokenizer vocab size {len(tokenizer)} does not match "
            f"model vocab_size {model_config.vocab_size}"
        )
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    results = {}
    for condition in ("no_memory", "memory"):
        results[condition] = _run_condition(
            condition=condition,
            model=model,
            tokenizer=tokenizer,
            steps=args.steps,
            eval_every=args.eval_every,
            eval_steps=args.eval_steps,
            learning_rate=args.learning_rate,
            device=device,
            port_rank=args.port_rank,
            gate_init_bias=args.gate_init_bias,
            num_lanes=args.num_lanes,
            eval_lanes=args.eval_lanes,
            session_length_steps=args.session_length_steps,
            prior_query_probability=args.prior_query_probability,
            port_seed=args.port_seed,
            data_seed=args.data_seed,
            eval_seed=args.eval_seed,
            level=args.level,
            retrieval_loss_weight=args.retrieval_loss_weight,
            temperature=args.temperature,
            recency_weight=args.recency_weight,
            recency_decay=args.recency_decay,
            value_source=args.value_source,
            compile_config=compile_config,
            mixed_precision=args.mixed_precision,
            pad_token_id=pad_token_id,
            unfreeze_sonic_at_step=args.unfreeze_sonic_at_step,
            sonic_learning_rate=args.sonic_learning_rate,
        )

    final_no_memory = results["no_memory"]["history"][-1]
    final_memory = results["memory"]["history"][-1]
    output = {
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_update": checkpoint.get("update"),
        "checkpoint_tokens_seen": checkpoint.get("tokens_seen"),
        "model_config": model_config.to_dict(),
        "steps": args.steps,
        "num_lanes": args.num_lanes,
        "eval_lanes": args.eval_lanes,
        "session_length_steps": args.session_length_steps,
        "prior_query_probability": args.prior_query_probability,
        "unfreeze_sonic_at_step": args.unfreeze_sonic_at_step,
        "sonic_learning_rate": (
            args.sonic_learning_rate
            if args.unfreeze_sonic_at_step is not None
            else None
        ),
        "level": args.level,
        "learning_rate": args.learning_rate,
        "gate_init_bias": args.gate_init_bias,
        "port_rank": args.port_rank,
        "retrieval_loss_weight": args.retrieval_loss_weight,
        "temperature": args.temperature,
        "recency_weight": args.recency_weight,
        "recency_decay": args.recency_decay,
        "value_source": args.value_source,
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        "mixed_precision": args.mixed_precision,
        "scan_backend": args.scan_backend,
        "packed_projections": args.packed_projections,
        "matmul_precision": args.matmul_precision,
        "results": results,
        "final_no_memory_prior_value_accuracy": final_no_memory[
            "eval_prior_value_accuracy"
        ],
        "final_memory_prior_value_accuracy": final_memory["eval_prior_value_accuracy"],
        "final_memory_current_value_accuracy": final_memory[
            "eval_current_value_accuracy"
        ],
        "final_memory_prior_retrieval_accuracy": final_memory.get(
            "eval_prior_retrieval_accuracy"
        ),
        "memory_helped_on_prior_queries": (
            (final_memory["eval_prior_value_accuracy"] or 0.0)
            > (final_no_memory["eval_prior_value_accuracy"] or 0.0) + 0.1
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
