"""Eighth training-viability probe, and the first test of a real basic
mechanism rather than plain retrieval: can the system learn **recency**,
not just topical relevance - given several basins about the *same*
(subject, noun) pair, does a query find the most recent one, not just
any relevant one?

Per the user's explicit priority: retrieval (finding *a* relevant fact
among distractors) is the easier problem and is already validated
(`sonic-pretrained-memory-discrimination-probe.md`,
`sonic-pretrained-memory-learned-retrieval-probe.md`). Overwriting -
tracking a value that changes over time, the way a real world model has
to - is the harder, more basic mechanism, and untested until now.
Explicitly *not* yet attempting cross-episode/session persistence (state
carried across many separate examples) - that is the next tier after
this, per the same conversation; this probe stays within one episode's
scope, matching every probe so far.

Uses the real, vendored `distractor_reasoning.correction` generator
(`Kevin owns 7 parrots. / Correction: Kevin owns 4 parrots. / Maria owns
9 parrots. / Q: How many parrots does Kevin own? / A: 4.`) rather than
a hand-built chain - it already implements exactly the adversarial
construction this question needs: decoys placed *after* the real
correction specifically to defeat a "just grab the last number
mentioned anywhere" shortcut, a same-subject-different-noun decoy (binds
the correction to the right attribute, not just the right person), and
critically a `decoy_observation` - a mere *sighting* of a different
number near the subject, which must never override the tracked value
(observation vs. possession, a distinction this project's synthetic
probes have exercised since the very first training-viability probe).

**Why this is a genuinely open question, not a foregone conclusion:**
`BasinMemory` as built has no representation of write order or time -
a basin's position is a pure function of what was written, not when.
Each statement in a correction chain (initial value, then one or more
corrections) gets its own basin here, exactly like
`sonic_distractor_generator_per_fact_memory_probe.py`'s per-fact writes.
If the trainable retrieval key learns to prefer the most recent
statement, it can only be doing so via a *learnable correlate* of
recency - e.g. correction-phrased text ("Actually,"/"Wait,") looking
different from initial-assertion text - not from any explicit ordering
signal, because there isn't one. Whether that correlate is learnable
well enough to reliably track the *true* current value (not just "the
most correction-sounding one") is exactly what this probe measures,
honestly reported either way.

Same frozen-Sonic, per-fact-write, `SonicKeyProjection` +
`SonicMemoryPort` design as the per-fact probe - only the data and the
retrieval target (most recent matching statement, not "the one fact
about this pair") changed.

**Two additions from the canonical run's result**
(`sonic-correction-memory-probe.md`), both optional and independently
switchable so their effects can be told apart:

1. **Separate key and value projections** (`--separate-value-projection`,
   default on). The canonical run reused one `SonicKeyProjection` output
   as both the retrieval key *and* (via `bank.centers[...]`) the
   injected value - the same simplification
   `sonic-distractor-generator-per-fact-memory-probe.md` diagnosed as a
   likely cause of low answer accuracy despite working retrieval: a
   vector trained hard for discriminability has no reason to preserve
   the content needed to decode the right answer. This adds a second,
   independently-initialized `SonicKeyProjection` instance trained only
   through the decode loss, storing its output alongside (not inside)
   `BasinMemory` - the bank still only sees keys.

2. **An explicit write-order recency bonus** (`--recency-weight`,
   default 0.0 = off). This project already has a decaying-trace
   mechanism for biasing routing toward recent activity
   (`persistent_context.py`'s `PersistentBasinContext`), but its
   `alpha`/`decay` blend operates on a *sequence of queries*, decaying a
   running trace of what was recently *routed to* - this probe has one
   query per episode, not a query sequence, so that class doesn't apply
   as built. The same underlying idea (an explicit decay, not a learned
   one) adapts directly to what this task actually has: known write
   order. When `--recency-weight` is positive, retrieval reads back
   every candidate (`top_k=len(statements)`) and re-ranks by
   `similarity + recency_weight * recency_decay ** (age in write-order
   steps)`, favoring later-written basins.

   This is a genuine double-edged fix, not a free win, and is reported
   honestly either way: `decoy_observation` statements are deliberately
   placed *after* the real correction specifically to test whether a
   mere sighting gets wrongly treated as authoritative. A blind
   "most-recently-written wins" bonus rewards exactly that wrong
   behavior whenever an observation decoy happens to be the last
   statement in the chain. Kept off by default and only added
   deliberately, isolated from the key/value split, so its effect - help,
   hurt, or a wash - can be measured on its own.

3. **`eval_value_accuracy`, separate from `eval_answer_accuracy`.**
   `answer_correct` requires teacher-forced exact match on *every* token
   in the rendered answer - "A:", the subject name, the relation word,
   the quantity, and the noun's singular/plural form - so a single wrong
   digit anywhere fails the whole check, same as getting the entity name
   wrong while the number was right. That is a real, separate question
   from whether the memory system got the *value* correct, and the
   design doc's own SS37 metrics list already calls this out explicitly
   ("Value accuracy: correct value regardless of lexical entity decoding
   ... particularly useful given Sonic's current wrong-name/right-value
   behavior"). `_build_episode` now reconstructs the answer piecewise
   (prefix / quantity / suffix - `ANSWER_TEMPLATES` has exactly one
   template, so this is an exact reconstruction, not a guess) and tracks
   the quantity digits' own token span, so `value_correct` can be scored
   independently of the surrounding sentence.
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
from dataclasses import dataclass
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

# Fixed padding shapes for --compile: model.compile_layers uses
# fullgraph=True, dynamic=False (matching production's own usage - a
# packed, fixed-length 8192-context batch never changes shape mid-run),
# which means every distinct input shape triggers its own recompile and
# a fullgraph compile hard-fails once more than 8 distinct shapes have
# been seen. Every statement/question/answer in this generator's level-5
# output measured well under these bounds across 2000 sampled episodes
# (max 7 statements, 13 tokens/statement, 25 QA tokens) - padding every
# call to these fixed, generous ceilings makes every forward_hidden call
# in an entire run exactly the same shape, so compile traces once and
# reuses that graph for the rest of the run instead of recompiling per
# episode. This is also literally the same prerequisite real batching
# needs (a single call over multiple sequences requires them to share one
# shape) - compile and batching solve the same problem, not two.
_MAX_STATEMENTS_PER_EPISODE = 8
_MAX_STATEMENT_TOKENS = 16
_MAX_QA_TOKENS = 32


def _pad_batch(
    sequences: list[list[int]], *, rows: int, width: int, pad_id: int, device: str
) -> torch.Tensor:
    """Right-pad every sequence to `width` tokens and the batch to `rows`
    sequences (with all-pad dummy rows), so the returned tensor always has
    exactly shape [rows, width] regardless of the real content. Right
    padding (not left) keeps every real token at its original position -
    required since callers slice back specific spans (value_span) computed
    against the unpadded sequence."""
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
    be tracked - the write-side analogue of the answer's own `value_span`.
    Concatenating the three pieces reproduces `render_statement`'s output
    exactly."""
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
class Episode:
    statement_ids: list[list[int]]
    # [start, end) per statement, into that statement's own token ids -
    # the write-side analogue of value_span below, used only when
    # --value-source=quantity pools from the digit's own hidden state
    # instead of the whole statement.
    statement_value_spans: list[tuple[int, int]]
    question_ids: list[int]
    answer_ids: list[int]
    target_statement_index: int
    last_statement_is_decoy_observation: bool
    # [start, end) into answer_ids covering just the quantity digits - lets
    # _run_episodes score "did it get the number right" separately from
    # "did it match the entire templated sentence token-for-token." See
    # the module docstring: a single wrong digit anywhere fails the
    # all-positions exact-match answer_correct check even when the
    # quantity itself, or everything else, was right.
    value_span: tuple[int, int]


def _build_episode(
    generator: random.Random, tokenizer: Any, *, level: int
) -> Episode | None:
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
        return None  # excluded by construction at level 5, checked defensively

    # generate_correction_example picks one of HAVING_RELATIONS internally
    # (needed to render every statement identically) but doesn't expose it
    # on CorrectionExample - recover it from the one place it's visible,
    # example.answer_text, which always follows "{subject} {relation}
    # {quantity} {noun_phrase}." for a non-UNKNOWN case (guaranteed above).
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
    target_index = None
    for index, statement in enumerate(example.statements):
        if statement.kind == "decoy_observation":
            continue
        if (statement.subject, statement.noun) == target_key:
            target_index = index
    if target_index is None:
        return None

    relation_base = HAVING_RELATIONS_PLURAL[relation]
    question_text = (
        f"Q: How many {nouns[example.query.target].plural} "
        f"does {example.query.subject} {relation_base}?"
    )

    # Rebuild the answer piecewise (ANSWER_TEMPLATES has exactly one
    # template, "{subject} {relation} {quantity} {noun_phrase}.", so this
    # reconstruction is exact, not a guess) instead of tokenizing
    # example.answer_text as one block, specifically to get the quantity
    # digits' own token span for value_span - answer_correct alone can't
    # tell "got the number right" apart from "got the whole templated
    # sentence right."
    quantity = example.answer_value
    noun_info = nouns[example.query.target]
    noun_phrase = noun_info.singular if quantity == 1 else noun_info.plural
    prefix_ids = tokenizer.encode(
        f"\nA: {example.query.subject} {relation} ", add_special_tokens=False
    )
    quantity_ids = tokenizer.encode(str(quantity), add_special_tokens=False)
    suffix_ids = tokenizer.encode(f" {noun_phrase}.", add_special_tokens=False)
    answer_ids = prefix_ids + quantity_ids + suffix_ids
    value_span = (len(prefix_ids), len(prefix_ids) + len(quantity_ids))

    return Episode(
        statement_ids=statement_ids,
        statement_value_spans=statement_value_spans,
        question_ids=tokenizer.encode(question_text, add_special_tokens=True),
        answer_ids=answer_ids,
        target_statement_index=target_index,
        last_statement_is_decoy_observation=(
            example.statements[-1].kind == "decoy_observation"
        ),
        value_span=value_span,
    )


@dataclass(frozen=True)
class EpisodeResult:
    decode_loss: torch.Tensor
    retrieval_loss: torch.Tensor | None
    answer_correct: bool
    value_correct: bool
    retrieval_correct: bool | None


def _run_episodes(
    *,
    model: SonicSSMLanguageModel,
    episodes: list[Episode],
    device: str,
    port: SonicMemoryPort | None,
    key_projection: SonicKeyProjection | None,
    value_projection: SonicKeyProjection | None,
    temperature: float,
    recency_weight: float,
    recency_decay: float,
    value_source: str = "full",
    mixed_precision: str = "fp32",
    pad_token_id: int = 0,
    sonic_requires_grad: bool = False,
) -> list[EpisodeResult]:
    """Batched across every episode in `episodes` - two forward_hidden
    calls total (one for every episode's statements, one for every
    episode's question+answer), not two per episode. Each episode still
    gets its own independent BasinMemory bank and its own independent
    statement/query encodings (a separate batch row per statement, exactly
    as the single-episode version did) - batching here only means many
    independent rows share one forward call, not that episodes see each
    other's content; nothing about what's being measured changes, only
    how many Python-loop iterations and separate compiled-graph shapes it
    costs to measure it.

    `sonic_requires_grad`: every prior probe in this line kept Sonic
    frozen, so both forward_hidden calls were unconditionally wrapped in
    torch.no_grad() - harmless when nothing downstream needs a gradient
    through them, but a hard barrier that silently makes
    --unfreeze-sonic-at-step a no-op if left in place: a tensor produced
    inside torch.no_grad()
    carries no grad_fn regardless of whether the parameters that produced
    it require grad, so nothing downstream can backprop into Sonic's own
    weights through it. Skipping no_grad here (nullcontext instead) is
    what actually lets that happen."""
    autocast_enabled = mixed_precision == "bf16" and device in ("cuda", "cpu")
    grad_context = (
        contextlib.nullcontext() if sonic_requires_grad else torch.no_grad()
    )

    def _autocast() -> torch.autocast:
        return torch.autocast(
            device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled
        )

    num_episodes = len(episodes)
    banks: list[BasinMemory | None] = [None] * num_episodes
    per_episode_keys: list[list[torch.Tensor]] = [[] for _ in range(num_episodes)]
    per_episode_values: list[list[torch.Tensor]] = [[] for _ in range(num_episodes)]

    if port is not None:
        assert key_projection is not None
        # Every episode's statements, padded per-statement to
        # _MAX_STATEMENT_TOKENS and per-episode to
        # _MAX_STATEMENTS_PER_EPISODE dummy rows, then stacked
        # episode-major: rows [0:_MAX_STATEMENTS_PER_EPISODE] are episode
        # 0's, the next block is episode 1's, and so on - a fixed
        # [num_episodes * _MAX_STATEMENTS_PER_EPISODE, _MAX_STATEMENT_TOKENS]
        # shape for any given num_episodes (constant per call site: always
        # episodes_per_step for training, always eval_episodes for eval),
        # so --compile only ever sees (at most) two distinct shapes for
        # this call across a whole run.
        statement_batch = torch.cat(
            [
                _pad_batch(
                    episode.statement_ids,
                    rows=_MAX_STATEMENTS_PER_EPISODE,
                    width=_MAX_STATEMENT_TOKENS,
                    pad_id=pad_token_id,
                    device=device,
                )
                for episode in episodes
            ],
            dim=0,
        )
        with grad_context, _autocast():
            statement_hidden_batch, _ = model.forward_hidden(statement_batch)
        statement_hidden_batch = statement_hidden_batch.float()

        for episode_index, episode in enumerate(episodes):
            bank = BasinMemory(dimension=model.config.d_model, novelty_threshold=0.0)
            banks[episode_index] = bank
            row_offset = episode_index * _MAX_STATEMENTS_PER_EPISODE
            for statement_index, (statement_ids, value_span) in enumerate(
                zip(episode.statement_ids, episode.statement_value_spans, strict=True)
            ):
                # Only the real (unpadded) token positions - right padding
                # never influences an earlier position's hidden state in a
                # causal recurrent model, so slicing to the true length
                # here reproduces the unpadded, unbatched call exactly.
                statement_hidden = statement_hidden_batch[
                    row_offset + statement_index, : len(statement_ids)
                ]
                # The key always pools the whole statement - discrimination
                # benefits from the full sentence's context. The value,
                # when value_source="quantity", pools only the digit's own
                # hidden state instead - testing whether whole-sentence
                # pooling was diluting the one token that actually matters
                # (see the module docstring and
                # sonic-vanilla-correction-baseline.md).
                pooled = statement_hidden.mean(dim=0, keepdim=True)
                if value_source == "quantity":
                    start, end = value_span
                    value_pooled = statement_hidden[start:end].mean(
                        dim=0, keepdim=True
                    )
                else:
                    value_pooled = pooled
                statement_key = key_projection(pooled).squeeze(0)
                per_episode_keys[episode_index].append(statement_key)
                bank.write(statement_key.detach(), "statement", Source("user"))
                per_episode_values[episode_index].append(
                    value_projection(value_pooled).squeeze(0)
                    if value_projection is not None
                    else statement_key
                )

    qa_batch = torch.cat(
        [
            _pad_batch(
                [episode.question_ids + episode.answer_ids],
                rows=1,
                width=_MAX_QA_TOKENS,
                pad_id=pad_token_id,
                device=device,
            )
            for episode in episodes
        ],
        dim=0,
    )
    with grad_context, _autocast():
        qa_hidden_batch, _ = model.forward_hidden(qa_batch)
    qa_hidden_batch = qa_hidden_batch.float()  # see the write loop's identical cast

    results: list[EpisodeResult] = []
    for episode_index, episode in enumerate(episodes):
        qa_len = len(episode.question_ids) + len(episode.answer_ids)
        # [1, qa_len, d] - keep the batch dim so the rest of the logic
        # below (unchanged from the single-episode version) sees exactly
        # the shape it always did.
        hidden = qa_hidden_batch[episode_index : episode_index + 1, :qa_len, :]
        answer_start = len(episode.question_ids)
        answer_hidden = hidden[:, answer_start:, :]

        retrieval_correct = None
        retrieval_loss = None
        if port is not None:
            assert key_projection is not None
            bank = banks[episode_index]
            assert bank is not None
            stacked_keys = torch.stack(per_episode_keys[episode_index])
            query_pooled = hidden[0, :answer_start].mean(dim=0, keepdim=True)
            query_key = key_projection(query_pooled).squeeze(0)
            num_statements = len(episode.statement_ids)
            top_k = num_statements if recency_weight > 0 else 1
            hits = bank.read(query_key.detach(), top_k=top_k)
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
                per_episode_values[episode_index][best_hit.basin_id].unsqueeze(0)
                if best_hit is not None
                else torch.zeros(1, model.config.d_model, device=device)
            )
            retrieval_correct = bool(
                best_hit is not None
                and best_hit.basin_id == episode.target_statement_index
            )
            retrieval_loss = info_nce_retrieval_loss(
                query_key,
                stacked_keys,
                episode.target_statement_index,
                temperature=temperature,
            )
            answer_hidden = port(answer_hidden, retrieved)

        logits = model.logits_from_hidden(answer_hidden)
        target_ids = torch.tensor(episode.answer_ids, device=device)
        predicted_logits = logits[0, :-1, :]
        predicted_targets = target_ids[1:]
        decode_loss = F.cross_entropy(predicted_logits, predicted_targets)
        answer_correct = bool(
            (predicted_logits.argmax(dim=-1) == predicted_targets).all().item()
        )

        # value_span indexes answer_ids directly; predicted_targets[i] ==
        # answer_ids[i+1] (shifted-by-one next-token prediction), so the
        # matching slice into predicted_targets is (start-1, end-1).
        value_start = max(episode.value_span[0] - 1, 0)
        value_end = episode.value_span[1] - 1
        value_correct = bool(
            (
                predicted_logits[value_start:value_end].argmax(dim=-1)
                == predicted_targets[value_start:value_end]
            )
            .all()
            .item()
        )

        results.append(
            EpisodeResult(
                decode_loss=decode_loss,
                retrieval_loss=retrieval_loss,
                answer_correct=answer_correct,
                value_correct=value_correct,
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
    retrieval_loss_weight: float,
    temperature: float,
    separate_value_projection: bool,
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
        trainable = list(port.parameters()) + list(key_projection.parameters())
        if separate_value_projection:
            value_projection = SonicKeyProjection(
                SonicKeyProjectionConfig(dim=model.config.d_model)
            ).to(device)
            trainable += list(value_projection.parameters())
        param_groups: list[dict[str, Any]] = [
            {"params": trainable, "lr": learning_rate}
        ]
        if unfreeze_sonic_at_step is not None:
            # Only the "memory" condition unfreezes - "no_memory" stays the
            # existing frozen baseline, unaffected, since it runs first
            # (see run()'s condition loop) and this only touches the
            # condition that follows. Sonic's params join the optimizer's
            # param groups now, at their own much smaller LR than the
            # freshly-initialized port/projections' 1e-3 - naive
            # fine-tuning at adapter-scale LR risks catastrophic
            # forgetting - but requires_grad stays False until step
            # unfreeze_sonic_at_step is reached (see the training loop
            # below): Adam accepts params that don't currently require
            # grad in a param group and simply skips them in .step() until
            # they start producing gradients, so adding the group here
            # unconditionally (rather than only once the threshold hits)
            # is safe and keeps the optimizer's state simple.
            param_groups.append(
                {"params": list(model.parameters()), "lr": sonic_learning_rate}
            )
        optimizer = torch.optim.Adam(param_groups)
    sonic_unfrozen = False  # tracks whether the threshold below has fired yet

    train_generator = random.Random(data_seed)
    heartbeat_every = max(1, min(eval_every, 20))
    start_time = time.monotonic()
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        # Once per logical training iteration, before its first compiled
        # forward call - required for CUDA-graph compile modes under
        # gradient accumulation (see mark_compiled_iteration_begin's own
        # docstring); a documented no-op for compile_config.mode
        # "max-autotune-no-cudagraphs" (this project's default), compile
        # disabled, or a non-CUDA device.
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
            step_episodes: list[Episode] = []
            while len(step_episodes) < episodes_per_step:
                episode = _build_episode(train_generator, tokenizer, level=level)
                if episode is not None:
                    step_episodes.append(episode)
            step_results = _run_episodes(
                model=model,
                episodes=step_episodes,
                device=device,
                port=port,
                key_projection=key_projection,
                value_projection=value_projection,
                temperature=temperature,
                recency_weight=recency_weight,
                recency_decay=recency_decay,
                value_source=value_source,
                mixed_precision=mixed_precision,
                pad_token_id=pad_token_id,
                sonic_requires_grad=sonic_requires_grad,
            )
            losses = [
                result.decode_loss + retrieval_loss_weight * result.retrieval_loss
                for result in step_results
            ]
            (sum(losses) / len(losses)).backward()
            optimizer.step()
        else:
            episodes_collected = 0
            while episodes_collected < episodes_per_step:
                if _build_episode(train_generator, tokenizer, level=level) is not None:
                    episodes_collected += 1

        if step % eval_every == 0 or step == steps:
            # Eval is a separate logical iteration stream from the training
            # loop above (same rationale as the marker call at the top of
            # this loop).
            mark_compiled_iteration_begin(compile_config, device)
            with torch.no_grad():
                eval_results = _run_episodes(
                    model=model,
                    episodes=eval_episodes,
                    device=device,
                    port=port,
                    key_projection=key_projection,
                    value_projection=value_projection,
                    temperature=temperature,
                    recency_weight=recency_weight,
                    recency_decay=recency_decay,
                    value_source=value_source,
                    mixed_precision=mixed_precision,
                    pad_token_id=pad_token_id,
                    sonic_requires_grad=sonic_requires_grad,
                )
            row = {
                "step": step,
                "eval_answer_accuracy": sum(r.answer_correct for r in eval_results)
                / len(eval_results),
                "eval_value_accuracy": sum(r.value_correct for r in eval_results)
                / len(eval_results),
                "eval_decode_loss_mean": sum(r.decode_loss.item() for r in eval_results)
                / len(eval_results),
            }
            if port is not None:
                row["eval_retrieval_accuracy"] = sum(
                    r.retrieval_correct for r in eval_results
                ) / len(eval_results)
                # Isolated check for the double-edged case the module
                # docstring flags: episodes where the last-written
                # statement is a decoy_observation - a naive
                # write-order recency bonus is expected to hurt here
                # specifically, since it rewards treating a sighting as
                # authoritative.
                trap_pairs = [
                    (episode, result)
                    for episode, result in zip(eval_episodes, eval_results, strict=True)
                    if episode.last_statement_is_decoy_observation
                ]
                if trap_pairs:
                    row["eval_retrieval_accuracy_observation_trap_cases"] = sum(
                        result.retrieval_correct for _episode, result in trap_pairs
                    ) / len(trap_pairs)
                    row["observation_trap_case_count"] = len(trap_pairs)
            history.append(row)
            elapsed = time.monotonic() - start_time
            print(
                f"[{condition}] eval step {step}/{steps} elapsed={elapsed:.1f}s "
                f"answer_acc={row['eval_answer_accuracy']:.3f} "
                f"value_acc={row['eval_value_accuracy']:.3f} "
                f"retrieval_acc={row.get('eval_retrieval_accuracy')}",
                flush=True,
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
    parser.add_argument("--retrieval-loss-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--no-separate-value-projection",
        dest="separate_value_projection",
        action="store_false",
        help=(
            "Revert to the canonical run's design: one SonicKeyProjection "
            "output used as both retrieval key and injected value."
        ),
    )
    parser.set_defaults(separate_value_projection=True)
    parser.add_argument(
        "--recency-weight",
        type=float,
        default=0.0,
        help=(
            "Explicit write-order recency bonus added to retrieval ranking "
            "(0 = off, matching the canonical run). Double-edged - see the "
            "module docstring's decoy_observation caveat."
        ),
    )
    parser.add_argument("--recency-decay", type=float, default=0.7)
    parser.add_argument(
        "--value-source",
        choices=("full", "quantity"),
        default="full",
        help=(
            "'full' (canonical): value_projection pools the whole rendered "
            "statement. 'quantity': pools only the digit's own hidden "
            "state - tests whether whole-sentence pooling was diluting "
            "the one token that actually matters "
            "(sonic-vanilla-correction-baseline.md)."
        ),
    )
    parser.add_argument(
        "--compile",
        choices=("none", "layers", "blocks"),
        default="none",
        help=(
            "Sonic's own scoped compile path (model.compile_layers / "
            "compile_ssm_blocks) - the production entrypoint explicitly "
            "disables torch.compile-the-whole-model for Sonic in favor of "
            "this. 'none' (default) preserves every prior run's eager "
            "behavior exactly; 'layers' is production's default choice."
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
        help="Only used when --compile is not 'none'.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help=(
            "bf16 autocasts Sonic's own forward pass only - the trainable "
            "port/key/value projections stay fp32 regardless (see "
            "_run_episodes's cast-back-to-float comments)."
        ),
    )
    parser.add_argument(
        "--scan-backend",
        choices=("doubling", "hgrn"),
        default="doubling",
        help="doubling (default) is a real batched parallel scan already "
        "used by every prior probe in this line; hgrn is the Triton "
        "kernel path, CUDA-only.",
    )
    parser.add_argument(
        "--packed-projections",
        action="store_true",
        help=(
            "Fuse the SSM's b/c/dt input projections into one matmul. "
            "Requires n_heads=1 (true for this project's checkpoint). "
            "configure_execution's own docstring guarantees this does not "
            "change model semantics - a pure execution optimization."
        ),
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
            "projections in the 'memory' condition only ('no_memory' stays "
            "the existing frozen baseline), starting at this step - "
            "'--unfreeze-sonic-at-step 1' unfreezes immediately (found to "
            "collapse value accuracy: sonic-correction-unfreeze-sonic.md); "
            "a later value stages it - the port/projections converge with "
            "Sonic frozen first, only then does Sonic start moving. "
            "Omit (default) to keep Sonic frozen throughout, as in every "
            "prior run in this line."
        ),
    )
    parser.add_argument(
        "--sonic-learning-rate",
        type=float,
        default=1e-5,
        help=(
            "Separate, much smaller LR than --learning-rate's default 1e-3 "
            "for Sonic's own ~100M already-pretrained parameters - naive "
            "fine-tuning at adapter-scale LR risks catastrophic forgetting."
        ),
    )
    parser.add_argument("--device", default=None)
    return parser


def _build_eval_set(
    generator: random.Random, tokenizer: Any, *, level: int, count: int
) -> list[Episode]:
    episodes: list[Episode] = []
    while len(episodes) < count:
        episode = _build_episode(generator, tokenizer, level=level)
        if episode is not None:
            episodes.append(episode)
    return episodes


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

    eval_generator = random.Random(args.eval_seed)
    eval_episodes = _build_eval_set(
        eval_generator, tokenizer, level=args.level, count=args.eval_episodes
    )

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
            retrieval_loss_weight=args.retrieval_loss_weight,
            temperature=args.temperature,
            separate_value_projection=args.separate_value_projection,
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
        "separate_value_projection": args.separate_value_projection,
        "recency_weight": args.recency_weight,
        "recency_decay": args.recency_decay,
        "value_source": args.value_source,
        "unfreeze_sonic_at_step": args.unfreeze_sonic_at_step,
        "sonic_learning_rate": (
            args.sonic_learning_rate
            if args.unfreeze_sonic_at_step is not None
            else None
        ),
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        "mixed_precision": args.mixed_precision,
        "scan_backend": args.scan_backend,
        "packed_projections": args.packed_projections,
        "matmul_precision": args.matmul_precision,
        "steps": args.steps,
        "episodes_per_step": args.episodes_per_step,
        "level": args.level,
        "learning_rate": args.learning_rate,
        "gate_init_bias": args.gate_init_bias,
        "port_rank": args.port_rank,
        "retrieval_loss_weight": args.retrieval_loss_weight,
        "temperature": args.temperature,
        "eval_episodes": args.eval_episodes,
        "results": results,
        "final_no_memory_answer_accuracy": final_no_memory["eval_answer_accuracy"],
        "final_memory_answer_accuracy": final_memory["eval_answer_accuracy"],
        "final_no_memory_value_accuracy": final_no_memory["eval_value_accuracy"],
        "final_memory_value_accuracy": final_memory["eval_value_accuracy"],
        "final_memory_retrieval_accuracy": final_memory.get(
            "eval_retrieval_accuracy"
        ),
        "final_memory_retrieval_accuracy_observation_trap_cases": final_memory.get(
            "eval_retrieval_accuracy_observation_trap_cases"
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
