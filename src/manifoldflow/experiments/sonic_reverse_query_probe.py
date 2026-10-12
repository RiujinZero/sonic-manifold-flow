"""Thirteenth probe: does the memory mechanism generalize to *reverse*
queries - "Who has 8 guitars?" instead of "How many guitars does Kevin
have?" - and if not, is the gap in retrieval or in decode?

Every probe in this project's whole correction/session/batch-lane line
asks the same query shape: subject given, quantity retrieved. The
`distractor_reasoning` generator has no reverse-query concept at all
(`Query` is hardcoded `(subject, target) -> quantity`
in `distractor_reasoning/query.py`) - this probe builds the reverse
framing itself, on top of the *same* generated episodes every other probe
in this line uses, rather than modifying the vendored generator.

Reverse queries are a genuinely different task on two axes at once, not
just a harder version of the same one:

- **Retrieval**: the query's own representation is built from
  quantity+object ("8 guitars"), not subject+object ("Kevin's guitars") -
  the trained key space was only ever asked to make *subject*-anchored
  queries close to the right statement's key, never quantity-anchored
  ones.
- **Decode**: the answer is an open-ended name (drawn from a small,
  bounded 20-name training vocabulary - see `distractor_reasoning.vocab.
  TRAIN_NAMES` - so bounded, not truly open-vocabulary, but still
  multi-token and higher-entropy than a single digit), not a low-
  cardinality quantity. This reopens the "is Sonic's own decode readout
  capable enough" question this project already had to answer once for
  digits (`sonic-vanilla-correction-baseline.md`,
  `sonic-correction-quantity-pooling-fix.md`).

Conflating both in one experiment would make a negative result
undiagnosable - if reverse queries fail, is it because the mechanism
can't *find* the right fact when queried by value, or because Sonic
can't *say* the name once it has the right fact handed to it? This probe
runs two modes to separate them:

- `--mode retrieval`: trains and evaluates *forward* queries exactly as
  `sonic_correction_memory_probe.py`'s already-validated setup does (the
  key space never sees a reverse query during training), but at eval
  time *also* scores a zero-shot reverse-query retrieval check: does the
  same trained key space, queried by quantity+object instead of
  subject+object, still find the right basin? Answers "does retrieval
  generalize to the reverse direction for free."
- `--mode decode`: skips real retrieval entirely - always injects the
  *correct* statement's own value vector (oracle, bypassing
  `BasinMemory.read()`) - and trains only the port + value projection to
  decode the *subject name* from the reverse question framing. Answers
  "can Sonic's decode readout handle an open-ended-ish target at all,
  independent of whether retrieval itself works."

Reverse queries can be genuinely ambiguous (two different subjects
legitimately having the same quantity of two different things) in a way
forward queries never are (a subject's own possessions are already
disambiguated by the correction-chain resolution). `_build_episode`
rejects (returns `None`, retried by the caller like every other
rejection case in this project's probes) any episode where the target
quantity is not unique among the episode's own possession-kind
statements, so every reverse query built here has exactly one correct
answer - but only when `require_unique_reverse=True`. `--mode decode`
trains directly on the reverse answer and always needs this; `--mode
retrieval` never trains on it at all (only scores a report-only
zero-shot metric with it) and runs with the rejection off, since
applying it there did nothing but silently bias that mode's own
training/eval population toward episodes with fewer decoys and
corrections (rejection probability grows with statement count) - see
`sonic-reverse-query-unfreeze-population-bias.md`.
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

# Same fixed padding shapes as sonic_correction_memory_probe.py, same
# rationale (measured against this generator's level-5 output) - kept
# generous enough to also cover the reverse question/answer, which is
# shorter than the forward one (no digit, just "Who has N X?" / a name).
_MAX_STATEMENTS_PER_EPISODE = 8
_MAX_STATEMENT_TOKENS = 16
_MAX_QA_TOKENS = 32


def _pad_batch(
    sequences: list[list[int]], *, rows: int, width: int, pad_id: int, device: str
) -> torch.Tensor:
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


def _recency_read(
    bank: BasinMemory,
    query_key: torch.Tensor,
    *,
    num_statements: int,
    recency_weight: float,
    recency_decay: float,
) -> Any:
    """Same write-order recency re-ranking sonic_correction_memory_probe.py
    validated (~60% -> ~83% retrieval accuracy on the identical
    plateau) - re-reads all candidates and re-ranks by
    `similarity + recency_weight * recency_decay ** age` instead of
    taking the bare top-1 similarity hit. `recency_weight=0.0` (default)
    reproduces the plain top-1 read exactly."""
    top_k = num_statements if recency_weight > 0 else 1
    hits = bank.read(query_key.detach(), top_k=top_k)
    if hits and recency_weight > 0:
        return max(
            hits,
            key=lambda hit: hit.score
            + recency_weight * recency_decay ** (num_statements - 1 - hit.basin_id),
        )
    return hits[0] if hits else None


def _render_statement_pieces(
    statement: Any, generator: random.Random, nouns: Any, relation: str
) -> tuple[str, str, str]:
    """Reimplements `render_statement`'s exact template logic - see
    sonic_correction_memory_probe.py's identical function for the full
    rationale."""
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
    statement_value_spans: list[tuple[int, int]]
    target_statement_index: int
    last_statement_is_decoy_observation: bool
    # forward: "How many guitars does Kevin have?" / "Kevin has 8 guitars."
    question_ids: list[int]
    answer_ids: list[int]
    value_span: tuple[int, int]
    # reverse: "Who has 8 guitars?" / "Kevin."
    reverse_question_ids: list[int]
    reverse_answer_ids: list[int]


def _build_episode(
    generator: random.Random,
    tokenizer: Any,
    *,
    level: int,
    require_unique_reverse: bool = True,
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
    target_index = None
    for index, statement in enumerate(example.statements):
        if statement.kind == "decoy_observation":
            continue
        if (statement.subject, statement.noun) == target_key:
            target_index = index
    if target_index is None:
        return None

    quantity = example.answer_value
    # Reverse queries need a unique answer: reject if any other
    # possession-kind statement (a different subject may have been
    # generated with the same quantity for a different noun, or even a
    # decoy_correction on a different subject/noun) shares this exact
    # quantity - forward queries never need this check since a subject's
    # own resolved value is already disambiguated by the correction
    # chain, but "who has 8 X" is only well-posed if 8 is the unique
    # quantity among what's actually written.
    #
    # require_unique_reverse=False (used by --mode retrieval, which never
    # trains on reverse_answer_ids at all - only scores a report-only
    # zero-shot metric with it) skips this rejection. Applying it
    # unconditionally silently biased retrieval mode's training/eval
    # population toward episodes with fewer decoys/corrections (rejection
    # probability grows with statement count), a survivorship bias with
    # no relevance to what that mode actually trains on - see
    # sonic-reverse-query-unfreeze-population-bias.md.
    if require_unique_reverse:
        target_statement = example.statements[target_index]
        other_quantities = [
            statement.quantity
            for statement in example.statements
            if statement.kind != "decoy_observation"
            and statement is not target_statement
        ]
        if quantity in other_quantities:
            return None

    noun_info = nouns[example.query.target]
    noun_phrase = noun_info.singular if quantity == 1 else noun_info.plural
    relation_base = HAVING_RELATIONS_PLURAL[relation]

    question_text = (
        f"Q: How many {noun_info.plural} "
        f"does {example.query.subject} {relation_base}?"
    )
    prefix_ids = tokenizer.encode(
        f"\nA: {example.query.subject} {relation} ", add_special_tokens=False
    )
    quantity_ids = tokenizer.encode(str(quantity), add_special_tokens=False)
    suffix_ids = tokenizer.encode(f" {noun_phrase}.", add_special_tokens=False)
    answer_ids = prefix_ids + quantity_ids + suffix_ids
    value_span = (len(prefix_ids), len(prefix_ids) + len(quantity_ids))

    reverse_question_text = f"Q: Who {relation} {quantity} {noun_phrase}?"
    reverse_answer_text = f"\nA: {example.query.subject}."

    return Episode(
        statement_ids=statement_ids,
        statement_value_spans=statement_value_spans,
        target_statement_index=target_index,
        last_statement_is_decoy_observation=(
            example.statements[-1].kind == "decoy_observation"
        ),
        question_ids=tokenizer.encode(question_text, add_special_tokens=True),
        answer_ids=answer_ids,
        value_span=value_span,
        reverse_question_ids=tokenizer.encode(
            reverse_question_text, add_special_tokens=True
        ),
        reverse_answer_ids=tokenizer.encode(
            reverse_answer_text, add_special_tokens=False
        ),
    )


@dataclass(frozen=True)
class RetrievalEpisodeResult:
    decode_loss: torch.Tensor
    retrieval_loss: torch.Tensor
    answer_correct: bool
    value_correct: bool
    retrieval_correct: bool
    reverse_retrieval_correct: bool


def _run_episodes_retrieval_mode(
    *,
    model: SonicSSMLanguageModel,
    episodes: list[Episode],
    device: str,
    port: SonicMemoryPort,
    key_projection: SonicKeyProjection,
    value_projection: SonicKeyProjection,
    temperature: float,
    pad_token_id: int,
    sonic_requires_grad: bool = False,
    recency_weight: float = 0.0,
    recency_decay: float = 0.7,
) -> list[RetrievalEpisodeResult]:
    """Forward-only training and forward decode scoring (identical shape
    to sonic_correction_memory_probe.py's validated design), plus a
    zero-shot reverse-query retrieval check at the end - the reverse
    query never contributes to any loss here, only to a reported metric.

    `sonic_requires_grad`: see sonic_correction_memory_probe.py's
    identical parameter - a tensor produced inside torch.no_grad()
    carries no grad_fn regardless of the model parameters' own
    requires_grad, which would silently make --unfreeze-sonic-at-step a
    no-op if these calls stayed hardcoded to torch.no_grad().

    `recency_weight`/`recency_decay`: the same write-order recency
    re-ranking sonic_correction_memory_probe.py already validated
    (sonic-correction-key-value-split-and-recency-bonus.md: ~60% -> ~83%
    retrieval accuracy) - this probe's own forward retrieval plateaued at
    the identical ~60% band the correction probe saw *before* that fix,
    because this design was built from the pre-recency-bonus retrieval
    path and never inherited it. Applied to both the forward query and
    the zero-shot reverse-query check, since both read the same bank
    with the same strategy."""
    grad_context = (
        contextlib.nullcontext() if sonic_requires_grad else torch.no_grad()
    )
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
    with grad_context:
        statement_hidden_batch, _ = model.forward_hidden(statement_batch)
    statement_hidden_batch = statement_hidden_batch.float()

    per_episode_keys: list[list[torch.Tensor]] = []
    per_episode_values: list[list[torch.Tensor]] = []
    banks: list[BasinMemory] = []
    for episode_index, episode in enumerate(episodes):
        bank = BasinMemory(dimension=model.config.d_model, novelty_threshold=0.0)
        keys: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        row_offset = episode_index * _MAX_STATEMENTS_PER_EPISODE
        for statement_index, statement_ids in enumerate(episode.statement_ids):
            statement_hidden = statement_hidden_batch[
                row_offset + statement_index, : len(statement_ids)
            ]
            pooled = statement_hidden.mean(dim=0, keepdim=True)
            key = key_projection(pooled).squeeze(0)
            keys.append(key)
            bank.write(key.detach(), "statement", Source("user"))
            values.append(value_projection(pooled).squeeze(0))
        banks.append(bank)
        per_episode_keys.append(keys)
        per_episode_values.append(values)

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
    with grad_context:
        qa_hidden_batch, _ = model.forward_hidden(qa_batch)
    qa_hidden_batch = qa_hidden_batch.float()

    reverse_qa_batch = torch.cat(
        [
            _pad_batch(
                [episode.reverse_question_ids],
                rows=1,
                width=_MAX_QA_TOKENS,
                pad_id=pad_token_id,
                device=device,
            )
            for episode in episodes
        ],
        dim=0,
    )
    with grad_context:
        reverse_hidden_batch, _ = model.forward_hidden(reverse_qa_batch)
    reverse_hidden_batch = reverse_hidden_batch.float()

    results: list[RetrievalEpisodeResult] = []
    for episode_index, episode in enumerate(episodes):
        qa_len = len(episode.question_ids) + len(episode.answer_ids)
        hidden = qa_hidden_batch[episode_index : episode_index + 1, :qa_len, :]
        answer_start = len(episode.question_ids)
        answer_hidden = hidden[:, answer_start:, :]

        stacked_keys = torch.stack(per_episode_keys[episode_index])
        query_pooled = hidden[0, :answer_start].mean(dim=0, keepdim=True)
        query_key = key_projection(query_pooled).squeeze(0)
        best_hit = _recency_read(
            banks[episode_index],
            query_key,
            num_statements=len(episode.statement_ids),
            recency_weight=recency_weight,
            recency_decay=recency_decay,
        )
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

        # Zero-shot reverse retrieval: never trained on, no loss - purely
        # a generalization check using the same key_projection.
        reverse_len = len(episode.reverse_question_ids)
        reverse_query_pooled = reverse_hidden_batch[
            episode_index, :reverse_len
        ].mean(dim=0, keepdim=True)
        reverse_query_key = key_projection(reverse_query_pooled).squeeze(0)
        reverse_best_hit = _recency_read(
            banks[episode_index],
            reverse_query_key,
            num_statements=len(episode.statement_ids),
            recency_weight=recency_weight,
            recency_decay=recency_decay,
        )
        reverse_retrieval_correct = bool(
            reverse_best_hit is not None
            and reverse_best_hit.basin_id == episode.target_statement_index
        )

        results.append(
            RetrievalEpisodeResult(
                decode_loss=decode_loss,
                retrieval_loss=retrieval_loss,
                answer_correct=answer_correct,
                value_correct=value_correct,
                retrieval_correct=retrieval_correct,
                reverse_retrieval_correct=reverse_retrieval_correct,
            )
        )
    return results


@dataclass(frozen=True)
class DecodeEpisodeResult:
    decode_loss: torch.Tensor
    name_correct: bool
    corrupted: bool


def _run_episodes_decode_mode(
    *,
    model: SonicSSMLanguageModel,
    episodes: list[Episode],
    device: str,
    port: SonicMemoryPort,
    value_projection: SonicKeyProjection,
    pad_token_id: int,
    sonic_requires_grad: bool = False,
    corruption_rate: float = 0.0,
    corruption_rng: random.Random | None = None,
) -> list[DecodeEpisodeResult]:
    """Oracle retrieval (the correct statement's own value, bypassing
    BasinMemory.read() entirely - no key_projection, no bank, no
    retrieval_loss) - isolates whether Sonic's decode readout can produce
    the right subject name at all, independent of whether real retrieval
    would have found it. `sonic_requires_grad`: see
    _run_episodes_retrieval_mode's identical parameter.

    `corruption_rate`/`corruption_rng`: the ablation this probe's own
    unfreeze results motivated (sonic-reverse-query-retrieval-freeze-
    corrected.md) - retrieval mode's real bank.read() occasionally grabs
    the *wrong* statement, and unfreezing Sonic there hurt decode twice
    while decode mode's oracle-always-correct unfreeze helped once. This
    reproduces that specific failure mode (a plausible-but-wrong
    statement from the same episode, not generic noise) at a controlled
    rate, isolating whether wrong-but-confident content - not anything
    else about joint real-retrieval training - is what breaks decode
    under unfreezing."""
    grad_context = (
        contextlib.nullcontext() if sonic_requires_grad else torch.no_grad()
    )
    source_indices: list[int] = []
    corrupted_flags: list[bool] = []
    for episode in episodes:
        other_indices = [
            index
            for index in range(len(episode.statement_ids))
            if index != episode.target_statement_index
        ]
        if (
            corruption_rate > 0
            and other_indices
            and corruption_rng is not None
            and corruption_rng.random() < corruption_rate
        ):
            source_indices.append(corruption_rng.choice(other_indices))
            corrupted_flags.append(True)
        else:
            source_indices.append(episode.target_statement_index)
            corrupted_flags.append(False)

    statement_batch = torch.cat(
        [
            _pad_batch(
                [episode.statement_ids[source_indices[episode_index]]],
                rows=1,
                width=_MAX_STATEMENT_TOKENS,
                pad_id=pad_token_id,
                device=device,
            )
            for episode_index, episode in enumerate(episodes)
        ],
        dim=0,
    )
    with grad_context:
        statement_hidden_batch, _ = model.forward_hidden(statement_batch)
    statement_hidden_batch = statement_hidden_batch.float()

    target_values: list[torch.Tensor] = []
    for episode_index, episode in enumerate(episodes):
        source_len = len(episode.statement_ids[source_indices[episode_index]])
        statement_hidden = statement_hidden_batch[episode_index, :source_len]
        pooled = statement_hidden.mean(dim=0, keepdim=True)
        target_values.append(value_projection(pooled).squeeze(0))

    reverse_qa_batch = torch.cat(
        [
            _pad_batch(
                [episode.reverse_question_ids + episode.reverse_answer_ids],
                rows=1,
                width=_MAX_QA_TOKENS,
                pad_id=pad_token_id,
                device=device,
            )
            for episode in episodes
        ],
        dim=0,
    )
    with grad_context:
        qa_hidden_batch, _ = model.forward_hidden(reverse_qa_batch)
    qa_hidden_batch = qa_hidden_batch.float()

    results: list[DecodeEpisodeResult] = []
    for episode_index, episode in enumerate(episodes):
        qa_len = len(episode.reverse_question_ids) + len(episode.reverse_answer_ids)
        hidden = qa_hidden_batch[episode_index : episode_index + 1, :qa_len, :]
        answer_start = len(episode.reverse_question_ids)
        answer_hidden = hidden[:, answer_start:, :]

        retrieved = target_values[episode_index].unsqueeze(0)
        answer_hidden = port(answer_hidden, retrieved)

        logits = model.logits_from_hidden(answer_hidden)
        target_ids = torch.tensor(episode.reverse_answer_ids, device=device)
        predicted_logits = logits[0, :-1, :]
        predicted_targets = target_ids[1:]
        decode_loss = F.cross_entropy(predicted_logits, predicted_targets)
        name_correct = bool(
            (predicted_logits.argmax(dim=-1) == predicted_targets).all().item()
        )
        results.append(
            DecodeEpisodeResult(
                decode_loss=decode_loss,
                name_correct=name_correct,
                corrupted=corrupted_flags[episode_index],
            )
        )
    return results


def _build_eval_set(
    generator: random.Random,
    tokenizer: Any,
    *,
    level: int,
    count: int,
    require_unique_reverse: bool = True,
) -> list[Episode]:
    episodes: list[Episode] = []
    while len(episodes) < count:
        episode = _build_episode(
            generator,
            tokenizer,
            level=level,
            require_unique_reverse=require_unique_reverse,
        )
        if episode is not None:
            episodes.append(episode)
    return episodes


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.unfreeze_sonic_at_step is not None
        and args.unfreeze_accuracy_threshold is not None
    ):
        raise ValueError(
            "--unfreeze-sonic-at-step and --unfreeze-accuracy-threshold "
            "are mutually exclusive - pick one trigger"
        )
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
    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    torch.manual_seed(args.port_seed)
    port = SonicMemoryPort(
        SonicMemoryPortConfig(
            d_model=model_config.d_model,
            memory_dim=model_config.d_model,
            rank=args.port_rank,
            gate_init_bias=args.gate_init_bias,
        )
    ).to(device)
    value_projection = SonicKeyProjection(
        SonicKeyProjectionConfig(dim=model_config.d_model)
    ).to(device)

    # retrieval mode never trains on reverse_answer_ids (only scores a
    # report-only zero-shot metric with it), so it doesn't need
    # _build_episode's reverse-uniqueness rejection - see that function's
    # docstring comment. decode mode strictly needs it (a well-posed
    # reverse answer is the whole point of what it trains on).
    require_unique_reverse = args.mode == "decode"
    eval_generator = random.Random(args.eval_seed)
    eval_episodes = _build_eval_set(
        eval_generator,
        tokenizer,
        level=args.level,
        count=args.eval_episodes,
        require_unique_reverse=require_unique_reverse,
    )
    train_generator = random.Random(args.data_seed)

    heartbeat_every = max(1, min(args.eval_every, 20))
    start_time = time.monotonic()
    history: list[dict[str, Any]] = []

    if args.mode == "retrieval":
        key_projection = SonicKeyProjection(
            SonicKeyProjectionConfig(dim=model_config.d_model)
        ).to(device)
        param_groups: list[dict[str, Any]] = [
            {
                "params": list(port.parameters())
                + list(key_projection.parameters())
                + list(value_projection.parameters()),
                "lr": args.learning_rate,
            }
        ]
        if (
            args.unfreeze_sonic_at_step is not None
            or args.unfreeze_accuracy_threshold is not None
        ):
            # Same design as sonic_correction_memory_probe.py's
            # --unfreeze-sonic-at-step: Sonic's params join the optimizer
            # now, at their own smaller LR, but requires_grad stays False
            # until the training loop below flips it - either at the
            # fixed step or once --unfreeze-accuracy-threshold's
            # convergence condition fires - Adam safely skips params with
            # no gradient in the meantime.
            param_groups.append(
                {"params": list(model.parameters()), "lr": args.sonic_learning_rate}
            )
        optimizer = torch.optim.Adam(param_groups)
        sonic_unfrozen = False
        actual_unfreeze_step: int | None = None
        consecutive_high_evals = 0
        for step in range(1, args.steps + 1):
            if step == 1 or step % heartbeat_every == 0:
                print(
                    f"step {step}/{args.steps} "
                    f"elapsed={time.monotonic() - start_time:.1f}s",
                    flush=True,
                )
            if (
                args.unfreeze_sonic_at_step is not None
                and not sonic_unfrozen
                and step >= args.unfreeze_sonic_at_step
            ):
                for parameter in model.parameters():
                    parameter.requires_grad_(True)
                sonic_unfrozen = True
                actual_unfreeze_step = step
                freeze_note = ""
                if args.freeze_retrieval_on_unfreeze:
                    for parameter in key_projection.parameters():
                        parameter.requires_grad_(False)
                    freeze_note = ", freezing key_projection"
                weight_note = (
                    f", post_unfreeze_retrieval_loss_weight="
                    f"{args.post_unfreeze_retrieval_loss_weight}"
                    if args.post_unfreeze_retrieval_loss_weight is not None
                    else ""
                )
                print(
                    f"unfreezing Sonic at step {step} "
                    f"(sonic_learning_rate={args.sonic_learning_rate})"
                    f"{freeze_note}{weight_note}",
                    flush=True,
                )
            optimizer.zero_grad()
            step_episodes: list[Episode] = []
            while len(step_episodes) < args.episodes_per_step:
                episode = _build_episode(
                    train_generator,
                    tokenizer,
                    level=args.level,
                    require_unique_reverse=require_unique_reverse,
                )
                if episode is not None:
                    step_episodes.append(episode)
            step_results = _run_episodes_retrieval_mode(
                model=model,
                episodes=step_episodes,
                device=device,
                port=port,
                key_projection=key_projection,
                value_projection=value_projection,
                temperature=args.temperature,
                pad_token_id=pad_token_id,
                sonic_requires_grad=sonic_unfrozen,
                recency_weight=args.recency_weight,
                recency_decay=args.recency_decay,
            )
            active_retrieval_loss_weight = (
                args.post_unfreeze_retrieval_loss_weight
                if sonic_unfrozen
                and args.post_unfreeze_retrieval_loss_weight is not None
                else args.retrieval_loss_weight
            )
            losses = [
                r.decode_loss + active_retrieval_loss_weight * r.retrieval_loss
                for r in step_results
            ]
            (sum(losses) / len(losses)).backward()
            optimizer.step()

            if step % args.eval_every == 0 or step == args.steps:
                with torch.no_grad():
                    eval_results = _run_episodes_retrieval_mode(
                        model=model,
                        episodes=eval_episodes,
                        device=device,
                        port=port,
                        key_projection=key_projection,
                        value_projection=value_projection,
                        temperature=args.temperature,
                        pad_token_id=pad_token_id,
                        sonic_requires_grad=False,
                        recency_weight=args.recency_weight,
                        recency_decay=args.recency_decay,
                    )
                row = {
                    "step": step,
                    "eval_value_accuracy": sum(
                        r.value_correct for r in eval_results
                    )
                    / len(eval_results),
                    "eval_retrieval_accuracy": sum(
                        r.retrieval_correct for r in eval_results
                    )
                    / len(eval_results),
                    "eval_reverse_retrieval_accuracy": sum(
                        r.reverse_retrieval_correct for r in eval_results
                    )
                    / len(eval_results),
                }
                history.append(row)
                streak_note = ""
                if args.unfreeze_accuracy_threshold is not None and not sonic_unfrozen:
                    # Gate the unfreeze on the *adapter's own* retrieval
                    # reliability actually converging, not on a fixed step
                    # count reaching an arbitrary threshold - see
                    # sonic-reverse-query-staged-unfreeze.md: a step-based
                    # trigger fired at step 2000 in this same mode while
                    # forward retrieval was still stuck at ~60%, well short
                    # of "reliable," and unfreezing there did not rescue
                    # decode (it cost a few points instead). Requires
                    # `--unfreeze-patience` consecutive eval checkpoints at
                    # or above threshold before firing, so a single lucky
                    # noisy checkpoint can't trigger it early.
                    if row["eval_retrieval_accuracy"] >= args.unfreeze_accuracy_threshold:
                        consecutive_high_evals += 1
                    else:
                        consecutive_high_evals = 0
                    streak_note = (
                        f" unfreeze_streak={consecutive_high_evals}"
                        f"/{args.unfreeze_patience}"
                    )
                    if consecutive_high_evals >= args.unfreeze_patience:
                        for parameter in model.parameters():
                            parameter.requires_grad_(True)
                        sonic_unfrozen = True
                        actual_unfreeze_step = step
                        freeze_note = ""
                        if args.freeze_retrieval_on_unfreeze:
                            for parameter in key_projection.parameters():
                                parameter.requires_grad_(False)
                            freeze_note = ", freezing key_projection"
                        weight_note = (
                            f", post_unfreeze_retrieval_loss_weight="
                            f"{args.post_unfreeze_retrieval_loss_weight}"
                            if args.post_unfreeze_retrieval_loss_weight is not None
                            else ""
                        )
                        print(
                            f"unfreezing Sonic at step {step} "
                            f"(retrieval_acc={row['eval_retrieval_accuracy']:.3f} "
                            f"held for {args.unfreeze_patience} evals, "
                            f"sonic_learning_rate={args.sonic_learning_rate})"
                            f"{freeze_note}{weight_note}",
                            flush=True,
                        )
                print(
                    f"eval step {step}/{args.steps} "
                    f"elapsed={time.monotonic() - start_time:.1f}s "
                    f"value_acc={row['eval_value_accuracy']:.3f} "
                    f"retrieval_acc={row['eval_retrieval_accuracy']:.3f} "
                    f"reverse_retrieval_acc="
                    f"{row['eval_reverse_retrieval_accuracy']:.3f}"
                    f"{streak_note}",
                    flush=True,
                )
    else:
        param_groups = [
            {
                "params": list(port.parameters()) + list(value_projection.parameters()),
                "lr": args.learning_rate,
            }
        ]
        if (
            args.unfreeze_sonic_at_step is not None
            or args.unfreeze_accuracy_threshold is not None
        ):
            param_groups.append(
                {"params": list(model.parameters()), "lr": args.sonic_learning_rate}
            )
        optimizer = torch.optim.Adam(param_groups)
        sonic_unfrozen = False
        actual_unfreeze_step = None
        consecutive_high_evals = 0
        corruption_generator = random.Random(args.data_seed + 1)
        for step in range(1, args.steps + 1):
            if step == 1 or step % heartbeat_every == 0:
                print(
                    f"step {step}/{args.steps} "
                    f"elapsed={time.monotonic() - start_time:.1f}s",
                    flush=True,
                )
            if (
                args.unfreeze_sonic_at_step is not None
                and not sonic_unfrozen
                and step >= args.unfreeze_sonic_at_step
            ):
                for parameter in model.parameters():
                    parameter.requires_grad_(True)
                sonic_unfrozen = True
                actual_unfreeze_step = step
                print(
                    f"unfreezing Sonic at step {step} "
                    f"(sonic_learning_rate={args.sonic_learning_rate})",
                    flush=True,
                )
            optimizer.zero_grad()
            step_episodes = []
            while len(step_episodes) < args.episodes_per_step:
                episode = _build_episode(
                    train_generator,
                    tokenizer,
                    level=args.level,
                    require_unique_reverse=require_unique_reverse,
                )
                if episode is not None:
                    step_episodes.append(episode)
            step_results = _run_episodes_decode_mode(
                model=model,
                episodes=step_episodes,
                device=device,
                port=port,
                value_projection=value_projection,
                pad_token_id=pad_token_id,
                sonic_requires_grad=sonic_unfrozen,
                corruption_rate=args.corruption_rate,
                corruption_rng=corruption_generator,
            )
            losses = [r.decode_loss for r in step_results]
            (sum(losses) / len(losses)).backward()
            optimizer.step()

            if step % args.eval_every == 0 or step == args.steps:
                with torch.no_grad():
                    eval_results = _run_episodes_decode_mode(
                        model=model,
                        episodes=eval_episodes,
                        device=device,
                        port=port,
                        value_projection=value_projection,
                        pad_token_id=pad_token_id,
                        sonic_requires_grad=False,
                        corruption_rate=args.corruption_rate,
                        corruption_rng=corruption_generator,
                    )
                corrupted_results = [r for r in eval_results if r.corrupted]
                clean_results = [r for r in eval_results if not r.corrupted]
                row = {
                    "step": step,
                    "eval_name_accuracy": sum(
                        r.name_correct for r in eval_results
                    )
                    / len(eval_results),
                    "eval_name_accuracy_clean": (
                        sum(r.name_correct for r in clean_results)
                        / len(clean_results)
                        if clean_results
                        else None
                    ),
                    "eval_name_accuracy_corrupted": (
                        sum(r.name_correct for r in corrupted_results)
                        / len(corrupted_results)
                        if corrupted_results
                        else None
                    ),
                }
                history.append(row)
                streak_note = ""
                if args.unfreeze_accuracy_threshold is not None and not sonic_unfrozen:
                    if row["eval_name_accuracy"] >= args.unfreeze_accuracy_threshold:
                        consecutive_high_evals += 1
                    else:
                        consecutive_high_evals = 0
                    streak_note = (
                        f" unfreeze_streak={consecutive_high_evals}"
                        f"/{args.unfreeze_patience}"
                    )
                    if consecutive_high_evals >= args.unfreeze_patience:
                        for parameter in model.parameters():
                            parameter.requires_grad_(True)
                        sonic_unfrozen = True
                        actual_unfreeze_step = step
                        print(
                            f"unfreezing Sonic at step {step} "
                            f"(name_acc={row['eval_name_accuracy']:.3f} "
                            f"held for {args.unfreeze_patience} evals, "
                            f"sonic_learning_rate={args.sonic_learning_rate})",
                            flush=True,
                        )
                corruption_note = (
                    f" name_acc_clean={row['eval_name_accuracy_clean']:.3f}"
                    f" name_acc_corrupted={row['eval_name_accuracy_corrupted']:.3f}"
                    if args.corruption_rate > 0
                    and row["eval_name_accuracy_clean"] is not None
                    and row["eval_name_accuracy_corrupted"] is not None
                    else ""
                )
                print(
                    f"eval step {step}/{args.steps} "
                    f"elapsed={time.monotonic() - start_time:.1f}s "
                    f"name_acc={row['eval_name_accuracy']:.3f}"
                    f"{corruption_note}"
                    f"{streak_note}",
                    flush=True,
                )

    output = {
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_update": checkpoint.get("update"),
        "checkpoint_tokens_seen": checkpoint.get("tokens_seen"),
        "model_config": model_config.to_dict(),
        "mode": args.mode,
        "steps": args.steps,
        "episodes_per_step": args.episodes_per_step,
        "level": args.level,
        "learning_rate": args.learning_rate,
        "gate_init_bias": args.gate_init_bias,
        "port_rank": args.port_rank,
        "retrieval_loss_weight": args.retrieval_loss_weight,
        "recency_weight": args.recency_weight,
        "recency_decay": args.recency_decay,
        "freeze_retrieval_on_unfreeze": args.freeze_retrieval_on_unfreeze,
        "corruption_rate": args.corruption_rate,
        "post_unfreeze_retrieval_loss_weight": args.post_unfreeze_retrieval_loss_weight,
        "temperature": args.temperature,
        "eval_episodes": args.eval_episodes,
        "unfreeze_sonic_at_step": args.unfreeze_sonic_at_step,
        "unfreeze_accuracy_threshold": args.unfreeze_accuracy_threshold,
        "unfreeze_patience": args.unfreeze_patience,
        "actual_unfreeze_step": actual_unfreeze_step,
        "sonic_learning_rate": (
            args.sonic_learning_rate
            if args.unfreeze_sonic_at_step is not None
            or args.unfreeze_accuracy_threshold is not None
            else None
        ),
        "history": history,
        "final": history[-1] if history else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-dir", type=Path, default=_DEFAULT_TOKENIZER_DIR)
    parser.add_argument(
        "--mode",
        choices=("retrieval", "decode"),
        required=True,
        help=(
            "'retrieval': train forward-only (as sonic_correction_memory_probe.py "
            "does), score zero-shot reverse-query retrieval at eval time - no "
            "loss ever sees a reverse query. 'decode': oracle-correct "
            "injection (bypasses real retrieval entirely), train only to "
            "decode the subject name from the reverse question."
        ),
    )
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
        "--unfreeze-sonic-at-step",
        type=int,
        default=None,
        help=(
            "Unfreeze Sonic's own parameters at this step, staged after "
            "the adapter (port/key/value projections) has had time to "
            "converge with Sonic frozen - the design validated in "
            "sonic-correction-staged-unfreeze.md and "
            "sonic-batch-lane-staged-unfreeze.md. None (default) keeps "
            "Sonic frozen for the whole run."
        ),
    )
    parser.add_argument("--sonic-learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--unfreeze-accuracy-threshold",
        type=float,
        default=None,
        help=(
            "Alternative to --unfreeze-sonic-at-step: unfreeze Sonic once "
            "the adapter's own eval metric (retrieval accuracy in "
            "--mode retrieval, name accuracy in --mode decode) has been "
            "at or above this threshold for --unfreeze-patience "
            "consecutive eval checkpoints, rather than at a fixed step - "
            "see sonic-reverse-query-staged-unfreeze.md: a fixed step "
            "count unfroze retrieval mode before its own retrieval "
            "accuracy had actually converged, and staging did not help "
            "decode there. Mutually exclusive with "
            "--unfreeze-sonic-at-step."
        ),
    )
    parser.add_argument("--unfreeze-patience", type=int, default=3)
    parser.add_argument(
        "--recency-weight",
        type=float,
        default=0.0,
        help=(
            "Write-order recency bonus for retrieval re-ranking, ported "
            "from sonic_correction_memory_probe.py "
            "(sonic-correction-key-value-split-and-recency-bonus.md: "
            "roughly 60pct to 83pct retrieval accuracy on the same "
            "plateau this probe's forward retrieval was stuck at). "
            "0.0 (default) "
            "reproduces the original plain top-1 read."
        ),
    )
    parser.add_argument("--recency-decay", type=float, default=0.7)
    parser.add_argument(
        "--freeze-retrieval-on-unfreeze",
        action="store_true",
        help=(
            "Only meaningful in --mode retrieval: at the same moment "
            "Sonic unfreezes (fixed step or accuracy-gated), also freeze "
            "key_projection so the retrieval decision stops shifting "
            "under Sonic's newly-moving weights - decode (port + "
            "value_projection + Sonic) stays trainable. Tests whether "
            "letting retrieval keep training simultaneously with Sonic "
            "was itself destabilizing decode, independent of retrieval "
            "having already reached a reliable accuracy band."
        ),
    )
    parser.add_argument(
        "--post-unfreeze-retrieval-loss-weight",
        type=float,
        default=None,
        help=(
            "Only meaningful in --mode retrieval with an unfreeze "
            "trigger set: use this retrieval_loss_weight instead of "
            "--retrieval-loss-weight once Sonic unfreezes (before that "
            "point, --retrieval-loss-weight always applies, since "
            "key_projection still needs the contrastive signal to "
            "converge in the first place). Tests whether the InfoNCE "
            "retrieval loss - still a function of Sonic's own hidden "
            "states, and still able to pull Sonic's weights toward a "
            "decode-irrelevant separability objective even with "
            "key_projection's own parameters frozen - is what's "
            "competing with decode_loss post-unfreeze "
            "(sonic-reverse-query-corruption-ablation.md ruled out wrong "
            "retrieved content as the cause; this tests the leading "
            "remaining candidate). None (default) keeps "
            "--retrieval-loss-weight unchanged after unfreezing, "
            "matching every earlier run this session."
        ),
    )
    parser.add_argument(
        "--corruption-rate",
        type=float,
        default=0.0,
        help=(
            "Only meaningful in --mode decode: with this probability, "
            "inject a random OTHER statement's value instead of the "
            "target's own - mimicking a real bank.read() grabbing the "
            "wrong (but plausible) statement, decode mode's oracle "
            "setup otherwise never has to contend with. The ablation "
            "sonic-reverse-query-retrieval-freeze-corrected.md's "
            "unresolved 'what breaks decode when Sonic unfreezes on "
            "real retrieval' question motivated: if unfreezing collapses "
            "decode here too once retrieval is imperfect, wrong content "
            "(not anything else about joint real-retrieval training) is "
            "the mechanism. 0.0 (default) reproduces the original "
            "always-correct oracle."
        ),
    )
    parser.add_argument("--device", default=None)
    return parser


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
    summary = {k: v for k, v in metrics.items() if k != "history"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
