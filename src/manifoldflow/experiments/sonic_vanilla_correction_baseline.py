"""Eleventh probe, and a different baseline than every `no_memory`
condition run so far: how accurate is **vanilla Sonic** - no
`SonicMemoryPort`, no `BasinMemory`, no separate write/query passes at
all - when the whole correction chain and its query sit in *one*
continuous sequence, letting Sonic's own recurrent state carry the
answer forward the way an ordinary language model naturally would?

Every earlier probe's `no_memory` condition answers a query from a
*fresh* forward pass with no access to the statements at all - a
structural floor proving the task cannot be solved without a memory
channel, not a measure of Sonic's own native capability. This probe
measures that capability directly: statements, question, and answer
tokenized and concatenated into one sequence, one `forward_hidden` call,
Sonic's own in-context processing doing all the work. No `BasinMemory`
read/write, no port, no training - this is pure inference, no optimizer
needed, since nothing here is trainable.

Motivation, stated directly by the user: retrieval accuracy climbing to
83% surprised them, since vanilla Sonic isn't reportedly very accurate
even at retrieving *words* on its own - if that reputation holds up
here, ManifoldFlow's retrieval number looks considerably more
significant by contrast, and if it doesn't, that's equally worth
knowing honestly. This also gives the numeric-value question
(`sonic-correction-value-accuracy-addendum.md`) an architecture-agnostic
data point: if vanilla in-context Sonic *also* struggles specifically
with digits (not just topic/entity words) while succeeding at everything
else, that points at something about Sonic's own numeric handling in
general, not specifically the memory injection mechanism; if vanilla
Sonic handles digits fine in-context, that isolates the problem to the
injection path.

Reuses `sonic_correction_memory_probe.py`'s exact episode content
(same `distractor_reasoning.correction` generator, same statements,
same value_span token tracking) so results are directly comparable -
only how the statements reach the answer differs.
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
    statement_ids: list[list[int]]
    question_ids: list[int]
    answer_ids: list[int]
    value_span: tuple[int, int]


def _build_episode(
    generator: random.Random, tokenizer: Any, *, level: int
) -> Episode | None:
    from distractor_reasoning.correction import (
        HAVING_RELATIONS_PLURAL,
        generate_correction_example,
        render_statement,
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
    statement_ids = [
        tokenizer.encode(
            render_statement(statement, generator, nouns, relation),
            add_special_tokens=(index == 0),
        )
        for index, statement in enumerate(example.statements)
    ]

    relation_base = HAVING_RELATIONS_PLURAL[relation]
    question_text = (
        f"\nQ: How many {nouns[example.query.target].plural} "
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

    return Episode(
        statement_ids=statement_ids,
        question_ids=tokenizer.encode(question_text, add_special_tokens=False),
        answer_ids=prefix_ids + quantity_ids + suffix_ids,
        value_span=(len(prefix_ids), len(prefix_ids) + len(quantity_ids)),
    )


@torch.no_grad()
def _score_episode(
    *, model: SonicSSMLanguageModel, episode: Episode, device: str
) -> tuple[bool, bool]:
    """Returns (answer_correct, value_correct). One continuous sequence:
    all statements, then the question, then the answer - Sonic's own
    recurrent state is the only thing carrying information from the
    statements to the answer, exactly as it would for any ordinary
    prompt."""
    all_statement_ids = [tok for ids in episode.statement_ids for tok in ids]
    full_ids = all_statement_ids + episode.question_ids + episode.answer_ids
    ids_tensor = torch.tensor([full_ids], device=device)
    hidden, _ = model.forward_hidden(ids_tensor)
    logits = model.logits_from_hidden(hidden)

    answer_len = len(episode.answer_ids)
    answer_start_in_full = len(full_ids) - answer_len
    # Position (answer_start_in_full - 1) is the last question token's
    # hidden state, predicting the first answer token - included, since
    # excluding it would (as in every earlier probe) skip scoring the
    # first answer token entirely.
    predicted_logits = logits[0, answer_start_in_full - 1 : -1, :]
    predicted_targets = ids_tensor[0, answer_start_in_full:]
    answer_correct = bool(
        (predicted_logits.argmax(dim=-1) == predicted_targets).all().item()
    )
    value_start, value_end = episode.value_span
    value_correct = bool(
        (
            predicted_logits[value_start:value_end].argmax(dim=-1)
            == predicted_targets[value_start:value_end]
        )
        .all()
        .item()
    )
    return answer_correct, value_correct


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-dir", type=Path, default=_DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--eval-seed", type=int, default=31)
    parser.add_argument("--level", type=int, default=5)
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

    generator = random.Random(args.eval_seed)
    answer_correct_count = 0
    value_correct_count = 0
    scored = 0
    while scored < args.num_episodes:
        episode = _build_episode(generator, tokenizer, level=args.level)
        if episode is None:
            continue
        answer_correct, value_correct = _score_episode(
            model=model, episode=episode, device=device
        )
        answer_correct_count += answer_correct
        value_correct_count += value_correct
        scored += 1

    output = {
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_update": checkpoint.get("update"),
        "checkpoint_tokens_seen": checkpoint.get("tokens_seen"),
        "model_config": model_config.to_dict(),
        "level": args.level,
        "num_episodes": args.num_episodes,
        "vanilla_answer_accuracy": answer_correct_count / scored,
        "vanilla_value_accuracy": value_correct_count / scored,
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
    print(json.dumps(metrics, indent=2, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
