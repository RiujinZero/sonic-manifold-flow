"""Phase 5: continual write lifecycle (design-v0.3.md SS9 Phase 5, exit
gate: "New memories are immediately usable; corrections replace obsolete
values; unrelated competence does not degrade materially.")

Simulates a non-IID stream: write a location fact for every entity in
random order, then - later in the same stream, also in random order -
write a *correction* (a new location, `Source("corrected")`,
`supersedes=` the original record) for half the entities, leaving the
other half untouched as the "unrelated competence" control group. All
writes go through the real `QwenRepresentationAdapter` and `BasinMemory`
(`continuous_memory.py`, Phase 5's lifecycle: trust-tier gating,
tiering, supersession).

Three checks, matching the exit gate's three clauses exactly:

1. Immediate usability: right after each original write, query that same
   fact's paraphrase and confirm it is found - before any correction or
   distraction has happened.
2. Corrections replace obsolete values: for corrected entities, does the
   new paraphrase retrieve the new fact, and does the *old* basin's
   payload list come back empty (correctly cleared, not lingering)? This
   also surfaces a real, honest limitation worth stating plainly: a
   correction creates a *new* basin (different content, different key) -
   querying the *old* phrasing does not get redirected to the new answer,
   because nothing here does subject-level indirection. "Replaces" means
   the old statement stops being returned, not that old queries find the
   new one.
3. Unrelated competence: after all originals and all corrections have
   been written (a much larger, grown bank than any single write), do the
   *uncorrected* entities' facts still retrieve correctly?
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

from manifoldflow.continuous_memory import BasinMemory, Source
from manifoldflow.experiments.qwen_hard_negative_probe import _generate_entity_names
from manifoldflow.experiments.qwen_representation_sweep import _LOCATIONS
from manifoldflow.qwen_adapter import (
    QwenRepresentationAdapter,
    QwenRepresentationConfig,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--entity-count", type=int, default=30)
    parser.add_argument("--dataset-seed", type=int, default=17)
    parser.add_argument("--stream-seed", type=int, default=29)
    parser.add_argument("--device", default=None)
    return parser


@dataclass(frozen=True)
class LifecycleDataset:
    entities: list[str]
    original_facts: list[str]
    original_paraphrases: list[str]
    corrected_facts: list[str]
    corrected_paraphrases: list[str]


def _generate_lifecycle_dataset(seed: int, entity_count: int) -> LifecycleDataset:
    generator = random.Random(seed)
    entities = _generate_entity_names(entity_count, seed)
    original_facts, original_paraphrases = [], []
    corrected_facts, corrected_paraphrases = [], []
    for entity in entities:
        location_a, location_b = generator.sample(_LOCATIONS, 2)
        original_facts.append(f"{entity} is currently in {location_a}.")
        original_paraphrases.append(f"Where is {entity}? In {location_a}.")
        corrected_facts.append(f"{entity} is currently in {location_b}.")
        corrected_paraphrases.append(f"Where is {entity}? In {location_b}.")
    return LifecycleDataset(
        entities=entities,
        original_facts=original_facts,
        original_paraphrases=original_paraphrases,
        corrected_facts=corrected_facts,
        corrected_paraphrases=corrected_paraphrases,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _generate_lifecycle_dataset(args.dataset_seed, args.entity_count)
    original_facts = dataset.original_facts
    original_paraphrases = dataset.original_paraphrases
    corrected_facts = dataset.corrected_facts
    corrected_paraphrases = dataset.corrected_paraphrases
    entity_count = len(dataset.entities)

    adapter = QwenRepresentationAdapter(
        QwenRepresentationConfig(model_name=args.model_name, device=args.device)
    )
    memory = BasinMemory(
        dimension=adapter.config.output_dim,
        novelty_threshold=0.0,  # exact-fact-per-basin, matching Phase 2/3's discipline
        promotion_threshold=2,
        # Expiration is already unit-tested (test_continuous_memory.py);
        # kept out of scope for this probe's own real-model evaluation.
        microbasin_max_age=10**9,
    )

    stream = random.Random(args.stream_seed)
    original_indices = list(range(entity_count))
    stream.shuffle(original_indices)
    corrected_indices = set(stream.sample(range(entity_count), entity_count // 2))

    # --- Phase A: write every original fact, checking immediate usability inline ---
    original_basin_by_entity: dict[int, int] = {}
    immediate_usability_hits = 0
    for index in original_indices:
        key = adapter.encode_event(original_facts[index])
        basin_id = memory.write(key, original_facts[index], Source("user"))
        original_basin_by_entity[index] = int(basin_id)

        paraphrase_key = adapter.encode_event(original_paraphrases[index])
        hits = memory.read(paraphrase_key, top_k=1)
        if hits and original_facts[index] in hits[0].payloads:
            immediate_usability_hits += 1
    immediate_usability_rate = immediate_usability_hits / entity_count

    # --- Phase B: write corrections for half the entities, later in the stream ---
    correction_order = list(corrected_indices)
    stream.shuffle(correction_order)
    for index in correction_order:
        key = adapter.encode_event(corrected_facts[index])
        basin_id = original_basin_by_entity[index]
        memory.write(
            key, corrected_facts[index], Source("corrected"), supersedes=(basin_id, 0)
        )

    # --- Check: corrections replace obsolete values ---
    correction_found = 0
    stale_cleared = 0
    for index in corrected_indices:
        new_key = adapter.encode_event(corrected_paraphrases[index])
        new_hits = memory.read(new_key, top_k=1)
        if new_hits and corrected_facts[index] in new_hits[0].payloads:
            correction_found += 1
        old_basin_id = original_basin_by_entity[index]
        if memory.payloads_for(old_basin_id) == []:
            stale_cleared += 1
    correction_replaces_rate = correction_found / len(corrected_indices)
    stale_cleared_rate = stale_cleared / len(corrected_indices)

    # --- Check: unrelated competence, after the bank has grown and changed ---
    uncorrected_indices = [i for i in range(entity_count) if i not in corrected_indices]
    unrelated_hits = 0
    for index in uncorrected_indices:
        paraphrase_key = adapter.encode_event(original_paraphrases[index])
        hits = memory.read(paraphrase_key, top_k=1)
        if hits and original_facts[index] in hits[0].payloads:
            unrelated_hits += 1
    unrelated_competence_rate = unrelated_hits / len(uncorrected_indices)

    tier_counts = {"recent": 0, "consolidated": 0}
    for basin_id in range(memory.basin_count):
        tier_counts[memory.tier(basin_id)] += 1

    result = {
        "model_name": args.model_name,
        "entity_count": entity_count,
        "corrected_entity_count": len(corrected_indices),
        "final_basin_count": memory.basin_count,
        "tier_counts": tier_counts,
        "retrieval_manifest": adapter.manifest(),
        "immediate_usability_rate": immediate_usability_rate,
        "correction_replaces_rate": correction_replaces_rate,
        "stale_record_cleared_rate": stale_cleared_rate,
        "unrelated_competence_rate": unrelated_competence_rate,
        "phase_5_exit_gate_passed": (
            immediate_usability_rate == 1.0
            and correction_replaces_rate == 1.0
            and stale_cleared_rate == 1.0
            and unrelated_competence_rate >= 0.9
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    # See imagenet_stream_extract.py's main() for the confirmed root cause
    # (a leaked multiprocessing semaphore inside heavy optional dependencies
    # stalls normal interpreter shutdown) and why both outcomes force-exit.
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
