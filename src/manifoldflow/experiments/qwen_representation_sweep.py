"""Phase 1 representation selection sweep (design-v0.3.md SS4.2, SS9 Phase 1).

Compares `{final, ~75% depth, ~50% depth}` x `{last_token, mean}` pooling -
six configurations - on stateless paraphrase retrieval only, never on
persistence scores, per SS4.2's explicit constraint against confounding
representation quality with the memory mechanism under test. For each
configuration, a bank of nonce-entity fact statements is encoded once, a
held-out paraphrase of each fact is encoded, and retrieval asks: does the
paraphrase's nearest bank neighbor recover its true source fact, among many
same-template distractor facts?

All three candidate layers are captured from a *single* forward pass per
text via simultaneous hooks (`output_hidden_states=True` would materialize
every layer for no benefit here) - both pooling rules are then computed
post-hoc from the same captured tensors, so this needs exactly one forward
pass per text, not one per configuration.

Reports each configuration against two controls required by the Phase 1
exit gate: a random-vector control (same retrieval protocol, no text
signal at all) and a lexical control (`lexical_baseline.py`, raw token
overlap, no model at all).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from manifoldflow.lexical_baseline import lexical_retrieval_topk_accuracy
from manifoldflow.representation import (
    FixedSeededProjection,
    layer_index_for_fraction,
    pool_hidden_states,
    retrieval_topk_accuracy,
)

_LAYER_FRACTIONS = (1.0, 0.75, 0.5)
_POOLINGS = ("last_token", "mean")

_ENTITIES = [
    "Neral", "Tovan", "Mira", "Selka", "Bren", "Dorna", "Ilvet", "Quorin",
    "Ashra", "Talvin", "Perik", "Wenna", "Corvis", "Idrel", "Sannu", "Vekka",
    "Ornic", "Halda", "Tresk", "Amory", "Bethic", "Corlan", "Druse", "Efrin",
]
_OBJECTS = [
    "copper glider", "striped lantern", "brass compass", "wooden crate",
    "silver whistle", "folded map", "iron key", "glass vial",
]
_LOCATIONS = [f"Vault {number}" for number in range(1, 9)] + [
    f"Box {number}" for number in range(1, 9)
]


def _generate_dataset(seed: int, pairs_per_family: int) -> tuple[list[str], list[str]]:
    """Return `(facts, paraphrases)`, index-aligned, across three template
    families (ownership, placement, relocation) with nonce entities so
    Qwen cannot answer from pretrained knowledge (SS8.2)."""
    generator = random.Random(seed)
    entities = generator.sample(_ENTITIES, pairs_per_family * 3)
    facts: list[str] = []
    paraphrases: list[str] = []

    for entity in entities[:pairs_per_family]:
        obj = generator.choice(_OBJECTS)
        facts.append(f"{entity} owns the {obj}.")
        paraphrases.append(f"The {obj} belongs to {entity}.")

    for entity in entities[pairs_per_family : 2 * pairs_per_family]:
        obj = generator.choice(_OBJECTS)
        loc = generator.choice(_LOCATIONS)
        facts.append(f"{entity} placed the {obj} in {loc}.")
        paraphrases.append(f"In {loc} is where {entity} put the {obj}.")

    for entity in entities[2 * pairs_per_family :]:
        loc_a, loc_b = generator.sample(_LOCATIONS, 2)
        facts.append(f"{entity} moved from {loc_a} to {loc_b}.")
        paraphrases.append(f"{entity} relocated, leaving {loc_a} for {loc_b}.")

    return facts, paraphrases


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--pairs-per-family", type=int, default=8)
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--dataset-seed", type=int, default=17)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--material-margin",
        type=float,
        default=0.10,
        help="minimum top-1 accuracy margin over each control to count as "
        "'materially exceeds' for the Phase 1 exit gate",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.pairs_per_family <= 0:
        raise ValueError("pairs_per_family must be positive")
    if args.pairs_per_family * 3 > len(_ENTITIES):
        raise ValueError(
            f"pairs_per_family={args.pairs_per_family} needs "
            f"{args.pairs_per_family * 3} distinct entities across three "
            f"families but only {len(_ENTITIES)} are defined"
        )
    try:
        import transformers
    except ImportError as error:
        raise RuntimeError(
            "The Qwen representation sweep requires the qwen extra: "
            "python -m pip install -e '.[qwen]'"
        ) from error

    facts, paraphrases = _generate_dataset(args.dataset_seed, args.pairs_per_family)
    bank_size = len(facts)
    all_texts = facts + paraphrases

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    layer_indices = {
        fraction: layer_index_for_fraction(num_layers, fraction)
        for fraction in _LAYER_FRACTIONS
    }
    distinct_indices = sorted(set(layer_indices.values()))

    captured: dict[int, Tensor] = {}

    def _make_hook(index: int):
        def _hook(module: Any, inputs: Any, output: Tensor) -> None:
            captured[index] = output

        return _hook

    handles = [
        model.model.layers[index].register_forward_hook(_make_hook(index))
        for index in distinct_indices
    ]

    projection = FixedSeededProjection(
        hidden_size, args.output_dim, seed=args.projection_seed
    ).to(device)

    pooled_by_config: dict[tuple[int, str], list[Tensor]] = {
        (index, pooling): [] for index in distinct_indices for pooling in _POOLINGS
    }
    forward_latencies_ms: list[float] = []

    with torch.no_grad():
        for text in all_texts:
            messages = [{"role": "user", "content": text}]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            ).to(device)

            started = time.perf_counter()
            model.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
            )
            forward_latencies_ms.append((time.perf_counter() - started) * 1000)

            for index in distinct_indices:
                hidden = captured[index].float()
                for pooling in _POOLINGS:
                    pooled = pool_hidden_states(
                        hidden, encoded["attention_mask"], pooling=pooling
                    )
                    pooled_by_config[(index, pooling)].append(pooled.squeeze(0).cpu())

    for handle in handles:
        handle.remove()

    true_index = torch.arange(bank_size)
    top_k = min(3, bank_size)

    configs: dict[str, Any] = {}
    for fraction in _LAYER_FRACTIONS:
        index = layer_indices[fraction]
        for pooling in _POOLINGS:
            stacked = torch.stack(pooled_by_config[(index, pooling)]).to(device)
            keys = projection(stacked).cpu()
            bank_keys, query_keys = keys[:bank_size], keys[bank_size:]
            configs[f"layer_fraction={fraction:g}_pooling={pooling}"] = {
                "layer_fraction": fraction,
                "layer_index": index,
                "pooling": pooling,
                "top1_retrieval_accuracy": retrieval_topk_accuracy(
                    query_keys, bank_keys, true_index, k=1
                ),
                "top3_retrieval_accuracy": retrieval_topk_accuracy(
                    query_keys, bank_keys, true_index, k=top_k
                ),
            }

    random_generator = torch.Generator().manual_seed(args.dataset_seed)
    random_keys = F.normalize(
        torch.randn(2 * bank_size, args.output_dim, generator=random_generator), dim=-1
    )
    random_control = {
        "top1_retrieval_accuracy": retrieval_topk_accuracy(
            random_keys[bank_size:], random_keys[:bank_size], true_index, k=1
        ),
        "top3_retrieval_accuracy": retrieval_topk_accuracy(
            random_keys[bank_size:], random_keys[:bank_size], true_index, k=top_k
        ),
    }
    lexical_control = {
        "top1_retrieval_accuracy": lexical_retrieval_topk_accuracy(
            paraphrases, facts, list(range(bank_size)), k=1
        ),
        "top3_retrieval_accuracy": lexical_retrieval_topk_accuracy(
            paraphrases, facts, list(range(bank_size)), k=top_k
        ),
    }

    best_key = max(configs, key=lambda name: configs[name]["top1_retrieval_accuracy"])
    best = configs[best_key]
    materially_exceeds = (
        best["top1_retrieval_accuracy"]
        >= random_control["top1_retrieval_accuracy"] + args.material_margin
        and best["top1_retrieval_accuracy"]
        >= lexical_control["top1_retrieval_accuracy"] + args.material_margin
    )

    ordered = sorted(forward_latencies_ms)

    def _percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, int(fraction * len(ordered)))
        return ordered[index]

    result = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": device,
        "model_name": args.model_name,
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "bank_size": bank_size,
        "material_margin": args.material_margin,
        "configs": configs,
        "random_control": random_control,
        "lexical_control": lexical_control,
        "selected_config": best_key,
        "selected_config_details": best,
        "phase_1_exit_gate_passed": materially_exceeds,
        "forward_pass_latency_ms": {
            "mean": sum(forward_latencies_ms) / len(forward_latencies_ms),
            "p50": _percentile(0.50),
            "p95": _percentile(0.95),
        },
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
