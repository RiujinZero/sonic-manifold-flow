"""Phase 1 representation selection sweep for Sonic (design-v0.3.md
SS4.2, SS9 Phase 1), mirroring `qwen_representation_sweep.py` exactly and
reusing its dataset generator so the two models' results are directly
comparable, cell for cell.

Sonic has no discrete attention layers to hook the same way a
transformer does, but `SonicSSMLayer.forward(hidden, state) -> (hidden,
state)` still returns a hookable per-layer hidden tensor, so the same
targeted-hook, single-forward-pass-captures-every-candidate-layer pattern
applies unchanged. All three candidate layers use the *raw* (pre-
`final_norm`) hook capture uniformly, including the "final" depth
fraction - `forward_hidden`'s own official output applies `final_norm`
only to the last layer, which would make the "final" cell methodologically
inconsistent with the other two if it alone got extra normalization the
sweep is not otherwise testing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from manifoldflow.experiments.qwen_representation_sweep import _generate_dataset
from manifoldflow.lexical_baseline import lexical_retrieval_topk_accuracy
from manifoldflow.representation import (
    FixedSeededProjection,
    layer_index_for_fraction,
    pool_hidden_states,
    retrieval_topk_accuracy,
)
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel

_LAYER_FRACTIONS = (1.0, 0.75, 0.5)
_POOLINGS = ("last_token", "mean")
_DEFAULT_CHECKPOINT = Path("checkpoints/sonic_ssm/sonic_ssm_128k_steps.pt")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--pairs-per-family", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=128)
    parser.add_argument("--projection-seed", type=int, default=0)
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

    facts, paraphrases = _generate_dataset(args.dataset_seed, args.pairs_per_family)
    bank_size = len(facts)
    all_texts = facts + paraphrases

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model_config = SonicSSMConfig(**state["model_config"])
    model = SonicSSMLanguageModel(model_config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    transformers = _require_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "resources/tokenizers/gtr_arw_english_50k"
    )

    num_layers = model_config.n_layers
    hidden_size = model_config.d_model
    layer_indices = {
        fraction: layer_index_for_fraction(num_layers, fraction)
        for fraction in _LAYER_FRACTIONS
    }
    distinct_indices = sorted(set(layer_indices.values()))

    captured: dict[int, Tensor] = {}

    def _make_hook(index: int):
        def _hook(module: Any, inputs: Any, output: Any) -> None:
            captured[index] = output[0]

        return _hook

    handles = [
        model.layers[index].register_forward_hook(_make_hook(index))
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
            ids = tokenizer.encode(text, add_special_tokens=True)
            input_ids = torch.tensor([ids], device=device)
            attention_mask = torch.ones_like(input_ids)

            started = time.perf_counter()
            model.forward_hidden(input_ids)
            forward_latencies_ms.append((time.perf_counter() - started) * 1000)

            for index in distinct_indices:
                hidden = captured[index].float()
                for pooling in _POOLINGS:
                    pooled = pool_hidden_states(hidden, attention_mask, pooling=pooling)
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
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_update": state.get("update"),
        "num_layers": num_layers,
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


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "The Sonic representation sweep requires the qwen extra "
            "(shared tokenizer dependency): python -m pip install -e '.[qwen]'"
        ) from error
    return transformers


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
