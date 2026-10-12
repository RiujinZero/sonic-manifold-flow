"""Evaluate an oracle Gemma world adapter on a held-out synthetic split."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from manifoldflow.experiments.world_train_oracle import _oracle_weights, _tokenize
from manifoldflow.gemma_world import (
    GemmaQueryProjector,
    enable_upper_layer_lora,
    load_gemma_world_model,
)
from manifoldflow.world_evaluation import (
    answer_candidates_for_question,
    world_fact_match,
)
from manifoldflow.world_synthetic_training import (
    OracleWorldBatch,
    collate_oracle_examples,
    load_oracle_examples,
)


def _token_metrics(
    world_model: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor,
    values: Tensor,
    keys: Tensor,
    memory_mask: Tensor,
    weights: Tensor,
) -> tuple[float, int, int]:
    prepared = world_model.prepare_inputs(
        input_ids,
        attention_mask,
        values,
        memory_keys=keys,
        labels=labels,
        memory_mask=memory_mask,
        retrieval_weights=weights,
    )
    output = world_model.model(
        inputs_embeds=prepared.inputs_embeds,
        attention_mask=prepared.attention_mask,
        return_dict=True,
    )
    targets = prepared.labels[:, 1:]
    logits = output.logits[:, :-1].float()
    valid = targets != -100
    losses = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), targets, ignore_index=-100, reduction="none"
    )
    correct = (logits.argmax(dim=-1) == targets) & valid
    return float(losses[valid].sum()), int(correct.sum()), int(valid.sum())


def _normalize_answer(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", value.lower()).split())


def _world_mismatched_indices(examples: list[Any], device: torch.device) -> Tensor:
    """Select negatives from a different episode with a different world fact."""
    if len(examples) <= 1:
        return torch.zeros(len(examples), dtype=torch.long, device=device)
    result: list[int] = []
    start_offset = max(1, len(examples) // 2)
    for row, example in enumerate(examples):
        candidates = [
            (row + start_offset + offset) % len(examples)
            for offset in range(len(examples) - 1)
        ]
        result.append(
            next(
                (
                    item
                    for item in candidates
                    if examples[item].episode_id != example.episode_id
                    and examples[item].answer != example.answer
                ),
                candidates[0],
            )
        )
    return torch.tensor(result, dtype=torch.long, device=device)


@torch.inference_mode()
def _ranked_answer_metrics(
    world_model: Any,
    tokenizer: Any,
    examples: list[Any],
    candidates: tuple[str, ...],
    *,
    device: torch.device,
    max_length: int,
    prompt_format: str,
    response_instruction: str | None,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    batch = collate_oracle_examples(examples)
    batch = OracleWorldBatch(
        batch.questions,
        batch.answers,
        batch.entity_ids,
        batch.appearance_keys.to(device),
        batch.state_values.to(device),
        batch.memory_mask.to(device),
        batch.evidence_mask.to(device),
    )
    negative_indices = _world_mismatched_indices(examples, device)
    hits = {"correct": 0, "shuffled": 0, "disabled": 0}
    category_hits: dict[str, int] = {}
    category_totals: dict[str, int] = {}
    samples: list[dict[str, str]] = []
    for row, (question, expected) in enumerate(
        zip(batch.questions, batch.answers, strict=True)
    ):
        question_candidates = answer_candidates_for_question(question, candidates)
        candidate_count = len(question_candidates)
        input_ids, attention_mask, labels = _tokenize(
            tokenizer,
            (question,) * candidate_count,
            question_candidates,
            device=device,
            max_length=max_length,
            prompt_format=prompt_format,
            response_instruction=response_instruction,
        )
        current_mask = batch.evidence_mask[row : row + 1].expand(candidate_count, -1)
        current_weights = current_mask.to(
            world_model.port.value_projection[0].weight.dtype
        )
        current_weights = current_weights / current_weights.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        negative_row = int(negative_indices[row])
        modes = {
            "correct": (
                batch.state_values[row : row + 1],
                batch.appearance_keys[row : row + 1],
                current_mask,
            ),
            "shuffled": (
                batch.state_values[negative_row : negative_row + 1],
                batch.appearance_keys[negative_row : negative_row + 1],
                current_mask,
            ),
            "disabled": (
                batch.state_values[row : row + 1],
                batch.appearance_keys[row : row + 1],
                torch.zeros_like(current_mask),
            ),
        }
        predictions: dict[str, str] = {}
        for name, (values, keys, memory_mask) in modes.items():
            prepared = world_model.prepare_inputs(
                input_ids,
                attention_mask,
                values.expand(candidate_count, -1, -1),
                memory_keys=keys.expand(candidate_count, -1, -1),
                labels=labels,
                memory_mask=memory_mask,
                retrieval_weights=current_weights,
            )
            output = world_model.model(
                inputs_embeds=prepared.inputs_embeds,
                attention_mask=prepared.attention_mask,
                return_dict=True,
            )
            targets = prepared.labels[:, 1:]
            losses = torch.nn.functional.cross_entropy(
                output.logits[:, :-1].float().transpose(1, 2),
                targets,
                ignore_index=-100,
                reduction="none",
            )
            valid = targets != -100
            candidate_losses = losses.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            prediction = question_candidates[int(candidate_losses.argmin())]
            predictions[name] = prediction
            hits[name] += int(prediction == expected)
            if name == "correct":
                category = str(examples[row].category)
                category_totals[category] = category_totals.get(category, 0) + 1
                category_hits[category] = category_hits.get(category, 0) + int(
                    prediction == expected
                )
        if len(samples) < 12:
            samples.append({"question": question, "expected": expected, **predictions})
    denominator = max(len(examples), 1)
    metrics = {name: value / denominator for name, value in hits.items()}
    metrics.update(
        {
            f"correct_category_{category}": category_hits.get(category, 0) / total
            for category, total in category_totals.items()
        }
    )
    return metrics, samples


@torch.inference_mode()
def _generated_metrics(
    world_model: Any,
    tokenizer: Any,
    examples: list[Any],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    prompt_format: str,
    response_instruction: str | None,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    hits = {"correct": 0, "shuffled": 0, "disabled": 0}
    fact_hits = {"correct": 0, "shuffled": 0, "disabled": 0}
    samples: list[dict[str, str]] = []
    for start in range(0, len(examples), batch_size):
        batch = collate_oracle_examples(examples[start : start + batch_size])
        batch = OracleWorldBatch(
            batch.questions,
            batch.answers,
            batch.entity_ids,
            batch.appearance_keys.to(device),
            batch.state_values.to(device),
            batch.memory_mask.to(device),
            batch.evidence_mask.to(device),
        )
        input_ids, attention_mask, _ = _tokenize(
            tokenizer,
            batch.questions,
            None,
            device=device,
            max_length=max_length,
            prompt_format=prompt_format,
            response_instruction=response_instruction,
        )
        weights = _oracle_weights(
            batch, world_model.port.value_projection[0].weight.dtype
        )
        negative_indices = _world_mismatched_indices(
            examples[start : start + batch_size], device
        )
        modes = {
            "correct": (
                batch.state_values,
                batch.appearance_keys,
                batch.evidence_mask,
            ),
            "shuffled": (
                batch.state_values.index_select(0, negative_indices),
                batch.appearance_keys.index_select(0, negative_indices),
                batch.evidence_mask,
            ),
            "disabled": (
                batch.state_values,
                batch.appearance_keys,
                torch.zeros_like(batch.evidence_mask),
            ),
        }
        outputs: dict[str, list[str]] = {}
        for name, (values, keys, memory_mask) in modes.items():
            generated = world_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                memory_values=values,
                memory_keys=keys,
                memory_mask=memory_mask,
                retrieval_weights=weights,
                generation_kwargs={
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "pad_token_id": tokenizer.pad_token_id,
                },
            )
            outputs[name] = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for expected, prediction in zip(batch.answers, outputs[name], strict=True):
                normalized_expected = _normalize_answer(expected)
                normalized_prediction = _normalize_answer(prediction)
                hits[name] += int(normalized_expected in normalized_prediction)
                fact_hits[name] += int(world_fact_match(expected, prediction))
        if len(samples) < 12:
            for row, (question, expected) in enumerate(
                zip(batch.questions, batch.answers, strict=True)
            ):
                if len(samples) >= 12:
                    break
                samples.append(
                    {
                        "question": question,
                        "expected": expected,
                        **{
                            name: predictions[row]
                            for name, predictions in outputs.items()
                        },
                    }
                )
    denominator = max(len(examples), 1)
    metrics = {name: value / denominator for name, value in hits.items()}
    metrics.update(
        {f"{name}_fact": value / denominator for name, value in fact_hits.items()}
    )
    return metrics, samples


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_name = str(checkpoint["model"])
    prompt_format = args.prompt_format or str(
        checkpoint.get("args", {}).get("prompt_format", "plain")
    )
    response_instruction = checkpoint.get("args", {}).get("response_instruction")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    world_model = load_gemma_world_model(model_name, device=args.device)
    if checkpoint.get("lora") is not None:
        checkpoint_args = checkpoint.get("args", {})
        enable_upper_layer_lora(
            world_model,
            rank=int(checkpoint_args.get("lora_rank", 8)),
            alpha=int(checkpoint_args.get("lora_alpha", 16)),
        )
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(world_model.model, checkpoint["lora"])
        world_model.model.eval()
    world_model.port.load_state_dict(checkpoint["port"])
    world_model.port.eval()
    query_projector = GemmaQueryProjector(
        hidden_dimension=world_model.port.hidden_dimension,
        key_dimension=128,
    ).to(device)
    query_projector.load_state_dict(checkpoint["query_projector"])
    query_projector.eval()
    examples = load_oracle_examples(args.dataset, args.split)[: args.max_examples]
    retrieval_rows = 0
    top1_hits = 0
    correct_totals = [0.0, 0, 0]
    shuffled_totals = [0.0, 0, 0]
    disabled_totals = [0.0, 0, 0]
    for start in range(0, len(examples), args.batch_size):
        batch = collate_oracle_examples(examples[start : start + args.batch_size])
        batch = OracleWorldBatch(
            batch.questions,
            batch.answers,
            batch.entity_ids,
            batch.appearance_keys.to(device),
            batch.state_values.to(device),
            batch.memory_mask.to(device),
            batch.evidence_mask.to(device),
        )
        input_ids, attention_mask, labels = _tokenize(
            tokenizer,
            batch.questions,
            batch.answers,
            device=device,
            max_length=args.max_length,
            prompt_format=prompt_format,
            response_instruction=response_instruction,
        )
        question_ids, question_mask, _ = _tokenize(
            tokenizer,
            batch.questions,
            None,
            device=device,
            max_length=args.max_length,
            prompt_format=prompt_format,
            response_instruction=None,
        )
        query_keys = query_projector(
            world_model.encode_query(question_ids, question_mask)
        )
        scores = torch.einsum(
            "bd,bnd->bn",
            torch.nn.functional.normalize(query_keys, dim=-1),
            torch.nn.functional.normalize(batch.appearance_keys, dim=-1),
        ).masked_fill(~batch.memory_mask, float("-inf"))
        has_evidence = batch.evidence_mask.any(dim=1)
        winners = scores.argmax(dim=1)
        rows = torch.arange(scores.shape[0], device=device)
        top1_hits += int((batch.evidence_mask[rows, winners] & has_evidence).sum())
        retrieval_rows += int(has_evidence.sum())
        weights = _oracle_weights(
            batch, world_model.port.value_projection[0].weight.dtype
        )
        negative_indices = _world_mismatched_indices(
            examples[start : start + args.batch_size], device
        )
        modes = (
            (
                correct_totals,
                batch.state_values,
                batch.appearance_keys,
                batch.evidence_mask,
            ),
            (
                shuffled_totals,
                batch.state_values.index_select(0, negative_indices),
                batch.appearance_keys.index_select(0, negative_indices),
                batch.evidence_mask,
            ),
            (
                disabled_totals,
                batch.state_values,
                batch.appearance_keys,
                torch.zeros_like(batch.evidence_mask),
            ),
        )
        for totals, values, keys, memory_mask in modes:
            values_to_add = _token_metrics(
                world_model,
                input_ids,
                attention_mask,
                labels,
                values,
                keys,
                memory_mask,
                weights,
            )
            for index, value in enumerate(values_to_add):
                totals[index] += value
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "examples": len(examples),
        "retrieval_top1": top1_hits / max(retrieval_rows, 1),
    }
    for name, totals in (
        ("correct", correct_totals),
        ("shuffled", shuffled_totals),
        ("disabled", disabled_totals),
    ):
        result[f"{name}_loss"] = totals[0] / max(totals[2], 1)
        result[f"{name}_token_accuracy"] = totals[1] / max(totals[2], 1)
    if args.generate_examples > 0:
        generation_count = min(args.generate_examples, len(examples))
        if generation_count <= 1:
            generated_examples = examples[:generation_count]
        else:
            generated_examples = [
                examples[round(index * (len(examples) - 1) / (generation_count - 1))]
                for index in range(generation_count)
            ]
        generated, samples = _generated_metrics(
            world_model,
            tokenizer,
            generated_examples,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.generation_max_new_tokens,
            prompt_format=prompt_format,
            response_instruction=response_instruction,
        )
        for name, accuracy in generated.items():
            if name.endswith("_fact"):
                result[f"{name.removesuffix('_fact')}_generated_fact_accuracy"] = (
                    accuracy
                )
            else:
                result[f"{name}_generated_accuracy"] = accuracy
        result["generated_examples"] = len(generated_examples)
        result["generation_samples"] = samples
    if args.rank_examples > 0:
        rank_count = min(args.rank_examples, len(examples))
        ranked_examples = [
            examples[round(index * (len(examples) - 1) / max(rank_count - 1, 1))]
            for index in range(rank_count)
        ]
        training_questions = Path(args.dataset) / "train" / "questions.jsonl"
        candidates = tuple(
            sorted(
                {
                    str(json.loads(line)["answer"])
                    for line in training_questions.read_text(
                        encoding="utf-8"
                    ).splitlines()
                }
            )
        )
        ranked, ranked_samples = _ranked_answer_metrics(
            world_model,
            tokenizer,
            ranked_examples,
            candidates,
            device=device,
            max_length=args.max_length,
            prompt_format=prompt_format,
            response_instruction=response_instruction,
        )
        for name, accuracy in ranked.items():
            if name.startswith("correct_category_"):
                category = name.removeprefix("correct_category_")
                result[f"ranked_{category}_accuracy"] = accuracy
            else:
                result[f"{name}_ranked_accuracy"] = accuracy
        result["ranked_examples"] = len(ranked_examples)
        result["ranked_candidates"] = len(candidates)
        result["ranked_samples"] = ranked_samples
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/world-sim")
    parser.add_argument("--checkpoint", default="checkpoints/world/oracle-port.pt")
    parser.add_argument("--output", default="checkpoints/world/oracle-validation.json")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--prompt-format", choices=("chat", "plain"))
    parser.add_argument("--generate-examples", type=int, default=0)
    parser.add_argument("--generation-max-new-tokens", type=int, default=24)
    parser.add_argument("--rank-examples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
