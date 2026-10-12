"""Train the frozen-SigLIP-to-ManifoldFlow identity projection."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from manifoldflow.world_vision import (
    CrossModalKeyProjector,
    SiglipVisionLanguageEncoder,
)
from manifoldflow.world_vision_training import (
    VisionCropExample,
    frame_retrieval_accuracy,
    load_visible_crop_examples,
    mean_target_cosine,
    open_crop,
    temporal_reidentification_accuracy,
)


def _extract_features(
    encoder: SiglipVisionLanguageEncoder,
    examples: list[VisionCropExample],
    *,
    batch_size: int,
) -> dict[str, Any]:
    features: list[Tensor] = []
    for start in range(0, len(examples), batch_size):
        images = [open_crop(item) for item in examples[start : start + batch_size]]
        features.append(encoder.encode_images(images))
    return {
        "features": torch.cat(features),
        "target_keys": torch.stack([item.target_key for item in examples]),
        "entity_ids": [item.entity_id for item in examples],
        "episode_ids": [item.episode_id for item in examples],
        "frame_ids": [item.frame_id for item in examples],
        "descriptions": [item.description for item in examples],
    }


def _load_or_extract(
    path: Path,
    encoder: SiglipVisionLanguageEncoder,
    examples: list[VisionCropExample],
    *,
    batch_size: int,
) -> dict[str, Any]:
    if path.exists():
        result = torch.load(path, map_location="cpu", weights_only=False)
        if "episode_ids" not in result:
            result["episode_ids"] = [
                entity_id.split(":", maxsplit=1)[0]
                for entity_id in result["entity_ids"]
            ]
        return result
    result = _extract_features(encoder, examples, batch_size=batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return result


def _entity_rows(entity_ids: list[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for row, entity_id in enumerate(entity_ids):
        result.setdefault(entity_id, []).append(row)
    return result


def _text_prototypes(
    encoder: SiglipVisionLanguageEncoder, payload: dict[str, Any]
) -> tuple[Tensor, Tensor, list[str]]:
    descriptions = sorted(set(payload["descriptions"]))
    features = encoder.encode_texts(
        [f"a photo of a {description}" for description in descriptions]
    )
    targets = []
    for description in descriptions:
        rows = [
            index
            for index, value in enumerate(payload["descriptions"])
            if value == description
        ]
        target = payload["target_keys"].index_select(0, torch.tensor(rows)).mean(dim=0)
        targets.append(torch.nn.functional.normalize(target, dim=0))
    return features, torch.stack(targets), descriptions


@torch.inference_mode()
def _evaluate(
    projector: CrossModalKeyProjector,
    payload: dict[str, Any],
    text_payload: tuple[Tensor, Tensor, list[str]],
    device: torch.device,
) -> dict[str, float]:
    projected = projector(payload["features"].to(device)).cpu()
    targets = payload["target_keys"]
    text_features, text_targets, _ = text_payload
    projected_text = projector(text_features.to(device)).cpu()
    text_scores = projected_text @ text_targets.T
    return {
        "target_cosine": mean_target_cosine(projected, targets),
        "frame_retrieval_top1": frame_retrieval_accuracy(
            projected, targets, payload["frame_ids"]
        ),
        "temporal_reidentification_top1": temporal_reidentification_accuracy(
            projected, payload["episode_ids"], payload["entity_ids"]
        ),
        "text_target_cosine": mean_target_cosine(projected_text, text_targets),
        "text_prototype_top1": float(
            (text_scores.argmax(dim=1) == torch.arange(len(text_targets)))
            .float()
            .mean()
        ),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size <= 1:
        raise ValueError("batch_size must exceed one for contrastive training")
    randomizer = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    encoder = SiglipVisionLanguageEncoder(args.model, device=device, dtype=dtype)
    train_examples = load_visible_crop_examples(
        args.dataset,
        "train",
        frame_stride=args.frame_stride,
        max_examples=args.max_train_examples,
    )
    validation_examples = load_visible_crop_examples(
        args.dataset,
        "validation",
        frame_stride=args.frame_stride,
        max_examples=args.max_validation_examples,
    )
    cache = Path(args.cache_dir)
    train_payload = _load_or_extract(
        cache / "train.pt", encoder, train_examples, batch_size=args.feature_batch_size
    )
    validation_payload = _load_or_extract(
        cache / "validation.pt",
        encoder,
        validation_examples,
        batch_size=args.feature_batch_size,
    )
    train_text = _text_prototypes(encoder, train_payload)
    validation_text = _text_prototypes(encoder, validation_payload)
    projector = CrossModalKeyProjector(encoder.feature_dimension, 128).to(device)
    if args.init_checkpoint:
        projector.load_state_dict(
            torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
        )
    projector.train()
    optimizer = torch.optim.AdamW(
        projector.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    entity_rows = _entity_rows(train_payload["entity_ids"])
    entities = list(entity_rows)
    if len(entities) < args.batch_size:
        raise ValueError("not enough unique entities for one batch")
    features = train_payload["features"]
    targets = train_payload["target_keys"]
    text_features = train_text[0].to(device)
    text_targets = train_text[1].to(device)
    history: list[dict[str, float | int]] = []
    for step in range(1, args.steps + 1):
        chosen_entities = randomizer.sample(entities, args.batch_size)
        rows = torch.tensor(
            [randomizer.choice(entity_rows[item]) for item in chosen_entities],
            dtype=torch.long,
        )
        batch_features = features.index_select(0, rows).to(device)
        batch_targets = torch.nn.functional.normalize(
            targets.index_select(0, rows).to(device), dim=-1
        )
        projected = projector(batch_features)
        logits = projected @ batch_targets.T / args.temperature
        labels = torch.arange(args.batch_size, device=device)
        contrastive = torch.nn.functional.cross_entropy(logits, labels)
        cosine = 1.0 - (projected * batch_targets).sum(dim=-1).mean()
        projected_text = projector(text_features)
        text_logits = projected_text @ text_targets.T / args.temperature
        text_labels = torch.arange(len(text_targets), device=device)
        text_contrastive = torch.nn.functional.cross_entropy(text_logits, text_labels)
        text_cosine = 1.0 - (projected_text * text_targets).sum(dim=-1).mean()
        text_loss = text_contrastive + args.cosine_weight * text_cosine
        loss = contrastive + args.cosine_weight * cosine + args.text_weight * text_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "contrastive": float(contrastive.detach()),
                "cosine": float(1.0 - cosine.detach()),
                "text_loss": float(text_loss.detach()),
                "text_cosine": float(1.0 - text_cosine.detach()),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    projector.eval()
    result = {
        "model": args.model,
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "validation": _evaluate(projector, validation_payload, validation_text, device),
        "history": history,
        "args": vars(args),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), output)
    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="artifacts/world-sim")
    parser.add_argument("--output", default="checkpoints/world/siglip-key-projector.pt")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--cache-dir", default="artifacts/world-siglip-cache")
    parser.add_argument("--model", default="google/siglip2-base-patch16-224")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-train-examples", type=int, default=12000)
    parser.add_argument("--max-validation-examples", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--cosine-weight", type=float, default=0.5)
    parser.add_argument("--text-weight", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
