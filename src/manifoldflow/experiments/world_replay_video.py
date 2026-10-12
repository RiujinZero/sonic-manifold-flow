"""Replay prerecorded footage through the local visual world-model pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import torch

from manifoldflow.world_answer import EvidenceAnswerer, GemmaLatentAnswerer
from manifoldflow.world_dataset import CLASSES
from manifoldflow.world_events import TabletopEventEngine
from manifoldflow.world_live import VisionFrameProcessor
from manifoldflow.world_memory import WorldMemory
from manifoldflow.world_video import (
    AnnotatedReplayProcessor,
    annotation_template,
    evaluate_replay,
    iter_video_frames,
    load_annotations,
    replay_deterministic,
    replay_realtime,
    write_replay_artifacts,
)
from manifoldflow.world_vision import (
    CrossModalKeyProjector,
    MultiClassByteTracker,
    SiglipVisionLanguageEncoder,
    YoloXDetector,
)


def _class_threshold(value: str) -> tuple[str, float]:
    class_name, separator, raw_threshold = value.partition("=")
    if not separator or class_name not in CLASSES:
        raise argparse.ArgumentTypeError(
            f"expected CLASS=VALUE where CLASS is one of {', '.join(CLASSES)}"
        )
    try:
        threshold = float(raw_threshold)
    except ValueError as error:
        raise argparse.ArgumentTypeError("class threshold must be numeric") from error
    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("class threshold must be between zero and one")
    return class_name, threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("detector_checkpoint")
    parser.add_argument("projector_checkpoint")
    parser.add_argument("video")
    parser.add_argument("--annotations")
    parser.add_argument("--gemma-checkpoint")
    parser.add_argument("--gemma-model")
    parser.add_argument("--query-checkpoint")
    parser.add_argument("--free-generation", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--mode", choices=("deterministic", "realtime"), default="deterministic"
    )
    parser.add_argument("--send-fps", type=float, default=5.0)
    parser.add_argument("--playback-rate", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--annotation-tolerance", type=float, default=0.35)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--detector-confidence", type=float, default=0.3)
    parser.add_argument("--track-confidence", type=float, default=0.35)
    parser.add_argument(
        "--class-detector-confidence",
        action="append",
        type=_class_threshold,
        default=[],
        metavar="CLASS=VALUE",
    )
    parser.add_argument(
        "--class-track-confidence",
        action="append",
        type=_class_threshold,
        default=[],
        metavar="CLASS=VALUE",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--session-id", default="video-evaluation")
    return parser


def _build_processor(
    args: argparse.Namespace,
    cup_labels: tuple[str, ...],
) -> tuple[
    VisionFrameProcessor,
    WorldMemory,
    SiglipVisionLanguageEncoder,
    CrossModalKeyProjector,
    torch.dtype,
]:
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    encoder = SiglipVisionLanguageEncoder(device=args.device, dtype=dtype)
    projector = CrossModalKeyProjector(encoder.feature_dimension, 128)
    projector.load_state_dict(
        torch.load(args.projector_checkpoint, map_location="cpu", weights_only=True)
    )
    projector.eval().to(args.device)
    memory = WorldMemory()
    return (
        VisionFrameProcessor(
            YoloXDetector(
                args.detector_checkpoint,
                CLASSES,
                confidence=args.detector_confidence,
                class_confidences=dict(args.class_detector_confidence),
                device=args.device,
            ),
            encoder,
            projector,
            memory,
            TabletopEventEngine(memory),
            lambda: MultiClassByteTracker(
                CLASSES,
                track_threshold=args.track_confidence,
                class_track_thresholds=dict(args.class_track_confidence),
            ),
            cup_labels=cup_labels,
        ),
        memory,
        encoder,
        projector,
        dtype,
    )


def _build_gemma_answerer(
    args: argparse.Namespace,
    memory: WorldMemory,
    encoder: SiglipVisionLanguageEncoder,
    projector: CrossModalKeyProjector,
    dtype: torch.dtype,
) -> GemmaLatentAnswerer:
    from transformers import AutoTokenizer

    from manifoldflow.gemma_world import (
        GemmaQueryProjector,
        enable_upper_layer_lora,
        load_gemma_world_model,
    )

    checkpoint = torch.load(
        Path(args.gemma_checkpoint), map_location="cpu", weights_only=False
    )
    bundled = isinstance(checkpoint, dict) and "port" in checkpoint
    model_name = args.gemma_model or (
        str(checkpoint.get("model", "google/gemma-3-270m-it"))
        if bundled
        else "google/gemma-3-270m-it"
    )
    model = load_gemma_world_model(model_name, device=args.device, dtype=dtype)
    checkpoint_args = checkpoint.get("args", {}) if bundled else {}
    if bundled and checkpoint.get("lora") is not None:
        enable_upper_layer_lora(
            model,
            rank=int(checkpoint_args.get("lora_rank", 8)),
            alpha=int(checkpoint_args.get("lora_alpha", 16)),
        )
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(model.model, checkpoint["lora"])
    model.port.load_state_dict(checkpoint["port"] if bundled else checkpoint)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    query_projector = None
    query_state = checkpoint.get("query_projector") if bundled else None
    if query_state is not None or args.query_checkpoint:
        query_projector = GemmaQueryProjector(
            hidden_dimension=model.port.hidden_dimension
        ).to(args.device)
        query_projector.load_state_dict(
            query_state
            if query_state is not None
            else torch.load(
                Path(args.query_checkpoint),
                map_location=args.device,
                weights_only=True,
            )
        )
        query_projector.eval()
    return GemmaLatentAnswerer(
        memory,
        encoder,
        projector,
        model,
        tokenizer,
        query_projector=query_projector,
        response_instruction=checkpoint_args.get("response_instruction"),
        prompt_format=str(checkpoint_args.get("prompt_format", "chat")),
        constrained=not args.free_generation,
    )


def run(args: argparse.Namespace) -> dict[str, Path]:
    video = Path(args.video)
    output = Path(args.output_dir or f"artifacts/world-video-runs/{video.stem}")
    annotations = load_annotations(args.annotations) if args.annotations else None
    annotated_cup_labels = tuple(
        dict.fromkeys(
            str(entity["label"])
            for checkpoint in (annotations or {}).get("checkpoints", [])
            for entity in checkpoint.get("entities", [])
            if entity.get("class") == "cup" and entity.get("label")
        )
    )
    cup_labels = annotated_cup_labels or ("red", "blue", "yellow")
    vision_processor, memory, encoder, projector, dtype = _build_processor(
        args, cup_labels
    )
    processor = vision_processor
    if annotations and annotations.get("questions"):
        answerer = (
            _build_gemma_answerer(args, memory, encoder, projector, dtype)
            if args.gemma_checkpoint
            else EvidenceAnswerer(memory)
        )
        processor = AnnotatedReplayProcessor(
            vision_processor, answerer, annotations["questions"]
        )
    frames = iter_video_frames(
        video,
        send_fps=args.send_fps,
        jpeg_quality=args.jpeg_quality,
        max_frames=args.max_frames,
    )
    try:
        if args.mode == "realtime":
            replay = asyncio.run(
                replay_realtime(
                    frames,
                    processor,
                    session_id=args.session_id,
                    playback_rate=args.playback_rate,
                )
            )
        else:
            replay = replay_deterministic(frames, processor, session_id=args.session_id)
    finally:
        vision_processor.close_session(args.session_id)

    evaluation = None
    if annotations:
        evaluation = evaluate_replay(
            replay,
            annotations,
            tolerance_s=args.annotation_tolerance,
            iou_threshold=args.iou_threshold,
        )
    paths = write_replay_artifacts(output, replay, evaluation=evaluation, video=video)
    if not args.annotations:
        template_path = output / "annotations.template.json"
        template_path.write_text(
            json.dumps(annotation_template(video.name), indent=2) + "\n",
            encoding="utf-8",
        )
        paths["annotation_template"] = template_path
    return paths


def main() -> None:
    paths = run(build_parser().parse_args())
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


if __name__ == "__main__":
    main()
