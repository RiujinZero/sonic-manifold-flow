"""Environment-driven assembly for the public world-model application."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from manifoldflow.public_demo import create_app
from manifoldflow.world_answer import ConversationalAnswerer, GemmaLatentAnswerer
from manifoldflow.world_conversation import ConversationMemory
from manifoldflow.world_demo import DemoRuntime
from manifoldflow.world_live import VisionFrameProcessor
from manifoldflow.world_vision import (
    CrossModalKeyProjector,
    MultiClassByteTracker,
    SiglipVisionLanguageEncoder,
    YoloXDetector,
)

CLASSES = ("toy_car", "occluder_box", "cup", "ball")

# Seeded into every new session's ConversationMemory (world_demo.py's
# DemoRuntime.new_session -> ConversationalAnswerer.seed_session), so a
# handful of common general-knowledge questions are answered by injected
# vector -- the mechanism this whole project demonstrates -- rather than
# falling back on the recall LoRA's own parametric knowledge, which the
# training needed for reliable correction-tracking measurably damages
# (2026-09-02 dev journal: general trivia collapsed to single
# training-vocabulary nouns after the step budget that made corrections
# work).
#
# Disabled by default (MANIFOLDFLOW_ENABLE_PREFILL=1 to turn on): live
# testing found the recall port has no learned decode behavior at all for
# non-quantity-tracking facts -- injecting "The capital of France is
# Paris." produced repetition-loop garbage ("That's 5 5 5"), not a wrong-
# but-parseable answer, regardless of retrieval working correctly. The
# mechanism itself (storage, retrieval, seed_session) is sound and
# covered by tests; what's missing is training data with general-fact
# examples in a comparable format, not just quantity-tracking ones. Kept
# wired up and gated, not removed, so it activates once that training
# exists instead of being rebuilt from scratch.
_PREFILL_FACTS: tuple[str, ...] = (
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo.",
    "The capital of Italy is Rome.",
    "The sky is blue.",
    "Grass is green.",
    "Two plus two equals four.",
    "There are seven days in a week.",
    "Water freezes at zero degrees Celsius.",
)


def build_app_from_environment():
    """Build curated-only mode or the complete GPU stack when weights are set."""
    runtime = DemoRuntime()
    detector_path = os.environ.get("MANIFOLDFLOW_YOLOX_CHECKPOINT")
    projector_path = os.environ.get("MANIFOLDFLOW_PROJECTOR_CHECKPOINT")
    if not detector_path or not projector_path:
        return create_app(runtime)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    encoder = SiglipVisionLanguageEncoder(device=device, dtype=dtype)
    projector = CrossModalKeyProjector(encoder.feature_dimension, 128)
    projector.load_state_dict(
        torch.load(Path(projector_path), map_location="cpu", weights_only=True)
    )
    projector.eval().to(device)
    detector = YoloXDetector(detector_path, CLASSES, device=device)
    live = VisionFrameProcessor(
        detector,
        encoder,
        projector,
        runtime.memory,
        runtime.engine,
        lambda: MultiClassByteTracker(CLASSES),
    )

    model = None
    tokenizer = None
    checkpoint_path = os.environ.get("MANIFOLDFLOW_GEMMA_CHECKPOINT") or os.environ.get(
        "MANIFOLDFLOW_GEMMA_PORT_CHECKPOINT"
    )
    if checkpoint_path:
        from transformers import AutoTokenizer

        from manifoldflow.gemma_world import (
            GemmaQueryProjector,
            enable_upper_layer_lora,
            load_gemma_world_model,
        )

        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=False
        )
        bundled = isinstance(checkpoint, dict) and "port" in checkpoint
        model_name = os.environ.get("MANIFOLDFLOW_GEMMA_MODEL") or (
            str(checkpoint.get("model", "google/gemma-3-270m-it"))
            if bundled
            else "google/gemma-3-270m-it"
        )
        model = load_gemma_world_model(model_name, device=device, dtype=dtype)
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
        query_path = os.environ.get("MANIFOLDFLOW_QUERY_CHECKPOINT")
        if query_state is not None or query_path:
            query_projector = GemmaQueryProjector(
                hidden_dimension=model.port.hidden_dimension
            ).to(device)
            query_projector.load_state_dict(
                query_state
                if query_state is not None
                else torch.load(
                    Path(query_path), map_location=device, weights_only=True
                )
            )
            query_projector.eval()
        runtime.answerer = GemmaLatentAnswerer(
            runtime.memory,
            encoder,
            projector,
            model,
            tokenizer,
            query_projector=query_projector,
            response_instruction=checkpoint_args.get("response_instruction"),
            prompt_format=str(checkpoint_args.get("prompt_format", "chat")),
            constrained=os.environ.get("MANIFOLDFLOW_GEMMA_FREE_GENERATION") != "1",
        )

    # Language persistence: recall what the user said earlier, alongside the
    # existing visual world-state answering. Retrieval reuses the same
    # trained SigLIP2 text projector as identity keys (no new training
    # needed there), but recall generation is a *separate* port + LoRA from
    # the visual one -- its value vectors are Gemma's own 640-d pooled
    # hidden states (state_dimension=640), not the 256-d typed world-state
    # vectors the visual port expects, so the two cannot share a checkpoint.
    # Falls through to the visual answerer unchanged when nothing relevant
    # was said before, and falls back to quoting the stored text (no
    # injection) when no recall checkpoint is configured.
    recall_model = None
    recall_tokenizer = None
    recall_args: dict = {}
    recall_checkpoint_path = os.environ.get("MANIFOLDFLOW_GEMMA_RECALL_CHECKPOINT")
    if recall_checkpoint_path:
        from transformers import AutoTokenizer

        from manifoldflow.gemma_world import (
            LatentMemoryPort,
            enable_upper_layer_lora,
            load_gemma_world_model,
        )

        recall_checkpoint = torch.load(
            Path(recall_checkpoint_path), map_location="cpu", weights_only=False
        )
        recall_model_name = os.environ.get(
            "MANIFOLDFLOW_GEMMA_MODEL", "google/gemma-3-270m-it"
        )
        recall_model = load_gemma_world_model(
            recall_model_name, device=device, dtype=dtype
        )
        recall_model.port = LatentMemoryPort(
            state_dimension=640, hidden_dimension=recall_model.port.hidden_dimension
        ).to(device)
        recall_args = recall_checkpoint.get("args", {})
        if recall_checkpoint.get("lora") is not None:
            enable_upper_layer_lora(
                recall_model,
                rank=int(recall_args.get("lora_rank", 8)),
                alpha=int(recall_args.get("lora_alpha", 16)),
            )
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(recall_model.model, recall_checkpoint["lora"])
        recall_model.port.load_state_dict(recall_checkpoint["port"])
        recall_model.eval()
        recall_tokenizer = AutoTokenizer.from_pretrained(recall_model_name)
        if recall_tokenizer.pad_token_id is None:
            recall_tokenizer.pad_token = recall_tokenizer.eos_token
        recall_tokenizer.padding_side = "right"

    runtime.answerer = ConversationalAnswerer(
        runtime.answerer,
        ConversationMemory(key_dimension=128, value_dimension=640),
        encoder,
        projector,
        model=recall_model,
        tokenizer=recall_tokenizer,
        # Must equal the checkpoint's own training-time max_memory_slots,
        # not just be large enough: GemmaWorldModel never passes
        # position_ids, so Gemma3 derives them from plain sequence length,
        # ignoring attention_mask -- a different injected memory-slot count
        # shifts every later token's position embedding away from what the
        # port/LoRA were trained under. Read from the checkpoint rather
        # than hardcoded so retraining with a different width can't
        # silently drift out of sync with this call site again.
        top_k=int(recall_args.get("max_memory_slots", 8)),
        prefill_facts=(
            _PREFILL_FACTS if os.environ.get("MANIFOLDFLOW_ENABLE_PREFILL") == "1" else ()
        ),
    )
    return create_app(runtime, live)


def main() -> None:
    try:
        import uvicorn
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Run pip install -e '.[world]' first") from error
    uvicorn.run(
        build_app_from_environment(),
        host=os.environ.get("MANIFOLDFLOW_HOST", "127.0.0.1"),
        port=int(os.environ.get("MANIFOLDFLOW_PORT", "8000")),
        proxy_headers=True,
    )
