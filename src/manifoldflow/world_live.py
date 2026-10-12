"""Live JPEG-to-world-state processor for the public browser demo."""

from __future__ import annotations

import io
import threading
import time
from collections.abc import Callable
from typing import Any

import torch

from manifoldflow.world_events import TabletopEventEngine
from manifoldflow.world_identity import TemporalLabelEvidence
from manifoldflow.world_memory import WorldMemory
from manifoldflow.world_types import ObservationEvent, zone_for_box
from manifoldflow.world_vision import (
    CrossModalKeyProjector,
    Detector,
    MultiClassByteTracker,
    VisionLanguageEncoder,
    crop_detection,
)


class VisionFrameProcessor:
    """Compose detector, ByteTrack, SigLIP2, and ManifoldFlow per session."""

    def __init__(
        self,
        detector: Detector,
        encoder: VisionLanguageEncoder,
        projector: CrossModalKeyProjector,
        memory: WorldMemory,
        engine: TabletopEventEngine,
        tracker_factory: Callable[[], MultiClassByteTracker],
        cup_labels: tuple[str, ...] = ("red", "blue", "yellow"),
    ) -> None:
        if not cup_labels or any(not item.strip() for item in cup_labels):
            raise ValueError("cup_labels must contain non-empty labels")
        self.detector = detector
        self.encoder = encoder
        self.projector = projector
        self.memory = memory
        self.engine = engine
        self.tracker_factory = tracker_factory
        self.cup_labels = cup_labels
        self._trackers: dict[str, MultiClassByteTracker] = {}
        self._frame_indices: dict[str, int] = {}
        self._inference_lock = threading.Lock()
        self._cup_label_keys = None
        self._label_evidence: dict[tuple[str, str], TemporalLabelEvidence] = {}

    def process(self, session_id: str, jpeg: bytes) -> dict[str, Any]:
        """Process a live frame using its wall-clock arrival time."""
        return self.process_at(session_id, jpeg, timestamp=time.time())

    def process_at(
        self, session_id: str, jpeg: bytes, *, timestamp: float
    ) -> dict[str, Any]:
        """Process a frame at an explicit timestamp for deterministic replay."""
        if timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        try:
            from PIL import Image
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("Live processing requires Pillow") from error
        if not jpeg or len(jpeg) > 1_000_000:
            raise ValueError("frame must be a JPEG/WebP image no larger than 1 MB")
        with Image.open(io.BytesIO(jpeg)) as loaded:
            image = loaded.convert("RGB")
        if image.width > 1280 or image.height > 720:
            image.thumbnail((1280, 720))
        with self._inference_lock:
            detections = self.detector.detect(image)
            crops = [crop_detection(image, item) for item in detections]
            features = self.encoder.encode_images(crops)
            projector_device = next(self.projector.parameters()).device
            keys = self.projector(features.to(projector_device)).detach().cpu()
            tracker = self._trackers.setdefault(session_id, self.tracker_factory())
            tracked, keys, tracked_indices = tracker.update_with_features(
                detections, keys
            )
            crops = [crops[index] for index in tracked_indices]
            cup_probabilities = self._cup_probabilities(
                tracked, keys, projector_device
            )
        frame_index = self._frame_indices.get(session_id, -1) + 1
        self._frame_indices[session_id] = frame_index
        observations = [
            ObservationEvent(
                session_id=session_id,
                frame_index=frame_index,
                timestamp=timestamp,
                tracker_id=item.tracker_id,
                class_name=item.class_name,
                box_xyxy=item.box_xyxy,
                confidence=item.confidence,
                appearance_key=key,
                zone=self._zone(item.box_xyxy),
            )
            for item, key in zip(tracked, keys, strict=True)
        ]
        self.engine.process_frame(session_id, frame_index, timestamp, observations)
        snapshot = self.memory.snapshot(session_id)
        cup_labels: list[str] = []
        for detection, probabilities, crop in zip(
            tracked, cup_probabilities, crops, strict=True
        ):
            if probabilities is None:
                cup_labels.append("")
                continue
            record = next(
                (
                    item
                    for item in snapshot.entities
                    if detection.tracker_id in item.tracker_ids
                ),
                None,
            )
            if record is not None:
                evidence = self._label_evidence.setdefault(
                    (session_id, record.entity_id),
                    TemporalLabelEvidence(self.cup_labels),
                )
                weight = detection.confidence * self._crop_quality(crop)
                label, confidence = evidence.update(probabilities, weight)
                if label:
                    self.memory.set_metadata(
                        session_id, record.entity_id, "label", label
                    )
                self.memory.set_metadata(
                    session_id, record.entity_id, "label_confidence", confidence
                )
                cup_labels.append(label)
            else:
                cup_labels.append("")
        snapshot = self.memory.snapshot(session_id)
        return {
            "frame": frame_index,
            "timestamp": timestamp,
            "observations": [
                {
                    "tracker_id": item.tracker_id,
                    "class": item.class_name,
                    "box": item.box_xyxy,
                    "confidence": item.confidence,
                    "label": label,
                    "zone": self._zone(item.box_xyxy),
                    "visibility": "visible",
                }
                for item, label in zip(tracked, cup_labels, strict=True)
            ],
            "entities": [
                {
                    "id": item.entity_id,
                    "class": item.class_name,
                    "box": item.box_xyxy,
                    "zone": item.zone,
                    "visibility": item.visibility.value,
                    "confidence": item.confidence,
                    "version": item.version,
                    "label": item.metadata.get("label", ""),
                    "label_confidence": item.metadata.get("label_confidence", 0.0),
                }
                for item in snapshot.entities
            ],
            "relations": [
                {
                    "subject": item.subject_id,
                    "relation": item.relation,
                    "target": item.target_id,
                    "active": item.active,
                    "confidence": item.confidence,
                }
                for item in snapshot.relations
            ],
            "events": [
                {
                    "type": item.event_type.value,
                    "entity": item.entity_id,
                    "related": item.related_entity_id,
                    "frame": item.frame_index,
                }
                for item in snapshot.events[-20:]
            ],
        }

    def close_session(self, session_id: str) -> None:
        self._trackers.pop(session_id, None)
        self._frame_indices.pop(session_id, None)
        for key in [item for item in self._label_evidence if item[0] == session_id]:
            del self._label_evidence[key]

    def _cup_probabilities(self, tracked, keys, projector_device):
        if self._cup_label_keys is None:
            text = self.encoder.encode_texts(
                [f"a photo of a {label} cup" for label in self.cup_labels]
            )
            self._cup_label_keys = (
                self.projector(text.to(projector_device)).detach().cpu()
            )
        result: list[torch.Tensor | None] = []
        for detection, key in zip(tracked, keys, strict=True):
            if detection.class_name != "cup":
                result.append(None)
                continue
            logits = key @ self._cup_label_keys.T
            result.append(torch.softmax(logits / 0.07, dim=-1))
        return result

    @staticmethod
    def _crop_quality(crop: Any) -> float:
        from PIL import ImageFilter, ImageStat

        gray = crop.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_variance = float(ImageStat.Stat(edges).var[0])
        sharpness = edge_variance / (edge_variance + 250.0)
        area = max(1, crop.width * crop.height)
        scale = min(1.0, (area**0.5) / 80.0)
        return max(0.1, (0.5 + 0.5 * sharpness) * scale)

    @staticmethod
    def _zone(box: tuple[float, float, float, float]) -> str:
        return zone_for_box(box)
