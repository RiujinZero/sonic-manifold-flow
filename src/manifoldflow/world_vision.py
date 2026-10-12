"""Vision and vision-language adapters for the tabletop world-model path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from manifoldflow.world_types import normalized


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackedDetection(Detection):
    tracker_id: int


@dataclass
class _AssociationTrack:
    box: tuple[float, float, float, float]
    missed: int = 0
    appearance_key: Tensor | None = None
    center: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)


class Detector(Protocol):
    def detect(self, image: Any) -> list[Detection]: ...


class VisionLanguageEncoder(Protocol):
    feature_dimension: int

    def encode_images(self, images: Sequence[Any]) -> Tensor: ...

    def encode_texts(self, texts: Sequence[str]) -> Tensor: ...


class CrossModalKeyProjector(nn.Module):
    """Shared projection that preserves SigLIP's aligned image/text space."""

    def __init__(self, feature_dimension: int, key_dimension: int = 128) -> None:
        super().__init__()
        if feature_dimension <= 0 or key_dimension <= 0:
            raise ValueError("projection dimensions must be positive")
        self.projection = nn.Linear(feature_dimension, key_dimension, bias=False)

    def forward(self, features: Tensor) -> Tensor:
        return torch.nn.functional.normalize(self.projection(features.float()), dim=-1)


class SiglipVisionLanguageEncoder:
    """Lazy optional wrapper around the frozen SigLIP2 image/text encoders."""

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-224",
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "SigLIP2 requires the world extra: pip install -e '.[world]'"
            ) from error
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(model_name, dtype=dtype).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.feature_dimension = int(self.model.config.text_config.projection_size)

    @torch.inference_mode()
    def encode_images(self, images: Sequence[Any]) -> Tensor:
        if not images:
            return torch.empty((0, self.feature_dimension))
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        output = self.model.get_image_features(**inputs)
        features = output if isinstance(output, Tensor) else output.pooler_output
        return torch.nn.functional.normalize(features.float(), dim=-1).cpu()

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str]) -> Tensor:
        if not texts:
            return torch.empty((0, self.feature_dimension))
        inputs = self.processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        output = self.model.get_text_features(**inputs)
        features = output if isinstance(output, Tensor) else output.pooler_output
        return torch.nn.functional.normalize(features.float(), dim=-1).cpu()


class YoloXDetector:
    """Official YOLOX checkpoint adapter with normalized box output."""

    def __init__(
        self,
        checkpoint: str | Path,
        class_names: Sequence[str],
        *,
        model_name: str = "yolox-nano",
        confidence: float = 0.3,
        class_confidences: Mapping[str, float] | None = None,
        nms_threshold: float = 0.45,
        device: str | torch.device = "cuda",
    ) -> None:
        try:
            from yolox.data.data_augment import ValTransform
            from yolox.exp import get_exp
            from yolox.utils import postprocess
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "YOLOX requires the world extra and the pinned YOLOX source"
            ) from error
        if not class_names:
            raise ValueError("class_names must be non-empty")
        self.class_names = tuple(class_names)
        unknown_classes = set(class_confidences or ()) - set(self.class_names)
        if unknown_classes:
            raise ValueError(
                f"unknown class confidence keys: {sorted(unknown_classes)}"
            )
        self.confidence = confidence
        self.class_confidences = dict(class_confidences or {})
        thresholds = [confidence, *self.class_confidences.values()]
        if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
            raise ValueError("confidence thresholds must be between zero and one")
        self._postprocess_confidence = min(thresholds)
        self.nms_threshold = nms_threshold
        self.device = torch.device(device)
        self._postprocess = postprocess
        self._transform = ValTransform(legacy=False)
        self._exp = get_exp(None, model_name)
        self._exp.num_classes = len(class_names)
        self.model = self._exp.get_model().to(self.device).eval()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = payload.get("model", payload) if isinstance(payload, dict) else payload
        self.model.load_state_dict(state)

    @torch.inference_mode()
    def detect(self, image: Any) -> list[Detection]:
        import numpy as np

        array = np.asarray(image)
        height, width = array.shape[:2]
        transformed, _ = self._transform(array, None, self._exp.test_size)
        value = torch.from_numpy(transformed).unsqueeze(0).float().to(self.device)
        output = self._postprocess(
            self.model(value),
            len(self.class_names),
            self._postprocess_confidence,
            self.nms_threshold,
            class_agnostic=True,
        )[0]
        if output is None:
            return []
        ratio = min(self._exp.test_size[0] / height, self._exp.test_size[1] / width)
        output[:, :4] /= ratio
        detections: list[Detection] = []
        for row in output.cpu():
            class_index = int(row[6].item())
            if not 0 <= class_index < len(self.class_names):
                continue
            class_name = self.class_names[class_index]
            score = float((row[4] * row[5]).item())
            if score < self.class_confidences.get(class_name, self.confidence):
                continue
            x1, y1, x2, y2 = row[:4].tolist()
            normalized_box = (
                max(0.0, min(1.0, x1 / width)),
                max(0.0, min(1.0, y1 / height)),
                max(0.0, min(1.0, x2 / width)),
                max(0.0, min(1.0, y2 / height)),
            )
            if (
                normalized_box[2] <= normalized_box[0]
                or normalized_box[3] <= normalized_box[1]
            ):
                continue
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=score,
                    box_xyxy=normalized_box,
                )
            )
        return detections


class MultiClassByteTracker:
    """MIT ByteTrack adapter with independent identity lanes per object class."""

    def __init__(
        self,
        class_names: Sequence[str],
        *,
        track_threshold: float = 0.35,
        class_track_thresholds: Mapping[str, float] | None = None,
        match_threshold: float = 0.8,
        frame_rate: int = 10,
        track_buffer: int = 30,
        appearance_match_threshold: float = 0.3,
        association_margin: float = 0.08,
    ) -> None:
        try:
            from yolox.tracker.byte_tracker import BYTETracker
        except ImportError:  # Newer official YOLOX releases removed this package.
            BYTETracker = None
        self._class_names = tuple(class_names)
        unknown_classes = set(class_track_thresholds or ()) - set(self._class_names)
        if unknown_classes:
            raise ValueError(f"unknown class track keys: {sorted(unknown_classes)}")
        self._track_thresholds = {
            name: float((class_track_thresholds or {}).get(name, track_threshold))
            for name in self._class_names
        }
        if any(not 0.0 <= value <= 1.0 for value in self._track_thresholds.values()):
            raise ValueError("track thresholds must be between zero and one")
        if appearance_match_threshold < 0 or association_margin < 0:
            raise ValueError("association thresholds must be non-negative")
        if BYTETracker is not None:
            self._trackers = {
                name: BYTETracker(
                    SimpleNamespace(
                        track_thresh=self._track_thresholds[name],
                        track_buffer=track_buffer,
                        match_thresh=match_threshold,
                        mot20=False,
                    ),
                    frame_rate=frame_rate,
                )
                for name in self._class_names
            }
        else:
            self._trackers = None
        self._fallback_iou_threshold = max(0.1, 1.0 - match_threshold)
        self._appearance_match_threshold = appearance_match_threshold
        self._association_margin = association_margin
        self._track_buffer = track_buffer
        self._fallback_tracks: dict[str, dict[int, _AssociationTrack]] = {
            name: {} for name in class_names
        }
        self._next_local_id: dict[str, int] = {name: 1 for name in class_names}
        self._global_ids: dict[tuple[str, int], int] = {}
        self._next_id = 1

    def update(self, detections: Sequence[Detection]) -> list[TrackedDetection]:
        if self._trackers is None:
            return self._fallback_update(detections)
        import numpy as np

        tracked: list[TrackedDetection] = []
        for class_name, tracker in self._trackers.items():
            selected = [item for item in detections if item.class_name == class_name]
            rows = np.asarray(
                [(*item.box_xyxy, item.confidence) for item in selected],
                dtype=np.float32,
            ).reshape(-1, 5)
            targets = tracker.update(rows, (1, 1), (1, 1))
            for target in targets:
                local_id = int(target.track_id)
                identity = (class_name, local_id)
                if identity not in self._global_ids:
                    self._global_ids[identity] = self._next_id
                    self._next_id += 1
                x, y, width, height = (float(item) for item in target.tlwh)
                tracked.append(
                    TrackedDetection(
                        class_name=class_name,
                        confidence=float(target.score),
                        box_xyxy=(x, y, x + width, y + height),
                        tracker_id=self._global_ids[identity],
                    )
                )
        return tracked

    def _fallback_update(
        self, detections: Sequence[Detection]
    ) -> list[TrackedDetection]:
        tracked, _ = self._associate(detections, None)
        return tracked

    def update_with_features(
        self, detections: Sequence[Detection], appearance_keys: Tensor
    ) -> tuple[list[TrackedDetection], Tensor, list[int]]:
        """Associate detections using appearance, motion, and overlap evidence."""
        if appearance_keys.ndim != 2 or appearance_keys.shape[0] != len(detections):
            raise ValueError("appearance_keys must have one row per detection")
        tracked, indices = self._associate(detections, appearance_keys)
        return tracked, appearance_keys[indices], indices

    def _associate(
        self,
        detections: Sequence[Detection],
        appearance_keys: Tensor | None,
    ) -> tuple[list[TrackedDetection], list[int]]:
        tracked: list[TrackedDetection] = []
        tracked_indices: list[int] = []
        for class_name in self._class_names:
            states = self._fallback_tracks[class_name]
            for state in states.values():
                state.missed += 1
            selected = sorted(
                (
                    (index, item)
                    for index, item in enumerate(detections)
                    if item.class_name == class_name
                    and item.confidence >= self._track_thresholds[class_name]
                ),
                key=lambda pair: pair[1].confidence,
                reverse=True,
            )
            claimed: set[int] = set()
            for detection_index, detection in selected:
                key = (
                    None
                    if appearance_keys is None
                    else normalized(appearance_keys[detection_index])
                )
                candidates = sorted(
                    (
                        self._association_score(detection.box_xyxy, key, state),
                        local_id,
                    )
                    for local_id, state in states.items()
                    if local_id not in claimed
                )
                candidates.reverse()
                score, local_id = candidates[0] if candidates else (0.0, 0)
                second_score = candidates[1][0] if len(candidates) > 1 else 0.0
                prior = states.get(local_id)
                strong_overlap = bool(
                    prior
                    and self._iou(detection.box_xyxy, prior.box)
                    >= max(0.5, self._fallback_iou_threshold)
                )
                ambiguous = (
                    len(candidates) > 1
                    and score - second_score < self._association_margin
                    and not strong_overlap
                )
                if score < self._fallback_iou_threshold or ambiguous:
                    local_id = self._next_local_id[class_name]
                    self._next_local_id[class_name] += 1
                    prior = None
                center = self._center(detection.box_xyxy)
                velocity = (
                    (0.0, 0.0)
                    if prior is None
                    else (center[0] - prior.center[0], center[1] - prior.center[1])
                )
                updated_key = key
                if (
                    prior is not None
                    and key is not None
                    and prior.appearance_key is not None
                ):
                    updated_key = normalized(0.8 * prior.appearance_key + 0.2 * key)
                states[local_id] = _AssociationTrack(
                    box=detection.box_xyxy,
                    appearance_key=updated_key,
                    center=center,
                    velocity=velocity,
                )
                claimed.add(local_id)
                identity = (class_name, local_id)
                if identity not in self._global_ids:
                    self._global_ids[identity] = self._next_id
                    self._next_id += 1
                tracked.append(
                    TrackedDetection(
                        class_name=class_name,
                        confidence=detection.confidence,
                        box_xyxy=detection.box_xyxy,
                        tracker_id=self._global_ids[identity],
                    )
                )
                tracked_indices.append(detection_index)
            for local_id, state in list(states.items()):
                if state.missed > self._track_buffer:
                    states.pop(local_id)
        if not tracked_indices:
            return [], []
        order = sorted(range(len(tracked_indices)), key=tracked_indices.__getitem__)
        return (
            [tracked[index] for index in order],
            [tracked_indices[index] for index in order],
        )

    def _association_score(
        self,
        box: tuple[float, float, float, float],
        key: Tensor | None,
        state: _AssociationTrack,
    ) -> float:
        overlap = self._iou(box, state.box)
        center = self._center(box)
        predicted = (
            state.center[0] + state.velocity[0],
            state.center[1] + state.velocity[1],
        )
        distance = (
            (center[0] - predicted[0]) ** 2 + (center[1] - predicted[1]) ** 2
        ) ** 0.5
        motion = max(0.0, 1.0 - distance / 0.5)
        if key is None or state.appearance_key is None:
            return max(overlap, 0.25 * overlap + 0.15 * motion)
        appearance_distance = float((key - state.appearance_key).square().sum().item())
        if appearance_distance > self._appearance_match_threshold:
            if overlap < self._fallback_iou_threshold:
                return 0.0
            return 0.25 * overlap + 0.10 * motion
        appearance = max(0.0, 1.0 - appearance_distance / 2.0)
        return 0.55 * appearance + 0.25 * overlap + 0.20 * motion

    @staticmethod
    def _center(
        box: tuple[float, float, float, float]
    ) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @staticmethod
    def _iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = (first[2] - first[0]) * (first[3] - first[1])
        second_area = (second[2] - second[0]) * (second[3] - second[1])
        return intersection / max(first_area + second_area - intersection, 1e-8)


def crop_detection(image: Any, detection: Detection, padding: float = 0.05) -> Any:
    """Crop a normalized detection box from a PIL-compatible image."""
    width, height = image.size
    x1, y1, x2, y2 = detection.box_xyxy
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    pixels = (
        max(0, min(width, round((x1 - pad_x) * width))),
        max(0, min(height, round((y1 - pad_y) * height))),
        max(0, min(width, round((x2 + pad_x) * width))),
        max(0, min(height, round((y2 + pad_y) * height))),
    )
    if pixels[2] <= pixels[0] or pixels[3] <= pixels[1]:
        raise ValueError(
            f"detection box falls outside image bounds: {detection.box_xyxy}"
        )
    return image.crop(pixels)


def project_single(projector: CrossModalKeyProjector, feature: Tensor) -> Tensor:
    if feature.ndim != 1:
        raise ValueError("feature must be one-dimensional")
    return normalized(projector(feature[None, :])[0])
