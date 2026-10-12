"""Typed records shared by the tabletop world-model components."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor


class VisibilityState(StrEnum):
    VISIBLE = "visible"
    OCCLUDED = "occluded"
    CONTAINED = "contained"
    MISSING = "missing"
    UNKNOWN = "unknown"


class EventType(StrEnum):
    APPEARED = "appeared"
    MOVED = "moved"
    OCCLUDED = "occluded"
    REAPPEARED = "reappeared"
    DISAPPEARED = "disappeared"
    CONTAINED_BY = "contained_by"
    RELEASED = "released"
    ZONE_CHANGED = "zone_changed"


@dataclass(frozen=True)
class ObservationEvent:
    session_id: str
    frame_index: int
    timestamp: float
    tracker_id: int
    class_name: str
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    appearance_key: Tensor
    position: tuple[float, float] | None = None
    velocity: tuple[float, float] = (0.0, 0.0)
    zone: str = "unknown"

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if self.appearance_key.ndim != 1:
            raise ValueError("appearance_key must be one-dimensional")
        x1, y1, x2, y2 = self.box_xyxy
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box_xyxy must have positive width and height")

    @property
    def center(self) -> tuple[float, float]:
        if self.position is not None:
            return self.position
        x1, y1, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class WorldRecord:
    entity_id: str
    session_id: str
    class_name: str
    appearance_key: Tensor
    state_value: Tensor
    box_xyxy: tuple[float, float, float, float]
    position: tuple[float, float]
    velocity: tuple[float, float]
    zone: str
    visibility: VisibilityState
    confidence: float
    first_seen: float
    last_seen: float
    version: int = 1
    observation_count: int = 1
    tracker_ids: set[int] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone(self) -> WorldRecord:
        return WorldRecord(
            entity_id=self.entity_id,
            session_id=self.session_id,
            class_name=self.class_name,
            appearance_key=self.appearance_key.detach().clone(),
            state_value=self.state_value.detach().clone(),
            box_xyxy=self.box_xyxy,
            position=self.position,
            velocity=self.velocity,
            zone=self.zone,
            visibility=self.visibility,
            confidence=self.confidence,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
            version=self.version,
            observation_count=self.observation_count,
            tracker_ids=set(self.tracker_ids),
            metadata=dict(self.metadata),
        )


@dataclass
class RelationRecord:
    subject_id: str
    relation: str
    target_id: str
    confidence: float
    started_at: float
    ended_at: float | None = None
    version: int = 1

    @property
    def active(self) -> bool:
        return self.ended_at is None

    def clone(self) -> RelationRecord:
        return RelationRecord(**vars(self))


@dataclass(frozen=True)
class WorldEvent:
    event_type: EventType
    entity_id: str
    timestamp: float
    frame_index: int
    related_entity_id: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class WorldHit:
    entity_id: str
    score: float
    record: WorldRecord
    related: tuple[WorldRecord, ...] = ()


@dataclass(frozen=True)
class WorldSnapshot:
    session_id: str
    entities: tuple[WorldRecord, ...]
    relations: tuple[RelationRecord, ...]
    events: tuple[WorldEvent, ...]

    def entity(self, entity_id: str) -> WorldRecord | None:
        return next(
            (item for item in self.entities if item.entity_id == entity_id), None
        )


def normalized(vector: Tensor) -> Tensor:
    vector = vector.detach().float().flatten()
    return torch.nn.functional.normalize(vector, dim=0, eps=1e-8)


ZONES: tuple[str, ...] = (
    "upper_left", "upper_center", "upper_right",
    "middle_left", "middle_center", "middle_right",
    "lower_left", "lower_center", "lower_right",
)


def zone_for_box(box_xyxy: tuple[float, float, float, float]) -> str:
    """3x3 grid zone label ("upper_left", ..., "lower_right") from a
    normalized [0,1] xyxy box's center. Single shared definition so the
    live camera, the tabletop simulator, and the trained state encoding
    can never drift out of sync on what a zone name means."""
    x1, y1, x2, y2 = box_xyxy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    column = "left" if center_x < 1 / 3 else "center" if center_x < 2 / 3 else "right"
    row = "upper" if center_y < 1 / 3 else "middle" if center_y < 2 / 3 else "lower"
    return f"{row}_{column}"


def seeded_concept_key(value: str, *, dimension: int = 128) -> Tensor:
    """SHA256-seeded, reproducible identity vector for a concept string
    ("class:cup", "attribute:red", ...) -- same string always produces the
    same vector, in any process, any session, forever. This is the actual
    identity-key space CrossModalKeyProjector and GemmaQueryProjector were
    trained to route into (see TabletopSimulator._appearance_key, which
    this function's blend below matches exactly) -- anything constructing
    a WorldRecord/ObservationEvent outside the simulator (curated demo
    scenarios, live camera labeling) needs to land in this same space, not
    an arbitrary one unrelated to it. See the 2026-09-02 dev journal entry
    diagnosing world_demo.py's curated scenarios using one-hot placeholder
    keys instead, which made their retrieval scores ~1e-6 (garbage)
    regardless of which oracle checkpoint was deployed."""
    digest = hashlib.sha256(value.encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big"))
    return torch.nn.functional.normalize(
        torch.randn(dimension, generator=generator), dim=0
    )


def concept_appearance_key(
    class_name: str,
    attribute: str = "",
    instance: str = "",
    *,
    dimension: int = 128,
) -> Tensor:
    """80% class + 15% attribute + 5% instance-disambiguation blend --
    TabletopSimulator._appearance_key's exact scheme, factored out so
    non-simulator callers can produce keys in the same trained space.
    instance should be a string unique to one entity within one scene
    (e.g. an episode/session id plus the entity's own id) when several
    entities share both class and attribute and need to stay
    distinguishable; leave it blank when that doesn't apply."""
    class_key = seeded_concept_key(f"class:{class_name}", dimension=dimension)
    attribute_key = (
        seeded_concept_key(f"attribute:{attribute}", dimension=dimension)
        if attribute
        else torch.zeros_like(class_key)
    )
    instance_key = (
        seeded_concept_key(f"instance:{instance}", dimension=dimension)
        if instance
        else torch.zeros_like(class_key)
    )
    return torch.nn.functional.normalize(
        0.80 * class_key + 0.15 * attribute_key + 0.05 * instance_key, dim=0
    )
