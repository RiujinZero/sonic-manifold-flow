"""Session-isolated ManifoldFlow memory for persistent visual entities."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor

from manifoldflow.bank import BasinState
from manifoldflow.index import ExactMainDeltaIndex
from manifoldflow.world_types import (
    ZONES,
    EventType,
    ObservationEvent,
    RelationRecord,
    VisibilityState,
    WorldEvent,
    WorldHit,
    WorldRecord,
    WorldSnapshot,
    normalized,
)

_WORLD_CLASSES = ("toy_car", "occluder_box", "cup", "ball")
_WORLD_RELATIONS = {"occluded_by": 0, "contained_by": 1}


@dataclass
class _SessionState:
    entities: dict[str, WorldRecord] = field(default_factory=dict)
    tracker_to_entity: dict[int, str] = field(default_factory=dict)
    relations: list[RelationRecord] = field(default_factory=list)
    events: list[WorldEvent] = field(default_factory=list)
    bank: BasinState | None = None
    index: ExactMainDeltaIndex | None = None
    basin_to_entity: list[str] = field(default_factory=list)
    entity_to_basin: dict[str, int] = field(default_factory=dict)
    next_entity: int = 1


class WorldStateEncoder:
    """Deterministic typed-state encoder; trainable ports can replace it later."""

    def __init__(self, dimension: int = 256) -> None:
        if dimension < 16:
            raise ValueError("state dimension must be at least 16")
        self.dimension = dimension

    def encode(self, record: WorldRecord) -> Tensor:
        value = torch.zeros(self.dimension, dtype=torch.float32)
        x1, y1, x2, y2 = record.box_xyxy
        visibility = list(VisibilityState).index(record.visibility)
        value[:12] = torch.tensor(
            [
                record.position[0],
                record.position[1],
                record.velocity[0],
                record.velocity[1],
                x1,
                y1,
                x2,
                y2,
                record.confidence,
                float(visibility) / max(len(VisibilityState) - 1, 1),
                min(record.observation_count / 100.0, 1.0),
                min(record.version / 100.0, 1.0),
            ]
        )
        relations_as_subject = record.metadata.get("_relations_as_subject", {})
        relations_as_target = record.metadata.get("_relations_as_target", {})
        relation_count = 0
        if isinstance(relations_as_subject, dict):
            for relation, confidence in relations_as_subject.items():
                index = _WORLD_RELATIONS.get(str(relation))
                if index is not None and self.dimension > 16 + index:
                    value[16 + index] = float(confidence)
                    relation_count += 1
        if isinstance(relations_as_target, dict):
            for relation, confidence in relations_as_target.items():
                index = _WORLD_RELATIONS.get(str(relation))
                if index is not None and self.dimension > 20 + index:
                    value[20 + index] = float(confidence)
                    relation_count += 1
        if self.dimension > 12:
            value[12] = min(relation_count / 4.0, 1.0)
        if self.dimension >= 33 and record.zone in ZONES:
            value[24 + ZONES.index(record.zone)] = 1.0
        if self.dimension >= 37 and record.class_name in _WORLD_CLASSES:
            value[33 + _WORLD_CLASSES.index(record.class_name)] = 1.0
        return value


class WorldMemory:
    """Small exact-RBF bank with explicit entity upserts and graph expansion."""

    def __init__(
        self,
        key_dimension: int = 128,
        state_dimension: int = 256,
        reid_threshold: float = 0.28,
        reid_margin: float = 0.03,
        reid_spatial_threshold: float = 0.4,
        class_reid_thresholds: Mapping[str, float] | None = None,
        rbf_temperature: float = 0.15,
    ) -> None:
        if key_dimension <= 0 or state_dimension <= 0:
            raise ValueError("dimensions must be positive")
        if min(reid_threshold, reid_margin, reid_spatial_threshold) < 0:
            raise ValueError("routing thresholds must be positive")
        if rbf_temperature <= 0:
            raise ValueError("rbf_temperature must be positive")
        unknown_classes = set(class_reid_thresholds or ()) - set(_WORLD_CLASSES)
        if unknown_classes:
            raise ValueError(f"unknown re-identification classes: {unknown_classes}")
        self.key_dimension = key_dimension
        self.state_dimension = state_dimension
        self.reid_threshold = reid_threshold
        self.reid_margin = reid_margin
        self.reid_spatial_threshold = reid_spatial_threshold
        self.class_reid_thresholds = dict(class_reid_thresholds or {})
        self.rbf_temperature = rbf_temperature
        self._sessions: dict[str, _SessionState] = {}
        self._lock = threading.RLock()
        self._encoder = WorldStateEncoder(state_dimension)

    def _session(self, session_id: str) -> _SessionState:
        if not session_id:
            raise ValueError("session_id must be non-empty")
        return self._sessions.setdefault(session_id, _SessionState())

    def _validate_key(self, key: Tensor) -> Tensor:
        if key.shape != (self.key_dimension,):
            raise ValueError(f"appearance key must have shape [{self.key_dimension}]")
        return normalized(key)

    @torch.no_grad()
    def upsert_observation(self, event: ObservationEvent) -> tuple[WorldRecord, bool]:
        key = self._validate_key(event.appearance_key)
        with self._lock:
            session = self._session(event.session_id)
            entity_id = session.tracker_to_entity.get(event.tracker_id)
            if entity_id is None:
                entity_id = self._reidentify(
                    session,
                    event.class_name,
                    key,
                    timestamp=event.timestamp,
                    position=event.center,
                )
            created = entity_id is None
            if created:
                entity_id = f"{event.class_name}-{session.next_entity:04d}"
                session.next_entity += 1
                record = WorldRecord(
                    entity_id=entity_id,
                    session_id=event.session_id,
                    class_name=event.class_name,
                    appearance_key=key,
                    state_value=torch.zeros(self.state_dimension),
                    box_xyxy=event.box_xyxy,
                    position=event.center,
                    velocity=event.velocity,
                    zone=event.zone,
                    visibility=VisibilityState.VISIBLE,
                    confidence=event.confidence,
                    first_seen=event.timestamp,
                    last_seen=event.timestamp,
                    tracker_ids={event.tracker_id},
                )
                session.entities[entity_id] = record
                if session.bank is None:
                    session.bank = BasinState.from_statistics(
                        key[None, :],
                        torch.ones(1, dtype=torch.long),
                        torch.zeros_like(key)[None, :],
                    )
                    session.index = ExactMainDeltaIndex(session.bank.mu)
                    basin_id = 0
                else:
                    basin_id = session.bank.append_statistics(
                        key, count=1, m2=torch.zeros_like(key)
                    )
                    assert session.index is not None
                    session.index.upsert(
                        torch.tensor([basin_id]), session.bank.mu[basin_id][None, :]
                    )
                session.basin_to_entity.append(entity_id)
                session.entity_to_basin[entity_id] = basin_id
            else:
                record = session.entities[entity_id]
                assert session.bank is not None and session.index is not None
                basin_id = session.entity_to_basin[entity_id]
                session.bank.update_winners(key[None, :], torch.tensor([basin_id]))
                session.bank.mu[basin_id] = normalized(session.bank.mu[basin_id])
                session.index.upsert(
                    torch.tensor([basin_id]), session.bank.mu[basin_id][None, :]
                )
                record.appearance_key = session.bank.mu[basin_id].detach().clone()
                record.box_xyxy = event.box_xyxy
                record.position = event.center
                record.velocity = event.velocity
                record.zone = event.zone
                record.visibility = VisibilityState.VISIBLE
                record.confidence = event.confidence
                record.last_seen = event.timestamp
                record.version += 1
                record.observation_count += 1
                record.tracker_ids.add(event.tracker_id)
            session.tracker_to_entity[event.tracker_id] = entity_id
            record.state_value = self._encoder.encode(record)
            return record.clone(), created

    def _reidentify(
        self,
        session: _SessionState,
        class_name: str,
        key: Tensor,
        *,
        timestamp: float | None = None,
        position: tuple[float, float] | None = None,
    ) -> str | None:
        if session.index is None or session.bank is None:
            return None
        candidate_count = session.bank.mu.shape[0]
        basin_ids = session.index.query(key[None, :], candidate_count)[0].tolist()
        candidates: list[tuple[float, str]] = []
        for basin_id in basin_ids:
            entity_id = session.basin_to_entity[basin_id]
            record = session.entities[entity_id]
            if record.class_name != class_name:
                continue
            if timestamp is not None and record.last_seen >= timestamp:
                continue
            if position is not None and record.visibility == VisibilityState.VISIBLE:
                spatial_distance = (
                    (record.position[0] - position[0]) ** 2
                    + (record.position[1] - position[1]) ** 2
                ) ** 0.5
                if spatial_distance > self.reid_spatial_threshold:
                    continue
            distance = float((session.bank.mu[basin_id] - key).square().sum().item())
            candidates.append((distance, entity_id))
        candidates.sort()
        if not candidates:
            return None
        winner_distance, winner_id = candidates[0]
        threshold = self.class_reid_thresholds.get(class_name, self.reid_threshold)
        if winner_distance > threshold:
            return None
        if (
            len(candidates) > 1
            and candidates[1][0] - winner_distance < self.reid_margin
        ):
            return None
        return winner_id

    def entity_for_tracker(self, session_id: str, tracker_id: int) -> str | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return (
                None if session is None else session.tracker_to_entity.get(tracker_id)
            )

    def match_identity(
        self,
        session_id: str,
        class_name: str,
        appearance_key: Tensor,
        *,
        timestamp: float | None = None,
        position: tuple[float, float] | None = None,
    ) -> str | None:
        """Resolve an image latent without mutating its basin statistics."""
        key = self._validate_key(appearance_key)
        with self._lock:
            session = self._sessions.get(session_id)
            return (
                None
                if session is None
                else self._reidentify(
                    session,
                    class_name,
                    key,
                    timestamp=timestamp,
                    position=position,
                )
            )

    def set_visibility(
        self, session_id: str, entity_id: str, visibility: VisibilityState
    ) -> WorldRecord:
        with self._lock:
            record = self._session(session_id).entities[entity_id]
            record.visibility = visibility
            record.version += 1
            record.state_value = self._encoder.encode(record)
            return record.clone()

    def set_metadata(
        self, session_id: str, entity_id: str, key: str, value: object
    ) -> None:
        with self._lock:
            self._session(session_id).entities[entity_id].metadata[key] = value

    def set_relation(
        self,
        session_id: str,
        subject_id: str,
        relation: str,
        target_id: str,
        confidence: float,
        timestamp: float,
    ) -> RelationRecord:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        with self._lock:
            session = self._session(session_id)
            if subject_id not in session.entities or target_id not in session.entities:
                raise KeyError("both relation entities must exist in the session")
            for existing in session.relations:
                if existing.active and existing.subject_id == subject_id:
                    if (
                        existing.relation == relation
                        and existing.target_id == target_id
                    ):
                        existing.confidence = confidence
                        existing.version += 1
                        self._refresh_relation_state(session)
                        return existing.clone()
                    existing.ended_at = timestamp
            result = RelationRecord(
                subject_id=subject_id,
                relation=relation,
                target_id=target_id,
                confidence=confidence,
                started_at=timestamp,
            )
            session.relations.append(result)
            self._refresh_relation_state(session)
            return result.clone()

    def clear_relation(
        self, session_id: str, subject_id: str, timestamp: float
    ) -> None:
        with self._lock:
            session = self._session(session_id)
            for relation in session.relations:
                if relation.active and relation.subject_id == subject_id:
                    relation.ended_at = timestamp
                    relation.version += 1
            self._refresh_relation_state(session)

    def _refresh_relation_state(self, session: _SessionState) -> None:
        for record in session.entities.values():
            record.metadata.pop("_relations_as_subject", None)
            record.metadata.pop("_relations_as_target", None)
        for relation in session.relations:
            if not relation.active:
                continue
            subject = session.entities[relation.subject_id]
            target = session.entities[relation.target_id]
            subject.metadata.setdefault("_relations_as_subject", {})[
                relation.relation
            ] = relation.confidence
            target.metadata.setdefault("_relations_as_target", {})[
                relation.relation
            ] = relation.confidence
        for record in session.entities.values():
            record.state_value = self._encoder.encode(record)

    def append_event(self, session_id: str, event: WorldEvent) -> None:
        with self._lock:
            session = self._session(session_id)
            session.events.append(event)
            if len(session.events) > 1_000:
                del session.events[:-1_000]

    @torch.no_grad()
    def query(
        self, session_id: str, query_keys: Tensor, top_k: int = 4
    ) -> list[WorldHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if query_keys.ndim == 1:
            query_keys = query_keys[None, :]
        if query_keys.ndim != 2 or query_keys.shape[1] != self.key_dimension:
            raise ValueError(
                f"query_keys must have shape [routes, {self.key_dimension}]"
            )
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.entities:
                return []
            assert session.bank is not None and session.index is not None
            queries = torch.stack([normalized(item) for item in query_keys])
            route_k = min(max(top_k * 2, top_k), session.bank.mu.shape[0])
            routed = session.index.query(queries, route_k)
            candidate_ids = torch.unique(routed.flatten())
            centers = session.bank.mu[candidate_ids]
            squared = (queries[:, None, :] - centers[None, :, :]).square().sum(-1)
            best_distance = squared.amin(dim=0)
            scores = torch.exp(-best_distance / self.rbf_temperature)
            count = min(top_k, candidate_ids.numel())
            local_indices = torch.topk(scores, count).indices.tolist()
            hits: list[WorldHit] = []
            for local_index in local_indices:
                basin_id = int(candidate_ids[local_index].item())
                record = session.entities[session.basin_to_entity[basin_id]]
                related_ids: set[str] = set()
                for relation in session.relations:
                    if not relation.active:
                        continue
                    if relation.subject_id == record.entity_id:
                        related_ids.add(relation.target_id)
                    if relation.target_id == record.entity_id:
                        related_ids.add(relation.subject_id)
                related = tuple(session.entities[item].clone() for item in related_ids)
                hits.append(
                    WorldHit(
                        entity_id=record.entity_id,
                        score=float(scores[local_index].item()),
                        record=record.clone(),
                        related=related,
                    )
                )
            return hits

    def basin_state(self, session_id: str) -> BasinState | None:
        """Expose a detached latent-bank snapshot for checkpointing and analysis."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.bank is None:
                return None
            bank = session.bank
            return BasinState.from_serialized(
                {
                    "mu": bank.mu.detach().clone(),
                    "raw_rate": bank.raw_rate.detach().clone(),
                    "bias": bank.bias.detach().clone(),
                    "count": bank.count.detach().clone(),
                    "m2": bank.m2.detach().clone(),
                    "usage": bank.usage.detach().clone(),
                    "version": bank.version,
                }
            )

    def snapshot(self, session_id: str) -> WorldSnapshot:
        with self._lock:
            session = self._sessions.get(session_id, _SessionState())
            return WorldSnapshot(
                session_id=session_id,
                entities=tuple(item.clone() for item in session.entities.values()),
                relations=tuple(item.clone() for item in session.relations),
                events=tuple(session.events),
            )

    def serializable_session(self, session_id: str) -> dict[str, object]:
        """Serialize typed records together with the original latent basin bank."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.bank is None:
                raise KeyError(f"session {session_id!r} has no world state")
            bank = session.bank
            return {
                "format_version": 1,
                "key_dimension": self.key_dimension,
                "state_dimension": self.state_dimension,
                "reid_threshold": self.reid_threshold,
                "rbf_temperature": self.rbf_temperature,
                "session_id": session_id,
                "bank": {
                    "mu": bank.mu.detach().cpu(),
                    "raw_rate": bank.raw_rate.detach().cpu(),
                    "bias": bank.bias.detach().cpu(),
                    "count": bank.count.detach().cpu(),
                    "m2": bank.m2.detach().cpu(),
                    "usage": bank.usage.detach().cpu(),
                    "version": bank.version,
                },
                "basin_to_entity": list(session.basin_to_entity),
                "next_entity": session.next_entity,
                "entities": [
                    self._serialize_record(item) for item in session.entities.values()
                ],
                "relations": [
                    {
                        "subject_id": item.subject_id,
                        "relation": item.relation,
                        "target_id": item.target_id,
                        "confidence": item.confidence,
                        "started_at": item.started_at,
                        "ended_at": item.ended_at,
                        "version": item.version,
                    }
                    for item in session.relations
                ],
                "events": [
                    {
                        "event_type": item.event_type.value,
                        "entity_id": item.entity_id,
                        "timestamp": item.timestamp,
                        "frame_index": item.frame_index,
                        "related_entity_id": item.related_entity_id,
                        "detail": item.detail,
                    }
                    for item in session.events
                ],
            }

    @classmethod
    def from_serializable_session(
        cls, payload: dict[str, object], *, session_id: str | None = None
    ) -> WorldMemory:
        if payload.get("format_version") != 1:
            raise ValueError("unsupported world-memory format")
        memory = cls(
            key_dimension=int(payload["key_dimension"]),
            state_dimension=int(payload["state_dimension"]),
            reid_threshold=float(payload["reid_threshold"]),
            rbf_temperature=float(payload["rbf_temperature"]),
        )
        target_session = session_id or str(payload["session_id"])
        bank_payload = payload["bank"]
        if not isinstance(bank_payload, dict):
            raise ValueError("bank must be a mapping")
        bank = BasinState.from_serialized(bank_payload)
        state = _SessionState(bank=bank, index=ExactMainDeltaIndex(bank.mu))
        basin_entities = payload["basin_to_entity"]
        if not isinstance(basin_entities, list):
            raise ValueError("basin_to_entity must be a list")
        state.basin_to_entity = [str(item) for item in basin_entities]
        state.entity_to_basin = {
            entity_id: index for index, entity_id in enumerate(state.basin_to_entity)
        }
        state.next_entity = int(payload["next_entity"])
        for value in payload["entities"]:  # type: ignore[union-attr]
            record = cls._deserialize_record(value, target_session)
            state.entities[record.entity_id] = record
            for tracker_id in record.tracker_ids:
                state.tracker_to_entity[tracker_id] = record.entity_id
        state.relations = [RelationRecord(**value) for value in payload["relations"]]  # type: ignore[arg-type,union-attr]
        state.events = [
            WorldEvent(
                event_type=EventType(value["event_type"]),
                **{  # type: ignore[index]
                    key: item
                    for key, item in value.items()  # type: ignore[union-attr]
                    if key != "event_type"
                },
            )
            for value in payload["events"]  # type: ignore[union-attr]
        ]
        memory._sessions[target_session] = state
        return memory

    @staticmethod
    def _serialize_record(record: WorldRecord) -> dict[str, object]:
        return {
            "entity_id": record.entity_id,
            "class_name": record.class_name,
            "appearance_key": record.appearance_key.detach().cpu(),
            "state_value": record.state_value.detach().cpu(),
            "box_xyxy": record.box_xyxy,
            "position": record.position,
            "velocity": record.velocity,
            "zone": record.zone,
            "visibility": record.visibility.value,
            "confidence": record.confidence,
            "first_seen": record.first_seen,
            "last_seen": record.last_seen,
            "version": record.version,
            "observation_count": record.observation_count,
            "tracker_ids": sorted(record.tracker_ids),
            "metadata": dict(record.metadata),
        }

    @staticmethod
    def _deserialize_record(value: object, session_id: str) -> WorldRecord:
        if not isinstance(value, dict):
            raise ValueError("entity record must be a mapping")
        return WorldRecord(
            entity_id=str(value["entity_id"]),
            session_id=session_id,
            class_name=str(value["class_name"]),
            appearance_key=value["appearance_key"],  # type: ignore[arg-type]
            state_value=value["state_value"],  # type: ignore[arg-type]
            box_xyxy=tuple(value["box_xyxy"]),  # type: ignore[arg-type]
            position=tuple(value["position"]),  # type: ignore[arg-type]
            velocity=tuple(value["velocity"]),  # type: ignore[arg-type]
            zone=str(value["zone"]),
            visibility=VisibilityState(str(value["visibility"])),
            confidence=float(value["confidence"]),
            first_seen=float(value["first_seen"]),
            last_seen=float(value["last_seen"]),
            version=int(value["version"]),
            observation_count=int(value["observation_count"]),
            tracker_ids={int(item) for item in value["tracker_ids"]},  # type: ignore[union-attr]
            metadata=dict(value["metadata"]),  # type: ignore[arg-type]
        )

    def close_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None
