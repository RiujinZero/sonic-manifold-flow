"""Deterministic tabletop event inference over tracked detections."""

from __future__ import annotations

from dataclasses import dataclass, field

from manifoldflow.world_memory import WorldMemory
from manifoldflow.world_types import (
    EventType,
    ObservationEvent,
    VisibilityState,
    WorldEvent,
)


def _expanded_contains(
    outer: tuple[float, float, float, float],
    point: tuple[float, float],
    margin: float,
) -> bool:
    x1, y1, x2, y2 = outer
    width, height = x2 - x1, y2 - y1
    return (
        x1 - width * margin <= point[0] <= x2 + width * margin
        and y1 - height * margin <= point[1] <= y2 + height * margin
    )


@dataclass
class _TrackState:
    entity_id: str
    last_event: ObservationEvent
    missing_frames: int = 0
    missing_since: float | None = None
    pending_occluder: str | None = None
    relation_scores: dict[tuple[str, str], float] = field(default_factory=dict)
    visibility_decided: bool = False
    disappeared_emitted: bool = False
    reappear_frames: int = 0
    reappear_since: float | None = None


@dataclass
class TabletopEventEngine:
    memory: WorldMemory
    confirm_frames: int | None = None
    lost_frames: int | None = None
    confirm_seconds: float = 0.4
    lost_seconds: float = 2.0
    movement_threshold: float = 0.03
    containment_margin: float = 0.15
    minimum_relation_confidence: float = 0.55
    relation_margin: float = 0.12
    _tracks: dict[str, dict[int, _TrackState]] = field(default_factory=dict, init=False)
    _pending_relations: dict[tuple[str, str], int] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        if (self.confirm_frames is None) != (self.lost_frames is None):
            raise ValueError("confirm_frames and lost_frames must be set together")
        if self.confirm_frames is not None and (
            self.confirm_frames <= 0 or self.lost_frames < self.confirm_frames
        ):
            raise ValueError("lost_frames must be >= confirm_frames > 0")
        if self.confirm_seconds <= 0 or self.lost_seconds < self.confirm_seconds:
            raise ValueError("lost_seconds must be >= confirm_seconds > 0")
        if not 0.0 <= self.minimum_relation_confidence <= 1.0:
            raise ValueError("minimum_relation_confidence must be between zero and one")
        if self.relation_margin < 0:
            raise ValueError("relation_margin must be non-negative")

    def process_frame(
        self,
        session_id: str,
        frame_index: int,
        timestamp: float,
        observations: list[ObservationEvent],
    ) -> list[WorldEvent]:
        if any(item.session_id != session_id for item in observations):
            raise ValueError("all observations must belong to session_id")
        tracks = self._tracks.setdefault(session_id, {})
        seen: set[int] = set()
        emitted: list[WorldEvent] = []

        for observation in observations:
            seen.add(observation.tracker_id)
            previous = tracks.get(observation.tracker_id)
            prior_entity = self.memory.entity_for_tracker(
                session_id, observation.tracker_id
            ) or self.memory.match_identity(
                session_id,
                observation.class_name,
                observation.appearance_key,
                timestamp=observation.timestamp,
                position=observation.center,
            )
            prior_visibility = None
            if prior_entity is not None:
                prior_record = self.memory.snapshot(session_id).entity(prior_entity)
                prior_visibility = (
                    None if prior_record is None else prior_record.visibility
                )
            reappearing = prior_entity is not None and prior_visibility in {
                VisibilityState.OCCLUDED,
                VisibilityState.CONTAINED,
                VisibilityState.MISSING,
            }
            if reappearing and previous is not None and previous.visibility_decided:
                # A single noisy detection should not discard an already
                # decided relation. Require the same sustained-evidence bar
                # used to confirm loss before trusting a reappearance too;
                # otherwise treat this observation as a likely false positive.
                # Skip committing it to memory at all, so it cannot disturb
                # the established relation, visibility, or the entity's
                # tracked position/appearance statistics either.
                previous.reappear_frames += 1
                previous.reappear_since = previous.reappear_since or timestamp
                if not self._reappearance_confirmed(previous, timestamp):
                    continue
            record, created = self.memory.upsert_observation(observation)
            if created:
                event_type = EventType.APPEARED
            elif reappearing:
                event_type = EventType.REAPPEARED
                self.memory.clear_relation(session_id, record.entity_id, timestamp)
                self.memory.set_metadata(
                    session_id, record.entity_id, "relation_candidates", []
                )
            else:
                dx = record.position[0] - (
                    previous.last_event.center[0] if previous else record.position[0]
                )
                dy = record.position[1] - (
                    previous.last_event.center[1] if previous else record.position[1]
                )
                event_type = (
                    EventType.MOVED
                    if dx * dx + dy * dy >= self.movement_threshold**2
                    else None
                )
            for stale_tracker in [
                item
                for item, track in tracks.items()
                if item != observation.tracker_id
                and track.entity_id == record.entity_id
            ]:
                del tracks[stale_tracker]
                seen.discard(stale_tracker)
            tracks[observation.tracker_id] = _TrackState(record.entity_id, observation)
            if event_type is not None:
                emitted.append(
                    WorldEvent(event_type, record.entity_id, timestamp, frame_index)
                )

        visible_by_class: dict[str, list[tuple[str, ObservationEvent]]] = {}
        for tracker_id in seen:
            state = tracks[tracker_id]
            visible_by_class.setdefault(state.last_event.class_name, []).append(
                (state.entity_id, state.last_event)
            )

        for tracker_id, state in list(tracks.items()):
            if tracker_id in seen:
                continue
            state.missing_frames += 1
            state.missing_since = state.missing_since or timestamp
            state.reappear_frames = 0
            state.reappear_since = None
            for relation_name, target_id, confidence in self._occlusion_candidates(
                state, visible_by_class
            ):
                key = (relation_name, target_id)
                state.relation_scores[key] = max(
                    confidence, state.relation_scores.get(key, 0.0)
                )
            if not state.visibility_decided and self._confirmation_due(
                state, timestamp
            ):
                hypotheses = self._relation_hypotheses(state)
                if hypotheses:
                    self.memory.set_metadata(
                        session_id,
                        state.entity_id,
                        "relation_candidates",
                        hypotheses,
                    )
                relation = self._select_relation(state)
                if relation is not None:
                    relation_name, target_id, confidence = relation
                    visibility = (
                        VisibilityState.CONTAINED
                        if relation_name == "contained_by"
                        else VisibilityState.OCCLUDED
                    )
                    self.memory.set_visibility(session_id, state.entity_id, visibility)
                    self.memory.set_relation(
                        session_id,
                        state.entity_id,
                        relation_name,
                        target_id,
                        confidence,
                        timestamp,
                    )
                    state.pending_occluder = target_id
                    event_type = (
                        EventType.CONTAINED_BY
                        if relation_name == "contained_by"
                        else EventType.OCCLUDED
                    )
                    emitted.append(
                        WorldEvent(
                            event_type,
                            state.entity_id,
                            timestamp,
                            frame_index,
                            target_id,
                        )
                    )
                else:
                    self.memory.set_visibility(
                        session_id, state.entity_id, VisibilityState.MISSING
                    )
                state.visibility_decided = True
            if self._loss_due(state, timestamp) and not state.disappeared_emitted:
                state.disappeared_emitted = True
                if state.pending_occluder is not None:
                    continue
                emitted.append(
                    WorldEvent(
                        EventType.DISAPPEARED,
                        state.entity_id,
                        timestamp,
                        frame_index,
                    )
                )

        for event in emitted:
            self.memory.append_event(session_id, event)
        return emitted

    def _confirmation_due(self, state: _TrackState, timestamp: float) -> bool:
        if self.confirm_frames is not None:
            return state.missing_frames >= self.confirm_frames
        return timestamp - state.last_event.timestamp >= self.confirm_seconds

    def _reappearance_confirmed(self, state: _TrackState, timestamp: float) -> bool:
        if self.confirm_frames is not None:
            return state.reappear_frames >= self.confirm_frames
        since = state.reappear_since
        return since is not None and timestamp - since >= self.confirm_seconds

    def _loss_due(self, state: _TrackState, timestamp: float) -> bool:
        if self.lost_frames is not None:
            return state.missing_frames >= self.lost_frames
        return timestamp - state.last_event.timestamp >= self.lost_seconds

    def _occlusion_candidates(
        self,
        state: _TrackState,
        visible: dict[str, list[tuple[str, ObservationEvent]]],
    ) -> list[tuple[str, str, float]]:
        candidates: list[tuple[str, str, float]] = []
        if state.last_event.class_name == "ball":
            for entity_id, cup in visible.get("cup", []):
                score = self._cover_score(
                    state.last_event, cup, self.containment_margin
                )
                if score is not None:
                    candidates.append(("contained_by", entity_id, score))
        if state.last_event.class_name == "toy_car":
            for entity_id, box in visible.get("occluder_box", []):
                score = self._cover_score(state.last_event, box, 0.35)
                if score is not None:
                    candidates.append(("occluded_by", entity_id, score))
        return candidates

    def _select_relation(self, state: _TrackState) -> tuple[str, str, float] | None:
        ranked = sorted(
            (
                (confidence, relation_name, target_id)
                for (relation_name, target_id), confidence in (
                    state.relation_scores.items()
                )
            ),
            reverse=True,
        )
        if not ranked or ranked[0][0] < self.minimum_relation_confidence:
            return None
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < self.relation_margin:
            return None
        confidence, relation_name, target_id = ranked[0]
        return relation_name, target_id, confidence

    @staticmethod
    def _relation_hypotheses(state: _TrackState) -> list[dict[str, object]]:
        return [
            {
                "relation": relation_name,
                "target": target_id,
                "confidence": confidence,
            }
            for (relation_name, target_id), confidence in sorted(
                state.relation_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

    @staticmethod
    def _cover_score(
        subject: ObservationEvent,
        cover: ObservationEvent,
        margin: float,
    ) -> float | None:
        if not _expanded_contains(cover.box_xyxy, subject.center, margin):
            return None
        x1, y1, x2, y2 = cover.box_xyxy
        center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        half_width = max((x2 - x1) * (0.5 + margin), 1e-6)
        half_height = max((y2 - y1) * (0.5 + margin), 1e-6)
        offset = min(
            1.0,
            ((subject.center[0] - center[0]) / half_width) ** 2
            + ((subject.center[1] - center[1]) / half_height) ** 2,
        )
        geometry = 1.0 - 0.5 * offset
        return min(
            1.0,
            0.45 * subject.confidence + 0.35 * cover.confidence + 0.20 * geometry,
        )

    def close_session(self, session_id: str) -> None:
        self._tracks.pop(session_id, None)
        for key in [item for item in self._pending_relations if item[0] == session_id]:
            del self._pending_relations[key]
        self.memory.close_session(session_id)
