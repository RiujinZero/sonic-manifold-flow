"""Deterministic scenarios and runtime used by the public review surface."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from manifoldflow.world_answer import Answerer, EvidenceAnswerer
from manifoldflow.world_events import TabletopEventEngine
from manifoldflow.world_memory import WorldMemory
from manifoldflow.world_types import (
    ObservationEvent,
    WorldRecord,
    concept_appearance_key,
    zone_for_box,
)

# TabletopSimulator's own key generation uses these shorter identity
# strings ("car"/"box"), not the WorldRecord.class_name values
# ("toy_car"/"occluder_box") used everywhere else -- see _observation's
# docstring below.
_IDENTITY_CLASS = {"toy_car": "car", "occluder_box": "box"}

_SCENARIO_ASSETS = Path(__file__).with_name("static") / "scenarios"


@dataclass(frozen=True)
class SceneObject:
    tracker_id: int
    class_name: str
    box: tuple[float, float, float, float]
    zone: str
    label: str = ""
    # Only set for scenarios loaded from a recorded episode
    # (_load_recorded_scenario) -- the exact key TabletopSimulator's own
    # generation stored, byte-for-byte, rather than an approximation
    # reconstructed from class+label alone (see _observation).
    appearance_key: tuple[float, ...] | None = None


@dataclass(frozen=True)
class CuratedScenario:
    scenario_id: str
    title: str
    description: str
    frames: tuple[tuple[SceneObject, ...], ...]
    # Real rendered frame per index, from a recorded episode; empty for
    # hand-authored scenarios, which fall back to a flat canvas fill.
    images: tuple[str | None, ...] = ()


def _scene_object(
    tracker_id: int,
    class_name: str,
    box: tuple[float, float, float, float],
    label: str = "",
) -> SceneObject:
    """zone is derived from the box, not hand-typed, so curated scenarios
    can never drift out of sync with the 3x3 grid live/synthetic sources
    compute (zone_for_box in world_types.py)."""
    return SceneObject(tracker_id, class_name, box, zone_for_box(box), label)


def _car_scenario() -> CuratedScenario:
    box = _scene_object(20, "occluder_box", (0.42, 0.25, 0.62, 0.78))
    frames: list[tuple[SceneObject, ...]] = []
    for x in (0.12, 0.24, 0.36):
        frames.append((_scene_object(10, "toy_car", (x, 0.51, x + 0.12, 0.65)), box))
    frames.extend([(box,), (box,), (box,), (box,)])
    for x in (0.58, 0.70, 0.80):
        frames.append((_scene_object(99, "toy_car", (x, 0.51, x + 0.12, 0.65)), box))
    return CuratedScenario(
        "car-occlusion",
        "Object permanence",
        "A toy car passes behind a box and reappears with a new tracker ID.",
        tuple(frames),
    )


def _cup_scenario() -> CuratedScenario:
    ball = _scene_object(30, "ball", (0.18, 0.58, 0.24, 0.66))
    red = _scene_object(40, "cup", (0.13, 0.38, 0.29, 0.70), "red")
    blue = _scene_object(41, "cup", (0.42, 0.38, 0.58, 0.70), "blue")
    yellow = _scene_object(42, "cup", (0.71, 0.38, 0.87, 0.70), "yellow")
    frames = [
        (ball, red, blue, yellow),
        (ball, red, blue, yellow),
        (red, blue, yellow),
        (red, blue, yellow),
        (red, blue, yellow),
        (
            _scene_object(40, "cup", (0.28, 0.38, 0.44, 0.70), "red"),
            _scene_object(41, "cup", (0.13, 0.38, 0.29, 0.70), "blue"),
            yellow,
        ),
        (
            _scene_object(40, "cup", (0.61, 0.38, 0.77, 0.70), "red"),
            _scene_object(41, "cup", (0.13, 0.38, 0.29, 0.70), "blue"),
            _scene_object(42, "cup", (0.42, 0.38, 0.58, 0.70), "yellow"),
        ),
        (
            _scene_object(40, "cup", (0.71, 0.38, 0.87, 0.70), "red"),
            _scene_object(41, "cup", (0.13, 0.38, 0.29, 0.70), "blue"),
            _scene_object(42, "cup", (0.42, 0.38, 0.58, 0.70), "yellow"),
        ),
    ]
    return CuratedScenario(
        "cup-shuffle",
        "Persistent relationships",
        "A ball is covered by the red cup before the cups are shuffled.",
        tuple(frames),
    )


def _load_recorded_scenario(
    scenario_id: str, title: str, description: str
) -> CuratedScenario | None:
    """Real rendered frames extracted from the synthetic training corpus
    (scripts/extract_demo_scenario.py) instead of the flat-color canvas
    the hand-authored scenarios use -- a reviewer can actually see the
    scene, not just its box overlays. Entities carry the exact
    appearance_key TabletopSimulator generated for them, not an
    approximation, so retrieval sees precisely the same identity vectors
    training did. Falls through to the hand-authored version (returns
    None) if the asset bundle isn't present, so the demo still runs
    without it."""
    asset_dir = _SCENARIO_ASSETS / scenario_id
    manifest_path = asset_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames: list[tuple[SceneObject, ...]] = []
    images: list[str | None] = []
    for frame in manifest["frames"]:
        visible = [
            SceneObject(
                tracker_id=entity["tracker_id"],
                class_name=entity["class_name"],
                box=tuple(entity["box_xyxy"]),
                zone=entity["zone"],
                label=entity["label"],
                appearance_key=tuple(entity["appearance_key"]),
            )
            for entity in frame["entities"]
            if entity["visibility"] == "visible"
        ]
        frames.append(tuple(visible))
        images.append(f"/static/scenarios/{scenario_id}/images/{frame['image']}")
    return CuratedScenario(scenario_id, title, description, tuple(frames), tuple(images))


SCENARIOS = {
    item.scenario_id: item
    for item in (
        _load_recorded_scenario(
            "car-occlusion",
            "Object permanence",
            "A toy car passes behind a box and reappears with a new tracker ID.",
        )
        or _car_scenario(),
        _load_recorded_scenario(
            "cup-shuffle",
            "Persistent relationships",
            "A ball is covered by the red cup before the cups are shuffled.",
        )
        or _cup_scenario(),
    )
}


class DemoRuntime:
    def __init__(self, answerer: Answerer | None = None) -> None:
        self.memory = WorldMemory(key_dimension=128)
        # confirm_seconds's 0.4s default requires two missed frames at the
        # 5fps (0.2s/frame) cadence both curated scenarios and live camera
        # use before committing an occlusion -- a deliberate debounce
        # against a single noisy YOLOX miss on live camera, but pure,
        # unnecessary lag for curated scenarios (a missing observation
        # there is never a detection glitch, it is definitively real,
        # since the frames are recorded, not detected). Reduced below one
        # frame interval so a single missing frame commits immediately;
        # this also makes live camera's state track what the detector is
        # actually reporting more closely instead of masking its
        # instability behind a delay, which given live detection is
        # already caveated as unreliable is a reasonable trade to make.
        self.engine = TabletopEventEngine(self.memory, confirm_seconds=0.15)
        self.answerer = answerer or EvidenceAnswerer(self.memory)
        self._scenario_sessions: dict[str, tuple[str, int]] = {}

    def new_session(self) -> str:
        session_id = secrets.token_urlsafe(24)
        seed_session = getattr(self.answerer, "seed_session", None)
        if seed_session is not None:
            seed_session(session_id)
        return session_id

    def scenario_catalog(self) -> list[dict[str, Any]]:
        return [
            {
                "id": item.scenario_id,
                "title": item.title,
                "description": item.description,
                "frames": len(item.frames),
            }
            for item in SCENARIOS.values()
        ]

    def render_scenario(
        self, session_id: str, scenario_id: str, frame: int
    ) -> dict[str, Any]:
        scenario = SCENARIOS[scenario_id]
        target = max(0, min(frame, len(scenario.frames) - 1))
        state = self._scenario_sessions.get(session_id)
        if state is None or state[0] != scenario_id or target < state[1]:
            self.engine.close_session(session_id)
            start = 0
        else:
            start = state[1] + 1
        events = []
        for index in range(start, target + 1):
            observations = [
                self._observation(session_id, index, item)
                for item in scenario.frames[index]
            ]
            events.extend(
                self.engine.process_frame(
                    session_id, index, float(index) / 5.0, observations
                )
            )
            self._apply_labels(session_id, scenario.frames[index])
        self._scenario_sessions[session_id] = (scenario_id, target)
        snapshot = self.memory.snapshot(session_id)
        image = scenario.images[target] if target < len(scenario.images) else None
        return {
            "scenario": scenario_id,
            "frame": target,
            "frame_count": len(scenario.frames),
            "image": image,
            "entities": [self._record_json(item) for item in snapshot.entities],
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

    def answer(self, session_id: str, question: str) -> dict[str, Any]:
        return self.answerer.answer(session_id, question)

    def close_session(self, session_id: str) -> None:
        self._scenario_sessions.pop(session_id, None)
        self.engine.close_session(session_id)
        close_answerer_session = getattr(self.answerer, "close_session", None)
        if close_answerer_session is not None:
            close_answerer_session(session_id)

    def _observation(
        self, session_id: str, frame: int, item: SceneObject
    ) -> ObservationEvent:
        # concept_appearance_key, not an arbitrary one-hot placeholder: this
        # is the actual identity-key space CrossModalKeyProjector and
        # GemmaQueryProjector were trained to route into (see the
        # 2026-09-02 dev journal entry -- a one-hot key here made every
        # curated-scenario retrieval score ~1e-6 regardless of which
        # oracle checkpoint was deployed, since it shared no structure
        # with what retrieval was ever trained against).
        #
        # _IDENTITY_CLASS maps WorldRecord.class_name to the shorter
        # identity string TabletopSimulator's own key generation actually
        # used ("car"/"box", not "toy_car"/"occluder_box") -- cup/ball
        # already match class_name verbatim, so only car and box need
        # translating. class_name itself is left alone everywhere else
        # (WorldStateEncoder's one-hot, event logic, etc. all correctly
        # expect the full "toy_car"/"occluder_box" form); this mapping is
        # scoped to key generation only.
        #
        # Recorded scenarios (_load_recorded_scenario) carry the exact
        # key TabletopSimulator generated for that entity -- use it
        # verbatim rather than reconstructing an approximation.
        key = (
            torch.tensor(item.appearance_key)
            if item.appearance_key is not None
            else concept_appearance_key(
                _IDENTITY_CLASS.get(item.class_name, item.class_name), item.label
            )
        )
        return ObservationEvent(
            session_id=session_id,
            frame_index=frame,
            timestamp=float(frame) / 5.0,
            tracker_id=item.tracker_id,
            class_name=item.class_name,
            box_xyxy=item.box,
            confidence=0.96,
            appearance_key=key,
            zone=item.zone,
        )

    def _apply_labels(self, session_id: str, objects: tuple[SceneObject, ...]) -> None:
        snapshot = self.memory.snapshot(session_id)
        for item in objects:
            if not item.label:
                continue
            entity_id = self.memory.entity_for_tracker(session_id, item.tracker_id)
            record = next(
                (value for value in snapshot.entities if value.entity_id == entity_id),
                None,
            )
            if record is not None:
                # Metadata is display-only and does not change the latent basin.
                self.memory.set_metadata(
                    session_id, record.entity_id, "label", item.label
                )

    @staticmethod
    def _record_json(item: WorldRecord) -> dict[str, Any]:
        return {
            "id": item.entity_id,
            "class": item.class_name,
            "box": item.box_xyxy,
            "zone": item.zone,
            "visibility": item.visibility.value,
            "confidence": item.confidence,
            "version": item.version,
            "label": item.metadata.get("label", ""),
        }
