"""Deterministic symbolic tabletop worlds for oracle and rendered training."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import StrEnum

import torch
from torch import Tensor

from manifoldflow.world_types import VisibilityState, concept_appearance_key, zone_for_box


class ScenarioKind(StrEnum):
    CAR_OCCLUSION = "car_occlusion"
    CUP_SHUFFLE = "cup_shuffle"


@dataclass(frozen=True)
class SimEntity:
    entity_id: str
    tracker_id: int
    class_name: str
    box_xyxy: tuple[float, float, float, float]
    color: tuple[int, int, int]
    visibility: VisibilityState = VisibilityState.VISIBLE
    zone: str = "unknown"
    label: str = ""
    velocity: tuple[float, float] = (0.0, 0.0)
    appearance_key: Tensor | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def visible(self) -> bool:
        return self.visibility == VisibilityState.VISIBLE


@dataclass(frozen=True)
class SimRelation:
    subject_id: str
    relation: str
    target_id: str
    confidence: float = 1.0


@dataclass(frozen=True)
class SimFrame:
    frame_index: int
    timestamp: float
    entities: tuple[SimEntity, ...]
    relations: tuple[SimRelation, ...] = ()
    events: tuple[str, ...] = ()

    @property
    def visible_entities(self) -> tuple[SimEntity, ...]:
        return tuple(item for item in self.entities if item.visible)

    def entity(self, entity_id: str) -> SimEntity:
        return next(item for item in self.entities if item.entity_id == entity_id)


@dataclass(frozen=True)
class SimQuestion:
    frame_index: int
    question: str
    answer: str
    evidence_entity_ids: tuple[str, ...]
    category: str


@dataclass(frozen=True)
class SimEpisode:
    episode_id: str
    scenario: ScenarioKind
    seed: int
    frames: tuple[SimFrame, ...]
    questions: tuple[SimQuestion, ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class SimulatorConfig:
    key_dimension: int = 128
    frames_per_episode: int = 24
    fps: float = 5.0
    ambiguous_probability: float = 0.1

    def __post_init__(self) -> None:
        if self.key_dimension <= 0 or self.frames_per_episode < 16 or self.fps <= 0:
            raise ValueError("invalid simulator dimensions or timing")
        if not 0.0 <= self.ambiguous_probability <= 1.0:
            raise ValueError("ambiguous_probability must be between 0 and 1")


class TabletopSimulator:
    """Produce exact world trajectories without depending on a physics engine."""

    def __init__(self, config: SimulatorConfig | None = None) -> None:
        self.config = config or SimulatorConfig()

    def generate(
        self, scenario: ScenarioKind | str, *, seed: int, episode_id: str | None = None
    ) -> SimEpisode:
        kind = ScenarioKind(scenario)
        identity = episode_id or f"{kind.value}-{seed:08d}"
        if kind == ScenarioKind.CAR_OCCLUSION:
            return self._car_episode(seed, identity)
        return self._cup_episode(seed, identity)

    def _car_episode(self, seed: int, episode_id: str) -> SimEpisode:
        rng = random.Random(seed)
        count = self.config.frames_per_episode
        direction = rng.choice((-1, 1))
        box_center = rng.uniform(0.43, 0.57)
        box_width = rng.uniform(0.18, 0.24)
        box_height = rng.uniform(0.34, 0.44)
        box_y = rng.uniform(0.34, 0.42)
        start_x, end_x = (0.05, 0.84) if direction > 0 else (0.84, 0.05)
        car_width = rng.uniform(0.10, 0.14)
        car_height = rng.uniform(0.10, 0.14)
        car_y = box_y + box_height * rng.uniform(0.47, 0.58)
        ambiguous = rng.random() < self.config.ambiguous_probability
        car_key = self._appearance_key(seed, "car")
        box_key = self._appearance_key(seed, "box")
        car_color = rng.choice(((222, 67, 61), (39, 113, 205), (236, 166, 38)))
        box_color = rng.choice(((121, 86, 55), (87, 101, 112), (145, 106, 72)))
        frames: list[SimFrame] = []
        was_hidden = False
        reappeared = False
        for frame_index in range(count):
            progress = frame_index / (count - 1)
            x = start_x + (end_x - start_x) * self._smoothstep(progress)
            car_box = (x, car_y, x + car_width, car_y + car_height)
            box_box = (
                box_center - box_width / 2,
                box_y,
                box_center + box_width / 2,
                box_y + box_height,
            )
            car_center = (car_box[0] + car_box[2]) / 2
            behind_box = box_box[0] <= car_center <= box_box[2]
            if ambiguous:
                hidden = count // 3 <= frame_index < count // 2
                relation = ()
            else:
                hidden = behind_box
                relation = (
                    (
                        SimRelation(
                            f"{episode_id}:car", "occluded_by", f"{episode_id}:box"
                        ),
                    )
                    if hidden
                    else ()
                )
            events: list[str] = []
            if frame_index == 0:
                events.extend(("car:appeared", "box:appeared"))
            if hidden and not was_hidden:
                events.append("car:occluded" if not ambiguous else "car:missing")
            if not hidden and was_hidden:
                events.append("car:reappeared")
                reappeared = True
            tracker_id = 101 if not reappeared else 909
            velocity = ((end_x - start_x) / (count - 1), 0.0)
            car = SimEntity(
                f"{episode_id}:car",
                tracker_id,
                "toy_car",
                car_box,
                car_color,
                VisibilityState.MISSING
                if ambiguous and hidden
                else (VisibilityState.OCCLUDED if hidden else VisibilityState.VISIBLE),
                zone_for_box(car_box),
                velocity=velocity,
                appearance_key=car_key,
            )
            box = SimEntity(
                f"{episode_id}:box",
                202,
                "occluder_box",
                box_box,
                box_color,
                zone=zone_for_box(box_box),
                appearance_key=box_key,
            )
            frames.append(
                SimFrame(
                    frame_index,
                    frame_index / self.config.fps,
                    (car, box),
                    relation,
                    tuple(events),
                )
            )
            was_hidden = hidden
        return SimEpisode(
            episode_id,
            ScenarioKind.CAR_OCCLUSION,
            seed,
            tuple(frames),
            self._car_questions(tuple(frames), ambiguous),
            {"direction": direction, "ambiguous": ambiguous},
        )

    def _cup_episode(self, seed: int, episode_id: str) -> SimEpisode:
        rng = random.Random(seed)
        count = self.config.frames_per_episode
        colors = (
            ("red", (210, 57, 54)),
            ("blue", (44, 111, 202)),
            ("yellow", (231, 184, 50)),
        )
        slots = (0.22, 0.50, 0.78)
        target_index = rng.randrange(3)
        permutation = list(range(3))
        while permutation == [0, 1, 2]:
            rng.shuffle(permutation)
        ambiguous = rng.random() < self.config.ambiguous_probability
        cover_frame = 5
        shuffle_start = 9
        shuffle_end = count - 3
        entity_ids = [f"{episode_id}:cup:{name}" for name, _ in colors]
        keys = [self._appearance_key(seed, f"cup:{name}") for name, _ in colors]
        ball_key = self._appearance_key(seed, "ball")
        frames: list[SimFrame] = []
        for frame_index in range(count):
            cup_entities: list[SimEntity] = []
            for cup_index, ((label, color), entity_id, key) in enumerate(
                zip(colors, entity_ids, keys, strict=True)
            ):
                destination_slot = permutation.index(cup_index)
                if frame_index < shuffle_start:
                    center_x = slots[cup_index]
                elif frame_index >= shuffle_end:
                    center_x = slots[destination_slot]
                else:
                    progress = (frame_index - shuffle_start) / (
                        shuffle_end - shuffle_start
                    )
                    center_x = slots[cup_index] + (
                        slots[destination_slot] - slots[cup_index]
                    ) * self._smoothstep(progress)
                covering = frame_index >= cover_frame
                top = 0.43 if covering else 0.31
                cup_box = (center_x - 0.085, top, center_x + 0.085, top + 0.27)
                cup_entities.append(
                    SimEntity(
                        entity_id,
                        300 + cup_index,
                        "cup",
                        cup_box,
                        color,
                        zone=zone_for_box(cup_box),
                        label=label,
                        velocity=(0.0, 0.0),
                        appearance_key=key,
                    )
                )
            target_cup = cup_entities[target_index]
            ball_center = (
                target_cup.center
                if frame_index >= cover_frame
                else (
                    slots[target_index],
                    0.65,
                )
            )
            ball_box = (
                ball_center[0] - 0.035,
                ball_center[1] - 0.035,
                ball_center[0] + 0.035,
                ball_center[1] + 0.035,
            )
            contained = frame_index >= cover_frame and not ambiguous
            visibility = (
                VisibilityState.CONTAINED
                if contained
                else VisibilityState.MISSING
                if frame_index >= cover_frame
                else VisibilityState.VISIBLE
            )
            ball = SimEntity(
                f"{episode_id}:ball",
                404,
                "ball",
                ball_box,
                (240, 219, 68),
                visibility,
                target_cup.zone,
                appearance_key=ball_key,
            )
            relations = (
                (SimRelation(ball.entity_id, "contained_by", target_cup.entity_id),)
                if contained
                else ()
            )
            events: list[str] = []
            if frame_index == 0:
                events.extend(("ball:appeared", "cups:appeared"))
            if frame_index == cover_frame:
                events.append("ball:contained" if contained else "ball:missing")
            if frame_index == shuffle_start:
                events.append("cups:shuffle_started")
            if frame_index == shuffle_end:
                events.append("cups:shuffle_finished")
            frames.append(
                SimFrame(
                    frame_index,
                    frame_index / self.config.fps,
                    (ball, *cup_entities),
                    relations,
                    tuple(events),
                )
            )
        return SimEpisode(
            episode_id,
            ScenarioKind.CUP_SHUFFLE,
            seed,
            tuple(frames),
            self._cup_questions(tuple(frames), target_index, colors, ambiguous),
            {
                "target_cup": colors[target_index][0],
                "permutation": permutation,
                "ambiguous": ambiguous,
            },
        )

    def _car_questions(
        self, frames: tuple[SimFrame, ...], ambiguous: bool
    ) -> tuple[SimQuestion, ...]:
        questions: list[SimQuestion] = []
        for frame in frames:
            car = next(item for item in frame.entities if item.class_name == "toy_car")
            box = next(
                item for item in frame.entities if item.class_name == "occluder_box"
            )
            if car.visibility == VisibilityState.VISIBLE:
                answer = f"The car is visible in the {car.zone.replace('_', ' ')} zone."
                evidence = (car.entity_id,)
            elif car.visibility == VisibilityState.OCCLUDED:
                answer = "The car is hidden behind the box."
                evidence = (car.entity_id, box.entity_id)
            else:
                answer = "The car's current location is unknown."
                evidence = (car.entity_id,)
            questions.append(
                SimQuestion(
                    frame.frame_index, "Where is the car?", answer, evidence, "location"
                )
            )
            if frame.frame_index in {0, len(frames) // 2, len(frames) - 1}:
                questions.append(
                    SimQuestion(
                        frame.frame_index,
                        "Is the car visible?",
                        "Yes." if car.visible else "No.",
                        (car.entity_id,),
                        "visibility",
                    )
                )
        if ambiguous:
            middle = len(frames) // 2
            middle_car = next(
                item for item in frames[middle].entities if item.class_name == "toy_car"
            )
            questions.append(
                SimQuestion(
                    middle,
                    "What is hiding the car?",
                    "There is not enough evidence to determine that.",
                    (middle_car.entity_id,),
                    "unknown",
                )
            )
        return tuple(questions)

    def _cup_questions(
        self,
        frames: tuple[SimFrame, ...],
        target_index: int,
        colors: tuple[tuple[str, tuple[int, int, int]], ...],
        ambiguous: bool,
    ) -> tuple[SimQuestion, ...]:
        questions: list[SimQuestion] = []
        target_label = colors[target_index][0]
        for frame in frames:
            ball = next(item for item in frame.entities if item.class_name == "ball")
            target = next(item for item in frame.entities if item.label == target_label)
            if ball.visibility == VisibilityState.VISIBLE:
                question = "Where is the ball?"
                answer = f"The ball is visible in the {ball.zone.replace('_', ' ')} zone."
                evidence = (ball.entity_id,)
                category = "location"
            elif ambiguous:
                question = "Which cup contains the ball?"
                answer = "There is not enough evidence to identify a cup."
                evidence = (ball.entity_id,)
                category = "unknown"
            else:
                question = "Which cup contains the ball?"
                answer = f"The ball is under the {target_label} cup."
                evidence = (ball.entity_id, target.entity_id)
                category = "containment"
            questions.append(
                SimQuestion(
                    frame.frame_index,
                    question,
                    answer,
                    evidence,
                    category,
                )
            )
            if frame.frame_index in {0, len(frames) - 1}:
                questions.append(
                    SimQuestion(
                        frame.frame_index,
                        f"Where is the {target_label} cup?",
                        f"The {target_label} cup is in the {target.zone.replace('_', ' ')} zone.",
                        (target.entity_id,),
                        "location",
                    )
                )
        return tuple(questions)

    def _appearance_key(self, seed: int, identity: str) -> Tensor:
        class_name, _, attribute = identity.partition(":")
        return concept_appearance_key(
            class_name,
            attribute,
            instance=f"{seed}:{identity}",
            dimension=self.config.key_dimension,
        )

    @staticmethod
    def _smoothstep(value: float) -> float:
        return value * value * (3.0 - 2.0 * value)



def corrupt_tracker_ids(
    episode: SimEpisode, seed: int, probability: float
) -> SimEpisode:
    """Create a retrieval-training copy with deterministic tracker-ID resets."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    rng = random.Random(seed)
    replacements: dict[tuple[int, str], int] = {}
    frames: list[SimFrame] = []
    for frame in episode.frames:
        entities: list[SimEntity] = []
        for entity in frame.entities:
            tracker_id = entity.tracker_id
            if entity.visible and rng.random() < probability:
                key = (frame.frame_index, entity.entity_id)
                replacements[key] = rng.randint(10_000, 99_999)
                tracker_id = replacements[key]
            entities.append(replace(entity, tracker_id=tracker_id))
        frames.append(replace(frame, entities=tuple(entities)))
    return replace(episode, frames=tuple(frames))
