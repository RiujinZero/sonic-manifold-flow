import torch

from manifoldflow.world_simulator import (
    ScenarioKind,
    SimulatorConfig,
    TabletopSimulator,
    corrupt_tracker_ids,
)
from manifoldflow.world_types import VisibilityState


def test_car_simulation_is_deterministic_and_changes_tracker_after_occlusion() -> None:
    simulator = TabletopSimulator(SimulatorConfig(ambiguous_probability=0.0))
    first = simulator.generate(ScenarioKind.CAR_OCCLUSION, seed=44)
    second = simulator.generate(ScenarioKind.CAR_OCCLUSION, seed=44)

    first_boxes = [frame.entities[0].box_xyxy for frame in first.frames]
    second_boxes = [frame.entities[0].box_xyxy for frame in second.frames]
    cars = [frame.entities[0] for frame in first.frames]
    assert first_boxes == second_boxes
    assert torch.equal(
        cars[0].appearance_key, second.frames[0].entities[0].appearance_key
    )
    assert any(item.visibility == VisibilityState.OCCLUDED for item in cars)
    assert {item.tracker_id for item in cars} == {101, 909}
    assert len({item.entity_id for item in cars}) == 1


def test_oracle_keys_share_semantic_structure_across_episodes() -> None:
    simulator = TabletopSimulator()
    first = simulator.generate(ScenarioKind.CAR_OCCLUSION, seed=1)
    second = simulator.generate(ScenarioKind.CAR_OCCLUSION, seed=2)
    car_one, box_one = first.frames[0].entities
    car_two = second.frames[0].entities[0]

    same_class = torch.dot(car_one.appearance_key, car_two.appearance_key)
    unrelated = torch.dot(car_one.appearance_key, box_one.appearance_key)

    assert same_class > 0.9
    assert same_class > unrelated


def test_ambiguous_car_disappearance_has_no_false_occluder_relation() -> None:
    simulator = TabletopSimulator(SimulatorConfig(ambiguous_probability=1.0))
    episode = simulator.generate(ScenarioKind.CAR_OCCLUSION, seed=3)

    missing = [
        frame
        for frame in episode.frames
        if frame.entities[0].visibility == VisibilityState.MISSING
    ]
    assert missing
    assert all(frame.relations == () for frame in missing)
    assert any(question.category == "unknown" for question in episode.questions)


def test_ball_relation_follows_persistent_cup_identity_through_shuffle() -> None:
    simulator = TabletopSimulator(SimulatorConfig(ambiguous_probability=0.0))
    episode = next(
        candidate
        for seed in range(100)
        if (candidate := simulator.generate(ScenarioKind.CUP_SHUFFLE, seed=seed))
        .metadata["permutation"]
        .index(  # type: ignore[union-attr]
            ("red", "blue", "yellow").index(candidate.metadata["target_cup"])
        )
        != ("red", "blue", "yellow").index(candidate.metadata["target_cup"])
    )
    contained = [frame for frame in episode.frames if frame.relations]
    target_ids = {frame.relations[0].target_id for frame in contained}
    first_target = contained[0].entity(next(iter(target_ids)))
    last_target = contained[-1].entity(next(iter(target_ids)))

    assert len(target_ids) == 1
    assert first_target.center != last_target.center
    assert all(
        frame.entity(frame.relations[0].subject_id).visibility
        == VisibilityState.CONTAINED
        for frame in contained
    )
    visible_questions = [
        question
        for question in episode.questions
        if episode.frames[question.frame_index]
        .entity(f"{episode.episode_id}:ball")
        .visibility
        == VisibilityState.VISIBLE
        and question.evidence_entity_ids == (f"{episode.episode_id}:ball",)
    ]
    assert visible_questions
    assert all(
        question.question == "Where is the ball?" for question in visible_questions
    )


def test_tracker_corruption_preserves_entity_and_appearance() -> None:
    simulator = TabletopSimulator(SimulatorConfig(ambiguous_probability=0.0))
    episode = simulator.generate(ScenarioKind.CAR_OCCLUSION, seed=8)
    corrupted = corrupt_tracker_ids(episode, seed=2, probability=1.0)

    original = episode.frames[0].entities[0]
    changed = corrupted.frames[0].entities[0]
    assert original.entity_id == changed.entity_id
    assert original.tracker_id != changed.tracker_id
    assert torch.equal(original.appearance_key, changed.appearance_key)
