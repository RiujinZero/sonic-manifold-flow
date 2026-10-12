from manifoldflow.world_demo import DemoRuntime
from manifoldflow.world_types import VisibilityState


def test_curated_car_scene_preserves_identity_after_tracker_change() -> None:
    # car-occlusion is now a recorded episode (episode-000870, extracted by
    # scripts/extract_demo_scenario.py) rather than hand-authored frames --
    # occluded frames 9-13, tracker changes 101 -> 909 on reappearance at
    # frame 14 (real values from that episode's own manifest, not the old
    # hand-picked 10/99).
    runtime = DemoRuntime()
    session = runtime.new_session()
    for frame in range(15):
        result = runtime.render_scenario(session, "car-occlusion", frame)

    cars = [
        item
        for item in runtime.memory.snapshot(session).entities
        if item.class_name == "toy_car"
    ]
    answer = runtime.answer(session, "Where is the car?")
    assert len(cars) == 1
    assert cars[0].tracker_ids == {101, 909}
    assert cars[0].visibility == VisibilityState.VISIBLE
    assert "center" in answer["answer"]
    assert any(event["type"] == "reappeared" for event in result["events"])


def test_curated_cup_scene_keeps_ball_relation_with_yellow_cup() -> None:
    # cup-shuffle is now a recorded episode (episode-000171) whose ball is
    # contained by the yellow cup at frame 5 -- real value, not the old
    # hand-authored red-cup case.
    runtime = DemoRuntime()
    session = runtime.new_session()
    for frame in range(8):
        runtime.render_scenario(session, "cup-shuffle", frame)

    answer = runtime.answer(session, "Which cup contains the ball?")
    snapshot = runtime.memory.snapshot(session)
    relation = next(item for item in snapshot.relations if item.active)
    cup = snapshot.entity(relation.target_id)
    assert cup is not None
    assert cup.metadata["label"] == "yellow"
    assert "yellow" in answer["answer"]


def test_rewinding_replays_from_a_clean_world() -> None:
    runtime = DemoRuntime()
    session = runtime.new_session()
    runtime.render_scenario(session, "car-occlusion", 9)
    runtime.render_scenario(session, "car-occlusion", 0)

    snapshot = runtime.memory.snapshot(session)
    cars = [item for item in snapshot.entities if item.class_name == "toy_car"]
    assert len(cars) == 1
    assert cars[0].tracker_ids == {101}
