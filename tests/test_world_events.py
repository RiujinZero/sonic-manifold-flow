import torch

from manifoldflow.world_events import TabletopEventEngine
from manifoldflow.world_memory import WorldMemory
from manifoldflow.world_types import EventType, ObservationEvent, VisibilityState


def _obs(
    frame: int,
    tracker: int,
    class_name: str,
    box: tuple[float, float, float, float],
    key_index: int,
) -> ObservationEvent:
    key = torch.zeros(4)
    key[key_index] = 1.0
    return ObservationEvent(
        session_id="s",
        frame_index=frame,
        timestamp=float(frame),
        tracker_id=tracker,
        class_name=class_name,
        box_xyxy=box,
        confidence=0.95,
        appearance_key=key,
    )


def test_ball_disappearance_under_cup_creates_containment_relation() -> None:
    memory = WorldMemory(key_dimension=4, state_dimension=16)
    engine = TabletopEventEngine(memory, confirm_frames=2, lost_frames=4)
    ball = _obs(0, 1, "ball", (0.4, 0.4, 0.5, 0.5), 0)
    cup = _obs(0, 2, "cup", (0.35, 0.3, 0.55, 0.6), 1)
    engine.process_frame("s", 0, 0.0, [ball, cup])
    engine.process_frame("s", 1, 1.0, [_obs(1, 2, "cup", cup.box_xyxy, 1)])
    events = engine.process_frame("s", 2, 2.0, [_obs(2, 2, "cup", cup.box_xyxy, 1)])

    snapshot = memory.snapshot("s")
    ball_record = next(item for item in snapshot.entities if item.class_name == "ball")
    assert [item.event_type for item in events] == [EventType.CONTAINED_BY]
    assert ball_record.visibility == VisibilityState.CONTAINED
    assert snapshot.relations[0].relation == "contained_by"


def test_single_frame_reappearance_blip_does_not_clear_relation() -> None:
    memory = WorldMemory(key_dimension=4, state_dimension=16)
    engine = TabletopEventEngine(memory, confirm_frames=2, lost_frames=4)
    ball = _obs(0, 1, "ball", (0.4, 0.4, 0.5, 0.5), 0)
    cup = _obs(0, 2, "cup", (0.35, 0.3, 0.55, 0.6), 1)
    engine.process_frame("s", 0, 0.0, [ball, cup])
    engine.process_frame("s", 1, 1.0, [_obs(1, 2, "cup", cup.box_xyxy, 1)])
    engine.process_frame("s", 2, 2.0, [_obs(2, 2, "cup", cup.box_xyxy, 1)])
    snapshot = memory.snapshot("s")
    assert snapshot.relations[0].active

    # A single stray detection reusing the ball's tracker id, far from the
    # cup, should not be trusted enough to discard the established relation.
    blip = _obs(3, 1, "ball", (0.9, 0.9, 0.95, 0.95), 0)
    events = engine.process_frame(
        "s", 3, 3.0, [blip, _obs(3, 2, "cup", cup.box_xyxy, 1)]
    )
    assert EventType.REAPPEARED not in [item.event_type for item in events]

    # The blip does not repeat, so the pending evidence must reset rather
    # than accumulate across a gap.
    engine.process_frame("s", 4, 4.0, [_obs(4, 2, "cup", cup.box_xyxy, 1)])

    snapshot = memory.snapshot("s")
    ball_record = next(item for item in snapshot.entities if item.class_name == "ball")
    assert ball_record.visibility == VisibilityState.CONTAINED
    assert snapshot.relations[0].active
    assert snapshot.relations[0].relation == "contained_by"


def test_sustained_reappearance_still_clears_relation() -> None:
    memory = WorldMemory(key_dimension=4, state_dimension=16)
    engine = TabletopEventEngine(memory, confirm_frames=2, lost_frames=4)
    ball = _obs(0, 1, "ball", (0.4, 0.4, 0.5, 0.5), 0)
    cup = _obs(0, 2, "cup", (0.35, 0.3, 0.55, 0.6), 1)
    engine.process_frame("s", 0, 0.0, [ball, cup])
    engine.process_frame("s", 1, 1.0, [_obs(1, 2, "cup", cup.box_xyxy, 1)])
    engine.process_frame("s", 2, 2.0, [_obs(2, 2, "cup", cup.box_xyxy, 1)])

    # Two consecutive genuine reappearance frames should still clear the
    # relation once they clear the same confirm_frames bar used for loss.
    engine.process_frame(
        "s", 3, 3.0, [_obs(3, 1, "ball", ball.box_xyxy, 0), _obs(3, 2, "cup", cup.box_xyxy, 1)]
    )
    events = engine.process_frame(
        "s", 4, 4.0, [_obs(4, 1, "ball", ball.box_xyxy, 0), _obs(4, 2, "cup", cup.box_xyxy, 1)]
    )

    assert EventType.REAPPEARED in [item.event_type for item in events]
    snapshot = memory.snapshot("s")
    assert not snapshot.relations[0].active


def test_car_disappearing_at_box_is_retained_as_occluded() -> None:
    memory = WorldMemory(key_dimension=4, state_dimension=16)
    engine = TabletopEventEngine(memory, confirm_frames=2, lost_frames=4)
    car = _obs(0, 1, "toy_car", (0.4, 0.4, 0.5, 0.5), 0)
    box = _obs(0, 2, "occluder_box", (0.35, 0.2, 0.65, 0.7), 1)
    engine.process_frame("s", 0, 0.0, [car, box])
    engine.process_frame("s", 1, 1.0, [_obs(1, 2, "occluder_box", box.box_xyxy, 1)])
    events = engine.process_frame(
        "s", 2, 2.0, [_obs(2, 2, "occluder_box", box.box_xyxy, 1)]
    )

    assert events[0].event_type == EventType.OCCLUDED
    car_record = next(
        item for item in memory.snapshot("s").entities if item.class_name == "toy_car"
    )
    assert car_record.visibility == VisibilityState.OCCLUDED


def test_timestamp_confirmation_is_stable_across_frame_rates() -> None:
    def containment_time(step: float) -> float:
        memory = WorldMemory(key_dimension=4, state_dimension=16)
        engine = TabletopEventEngine(memory, confirm_seconds=0.4)
        ball = _obs(0, 1, "ball", (0.4, 0.4, 0.5, 0.5), 0)
        cup = _obs(0, 2, "cup", (0.35, 0.3, 0.55, 0.6), 1)
        engine.process_frame("s", 0, 0.0, [ball, cup])
        index = 1
        timestamp = step
        while timestamp <= 0.8:
            current_cup = ObservationEvent(
                **{
                    **vars(_obs(index, 2, "cup", cup.box_xyxy, 1)),
                    "timestamp": timestamp,
                }
            )
            events = engine.process_frame("s", index, timestamp, [current_cup])
            for event in events:
                if event.event_type == EventType.CONTAINED_BY:
                    return event.timestamp
            index += 1
            timestamp += step
        raise AssertionError("containment was not inferred")

    slow = containment_time(0.2)
    fast = containment_time(0.1)

    assert slow >= 0.4
    assert fast >= 0.4
    assert abs(slow - fast) <= 0.1


def test_ambiguous_overlapping_cups_do_not_create_hard_relation() -> None:
    memory = WorldMemory(key_dimension=4, state_dimension=16)
    engine = TabletopEventEngine(
        memory,
        confirm_frames=2,
        lost_frames=4,
        relation_margin=0.1,
    )
    ball = _obs(0, 1, "ball", (0.47, 0.45, 0.53, 0.51), 0)
    left = _obs(0, 2, "cup", (0.35, 0.3, 0.55, 0.65), 1)
    right = _obs(0, 3, "cup", (0.45, 0.3, 0.65, 0.65), 2)
    engine.process_frame("s", 0, 0.0, [ball, left, right])
    engine.process_frame(
        "s",
        1,
        1.0,
        [
            _obs(1, 2, "cup", left.box_xyxy, 1),
            _obs(1, 3, "cup", right.box_xyxy, 2),
        ],
    )
    engine.process_frame(
        "s",
        2,
        2.0,
        [
            _obs(2, 2, "cup", left.box_xyxy, 1),
            _obs(2, 3, "cup", right.box_xyxy, 2),
        ],
    )

    snapshot = memory.snapshot("s")
    ball_record = next(item for item in snapshot.entities if item.class_name == "ball")
    assert snapshot.relations == ()
    assert ball_record.visibility == VisibilityState.MISSING
    assert len(ball_record.metadata["relation_candidates"]) == 2
