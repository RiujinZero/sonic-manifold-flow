import torch

from manifoldflow.world_events import TabletopEventEngine
from manifoldflow.world_memory import WorldMemory
from manifoldflow.world_types import ObservationEvent


def _observation(tracker_id: int, box, frame: int = 0) -> ObservationEvent:
    key = torch.zeros(128)
    key[0] = 1.0
    return ObservationEvent(
        session_id="real-video",
        frame_index=frame,
        timestamp=float(frame),
        tracker_id=tracker_id,
        class_name="cup",
        box_xyxy=box,
        confidence=0.9,
        appearance_key=key,
        zone="left",
    )


def test_new_tracker_identity_merge_removes_stale_tracker_from_seen_set() -> None:
    memory = WorldMemory()
    engine = TabletopEventEngine(memory)

    engine.process_frame(
        "real-video",
        0,
        0.0,
        [_observation(1, (0.1, 0.2, 0.2, 0.5))],
    )
    engine.process_frame(
        "real-video",
        1,
        1.0,
        [_observation(2, (0.11, 0.2, 0.21, 0.5), frame=1)],
    )

    assert len(memory.snapshot("real-video").entities) == 1
    assert memory.entity_for_tracker("real-video", 2) is not None
