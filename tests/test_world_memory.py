import torch

from manifoldflow.world_memory import WorldMemory
from manifoldflow.world_types import ObservationEvent, VisibilityState


def _key(index: int, dimension: int = 4) -> torch.Tensor:
    value = torch.zeros(dimension)
    value[index] = 1.0
    return value


def _observation(
    session: str,
    tracker: int,
    key: torch.Tensor,
    *,
    class_name: str = "toy_car",
    frame: int = 0,
) -> ObservationEvent:
    return ObservationEvent(
        session_id=session,
        frame_index=frame,
        timestamp=float(frame),
        tracker_id=tracker,
        class_name=class_name,
        box_xyxy=(0.1, 0.1, 0.3, 0.3),
        confidence=0.9,
        appearance_key=key,
        zone="upper_left",
    )


def test_world_memory_uses_existing_basin_state_for_image_latents() -> None:
    memory = WorldMemory(key_dimension=4, state_dimension=16)
    first, created = memory.upsert_observation(_observation("a", 1, _key(0)))
    second, reinforced = memory.upsert_observation(
        _observation("a", 1, torch.tensor([0.99, 0.01, 0.0, 0.0]), frame=1)
    )

    bank = memory.basin_state("a")
    assert created is True
    assert reinforced is False
    assert first.entity_id == second.entity_id
    assert bank is not None
    assert bank.mu.shape == (1, 4)
    assert bank.count.tolist() == [2]


def test_new_tracker_reidentifies_against_latent_basin() -> None:
    memory = WorldMemory(key_dimension=4, state_dimension=16, reid_threshold=0.1)
    first, _ = memory.upsert_observation(_observation("a", 1, _key(0)))
    memory.set_visibility("a", first.entity_id, VisibilityState.OCCLUDED)
    returned, created = memory.upsert_observation(
        _observation("a", 99, torch.tensor([0.999, 0.001, 0.0, 0.0]), frame=5)
    )

    assert created is False
    assert returned.entity_id == first.entity_id
    assert returned.tracker_ids == {1, 99}


def test_ambiguous_reidentification_creates_new_entity() -> None:
    memory = WorldMemory(
        key_dimension=4,
        state_dimension=16,
        reid_threshold=0.7,
        reid_margin=0.1,
    )
    first, _ = memory.upsert_observation(_observation("a", 1, _key(0)))
    second, _ = memory.upsert_observation(_observation("a", 2, _key(1)))
    memory.set_visibility("a", first.entity_id, VisibilityState.OCCLUDED)
    memory.set_visibility("a", second.entity_id, VisibilityState.OCCLUDED)

    returned, created = memory.upsert_observation(
        _observation("a", 99, torch.tensor([0.72, 0.69, 0.0, 0.0]), frame=5)
    )

    assert created is True
    assert returned.entity_id not in {first.entity_id, second.entity_id}


def test_simultaneous_similar_objects_do_not_merge() -> None:
    memory = WorldMemory(key_dimension=4, state_dimension=16, reid_threshold=0.5)
    first, _ = memory.upsert_observation(_observation("a", 1, _key(0)))
    second, created = memory.upsert_observation(
        _observation("a", 2, torch.tensor([0.999, 0.001, 0.0, 0.0]))
    )

    assert created is True
    assert second.entity_id != first.entity_id


def test_sessions_are_structurally_isolated() -> None:
    memory = WorldMemory(key_dimension=4, state_dimension=16)
    memory.upsert_observation(_observation("a", 1, _key(0)))
    memory.upsert_observation(_observation("b", 1, _key(1)))

    hits = memory.query("a", _key(1), top_k=4)

    assert len(hits) == 1
    assert hits[0].record.session_id == "a"
    assert memory.snapshot("b").entities[0].session_id == "b"


def test_query_expands_active_relations() -> None:
    memory = WorldMemory(key_dimension=4, state_dimension=16)
    ball, _ = memory.upsert_observation(
        _observation("a", 1, _key(0), class_name="ball")
    )
    cup, _ = memory.upsert_observation(_observation("a", 2, _key(1), class_name="cup"))
    memory.set_relation("a", ball.entity_id, "contained_by", cup.entity_id, 0.9, 1.0)

    hits = memory.query("a", _key(0), top_k=1)

    assert hits[0].entity_id == ball.entity_id
    assert [item.entity_id for item in hits[0].related] == [cup.entity_id]


def test_relation_state_features_follow_active_graph() -> None:
    memory = WorldMemory(key_dimension=4, state_dimension=256)
    ball, _ = memory.upsert_observation(
        _observation("a", 1, _key(0), class_name="ball")
    )
    cup, _ = memory.upsert_observation(_observation("a", 2, _key(1), class_name="cup"))

    memory.set_relation("a", ball.entity_id, "contained_by", cup.entity_id, 0.9, 1.0)
    snapshot = memory.snapshot("a")
    assert snapshot.entity(ball.entity_id).state_value[17] == 0.9
    assert snapshot.entity(cup.entity_id).state_value[21] == 0.9
    assert snapshot.entity(ball.entity_id).state_value[12] == 0.25

    memory.clear_relation("a", ball.entity_id, 2.0)
    snapshot = memory.snapshot("a")
    assert snapshot.entity(ball.entity_id).state_value[17] == 0
    assert snapshot.entity(cup.entity_id).state_value[21] == 0
    assert snapshot.entity(ball.entity_id).state_value[12] == 0


def test_close_session_erases_latents_and_records() -> None:
    memory = WorldMemory(key_dimension=4, state_dimension=16)
    memory.upsert_observation(_observation("a", 1, _key(0)))

    assert memory.close_session("a") is True
    assert memory.snapshot("a").entities == ()
    assert memory.basin_state("a") is None
