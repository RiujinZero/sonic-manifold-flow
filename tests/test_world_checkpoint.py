import torch

from manifoldflow.world_checkpoint import load_world_bundle, publish_world_bundle
from manifoldflow.world_memory import WorldMemory
from manifoldflow.world_types import ObservationEvent


def test_world_bundle_round_trip_preserves_image_latent_bank(tmp_path) -> None:
    memory = WorldMemory(key_dimension=4, state_dimension=16)
    key = torch.tensor([1.0, 0.0, 0.0, 0.0])
    memory.upsert_observation(
        ObservationEvent(
            session_id="source",
            frame_index=0,
            timestamp=1.0,
            tracker_id=7,
            class_name="toy_car",
            box_xyxy=(0.1, 0.2, 0.3, 0.4),
            confidence=0.9,
            appearance_key=key,
        )
    )
    module = torch.nn.Linear(2, 2)
    published = publish_world_bundle(
        tmp_path,
        memory=memory,
        session_id="source",
        modules={"adapter": module},
        manifest={"model": "test"},
    )

    restored_module = torch.nn.Linear(2, 2)
    restored, manifest = load_world_bundle(
        tmp_path, modules={"adapter": restored_module}, session_id="restored"
    )

    assert published.name == "world-v000000.pt"
    assert manifest == {"model": "test"}
    assert torch.equal(restored.basin_state("restored").mu, key[None, :])
    assert restored.snapshot("restored").entities[0].session_id == "restored"
    for expected, actual in zip(
        module.parameters(), restored_module.parameters(), strict=True
    ):
        assert torch.equal(expected, actual)
