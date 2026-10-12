import torch

from manifoldflow.experiments.world_export_yolox import export_checkpoint


def test_export_yolox_keeps_only_model_tensors(tmp_path) -> None:
    source = tmp_path / "training.pt"
    destination = tmp_path / "runtime.pt"
    torch.save(
        {
            "model": {"layer.weight": torch.ones(2, 2)},
            "optimizer": {"unsafe_for_runtime": True},
        },
        source,
    )

    export_checkpoint(source, destination)

    result = torch.load(destination, weights_only=True)
    assert list(result) == ["layer.weight"]
