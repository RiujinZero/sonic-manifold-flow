import torch

from manifoldflow import (
    BasinState,
    ContinualStreamSpec,
    class_conditioned_shift,
    evaluate_continual_latent_stream,
)


def test_class_conditioned_shift_is_deterministic_and_normalized() -> None:
    latent = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    label = torch.tensor([0, 1])
    directions = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    first = class_conditioned_shift(
        latent,
        label,
        directions,
        shift_scale=0.2,
        noise_std=0.01,
        generator=torch.Generator().manual_seed(4),
    )
    second = class_conditioned_shift(
        latent,
        label,
        directions,
        shift_scale=0.2,
        noise_std=0.01,
        generator=torch.Generator().manual_seed(4),
    )

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.norm(dim=-1), torch.ones(2))


def test_continual_stream_reports_adaptation_for_all_controls() -> None:
    train_latent = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [-1.0, 0.0],
            [-0.9, -0.1],
            [-0.8, -0.2],
        ]
    )
    train_latent = torch.nn.functional.normalize(train_latent, dim=-1)
    train_label = torch.tensor([0, 0, 0, 1, 1, 1])
    test_latent = train_latent[[0, 1, 3, 4]]
    test_label = train_label[[0, 1, 3, 4]]
    bank = BasinState.from_statistics(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
        torch.tensor([3, 3]),
        torch.zeros(2, 2),
    )
    prototype_bank = BasinState.from_statistics(
        torch.tensor([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [-0.8, -0.2]]),
        torch.tensor([2, 1, 2, 1]),
        torch.zeros(4, 2),
    )

    result = evaluate_continual_latent_stream(
        bank,
        torch.tensor([[3, 0], [0, 3]]),
        prototype_bank,
        torch.tensor([[2, 0], [1, 0], [0, 2], [0, 1]]),
        train_latent,
        train_label,
        test_latent,
        test_label,
        spec=ContinualStreamSpec(
            classes_per_task=1,
            stream_samples_per_class=2,
            shift_scale=0.1,
            latent_noise_std=0.0,
            novelty_quantile=0.8,
            novelty_persistence=2,
            novelty_support_multiplier=1.0,
            basin_growth_budget=2,
            seed=3,
        ),
    )

    assert result["tasks"] == [[0], [1]]
    methods = result["methods"]
    assert set(methods) == {
        "frozen",
        "local_fixed",
        "local_novelty",
        "prototype_frozen",
        "prototype_online",
        "rbf_frozen",
        "rbf_online",
        "centroid_frozen",
        "centroid_online",
    }
    for method in methods.values():
        assert len(method["stages"]) == 3
        assert len(method["task_updates"]) == 2
        assert 0 <= method["summary"]["final_shifted_accuracy"] <= 1
    assert methods["frozen"]["summary"]["final_delta_size"] == 0
    assert methods["local_fixed"]["summary"]["final_delta_size"] > 0
    assert (
        methods["frozen"]["summary"]["stored_vector_values"]
        == methods["prototype_online"]["summary"]["stored_vector_values"]
    )
    assert methods["centroid_online"]["summary"]["stored_vector_values"] == 4
