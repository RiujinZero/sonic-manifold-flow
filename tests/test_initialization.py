import torch

from manifoldflow.initialization import (
    initialize_basin_bank,
    mini_batch_kmeans,
    nearest_center_assignment,
)


def _separated_latents() -> tuple[torch.Tensor, torch.Tensor]:
    left = torch.tensor([[-3.0, -0.1], [-2.8, 0.0], [-2.6, 0.1], [-2.4, 0.0]])
    right = -left
    latent = torch.cat((left, right))
    label = torch.cat(
        (torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long))
    )
    return latent, label


def test_nearest_assignment_is_invariant_to_both_chunk_sizes() -> None:
    latent = torch.tensor([[0.1, 0.0], [4.9, 0.0], [9.0, 0.0]])
    centers = torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    full = nearest_center_assignment(latent, centers)
    chunked = nearest_center_assignment(
        latent, centers, query_chunk_size=1, center_chunk_size=1
    )
    assert full.ids.tolist() == [0, 1, 2]
    torch.testing.assert_close(chunked.ids, full.ids)
    torch.testing.assert_close(chunked.squared_distance, full.squared_distance)


def test_mini_batch_kmeans_is_seed_deterministic() -> None:
    latent, _ = _separated_latents()
    first = mini_batch_kmeans(latent, clusters=2, batch_size=4, steps=10, seed=9)
    second = mini_batch_kmeans(latent, clusters=2, batch_size=4, steps=10, seed=9)
    torch.testing.assert_close(first.centers, second.centers)
    torch.testing.assert_close(first.update_count, second.update_count)


def test_initialized_bank_has_exact_statistics_and_values() -> None:
    latent, label = _separated_latents()
    bank = initialize_basin_bank(
        latent,
        label,
        basins=2,
        classes=2,
        mini_batch_size=4,
        mini_batch_steps=20,
        refinement_steps=3,
        seed=5,
    )

    assert bank.active_basins == 2
    assert bank.state.count.sum().item() == latent.shape[0]
    assert bank.class_count.sum().item() == latent.shape[0]
    for basin_id in range(bank.active_basins):
        members = latent[bank.assignment == basin_id]
        torch.testing.assert_close(bank.state.mu[basin_id], members.mean(dim=0))
        torch.testing.assert_close(
            bank.state.m2[basin_id],
            (members - members.mean(dim=0)).square().sum(dim=0),
        )
    prediction = bank.predict_nearest_value(latent)
    assert torch.equal(prediction, label)
    assert bank.inertia >= 0
    assert bank.variance_scale > 0


def test_empty_duplicate_centers_are_pruned() -> None:
    latent = torch.tensor([[0.0, 0.0]] * 4 + [[1.0, 1.0]] * 4)
    label = torch.tensor([0] * 4 + [1] * 4)
    bank = initialize_basin_bank(
        latent,
        label,
        basins=4,
        classes=2,
        mini_batch_size=8,
        mini_batch_steps=3,
        refinement_steps=2,
    )
    assert bank.active_basins == 2
    assert torch.all(bank.state.count > 0)
    assert bank.serializable()["requested_basins"] == 4
