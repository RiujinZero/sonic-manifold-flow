import torch

from manifoldflow import BasinState, ExactIndex, exact_bank_flow, positive_rate


def test_exact_index_returns_nearest_centers_in_order() -> None:
    centers = torch.tensor([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]])
    index = ExactIndex(centers)
    ids = index.query(torch.tensor([[1.8, 0.0]]), k=2)
    assert ids.tolist() == [[1, 0]]


def test_welford_update_is_local_and_exact() -> None:
    bank = BasinState.from_centers(torch.tensor([[0.0, 0.0], [10.0, 10.0]]))
    observations = torch.tensor([[2.0, 4.0], [4.0, 8.0]])
    bank.update_winners(observations, torch.tensor([0, 0]))

    torch.testing.assert_close(bank.mu[0], torch.tensor([3.0, 6.0]))
    torch.testing.assert_close(bank.m2[0], torch.tensor([2.0, 8.0]))
    assert bank.count.tolist() == [2, 0]
    assert bank.usage.tolist() == [2, 0]
    torch.testing.assert_close(bank.mu[1], torch.tensor([10.0, 10.0]))
    assert torch.all(positive_rate(bank.raw_rate) > 0)


def test_from_statistics_derives_rates_and_preserves_counts() -> None:
    centers = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    count = torch.tensor([2, 1])
    m2 = torch.tensor([[2.0, 8.0], [0.0, 0.0]])
    bank = BasinState.from_statistics(
        centers,
        count,
        m2,
        epsilon=1e-4,
        singleton_rate=1.0,
    )

    expected_first = 1.0 / (m2[0] + 1e-4)
    torch.testing.assert_close(positive_rate(bank.raw_rate)[0], expected_first)
    torch.testing.assert_close(positive_rate(bank.raw_rate)[1], torch.ones(2))
    assert bank.count.tolist() == [2, 1]
    assert bank.usage.tolist() == [2, 1]


def test_serialized_bank_roundtrip_restores_core_state() -> None:
    original = BasinState.from_centers(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    payload = {
        "mu": original.mu,
        "raw_rate": original.raw_rate,
        "bias": original.bias,
        "count": original.count,
        "m2": original.m2,
        "usage": original.usage,
        "version": 7,
        "ignored_value_state": torch.ones(2),
    }
    restored = BasinState.from_serialized(payload)

    assert restored.version == 7
    torch.testing.assert_close(restored.mu, original.mu)
    torch.testing.assert_close(restored.raw_rate, original.raw_rate)


def test_gather_moves_only_selected_rows() -> None:
    bank = BasinState.from_centers(torch.arange(12, dtype=torch.float32).reshape(4, 3))
    selected = bank.gather(torch.tensor([[3, 1]]))
    assert selected["mu"].shape == (1, 2, 3)
    torch.testing.assert_close(selected["mu"][0, 0], bank.mu[3])
    torch.testing.assert_close(selected["mu"][0, 1], bank.mu[1])


def test_append_statistics_adds_populated_basin() -> None:
    bank = BasinState.from_centers(torch.tensor([[0.0, 0.0]]))

    new_id = bank.append_statistics(
        torch.tensor([4.0, 6.0]),
        3,
        torch.tensor([2.0, 8.0]),
    )

    assert new_id == 1
    assert bank.count.tolist() == [0, 3]
    assert bank.usage.tolist() == [0, 3]
    torch.testing.assert_close(bank.mu[1], torch.tensor([4.0, 6.0]))


def test_exact_bank_flow_reranks_and_returns_local_ids() -> None:
    bank = BasinState.from_centers(torch.tensor([[0.0, 0.0], [2.0, 0.0], [5.0, 0.0]]))
    result = exact_bank_flow(
        torch.tensor([[1.8, 0.0]]), bank, candidates=3, top_k=2, t=0.5
    )

    assert result.basin_ids.tolist() == [[1, 0]]
    assert result.latent.shape == (1, 2)
    torch.testing.assert_close(result.weight.sum(-1), torch.ones(1))
