import pytest
import torch

from manifoldflow.representation import (
    FixedSeededProjection,
    layer_index_for_fraction,
    pool_hidden_states,
    retrieval_topk_accuracy,
)


def test_mean_pooling_averages_only_content_tokens() -> None:
    hidden = torch.zeros(1, 3, 2)
    hidden[0, 0] = torch.tensor([2.0, 4.0])
    hidden[0, 1] = torch.tensor([4.0, 8.0])
    hidden[0, 2] = torch.tensor([100.0, 100.0])  # padding, must be excluded
    mask = torch.tensor([[1, 1, 0]])

    pooled = pool_hidden_states(hidden, mask, pooling="mean")

    assert torch.allclose(pooled, torch.tensor([[3.0, 6.0]]))


def test_last_token_pooling_selects_final_content_token_per_row() -> None:
    hidden = torch.zeros(2, 3, 2)
    hidden[0, 1] = torch.tensor([9.0, 9.0])  # row 0 has 2 content tokens
    hidden[1, 2] = torch.tensor([5.0, 5.0])  # row 1 has 3 content tokens
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    pooled = pool_hidden_states(hidden, mask, pooling="last_token")

    assert torch.allclose(pooled, torch.tensor([[9.0, 9.0], [5.0, 5.0]]))


def test_pool_hidden_states_rejects_unknown_pooling_rule() -> None:
    hidden = torch.zeros(1, 2, 2)
    mask = torch.ones(1, 2)
    with pytest.raises(ValueError):
        pool_hidden_states(hidden, mask, pooling="max")


def test_pool_hidden_states_rejects_mismatched_mask_shape() -> None:
    hidden = torch.zeros(1, 2, 2)
    mask = torch.ones(1, 3)
    with pytest.raises(ValueError):
        pool_hidden_states(hidden, mask, pooling="mean")


def test_fixed_seeded_projection_is_deterministic_across_instances() -> None:
    first = FixedSeededProjection(16, 4, seed=17)
    second = FixedSeededProjection(16, 4, seed=17)

    assert torch.equal(first.projection, second.projection)


def test_fixed_seeded_projection_differs_across_seeds() -> None:
    first = FixedSeededProjection(16, 4, seed=1)
    second = FixedSeededProjection(16, 4, seed=2)

    assert not torch.equal(first.projection, second.projection)


def test_fixed_seeded_projection_output_is_l2_normalized() -> None:
    projection = FixedSeededProjection(16, 4, seed=0)
    features = torch.randn(5, 16)

    output = projection(features)

    assert output.shape == (5, 4)
    assert torch.allclose(output.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_fixed_seeded_projection_rejects_wrong_input_width() -> None:
    projection = FixedSeededProjection(16, 4, seed=0)
    with pytest.raises(ValueError):
        projection(torch.randn(2, 8))


def test_fixed_seeded_projection_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError):
        FixedSeededProjection(0, 4, seed=0)
    with pytest.raises(ValueError):
        FixedSeededProjection(16, 0, seed=0)


@pytest.mark.parametrize(
    ("num_layers", "fraction", "expected"),
    [
        (32, 1.0, 31),
        (32, 0.75, 23),
        (32, 0.5, 15),
        (1, 1.0, 0),
    ],
)
def test_layer_index_for_fraction(
    num_layers: int, fraction: float, expected: int
) -> None:
    assert layer_index_for_fraction(num_layers, fraction) == expected


def test_layer_index_for_fraction_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        layer_index_for_fraction(0, 0.5)
    with pytest.raises(ValueError):
        layer_index_for_fraction(32, 0.0)
    with pytest.raises(ValueError):
        layer_index_for_fraction(32, 1.5)


def test_retrieval_topk_accuracy_finds_exact_matches_at_k1() -> None:
    bank = torch.eye(4)
    query = torch.eye(4)
    true_index = torch.arange(4)

    accuracy = retrieval_topk_accuracy(query, bank, true_index, k=1)

    assert accuracy == 1.0


def test_retrieval_topk_accuracy_counts_partial_hits() -> None:
    bank = torch.eye(3)
    # query 0 is nearest to bank[0] (correct); query 1 is nearest to bank[0]
    # but its true match is bank[1], so only k=1 for query 0 hits.
    query = torch.tensor([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]])
    true_index = torch.tensor([0, 1])

    assert retrieval_topk_accuracy(query, bank, true_index, k=1) == 0.5
    assert retrieval_topk_accuracy(query, bank, true_index, k=2) == 1.0


def test_retrieval_topk_accuracy_rejects_mismatched_shapes() -> None:
    bank = torch.eye(4)
    query = torch.eye(4)
    with pytest.raises(ValueError):
        retrieval_topk_accuracy(query, bank, torch.arange(3), k=1)
    with pytest.raises(ValueError):
        retrieval_topk_accuracy(query, torch.eye(3), torch.arange(4), k=1)


def test_retrieval_topk_accuracy_rejects_invalid_k() -> None:
    bank = torch.eye(4)
    query = torch.eye(4)
    true_index = torch.arange(4)
    with pytest.raises(ValueError):
        retrieval_topk_accuracy(query, bank, true_index, k=0)
    with pytest.raises(ValueError):
        retrieval_topk_accuracy(query, bank, true_index, k=5)
