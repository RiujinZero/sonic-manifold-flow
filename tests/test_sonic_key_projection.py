import pytest
import torch
import torch.nn.functional as F

from manifoldflow.sonic_key_projection import (
    SonicKeyProjection,
    SonicKeyProjectionConfig,
    info_nce_retrieval_loss,
)


def test_identity_at_init() -> None:
    # The whole point of identity init: a freshly constructed projection
    # must reproduce the untrained baseline exactly, not regress it before
    # training has done anything.
    projection = SonicKeyProjection(SonicKeyProjectionConfig(dim=8))
    pooled = torch.randn(4, 8)

    output = projection(pooled)

    assert torch.allclose(output, F.normalize(pooled, dim=-1), atol=1e-6)


def test_output_shape_and_is_normalized() -> None:
    projection = SonicKeyProjection(SonicKeyProjectionConfig(dim=8))
    pooled = torch.randn(3, 8)

    output = projection(pooled)

    assert output.shape == (3, 8)
    assert torch.allclose(output.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_rejects_wrong_shape() -> None:
    projection = SonicKeyProjection(SonicKeyProjectionConfig(dim=8))
    with pytest.raises(ValueError):
        projection(torch.randn(1, 3, 8))
    with pytest.raises(ValueError):
        projection(torch.randn(4, 5))


def test_config_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError):
        SonicKeyProjectionConfig(dim=0)


def test_gradients_reach_projection_weight() -> None:
    projection = SonicKeyProjection(SonicKeyProjectionConfig(dim=8))
    pooled = torch.randn(5, 8)
    query_key = projection(pooled[0:1]).squeeze(0)
    candidate_keys = projection(pooled)

    loss = info_nce_retrieval_loss(query_key, candidate_keys, 0, temperature=0.1)
    loss.backward()

    assert projection.projection.weight.grad is not None
    assert projection.projection.weight.grad.abs().sum().item() > 0


def test_info_nce_loss_is_low_when_correct_candidate_dominates() -> None:
    query_key = F.normalize(torch.tensor([1.0, 0.0, 0.0]), dim=-1)
    candidate_keys = F.normalize(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), dim=-1
    )

    loss = info_nce_retrieval_loss(query_key, candidate_keys, 0, temperature=0.1)

    assert loss.item() < 0.01


def test_info_nce_loss_is_high_when_correct_candidate_is_worst() -> None:
    query_key = F.normalize(torch.tensor([1.0, 0.0, 0.0]), dim=-1)
    candidate_keys = F.normalize(
        torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]), dim=-1
    )

    # correct_index=0 names the worst-scoring candidate (anti-parallel to
    # the query) as correct - the loss should reflect a bad prediction.
    loss = info_nce_retrieval_loss(query_key, candidate_keys, 0, temperature=0.1)

    assert loss.item() > 2.0


def test_info_nce_rejects_non_positive_temperature() -> None:
    query_key = F.normalize(torch.randn(4), dim=-1)
    candidate_keys = F.normalize(torch.randn(3, 4), dim=-1)
    with pytest.raises(ValueError):
        info_nce_retrieval_loss(query_key, candidate_keys, 0, temperature=0.0)


def test_info_nce_rejects_out_of_range_correct_index() -> None:
    query_key = F.normalize(torch.randn(4), dim=-1)
    candidate_keys = F.normalize(torch.randn(3, 4), dim=-1)
    with pytest.raises(ValueError):
        info_nce_retrieval_loss(query_key, candidate_keys, 3, temperature=0.1)
    with pytest.raises(ValueError):
        info_nce_retrieval_loss(query_key, candidate_keys, -1, temperature=0.1)
