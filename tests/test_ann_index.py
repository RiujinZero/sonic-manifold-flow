import torch

from manifoldflow import AnnMainDeltaIndex, ExactIndex


def test_ann_main_delta_merges_new_rows_and_suppresses_stale_main_rows() -> None:
    centers = torch.tensor([[0.0], [10.0], [20.0]])
    index = AnnMainDeltaIndex(centers, ExactIndex(centers))

    index.upsert(torch.tensor([0]), torch.tensor([[30.0]]))
    index.upsert(torch.tensor([3]), torch.tensor([[0.25]]))

    assert index.query(torch.tensor([[0.0]]), 2).tolist() == [[3, 1]]
    assert index.query(torch.tensor([[29.0]]), 1).item() == 0
    assert index.size == 4
    assert index.delta_size == 2


def test_ann_main_delta_validates_matching_main_index() -> None:
    centers = torch.tensor([[0.0], [1.0]])

    try:
        AnnMainDeltaIndex(centers, ExactIndex(torch.tensor([[0.0, 1.0]])))
    except ValueError as error:
        assert "match" in str(error)
    else:
        raise AssertionError("mismatched index should fail")
