import torch

from manifoldflow import OnlineClassCentroid, OnlinePrototypeMemory


def test_online_prototype_updates_only_winner_and_supports_rbf() -> None:
    memory = OnlinePrototypeMemory(
        centers=torch.tensor([[0.0], [10.0]]),
        count=torch.tensor([2, 2]),
        class_count=torch.tensor([[2, 0], [0, 2]]),
    )

    winners = memory.update(torch.tensor([[2.0]]), torch.tensor([0]))

    assert winners.tolist() == [0]
    torch.testing.assert_close(memory.centers, torch.tensor([[2.0 / 3.0], [10.0]]))
    assert memory.class_count.tolist() == [[3, 0], [0, 2]]
    assert memory.index.delta_size == 1
    assert memory.predict_nearest(torch.tensor([[0.5], [9.0]])).tolist() == [0, 1]
    assert (
        memory.predict_rbf(torch.tensor([[9.0]]), top_k=2, temperature=0.1).item() == 1
    )


def test_online_class_centroid_updates_only_labeled_class() -> None:
    latent = torch.tensor([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
    label = torch.tensor([0, 0, 1, 1])
    memory = OnlineClassCentroid.from_data(latent, label, classes=2)
    untouched = memory.centers[1].clone()

    memory.update(torch.tensor([[4.0, 0.0]]), torch.tensor([0]))

    torch.testing.assert_close(memory.centers[0], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(memory.centers[1], untouched)
    assert memory.count.tolist() == [3, 2]
    assert memory.touched == {0}
    assert memory.predict(torch.tensor([[2.1, 0.0], [11.0, 0.0]])).tolist() == [0, 1]
