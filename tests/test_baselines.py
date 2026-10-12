import pytest
import torch

from manifoldflow.baselines import (
    classification_accuracy,
    exact_knn_predict,
    fit_linear_probe,
    fit_nearest_centroid,
    knn_class_scores,
    linear_probe_predict,
    neighborhood_preservation,
    top_k_accuracy,
)


def test_nearest_centroid_classifies_separated_clusters() -> None:
    latent = torch.tensor([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    label = torch.tensor([0, 0, 1, 1])
    classifier = fit_nearest_centroid(latent, label)
    prediction = classifier.predict(torch.tensor([[-3.0, 0.0], [3.0, 0.0]]))
    assert prediction.tolist() == [0, 1]


def test_exact_knn_is_invariant_to_reference_chunking() -> None:
    reference = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [4.8, 0.0], [5.0, 0.0]]
    )
    label = torch.tensor([0, 0, 0, 1, 1])
    query = torch.tensor([[0.15, 0.0], [4.9, 0.0]])

    unchunked = exact_knn_predict(reference, label, query, k=3)
    chunked = exact_knn_predict(
        reference,
        label,
        query,
        k=3,
        query_chunk_size=1,
        reference_chunk_size=2,
    )
    assert unchunked.tolist() == [0, 1]
    torch.testing.assert_close(chunked, unchunked)


def test_knn_class_scores_matches_exact_knn_predict_argmax() -> None:
    reference = torch.tensor(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [4.8, 0.0], [5.0, 0.0]]
    )
    label = torch.tensor([0, 0, 0, 1, 1])
    query = torch.tensor([[0.15, 0.0], [4.9, 0.0]])

    scores = knn_class_scores(reference, label, query, k=3, classes=2)
    predicted = exact_knn_predict(reference, label, query, k=3)

    assert scores.argmax(dim=-1).tolist() == predicted.tolist()


def test_knn_class_scores_reports_vote_counts_per_class() -> None:
    reference = torch.tensor([[0.0], [0.1], [0.2], [10.0]])
    label = torch.tensor([0, 0, 1, 2])
    query = torch.tensor([[0.05]])

    scores = knn_class_scores(reference, label, query, k=3, classes=3)

    assert scores.tolist() == [[2.0, 1.0, 0.0]]


def test_top_k_accuracy_checks_target_within_top_k() -> None:
    # Row 0: target 2 scores 0.5, 2nd-highest -> in top-2, not top-1.
    # Row 1: target 0 scores 0.1, lowest -> outside top-2 too.
    scores = torch.tensor([[0.9, 0.1, 0.5], [0.1, 0.9, 0.2]])
    target = torch.tensor([2, 0])

    assert top_k_accuracy(scores, target, k=1) == 0.0
    assert top_k_accuracy(scores, target, k=2) == 0.5
    assert top_k_accuracy(scores, target, k=3) == 1.0


def test_top_k_accuracy_rejects_invalid_k() -> None:
    scores = torch.tensor([[0.9, 0.1]])
    target = torch.tensor([0])

    with pytest.raises(ValueError):
        top_k_accuracy(scores, target, k=0)
    with pytest.raises(ValueError):
        top_k_accuracy(scores, target, k=3)


def test_linear_probe_learns_separable_latents() -> None:
    negative = torch.stack((torch.linspace(-2.0, -1.0, 20), torch.zeros(20)), dim=1)
    positive = torch.stack((torch.linspace(1.0, 2.0, 20), torch.zeros(20)), dim=1)
    latent = torch.cat((negative, positive))
    label = torch.cat(
        (torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long))
    )
    result = fit_linear_probe(
        latent,
        label,
        classes=2,
        epochs=20,
        batch_size=10,
        learning_rate=0.05,
    )
    prediction = linear_probe_predict(result.model, latent)
    assert classification_accuracy(prediction, label) == 1.0
    assert result.loss[-1] < result.loss[0]


def test_neighborhood_preservation_is_one_for_identical_geometry() -> None:
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [7.0, 0.0], [12.0, 0.0]])
    overlap = neighborhood_preservation(points, points * 2.0, k=2, query_chunk_size=2)
    assert overlap == 1.0
