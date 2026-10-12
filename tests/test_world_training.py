import torch

from manifoldflow.world_training import (
    answer_mismatched_indices,
    combine_world_losses,
    masked_candidate_retrieval_loss,
    memory_use_contrast,
    retrieval_info_nce,
    should_enable_lora,
)


def test_answer_mismatched_indices_avoid_same_answer_when_possible() -> None:
    answers = ("left", "left", "right", "unknown")
    indices = answer_mismatched_indices(answers)

    assert indices.shape == (4,)
    assert all(answers[row] != answers[index] for row, index in enumerate(indices))


def test_retrieval_info_nce_prefers_matching_latents() -> None:
    matching = torch.eye(3)
    wrong = torch.flip(torch.eye(3), dims=(0,))
    assert retrieval_info_nce(matching, matching) < retrieval_info_nce(matching, wrong)


def test_masked_candidate_retrieval_supports_multiple_positives() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor(
        [
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        ]
    )
    candidate_mask = torch.tensor([[True, True, True], [True, True, False]])
    evidence = torch.tensor([[True, True, False], [False, True, False]])

    matched = masked_candidate_retrieval_loss(
        query, candidates, candidate_mask, evidence
    )
    mismatched = masked_candidate_retrieval_loss(
        query.flip(1), candidates, candidate_mask, evidence
    )

    assert matched < mismatched


def test_masked_candidate_retrieval_ignores_unknown_rows() -> None:
    query = torch.tensor([[1.0, 0.0]], requires_grad=True)
    loss = masked_candidate_retrieval_loss(
        query,
        torch.tensor([[[1.0, 0.0]]]),
        torch.tensor([[True]]),
        torch.tensor([[False]]),
    )

    assert loss.item() == 0.0
    loss.backward()
    assert query.grad is not None


def test_memory_contrast_penalizes_ignoring_correct_memory() -> None:
    assert memory_use_contrast(torch.tensor(1.0), torch.tensor(1.1)).item() > 0
    assert memory_use_contrast(torch.tensor(0.2), torch.tensor(1.2)).item() == 0


def test_combined_loss_and_lora_gate() -> None:
    loss = combine_world_losses(torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0))
    assert torch.isclose(loss.total, torch.tensor(1.8))
    assert should_enable_lora(0.96, 0.80) is True
    assert should_enable_lora(0.90, 0.80) is False
