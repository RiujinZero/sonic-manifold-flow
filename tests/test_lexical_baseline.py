import pytest

from manifoldflow.lexical_baseline import (
    jaccard_token_similarity,
    lexical_retrieval_topk_accuracy,
)


def test_jaccard_token_similarity_identical_text_is_one() -> None:
    assert jaccard_token_similarity("the copper glider", "the copper glider") == 1.0


def test_jaccard_token_similarity_disjoint_text_is_zero() -> None:
    assert jaccard_token_similarity("copper glider", "striped lantern") == 0.0


def test_jaccard_token_similarity_partial_overlap() -> None:
    # {the, copper, glider} vs {the, copper, lantern}: intersection 2, union 4
    assert jaccard_token_similarity("the copper glider", "the copper lantern") == 0.5


def test_jaccard_token_similarity_empty_text_is_zero() -> None:
    assert jaccard_token_similarity("", "the copper glider") == 0.0


def test_lexical_retrieval_topk_accuracy_finds_high_overlap_match() -> None:
    bank = ["Neral owns the copper glider.", "Tovan placed the lantern in Vault 7."]
    queries = ["Neral owns a copper glider."]
    true_index = [0]

    accuracy = lexical_retrieval_topk_accuracy(queries, bank, true_index, k=1)

    assert accuracy == 1.0


def test_lexical_retrieval_topk_accuracy_fooled_by_higher_overlap_distractor() -> None:
    bank = [
        "neral owns the copper glider",
        "tovan placed the striped lantern in the vault",
    ]
    # Shares only "the" with the true match (bank[0], jaccard 1/9) but every
    # token with the distractor (bank[1], jaccard 5/7) - lexical overlap
    # ranks the wrong entry first, unlike a representation that captures
    # meaning rather than surface form.
    queries = ["the striped lantern in the vault"]
    true_index = [0]

    accuracy = lexical_retrieval_topk_accuracy(queries, bank, true_index, k=1)

    assert accuracy == 0.0


def test_lexical_retrieval_topk_accuracy_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        lexical_retrieval_topk_accuracy([], ["a"], [0], k=1)
    with pytest.raises(ValueError):
        lexical_retrieval_topk_accuracy(["a"], [], [0], k=1)


def test_lexical_retrieval_topk_accuracy_rejects_mismatched_true_index() -> None:
    with pytest.raises(ValueError):
        lexical_retrieval_topk_accuracy(["a", "b"], ["a", "b"], [0], k=1)


def test_lexical_retrieval_topk_accuracy_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        lexical_retrieval_topk_accuracy(["a"], ["a", "b"], [0], k=0)
    with pytest.raises(ValueError):
        lexical_retrieval_topk_accuracy(["a"], ["a", "b"], [0], k=3)
