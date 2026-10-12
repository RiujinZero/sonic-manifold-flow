import pytest
import torch

from manifoldflow.world_conversation import ConversationMemory


def _key(index: int, dimension: int = 4) -> torch.Tensor:
    value = torch.zeros(dimension)
    value[index] = 1.0
    return value


def _value(seed: float, dimension: int = 6) -> torch.Tensor:
    return torch.full((dimension,), seed)


def test_recall_returns_the_semantically_closest_utterance() -> None:
    memory = ConversationMemory(key_dimension=4, value_dimension=6)
    memory.remember(
        "s", "My favorite color is blue.", _key(0), _value(1.0), timestamp=0.0
    )
    memory.remember("s", "I have a dog named Rex.", _key(2), _value(2.0), timestamp=1.0)

    hits = memory.recall("s", _key(0), top_k=1)

    assert len(hits) == 1
    assert hits[0].utterance.text == "My favorite color is blue."
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    assert torch.equal(hits[0].utterance.value, _value(1.0))


def test_recall_ranks_multiple_hits_by_similarity() -> None:
    memory = ConversationMemory(key_dimension=4, value_dimension=6)
    memory.remember(
        "s", "My favorite color is blue.", _key(0), _value(1.0), timestamp=0.0
    )
    memory.remember(
        "s",
        "My favorite color is teal.",
        torch.tensor([0.9, 0.1, 0.0, 0.0]),
        _value(2.0),
        timestamp=1.0,
    )
    memory.remember("s", "I have a dog named Rex.", _key(2), _value(3.0), timestamp=2.0)

    hits = memory.recall("s", _key(0), top_k=2)

    assert [hit.utterance.text for hit in hits] == [
        "My favorite color is blue.",
        "My favorite color is teal.",
    ]
    assert hits[0].score >= hits[1].score


def test_recall_is_scoped_per_session() -> None:
    memory = ConversationMemory(key_dimension=4, value_dimension=6)
    memory.remember("a", "Session a fact.", _key(0), _value(1.0), timestamp=0.0)

    assert memory.recall("b", _key(0), top_k=1) == []


def test_recall_with_no_history_returns_empty() -> None:
    memory = ConversationMemory(key_dimension=4, value_dimension=6)
    assert memory.recall("empty-session", _key(0), top_k=1) == []


def test_history_preserves_insertion_order() -> None:
    memory = ConversationMemory(key_dimension=4, value_dimension=6)
    memory.remember("s", "first", _key(0), _value(1.0), timestamp=0.0)
    memory.remember("s", "second", _key(1), _value(2.0), timestamp=1.0)
    memory.remember("s", "third", _key(2), _value(3.0), timestamp=2.0)

    assert [item.text for item in memory.history("s")] == ["first", "second", "third"]


def test_close_session_clears_stored_utterances() -> None:
    memory = ConversationMemory(key_dimension=4, value_dimension=6)
    memory.remember("s", "will be forgotten", _key(0), _value(1.0), timestamp=0.0)

    memory.close_session("s")

    assert memory.history("s") == ()
    assert memory.recall("s", _key(0), top_k=1) == []


def test_empty_text_is_rejected() -> None:
    memory = ConversationMemory(key_dimension=4, value_dimension=6)
    with pytest.raises(ValueError, match="non-empty"):
        memory.remember("s", "   ", _key(0), _value(1.0), timestamp=0.0)


def test_wrong_key_dimension_is_rejected() -> None:
    memory = ConversationMemory(key_dimension=4, value_dimension=6)
    with pytest.raises(ValueError, match="shape"):
        memory.remember("s", "text", torch.zeros(8), _value(1.0), timestamp=0.0)


def test_wrong_value_dimension_is_rejected() -> None:
    memory = ConversationMemory(key_dimension=4, value_dimension=6)
    with pytest.raises(ValueError, match="shape"):
        memory.remember("s", "text", _key(0), torch.zeros(3), timestamp=0.0)
