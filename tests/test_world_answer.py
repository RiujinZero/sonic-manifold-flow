from types import SimpleNamespace

import torch
from torch import nn

from manifoldflow.world_answer import (
    ConversationalAnswerer,
    EvidenceAnswerer,
    GemmaLatentAnswerer,
)
from manifoldflow.world_conversation import ConversationMemory
from manifoldflow.world_types import VisibilityState


class _Encoder:
    def encode_texts(self, texts):
        return torch.ones(len(texts), 2)


class _Memory:
    def __init__(self, hits):
        self.hits = hits

    def query(self, session_id, routes, top_k):
        assert session_id == "demo"
        assert routes.shape == (1, 4)
        assert top_k == 2
        return self.hits


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))


class _CapturingAnswerer(GemmaLatentAnswerer):
    def _rank_answer(self, question, values, keys, memory_mask, weights, confidence):
        self.captured = (values, keys, memory_mask, weights, confidence)
        return "The ball is visible in the left zone."


def test_gemma_answerer_sends_identity_keys_to_ranker() -> None:
    record = SimpleNamespace(
        state_value=torch.arange(256, dtype=torch.float32),
        appearance_key=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        confidence=0.8,
    )
    hit = SimpleNamespace(record=record, score=0.75, entity_id="entity-1")
    projector = nn.Linear(2, 4, bias=False)
    answerer = _CapturingAnswerer(
        _Memory([hit]),
        _Encoder(),
        projector,
        _Model(),
        tokenizer=None,
        max_memory=2,
    )

    result = answerer.answer("demo", "Where is the ball?")

    _, keys, mask, weights, _ = answerer.captured
    assert torch.equal(keys[0], record.appearance_key)
    assert not keys[1].any()
    assert mask.tolist() == [[True, False]]
    assert weights.tolist() == [[1.0, 0.0]]
    assert result["mode"] == "gemma-ranked"
    assert result["evidence"] == ["entity-1"]


def test_gemma_answerer_drops_irrelevant_hits_below_relevance_floor() -> None:
    record = SimpleNamespace(
        state_value=torch.arange(256, dtype=torch.float32),
        appearance_key=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        confidence=0.8,
    )
    # A distant nearest-neighbor match, e.g. the only entity in a scene
    # unrelated to the question ("Hello"). WorldMemory.query always returns
    # its closest match regardless of distance; the answerer must not treat
    # this as usable evidence.
    hit = SimpleNamespace(record=record, score=0.05, entity_id="entity-1")
    projector = nn.Linear(2, 4, bias=False)
    answerer = _CapturingAnswerer(
        _Memory([hit]),
        _Encoder(),
        projector,
        _Model(),
        tokenizer=None,
        max_memory=2,
        min_relevance_score=0.15,
    )

    result = answerer.answer("demo", "Hello")

    assert not hasattr(answerer, "captured")
    assert result["answer"] == "I do not have enough evidence."
    assert result["evidence"] == []


def test_evidence_answerer_answers_labeled_cup_location() -> None:
    cup = SimpleNamespace(
        entity_id="cup-1",
        class_name="cup",
        metadata={"label": "red"},
        zone="right",
        visibility=VisibilityState.VISIBLE,
    )
    memory = SimpleNamespace(snapshot=lambda _: SimpleNamespace(entities=(cup,)))

    result = EvidenceAnswerer(memory).answer("demo", "Where is the red cup?")

    assert result["answer"] == "The red cup is in the right zone."
    assert result["evidence"] == ["cup-1"]


class _TextKeyEncoder:
    """Maps specific texts to fixed keys so recall similarity is controllable."""

    def __init__(self, keys: dict[str, torch.Tensor], default: torch.Tensor):
        self.keys = keys
        self.default = default

    def encode_texts(self, texts):
        return torch.stack([self.keys.get(text, self.default) for text in texts])


class _InnerAnswerer:
    def __init__(self):
        self.calls = []

    def answer(self, session_id, question):
        self.calls.append((session_id, question))
        return {"answer": "fallback", "evidence": [], "mode": "fallback"}


def _identity_projector(dimension: int = 4) -> nn.Linear:
    projector = nn.Linear(dimension, dimension, bias=False)
    with torch.no_grad():
        projector.weight.copy_(torch.eye(dimension))
    return projector


def test_conversational_answerer_recalls_via_quote_back_without_gemma() -> None:
    statement = "My favorite color is blue."
    question = "What is my favorite color?"
    encoder = _TextKeyEncoder(
        {
            statement: torch.tensor([1.0, 0.0, 0.0, 0.0]),
            question: torch.tensor([0.95, 0.05, 0.0, 0.0]),
        },
        default=torch.zeros(4),
    )
    inner = _InnerAnswerer()
    answerer = ConversationalAnswerer(
        inner, ConversationMemory(key_dimension=4), encoder, _identity_projector()
    )

    answerer.answer("s", statement)  # first-ever turn: no history, falls through
    result = answerer.answer("s", question)

    assert result["mode"] == "conversation-recall"
    assert statement in result["answer"]
    assert inner.calls == [("s", statement)]


def test_conversational_answerer_falls_through_with_no_history() -> None:
    encoder = _TextKeyEncoder({}, default=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    inner = _InnerAnswerer()
    answerer = ConversationalAnswerer(
        inner, ConversationMemory(key_dimension=4), encoder, _identity_projector()
    )

    result = answerer.answer("s", "Where is the ball?")

    assert result == {"answer": "fallback", "evidence": [], "mode": "fallback"}
    assert inner.calls == [("s", "Where is the ball?")]


def test_conversational_answerer_falls_through_below_threshold() -> None:
    encoder = _TextKeyEncoder(
        {
            "My favorite color is blue.": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "Where is the ball?": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        },
        default=torch.zeros(4),
    )
    inner = _InnerAnswerer()
    answerer = ConversationalAnswerer(
        inner, ConversationMemory(key_dimension=4), encoder, _identity_projector()
    )

    answerer.answer("s", "My favorite color is blue.")  # first turn: no history
    result = answerer.answer("s", "Where is the ball?")

    assert result["mode"] == "fallback"
    assert inner.calls == [
        ("s", "My favorite color is blue."),
        ("s", "Where is the ball?"),
    ]


class _FakeCausalModel(nn.Module):
    def __init__(self, output_ids: torch.Tensor):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.output_ids = output_ids
        self.generate_calls: list[dict] = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return self.output_ids


class _FakeGemmaWorldModel(nn.Module):
    """Duck-types GemmaWorldModel's external surface: .model (plain chat),
    .encode_query (value/query encoding), and .generate (port injection)."""

    def __init__(self, output_ids: torch.Tensor, encode_output: torch.Tensor | None = None):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.model = _FakeCausalModel(output_ids)
        self.output_ids = output_ids
        self._encode_output = (
            encode_output if encode_output is not None else torch.zeros(1, 640)
        )
        self.generate_calls: list[dict] = []
        self.encode_calls: list[dict] = []

    @property
    def generate_calls_on_causal(self) -> list[dict]:
        return self.model.generate_calls

    def encode_query(self, input_ids, attention_mask):
        self.encode_calls.append(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )
        return self._encode_output

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return self.output_ids


class _FakeTokenizer:
    def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
        assert not tokenize
        assert add_generation_prompt
        return "PROMPT"

    def __call__(self, text, return_tensors, add_special_tokens=True):
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens
        return " Your favorite color is blue. "


class _AbstainingAnswerer:
    def answer(self, session_id, question):
        return {"answer": "I do not have enough evidence.", "evidence": [], "mode": "gemma"}


def test_conversational_answerer_replaces_abstention_with_plain_chat() -> None:
    encoder = _TextKeyEncoder({}, default=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    fake_model = _FakeGemmaWorldModel(torch.tensor([[1, 2, 3, 4, 5]]))
    answerer = ConversationalAnswerer(
        _AbstainingAnswerer(),
        ConversationMemory(key_dimension=4),
        encoder,
        _identity_projector(),
        model=fake_model,
        tokenizer=_FakeTokenizer(),
    )

    result = answerer.answer("s", "Hello")

    assert result["mode"] == "chat"
    assert result["answer"] == "Your favorite color is blue."
    assert len(fake_model.model.generate_calls) == 1
    assert not fake_model.generate_calls


def test_conversational_answerer_keeps_honest_abstention_for_world_question() -> None:
    # "Where is the car?" is a legitimate world question -- if nothing has
    # been detected yet, the honest answer is "no evidence", not Gemma
    # free-chatting about itself as if this were small talk.
    encoder = _TextKeyEncoder({}, default=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    fake_model = _FakeGemmaWorldModel(torch.tensor([[1, 2, 3, 4, 5]]))
    answerer = ConversationalAnswerer(
        _AbstainingAnswerer(),
        ConversationMemory(key_dimension=4),
        encoder,
        _identity_projector(),
        model=fake_model,
        tokenizer=_FakeTokenizer(),
    )

    result = answerer.answer("s", "Where is the car?")

    assert result == {
        "answer": "I do not have enough evidence.",
        "evidence": [],
        "mode": "gemma",
    }
    assert not fake_model.model.generate_calls
    assert not fake_model.generate_calls


def test_conversational_answerer_keeps_non_abstaining_inner_answer() -> None:
    encoder = _TextKeyEncoder({}, default=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    fake_model = _FakeGemmaWorldModel(torch.tensor([[1, 2, 3, 4, 5]]))
    answerer = ConversationalAnswerer(
        _InnerAnswerer(),
        ConversationMemory(key_dimension=4),
        encoder,
        _identity_projector(),
        model=fake_model,
        tokenizer=_FakeTokenizer(),
    )

    result = answerer.answer("s", "Where is the ball?")

    assert result == {"answer": "fallback", "evidence": [], "mode": "fallback"}
    assert not fake_model.model.generate_calls
    assert not fake_model.generate_calls


def test_conversational_answerer_phrases_recall_with_gemma_when_available() -> None:
    statement = "My favorite color is blue."
    question = "What is my favorite color?"
    encoder = _TextKeyEncoder(
        {
            statement: torch.tensor([1.0, 0.0, 0.0, 0.0]),
            question: torch.tensor([0.95, 0.05, 0.0, 0.0]),
        },
        default=torch.zeros(4),
    )
    fake_model = _FakeGemmaWorldModel(torch.tensor([[1, 2, 3, 4, 5]]))
    answerer = ConversationalAnswerer(
        _InnerAnswerer(),
        ConversationMemory(key_dimension=4),
        encoder,
        _identity_projector(),
        model=fake_model,
        tokenizer=_FakeTokenizer(),
    )

    answerer.answer("s", statement)
    result = answerer.answer("s", question)

    assert result["answer"] == "Your favorite color is blue."
    assert result["mode"] == "conversation-recall"
    assert len(fake_model.generate_calls) == 1
    assert not fake_model.model.generate_calls
    generate_kwargs = fake_model.generate_calls[0]
    assert generate_kwargs["memory_values"].shape == (1, 4, 640)
    assert generate_kwargs["memory_mask"].tolist() == [[True, False, False, False]]


def test_seed_session_makes_prefill_facts_recallable() -> None:
    fact = "The capital of France is Paris."
    question = "What is the capital of France?"
    encoder = _TextKeyEncoder(
        {
            fact: torch.tensor([1.0, 0.0, 0.0, 0.0]),
            question: torch.tensor([0.95, 0.05, 0.0, 0.0]),
        },
        default=torch.zeros(4),
    )
    fake_model = _FakeGemmaWorldModel(torch.tensor([[1, 2, 3, 4, 5]]))
    answerer = ConversationalAnswerer(
        _InnerAnswerer(),
        ConversationMemory(key_dimension=4),
        encoder,
        _identity_projector(),
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        prefill_facts=(fact,),
    )

    answerer.seed_session("s")
    result = answerer.answer("s", question)

    assert result["mode"] == "conversation-recall"
    assert result["evidence"] == ["utterance-0001"]
    assert answerer.memory.history("s")[0].text == fact
    assert answerer.memory.history("s")[0].timestamp == 0.0


def test_seed_session_caches_encoding_across_sessions() -> None:
    encoder = _TextKeyEncoder({}, default=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    fake_model = _FakeGemmaWorldModel(torch.tensor([[1, 2, 3, 4, 5]]))
    answerer = ConversationalAnswerer(
        _InnerAnswerer(),
        ConversationMemory(key_dimension=4),
        encoder,
        _identity_projector(),
        model=fake_model,
        tokenizer=_FakeTokenizer(),
        prefill_facts=("Fact one.", "Fact two."),
    )

    answerer.seed_session("session-a")
    answerer.seed_session("session-b")

    assert len(fake_model.encode_calls) == 2  # once per fact, not per session
    assert [item.text for item in answerer.memory.history("session-a")] == [
        "Fact one.",
        "Fact two.",
    ]
    assert [item.text for item in answerer.memory.history("session-b")] == [
        "Fact one.",
        "Fact two.",
    ]


def test_seed_session_is_noop_without_model_or_prefill_facts() -> None:
    encoder = _TextKeyEncoder({}, default=torch.tensor([1.0, 0.0, 0.0, 0.0]))
    answerer = ConversationalAnswerer(
        _InnerAnswerer(),
        ConversationMemory(key_dimension=4),
        encoder,
        _identity_projector(),
        prefill_facts=("Fact one.",),
    )

    answerer.seed_session("s")  # no model configured -- must not crash

    assert answerer.memory.history("s") == ()
