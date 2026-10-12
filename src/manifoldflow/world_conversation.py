"""Session-scoped memory for user statements, keyed the same way as WorldMemory.

Reuses the same basin/index architecture WorldMemory uses for persistent
visual entities, applied instead to text: a user utterance is stored under
a semantic key (the shared SigLIP2 text/image projection already trained for
CrossModalKeyProjector) and later recalled by nearest-neighbor similarity to
a new question's key. No training is required for storage or retrieval.

Each utterance also carries a value vector -- Gemma's own pooled hidden
state over the utterance text, mirroring Sonic's basin-write mechanism
(`manifoldflow_production_train.py`) rather than a hand-designed encoding.
Retrieval (this module) needs no training; faithfully decoding an answer
from the retrieved value, injected as a continuous token through a
LatentMemoryPort exactly like visual entity state already is, does -- that
half lives in ConversationalAnswerer and its dedicated training script.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch
from torch import Tensor

from manifoldflow.bank import BasinState
from manifoldflow.index import ExactMainDeltaIndex
from manifoldflow.world_types import normalized


@dataclass(frozen=True)
class Utterance:
    utterance_id: str
    session_id: str
    text: str
    timestamp: float
    speaker: str
    value: Tensor


@dataclass(frozen=True)
class ConversationHit:
    utterance: Utterance
    score: float


@dataclass
class _ConversationSession:
    utterances: dict[str, Utterance] = field(default_factory=dict)
    bank: BasinState | None = None
    index: ExactMainDeltaIndex | None = None
    basin_to_utterance: list[str] = field(default_factory=list)
    next_id: int = 1


class ConversationMemory:
    """Store and recall user statements by semantic similarity, per session."""

    def __init__(
        self,
        key_dimension: int = 128,
        value_dimension: int = 640,
        rbf_temperature: float = 0.5,
    ) -> None:
        if key_dimension <= 0:
            raise ValueError("key_dimension must be positive")
        if value_dimension <= 0:
            raise ValueError("value_dimension must be positive")
        if rbf_temperature <= 0:
            raise ValueError("rbf_temperature must be positive")
        self.key_dimension = key_dimension
        self.value_dimension = value_dimension
        self.rbf_temperature = rbf_temperature
        self._sessions: dict[str, _ConversationSession] = {}
        self._lock = threading.Lock()

    def _validate_key(self, key: Tensor) -> Tensor:
        if key.shape != (self.key_dimension,):
            raise ValueError(f"key must have shape [{self.key_dimension}]")
        return normalized(key)

    def _validate_value(self, value: Tensor) -> Tensor:
        if value.shape != (self.value_dimension,):
            raise ValueError(f"value must have shape [{self.value_dimension}]")
        return value.detach().float().cpu()

    def remember(
        self,
        session_id: str,
        text: str,
        key: Tensor,
        value: Tensor,
        *,
        timestamp: float,
        speaker: str = "user",
    ) -> Utterance:
        if not text.strip():
            raise ValueError("text must be non-empty")
        validated_key = self._validate_key(key)
        validated_value = self._validate_value(value)
        with self._lock:
            session = self._sessions.setdefault(session_id, _ConversationSession())
            utterance_id = f"utterance-{session.next_id:04d}"
            session.next_id += 1
            utterance = Utterance(
                utterance_id, session_id, text, timestamp, speaker, validated_value
            )
            session.utterances[utterance_id] = utterance
            if session.bank is None:
                session.bank = BasinState.from_statistics(
                    validated_key[None, :],
                    torch.ones(1, dtype=torch.long),
                    torch.zeros_like(validated_key)[None, :],
                )
                session.index = ExactMainDeltaIndex(session.bank.mu)
                basin_id = 0
            else:
                basin_id = session.bank.append_statistics(
                    validated_key, count=1, m2=torch.zeros_like(validated_key)
                )
                assert session.index is not None
                session.index.upsert(
                    torch.tensor([basin_id]), session.bank.mu[basin_id][None, :]
                )
            session.basin_to_utterance.append(utterance_id)
            return utterance

    def recall(
        self, session_id: str, query_key: Tensor, top_k: int = 3
    ) -> list[ConversationHit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        validated_key = self._validate_key(query_key)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or not session.utterances:
                return []
            assert session.bank is not None and session.index is not None
            route_k = min(max(top_k * 2, top_k), session.bank.mu.shape[0])
            routed = session.index.query(validated_key[None, :], route_k)[0]
            distances = (
                (session.bank.mu[routed] - validated_key[None, :]).square().sum(-1)
            )
            scores = torch.exp(-distances / self.rbf_temperature)
            count = min(top_k, routed.numel())
            local_indices = torch.topk(scores, count).indices.tolist()
            hits: list[ConversationHit] = []
            for local_index in local_indices:
                basin_id = int(routed[local_index].item())
                utterance = session.utterances[session.basin_to_utterance[basin_id]]
                hits.append(
                    ConversationHit(utterance, float(scores[local_index].item()))
                )
            return hits

    def history(self, session_id: str) -> tuple[Utterance, ...]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return ()
            return tuple(
                session.utterances[uid] for uid in session.basin_to_utterance
            )

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
