"""Question answering over immutable world snapshots and latent retrieval."""

from __future__ import annotations

import time
from typing import Any, Protocol

import torch
from torch import Tensor

from manifoldflow.gemma_world import GemmaQueryProjector, GemmaWorldModel
from manifoldflow.world_conversation import ConversationHit, ConversationMemory
from manifoldflow.world_evaluation import (
    WORLD_FACT_CANDIDATES,
    answer_candidates_for_question,
)
from manifoldflow.world_memory import WorldMemory
from manifoldflow.world_types import VisibilityState, WorldSnapshot
from manifoldflow.world_vision import CrossModalKeyProjector, VisionLanguageEncoder


class Answerer(Protocol):
    def answer(self, session_id: str, question: str) -> dict[str, Any]: ...


def _zone_phrase(zone: str) -> str:
    """"upper_left" -> "upper left": WorldRecord.zone stays underscore-joined
    (matching ZONES/zone_for_box in world_types.py) since that's what the
    trained state encoding indexes into; only rendered answer text needs
    natural spacing."""
    return zone.replace("_", " ")


class EvidenceAnswerer:
    """Deterministic fail-safe for curated demos and degraded GPU operation."""

    def __init__(self, memory: WorldMemory) -> None:
        self.memory = memory

    def answer(self, session_id: str, question: str) -> dict[str, Any]:
        snapshot = self.memory.snapshot(session_id)
        text, evidence = self._answer_snapshot(snapshot, question.lower())
        return {"answer": text, "evidence": evidence, "mode": "evidence"}

    def _answer_snapshot(
        self, snapshot: WorldSnapshot, question: str
    ) -> tuple[str, list[str]]:
        if "ball" in question:
            ball = next(
                (item for item in snapshot.entities if item.class_name == "ball"), None
            )
            if ball is None:
                return "I do not have evidence of a ball.", []
            relation = next(
                (
                    item
                    for item in snapshot.relations
                    if item.active
                    and item.subject_id == ball.entity_id
                    and item.relation == "contained_by"
                ),
                None,
            )
            if relation is not None:
                cup = snapshot.entity(relation.target_id)
                label = (
                    cup.metadata.get("label", cup.entity_id)
                    if cup
                    else relation.target_id
                )
                return f"The ball is under the {label} cup.", [
                    ball.entity_id,
                    relation.target_id,
                ]
            if ball.visibility == VisibilityState.VISIBLE:
                return f"The ball is visible in the {_zone_phrase(ball.zone)} zone.", [
                    ball.entity_id
                ]
            return "I cannot determine which cup contains the ball.", [ball.entity_id]
        if "car" in question:
            car = next(
                (item for item in snapshot.entities if item.class_name == "toy_car"),
                None,
            )
            if car is None:
                return "I do not have evidence of a toy car.", []
            if car.visibility == VisibilityState.OCCLUDED:
                return "The car is currently hidden behind the box.", [car.entity_id]
            if car.visibility == VisibilityState.VISIBLE:
                return f"The car is visible in the {_zone_phrase(car.zone)} zone.", [
                    car.entity_id
                ]
            return "The car was seen, but its current location is unknown.", [
                car.entity_id
            ]
        if "cup" in question:
            cups = [item for item in snapshot.entities if item.class_name == "cup"]
            requested_label = next(
                (label for label in ("red", "blue", "yellow") if label in question),
                None,
            )
            if requested_label is not None:
                cup = next(
                    (
                        item
                        for item in cups
                        if item.metadata.get("label") == requested_label
                    ),
                    None,
                )
                if cup is None:
                    return f"I do not have evidence of a {requested_label} cup.", []
                if cup.visibility == VisibilityState.VISIBLE:
                    return (
                        f"The {requested_label} cup is in the "
                        f"{_zone_phrase(cup.zone)} zone.",
                        [cup.entity_id],
                    )
                return (
                    f"The {requested_label} cup is currently {cup.visibility.value}.",
                    [cup.entity_id],
                )
            if not cups:
                return "I do not have evidence of any cups.", []
            summary = ", ".join(
                f"{item.metadata.get('label', item.entity_id)} in {_zone_phrase(item.zone)}"
                for item in cups
            )
            return f"Observed cups: {summary}.", [item.entity_id for item in cups]
        if "moved" in question or "happened" in question:
            recent = snapshot.events[-5:]
            if not recent:
                return "No world events have been observed yet.", []
            summary = ", ".join(item.event_type.value for item in recent)
            return f"Recent events: {summary}.", [item.entity_id for item in recent]
        return (
            "I can answer questions about the car, ball, cups, and recent movement.",
            [],
        )


class GemmaLatentAnswerer:
    """Retrieve with SigLIP text keys and answer through Gemma latent tokens.

    WorldMemory.query always returns its nearest neighbors regardless of how
    distant they are, so a question with nothing to do with the scene ("Hello")
    still retrieves whatever entities exist and forces them into Gemma's
    context as if they were relevant. min_relevance_score drops hits below
    that similarity floor before answering, so an irrelevant question falls
    back to "not enough evidence" instead of a confused answer grounded in
    noise.

    The default was recalibrated 2026-09-02 against real measured scores,
    not guessed: relevant questions against a live curated scenario score
    0.0003-0.006 (e.g. "Where is the red cup?" -> 0.0051 for the actual red
    cup), while genuinely irrelevant ones ("Hello", "What is the capital of
    France?") score 0.0-0.0001 -- a clean, well-separated gap. The prior
    default of 0.15 silently filtered out every real match too (retrieval
    was separately broken at the time -- see world_types.concept_appearance_key
    and the 2026-09-02 dev journal entry -- so no threshold could have
    passed anything through), which is how a threshold two orders of
    magnitude too strict went unnoticed.
    """

    def __init__(
        self,
        memory: WorldMemory,
        encoder: VisionLanguageEncoder,
        projector: CrossModalKeyProjector,
        model: GemmaWorldModel,
        tokenizer: Any,
        query_projector: GemmaQueryProjector | None = None,
        max_memory: int = 4,
        response_instruction: str | None = None,
        prompt_format: str = "chat",
        constrained: bool = True,
        min_relevance_score: float = 0.0002,
    ) -> None:
        self.memory = memory
        self.encoder = encoder
        self.projector = projector
        self.model = model
        self.tokenizer = tokenizer
        self.query_projector = query_projector
        self.max_memory = max_memory
        self.response_instruction = response_instruction
        self.prompt_format = prompt_format
        self.constrained = constrained
        self.min_relevance_score = min_relevance_score

    @torch.inference_mode()
    def answer(self, session_id: str, question: str) -> dict[str, Any]:
        feature = self.encoder.encode_texts([question])
        projector_device = next(self.projector.parameters()).device
        query_key = self.projector(feature.to(projector_device))[0].cpu()
        query_routes = [query_key]
        if self.query_projector is not None:
            if self.prompt_format == "chat":
                query_prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                query_prompt = f"Question: {question}\nAnswer:"
            tokens = self.tokenizer(
                query_prompt,
                add_special_tokens=self.prompt_format != "chat",
                return_tensors="pt",
            )
            device = next(self.model.parameters()).device
            hidden = self.model.encode_query(
                tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
            )
            route = self.query_projector(hidden)[0].detach().cpu()
            query_routes.append(route)
        hits = self.memory.query(
            session_id, torch.stack(query_routes), top_k=self.max_memory
        )
        hits = [item for item in hits if item.score >= self.min_relevance_score]
        if not hits:
            return {
                "answer": "I do not have enough evidence.",
                "evidence": [],
                "mode": "gemma",
            }
        values = torch.stack([item.record.state_value for item in hits])
        keys = torch.stack([item.record.appearance_key for item in hits])
        weights = torch.tensor([item.score for item in hits])
        confidence = torch.tensor([item.record.confidence for item in hits])
        count = values.shape[0]
        if count < self.max_memory:
            values = torch.nn.functional.pad(values, (0, 0, 0, self.max_memory - count))
            keys = torch.nn.functional.pad(keys, (0, 0, 0, self.max_memory - count))
            weights = torch.nn.functional.pad(weights, (0, self.max_memory - count))
            confidence = torch.nn.functional.pad(
                confidence, (0, self.max_memory - count)
            )
        device = next(self.model.parameters()).device
        memory_mask = (weights > 0)[None, :].to(device)
        weight_batch = weights[None, :].to(device)
        weight_batch = weight_batch / weight_batch.sum(dim=1, keepdim=True).clamp_min(
            1e-8
        )
        if self.constrained:
            answer = self._rank_answer(
                question,
                values.to(device),
                keys.to(device),
                memory_mask,
                weight_batch,
                confidence.to(device),
            )
            return {
                "answer": answer,
                "evidence": [item.entity_id for item in hits],
                "mode": "gemma-ranked",
            }
        prompt = self._prompt(question)
        tokens = self.tokenizer(
            prompt,
            add_special_tokens=self.prompt_format != "chat",
            return_tensors="pt",
        )
        output = self.model.generate(
            input_ids=tokens["input_ids"].to(device),
            attention_mask=tokens["attention_mask"].to(device),
            memory_values=values[None, :].to(device),
            memory_keys=keys[None, :].to(device),
            memory_mask=memory_mask,
            retrieval_weights=weight_batch,
            confidence=confidence[None, :].to(device),
            generation_kwargs={"max_new_tokens": 48, "do_sample": False},
        )
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return {
            "answer": answer,
            "evidence": [item.entity_id for item in hits],
            "mode": "gemma",
        }

    def _prompt(self, question: str) -> str:
        content = (
            f"{self.response_instruction}\n\nQuestion: {question}"
            if self.response_instruction
            else question
        )
        if self.prompt_format == "chat":
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"Question: {content}\nAnswer:"

    def _candidate_tokens(
        self, question: str, candidates: tuple[str, ...], device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor]:
        content = (
            f"{self.response_instruction}\n\nQuestion: {question}"
            if self.response_instruction
            else question
        )
        if self.prompt_format == "chat":
            conversation = [{"role": "user", "content": content}]
            prompt = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            texts = [
                self.tokenizer.apply_chat_template(
                    [*conversation, {"role": "assistant", "content": candidate}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for candidate in candidates
            ]
            add_special_tokens = False
        else:
            prompt = f"Question: {content}\nAnswer:"
            texts = [f"{prompt} {candidate}" for candidate in candidates]
            add_special_tokens = True
        encoded = self.tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            padding=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        )
        prompt_length = len(
            self.tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=96,
            )["input_ids"]
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        labels[:, :prompt_length] = -100
        return input_ids, attention_mask, labels

    def _rank_answer(
        self,
        question: str,
        values: Tensor,
        keys: Tensor,
        memory_mask: Tensor,
        weights: Tensor,
        confidence: Tensor,
    ) -> str:
        candidates = answer_candidates_for_question(question, WORLD_FACT_CANDIDATES)
        device = values.device
        input_ids, attention_mask, labels = self._candidate_tokens(
            question, candidates, device
        )
        count = len(candidates)
        prepared = self.model.prepare_inputs(
            input_ids,
            attention_mask,
            values[None, :].expand(count, -1, -1),
            memory_keys=keys[None, :].expand(count, -1, -1),
            labels=labels,
            memory_mask=memory_mask.expand(count, -1),
            retrieval_weights=weights.expand(count, -1),
            confidence=confidence[None, :].expand(count, -1),
        )
        output = self.model.model(
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=prepared.attention_mask,
            return_dict=True,
        )
        targets = prepared.labels[:, 1:]
        losses = torch.nn.functional.cross_entropy(
            output.logits[:, :-1].float().transpose(1, 2),
            targets,
            ignore_index=-100,
            reduction="none",
        )
        valid = targets != -100
        mean_loss = losses.sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return candidates[int(mean_loss.argmin())]


_ABSTENTION_PHRASES = (
    "enough evidence",
    "do not have evidence",
    "cannot determine",
    "can't determine",
    "unable to determine",
)

# If a question mentions the scene's own vocabulary, an abstention is an
# honest "no evidence yet" and must be kept, not replaced with free chat --
# otherwise "Where is the car?" with nothing detected yet gets answered by
# Gemma talking about itself instead of admitting it doesn't know. Only
# questions with none of this vocabulary are treated as general chat.
_WORLD_TOPIC_KEYWORDS = (
    "ball",
    "car",
    "cup",
    "box",
    "red",
    "blue",
    "yellow",
    "visible",
    "view",
    "camera",
    "hiding",
    "hidden",
    "occlude",
    "conceal",
    "contain",
)


class ConversationalAnswerer:
    """Add persistent recall and general chat on top of any Answerer.

    Every question is stored so later questions can recall it, the same way
    Sonic writes a basin: the utterance's key is the shared SigLIP2 text
    embedding (used only for retrieval routing), and its value is Gemma's
    own pooled hidden state over the utterance text (encode_value, mirroring
    GemmaWorldModel.encode_query). On recall, that value is injected back
    into Gemma as a continuous memory token through the same LatentMemoryPort
    mechanism GemmaLatentAnswerer already uses for visual world facts --
    not replayed as text in the prompt. This is what actually needs
    training: retrieval requires none (the frozen SigLIP2 key space already
    routes correctly), but faithfully decoding an answer from an injected
    value does, via a dedicated port and LoRA trained on conversational
    recall (see the recall training script; the port here must have
    state_dimension == memory.value_dimension).

    A question is answered from conversational recall only when the closest
    stored utterance is semantically close enough (recall_threshold);
    otherwise it falls through to the wrapped answerer. The wrapped visual
    answerer forces every question through closed-world facts about
    detected objects, so a question with nothing to do with the scene
    ("Hello") gets an irrelevant entity injected as context and, correctly,
    an abstention -- but that reads as a broken chatbot to a user who just
    said hi. When the wrapped answerer abstains, a Gemma model is available,
    and the question doesn't mention the scene's own vocabulary at all,
    this generates a plain conversational reply (no memory injected, since
    there is nothing relevant to inject) instead of surfacing the
    abstention. A question that does mention the scene ("Where is the
    car?") keeps its honest abstention even with nothing detected yet,
    rather than having Gemma talk about itself instead of admitting it
    doesn't know.
    """

    def __init__(
        self,
        inner: Answerer,
        memory: ConversationMemory,
        encoder: VisionLanguageEncoder,
        projector: CrossModalKeyProjector,
        model: GemmaWorldModel | None = None,
        tokenizer: Any | None = None,
        recall_threshold: float = 0.38,
        top_k: int = 4,
        prefill_facts: tuple[str, ...] = (),
    ) -> None:
        if not 0.0 <= recall_threshold <= 1.0:
            raise ValueError("recall_threshold must be between zero and one")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.inner = inner
        self.memory = memory
        self.encoder = encoder
        self.projector = projector
        self.model = model
        self.tokenizer = tokenizer
        self.recall_threshold = recall_threshold
        self.top_k = top_k
        self.prefill_facts = prefill_facts
        self._prefill_cache: list[tuple[str, Tensor, Tensor]] | None = None

    @torch.inference_mode()
    def seed_session(self, session_id: str) -> None:
        """Ground general-knowledge questions through the same injected-
        vector mechanism corrections use, rather than the LoRA'd model's
        own parametric knowledge -- which the training this session ran
        measurably damages at the rank/step budget needed to make
        corrections reliable (2026-09-02 dev journal: capital-of-France
        and color questions collapsed to single training-vocabulary nouns
        after 16k steps). A persistent memory layer grounding answers in
        stored facts rather than parametric recall is the mechanism this
        whole project is built around; this is that mechanism applied to
        the model's own general-knowledge gaps, not a new one. Keys/values
        are computed once (cached) and copied into every new session, not
        recomputed per session -- cheap, since the fact set is fixed and
        small.
        """
        if not self.prefill_facts or self.model is None or self.tokenizer is None:
            return
        if self._prefill_cache is None:
            device = next(self.projector.parameters()).device
            entries: list[tuple[str, Tensor, Tensor]] = []
            for fact in self.prefill_facts:
                feature = self.encoder.encode_texts([fact])
                key = self.projector(feature.to(device))[0].detach().cpu()
                value = self._encode_value(fact)
                entries.append((fact, key, value))
            self._prefill_cache = entries
        for fact, key, value in self._prefill_cache:
            self.memory.remember(
                session_id, fact, key, value, timestamp=0.0, speaker="system"
            )

    @torch.inference_mode()
    def answer(self, session_id: str, question: str) -> dict[str, Any]:
        feature = self.encoder.encode_texts([question])
        device = next(self.projector.parameters()).device
        key = self.projector(feature.to(device))[0].detach().cpu()

        # A world question ("Where is the car?") that happens to repeat a
        # prior turn's exact or near-exact wording will always score above
        # recall_threshold against that earlier occurrence of itself --
        # every question gets stored via remember() below, world-directed
        # or not, so a second "Where is the car?" recalls the first
        # instead of re-querying current state. Gate recall out entirely
        # for anything world-shaped; GemmaLatentAnswerer is the trained,
        # authoritative source for these topics, and conversational recall
        # was never trained on them at all -- there is no legitimate case
        # where recall should outrank a fresh visual query here. The
        # question is still stored (harmless, and still useful if a
        # genuinely fact-bearing statement happens to match this
        # vocabulary too), just not treated as a candidate this turn.
        looks_like_world_question = self._looks_like_world_question(question)
        hits = (
            []
            if looks_like_world_question
            else self.memory.recall(session_id, key, top_k=self.top_k)
        )
        value = self._encode_value(question)
        self.memory.remember(session_id, question, key, value, timestamp=time.time())

        if not hits or hits[0].score < self.recall_threshold:
            result = self.inner.answer(session_id, question)
            if (
                self.model is not None
                and self.tokenizer is not None
                and self._is_abstention(str(result.get("answer", "")))
                and not looks_like_world_question
            ):
                return {
                    "answer": self._generate_plain(question),
                    "evidence": result.get("evidence", []),
                    "mode": "chat",
                }
            return result

        if self.model is not None and self.tokenizer is not None:
            answer_text = self._generate_with_port(question, hits)
        else:
            answer_text = self._quote_back(hits)
        return {
            "answer": answer_text,
            "evidence": [item.utterance.utterance_id for item in hits],
            "mode": "conversation-recall",
        }

    def close_session(self, session_id: str) -> None:
        self.memory.close_session(session_id)
        close_inner_session = getattr(self.inner, "close_session", None)
        if close_inner_session is not None:
            close_inner_session(session_id)

    @staticmethod
    def _quote_back(hits: list[ConversationHit]) -> str:
        quotes = "; ".join(f'"{item.utterance.text}"' for item in hits)
        return f"Earlier you told me: {quotes}"

    @staticmethod
    def _normalize_recency(timestamps: Tensor) -> Tensor:
        if timestamps.numel() <= 1:
            return torch.zeros_like(timestamps)
        ranks = timestamps.argsort().argsort().to(timestamps.dtype)
        return ranks / (timestamps.numel() - 1)

    @staticmethod
    def _is_abstention(text: str) -> bool:
        lowered = text.lower()
        return any(phrase in lowered for phrase in _ABSTENTION_PHRASES)

    @staticmethod
    def _looks_like_world_question(question: str) -> bool:
        lowered = question.lower()
        return any(keyword in lowered for keyword in _WORLD_TOPIC_KEYWORDS)

    def _encode_value(self, text: str) -> Tensor:
        if self.model is None:
            return torch.zeros(self.memory.value_dimension)
        tokens = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        hidden = self.model.encode_query(
            tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
        )
        return hidden[0].detach().cpu()

    def _generate_plain(self, question: str) -> str:
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = self.tokenizer(chat, add_special_tokens=False, return_tensors="pt")
        device = next(self.model.parameters()).device
        tokens = {name: value.to(device) for name, value in tokens.items()}
        output = self.model.model.generate(
            **tokens, max_new_tokens=64, do_sample=False
        )
        return self.tokenizer.decode(
            output[0][tokens["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    def _generate_with_port(self, question: str, hits: list[ConversationHit]) -> str:
        device = next(self.model.parameters()).device
        values = torch.stack([item.utterance.value for item in hits]).to(device)
        # Uniform, not similarity-score-based: world_train_recall_port.py's
        # _encode_values gives every valid memory slot equal weight
        # (memory_mask / count) in 100% of training -- the port's
        # retrieval_weights input never carried a discriminative signal
        # during training, so feeding it real (highly variable) SigLIP2
        # scores here is a distribution the model never saw. Measured
        # directly: uniform weights raised a 15-episode live-pipeline
        # sample from 7/15 to 10/15 correct with an otherwise-identical
        # rank-32 checkpoint.
        weights = torch.ones(len(hits), device=device) / len(hits)
        timestamps = torch.tensor(
            [item.utterance.timestamp for item in hits], device=device
        )
        recency = self._normalize_recency(timestamps)
        memory_mask = torch.ones(len(hits), dtype=torch.bool, device=device)
        count = len(hits)
        if count < self.top_k:
            pad = self.top_k - count
            values = torch.nn.functional.pad(values, (0, 0, 0, pad))
            weights = torch.nn.functional.pad(weights, (0, pad))
            recency = torch.nn.functional.pad(recency, (0, pad))
            memory_mask = torch.nn.functional.pad(memory_mask, (0, pad))
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = self.tokenizer(chat, add_special_tokens=False, return_tensors="pt")
        tokens = {name: value.to(device) for name, value in tokens.items()}
        output = self.model.generate(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            memory_values=values[None, :],
            memory_mask=memory_mask[None, :],
            recency=recency[None, :],
            retrieval_weights=weights[None, :],
            generation_kwargs={"max_new_tokens": 64, "do_sample": False},
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
