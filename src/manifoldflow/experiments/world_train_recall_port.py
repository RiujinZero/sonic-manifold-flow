"""Train Gemma's conversational-recall port + LoRA on correction chains.

This is the "proper Sonic version" of conversational recall: rather than
replaying past user statements as text in the prompt (which forgets under
long context, the same failure R2/R3 hit on the visual side), each stored
utterance is written as a value vector -- Gemma's own pooled hidden state
over the statement text, computed the same way encode_query pools a
question -- and read back by injecting it as a continuous memory token
through LatentMemoryPort, exactly like GemmaLatentAnswerer already does for
visual world facts (world_answer.ConversationalAnswerer /
world_conversation.ConversationMemory implement the storage and injection
side; this script trains the port + a dedicated LoRA to decode faithfully
from what gets injected).

Retrieval itself needs no training -- ConversationMemory routes on the
already-trained, frozen SigLIP2 text key space, and that routing was
already validated for the visual path. What is untrained, and what this
script targets, is faithful decoding under distractors: given a
question and a handful of injected candidate values -- most of them
irrelevant, some of them stale corrections of the same fact -- can the
model select the *last* statement that actually answers the question and
ignore the rest, instead of a tiny model's default behavior of grabbing
whatever's topically closest or most recently phrased. This is the exact
failure the user hit outside of training ("Kevin having 5 dogs when I said
5 dogs barked").

Uses the vendored, already-validated `distractor_reasoning.correction`
generator (see sonic_correction_memory_probe.py for the precedent this
script's data pipeline follows) rather than hand-built examples: it already
implements the adversarial construction needed here -- decoys placed after
the real correction to defeat "grab the last number mentioned anywhere",
a same-subject-different-noun decoy that binds the correction to the right
attribute, and decoy_observation statements (a mere sighting that must
never override the tracked value).

Every statement's value is injected together as one candidate set (not
just the oracle-correct one) -- unlike world_train_oracle.py, which trains
answer-decode only against pre-filtered ground-truth evidence. Real
inference retrieval will not perfectly filter out topically-similar
distractors (a SigLIP2 text embedding cannot tell "Kevin owns 4 parrots"
apart from "Kevin owns 7 parrots" as cleanly as it tells "parrots" apart
from "toy car"), so training must include the same mixed, undifferentiated
candidate set the port will actually see, and rely on the decode loss
itself -- plus an explicit write-order recency signal, since "corrections"
are fundamentally about which statement came last -- to teach selection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from manifoldflow.gemma_world import (
    GemmaWorldModel,
    LatentMemoryPort,
    enable_upper_layer_lora,
    load_gemma_world_model,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "vendor" / "GTR-sonic-ssm"))


@dataclass(frozen=True)
class RecallEpisode:
    statements: tuple[str, ...]
    question: str
    answer: str
    target_index: int


def _build_correction_episode(
    rng: random.Random, level: int, *, max_memory_slots: int, reverse_fraction: float = 0.0
) -> RecallEpisode | None:
    """distractor_reasoning.correction: tracking which of several
    same-subject-same-noun statements is current, the exact shape of the
    demo's motivating failure ("Kevin has 5 dogs" from "5 dogs barked").
    Every statement follows one "{subject} {relation} {quantity}
    {noun_phrase}." template family (plus a fixed set of
    "Actually,"/"Wait,"/"Correction:" prefixes) -- this generator alone
    cannot teach the model to generalize past that one shape, which is
    exactly what _build_general_episode below is for.

    reverse_fraction controls how often the question is framed in reverse
    -- "Who owns 8 guitars?" (answer: the subject's name) or "What does
    Kevin own 8 of?" (answer: the noun) instead of the forward "How many
    guitars does Kevin own?" (answer: the quantity). Every prior
    generator in this project asked only the forward direction; live
    testing found that gap directly ("who has 17 cats?" failed outright
    on a checkpoint that answered the forward phrasing of the identical
    fact perfectly). Not a new generator -- this is a direct port of the
    text-construction logic already built and validated in
    sonic_reverse_query_probe.py / sonic_noun_target_probe.py (see
    session_builder.py's own docstring, which already reuses those same
    two probes' _build_episode for exactly this reason). Those probes
    tokenize with the GTR/Sonic tokenizer for Sonic's own training; the
    reverse/noun-target question and answer text they build is plain
    Python string formatting from the same CorrectionExample this
    function already has, with no tokenizer involved until the very end
    -- reusable here verbatim, just tokenized with Gemma's tokenizer
    downstream like every other episode in this script, not decoded from
    one tokenizer's ids into another's."""
    from distractor_reasoning.correction import (
        HAVING_RELATIONS_PLURAL,
        generate_correction_example,
        render_statement,
    )
    from distractor_reasoning.query import UNKNOWN
    from distractor_reasoning.vocab import HAVING_RELATIONS, nouns_for

    nouns = nouns_for("train")
    example = generate_correction_example(rng, level=level, split="train")
    if example.answer_value == UNKNOWN:
        return None  # honest-abstention cases are out of scope for this pass
    if len(example.statements) > max_memory_slots:
        return None

    relation = next(
        candidate
        for candidate in HAVING_RELATIONS
        if f" {candidate} " in example.answer_text
    )
    target_key = (example.query.subject, example.query.target)
    target_index = None
    for index, statement in enumerate(example.statements):
        if statement.kind == "decoy_observation":
            continue
        if (statement.subject, statement.noun) == target_key:
            target_index = index
    if target_index is None:
        return None

    quantity = example.answer_value
    relation_base = HAVING_RELATIONS_PLURAL[relation]
    noun_info = nouns[example.query.target]
    noun_phrase = noun_info.singular if quantity == 1 else noun_info.plural

    question_type = "forward"
    if rng.random() < reverse_fraction:
        question_type = rng.choice(("reverse_subject", "reverse_noun"))

    if question_type == "reverse_subject":
        # Well-posed only if this quantity is unique among the episode's
        # own possession-kind statements -- see
        # sonic_reverse_query_probe.py's require_unique_reverse for the
        # full rationale (a decoy_fact could coincidentally share the
        # same number for a different subject/noun).
        target_statement = example.statements[target_index]
        other_quantities = [
            statement.quantity
            for statement in example.statements
            if statement.kind != "decoy_observation" and statement is not target_statement
        ]
        if quantity in other_quantities:
            return None
        question = f"Who {relation} {quantity} {noun_phrase}?"
        answer = f"{example.query.subject}."
    elif question_type == "reverse_noun":
        question = f"What does {example.query.subject} {relation_base} {quantity} of?"
        answer = f"{noun_phrase}."
    else:
        question = (
            f"How many {noun_info.plural} does {example.query.subject} {relation_base}?"
        )
        answer = example.answer_text.removeprefix("A: ")

    statements = tuple(
        render_statement(statement, rng, nouns, relation)
        for statement in example.statements
    )
    return RecallEpisode(
        statements=statements, question=question, answer=answer, target_index=target_index
    )


def _build_general_episode(
    rng: random.Random, level: int, *, max_memory_slots: int
) -> RecallEpisode | None:
    """distractor_reasoning.generate (via world.py's symbolic Fact/World):
    a *static* set of facts (some relevant, many distractors across six
    distinct distractor kinds -- subject_relation, relation_object,
    subject_object, subject_only, object_only, category_confusable -- plus
    filler facts about unrelated people, plus category-generalization
    questions from level 5), one targeted query. No correction tracking at
    all -- this generator complements correction.py rather than
    overlapping it, specifically to break the single-template-family
    overfitting risk correction.py alone carries (see the
    2026-09-02 dev journal entry's phrasing-distribution-gap finding).
    Levels capped at 5 here to stay inside this script's existing
    single-integer-answer, always-answerable contract -- aggregation
    (level 6+), explicit zero (7+), and reported-fact disambiguation (9+)
    are real, harder task shapes this generator also supports, deliberately
    left for a follow-up rather than folded in silently."""
    from distractor_reasoning.generate import generate_example
    from distractor_reasoning.query import UNKNOWN
    from distractor_reasoning.render import render_fact, render_question
    from distractor_reasoning.vocab import nouns_for

    nouns = nouns_for("train")
    example = generate_example(rng, level=level, split="train")
    if example.answer_value == UNKNOWN:
        return None
    if len(example.world.facts) > max_memory_slots:
        return None

    target_index = next(
        (
            index
            for index, fact in enumerate(example.world.facts)
            if fact.is_having
            and not fact.is_reported
            and fact.subject == example.query.subject
            and (
                fact.noun == example.query.target
                if not example.query.is_category
                else nouns[fact.noun].categories
                and example.query.target in nouns[fact.noun].categories
            )
        ),
        None,
    )
    if target_index is None:
        return None

    statements = tuple(
        render_fact(fact, rng, "train", nouns) for fact in example.world.facts
    )
    question = render_question(example.query, rng, "train", nouns)
    answer = example.answer_text.removeprefix("A: ")
    return RecallEpisode(
        statements=statements, question=question, answer=answer, target_index=target_index
    )


def _build_episode(
    rng: random.Random,
    *,
    correction_level: int,
    general_level: int,
    general_fraction: float,
    reverse_fraction: float,
    max_memory_slots: int,
) -> RecallEpisode | None:
    if rng.random() < general_fraction:
        return _build_general_episode(rng, general_level, max_memory_slots=max_memory_slots)
    return _build_correction_episode(
        rng,
        correction_level,
        max_memory_slots=max_memory_slots,
        reverse_fraction=reverse_fraction,
    )


def _build_batch(
    rng: random.Random,
    count: int,
    *,
    correction_level: int,
    general_level: int,
    general_fraction: float,
    reverse_fraction: float,
    max_memory_slots: int,
) -> list[RecallEpisode]:
    episodes: list[RecallEpisode] = []
    while len(episodes) < count:
        episode = _build_episode(
            rng,
            correction_level=correction_level,
            general_level=general_level,
            general_fraction=general_fraction,
            reverse_fraction=reverse_fraction,
            max_memory_slots=max_memory_slots,
        )
        if episode is not None:
            episodes.append(episode)
    return episodes


def _encode_values(
    world_model: GemmaWorldModel,
    tokenizer: Any,
    episodes: list[RecallEpisode],
    *,
    device: torch.device,
    max_length: int,
    requires_grad: bool,
    max_memory_slots: int,
) -> tuple[Tensor, Tensor]:
    """Pool every statement in every episode into a value vector, batched
    across the whole step. Same last-token pooling encode_query uses (not
    mean-pooling) so write-side and read-side (ConversationalAnswerer.
    _encode_value) stay consistent -- both go through the same LoRA'd
    model. Returns (values, mask) padded to [episodes, max_memory_slots, d].

    max_memory_slots must match ConversationalAnswerer's top_k at inference
    exactly, not just be "big enough": GemmaWorldModel never passes
    position_ids, so Gemma3 falls back to a plain arange(seq_len) that
    ignores attention_mask entirely (confirmed against
    transformers.models.gemma3.modeling_gemma3.Gemma3TextModel.forward) --
    every token after the memory block gets shifted by the *total* memory
    slot count, valid or padded, not by how many slots actually held real
    content. Training with 8 slots and answering with top_k=4 live silently
    trains and infers under different position embeddings for every real
    token in the prompt, which measured as a systematic wrong-answer
    pattern (converging on "1" regardless of the true corrected value) on
    the exact live multi-turn correction scenario this training targets."""
    flat_texts = [
        statement for episode in episodes for statement in episode.statements
    ]
    encoded = tokenizer(
        flat_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        output = world_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states[-1]
        positions = attention_mask.long().sum(dim=1).sub(1).clamp_min(0)
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        # The port itself stays fp32 (see train()'s comment); cast its
        # input to match rather than downcasting the port.
        pooled = hidden[rows, positions].float()

    hidden_dimension = pooled.shape[-1]
    values = torch.zeros(
        len(episodes),
        max_memory_slots,
        hidden_dimension,
        device=device,
        dtype=pooled.dtype,
    )
    mask = torch.zeros(len(episodes), max_memory_slots, device=device, dtype=torch.bool)
    cursor = 0
    for row, episode in enumerate(episodes):
        count = len(episode.statements)
        values[row, :count] = pooled[cursor : cursor + count]
        mask[row, :count] = True
        cursor += count
    return values, mask


def _recency(mask: Tensor) -> Tensor:
    """Normalized write-order position per row: 0 for the first statement,
    1 for the last valid one -- the explicit signal a correction chain
    needs, since BasinMemory-style retrieval has no notion of write order
    on its own (see ConversationalAnswerer._normalize_recency, the
    inference-time counterpart)."""
    counts = mask.sum(dim=1, keepdim=True).clamp_min(1)
    positions = torch.arange(mask.shape[1], device=mask.device, dtype=torch.float32)
    return (positions[None, :] / (counts - 1).clamp_min(1)).clamp(max=1.0) * mask


def _scheduled_sampling_decode_loss(
    world_model: GemmaWorldModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    values: Tensor,
    memory_mask: Tensor,
    weights: Tensor,
    recency: Tensor,
    labels: Tensor,
    *,
    sampling_prob: float,
) -> Tensor:
    """Teacher-forced training only ever shows the model correct preceding
    answer tokens as input, never its own predictions -- exactly what free
    generation can't provide, since a wrong token early on has to be lived
    with, not corrected by a coach. This measurably matters here:
    eval_teacher_forced_accuracy sat at 85-97% for 8,000 steps of continued
    training while eval_generation_accuracy stayed flat around 20%, and
    switching decoding strategy (greedy vs. beam search) changed nothing --
    the learned distribution itself is confidently wrong under free
    generation, not just unluckily decoded.

    Implements scheduled sampling (Bengio et al. 2015) in the standard,
    cheap two-forward-pass form for transformers (no autoregressive
    step-by-step re-encoding): a first, no-grad pass gets the model's own
    next-token predictions at every position; some fraction of the answer
    positions (not the first, which always has to condition on the real
    prompt) then get their *input* embedding replaced with the model's own
    predicted-token embedding from the preceding position instead of the
    ground-truth token's embedding, before a second, gradient-carrying pass
    computes loss against the unchanged ground-truth labels. sampling_prob
    is 0 early in training (pure teacher forcing, correct when the model's
    own predictions are still mostly noise) and ramps up as training
    progresses (see train()'s schedule), progressively narrowing the gap
    between what training shows the model and what real generation does.
    """
    if sampling_prob <= 0.0:
        return world_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            memory_values=values,
            memory_mask=memory_mask,
            retrieval_weights=weights,
            recency=recency,
        ).loss

    prepared = world_model.prepare_inputs(
        input_ids,
        attention_mask,
        values,
        labels=labels,
        memory_mask=memory_mask,
        retrieval_weights=weights,
        recency=recency,
    )
    with torch.no_grad():
        first_pass = world_model.model(
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=prepared.attention_mask,
            return_dict=True,
        )
        predicted_ids = first_pass.logits.argmax(dim=-1)

    answer_mask = prepared.labels != -100
    # A position is substitutable only if it and its immediate predecessor
    # are both answer tokens -- the first answer token's input is always
    # the real last prompt token, never a substitution candidate, since
    # there is no "answer so far" to have gotten wrong yet.
    substitutable = answer_mask.clone()
    substitutable[:, 1:] &= answer_mask[:, :-1]
    substitutable[:, 0] = False
    substitute = substitutable & (
        torch.rand(substitutable.shape, device=substitutable.device) < sampling_prob
    )

    embedding_table = world_model.model.get_input_embeddings()
    predicted_embeds = embedding_table(predicted_ids).to(prepared.inputs_embeds.dtype)
    # predicted_ids[:, t] is the model's own prediction *for* position t+1;
    # substituting at position p therefore reads the prediction at p - 1.
    shifted_predicted_embeds = torch.zeros_like(prepared.inputs_embeds)
    shifted_predicted_embeds[:, 1:] = predicted_embeds[:, :-1]
    mixed_embeds = torch.where(
        substitute.unsqueeze(-1), shifted_predicted_embeds, prepared.inputs_embeds
    )

    second_pass = world_model.model(
        inputs_embeds=mixed_embeds,
        attention_mask=prepared.attention_mask,
        labels=prepared.labels,
        return_dict=True,
    )
    return second_pass.loss


def _tokenize_qa(
    tokenizer: Any,
    episodes: list[RecallEpisode],
    *,
    device: torch.device,
    max_length: int,
    with_answer: bool,
) -> tuple[Tensor, Tensor, Tensor | None]:
    conversations = [
        [{"role": "user", "content": episode.question}] for episode in episodes
    ]
    prompts = [
        tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        for conversation in conversations
    ]
    texts = prompts
    if with_answer:
        texts = [
            tokenizer.apply_chat_template(
                [*conversation, {"role": "assistant", "content": episode.answer}],
                tokenize=False,
                add_generation_prompt=False,
            )
            for conversation, episode in zip(conversations, episodes, strict=True)
        ]
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if not with_answer:
        return input_ids, attention_mask, None
    prompt_lengths = [
        len(
            tokenizer(
                prompt, add_special_tokens=False, truncation=True, max_length=max_length
            )["input_ids"]
        )
        for prompt in prompts
    ]
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    for row, length in enumerate(prompt_lengths):
        labels[row, : min(length, labels.shape[1])] = -100
    return input_ids, attention_mask, labels


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if args.scheduled_sampling_warmup_steps is None:
        args.scheduled_sampling_warmup_steps = max(
            1, args.steps - args.scheduled_sampling_start_step
        )
    rng = random.Random(args.seed)
    eval_rng = random.Random(args.eval_seed)
    eval_episodes = _build_batch(
        eval_rng,
        args.eval_episodes,
        correction_level=args.level,
        general_level=args.general_level,
        general_fraction=args.general_fraction,
        reverse_fraction=args.reverse_fraction,
        max_memory_slots=args.max_memory_slots,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    world_model = load_gemma_world_model(args.model, device=args.device, dtype=dtype)
    # The port stays fp32 -- prepare_inputs already casts its *output* to
    # the model's bf16 embeddings before combining them, so fp32 here loses
    # nothing. Casting the whole module to bf16 (as an earlier version of
    # this script did) also casts the scalar `gate` parameter, and Adam's
    # small per-step updates to a bf16 scalar underflow to nothing --
    # confirmed by a run where port_gate was bit-identical to its init
    # value after 150 steps.
    world_model.port = LatentMemoryPort(
        state_dimension=world_model.port.hidden_dimension,
        hidden_dimension=world_model.port.hidden_dimension,
    ).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if str(initial["model"]) != args.model:
            raise ValueError("initial checkpoint uses a different base model")
        world_model.port.load_state_dict(initial["port"])
        print(f"initialized port from {args.init_checkpoint}", flush=True)

    enable_upper_layer_lora(world_model, rank=args.lora_rank, alpha=args.lora_alpha)
    if args.init_checkpoint and initial.get("lora") is not None:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(world_model.model, initial["lora"])
    world_model.model.train()
    world_model.port.train()

    trainable_model_parameters = [
        parameter for parameter in world_model.model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [*world_model.port.parameters(), *trainable_model_parameters],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []

    for step in range(1, args.steps + 1):
        episodes = _build_batch(
            rng,
            args.batch_size,
            correction_level=args.level,
            general_level=args.general_level,
            general_fraction=args.general_fraction,
            reverse_fraction=args.reverse_fraction,
            max_memory_slots=args.max_memory_slots,
        )
        values, memory_mask = _encode_values(
            world_model,
            tokenizer,
            episodes,
            device=device,
            max_length=args.max_length,
            requires_grad=True,
            max_memory_slots=args.max_memory_slots,
        )
        recency = _recency(memory_mask)
        weights = memory_mask.to(values.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)

        input_ids, attention_mask, labels = _tokenize_qa(
            tokenizer, episodes, device=device, max_length=args.max_length, with_answer=True
        )
        ramp_progress = (step - args.scheduled_sampling_start_step) / max(
            1, args.scheduled_sampling_warmup_steps
        )
        sampling_prob = args.scheduled_sampling_max_prob * min(1.0, max(0.0, ramp_progress))
        decode_loss = _scheduled_sampling_decode_loss(
            world_model,
            input_ids,
            attention_mask,
            values,
            memory_mask,
            weights,
            recency,
            labels,
            sampling_prob=sampling_prob,
        )

        # Hard negative: shuffle each episode's memory set with another
        # episode's (different answer, by construction of the batch's
        # independent draws) -- if the port/LoRA is actually reading the
        # injected values rather than pattern-matching the question alone,
        # decode loss should be markedly worse with the wrong memory.
        shuffled_indices = torch.roll(torch.arange(len(episodes)), shifts=1)
        shuffled = world_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            memory_values=values.index_select(0, shuffled_indices.to(device)),
            memory_mask=memory_mask.index_select(0, shuffled_indices.to(device)),
            retrieval_weights=weights.index_select(0, shuffled_indices.to(device)),
            recency=recency.index_select(0, shuffled_indices.to(device)),
        ).loss
        contrast = torch.relu(decode_loss - shuffled + args.contrast_margin)

        loss = decode_loss + args.contrast_weight * contrast
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*world_model.port.parameters(), *trainable_model_parameters], 1.0
        )
        optimizer.step()

        row = {
            "step": step,
            "total": float(loss.detach()),
            "decode": float(decode_loss.detach()),
            "contrast": float(contrast.detach()),
            "port_gate": float(torch.sigmoid(world_model.port.gate.detach())),
            "sampling_prob": sampling_prob,
        }
        if step == 1 or step % args.log_every == 0:
            print(json.dumps(row), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            eval_row = _evaluate(
                world_model,
                tokenizer,
                eval_episodes,
                device,
                args.max_length,
                max_memory_slots=args.max_memory_slots,
            )
            eval_row.update(
                _evaluate_generation(
                    world_model,
                    tokenizer,
                    eval_episodes[: args.eval_generation_episodes],
                    device,
                    max_memory_slots=args.max_memory_slots,
                )
            )
            eval_row["step"] = step
            print(json.dumps(eval_row), flush=True)
            row.update(eval_row)
        history.append(row)

    lora_state = None
    from peft import get_peft_model_state_dict

    lora_state = get_peft_model_state_dict(world_model.model)
    torch.save(
        {
            "format_version": 1,
            "state_schema": "world-recall-port-v1",
            "model": args.model,
            "port": world_model.port.state_dict(),
            "lora": lora_state,
            "optimizer": optimizer.state_dict(),
            "history": history,
            "args": vars(args),
        },
        output,
    )
    print(f"saved recall port checkpoint to {output}", flush=True)


@torch.no_grad()
def _evaluate(
    world_model: GemmaWorldModel,
    tokenizer: Any,
    episodes: list[RecallEpisode],
    device: torch.device,
    max_length: int,
    *,
    max_memory_slots: int,
) -> dict[str, float]:
    world_model.model.eval()
    values, memory_mask = _encode_values(
        world_model,
        tokenizer,
        episodes,
        device=device,
        max_length=max_length,
        requires_grad=False,
        max_memory_slots=max_memory_slots,
    )
    recency = _recency(memory_mask)
    weights = memory_mask.to(values.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    input_ids, attention_mask, labels = _tokenize_qa(
        tokenizer, episodes, device=device, max_length=max_length, with_answer=True
    )
    # world_model(...) returns the underlying causal LM's output, whose
    # logits are as long as the *combined* (memory-token-prefixed)
    # sequence, not the raw input_ids/labels passed in -- prepare_inputs
    # separately, so the labels compared against argmax are the same ones
    # the loss was actually computed against.
    prepared = world_model.prepare_inputs(
        input_ids,
        attention_mask,
        values,
        labels=labels,
        memory_mask=memory_mask,
        retrieval_weights=weights,
        recency=recency,
    )
    output = world_model.model(
        inputs_embeds=prepared.inputs_embeds,
        attention_mask=prepared.attention_mask,
        labels=prepared.labels,
        return_dict=True,
    )
    predicted = output.logits[:, :-1].argmax(dim=-1)
    targets = prepared.labels[:, 1:]
    valid = targets != -100
    correct_tokens = ((predicted == targets) | ~valid).all(dim=1)
    world_model.model.train()
    return {
        "eval_decode_loss": float(output.loss.detach()),
        # Teacher-forced: each predicted token is conditioned on the
        # *correct* preceding answer tokens, not on what the model itself
        # would have generated so far. This is the standard, cheap
        # per-step training signal, but it is an optimistic proxy for real
        # quality -- see eval_generation_accuracy below, which measures the
        # thing that actually matters for the demo.
        "eval_teacher_forced_accuracy": float(correct_tokens.float().mean()),
    }


@torch.no_grad()
def _evaluate_generation(
    world_model: GemmaWorldModel,
    tokenizer: Any,
    episodes: list[RecallEpisode],
    device: torch.device,
    *,
    max_memory_slots: int,
    max_new_tokens: int = 32,
) -> dict[str, float]:
    """Free-generation accuracy: no coaching from the correct answer at
    each step, matching how ConversationalAnswerer actually calls
    world_model.generate() in production. A model can score high on
    teacher-forced next-token accuracy while still failing badly here,
    because teacher-forcing recovers from a wrong token at step t by being
    handed the *correct* token at t+1 regardless -- free generation has to
    live with whatever it actually produced. One episode at a time (not
    batched): transformers requires left-padding for correct batched
    generation, but this script tokenizes right-padded throughout for the
    teacher-forced path above, and this metric is for periodic eval, not
    the training hot loop, so the slower one-at-a-time cost is acceptable.
    """
    world_model.model.eval()
    correct = 0
    for episode in episodes:
        values, memory_mask = _encode_values(
            world_model,
            tokenizer,
            [episode],
            device=device,
            max_length=96,
            requires_grad=False,
            max_memory_slots=max_memory_slots,
        )
        recency = _recency(memory_mask)
        weights = memory_mask.to(values.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": episode.question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        tokens = tokenizer(chat, add_special_tokens=False, return_tensors="pt")
        tokens = {name: value.to(device) for name, value in tokens.items()}
        output_ids = world_model.generate(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
            memory_values=values,
            memory_mask=memory_mask,
            retrieval_weights=weights,
            recency=recency,
            generation_kwargs={"max_new_tokens": max_new_tokens, "do_sample": False},
        )
        generated = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        if generated.rstrip(".").lower() == episode.answer.rstrip(".").lower():
            correct += 1
    world_model.model.train()
    return {"eval_generation_accuracy": correct / len(episodes)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="checkpoints/world/recall-port.pt")
    parser.add_argument("--model", default="google/gemma-3-270m-it")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument(
        "--scheduled-sampling-max-prob",
        type=float,
        default=0.0,
        help=(
            "Probability, at full ramp, of substituting the model's own "
            "predicted token (from a no-grad first pass) for the "
            "ground-truth token when feeding a non-first answer position's "
            "input embedding in the gradient-carrying second pass. 0 "
            "disables scheduled sampling entirely (pure teacher forcing, "
            "the original behavior). See _scheduled_sampling_decode_loss. "
            "0.4 from step 0 measurably hurt (0.125 vs. 0.2375 "
            "eval_generation_accuracy on an 80-episode comparison against "
            "a plain-teacher-forced checkpoint of otherwise identical "
            "budget) -- early in training the model's own predictions are "
            "close to random, so substituting them trains against near-"
            "garbage context before the base task is even learned. A "
            "lower value and/or --scheduled-sampling-start-step > 0 is "
            "worth trying before assuming the technique itself doesn't "
            "help here."
        ),
    )
    parser.add_argument(
        "--scheduled-sampling-start-step",
        type=int,
        default=0,
        help=(
            "Step at which the sampling_prob ramp begins (0 before this "
            "many steps). Letting the model reach reasonable teacher-forced "
            "accuracy on pure teacher forcing first, before introducing its "
            "own (by-then less noisy) predictions as input, is the more "
            "standard reading of Bengio et al. 2015's schedule than "
            "ramping from step 0."
        ),
    )
    parser.add_argument(
        "--scheduled-sampling-warmup-steps",
        type=int,
        default=None,
        help=(
            "Steps, after --scheduled-sampling-start-step, to linearly ramp "
            "sampling_prob from 0 to --scheduled-sampling-max-prob. "
            "Defaults to --steps (ramp across the whole remaining run)."
        ),
    )
    parser.add_argument(
        "--max-memory-slots",
        type=int,
        default=8,
        help=(
            "Fixed memory-token slot count injected per example (padded, "
            "not just a cap). GemmaWorldModel never passes position_ids, "
            "so Gemma3 defaults to arange(seq_len), ignoring attention_mask "
            "-- every later token's position embedding depends on this "
            "exact count, valid slots or padding alike. Must match "
            "ConversationalAnswerer's top_k at inference exactly; saved "
            "into the checkpoint's args so world_app.py can read it back "
            "rather than relying on both sides hardcoding the same number."
        ),
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        choices=range(1, 7),
        help="distractor_reasoning.correction difficulty (correction-tracking episodes).",
    )
    parser.add_argument(
        "--general-level",
        type=int,
        default=5,
        choices=range(1, 11),
        help=(
            "distractor_reasoning.generate difficulty (static many-facts "
            "episodes: six distractor kinds, category-generalization "
            "questions from level 5). Capped at 5 by this script's own "
            "always-answerable, single-integer-answer contract -- "
            "aggregation/zero/reported-fact levels (6+) are real but "
            "deliberately not folded in here yet."
        ),
    )
    parser.add_argument(
        "--general-fraction",
        type=float,
        default=0.5,
        help=(
            "Fraction of episodes drawn from distractor_reasoning.generate "
            "instead of distractor_reasoning.correction. correction.py "
            "alone is one template family ('{subject} {relation} "
            "{quantity} {noun_phrase}.' plus a fixed correction-prefix "
            "set); this generator's distinct fact/distractor/question "
            "shapes exist specifically to break single-template "
            "overfitting risk (2026-09-02 dev journal: hand-typed live "
            "sessions scored well below this generator's own held-out "
            "eval when training used correction.py exclusively). 0 "
            "disables the general generator entirely (the original "
            "correction-only behavior)."
        ),
    )
    parser.add_argument(
        "--reverse-fraction",
        type=float,
        default=0.3,
        help=(
            "Fraction of distractor_reasoning.correction episodes framed "
            "as a reverse query -- 'Who owns 8 guitars?' (answer: the "
            "subject's name) or 'What does Kevin own 8 of?' (answer: the "
            "noun) -- instead of the forward 'How many guitars does Kevin "
            "own?' (answer: the quantity). Every generator in this script "
            "previously asked only the forward direction; live testing "
            "found this gap directly (a checkpoint that answered the "
            "forward phrasing of a fact perfectly still failed 'who has "
            "17 cats?' about the identical fact). Ported from the text "
            "construction already built and validated in "
            "sonic_reverse_query_probe.py / sonic_noun_target_probe.py "
            "(see session_builder.py's own docstring), not a new "
            "generator. Only applies to correction.py episodes, not the "
            "general generator, which has no notion of query direction "
            "at all. 0 disables (forward-only, the original behavior)."
        ),
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--contrast-weight", type=float, default=0.10)
    parser.add_argument("--contrast-margin", type=float, default=0.4)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument(
        "--eval-generation-episodes",
        type=int,
        default=24,
        help=(
            "How many of --eval-episodes to also free-generate on (one at "
            "a time, so keep this well below --eval-episodes if steps are "
            "frequent). See _evaluate_generation's docstring for why this "
            "and eval_teacher_forced_accuracy can diverge sharply."
        ),
    )
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=31)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
