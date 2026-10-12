"""Stability check for fine-tuning Sonic jointly with ManifoldFlow -
the direct prerequisite for a real "does ManifoldFlow help Sonic converge
faster than vanilla Sonic" comparison.

Every probe before this one kept Sonic **frozen** (`sonic-pretrained-memory-probe.md`,
`sonic-pretrained-memory-discrimination-probe.md`, and the learned-retrieval
probe all trained only `SonicMemoryPort` and/or `SonicKeyProjection`
against a fixed backbone). None of them has ever backpropagated into
Sonic's own weights while the memory mechanism is active. A convergence
race requires exactly that - "does Sonic converge faster" only means
something if Sonic's parameters are moving - so before running one, this
checks the much cheaper failure mode first: does joint training even
stay numerically well-behaved, or does combining full-model fine-tuning
with a freshly-initialized port and key projection destabilize either
side.

This is *not* an accuracy or convergence-speed benchmark. It runs one
condition (Sonic + `SonicMemoryPort` + `SonicKeyProjection`, all
trainable) for a short, fast budget and watches four things a longer run
would be expensive to discover trouble in only after the fact:

1. **No NaN/Inf** anywhere in the loss or in any parameter's gradient,
   checked explicitly every step, not just eyeballed from a loss curve.
2. **Gradient norms stay bounded** for Sonic's own parameters and for
   the memory components separately - reported separately because they
   are expected to behave very differently (Sonic's gradients should be
   small relative to a pretrained model's own weight scale; the freshly
   initialized port/projection's gradients should look like the earlier
   frozen-backbone probes').
3. **Task loss actually decreases** - confirms training is doing
   something sensible, not just failing to explode.
4. **Held-out ordinary-language loss does not blow up** - a fixed set of
   generic English sentences, unrelated to the synthetic recall task,
   scored before and periodically during training. This is a cheap,
   partial proxy for catastrophic forgetting (not a real perplexity
   benchmark on held-out corpus data - there is no such corpus wired up
   yet) - a sharp rise here would be a real warning sign worth stopping
   for; a small run being flat is reassuring but not proof a longer
   fine-tune stays safe.

Sonic and the memory components use separate learning rates via Adam
parameter groups - a much smaller one for Sonic's already-pretrained
weights, a larger one for the freshly-initialized port/projection - the
standard practice for fine-tuning a pretrained backbone alongside new
modules, and itself part of what "stability" means here: using one
shared learning rate for both would be the more likely way to destabilize
training, not a neutral default.

Gradient checkpointing is disabled (`gradient_checkpointing_disable()`):
on CPU, with no GPU memory pressure to trade against, checkpointing only
adds a second forward pass per layer during backward for no benefit here
- disabling it roughly halved wall-clock time in a timing check.

Reuses the discrimination/learned-retrieval probes' episode generation
(same entities/objects, same fact/query template, same
two-forward-pass-per-fact separation with fresh state) so this probe's
task is directly comparable to theirs - only what's trainable differs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from manifoldflow.continuous_memory import BasinMemory, Source
from manifoldflow.sonic_key_projection import (
    SonicKeyProjection,
    SonicKeyProjectionConfig,
    info_nce_retrieval_loss,
)
from manifoldflow.sonic_memory_port import SonicMemoryPort, SonicMemoryPortConfig
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel

_DEFAULT_CHECKPOINT = Path("checkpoints/sonic_ssm/sonic_ssm_128k_steps.pt")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TOKENIZER_DIR = (
    _PROJECT_ROOT / "resources" / "tokenizers" / "gtr_arw_english_50k"
)

_ENTITIES = ["Neral", "Tovan", "Mira", "Selka", "Bren", "Dorna", "Ilvet", "Quorin"]
_OBJECTS = ["glider", "lantern", "compass", "crate", "whistle", "map", "key", "vial"]

# A fixed, task-unrelated sample - not a real held-out corpus, just enough
# to notice if fine-tuning is visibly wrecking ordinary language modeling
# during a short stability run.
_HELD_OUT_SENTENCES = [
    "The river winds slowly through the valley before reaching the sea.",
    "Scientists have long debated the origins of the solar system.",
    "She opened the window to let the morning air into the room.",
    "The committee will meet on Thursday to discuss the new proposal.",
    "A gentle rain fell over the city as the streetlights flickered on.",
    "He spent the afternoon repairing the old wooden fence behind the barn.",
    "The recipe calls for two cups of flour and a pinch of salt.",
    "Economic growth slowed slightly in the third quarter of the year.",
    "The children built a small fort out of blankets and cushions.",
    "Researchers published their findings in a peer-reviewed journal.",
]


@dataclass(frozen=True)
class Fact:
    entity: str
    obj: str
    fact_ids: list[int]


@dataclass(frozen=True)
class Episode:
    facts: list[Fact]
    query_entity: str
    query_ids: list[int]
    answer_ids: list[int]

    @property
    def answer_start_index(self) -> int:
        return len(self.query_ids)

    @property
    def full_query_and_answer_ids(self) -> list[int]:
        return self.query_ids + self.answer_ids


def _generate_episode(
    generator: random.Random, tokenizer: Any, facts_per_episode: int
) -> Episode:
    entities = generator.sample(_ENTITIES, facts_per_episode)
    facts = [
        Fact(
            entity=entity,
            obj=(obj := generator.choice(_OBJECTS)),
            fact_ids=tokenizer.encode(
                f"{entity} owns the {obj}.", add_special_tokens=True
            ),
        )
        for entity in entities
    ]
    query_fact = generator.choice(facts)
    query_ids = tokenizer.encode(
        f"What does {query_fact.entity} own?", add_special_tokens=True
    )
    answer_ids = tokenizer.encode(f" the {query_fact.obj}", add_special_tokens=False)
    return Episode(
        facts=facts,
        query_entity=query_fact.entity,
        query_ids=query_ids,
        answer_ids=answer_ids,
    )


def _episode_step(
    *,
    model: SonicSSMLanguageModel,
    episode: Episode,
    port: SonicMemoryPort,
    key_projection: SonicKeyProjection,
    retrieval_loss_weight: float,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]:
    """Unlike the frozen-backbone probes, Sonic's own forward passes are
    *not* wrapped in `no_grad` here - gradient must reach Sonic's weights
    for this to be a fine-tuning stability check at all."""
    bank = BasinMemory(dimension=model.config.d_model, novelty_threshold=0.0)
    entity_to_basin: dict[str, int] = {}
    fact_keys: list[torch.Tensor] = []
    correct_index = None
    for index, fact in enumerate(episode.facts):
        fact_tensor = torch.tensor([fact.fact_ids])
        fact_hidden, _fact_states = model.forward_hidden(fact_tensor)
        pooled = fact_hidden[0].mean(dim=0, keepdim=True)
        fact_key = key_projection(pooled).squeeze(0)
        fact_keys.append(fact_key)
        basin_id = int(bank.write(fact_key.detach(), "fact", Source("user")))
        entity_to_basin[fact.entity] = basin_id
        if fact.entity == episode.query_entity:
            correct_index = index
    assert correct_index is not None

    qa_tensor = torch.tensor([episode.full_query_and_answer_ids])
    hidden, _states = model.forward_hidden(qa_tensor)
    answer_hidden = hidden[:, episode.answer_start_index :, :]

    query_pooled = hidden[0, : episode.answer_start_index].mean(dim=0, keepdim=True)
    query_key = key_projection(query_pooled).squeeze(0)
    hits = bank.read(query_key.detach(), top_k=1)
    retrieved = (
        bank.centers[hits[0].basin_id].unsqueeze(0)
        if hits
        else torch.zeros(1, model.config.d_model)
    )
    retrieval_correct = bool(
        hits and hits[0].basin_id == entity_to_basin[episode.query_entity]
    )
    retrieval_loss = info_nce_retrieval_loss(
        query_key, torch.stack(fact_keys), correct_index, temperature=temperature
    )
    answer_hidden = port(answer_hidden, retrieved)

    logits = model.logits_from_hidden(answer_hidden)
    target_ids = torch.tensor(episode.answer_ids)
    predicted_logits = logits[0, :-1, :]
    predicted_targets = target_ids[1:]
    decode_loss = F.cross_entropy(predicted_logits, predicted_targets)
    answer_correct = bool(
        (predicted_logits.argmax(dim=-1) == predicted_targets).all().item()
    )

    total_loss = decode_loss + retrieval_loss_weight * retrieval_loss
    return total_loss, decode_loss, retrieval_loss, answer_correct, retrieval_correct


@torch.no_grad()
def _held_out_lm_loss(model: SonicSSMLanguageModel, tokenizer: Any) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for sentence in _HELD_OUT_SENTENCES:
        ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True)])
        hidden, _ = model.forward_hidden(ids)
        logits = model.logits_from_hidden(hidden)
        predicted_logits = logits[0, :-1, :]
        predicted_targets = ids[0, 1:]
        losses.append(F.cross_entropy(predicted_logits, predicted_targets).item())
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def _grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += parameter.grad.detach().float().norm().item() ** 2
    return math.sqrt(total)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-dir", type=Path, default=_DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--held-out-every", type=int, default=25)
    parser.add_argument("--sonic-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-seed", type=int, default=29)
    parser.add_argument("--port-seed", type=int, default=17)
    parser.add_argument("--port-rank", type=int, default=64)
    parser.add_argument("--gate-init-bias", type=float, default=0.0)
    parser.add_argument("--episodes-per-step", type=int, default=4)
    parser.add_argument("--facts-per-episode", type=int, default=2)
    parser.add_argument("--retrieval-loss-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= args.facts_per_episode <= len(_ENTITIES):
        raise ValueError(f"facts_per_episode must be in [1, {len(_ENTITIES)}]")

    transformers = _require_transformers()

    checkpoint = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=False
    )
    model_config = SonicSSMConfig(**checkpoint["model_config"])
    model = SonicSSMLanguageModel(model_config)
    model.load_state_dict(checkpoint["model"])
    model.gradient_checkpointing_disable()
    model.train()  # Sonic is fully trainable in this probe - not frozen.

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    if len(tokenizer) != model_config.vocab_size:
        raise ValueError(
            f"tokenizer vocab size {len(tokenizer)} does not match "
            f"model vocab_size {model_config.vocab_size}"
        )

    torch.manual_seed(args.port_seed)
    port = SonicMemoryPort(
        SonicMemoryPortConfig(
            d_model=model_config.d_model,
            memory_dim=model_config.d_model,
            rank=args.port_rank,
            gate_init_bias=args.gate_init_bias,
        )
    )
    key_projection = SonicKeyProjection(
        SonicKeyProjectionConfig(dim=model_config.d_model)
    )

    sonic_parameters = list(model.parameters())
    head_parameters = list(port.parameters()) + list(key_projection.parameters())
    optimizer = torch.optim.Adam(
        [
            {"params": sonic_parameters, "lr": args.sonic_learning_rate},
            {"params": head_parameters, "lr": args.head_learning_rate},
        ]
    )

    initial_held_out_loss = _held_out_lm_loss(model, tokenizer)
    history: list[dict[str, Any]] = []
    train_generator = random.Random(args.data_seed)
    nan_or_inf_detected = False
    detection_step: int | None = None

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        step_losses = []
        step_decode_losses = []
        step_retrieval_losses = []
        step_correct = 0
        step_retrieval_correct = 0
        for _ in range(args.episodes_per_step):
            episode = _generate_episode(
                train_generator, tokenizer, args.facts_per_episode
            )
            total_loss, decode_loss, retrieval_loss, correct, retrieval_correct = (
                _episode_step(
                    model=model,
                    episode=episode,
                    port=port,
                    key_projection=key_projection,
                    retrieval_loss_weight=args.retrieval_loss_weight,
                    temperature=args.temperature,
                )
            )
            (total_loss / args.episodes_per_step).backward()
            step_losses.append(total_loss.item())
            step_decode_losses.append(decode_loss.item())
            step_retrieval_losses.append(retrieval_loss.item())
            step_correct += correct
            step_retrieval_correct += retrieval_correct

        sonic_grad_norm = _grad_norm(sonic_parameters)
        head_grad_norm = _grad_norm(head_parameters)
        step_nan_or_inf = not (
            math.isfinite(sonic_grad_norm) and math.isfinite(head_grad_norm)
        )
        if step_nan_or_inf and not nan_or_inf_detected:
            nan_or_inf_detected = True
            detection_step = step

        optimizer.step()

        if step % args.log_every == 0 or step == 1 or step_nan_or_inf:
            row: dict[str, Any] = {
                "step": step,
                "total_loss_mean": sum(step_losses) / len(step_losses),
                "decode_loss_mean": sum(step_decode_losses) / len(step_decode_losses),
                "retrieval_loss_mean": sum(step_retrieval_losses)
                / len(step_retrieval_losses),
                "answer_accuracy": step_correct / args.episodes_per_step,
                "retrieval_accuracy": step_retrieval_correct / args.episodes_per_step,
                "sonic_grad_norm": sonic_grad_norm,
                "head_grad_norm": head_grad_norm,
                "nan_or_inf_this_step": step_nan_or_inf,
            }
            if step % args.held_out_every == 0 or step == 1 or step == args.steps:
                row["held_out_lm_loss"] = _held_out_lm_loss(model, tokenizer)
            history.append(row)

        if nan_or_inf_detected and step - detection_step >= 5:
            break  # stop early once instability is confirmed, not just a one-off

    final_held_out_loss = _held_out_lm_loss(model, tokenizer)
    output = {
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_update": checkpoint.get("update"),
        "model_config": model_config.to_dict(),
        "steps_requested": args.steps,
        "steps_completed": history[-1]["step"] if history else 0,
        "sonic_learning_rate": args.sonic_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "episodes_per_step": args.episodes_per_step,
        "facts_per_episode": args.facts_per_episode,
        "retrieval_loss_weight": args.retrieval_loss_weight,
        "temperature": args.temperature,
        "initial_held_out_lm_loss": initial_held_out_loss,
        "final_held_out_lm_loss": final_held_out_loss,
        "held_out_lm_loss_delta": final_held_out_loss - initial_held_out_loss,
        "nan_or_inf_detected": nan_or_inf_detected,
        "nan_or_inf_detection_step": detection_step,
        "history": history,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def _require_transformers() -> Any:
    try:
        import transformers
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "This probe requires the qwen extra (shared tokenizer dependency): "
            "python -m pip install -e '.[qwen]'"
        ) from error
    return transformers


def main() -> None:
    args = _parser().parse_args()
    try:
        metrics = run(args)
    except BaseException:
        sys.stdout.flush()
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    summary = {k: v for k, v in metrics.items() if k != "history"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
