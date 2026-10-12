"""Third training-viability probe: multi-fact discrimination, not just
decode-ability.

`sonic_pretrained_memory_probe.py` proved the mechanism can *decode* a
written value once retrieval is trivial - its bank held exactly one fact
per episode, so `bank.read()` always returned the only basin present
regardless of how good the query key was. That is a real result (100%
accuracy, frozen checkpoint, trained port) but a narrow one: it never
tested whether the system can find the *right* fact among competitors,
which is the actual point of a persistent memory bank - "recall and
persistent state working" (the user's own framing for why this line of
work matters more than staying spec-pure) means more than one fact has
to coexist and be told apart.

This probe writes several facts about *different* entities into the same
per-episode bank (still fresh per episode - a longer-lived,
multi-episode-persistent bank is a further step this one does not take),
then queries about one of them and checks two things separately:
`retrieval_accuracy` (did `bank.read()` return the basin actually written
for the queried entity - a real, non-trivial question now that the bank
holds several basins) and `answer_accuracy` (did the port + frozen Sonic
then decode the right object from whatever was retrieved). Splitting
these two numbers matters because they can fail independently: retrieval
could work while decoding fails, or vice versa, and the fix for each is
different.

Same frozen-Sonic, trained-port-only design as the pretrained probe (see
that script's docstring for why freezing sidesteps design-v0.4 SS6.3's
detached-write question rather than resolving it), same real checkpoint
and tokenizer, same two-forward-pass-per-fact separation with fresh state.
`novelty_threshold=0.0` on the per-episode `BasinMemory` means every
distinct fact write gets its own basin (only a bit-exact duplicate key
would merge), so `facts_per_episode` writes reliably produce
`facts_per_episode` basins - this probe is about whether the *query*
finds the right one among them, not about `BasinMemory`'s own novelty
threshold, which existing Phase 2 probes already cover.
"""

from __future__ import annotations

import argparse
import json
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
from manifoldflow.sonic_memory_port import SonicMemoryPort, SonicMemoryPortConfig
from manifoldflow.sonic_ssm import SonicSSMConfig, SonicSSMLanguageModel

_DEFAULT_CHECKPOINT = Path("checkpoints/sonic_ssm/sonic_ssm_128k_steps.pt")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_TOKENIZER_DIR = (
    _PROJECT_ROOT / "resources" / "tokenizers" / "gtr_arw_english_50k"
)

_ENTITIES = ["Neral", "Tovan", "Mira", "Selka", "Bren", "Dorna", "Ilvet", "Quorin"]
_OBJECTS = ["glider", "lantern", "compass", "crate", "whistle", "map", "key", "vial"]


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
    device: str,
    port: SonicMemoryPort | None,
) -> tuple[torch.Tensor, bool, bool | None]:
    """Returns (loss, answer_correct, retrieval_correct). `port=None` is
    the no-memory control condition (`retrieval_correct=None` - there is
    no bank in that condition). Sonic is always frozen (`no_grad`); the
    caller wraps eval calls in `torch.no_grad()` so the port's own
    forward doesn't build a graph either."""
    retrieval_correct: bool | None = None
    if port is not None:
        bank = BasinMemory(dimension=model.config.d_model, novelty_threshold=0.0)
        entity_to_basin: dict[str, int] = {}
        for fact in episode.facts:
            fact_tensor = torch.tensor([fact.fact_ids], device=device)
            with torch.no_grad():
                fact_hidden, _fact_states = model.forward_hidden(fact_tensor)
            fact_repr = F.normalize(fact_hidden[0].mean(dim=0, keepdim=True), dim=-1)
            basin_id = int(bank.write(fact_repr.squeeze(0), "fact", Source("user")))
            entity_to_basin[fact.entity] = basin_id

    qa_tensor = torch.tensor([episode.full_query_and_answer_ids], device=device)
    with torch.no_grad():
        hidden, _states = model.forward_hidden(qa_tensor)
    answer_hidden = hidden[:, episode.answer_start_index :, :]

    if port is not None:
        query_repr = F.normalize(
            hidden[0, : episode.answer_start_index].mean(dim=0, keepdim=True),
            dim=-1,
        )
        hits = bank.read(query_repr.squeeze(0), top_k=1)
        retrieved = (
            bank.centers[hits[0].basin_id].unsqueeze(0)
            if hits
            else torch.zeros(1, model.config.d_model, device=device)
        )
        retrieval_correct = bool(
            hits and hits[0].basin_id == entity_to_basin[episode.query_entity]
        )
        answer_hidden = port(answer_hidden, retrieved)

    logits = model.logits_from_hidden(answer_hidden)
    target_ids = torch.tensor(episode.answer_ids, device=device)
    predicted_logits = logits[0, :-1, :]
    predicted_targets = target_ids[1:]
    loss = F.cross_entropy(predicted_logits, predicted_targets)
    answer_correct = bool(
        (predicted_logits.argmax(dim=-1) == predicted_targets).all().item()
    )
    return loss, answer_correct, retrieval_correct


def _run_condition(
    *,
    condition: str,
    model: SonicSSMLanguageModel,
    tokenizer: Any,
    steps: int,
    eval_every: int,
    eval_episodes: int,
    learning_rate: float,
    device: str,
    data_seed: int,
    port_rank: int,
    gate_init_bias: float,
    episodes_per_step: int,
    port_seed: int,
    facts_per_episode: int,
) -> dict[str, Any]:
    port = None
    optimizer = None
    if condition == "memory":
        torch.manual_seed(port_seed)
        port = SonicMemoryPort(
            SonicMemoryPortConfig(
                d_model=model.config.d_model,
                memory_dim=model.config.d_model,
                rank=port_rank,
                gate_init_bias=gate_init_bias,
            )
        ).to(device)
        optimizer = torch.optim.Adam(port.parameters(), lr=learning_rate)

    train_generator = random.Random(data_seed)
    history: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        if port is not None:
            optimizer.zero_grad()
            for _ in range(episodes_per_step):
                episode = _generate_episode(
                    train_generator, tokenizer, facts_per_episode
                )
                loss, _correct, _retrieval = _episode_step(
                    model=model, episode=episode, device=device, port=port
                )
                (loss / episodes_per_step).backward()
            optimizer.step()
        else:
            for _ in range(episodes_per_step):
                _generate_episode(train_generator, tokenizer, facts_per_episode)

        if step % eval_every == 0 or step == steps:
            eval_generator = random.Random(data_seed + 1_000_000 + step)
            eval_losses = []
            correct_count = 0
            retrieval_correct_count = 0
            retrieval_checked_count = 0
            with torch.no_grad():
                for _ in range(eval_episodes):
                    episode = _generate_episode(
                        eval_generator, tokenizer, facts_per_episode
                    )
                    loss, correct, retrieval_correct = _episode_step(
                        model=model, episode=episode, device=device, port=port
                    )
                    eval_losses.append(loss.item())
                    correct_count += correct
                    if retrieval_correct is not None:
                        retrieval_checked_count += 1
                        retrieval_correct_count += retrieval_correct
            row = {
                "step": step,
                "eval_loss_mean": sum(eval_losses) / len(eval_losses),
                "eval_answer_accuracy": correct_count / eval_episodes,
            }
            if retrieval_checked_count:
                row["eval_retrieval_accuracy"] = (
                    retrieval_correct_count / retrieval_checked_count
                )
            history.append(row)

    return {"condition": condition, "history": history}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-dir", type=Path, default=_DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-seed", type=int, default=29)
    parser.add_argument("--port-seed", type=int, default=17)
    parser.add_argument("--port-rank", type=int, default=64)
    parser.add_argument("--gate-init-bias", type=float, default=0.0)
    parser.add_argument("--episodes-per-step", type=int, default=8)
    parser.add_argument("--facts-per-episode", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= args.facts_per_episode <= len(_ENTITIES):
        raise ValueError(f"facts_per_episode must be in [1, {len(_ENTITIES)}]")

    transformers = _require_transformers()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        args.checkpoint_path, map_location="cpu", weights_only=False
    )
    model_config = SonicSSMConfig(**checkpoint["model_config"])
    model = SonicSSMLanguageModel(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.tokenizer_dir))
    if len(tokenizer) != model_config.vocab_size:
        raise ValueError(
            f"tokenizer vocab size {len(tokenizer)} does not match "
            f"model vocab_size {model_config.vocab_size}"
        )

    results = {}
    for condition in ("no_memory", "memory"):
        results[condition] = _run_condition(
            condition=condition,
            model=model,
            tokenizer=tokenizer,
            steps=args.steps,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            learning_rate=args.learning_rate,
            device=device,
            data_seed=args.data_seed,
            port_rank=args.port_rank,
            gate_init_bias=args.gate_init_bias,
            episodes_per_step=args.episodes_per_step,
            port_seed=args.port_seed,
            facts_per_episode=args.facts_per_episode,
        )

    final_no_memory = results["no_memory"]["history"][-1]
    final_memory = results["memory"]["history"][-1]
    output = {
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_update": checkpoint.get("update"),
        "checkpoint_tokens_seen": checkpoint.get("tokens_seen"),
        "model_config": model_config.to_dict(),
        "steps": args.steps,
        "episodes_per_step": args.episodes_per_step,
        "learning_rate": args.learning_rate,
        "gate_init_bias": args.gate_init_bias,
        "port_rank": args.port_rank,
        "facts_per_episode": args.facts_per_episode,
        "results": results,
        "final_no_memory_accuracy": final_no_memory["eval_answer_accuracy"],
        "final_memory_accuracy": final_memory["eval_answer_accuracy"],
        "final_memory_retrieval_accuracy": final_memory.get("eval_retrieval_accuracy"),
        "memory_measurably_helped": (
            final_memory["eval_answer_accuracy"]
            > final_no_memory["eval_answer_accuracy"] + 0.1
        ),
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
    summary = {k: v for k, v in metrics.items() if k != "results"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
