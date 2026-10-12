"""First training viability test for design-v0.4.md's core hypothesis:
"A 100M Sonic can learn substantially stronger long-horizon binding...
when a sparse geometric memory is present from the first training stage
and is treated as native recurrent state." This is the smallest version
of that question that can be answered with real gradients and a real
`BasinMemory`, not the full curriculum design-v0.4.md eventually wants.

Two Sonic instances, identical architecture and identical random init
(same seed), trained on the same synthetic episodes for the same number
of steps, differing only in one thing: whether a `SonicMemoryPort`
reads a real, per-episode `BasinMemory` write/read cycle before the
answer span, or the model answers from its own forward pass alone.

**The fact and the query/answer are run as two separate `forward_hidden`
calls, each starting from fresh (`states=None`) recurrent state, with no
shared context between them.** An earlier version of this script ran
them as one contiguous sequence and found both conditions converge to
100% - a real result, but an uninformative one: a plain recurrent SSM
can just carry the fact forward through its own within-sequence hidden
state and never needs external memory to answer a question about
something it saw a dozen tokens ago in the *same* forward pass. That
tests in-context recall, not persistent memory. Splitting the two calls
means the *only* channel that can carry the entity-object binding from
the fact pass to the query pass is the external bank+port in the
"memory" condition; the "no_memory" condition's query pass never sees
the fact at all and is reduced to an 8-way guess (12.5%) - a real,
adversarial floor, not a strawman.

Deliberately minimal in ways worth naming, not hiding:

- **Single fact per episode**, so retrieval is trivial (the bank always
  has exactly one thing to find) - this tests "does gradient flow
  through the port, and does having a stored value at all help
  reconstruct it," not "does the query correctly discriminate among
  competing facts" the way Phase 2/3's probes do. That is a harder,
  natural follow-up, not this script's job.
- **Fresh, small `SonicSSMConfig`, trained from random init** - not the
  real 100M checkpoint. Fast iteration, and whether the mechanism trains
  at all does not need real pretraining to test.
- **A tiny, task-specific word-level vocabulary** - not the real 50K
  tokenizer, unnecessary size and complexity for a toy recall task.
- **The memory port is applied once, to the whole answer span** after
  Sonic's final hidden state - not design-v0.4's literal "one or two
  selected mid-stack depths." `sonic_ssm.py` stays untouched.
- **The written/retrieved "value" is Sonic's own raw hidden state at the
  fact boundary** (L2-normalized), reused as both the `BasinMemory` key
  and the injected content - not a separate frozen codec producing a
  distinct D_value latent the way design-v0.4's SS3.1 eventually wants.

`--detach-write` (default `true`) controls one specific, deliberately
exposed deviation point. `true` matches SS6.3's spec ("the retrieved
basin values are detached constants") literally: the fact-pass
representation that gets written to the bank carries no gradient back to
Sonic, so nothing ever tells Sonic's own weights to make that
representation more legible for retrieval - only the port learns, from
whatever structure happens to already be there. `false` is a documented
non-spec diagnostic: it lets gradient flow through the write-side
representation too, isolating whether *that specific detachment* is what
blocks learning, separate from any other possible defect in the
mechanism (two-pass separation, gate, injection point, task design). Both
conditions were run for real; see the results doc for what each showed.
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

_ENTITIES = ["Neral", "Tovan", "Mira", "Selka", "Bren", "Dorna", "Ilvet", "Quorin"]
_OBJECTS = ["glider", "lantern", "compass", "crate", "whistle", "map", "key", "vial"]
_FIXED_VOCAB = ["owns", "the", ".", "What", "does", "own", "?", "<pad>"]


class ToyTokenizer:
    """A tiny closed word-level vocabulary for this synthetic task only -
    not the real 50K GTR tokenizer, which would dominate a small model's
    parameter count for no benefit here."""

    def __init__(self) -> None:
        vocab = sorted(set(_ENTITIES) | set(_OBJECTS) | set(_FIXED_VOCAB))
        self.vocab = vocab
        self.token_to_id = {token: index for index, token in enumerate(vocab)}

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id[token] for token in tokens]


@dataclass(frozen=True)
class Episode:
    fact_tokens: list[str]
    query_tokens: list[str]
    answer_tokens: list[str]

    @property
    def fact_end_index(self) -> int:
        return len(self.fact_tokens) - 1

    @property
    def query_and_answer_tokens(self) -> list[str]:
        return self.query_tokens + self.answer_tokens

    @property
    def query_end_index(self) -> int:
        return len(self.query_tokens) - 1

    @property
    def answer_start_index(self) -> int:
        return self.query_end_index + 1


def _generate_episode(generator: random.Random) -> Episode:
    entity = generator.choice(_ENTITIES)
    obj = generator.choice(_OBJECTS)
    return Episode(
        fact_tokens=[entity, "owns", "the", obj, "."],
        query_tokens=["What", "does", entity, "own", "?"],
        answer_tokens=["the", obj],
    )


def _episode_step(
    *,
    model: SonicSSMLanguageModel,
    tokenizer: ToyTokenizer,
    episode: Episode,
    device: str,
    port: SonicMemoryPort | None,
    detach_write: bool = True,
) -> tuple[torch.Tensor, bool]:
    """Run one episode; returns (loss, answer_fully_correct). `port=None`
    is the no-memory control condition. The fact and the query/answer are
    two independent `forward_hidden` calls (fresh state each) - see the
    module docstring for why that separation is what makes this a real
    test of the memory channel rather than in-context recall.
    `detach_write=False` is the non-spec diagnostic described there."""
    if port is not None:
        fact_ids = torch.tensor(
            [tokenizer.encode(episode.fact_tokens)], device=device
        )
        fact_hidden, _fact_states = model.forward_hidden(fact_ids)
        # Mean pooling over the fact span, not the last token: Sonic Phase 1
        # (sonic-phase1-representation-sweep.md) found mean pooling beats
        # last-token pooling at every depth for this same architecture
        # (91.7%/91.7%/83.3% vs 58.3%/41.7%/41.7%) - a debug check here
        # confirmed the same pattern (last-token gave >0.96 cosine similarity
        # between *different* facts, near-unusable as a signal).
        pooled = fact_hidden[0].mean(dim=0, keepdim=True)
        fact_repr = F.normalize(pooled if not detach_write else pooled.detach(), dim=-1)
        if detach_write:
            bank = BasinMemory(dimension=model.config.d_model, novelty_threshold=0.0)
            bank.write(fact_repr.squeeze(0), "fact", Source("user"))

    query_answer_ids = torch.tensor(
        [tokenizer.encode(episode.query_and_answer_tokens)], device=device
    )
    hidden, _states = model.forward_hidden(query_answer_ids)
    answer_hidden = hidden[:, episode.answer_start_index :, :]

    if port is not None:
        if detach_write:
            query_repr = F.normalize(
                hidden[0, : episode.answer_start_index]
                .mean(dim=0, keepdim=True)
                .detach(),
                dim=-1,
            )
            hits = bank.read(query_repr.squeeze(0), top_k=1)
            retrieved = (
                bank.centers[hits[0].basin_id].unsqueeze(0)
                if hits
                else torch.zeros(1, model.config.d_model, device=device)
            )
        else:
            # No bank at all in this diagnostic: fact_repr is injected
            # directly, undetached, isolating the effect of detachment
            # itself from BasinMemory's (trivial, single-basin) routing.
            retrieved = fact_repr
        answer_hidden = port(answer_hidden, retrieved)

    logits = model.logits_from_hidden(answer_hidden)
    target_ids = torch.tensor(
        tokenizer.encode(episode.answer_tokens), device=device
    )
    # Position i's logits predict token i+1; the last answer token has no
    # next-token target within this sequence, so it is excluded - standard
    # shifted causal-LM loss, not a memory-specific choice.
    predicted_logits = logits[0, :-1, :]
    predicted_targets = target_ids[1:]
    loss = F.cross_entropy(predicted_logits, predicted_targets)
    correct = bool((predicted_logits.argmax(dim=-1) == predicted_targets).all().item())
    return loss, correct


def _train_condition(
    *,
    condition: str,
    seed: int,
    config: SonicSSMConfig,
    tokenizer: ToyTokenizer,
    steps: int,
    eval_every: int,
    eval_episodes: int,
    learning_rate: float,
    device: str,
    data_seed: int,
    port_rank: int,
    gate_init_bias: float,
    episodes_per_step: int,
    detach_write: bool,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = SonicSSMLanguageModel(config).to(device)
    port = None
    parameters = list(model.parameters())
    if condition == "memory":
        port = SonicMemoryPort(
            SonicMemoryPortConfig(
                d_model=config.d_model,
                memory_dim=config.d_model,
                rank=port_rank,
                gate_init_bias=gate_init_bias,
            )
        ).to(device)
        parameters += list(port.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)

    train_generator = random.Random(data_seed)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, steps + 1):
        # Gradient accumulation over `episodes_per_step` independent episodes
        # (each with its own fresh per-episode BasinMemory bank) rather than
        # a real batched forward pass - BasinMemory's write/read operate on
        # one key at a time, so this is the simplest way to get a
        # lower-variance gradient estimate. Motivated by a real observation:
        # batch=1 SGD produced a brief, real accuracy spike (26%, well below
        # chance loss) that then collapsed back to the chance floor by the
        # next eval - a sign the memory pathway can carry usable signal but
        # single-episode updates are too noisy to consolidate it.
        optimizer.zero_grad()
        for _ in range(episodes_per_step):
            episode = _generate_episode(train_generator)
            loss, _correct = _episode_step(
                model=model,
                tokenizer=tokenizer,
                episode=episode,
                device=device,
                port=port,
                detach_write=detach_write,
            )
            (loss / episodes_per_step).backward()
        optimizer.step()

        if step % eval_every == 0 or step == steps:
            model.eval()
            eval_generator = random.Random(data_seed + 1_000_000 + step)
            eval_losses = []
            correct_count = 0
            with torch.no_grad():
                for _ in range(eval_episodes):
                    episode = _generate_episode(eval_generator)
                    loss, correct = _episode_step(
                        model=model,
                        tokenizer=tokenizer,
                        episode=episode,
                        device=device,
                        port=port,
                        detach_write=detach_write,
                    )
                    eval_losses.append(loss.item())
                    correct_count += correct
            model.train()
            history.append(
                {
                    "step": step,
                    "eval_loss_mean": sum(eval_losses) / len(eval_losses),
                    "eval_answer_accuracy": correct_count / eval_episodes,
                }
            )

    return {"condition": condition, "history": history}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--data-seed", type=int, default=29)
    parser.add_argument("--port-rank", type=int, default=16)
    parser.add_argument(
        "--episodes-per-step",
        type=int,
        default=1,
        help=(
            "Gradient-accumulation batch size (independent episodes "
            "averaged per optimizer step)."
        ),
    )
    parser.add_argument(
        "--gate-init-bias",
        type=float,
        default=-4.0,
        help=(
            "SonicMemoryPortConfig.gate_init_bias. The default (-4.0) mirrors "
            "sonic_ssm.py's own memory_gate_init_bias, chosen there to protect "
            "an already-trained checkpoint being grafted onto - not this "
            "script's from-scratch case, which sonic_ssm.py's own docstring "
            "says can (often should) open the gate further, e.g. 0.0."
        ),
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument("--num-modes", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-detach-write",
        dest="detach_write",
        action="store_false",
        help=(
            "Non-spec diagnostic: let gradient flow through the write-side "
            "representation instead of treating it as a detached constant "
            "(design-v0.4.md SS6.3). See the module docstring."
        ),
    )
    parser.set_defaults(detach_write=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = ToyTokenizer()
    config = SonicSSMConfig(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        lexical_dim=args.d_model,
        n_layers=args.n_layers,
        state_dim=args.state_dim,
        num_modes=args.num_modes,
        sparsity_target=0.5,
        activation_checkpointing=False,
    )

    results = {}
    for condition in ("no_memory", "memory"):
        results[condition] = _train_condition(
            condition=condition,
            seed=args.seed,
            config=config,
            tokenizer=tokenizer,
            steps=args.steps,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            learning_rate=args.learning_rate,
            device=args.device,
            data_seed=args.data_seed,
            port_rank=args.port_rank,
            gate_init_bias=args.gate_init_bias,
            episodes_per_step=args.episodes_per_step,
            detach_write=args.detach_write,
        )

    final_no_memory = results["no_memory"]["history"][-1]
    final_memory = results["memory"]["history"][-1]
    output = {
        "vocab_size": len(tokenizer),
        "model_config": config.to_dict(),
        "steps": args.steps,
        "episodes_per_step": args.episodes_per_step,
        "learning_rate": args.learning_rate,
        "gate_init_bias": args.gate_init_bias,
        "detach_write": args.detach_write,
        "results": results,
        "final_no_memory_accuracy": final_no_memory["eval_answer_accuracy"],
        "final_memory_accuracy": final_memory["eval_answer_accuracy"],
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


def main() -> None:
    args = _parser().parse_args()
    try:
        metrics = run(args)
    except BaseException:
        sys.stdout.flush()
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    summary = {
        k: v for k, v in metrics.items() if k != "results"
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
