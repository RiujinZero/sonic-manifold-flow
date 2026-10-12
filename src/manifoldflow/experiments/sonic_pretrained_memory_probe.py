"""Second training-viability probe, starting from the real checkpoint
instead of a random init - a direct response to what
`sonic_memory_training_probe.py` found.

That probe's canonical (spec-faithful, SS6.3 "detached constant") result
was flat at the chance floor, and a non-spec diagnostic that let gradient
flow through the write-side representation isolated why: a freshly
initialized Sonic's hidden states for different facts were barely
separable (>0.96 cosine similarity between *different* facts at init),
and with the write side detached, nothing ever told Sonic's weights to
make that representation more legible - only the port could learn, from
whatever weak structure happened to already exist. The user's direction
here overrides the design doc's literal path where the two conflict:
recall and persistent state working matters more than staying spec-pure,
so this probe starts from `checkpoints/sonic_ssm/sonic_ssm_128k_steps.pt`
(6.1B real training tokens) instead of a random init, testing whether an
*already-trained* representation is legible enough for the port to
bootstrap from.

Sonic is loaded **frozen** (`requires_grad_(False)`, wrapped in
`torch.no_grad()` throughout) - not fine-tuned. This is a deliberate
simplification, not an attempt to dodge SS6.3: with Sonic frozen, there
is no gradient path into its weights either way, so the
detached-vs-undetached distinction the first probe's diagnostic turned on
is moot here by construction - there is nothing for gradient to reach
even if it were allowed through. What this probe isolates cleanly is the
first probe's actual open question: is a pretrained representation
*already* legible enough for a small trainable port to exploit, without
needing to fine-tune the 100M-parameter backbone at all. Only
`SonicMemoryPort`'s own parameters train.

Same two-forward-pass separation as the first probe (fact and
query/answer run independently, fresh state each, so external memory is
the only channel that can carry the binding - see that script's docstring
for why an earlier single-sequence version was uninformative) and the
same single-fact-per-episode simplification (retrieval is trivial; this
tests decode-ability, not discrimination among competing facts).

Uses the real `gtr_arw_english_50k` tokenizer instead of a toy word-level
vocabulary, so entities/objects are now real BPE token sequences
(possibly multi-token) rather than one word = one id. The `no_memory`
condition runs entirely under `torch.no_grad()`: nothing trains there
either way (Sonic frozen, no port), so it is a pure fixed baseline, not a
second training run.
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
class Episode:
    fact_ids: list[int]
    query_ids: list[int]
    answer_ids: list[int]

    @property
    def answer_start_index(self) -> int:
        return len(self.query_ids)

    @property
    def full_query_and_answer_ids(self) -> list[int]:
        return self.query_ids + self.answer_ids


def _generate_episode(generator: random.Random, tokenizer: Any) -> Episode:
    entity = generator.choice(_ENTITIES)
    obj = generator.choice(_OBJECTS)
    # Fact/query and answer are tokenized separately and concatenated as id
    # lists, not by encoding one merged string - side-steps BPE merge
    # boundary effects at the concatenation point (a documented
    # simplification, matching how the first probe's toy tokenizer avoided
    # the same issue by construction).
    fact_ids = tokenizer.encode(f"{entity} owns the {obj}.", add_special_tokens=True)
    query_ids = tokenizer.encode(f"What does {entity} own?", add_special_tokens=True)
    answer_ids = tokenizer.encode(f" the {obj}", add_special_tokens=False)
    return Episode(fact_ids=fact_ids, query_ids=query_ids, answer_ids=answer_ids)


def _episode_step(
    *,
    model: SonicSSMLanguageModel,
    episode: Episode,
    device: str,
    port: SonicMemoryPort | None,
) -> tuple[torch.Tensor, bool]:
    """Run one episode; returns (loss, answer_fully_correct). `port=None`
    is the no-memory control condition. Sonic is always frozen, so its own
    forward passes are always run under `no_grad`; whether the *caller* is
    in a grad-enabled context (training) or not (eval) determines whether
    the port's own forward - the only place gradient can go - builds a
    graph. Callers are responsible for wrapping eval calls in
    `torch.no_grad()`."""
    if port is not None:
        fact_tensor = torch.tensor([episode.fact_ids], device=device)
        with torch.no_grad():  # Sonic is frozen; no graph needed here
            fact_hidden, _fact_states = model.forward_hidden(fact_tensor)
        fact_repr = F.normalize(fact_hidden[0].mean(dim=0, keepdim=True), dim=-1)
        bank = BasinMemory(dimension=model.config.d_model, novelty_threshold=0.0)
        bank.write(fact_repr.squeeze(0), "fact", Source("user"))

    qa_tensor = torch.tensor([episode.full_query_and_answer_ids], device=device)
    with torch.no_grad():  # Sonic is frozen; no graph needed here either
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
        answer_hidden = port(answer_hidden, retrieved)

    logits = model.logits_from_hidden(answer_hidden)
    target_ids = torch.tensor(episode.answer_ids, device=device)
    # Position i's logits predict token i+1; the last answer token has no
    # next-token target within this sequence, so it is excluded. Excludes
    # predicting answer_ids[0] (" the") too - that token is not
    # object-identity-bearing and is predicted from the query pass's last
    # (un-injected) hidden state, not scored here.
    predicted_logits = logits[0, :-1, :]
    predicted_targets = target_ids[1:]
    loss = F.cross_entropy(predicted_logits, predicted_targets)
    correct = bool((predicted_logits.argmax(dim=-1) == predicted_targets).all().item())
    return loss, correct


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
                episode = _generate_episode(train_generator, tokenizer)
                loss, _correct = _episode_step(
                    model=model, episode=episode, device=device, port=port
                )
                (loss / episodes_per_step).backward()
            optimizer.step()
        else:
            # Nothing trains in the no-memory condition (Sonic frozen, no
            # port) - advance the generator identically so both
            # conditions' eval episodes line up step-for-step, but skip
            # the wasted forward passes.
            for _ in range(episodes_per_step):
                _generate_episode(train_generator, tokenizer)

        if step % eval_every == 0 or step == steps:
            eval_generator = random.Random(data_seed + 1_000_000 + step)
            eval_losses = []
            correct_count = 0
            with torch.no_grad():
                for _ in range(eval_episodes):
                    episode = _generate_episode(eval_generator, tokenizer)
                    loss, correct = _episode_step(
                        model=model, episode=episode, device=device, port=port
                    )
                    eval_losses.append(loss.item())
                    correct_count += correct
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
    parser.add_argument("--checkpoint-path", type=Path, default=_DEFAULT_CHECKPOINT)
    parser.add_argument("--tokenizer-dir", type=Path, default=_DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-seed", type=int, default=29)
    parser.add_argument("--port-seed", type=int, default=17)
    parser.add_argument("--port-rank", type=int, default=64)
    parser.add_argument("--gate-init-bias", type=float, default=0.0)
    parser.add_argument("--episodes-per-step", type=int, default=8)
    parser.add_argument("--device", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
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
