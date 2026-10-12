"""Delayed-recall integration benchmark (SGBM design spec section 26.1).

This is benchmark *infrastructure*: a synthetic episode generator plus a
walkthrough that exercises a real ``GatorLinearPrefixModel`` end to end
(prefill -> teacher-forced fact commit -> filler commits -> a query block
compared with and without memory). It does not by itself demonstrate the
design spec's acceptance criterion #4 ("memory improves delayed fact
recall") — that requires actually *training* the memory submodules'
learned projections (query/key/value/predictor/summarizer/injector) on
enough of these episodes first. An untrained, randomly-initialized model
has no learned reason to write or retrieve anything useful; running this
harness against one only proves the plumbing works end to end, which is
this module's actual scope (SGBM Phase G).

Note on placement: mirrors the rest of ``gator/continual/`` — the spec's
file layout would put this under ``gator/training/``, unavailable here
since ``gator/training.py`` already exists as a module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from gator.continual.episodic_stream import Episode
from gator.continual.task_schedule import InterleavedExample, ModularArithmeticTask
from gator.memory.config import SurpriseMemoryConfig
from gator.memory.runtime import MemoryRuntime

__all__ = [
    "RecallEpisode",
    "DelayedRecallBenchmark",
    "evaluate_delayed_recall",
    "evaluate_task_accuracy",
    "evaluate_interleaved_accuracy",
]


@dataclass
class RecallEpisode:
    """One synthetic "store a fact, bury it in filler, ask for it back" episode."""

    fact_tokens: Tensor
    filler_blocks: list[Tensor]
    episode_id: str

    def to_episode(self) -> Episode[Tensor]:
        """Adapt to the generic ``EpisodicBatchScheduler`` block sequence."""

        return Episode(
            episode_id=self.episode_id,
            blocks=[self.fact_tokens, *self.filler_blocks],
        )


class DelayedRecallBenchmark:
    """Generates :class:`RecallEpisode` instances over a reserved synthetic vocabulary.

    Token IDs below ``filler_start`` are reserved as structural markers
    (fact/query); everything at or above it is ordinary "content" used for
    both the fact and filler blocks, so a model cannot trivially distinguish
    a fact block from filler by token range alone.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        block_size: int,
        query_marker: int = 0,
        filler_start: int = 1,
        seed: int = 0,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if filler_start <= query_marker:
            raise ValueError("filler_start must be greater than query_marker")
        if vocab_size <= filler_start:
            raise ValueError("vocab_size must leave room for content tokens above filler_start")
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.query_marker = query_marker
        self.filler_start = filler_start
        self._generator = torch.Generator().manual_seed(seed)

    def _random_content_block(self) -> Tensor:
        return torch.randint(
            self.filler_start, self.vocab_size, (self.block_size,), generator=self._generator
        )

    def query_block(self) -> Tensor:
        return torch.full((self.block_size,), self.query_marker, dtype=torch.long)

    def generate(self, count: int, *, filler_blocks: int) -> list[RecallEpisode]:
        if count <= 0:
            raise ValueError("count must be positive")
        if filler_blocks < 0:
            raise ValueError("filler_blocks must be nonnegative")
        episodes = []
        for index in range(count):
            fact = self._random_content_block()
            fillers = [self._random_content_block() for _ in range(filler_blocks)]
            episodes.append(
                RecallEpisode(fact_tokens=fact, filler_blocks=fillers, episode_id=f"recall-{index}")
            )
        return episodes


@torch.no_grad()
def evaluate_delayed_recall(
    model,
    episode: RecallEpisode,
    *,
    memory_config: SurpriseMemoryConfig,
) -> dict[str, Tensor]:
    """Walk one episode through ``model`` and compare query-block recall
    with vs. without memory.

    The fact and every filler block are committed via teacher-forced
    proposals (``output_embeddings`` fixed to the block's own token
    embeddings) so the walkthrough always presents the intended synthetic
    transcript, independent of what the model would have generated on its
    own — the query block, by contrast, is a genuine free-form proposal
    (no teacher forcing), since recall is exactly what is being measured.
    """

    device = next(model.parameters()).device
    block_size = episode.fact_tokens.shape[0]

    fact = episode.fact_tokens.to(device).unsqueeze(0)
    cache = model.prefill_cache(fact, block_size=block_size)
    runtime = MemoryRuntime.allocate(1, memory_config, device=device)

    fact_embeddings = model.token_embedding(fact)
    model.forward_cached(cache, output_embeddings=fact_embeddings, memory=runtime)
    cache = model.append_cache(cache, fact)

    for filler in episode.filler_blocks:
        filler_ids = filler.to(device).unsqueeze(0)
        filler_embeddings = model.token_embedding(filler_ids)
        model.forward_cached(cache, output_embeddings=filler_embeddings, memory=runtime)
        cache = model.append_cache(cache, filler_ids)

    logits_with_memory = model.forward_cached(cache, memory=runtime)
    predicted_with_memory = logits_with_memory.argmax(dim=-1)

    logits_without_memory = model.forward_cached(cache)
    predicted_without_memory = logits_without_memory.argmax(dim=-1)

    target = fact
    accuracy_with_memory = (predicted_with_memory == target).float().mean()
    accuracy_without_memory = (predicted_without_memory == target).float().mean()

    return {
        "accuracy_with_memory": accuracy_with_memory,
        "accuracy_without_memory": accuracy_without_memory,
        "predicted_with_memory": predicted_with_memory.squeeze(0),
        "predicted_without_memory": predicted_without_memory.squeeze(0),
        "target": target.squeeze(0),
    }


@torch.no_grad()
def evaluate_task_accuracy(
    model,
    task: ModularArithmeticTask,
    *,
    count: int,
    input_length: int,
    generator: torch.Generator,
) -> float:
    """Probe a model's accuracy on one synthetic task family (Procoim-S
    design spec section 24.2's per-phase accuracy measurements). As with
    :func:`evaluate_delayed_recall`, this measures whatever the model
    currently does — it does not train it; acquisition/forgetting/recovery
    only become meaningful numbers once real A-B-A training has actually
    run between calls.
    """

    examples = task.generate(count, input_length=input_length, generator=generator)
    inputs = torch.stack([example[0] for example in examples])
    targets = torch.stack([example[1] for example in examples])
    logits = model(inputs, output_slots=1)
    predicted = logits.argmax(dim=-1).squeeze(-1)
    return (predicted == targets).float().mean().item()


@torch.no_grad()
def evaluate_interleaved_accuracy(
    model,
    stream: list[InterleavedExample],
) -> dict[str, float]:
    """Per-task-family accuracy over an interleaved example stream (design
    spec section 24.3)."""

    correct: dict[str, int] = {}
    total: dict[str, int] = {}
    for example in stream:
        logits = model(example.inputs.unsqueeze(0), output_slots=1)
        predicted = logits.argmax(dim=-1).squeeze()
        total[example.task_name] = total.get(example.task_name, 0) + 1
        if int(predicted.item()) == int(example.target.item()):
            correct[example.task_name] = correct.get(example.task_name, 0) + 1
    return {name: correct.get(name, 0) / total[name] for name in total}
