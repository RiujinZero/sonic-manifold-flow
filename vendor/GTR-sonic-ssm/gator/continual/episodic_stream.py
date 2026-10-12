"""Episodic batch scheduling for SGBM training (design spec section 18).

Note on placement: the SGBM design spec's proposed file layout (section 23)
puts this at ``gator/training/episodic_batching.py``. ``gator/training.py``
already exists in this repo as a module, not a package, so that exact path
is unavailable without restructuring unrelated shared code. This lives
under ``gator/continual/`` instead, alongside the rest of the continual-
learning support built for Procoim-S.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import Tensor

__all__ = [
    "Episode",
    "BatchStep",
    "EpisodicBatchScheduler",
    "assert_no_cross_episode_leakage",
]

BlockT = TypeVar("BlockT")


@dataclass
class Episode(Generic[BlockT]):
    """A coherent, ordered sequence of blocks that must be processed in order
    with shared, persistent runtime state (design spec section 18)."""

    episode_id: str
    blocks: Sequence[BlockT]

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("an episode must contain at least one block")


@dataclass
class _LaneState(Generic[BlockT]):
    episode: Episode[BlockT]
    cursor: int = 0


@dataclass
class BatchStep(Generic[BlockT]):
    """One step's worth of per-lane blocks plus lane bookkeeping.

    ``active_mask[i]`` is False when lane ``i`` has no data this step (all
    episodes exhausted); ``reset_mask[i]`` is True exactly when lane ``i``
    just started a brand-new episode — the caller's cue to call
    ``MemoryRuntime.reset_lanes(reset_mask)`` before using this step's data.
    """

    blocks: list[BlockT | None]
    active_mask: Tensor
    reset_mask: Tensor
    episode_ids: list[str | None]


class EpisodicBatchScheduler(Generic[BlockT]):
    """Assigns episodes to fixed batch lanes and advances them block by block.

    Each lane stays bound to one episode across consecutive
    :meth:`next_batch` calls — never reshuffled mid-episode — and a lane
    only receives ``reset_mask=True`` when it starts a genuinely new
    episode. This is what lets a caller reset only the lanes whose episode
    actually ended (design spec section 18) rather than resetting the whole
    batch's memory on every step.
    """

    def __init__(self, episodes: Iterable[Episode[BlockT]], *, batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._queue: deque[Episode[BlockT]] = deque(episodes)
        self.batch_size = batch_size
        self._lanes: list[_LaneState[BlockT] | None] = [None] * batch_size

    def has_more(self) -> bool:
        return bool(self._queue) or any(
            lane is not None and lane.cursor < len(lane.episode.blocks)
            for lane in self._lanes
        )

    def next_batch(self) -> BatchStep[BlockT]:
        blocks: list[BlockT | None] = [None] * self.batch_size
        reset = [False] * self.batch_size
        active = [False] * self.batch_size
        episode_ids: list[str | None] = [None] * self.batch_size

        for lane_index in range(self.batch_size):
            lane = self._lanes[lane_index]
            if lane is not None and lane.cursor >= len(lane.episode.blocks):
                lane = None
                self._lanes[lane_index] = None
            if lane is None:
                if not self._queue:
                    continue
                lane = _LaneState(episode=self._queue.popleft())
                self._lanes[lane_index] = lane
                reset[lane_index] = True
            blocks[lane_index] = lane.episode.blocks[lane.cursor]
            active[lane_index] = True
            episode_ids[lane_index] = lane.episode.episode_id
            lane.cursor += 1

        return BatchStep(
            blocks=blocks,
            active_mask=torch.tensor(active, dtype=torch.bool),
            reset_mask=torch.tensor(reset, dtype=torch.bool),
            episode_ids=episode_ids,
        )


def assert_no_cross_episode_leakage(
    previous_episode_ids: Sequence[str | None],
    step: BatchStep,
) -> None:
    """Raise if any lane's episode changed without a ``reset_mask`` flag.

    A concrete no-leakage assertion (design spec section 18): memory from
    one episode must never silently continue as if it belonged to another.
    """

    if len(previous_episode_ids) != len(step.episode_ids):
        raise ValueError("episode id lists must have matching batch size")
    for lane_index, (previous, current) in enumerate(
        zip(previous_episode_ids, step.episode_ids, strict=True)
    ):
        if current is None:
            continue
        if previous != current and not bool(step.reset_mask[lane_index]):
            raise RuntimeError(
                f"lane {lane_index} switched from episode {previous!r} to "
                f"{current!r} without a reset_mask flag -- cross-episode "
                "memory leakage"
            )
