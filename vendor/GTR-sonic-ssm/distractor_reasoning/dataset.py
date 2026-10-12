"""Async CPU generation for training integration.

``ExampleQueue`` runs one or more background daemon threads that keep
generating fresh examples into a bounded queue, so a training loop's
``get_batch`` call is (almost always) just a queue pop, not a blocking
generation call -- generation never stalls GPU-side training as long as
consumption doesn't outrun the producers for long. Mirrors the
prefetch-thread-pool pattern already used for the SmolLM3 recipe
(``gator/smollm3_recipe.py``).
"""

from __future__ import annotations

import queue
import random
import threading
from typing import Any, Callable, Optional, Union

from .difficulty import DIFFICULTY_LEVELS
from .generate import Example, generate_example

LevelSchedule = Union[int, Callable[[int], int]]
GeneratorFn = Callable[[random.Random, int, str, str], Any]


class ExampleQueue:
    """Background-thread producer of ``Example``s.

    ``level_schedule`` controls the difficulty curriculum: a fixed int
    pins every example to that level; a callable ``step -> level`` (called
    with each worker's own local step counter) allows an arbitrary
    curriculum (e.g. ramping up over training); ``None`` (the default)
    uniformly samples a level in ``[min_level, max_level]`` each time.

    ``distractor_fraction``, when set, overrides the level choice with a
    per-example coin flip: with that probability, sample only from levels
    that actually include distractors (level >= 3); otherwise sample only
    from distractor-free levels (1-2). This is the knob the spec asks for
    ("a configurable fraction of each SFT batch should consist of
    distractor examples") independent of whatever curriculum
    ``level_schedule`` is doing for category/aggregation/reported-fact
    difficulty.

    ``generator_fn`` defaults to the Binding generator
    (``generate.generate_example``) but accepts any ``(rng, level, split,
    style) -> example`` callable with the same shape -- e.g.
    ``state_update.generate_state_update_example`` -- so this same
    background-thread infrastructure serves every generator in the suite.
    Note the State-Update generator's difficulty levels only go up to 8;
    pass a matching ``max_level`` when using it here.
    """

    def __init__(
        self,
        *,
        seed: int = 0,
        min_level: int = 1,
        max_level: int = 10,
        level_schedule: Optional[LevelSchedule] = None,
        distractor_fraction: Optional[float] = None,
        style: str = "qa",
        split: str = "train",
        queue_size: int = 512,
        num_workers: int = 2,
        generator_fn: GeneratorFn = generate_example,
    ) -> None:
        if not 1 <= min_level <= max_level <= 10:
            raise ValueError("require 1 <= min_level <= max_level <= 10")
        if distractor_fraction is not None and not 0 <= distractor_fraction <= 1:
            raise ValueError("distractor_fraction must be in [0, 1]")
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        self.min_level = min_level
        self.max_level = max_level
        self.level_schedule = level_schedule
        self.distractor_fraction = distractor_fraction
        self.style = style
        self.split = split
        self.generator_fn = generator_fn

        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(
                target=self._worker, args=(seed + i,), daemon=True,
                name=f"distractor-reasoning-worker-{i}",
            )
            for i in range(num_workers)
        ]
        for thread in self._threads:
            thread.start()

    def _choose_level(self, step: int, rng: random.Random) -> int:
        if self.distractor_fraction is not None:
            with_distractors = rng.random() < self.distractor_fraction
            low, high = (3, self.max_level) if with_distractors else (1, 2)
            low = max(low, self.min_level)
            high = min(high, self.max_level)
            if low > high:
                low, high = self.min_level, self.max_level
            return rng.randint(low, high)
        if self.level_schedule is None:
            return rng.randint(self.min_level, self.max_level)
        if callable(self.level_schedule):
            level = self.level_schedule(step)
            return max(self.min_level, min(self.max_level, level))
        return self.level_schedule

    def _worker(self, seed: int) -> None:
        rng = random.Random(seed)
        step = 0
        while not self._stop.is_set():
            level = self._choose_level(step, rng)
            example = self.generator_fn(rng, level, split=self.split, style=self.style)
            while not self._stop.is_set():
                try:
                    self._queue.put(example, timeout=0.5)
                    break
                except queue.Full:
                    continue
            step += 1

    def get(self, timeout: Optional[float] = None) -> Any:
        return self._queue.get(timeout=timeout)

    def get_batch(self, batch_size: int, timeout: Optional[float] = None) -> list[Any]:
        return [self.get(timeout=timeout) for _ in range(batch_size)]

    def qsize(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2.0)

    def __enter__(self) -> "ExampleQueue":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = ["DIFFICULTY_LEVELS", "Example", "ExampleQueue", "generate_example"]
