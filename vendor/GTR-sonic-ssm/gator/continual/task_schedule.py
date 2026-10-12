"""Synthetic task families and A-B-A / interleaved scheduling (Procoim-S
design spec sections 24.2-24.3).

``ModularArithmeticTask`` is a synthetic stand-in for "mod-12 prefix
addition" vs. "a different modulus or operation" (spec section 24.2's
example) — deliberately simple so the benchmark harness is independently
testable without depending on this repo's real prefix-task data generator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "ModularArithmeticTask",
    "ABAPhase",
    "ABASchedule",
    "build_aba_schedule",
    "InterleavedExample",
    "generate_interleaved_stream",
]


@dataclass(frozen=True)
class ModularArithmeticTask:
    """``target = (sum(inputs) * scale + offset) mod modulus``.

    Two tasks with different ``modulus``/``offset``/``scale`` are
    "different task families" in the sense section 24.2/24.3 need: same
    input shape and vocabulary, genuinely different target rule.
    """

    name: str
    modulus: int
    offset: int = 0
    scale: int = 1

    def __post_init__(self) -> None:
        if self.modulus <= 1:
            raise ValueError("modulus must be greater than 1")

    def target(self, inputs: Tensor) -> Tensor:
        return (inputs.sum(dim=-1) * self.scale + self.offset) % self.modulus

    def generate(
        self, count: int, *, input_length: int, generator: torch.Generator
    ) -> list[tuple[Tensor, Tensor]]:
        if count <= 0:
            raise ValueError("count must be positive")
        if input_length <= 0:
            raise ValueError("input_length must be positive")
        examples = []
        for _ in range(count):
            inputs = torch.randint(0, self.modulus, (input_length,), generator=generator)
            examples.append((inputs, self.target(inputs)))
        return examples


@dataclass(frozen=True)
class ABAPhase:
    name: str
    task: ModularArithmeticTask
    example_count: int


@dataclass(frozen=True)
class ABASchedule:
    """Phase A -> Phase B -> Phase A return (design spec section 24.2)."""

    phase_a: ABAPhase
    phase_b: ABAPhase
    phase_a_return: ABAPhase


def build_aba_schedule(
    task_a: ModularArithmeticTask,
    task_b: ModularArithmeticTask,
    *,
    examples_per_phase: int,
) -> ABASchedule:
    if examples_per_phase <= 0:
        raise ValueError("examples_per_phase must be positive")
    if task_a.name == task_b.name:
        raise ValueError("task_a and task_b must have distinct names")
    return ABASchedule(
        phase_a=ABAPhase(name="A", task=task_a, example_count=examples_per_phase),
        phase_b=ABAPhase(name="B", task=task_b, example_count=examples_per_phase),
        phase_a_return=ABAPhase(name="A-return", task=task_a, example_count=examples_per_phase),
    )


@dataclass(frozen=True)
class InterleavedExample:
    task_name: str
    inputs: Tensor
    target: Tensor


def generate_interleaved_stream(
    tasks: list[ModularArithmeticTask],
    *,
    count: int,
    input_length: int,
    generator: torch.Generator,
) -> list[InterleavedExample]:
    """Randomly alternate task families example by example (design spec
    section 24.3: "tests whether Procoim-S can maintain a distribution
    rather than one sequential task")."""

    if not tasks:
        raise ValueError("tasks must not be empty")
    if count <= 0:
        raise ValueError("count must be positive")
    stream = []
    for _ in range(count):
        task_index = int(torch.randint(0, len(tasks), (1,), generator=generator).item())
        task = tasks[task_index]
        inputs, target = task.generate(1, input_length=input_length, generator=generator)[0]
        stream.append(InterleavedExample(task_name=task.name, inputs=inputs, target=target))
    return stream
