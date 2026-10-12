"""Reproducible non-IID latent-stream evaluation for local basin adaptation."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from manifoldflow.bank import BasinState
from manifoldflow.streaming import OnlineBasinMemory, calibrate_novelty_threshold
from manifoldflow.streaming_baselines import OnlineClassCentroid, OnlinePrototypeMemory


@dataclass(frozen=True)
class ContinualStreamSpec:
    classes_per_task: int = 2
    stream_samples_per_class: int = 200
    shift_scale: float = 0.25
    latent_noise_std: float = 0.02
    novelty_quantile: float = 0.95
    novelty_persistence: int = 3
    novelty_support_multiplier: float = 1.5
    basin_growth_budget: int = 250
    seed: int = 17

    def validate(self, *, classes: int) -> None:
        if not 1 <= self.classes_per_task <= classes:
            raise ValueError("classes_per_task must be in [1, classes]")
        if classes % self.classes_per_task:
            raise ValueError("classes must divide evenly into tasks")
        if self.stream_samples_per_class <= 0:
            raise ValueError("stream_samples_per_class must be positive")
        if self.shift_scale < 0 or self.latent_noise_std < 0:
            raise ValueError("shift and noise scales must be non-negative")
        if not 0 < self.novelty_quantile < 1:
            raise ValueError("novelty_quantile must be in (0, 1)")
        if self.novelty_persistence < 2:
            raise ValueError("novelty_persistence must be >= 2")
        if self.novelty_support_multiplier <= 0 or self.basin_growth_budget < 0:
            raise ValueError("novelty support must be positive and budget non-negative")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class _MethodView:
    predict: Callable[[Tensor], Tensor]
    state: Callable[[], dict[str, int]]
    stored_vector_values: Callable[[], int]
    stored_value_scalars: Callable[[], int]


def _validate_latents(latent: Tensor, label: Tensor, *, classes: int) -> None:
    if latent.ndim != 2 or latent.shape[0] == 0:
        raise ValueError("latent must have shape [samples, d]")
    if label.shape != (latent.shape[0],) or label.dtype != torch.long:
        raise ValueError("label must be long with shape [samples]")
    if torch.any(label < 0) or torch.any(label >= classes):
        raise ValueError("labels must be in [0, classes)")


@torch.no_grad()
def class_conditioned_shift(
    latent: Tensor,
    label: Tensor,
    directions: Tensor,
    *,
    shift_scale: float,
    noise_std: float,
    generator: torch.Generator,
) -> Tensor:
    """Apply deterministic class drift and noise, then restore unit norm."""
    if directions.ndim != 2 or directions.shape[1] != latent.shape[1]:
        raise ValueError("directions must have shape [classes, latent dimension]")
    _validate_latents(latent, label, classes=directions.shape[0])
    if shift_scale < 0 or noise_std < 0:
        raise ValueError("shift_scale and noise_std must be non-negative")
    noise = torch.randn(
        latent.shape,
        generator=generator,
        device=latent.device,
        dtype=latent.dtype,
    )
    shifted = latent + directions.to(latent.device)[label] * shift_scale
    shifted += noise * noise_std
    return F.normalize(shifted, p=2, dim=-1)


def _task_classes(classes: int, classes_per_task: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(range(start, start + classes_per_task))
        for start in range(0, classes, classes_per_task)
    )


def _stream_indices(
    label: Tensor,
    task: tuple[int, ...],
    *,
    samples_per_class: int,
    generator: torch.Generator,
) -> Tensor:
    selected = []
    for class_id in task:
        available = (label == class_id).nonzero(as_tuple=False).flatten()
        if available.shape[0] < samples_per_class:
            raise ValueError(
                f"class {class_id} has fewer than {samples_per_class} stream samples"
            )
        order = torch.randperm(available.shape[0], generator=generator)
        selected.append(available[order[:samples_per_class]])
    combined = torch.cat(selected)
    return combined[torch.randperm(combined.shape[0], generator=generator)]


@torch.no_grad()
def _evaluate_method(
    view: _MethodView,
    clean_latent: Tensor,
    shifted_latent: Tensor,
    label: Tensor,
    *,
    classes: int,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for condition, latent in (("clean", clean_latent), ("shifted", shifted_latent)):
        prediction = view.predict(latent)
        correct = prediction == label
        per_class = []
        for class_id in range(classes):
            selected = label == class_id
            per_class.append(float(correct[selected].float().mean().item()))
        result[condition] = {
            "accuracy": float(correct.float().mean().item()),
            "per_class_accuracy": per_class,
        }
    result.update(view.state())
    result["stored_vector_values"] = view.stored_vector_values()
    result["stored_value_scalars"] = view.stored_value_scalars()
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _summarize_method(
    stages: list[dict[str, object]],
    tasks: tuple[tuple[int, ...], ...],
    task_updates: list[dict[str, object]],
) -> dict[str, object]:
    adaptation = []
    for task_index, task in enumerate(tasks):
        before = stages[task_index]["shifted"]
        after = stages[task_index + 1]["shifted"]
        assert isinstance(before, dict) and isinstance(after, dict)
        before_per_class = before["per_class_accuracy"]
        after_per_class = after["per_class_accuracy"]
        assert isinstance(before_per_class, list) and isinstance(after_per_class, list)
        adaptation.append(
            _mean(
                [
                    float(after_per_class[class_id]) - float(before_per_class[class_id])
                    for class_id in task
                ]
            )
        )
    final_shifted = stages[-1]["shifted"]
    final_clean = stages[-1]["clean"]
    initial_clean = stages[0]["clean"]
    assert isinstance(final_shifted, dict)
    assert isinstance(final_clean, dict)
    assert isinstance(initial_clean, dict)
    final_per_class = final_shifted["per_class_accuracy"]
    assert isinstance(final_per_class, list)
    forgetting = []
    for task_index, task in enumerate(tasks):
        for class_id in task:
            history = []
            for stage in stages[task_index + 1 :]:
                shifted = stage["shifted"]
                assert isinstance(shifted, dict)
                per_class = shifted["per_class_accuracy"]
                assert isinstance(per_class, list)
                history.append(float(per_class[class_id]))
            forgetting.append(max(history) - float(final_per_class[class_id]))
    return {
        "initial_clean_accuracy": float(initial_clean["accuracy"]),
        "final_clean_accuracy": float(final_clean["accuracy"]),
        "clean_accuracy_change": float(final_clean["accuracy"])
        - float(initial_clean["accuracy"]),
        "final_shifted_accuracy": float(final_shifted["accuracy"]),
        "mean_task_adaptation": _mean(adaptation),
        "mean_forgetting": _mean(forgetting),
        "initial_basins": int(stages[0]["basins"]),
        "final_basins": int(stages[-1]["basins"]),
        "basins_created": int(stages[-1]["basins"]) - int(stages[0]["basins"]),
        "final_delta_size": int(stages[-1]["delta_size"]),
        "final_pending_novelty": int(stages[-1]["pending_novelty"]),
        "final_updated_rows": int(stages[-1]["updated_rows"]),
        "stored_vector_values": int(stages[-1]["stored_vector_values"]),
        "stored_value_scalars": int(stages[-1]["stored_value_scalars"]),
        "novel_observations": sum(
            int(item["novel_observations"]) for item in task_updates
        ),
        "created_observations": sum(
            int(item["created_observations"]) for item in task_updates
        ),
        "capacity_rejections": sum(
            int(item["capacity_rejections"]) for item in task_updates
        ),
        "task_adaptation": adaptation,
    }


@torch.no_grad()
def evaluate_continual_latent_stream(
    bank: BasinState,
    class_count: Tensor,
    prototype_bank: BasinState,
    prototype_class_count: Tensor,
    train_latent: Tensor,
    train_label: Tensor,
    test_latent: Tensor,
    test_label: Tensor,
    *,
    spec: ContinualStreamSpec,
    variance_scale: float = 1.0,
    variance_epsilon: float = 1e-4,
    rate_min: float = 1e-4,
    rate_max: float = 32.0,
    rbf_top_k: int = 4,
    rbf_temperature: float = 0.1,
) -> dict[str, object]:
    """Compare sparse local adaptation with matched online memory controls."""
    if class_count.ndim != 2 or class_count.shape[0] != bank.mu.shape[0]:
        raise ValueError("class_count must have shape [basins, classes]")
    classes = class_count.shape[1]
    if (
        prototype_class_count.ndim != 2
        or prototype_class_count.shape[0] != prototype_bank.mu.shape[0]
        or prototype_class_count.shape[1] != classes
    ):
        raise ValueError("prototype_class_count must match its bank and classes")
    manifold_vector_values = bank.mu.numel() + bank.raw_rate.numel()
    prototype_vector_values = prototype_bank.mu.numel()
    if manifold_vector_values != prototype_vector_values:
        raise ValueError(
            "prototype centers must match manifold center-plus-rate values"
        )
    if not 1 <= rbf_top_k <= prototype_bank.mu.shape[0] or rbf_temperature <= 0:
        raise ValueError("RBF controls have invalid top_k or temperature")
    spec.validate(classes=classes)
    _validate_latents(train_latent, train_label, classes=classes)
    _validate_latents(test_latent, test_label, classes=classes)
    if (
        train_latent.shape[1] != bank.mu.shape[1]
        or test_latent.shape[1] != bank.mu.shape[1]
        or prototype_bank.mu.shape[1] != bank.mu.shape[1]
    ):
        raise ValueError("latent dimensions must match the bank")

    direction_generator = torch.Generator().manual_seed(spec.seed + 1000)
    directions = F.normalize(
        torch.randn(classes, bank.mu.shape[1], generator=direction_generator),
        p=2,
        dim=-1,
    )
    shifted_train = class_conditioned_shift(
        train_latent,
        train_label,
        directions,
        shift_scale=spec.shift_scale,
        noise_std=spec.latent_noise_std,
        generator=torch.Generator().manual_seed(spec.seed + 2000),
    )
    shifted_test = class_conditioned_shift(
        test_latent,
        test_label,
        directions,
        shift_scale=spec.shift_scale,
        noise_std=spec.latent_noise_std,
        generator=torch.Generator().manual_seed(spec.seed + 3000),
    )
    novelty_threshold = calibrate_novelty_threshold(
        train_latent,
        bank.mu,
        quantile=spec.novelty_quantile,
    )
    support_radius = math.sqrt(novelty_threshold) * spec.novelty_support_multiplier
    manifold_common = {
        "novelty_persistence": spec.novelty_persistence,
        "novelty_support_radius": support_radius,
        "class_count": class_count,
        "variance_scale": variance_scale,
        "variance_epsilon": variance_epsilon,
        "rate_min": rate_min,
        "rate_max": rate_max,
    }
    initial_basins = bank.mu.shape[0]
    basin_memories = {
        "frozen": OnlineBasinMemory(
            copy.deepcopy(bank),
            novelty_threshold=float("inf"),
            max_basins=initial_basins,
            **manifold_common,
        ),
        "local_fixed": OnlineBasinMemory(
            copy.deepcopy(bank),
            novelty_threshold=float("inf"),
            max_basins=initial_basins,
            **manifold_common,
        ),
        "local_novelty": OnlineBasinMemory(
            copy.deepcopy(bank),
            novelty_threshold=novelty_threshold,
            max_basins=initial_basins + spec.basin_growth_budget,
            **manifold_common,
        ),
    }
    prototype_memories = {
        "prototype_frozen": OnlinePrototypeMemory(
            prototype_bank.mu, prototype_bank.count, prototype_class_count
        ),
        "prototype_online": OnlinePrototypeMemory(
            prototype_bank.mu, prototype_bank.count, prototype_class_count
        ),
    }
    centroid_initial = OnlineClassCentroid.from_data(
        train_latent, train_label, classes=classes
    )
    centroid_memories = {
        "centroid_frozen": copy.deepcopy(centroid_initial),
        "centroid_online": copy.deepcopy(centroid_initial),
    }

    def basin_state(memory: OnlineBasinMemory) -> dict[str, int]:
        return {
            "basins": memory.bank.mu.shape[0],
            "delta_size": memory.index.delta_size,
            "pending_novelty": memory.stager.pending_count,
            "updated_rows": memory.index.delta_size,
        }

    def prototype_state(memory: OnlinePrototypeMemory) -> dict[str, int]:
        return {
            "basins": memory.centers.shape[0],
            "delta_size": memory.index.delta_size,
            "pending_novelty": 0,
            "updated_rows": memory.index.delta_size,
        }

    def centroid_state(memory: OnlineClassCentroid) -> dict[str, int]:
        return {
            "basins": memory.classes,
            "delta_size": 0,
            "pending_novelty": 0,
            "updated_rows": len(memory.touched),
        }

    views: dict[str, _MethodView] = {}
    for name, memory in basin_memories.items():
        views[name] = _MethodView(
            predict=memory.predict_nearest_value,
            state=lambda memory=memory: basin_state(memory),
            stored_vector_values=lambda memory=memory: (
                memory.bank.mu.numel() + memory.bank.raw_rate.numel()
            ),
            stored_value_scalars=lambda memory=memory: (
                0 if memory.class_count is None else memory.class_count.numel()
            ),
        )
    prototype_frozen = prototype_memories["prototype_frozen"]
    prototype_online = prototype_memories["prototype_online"]
    for name, memory in prototype_memories.items():
        views[name] = _MethodView(
            predict=memory.predict_nearest,
            state=lambda memory=memory: prototype_state(memory),
            stored_vector_values=lambda memory=memory: memory.centers.numel(),
            stored_value_scalars=lambda memory=memory: (
                memory.count.numel() + memory.class_count.numel()
            ),
        )
    views["rbf_frozen"] = _MethodView(
        predict=lambda latent: prototype_frozen.predict_rbf(
            latent, top_k=rbf_top_k, temperature=rbf_temperature
        ),
        state=lambda: prototype_state(prototype_frozen),
        stored_vector_values=lambda: prototype_frozen.centers.numel(),
        stored_value_scalars=lambda: (
            prototype_frozen.count.numel() + prototype_frozen.class_count.numel()
        ),
    )
    views["rbf_online"] = _MethodView(
        predict=lambda latent: prototype_online.predict_rbf(
            latent, top_k=rbf_top_k, temperature=rbf_temperature
        ),
        state=lambda: prototype_state(prototype_online),
        stored_vector_values=lambda: prototype_online.centers.numel(),
        stored_value_scalars=lambda: (
            prototype_online.count.numel() + prototype_online.class_count.numel()
        ),
    )
    for name, memory in centroid_memories.items():
        views[name] = _MethodView(
            predict=memory.predict,
            state=lambda memory=memory: centroid_state(memory),
            stored_vector_values=lambda memory=memory: memory.centers.numel(),
            stored_value_scalars=lambda memory=memory: memory.count.numel(),
        )

    source_for_method = {
        "frozen": "frozen",
        "local_fixed": "local_fixed",
        "local_novelty": "local_novelty",
        "prototype_frozen": "prototype_frozen",
        "prototype_online": "prototype_online",
        "rbf_frozen": "prototype_frozen",
        "rbf_online": "prototype_online",
        "centroid_frozen": "centroid_frozen",
        "centroid_online": "centroid_online",
    }
    tasks = _task_classes(classes, spec.classes_per_task)
    stream_generator = torch.Generator().manual_seed(spec.seed + 4000)
    stages = {
        name: [
            _evaluate_method(
                view,
                test_latent,
                shifted_test,
                test_label,
                classes=classes,
            )
        ]
        for name, view in views.items()
    }
    task_updates: dict[str, list[dict[str, object]]] = {name: [] for name in views}
    for task in tasks:
        indices = _stream_indices(
            train_label,
            task,
            samples_per_class=spec.stream_samples_per_class,
            generator=stream_generator,
        )
        stream_latent = shifted_train[indices]
        stream_label = train_label[indices]
        empty_update = {
            "classes": list(task),
            "samples": indices.shape[0],
            "novel_observations": 0,
            "created_observations": 0,
            "capacity_rejections": 0,
        }
        source_updates: dict[str, dict[str, object]] = {
            "frozen": empty_update,
            "prototype_frozen": empty_update,
            "centroid_frozen": empty_update,
        }
        for source in ("local_fixed", "local_novelty"):
            update = basin_memories[source].update(stream_latent, stream_label)
            source_updates[source] = {
                **empty_update,
                "novel_observations": int(update.novel.sum().item()),
                "created_observations": int(update.created.sum().item()),
                "capacity_rejections": int(update.capacity_rejected.sum().item()),
            }
        prototype_online.update(stream_latent, stream_label)
        source_updates["prototype_online"] = empty_update
        centroid_memories["centroid_online"].update(stream_latent, stream_label)
        source_updates["centroid_online"] = empty_update
        for name, view in views.items():
            task_updates[name].append(dict(source_updates[source_for_method[name]]))
            stages[name].append(
                _evaluate_method(
                    view,
                    test_latent,
                    shifted_test,
                    test_label,
                    classes=classes,
                )
            )
    return {
        "spec": {
            "classes_per_task": spec.classes_per_task,
            "stream_samples_per_class": spec.stream_samples_per_class,
            "shift_scale": spec.shift_scale,
            "latent_noise_std": spec.latent_noise_std,
            "novelty_quantile": spec.novelty_quantile,
            "novelty_threshold": novelty_threshold,
            "novelty_support_radius": support_radius,
            "novelty_support_multiplier": spec.novelty_support_multiplier,
            "novelty_persistence": spec.novelty_persistence,
            "basin_growth_budget": spec.basin_growth_budget,
            "rbf_top_k": rbf_top_k,
            "rbf_temperature": rbf_temperature,
            "manifold_vector_values": manifold_vector_values,
            "prototype_vector_values": prototype_vector_values,
            "seed": spec.seed,
        },
        "tasks": [list(task) for task in tasks],
        "methods": {
            name: {
                "summary": _summarize_method(stages[name], tasks, task_updates[name]),
                "stages": stages[name],
                "task_updates": task_updates[name],
            }
            for name in views
        },
    }
