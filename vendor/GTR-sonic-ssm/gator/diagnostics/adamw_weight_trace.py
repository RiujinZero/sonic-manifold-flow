"""Interval tracing of AdamW's exact weight-update geometry.

The tracer is intentionally opt-in. It records the small benchmark model in
enough detail to compare AdamW's learned update direction with forward-only
perturbation policies without changing the optimizer itself. Boundary mode
only touches tensors at trace intervals; an explicitly enabled deep mode also
accumulates exact per-step path statistics at substantial CPU cost.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AdamWTraceSnapshot:
    """One persisted interval and its compact human-readable summary."""

    step: int
    interval_start_step: int
    interval_steps: int
    artifact_path: Path | None
    summary: dict[str, Any]

    def log_line(self) -> str:
        global_stats = self.summary["global"]
        cosine = global_stats["cos_delta_endpoint_adam_update"]
        grad_cosine = global_stats["mean_step_update_negative_gradient_cosine"]
        if grad_cosine is None:
            grad_cosine = global_stats["cos_delta_negative_mean_gradient"]
        path_efficiency = global_stats["path_efficiency"]
        artifact = (
            "none" if self.artifact_path is None else str(self.artifact_path)
        )
        groups = ",".join(
            (
                f"{name}:d={metrics['delta_rms']:.2e}"
                f"/eff={_format_optional(metrics['path_efficiency'])}"
            )
            for name, metrics in self.summary["groups"].items()
        )
        return (
            f"weight_trace step={self.step} "
            f"interval={self.interval_start_step}:{self.step} "
            f"interval_steps={self.interval_steps} "
            f"delta_norm={global_stats['delta_norm']:.6g} "
            f"delta_rms={global_stats['delta_rms']:.6g} "
            f"relative_delta={global_stats['relative_delta_norm']:.6g} "
            f"path_efficiency={_format_optional(path_efficiency)} "
            f"adam_cos={_format_optional(cosine)} "
            f"negative_grad_cos={_format_optional(grad_cosine)} "
            f"clip_rate={self.summary['clip_rate']:.4f} "
            f"groups=[{groups}] artifact={artifact!r}"
        )


@dataclass(frozen=True)
class _ParameterRecord:
    name: str
    parameter: nn.Parameter
    semantic_group: str
    optimizer_group: int


def _format_optional(value: float | None) -> str:
    return "nan" if value is None else f"{value:.6g}"


def _cosine(
    left_dot_right: float,
    left_norm: float,
    right_norm: float,
) -> float | None:
    denominator = left_norm * right_norm
    if denominator <= 0.0:
        return None
    return max(-1.0, min(1.0, left_dot_right / denominator))


def _optimizer_step_value(state: Mapping[str, Any]) -> int:
    value = state.get("step", 0)
    if isinstance(value, Tensor):
        return int(value.detach().cpu())
    return int(value)


class AdamWWeightTracer:
    """Trace exact AdamW weight displacement and endpoint optimizer state."""

    format_version = 2
    _all_group = "__all__"

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.AdamW,
        output_dir: str | Path,
        *,
        interval: int = 500,
        save_tensors: bool = True,
        step_statistics: bool = False,
    ) -> None:
        if not isinstance(optimizer, torch.optim.AdamW):
            raise TypeError("AdamWWeightTracer requires torch.optim.AdamW")
        if isinstance(interval, bool) or not isinstance(interval, int):
            raise TypeError("trace interval must be an integer")
        if interval <= 0:
            raise ValueError("trace interval must be positive")

        self.model = model
        self.optimizer = optimizer
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.save_tensors = bool(save_tensors)
        self.step_statistics = bool(step_statistics)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / "summary.jsonl"
        if self.summary_path.exists() or any(
            self.output_dir.glob("trace_step_*.pt")
        ):
            raise FileExistsError(
                f"AdamW trace directory is not empty: {self.output_dir}"
            )

        optimizer_groups: dict[int, int] = {}
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter in group["params"]:
                identity = id(parameter)
                if identity in optimizer_groups:
                    raise ValueError(
                        "an AdamW parameter appears in multiple groups"
                    )
                optimizer_groups[identity] = group_index

        semantic_groups: dict[int, str] = {}
        group_provider = getattr(model, "procoim_parameter_groups", None)
        if callable(group_provider):
            provided = group_provider()
            if not isinstance(provided, Mapping):
                raise TypeError(
                    "procoim_parameter_groups() must return a mapping"
                )
            for group_name, parameters in provided.items():
                for parameter in parameters:
                    identity = id(parameter)
                    if (
                        identity in semantic_groups
                        and semantic_groups[identity] != str(group_name)
                    ):
                        raise ValueError(
                            "semantic parameter groups must be disjoint"
                        )
                    semantic_groups[identity] = str(group_name)

        records: list[_ParameterRecord] = []
        for name, parameter in model.named_parameters():
            identity = id(parameter)
            if identity not in optimizer_groups:
                raise ValueError(
                    f"model parameter {name!r} is absent from AdamW"
                )
            records.append(
                _ParameterRecord(
                    name=name,
                    parameter=parameter,
                    semantic_group=semantic_groups.get(
                        identity, "ungrouped"
                    ),
                    optimizer_group=optimizer_groups[identity],
                )
            )
        if not records:
            raise ValueError("cannot trace a model without parameters")
        self.records = tuple(records)
        self._record_by_name = {
            record.name: record for record in self.records
        }
        self._group_names = tuple(
            sorted({record.semantic_group for record in self.records})
        )

        self._previous_parameters = self._cpu_parameters()
        self._previous_delta: dict[str, Tensor] | None = None
        self._before_step: dict[str, Tensor] | None = None
        self._last_snapshot_step: int | None = None
        self._window_start_step = 0
        self._reset_window_accumulators()

    def _cpu_parameters(self) -> dict[str, Tensor]:
        return {
            record.name: (
                record.parameter.detach().to(
                    device="cpu", dtype=torch.float32
                ).clone()
            )
            for record in self.records
        }

    def _cpu_gradients(self) -> dict[str, Tensor]:
        gradients: dict[str, Tensor] = {}
        for record in self.records:
            gradient = record.parameter.grad
            if gradient is None:
                value = torch.zeros_like(
                    record.parameter,
                    device="cpu",
                    dtype=torch.float32,
                )
            else:
                value = gradient.detach()
                if value.is_sparse:
                    value = value.to_dense()
                value = value.to(
                    device="cpu",
                    dtype=torch.float32,
                ).clone()
            gradients[record.name] = value
        return gradients

    def _reset_window_accumulators(self) -> None:
        self._window_steps = 0
        self._loss_sum = 0.0
        self._preclip_norm_sum = 0.0
        self._preclip_norm_square_sum = 0.0
        self._clip_count = 0
        self._endpoint_gradient: dict[str, Tensor] | None = None
        if not self.step_statistics:
            self._gradient_sum = {}
            self._gradient_square_sum = {}
            self._update_square_sum = {}
            self._path_length = {}
            self._step_update_gradient_cosine_sum = {}
            self._step_update_gradient_cosine_count = {}
            return
        self._gradient_sum = {
            record.name: torch.zeros_like(
                record.parameter, dtype=torch.float32
            )
            for record in self.records
        }
        self._gradient_square_sum = {
            record.name: torch.zeros_like(
                record.parameter, dtype=torch.float32
            )
            for record in self.records
        }
        self._update_square_sum = {
            record.name: torch.zeros_like(
                record.parameter, dtype=torch.float32
            )
            for record in self.records
        }
        scalar_device = self.records[0].parameter.device
        statistic_groups = (self._all_group, *self._group_names)
        self._path_length = {
            name: torch.zeros((), device=scalar_device)
            for name in statistic_groups
        }
        self._step_update_gradient_cosine_sum = {
            name: torch.zeros((), device=scalar_device)
            for name in statistic_groups
        }
        self._step_update_gradient_cosine_count = {
            name: torch.zeros((), device=scalar_device)
            for name in statistic_groups
        }

    @torch.no_grad()
    def before_optimizer_step(
        self,
        *,
        step: int,
        loss: float,
        preclip_gradient_norm: float,
        gradient_clip: float,
        force_snapshot: bool = False,
    ) -> None:
        """Capture the exact clipped gradients AdamW is about to consume."""

        if self._before_step is not None:
            raise RuntimeError("AdamW tracer step is already in progress")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("traced optimizer step must be a positive integer")
        if not math.isfinite(loss):
            raise ValueError("traced loss must be finite")
        if not math.isfinite(preclip_gradient_norm):
            raise ValueError("traced gradient norm must be finite")
        if not math.isfinite(gradient_clip) or gradient_clip <= 0:
            raise ValueError("gradient clip must be finite and positive")

        self._window_steps += 1
        self._loss_sum += float(loss)
        self._preclip_norm_sum += float(preclip_gradient_norm)
        self._preclip_norm_square_sum += float(preclip_gradient_norm) ** 2
        if preclip_gradient_norm > gradient_clip:
            self._clip_count += 1

        if force_snapshot or step % self.interval == 0:
            self._endpoint_gradient = self._cpu_gradients()
        if not self.step_statistics:
            return

        before: dict[str, Tensor] = {}
        for record in self.records:
            parameter = record.parameter
            before[record.name] = parameter.detach().to(
                dtype=torch.float32
            ).clone()
            gradient = parameter.grad
            if gradient is None:
                continue
            value = gradient.detach()
            if value.is_sparse:
                value = value.to_dense()
            value = value.to(dtype=torch.float32)
            self._gradient_sum[record.name].add_(value)
            self._gradient_square_sum[record.name].addcmul_(
                value, value
            )

        self._before_step = before

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        """Measure the exact parameter update produced by AdamW."""

        if not self.step_statistics:
            return
        if self._before_step is None:
            raise RuntimeError("AdamW tracer has no pending optimizer step")
        group_update_square: dict[str, Tensor] = {}
        group_gradient_square: dict[str, Tensor] = {}
        group_update_negative_gradient_dot: dict[str, Tensor] = {}
        device = self.records[0].parameter.device
        for group_name in (self._all_group, *self._group_names):
            group_update_square[group_name] = torch.zeros((), device=device)
            group_gradient_square[group_name] = torch.zeros((), device=device)
            group_update_negative_gradient_dot[group_name] = torch.zeros(
                (), device=device
            )

        for record in self.records:
            parameter = record.parameter
            update = (
                parameter.detach().to(dtype=torch.float32)
                - self._before_step[record.name]
            )
            self._update_square_sum[record.name].addcmul_(update, update)
            gradient = parameter.grad
            if gradient is None:
                gradient_value = torch.zeros_like(update)
            else:
                gradient_value = gradient.detach()
                if gradient_value.is_sparse:
                    gradient_value = gradient_value.to_dense()
                gradient_value = gradient_value.to(dtype=torch.float32)

            update_square = update.square().sum()
            gradient_square = gradient_value.square().sum()
            update_negative_gradient_dot = (
                update * -gradient_value
            ).sum()
            for group_name in (
                self._all_group,
                record.semantic_group,
            ):
                group_update_square[group_name].add_(update_square)
                group_gradient_square[group_name].add_(gradient_square)
                group_update_negative_gradient_dot[group_name].add_(
                    update_negative_gradient_dot
                )

        for group_name in (self._all_group, *self._group_names):
            update_norm = group_update_square[group_name].sqrt()
            gradient_norm = group_gradient_square[group_name].sqrt()
            denominator = update_norm * gradient_norm
            cosine = torch.where(
                denominator > 0,
                group_update_negative_gradient_dot[group_name]
                / denominator.clamp_min(torch.finfo(torch.float32).tiny),
                torch.zeros_like(denominator),
            )
            self._path_length[group_name].add_(update_norm)
            self._step_update_gradient_cosine_sum[group_name].add_(cosine)
            self._step_update_gradient_cosine_count[group_name].add_(
                (denominator > 0).to(dtype=torch.float32)
            )
        self._before_step = None

    def maybe_snapshot(
        self,
        step: int,
        *,
        force: bool = False,
        context: Mapping[str, Any] | None = None,
    ) -> AdamWTraceSnapshot | None:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("trace step must be an integer")
        if step < 0:
            raise ValueError("trace step must be nonnegative")
        if self._last_snapshot_step == step:
            return None
        if not force and step % self.interval != 0:
            return None
        return self._snapshot(step, context=context)

    @torch.no_grad()
    def _snapshot(
        self,
        step: int,
        *,
        context: Mapping[str, Any] | None,
    ) -> AdamWTraceSnapshot:
        if self._before_step is not None:
            raise RuntimeError("cannot snapshot during an optimizer step")
        if step < self._window_start_step:
            raise ValueError("trace steps must be monotonic")

        current = self._cpu_parameters()
        delta = {
            name: current[name] - self._previous_parameters[name]
            for name in current
        }
        if self.step_statistics:
            divisor = max(self._window_steps, 1)
            mean_gradient = {
                name: value.detach().cpu() / divisor
                for name, value in self._gradient_sum.items()
            }
            mean_gradient_square = {
                name: value.detach().cpu() / divisor
                for name, value in self._gradient_square_sum.items()
            }
            rms_gradient = {
                name: value.clamp_min(0.0).sqrt()
                for name, value in mean_gradient_square.items()
            }
            mean_update_square = {
                name: value.detach().cpu() / divisor
                for name, value in self._update_square_sum.items()
            }
            rms_update = {
                name: value.clamp_min(0.0).sqrt()
                for name, value in mean_update_square.items()
            }
        else:
            if self._window_steps > 0 and self._endpoint_gradient is None:
                raise RuntimeError(
                    "boundary trace snapshot was not announced before "
                    "the endpoint optimizer step"
                )
            endpoint = self._endpoint_gradient or {
                name: torch.zeros_like(value)
                for name, value in current.items()
            }
            mean_gradient = {
                name: value.clone() for name, value in endpoint.items()
            }
            mean_gradient_square = {
                name: value.square() for name, value in endpoint.items()
            }
            rms_gradient = {
                name: value.abs() for name, value in endpoint.items()
            }
            rms_update = {
                name: torch.zeros_like(value)
                for name, value in current.items()
            }

        exp_avg: dict[str, Tensor] = {}
        exp_avg_sq: dict[str, Tensor] = {}
        endpoint_adam_update: dict[str, Tensor] = {}
        optimizer_steps: dict[str, int] = {}
        for record in self.records:
            state = self.optimizer.state.get(record.parameter, {})
            first = state.get("exp_avg")
            second = state.get("exp_avg_sq")
            first_cpu = (
                torch.zeros_like(current[record.name])
                if first is None
                else first.detach().to(
                    device="cpu", dtype=torch.float32
                ).clone()
            )
            second_cpu = (
                torch.zeros_like(current[record.name])
                if second is None
                else second.detach().to(
                    device="cpu", dtype=torch.float32
                ).clone()
            )
            exp_avg[record.name] = first_cpu
            exp_avg_sq[record.name] = second_cpu
            optimizer_step = _optimizer_step_value(state)
            optimizer_steps[record.name] = optimizer_step
            group = self.optimizer.param_groups[record.optimizer_group]
            beta1, beta2 = group["betas"]
            if optimizer_step > 0:
                first_hat = first_cpu / (1.0 - beta1**optimizer_step)
                second_hat = second_cpu / (1.0 - beta2**optimizer_step)
                adam_direction = first_hat / (
                    second_hat.sqrt() + float(group["eps"])
                )
            else:
                adam_direction = torch.zeros_like(first_cpu)
            endpoint_adam_update[record.name] = -float(group["lr"]) * (
                adam_direction
                + float(group["weight_decay"]) * current[record.name]
            )

        names_by_group: dict[str, list[str]] = defaultdict(list)
        for record in self.records:
            names_by_group[record.semantic_group].append(record.name)
        global_stats = self._vector_statistics(
            tuple(current),
            current=current,
            delta=delta,
            mean_gradient=mean_gradient,
            mean_gradient_square=mean_gradient_square,
            rms_update=rms_update,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            endpoint_adam_update=endpoint_adam_update,
            statistic_group=self._all_group,
        )
        group_stats = {
            group_name: self._vector_statistics(
                tuple(names),
                current=current,
                delta=delta,
                mean_gradient=mean_gradient,
                mean_gradient_square=mean_gradient_square,
                rms_update=rms_update,
                exp_avg=exp_avg,
                exp_avg_sq=exp_avg_sq,
                endpoint_adam_update=endpoint_adam_update,
                statistic_group=group_name,
            )
            for group_name, names in sorted(names_by_group.items())
        }
        summary: dict[str, Any] = {
            "format": "gator-adamw-weight-trace-summary-v2",
            "step": step,
            "interval_start_step": self._window_start_step,
            "step_statistics": self.step_statistics,
            "gradient_statistic": (
                "window_mean"
                if self.step_statistics
                else "endpoint_clipped_gradient"
            ),
            "interval_steps": self._window_steps,
            "mean_training_loss": (
                None
                if self._window_steps == 0
                else self._loss_sum / self._window_steps
            ),
            "mean_preclip_gradient_norm": (
                None
                if self._window_steps == 0
                else self._preclip_norm_sum / self._window_steps
            ),
            "rms_preclip_gradient_norm": (
                None
                if self._window_steps == 0
                else math.sqrt(
                    self._preclip_norm_square_sum / self._window_steps
                )
            ),
            "clip_rate": self._clip_count / max(self._window_steps, 1),
            "global": global_stats,
            "groups": group_stats,
            "context": dict(context or {}),
        }

        artifact_path: Path | None = None
        if self.save_tensors:
            artifact_path = (
                self.output_dir / f"trace_step_{step:09d}.pt"
            )
            temporary = artifact_path.with_suffix(".pt.tmp")
            payload = {
                "format": "gator-adamw-weight-trace",
                "version": self.format_version,
                "summary": summary,
                "parameter_groups": {
                    record.name: record.semantic_group
                    for record in self.records
                },
                "optimizer_groups": {
                    record.name: record.optimizer_group
                    for record in self.records
                },
                "optimizer_steps": optimizer_steps,
                "parameters": current,
                "interval_delta": delta,
                "mean_gradient": mean_gradient,
                "endpoint_gradient": (
                    {
                        name: value.clone()
                        for name, value in self._endpoint_gradient.items()
                    }
                    if self._endpoint_gradient is not None
                    else {
                        name: torch.zeros_like(value)
                        for name, value in current.items()
                    }
                ),
                "rms_gradient": rms_gradient,
                "rms_update": rms_update,
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
                "endpoint_adam_update": endpoint_adam_update,
            }
            torch.save(payload, temporary)
            temporary.replace(artifact_path)

        with self.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(summary, sort_keys=True, allow_nan=False) + "\n"
            )

        result = AdamWTraceSnapshot(
            step=step,
            interval_start_step=self._window_start_step,
            interval_steps=self._window_steps,
            artifact_path=artifact_path,
            summary=summary,
        )
        self._previous_parameters = current
        self._previous_delta = {
            name: value.clone() for name, value in delta.items()
        }
        self._last_snapshot_step = step
        self._window_start_step = step
        self._reset_window_accumulators()
        return result

    def _vector_statistics(
        self,
        names: tuple[str, ...],
        *,
        current: Mapping[str, Tensor],
        delta: Mapping[str, Tensor],
        mean_gradient: Mapping[str, Tensor],
        mean_gradient_square: Mapping[str, Tensor],
        rms_update: Mapping[str, Tensor],
        exp_avg: Mapping[str, Tensor],
        exp_avg_sq: Mapping[str, Tensor],
        endpoint_adam_update: Mapping[str, Tensor],
        statistic_group: str,
    ) -> dict[str, Any]:
        count = sum(current[name].numel() for name in names)
        weight_square = math.fsum(
            float(current[name].square().sum()) for name in names
        )
        delta_square = math.fsum(
            float(delta[name].square().sum()) for name in names
        )
        gradient_square = math.fsum(
            float(mean_gradient[name].square().sum()) for name in names
        )
        gradient_element_square = math.fsum(
            float(mean_gradient_square[name].sum()) for name in names
        )
        rms_update_square = math.fsum(
            float(rms_update[name].square().sum()) for name in names
        )
        first_moment_square = math.fsum(
            float(exp_avg[name].square().sum()) for name in names
        )
        second_moment_sum = math.fsum(
            float(exp_avg_sq[name].sum()) for name in names
        )
        endpoint_square = math.fsum(
            float(endpoint_adam_update[name].square().sum())
            for name in names
        )
        delta_endpoint_dot = math.fsum(
            float((delta[name] * endpoint_adam_update[name]).sum())
            for name in names
        )
        delta_negative_gradient_dot = math.fsum(
            float((delta[name] * -mean_gradient[name]).sum())
            for name in names
        )
        changed = sum(
            int(delta[name].ne(0).sum()) for name in names
        )
        delta_sum = math.fsum(float(delta[name].sum()) for name in names)
        delta_max = max(
            float(delta[name].abs().max()) for name in names
        )
        weight_norm = math.sqrt(weight_square)
        delta_norm = math.sqrt(delta_square)
        gradient_norm = math.sqrt(gradient_square)
        endpoint_norm = math.sqrt(endpoint_square)
        previous_norm = 0.0
        delta_previous_dot = 0.0
        if self._previous_delta is not None:
            previous_square = math.fsum(
                float(self._previous_delta[name].square().sum())
                for name in names
            )
            previous_norm = math.sqrt(previous_square)
            delta_previous_dot = math.fsum(
                float((delta[name] * self._previous_delta[name]).sum())
                for name in names
            )
        if self.step_statistics:
            path_length: float | None = float(
                self._path_length[statistic_group].detach().cpu()
            )
            cosine_count = float(
                self._step_update_gradient_cosine_count[
                    statistic_group
                ].detach().cpu()
            )
            step_gradient_cosine = (
                None
                if cosine_count == 0
                else float(
                    self._step_update_gradient_cosine_sum[
                        statistic_group
                    ].detach().cpu()
                )
                / cosine_count
            )
        else:
            path_length = None
            step_gradient_cosine = None
        return {
            "parameter_count": count,
            "weight_norm": weight_norm,
            "delta_norm": delta_norm,
            "delta_rms": delta_norm / math.sqrt(max(count, 1)),
            "relative_delta_norm": delta_norm / max(weight_norm, 1e-30),
            "delta_mean": delta_sum / max(count, 1),
            "delta_max_abs": delta_max,
            "changed_fraction": changed / max(count, 1),
            "mean_gradient_norm": gradient_norm,
            "gradient_rms": math.sqrt(
                gradient_element_square / max(count, 1)
            ),
            "rms_step_update": (
                math.sqrt(rms_update_square / max(count, 1))
                if self.step_statistics
                else None
            ),
            "first_moment_norm": math.sqrt(first_moment_square),
            "second_moment_rms": math.sqrt(
                second_moment_sum / max(count, 1)
            ),
            "endpoint_adam_update_norm": endpoint_norm,
            "path_length": path_length,
            "path_efficiency": (
                None
                if path_length is None
                else 0.0
                if path_length == 0.0
                else delta_norm / path_length
            ),
            "mean_step_update_norm": (
                None
                if path_length is None
                else 0.0
                if self._window_steps == 0
                else path_length / self._window_steps
            ),
            "cos_delta_previous": _cosine(
                delta_previous_dot, delta_norm, previous_norm
            ),
            "cos_delta_negative_mean_gradient": _cosine(
                delta_negative_gradient_dot,
                delta_norm,
                gradient_norm,
            ),
            "cos_delta_endpoint_adam_update": _cosine(
                delta_endpoint_dot,
                delta_norm,
                endpoint_norm,
            ),
            "mean_step_update_negative_gradient_cosine": (
                step_gradient_cosine
            ),
        }
