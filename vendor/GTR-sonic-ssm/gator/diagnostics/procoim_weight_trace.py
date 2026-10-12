"""Interval tracing of Procoim's exact parameter-space displacement.

Mirrors ``gator.diagnostics.adamw_weight_trace.AdamWWeightTracer`` closely
enough that ``Runbooks/colab_cells/_analyze_trace_geometry.py``'s
methodology (global update-vector cosine with the previous interval;
per-parameter rank-4/rank-8 SVD energy capture) runs unmodified on Procoim
traces too -- that analysis only reads ``payload["interval_delta"]``, a
dict of per-parameter tensors giving the net parameter change since the
previous snapshot.

Procoim has no gradients and no AdamW-style Adam moments in parameter
space (``--procoim-directional-adam``'s own Adam-like moments live in a
K-dimensional coefficient space, not per-parameter), so this tracer is
much smaller than ``AdamWWeightTracer``: it records only what a Procoim
"step" actually produces -- the accepted, verified parameter delta --
snapshotted every ``interval`` forward calls (Procoim's own step unit,
directly analogous to how AdamW's ``--steps``/trace-interval count
optimizer steps).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ProcoimTraceSnapshot:
    """One persisted interval and its compact human-readable summary."""

    step: int
    interval_start_step: int
    interval_forwards: int
    artifact_path: Path | None
    summary: dict[str, Any]

    def log_line(self) -> str:
        global_stats = self.summary["global"]
        cosine = global_stats["cos_delta_previous"]
        artifact = (
            "none" if self.artifact_path is None else str(self.artifact_path)
        )
        groups = ",".join(
            f"{name}:d={metrics['delta_rms']:.2e}"
            for name, metrics in self.summary["groups"].items()
        )
        return (
            f"procoim_weight_trace step={self.step} "
            f"interval={self.interval_start_step}:{self.step} "
            f"interval_forwards={self.interval_forwards} "
            f"delta_norm={global_stats['delta_norm']:.6g} "
            f"delta_rms={global_stats['delta_rms']:.6g} "
            f"relative_delta={global_stats['relative_delta_norm']:.6g} "
            f"prev_cos={_format_optional(cosine)} "
            f"groups=[{groups}] artifact={artifact!r}"
        )


def _format_optional(value: float | None) -> str:
    return "nan" if value is None else f"{value:.6g}"


def _cosine(dot: float, left_norm: float, right_norm: float) -> float | None:
    denominator = left_norm * right_norm
    if denominator <= 0.0:
        return None
    return max(-1.0, min(1.0, dot / denominator))


class ProcoimWeightTracer:
    """Trace Procoim's exact parameter-space displacement per interval."""

    format_version = 1

    def __init__(
        self,
        model: nn.Module,
        output_dir: str | Path,
        *,
        interval: int = 500,
        save_tensors: bool = True,
    ) -> None:
        if isinstance(interval, bool) or not isinstance(interval, int):
            raise TypeError("trace interval must be an integer")
        if interval <= 0:
            raise ValueError("trace interval must be positive")

        self.model = model
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.save_tensors = bool(save_tensors)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.output_dir / "summary.jsonl"
        if self.summary_path.exists() or any(
            self.output_dir.glob("trace_step_*.pt")
        ):
            raise FileExistsError(
                f"Procoim trace directory is not empty: {self.output_dir}"
            )

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

        self._names: list[str] = []
        self._parameters: dict[str, nn.Parameter] = {}
        self._semantic_group: dict[str, str] = {}
        for name, parameter in model.named_parameters():
            self._names.append(name)
            self._parameters[name] = parameter
            self._semantic_group[name] = semantic_groups.get(
                id(parameter), "ungrouped"
            )
        if not self._names:
            raise ValueError("cannot trace a model without parameters")
        self._group_names = tuple(sorted(set(self._semantic_group.values())))

        self._previous_parameters = self._cpu_parameters()
        self._previous_delta: dict[str, Tensor] | None = None
        self._last_snapshot_step = 0
        self._window_start_step = 0

    def _cpu_parameters(self) -> dict[str, Tensor]:
        return {
            name: parameter.detach().to(
                device="cpu", dtype=torch.float32
            ).clone()
            for name, parameter in self._parameters.items()
        }

    def maybe_snapshot(
        self,
        step: int,
        *,
        force: bool = False,
        context: Mapping[str, Any] | None = None,
    ) -> ProcoimTraceSnapshot | None:
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError("trace step must be an integer")
        if step < 0:
            raise ValueError("trace step must be nonnegative")
        if step < self._last_snapshot_step:
            raise ValueError("trace steps must be monotonic")
        if step == self._last_snapshot_step and not force:
            return None
        crossed = (
            step // self.interval
            > self._window_start_step // self.interval
        )
        if not force and not crossed:
            return None
        return self._snapshot(step, context=context)

    @torch.no_grad()
    def _snapshot(
        self,
        step: int,
        *,
        context: Mapping[str, Any] | None,
    ) -> ProcoimTraceSnapshot:
        current = self._cpu_parameters()
        delta = {
            name: current[name] - self._previous_parameters[name]
            for name in current
        }

        names_by_group: dict[str, list[str]] = {
            group_name: [] for group_name in self._group_names
        }
        for name in self._names:
            names_by_group[self._semantic_group[name]].append(name)

        global_stats = self._vector_statistics(
            tuple(current), current=current, delta=delta
        )
        group_stats = {
            group_name: self._vector_statistics(
                tuple(names), current=current, delta=delta
            )
            for group_name, names in sorted(names_by_group.items())
            if names
        }

        summary: dict[str, Any] = {
            "format": "gator-procoim-weight-trace-summary-v1",
            "step": step,
            "interval_start_step": self._window_start_step,
            "interval_forwards": step - self._window_start_step,
            "global": global_stats,
            "groups": group_stats,
            "context": dict(context or {}),
        }

        artifact_path: Path | None = None
        if self.save_tensors:
            artifact_path = self.output_dir / f"trace_step_{step:09d}.pt"
            temporary = artifact_path.with_suffix(".pt.tmp")
            payload = {
                "format": "gator-procoim-weight-trace",
                "version": self.format_version,
                "summary": summary,
                "parameter_groups": dict(self._semantic_group),
                "parameters": current,
                "interval_delta": delta,
            }
            torch.save(payload, temporary)
            temporary.replace(artifact_path)

        with self.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(summary, sort_keys=True, allow_nan=False) + "\n"
            )

        result = ProcoimTraceSnapshot(
            step=step,
            interval_start_step=self._window_start_step,
            interval_forwards=step - self._window_start_step,
            artifact_path=artifact_path,
            summary=summary,
        )
        self._previous_parameters = current
        self._previous_delta = {
            name: value.clone() for name, value in delta.items()
        }
        self._last_snapshot_step = step
        self._window_start_step = step
        return result

    def _vector_statistics(
        self,
        names: tuple[str, ...],
        *,
        current: Mapping[str, Tensor],
        delta: Mapping[str, Tensor],
    ) -> dict[str, Any]:
        count = sum(current[name].numel() for name in names)
        weight_square = math.fsum(
            float(current[name].square().sum()) for name in names
        )
        delta_square = math.fsum(
            float(delta[name].square().sum()) for name in names
        )
        delta_sum = math.fsum(float(delta[name].sum()) for name in names)
        delta_max = max(
            (float(delta[name].abs().max()) for name in names),
            default=0.0,
        )
        changed = sum(int(delta[name].ne(0).sum()) for name in names)
        weight_norm = math.sqrt(weight_square)
        delta_norm = math.sqrt(delta_square)
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
        return {
            "parameter_count": count,
            "weight_norm": weight_norm,
            "delta_norm": delta_norm,
            "delta_rms": delta_norm / math.sqrt(max(count, 1)),
            "relative_delta_norm": delta_norm / max(weight_norm, 1e-30),
            "delta_mean": delta_sum / max(count, 1),
            "delta_max_abs": delta_max,
            "changed_fraction": changed / max(count, 1),
            "cos_delta_previous": _cosine(
                delta_previous_dot, delta_norm, previous_norm
            ),
        }
