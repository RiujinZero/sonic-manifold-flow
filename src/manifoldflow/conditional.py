"""Confidence, corruption, and semantic-boundary gating for analytic flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from manifoldflow.bank import BasinState
from manifoldflow.flow import analytic_topk_flow, positive_rate
from manifoldflow.index import ExactIndex
from manifoldflow.routing import rerank_topk


@dataclass(frozen=True)
class ConditionalGateSpec:
    """Thresholds for one conditional-flow operating point."""

    name: str
    confidence_threshold: float
    distance_threshold: float
    semantic_margin_threshold: float | None = None


@dataclass(frozen=True)
class RoutedFlowEvidence:
    """One routed flow plus the reusable evidence needed by gate variants."""

    flowed_latent: Tensor
    confidence: Tensor
    local_distance: Tensor
    semantic_margin: Tensor | None


@dataclass(frozen=True)
class ConditionalFlowResult:
    """Selective flow output and the evidence used by its gate."""

    latent: Tensor
    flowed_latent: Tensor
    active: Tensor
    confidence: Tensor
    local_distance: Tensor
    semantic_margin: Tensor | None


@dataclass(frozen=True)
class ConditionalGateCalibration:
    """Clean-training calibration for one conditional-flow gate."""

    distance_threshold: float
    confidence_threshold: float
    distance_quantile: float
    samples: int
    confident_fraction: float
    activation_fraction: float
    semantic_margin_threshold: float | None = None
    semantic_eligible_fraction: float = 1.0
    name: str = "conditional_flow"

    @property
    def spec(self) -> ConditionalGateSpec:
        return ConditionalGateSpec(
            name=self.name,
            confidence_threshold=self.confidence_threshold,
            distance_threshold=self.distance_threshold,
            semantic_margin_threshold=self.semantic_margin_threshold,
        )


def _validate_class_probability(
    class_probability: Tensor | None,
    bank: BasinState,
) -> None:
    if class_probability is None:
        return
    if (
        class_probability.ndim != 2
        or class_probability.shape[0] != bank.mu.shape[0]
        or class_probability.shape[1] < 2
    ):
        raise ValueError(
            "class_probability must have shape [basins, classes] with classes >= 2"
        )
    if torch.any(class_probability < 0):
        raise ValueError("class_probability must be non-negative")
    row_sum = class_probability.sum(dim=1)
    if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-5, rtol=1e-5):
        raise ValueError("class_probability rows must sum to one")


def _validate_spec(spec: ConditionalGateSpec, *, has_semantics: bool) -> None:
    if not spec.name:
        raise ValueError("conditional gate names must be non-empty")
    if not 0 <= spec.confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if spec.distance_threshold < 0:
        raise ValueError("distance_threshold must be non-negative")
    if spec.semantic_margin_threshold is not None:
        if not has_semantics:
            raise ValueError("semantic gating requires basin class probabilities")
        if not 0 <= spec.semantic_margin_threshold <= 1:
            raise ValueError("semantic_margin_threshold must be in [0, 1]")


@torch.no_grad()
def routed_flow_evidence(
    latent: Tensor,
    bank: BasinState,
    *,
    candidates: int = 64,
    top_k: int = 4,
    flow_time: float = 0.1,
    temperature: float = 0.1,
    index: ExactIndex | None = None,
    class_probability: Tensor | None = None,
) -> RoutedFlowEvidence:
    """Route and flow once, optionally measuring a soft class-mixture margin."""
    if flow_time <= 0 or temperature <= 0:
        raise ValueError("flow_time and temperature must be positive")
    candidates = min(candidates, bank.mu.shape[0])
    if not 1 <= top_k <= candidates:
        raise ValueError("top_k cannot exceed candidates")
    _validate_class_probability(class_probability, bank)
    index = index if index is not None else ExactIndex(bank.mu)
    candidate_ids = index.query(latent, candidates)
    ids, mu, raw_rate, bias, _ = rerank_topk(latent, candidate_ids, bank, top_k=top_k)
    flowed, _, weight = analytic_topk_flow(
        latent,
        mu,
        raw_rate,
        bias,
        t=flow_time,
        temperature=temperature,
    )
    confidence, dominant = weight.max(dim=-1)
    dominant_mu = mu.gather(
        1, dominant[:, None, None].expand(-1, 1, mu.shape[-1])
    ).squeeze(1)
    rate = positive_rate(raw_rate)
    dominant_rate = rate.gather(
        1, dominant[:, None, None].expand(-1, 1, rate.shape[-1])
    ).squeeze(1)
    local_distance = 0.5 * (dominant_rate * (latent - dominant_mu).square()).sum(dim=-1)

    semantic_margin = None
    if class_probability is not None:
        basin_probability = class_probability.to(latent.device)[ids]
        routed_probability = (weight[..., None] * basin_probability).sum(dim=1)
        leading = routed_probability.topk(k=2, dim=-1).values
        semantic_margin = leading[:, 0] - leading[:, 1]
    return RoutedFlowEvidence(
        flowed_latent=flowed,
        confidence=confidence,
        local_distance=local_distance,
        semantic_margin=semantic_margin,
    )


def apply_conditional_gate(
    latent: Tensor,
    evidence: RoutedFlowEvidence,
    spec: ConditionalGateSpec,
) -> ConditionalFlowResult:
    """Apply one gate without repeating retrieval or analytic flow."""
    _validate_spec(spec, has_semantics=evidence.semantic_margin is not None)
    active = (evidence.confidence >= spec.confidence_threshold) & (
        evidence.local_distance >= spec.distance_threshold
    )
    if spec.semantic_margin_threshold is not None:
        assert evidence.semantic_margin is not None
        active &= evidence.semantic_margin >= spec.semantic_margin_threshold
    output = torch.where(active[:, None], evidence.flowed_latent, latent)
    return ConditionalFlowResult(
        latent=output,
        flowed_latent=evidence.flowed_latent,
        active=active,
        confidence=evidence.confidence,
        local_distance=evidence.local_distance,
        semantic_margin=evidence.semantic_margin,
    )


@torch.no_grad()
def conditional_flow(
    latent: Tensor,
    bank: BasinState,
    *,
    confidence_threshold: float,
    distance_threshold: float,
    semantic_margin_threshold: float | None = None,
    candidates: int = 64,
    top_k: int = 4,
    flow_time: float = 0.1,
    temperature: float = 0.1,
    index: ExactIndex | None = None,
    class_probability: Tensor | None = None,
) -> ConditionalFlowResult:
    """Flow only confident, far, and semantically unambiguous routes."""
    evidence = routed_flow_evidence(
        latent,
        bank,
        candidates=candidates,
        top_k=top_k,
        flow_time=flow_time,
        temperature=temperature,
        index=index,
        class_probability=class_probability,
    )
    return apply_conditional_gate(
        latent,
        evidence,
        ConditionalGateSpec(
            name="conditional_flow",
            confidence_threshold=confidence_threshold,
            distance_threshold=distance_threshold,
            semantic_margin_threshold=semantic_margin_threshold,
        ),
    )


@torch.no_grad()
def _clean_gate_evidence(
    clean_latent: Tensor,
    bank: BasinState,
    *,
    candidates: int,
    top_k: int,
    flow_time: float,
    temperature: float,
    chunk_size: int,
    max_samples: int | None,
    class_probability: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, int]:
    if clean_latent.ndim != 2 or clean_latent.shape[1] != bank.mu.shape[1]:
        raise ValueError("clean_latent must match the bank dimension")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when supplied")
    sample = clean_latent if max_samples is None else clean_latent[:max_samples]
    index = ExactIndex(bank.mu)
    confidences = []
    distances = []
    margins = []
    for start in range(0, sample.shape[0], chunk_size):
        evidence = routed_flow_evidence(
            sample[start : start + chunk_size],
            bank,
            candidates=candidates,
            top_k=top_k,
            flow_time=flow_time,
            temperature=temperature,
            index=index,
            class_probability=class_probability,
        )
        confidences.append(evidence.confidence.cpu())
        distances.append(evidence.local_distance.cpu())
        if evidence.semantic_margin is not None:
            margins.append(evidence.semantic_margin.cpu())
    confidence = torch.cat(confidences)
    local_distance = torch.cat(distances)
    semantic_margin = torch.cat(margins) if margins else torch.ones_like(confidence)
    return confidence, local_distance, semantic_margin, sample.shape[0]


def _calibration_from_evidence(
    confidence: Tensor,
    local_distance: Tensor,
    semantic_margin: Tensor,
    *,
    samples: int,
    confidence_threshold: float,
    distance_quantile: float,
    semantic_margin_threshold: float | None,
    name: str,
) -> ConditionalGateCalibration:
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0 < distance_quantile < 1:
        raise ValueError("distance_quantile must be in (0, 1)")
    if (
        semantic_margin_threshold is not None
        and not 0 <= semantic_margin_threshold <= 1
    ):
        raise ValueError("semantic_margin_threshold must be in [0, 1]")
    confident = confidence >= confidence_threshold
    semantic = torch.ones_like(confident)
    if semantic_margin_threshold is not None:
        semantic = semantic_margin >= semantic_margin_threshold
    eligible = confident & semantic
    if not bool(eligible.any()):
        raise ValueError("no clean calibration samples pass the route gates")
    distance_threshold = float(
        torch.quantile(local_distance[eligible], distance_quantile).item()
    )
    active = eligible & (local_distance >= distance_threshold)
    return ConditionalGateCalibration(
        distance_threshold=distance_threshold,
        confidence_threshold=confidence_threshold,
        distance_quantile=distance_quantile,
        samples=samples,
        confident_fraction=float(confident.float().mean().item()),
        activation_fraction=float(active.float().mean().item()),
        semantic_margin_threshold=semantic_margin_threshold,
        semantic_eligible_fraction=float(eligible.float().mean().item()),
        name=name,
    )


@torch.no_grad()
def calibrate_conditional_gate(
    clean_latent: Tensor,
    bank: BasinState,
    *,
    confidence_threshold: float = 0.8,
    distance_quantile: float = 0.9,
    semantic_margin_threshold: float | None = None,
    candidates: int = 64,
    top_k: int = 4,
    flow_time: float = 0.1,
    temperature: float = 0.1,
    chunk_size: int = 512,
    max_samples: int | None = 10000,
    class_probability: Tensor | None = None,
    name: str = "conditional_flow",
) -> ConditionalGateCalibration:
    """Set the distance threshold from clean latents passing route gates."""
    if semantic_margin_threshold is not None and class_probability is None:
        raise ValueError("semantic gating requires basin class probabilities")
    confidence, distance, margin, samples = _clean_gate_evidence(
        clean_latent,
        bank,
        candidates=candidates,
        top_k=top_k,
        flow_time=flow_time,
        temperature=temperature,
        chunk_size=chunk_size,
        max_samples=max_samples,
        class_probability=class_probability,
    )
    return _calibration_from_evidence(
        confidence,
        distance,
        margin,
        samples=samples,
        confidence_threshold=confidence_threshold,
        distance_quantile=distance_quantile,
        semantic_margin_threshold=semantic_margin_threshold,
        name=name,
    )


def semantic_gate_name(
    confidence: float,
    quantile: float,
    semantic_margin: float,
    flow_time: float,
) -> str:
    return (
        f"semantic_flow_c{confidence:g}_q{quantile:g}_"
        f"m{semantic_margin:g}_t{flow_time:g}"
    )


def semantic_hybrid_name(gate_name: str, rbf_strength: float) -> str:
    """Name an RBF-default policy specialized by one semantic flow gate."""
    if not gate_name.startswith("semantic_flow_"):
        raise ValueError("semantic hybrid gates must use a semantic_flow name")
    if not 0 < rbf_strength <= 1:
        raise ValueError("rbf_strength must be in (0, 1]")
    suffix = gate_name.removeprefix("semantic_flow_")
    return f"semantic_hybrid_rbf{rbf_strength:g}_{suffix}"


@torch.no_grad()
def calibrate_conditional_gate_grid(
    clean_latent: Tensor,
    bank: BasinState,
    class_probability: Tensor,
    *,
    confidence_thresholds: tuple[float, ...],
    distance_quantiles: tuple[float, ...],
    semantic_margin_thresholds: tuple[float, ...],
    candidates: int = 64,
    top_k: int = 4,
    flow_time: float = 0.1,
    temperature: float = 0.1,
    chunk_size: int = 512,
    max_samples: int | None = 10000,
) -> tuple[ConditionalGateCalibration, ...]:
    """Calibrate a semantic gate grid with one clean-latent routing pass."""
    if (
        not confidence_thresholds
        or not distance_quantiles
        or not semantic_margin_thresholds
    ):
        raise ValueError("conditional gate grids must be non-empty")
    confidence, distance, margin, samples = _clean_gate_evidence(
        clean_latent,
        bank,
        candidates=candidates,
        top_k=top_k,
        flow_time=flow_time,
        temperature=temperature,
        chunk_size=chunk_size,
        max_samples=max_samples,
        class_probability=class_probability,
    )
    result = []
    for confidence_threshold in confidence_thresholds:
        for distance_quantile in distance_quantiles:
            for semantic_margin_threshold in semantic_margin_thresholds:
                result.append(
                    _calibration_from_evidence(
                        confidence,
                        distance,
                        margin,
                        samples=samples,
                        confidence_threshold=confidence_threshold,
                        distance_quantile=distance_quantile,
                        semantic_margin_threshold=semantic_margin_threshold,
                        name=semantic_gate_name(
                            confidence_threshold,
                            distance_quantile,
                            semantic_margin_threshold,
                            flow_time,
                        ),
                    )
                )
    return tuple(result)
