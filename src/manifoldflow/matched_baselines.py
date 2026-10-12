"""Matched clean/noisy evaluation for flow and simpler prototype memories."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from manifoldflow.bank import BasinState
from manifoldflow.conditional import (
    ConditionalGateSpec,
    apply_conditional_gate,
    routed_flow_evidence,
    semantic_hybrid_name,
)
from manifoldflow.flow import analytic_topk_flow
from manifoldflow.index import ExactIndex
from manifoldflow.models import CompactImageAutoencoder
from manifoldflow.normalization import FrozenDiagonalWhitening
from manifoldflow.prototype import (
    contract_toward,
    nearest_prototype_target,
    rbf_prototype_target,
)
from manifoldflow.routing import rerank_topk


def _soft_flow_variants(
    latent: Tensor,
    bank: BasinState,
    index: ExactIndex,
    *,
    candidates: int,
    top_k: int,
    times: tuple[float, ...],
    temperature: float,
) -> dict[str, Tensor]:
    candidate_ids = index.query(latent, candidates)
    _, mu, raw_rate, bias, _ = rerank_topk(latent, candidate_ids, bank, top_k=top_k)
    result = {}
    for flow_time in times:
        flowed, _, _ = analytic_topk_flow(
            latent,
            mu,
            raw_rate,
            bias,
            t=flow_time,
            temperature=temperature,
        )
        result[f"flow_soft_t{flow_time:g}"] = flowed
    return result


def _prototype_variants(
    latent: Tensor,
    prototype_centers: Tensor,
    class_centers: Tensor,
    prototype_index: ExactIndex,
    class_index: ExactIndex,
    *,
    strengths: tuple[float, ...],
    rbf_top_k: int,
    rbf_temperature: float,
) -> dict[str, Tensor]:
    nearest_target = nearest_prototype_target(
        latent, prototype_centers, index=prototype_index
    )
    rbf_target = rbf_prototype_target(
        latent,
        prototype_centers,
        top_k=rbf_top_k,
        temperature=rbf_temperature,
        index=prototype_index,
    )
    class_target = nearest_prototype_target(latent, class_centers, index=class_index)
    result = {"vq_projection": nearest_target}
    for strength in strengths:
        suffix = f"a{strength:g}"
        result[f"prototype_{suffix}"] = contract_toward(
            latent, nearest_target, strength=strength
        )
        result[f"rbf_{suffix}"] = contract_toward(latent, rbf_target, strength=strength)
        result[f"class_centroid_{suffix}"] = contract_toward(
            latent, class_target, strength=strength
        )
    return result


@torch.no_grad()
def evaluate_matched_baselines(
    model: CompactImageAutoencoder,
    normalizer: FrozenDiagonalWhitening,
    probe: nn.Module,
    flow_bank: BasinState,
    prototype_centers: Tensor,
    class_centers: Tensor,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    device: torch.device | str,
    flow_times: tuple[float, ...] = (0.05, 0.1),
    strengths: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 1.0),
    candidates: int = 64,
    flow_top_k: int = 4,
    flow_temperature: float = 0.1,
    rbf_top_k: int = 4,
    rbf_temperature: float = 0.1,
    flow_class_probability: Tensor | None = None,
    conditional_gates: tuple[ConditionalGateSpec, ...] | None = None,
    conditional_confidence_threshold: float | None = None,
    conditional_distance_threshold: float | None = None,
    conditional_flow_time: float = 0.1,
    diagnostic_rbf_strength: float = 0.2,
    include_semantic_hybrids: bool = False,
    image_noise_std: float = 0.2,
    skip_samples: int = 0,
    max_samples: int | None = 2000,
    seed: int = 17,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate flow and prototype transforms with one decoder and probe."""
    if not flow_times or any(value <= 0 for value in flow_times):
        raise ValueError("flow_times must contain positive values")
    if not strengths or any(not 0 < value <= 1 for value in strengths):
        raise ValueError("strengths must contain values in (0, 1]")
    if image_noise_std < 0:
        raise ValueError("image_noise_std must be non-negative")
    if skip_samples < 0:
        raise ValueError("skip_samples must be non-negative")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when supplied")
    candidates = min(candidates, flow_bank.mu.shape[0])
    if not 1 <= flow_top_k <= candidates:
        raise ValueError("flow_top_k cannot exceed candidates")
    legacy_conditional = (
        conditional_confidence_threshold is not None
        or conditional_distance_threshold is not None
    )
    if legacy_conditional and (
        conditional_confidence_threshold is None
        or conditional_distance_threshold is None
    ):
        raise ValueError("both conditional thresholds must be supplied")
    if conditional_gates is not None and legacy_conditional:
        raise ValueError("conditional_gates cannot be combined with legacy thresholds")
    if conditional_gates is None and legacy_conditional:
        conditional_gates = (
            ConditionalGateSpec(
                name=f"conditional_flow_t{conditional_flow_time:g}",
                confidence_threshold=conditional_confidence_threshold,
                distance_threshold=conditional_distance_threshold,
            ),
        )
    conditional_gates = conditional_gates or ()
    gate_names = [spec.name for spec in conditional_gates]
    if len(set(gate_names)) != len(gate_names):
        raise ValueError("conditional gate names must be unique")
    if any(spec.semantic_margin_threshold is not None for spec in conditional_gates):
        if flow_class_probability is None:
            raise ValueError("semantic gates require flow_class_probability")
    conditional_enabled = bool(conditional_gates)
    if conditional_enabled and conditional_flow_time not in flow_times:
        raise ValueError("conditional_flow_time must be present in flow_times")
    if conditional_enabled and diagnostic_rbf_strength not in strengths:
        raise ValueError("diagnostic_rbf_strength must be present in strengths")
    if include_semantic_hybrids and not conditional_enabled:
        raise ValueError("semantic hybrids require conditional_gates")
    if include_semantic_hybrids and any(
        spec.semantic_margin_threshold is None for spec in conditional_gates
    ):
        raise ValueError("semantic hybrids require semantic conditional gates")

    model.to(device).eval()
    normalizer.to(device).eval()
    probe.to(device).eval()
    flow_index = ExactIndex(flow_bank.mu)
    prototype_index = ExactIndex(prototype_centers)
    class_index = ExactIndex(class_centers)
    generator = torch.Generator(device=device).manual_seed(seed)
    totals: dict[str, dict[str, dict[str, float]]] = {}
    sample_count = pixel_count = 0
    skip_remaining = skip_samples

    for image, label in batches:
        if skip_remaining >= image.shape[0]:
            skip_remaining -= image.shape[0]
            continue
        if skip_remaining:
            image, label = image[skip_remaining:], label[skip_remaining:]
            skip_remaining = 0
        if max_samples is not None:
            remaining = max_samples - sample_count
            if remaining <= 0:
                break
            image, label = image[:remaining], label[:remaining]
        image = image.to(device)
        label = label.to(device)
        noise = torch.randn(
            image.shape,
            generator=generator,
            device=device,
            dtype=image.dtype,
        )
        noisy_image = (image + noise * image_noise_std).clamp(0, 1)

        for condition, model_input in (("clean", image), ("noisy", noisy_image)):
            z0 = normalizer(model.encode(model_input))
            variants = {"t0": z0}
            variants.update(
                _soft_flow_variants(
                    z0,
                    flow_bank,
                    flow_index,
                    candidates=candidates,
                    top_k=flow_top_k,
                    times=flow_times,
                    temperature=flow_temperature,
                )
            )
            variants.update(
                _prototype_variants(
                    z0,
                    prototype_centers,
                    class_centers,
                    prototype_index,
                    class_index,
                    strengths=strengths,
                    rbf_top_k=rbf_top_k,
                    rbf_temperature=rbf_temperature,
                )
            )
            conditional_active: dict[str, Tensor] = {}
            if conditional_enabled:
                evidence = routed_flow_evidence(
                    z0,
                    flow_bank,
                    candidates=candidates,
                    top_k=flow_top_k,
                    flow_time=conditional_flow_time,
                    temperature=flow_temperature,
                    index=flow_index,
                    class_probability=flow_class_probability,
                )
                for spec in conditional_gates:
                    conditional = apply_conditional_gate(z0, evidence, spec)
                    if spec.name in variants:
                        raise ValueError(
                            f"conditional gate name {spec.name!r} "
                            "collides with a baseline"
                        )
                    variants[spec.name] = conditional.latent
                    conditional_active[spec.name] = conditional.active
                    if include_semantic_hybrids:
                        rbf_name = f"rbf_a{diagnostic_rbf_strength:g}"
                        hybrid_name = semantic_hybrid_name(
                            spec.name, diagnostic_rbf_strength
                        )
                        if hybrid_name in variants:
                            raise ValueError(
                                f"semantic hybrid name {hybrid_name!r} "
                                "collides with a baseline"
                            )
                        variants[hybrid_name] = torch.where(
                            conditional.active[:, None],
                            conditional.flowed_latent,
                            variants[rbf_name],
                        )
                        conditional_active[hybrid_name] = conditional.active
            predictions = {}
            for variant, latent in variants.items():
                entry = totals.setdefault(variant, {}).setdefault(
                    condition,
                    {"squared_error": 0.0, "correct": 0.0, "displacement": 0.0},
                )
                reconstruction = model.decode(latent)
                entry["squared_error"] += float(
                    (reconstruction - image).square().sum().item()
                )
                prediction = probe(latent).argmax(dim=-1)
                predictions[variant] = prediction
                entry["correct"] += float((prediction == label).sum().item())
                entry["displacement"] += float((latent - z0).norm(dim=-1).sum().item())

            if conditional_enabled:
                rbf_name = f"rbf_a{diagnostic_rbf_strength:g}"
                flow_name = f"flow_soft_t{conditional_flow_time:g}"
                rbf_correct = predictions[rbf_name] == label
                targets = [(flow_name, torch.ones_like(rbf_correct, dtype=torch.bool))]
                targets.extend(conditional_active.items())
                for target_name, mask in targets:
                    target_correct = predictions[target_name] == label
                    entry = totals[target_name][condition]
                    entry["rbf_correct_to_wrong"] = entry.get(
                        "rbf_correct_to_wrong", 0.0
                    ) + float((mask & rbf_correct & ~target_correct).sum().item())
                    entry["rbf_wrong_to_correct"] = entry.get(
                        "rbf_wrong_to_correct", 0.0
                    ) + float((mask & ~rbf_correct & target_correct).sum().item())
                for conditional_name, active in conditional_active.items():
                    conditional_entry = totals[conditional_name][condition]
                    conditional_entry["activation"] = conditional_entry.get(
                        "activation", 0.0
                    ) + float(active.sum().item())

        sample_count += image.shape[0]
        pixel_count += image.numel()

    if sample_count == 0:
        raise ValueError("evaluation slice produced no samples")
    result = {}
    for variant, conditions in totals.items():
        result[variant] = {}
        for condition, value in conditions.items():
            result[variant][condition] = {
                "reconstruction_mse": value["squared_error"] / pixel_count,
                "classification_accuracy": value["correct"] / sample_count,
                "mean_latent_displacement": value["displacement"] / sample_count,
            }
            for count_name in (
                "rbf_correct_to_wrong",
                "rbf_wrong_to_correct",
                "activation",
            ):
                if count_name in value:
                    result[variant][condition][f"{count_name}_count"] = value[
                        count_name
                    ]
                    result[variant][condition][f"{count_name}_fraction"] = (
                        value[count_name] / sample_count
                    )
    return result
