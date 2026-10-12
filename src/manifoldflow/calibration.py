"""Decoder calibration and matched flow ablations for the frozen backbone."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from manifoldflow.bank import BasinState
from manifoldflow.flow import analytic_topk_flow
from manifoldflow.index import ExactIndex
from manifoldflow.models import CompactImageAutoencoder
from manifoldflow.normalization import FrozenDiagonalWhitening
from manifoldflow.prototype import (
    contract_toward,
    nearest_prototype_target,
    rbf_prototype_target,
)
from manifoldflow.routing import exact_bank_flow, rerank_topk


@dataclass(frozen=True)
class DecoderCalibrationResult:
    """Per-epoch decoder losses for normalized and flowed latent branches."""

    total_loss: tuple[float, ...]
    normalized_loss: tuple[float, ...]
    flowed_loss: tuple[float, ...]


@dataclass(frozen=True)
class NeutralDecoderCalibrationResult:
    """Decoder calibration losses with balanced transform exposure."""

    total_loss: tuple[float, ...]
    normalized_loss: tuple[float, ...]
    transformed_loss: tuple[float, ...]
    branch_loss: dict[str, tuple[float, ...]]


def _validate_calibration_settings(
    *,
    epochs: int,
    learning_rate: float,
    flow_time: float,
    latent_noise_std: float,
    normalized_weight: float,
    flowed_weight: float,
) -> None:
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if flow_time < 0 or latent_noise_std < 0:
        raise ValueError("flow_time and latent_noise_std must be non-negative")
    if normalized_weight < 0 or flowed_weight < 0:
        raise ValueError("calibration weights must be non-negative")
    if normalized_weight + flowed_weight <= 0:
        raise ValueError("at least one calibration branch must have positive weight")


def _perturb_latent(
    latent: Tensor,
    *,
    standard_deviation: float,
    l2_normalize: bool,
) -> Tensor:
    if standard_deviation == 0:
        return latent
    perturbed = latent + torch.randn_like(latent) * standard_deviation
    if l2_normalize:
        perturbed = F.normalize(perturbed, p=2, dim=-1)
    return perturbed


def calibrate_decoder(
    model: CompactImageAutoencoder,
    normalizer: FrozenDiagonalWhitening,
    bank: BasinState,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    device: torch.device | str,
    epochs: int = 5,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    flow_time: float = 0.25,
    latent_noise_std: float = 0.02,
    candidates: int = 64,
    normalized_weight: float = 0.5,
    flowed_weight: float = 0.5,
    max_batches: int | None = None,
) -> DecoderCalibrationResult:
    """Fit only the decoder to normalized and gently contracted latents."""
    _validate_calibration_settings(
        epochs=epochs,
        learning_rate=learning_rate,
        flow_time=flow_time,
        latent_noise_std=latent_noise_std,
        normalized_weight=normalized_weight,
        flowed_weight=flowed_weight,
    )
    if candidates <= 0:
        raise ValueError("candidates must be positive")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when supplied")

    candidates = min(candidates, bank.mu.shape[0])
    model.to(device)
    normalizer.to(device).eval()
    model.freeze_encoder()
    decoder_parameters = model.train_decoder()
    optimizer = torch.optim.AdamW(
        decoder_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_history = []
    normalized_history = []
    flowed_history = []
    weight_total = normalized_weight + flowed_weight
    for _ in range(epochs):
        loss_sum = normalized_sum = flowed_sum = 0.0
        sample_count = 0
        for batch_index, (image, _) in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            image = image.to(device)
            with torch.no_grad():
                z0 = normalizer(model.encode(image))
                flow_input = _perturb_latent(
                    z0,
                    standard_deviation=latent_noise_std,
                    l2_normalize=normalizer.l2_normalize,
                )
                flowed = exact_bank_flow(
                    flow_input,
                    bank,
                    candidates=candidates,
                    top_k=1,
                    t=flow_time,
                ).latent

            optimizer.zero_grad(set_to_none=True)
            normalized_loss = F.mse_loss(model.decode(z0), image)
            flowed_loss = F.mse_loss(model.decode(flowed), image)
            loss = (
                normalized_weight * normalized_loss + flowed_weight * flowed_loss
            ) / weight_total
            loss.backward()
            optimizer.step()

            count = image.shape[0]
            loss_sum += float(loss.detach()) * count
            normalized_sum += float(normalized_loss.detach()) * count
            flowed_sum += float(flowed_loss.detach()) * count
            sample_count += count

        if sample_count == 0:
            raise ValueError("calibration batches produced no samples")
        total_history.append(loss_sum / sample_count)
        normalized_history.append(normalized_sum / sample_count)
        flowed_history.append(flowed_sum / sample_count)

    return DecoderCalibrationResult(
        total_loss=tuple(total_history),
        normalized_loss=tuple(normalized_history),
        flowed_loss=tuple(flowed_history),
    )


def calibrate_neutral_decoder(
    model: CompactImageAutoencoder,
    normalizer: FrozenDiagonalWhitening,
    flow_bank: BasinState,
    prototype_centers: Tensor,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    device: torch.device | str,
    epochs: int = 5,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    latent_noise_std: float = 0.02,
    flow_times: tuple[float, ...] = (0.05, 0.1),
    prototype_strengths: tuple[float, ...] = (0.1, 0.2),
    candidates: int = 64,
    flow_top_k: int = 4,
    flow_temperature: float = 0.1,
    rbf_top_k: int = 4,
    rbf_temperature: float = 0.1,
    normalized_weight: float = 0.5,
    transformed_weight: float = 0.5,
    max_batches: int | None = None,
) -> NeutralDecoderCalibrationResult:
    """Calibrate one decoder with balanced flow/prototype/RBF exposure.

    Every batch includes the untransformed normalized branch and exactly one
    transformed branch. Transform branches rotate deterministically so decoder
    compute stays bounded while each family receives equal exposure.
    """
    _validate_calibration_settings(
        epochs=epochs,
        learning_rate=learning_rate,
        flow_time=min(flow_times, default=-1.0),
        latent_noise_std=latent_noise_std,
        normalized_weight=normalized_weight,
        flowed_weight=transformed_weight,
    )
    if not flow_times or any(value <= 0 for value in flow_times):
        raise ValueError("flow_times must contain positive values")
    if not prototype_strengths or any(
        not 0 < value <= 1 for value in prototype_strengths
    ):
        raise ValueError("prototype_strengths must contain values in (0, 1]")
    if prototype_centers.ndim != 2 or prototype_centers.shape[1] != model.latent_dim:
        raise ValueError("prototype_centers must match the model latent width")
    if candidates <= 0 or not 1 <= flow_top_k <= min(candidates, flow_bank.mu.shape[0]):
        raise ValueError("flow routing sizes are invalid")
    if not 1 <= rbf_top_k <= prototype_centers.shape[0]:
        raise ValueError("rbf_top_k cannot exceed the prototype count")
    if flow_temperature <= 0 or rbf_temperature <= 0:
        raise ValueError("routing temperatures must be positive")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when supplied")

    candidates = min(candidates, flow_bank.mu.shape[0])
    branches = (
        tuple((f"flow_t{value:g}", "flow", value) for value in flow_times)
        + tuple(
            (f"prototype_a{value:g}", "prototype", value)
            for value in prototype_strengths
        )
        + tuple((f"rbf_a{value:g}", "rbf", value) for value in prototype_strengths)
    )
    flow_index = ExactIndex(flow_bank.mu)
    prototype_index = ExactIndex(prototype_centers)
    model.to(device)
    normalizer.to(device).eval()
    model.freeze_encoder()
    decoder_parameters = model.train_decoder()
    optimizer = torch.optim.AdamW(
        decoder_parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    total_history = []
    normalized_history = []
    transformed_history = []
    branch_history = {name: [] for name, _, _ in branches}
    weight_total = normalized_weight + transformed_weight
    branch_cursor = 0
    for _ in range(epochs):
        loss_sum = normalized_sum = transformed_sum = 0.0
        branch_sum = {name: 0.0 for name, _, _ in branches}
        branch_count = {name: 0 for name, _, _ in branches}
        sample_count = 0
        for batch_index, (image, _) in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            image = image.to(device)
            branch_name, branch_kind, branch_value = branches[
                branch_cursor % len(branches)
            ]
            branch_cursor += 1
            with torch.no_grad():
                z0 = normalizer(model.encode(image))
                transform_input = _perturb_latent(
                    z0,
                    standard_deviation=latent_noise_std,
                    l2_normalize=normalizer.l2_normalize,
                )
                if branch_kind == "flow":
                    candidate_ids = flow_index.query(transform_input, candidates)
                    _, mu, raw_rate, bias, _ = rerank_topk(
                        transform_input,
                        candidate_ids,
                        flow_bank,
                        top_k=flow_top_k,
                    )
                    transformed, _, _ = analytic_topk_flow(
                        transform_input,
                        mu,
                        raw_rate,
                        bias,
                        t=branch_value,
                        temperature=flow_temperature,
                    )
                elif branch_kind == "prototype":
                    target = nearest_prototype_target(
                        transform_input,
                        prototype_centers,
                        index=prototype_index,
                    )
                    transformed = contract_toward(
                        transform_input, target, strength=branch_value
                    )
                else:
                    target = rbf_prototype_target(
                        transform_input,
                        prototype_centers,
                        top_k=rbf_top_k,
                        temperature=rbf_temperature,
                        index=prototype_index,
                    )
                    transformed = contract_toward(
                        transform_input, target, strength=branch_value
                    )

            optimizer.zero_grad(set_to_none=True)
            normalized_loss = F.mse_loss(model.decode(z0), image)
            transformed_loss = F.mse_loss(model.decode(transformed), image)
            loss = (
                normalized_weight * normalized_loss
                + transformed_weight * transformed_loss
            ) / weight_total
            loss.backward()
            optimizer.step()

            count = image.shape[0]
            loss_sum += float(loss.detach()) * count
            normalized_sum += float(normalized_loss.detach()) * count
            transformed_sum += float(transformed_loss.detach()) * count
            branch_sum[branch_name] += float(transformed_loss.detach()) * count
            branch_count[branch_name] += count
            sample_count += count

        if sample_count == 0:
            raise ValueError("calibration batches produced no samples")
        total_history.append(loss_sum / sample_count)
        normalized_history.append(normalized_sum / sample_count)
        transformed_history.append(transformed_sum / sample_count)
        for name, _, _ in branches:
            if branch_count[name] == 0:
                raise ValueError(f"calibration branch {name!r} received no samples")
            branch_history[name].append(branch_sum[name] / branch_count[name])

    return NeutralDecoderCalibrationResult(
        total_loss=tuple(total_history),
        normalized_loss=tuple(normalized_history),
        transformed_loss=tuple(transformed_history),
        branch_loss={name: tuple(values) for name, values in branch_history.items()},
    )


def _flow_variants(
    latent: Tensor,
    bank: BasinState,
    index: ExactIndex,
    *,
    candidates: int,
    soft_top_k: int,
    times: tuple[float, ...],
    temperature: float,
) -> dict[str, Tensor]:
    candidate_ids = index.query(latent, candidates)
    variants = {"t0": latent}
    for prefix, top_k in (("hard", 1), ("soft", soft_top_k)):
        _, mu, raw_rate, bias, _ = rerank_topk(
            latent,
            candidate_ids,
            bank,
            top_k=top_k,
        )
        for flow_time in times:
            flowed, _, _ = analytic_topk_flow(
                latent,
                mu,
                raw_rate,
                bias,
                t=flow_time,
                temperature=temperature,
            )
            variants[f"{prefix}_t{flow_time:g}"] = flowed
    return variants


@torch.no_grad()
def evaluate_flow_ablation(
    model: CompactImageAutoencoder,
    normalizer: FrozenDiagonalWhitening,
    probe: nn.Module,
    bank: BasinState,
    batches: Iterable[tuple[Tensor, Tensor]],
    *,
    device: torch.device | str,
    times: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0),
    candidates: int = 64,
    soft_top_k: int = 4,
    temperature: float = 0.1,
    image_noise_std: float = 0.2,
    max_samples: int | None = 2000,
    seed: int = 17,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compare no-flow, hard-flow, and soft-flow on clean/noisy inputs."""
    if not times or any(value <= 0 for value in times):
        raise ValueError("times must contain positive values")
    if candidates <= 0 or soft_top_k <= 0:
        raise ValueError("candidates and soft_top_k must be positive")
    if image_noise_std < 0:
        raise ValueError("image_noise_std must be non-negative")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive when supplied")
    candidates = min(candidates, bank.mu.shape[0])
    if soft_top_k > candidates:
        raise ValueError("soft_top_k cannot exceed the candidate count")

    model.to(device).eval()
    normalizer.to(device).eval()
    probe.to(device).eval()
    index = ExactIndex(bank.mu)
    generator = torch.Generator(device=device).manual_seed(seed)
    totals: dict[str, dict[str, dict[str, float]]] = {}
    sample_count = pixel_count = 0

    for image, label in batches:
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
            variants = _flow_variants(
                z0,
                bank,
                index,
                candidates=candidates,
                soft_top_k=soft_top_k,
                times=times,
                temperature=temperature,
            )
            for variant, latent in variants.items():
                entry = totals.setdefault(variant, {}).setdefault(
                    condition,
                    {"squared_error": 0.0, "correct": 0.0, "displacement": 0.0},
                )
                reconstruction = model.decode(latent)
                entry["squared_error"] += float(
                    (reconstruction - image).square().sum().item()
                )
                entry["correct"] += float(
                    (probe(latent).argmax(dim=-1) == label).sum().item()
                )
                entry["displacement"] += float((latent - z0).norm(dim=-1).sum().item())

        sample_count += image.shape[0]
        pixel_count += image.numel()

    if sample_count == 0:
        raise ValueError("evaluation batches produced no samples")
    result = {}
    for variant, conditions in totals.items():
        result[variant] = {}
        for condition, value in conditions.items():
            result[variant][condition] = {
                "reconstruction_mse": value["squared_error"] / pixel_count,
                "classification_accuracy": value["correct"] / sample_count,
                "mean_latent_displacement": value["displacement"] / sample_count,
            }
    return result
