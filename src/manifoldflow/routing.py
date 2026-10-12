"""Two-stage candidate generation, energy reranking, and local flow."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from manifoldflow.bank import BasinState
from manifoldflow.flow import analytic_topk_flow, basin_energy
from manifoldflow.index import ExactIndex


@dataclass(frozen=True)
class FlowResult:
    """Observable output of one routed analytic segment."""

    latent: Tensor
    basin_ids: Tensor
    energy: Tensor
    weight: Tensor


def _take_candidate_rows(value: Tensor, indices: Tensor) -> Tensor:
    if value.ndim == 2:
        return value.gather(1, indices)
    if value.ndim == 3:
        return value.gather(1, indices[..., None].expand(-1, -1, value.shape[-1]))
    raise ValueError("candidate tensors must have rank 2 or 3")


def rerank_topk(
    z0: Tensor,
    candidate_ids: Tensor,
    bank: BasinState,
    *,
    top_k: int,
    rate_min: float = 1e-4,
    rate_max: float = 32.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Rerank retrieved candidates by their true anisotropic energy."""
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != z0.shape[0]:
        raise ValueError("candidate_ids must have shape [batch, candidates]")
    if not 1 <= top_k <= candidate_ids.shape[1]:
        raise ValueError("top_k must be in [1, candidate count]")

    records = bank.gather(candidate_ids, device=z0.device)
    candidate_energy = basin_energy(
        z0,
        records["mu"],
        records["raw_rate"],
        records["bias"],
        rate_min=rate_min,
        rate_max=rate_max,
    )
    keep = candidate_energy.topk(k=top_k, dim=-1, largest=False, sorted=True).indices
    return (
        _take_candidate_rows(records["ids"], keep),
        _take_candidate_rows(records["mu"], keep),
        _take_candidate_rows(records["raw_rate"], keep),
        _take_candidate_rows(records["bias"], keep),
        _take_candidate_rows(candidate_energy, keep),
    )


def exact_bank_flow(
    z0: Tensor,
    bank: BasinState,
    *,
    candidates: int = 64,
    top_k: int = 4,
    t: float | Tensor = 1.0,
    temperature: float = 0.1,
    rate_min: float = 1e-4,
    rate_max: float = 32.0,
) -> FlowResult:
    """Run the complete exact-routing reference path for one segment."""
    if candidates > bank.mu.shape[0]:
        raise ValueError("candidates cannot exceed bank size")
    index = ExactIndex(bank.mu)
    candidate_ids = index.query(z0.detach(), candidates)
    ids, mu, raw_rate, bias, _ = rerank_topk(
        z0,
        candidate_ids,
        bank,
        top_k=top_k,
        rate_min=rate_min,
        rate_max=rate_max,
    )
    latent, energy, weight = analytic_topk_flow(
        z0,
        mu,
        raw_rate,
        bias,
        t=t,
        temperature=temperature,
        rate_min=rate_min,
        rate_max=rate_max,
    )
    return FlowResult(latent=latent, basin_ids=ids, energy=energy, weight=weight)
