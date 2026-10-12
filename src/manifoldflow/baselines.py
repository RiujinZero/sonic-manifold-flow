"""Matched representation baselines for the Phase A evaluation gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def classification_accuracy(prediction: Tensor, target: Tensor) -> float:
    """Return scalar classification accuracy after strict shape validation."""
    if prediction.shape != target.shape or prediction.ndim != 1:
        raise ValueError("prediction and target must have shape [samples]")
    if target.numel() == 0:
        raise ValueError("cannot evaluate an empty target")
    return float((prediction == target).float().mean().item())


def top_k_accuracy(scores: Tensor, target: Tensor, *, k: int) -> float:
    """Return top-k accuracy: does the target rank among the k highest scores."""
    if scores.ndim != 2 or target.shape != (scores.shape[0],):
        raise ValueError("scores must be [samples, classes] matching target")
    if target.numel() == 0:
        raise ValueError("cannot evaluate an empty target")
    if not 1 <= k <= scores.shape[1]:
        raise ValueError("k must be in [1, classes]")
    top = scores.topk(k, dim=-1).indices
    return float((top == target[:, None]).any(dim=-1).float().mean().item())


@dataclass(frozen=True)
class CentroidClassifier:
    """Nearest-class-centroid state fitted from normalized latents."""

    centers: Tensor
    classes: Tensor

    @torch.no_grad()
    def predict(self, latent: Tensor, *, chunk_size: int = 4096) -> Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.centers.shape[1]:
            raise ValueError("latent width must match centroid width")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        predictions = []
        for start in range(0, latent.shape[0], chunk_size):
            query = latent[start : start + chunk_size]
            distance = torch.cdist(query, self.centers.to(query.device))
            predictions.append(self.classes.to(query.device)[distance.argmin(dim=1)])
        return torch.cat(predictions).to(latent.device)


def fit_nearest_centroid(latent: Tensor, label: Tensor) -> CentroidClassifier:
    """Fit one exact mean vector per observed class."""
    _validate_reference(latent, label)
    classes = label.unique(sorted=True)
    centers = torch.stack([latent[label == value].mean(dim=0) for value in classes])
    return CentroidClassifier(centers=centers, classes=classes)


@torch.no_grad()
def knn_class_scores(
    reference: Tensor,
    reference_label: Tensor,
    query: Tensor,
    *,
    k: int,
    classes: int,
    query_chunk_size: int = 512,
    reference_chunk_size: int = 8192,
) -> Tensor:
    """Return per-class vote counts among each query's k nearest neighbors.

    A `[samples, classes]` score matrix, not just the majority label, so
    top-k accuracy (not only top-1) can be computed from exact k-NN.
    """
    _validate_reference(reference, reference_label)
    if query.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("query must have shape [samples, reference width]")
    if not 1 <= k <= reference.shape[0]:
        raise ValueError("k must be in [1, reference sample count]")
    if classes <= 0:
        raise ValueError("classes must be positive")
    if query_chunk_size <= 0 or reference_chunk_size <= 0:
        raise ValueError("chunk sizes must be positive")

    result = []
    for query_start in range(0, query.shape[0], query_chunk_size):
        local_query = query[query_start : query_start + query_chunk_size]
        rows = local_query.shape[0]
        best_distance = torch.full(
            (rows, k),
            torch.inf,
            dtype=local_query.dtype,
            device=local_query.device,
        )
        best_label = torch.zeros(
            (rows, k), dtype=reference_label.dtype, device=local_query.device
        )

        for reference_start in range(0, reference.shape[0], reference_chunk_size):
            local_reference = reference[
                reference_start : reference_start + reference_chunk_size
            ].to(local_query.device)
            local_label = reference_label[
                reference_start : reference_start + reference_chunk_size
            ].to(local_query.device)
            distance = torch.cdist(local_query, local_reference)
            expanded_label = local_label[None, :].expand(rows, -1)
            candidate_distance = torch.cat((best_distance, distance), dim=1)
            candidate_label = torch.cat((best_label, expanded_label), dim=1)
            best_distance, keep = candidate_distance.topk(
                k=k, dim=1, largest=False, sorted=False
            )
            best_label = candidate_label.gather(1, keep)

        scores = torch.zeros(
            (rows, classes), dtype=torch.float32, device=local_query.device
        )
        scores.scatter_add_(
            1, best_label, torch.ones_like(best_label, dtype=torch.float32)
        )
        result.append(scores)
    return torch.cat(result).to(query.device)


@torch.no_grad()
def exact_knn_predict(
    reference: Tensor,
    reference_label: Tensor,
    query: Tensor,
    *,
    k: int = 1,
    query_chunk_size: int = 512,
    reference_chunk_size: int = 8192,
) -> Tensor:
    """Compute exact Euclidean k-NN majority vote with bounded working memory."""
    classes = int(reference_label.max().item()) + 1
    scores = knn_class_scores(
        reference,
        reference_label,
        query,
        k=k,
        classes=classes,
        query_chunk_size=query_chunk_size,
        reference_chunk_size=reference_chunk_size,
    )
    return scores.argmax(dim=-1).to(dtype=reference_label.dtype)


@torch.no_grad()
def neighborhood_preservation(
    reference_space: Tensor,
    latent_space: Tensor,
    *,
    k: int = 10,
    query_chunk_size: int = 128,
) -> float:
    """Measure mean top-k neighbor overlap between input and latent spaces.

    The metric is intentionally direct and bounded to a configured sample. Each
    sample's self-match is excluded before comparing exact neighbor identities.
    """
    if reference_space.ndim != 2 or latent_space.ndim != 2:
        raise ValueError("reference_space and latent_space must be matrices")
    if reference_space.shape[0] != latent_space.shape[0]:
        raise ValueError("spaces must contain the same samples in the same order")
    if not 1 <= k < reference_space.shape[0]:
        raise ValueError("k must be positive and smaller than the sample count")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    overlaps = []
    for start in range(0, reference_space.shape[0], query_chunk_size):
        stop = min(start + query_chunk_size, reference_space.shape[0])
        input_distance = torch.cdist(reference_space[start:stop], reference_space)
        latent_distance = torch.cdist(latent_space[start:stop], latent_space)
        local_rows = torch.arange(stop - start, device=input_distance.device)
        global_rows = torch.arange(start, stop, device=input_distance.device)
        input_distance[local_rows, global_rows] = torch.inf
        latent_distance[local_rows, global_rows] = torch.inf
        input_neighbor = input_distance.topk(k, largest=False).indices
        latent_neighbor = latent_distance.topk(k, largest=False).indices
        overlap = (
            (input_neighbor[:, :, None] == latent_neighbor[:, None, :])
            .any(dim=-1)
            .float()
            .mean(dim=-1)
        )
        overlaps.append(overlap)
    return float(torch.cat(overlaps).mean().item())


@dataclass(frozen=True)
class LinearProbeResult:
    """Frozen-backbone linear classifier and its epoch loss history."""

    model: nn.Linear
    loss: tuple[float, ...]


def fit_linear_probe(
    latent: Tensor,
    label: Tensor,
    *,
    classes: int,
    epochs: int = 50,
    batch_size: int = 512,
    learning_rate: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 17,
    device: torch.device | str = "cpu",
) -> LinearProbeResult:
    """Fit a small linear probe without changing the representation."""
    _validate_reference(latent, label)
    if classes <= int(label.max().item()):
        raise ValueError("classes must exceed every label value")
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")

    generator = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = nn.Linear(latent.shape[1], classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    history = []
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(latent.shape[0], generator=generator)
        loss_sum = 0.0
        for start in range(0, latent.shape[0], batch_size):
            rows = permutation[start : start + batch_size]
            feature = latent[rows].to(device)
            target = label[rows].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(feature), target)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * rows.numel()
        history.append(loss_sum / latent.shape[0])
    model.eval()
    return LinearProbeResult(model=model, loss=tuple(history))


@torch.no_grad()
def linear_probe_predict(model: nn.Linear, latent: Tensor) -> Tensor:
    """Predict labels while respecting the probe's current device."""
    device = model.weight.device
    return model(latent.to(device)).argmax(dim=1).to(latent.device)


def _validate_reference(latent: Tensor, label: Tensor) -> None:
    if latent.ndim != 2 or latent.shape[0] == 0:
        raise ValueError("latent must have shape [samples, d] with samples > 0")
    if label.shape != (latent.shape[0],):
        raise ValueError("label must have shape [samples]")
    if label.dtype != torch.long:
        raise ValueError("label must use torch.long dtype")
