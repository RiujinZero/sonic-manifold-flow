"""Simple online controls for continual sparse-memory experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from manifoldflow.index import ExactMainDeltaIndex


@dataclass
class OnlinePrototypeMemory:
    """Online nearest-prototype means with nearest and RBF value readouts."""

    centers: Tensor
    count: Tensor
    class_count: Tensor
    index: ExactMainDeltaIndex = field(init=False)

    def __post_init__(self) -> None:
        if self.centers.ndim != 2 or self.centers.shape[0] == 0:
            raise ValueError("centers must have shape [prototypes, d]")
        rows = self.centers.shape[0]
        if self.count.shape != (rows,) or torch.any(self.count <= 0):
            raise ValueError("count must be positive with shape [prototypes]")
        if (
            self.class_count.ndim != 2
            or self.class_count.shape[0] != rows
            or torch.any(self.class_count < 0)
            or torch.any(self.class_count.sum(dim=1) <= 0)
        ):
            raise ValueError("class_count must be non-empty with one row per prototype")
        self.centers = self.centers.detach().clone()
        self.count = self.count.detach().clone()
        self.class_count = self.class_count.detach().clone()
        self.index = ExactMainDeltaIndex(self.centers)

    @property
    def dimension(self) -> int:
        return self.centers.shape[1]

    @torch.no_grad()
    def predict_nearest(self, latent: Tensor) -> Tensor:
        ids = self.index.query(latent, 1).squeeze(1)
        return self.class_count.to(latent.device)[ids].argmax(dim=-1)

    @torch.no_grad()
    def predict_rbf(
        self,
        latent: Tensor,
        *,
        top_k: int = 4,
        temperature: float = 0.1,
    ) -> Tensor:
        if not 1 <= top_k <= self.index.size or temperature <= 0:
            raise ValueError("top_k and temperature are outside valid bounds")
        ids = self.index.query(latent, top_k)
        local = self.centers.to(device=latent.device, dtype=latent.dtype)[ids]
        squared_distance = (latent[:, None, :] - local).square().sum(dim=-1)
        weight = torch.softmax(-0.5 * squared_distance / temperature, dim=-1)
        count = self.class_count.to(latent.device)[ids].float()
        probability = count / count.sum(dim=-1, keepdim=True).clamp_min(1)
        return (weight[..., None] * probability).sum(dim=1).argmax(dim=-1)

    @torch.no_grad()
    def update(self, observations: Tensor, labels: Tensor) -> Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.dimension:
            raise ValueError("observations must have shape [batch, dimension]")
        if labels.shape != (observations.shape[0],) or labels.dtype != torch.long:
            raise ValueError("labels must be long with shape [batch]")
        if torch.any(labels < 0) or torch.any(labels >= self.class_count.shape[1]):
            raise ValueError("labels are outside the prototype class width")
        winners = []
        for observation, label in zip(observations, labels, strict=True):
            winner = int(self.index.query(observation[None, :], 1).item())
            value = observation.to(device=self.centers.device, dtype=self.centers.dtype)
            new_count = self.count[winner] + 1
            self.centers[winner] += (value - self.centers[winner]) / new_count.to(
                self.centers.dtype
            )
            self.count[winner] = new_count
            self.class_count[winner, int(label.item())] += 1
            self.index.upsert(
                torch.tensor([winner], device=self.centers.device),
                self.centers[winner][None, :],
            )
            winners.append(winner)
        return torch.tensor(winners, dtype=torch.long, device=observations.device)


@dataclass
class OnlineClassCentroid:
    """Supervised streaming mean for each class."""

    centers: Tensor
    count: Tensor
    touched: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.centers.ndim != 2 or self.centers.shape[0] < 2:
            raise ValueError("centers must have shape [classes, d] with classes >= 2")
        if self.count.shape != (self.centers.shape[0],) or torch.any(self.count <= 0):
            raise ValueError("count must be positive with shape [classes]")

    @classmethod
    @torch.no_grad()
    def from_data(
        cls,
        latent: Tensor,
        label: Tensor,
        *,
        classes: int,
    ) -> OnlineClassCentroid:
        if latent.ndim != 2 or latent.shape[0] == 0:
            raise ValueError("latent must have shape [samples, d]")
        if label.shape != (latent.shape[0],) or label.dtype != torch.long:
            raise ValueError("label must be long with shape [samples]")
        if classes < 2 or torch.any(label < 0) or torch.any(label >= classes):
            raise ValueError("classes and labels are incompatible")
        count = torch.bincount(label, minlength=classes)
        if torch.any(count == 0):
            raise ValueError("every class must have at least one observation")
        total = torch.zeros(
            classes,
            latent.shape[1],
            dtype=latent.dtype,
            device=latent.device,
        )
        total.index_add_(0, label, latent)
        centers = total / count.to(latent.dtype)[:, None]
        return cls(centers=centers, count=count)

    @property
    def dimension(self) -> int:
        return self.centers.shape[1]

    @property
    def classes(self) -> int:
        return self.centers.shape[0]

    @torch.no_grad()
    def predict(self, latent: Tensor) -> Tensor:
        if latent.ndim != 2 or latent.shape[1] != self.dimension:
            raise ValueError("latent must have shape [samples, dimension]")
        centers = self.centers.to(device=latent.device, dtype=latent.dtype)
        return torch.cdist(latent, centers).argmin(dim=-1)

    @torch.no_grad()
    def update(self, observations: Tensor, labels: Tensor) -> None:
        if observations.ndim != 2 or observations.shape[1] != self.dimension:
            raise ValueError("observations must have shape [batch, dimension]")
        if labels.shape != (observations.shape[0],) or labels.dtype != torch.long:
            raise ValueError("labels must be long with shape [batch]")
        if torch.any(labels < 0) or torch.any(labels >= self.classes):
            raise ValueError("labels must be in [0, classes)")
        for observation, label in zip(observations, labels, strict=True):
            class_id = int(label.item())
            value = observation.to(device=self.centers.device, dtype=self.centers.dtype)
            new_count = self.count[class_id] + 1
            self.centers[class_id] += (value - self.centers[class_id]) / new_count.to(
                self.centers.dtype
            )
            self.count[class_id] = new_count
            self.touched.add(class_id)
