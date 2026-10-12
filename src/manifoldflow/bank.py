"""CPU-oriented basin records and local streaming statistical updates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor


def _inverse_softplus(value: Tensor) -> Tensor:
    """Numerically stable inverse of softplus for strictly positive values."""
    return value + torch.log(-torch.expm1(-value))


@dataclass
class BasinState:
    """Dense MVP storage for a versioned collection of basin records.

    Production shards may use quantized or memory-mapped storage. This class
    keeps lifecycle and math explicit while the bank is still small enough for
    exact-routing experiments.
    """

    mu: Tensor
    raw_rate: Tensor
    bias: Tensor
    count: Tensor
    m2: Tensor
    usage: Tensor
    version: int = 0

    def __post_init__(self) -> None:
        if self.mu.ndim != 2 or self.mu.shape[0] == 0:
            raise ValueError("mu must have shape [basins, d] with basins > 0")
        if self.raw_rate.shape != self.mu.shape or self.m2.shape != self.mu.shape:
            raise ValueError("raw_rate and m2 must match mu")
        rows = self.mu.shape[0]
        for name, value in (
            ("bias", self.bias),
            ("count", self.count),
            ("usage", self.usage),
        ):
            if value.shape != (rows,):
                raise ValueError(f"{name} must have shape [basins]")
        if torch.any(self.count < 0):
            raise ValueError("count must be non-negative")
        if torch.any(self.m2 < 0):
            raise ValueError("m2 must be non-negative")

    @classmethod
    def from_centers(
        cls,
        centers: Tensor,
        *,
        initial_rate: float = 1.0,
        rate_min: float = 1e-4,
    ) -> BasinState:
        """Initialize empty sufficient statistics around supplied centers."""
        if initial_rate <= rate_min:
            raise ValueError("initial_rate must be greater than rate_min")
        centers = centers.detach().clone()
        raw_value = _inverse_softplus(
            torch.tensor(
                initial_rate - rate_min,
                dtype=centers.dtype,
                device=centers.device,
            )
        )
        return cls(
            mu=centers,
            raw_rate=raw_value.expand_as(centers).clone(),
            bias=torch.zeros(
                centers.shape[0], dtype=centers.dtype, device=centers.device
            ),
            count=torch.zeros(
                centers.shape[0], dtype=torch.long, device=centers.device
            ),
            m2=torch.zeros_like(centers),
            usage=torch.zeros(
                centers.shape[0], dtype=torch.long, device=centers.device
            ),
        )

    @classmethod
    def from_statistics(
        cls,
        centers: Tensor,
        count: Tensor,
        m2: Tensor,
        *,
        bias: Tensor | None = None,
        variance_scale: float = 1.0,
        epsilon: float = 1e-4,
        rate_min: float = 1e-4,
        rate_max: float = 32.0,
        singleton_rate: float = 1.0,
    ) -> BasinState:
        """Build populated basin records from exact sufficient statistics."""
        if centers.ndim != 2 or centers.shape[0] == 0:
            raise ValueError("centers must have shape [basins, d] with basins > 0")
        if count.shape != (centers.shape[0],) or torch.any(count <= 0):
            raise ValueError("count must be positive with shape [basins]")
        if m2.shape != centers.shape or torch.any(m2 < 0):
            raise ValueError("m2 must be non-negative and match centers")
        if variance_scale <= 0 or epsilon <= 0:
            raise ValueError("variance_scale and epsilon must be positive")
        if not rate_min < singleton_rate <= rate_max:
            raise ValueError("singleton_rate must be in (rate_min, rate_max]")

        centers = centers.detach().clone()
        count = count.to(device=centers.device, dtype=torch.long).detach().clone()
        m2 = m2.to(device=centers.device, dtype=centers.dtype).detach().clone()
        denominator = (count - 1).clamp_min(1).to(centers.dtype)[:, None]
        variance = m2 / denominator
        estimated_rate = (variance_scale / (variance + epsilon)).clamp(
            min=rate_min, max=rate_max
        )
        default_rate = torch.full_like(estimated_rate, singleton_rate)
        rate = torch.where(count[:, None] > 1, estimated_rate, default_rate)
        raw_rate = _inverse_softplus(
            (rate - rate_min).clamp_min(torch.finfo(rate.dtype).tiny)
        )

        if bias is None:
            bias = torch.zeros(
                centers.shape[0], dtype=centers.dtype, device=centers.device
            )
        elif bias.shape != (centers.shape[0],):
            raise ValueError("bias must have shape [basins]")
        else:
            bias = bias.to(device=centers.device, dtype=centers.dtype).detach().clone()

        return cls(
            mu=centers,
            raw_rate=raw_rate,
            bias=bias,
            count=count,
            m2=m2,
            usage=count.clone(),
        )

    @classmethod
    def from_serialized(cls, payload: Mapping[str, object]) -> BasinState:
        """Restore the tensor fields emitted by initialized-bank serialization."""
        tensor_names = ("mu", "raw_rate", "bias", "count", "m2", "usage")
        tensors = {}
        for name in tensor_names:
            value = payload.get(name)
            if not isinstance(value, Tensor):
                raise ValueError(f"serialized bank field {name!r} must be a tensor")
            tensors[name] = value
        version = payload.get("version", 0)
        if not isinstance(version, int):
            raise ValueError("serialized bank version must be an integer")
        return cls(**tensors, version=version)

    def gather(
        self,
        ids: Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> dict[str, Tensor]:
        """Gather candidate records and optionally transfer only those rows."""
        target = device if device is not None else self.mu.device
        bank_ids = ids.to(self.mu.device)
        return {
            "mu": self.mu[bank_ids].to(target),
            "raw_rate": self.raw_rate[bank_ids].to(target),
            "bias": self.bias[bank_ids].to(target),
            "ids": ids.to(target),
        }

    @torch.no_grad()
    def append_statistics(
        self,
        center: Tensor,
        count: int,
        m2: Tensor,
        *,
        bias: float = 0.0,
        variance_scale: float = 1.0,
        epsilon: float = 1e-4,
        rate_min: float = 1e-4,
        rate_max: float = 32.0,
        singleton_rate: float = 1.0,
    ) -> int:
        """Append one basin initialized from staged sufficient statistics."""
        if center.shape != (self.mu.shape[1],) or m2.shape != center.shape:
            raise ValueError("center and m2 must have shape [bank.dimension]")
        if count <= 0:
            raise ValueError("count must be positive")
        staged = BasinState.from_statistics(
            center.to(device=self.mu.device, dtype=self.mu.dtype)[None, :],
            torch.tensor([count], device=self.count.device, dtype=torch.long),
            m2.to(device=self.m2.device, dtype=self.m2.dtype)[None, :],
            bias=torch.tensor([bias], device=self.bias.device, dtype=self.bias.dtype),
            variance_scale=variance_scale,
            epsilon=epsilon,
            rate_min=rate_min,
            rate_max=rate_max,
            singleton_rate=singleton_rate,
        )
        new_id = self.mu.shape[0]
        for field in ("mu", "raw_rate", "bias", "count", "m2", "usage"):
            setattr(
                self, field, torch.cat((getattr(self, field), getattr(staged, field)))
            )
        return new_id

    @torch.no_grad()
    def update_winners(
        self,
        observations: Tensor,
        winner_ids: Tensor,
        *,
        variance_scale: float = 1.0,
        epsilon: float = 1e-4,
        rate_min: float = 1e-4,
        rate_max: float = 32.0,
    ) -> None:
        """Update only assigned rows with sequential Welford statistics.

        Repeated IDs within a batch are processed in stream order so the result
        matches online updates exactly.
        """
        if observations.ndim != 2 or observations.shape[1] != self.mu.shape[1]:
            raise ValueError("observations must have shape [batch, d]")
        if winner_ids.shape != (observations.shape[0],):
            raise ValueError("winner_ids must have shape [batch]")
        if variance_scale <= 0 or epsilon <= 0:
            raise ValueError("variance_scale and epsilon must be positive")

        for observation, winner in zip(observations, winner_ids, strict=True):
            index = int(winner.item())
            if not 0 <= index < self.mu.shape[0]:
                raise IndexError(f"winner basin {index} is outside the bank")

            value = observation.to(device=self.mu.device, dtype=self.mu.dtype)
            new_count = self.count[index] + 1
            delta = value - self.mu[index]
            new_mu = self.mu[index] + delta / new_count.to(self.mu.dtype)
            self.m2[index] += delta * (value - new_mu)
            self.mu[index] = new_mu
            self.count[index] = new_count
            self.usage[index] += 1

            if new_count > 1:
                variance = self.m2[index] / (new_count - 1).to(self.mu.dtype)
                rate = (variance_scale / (variance + epsilon)).clamp(
                    min=rate_min, max=rate_max
                )
                positive_part = (rate - rate_min).clamp_min(
                    torch.finfo(rate.dtype).tiny
                )
                self.raw_rate[index] = _inverse_softplus(positive_part)
