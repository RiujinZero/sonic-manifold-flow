"""Frozen latent normalization used to stabilize retrieval geometry."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class FrozenDiagonalWhitening(nn.Module):
    """Fit diagonal whitening once, then expose immutable transform buffers."""

    def __init__(
        self,
        dimension: int,
        *,
        epsilon: float = 1e-5,
        l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.dimension = dimension
        self.epsilon = epsilon
        self.l2_normalize = l2_normalize
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("inverse_std", torch.ones(dimension))
        self.register_buffer("is_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, latent: Tensor) -> None:
        """Fit population mean and diagonal variance from training latents."""
        self._validate(latent)
        if latent.shape[0] < 2:
            raise ValueError("at least two latents are required to fit whitening")
        mean = latent.mean(dim=0)
        variance = latent.var(dim=0, unbiased=False)
        self.mean.copy_(mean.to(device=self.mean.device, dtype=self.mean.dtype))
        self.inverse_std.copy_(
            torch.rsqrt(variance + self.epsilon).to(
                device=self.inverse_std.device,
                dtype=self.inverse_std.dtype,
            )
        )
        self.is_fitted.fill_(True)

    def forward(self, latent: Tensor) -> Tensor:
        self._validate(latent)
        if not bool(self.is_fitted):
            raise RuntimeError("whitening must be fitted before use")
        normalized = (latent - self.mean) * self.inverse_std
        if self.l2_normalize:
            normalized = F.normalize(normalized, p=2, dim=-1)
        return normalized

    def _validate(self, latent: Tensor) -> None:
        if latent.ndim != 2 or latent.shape[1] != self.dimension:
            raise ValueError("latent must have shape [batch, dimension]")
