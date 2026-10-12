"""Frozen 224px discriminative backbone adapter for large-image entry points.

The MVP encoder (`CompactImageAutoencoder`) is a small paired autoencoder
sized for 28x28/32x32 inputs. An ImageNet-shaped entry point instead needs a
frozen, standard 224px classification backbone with no paired decoder (design
v0.2 SS2: "A frozen discriminative encoder is suitable for retrieval/
classification but cannot be assumed to have a compatible decoder"), reduced
to a bounded normalized latent width with a fixed projection rather than a
trained one, matching SS2's "Fixed whitening or learned normalization frozen
with encoder" requirement.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_FEATURE_DIM = 512  # ResNet-18 pooled feature width


class Frozen224Backbone(nn.Module):
    """Frozen ResNet-18 feature extractor plus a fixed PCA projection.

    The backbone is always in eval mode with gradients disabled. Call
    `fit_projection` once on a calibration feature sample to freeze a PCA
    basis into buffers before `encode` is usable; there is no further
    training after that, matching `FrozenDiagonalWhitening`'s fit-once
    pattern.
    """

    def __init__(self, *, projected_dim: int, pretrained: bool = True) -> None:
        super().__init__()
        if projected_dim <= 0:
            raise ValueError("projected_dim must be positive")
        if projected_dim > _FEATURE_DIM:
            raise ValueError(
                f"projected_dim cannot exceed the backbone feature dim {_FEATURE_DIM}"
            )
        self.projected_dim = projected_dim
        self.feature_dim = _FEATURE_DIM
        self._backbone = _build_resnet18(pretrained=pretrained)
        for parameter in self._backbone.parameters():
            parameter.requires_grad_(False)
        self.eval()
        self.register_buffer(
            "projection", torch.zeros(self.feature_dim, projected_dim)
        )
        self.register_buffer("projection_mean", torch.zeros(self.feature_dim))
        self.register_buffer("is_fitted", torch.tensor(False))

    @torch.no_grad()
    def extract_features(self, image: Tensor) -> Tensor:
        """Run only the frozen backbone, returning raw pooled features."""
        self._validate_images(image)
        return self._backbone(image)

    @torch.no_grad()
    def fit_projection(self, features: Tensor) -> None:
        """Fit a fixed PCA projection once from a calibration feature sample."""
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape [batch, {self.feature_dim}]")
        if features.shape[0] <= self.projected_dim:
            raise ValueError("calibration sample must exceed projected_dim rows")
        mean = features.mean(dim=0)
        _, _, v = torch.linalg.svd(features - mean, full_matrices=False)
        components = v[: self.projected_dim].transpose(0, 1)
        target = self.projection_mean
        self.projection_mean.copy_(mean.to(device=target.device, dtype=target.dtype))
        self.projection.copy_(
            components.to(device=self.projection.device, dtype=self.projection.dtype)
        )
        self.is_fitted.fill_(True)

    @torch.no_grad()
    def encode(self, image: Tensor) -> Tensor:
        """Encode a batch of 224px images into the fixed projected dimension."""
        if not bool(self.is_fitted):
            raise RuntimeError("fit_projection must run before encode()")
        features = self.extract_features(image)
        return (features - self.projection_mean) @ self.projection

    def freeze(self) -> None:
        """Interface parity with `CompactImageAutoencoder.freeze()`; already frozen."""
        self.eval()

    def _validate_images(self, image: Tensor) -> None:
        expected = (3, 224, 224)
        if image.ndim != 4 or image.shape[1:] != expected:
            raise ValueError(f"image must have shape [batch, {expected}]")


def _build_resnet18(*, pretrained: bool) -> nn.Module:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "The frozen 224px backbone requires the vision extra: "
            "python -m pip install -e '.[vision]'"
        ) from error
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    return model


def imagenet_eval_transform() -> Any:
    """Standard resize/crop/normalize pipeline expected by the frozen backbone."""
    try:
        from torchvision import transforms
    except ImportError as error:  # pragma: no cover - depends on optional extra
        raise RuntimeError(
            "The frozen 224px backbone requires the vision extra: "
            "python -m pip install -e '.[vision]'"
        ) from error
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
