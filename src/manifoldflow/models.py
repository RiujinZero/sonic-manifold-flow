"""Compact paired image encoders and decoders for Phase A experiments."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CompactImageAutoencoder(nn.Module):
    """Small convolutional autoencoder with an explicit latent bottleneck.

    The same architecture supports 28x28 grayscale and 32x32 RGB inputs. The
    decoder is paired with the encoder and resizes its final 32x32 activation to
    the requested input size when necessary.
    """

    def __init__(
        self,
        *,
        input_channels: int,
        input_size: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if input_size < 16:
            raise ValueError("input_size must be at least 16")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")

        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.encoder_projection = nn.Linear(128 * 4 * 4, latent_dim)

        self.decoder_projection = nn.Linear(latent_dim, 128 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.ConvTranspose2d(
                32,
                input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

    def encode(self, image: Tensor) -> Tensor:
        """Map a batch of images to unnormalized latent vectors."""
        self._validate_images(image)
        features = self.encoder_conv(image)
        return self.encoder_projection(features.flatten(start_dim=1))

    def decode(self, latent: Tensor) -> Tensor:
        """Decode compatible latents back into the input image domain."""
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError("latent must have shape [batch, latent_dim]")
        features = self.decoder_projection(latent).reshape(-1, 128, 4, 4)
        reconstruction = self.decoder_conv(features)
        if reconstruction.shape[-2:] != (self.input_size, self.input_size):
            reconstruction = F.interpolate(
                reconstruction,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        return torch.sigmoid(reconstruction)

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(image)
        return self.decode(latent), latent

    def freeze(self) -> None:
        """Freeze the paired backbone before the primary online-learning run."""
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def freeze_encoder(self) -> None:
        """Freeze only the representation path during decoder calibration."""
        self.encoder_conv.eval()
        self.encoder_projection.eval()
        for module in (self.encoder_conv, self.encoder_projection):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train_decoder(self) -> tuple[nn.Parameter, ...]:
        """Enable decoder training and return its optimizer parameter set."""
        self.decoder_projection.train()
        self.decoder_conv.train()
        parameters = tuple(self.decoder_projection.parameters()) + tuple(
            self.decoder_conv.parameters()
        )
        for parameter in parameters:
            parameter.requires_grad_(True)
        return parameters

    def _validate_images(self, image: Tensor) -> None:
        expected = (self.input_channels, self.input_size, self.input_size)
        if image.ndim != 4 or image.shape[1:] != expected:
            raise ValueError(f"image must have shape [batch, {expected}]")


def trainable_parameter_count(model: nn.Module) -> int:
    """Return the number of parameters that currently receive gradients."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
