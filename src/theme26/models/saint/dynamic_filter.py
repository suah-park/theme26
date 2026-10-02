"""Dynamic filter generator for SAINT anisotropic interpolation."""

from __future__ import annotations

import torch
from torch import nn


class DynamicFilterGenerator(nn.Module):
    """Map FDM ``[B,R,K,K]`` to filters ``[B,R,C,K,K]``."""

    def __init__(self, *, feature_channels: int, kernel_size: int) -> None:
        super().__init__()
        if feature_channels < 1:
            raise ValueError("feature_channels must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.feature_channels = feature_channels
        self.kernel_size = kernel_size
        self.meta_network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, feature_channels, 3, padding=1),
        )

    def forward(self, fdm: torch.Tensor) -> torch.Tensor:
        """Generate one feature filter for every subpixel position."""

        if fdm.ndim != 4:
            raise ValueError("fdm must have shape [B,R,K,K]")
        batch_size, factor, height, width = fdm.shape
        if height != self.kernel_size or width != self.kernel_size:
            raise ValueError(
                f"fdm kernel must be {self.kernel_size}x{self.kernel_size}, "
                f"got {height}x{width}"
            )
        if not fdm.is_floating_point():
            raise TypeError("fdm must be floating point")
        if not torch.isfinite(fdm).all():
            raise ValueError("fdm contains NaN or Inf")

        generated = self.meta_network(fdm.reshape(batch_size * factor, 1, height, width))
        return generated.reshape(
            batch_size,
            factor,
            self.feature_channels,
            self.kernel_size,
            self.kernel_size,
        )
