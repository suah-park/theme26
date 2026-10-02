"""Anisotropic Meta Interpolation (AMI) from SAINT."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .dynamic_filter import DynamicFilterGenerator


class _ResidualDenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            growth_rate,
            kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.conv(inputs), inplace=True)
        return torch.cat((inputs, features), dim=1)


class _ResidualDenseBlock(nn.Module):
    def __init__(
        self,
        feature_channels: int,
        growth_rate: int,
        num_layers: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        layers = []
        for index in range(num_layers):
            layers.append(
                _ResidualDenseLayer(
                    feature_channels + index * growth_rate,
                    growth_rate,
                    kernel_size,
                )
            )
        self.layers = nn.Sequential(*layers)
        self.local_fusion = nn.Conv2d(
            feature_channels + num_layers * growth_rate,
            feature_channels,
            1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.local_fusion(self.layers(inputs)) + inputs


class AnisotropicMetaInterpolator(nn.Module):
    """Upsample only the sparse z dimension of a 3-channel marginal slice."""

    def __init__(
        self,
        *,
        factor: int = 4,
        feature_channels: int = 64,
        kernel_size: int = 3,
        num_blocks: int = 6,
        layers_per_block: int = 8,
        growth_rate: int = 32,
    ) -> None:
        super().__init__()
        if factor < 2:
            raise ValueError("factor must be at least 2")
        if feature_channels < 1 or growth_rate < 1:
            raise ValueError("feature_channels and growth_rate must be positive")
        if num_blocks < 1 or layers_per_block < 1:
            raise ValueError("num_blocks and layers_per_block must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.factor = factor
        self.feature_channels = feature_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        self.shallow_1 = nn.Conv2d(3, feature_channels, kernel_size, padding=padding)
        self.shallow_2 = nn.Conv2d(
            feature_channels,
            feature_channels,
            kernel_size,
            padding=padding,
        )
        self.blocks = nn.ModuleList(
            [
                _ResidualDenseBlock(
                    feature_channels,
                    growth_rate,
                    layers_per_block,
                    kernel_size,
                )
                for _ in range(num_blocks)
            ]
        )
        self.global_fusion = nn.Sequential(
            nn.Conv2d(num_blocks * feature_channels, feature_channels, 1),
            nn.Conv2d(feature_channels, feature_channels, kernel_size, padding=padding),
        )
        self.filter_generator = DynamicFilterGenerator(
            feature_channels=feature_channels,
            kernel_size=kernel_size,
        )

    def _validate_inputs(self, marginal: torch.Tensor, fdm: torch.Tensor) -> None:
        if marginal.ndim != 4 or marginal.shape[1] != 3:
            raise ValueError("marginal must have shape [B,3,L,Z_lr]")
        if not marginal.is_floating_point():
            raise TypeError("marginal must be floating point")
        if not torch.isfinite(marginal).all():
            raise ValueError("marginal contains NaN or Inf")
        if fdm.ndim != 4 or fdm.shape[1:] != (
            self.factor,
            self.kernel_size,
            self.kernel_size,
        ):
            raise ValueError(
                "fdm must have shape [1|B,factor,kernel_size,kernel_size]"
            )
        if fdm.shape[0] not in (1, marginal.shape[0]):
            raise ValueError("fdm batch must be 1 or match the marginal batch")

    def forward_subpixels(self, marginal: torch.Tensor, fdm: torch.Tensor) -> torch.Tensor:
        """Return pre-shuffle subpixel channels shaped ``[B,R,L,Z_lr]``."""

        self._validate_inputs(marginal, fdm)
        shallow = self.shallow_1(marginal)
        features = self.shallow_2(shallow)
        block_outputs = []
        for block in self.blocks:
            features = block(features)
            block_outputs.append(features)
        features = self.global_fusion(torch.cat(block_outputs, dim=1)) + shallow

        if fdm.shape[0] == 1 and marginal.shape[0] != 1:
            fdm = fdm.expand(marginal.shape[0], -1, -1, -1)
        # The released SAINT code predicts every subpixel phase, including
        # phase zero.  Exact observed-slice preservation is imposed later by
        # compose_full_volume rather than hidden inside this learned module.
        filters = self.filter_generator(fdm).flatten(2)
        patches = F.unfold(
            features,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
        )
        subpixels = torch.bmm(filters, patches)
        return subpixels.reshape(
            marginal.shape[0],
            self.factor,
            marginal.shape[2],
            marginal.shape[3],
        )

    def forward(self, marginal: torch.Tensor, fdm: torch.Tensor) -> torch.Tensor:
        """Map ``[B,3,L,Z_lr]`` and ``[1|B,R,K,K]`` to ``[B,1,L,R*Z_lr]``."""

        subpixels = self.forward_subpixels(marginal, fdm)
        batch_size, factor, length, z_lr = subpixels.shape
        dense = subpixels.permute(0, 2, 3, 1).reshape(
            batch_size,
            length,
            z_lr * factor,
        )
        return dense.unsqueeze(1)
