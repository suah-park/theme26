"""Residual Fusion Network (RFN) from SAINT."""

from __future__ import annotations

import torch
from torch import nn

from .ami import _ResidualDenseBlock


class ResidualFusionNetwork(nn.Module):
    """Fuse sagittal and coronal dense volumes with residual correction."""

    def __init__(
        self,
        *,
        feature_channels: int = 64,
        kernel_size: int = 3,
        num_blocks: int = 4,
        layers_per_block: int = 6,
        growth_rate: int = 12,
    ) -> None:
        super().__init__()
        if feature_channels < 1 or growth_rate < 1:
            raise ValueError("feature_channels and growth_rate must be positive")
        if num_blocks < 1 or layers_per_block < 1:
            raise ValueError("num_blocks and layers_per_block must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.feature_channels = feature_channels
        padding = kernel_size // 2
        self.shallow_1 = nn.Conv2d(2, feature_channels, kernel_size, padding=padding)
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
        self.output = nn.Conv2d(feature_channels, 1, kernel_size, padding=padding)

    def forward_slices(self, inputs: torch.Tensor) -> torch.Tensor:
        """Fuse axial slices shaped ``[N,2,X,Y]`` and return ``[N,1,X,Y]``."""

        if inputs.ndim != 4 or inputs.shape[1] != 2:
            raise ValueError("RFN slice input must have shape [N,2,X,Y]")
        shallow = self.shallow_1(inputs)
        features = self.shallow_2(shallow)
        block_outputs = []
        for block in self.blocks:
            features = block(features)
            block_outputs.append(features)
        residual = self.output(
            self.global_fusion(torch.cat(block_outputs, dim=1)) + shallow
        )
        return inputs.mean(dim=1, keepdim=True) + residual

    def forward(self, sagittal: torch.Tensor, coronal: torch.Tensor) -> torch.Tensor:
        """Map two ``[B,1,X,Y,Z]`` volumes to fused ``[B,1,X,Y,Z]``."""

        if sagittal.shape != coronal.shape:
            raise ValueError("sagittal and coronal volumes must have identical shapes")
        if sagittal.ndim != 5 or sagittal.shape[1] != 1:
            raise ValueError("directional volumes must have shape [B,1,X,Y,Z]")
        if not sagittal.is_floating_point() or not coronal.is_floating_point():
            raise TypeError("directional volumes must be floating point")

        batch_size, _, x_size, y_size, z_size = sagittal.shape
        axial_pairs = torch.cat((sagittal, coronal), dim=1)
        axial_pairs = axial_pairs.permute(0, 4, 1, 2, 3).reshape(
            batch_size * z_size,
            2,
            x_size,
            y_size,
        )
        fused = self.forward_slices(axial_pairs)
        return fused.reshape(batch_size, z_size, 1, x_size, y_size).permute(
            0, 2, 3, 4, 1
        )
