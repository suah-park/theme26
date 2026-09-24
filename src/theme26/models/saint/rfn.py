"""Residual Fusion Network contract."""

from __future__ import annotations

import torch
from torch import nn


class ResidualFusionNetwork(nn.Module):
    """Fuse sagittal and coronal dense volumes with residual correction."""

    def __init__(self, *, feature_channels: int = 64) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        raise NotImplementedError("model owner: implement RFN without hard-coded spatial sizes")

    def forward(self, sagittal: torch.Tensor, coronal: torch.Tensor) -> torch.Tensor:
        """Map two ``[B,1,X,Y,Z]`` volumes to fused ``[B,1,X,Y,Z]``."""

        raise NotImplementedError
