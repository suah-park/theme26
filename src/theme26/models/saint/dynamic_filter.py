"""Dynamic filter generator contract."""

from __future__ import annotations

import torch
from torch import nn


class DynamicFilterGenerator(nn.Module):
    """Map FDM ``[B,R,K,K]`` to filters ``[B,R,C,K,K]``."""

    def __init__(self, *, feature_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.kernel_size = kernel_size
        raise NotImplementedError("model owner: implement dynamic filter generator layers")

    def forward(self, fdm: torch.Tensor) -> torch.Tensor:
        """Generate one feature filter for every subpixel position."""

        raise NotImplementedError
