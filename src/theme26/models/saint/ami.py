"""Anisotropic Meta Interpolation contract."""

from __future__ import annotations

import torch
from torch import nn


class AnisotropicMetaInterpolator(nn.Module):
    """Upsample only the sparse z dimension of a 3-channel marginal slice."""

    def __init__(self, *, factor: int = 4, feature_channels: int = 64, kernel_size: int = 3) -> None:
        super().__init__()
        self.factor = factor
        self.feature_channels = feature_channels
        self.kernel_size = kernel_size
        raise NotImplementedError("model owner: implement SAINT feature learner and anisotropic rearrangement")

    def forward(self, marginal: torch.Tensor, fdm: torch.Tensor) -> torch.Tensor:
        """Map ``[B,3,L,Z_lr]`` and ``[1|B,R,K,K]`` to ``[B,1,L,R*Z_lr]``."""

        raise NotImplementedError
