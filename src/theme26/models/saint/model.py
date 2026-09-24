"""End-to-end SAINT reconstruction contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ReconstructionOutput:
    """Raw learned outputs before observed-slice reinsertion."""

    prediction: torch.Tensor
    sagittal: torch.Tensor
    coronal: torch.Tensor


class SaintReconstructor(nn.Module):
    """Reconstruct a dense grayscale volume from compressed sparse-z input."""

    def __init__(self, *, factor: int = 4) -> None:
        super().__init__()
        self.factor = factor
        raise NotImplementedError("model owner: connect shared AMI, directional assembly and RFN")

    def forward(self, sparse_volume: torch.Tensor, fdm: torch.Tensor) -> ReconstructionOutput:
        """Map ``[B,1,X,Y,Z_lr]`` to three ``[B,1,X,Y,R*Z_lr]`` outputs."""

        raise NotImplementedError
