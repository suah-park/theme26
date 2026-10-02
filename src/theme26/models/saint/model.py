"""End-to-end grayscale SAINT reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .ami import AnisotropicMetaInterpolator
from .rfn import ResidualFusionNetwork


@dataclass(frozen=True)
class ReconstructionOutput:
    """Raw learned outputs before observed-slice reinsertion."""

    prediction: torch.Tensor
    sagittal: torch.Tensor
    coronal: torch.Tensor


class SaintReconstructor(nn.Module):
    """Reconstruct a dense grayscale volume from compressed sparse-z input."""

    def __init__(
        self,
        *,
        factor: int = 4,
        feature_channels: int = 64,
        ami_num_blocks: int = 6,
        ami_layers_per_block: int = 8,
        ami_growth_rate: int = 32,
        rfn_num_blocks: int = 4,
        rfn_layers_per_block: int = 6,
        rfn_growth_rate: int = 12,
        marginal_batch_size: int | None = None,
    ) -> None:
        super().__init__()
        if marginal_batch_size is not None and marginal_batch_size < 1:
            raise ValueError("marginal_batch_size must be positive or None")
        self.factor = factor
        self.marginal_batch_size = marginal_batch_size
        self.ami = AnisotropicMetaInterpolator(
            factor=factor,
            feature_channels=feature_channels,
            num_blocks=ami_num_blocks,
            layers_per_block=ami_layers_per_block,
            growth_rate=ami_growth_rate,
        )
        self.rfn = ResidualFusionNetwork(
            feature_channels=feature_channels,
            num_blocks=rfn_num_blocks,
            layers_per_block=rfn_layers_per_block,
            growth_rate=rfn_growth_rate,
        )

    @staticmethod
    def _neighbor_triplets(volume: torch.Tensor, axis: int) -> torch.Tensor:
        """Return neighbouring planes with replicated boundaries.

        ``volume`` is ``[B,X,Y,Z]``.  Axis 1 produces ``[B*X,3,Y,Z]`` and
        axis 2 produces ``[B*Y,3,X,Z]``.
        """

        size = volume.shape[axis]
        indices = torch.arange(size, device=volume.device)
        neighbours = (
            (indices - 1).clamp_min(0),
            indices,
            (indices + 1).clamp_max(size - 1),
        )
        if axis == 1:
            stacked = torch.stack([volume[:, index] for index in neighbours], dim=2)
            return stacked.reshape(-1, 3, volume.shape[2], volume.shape[3])
        if axis == 2:
            reordered = volume.permute(0, 2, 1, 3)
            stacked = torch.stack([reordered[:, index] for index in neighbours], dim=2)
            return stacked.reshape(-1, 3, volume.shape[1], volume.shape[3])
        raise ValueError("axis must be 1 (X) or 2 (Y)")

    def _run_ami(self, marginal: torch.Tensor, fdm: torch.Tensor) -> torch.Tensor:
        chunk_size = self.marginal_batch_size or marginal.shape[0]
        outputs = []
        for start in range(0, marginal.shape[0], chunk_size):
            stop = min(start + chunk_size, marginal.shape[0])
            chunk_fdm = fdm
            if fdm.shape[0] != 1:
                chunk_fdm = fdm[start:stop]
            outputs.append(self.ami(marginal[start:stop], chunk_fdm))
        return torch.cat(outputs, dim=0)

    def forward(self, sparse_volume: torch.Tensor, fdm: torch.Tensor) -> ReconstructionOutput:
        """Map ``[B,1,X,Y,Z_lr]`` to three ``[B,1,X,Y,R*Z_lr]`` outputs."""

        if sparse_volume.ndim != 5 or sparse_volume.shape[1] != 1:
            raise ValueError("sparse_volume must have shape [B,1,X,Y,Z_lr]")
        if not sparse_volume.is_floating_point():
            raise TypeError("sparse_volume must be floating point")
        if not torch.isfinite(sparse_volume).all():
            raise ValueError("sparse_volume contains NaN or Inf")
        if fdm.shape[0] not in (1, sparse_volume.shape[0]):
            raise ValueError(
                "end-to-end reconstruction accepts a shared FDM or one FDM per volume"
            )

        batch_size, _, x_size, y_size, z_lr = sparse_volume.shape
        volume = sparse_volume[:, 0]
        sagittal_input = self._neighbor_triplets(volume, axis=1)
        coronal_input = self._neighbor_triplets(volume, axis=2)

        if fdm.shape[0] != 1:
            sagittal_fdm = fdm.repeat_interleave(x_size, dim=0)
            coronal_fdm = fdm.repeat_interleave(y_size, dim=0)
        else:
            sagittal_fdm = coronal_fdm = fdm

        sagittal_marginal = self._run_ami(sagittal_input, sagittal_fdm)
        coronal_marginal = self._run_ami(coronal_input, coronal_fdm)
        z_hr = z_lr * self.factor

        sagittal = sagittal_marginal.reshape(
            batch_size,
            x_size,
            y_size,
            z_hr,
        ).unsqueeze(1)
        coronal = coronal_marginal.reshape(
            batch_size,
            y_size,
            x_size,
            z_hr,
        ).permute(0, 2, 1, 3).unsqueeze(1)
        prediction = self.rfn(sagittal, coronal)
        return ReconstructionOutput(
            prediction=prediction,
            sagittal=sagittal,
            coronal=coronal,
        )
