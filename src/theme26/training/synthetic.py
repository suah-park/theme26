"""Small deterministic datasets for CPU/GPU training-loop smoke tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from theme26.models.saint import build_filter_distance_matrix


def _smooth(image: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(image.unsqueeze(0), kernel_size=3, stride=1, padding=1)[0]


class SyntheticAmiDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic marginal pairs used only to verify E1 AMI training."""

    def __init__(
        self,
        *,
        samples: int,
        length: int,
        z_lr: int,
        factor: int,
        spacing_um_xyz: tuple[float, float, float],
        seed: int,
    ) -> None:
        if min(samples, length, z_lr) < 1:
            raise ValueError("samples, length and z_lr must be positive")
        if factor < 2:
            raise ValueError("factor must be at least 2")
        self.samples = samples
        self.length = length
        self.z_lr = z_lr
        self.factor = factor
        self.seed = seed
        self.fdm = build_filter_distance_matrix(
            factor=factor,
            kernel_size=3,
            hr_spacing_um_xyz=spacing_um_xyz,
            marginal_axis="sagittal",
        )[0]

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        target = _smooth(
            torch.rand(1, self.length, self.z_lr * self.factor, generator=generator)
        )
        centre = target[..., :: self.factor]
        marginal = torch.cat(
            (
                (centre * 0.97).clamp(0.0, 1.0),
                centre,
                (centre * 1.03).clamp(0.0, 1.0),
            ),
            dim=0,
        )
        indices = torch.arange(target.shape[-1])
        primary = (indices > 0) & (indices < (self.z_lr - 1) * self.factor) & (indices % self.factor != 0)
        return {"input": marginal, "target": target, "fdm": self.fdm.clone(),
                "primary_mask": primary.view(1, 1, -1).expand_as(target)}


class SyntheticRfnDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic axial pairs used only to verify E1 RFN training."""

    def __init__(self, *, samples: int, x_size: int, y_size: int, seed: int) -> None:
        if min(samples, x_size, y_size) < 1:
            raise ValueError("samples, x_size and y_size must be positive")
        self.samples = samples
        self.x_size = x_size
        self.y_size = y_size
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        target = _smooth(
            torch.rand(1, self.x_size, self.y_size, generator=generator)
        )
        row_bias = torch.linspace(-0.08, 0.08, self.x_size).view(1, self.x_size, 1)
        column_bias = torch.linspace(0.08, -0.08, self.y_size).view(1, 1, self.y_size)
        sagittal = (target + row_bias).clamp(0.0, 1.0)
        coronal = (target + column_bias).clamp(0.0, 1.0)
        return {
            "input": torch.cat((sagittal, coronal), dim=0),
            "target": target,
            # Synthetic RFN samples represent missing axial slices only.
            "primary_mask": torch.ones_like(target, dtype=torch.bool),
        }
