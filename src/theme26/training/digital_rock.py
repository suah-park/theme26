"""Digital-rock adapters that turn canonical volumes into SAINT E1 batches."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from theme26.contracts import CropSpec, SamplingSpec, VolumeMetadata
from theme26.data import (
    extract_volume_crop,
    make_sparse_sample,
    normalize_grayscale,
    read_raw_volume,
)
from theme26.models.saint import build_filter_distance_matrix

MarginalAxis = Literal["sagittal", "coronal"]


class DigitalRockAmiDataset(Dataset[dict[str, torch.Tensor]]):
    """Create shared-AMI marginal samples from one normalized dense crop.

    Sagittal samples are fixed-X ``[Y,Z]`` planes and coronal samples are
    fixed-Y ``[X,Z]`` planes.  Each input has three neighboring planes as
    channels; boundary neighbors use replication.
    """

    def __init__(
        self,
        dense_crop: np.ndarray,
        *,
        sampling: SamplingSpec,
        spacing_um_xyz: tuple[float, float, float],
        kernel_size: int = 3,
        directions: Sequence[MarginalAxis] = ("sagittal", "coronal"),
    ) -> None:
        dense = np.asarray(dense_crop)
        if dense.ndim != 3 or dense.dtype != np.float32:
            raise TypeError("dense_crop must be float32 [X,Y,Z]")
        if not np.isfinite(dense).all() or np.any((dense < 0.0) | (dense > 1.0)):
            raise ValueError("dense_crop must contain finite values in [0,1]")
        if not directions or any(
            item not in ("sagittal", "coronal") for item in directions
        ):
            raise ValueError("directions must contain sagittal and/or coronal")

        self.dense = np.ascontiguousarray(dense)
        self.sample = make_sparse_sample(self.dense, sampling=sampling)
        self.directions = tuple(directions)
        self._entries = tuple(
            (direction, index)
            for direction in self.directions
            for index in range(self.dense.shape[0 if direction == "sagittal" else 1])
        )
        self._fdm = {
            direction: build_filter_distance_matrix(
                factor=sampling.factor,
                kernel_size=kernel_size,
                hr_spacing_um_xyz=spacing_um_xyz,
                marginal_axis=direction,
            )[0]
            for direction in self.directions
        }

    @classmethod
    def from_raw(
        cls,
        path: str | Path,
        *,
        metadata: VolumeMetadata,
        crop: CropSpec,
        normalization_lower: float,
        normalization_upper: float,
        sampling: SamplingSpec,
        kernel_size: int = 3,
        directions: Sequence[MarginalAxis] = ("sagittal", "coronal"),
    ) -> "DigitalRockAmiDataset":
        """Load only the requested raw crop and construct an AMI dataset."""

        raw = read_raw_volume(path, metadata=metadata)
        raw_crop = extract_volume_crop(raw, crop=crop)
        dense_crop = normalize_grayscale(
            raw_crop, lower=normalization_lower, upper=normalization_upper
        )
        return cls(
            dense_crop,
            sampling=sampling,
            spacing_um_xyz=metadata.voxel_spacing_um_xyz,
            kernel_size=kernel_size,
            directions=directions,
        )

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _neighbor_indices(index: int, size: int) -> tuple[int, int, int]:
        return max(index - 1, 0), index, min(index + 1, size - 1)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        direction, index = self._entries[item]
        if direction == "sagittal":
            neighbors = self._neighbor_indices(index, self.dense.shape[0])
            marginal = self.sample.sparse_volume[list(neighbors), :, :]
            target = self.dense[index, :, :][None, ...]
        else:
            neighbors = self._neighbor_indices(index, self.dense.shape[1])
            marginal = self.sample.sparse_volume[:, list(neighbors), :].transpose(
                1, 0, 2
            )
            target = self.dense[:, index, :][None, ...]

        return {
            "input": torch.from_numpy(np.ascontiguousarray(marginal)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "fdm": self._fdm[direction].clone(),
        }
