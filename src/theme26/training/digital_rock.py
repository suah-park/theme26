"""Digital-rock adapters that turn canonical volumes into SAINT E1 batches."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from theme26.contracts import CropSpec, SamplingSpec, VolumeMetadata
from theme26.data import (
    CropRecord,
    extract_volume_crop,
    make_sparse_sample,
    normalize_grayscale,
    read_raw_volume,
)
from theme26.models.saint import build_filter_distance_matrix

MarginalAxis = Literal["sagittal", "coronal"]


def _validate_directions(
    directions: Sequence[MarginalAxis],
) -> tuple[MarginalAxis, ...]:
    if not directions or any(
        item not in ("sagittal", "coronal") for item in directions
    ):
        raise ValueError("directions must contain sagittal and/or coronal")
    return tuple(directions)


def _neighbor_indices(index: int, size: int) -> tuple[int, int, int]:
    return max(index - 1, 0), index, min(index + 1, size - 1)


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

        self.dense = np.ascontiguousarray(dense)
        self.sample = make_sparse_sample(self.dense, sampling=sampling)
        self.directions = _validate_directions(directions)
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

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        direction, index = self._entries[item]
        if direction == "sagittal":
            neighbors = _neighbor_indices(index, self.dense.shape[0])
            marginal = self.sample.sparse_volume[list(neighbors), :, :]
            target = self.dense[index, :, :][None, ...]
        else:
            neighbors = _neighbor_indices(index, self.dense.shape[1])
            marginal = self.sample.sparse_volume[:, list(neighbors), :].transpose(
                1, 0, 2
            )
            target = self.dense[:, index, :][None, ...]

        return {
            "input": torch.from_numpy(np.ascontiguousarray(marginal)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "fdm": self._fdm[direction].clone(),
            "primary_mask": torch.from_numpy(np.ascontiguousarray(
                self.sample.primary_metric_mask[index] if direction == "sagittal"
                else self.sample.primary_metric_mask[:, index, :]
            )).unsqueeze(0),
        }


class ManifestAmiDataset(Dataset[dict[str, torch.Tensor]]):
    """Read AMI marginals lazily from many manifest crops in one raw volume."""

    def __init__(
        self,
        raw_volume: np.ndarray,
        *,
        records: Sequence[CropRecord],
        metadata: VolumeMetadata,
        normalization_lower: float,
        normalization_upper: float,
        kernel_size: int = 3,
        directions: Sequence[MarginalAxis] = ("sagittal", "coronal"),
    ) -> None:
        raw = np.asarray(raw_volume)
        if raw.ndim != 3 or tuple(raw.shape) != metadata.shape_xyz:
            raise ValueError(
                f"raw volume must have canonical shape {metadata.shape_xyz}, got {raw.shape}"
            )
        if not records:
            raise ValueError("records cannot be empty")
        if not np.isfinite(normalization_lower) or not np.isfinite(
            normalization_upper
        ):
            raise ValueError("normalization bounds must be finite")
        if normalization_lower >= normalization_upper:
            raise ValueError("normalization_lower must be below normalization_upper")

        self.raw = raw
        self.records = tuple(records)
        self.metadata = metadata
        self.normalization_lower = float(normalization_lower)
        self.normalization_upper = float(normalization_upper)
        self.directions = _validate_directions(directions)
        self._entries = tuple(
            (record_index, direction, marginal_index)
            for record_index, record in enumerate(self.records)
            for direction in self.directions
            for marginal_index in range(
                record.crop.shape_xyz[0 if direction == "sagittal" else 1]
            )
        )
        self._fdm = {
            (record.sampling.factor, direction): build_filter_distance_matrix(
                factor=record.sampling.factor,
                kernel_size=kernel_size,
                hr_spacing_um_xyz=metadata.voxel_spacing_um_xyz,
                marginal_axis=direction,
            )[0]
            for record in self.records
            for direction in self.directions
        }
        for record in self.records:
            if record.source_core_id != metadata.source_core_id:
                raise ValueError(
                    f"record core {record.source_core_id!r} does not match "
                    f"metadata core {metadata.source_core_id!r}"
                )
            record.crop.validate_inside(metadata.shape_xyz)

    @classmethod
    def from_raw(
        cls,
        path: str | Path,
        *,
        records: Sequence[CropRecord],
        metadata: VolumeMetadata,
        normalization_lower: float,
        normalization_upper: float,
        kernel_size: int = 3,
        directions: Sequence[MarginalAxis] = ("sagittal", "coronal"),
    ) -> "ManifestAmiDataset":
        raw = read_raw_volume(path, metadata=metadata)
        return cls(
            raw,
            records=records,
            metadata=metadata,
            normalization_lower=normalization_lower,
            normalization_upper=normalization_upper,
            kernel_size=kernel_size,
            directions=directions,
        )

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        record_index, direction, marginal_index = self._entries[item]
        record = self.records[record_index]
        ox, oy, oz = record.crop.origin_xyz
        sx, sy, sz = record.crop.shape_xyz
        observed_z = oz + np.asarray(
            record.sampling.observed_indices(sz), dtype=np.intp
        )

        if direction == "sagittal":
            neighbors = ox + np.asarray(
                _neighbor_indices(marginal_index, sx), dtype=np.intp
            )
            y_indices = np.arange(oy, oy + sy, dtype=np.intp)
            marginal_raw = self.raw[np.ix_(neighbors, y_indices, observed_z)]
            target_raw = self.raw[
                ox + marginal_index, oy : oy + sy, oz : oz + sz
            ][None, ...]
        else:
            neighbors = oy + np.asarray(
                _neighbor_indices(marginal_index, sy), dtype=np.intp
            )
            x_indices = np.arange(ox, ox + sx, dtype=np.intp)
            marginal_raw = self.raw[np.ix_(x_indices, neighbors, observed_z)].transpose(
                1, 0, 2
            )
            target_raw = self.raw[
                ox : ox + sx, oy + marginal_index, oz : oz + sz
            ][None, ...]

        marginal = normalize_grayscale(
            marginal_raw,
            lower=self.normalization_lower,
            upper=self.normalization_upper,
        )
        target = normalize_grayscale(
            target_raw,
            lower=self.normalization_lower,
            upper=self.normalization_upper,
        )
        return {
            "input": torch.from_numpy(marginal),
            "target": torch.from_numpy(target),
            "fdm": self._fdm[(record.sampling.factor, direction)].clone(),
            "primary_mask": torch.from_numpy(np.broadcast_to(
                np.isin(np.arange(sz), record.sampling.bracketed_missing_indices(sz)),
                target.shape,
            ).copy()),
        }
