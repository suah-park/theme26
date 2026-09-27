"""Sparse-z sampling, E0 interpolation and data-consistency contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from theme26.contracts import SamplingSpec


@dataclass(frozen=True)
class SparseSample:
    """One NumPy ``[X,Y,Z]`` sample and masks aligned to its HR grid.

    A DataLoader adds channel and batch dimensions, converting these masks to
    Torch ``[B,1,X,Y,Z]``. ``primary_metric_mask`` excludes unbracketed boundary
    positions from the headline interpolation comparison.
    """

    sparse_volume: npt.NDArray[np.float32]
    observed_indices: tuple[int, ...]
    observed_mask: npt.NDArray[np.bool_]
    missing_mask: npt.NDArray[np.bool_]
    primary_metric_mask: npt.NDArray[np.bool_]
    hr_shape_xyz: tuple[int, int, int]
    sampling: SamplingSpec


def make_sparse_sample(
    dense: npt.NDArray[np.float32], *, sampling: SamplingSpec
) -> SparseSample:
    """Compress z and return HR-grid observed, missing and primary metric masks."""

    array = np.asarray(dense)
    if array.ndim != 3:
        raise ValueError(f"dense volume must be 3D [X,Y,Z], got shape={array.shape}")
    if array.dtype != np.float32:
        raise TypeError(f"dense volume must be float32, got {array.dtype}")
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError("dense volume must contain finite values in [0,1]")

    z_size = int(array.shape[2])
    observed_indices = sampling.observed_indices(z_size)
    sparse_volume = np.ascontiguousarray(array[..., observed_indices], dtype=np.float32)
    observed_mask = np.zeros(array.shape, dtype=np.bool_)
    observed_mask[..., observed_indices] = True
    primary_metric_mask = np.zeros(array.shape, dtype=np.bool_)
    primary_metric_mask[..., sampling.bracketed_missing_indices(z_size)] = True
    return SparseSample(
        sparse_volume=sparse_volume,
        observed_indices=observed_indices,
        observed_mask=observed_mask,
        missing_mask=~observed_mask,
        primary_metric_mask=primary_metric_mask,
        hr_shape_xyz=tuple(int(value) for value in array.shape),
        sampling=sampling,
    )


def linear_interpolate_z(sample: SparseSample) -> npt.NDArray[np.float32]:
    """Return E0 dense ``[X,Y,Z_hr]`` using linear interpolation and boundary policy."""

    output = np.empty(sample.hr_shape_xyz, dtype=np.float32)
    observed = sample.observed_indices
    sparse = sample.sparse_volume
    if sparse.shape[:2] != sample.hr_shape_xyz[:2] or sparse.shape[2] != len(observed):
        raise ValueError("sparse volume shape is inconsistent with its metadata")

    first = observed[0]
    output[..., : first + 1] = sparse[..., 0, None]
    for sparse_index, (left, right) in enumerate(zip(observed, observed[1:])):
        left_slice = sparse[..., sparse_index]
        right_slice = sparse[..., sparse_index + 1]
        width = right - left
        for z_index in range(left, right + 1):
            weight = np.float32((z_index - left) / width)
            output[..., z_index] = (
                (np.float32(1.0) - weight) * left_slice + weight * right_slice
            )
    last = observed[-1]
    output[..., last:] = sparse[..., -1, None]
    return output


def compose_full_volume(
    prediction: npt.NDArray[np.float32], sample: SparseSample
) -> npt.NDArray[np.float32]:
    """Copy observed sparse slices into an HR-grid prediction for 3D evaluation."""

    array = np.asarray(prediction)
    if array.shape != sample.hr_shape_xyz:
        raise ValueError(
            f"prediction shape must be {sample.hr_shape_xyz}, got {array.shape}"
        )
    if array.dtype != np.float32:
        raise TypeError(f"prediction must be float32, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError("prediction contains non-finite values")
    output = np.ascontiguousarray(array, dtype=np.float32).copy()
    output[..., sample.observed_indices] = sample.sparse_volume
    return output
