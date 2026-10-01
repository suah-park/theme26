"""Sparse-z sampling, E0 interpolation and data-consistency contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from theme26.contracts import SamplingSpec


def _validate_dense(dense: npt.NDArray[np.float32], *, name: str = "dense") -> None:
    if not isinstance(dense, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if dense.ndim != 3:
        raise ValueError(f"{name} must have shape [X,Y,Z], got {dense.shape}")
    if dense.dtype != np.dtype(np.float32):
        raise TypeError(f"{name} must be float32, got {dense.dtype}")
    if not np.isfinite(dense).all() or np.any((dense < 0.0) | (dense > 1.0)):
        raise ValueError(f"{name} must be finite and within [0,1]")


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

    _validate_dense(dense)
    observed_indices = sampling.observed_indices(dense.shape[2])
    observed_mask = np.zeros(dense.shape, dtype=np.bool_)
    observed_mask[..., observed_indices] = True
    missing_mask = ~observed_mask
    primary_metric_mask = np.zeros(dense.shape, dtype=np.bool_)
    bracketed = sampling.bracketed_missing_indices(dense.shape[2])
    primary_metric_mask[..., bracketed] = True
    return SparseSample(
        sparse_volume=dense[..., observed_indices].copy(),
        observed_indices=observed_indices,
        observed_mask=observed_mask,
        missing_mask=missing_mask,
        primary_metric_mask=primary_metric_mask,
        hr_shape_xyz=tuple(int(size) for size in dense.shape),
        sampling=sampling,
    )


def linear_interpolate_z(sample: SparseSample) -> npt.NDArray[np.float32]:
    """Return E0 dense ``[X,Y,Z_hr]`` using linear interpolation and boundary policy."""

    _validate_sparse_sample(sample)
    result = np.empty(sample.hr_shape_xyz, dtype=np.float32)
    observed = sample.observed_indices
    sparse = sample.sparse_volume
    for sparse_z, hr_z in enumerate(observed):
        result[..., hr_z] = sparse[..., sparse_z]
    first, last = observed[0], observed[-1]
    result[..., :first] = sparse[..., :1]
    for left_pos, (left, right) in enumerate(zip(observed, observed[1:])):
        left_slice, right_slice = sparse[..., left_pos], sparse[..., left_pos + 1]
        for z in range(left + 1, right):
            weight = np.float32((z - left) / (right - left))
            result[..., z] = left_slice * (np.float32(1.0) - weight) + right_slice * weight
    result[..., last + 1 :] = sparse[..., -1:]
    return result


def compose_full_volume(
    prediction: npt.NDArray[np.float32], sample: SparseSample
) -> npt.NDArray[np.float32]:
    """Copy observed sparse slices into an HR-grid prediction for 3D evaluation."""

    _validate_dense(prediction, name="prediction")
    _validate_sparse_sample(sample)
    if prediction.shape != sample.hr_shape_xyz:
        raise ValueError(f"prediction shape {prediction.shape} != HR shape {sample.hr_shape_xyz}")
    full = prediction.copy()
    full[..., sample.observed_indices] = sample.sparse_volume
    return full


def _validate_sparse_sample(sample: SparseSample) -> None:
    if not isinstance(sample, SparseSample):
        raise TypeError("sample must be a SparseSample")
    _validate_dense(sample.sparse_volume, name="sparse_volume")
    if len(sample.hr_shape_xyz) != 3 or any(size <= 0 for size in sample.hr_shape_xyz):
        raise ValueError(f"invalid HR shape {sample.hr_shape_xyz}")
    observed = sample.sampling.observed_indices(sample.hr_shape_xyz[2])
    expected_sparse_shape = sample.hr_shape_xyz[:2] + (len(observed),)
    if sample.observed_indices != observed or sample.sparse_volume.shape != expected_sparse_shape:
        raise ValueError("SparseSample observed indices or sparse volume shape are inconsistent with sampling")
    for name, mask in (("observed_mask", sample.observed_mask), ("missing_mask", sample.missing_mask), ("primary_metric_mask", sample.primary_metric_mask)):
        if not isinstance(mask, np.ndarray) or mask.dtype != np.dtype(np.bool_) or mask.shape != sample.hr_shape_xyz:
            raise ValueError(f"{name} must be a bool HR-grid mask with shape {sample.hr_shape_xyz}")
    if np.any(sample.observed_mask & sample.missing_mask) or not np.all(sample.observed_mask | sample.missing_mask):
        raise ValueError("observed_mask and missing_mask must be complementary")
    if np.any(sample.primary_metric_mask & ~sample.missing_mask):
        raise ValueError("primary_metric_mask must only contain missing positions")
    expected_observed = np.zeros(sample.hr_shape_xyz, dtype=np.bool_)
    expected_observed[..., observed] = True
    expected_primary = np.zeros(sample.hr_shape_xyz, dtype=np.bool_)
    expected_primary[..., sample.sampling.bracketed_missing_indices(sample.hr_shape_xyz[2])] = True
    if not np.array_equal(sample.observed_mask, expected_observed):
        raise ValueError("observed_mask does not match observed_indices")
    if not np.array_equal(sample.primary_metric_mask, expected_primary):
        raise ValueError("primary_metric_mask does not match bracketed missing indices")
