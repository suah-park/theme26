"""Sparse-z sampling, E0 interpolation and data-consistency contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from theme26.contracts import SamplingSpec


@dataclass(frozen=True)
class SparseSample:
    """Compressed sparse input plus masks aligned to the original HR grid."""

    sparse_volume: npt.NDArray[np.float32]
    observed_indices: tuple[int, ...]
    observed_mask: npt.NDArray[np.bool_]
    missing_mask: npt.NDArray[np.bool_]
    hr_shape_xyz: tuple[int, int, int]
    sampling: SamplingSpec


def make_sparse_sample(
    dense: npt.NDArray[np.float32], *, sampling: SamplingSpec
) -> SparseSample:
    """Compress z using ``sampling`` and return HR-grid masks and indices."""

    raise NotImplementedError("data owner: implement x4 sparse-z sampling")


def linear_interpolate_z(sample: SparseSample) -> npt.NDArray[np.float32]:
    """Return E0 dense ``[X,Y,Z_hr]`` using linear interpolation and boundary policy."""

    raise NotImplementedError("data owner: implement deterministic E0 interpolation")


def compose_full_volume(
    prediction: npt.NDArray[np.float32], sample: SparseSample
) -> npt.NDArray[np.float32]:
    """Copy observed sparse slices into an HR-grid prediction for 3D evaluation."""

    raise NotImplementedError("data owner: implement observed-slice data consistency")
