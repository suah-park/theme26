"""Aligned grayscale/binary crop contracts."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from theme26.contracts import CropSpec


def extract_volume_crop(
    volume: npt.NDArray[np.generic], *, crop: CropSpec
) -> npt.NDArray[np.generic]:
    """Extract one copied ``[X,Y,Z]`` crop while preserving its dtype."""

    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"volume must be 3D [X,Y,Z], got shape={array.shape}")
    crop.validate_inside(tuple(int(value) for value in array.shape))
    ox, oy, oz = crop.origin_xyz
    sx, sy, sz = crop.shape_xyz
    return np.ascontiguousarray(array[ox : ox + sx, oy : oy + sy, oz : oz + sz])


def extract_aligned_crop(
    grayscale: npt.NDArray[np.float32],
    binary: npt.NDArray[np.bool_],
    *,
    crop: CropSpec,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Extract identically aligned ``[X,Y,Z]`` grayscale and pore-mask crops."""

    gray = np.asarray(grayscale)
    pore = np.asarray(binary)
    if gray.shape != pore.shape:
        raise ValueError(f"grayscale/binary shape mismatch: {gray.shape} vs {pore.shape}")
    if gray.ndim != 3:
        raise ValueError(f"volumes must be 3D [X,Y,Z], got shape={gray.shape}")
    if gray.dtype != np.float32:
        raise TypeError(f"grayscale must be float32, got {gray.dtype}")
    if pore.dtype != np.bool_:
        raise TypeError(f"binary pore mask must be bool, got {pore.dtype}")
    if not np.isfinite(gray).all() or np.any((gray < 0.0) | (gray > 1.0)):
        raise ValueError("grayscale must contain finite values in [0,1]")
    return (
        extract_volume_crop(gray, crop=crop),
        extract_volume_crop(pore, crop=crop),
    )


def sample_training_crop(
    grayscale: npt.NDArray[np.float32],
    binary: npt.NDArray[np.bool_],
    *,
    crop_shape_xyz: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], CropSpec]:
    """Sample one reproducible aligned training crop and return its coordinates."""

    gray = np.asarray(grayscale)
    pore = np.asarray(binary)
    if gray.shape != pore.shape or gray.ndim != 3:
        raise ValueError(
            f"aligned 3D grayscale/binary volumes are required, got {gray.shape} and {pore.shape}"
        )
    if len(crop_shape_xyz) != 3 or any(size <= 0 for size in crop_shape_xyz):
        raise ValueError("crop_shape_xyz must contain three positive values")
    if any(size > limit for size, limit in zip(crop_shape_xyz, gray.shape)):
        raise ValueError(f"crop shape {crop_shape_xyz} exceeds volume shape {gray.shape}")
    origin = tuple(
        int(rng.integers(0, limit - size + 1))
        for size, limit in zip(crop_shape_xyz, gray.shape)
    )
    crop = CropSpec(origin_xyz=origin, shape_xyz=crop_shape_xyz)
    gray_crop, pore_crop = extract_aligned_crop(gray, pore, crop=crop)
    return gray_crop, pore_crop, crop
