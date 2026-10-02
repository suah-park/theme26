"""Aligned grayscale/binary crop contracts."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from theme26.contracts import CropSpec


def _validate_inputs(grayscale: npt.NDArray[np.float32], binary: npt.NDArray[np.bool_]) -> None:
    if not isinstance(grayscale, np.ndarray) or not isinstance(binary, np.ndarray):
        raise TypeError("grayscale and binary must be NumPy arrays")
    if grayscale.ndim != 3 or binary.ndim != 3:
        raise ValueError("grayscale and binary must both have shape [X,Y,Z]")
    if grayscale.shape != binary.shape:
        raise ValueError(f"grayscale/binary shape mismatch: {grayscale.shape} != {binary.shape}")
    if grayscale.dtype != np.dtype(np.float32):
        raise TypeError(f"grayscale must be float32, got {grayscale.dtype}")
    if binary.dtype != np.dtype(np.bool_):
        raise TypeError(f"binary must be bool, got {binary.dtype}")
    if not np.isfinite(grayscale).all() or np.any((grayscale < 0.0) | (grayscale > 1.0)):
        raise ValueError("grayscale must be finite and within [0,1]")


def extract_aligned_crop(
    grayscale: npt.NDArray[np.float32],
    binary: npt.NDArray[np.bool_],
    *,
    crop: CropSpec,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Extract identically aligned ``[X,Y,Z]`` grayscale and pore-mask crops."""

    _validate_inputs(grayscale, binary)
    crop.validate_inside(grayscale.shape)
    x0, y0, z0 = crop.origin_xyz
    sx, sy, sz = crop.shape_xyz
    region = np.s_[x0 : x0 + sx, y0 : y0 + sy, z0 : z0 + sz]
    # Copies make the crop independent of the potentially gigabyte-sized source.
    return grayscale[region].copy(), binary[region].copy()


def sample_training_crop(
    grayscale: npt.NDArray[np.float32],
    binary: npt.NDArray[np.bool_],
    *,
    crop_shape_xyz: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], CropSpec]:
    """Sample one reproducible aligned training crop and return its coordinates."""

    _validate_inputs(grayscale, binary)
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if len(crop_shape_xyz) != 3 or any(not isinstance(size, (int, np.integer)) or size <= 0 for size in crop_shape_xyz):
        raise ValueError(f"crop_shape_xyz must contain three positive integers, got {crop_shape_xyz!r}")
    if any(size > limit for size, limit in zip(crop_shape_xyz, grayscale.shape)):
        raise ValueError(f"crop shape {crop_shape_xyz} exceeds volume shape {grayscale.shape}")
    origin = tuple(
        int(rng.integers(0, limit - size + 1))
        for size, limit in zip(crop_shape_xyz, grayscale.shape)
    )
    crop = CropSpec(origin_xyz=origin, shape_xyz=tuple(int(size) for size in crop_shape_xyz))
    gray_crop, binary_crop = extract_aligned_crop(grayscale, binary, crop=crop)
    return gray_crop, binary_crop, crop


def extract_volume_crop(volume, *, crop: CropSpec):
    """Copy one XYZ crop without loading the whole RAW into RAM."""
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError('volume must be 3D [X,Y,Z]')
    crop.validate_inside(tuple(array.shape))
    x,y,z = crop.origin_xyz
    sx,sy,sz = crop.shape_xyz
    return array[x:x+sx,y:y+sy,z:z+sz].copy()
