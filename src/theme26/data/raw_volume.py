"""Raw volume I/O and canonical grayscale/binary conversion contracts."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt

from theme26.contracts import VolumeMetadata


def read_raw_volume(path: str | Path, *, metadata: VolumeMetadata) -> npt.NDArray[np.generic]:
    """Read one raw volume as ``[X,Y,Z]`` using explicit metadata.

    Returns the source dtype without normalization. Implementations must verify
    file size, shape, dtype and endianness rather than inferring them.
    """

    raw_path = Path(path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw volume does not exist: {raw_path}")

    dtype = np.dtype(metadata.dtype)
    if dtype.kind not in "buif":
        raise TypeError(f"raw volume dtype must be numeric, got {dtype}")
    if metadata.endianness == "not-applicable":
        if dtype.itemsize != 1:
            raise ValueError("endianness='not-applicable' is valid only for one-byte dtypes")
    else:
        dtype = dtype.newbyteorder("<" if metadata.endianness == "little" else ">")

    expected_bytes = math.prod(metadata.shape_xyz) * dtype.itemsize
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"raw volume size mismatch: expected {expected_bytes} bytes for "
            f"shape={metadata.shape_xyz}, dtype={dtype}; found {actual_bytes}"
        )

    storage_shape = (
        metadata.shape_xyz
        if metadata.storage_order == "XYZ"
        else tuple(reversed(metadata.shape_xyz))
    )
    mapped = np.memmap(raw_path, dtype=dtype, mode="r", shape=storage_shape, order="C")
    if metadata.storage_order == "ZYX":
        return mapped.transpose(2, 1, 0)
    return mapped


def normalize_grayscale(
    volume: npt.NDArray[np.generic], *, lower: float, upper: float
) -> npt.NDArray[np.float32]:
    """Apply frozen train-derived clipping/scaling and return float32 ``[0,1]``."""

    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError(f"grayscale volume must be 3D [X,Y,Z], got shape={array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"grayscale volume must be numeric, got dtype={array.dtype}")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("normalization bounds must be finite and satisfy lower < upper")
    if not np.isfinite(array).all():
        raise ValueError("grayscale volume contains non-finite values")

    normalized = np.asarray(array, dtype=np.float32)
    normalized = np.clip(normalized, lower, upper)
    normalized = (normalized - np.float32(lower)) / np.float32(upper - lower)
    return np.ascontiguousarray(normalized, dtype=np.float32)


def to_pore_mask(
    binary: npt.NDArray[np.generic], *, source_pore_value: int
) -> npt.NDArray[np.bool_]:
    """Return canonical bool ``[X,Y,Z]`` with ``True`` meaning pore."""

    array = np.asarray(binary)
    if array.ndim != 3:
        raise ValueError(f"binary volume must be 3D [X,Y,Z], got shape={array.shape}")
    if not np.issubdtype(array.dtype, np.number) and array.dtype != np.bool_:
        raise TypeError(f"binary volume must be numeric or bool, got dtype={array.dtype}")
    unique = np.unique(array)
    if unique.size > 2:
        raise ValueError(f"binary volume contains more than two labels: {unique.tolist()}")
    if source_pore_value not in unique:
        raise ValueError(
            f"source_pore_value={source_pore_value} is absent from labels {unique.tolist()}"
        )
    return np.ascontiguousarray(array == source_pore_value, dtype=np.bool_)
