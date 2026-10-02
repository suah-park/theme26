"""Raw volume I/O and canonical grayscale/binary conversion contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from theme26.contracts import VolumeMetadata


def _source_dtype(metadata: VolumeMetadata) -> np.dtype[np.generic]:
    """Build the on-disk dtype from separately recorded dtype/byte order."""

    try:
        base_dtype = np.dtype(metadata.dtype)
    except TypeError as error:
        raise ValueError(f"unsupported raw dtype {metadata.dtype!r}") from error
    if base_dtype.hasobject or base_dtype.fields is not None:
        raise ValueError(f"raw dtype must be a plain numeric dtype, got {base_dtype!s}")

    if base_dtype.itemsize == 1:
        if metadata.endianness != "not-applicable":
            raise ValueError("one-byte raw dtypes require endianness='not-applicable'")
        return base_dtype
    if metadata.endianness not in {"little", "big"}:
        raise ValueError("multi-byte raw dtypes require endianness 'little' or 'big'")
    return base_dtype.newbyteorder("<" if metadata.endianness == "little" else ">")


def _require_volume(array: npt.NDArray[np.generic], *, name: str) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [X,Y,Z], got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must have a real numeric dtype, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must not contain NaN or Inf")


def read_raw_volume(path: str | Path, *, metadata: VolumeMetadata) -> npt.NDArray[np.generic]:
    """Read one raw volume as ``[X,Y,Z]`` using explicit metadata.

    Returns the source dtype without normalization. Implementations must verify
    file size, shape, dtype and endianness rather than inferring them.
    """

    raw_path = Path(path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"raw volume does not exist or is not a file: {raw_path}")

    dtype = _source_dtype(metadata)
    expected_bytes = int(np.prod(metadata.shape_xyz, dtype=np.int64)) * dtype.itemsize
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"raw file size mismatch for {raw_path}: expected {expected_bytes} bytes "
            f"for shape={metadata.shape_xyz}, dtype={dtype}; found {actual_bytes}"
        )

    if metadata.storage_order != 'XYZ':
        raise ValueError('RAW storage_order must be XYZ; this protocol does not transpose axes')
    return np.memmap(raw_path, dtype=dtype, mode='r', shape=metadata.shape_xyz, order='C')



def normalize_grayscale(
    volume: npt.NDArray[np.generic], *, lower: float, upper: float
) -> npt.NDArray[np.float32]:
    """Apply frozen train-derived clipping/scaling and return float32 ``[0,1]``."""

    _require_volume(volume, name="grayscale volume")
    if volume.dtype != np.dtype(np.uint8):
        raise TypeError(f"grayscale source must be uint8, got {volume.dtype}")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"normalization bounds must be finite and lower < upper, got {lower}, {upper}")

    normalized = (volume.astype(np.float32) - np.float32(lower)) / np.float32(upper - lower)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)


def to_pore_mask(
    binary: npt.NDArray[np.generic], *, source_pore_value: int
) -> npt.NDArray[np.bool_]:
    """Return canonical bool ``[X,Y,Z]`` with ``True`` meaning pore."""

    if not isinstance(binary, np.ndarray):
        raise TypeError("binary volume must be a NumPy array")
    if binary.ndim != 3:
        raise ValueError(f"binary volume must have shape [X,Y,Z], got {binary.shape}")
    if not isinstance(source_pore_value, (int, np.integer)) or source_pore_value not in (0, 1):
        raise ValueError(f"source_pore_value must be 0 or 1, got {source_pore_value!r}")
    if not np.issubdtype(binary.dtype, np.integer) and binary.dtype != np.dtype(np.bool_):
        raise TypeError(f"binary volume must have an integer dtype, got {binary.dtype}")
    if not np.isfinite(binary).all():
        raise ValueError("binary volume must not contain NaN or Inf")
    labels = np.unique(binary)
    if labels.size == 0 or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"binary volume may only contain labels 0 and 1, found {labels.tolist()}")
    return (binary == source_pore_value).astype(np.bool_, copy=False)
