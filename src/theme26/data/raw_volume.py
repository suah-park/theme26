"""Raw volume I/O and canonical grayscale/binary conversion contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

from theme26.contracts import VolumeMetadata


def read_raw_volume(path: str | Path, *, metadata: VolumeMetadata) -> npt.NDArray[np.generic]:
    """Read one raw volume as ``[X,Y,Z]`` using explicit metadata.

    Returns the source dtype without normalization. Implementations must verify
    file size, shape, dtype and endianness rather than inferring them.
    """

    raise NotImplementedError("data owner: implement raw file validation and loading")


def normalize_grayscale(
    volume: npt.NDArray[np.generic], *, lower: float, upper: float
) -> npt.NDArray[np.float32]:
    """Apply frozen train-derived clipping/scaling and return float32 ``[0,1]``."""

    raise NotImplementedError("data owner: implement frozen grayscale normalization")


def to_pore_mask(
    binary: npt.NDArray[np.generic], *, source_pore_value: int
) -> npt.NDArray[np.bool_]:
    """Return canonical bool ``[X,Y,Z]`` with ``True`` meaning pore."""

    raise NotImplementedError("data owner: implement binary-polarity validation and conversion")
