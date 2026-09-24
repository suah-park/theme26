"""Aligned grayscale/binary crop contracts."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from theme26.contracts import CropSpec


def extract_aligned_crop(
    grayscale: npt.NDArray[np.float32],
    binary: npt.NDArray[np.bool_],
    *,
    crop: CropSpec,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Extract identically aligned ``[X,Y,Z]`` grayscale and pore-mask crops."""

    raise NotImplementedError("data owner: implement aligned crop extraction and validation")


def sample_training_crop(
    grayscale: npt.NDArray[np.float32],
    binary: npt.NDArray[np.bool_],
    *,
    crop_shape_xyz: tuple[int, int, int],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], CropSpec]:
    """Sample one reproducible aligned training crop and return its coordinates."""

    raise NotImplementedError("data owner: implement training-only random crop sampling")
