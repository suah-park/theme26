"""Missing-slice grayscale metric contracts."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_image_metrics(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    *,
    mask: npt.NDArray[np.bool_],
    data_range: float = 1.0,
) -> dict[str, float]:
    """Return MAE, PSNR, SSIM and gradient error on an explicit HR-grid mask."""

    raise NotImplementedError("evaluation owner: implement frozen missing-slice image metrics")
