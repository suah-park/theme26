"""Hard-segmentation structure and topology metric contracts."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_topology_metrics(
    predicted_pore: npt.NDArray[np.bool_],
    target_pore: npt.NDArray[np.bool_],
    *,
    spacing_um_xyz: tuple[float, float, float],
) -> dict[str, float | bool]:
    """Return frozen pore-overlap, morphology, connectivity and percolation metrics."""

    raise NotImplementedError("evaluation owner: implement versioned 3D topology metrics")
