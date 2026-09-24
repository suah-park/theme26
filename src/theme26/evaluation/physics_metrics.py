"""Solver-facing physical metric contracts."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt


class PhysicsSolver(Protocol):
    """Adapter contract for a frozen external tortuosity/permeability solver."""

    name: str
    version: str

    def solve(
        self,
        pore_mask: npt.NDArray[np.bool_],
        *,
        spacing_um_xyz: tuple[float, float, float],
    ) -> dict[str, float | bool | str]:
        """Return JSON-serializable solver outputs for one pore volume."""


def compute_physics_metrics(
    predicted_pore: npt.NDArray[np.bool_],
    target_pore: npt.NDArray[np.bool_],
    *,
    spacing_um_xyz: tuple[float, float, float],
    solver: PhysicsSolver,
) -> dict[str, float | bool | str]:
    """Run the same frozen solver on prediction/target and report paired errors."""

    raise NotImplementedError("evaluation owner: implement solver adapter after protocol freeze")
