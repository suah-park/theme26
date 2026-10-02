"""Filter Distance Matrix used by SAINT's dynamic filter generator."""

from __future__ import annotations

import torch


_MARGINAL_AXES = {"sagittal", "coronal"}


def build_filter_distance_matrix(
    *,
    factor: int,
    kernel_size: int,
    hr_spacing_um_xyz: tuple[float, float, float],
    marginal_axis: str,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the physical-distance FDM with shape ``[1,factor,K,K]``.

    The implementation follows Algorithm 1 of the SAINT paper.  A feature-map
    location ``(h, w)`` corresponds to HR coordinate ``(h, factor*w)`` and
    sub-pixel channel ``c`` at the kernel centre corresponds to
    ``(centre, factor*centre + c)``.  The coordinate difference is scaled by
    the physical HR voxel spacing before its Euclidean norm is taken.

    ``marginal_axis="sagittal"`` means a fixed-X ``[Y,Z]`` plane and therefore
    uses ``sy`` as its first in-plane spacing.  ``"coronal"`` means a fixed-Y
    ``[X,Z]`` plane and uses ``sx``.  The second spacing is always ``sz``.
    """

    if factor < 2:
        raise ValueError("factor must be at least 2")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    if marginal_axis not in _MARGINAL_AXES:
        raise ValueError(f"marginal_axis must be one of {sorted(_MARGINAL_AXES)}")
    if len(hr_spacing_um_xyz) != 3 or any(value <= 0 for value in hr_spacing_um_xyz):
        raise ValueError("hr_spacing_um_xyz must contain three positive values")
    if not dtype.is_floating_point:
        raise TypeError("FDM dtype must be floating point")

    sx, sy, sz = (float(value) for value in hr_spacing_um_xyz)
    in_plane_spacing = sy if marginal_axis == "sagittal" else sx
    centre = kernel_size // 2

    row = torch.arange(kernel_size, device=device, dtype=dtype) - centre
    column = torch.arange(kernel_size, device=device, dtype=dtype) - centre
    subpixel = torch.arange(factor, device=device, dtype=dtype)

    delta_plane = row.view(1, kernel_size, 1) * in_plane_spacing
    delta_z = (
        factor * column.view(1, 1, kernel_size)
        - subpixel.view(factor, 1, 1)
    ) * sz
    distances = torch.sqrt(delta_plane.square() + delta_z.square())
    return distances.unsqueeze(0)
