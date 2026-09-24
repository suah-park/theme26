"""Filter Distance Matrix contract."""

from __future__ import annotations

import torch


def build_filter_distance_matrix(
    *,
    factor: int,
    kernel_size: int,
    hr_spacing_um_xyz: tuple[float, float, float],
    marginal_axis: str,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return physical-distance FDM with shape ``[1,factor,K,K]``.

    ``marginal_axis`` must identify sagittal/X-fixed or coronal/Y-fixed input so
    the correct in-plane spacing is used. Distances are expressed in um.
    """

    raise NotImplementedError("model owner: port and test the SAINT FDM definition")
