"""Finite-difference grayscale gradient losses."""

from __future__ import annotations

import torch

from .reconstruction import _broadcast_mask, _validate_pair


def _directional_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    dimension: int,
) -> torch.Tensor:
    before = [slice(None)] * prediction.ndim
    after = [slice(None)] * prediction.ndim
    before[dimension] = slice(None, -1)
    after[dimension] = slice(1, None)

    prediction_gradient = prediction[tuple(after)] - prediction[tuple(before)]
    target_gradient = target[tuple(after)] - target[tuple(before)]
    error = (prediction_gradient - target_gradient).abs()
    if mask is None:
        return error.mean()

    expanded_mask = _broadcast_mask(mask.to(device=prediction.device), prediction.shape)
    edge_mask = expanded_mask[tuple(after)] & expanded_mask[tuple(before)]
    if not edge_mask.any():
        raise ValueError("mask does not contain a valid neighbouring pair")
    return error.masked_select(edge_mask).mean()


def gradient_loss_2d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compare finite differences on tensors shaped ``[B,C,H,W]``."""

    _validate_pair(prediction, target)
    if prediction.ndim != 4:
        raise ValueError("2D gradient loss expects [B,C,H,W]")
    if prediction.shape[-2] < 2 or prediction.shape[-1] < 2:
        raise ValueError("H and W must both be at least 2")
    return _directional_loss(prediction, target, mask, dimension=-2) + _directional_loss(
        prediction,
        target,
        mask,
        dimension=-1,
    )


def gradient_loss_3d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compare XYZ finite differences on tensors shaped ``[B,C,X,Y,Z]``."""

    _validate_pair(prediction, target)
    if prediction.ndim != 5:
        raise ValueError("3D gradient loss expects [B,C,X,Y,Z]")
    if any(size < 2 for size in prediction.shape[-3:]):
        raise ValueError("X, Y and Z must all be at least 2")
    return sum(
        (_directional_loss(prediction, target, mask, dimension=dimension) for dimension in (-3, -2, -1)),
        start=prediction.new_zeros(()),
    )
