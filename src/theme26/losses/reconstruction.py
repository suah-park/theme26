"""Pixel-wise reconstruction losses."""

from __future__ import annotations

import torch


def _validate_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have identical shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("prediction and target must be floating point")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must not contain NaN or Inf")


def _broadcast_mask(mask: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if mask.dtype is not torch.bool:
        raise TypeError("mask must have dtype bool")
    try:
        expanded = torch.broadcast_to(mask, shape)
    except RuntimeError as error:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is not broadcastable to {tuple(shape)}"
        ) from error
    if not expanded.any():
        raise ValueError("mask must select at least one element")
    return expanded


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return scalar L1, optionally over a broadcast-compatible bool mask."""

    _validate_pair(prediction, target)
    absolute_error = (prediction - target).abs()
    if mask is None:
        return absolute_error.mean()
    expanded_mask = _broadcast_mask(mask.to(device=prediction.device), prediction.shape)
    return absolute_error.masked_select(expanded_mask).mean()
