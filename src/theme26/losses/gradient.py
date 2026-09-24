"""Finite-difference grayscale gradient loss contracts."""

from __future__ import annotations

import torch


def gradient_loss_2d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compare finite differences on tensors shaped ``[B,C,H,W]``."""

    raise NotImplementedError("loss owner: implement documented 2D finite differences")


def gradient_loss_3d(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compare XYZ finite differences on tensors shaped ``[B,C,X,Y,Z]``."""

    raise NotImplementedError("loss owner: implement optional assembled-volume gradient loss")
