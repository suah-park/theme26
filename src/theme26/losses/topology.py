"""Differentiable grayscale-to-topology loss contracts."""

from __future__ import annotations

import torch


def soft_pore_probability(
    grayscale: torch.Tensor, *, threshold: float, temperature: float
) -> torch.Tensor:
    """Map dark-pore grayscale ``[0,1]`` to soft pore probability."""

    raise NotImplementedError("loss owner: implement validated differentiable soft threshold")


def soft_dice_loss(
    probability: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """Return scalar soft Dice loss for pore probability and binary pore target."""

    raise NotImplementedError("loss owner: implement soft Dice with empty-mask tests")


def soft_cldice_loss_3d(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    iterations: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return scalar 3D soft-clDice connectivity surrogate."""

    raise NotImplementedError("loss owner: implement and phantom-test 3D soft skeletonization")


def topology_loss(
    probability: torch.Tensor,
    target: torch.Tensor,
    *,
    iterations: int,
    dice_weight: float = 0.5,
    cldice_weight: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Combine soft Dice and 3D soft-clDice using config-frozen iterations."""

    raise NotImplementedError("loss owner: implement weighted topology loss and weight validation")
