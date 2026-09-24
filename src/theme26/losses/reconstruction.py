"""Pixel-wise reconstruction loss contract."""

from __future__ import annotations

import torch


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return scalar L1, optionally over a broadcast-compatible bool mask."""

    raise NotImplementedError("loss owner: implement shape-safe masked L1")
