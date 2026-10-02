"""Checkpoint persistence for staged SAINT training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: object | None,
    epoch: int,
    validation_loss: float,
    config: Mapping[str, Any],
) -> Path:
    """Atomically save a restartable training checkpoint."""

    if epoch < 1:
        raise ValueError("epoch must be at least 1")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    scaler_state = None
    if scaler is not None and hasattr(scaler, "state_dict"):
        scaler_state = scaler.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "validation_loss": float(validation_loss),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler_state,
            "config": dict(config),
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: object | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model and optional optimizer/scaler state."""

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if (
        scaler is not None
        and checkpoint.get("scaler_state") is not None
        and hasattr(scaler, "load_state_dict")
    ):
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint
