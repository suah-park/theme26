"""Checkpoint persistence for staged SAINT training."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import torch
import numpy as np
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
    best_validation_loss: float | None = None,
    loader_generator: torch.Generator | None = None,
    run_metadata: Mapping[str, Any] | None = None,
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
            "best_validation_loss": float(validation_loss if best_validation_loss is None else best_validation_loss),
            "run_metadata": dict(run_metadata or {}),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "loader": loader_generator.get_state() if loader_generator is not None else None,
            },
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
    restore_rng: bool = False,
    loader_generator: torch.Generator | None = None,
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
    if restore_rng:
        states = checkpoint.get("rng_state")
        if states is None:
            raise ValueError("checkpoint has no RNG state; exact epoch-boundary resume is unavailable")
        random.setstate(states["python"])
        np.random.set_state(states["numpy"])
        torch.set_rng_state(states["torch"].cpu())
        if states["cuda"] is not None and torch.cuda.is_available():
            if len(states["cuda"]) != torch.cuda.device_count():
                raise ValueError("CUDA device count differs from checkpoint")
            torch.cuda.set_rng_state_all([state.cpu() for state in states["cuda"]])
        if loader_generator is not None:
            if states["loader"] is None:
                raise ValueError("checkpoint has no DataLoader generator state")
            loader_generator.set_state(states["loader"].cpu())
    return checkpoint
