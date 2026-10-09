"""Stage-aware E1 and E2 training and validation loops."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
import math
from typing import Callable, Iterable, Literal, Mapping

import torch
from torch import nn

from theme26.losses import gradient_loss_2d, masked_l1_loss

Stage = Literal["ami", "rfn"]
TensorBatch = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class EpochResult:
    """Aggregate scalar results from one pass through a loader."""

    loss: float
    steps: int
    primary_mae: float | None = None
    primary_voxels: int = 0
    l1_loss: float = 0.0
    gradient_loss: float = 0.0


def _move_batch(batch: TensorBatch, device: torch.device) -> dict[str, torch.Tensor]:
    required = {"input", "target"}
    missing = required - set(batch)
    if missing:
        raise KeyError(f"batch is missing required keys: {sorted(missing)}")
    return {
        name: tensor.to(device=device, non_blocking=True)
        for name, tensor in batch.items()
    }


def _forward_stage(model: nn.Module, batch: TensorBatch, stage: Stage) -> torch.Tensor:
    if stage == "ami":
        if "fdm" not in batch:
            raise KeyError("AMI batch requires an 'fdm' tensor")
        return model(batch["input"], batch["fdm"])
    if stage == "rfn":
        forward_slices = getattr(model, "forward_slices", None)
        if forward_slices is None:
            raise TypeError("RFN stage model must expose forward_slices(inputs)")
        return forward_slices(batch["input"])
    raise ValueError(f"unsupported E1 stage: {stage!r}")


def _run_epoch(
    model: nn.Module,
    loader: Iterable[TensorBatch],
    *,
    stage: Stage,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: object | None,
    use_amp: bool,
    gradient_clip_norm: float | None,
    max_steps: int | None,
    progress_callback: Callable[[int, int], None] | None,
    gradient_weight: float,
) -> EpochResult:
    if not math.isfinite(gradient_weight) or gradient_weight < 0:
        raise ValueError("gradient_weight must be finite and non-negative")
    training = optimizer is not None
    model.train(training)
    amp_enabled = use_amp and device.type == "cuda"
    total_loss = 0.0
    total_l1 = 0.0
    total_gradient = 0.0
    loss_voxels = 0
    primary_error = 0.0
    primary_voxels = 0
    steps = 0
    samples = 0

    selected_loader = islice(loader, max_steps) if max_steps is not None else loader
    for batch_index, raw_batch in enumerate(selected_loader):
        batch = _move_batch(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        grad_context = nullcontext() if training else torch.no_grad()
        with grad_context:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                prediction = _forward_stage(model, batch, stage)
                # E2 uses the entire raw output for both terms. Keep the
                # optional E1 mask behavior when gradient_weight is zero.
                l1 = masked_l1_loss(prediction, batch["target"],
                                    mask=batch.get("mask") if gradient_weight == 0 else None)
                gradient = (gradient_loss_2d(prediction, batch["target"])
                            if gradient_weight else None)
                loss = l1 + gradient_weight * gradient if gradient is not None else l1

        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {stage} loss at step {batch_index}")

        if training:
            scaler_enabled = bool(
                scaler is not None
                and getattr(scaler, "is_enabled", lambda: False)()
            )
            if scaler_enabled:
                scaler.scale(loss).backward()
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()

        count = (prediction.numel() if gradient_weight or "mask" not in batch else
                 int(torch.broadcast_to(batch["mask"], prediction.shape).sum()))
        total_loss += float(loss.detach()) * count
        total_l1 += float(l1.detach()) * count
        if gradient is not None:
            total_gradient += float(gradient.detach()) * count
        loss_voxels += count
        if "primary_mask" in batch:
            metric_mask = torch.broadcast_to(batch["primary_mask"], prediction.shape)
            if metric_mask.dtype != torch.bool:
                raise TypeError("primary_mask must be bool")
            metric_error = (prediction.detach().float().clamp(0, 1) - batch["target"].float()).abs()
            primary_error += float(metric_error.masked_select(metric_mask).sum())
            primary_voxels += int(metric_mask.sum())
        steps += 1
        samples += int(batch["input"].shape[0])
        if progress_callback is not None:
            progress_callback(steps, samples)

    if steps == 0:
        raise ValueError("loader produced no training or validation steps")
    return EpochResult(loss=total_loss / loss_voxels, steps=steps,
                       primary_mae=primary_error / primary_voxels if primary_voxels else None,
                       primary_voxels=primary_voxels,
                       l1_loss=total_l1 / loss_voxels,
                       gradient_loss=total_gradient / loss_voxels)


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[TensorBatch],
    optimizer: torch.optim.Optimizer,
    *,
    stage: Stage,
    device: torch.device,
    scaler: object | None = None,
    use_amp: bool = True,
    gradient_clip_norm: float | None = None,
    max_steps: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    gradient_weight: float = 0.0,
) -> EpochResult:
    """Train AMI or RFN with E1 L1 or E2 L1 plus image-gradient loss."""

    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive or None")
    return _run_epoch(
        model,
        loader,
        stage=stage,
        device=device,
        optimizer=optimizer,
        scaler=scaler,
        use_amp=use_amp,
        gradient_clip_norm=gradient_clip_norm,
        max_steps=max_steps,
        progress_callback=progress_callback,
        gradient_weight=gradient_weight,
    )


def validate_one_epoch(
    model: nn.Module,
    loader: Iterable[TensorBatch],
    *,
    stage: Stage,
    device: torch.device,
    use_amp: bool = True,
    max_steps: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    gradient_weight: float = 0.0,
) -> EpochResult:
    """Evaluate one epoch without updating model state."""

    return _run_epoch(
        model,
        loader,
        stage=stage,
        device=device,
        optimizer=None,
        scaler=None,
        use_amp=use_amp,
        gradient_clip_norm=None,
        max_steps=max_steps,
        progress_callback=progress_callback,
        gradient_weight=gradient_weight,
    )
