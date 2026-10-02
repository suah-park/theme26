"""Stage-aware E1 training and validation loops."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

import torch
from torch import nn

from theme26.losses import masked_l1_loss

Stage = Literal["ami", "rfn"]
TensorBatch = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class EpochResult:
    """Aggregate scalar results from one pass through a loader."""

    loss: float
    steps: int


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


def _e1_loss(model: nn.Module, batch: TensorBatch, stage: Stage) -> torch.Tensor:
    prediction = _forward_stage(model, batch, stage)
    target = batch["target"]
    if prediction.shape != target.shape:
        raise ValueError(
            f"{stage} prediction/target mismatch: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    return masked_l1_loss(prediction, target, mask=batch.get("mask"))


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
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    amp_enabled = use_amp and device.type == "cuda"
    total_loss = 0.0
    steps = 0

    for batch_index, raw_batch in enumerate(loader):
        if max_steps is not None and batch_index >= max_steps:
            break
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
                loss = _e1_loss(model, batch, stage)

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

        total_loss += float(loss.detach())
        steps += 1

    if steps == 0:
        raise ValueError("loader produced no training or validation steps")
    return EpochResult(loss=total_loss / steps, steps=steps)


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
) -> EpochResult:
    """Train AMI or RFN for one E1 epoch using L1 reconstruction loss."""

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
    )


def validate_one_epoch(
    model: nn.Module,
    loader: Iterable[TensorBatch],
    *,
    stage: Stage,
    device: torch.device,
    use_amp: bool = True,
    max_steps: int | None = None,
) -> EpochResult:
    """Evaluate one E1 epoch without updating model state."""

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
    )
