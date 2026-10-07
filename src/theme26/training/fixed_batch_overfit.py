"""Fixed train-marginal AMI optimization diagnostic; no held-out evaluation."""

from __future__ import annotations

import os
import struct
import time
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from theme26.losses import masked_l1_loss
from theme26.training.digital_rock import ManifestAmiDataset
from theme26.training.engine import train_one_epoch


def select_train_marginal(dataset: ManifestAmiDataset, *, sample_id: str,
                          direction: str, plane_index: int) -> dict[str, torch.Tensor]:
    """Select exactly one registered train marginal, with a batch dimension."""
    if direction not in ("sagittal", "coronal"):
        raise ValueError("direction must be sagittal or coronal")
    matches = [record for record in dataset.records if record.sample_id == sample_id]
    if len(matches) != 1 or matches[0].split != "train":
        raise ValueError("sample_id must identify exactly one train record")
    record = matches[0]
    extent = record.crop.shape_xyz[0 if direction == "sagittal" else 1]
    if not 0 <= plane_index < extent:
        raise ValueError(f"plane_index must be in [0,{extent})")
    entry = (dataset.records.index(record), direction, plane_index)
    return {key: value.unsqueeze(0).clone() for key, value in
            dataset[dataset._entries.index(entry)].items()}


def _write_png(path: Path, pixels: np.ndarray) -> None:
    """Write a small RGB comparison without an optional plotting dependency."""
    height, width, channels = pixels.shape
    if channels != 3 or pixels.dtype != np.uint8:
        raise ValueError("PNG pixels must be uint8 RGB")
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + kind + payload +
                struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))
    scanlines = b"".join(b"\0" + pixels[row].tobytes() for row in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" +
                     chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


def _save_comparison(path: Path, before: np.ndarray, after: np.ndarray,
                     target: np.ndarray, mask: np.ndarray) -> None:
    # 2x2: before | after / target | primary mask. Clip predictions for display only.
    def rgb(array: np.ndarray) -> np.ndarray:
        gray = np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)
    top = np.concatenate((rgb(before), rgb(after)), axis=1)
    bottom = np.concatenate((rgb(target), rgb(mask.astype(np.float32))), axis=1)
    _write_png(path, np.concatenate((top, bottom), axis=0))


def run_fixed_batch_overfit(model: torch.nn.Module, batch: dict[str, torch.Tensor],
                            *, config: dict[str, Any], output_dir: Path,
                            device: torch.device, metadata: dict[str, Any]) -> None:
    """Repeatedly optimize one unchanged AMI batch and save paired diagnostics."""
    training = config["training"]
    steps = int(training["optimizer_steps"])
    interval = int(training["log_interval_steps"])
    if steps < 1 or interval < 1:
        raise ValueError("optimizer_steps and log_interval_steps must be positive")
    if training.get("loss_region", "all") != "all":
        raise ValueError("fixed-batch diagnostic requires all-output L1")
    if int(training.get("batch_size", 1)) != 1:
        raise ValueError("fixed-batch diagnostic requires batch_size: 1")
    if any(value.shape[0] != 1 for value in batch.values()):
        raise ValueError("every batch tensor must have batch size one")
    fixed = {key: value.detach().clone().to(device) for key, value in batch.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]),
                                 weight_decay=float(training.get("weight_decay", 0)))
    elapsed_start = time.perf_counter()
    history: list[dict[str, float | int]] = []
    (output_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False),
                                                encoding="utf-8")

    def snapshot(step: int) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            prediction = model(fixed["input"], fixed["fdm"])
            loss = masked_l1_loss(prediction, fixed["target"])
        elapsed = time.perf_counter() - elapsed_start
        item = {"optimizer_step": step, "raw_all_output_l1": float(loss),
                "elapsed_seconds": elapsed}
        history.append(item)
        print(f"fixed_train step={step:03d}/{steps:03d} "
              f"raw_all_output_l1={item['raw_all_output_l1']:.6f} "
              f"elapsed_seconds={elapsed:.1f}", flush=True)
        temporary = output_dir / "history.yaml.tmp"
        temporary.write_text(yaml.safe_dump(history, sort_keys=False), encoding="utf-8")
        os.replace(temporary, output_dir / "history.yaml")
        return prediction.detach().cpu().numpy()[0, 0].copy()

    before = snapshot(0)
    np.save(output_dir / "input.npy", fixed["input"].cpu().numpy()[0])
    np.save(output_dir / "target.npy", fixed["target"].cpu().numpy()[0, 0])
    np.save(output_dir / "primary_mask.npy", fixed["primary_mask"].cpu().numpy()[0, 0])
    np.save(output_dir / "prediction_before_raw.npy", before)
    for step in range(1, steps + 1):
        train_one_epoch(model, [fixed], optimizer, stage="ami", device=device,
                        use_amp=bool(training.get("mixed_precision", False)),
                        gradient_clip_norm=training.get("gradient_clip_norm"))
        if step % interval == 0 or step == steps:
            after = snapshot(step)
    np.save(output_dir / "prediction_after_raw.npy", after)
    _save_comparison(output_dir / "comparison.png", before, after,
                     fixed["target"].cpu().numpy()[0, 0],
                     fixed["primary_mask"].cpu().numpy()[0, 0])
    metadata["initial_raw_all_output_l1"] = history[0]["raw_all_output_l1"]
    metadata["final_raw_all_output_l1"] = history[-1]["raw_all_output_l1"]
    metadata["comparison_png_layout"] = "before | after / target | primary mask"
    (output_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False),
                                                encoding="utf-8")
    temporary = output_dir / "checkpoint.pt.tmp"
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "config": config, "metadata": metadata, "optimizer_step": steps}, temporary)
    os.replace(temporary, output_dir / "checkpoint.pt")
    print(f"fixed_train_l1_change={history[0]['raw_all_output_l1']:.6f} -> "
          f"{history[-1]['raw_all_output_l1']:.6f} artifacts={output_dir}", flush=True)
