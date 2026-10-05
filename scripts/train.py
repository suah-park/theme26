"""Command-line entry point for staged SAINT training."""

from __future__ import annotations

import argparse
import os
import random
import sys
import copy
import subprocess
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from theme26.contracts import CropSpec, SamplingSpec, VolumeMetadata
from theme26.training import (
    DigitalRockAmiDataset,
    ManifestAmiDataset,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
    load_checkpoint,
)
from theme26.training.directional_cache import CachedRfnDataset, validate_cache
from theme26.training.workflow import (
    resolve_path, file_hash, read_berea_context,
    load_experiment_config as _load_config,
    build_stage_model as _build_model,
)
from theme26.training.synthetic import SyntheticAmiDataset, SyntheticRfnDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a theme26 SAINT experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to a versioned experiment config")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, help="Optional epoch override")
    parser.add_argument("--output-dir", type=Path, help="Optional artifact directory override")
    parser.add_argument("--resume", type=Path, help="Resume at the next epoch from last.pt")
    return parser


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def _build_loaders(
    config: dict[str, Any], stage: str, seed: int
) -> tuple[DataLoader, DataLoader]:
    data_config = config["data"]
    mode = data_config.get("mode")
    if mode == "berea_rfn_cache":
        if stage != "rfn":
            raise ValueError("berea_rfn_cache requires RFN stage")
        raw, records, normalization, entries = validate_cache(
            data_config["source"], resolve_path(data_config["cache_dir"]),
            resolve_path(data_config["ami_checkpoint"]),
        )
        train_dataset = CachedRfnDataset(raw, records, normalization, entries,
                                        resolve_path(data_config["cache_dir"]), split="train")
        validation_dataset = CachedRfnDataset(raw, records, normalization, entries,
                                             resolve_path(data_config["cache_dir"]), split="validation")
    elif mode == "berea_manifest":
        if stage != "ami":
            raise ValueError("data.mode='berea_manifest' supports only AMI training")
        raw_path, metadata, records, normalization_config, _ = read_berea_context(data_config)
        shared = {
            "path": raw_path,
            "metadata": metadata,
            "normalization_lower": float(normalization_config["lower"]),
            "normalization_upper": float(normalization_config["upper"]),
            "kernel_size": int(config["model"].get("kernel_size", 3)),
        }
        train_dataset = ManifestAmiDataset.from_raw(
            records=tuple(record for record in records if record.split == "train"),
            **shared,
        )
        validation_dataset = ManifestAmiDataset.from_raw(
            records=tuple(
                record for record in records if record.split == "validation"
            ),
            **shared,
        )
    elif mode == "raw_ami_smoke":
        if stage != "ami":
            raise ValueError("data.mode='raw_ami_smoke' supports only experiment.stage='ami'")
        path_env = str(data_config.get("path_env", "THEME26_GRAYSCALE_RAW"))
        raw_path = os.environ.get(path_env)
        if not raw_path:
            raise RuntimeError(f"set environment variable {path_env} to the grayscale .raw path")
        metadata_config = data_config["metadata"]
        metadata = VolumeMetadata(
            sample_id=str(metadata_config.get("sample_id", "berea")),
            source_core_id=str(metadata_config.get("source_core_id", "berea")),
            shape_xyz=tuple(int(value) for value in metadata_config["shape_xyz"]),
            voxel_spacing_um_xyz=tuple(
                float(value) for value in metadata_config["spacing_um_xyz"]
            ),
            dtype=str(metadata_config.get("dtype", "uint8")),
            endianness=str(metadata_config.get("endianness", "not-applicable")),
            normalization_id=str(
                data_config.get("normalization_id", "smoke_uint8_0_255")
            ),
            storage_order=str(metadata_config.get("storage_order", "XYZ")),
        )
        crop = CropSpec(
            origin_xyz=tuple(int(value) for value in data_config["crop_origin_xyz"]),
            shape_xyz=tuple(int(value) for value in data_config["crop_shape_xyz"]),
        )
        normalization = data_config.get("normalization", {"lower": 0, "upper": 255})
        dataset = DigitalRockAmiDataset.from_raw(
            raw_path,
            metadata=metadata,
            crop=crop,
            normalization_lower=float(normalization["lower"]),
            normalization_upper=float(normalization["upper"]),
            sampling=SamplingSpec(factor=int(config["model"].get("factor", 4))),
            kernel_size=int(config["model"].get("kernel_size", 3)),
        )
        requested = min(int(data_config.get("max_marginals", 8)), len(dataset))
        if requested < 2:
            raise ValueError("raw AMI smoke test requires at least two marginal samples")
        indices = torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(seed)
        )[:requested]
        split = max(1, requested - max(1, requested // 4))
        train_dataset = Subset(dataset, indices[:split].tolist())
        validation_dataset = Subset(dataset, indices[split:].tolist())
    elif mode == "synthetic" and stage == "ami":
        kwargs = {
            "length": int(data_config.get("length", 16)),
            "z_lr": int(data_config.get("z_lr", 8)),
            "factor": int(config["model"].get("factor", 4)),
            "spacing_um_xyz": tuple(
                float(value)
                for value in data_config.get("spacing_um_xyz", [2.25, 2.25, 2.25])
            ),
        }
        train_dataset = SyntheticAmiDataset(
            samples=int(data_config.get("train_samples", 8)), seed=seed, **kwargs
        )
        validation_dataset = SyntheticAmiDataset(
            samples=int(data_config.get("validation_samples", 4)),
            seed=seed + 10_000,
            **kwargs,
        )
    elif mode == "synthetic" and stage == "rfn":
        kwargs = {
            "x_size": int(data_config.get("x_size", 16)),
            "y_size": int(data_config.get("y_size", 16)),
        }
        train_dataset = SyntheticRfnDataset(
            samples=int(data_config.get("train_samples", 8)), seed=seed, **kwargs
        )
        validation_dataset = SyntheticRfnDataset(
            samples=int(data_config.get("validation_samples", 4)),
            seed=seed + 10_000,
            **kwargs,
        )
    else:
        raise ValueError(
            "data.mode must be synthetic, raw_ami_smoke, berea_manifest, or berea_rfn_cache"
        )

    batch_size = int(config["training"].get("batch_size", 1))
    num_workers = int(config["training"].get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("training.num_workers must be non-negative")
    pin_memory = bool(config["training"].get("pin_memory", False))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, validation_loader


def _make_scaler(enabled: bool) -> object | None:
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=True)


def main() -> None:
    args = build_parser().parse_args()
    config = _load_config(args.config)
    experiment = config["experiment"]
    training = config["training"]
    stage = str(experiment.get("stage", "ami"))
    if training.get("loss_region", "all") != "all":
        raise ValueError("E1 protocol uses all-output training L1")
    if training.get("checkpoint_metric", "primary_mae") != "primary_mae":
        raise ValueError("E1 best checkpoint must use clipped bracketed-missing MAE")
    if training.get("epochs") is None and args.epochs is None:
        raise ValueError("Set epochs after the GPU pilot; this config is an unfilled template")
    if training.get("batch_size", 1) is None:
        raise ValueError("Set batch_size after the GPU pilot")
    seed = int(experiment.get("seed", 0))
    _set_seed(seed)
    device = _resolve_device(args.device)

    model = _build_model(config, stage).to(device)
    train_loader, validation_loader = _build_loaders(config, stage, seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    use_amp = bool(training.get("mixed_precision", True))
    scaler = _make_scaler(use_amp and device.type == "cuda")
    epochs = args.epochs if args.epochs is not None else int(training["epochs"])
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    output_dir = args.output_dir or Path(
        config.get("output_dir", f"runs/{experiment.get('id', 'e1')}")
    )
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir

    best_validation = float("inf")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((output_dir / name).exists() for name in ("last.pt", "best.pt", "history.yaml")):
        raise FileExistsError("run directory already has results; use --resume or a new output directory")
    def compatible_config(value):
        value = copy.deepcopy(value)
        value.pop("output_dir", None)
        value["training"].pop("epochs", None)
        return value
    git_commit = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
                                             stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    metadata = {"git_commit": git_commit, "seed": seed, "device": str(device),
                "torch_version": str(torch.__version__), "config_file_sha256": file_hash(args.config),
                "checkpoint_metric": "primary_mae", "metric_range_policy": "clip_0_1"}
    source_data = config["data"].get("source", config["data"])
    if source_data.get("mode") == "berea_manifest":
        raw_path, _, _, _, protocol_id = read_berea_context(source_data)
        metadata.update(source_raw_sha256=file_hash(raw_path), protocol_sha256=protocol_id)
    start_epoch = 1
    history = []
    if args.resume:
        if args.resume.resolve().parent != output_dir.resolve() or args.resume.name != "last.pt":
            raise ValueError("resume requires this run directory's last.pt to preserve best checkpoint and history")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if compatible_config(checkpoint["config"]) != compatible_config(config):
            raise ValueError("resume config differs from checkpoint (only epoch budget/output directory may change)")
        for key in ("checkpoint_metric", "metric_range_policy", "source_raw_sha256", "protocol_sha256"):
            if checkpoint.get("run_metadata", {}).get(key) != metadata.get(key):
                raise ValueError(f"resume identity mismatch: {key}")
        checkpoint = load_checkpoint(args.resume, model=model, optimizer=optimizer, scaler=scaler,
                                     restore_rng=True, loader_generator=train_loader.generator)
        start_epoch = checkpoint["epoch"] + 1
        best_validation = checkpoint["best_validation_loss"]
        history = checkpoint.get("run_metadata", {}).get("history", [])
    if start_epoch > epochs:
        raise ValueError("epoch budget must exceed the saved epoch; --epochs is the total budget")
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            stage=stage,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            gradient_clip_norm=training.get("gradient_clip_norm"),
            max_steps=training.get("max_steps_per_epoch"),
        )
        validation_result = validate_one_epoch(
            model,
            validation_loader,
            stage=stage,
            device=device,
            use_amp=use_amp,
            max_steps=training.get("max_validation_steps"),
        )
        if validation_result.primary_mae is None:
            raise ValueError("validation has no bracketed missing voxels; increase pilot validation steps")
        selected_metric = validation_result.primary_mae
        print(
            f"epoch={epoch:03d} stage={stage} device={device.type} "
            f"train_l1={train_result.loss:.6f} "
            f"validation_l1={validation_result.loss:.6f} "
            f"validation_primary_mae={selected_metric:.6f}"
        )
        improved = selected_metric < best_validation
        best_validation = min(best_validation, selected_metric)
        history.append({"epoch": epoch, "train_l1": train_result.loss,
                        "validation_l1": validation_result.loss, "validation_primary_mae": selected_metric,
                        "primary_voxels": validation_result.primary_voxels,
                        "train_steps": train_result.steps, "validation_steps": validation_result.steps,
                        "elapsed_seconds": time.perf_counter() - started,
                        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
        checkpoint_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "scaler": scaler,
            "epoch": epoch,
            "validation_loss": selected_metric,
            "config": config,
            "best_validation_loss": best_validation,
            "loader_generator": train_loader.generator,
            "run_metadata": dict(metadata, history=history),
        }
        save_checkpoint(output_dir / "last.pt", **checkpoint_kwargs)
        if improved:
            save_checkpoint(output_dir / "best.pt", **checkpoint_kwargs)
        temporary_history = output_dir / "history.yaml.tmp"
        temporary_history.write_text(yaml.safe_dump(history, sort_keys=False), encoding="utf-8")
        os.replace(temporary_history, output_dir / "history.yaml")

    print(f"best_validation_primary_mae={best_validation:.6f}")
    print(f"artifacts={output_dir}")


if __name__ == "__main__":
    main()
