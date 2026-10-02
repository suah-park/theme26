"""Command-line entry point for staged SAINT training."""

from __future__ import annotations

import argparse
import os
import random
import sys
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
from theme26.data import read_crop_manifest, validate_spatial_split
from theme26.data.protocol import load_protocol
from theme26.data.manifest import validate_protocol_records
from theme26.models.saint import AnisotropicMetaInterpolator, ResidualFusionNetwork
from theme26.training import (
    DigitalRockAmiDataset,
    ManifestAmiDataset,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)
from theme26.training.synthetic import SyntheticAmiDataset, SyntheticRfnDataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a theme26 SAINT experiment")
    parser.add_argument("--config", type=Path, required=True, help="Path to a versioned experiment config")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, help="Optional epoch override")
    parser.add_argument("--output-dir", type=Path, help="Optional artifact directory override")
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    for section in ("experiment", "model", "data", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"config requires a mapping section named {section!r}")
    return config


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


def _build_model(config: dict[str, Any], stage: str) -> torch.nn.Module:
    model_config = config["model"]
    shared = {
        "feature_channels": int(model_config.get("feature_channels", 64)),
        "kernel_size": int(model_config.get("kernel_size", 3)),
    }
    if stage == "ami":
        return AnisotropicMetaInterpolator(
            factor=int(model_config.get("factor", 4)),
            num_blocks=int(model_config.get("num_blocks", 6)),
            layers_per_block=int(model_config.get("layers_per_block", 8)),
            growth_rate=int(model_config.get("growth_rate", 32)),
            **shared,
        )
    if stage == "rfn":
        return ResidualFusionNetwork(
            num_blocks=int(model_config.get("num_blocks", 4)),
            layers_per_block=int(model_config.get("layers_per_block", 6)),
            growth_rate=int(model_config.get("growth_rate", 12)),
            **shared,
        )
    raise ValueError("experiment.stage must be 'ami' or 'rfn'")


def _build_loaders(
    config: dict[str, Any], stage: str, seed: int
) -> tuple[DataLoader, DataLoader]:
    data_config = config["data"]
    mode = data_config.get("mode")
    if mode == "berea_manifest":
        if stage != "ami":
            raise ValueError("data.mode='berea_manifest' supports only AMI training")
        path_env = str(data_config.get("path_env", "THEME26_GRAYSCALE_RAW"))
        raw_path = os.environ.get(path_env)
        if not raw_path:
            raise RuntimeError(f"set environment variable {path_env} to the grayscale .raw path")
        protocol_path = Path(data_config.get('protocol_dir', 'config'))
        if not protocol_path.is_absolute():
            protocol_path = REPOSITORY_ROOT / protocol_path
        protocol, normalization_config, evaluation = load_protocol(protocol_path)
        metadata_config = data_config["metadata"]
        metadata = VolumeMetadata(
            sample_id=str(metadata_config.get("sample_id", "berea")),
            source_core_id=str(metadata_config.get("source_core_id", "berea_2d25um")),
            shape_xyz=tuple(int(value) for value in metadata_config["shape_xyz"]),
            voxel_spacing_um_xyz=tuple(
                float(value) for value in metadata_config["spacing_um_xyz"]
            ),
            dtype=str(metadata_config.get("dtype", "uint8")),
            endianness=str(metadata_config.get("endianness", "not-applicable")),
            normalization_id=str(normalization_config["normalization_id"]),
            storage_order=str(metadata_config.get("storage_order", "XYZ")),
        )
        manifest_path = Path(data_config["manifest"])
        if not manifest_path.is_absolute():
            manifest_path = REPOSITORY_ROOT / manifest_path
        records = read_crop_manifest(manifest_path)
        validate_protocol_records(records, protocol, evaluation)
        if (metadata.shape_xyz != tuple(protocol['source_shape_xyz'])
            or metadata.voxel_spacing_um_xyz != tuple(protocol['voxel_spacing_um_xyz'])
            or metadata.storage_order != protocol['storage_order']):
            raise ValueError('training metadata differs from canonical XYZ protocol')
        validate_spatial_split(
            records,
            volume_shape_xyz=metadata.shape_xyz,
            minimum_buffer_voxels=int(data_config.get("minimum_buffer_voxels", 32)),
        )
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
            "data.mode must be 'synthetic', 'raw_ami_smoke', or 'berea_manifest'"
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
    epochs = args.epochs or int(training.get("epochs", 1))
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    output_dir = args.output_dir or Path(
        config.get("output_dir", f"runs/{experiment.get('id', 'e1')}")
    )
    if not output_dir.is_absolute():
        output_dir = REPOSITORY_ROOT / output_dir

    best_validation = float("inf")
    for epoch in range(1, epochs + 1):
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
        print(
            f"epoch={epoch:03d} stage={stage} device={device.type} "
            f"train_l1={train_result.loss:.6f} "
            f"validation_l1={validation_result.loss:.6f}"
        )
        checkpoint_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "scaler": scaler,
            "epoch": epoch,
            "validation_loss": validation_result.loss,
            "config": config,
        }
        save_checkpoint(output_dir / "last.pt", **checkpoint_kwargs)
        if validation_result.loss < best_validation:
            best_validation = validation_result.loss
            save_checkpoint(output_dir / "best.pt", **checkpoint_kwargs)

    print(f"best_validation_l1={best_validation:.6f}")
    print(f"artifacts={output_dir}")


if __name__ == "__main__":
    main()
