"""Shared configuration, source identity and checkpoint model construction."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

from theme26.contracts import VolumeMetadata
from theme26.data import read_crop_manifest, validate_spatial_split
from theme26.data.manifest import validate_protocol_records
from theme26.data.protocol import load_protocol
from theme26.models.saint import AnisotropicMetaInterpolator, ResidualFusionNetwork

ROOT = Path(__file__).resolve().parents[3]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(value: dict) -> str:
    return hashlib.sha256(yaml.safe_dump(value, sort_keys=True).encode()).hexdigest()


def resolve_model_config(config: dict, stage: str) -> dict:
    if stage not in ("ami", "rfn"):
        raise ValueError("stage must be ami or rfn")
    values = dict(config["model"])
    if "settings" in values:
        if set(values) != {"settings", "profile"}:
            raise ValueError("shared model settings require exactly settings and profile; overrides are not allowed")
        profiles = yaml.safe_load(resolve_path(values["settings"]).read_text(encoding="utf-8"))
        profile = values["profile"]
        if not isinstance(profile, str) or not profile.startswith(stage + "_"):
            raise ValueError("model profile does not match stage")
        if not isinstance(profiles, dict) or profile not in profiles.get("profiles", {}):
            raise ValueError(f"unknown model profile: {profile}")
        values = dict(profiles["profiles"][profile])
    return values


def load_experiment_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config root must be a mapping")
    for section in ("experiment", "model", "data", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"config requires a mapping section named {section!r}")
    if "settings" in config["model"]:
        reference = dict(config["model"])
        config["model"] = resolve_model_config(config, config["experiment"].get("stage"))
        config["model_reference"] = reference
    return config


def build_stage_model(config: dict, stage: str):
    values = resolve_model_config(config, stage)
    if stage == "ami":
        return AnisotropicMetaInterpolator(**values)
    if stage == "rfn":
        return ResidualFusionNetwork(**values)
    raise ValueError("stage must be ami or rfn")


def read_berea_context(data: dict):
    protocol, normalization, evaluation = load_protocol(resolve_path(data.get("protocol_dir", "configs/data")))
    records = read_crop_manifest(resolve_path(data["manifest"]))
    validate_protocol_records(records, protocol, evaluation)
    values = data["metadata"]
    metadata = VolumeMetadata(
        sample_id=str(values.get("sample_id", "berea")), source_core_id=str(values.get("source_core_id", "berea_2d25um")),
        shape_xyz=tuple(int(v) for v in values["shape_xyz"]), voxel_spacing_um_xyz=tuple(float(v) for v in values["spacing_um_xyz"]),
        dtype=str(values.get("dtype", "uint8")), endianness=str(values.get("endianness", "not-applicable")),
        normalization_id=normalization["normalization_id"], storage_order=str(values.get("storage_order", "XYZ")),
    )
    if (metadata.shape_xyz != tuple(protocol["source_shape_xyz"])
        or metadata.voxel_spacing_um_xyz != tuple(protocol["voxel_spacing_um_xyz"])
        or metadata.storage_order != protocol["storage_order"]):
        raise ValueError("metadata differs from canonical protocol")
    validate_spatial_split(records, volume_shape_xyz=metadata.shape_xyz,
                           minimum_buffer_voxels=int(data.get("minimum_buffer_voxels", 32)))
    raw = os.environ.get(data.get("path_env", "THEME26_GRAYSCALE_RAW"))
    if not raw:
        raise RuntimeError(f"set environment variable {data.get('path_env', 'THEME26_GRAYSCALE_RAW')} to the grayscale .raw path")
    source_id = config_hash({"protocol": protocol, "normalization": normalization,
                             "evaluation": evaluation,
                             "manifest_sha256": file_hash(resolve_path(data["manifest"]))})
    return Path(raw), metadata, records, normalization, source_id
