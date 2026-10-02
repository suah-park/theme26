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


def build_stage_model(config: dict, stage: str):
    values = dict(config["model"])
    if "settings" in values:
        profiles = yaml.safe_load(resolve_path(values["settings"]).read_text(encoding="utf-8"))
        profile = values["profile"]
        if not profile.startswith(stage + "_"):
            raise ValueError("model profile does not match stage")
        values = dict(profiles["profiles"][profile])
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
        sample_id=values["sample_id"], source_core_id=values["source_core_id"],
        shape_xyz=tuple(values["shape_xyz"]), voxel_spacing_um_xyz=tuple(values["spacing_um_xyz"]),
        dtype=values["dtype"], endianness=values["endianness"],
        normalization_id=normalization["normalization_id"], storage_order=values["storage_order"],
    )
    if (metadata.shape_xyz != tuple(protocol["source_shape_xyz"])
        or metadata.voxel_spacing_um_xyz != tuple(protocol["voxel_spacing_um_xyz"])
        or metadata.storage_order != protocol["storage_order"]):
        raise ValueError("metadata differs from canonical protocol")
    validate_spatial_split(records, volume_shape_xyz=metadata.shape_xyz,
                           minimum_buffer_voxels=int(data.get("minimum_buffer_voxels", 32)))
    raw = os.environ.get(data.get("path_env", "THEME26_GRAYSCALE_RAW"))
    if not raw:
        raise ValueError("set the configured grayscale RAW environment variable")
    source_id = config_hash({"protocol": protocol, "normalization": normalization,
                             "evaluation": evaluation,
                             "manifest_sha256": file_hash(resolve_path(data["manifest"]))})
    return Path(raw), metadata, records, normalization, source_id
