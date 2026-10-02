"""Versioned, raw float32 AMI outputs for staged RFN training."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from theme26.data import extract_volume_crop, make_sparse_sample, normalize_grayscale, read_raw_volume
from theme26.models.saint.inference import reconstruct_directions
from .workflow import config_hash, file_hash, read_berea_context, resolve_path


def _record_identity(record) -> dict:
    return {"crop_id": record.sample_id, "split": record.split,
            "origin_xyz": list(record.crop.origin_xyz), "shape_xyz": list(record.crop.shape_xyz),
            "factor": record.sampling.factor, "offset": record.sampling.offset,
            "source_core_id": record.source_core_id}


def generate_cache(ami, ami_checkpoint: Path, data: dict, destination: Path, *,
                   device: torch.device, batch_size: int, use_amp: bool) -> Path:
    """Generate train/validation only; refuse existing cache directories."""
    raw_path, metadata, records, normalization, protocol_id = read_berea_context(data)
    if destination.exists():
        raise FileExistsError("cache destination exists; choose a new directory to avoid mixing AMI runs")
    raw = read_raw_volume(raw_path, metadata=metadata)
    source_hash = file_hash(raw_path)
    checkpoint = torch.load(ami_checkpoint, map_location="cpu", weights_only=False)
    provenance = checkpoint.get("run_metadata", {})
    if (checkpoint["config"]["experiment"]["stage"] != "ami"
        or checkpoint["config"]["data"] != data
        or provenance.get("source_raw_sha256") != source_hash
        or provenance.get("protocol_sha256") != protocol_id):
        raise ValueError("AMI checkpoint source identity differs from cache source")
    destination.mkdir(parents=True)
    entries = []
    for record in records:
        if record.split not in ("train", "validation"):
            continue
        dense = normalize_grayscale(extract_volume_crop(raw, crop=record.crop),
                                    lower=normalization["lower"], upper=normalization["upper"])
        sample = make_sparse_sample(dense, sampling=record.sampling)
        sagittal, coronal = reconstruct_directions(
            ami, sample.sparse_volume, sampling=record.sampling,
            spacing_um_xyz=metadata.voxel_spacing_um_xyz,
            batch_size=batch_size, device=device, use_amp=use_amp,
        )
        entry = _record_identity(record)
        # Integer prefixes avoid trusting crop IDs as filesystem paths.
        for axis, array in (("sagittal", sagittal), ("coronal", coronal)):
            name = f"{len(entries):03d}_{axis}.npy"
            np.save(destination / name, array, allow_pickle=False)
            entry[axis] = {"path": name, "sha256": file_hash(destination / name)}
        entries.append(entry)
        print(f"cache crop={record.sample_id} split={record.split}", flush=True)
    manifest = {"format": "theme26-ami-cache-v1", "complete": True,
                "axis_order": "XYZ", "dtype": "float32", "range_policy": "raw",
                "ami_checkpoint_sha256": file_hash(ami_checkpoint),
                "source_raw_sha256": source_hash, "protocol_sha256": protocol_id,
                "data_config_sha256": config_hash(data), "records": entries}
    # No complete manifest is written if generation fails part way through.
    path = destination / "manifest.yaml"
    temporary = destination / "manifest.yaml.tmp"
    temporary.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)
    return path


def validate_cache(data: dict, directory: Path, ami_checkpoint: Path):
    raw_path, metadata, records, normalization, protocol_id = read_berea_context(data)
    manifest = yaml.safe_load((directory / "manifest.yaml").read_text(encoding="utf-8"))
    expected = {"format": "theme26-ami-cache-v1", "complete": True, "axis_order": "XYZ",
                "dtype": "float32", "range_policy": "raw",
                "ami_checkpoint_sha256": file_hash(ami_checkpoint),
                "source_raw_sha256": file_hash(raw_path), "protocol_sha256": protocol_id,
                "data_config_sha256": config_hash(data)}
    if not isinstance(manifest, dict) or any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError("cache identity differs from AMI checkpoint, RAW, or data protocol")
    wanted = [record for record in records if record.split in ("train", "validation")]
    entries = manifest.get("records", [])
    if len(entries) != len(wanted):
        raise ValueError("cache must contain all train/validation crops and no test crops")
    for entry, record in zip(entries, wanted):
        if any(entry.get(k) != v for k, v in _record_identity(record).items()):
            raise ValueError("cache crop identity differs from manifest")
        for axis in ("sagittal", "coronal"):
            path = (directory / entry[axis]["path"]).resolve()
            if path.parent != directory.resolve():
                raise ValueError("cache file must stay inside cache directory")
            if file_hash(path) != entry[axis]["sha256"]:
                raise ValueError("cache array hash mismatch")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.float32 or array.shape != record.crop.shape_xyz:
                raise ValueError("cache array shape or dtype mismatch")
            if not np.isfinite(array).all():
                raise ValueError("cache array contains NaN/Inf")
    raw = read_raw_volume(raw_path, metadata=metadata)
    return raw, records, normalization, entries


class CachedRfnDataset(Dataset):
    """Raw directional [2,X,Y] → dense target [1,X,Y], per crop and Z.

Training includes every output phase. Validation samples bracketed missing
axial slices only, with the same voxel weights as the full-volume metric.
"""
    def __init__(self, raw, records, normalization, entries, directory: Path, *, split: str):
        self.raw = raw
        self.normalization = normalization
        self.entries = []
        self.arrays = {}
        record_by_id = {record.sample_id: record for record in records}
        for entry in entries:
            record = record_by_id[entry["crop_id"]]
            if record.split != split:
                continue
            self.arrays[record.sample_id] = tuple(np.load(directory / entry[axis]["path"],
                                                        mmap_mode="r", allow_pickle=False)
                                               for axis in ("sagittal", "coronal"))
            primary = set(record.sampling.bracketed_missing_indices(record.crop.shape_xyz[2]))
            z_indices = range(record.crop.shape_xyz[2]) if split == "train" else sorted(primary)
            self.entries.extend((record, z, z in primary) for z in z_indices)
        if not self.entries:
            raise ValueError(f"no {split} RFN samples")

    def __len__(self):
        return len(self.entries)

    def close(self):
        """Release directional memory maps; the shared RAW map belongs to caller."""
        for arrays in self.arrays.values():
            for array in arrays:
                array._mmap.close()
        self.arrays.clear()

    def __getitem__(self, index):
        record, z, primary = self.entries[index]
        sx, sy, _ = record.crop.shape_xyz
        ox, oy, oz = record.crop.origin_xyz
        sagittal, coronal = self.arrays[record.sample_id]
        inputs = np.stack((sagittal[..., z], coronal[..., z]))
        target = normalize_grayscale(self.raw[ox:ox + sx, oy:oy + sy, oz + z][None],
                                     lower=self.normalization["lower"], upper=self.normalization["upper"])
        return {"input": torch.from_numpy(inputs), "target": torch.from_numpy(target),
                "primary_mask": torch.full(target.shape, primary, dtype=torch.bool)}
