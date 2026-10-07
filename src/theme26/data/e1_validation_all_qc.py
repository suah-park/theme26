"""Compare E0, raw-direction AMI average, and RFN on registered validation crops."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from theme26.data import (compose_full_volume, extract_volume_crop,
                          linear_interpolate_z, make_sparse_sample,
                          normalize_grayscale, read_raw_volume)
from theme26.training.workflow import config_hash, file_hash, read_berea_context


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / reference


def _metrics(error_sums: dict[str, float], voxels: int) -> dict[str, float]:
    e0 = error_sums["e0"] / voxels
    ami = error_sums["ami_average"] / voxels
    rfn = error_sums["e1_rfn"] / voxels
    return {
        "e0_primary_mae": e0,
        "ami_average_primary_mae": ami,
        "e1_rfn_primary_mae": rfn,
        "ami_improvement_vs_e0_percent": _improvement(e0, ami),
        "rfn_improvement_vs_e0_percent": _improvement(e0, rfn),
        "rfn_minus_ami_primary_mae": rfn - ami,
        "rfn_improvement_vs_ami_percent": _improvement(ami, rfn),
    }


def _cache_array(cache_dir: Path, item: dict, shape: tuple[int, int, int]) -> np.ndarray:
    path = (cache_dir / item["path"]).resolve()
    _require(path.parent == cache_dir.resolve(), "cache array path escapes cache directory")
    _require(file_hash(path) == item["sha256"], f"cache array SHA256 mismatch: {path.name}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    _require(array.shape == shape and array.dtype == np.float32,
             f"cache array shape/dtype mismatch: {path.name}")
    _require(bool(np.isfinite(array).all()), f"cache array has NaN/Inf: {path.name}")
    return array


def _preflight(result_dir: Path, cache_dir: Path, checkpoint_path: Path, raw_argument: Path):
    output_manifest = yaml.safe_load((result_dir / "manifest.yaml").read_text(encoding="utf-8"))
    cache_manifest = yaml.safe_load((cache_dir / "manifest.yaml").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(checkpoint["config"]["experiment"]["stage"] == "ami", "checkpoint is not AMI")
    source_config = checkpoint["config"]["data"]
    os.environ[source_config.get("path_env", "THEME26_GRAYSCALE_RAW")] = str(raw_argument.resolve())
    raw_path, volume_metadata, records, normalization, protocol_sha = read_berea_context(source_config)
    raw_sha = file_hash(raw_path)
    checkpoint_sha = file_hash(checkpoint_path)
    _require(checkpoint["run_metadata"]["source_raw_sha256"] == raw_sha,
             "RAW differs from the AMI checkpoint")
    _require(checkpoint["run_metadata"]["protocol_sha256"] == protocol_sha,
             "protocol differs from the AMI checkpoint")
    for key, value in {
        "format": "theme26-ami-cache-v1", "complete": True,
        "axis_order": "XYZ", "dtype": "float32", "range_policy": "raw",
        "ami_checkpoint_sha256": checkpoint_sha, "source_raw_sha256": raw_sha,
        "protocol_sha256": protocol_sha,
        "data_config_sha256": config_hash(source_config),
    }.items():
        _require(cache_manifest.get(key) == value, f"cache manifest identity mismatch: {key}")
    _require(output_manifest.get("complete") is True, "validation output manifest is incomplete")
    expected = [record for record in records if record.split == "validation"]
    output_rows = output_manifest.get("records", [])
    cache_rows = [entry for entry in cache_manifest.get("records", [])
                  if entry.get("split") == "validation"]
    _require(len(expected) == len(output_rows) == len(cache_rows) == 9,
             "expected exactly nine validation crops in manifest/cache/output")
    _require(len({entry["crop_id"] for entry in cache_rows}) == 9,
             "cache has duplicate validation crop IDs")
    _require(len({entry["crop_id"] for entry in output_rows}) == 9,
             "output has duplicate validation crop IDs")
    cache_by_id = {entry["crop_id"]: entry for entry in cache_rows}
    shared_rfn_sha = None
    pairs = []
    for index, record in enumerate(expected):
        record_dir = result_dir / f"{index:03d}"
        metadata = yaml.safe_load((record_dir / "metadata.yaml").read_text(encoding="utf-8"))
        _require(output_rows[index] == metadata, f"output manifest differs from {record.sample_id} metadata")
        cache_entry = cache_by_id.get(record.sample_id)
        _require(cache_entry is not None, f"cache misses {record.sample_id}")
        identity = {
            "crop_id": record.sample_id, "split": "validation",
            "origin_xyz": list(record.crop.origin_xyz),
            "shape_xyz": list(record.crop.shape_xyz),
        }
        for key, value in identity.items():
            _require(metadata.get(key) == value and cache_entry.get(key) == value,
                     f"crop ID/coordinates mismatch for {record.sample_id}: {key}")
        for key, value in {"factor": record.sampling.factor, "offset": record.sampling.offset,
                           "source_core_id": record.source_core_id}.items():
            _require(cache_entry.get(key) == value,
                     f"cache crop sampling/core mismatch for {record.sample_id}: {key}")
        for key, value in {"sampling_factor": record.sampling.factor,
                           "sampling_offset": record.sampling.offset,
                           "axis_order": "XYZ", "dtype": "float32",
                           "normalization_id": volume_metadata.normalization_id,
                           "voxel_spacing_um_xyz": list(volume_metadata.voxel_spacing_um_xyz),
                           "ami_checkpoint_sha256": checkpoint_sha,
                           "source_raw_sha256": raw_sha,
                           "protocol_sha256": protocol_sha}.items():
            _require(metadata.get(key) == value,
                     f"output identity mismatch for {record.sample_id}: {key}")
        _require(metadata.get("observed_reinserted") is True,
                 f"output lacks observed reinsertion marker: {record.sample_id}")
        _require(metadata.get("primary_voxels") == 256 * 256 * 189,
                 f"primary count mismatch in metadata: {record.sample_id}")
        if shared_rfn_sha is None:
            shared_rfn_sha = metadata["rfn_checkpoint_sha256"]
        _require(metadata["rfn_checkpoint_sha256"] == shared_rfn_sha,
                 "validation crops use different RFN checkpoints")
        pairs.append((record, metadata, cache_entry, record_dir))
    _require({path.name for path in result_dir.iterdir() if path.is_dir()} - {"qc"} ==
             {f"{index:03d}" for index in range(9)},
             "unexpected crop directories in validation output")
    return (raw_path, volume_metadata, normalization, protocol_sha, raw_sha,
            checkpoint_sha, shared_rfn_sha, pairs)


def _error_sum(candidate: np.ndarray, dense: np.ndarray, mask: np.ndarray) -> float:
    difference = np.abs(candidate - dense)
    return float(np.sum(difference[mask], dtype=np.float64))


def run_validation_all_qc(*, result_dir: Path, cache_dir: Path,
                          ami_checkpoint: Path, raw: Path, output_dir: Path) -> None:
    """Compare all registered validation crops without changing saved predictions."""
    result_dir, cache_dir = result_dir.resolve(), cache_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"QC destination already exists: {output_dir}")
    (raw_path, volume_metadata, normalization, protocol_sha, raw_sha,
     checkpoint_sha, rfn_sha, pairs) = _preflight(
        result_dir, cache_dir, ami_checkpoint.resolve(), raw)
    print("identity preflight passed: nine validation crops, cache, output, AMI, RAW and protocol",
          flush=True)
    raw = read_raw_volume(raw_path, metadata=volume_metadata)
    rows = []
    total_sums = {"e0": 0.0, "ami_average": 0.0, "e1_rfn": 0.0}
    total_voxels = 0
    for record, metadata, cache_entry, record_dir in pairs:
        dense = normalize_grayscale(extract_volume_crop(raw, crop=record.crop),
                                    lower=float(normalization["lower"]),
                                    upper=float(normalization["upper"]))
        sample = make_sparse_sample(dense, sampling=record.sampling)
        mask = np.load(record_dir / "primary_mask.npy", mmap_mode="r", allow_pickle=False)
        _require(mask.dtype == np.bool_ and mask.shape == dense.shape and
                 np.array_equal(mask, sample.primary_metric_mask),
                 f"primary mask mismatch: {record.sample_id}")
        voxels = int(mask.sum())
        _require(voxels == metadata["primary_voxels"],
                 f"primary voxel count differs from metadata: {record.sample_id}")
        prediction = np.load(record_dir / "prediction.npy", mmap_mode="r", allow_pickle=False)
        _require(prediction.dtype == np.float32 and prediction.shape == dense.shape,
                 f"RFN prediction shape/dtype mismatch: {record.sample_id}")
        _require(bool(np.isfinite(prediction).all()) and
                 float(prediction.min()) >= 0.0 and float(prediction.max()) <= 1.0,
                 f"RFN prediction is not finite [0,1]: {record.sample_id}")
        _require(np.array_equal(prediction[..., sample.observed_indices],
                                sample.sparse_volume),
                 f"RFN observed slices differ from normalized RAW: {record.sample_id}")
        shape = tuple(record.crop.shape_xyz)
        sagittal = _cache_array(cache_dir, cache_entry["sagittal"], shape)
        coronal = _cache_array(cache_dir, cache_entry["coronal"], shape)
        sums = {}
        e0 = compose_full_volume(np.clip(linear_interpolate_z(sample), 0, 1), sample)
        sums["e0"] = _error_sum(e0, dense, mask)
        del e0
        # Average the two raw AMI directions before any clipping.
        ami_raw_average = (sagittal + coronal) * np.float32(0.5)
        ami = compose_full_volume(np.clip(ami_raw_average, 0, 1), sample)
        sums["ami_average"] = _error_sum(ami, dense, mask)
        del ami_raw_average, ami, sagittal, coronal
        e1 = compose_full_volume(np.clip(np.asarray(prediction), 0, 1), sample)
        _require(np.array_equal(e1, prediction),
                 f"RFN output changed under common clip/reinsertion: {record.sample_id}")
        sums["e1_rfn"] = _error_sum(e1, dense, mask)
        del e1
        _require(abs(sums["e1_rfn"] / voxels - metadata["primary_mae"]) <= 1e-8,
                 f"RFN primary MAE differs from metadata: {record.sample_id}")
        row = {"crop_id": record.sample_id,
               "crop_x": record.crop.origin_xyz[0], "crop_y": record.crop.origin_xyz[1],
               "crop_z": record.crop.origin_xyz[2], "primary_voxels": voxels,
               **_metrics(sums, voxels)}
        rows.append(row)
        total_voxels += voxels
        for key, value in sums.items():
            total_sums[key] += value
        print(f"crop={record.sample_id} E0={row['e0_primary_mae']:.9f} "
              f"AMI={row['ami_average_primary_mae']:.9f} "
              f"RFN={row['e1_rfn_primary_mae']:.9f}", flush=True)
        del prediction, mask, sample, dense
    overall = {"crop_id": "ALL_PRIMARY_VOXELS", "crop_x": "", "crop_y": "", "crop_z": "",
               "primary_voxels": total_voxels, **_metrics(total_sums, total_voxels)}
    output_dir.mkdir(parents=True, exist_ok=False)
    columns = list(rows[0])
    with (output_dir / "primary_mae_by_crop.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "overall_primary_voxel_mae.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(overall)
    summary = {
        "scope": "nine registered validation blocks from one Berea core; not nine independent rocks",
        "crop_count": len(rows), "primary_voxels": total_voxels,
        "normalization_id": normalization["normalization_id"],
        "normalization_lower_upper": [normalization["lower"], normalization["upper"]],
        "source_raw_sha256": raw_sha, "protocol_sha256": protocol_sha,
        "ami_checkpoint_sha256": checkpoint_sha, "rfn_checkpoint_sha256": rfn_sha,
        "cache_identity_checked": True, "output_identity_checked": True,
        "cache_direction_policy": "raw sagittal/coronal mean, then clip [0,1]",
        "common_policy": "clip [0,1], reinsert observed Z slices, score same bracketed-missing primary mask",
        "overall_weighting": "sum absolute error across all primary voxels / total primary voxels",
        "rfn_minus_ami_sign": "negative means RFN has lower MAE than AMI average",
        "overall": overall,
        "files": ["primary_mae_by_crop.csv", "overall_primary_voxel_mae.csv"],
    }
    (output_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False),
                                               encoding="utf-8")
    print(yaml.safe_dump({"qc_dir": str(output_dir), "overall": overall}, sort_keys=False))
