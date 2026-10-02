"""Versioned crop-manifest I/O and spatial leakage checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import yaml

from theme26.contracts import CropSpec, SamplingSpec, ShapeXYZ

Split = Literal["train", "validation", "test"]
_SPLITS = ("train", "validation", "test")
@dataclass(frozen=True)
class CropRecord:
    """One immutable spatial crop assignment."""

    manifest_version: str
    sample_id: str
    source_core_id: str
    split: Split
    crop: CropSpec
    sampling: SamplingSpec

    def __post_init__(self) -> None:
        if not self.manifest_version or not self.sample_id or not self.source_core_id:
            raise ValueError("manifest_version, sample_id and source_core_id are required")
        if self.split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}")


def write_crop_manifest(path: str | Path, records: Sequence[CropRecord]) -> None:
    """Write crop assignments in the canonical human-readable YAML format."""

    if not records:
        raise ValueError("crop manifest cannot be empty")
    versions = {record.manifest_version for record in records}
    if len(versions) != 1:
        raise ValueError("all crop records must use one manifest_version")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_version": versions.pop(),
        "records": [
            {
                "sample_id": record.sample_id,
                "source_core_id": record.source_core_id,
                "split": record.split,
                "crop_x": record.crop.origin_xyz[0],
                "crop_y": record.crop.origin_xyz[1],
                "crop_z": record.crop.origin_xyz[2],
                "size_x": record.crop.shape_xyz[0],
                "size_y": record.crop.shape_xyz[1],
                "size_z": record.crop.shape_xyz[2],
                "sampling_factor": record.sampling.factor,
                "sampling_offset": record.sampling.offset,
            }
            for record in records
        ],
    }
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def read_crop_manifest(path: str | Path) -> tuple[CropRecord, ...]:
    """Read a YAML manifest and reconstruct validated crop records."""

    manifest_path = Path(path)
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("crop manifest must contain a records list")
    version = payload.get("manifest_version")
    if not isinstance(version, str) or not version:
        raise ValueError("crop manifest must contain manifest_version")
    required = {
        "sample_id", "source_core_id", "split", "crop_x", "crop_y", "crop_z",
        "size_x", "size_y", "size_z", "sampling_factor", "sampling_offset",
    }
    records_list = payload["records"]
    for index, row in enumerate(records_list):
        if not isinstance(row, dict):
            raise ValueError(f"crop record {index} must be a mapping")
        missing = required - set(row)
        if missing:
            raise ValueError(f"crop record {index} is missing fields: {sorted(missing)}")
    records = tuple(
        CropRecord(
            manifest_version=version,
            sample_id=str(row["sample_id"]),
            source_core_id=str(row["source_core_id"]),
            split=row["split"],  # type: ignore[arg-type]
            crop=CropSpec(
                origin_xyz=(int(row["crop_x"]), int(row["crop_y"]), int(row["crop_z"])),
                shape_xyz=(int(row["size_x"]), int(row["size_y"]), int(row["size_z"])),
            ),
            sampling=SamplingSpec(
                factor=int(row["sampling_factor"]),
                offset=int(row["sampling_offset"]),
            ),
        )
        for row in records_list
    )
    if not records:
        raise ValueError("crop manifest cannot be empty")
    return records


def _separation(a: CropSpec, b: CropSpec) -> tuple[int, int, int]:
    gaps: list[int] = []
    for a_start, a_size, b_start, b_size in zip(
        a.origin_xyz, a.shape_xyz, b.origin_xyz, b.shape_xyz
    ):
        a_end = a_start + a_size
        b_end = b_start + b_size
        if a_end <= b_start:
            gaps.append(b_start - a_end)
        elif b_end <= a_start:
            gaps.append(a_start - b_end)
        else:
            gaps.append(-1)
    return tuple(gaps)  # type: ignore[return-value]


def validate_spatial_split(
    records: Sequence[CropRecord],
    *,
    volume_shape_xyz: ShapeXYZ,
    minimum_buffer_voxels: int = 0,
) -> None:
    """Reject out-of-bounds crops and cross-split overlap/leakage."""

    if minimum_buffer_voxels < 0:
        raise ValueError("minimum_buffer_voxels must be non-negative")
    if not records:
        raise ValueError("crop manifest cannot be empty")
    present = {record.split for record in records}
    if present != set(_SPLITS):
        raise ValueError(f"manifest must contain all splits, found {sorted(present)}")

    keys: set[tuple[str, Split, tuple[int, int, int]]] = set()
    for record in records:
        record.crop.validate_inside(volume_shape_xyz)
        key = (record.source_core_id, record.split, record.crop.origin_xyz)
        if key in keys:
            raise ValueError(f"duplicate crop assignment: {key}")
        keys.add(key)

    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            if left.source_core_id != right.source_core_id or left.split == right.split:
                continue
            separation = _separation(left.crop, right.crop)
            if max(separation) < minimum_buffer_voxels:
                raise ValueError(
                    "cross-split crops overlap or violate the spatial buffer: "
                    f"{left.sample_id}/{left.split} and {right.sample_id}/{right.split}; "
                    f"axis gaps={separation}"
                )


def records_from_protocol(config: dict, evaluation: dict, *, seed: int = 2026,
                          train_crops: int = 48) -> tuple[CropRecord, ...]:
    """Generate deterministic crop records from the canonical spatial protocol."""
    import numpy as np
    from .protocol import region_spec

    roi = region_spec(config['splits']['train'][0])
    size = tuple(config['crop_shape_xyz'])
    limits = tuple(width - crop + 1 for width, crop in zip(roi.shape_xyz, size))
    if train_crops < 1 or train_crops > int(np.prod(limits)):
        raise ValueError('invalid number of unique training crops')
    rng = np.random.default_rng(seed)
    origins = set()
    while len(origins) < train_crops:
        origins.add(tuple(start + int(rng.integers(limit))
                          for start, limit in zip(roi.origin_xyz, limits)))
    records = [CropRecord(config['manifest_version'], f'berea_train_{i:03d}',
                          'Berea', 'train', CropSpec(origin, size), SamplingSpec(4, 0))
               for i, origin in enumerate(sorted(origins))]
    records.extend(CropRecord(config['manifest_version'], row['crop_id'],
                              row['source_core_id'], row['split'],
                              CropSpec(tuple(row['origin_xyz']), tuple(row['shape_xyz'])),
                              SamplingSpec(row['sampling_factor'], row['sampling_offset']))
                   for row in evaluation['crops'])
    return tuple(records)


def validate_protocol_records(records: Sequence[CropRecord], config: dict,
                              evaluation: dict) -> None:
    """Reject stale manifests, out-of-region train crops and changed held-out blocks."""
    from .protocol import contains, region_spec

    validate_spatial_split(records, volume_shape_xyz=tuple(config['source_shape_xyz']),
                           minimum_buffer_voxels=config['buffer_voxels_x'])
    for record in records:
        if record.manifest_version != config['manifest_version']:
            raise ValueError('stale training manifest version; regenerate from protocol')
        regions = config['splits'][record.split]
        if not any(record.source_core_id == entry['source_core_id']
                   and contains(region_spec(entry), record.crop) for entry in regions):
            raise ValueError(f'crop outside canonical {record.split} region: {record.sample_id}')
        if record.crop.shape_xyz != tuple(config['crop_shape_xyz']) or record.sampling != SamplingSpec(4, 0):
            raise ValueError('crop shape or sampling differs from canonical protocol')
    expected = records_from_protocol(config, evaluation)
    held_out = lambda rows: {(r.sample_id, r.source_core_id, r.split, r.crop, r.sampling)
                             for r in rows if r.split != 'train'}
    if held_out(records) != held_out(expected):
        raise ValueError('held-out crops differ from canonical evaluation manifest')
