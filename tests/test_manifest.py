"""Tests for versioned crop manifests and spatial leakage checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from theme26.contracts import CropSpec, SamplingSpec
from theme26.data import (
    CropRecord,
    read_crop_manifest,
    validate_spatial_split,
    write_crop_manifest,
)
from theme26.data.protocol import load_protocol
from theme26.data.manifest import validate_protocol_records

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _record(sample_id: str, split: str, origin_x: int) -> CropRecord:
    return CropRecord(
        manifest_version="test_v1",
        sample_id=sample_id,
        source_core_id="berea",
        split=split,  # type: ignore[arg-type]
        crop=CropSpec((origin_x, 0, 0), (4, 4, 4)),
        sampling=SamplingSpec(factor=2),
    )


class ManifestTests(unittest.TestCase):
    def test_frozen_berea_manifest(self) -> None:
        records = read_crop_manifest(
            REPOSITORY_ROOT / "configs/data/berea_crops.yaml"
        )
        validate_spatial_split(
            records,
            volume_shape_xyz=(1000, 1000, 1000),
            minimum_buffer_voxels=32,
        )
        counts = {
            split: sum(record.split == split for record in records)
            for split in ("train", "validation", "test")
        }
        self.assertEqual(counts, {"train": 48, "validation": 9, "test": 9})
        config, normalization, evaluation = load_protocol(REPOSITORY_ROOT / 'configs/data')
        validate_protocol_records(records, config, evaluation)

    def test_old_manifest_version_and_changed_eval_coordinates_are_rejected(self):
        from dataclasses import replace
        records = read_crop_manifest(REPOSITORY_ROOT / 'configs/data/berea_crops.yaml')
        config, _, evaluation = load_protocol(REPOSITORY_ROOT / 'configs/data')
        stale = (replace(records[0], manifest_version='old-zyx'), *records[1:])
        with self.assertRaisesRegex(ValueError, 'stale'):
            validate_protocol_records(stale, config, evaluation)
        changed = list(records)
        changed[48] = replace(changed[48], crop=CropSpec((432, 1, 0), (256, 256, 256)))
        with self.assertRaisesRegex(ValueError, 'held-out'):
            validate_protocol_records(changed, config, evaluation)

    def test_round_trip_and_buffer_validation(self) -> None:
        records = (
            _record("train", "train", 0),
            _record("validation", "validation", 6),
            _record("test", "test", 12),
        )
        validate_spatial_split(
            records, volume_shape_xyz=(16, 4, 4), minimum_buffer_voxels=2
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.yaml"
            write_crop_manifest(path, records)
            self.assertEqual(read_crop_manifest(path), records)

    def test_cross_split_overlap_is_rejected(self) -> None:
        records = (
            _record("train", "train", 0),
            _record("validation", "validation", 3),
            _record("test", "test", 10),
        )
        with self.assertRaises(ValueError):
            validate_spatial_split(
                records, volume_shape_xyz=(16, 4, 4), minimum_buffer_voxels=0
            )

    def test_insufficient_buffer_is_rejected(self) -> None:
        records = (
            _record("train", "train", 0),
            _record("validation", "validation", 5),
            _record("test", "test", 10),
        )
        with self.assertRaises(ValueError):
            validate_spatial_split(
                records, volume_shape_xyz=(16, 4, 4), minimum_buffer_voxels=2
            )


if __name__ == "__main__":
    unittest.main()
