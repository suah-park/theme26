"""Dependency-free tests for the shared interface contract."""

from __future__ import annotations

import unittest

from theme26.contracts import CropSpec, SamplingSpec, VolumeMetadata


class ContractTests(unittest.TestCase):
    def test_primary_sampling_indices(self) -> None:
        spec = SamplingSpec(factor=4, offset=0)
        indices = spec.observed_indices(256)

        self.assertEqual(len(indices), 64)
        self.assertEqual(indices[:3], (0, 4, 8))
        self.assertEqual(indices[-1], 252)

    def test_crop_must_fit_volume(self) -> None:
        CropSpec((372, 372, 372)).validate_inside((1000, 1000, 1000))
        with self.assertRaises(ValueError):
            CropSpec((800, 800, 800)).validate_inside((1000, 1000, 1000))

    def test_metadata_rejects_wrong_axis(self) -> None:
        with self.assertRaises(ValueError):
            VolumeMetadata(
                sample_id="sample",
                source_core_id="core",
                shape_xyz=(1000, 1000, 1000),
                voxel_spacing_um_xyz=(2.25, 2.25, 2.25),
                dtype="uint8",
                endianness="not-applicable",
                axis_order="ZYX",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
