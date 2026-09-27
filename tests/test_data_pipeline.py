"""Tests for raw loading, aligned crops and sparse SAINT preparation."""

from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

import numpy as np

from theme26.contracts import CropSpec, SamplingSpec, VolumeMetadata
from theme26.data import (
    compose_full_volume,
    extract_aligned_crop,
    linear_interpolate_z,
    make_sparse_sample,
    normalize_grayscale,
    read_raw_volume,
    sample_training_crop,
    to_pore_mask,
)
from theme26.training import DigitalRockAmiDataset


class RawVolumeTests(unittest.TestCase):
    def test_zyx_storage_is_returned_as_canonical_xyz(self) -> None:
        canonical = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        metadata = VolumeMetadata(
            sample_id="sample",
            source_core_id="core",
            shape_xyz=(2, 3, 4),
            voxel_spacing_um_xyz=(1.0, 1.0, 1.0),
            dtype="uint8",
            endianness="not-applicable",
            storage_order="ZYX",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "volume.raw"
            canonical.transpose(2, 1, 0).tofile(path)
            loaded = read_raw_volume(path, metadata=metadata)
            np.testing.assert_array_equal(loaded, canonical)
            del loaded
            gc.collect()

    def test_size_mismatch_is_rejected(self) -> None:
        metadata = VolumeMetadata(
            sample_id="sample",
            source_core_id="core",
            shape_xyz=(2, 2, 2),
            voxel_spacing_um_xyz=(1.0, 1.0, 1.0),
            dtype="uint8",
            endianness="not-applicable",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "short.raw"
            np.arange(7, dtype=np.uint8).tofile(path)
            with self.assertRaises(ValueError):
                read_raw_volume(path, metadata=metadata)

    def test_normalization_and_pore_polarity(self) -> None:
        raw = np.array([0, 128, 255, 255], dtype=np.uint8).reshape(1, 2, 2)
        normalized = normalize_grayscale(raw, lower=0, upper=255)
        self.assertEqual(normalized.dtype, np.float32)
        self.assertAlmostEqual(float(normalized[0, 0, 1]), 128 / 255)
        pore = to_pore_mask((raw == 0).astype(np.uint8), source_pore_value=1)
        np.testing.assert_array_equal(pore, raw == 0)


class CropAndSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        x, y, z = np.indices((4, 5, 8), dtype=np.float32)
        self.dense = np.ascontiguousarray((x + y + z) / 14.0, dtype=np.float32)
        self.pore = np.ascontiguousarray(self.dense < 0.5)

    def test_aligned_and_seeded_crop(self) -> None:
        crop = CropSpec((1, 1, 2), (2, 3, 4))
        gray_crop, pore_crop = extract_aligned_crop(self.dense, self.pore, crop=crop)
        np.testing.assert_array_equal(gray_crop, self.dense[1:3, 1:4, 2:6])
        np.testing.assert_array_equal(pore_crop, self.pore[1:3, 1:4, 2:6])

        result_a = sample_training_crop(
            self.dense,
            self.pore,
            crop_shape_xyz=(2, 3, 4),
            rng=np.random.default_rng(7),
        )
        result_b = sample_training_crop(
            self.dense,
            self.pore,
            crop_shape_xyz=(2, 3, 4),
            rng=np.random.default_rng(7),
        )
        self.assertEqual(result_a[2], result_b[2])
        np.testing.assert_array_equal(result_a[0], result_b[0])

    def test_sparse_sampling_interpolation_and_consistency(self) -> None:
        linear = np.broadcast_to(
            np.arange(8, dtype=np.float32)[None, None, :] / 7.0,
            (2, 3, 8),
        ).copy()
        sample = make_sparse_sample(linear, sampling=SamplingSpec(factor=2))
        self.assertEqual(sample.sparse_volume.shape, (2, 3, 4))
        self.assertEqual(sample.observed_indices, (0, 2, 4, 6))
        self.assertEqual(int(sample.primary_metric_mask[0, 0].sum()), 3)
        interpolated = linear_interpolate_z(sample)
        np.testing.assert_allclose(interpolated[..., :7], linear[..., :7])
        np.testing.assert_allclose(interpolated[..., 7], linear[..., 6])

        prediction = np.zeros_like(linear)
        composed = compose_full_volume(prediction, sample)
        np.testing.assert_array_equal(
            composed[..., sample.observed_indices], sample.sparse_volume
        )

    def test_ami_dataset_shapes_for_both_marginal_directions(self) -> None:
        dataset = DigitalRockAmiDataset(
            self.dense,
            sampling=SamplingSpec(factor=2),
            spacing_um_xyz=(2.25, 2.25, 2.25),
        )
        self.assertEqual(len(dataset), 9)
        sagittal = dataset[0]
        coronal = dataset[4]
        self.assertEqual(tuple(sagittal["input"].shape), (3, 5, 4))
        self.assertEqual(tuple(sagittal["target"].shape), (1, 5, 8))
        self.assertEqual(tuple(coronal["input"].shape), (3, 4, 4))
        self.assertEqual(tuple(coronal["target"].shape), (1, 4, 8))
        self.assertEqual(tuple(sagittal["fdm"].shape), (2, 3, 3))


if __name__ == "__main__":
    unittest.main()
