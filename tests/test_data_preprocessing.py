"""CPU tests for raw validation, aligned crops and x4 sparse sampling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from theme26.contracts import CropSpec, SamplingSpec, VolumeMetadata
from theme26.data.crop_sampler import extract_aligned_crop, sample_training_crop
from theme26.data.raw_volume import normalize_grayscale, read_raw_volume, to_pore_mask
from theme26.data.sparse_sampling import compose_full_volume, linear_interpolate_z, make_sparse_sample


class RawVolumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metadata = VolumeMetadata(
            sample_id="tiny", source_core_id="tiny", shape_xyz=(2, 3, 4),
            voxel_spacing_um_xyz=(2.25, 2.25, 2.25), dtype="uint8", endianness="not-applicable"
        )

    def test_raw_load_validates_size_and_xyz_shape(self) -> None:
        source = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.raw"
            source.tofile(path)
            loaded = read_raw_volume(path, metadata=self.metadata)
            np.testing.assert_array_equal(loaded, source)
            path.write_bytes(b"too short")
            with self.assertRaises(ValueError):
                read_raw_volume(path, metadata=self.metadata)

    def test_normalization_is_float32_clipped_and_frozen(self) -> None:
        source = np.array([[[0, 10, 20, 30]]], dtype=np.uint8)
        actual = normalize_grayscale(source, lower=10.0, upper=20.0)
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, np.array([[[0.0, 0.0, 1.0, 1.0]]], dtype=np.float32))
        with self.assertRaises(ValueError):
            normalize_grayscale(source, lower=10, upper=10)

    def test_binary_polarity_and_label_validation(self) -> None:
        binary = np.array([[[0, 1], [1, 0]]], dtype=np.uint8)
        expected = np.array([[[True, False], [False, True]]])
        np.testing.assert_array_equal(to_pore_mask(binary, source_pore_value=0), expected)
        with self.assertRaises(ValueError):
            to_pore_mask(np.array([[[0, 2]]], dtype=np.uint8), source_pore_value=0)


class CropAndSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grayscale = np.linspace(0.0, 1.0, 8 * 7 * 10, dtype=np.float32).reshape(8, 7, 10)
        self.binary = (self.grayscale < 0.4)

    def test_aligned_crop_and_seeded_training_crop(self) -> None:
        crop, mask = extract_aligned_crop(self.grayscale, self.binary, crop=CropSpec((2, 1, 3), (3, 4, 5)))
        np.testing.assert_array_equal(crop, self.grayscale[2:5, 1:5, 3:8])
        np.testing.assert_array_equal(mask, self.binary[2:5, 1:5, 3:8])
        first = sample_training_crop(self.grayscale, self.binary, crop_shape_xyz=(3, 4, 5), rng=np.random.default_rng(7))
        second = sample_training_crop(self.grayscale, self.binary, crop_shape_xyz=(3, 4, 5), rng=np.random.default_rng(7))
        self.assertEqual(first[2], second[2])
        np.testing.assert_array_equal(first[0], second[0])

    def test_x4_sampling_masks_interpolation_and_data_consistency(self) -> None:
        dense = np.broadcast_to(np.arange(8, dtype=np.float32), (2, 3, 8)).copy() / 7.0
        sample = make_sparse_sample(dense, sampling=SamplingSpec(factor=4, offset=0))
        self.assertEqual(sample.sparse_volume.shape, (2, 3, 2))
        self.assertEqual(sample.observed_indices, (0, 4))
        self.assertEqual(int(sample.primary_metric_mask.sum()), 2 * 3 * 3)
        e0 = linear_interpolate_z(sample)
        np.testing.assert_allclose(e0[..., :5], dense[..., :5])
        np.testing.assert_array_equal(e0[..., 5:], np.broadcast_to(dense[..., 4:5], (2, 3, 3)))
        prediction = np.zeros_like(dense)
        full = compose_full_volume(prediction, sample)
        np.testing.assert_array_equal(full[..., (0, 4)], dense[..., (0, 4)])
        self.assertFalse(full[..., 1].any())

    def test_protocol_256_depth_has_64_observed_and_189_primary_slices(self) -> None:
        dense = np.zeros((1, 1, 256), dtype=np.float32)
        sample = make_sparse_sample(dense, sampling=SamplingSpec(factor=4, offset=0))
        self.assertEqual(sample.sparse_volume.shape, (1, 1, 64))
        self.assertEqual(len(sample.observed_indices), 64)
        self.assertEqual(int(sample.primary_metric_mask.sum()), 189)
        self.assertFalse(sample.primary_metric_mask[..., 253:].any())

    def test_rejects_invalid_contract_inputs(self) -> None:
        with self.assertRaises(TypeError):
            make_sparse_sample(self.grayscale.astype(np.float64), sampling=SamplingSpec())
        with self.assertRaises(ValueError):
            extract_aligned_crop(self.grayscale, self.binary[:-1], crop=CropSpec((0, 0, 0), (1, 1, 1)))


if __name__ == "__main__":
    unittest.main()
