"""Tests for on-demand memory-mapped train crop preprocessing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from theme26.data.dataset import CoreSource, TrainRandomCropDataset, preprocess_crop
from theme26.contracts import CropSpec


class DatasetTests(unittest.TestCase):
    def test_preprocess_and_train_sampler_use_aligned_raw_crops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (8, 8, 8)
            gray = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
            binary = (gray > 100).astype(np.uint8)
            gray_path, binary_path = root / "gray.raw", root / "binary.raw"
            gray.tofile(gray_path); binary.tofile(binary_path)
            source = CoreSource("tiny", "train", gray_path, binary_path, 0, shape)
            item = preprocess_crop(source, crop=CropSpec((1, 1, 1), (4, 4, 4)), normalization_lower=0, normalization_upper=255)
            self.assertEqual(item.grayscale.shape, (4, 4, 4))
            self.assertTrue(np.array_equal(item.pore_mask, binary[1:5, 1:5, 1:5] == 0))
            self.assertEqual(item.sparse.sparse_volume.shape, (4, 4, 1))
            dataset = TrainRandomCropDataset([source], normalization_lower=0, normalization_upper=255, crop_shape_xyz=(4, 4, 4))
            sampled = dataset.sample(np.random.default_rng(3))
            self.assertEqual(sampled.core_id, "tiny")
            self.assertEqual(sampled.grayscale.shape, (4, 4, 4))

    def test_train_dataset_rejects_eval_source(self) -> None:
        source = CoreSource("eval", "test", Path("gray"), Path("binary"), shape_xyz=(8, 8, 8))
        with self.assertRaises(ValueError):
            TrainRandomCropDataset([source], normalization_lower=0, normalization_upper=255)


if __name__ == "__main__":
    unittest.main()
