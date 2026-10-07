"""Small CPU coverage for fixed AMI input selection and diagnostic artifacts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from theme26.contracts import CropSpec, SamplingSpec, VolumeMetadata
from theme26.data.manifest import CropRecord
from theme26.models.saint import AnisotropicMetaInterpolator
from theme26.training.digital_rock import ManifestAmiDataset
from theme26.training.fixed_batch_overfit import run_fixed_batch_overfit, select_train_marginal


class FixedBatchOverfitTests(unittest.TestCase):
    def test_fixed_train_batch_and_saved_raw_predictions(self) -> None:
        raw = np.arange(8 * 8 * 8, dtype=np.uint8).reshape(8, 8, 8)
        metadata = VolumeMetadata(sample_id="tiny", source_core_id="tiny",
                                  shape_xyz=(8, 8, 8), voxel_spacing_um_xyz=(1, 1, 1),
                                  dtype="uint8", endianness="not-applicable",
                                  normalization_id="test", storage_order="XYZ")
        train = CropRecord("test", "train_0", "tiny", "train",
                           CropSpec((0, 0, 0), (8, 8, 8)), SamplingSpec(2, 0))
        held_out = CropRecord("test", "validation_0", "tiny", "validation",
                              CropSpec((0, 0, 0), (8, 8, 8)), SamplingSpec(2, 0))
        dataset = ManifestAmiDataset(raw, records=(train, held_out), metadata=metadata,
                                     normalization_lower=0, normalization_upper=255)
        with self.assertRaisesRegex(ValueError, "train record"):
            select_train_marginal(dataset, sample_id="validation_0",
                                  direction="sagittal", plane_index=2)
        batch = select_train_marginal(dataset, sample_id="train_0",
                                      direction="coronal", plane_index=3)
        original = {key: value.clone() for key, value in batch.items()}
        model = AnisotropicMetaInterpolator(factor=2, feature_channels=4,
                                            num_blocks=1, layers_per_block=1,
                                            growth_rate=2)
        config = {"training": {"loss_region": "all", "batch_size": 1,
                               "optimizer_steps": 2, "log_interval_steps": 1,
                               "learning_rate": 1e-3, "mixed_precision": False}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_fixed_batch_overfit(model, batch, config=config, output_dir=output,
                                    device=torch.device("cpu"), metadata={"split": "train"})
            for key, value in original.items():
                self.assertTrue(torch.equal(batch[key], value))
            self.assertTrue(np.array_equal(np.load(output / "input.npy"),
                                           original["input"].numpy()[0]))
            self.assertTrue(np.array_equal(np.load(output / "target.npy"),
                                           original["target"].numpy()[0, 0]))
            self.assertEqual(np.load(output / "primary_mask.npy").dtype, np.bool_)
            self.assertEqual(np.load(output / "prediction_before_raw.npy").shape, (8, 8))
            self.assertEqual(np.load(output / "prediction_after_raw.npy").shape, (8, 8))
            self.assertEqual((output / "comparison.png").read_bytes()[:8],
                             b"\x89PNG\r\n\x1a\n")
            self.assertTrue((output / "checkpoint.pt").exists())
            history = yaml.safe_load((output / "history.yaml").read_text())
            self.assertEqual([row["optimizer_step"] for row in history], [0, 1, 2])
            self.assertEqual(yaml.safe_load((output / "metadata.yaml").read_text())["split"],
                             "train")
