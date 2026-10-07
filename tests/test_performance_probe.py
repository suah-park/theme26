"""CPU coverage for bounded phase progress and persisted timing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from theme26.models.saint import AnisotropicMetaInterpolator
from theme26.training.engine import train_one_epoch, validate_one_epoch
from theme26.training.performance_probe import PhaseProbe
from theme26.training.synthetic import SyntheticAmiDataset


class PerformanceProbeTests(unittest.TestCase):
    def test_interval_also_records_final_partial_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            probe = PhaseProbe(epoch=1, phase="train", expected_steps=5,
                               device=torch.device("cpu"), output_dir=Path(directory),
                               log_interval_steps=2)
            probe.start()
            for step in range(1, 6):
                probe.on_step(step, step * 8)
            summary = probe.finish(5)
            events = yaml.safe_load((Path(directory) / "progress.yaml").read_text())
            self.assertEqual([row["step"] for row in events], [0, 2, 4, 5])
            self.assertEqual(summary["samples"], 40)

    def test_train_and_validation_phase_progress(self) -> None:
        dataset = SyntheticAmiDataset(samples=4, length=8, z_lr=4,
                                      factor=2, spacing_um_xyz=(1, 1, 1), seed=3)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True,
                                  generator=torch.Generator().manual_seed(5))
        validation_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        model = AnisotropicMetaInterpolator(factor=2, feature_channels=4,
                                            num_blocks=1, layers_per_block=1,
                                            growth_rate=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            train_probe = PhaseProbe(epoch=1, phase="train", expected_steps=4,
                                     device=torch.device("cpu"), output_dir=output)
            train_probe.start()
            train = train_one_epoch(model, train_loader, optimizer, stage="ami",
                                    device=torch.device("cpu"), use_amp=False,
                                    max_steps=4, progress_callback=train_probe.on_step)
            train_summary = train_probe.finish(train.steps)
            validation_probe = PhaseProbe(epoch=1, phase="validation", expected_steps=2,
                                          device=torch.device("cpu"), output_dir=output)
            validation_probe.start()
            validation = validate_one_epoch(model, validation_loader, stage="ami",
                                             device=torch.device("cpu"), use_amp=False,
                                             max_steps=2,
                                             progress_callback=validation_probe.on_step)
            validation_summary = validation_probe.finish(validation.steps)
            events = yaml.safe_load((output / "progress.yaml").read_text())
            self.assertEqual([(row["phase"], row["step"]) for row in events],
                             [("train", 0), ("train", 2), ("train", 4),
                              ("validation", 0), ("validation", 1), ("validation", 2)])
            self.assertEqual(train_summary["samples"], 4)
            self.assertEqual(validation_summary["samples"], 2)
            self.assertGreater(train_summary["samples_per_second"], 0)
            self.assertGreater(validation_summary["elapsed_seconds"], 0)
            self.assertIsNone(validation_summary["peak_gpu_allocated_bytes"])
