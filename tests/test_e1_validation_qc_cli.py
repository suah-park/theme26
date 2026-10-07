"""Small CPU checks for the E1 data-package QC entry points."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from theme26.data.cli import build_parser
from theme26.data.e1_validation_all_qc import _metrics
from theme26.data.e1_validation_crop_qc import _figure


class E1ValidationQcCliTests(unittest.TestCase):
    def test_crop_command_routes_paths_to_data_qc(self):
        args = build_parser().parse_args([
            "e1-validation-crop", "--result-dir", "result/000", "--raw", "berea.raw",
            "--output-dir", "new-qc", "--source-config", "source.yaml",
        ])
        with patch("theme26.data.e1_validation_crop_qc.run_validation_crop_qc") as run:
            args.handler(args)
        run.assert_called_once_with(result_dir=Path("result/000"), raw=Path("berea.raw"),
                                    output_dir=Path("new-qc"), source_config=Path("source.yaml"))

    def test_all_command_routes_paths_to_data_qc(self):
        args = build_parser().parse_args([
            "e1-validation-all", "--result-dir", "results", "--cache-dir", "cache",
            "--ami-checkpoint", "best.pt", "--raw", "berea.raw", "--output-dir", "new-qc",
        ])
        with patch("theme26.data.e1_validation_all_qc.run_validation_all_qc") as run:
            args.handler(args)
        run.assert_called_once_with(result_dir=Path("results"), cache_dir=Path("cache"),
                                    ami_checkpoint=Path("best.pt"), raw=Path("berea.raw"),
                                    output_dir=Path("new-qc"))

    def test_weighted_mae_and_shared_scale_figure(self):
        metrics = _metrics({"e0": 8.0, "ami_average": 4.0, "e1_rfn": 2.0}, 4)
        self.assertEqual(metrics["rfn_improvement_vs_e0_percent"], 75.0)
        self.assertEqual(metrics["rfn_minus_ami_primary_mae"], -0.5)
        plane = np.zeros((4, 4), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "figure.svg"
            _figure(path, title="Axial: missing Z=125", x_axis="X", y_axis="Y",
                    truth=plane, e0=plane, e1=plane, error_maximum=1.0)
            svg = path.read_text(encoding="utf-8")
            self.assertIn("Axial: missing Z=125", svg)
            self.assertIn("Shared grayscale [0,1]", svg)
            self.assertIn("Absolute error", svg)


if __name__ == "__main__":
    unittest.main()
