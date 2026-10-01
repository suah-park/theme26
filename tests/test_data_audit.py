"""Unit tests for the read-only 11 Sandstones audit primitives."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from theme26.data.audit import audit_core, discover_cores


class DataAuditTests(unittest.TestCase):
    def _write_core(self, root: Path, name: str, binary: np.ndarray) -> Path:
        core = root / name
        core.mkdir()
        grayscale = np.where(binary == 0, 20, 200).astype(np.uint8)
        grayscale.tofile(core / f"{name}_grayscale_filtered.raw")
        binary.tofile(core / f"{name}_binary.raw")
        return core

    def test_audit_reports_exact_stats_and_infers_dark_label_as_pore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = np.zeros((4, 4, 4), dtype=np.uint8)
            binary[::2] = 1
            core = self._write_core(root, "core_a", binary)
            result, crop = audit_core(core, shape_xyz=(4, 4, 4), chunk_x=2, qc_crop_origins_xyz=((0, 0, 0),))
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["binary"]["source_pore_value_inferred"], 0)
            self.assertEqual(result["grayscale"]["min"], 20)
            self.assertEqual(result["grayscale"]["max"], 200)
            self.assertEqual(result["grayscale"]["mean"], 110.0)
            self.assertEqual(result["binary"]["pore_fraction"], 0.5)
            self.assertIsNotNone(crop)
            assert crop is not None
            self.assertEqual(crop[0].shape, (4, 4, 4))

    def test_audit_rejects_nonbinary_labels_and_discovery_is_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            core = self._write_core(root, "z_core", np.full((4, 4, 4), 2, dtype=np.uint8))
            (root / "a_core").mkdir()
            self.assertEqual([item.name for item in discover_cores(root)], ["a_core", "z_core"])
            result, crop = audit_core(core, shape_xyz=(4, 4, 4), qc_crop_origins_xyz=((0, 0, 0),))
            self.assertEqual(result["status"], "fail")
            self.assertIsNone(crop)
            self.assertIn("binary labels must be exactly [0, 1]", result["failures"][0])


if __name__ == "__main__":
    unittest.main()
