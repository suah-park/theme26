"""CPU checks for E2 objective, staged settings, and saved loss records."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from theme26.losses import gradient_loss_2d
from theme26.training import train_one_epoch, validate_one_epoch


ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


class ScaleAmi(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs, fdm):
        return self.scale * inputs


class ScaleRfn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward_slices(self, inputs):
        return self.scale * inputs[:, :1]


class E2TrainingTests(unittest.TestCase):
    def setUp(self):
        image = torch.tensor([[[[0.0, 1.0, 2.0], [0.0, 0.0, 0.0]]]])
        self.ami_batch = {
            "input": image, "target": torch.zeros_like(image), "fdm": torch.zeros(1),
            "mask": torch.zeros_like(image, dtype=torch.bool),
            "primary_mask": torch.tensor([[[[False, False, True],
                                               [False, False, False]]]]),
        }
        self.rfn_batch = dict(self.ami_batch, input=torch.cat((image, image), dim=1))

    def test_axis_means_are_summed_and_e2_ignores_training_mask(self):
        image = self.ami_batch["input"]
        self.assertAlmostEqual(float(gradient_loss_2d(image, self.ami_batch["target"])), 1.5)
        result = validate_one_epoch(ScaleAmi(), [self.ami_batch], stage="ami",
                                    device=CPU, use_amp=False, gradient_weight=0.1)
        self.assertAlmostEqual(result.l1_loss, 0.5)
        self.assertAlmostEqual(result.gradient_loss, 1.5)
        self.assertAlmostEqual(result.loss, 0.65)
        self.assertAlmostEqual(result.primary_mae, 1.0)  # raw 2 is clipped for selection

    def test_ami_and_rfn_total_loss_backpropagates(self):
        for stage, model, batch in (("ami", ScaleAmi(), self.ami_batch),
                                    ("rfn", ScaleRfn(), self.rfn_batch)):
            with self.subTest(stage=stage):
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                result = train_one_epoch(model, [batch], optimizer, stage=stage,
                                         device=CPU, use_amp=False, gradient_weight=0.1)
                self.assertAlmostEqual(result.loss, result.l1_loss +
                                       0.1 * result.gradient_loss, places=6)
                self.assertAlmostEqual(float(model.scale.grad), 0.65, places=6)
                self.assertAlmostEqual(float(model.scale.detach()), 0.935, places=6)

    def test_e1_default_retains_masked_l1_without_gradient(self):
        batch = dict(self.ami_batch)
        batch["mask"] = batch["primary_mask"]
        result = validate_one_epoch(ScaleAmi(), [batch], stage="ami", device=CPU,
                                    use_amp=False)
        self.assertEqual(result.loss, 2.0)
        self.assertEqual(result.l1_loss, 2.0)
        self.assertEqual(result.gradient_loss, 0.0)

    def test_e2_configs_preserve_e1_protocol_and_separate_artifacts(self):
        def read(path):
            return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
        ami = read("configs/experiments/e2_ami_seed1_ep10.yaml")
        rfn = read("configs/experiments/e2_rfn_seed1_ep10.yaml")
        old_ami = read("configs/experiments/e1_ami_full_epoch1.yaml")
        old_rfn = read("configs/experiments/e1_rfn_ep10_full_epoch1.yaml")
        cache = read("configs/inference/e2_cache_seed1_ep10.yaml")
        self.assertEqual((ami["model"], ami["data"]), (old_ami["model"], old_ami["data"]))
        self.assertEqual(rfn["model"], old_rfn["model"])
        self.assertEqual(rfn["data"]["source"], ami["data"])
        for config, old in ((ami, old_ami), (rfn, old_rfn)):
            self.assertEqual(config["experiment"]["seed"], old["experiment"]["seed"])
            self.assertEqual(config["training"]["epochs"], 10)
            self.assertEqual(config["training"]["gradient_weight"], 0.1)
            self.assertEqual(config["training"]["batch_size"], 8)
            self.assertFalse(config["training"]["mixed_precision"])
            self.assertIsNone(config["training"]["max_steps_per_epoch"])
            self.assertIsNone(config["training"]["max_validation_steps"])
            for key in ("learning_rate", "loss_region", "checkpoint_metric"):
                self.assertEqual(config["training"][key], old["training"][key])
            self.assertNotEqual(config["output_dir"], old["output_dir"])
        self.assertEqual(rfn["data"]["ami_checkpoint"], ami["output_dir"] + "/best.pt")
        self.assertEqual(rfn["data"]["cache_dir"], cache["output_dir"])
        self.assertEqual(cache["range_policy"], "raw_intermediate_clip_final")
        self.assertEqual((cache["batch_size"], cache["mixed_precision"]), (1, False))
        for stage in ("ami", "rfn"):
            pilot = read(f"configs/experiments/e2_{stage}_gpu_pilot.yaml")
            full = ami if stage == "ami" else rfn
            self.assertEqual((pilot["model"], pilot["data"]),
                             (full["model"], full["data"]))
            self.assertEqual(pilot["training"]["gradient_weight"], 0.1)
            self.assertEqual((pilot["training"]["batch_size"], pilot["training"]["epochs"]),
                             (8, 1))
            self.assertEqual((pilot["training"]["max_steps_per_epoch"],
                              pilot["training"]["max_validation_steps"]), (2, 2))
            self.assertNotEqual(pilot["output_dir"], full["output_dir"])

    def test_cli_records_all_e2_losses_and_primary_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "experiment": {"id": "e2-cpu", "stage": "ami", "seed": 1},
                "model": {"factor": 4, "feature_channels": 4, "kernel_size": 3,
                          "num_blocks": 1, "layers_per_block": 1, "growth_rate": 2},
                "data": {"mode": "synthetic", "length": 5, "z_lr": 3,
                         "train_samples": 2, "validation_samples": 2},
                "training": {"batch_size": 1, "epochs": 1, "learning_rate": 1e-4,
                             "mixed_precision": False, "loss_region": "all",
                             "checkpoint_metric": "primary_mae", "gradient_weight": 0.1},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            run = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/train.py"),
                                  "--config", str(config_path), "--output-dir", str(root / "run"),
                                  "--device", "cpu"], cwd=ROOT, env=env,
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            history = yaml.safe_load((root / "run/history.yaml").read_text(encoding="utf-8"))[0]
            for prefix in ("train", "validation"):
                self.assertAlmostEqual(history[prefix + "_total_loss"],
                                       history[prefix + "_l1"] +
                                       0.1 * history[prefix + "_gradient_loss"], places=6)
                self.assertIn(prefix + "_total_loss", run.stdout)
            self.assertIn("validation_primary_mae", history)
            checkpoint = torch.load(root / "run/best.pt", map_location="cpu",
                                    weights_only=False)
            self.assertEqual(checkpoint["validation_loss"], history["validation_primary_mae"])
            self.assertEqual(checkpoint["run_metadata"]["gradient_weight"], 0.1)

            original_history = (root / "run/history.yaml").read_bytes()
            second = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/train.py"),
                                     "--config", str(config_path), "--output-dir", str(root / "run"),
                                     "--device", "cpu"], cwd=ROOT, env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("run directory already has results", second.stderr)
            self.assertEqual((root / "run/history.yaml").read_bytes(), original_history)


if __name__ == "__main__":
    unittest.main()
