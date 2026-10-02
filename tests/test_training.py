"""CPU tests for the staged E1 training engine."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError:  # pragma: no cover - dependency-free setup
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrainingTests(unittest.TestCase):
    def test_ami_epoch_and_checkpoint_round_trip(self) -> None:
        from theme26.models.saint import AnisotropicMetaInterpolator
        from theme26.training import load_checkpoint, save_checkpoint, train_one_epoch
        from theme26.training.synthetic import SyntheticAmiDataset

        dataset = SyntheticAmiDataset(
            samples=4,
            length=8,
            z_lr=4,
            factor=2,
            spacing_um_xyz=(1.0, 1.0, 1.0),
            seed=0,
        )
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = AnisotropicMetaInterpolator(
            factor=2,
            feature_channels=4,
            num_blocks=1,
            layers_per_block=1,
            growth_rate=2,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        result = train_one_epoch(
            model,
            loader,
            optimizer,
            stage="ami",
            device=torch.device("cpu"),
            use_amp=False,
        )

        self.assertEqual(result.steps, 2)
        self.assertGreater(result.loss, 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scaler=None,
                epoch=1,
                validation_loss=result.loss,
                config={"experiment": {"stage": "ami"}},
            )
            restored = load_checkpoint(path, model=model, optimizer=optimizer)
            self.assertEqual(restored["epoch"], 1)

    def test_rfn_epoch(self) -> None:
        from theme26.models.saint import ResidualFusionNetwork
        from theme26.training import train_one_epoch
        from theme26.training.synthetic import SyntheticRfnDataset

        dataset = SyntheticRfnDataset(samples=4, x_size=8, y_size=8, seed=0)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = ResidualFusionNetwork(
            feature_channels=4,
            num_blocks=1,
            layers_per_block=1,
            growth_rate=2,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        result = train_one_epoch(
            model,
            loader,
            optimizer,
            stage="rfn",
            device=torch.device("cpu"),
            use_amp=False,
        )

        self.assertEqual(result.steps, 2)
        self.assertGreater(result.loss, 0.0)


if __name__ == "__main__":
    unittest.main()
