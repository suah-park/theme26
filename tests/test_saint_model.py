"""CPU tests for the grayscale SAINT model implementation."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised on dependency-free setups
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SaintModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from theme26.models.saint import (
            AnisotropicMetaInterpolator,
            ResidualFusionNetwork,
            SaintReconstructor,
            build_filter_distance_matrix,
        )

        cls.AMI = AnisotropicMetaInterpolator
        cls.RFN = ResidualFusionNetwork
        cls.Saint = SaintReconstructor
        cls.build_fdm = staticmethod(build_filter_distance_matrix)

    def test_fdm_uses_physical_spacing_and_subpixel_position(self) -> None:
        fdm = self.build_fdm(
            factor=4,
            kernel_size=3,
            hr_spacing_um_xyz=(2.0, 3.0, 5.0),
            marginal_axis="sagittal",
        )

        self.assertEqual(tuple(fdm.shape), (1, 4, 3, 3))
        torch.testing.assert_close(fdm[0, :, 1, 1], torch.tensor([0.0, 5.0, 10.0, 15.0]))
        self.assertAlmostEqual(float(fdm[0, 0, 0, 1]), 3.0)

    def test_ami_shape_and_backward(self) -> None:
        model = self.AMI(
            factor=2,
            feature_channels=8,
            num_blocks=1,
            layers_per_block=2,
            growth_rate=4,
        )
        marginal = torch.rand(2, 3, 5, 4, requires_grad=True)
        fdm = self.build_fdm(
            factor=2,
            kernel_size=3,
            hr_spacing_um_xyz=(1.0, 1.0, 1.0),
            marginal_axis="sagittal",
        )

        output = model(marginal, fdm)

        self.assertEqual(tuple(output.shape), (2, 1, 5, 8))
        output.mean().backward()
        self.assertIsNotNone(marginal.grad)

    def test_rfn_shape_and_average_residual_path(self) -> None:
        model = self.RFN(
            feature_channels=8,
            num_blocks=1,
            layers_per_block=2,
            growth_rate=4,
        )
        sagittal = torch.rand(1, 1, 4, 5, 6)
        coronal = torch.rand(1, 1, 4, 5, 6)

        output = model(sagittal, coronal)

        self.assertEqual(tuple(output.shape), tuple(sagittal.shape))

    def test_end_to_end_shape_and_shared_ami(self) -> None:
        model = self.Saint(
            factor=2,
            feature_channels=8,
            ami_num_blocks=1,
            ami_layers_per_block=2,
            ami_growth_rate=4,
            rfn_num_blocks=1,
            rfn_layers_per_block=2,
            rfn_growth_rate=4,
            marginal_batch_size=2,
        )
        sparse = torch.rand(1, 1, 4, 5, 3, requires_grad=True)
        fdm = self.build_fdm(
            factor=2,
            kernel_size=3,
            hr_spacing_um_xyz=(1.0, 1.0, 1.0),
            marginal_axis="sagittal",
        )

        output = model(sparse, fdm)

        expected_shape = (1, 1, 4, 5, 6)
        self.assertEqual(tuple(output.prediction.shape), expected_shape)
        self.assertEqual(tuple(output.sagittal.shape), expected_shape)
        self.assertEqual(tuple(output.coronal.shape), expected_shape)
        output.prediction.mean().backward()
        self.assertIsNotNone(sparse.grad)


if __name__ == "__main__":
    unittest.main()
