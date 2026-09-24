"""CPU tests for reconstruction and gradient losses."""

from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised on dependency-free setups
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "PyTorch is not installed")
class LossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from theme26.losses import gradient_loss_2d, gradient_loss_3d, masked_l1_loss

        cls.gradient_2d = staticmethod(gradient_loss_2d)
        cls.gradient_3d = staticmethod(gradient_loss_3d)
        cls.l1 = staticmethod(masked_l1_loss)

    def test_masked_l1_uses_only_selected_voxels(self) -> None:
        prediction = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]], requires_grad=True)
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[[[True, False], [True, False]]]])

        loss = self.l1(prediction, target, mask=mask)

        self.assertAlmostEqual(float(loss.detach()), 1.0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)

    def test_gradient_2d_is_zero_for_equal_images(self) -> None:
        image = torch.rand(2, 1, 4, 5)
        loss = self.gradient_2d(image, image.clone())
        self.assertEqual(float(loss), 0.0)

    def test_gradient_3d_detects_changed_edge(self) -> None:
        target = torch.zeros(1, 1, 3, 3, 3)
        prediction = target.clone()
        prediction[..., 1:, :, :] = 1.0

        loss = self.gradient_3d(prediction, target)

        self.assertGreater(float(loss), 0.0)

    def test_empty_mask_is_rejected(self) -> None:
        image = torch.zeros(1, 1, 2, 2)
        with self.assertRaises(ValueError):
            self.l1(image, image, mask=torch.zeros_like(image, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
