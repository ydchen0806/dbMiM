from __future__ import annotations

import unittest

import torch

from train_finetune import make_photometric_consistency_view, photometric_consistency_loss


class AugmentationConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            "channels": 3,
            "gamma_range": [0.85, 1.2],
            "gain_range": [0.9, 1.1],
            "bias_range": [-0.05, 0.05],
            "noise_std": 0.02,
            "confidence_floor": 0.0,
        }

    def test_view_is_aligned_bounded_and_changed(self) -> None:
        torch.manual_seed(72)
        image = torch.linspace(0.0, 1.0, 2 * 1 * 4 * 8 * 8).reshape(2, 1, 4, 8, 8)
        view = make_photometric_consistency_view(image, self.cfg)
        self.assertEqual(view.shape, image.shape)
        self.assertGreaterEqual(float(view.min()), 0.0)
        self.assertLessEqual(float(view.max()), 1.0)
        self.assertGreater(float((view - image).abs().mean()), 0.005)

    def test_identical_logits_have_zero_loss(self) -> None:
        logits = torch.randn(2, 3, 4, 8, 8)
        loss = photometric_consistency_loss(logits, logits.clone(), self.cfg)
        self.assertEqual(float(loss), 0.0)

    def test_reference_branch_is_stopped(self) -> None:
        reference = torch.randn(2, 3, 4, 8, 8, requires_grad=True)
        view = torch.randn(2, 3, 4, 8, 8, requires_grad=True)
        loss = photometric_consistency_loss(reference, view, self.cfg)
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertIsNotNone(view.grad)
        self.assertGreater(float(view.grad.abs().sum()), 0.0)

    def test_invalid_range_fails_closed(self) -> None:
        image = torch.rand(1, 1, 4, 8, 8)
        with self.assertRaises(ValueError):
            make_photometric_consistency_view(image, {**self.cfg, "gain_range": [1.1, 0.9]})


if __name__ == "__main__":
    unittest.main()
