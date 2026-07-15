from __future__ import annotations

import unittest

import torch

from train_finetune import halo_consistency_loss, make_halo_consistency_view


class HaloConsistencyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            "channels": 3,
            "halo": [2, 4, 4],
            "padding_mode": "reflect",
            "confidence_floor": 0.0,
        }

    def test_view_preserves_shape_and_core_exactly(self) -> None:
        image = torch.linspace(0.0, 1.0, 2 * 1 * 8 * 24 * 24).reshape(2, 1, 8, 24, 24)
        view = make_halo_consistency_view(image, self.cfg)
        self.assertEqual(view.shape, image.shape)
        self.assertTrue(torch.equal(view[:, :, 2:-2, 4:-4, 4:-4], image[:, :, 2:-2, 4:-4, 4:-4]))
        self.assertGreater(float((view - image).abs().mean()), 0.0)

    def test_identical_logits_have_zero_loss(self) -> None:
        logits = torch.randn(2, 3, 8, 24, 24)
        loss = halo_consistency_loss(logits, logits.clone(), self.cfg)
        self.assertEqual(float(loss), 0.0)

    def test_only_core_contributes(self) -> None:
        reference = torch.zeros(1, 3, 8, 24, 24)
        view = reference.clone()
        view[:, :, :2] = 10.0
        view[:, :, -2:] = 10.0
        view[:, :, :, :4] = 10.0
        view[:, :, :, -4:] = 10.0
        view[:, :, :, :, :4] = 10.0
        view[:, :, :, :, -4:] = 10.0
        self.assertEqual(float(halo_consistency_loss(reference, view, self.cfg)), 0.0)

    def test_reference_branch_is_stopped(self) -> None:
        reference = torch.randn(1, 3, 8, 24, 24, requires_grad=True)
        view = torch.randn(1, 3, 8, 24, 24, requires_grad=True)
        loss = halo_consistency_loss(reference, view, self.cfg)
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertIsNotNone(view.grad)
        self.assertGreater(float(view.grad.abs().sum()), 0.0)

    def test_invalid_halo_fails_closed(self) -> None:
        image = torch.rand(1, 1, 8, 24, 24)
        with self.assertRaises(ValueError):
            make_halo_consistency_view(image, {**self.cfg, "halo": [4, 4, 4]})


if __name__ == "__main__":
    unittest.main()
