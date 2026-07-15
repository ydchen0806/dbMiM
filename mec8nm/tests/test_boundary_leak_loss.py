from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from dbmim.datasets import EMVolumeDataset
from train_finetune import instance_pair_boundary_leak_loss


def _logit(probability: float) -> float:
    return float(torch.logit(torch.tensor(probability)))


class InstancePairBoundaryLeakLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            "channels": 3,
            "tail_fraction": 1.0,
            "max_affinity": 0.25,
            "min_edges": 1,
            "max_pairs": 64,
        }

    def test_no_cross_instance_boundary_has_zero_loss(self) -> None:
        logits = torch.randn(1, 3, 2, 4, 4, requires_grad=True)
        labels = torch.ones(1, 2, 4, 4, dtype=torch.long)
        loss, parts = instance_pair_boundary_leak_loss(logits, labels, self.cfg)
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(parts["pair_count"]), 0.0)
        self.assertIsNotNone(logits.grad)
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_high_affinity_boundary_is_penalized(self) -> None:
        logits = torch.full((1, 3, 1, 3, 6), _logit(0.05), requires_grad=True)
        labels = torch.ones(1, 1, 3, 6, dtype=torch.long)
        labels[:, :, :, 3:] = 2
        with torch.no_grad():
            logits[:, 2, :, :, 3] = _logit(0.90)
        loss, parts = instance_pair_boundary_leak_loss(logits, labels, self.cfg)
        loss.backward()
        self.assertAlmostEqual(float(loss), (0.90 - 0.25) ** 2, places=6)
        self.assertEqual(float(parts["pair_count"]), 1.0)
        self.assertEqual(float(parts["edge_count"]), 3.0)
        self.assertEqual(float(parts["active_pair_fraction"]), 1.0)
        self.assertAlmostEqual(float(parts["tail_probability_mean"]), 0.90, places=6)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_low_affinity_boundary_is_not_penalized(self) -> None:
        logits = torch.full((1, 3, 1, 3, 6), _logit(0.05), requires_grad=True)
        labels = torch.ones(1, 1, 3, 6, dtype=torch.long)
        labels[:, :, :, 3:] = 2
        with torch.no_grad():
            logits[:, 2, :, :, 3] = _logit(0.20)
        loss, parts = instance_pair_boundary_leak_loss(logits, labels, self.cfg)
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(parts["pair_count"]), 1.0)
        self.assertEqual(float(parts["active_pair_fraction"]), 0.0)
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_instance_pairs_are_equal_weight_not_edge_weighted(self) -> None:
        logits = torch.full((1, 3, 1, 5, 6), _logit(0.05))
        labels = torch.zeros(1, 1, 5, 6, dtype=torch.long)
        labels[:, :, :3, :3] = 1
        labels[:, :, :3, 3:] = 2
        labels[:, :, 4:, :3] = 3
        labels[:, :, 4:, 3:] = 4
        logits[:, 2, :, :3, 3] = _logit(0.90)
        logits[:, 2, :, 4:, 3] = _logit(0.50)
        loss, parts = instance_pair_boundary_leak_loss(logits, labels, self.cfg)
        expected = 0.5 * ((0.90 - 0.25) ** 2 + (0.50 - 0.25) ** 2)
        self.assertAlmostEqual(float(loss), expected, places=6)
        self.assertEqual(float(parts["pair_count"]), 2.0)
        self.assertEqual(float(parts["edge_count"]), 4.0)

    def test_invalid_tail_fraction_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            instance_pair_boundary_leak_loss(
                torch.randn(1, 3, 1, 2, 2),
                torch.ones(1, 1, 2, 2, dtype=torch.long),
                {**self.cfg, "tail_fraction": 0.0},
            )


class UnwidenedLabelDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "volume.h5"
        shape = (4, 12, 12)
        raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        labels = np.ones(shape, dtype=np.int64)
        labels[:, :, 6:] = 2
        with h5py.File(self.path, "w") as handle:
            handle.create_dataset("raw", data=raw)
            handle.create_dataset("labels", data=labels)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def _dataset(self, *, augment: bool, return_unwidened_label: bool = True) -> EMVolumeDataset:
        return EMVolumeDataset(
            paths=[self.path],
            label_paths=[self.path],
            keys=["raw"],
            label_keys=["labels"],
            volume_size=(4, 12, 12),
            augment=augment,
            augment_rotate_xy=True,
            widen_border=True,
            widen_border_radius=1,
            return_unwidened_label=return_unwidened_label,
            seed=81,
        )

    def test_unwidened_label_preserves_true_instance_contact(self) -> None:
        item = self._dataset(augment=False)[0]
        self.assertEqual(set(torch.unique(item["unwidened_label"]).tolist()), {1, 2})
        self.assertIn(0, set(torch.unique(item["label"]).tolist()))
        self.assertGreater(
            int(torch.count_nonzero(item["unwidened_label"])),
            int(torch.count_nonzero(item["label"])),
        )

    def test_augmented_labels_remain_coordinate_aligned(self) -> None:
        torch.manual_seed(81)
        item = self._dataset(augment=True)[0]
        widened_foreground = item["label"] > 0
        self.assertTrue(torch.equal(item["label"][widened_foreground], item["unwidened_label"][widened_foreground]))
        self.assertGreater(
            int(torch.count_nonzero(item["unwidened_label"])),
            int(torch.count_nonzero(item["label"])),
        )

    def test_extra_label_does_not_change_default_augmentation_rng(self) -> None:
        torch.manual_seed(812)
        default_item = self._dataset(augment=True, return_unwidened_label=False)[0]
        default_next_random = torch.rand(())
        torch.manual_seed(812)
        guarded_item = self._dataset(augment=True, return_unwidened_label=True)[0]
        guarded_next_random = torch.rand(())
        self.assertTrue(torch.equal(default_item["image"], guarded_item["image"]))
        self.assertTrue(torch.equal(default_item["label"], guarded_item["label"]))
        self.assertEqual(float(default_next_random), float(guarded_next_random))


if __name__ == "__main__":
    unittest.main()
