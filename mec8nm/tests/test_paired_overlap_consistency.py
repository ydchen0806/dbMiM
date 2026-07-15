from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from dbmim.datasets import EMVolumeDataset, transform_paired_offset
from train_finetune import (
    finetune_loss,
    make_affinity_target,
    make_offset_affinity_valid_mask,
    paired_overlap_consistency_loss,
    paired_overlap_slices,
)


class PairedOverlapDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tempdir.name) / "volume.h5"
        shape = (12, 40, 40)
        raw = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        labels = np.arange(1, np.prod(shape) + 1, dtype=np.int64).reshape(shape)
        with h5py.File(self.path, "w") as handle:
            handle.create_dataset("raw", data=raw)
            handle.create_dataset("labels", data=labels)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def make_dataset(self, *, augment: bool) -> EMVolumeDataset:
        return EMVolumeDataset(
            paths=[self.path],
            label_paths=[self.path],
            keys=["raw"],
            label_keys=["labels"],
            volume_size=(8, 24, 24),
            length_multiplier=8,
            augment=augment,
            augment_rotate_xy=True,
            augment_gamma=True,
            augment_gamma_range=(0.5, 1.8),
            augment_noise_std=0.05,
            paired_overlap=True,
            paired_overlap_shift=(2, 4, 4),
            seed=78,
        )

    def assert_overlap_labels_align(self, item: dict[str, torch.Tensor]) -> None:
        reference_slice, paired_slice = paired_overlap_slices(
            item["label"].shape[-3:],
            item["paired_offset"].tolist(),
        )
        reference = item["label"][reference_slice]
        paired = item["paired_label"][paired_slice]
        self.assertEqual(reference.shape, (6, 20, 20))
        self.assertTrue(torch.equal(reference, paired))

    def test_real_hdf5_pair_has_exact_coordinate_overlap(self) -> None:
        item = self.make_dataset(augment=False)[0]
        self.assertEqual(item["image"].shape, (1, 8, 24, 24))
        self.assertEqual(item["paired_image"].shape, item["image"].shape)
        self.assertEqual(tuple(item["paired_offset"].abs().tolist()), (2, 4, 4))
        self.assert_overlap_labels_align(item)

    def test_shared_geometry_keeps_overlap_aligned(self) -> None:
        torch.manual_seed(78)
        dataset = self.make_dataset(augment=True)
        for index in range(len(dataset)):
            self.assert_overlap_labels_align(dataset[index])

    def test_invalid_shift_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            EMVolumeDataset(
                paths=[self.path],
                label_paths=[self.path],
                volume_size=(8, 24, 24),
                paired_overlap=True,
                paired_overlap_shift=(8, 4, 4),
            )


class PairedOverlapLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            "channels": 3,
            "expected_overlap": [6, 20, 20],
            "confidence_floor": 0.0,
        }
        self.offset = torch.tensor([[2, -4, 4]], dtype=torch.long)

    def test_aligned_identical_overlap_has_zero_loss(self) -> None:
        reference = torch.randn(1, 3, 8, 24, 24)
        paired = torch.randn_like(reference)
        reference_slice, paired_slice = paired_overlap_slices(reference.shape[-3:], self.offset[0])
        paired[(0, slice(None), *paired_slice)] = reference[(0, slice(None), *reference_slice)]
        loss = paired_overlap_consistency_loss(reference, paired, self.offset, self.cfg)
        self.assertEqual(float(loss), 0.0)

    def test_reference_branch_is_stopped(self) -> None:
        reference = torch.randn(1, 3, 8, 24, 24, requires_grad=True)
        paired = torch.randn(1, 3, 8, 24, 24, requires_grad=True)
        loss = paired_overlap_consistency_loss(reference, paired, self.offset, self.cfg)
        loss.backward()
        self.assertIsNone(reference.grad)
        self.assertIsNotNone(paired.grad)
        self.assertGreater(float(paired.grad.abs().sum()), 0.0)

    def test_target_disagreement_is_excluded(self) -> None:
        reference = torch.zeros(1, 3, 8, 24, 24)
        paired = torch.full_like(reference, 10.0)
        reference_target = torch.zeros_like(reference)
        paired_target = torch.ones_like(paired)
        loss = paired_overlap_consistency_loss(
            reference,
            paired,
            self.offset,
            self.cfg,
            reference_target=reference_target,
            paired_target=paired_target,
        )
        self.assertEqual(float(loss), 0.0)

    def test_invalid_expected_overlap_fails_closed(self) -> None:
        logits = torch.zeros(1, 3, 8, 24, 24)
        with self.assertRaises(ValueError):
            paired_overlap_consistency_loss(
                logits,
                logits,
                self.offset,
                {**self.cfg, "expected_overlap": [7, 20, 20]},
            )

    def test_offset_geometry_transform(self) -> None:
        transformed = transform_paired_offset(
            (2, 3, 5),
            rotate_xy_k=1,
            flip_x=True,
            flip_z=True,
        )
        self.assertEqual(transformed, (-2, -5, -3))

    def test_supervised_plus_overlap_backward_smoke(self) -> None:
        torch.manual_seed(78)
        model = torch.nn.Conv3d(1, 3, kernel_size=1)
        image = torch.rand(1, 1, 8, 24, 24)
        paired_image = torch.rand_like(image)
        labels = torch.randint(1, 5, (1, 8, 24, 24))
        paired_labels = torch.randint(1, 5, (1, 8, 24, 24))
        train_cfg = {
            "replicate_affinity_boundary": True,
            "loss": {
                "loss_type": "weighted_mse",
                "weight_alpha": 1.0,
                "channel_weights": [1.0, 1.0, 1.0],
            },
        }
        target = make_affinity_target(labels, train_cfg)
        paired_target = make_affinity_target(paired_labels, train_cfg)
        logits = model(image)
        paired_logits = model(paired_image)
        main_loss, _ = finetune_loss(logits, target, labels, train_cfg, image=image)
        overlap_loss = paired_overlap_consistency_loss(
            logits,
            paired_logits,
            self.offset,
            self.cfg,
            reference_valid_mask=make_offset_affinity_valid_mask(labels, train_cfg, 3),
            paired_valid_mask=make_offset_affinity_valid_mask(paired_labels, train_cfg, 3),
            reference_target=target,
            paired_target=paired_target,
        )
        total = main_loss + 0.05 * overlap_loss
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertIsNotNone(model.weight.grad)
        self.assertGreater(float(model.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
