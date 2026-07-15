from __future__ import annotations

import unittest

import torch

from dbmim.models import UNETREMAffinityNet
from train_finetune import discriminative_instance_embedding_loss


class DiscriminativeInstanceEmbeddingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            "feature_stride": [1, 1, 1],
            "min_voxels": 1,
            "max_instances": 8,
            "variance_margin": 0.20,
            "distance_margin": 1.0,
            "variance_weight": 1.0,
            "distance_weight": 1.0,
        }

    def test_compact_separated_instances_have_zero_loss(self) -> None:
        features = torch.zeros(1, 2, 2, 4, 4)
        features[:, 0, :, :, :2] = 1.0
        features[:, 0, :, :, 2:] = -1.0
        labels = torch.ones(1, 2, 4, 4, dtype=torch.long)
        labels[:, :, :, 2:] = 2
        loss, parts = discriminative_instance_embedding_loss(features, labels, self.cfg)
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(parts["instance_count"]), 2.0)

    def test_collapsed_instance_centroids_are_penalized(self) -> None:
        features = torch.zeros(1, 2, 2, 4, 4)
        features[:, 0] = 1.0
        labels = torch.ones(1, 2, 4, 4, dtype=torch.long)
        labels[:, :, :, 2:] = 2
        loss, parts = discriminative_instance_embedding_loss(features, labels, self.cfg)
        self.assertAlmostEqual(float(parts["variance_loss"]), 0.0, places=7)
        self.assertAlmostEqual(float(parts["distance_loss"]), 1.0, places=7)
        self.assertAlmostEqual(float(loss), 1.0, places=7)

    def test_background_is_ignored(self) -> None:
        features = torch.randn(1, 4, 2, 4, 4, requires_grad=True)
        labels = torch.zeros(1, 2, 4, 4, dtype=torch.long)
        loss, parts = discriminative_instance_embedding_loss(features, labels, self.cfg)
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(parts["instance_count"]), 0.0)
        self.assertIsNotNone(features.grad)
        self.assertEqual(float(features.grad.abs().sum()), 0.0)

    def test_noncompact_features_receive_gradients(self) -> None:
        torch.manual_seed(80)
        features = torch.randn(1, 4, 2, 4, 4, requires_grad=True)
        labels = torch.ones(1, 2, 4, 4, dtype=torch.long)
        labels[:, :, :, 2:] = 2
        loss, parts = discriminative_instance_embedding_loss(
            features,
            labels,
            {**self.cfg, "variance_margin": 0.0, "distance_margin": 1.5},
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(parts["variance_loss"]), 0.0)
        self.assertIsNotNone(features.grad)
        self.assertGreater(float(features.grad.abs().sum()), 0.0)

    def test_invalid_stride_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            discriminative_instance_embedding_loss(
                torch.randn(1, 4, 2, 4, 4),
                torch.ones(1, 2, 4, 4, dtype=torch.long),
                {**self.cfg, "feature_stride": [1, 0, 1]},
            )


class DecoderFeatureReturnTest(unittest.TestCase):
    def test_optional_features_do_not_change_logits_or_state(self) -> None:
        torch.manual_seed(80)
        model = UNETREMAffinityNet(
            in_channels=1,
            out_channels=3,
            volume_size=(8, 32, 32),
            patch_size=(4, 16, 16),
            embed_dim=24,
            depth=3,
            num_heads=3,
            feature_size=4,
            dropout=0.0,
            em_refine_depth=1,
            channel_bias_init=(-0.2, 0.0, 0.0),
        ).eval()
        image = torch.rand(1, 1, 8, 32, 32)
        state_keys = set(model.state_dict())
        with torch.no_grad():
            default_logits = model(image)
            returned_logits, features = model(image, return_features=True)
        self.assertTrue(torch.equal(default_logits, returned_logits))
        self.assertEqual(returned_logits.shape, (1, 3, 8, 32, 32))
        self.assertEqual(features.shape, (1, 4, 8, 32, 32))
        self.assertEqual(state_keys, set(model.state_dict()))


if __name__ == "__main__":
    unittest.main()
