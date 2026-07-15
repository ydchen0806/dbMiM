from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from dbmim.models import DecoderAwareDBMIM3DMAE, UNETREMAffinityNet
from train_pretrain import initialize_from_pretrained


def _model_kwargs() -> dict:
    return {
        "in_channels": 1,
        "out_channels": 3,
        "volume_size": (8, 32, 32),
        "patch_size": (4, 16, 16),
        "embed_dim": 24,
        "depth": 3,
        "num_heads": 3,
        "feature_size": 4,
        "dropout": 0.0,
        "em_refine_depth": 1,
        "channel_bias_init": (-0.2, 0.0, 0.0),
    }


class DecoderAwareDomainPretrainTest(unittest.TestCase):
    def test_segmentation_checkpoint_initializes_all_shared_modules(self) -> None:
        torch.manual_seed(73)
        source = UNETREMAffinityNet(**_model_kwargs())
        target = DecoderAwareDBMIM3DMAE(
            **_model_kwargs(),
            decoder_dim=24,
            mask_ratio=0.75,
            affinity_weight=0.1,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "source.pt"
            torch.save({"model": source.state_dict()}, checkpoint)
            loaded = initialize_from_pretrained(
                target,
                checkpoint,
                min_loaded_keys=len(source.state_dict()),
            )

        self.assertEqual(set(loaded), set(source.state_dict()))
        self.assertIn("head.z_head.weight", loaded)
        self.assertIn("decoder2.conv.conv.0.weight", loaded)
        self.assertNotIn("mask_token", loaded)
        self.assertNotIn("reconstruction_decoder.0.weight", loaded)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

    def test_decoder_aware_objective_is_finite_and_updates_shared_head(self) -> None:
        torch.manual_seed(73)
        model = DecoderAwareDBMIM3DMAE(
            **_model_kwargs(),
            decoder_dim=24,
            mask_ratio=0.75,
            structure_weight=0.2,
            affinity_weight=0.1,
        )
        image = torch.rand(1, 1, 8, 32, 32)
        output = model(image)
        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNotNone(output.affinity_loss)
        output.loss.backward()
        self.assertIsNotNone(model.head.z_head.weight.grad)
        self.assertGreater(float(model.head.z_head.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
