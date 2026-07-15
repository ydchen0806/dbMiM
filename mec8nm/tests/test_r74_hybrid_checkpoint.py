from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_mec8nm_r74_hybrid_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("mec8nm_r74_hybrid", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load R74 hybrid exporter: {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_hybrid_checkpoint = MODULE.build_hybrid_checkpoint
is_encoder_key = MODULE.is_encoder_key


class R74HybridCheckpointTest(unittest.TestCase):
    def test_replaces_only_encoder_keys(self) -> None:
        r57_state = {
            "pos_embed": torch.zeros(1, 4, 8),
            "patch_embed.proj.weight": torch.zeros(8, 1, 2, 2, 2),
            "encoder_blocks.0.norm1.weight": torch.zeros(8),
            "norm.weight": torch.zeros(8),
            "decoder2.up.weight": torch.full((8, 8, 2, 2, 2), 7.0),
            "head.z_head.weight": torch.full((1, 8, 1, 1, 1), 9.0),
        }
        encoder_keys = sorted(key for key in r57_state if is_encoder_key(key))
        adapted_state = {
            key: value + 1.0 for key, value in r57_state.items() if is_encoder_key(key)
        }
        adapted_state["mask_token"] = torch.zeros(1, 1, 8)
        payload, summary = build_hybrid_checkpoint(
            {"model": r57_state, "config": {"model": {}}},
            {
                "model": adapted_state,
                "global_step": 4000,
                "pretrained_loaded_keys": encoder_keys,
            },
            expected_adapted_step=4000,
            expected_encoder_keys=len(encoder_keys),
        )
        hybrid = payload["model"]
        for key in encoder_keys:
            self.assertTrue(torch.equal(hybrid[key], adapted_state[key]), key)
        self.assertTrue(torch.equal(hybrid["decoder2.up.weight"], r57_state["decoder2.up.weight"]))
        self.assertTrue(torch.equal(hybrid["head.z_head.weight"], r57_state["head.z_head.weight"]))
        self.assertEqual(summary["encoder_keys_replaced"], len(encoder_keys))
        self.assertEqual(summary["decoder_head_keys_preserved"], 2)

    def test_rejects_incomplete_lineage(self) -> None:
        r57_state = {"pos_embed": torch.zeros(1, 4, 8), "head.weight": torch.zeros(1)}
        with self.assertRaises(RuntimeError):
            build_hybrid_checkpoint(
                {"model": r57_state},
                {
                    "model": {"pos_embed": torch.ones(1, 4, 8)},
                    "global_step": 4000,
                    "pretrained_loaded_keys": [],
                },
                expected_adapted_step=4000,
                expected_encoder_keys=1,
            )


if __name__ == "__main__":
    unittest.main()
