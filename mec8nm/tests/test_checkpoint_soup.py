from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_mec8nm_checkpoint_soup.py"
SPEC = importlib.util.spec_from_file_location("mec8nm_checkpoint_soup", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load checkpoint soup builder: {SCRIPT}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_soups = MODULE.build_soups


class CheckpointSoupTest(unittest.TestCase):
    def make_payload(self, value: float, *, ema_decay: float | None) -> dict:
        model = {f"encoder_blocks.{index}.weight": torch.full((2, 2), value) for index in range(214)}
        return {
            "model": model,
            "global_step": 12000,
            "epoch": 5,
            "ema_decay": ema_decay,
            "config": {"seed": 1789, "model": {"architecture": "unetr_aniso_em"}},
        }

    def test_exact_linear_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.pt"
            ema = root / "ema.pt"
            torch.save(self.make_payload(2.0, ema_decay=None), anchor)
            torch.save(self.make_payload(6.0, ema_decay=0.999), ema)
            manifest = build_soups(anchor, ema, root / "out", [0.05, 0.20])
            self.assertEqual(len(manifest["records"]), 2)
            for record, expected in zip(manifest["records"], (2.2, 2.8)):
                payload = torch.load(record["path"], map_location="cpu", weights_only=False)
                self.assertEqual(payload["interpolation"]["changed_float_tensors"], 214)
                self.assertTrue(
                    torch.allclose(payload["model"]["encoder_blocks.0.weight"], torch.full((2, 2), expected))
                )

    def test_rejects_alpha_outside_bounded_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.pt"
            ema = root / "ema.pt"
            torch.save(self.make_payload(2.0, ema_decay=None), anchor)
            torch.save(self.make_payload(6.0, ema_decay=0.999), ema)
            with self.assertRaises(ValueError):
                build_soups(anchor, ema, root / "out", [0.25])

    def test_rejects_bad_ema_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            anchor = root / "anchor.pt"
            ema = root / "ema.pt"
            torch.save(self.make_payload(2.0, ema_decay=None), anchor)
            torch.save(self.make_payload(6.0, ema_decay=0.99), ema)
            with self.assertRaises(ValueError):
                build_soups(anchor, ema, root / "out", [0.05])


if __name__ == "__main__":
    unittest.main()
