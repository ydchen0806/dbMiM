#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"checkpoint has no model mapping: {path}")
    return payload


def validate_lineage(anchor: dict[str, Any], ema: dict[str, Any]) -> tuple[dict, dict]:
    if int(anchor.get("global_step", -1)) != 12000 or int(ema.get("global_step", -1)) != 12000:
        raise ValueError("R58 and R71 source checkpoints must both record global_step=12000")
    if anchor.get("ema_decay") not in (None, 0, 0.0):
        raise ValueError("R58 anchor unexpectedly records EMA weights")
    if abs(float(ema.get("ema_decay", -1.0)) - 0.999) > 1e-12:
        raise ValueError("R71 source must record ema_decay=0.999")

    for name, payload in (("R58", anchor), ("R71", ema)):
        cfg = payload.get("config", {})
        if int(cfg.get("seed", -1)) != 1789:
            raise ValueError(f"{name} source must record seed=1789")
        if cfg.get("model", {}).get("architecture") != "unetr_aniso_em":
            raise ValueError(f"{name} source must use unetr_aniso_em")

    anchor_model = anchor["model"]
    ema_model = ema["model"]
    if len(anchor_model) != 214 or set(anchor_model) != set(ema_model):
        raise ValueError("R58/R71 model topology must contain the same 214 keys")
    for key in anchor_model:
        left, right = anchor_model[key], ema_model[key]
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise TypeError(f"model state is not tensor-valued at {key}")
        if left.shape != right.shape:
            raise ValueError(f"source shape mismatch at {key}: {left.shape} vs {right.shape}")
        if not left.is_floating_point() and not torch.equal(left, right):
            raise ValueError(f"non-floating source state differs at {key}")
    return anchor_model, ema_model


def alpha_tag(alpha: float) -> str:
    return f"{alpha:.3f}".replace(".", "p")


def build_soups(
    anchor_path: Path,
    ema_path: Path,
    output_dir: Path,
    alphas: list[float],
    *,
    anchor_sha256: str | None = None,
    ema_sha256: str | None = None,
) -> dict[str, Any]:
    if not alphas or len(set(alphas)) != len(alphas):
        raise ValueError("alphas must be nonempty and unique")
    if any(not 0.0 < alpha <= 0.20 for alpha in alphas):
        raise ValueError(f"R76 alphas must satisfy 0 < alpha <= 0.20, got {alphas}")

    actual_anchor_sha = sha256(anchor_path)
    actual_ema_sha = sha256(ema_path)
    if anchor_sha256 is not None and actual_anchor_sha != anchor_sha256:
        raise ValueError(f"R58 SHA256 mismatch: {actual_anchor_sha}")
    if ema_sha256 is not None and actual_ema_sha != ema_sha256:
        raise ValueError(f"R71 SHA256 mismatch: {actual_ema_sha}")

    anchor = load_checkpoint(anchor_path)
    ema = load_checkpoint(ema_path)
    anchor_model, ema_model = validate_lineage(anchor, ema)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for alpha in sorted(alphas):
        model = {}
        changed = 0
        for key, anchor_value in anchor_model.items():
            ema_value = ema_model[key]
            if anchor_value.is_floating_point():
                value = torch.lerp(anchor_value.float(), ema_value.float(), float(alpha)).to(anchor_value.dtype)
                changed += int(not torch.equal(value, anchor_value))
            else:
                value = anchor_value.clone()
            model[key] = value
        tag = alpha_tag(alpha)
        output_path = output_dir / f"r58_r71_soup_alpha_{tag}.pt"
        payload = {
            "model": model,
            "global_step": 12000,
            "epoch": anchor.get("epoch"),
            "config": anchor.get("config", {}),
            "interpolation": {
                "alpha_r71_ema": float(alpha),
                "r58_sha256": actual_anchor_sha,
                "r71_ema_sha256": actual_ema_sha,
                "source_model_keys": len(model),
                "changed_float_tensors": changed,
            },
        }
        torch.save(payload, output_path)
        records.append(
            {
                "alpha_r71_ema": float(alpha),
                "path": str(output_path),
                "sha256": sha256(output_path),
                "bytes": output_path.stat().st_size,
                "changed_float_tensors": changed,
            }
        )

    manifest = {
        "r58": {"path": str(anchor_path), "sha256": actual_anchor_sha},
        "r71_ema": {"path": str(ema_path), "sha256": actual_ema_sha},
        "records": records,
    }
    manifest_path = output_dir / "checkpoint_soup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--ema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--anchor-sha256")
    parser.add_argument("--ema-sha256")
    args = parser.parse_args()
    build_soups(
        args.anchor,
        args.ema,
        args.output_dir,
        args.alphas,
        anchor_sha256=args.anchor_sha256,
        ema_sha256=args.ema_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
