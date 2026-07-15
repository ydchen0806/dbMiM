#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ENCODER_PREFIXES = ("pos_embed", "patch_embed.", "encoder_blocks.", "norm.")


def is_encoder_key(key: str) -> bool:
    return key == "pos_embed" or key.startswith(ENCODER_PREFIXES[1:])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_hybrid_checkpoint(
    r57_checkpoint: dict[str, Any],
    adapted_checkpoint: dict[str, Any],
    *,
    expected_adapted_step: int,
    expected_encoder_keys: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    r57_state = r57_checkpoint.get("model", r57_checkpoint)
    adapted_state = adapted_checkpoint.get("model", adapted_checkpoint)
    if not isinstance(r57_state, dict) or not isinstance(adapted_state, dict):
        raise TypeError("both checkpoints must contain model state dictionaries")

    adapted_step = int(adapted_checkpoint.get("global_step", -1))
    if adapted_step != int(expected_adapted_step):
        raise RuntimeError(
            f"adapted checkpoint step is {adapted_step}, expected {int(expected_adapted_step)}"
        )

    encoder_keys = sorted(key for key in r57_state if is_encoder_key(key))
    if len(encoder_keys) != int(expected_encoder_keys):
        raise RuntimeError(
            f"R57 contains {len(encoder_keys)} encoder keys, expected {int(expected_encoder_keys)}"
        )
    recorded_loaded = set(adapted_checkpoint.get("pretrained_loaded_keys", []))
    if recorded_loaded != set(encoder_keys):
        raise RuntimeError(
            "adapted checkpoint lineage does not match the complete R57 encoder: "
            f"missing={sorted(set(encoder_keys) - recorded_loaded)[:8]}, "
            f"extra={sorted(recorded_loaded - set(encoder_keys))[:8]}"
        )

    hybrid_state = {key: value.detach().cpu().clone() for key, value in r57_state.items()}
    changed_encoder_keys = []
    for key in encoder_keys:
        if key not in adapted_state:
            raise RuntimeError(f"adapted checkpoint is missing encoder key {key}")
        adapted_value = adapted_state[key]
        if tuple(adapted_value.shape) != tuple(r57_state[key].shape):
            raise RuntimeError(
                f"encoder shape mismatch for {key}: {tuple(adapted_value.shape)} "
                f"!= {tuple(r57_state[key].shape)}"
            )
        if not torch.equal(adapted_value.detach().cpu(), r57_state[key].detach().cpu()):
            changed_encoder_keys.append(key)
        hybrid_state[key] = adapted_value.detach().cpu().clone()
    if not changed_encoder_keys:
        raise RuntimeError("adapted checkpoint did not change any R57 encoder parameter")

    decoder_head_keys = sorted(key for key in r57_state if key not in set(encoder_keys))
    for key in decoder_head_keys:
        if not torch.equal(hybrid_state[key], r57_state[key].detach().cpu()):
            raise RuntimeError(f"R57 decoder/head key changed during hybrid export: {key}")

    payload = {
        "model": hybrid_state,
        "config": r57_checkpoint.get("config", {}),
        "global_step": adapted_step,
        "r74_provenance": {
            "method": "MEC raw-only encoder adaptation merged into untouched R57 decoder/head",
            "encoder_keys_replaced": encoder_keys,
            "changed_encoder_keys": changed_encoder_keys,
            "decoder_head_keys_preserved": decoder_head_keys,
            "adapted_pretrained_loaded_keys": sorted(recorded_loaded),
        },
    }
    summary = {
        "adapted_step": adapted_step,
        "hybrid_model_keys": len(hybrid_state),
        "encoder_keys_replaced": len(encoder_keys),
        "changed_encoder_keys": len(changed_encoder_keys),
        "decoder_head_keys_preserved": len(decoder_head_keys),
    }
    return payload, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the MEC R74 R57/encoder-adapt hybrid checkpoint")
    parser.add_argument("--r57", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-adapted-step", type=int, default=4000)
    parser.add_argument("--expected-encoder-keys", type=int, default=77)
    args = parser.parse_args()

    r57_checkpoint = torch.load(args.r57, map_location="cpu", weights_only=False)
    adapted_checkpoint = torch.load(args.adapted, map_location="cpu", weights_only=False)
    payload, summary = build_hybrid_checkpoint(
        r57_checkpoint,
        adapted_checkpoint,
        expected_adapted_step=args.expected_adapted_step,
        expected_encoder_keys=args.expected_encoder_keys,
    )
    payload["r74_provenance"].update(
        {
            "r57_path": str(args.r57),
            "r57_sha256": sha256_file(args.r57),
            "adapted_path": str(args.adapted),
            "adapted_sha256": sha256_file(args.adapted),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    summary.update(
        {
            "event": "r74_hybrid_checkpoint_built",
            "output": str(args.output),
            "output_sha256": sha256_file(args.output),
        }
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
