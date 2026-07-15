#!/usr/bin/env python3
from __future__ import annotations

import csv
import filecmp
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT.parent
TRAIN_IDS = {1, 2, 3, 4, 5, 6, 7, 9}
VAL_IDS = {8, 10}
TEST_IDS = {11, 12}


def _ids(records: list[dict]) -> set[int]:
    return {int(record["index"]) for record in records}


def _compare_file(local: Path, legacy: Path) -> None:
    if not local.is_file() or not legacy.is_file():
        raise RuntimeError(f"missing mirrored runtime file: {local} or {legacy}")
    if not filecmp.cmp(local, legacy, shallow=False):
        raise RuntimeError(f"runtime mirror drift: {local.relative_to(ROOT)} != {legacy}")


def verify_split() -> None:
    split_path = ROOT / "data/MEC_8nm/mec8nm_splits.json"
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    splits = payload["splits"]
    actual = {
        "train": _ids(splits["train"]),
        "val": _ids(splits["val"]),
        "test": _ids(splits["test"]),
    }
    expected = {"train": TRAIN_IDS, "val": VAL_IDS, "test": TEST_IDS}
    if actual != expected:
        raise RuntimeError(f"split contract changed: expected {expected}, got {actual}")
    if set.union(*actual.values()) != set(range(1, 13)):
        raise RuntimeError(f"split contract does not cover M01--M12 exactly: {actual}")


def verify_anchor() -> None:
    metrics_path = (
        ROOT
        / "evidence/mec8nm_remote_results/r58_balanced_crop_20260711/cremi_segmentation_metrics.csv"
    )
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 80:
        raise RuntimeError(f"R58 anchor must contain exactly 80 raw rows, found {len(rows)}")
    samples = {row["sample"] for row in rows}
    expected_samples = {"mec8nm_08.h5", "mec8nm_10.h5"}
    if samples != expected_samples:
        raise RuntimeError(f"R58 anchor sample mismatch: {samples}")
    if {row["backend"] for row in rows} != {"waterz"}:
        raise RuntimeError("R58 anchor contains a non-WaterZ backend")
    for row in rows:
        for field in ("adapted_rand_error", "voi_sum"):
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"non-finite R58 {field}: {row}")


def verify_no_large_data() -> None:
    forbidden = []
    for suffix in ("*.h5", "*.hdf", "*.hdf5", "*.pt", "*.pth", "*.ckpt"):
        forbidden.extend(ROOT.rglob(suffix))
    if forbidden:
        raise RuntimeError(f"large data/checkpoints copied into MEC workspace: {forbidden[:5]}")


def verify_mirrors() -> None:
    shared = [
        "train_finetune.py",
        "train_pretrain.py",
        "scripts/build_mec8nm_r74_hybrid_checkpoint.py",
        "scripts/build_mec8nm_checkpoint_soup.py",
        "scripts/evaluate_cremi_segmentation.py",
        "scripts/evaluate_cremi_blockwise_scale.py",
        "dbmim/datasets.py",
        "dbmim/metrics.py",
        "dbmim/models.py",
        "dbmim/postprocess.py",
        "dbmim/utils.py",
        "data/MEC_8nm/mec8nm_manifest.json",
        "data/MEC_8nm/mec8nm_splits.json",
    ]
    for relative in shared:
        _compare_file(ROOT / relative, LEGACY_ROOT / relative)
    _compare_file(ROOT / "skill/SKILL.md", LEGACY_ROOT / "dbmim-neuron-seg-skill/SKILL.md")
    _compare_file(ROOT / "skill/STATE.md", LEGACY_ROOT / "dbmim-neuron-seg-skill/STATE.md")


def main() -> None:
    verify_split()
    verify_anchor()
    verify_no_large_data()
    verify_mirrors()
    print(
        json.dumps(
            {
                "workspace": str(ROOT),
                "split_contract": (
                    "M01-07,M09 train; M08,M10 validation; "
                    "M11,M12 consumed designated replay"
                ),
                "r58_rows": 80,
                "runtime_mirrors": "in_sync",
                "large_assets_copied": False,
                "status": "ok",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
