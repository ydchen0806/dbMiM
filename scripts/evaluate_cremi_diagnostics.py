#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in [ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dbmim.datasets import labels_to_affinities
from dbmim.metrics import binary_iou_from_logits, dice_from_logits
from dbmim.postprocess import segmentation_metrics
from dbmim.utils import ensure_dir, load_config
from evaluate_cremi_segmentation import (
    build_model,
    list_cremi_files,
    metric_dict,
    normalize_crop_size,
    predict_affinities,
    read_cremi_crop,
    run_backend,
)


def _prob_to_logit(values: np.ndarray) -> torch.Tensor:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return torch.from_numpy(np.log(clipped / (1.0 - clipped)))


def affinity_stats(name: str, affinities: np.ndarray) -> dict[str, float | str]:
    stats: dict[str, float | str] = {"affinity_variant": name}
    for channel, channel_name in enumerate(["z", "y", "x"]):
        values = affinities[channel].reshape(-1).astype(np.float32, copy=False)
        stats[f"{channel_name}_mean"] = float(np.mean(values))
        stats[f"{channel_name}_std"] = float(np.std(values))
        stats[f"{channel_name}_p01"] = float(np.quantile(values, 0.01))
        stats[f"{channel_name}_p05"] = float(np.quantile(values, 0.05))
        stats[f"{channel_name}_p50"] = float(np.quantile(values, 0.50))
        stats[f"{channel_name}_p95"] = float(np.quantile(values, 0.95))
        stats[f"{channel_name}_p99"] = float(np.quantile(values, 0.99))
    return stats


def summarize_groups(
    records: list[dict[str, float | int | str]],
    group_keys: list[str],
    metric_keys: list[str],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[object, ...], list[dict[str, float | int | str]]] = {}
    for record in records:
        grouped.setdefault(tuple(record[key] for key in group_keys), []).append(record)
    rows: list[dict[str, float | int | str]] = []
    for key_tuple, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        mean_row: dict[str, float | int | str] = dict(zip(group_keys, key_tuple))
        mean_row["n"] = len(group_rows)
        for key in metric_keys:
            mean_row[key] = float(np.mean([float(r[key]) for r in group_rows])) if group_rows else float("nan")
        rows.append(mean_row)
    return rows


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(values: Iterable[float]) -> list[float]:
    return [float(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CREMI affinity calibration and VOI/ARAND decode behavior")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", nargs=3, type=int, default=[32, 320, 320])
    parser.add_argument("--stride", nargs=3, type=int, default=None)
    parser.add_argument("--z-thresholds", nargs="+", type=float, default=[0.85, 0.90, 0.95, 0.975, 0.99, 0.995])
    parser.add_argument(
        "--xy-thresholds",
        nargs="+",
        type=float,
        default=[0.90, 0.95, 0.975, 0.99, 0.995, 0.999],
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["graph_cc"],
        choices=["graph_cc", "cupy_graph_cc"],
    )
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ignore-label", type=int, default=0)
    parser.add_argument("--replicate-affinity-boundary", action="store_true")
    parser.add_argument("--metric-backend", choices=["internal", "skimage"], default="internal")
    parser.add_argument("--include-oracle-affinity", action="store_true")
    parser.add_argument("--include-inverted-affinity", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(args.output_dir)
    model = build_model(cfg, Path(args.checkpoint), device)
    model_window = tuple(int(v) for v in model.volume_size)
    stride = tuple(args.stride) if args.stride is not None else tuple(max(1, v // 2) for v in model_window)
    requested_crop_size = tuple(int(v) for v in args.crop_size)
    crop_size = normalize_crop_size(requested_crop_size, model_window)

    data_cfg = cfg.get("data", {})
    raw_keys = data_cfg.get("image_keys") or ["volumes/raw", "raw", "main"]
    label_keys = data_cfg.get("label_keys") or ["volumes/labels/neuron_ids", "labels", "label", "gt"]
    files = list_cremi_files(args.data_dir)
    if args.max_samples > 0:
        files = files[: args.max_samples]

    z_thresholds = parse_float_list(args.z_thresholds)
    xy_thresholds = parse_float_list(args.xy_thresholds)
    records: list[dict[str, float | int | str]] = []
    affinity_records: list[dict[str, float | int | str]] = []
    failures: list[dict[str, float | str]] = []

    for path in files:
        raw, label, crop = read_cremi_crop(path, crop_size, raw_keys, label_keys)
        print(
            {
                "sample": path.name,
                "raw_shape": list(raw.shape),
                "crop": [[sl.start, sl.stop] for sl in crop],
                "checkpoint": str(args.checkpoint),
            },
            flush=True,
        )
        t0 = time.perf_counter()
        pred_aff = predict_affinities(model, raw, model_window, stride, device)
        infer_sec = time.perf_counter() - t0
        oracle_aff = labels_to_affinities(
            torch.from_numpy(label.astype(np.int64))[None],
            replicate_boundary=args.replicate_affinity_boundary,
        ).numpy()[0]

        pred_logits = _prob_to_logit(pred_aff)
        oracle_target = torch.from_numpy(oracle_aff.astype(np.float32, copy=False))
        aff_dice = dice_from_logits(pred_logits, oracle_target)
        aff_iou = binary_iou_from_logits(pred_logits, oracle_target)

        variants: list[tuple[str, np.ndarray, float, float, float]] = [
            ("pred", pred_aff, float(aff_dice), float(aff_iou), float(infer_sec))
        ]
        if args.include_inverted_affinity:
            inverted = 1.0 - pred_aff
            inv_logits = _prob_to_logit(inverted)
            inv_dice = dice_from_logits(inv_logits, oracle_target)
            inv_iou = binary_iou_from_logits(inv_logits, oracle_target)
            variants.append(("inverted_pred", inverted, float(inv_dice), float(inv_iou), float(infer_sec)))
        if args.include_oracle_affinity:
            variants.append(("oracle", oracle_aff.astype(np.float32, copy=False), 1.0, 1.0, 0.0))

        label_unique = np.unique(label)
        for variant_name, affinities, variant_dice, variant_iou, variant_infer_sec in variants:
            stats = affinity_stats(variant_name, affinities)
            stats.update(
                {
                    "sample": path.name,
                    "label_unique": int(label_unique.size),
                    "label_foreground_fraction": float(np.mean(label != args.ignore_label)),
                    "affinity_dice": float(variant_dice),
                    "affinity_iou": float(variant_iou),
                }
            )
            affinity_records.append(stats)
            if args.diagnostics:
                print(stats, flush=True)

            for backend, z_threshold, xy_threshold in itertools.product(args.backends, z_thresholds, xy_thresholds):
                try:
                    t1 = time.perf_counter()
                    seg = run_backend(
                        affinities,
                        backend=backend,
                        threshold=0.0,
                        min_size=args.min_size,
                        seed_method="maxima_distance",
                        seed_distance=12,
                        boundary_threshold=0.5,
                        min_boundary=1,
                        score_mode="mean",
                        rag_quantile=0.25,
                        z_threshold=float(z_threshold),
                        xy_threshold=float(xy_threshold),
                    )
                    postprocess_sec = time.perf_counter() - t1
                    t2 = time.perf_counter()
                    metrics = segmentation_metrics(
                        seg,
                        label,
                        ignore_label=args.ignore_label,
                        backend=args.metric_backend,
                    )
                    metrics_sec = time.perf_counter() - t2
                except Exception as exc:
                    failure = {
                        "sample": path.name,
                        "affinity_variant": variant_name,
                        "backend": backend,
                        "z_threshold": float(z_threshold),
                        "xy_threshold": float(xy_threshold),
                        "error": repr(exc),
                    }
                    print(failure, flush=True)
                    failures.append(failure)
                    continue

                record: dict[str, float | int | str] = {
                    "sample": path.name,
                    "affinity_variant": variant_name,
                    "backend": backend,
                    "threshold": 0.0,
                    "z_threshold": float(z_threshold),
                    "xy_threshold": float(xy_threshold),
                    "min_size": int(args.min_size),
                    "affinity_dice": float(variant_dice),
                    "affinity_iou": float(variant_iou),
                    "inference_sec": float(variant_infer_sec),
                    "postprocess_sec": float(postprocess_sec),
                    "metrics_sec": float(metrics_sec),
                    **metric_dict(metrics),
                }
                print(record, flush=True)
                records.append(record)

    write_csv(output_dir / "cremi_diagnostic_metrics.csv", records)
    write_csv(output_dir / "cremi_affinity_diagnostics.csv", affinity_records)

    metric_keys = [
        "adapted_rand_error",
        "rand_fscore",
        "rand_precision",
        "rand_recall",
        "voi_split",
        "voi_merge",
        "voi_sum",
        "n_pred",
        "n_gt",
        "affinity_dice",
        "affinity_iou",
        "inference_sec",
        "postprocess_sec",
        "metrics_sec",
    ]
    group_keys = ["affinity_variant", "backend", "z_threshold", "xy_threshold", "min_size"]
    per_group = summarize_groups(records, group_keys, metric_keys)
    pred_rows = [row for row in per_group if row["affinity_variant"] == "pred"]
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "requested_crop_size": list(requested_crop_size),
        "crop_size": list(crop_size),
        "window_size": list(model_window),
        "stride": list(stride),
        "z_thresholds": z_thresholds,
        "xy_thresholds": xy_thresholds,
        "backends": args.backends,
        "min_size": int(args.min_size),
        "ignore_label": int(args.ignore_label),
        "num_records": len(records),
        "num_affinity_records": len(affinity_records),
        "metric_backend": args.metric_backend,
        "replicate_affinity_boundary": bool(args.replicate_affinity_boundary),
        "failures": failures,
        "per_affinity_backend_threshold": per_group,
    }
    if pred_rows:
        summary["best_pred_by_adapted_rand"] = min(pred_rows, key=lambda row: float(row["adapted_rand_error"]))
        summary["best_pred_by_voi_sum"] = min(pred_rows, key=lambda row: float(row["voi_sum"]))
    if per_group:
        summary["best_overall_by_adapted_rand"] = min(per_group, key=lambda row: float(row["adapted_rand_error"]))
        summary["best_overall_by_voi_sum"] = min(per_group, key=lambda row: float(row["voi_sum"]))
    (output_dir / "cremi_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cremi_diagnostic_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cremi_affinity_diagnostics.json").write_text(
        json.dumps(affinity_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
