#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dbmim.datasets import labels_to_affinities
from dbmim.metrics import binary_iou_from_logits, dice_from_logits
from dbmim.postprocess import segmentation_metrics, waterz_agglomeration, watershed_fragments
from dbmim.utils import ensure_dir, load_config
from evaluate_cremi_segmentation import (  # noqa: E402
    apply_cremi_boundary_ignore,
    build_model,
    build_postprocess,
    list_cremi_files,
    normalize_crop_size,
    predict_affinities,
    read_cremi_crop,
    sigmoid_np,
)


class AffinityCalibrator(torch.nn.Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.log_scale = torch.nn.Parameter(torch.zeros(1, int(channels), 1, 1, 1))
        self.bias = torch.nn.Parameter(torch.zeros(1, int(channels), 1, 1, 1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.log_scale.exp() + self.bias


def _safe_logit(prob: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp(1e-5, 1.0 - 1e-5)
    return torch.log(prob / (1.0 - prob))


def _mean_metric(rows: list[dict[str, float | int | str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn z/y/x affinity calibration, then evaluate waterz.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--stride", nargs=3, type=int, default=[16, 80, 80])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-samples", nargs="+", default=["sample_A_20160501.hdf", "sample_B_20160501.hdf"])
    parser.add_argument(
        "--eval-samples",
        nargs="+",
        default=["sample_A_20160501.hdf", "sample_B_20160501.hdf", "sample_C_20160501.hdf"],
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=0.25)
    parser.add_argument("--identity-weight", type=float, default=0.01)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--seed-distance", type=int, default=10)
    parser.add_argument("--waterz-scoring", default="hist_quantile")
    parser.add_argument("--ignore-label", type=int, default=0)
    parser.add_argument("--cremi-boundary-ignore-distance-xy", type=int, default=1)
    parser.add_argument("--cremi-boundary-ignore-distance-z", type=int, default=0)
    parser.add_argument("--replicate-affinity-boundary", action="store_true")
    parser.add_argument("--metric-backend", choices=["internal", "skimage"], default="skimage")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(cfg, Path(args.checkpoint), device)
    postprocess = build_postprocess(cfg, Path(args.checkpoint), device)
    model_window = tuple(int(v) for v in model.volume_size)
    crop_size = normalize_crop_size(tuple(int(v) for v in args.crop_size), model_window)
    stride = tuple(int(v) for v in args.stride)
    data_cfg = cfg.get("data", {})
    raw_keys = data_cfg.get("image_keys") or ["volumes/raw", "raw", "main"]
    label_keys = data_cfg.get("label_keys") or ["volumes/labels/neuron_ids", "labels", "label", "gt"]

    files_by_name = {path.name: path for path in list_cremi_files(args.data_dir)}
    eval_files = [files_by_name[name] for name in args.eval_samples if name in files_by_name]
    train_names = {name for name in args.train_samples if name in files_by_name}
    if not train_names:
        raise ValueError(f"no train samples matched {args.train_samples}")
    if not eval_files:
        raise ValueError(f"no eval samples matched {args.eval_samples}")

    cache: dict[str, dict[str, object]] = {}
    train_logits: list[torch.Tensor] = []
    train_targets: list[torch.Tensor] = []
    for path in eval_files:
        raw, label, _ = read_cremi_crop(path, crop_size, raw_keys, label_keys)
        t0 = time.perf_counter()
        logits = predict_affinities(
            model,
            raw,
            model_window,
            stride,
            device,
            channels=3,
            postprocess=postprocess,
            return_logits=True,
        ).astype(np.float32, copy=False)
        infer_sec = time.perf_counter() - t0
        target_aff = labels_to_affinities(
            torch.from_numpy(label.astype(np.int64))[None],
            replicate_boundary=bool(args.replicate_affinity_boundary),
        )[0].numpy().astype(np.float32, copy=False)
        cache[path.name] = {
            "label": label,
            "logits": logits,
            "target_aff": target_aff,
            "infer_sec": infer_sec,
        }
        print({"sample": path.name, "raw_shape": list(raw.shape), "inference_sec": infer_sec}, flush=True)
        if path.name in train_names:
            train_logits.append(torch.from_numpy(logits))
            train_targets.append(torch.from_numpy(target_aff))

    x = torch.cat([item.reshape(1, 3, -1) for item in train_logits], dim=2).reshape(1, 3, 1, 1, -1).to(device)
    y = torch.cat([item.reshape(1, 3, -1) for item in train_targets], dim=2).reshape(1, 3, 1, 1, -1).to(device)
    calibrator = AffinityCalibrator(channels=3).to(device)
    opt = torch.optim.AdamW(calibrator.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    pos_fraction = y.mean(dim=(0, 2, 3, 4), keepdim=True).clamp(0.02, 0.98)
    pos_weight = (1.0 - pos_fraction) / pos_fraction
    pos_weight = (float(args.positive_weight) * pos_weight).clamp(0.25, 16.0)

    history = []
    for epoch in range(int(args.epochs)):
        calibrated = calibrator(x)
        bce = F.binary_cross_entropy_with_logits(calibrated, y, pos_weight=pos_weight)
        prob = torch.sigmoid(calibrated)
        inter = (prob * y).sum(dim=(2, 3, 4))
        denom = prob.sum(dim=(2, 3, 4)) + y.sum(dim=(2, 3, 4))
        dice_loss = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
        identity = calibrator.log_scale.pow(2).mean() + calibrator.bias.pow(2).mean()
        loss = bce + float(args.dice_weight) * dice_loss + float(args.identity_weight) * identity
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch + 1 == int(args.epochs):
            row = {
                "epoch": epoch + 1,
                "loss": float(loss.detach().cpu()),
                "bce": float(bce.detach().cpu()),
                "dice_loss": float(dice_loss.detach().cpu()),
                "scale": calibrator.log_scale.detach().exp().cpu().view(-1).tolist(),
                "bias": calibrator.bias.detach().cpu().view(-1).tolist(),
            }
            history.append(row)
            print(row, flush=True)

    records = []
    affinity_records = []
    calibrator.eval()
    with torch.no_grad():
        for path in eval_files:
            item = cache[path.name]
            label = item["label"]  # type: ignore[assignment]
            metric_label, boundary_ignore_fraction = apply_cremi_boundary_ignore(
                label,  # type: ignore[arg-type]
                ignore_label=int(args.ignore_label),
                distance_xy=int(args.cremi_boundary_ignore_distance_xy),
                distance_z=int(args.cremi_boundary_ignore_distance_z),
            )
            logits_np = np.asarray(item["logits"], dtype=np.float32)
            target_np = np.asarray(item["target_aff"], dtype=np.float32)
            variants = {
                "pred": logits_np,
                "learned_calibrated": calibrator(torch.from_numpy(logits_np)[None].to(device))[0]
                .float()
                .cpu()
                .numpy(),
            }
            for variant_name, variant_logits in variants.items():
                variant_prob = sigmoid_np(variant_logits).astype(np.float32, copy=False)
                logits_t = torch.from_numpy(variant_logits)
                target_t = torch.from_numpy(target_np)
                affinity_records.append(
                    {
                        "sample": path.name,
                        "split": "calibration_train" if path.name in train_names else "calibration_holdout",
                        "affinity_variant": variant_name,
                        "affinity_dice": float(dice_from_logits(logits_t, target_t)),
                        "affinity_iou": float(binary_iou_from_logits(logits_t, target_t)),
                    }
                )
                t_frag = time.perf_counter()
                fragments = watershed_fragments(
                    variant_prob,
                    backend="mahotas",
                    seed_method="maxima_distance",
                    seed_distance=int(args.seed_distance),
                    boundary_threshold=float(args.boundary_threshold),
                )
                fragment_sec = time.perf_counter() - t_frag
                for threshold in args.thresholds:
                    t_pp = time.perf_counter()
                    seg = waterz_agglomeration(
                        variant_prob,
                        threshold=float(threshold),
                        fragments=fragments,
                        scoring_function=str(args.waterz_scoring),
                    )
                    postprocess_sec = time.perf_counter() - t_pp + fragment_sec
                    t_met = time.perf_counter()
                    metrics = segmentation_metrics(
                        seg,
                        metric_label,
                        ignore_label=int(args.ignore_label),
                        backend=args.metric_backend,
                    )
                    metrics_sec = time.perf_counter() - t_met
                    record = {
                        "sample": path.name,
                        "split": "calibration_train" if path.name in train_names else "calibration_holdout",
                        "affinity_variant": variant_name,
                        "backend": "waterz",
                        "threshold": float(threshold),
                        "boundary_threshold": float(args.boundary_threshold),
                        "seed_distance": int(args.seed_distance),
                        "waterz_scoring": str(args.waterz_scoring),
                        "boundary_ignore_fraction": float(boundary_ignore_fraction),
                        "inference_sec": float(item["infer_sec"]),
                        "postprocess_sec": float(postprocess_sec),
                        "metrics_sec": float(metrics_sec),
                        "adapted_rand_error": metrics.adapted_rand_error,
                        "rand_fscore": metrics.rand_fscore,
                        "rand_precision": metrics.rand_precision,
                        "rand_recall": metrics.rand_recall,
                        "voi_split": metrics.voi_split,
                        "voi_merge": metrics.voi_merge,
                        "voi_sum": metrics.voi_split + metrics.voi_merge,
                    }
                    print(record, flush=True)
                    records.append(record)

    group_keys = ["affinity_variant", "backend", "threshold", "boundary_threshold", "seed_distance", "waterz_scoring"]
    per_group = []
    grouped: dict[tuple[object, ...], list[dict[str, float | int | str]]] = {}
    for record in records:
        grouped.setdefault(tuple(record[key] for key in group_keys), []).append(record)
    for key_tuple, rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        row = dict(zip(group_keys, key_tuple))
        row["n"] = len(rows)
        for key in ["adapted_rand_error", "voi_sum", "voi_split", "voi_merge", "postprocess_sec", "metrics_sec"]:
            row[key] = _mean_metric(rows, key)
        per_group.append(row)
    holdout_rows = [row for row in records if row["split"] == "calibration_holdout"]
    holdout_grouped: dict[tuple[object, ...], list[dict[str, float | int | str]]] = {}
    for record in holdout_rows:
        holdout_grouped.setdefault(tuple(record[key] for key in group_keys), []).append(record)
    holdout_per_group = []
    for key_tuple, rows in sorted(holdout_grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        row = dict(zip(group_keys, key_tuple))
        row["n"] = len(rows)
        for key in ["adapted_rand_error", "voi_sum", "voi_split", "voi_merge", "postprocess_sec", "metrics_sec"]:
            row[key] = _mean_metric(rows, key)
        holdout_per_group.append(row)

    summary = {
        "checkpoint": str(args.checkpoint),
        "train_samples": list(args.train_samples),
        "eval_samples": [path.name for path in eval_files],
        "scale": calibrator.log_scale.detach().exp().cpu().view(-1).tolist(),
        "bias": calibrator.bias.detach().cpu().view(-1).tolist(),
        "history": history,
        "num_records": len(records),
        "affinity_records": affinity_records,
        "per_backend_threshold": per_group,
        "holdout_per_backend_threshold": holdout_per_group,
    }
    if per_group:
        summary["best_by_voi_sum"] = min(per_group, key=lambda row: float(row["voi_sum"]))
        summary["best_by_adapted_rand"] = min(per_group, key=lambda row: float(row["adapted_rand_error"]))
    if holdout_per_group:
        summary["holdout_best_by_voi_sum"] = min(holdout_per_group, key=lambda row: float(row["voi_sum"]))
        summary["holdout_best_by_adapted_rand"] = min(
            holdout_per_group, key=lambda row: float(row["adapted_rand_error"])
        )
    (output_dir / "learned_affinity_calibration.json").write_text(
        json.dumps(
            {
                "scale": summary["scale"],
                "bias": summary["bias"],
                "history": history,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "cremi_segmentation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "cremi_segmentation_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
