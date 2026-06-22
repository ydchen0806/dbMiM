#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from dbmim.models import LearnedRAGMergeScorer
from dbmim.postprocess import (
    agglomerate_fragments_by_scores,
    rag_boundary_features,
    rag_pair_labels_from_ground_truth,
    segmentation_metrics,
    watershed_fragments,
)
from dbmim.utils import ensure_dir, load_config
from evaluate_cremi_segmentation import (  # noqa: E402
    apply_cremi_boundary_ignore,
    build_model,
    build_postprocess,
    list_cremi_files,
    metric_dict,
    normalize_crop_size,
    predict_affinities,
    read_cremi_crop,
    sigmoid_np,
)


def _standardize_train_eval(
    train_features: np.ndarray,
    eval_features: list[np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], dict[str, list[float]]]:
    mean = train_features.mean(axis=0, keepdims=True).astype(np.float32)
    std = train_features.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-4] = 1.0
    return (
        (train_features - mean) / std,
        [(features - mean) / std for features in eval_features],
        {"mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist()},
    )


def train_edge_scorer(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    device: torch.device,
    hidden_dim: int,
    depth: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    positive_weight_cap: float,
) -> tuple[LearnedRAGMergeScorer, list[dict[str, float | int]]]:
    model = LearnedRAGMergeScorer(features.shape[1], hidden_dim=hidden_dim, depth=depth, dropout=dropout).to(device)
    x = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)
    y = torch.from_numpy(targets.astype(np.float32, copy=False)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    pos = float(targets.sum())
    neg = float(targets.size - targets.sum())
    pos_weight_value = min(float(positive_weight_cap), max(1.0, neg / max(pos, 1.0)))
    pos_weight = torch.tensor([pos_weight_value], device=device)
    n = int(x.shape[0])
    history: list[dict[str, float | int]] = []
    for epoch in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        losses = []
        model.train()
        for start in range(0, n, int(batch_size)):
            idx = perm[start : start + int(batch_size)]
            logits = model(x[idx])
            loss = F.binary_cross_entropy_with_logits(logits, y[idx], pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == int(epochs):
            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits)
                pred = (probs >= 0.5).float()
                tp = float(((pred == 1) & (y == 1)).sum().cpu())
                fp = float(((pred == 1) & (y == 0)).sum().cpu())
                fn = float(((pred == 0) & (y == 1)).sum().cpu())
                tn = float(((pred == 0) & (y == 0)).sum().cpu())
                precision = tp / max(tp + fp, 1.0)
                recall = tp / max(tp + fn, 1.0)
                acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
            row = {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "edge_acc": float(acc),
                "edge_precision": float(precision),
                "edge_recall": float(recall),
                "pos_weight": float(pos_weight_value),
            }
            print(row, flush=True)
            history.append(row)
    return model, history


def baseline_scores(features: np.ndarray, mode: str) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    mode = mode.lower()
    if mode == "mean":
        return features[:, :3].mean(axis=1)
    if mode == "xy_mean":
        return features[:, 1:3].mean(axis=1)
    if mode == "min":
        return features[:, 3:6].min(axis=1)
    if mode in {"q25_proxy", "conservative"}:
        return 0.5 * features[:, :3].mean(axis=1) + 0.5 * features[:, 3:6].min(axis=1)
    raise ValueError(f"unknown baseline score mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sparse learned RAG edge postprocessor.")
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
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--seed-distance", type=int, default=10)
    parser.add_argument("--min-boundary", type=int, default=4)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--merge-thresholds", nargs="+", type=float, default=[0.30, 0.40, 0.50, 0.60, 0.70])
    parser.add_argument("--baseline-thresholds", nargs="+", type=float, default=[0.10, 0.20, 0.30, 0.40, 0.50])
    parser.add_argument("--baseline-score-modes", nargs="+", default=["mean", "xy_mean", "conservative"])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--positive-weight-cap", type=float, default=32.0)
    parser.add_argument("--ignore-label", type=int, default=0)
    parser.add_argument("--cremi-boundary-ignore-distance-xy", type=int, default=1)
    parser.add_argument("--cremi-boundary-ignore-distance-z", type=int, default=0)
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
    train_names = {name for name in args.train_samples if name in files_by_name}
    eval_files = [files_by_name[name] for name in args.eval_samples if name in files_by_name]
    if not train_names:
        raise ValueError(f"no train samples matched {args.train_samples}")
    if not eval_files:
        raise ValueError(f"no eval samples matched {args.eval_samples}")

    cache: dict[str, dict[str, object]] = {}
    train_features_parts: list[np.ndarray] = []
    train_targets_parts: list[np.ndarray] = []
    eval_feature_refs: list[np.ndarray] = []
    sample_order: list[str] = []
    for path in eval_files:
        raw, label, crop = read_cremi_crop(path, crop_size, raw_keys, label_keys)
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
        )
        aff = sigmoid_np(logits).astype(np.float32, copy=False)
        infer_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        fragments = watershed_fragments(
            aff,
            backend="mahotas",
            seed_method="maxima_distance",
            seed_distance=int(args.seed_distance),
            boundary_threshold=float(args.boundary_threshold),
        )
        fragment_sec = time.perf_counter() - t1
        t2 = time.perf_counter()
        pairs, features, counts = rag_boundary_features(aff, fragments, min_boundary=int(args.min_boundary))
        rag_sec = time.perf_counter() - t2
        if path.name in train_names:
            targets = rag_pair_labels_from_ground_truth(
                pairs,
                fragments,
                label,
                ignore_label=int(args.ignore_label),
                positive_fraction=float(args.positive_fraction),
            )
            train_features_parts.append(features)
            train_targets_parts.append(targets)
            print(
                {
                    "sample": path.name,
                    "train_edges": int(targets.size),
                    "positive_fraction": float(targets.mean()) if targets.size else 0.0,
                },
                flush=True,
            )
        cache[path.name] = {
            "label": label,
            "fragments": fragments,
            "pairs": pairs,
            "features": features,
            "counts": counts,
            "infer_sec": infer_sec,
            "fragment_sec": fragment_sec,
            "rag_sec": rag_sec,
            "crop": np.array([[sl.start, sl.stop] for sl in crop], dtype=np.int64),
        }
        eval_feature_refs.append(features)
        sample_order.append(path.name)
        print(
            {
                "sample": path.name,
                "raw_shape": list(raw.shape),
                "num_fragments": int(fragments.max(initial=0)),
                "num_edges": int(pairs.size),
                "mean_boundary_count": float(np.mean(counts)) if counts.size else 0.0,
                "infer_sec": float(infer_sec),
                "fragment_sec": float(fragment_sec),
                "rag_sec": float(rag_sec),
            },
            flush=True,
        )

    train_features = np.concatenate(train_features_parts, axis=0).astype(np.float32, copy=False)
    train_targets = np.concatenate(train_targets_parts, axis=0).astype(np.float32, copy=False)
    train_features_std, eval_features_std, standardization = _standardize_train_eval(train_features, eval_feature_refs)
    for name, standardized in zip(sample_order, eval_features_std):
        cache[name]["features_std"] = standardized

    scorer, history = train_edge_scorer(
        train_features_std,
        train_targets,
        device=device,
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        positive_weight_cap=float(args.positive_weight_cap),
    )
    torch.save(
        {
            "model": scorer.state_dict(),
            "config": vars(args),
            "feature_dim": int(train_features.shape[1]),
            "standardization": standardization,
            "history": history,
        },
        output_dir / "sparse_edge_scorer.pt",
    )

    records: list[dict[str, float | int | str]] = []
    edge_records: list[dict[str, float | int | str]] = []
    scorer.eval()
    with torch.no_grad():
        for path in eval_files:
            item = cache[path.name]
            label = item["label"]  # type: ignore[assignment]
            fragments = np.asarray(item["fragments"])  # type: ignore[arg-type]
            pairs = np.asarray(item["pairs"])  # type: ignore[arg-type]
            features = np.asarray(item["features"], dtype=np.float32)  # type: ignore[arg-type]
            features_std = np.asarray(item["features_std"], dtype=np.float32)  # type: ignore[arg-type]
            metric_label, boundary_ignore_fraction = apply_cremi_boundary_ignore(
                label,  # type: ignore[arg-type]
                ignore_label=int(args.ignore_label),
                distance_xy=int(args.cremi_boundary_ignore_distance_xy),
                distance_z=int(args.cremi_boundary_ignore_distance_z),
            )
            x = torch.from_numpy(features_std).to(device)
            t_score = time.perf_counter()
            learned_scores = torch.sigmoid(scorer(x)).float().cpu().numpy()
            learned_score_sec = time.perf_counter() - t_score
            variants: list[tuple[str, np.ndarray, list[float], float]] = [
                ("learned_sparse_edge", learned_scores, [float(v) for v in args.merge_thresholds], learned_score_sec)
            ]
            for mode in args.baseline_score_modes:
                variants.append(
                    (
                        f"baseline_{mode}",
                        baseline_scores(features, str(mode)).astype(np.float32, copy=False),
                        [float(v) for v in args.baseline_thresholds],
                        0.0,
                    )
                )
            if path.name in train_names:
                targets = rag_pair_labels_from_ground_truth(
                    pairs,
                    fragments,
                    label,  # type: ignore[arg-type]
                    ignore_label=int(args.ignore_label),
                    positive_fraction=float(args.positive_fraction),
                )
            else:
                targets = None
            for variant_name, scores, thresholds, score_sec in variants:
                if targets is not None and targets.size:
                    for threshold in thresholds:
                        pred = scores >= float(threshold)
                        tp = float(np.sum(pred & (targets > 0.5)))
                        fp = float(np.sum(pred & (targets <= 0.5)))
                        fn = float(np.sum((~pred) & (targets > 0.5)))
                        tn = float(np.sum((~pred) & (targets <= 0.5)))
                        edge_records.append(
                            {
                                "sample": path.name,
                                "variant": variant_name,
                                "threshold": float(threshold),
                                "edge_precision": tp / max(tp + fp, 1.0),
                                "edge_recall": tp / max(tp + fn, 1.0),
                                "edge_acc": (tp + tn) / max(tp + tn + fp + fn, 1.0),
                                "positive_fraction": float(targets.mean()),
                            }
                        )
                for threshold in thresholds:
                    t_merge = time.perf_counter()
                    seg = agglomerate_fragments_by_scores(
                        fragments,
                        pairs,
                        scores,
                        threshold=float(threshold),
                    )
                    merge_sec = time.perf_counter() - t_merge
                    t_metric = time.perf_counter()
                    metrics = segmentation_metrics(
                        seg,
                        metric_label,
                        ignore_label=int(args.ignore_label),
                        backend=str(args.metric_backend),
                    )
                    metric_sec = time.perf_counter() - t_metric
                    record: dict[str, float | int | str] = {
                        "sample": path.name,
                        "split": "train" if path.name in train_names else "holdout",
                        "variant": variant_name,
                        "threshold": float(threshold),
                        "num_edges": int(pairs.size),
                        "num_fragments": int(fragments.max(initial=0)),
                        "inference_sec": float(item["infer_sec"]),  # type: ignore[arg-type]
                        "fragment_sec": float(item["fragment_sec"]),  # type: ignore[arg-type]
                        "rag_sec": float(item["rag_sec"]),  # type: ignore[arg-type]
                        "score_sec": float(score_sec),
                        "merge_sec": float(merge_sec),
                        "metrics_sec": float(metric_sec),
                        "boundary_ignore_fraction": float(boundary_ignore_fraction),
                        **metric_dict(metrics),
                    }
                    print(record, flush=True)
                    records.append(record)

    if records:
        with (output_dir / "sparse_edge_postprocess_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    if edge_records:
        with (output_dir / "sparse_edge_classifier_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(edge_records[0].keys()))
            writer.writeheader()
            writer.writerows(edge_records)

    group_keys = ["variant", "threshold"]
    holdout_grouped: dict[tuple[object, ...], list[dict[str, float | int | str]]] = {}
    all_grouped: dict[tuple[object, ...], list[dict[str, float | int | str]]] = {}
    for record in records:
        key = tuple(record[k] for k in group_keys)
        all_grouped.setdefault(key, []).append(record)
        if record["split"] == "holdout":
            holdout_grouped.setdefault(key, []).append(record)

    def summarize_grouped(grouped: dict[tuple[object, ...], list[dict[str, float | int | str]]]) -> list[dict[str, float | int | str]]:
        out: list[dict[str, float | int | str]] = []
        for key_tuple, rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
            row: dict[str, float | int | str] = dict(zip(group_keys, key_tuple))
            row["n"] = len(rows)
            for metric in [
                "adapted_rand_error",
                "rand_fscore",
                "rand_precision",
                "rand_recall",
                "voi_split",
                "voi_merge",
                "voi_sum",
                "score_sec",
                "merge_sec",
            ]:
                row[metric] = float(np.mean([float(r[metric]) for r in rows])) if rows else float("nan")
            out.append(row)
        return out

    per_group = summarize_grouped(all_grouped)
    holdout_per_group = summarize_grouped(holdout_grouped)
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "crop_size": list(crop_size),
        "stride": list(stride),
        "history": history,
        "standardization": standardization,
        "num_records": len(records),
        "num_edge_records": len(edge_records),
        "per_group": per_group,
        "holdout_per_group": holdout_per_group,
        "edge_records": edge_records,
    }
    if per_group:
        summary["best_by_voi_sum"] = min(per_group, key=lambda row: float(row["voi_sum"]))
        summary["best_by_adapted_rand"] = min(per_group, key=lambda row: float(row["adapted_rand_error"]))
    if holdout_per_group:
        summary["holdout_best_by_voi_sum"] = min(holdout_per_group, key=lambda row: float(row["voi_sum"]))
        summary["holdout_best_by_adapted_rand"] = min(
            holdout_per_group,
            key=lambda row: float(row["adapted_rand_error"]),
        )
    (output_dir / "sparse_edge_postprocess_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "sparse_edge_postprocess_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
