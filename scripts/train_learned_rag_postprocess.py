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

from dbmim.models import LearnedRAGMergeScorer
from dbmim.postprocess import (
    agglomerate_fragments_by_scores,
    rag_boundary_features,
    rag_pair_labels_from_ground_truth,
    segmentation_metrics,
    watershed_fragments,
)
from dbmim.utils import ensure_dir, load_config
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

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


def train_scorer(
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
) -> LearnedRAGMergeScorer:
    model = LearnedRAGMergeScorer(features.shape[1], hidden_dim=hidden_dim, depth=depth, dropout=dropout).to(device)
    x = torch.from_numpy(features.astype(np.float32, copy=False)).to(device)
    y = torch.from_numpy(targets.astype(np.float32, copy=False)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    pos = float(targets.sum())
    neg = float(targets.size - targets.sum())
    pos_weight = torch.tensor([max(1.0, neg / max(pos, 1.0))], device=device)
    n = int(x.shape[0])
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
            with torch.no_grad():
                pred = (torch.sigmoid(model(x)) >= 0.5).float()
                acc = float((pred == y).float().mean().cpu())
            print({"epoch": epoch + 1, "loss": float(np.mean(losses)), "edge_acc": acc}, flush=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a learned RAG merge postprocessor.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--stride", nargs=3, type=int, default=[16, 80, 80])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--train-samples", nargs="+", default=["sample_A_20160501.hdf", "sample_B_20160501.hdf"])
    parser.add_argument("--eval-samples", nargs="+", default=["sample_A_20160501.hdf", "sample_B_20160501.hdf", "sample_C_20160501.hdf"])
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--seed-distance", type=int, default=10)
    parser.add_argument("--min-boundary", type=int, default=4)
    parser.add_argument("--merge-thresholds", nargs="+", type=float, default=[0.30, 0.40, 0.50, 0.60, 0.70])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8192)
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

    files = list_cremi_files(args.data_dir)
    if int(args.max_samples) > 0:
        files = files[: int(args.max_samples)]
    files_by_name = {path.name: path for path in files}
    train_files = [files_by_name[name] for name in args.train_samples if name in files_by_name]
    eval_files = [files_by_name[name] for name in args.eval_samples if name in files_by_name]
    train_names = {path.name for path in train_files}
    if not train_files:
        raise ValueError(f"no train samples matched {args.train_samples}")
    if not eval_files:
        raise ValueError(f"no eval samples matched {args.eval_samples}")

    cache: dict[str, dict[str, object]] = {}
    feature_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
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
        pairs, features, counts = rag_boundary_features(aff, fragments, min_boundary=int(args.min_boundary))
        cache[path.name] = {
            "label": label,
            "aff": aff,
            "fragments": fragments,
            "pairs": pairs,
            "features": features,
            "counts": counts,
            "infer_sec": infer_sec,
            "fragment_sec": fragment_sec,
        }
        print(
            {
                "sample": path.name,
                "raw_shape": list(raw.shape),
                "pairs": int(pairs.size),
                "infer_sec": float(infer_sec),
                "fragment_sec": float(fragment_sec),
            },
            flush=True,
        )
        if path in train_files:
            targets = rag_pair_labels_from_ground_truth(
                pairs,
                fragments,
                label,
                ignore_label=int(args.ignore_label),
            )
            feature_parts.append(features)
            target_parts.append(targets)
            print(
                {
                    "sample": path.name,
                    "train_pairs": int(targets.size),
                    "positive_fraction": float(targets.mean()) if targets.size else 0.0,
                },
                flush=True,
            )
    train_features = np.concatenate(feature_parts, axis=0).astype(np.float32, copy=False)
    train_targets = np.concatenate(target_parts, axis=0).astype(np.float32, copy=False)
    scorer = train_scorer(
        train_features,
        train_targets,
        device=device,
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
    )
    torch.save(
        {
            "model": scorer.state_dict(),
            "config": vars(args),
            "feature_dim": int(train_features.shape[1]),
        },
        output_dir / "learned_rag_scorer.pt",
    )

    records = []
    scorer.eval()
    with torch.no_grad():
        for path in eval_files:
            item = cache[path.name]
            features = item["features"]  # type: ignore[assignment]
            pairs = item["pairs"]  # type: ignore[assignment]
            fragments = item["fragments"]  # type: ignore[assignment]
            label = item["label"]  # type: ignore[assignment]
            metric_label, boundary_ignore_fraction = apply_cremi_boundary_ignore(
                label,  # type: ignore[arg-type]
                ignore_label=int(args.ignore_label),
                distance_xy=int(args.cremi_boundary_ignore_distance_xy),
                distance_z=int(args.cremi_boundary_ignore_distance_z),
            )
            x = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(device)
            t0 = time.perf_counter()
            scores = torch.sigmoid(scorer(x)).float().cpu().numpy()
            score_sec = time.perf_counter() - t0
            for threshold in args.merge_thresholds:
                t1 = time.perf_counter()
                seg = agglomerate_fragments_by_scores(
                    np.asarray(fragments),
                    np.asarray(pairs),
                    scores,
                    threshold=float(threshold),
                )
                merge_sec = time.perf_counter() - t1
                t2 = time.perf_counter()
                metrics = segmentation_metrics(
                    seg,
                    metric_label,
                    ignore_label=int(args.ignore_label),
                    backend=args.metric_backend,
                )
                metrics_sec = time.perf_counter() - t2
                record = {
                    "sample": path.name,
                    "split": "postprocess_train" if path.name in train_names else "postprocess_holdout",
                    "backend": "learned_rag",
                    "threshold": float(threshold),
                    "boundary_threshold": float(args.boundary_threshold),
                    "seed_distance": int(args.seed_distance),
                    "min_boundary": int(args.min_boundary),
                    "n_pairs": int(np.asarray(pairs).size),
                    "boundary_ignore_fraction": float(boundary_ignore_fraction),
                    "inference_sec": float(item["infer_sec"]),
                    "fragment_sec": float(item["fragment_sec"]),
                    "score_sec": float(score_sec),
                    "postprocess_sec": float(score_sec + merge_sec),
                    "metrics_sec": float(metrics_sec),
                    "adapted_rand_error": metrics.adapted_rand_error,
                    "voi_split": metrics.voi_split,
                    "voi_merge": metrics.voi_merge,
                    "voi_sum": metrics.voi_split + metrics.voi_merge,
                    "rand_fscore": metrics.rand_fscore,
                    "rand_precision": metrics.rand_precision,
                    "rand_recall": metrics.rand_recall,
                }
                print(record, flush=True)
                records.append(record)
    summary_rows = []
    holdout_rows = []
    for threshold in args.merge_thresholds:
        rows = [row for row in records if float(row["threshold"]) == float(threshold)]
        if not rows:
            continue
        summary_rows.append(
            {
                "backend": "learned_rag",
                "threshold": float(threshold),
                "n": len(rows),
                "adapted_rand_error": float(np.mean([float(r["adapted_rand_error"]) for r in rows])),
                "voi_sum": float(np.mean([float(r["voi_sum"]) for r in rows])),
                "postprocess_sec": float(np.mean([float(r["postprocess_sec"]) for r in rows])),
            }
        )
        held = [row for row in rows if row.get("split") == "postprocess_holdout"]
        if held:
            holdout_rows.append(
                {
                    "backend": "learned_rag",
                    "threshold": float(threshold),
                    "n": len(held),
                    "adapted_rand_error": float(np.mean([float(r["adapted_rand_error"]) for r in held])),
                    "voi_sum": float(np.mean([float(r["voi_sum"]) for r in held])),
                    "postprocess_sec": float(np.mean([float(r["postprocess_sec"]) for r in held])),
                }
            )
    summary = {
        "records": len(records),
        "train_pairs": int(train_targets.size),
        "train_positive_fraction": float(train_targets.mean()) if train_targets.size else 0.0,
        "per_threshold": summary_rows,
        "holdout_per_threshold": holdout_rows,
    }
    if summary_rows:
        summary["best_by_voi_sum"] = min(summary_rows, key=lambda row: float(row["voi_sum"]))
        summary["best_by_adapted_rand"] = min(summary_rows, key=lambda row: float(row["adapted_rand_error"]))
    if holdout_rows:
        summary["holdout_best_by_voi_sum"] = min(holdout_rows, key=lambda row: float(row["voi_sum"]))
        summary["holdout_best_by_adapted_rand"] = min(
            holdout_rows, key=lambda row: float(row["adapted_rand_error"])
        )
    (output_dir / "learned_rag_records.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    (output_dir / "learned_rag_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
