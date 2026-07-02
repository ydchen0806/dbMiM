#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
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
    agglomerate_fragments_by_affinity,
    agglomerate_fragments_by_scores,
    cupy_agglomerate_fragments_by_scores,
    rag_boundary_features,
    rag_pair_labels_from_ground_truth,
    segmentation_metrics,
    waterz_agglomeration,
    waterz_agglomerations,
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
    read_cremi_crop,
    sigmoid_np,
    predict_affinities,
)


@dataclass
class RagSample:
    sample: str
    affinities: np.ndarray
    labels: np.ndarray
    metric_labels: np.ndarray
    fragments: np.ndarray
    packed_pairs: np.ndarray
    features: np.ndarray
    edge_targets: np.ndarray | None
    timings: dict[str, float]


def parse_triplet(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"expected one z/y/x triplet, got {len(values)} values")
    return float(values[0]), float(values[1]), float(values[2])


def select_named_files(files: list[Path], names: list[str]) -> list[Path]:
    if not names or names == ["all"]:
        return files
    by_name = {path.name: path for path in files}
    by_stem = {path.stem: path for path in files}
    selected: list[Path] = []
    for name in names:
        path = by_name.get(name) or by_stem.get(Path(name).stem)
        if path is None:
            raise FileNotFoundError(f"requested sample {name!r} not found; available={[p.name for p in files]}")
        selected.append(path)
    return selected


def calibrated_affinities(logits: np.ndarray, bias: tuple[float, float, float], temperature: float) -> np.ndarray:
    bias_arr = np.asarray(bias, dtype=np.float32).reshape(3, 1, 1, 1)
    return sigmoid_np((logits + bias_arr) / max(float(temperature), 1e-6)).astype(np.float32, copy=False)


def append_fragment_context_features(fragments: np.ndarray, packed_pairs: np.ndarray, features: np.ndarray) -> np.ndarray:
    fragments = np.asarray(fragments, dtype=np.uint64)
    packed_pairs = np.asarray(packed_pairs, dtype=np.int64)
    features = np.asarray(features, dtype=np.float32)
    if packed_pairs.size == 0:
        return features
    max_id = int(fragments.max(initial=0))
    base = max_id + 1
    flat = fragments.reshape(-1).astype(np.int64, copy=False)
    sizes = np.bincount(flat, minlength=max_id + 1).astype(np.float32, copy=False)
    left = (packed_pairs // base).astype(np.int64, copy=False)
    right = (packed_pairs % base).astype(np.int64, copy=False)
    left_size = sizes[np.clip(left, 0, max_id)]
    right_size = sizes[np.clip(right, 0, max_id)]
    min_size = np.minimum(left_size, right_size)
    max_size = np.maximum(left_size, right_size)
    size_context = np.stack(
        [
            np.log1p(left_size),
            np.log1p(right_size),
            np.log1p(min_size),
            np.log1p(max_size),
            min_size / np.maximum(max_size, 1.0),
            features[:, 12] / np.maximum(np.log1p(min_size), 1e-6),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    return np.concatenate([features, size_context], axis=1)


def rag_pair_labels_from_teacher_segmentation(
    packed_pairs: np.ndarray,
    fragments: np.ndarray,
    teacher: np.ndarray,
    *,
    ignore_label: int | None = 0,
    positive_fraction: float = 0.5,
) -> np.ndarray:
    packed_pairs = np.asarray(packed_pairs, dtype=np.int64)
    fragments = np.asarray(fragments, dtype=np.uint64)
    teacher = np.asarray(teacher)
    if fragments.shape != teacher.shape:
        raise ValueError(f"fragment/teacher shape mismatch: {fragments.shape} vs {teacher.shape}")
    max_id = int(fragments.max(initial=0))
    base = max_id + 1
    flat_frag = fragments.reshape(-1).astype(np.int64, copy=False)
    flat_teacher = teacher.reshape(-1).astype(np.int64, copy=False)
    keep = (flat_frag > 0) & (flat_frag <= max_id) & (flat_teacher > 0)
    if ignore_label is not None:
        keep &= flat_teacher != int(ignore_label)
    major = np.zeros(max_id + 1, dtype=np.int64)
    confidence = np.zeros(max_id + 1, dtype=np.float32)
    if np.any(keep):
        kept_frag = flat_frag[keep]
        kept_teacher = flat_teacher[keep]
        teacher_base = int(kept_teacher.max(initial=0)) + 1
        packed = kept_frag * teacher_base + kept_teacher
        unique, counts = np.unique(packed, return_counts=True)
        frag_ids = unique // teacher_base
        teacher_ids = unique % teacher_base
        totals = np.bincount(kept_frag, minlength=max_id + 1).astype(np.float32, copy=False)
        order = np.lexsort((-counts, frag_ids))
        sorted_frag = frag_ids[order]
        first = np.r_[True, sorted_frag[1:] != sorted_frag[:-1]]
        best = order[first]
        best_frag = frag_ids[best].astype(np.int64, copy=False)
        best_teacher = teacher_ids[best].astype(np.int64, copy=False)
        major[best_frag] = best_teacher
        confidence[best_frag] = counts[best].astype(np.float32, copy=False) / np.maximum(totals[best_frag], 1.0)
    out = np.zeros(packed_pairs.shape, dtype=np.float32)
    for idx, packed in enumerate(packed_pairs):
        left = int(packed // base)
        right = int(packed % base)
        if left <= 0 or right <= 0 or left > max_id or right > max_id:
            continue
        if major[left] > 0 and major[left] == major[right]:
            out[idx] = 1.0 if min(confidence[left], confidence[right]) >= float(positive_fraction) else 0.0
    return out


@torch.no_grad()
def build_rag_sample(
    path: Path,
    *,
    cfg: dict,
    model: torch.nn.Module,
    postprocess: torch.nn.Module | None,
    model_window: tuple[int, int, int],
    crop_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    device: torch.device,
    calibration_bias: tuple[float, float, float],
    calibration_temperature: float,
    seed_distance: int,
    boundary_threshold: float,
    min_boundary: int,
    ignore_label: int,
    cremi_boundary_ignore_distance_xy: int,
    cremi_boundary_ignore_distance_z: int,
    positive_fraction: float,
    append_context_features: bool,
    target_mode: str,
    teacher_waterz_threshold: float,
    teacher_waterz_thresholds: tuple[float, ...],
    teacher_waterz_scoring: str,
    with_targets: bool,
) -> RagSample:
    data_cfg = cfg.get("data", {})
    raw_keys = data_cfg.get("image_keys") or ["volumes/raw", "raw", "main"]
    label_keys = data_cfg.get("label_keys") or ["volumes/labels/neuron_ids", "labels", "label", "gt"]
    raw, label, crop = read_cremi_crop(path, crop_size, raw_keys, label_keys)
    print({"sample": path.name, "raw_shape": list(raw.shape), "crop": [[sl.start, sl.stop] for sl in crop]}, flush=True)

    timings: dict[str, float] = {}
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
    affinities = calibrated_affinities(logits, calibration_bias, calibration_temperature)
    timings["inference_sec"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    fragments = watershed_fragments(
        affinities,
        backend="mahotas",
        seed_method="maxima_distance",
        seed_distance=int(seed_distance),
        boundary_threshold=float(boundary_threshold),
    )
    timings["fragment_sec"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    packed_pairs, features, counts = rag_boundary_features(affinities, fragments, min_boundary=int(min_boundary))
    if append_context_features:
        features = append_fragment_context_features(fragments, packed_pairs, features)
    timings["rag_feature_sec"] = time.perf_counter() - t2
    print(
        {
            "sample": path.name,
            "num_fragments": int(fragments.max(initial=0)),
            "num_rag_edges": int(packed_pairs.size),
            "mean_boundary_count": float(np.mean(counts)) if counts.size else 0.0,
            **timings,
        },
        flush=True,
    )

    metric_labels, boundary_ignore_fraction = apply_cremi_boundary_ignore(
        label,
        ignore_label=int(ignore_label),
        distance_xy=int(cremi_boundary_ignore_distance_xy),
        distance_z=int(cremi_boundary_ignore_distance_z),
    )
    timings["boundary_ignore_fraction"] = float(boundary_ignore_fraction)

    edge_targets = None
    if with_targets:
        t3 = time.perf_counter()
        if target_mode == "waterz":
            teacher_seg = waterz_agglomeration(
                affinities,
                threshold=float(teacher_waterz_threshold),
                fragments=fragments,
                scoring_function=str(teacher_waterz_scoring),
            )
            edge_targets = rag_pair_labels_from_teacher_segmentation(
                packed_pairs,
                fragments,
                teacher_seg,
                ignore_label=int(ignore_label),
                positive_fraction=float(positive_fraction),
            )
        elif target_mode == "waterz_ladder":
            if not teacher_waterz_thresholds:
                raise ValueError("target_mode=waterz_ladder requires at least one teacher threshold")
            teacher_segmentations = waterz_agglomerations(
                affinities,
                thresholds=list(teacher_waterz_thresholds),
                fragments=fragments,
                scoring_function=str(teacher_waterz_scoring),
            )
            soft_targets = np.zeros(packed_pairs.shape, dtype=np.float32)
            for teacher_seg in teacher_segmentations:
                soft_targets += rag_pair_labels_from_teacher_segmentation(
                    packed_pairs,
                    fragments,
                    teacher_seg,
                    ignore_label=int(ignore_label),
                    positive_fraction=float(positive_fraction),
                )
            edge_targets = soft_targets / max(1, len(teacher_segmentations))
        else:
            edge_targets = rag_pair_labels_from_ground_truth(
                packed_pairs,
                fragments,
                label,
                ignore_label=int(ignore_label),
                positive_fraction=float(positive_fraction),
            )
        timings["edge_label_sec"] = time.perf_counter() - t3
        print(
            {
                "sample": path.name,
                "target_mode": str(target_mode),
                "teacher_waterz_threshold": float(teacher_waterz_threshold) if target_mode == "waterz" else "",
                "teacher_waterz_thresholds": list(teacher_waterz_thresholds)
                if target_mode == "waterz_ladder"
                else "",
                "edge_positive_fraction": float(np.mean(edge_targets)) if edge_targets.size else 0.0,
                "edge_hard_positive_fraction": float(np.mean(edge_targets >= 0.5)) if edge_targets.size else 0.0,
                "edge_label_sec": timings["edge_label_sec"],
            },
            flush=True,
        )

    return RagSample(
        sample=path.name,
        affinities=affinities,
        labels=label,
        metric_labels=metric_labels,
        fragments=fragments,
        packed_pairs=packed_pairs,
        features=features,
        edge_targets=edge_targets,
        timings=timings,
    )


def balanced_edge_subset(targets: np.ndarray, max_edges: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    targets = np.asarray(targets, dtype=np.float32)
    indices = np.arange(targets.shape[0])
    if max_edges <= 0 or targets.shape[0] <= max_edges:
        rng.shuffle(indices)
        return indices
    pos = np.flatnonzero(targets >= 0.5)
    neg = np.flatnonzero(targets < 0.5)
    if pos.size == 0 or neg.size == 0:
        return rng.choice(indices, size=int(max_edges), replace=False)
    half = int(max_edges) // 2
    pos_take = min(pos.size, half)
    neg_take = min(neg.size, int(max_edges) - pos_take)
    if neg_take < int(max_edges) - pos_take and pos.size > pos_take:
        pos_take = min(pos.size, int(max_edges) - neg_take)
    chosen = np.concatenate(
        [
            rng.choice(pos, size=pos_take, replace=False),
            rng.choice(neg, size=neg_take, replace=False),
        ]
    )
    rng.shuffle(chosen)
    return chosen.astype(np.int64, copy=False)


def train_scorer(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    depth: int,
    dropout: float,
    seed: int,
) -> tuple[LearnedRAGMergeScorer, np.ndarray, np.ndarray, dict[str, float | int]]:
    torch.manual_seed(int(seed))
    mean = features.mean(axis=0).astype(np.float32)
    std = np.maximum(features.std(axis=0).astype(np.float32), 1e-6)
    x_np = ((features - mean) / std).astype(np.float32, copy=False)
    y_np = targets.astype(np.float32, copy=False)
    model = LearnedRAGMergeScorer(
        in_features=int(x_np.shape[1]),
        hidden_dim=int(hidden_dim),
        depth=int(depth),
        dropout=float(dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    hard_pos = float(np.sum(y_np >= 0.5))
    hard_neg = float(np.sum(y_np < 0.5))
    pos_mass = float(np.sum(y_np))
    neg_mass = float(np.sum(1.0 - y_np))
    pos_weight = torch.tensor([neg_mass / max(pos_mass, 1.0)], dtype=torch.float32, device=device)
    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    rng = np.random.default_rng(int(seed))
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(epochs) + 1):
        order = rng.permutation(x.shape[0])
        losses: list[float] = []
        model.train()
        for start in range(0, order.size, int(batch_size)):
            batch_idx = order[start : start + int(batch_size)]
            xb = x[batch_idx].to(device, non_blocking=True)
            yb = y[batch_idx].to(device, non_blocking=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch % max(1, int(epochs) // 10) == 0 or epoch == int(epochs):
            model.eval()
            with torch.no_grad():
                eval_idx = order[: min(order.size, 100000)]
                logits = model(x[eval_idx].to(device)).float().cpu()
                probs = torch.sigmoid(logits).numpy()
                yy = y_np[eval_idx]
            row = {
                "epoch": int(epoch),
                "loss": float(np.mean(losses)) if losses else 0.0,
                "edge_prob_pos_mean": float(np.mean(probs[yy >= 0.5])) if np.any(yy >= 0.5) else 0.0,
                "edge_prob_neg_mean": float(np.mean(probs[yy < 0.5])) if np.any(yy < 0.5) else 0.0,
                "edge_accuracy_0p5": float(np.mean((probs >= 0.5) == (yy >= 0.5))) if yy.size else 0.0,
            }
            print(row, flush=True)
            history.append(row)
    stats: dict[str, float | int] = {
        "num_edges": int(y_np.size),
        "num_positive": int(hard_pos),
        "num_negative": int(hard_neg),
        "positive_fraction": float(hard_pos / max(hard_pos + hard_neg, 1.0)),
        "soft_positive_mass": float(pos_mass),
        "soft_negative_mass": float(neg_mass),
        "soft_positive_fraction": float(pos_mass / max(pos_mass + neg_mass, 1.0)),
        "pos_weight": float(pos_weight.item()),
        "final_loss": history[-1]["loss"] if history else 0.0,
        "final_edge_accuracy_0p5": history[-1]["edge_accuracy_0p5"] if history else 0.0,
    }
    return model, mean, std, stats


@torch.no_grad()
def score_edges(
    model: LearnedRAGMergeScorer,
    features: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    x_np = ((features - mean) / std).astype(np.float32, copy=False)
    scores: list[np.ndarray] = []
    for start in range(0, x_np.shape[0], int(batch_size)):
        xb = torch.from_numpy(x_np[start : start + int(batch_size)]).to(device)
        probs = torch.sigmoid(model(xb)).float().cpu().numpy()
        scores.append(probs.astype(np.float32, copy=False))
    return np.concatenate(scores) if scores else np.empty((0,), dtype=np.float32)


def boundary_mean_from_features(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] < 13:
        raise ValueError(f"expected RAG features with at least 13 columns, got {features.shape}")
    axis_ratio = np.clip(features[:, 9:12], 0.0, 1.0)
    denom = np.maximum(axis_ratio.sum(axis=1), 1e-6)
    return (features[:, 0:3] * axis_ratio).sum(axis=1) / denom


def signed_veto_score_variant(
    edge_scores: np.ndarray,
    features: np.ndarray,
    *,
    mode: str,
    mean_threshold: float,
    count_log_min: float,
    penalty: float,
) -> tuple[np.ndarray, float]:
    """Apply a simple signed-graph repulsive cue to learned merge scores.

    Low affinity across a sufficiently large fragment boundary is treated as a
    cannot-link cue. ``hard`` mode removes those edges; ``penalty`` mode lowers
    their scores so lower merge thresholds can still be explored.
    """

    edge_scores = np.asarray(edge_scores, dtype=np.float32)
    boundary_mean = boundary_mean_from_features(features)
    boundary_count_log = np.asarray(features[:, 12], dtype=np.float32)
    weak_boundary = (boundary_mean < float(mean_threshold)) & (boundary_count_log >= float(count_log_min))
    if mode == "hard":
        adjusted = edge_scores.copy()
        adjusted[weak_boundary] = -np.inf
    elif mode == "penalty":
        count_conf = np.clip(boundary_count_log / max(float(count_log_min), 1e-6), 0.0, 2.0)
        strength = np.clip(float(mean_threshold) - boundary_mean, 0.0, 1.0) * count_conf
        adjusted = edge_scores - float(penalty) * strength.astype(np.float32, copy=False)
    else:
        raise ValueError(f"unknown signed veto mode: {mode}")
    return adjusted.astype(np.float32, copy=False), float(np.mean(weak_boundary)) if weak_boundary.size else 0.0


def signed_reweight_score_variant(
    edge_scores: np.ndarray,
    features: np.ndarray,
    *,
    mode: str,
    alpha: float,
    gamma: float,
) -> tuple[np.ndarray, float]:
    """Continuously reweight merge scores with the signed boundary cue.

    The earlier veto variant only touched edges below a fixed boundary-mean
    threshold. This variant changes every candidate edge score, so it can alter
    the agglomeration order even when the weak-boundary veto set is sparse.
    """

    edge_scores = np.asarray(edge_scores, dtype=np.float32)
    boundary_mean = np.clip(boundary_mean_from_features(features), 0.0, 1.0).astype(np.float32, copy=False)
    if mode == "subtract":
        adjusted = edge_scores - float(alpha) * (1.0 - boundary_mean)
    elif mode == "multiply":
        adjusted = edge_scores * np.power(np.maximum(boundary_mean, 1e-6), float(gamma)).astype(
            np.float32, copy=False
        )
    elif mode == "blend":
        adjusted = (1.0 - float(alpha)) * edge_scores + float(alpha) * boundary_mean
    else:
        raise ValueError(f"unknown signed reweight mode: {mode}")
    changed_fraction = float(np.mean(np.abs(adjusted - edge_scores) > 1e-6)) if edge_scores.size else 0.0
    return adjusted.astype(np.float32, copy=False), changed_fraction


def _mean(rows: list[dict[str, float | int | str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a learned RAG merge scorer on CREMI affinities.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", nargs=3, type=int, default=[64, 512, 512])
    parser.add_argument("--eval-crop-size", nargs=3, type=int, default=None)
    parser.add_argument("--stride", nargs=3, type=int, default=[16, 80, 80])
    parser.add_argument("--train-samples", nargs="+", default=["sample_A_20160501.hdf", "sample_B_20160501.hdf"])
    parser.add_argument("--eval-samples", nargs="+", default=["all"])
    parser.add_argument("--calibration-biases", nargs=3, type=float, default=[-0.25, -0.50, -0.50])
    parser.add_argument("--calibration-temperature", type=float, default=1.0)
    parser.add_argument("--seed-distance", type=int, default=10)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--min-boundary", type=int, default=4)
    parser.add_argument("--positive-fraction", type=float, default=0.5)
    parser.add_argument("--target-mode", choices=["gt", "waterz", "waterz_ladder"], default="gt")
    parser.add_argument("--teacher-waterz-threshold", type=float, default=0.20)
    parser.add_argument("--teacher-waterz-thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--teacher-waterz-scoring", default="hist_quantile")
    parser.add_argument("--max-train-edges", type=int, default=600000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--append-context-features", action="store_true")
    parser.add_argument("--merge-backends", nargs="+", choices=["cpu", "gpu"], default=["cpu"])
    parser.add_argument("--merge-thresholds", nargs="+", type=float, default=[0.30, 0.40, 0.50, 0.60, 0.70])
    parser.add_argument("--signed-veto-modes", nargs="+", choices=["hard", "penalty"], default=[])
    parser.add_argument("--signed-veto-mean-thresholds", nargs="+", type=float, default=[])
    parser.add_argument("--signed-veto-count-log-mins", nargs="+", type=float, default=[0.0])
    parser.add_argument("--signed-veto-penalties", nargs="+", type=float, default=[1.0])
    parser.add_argument("--signed-reweight-modes", nargs="+", choices=["subtract", "multiply", "blend"], default=[])
    parser.add_argument("--signed-reweight-alphas", nargs="+", type=float, default=[])
    parser.add_argument("--signed-reweight-gammas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--baseline-z-thresholds", nargs="+", type=float, default=[0.20, 0.30])
    parser.add_argument("--baseline-xy-thresholds", nargs="+", type=float, default=[0.20, 0.30])
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--score-mode", nargs="+", default=["mean", "q25"])
    parser.add_argument("--rag-quantile", type=float, default=0.25)
    parser.add_argument("--metric-backend", choices=["internal", "skimage"], default="skimage")
    parser.add_argument("--ignore-label", type=int, default=0)
    parser.add_argument("--cremi-boundary-ignore-distance-xy", type=int, default=1)
    parser.add_argument("--cremi-boundary-ignore-distance-z", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=309)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(cfg, Path(args.checkpoint), device)
    postprocess = build_postprocess(cfg, Path(args.checkpoint), device)
    model_window = tuple(int(v) for v in model.volume_size)
    crop_size = normalize_crop_size(tuple(int(v) for v in args.crop_size), model_window)
    eval_crop_size = (
        normalize_crop_size(tuple(int(v) for v in args.eval_crop_size), model_window)
        if args.eval_crop_size is not None
        else crop_size
    )
    stride = tuple(int(v) for v in args.stride)
    calibration_bias = parse_triplet([float(v) for v in args.calibration_biases])
    teacher_waterz_thresholds = tuple(
        float(v) for v in (args.teacher_waterz_thresholds or [args.teacher_waterz_threshold])
    )
    files = list_cremi_files(args.data_dir)
    train_files = select_named_files(files, list(args.train_samples))
    eval_files = select_named_files(files, list(args.eval_samples))

    train_features: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    sample_summaries: list[dict[str, float | int | str]] = []
    for path in train_files:
        sample = build_rag_sample(
            path,
            cfg=cfg,
            model=model,
            postprocess=postprocess,
            model_window=model_window,
            crop_size=crop_size,
            stride=stride,
            device=device,
            calibration_bias=calibration_bias,
            calibration_temperature=float(args.calibration_temperature),
            seed_distance=int(args.seed_distance),
            boundary_threshold=float(args.boundary_threshold),
            min_boundary=int(args.min_boundary),
            ignore_label=int(args.ignore_label),
            cremi_boundary_ignore_distance_xy=int(args.cremi_boundary_ignore_distance_xy),
            cremi_boundary_ignore_distance_z=int(args.cremi_boundary_ignore_distance_z),
            positive_fraction=float(args.positive_fraction),
            append_context_features=bool(args.append_context_features),
            target_mode=str(args.target_mode),
            teacher_waterz_threshold=float(args.teacher_waterz_threshold),
            teacher_waterz_thresholds=teacher_waterz_thresholds,
            teacher_waterz_scoring=str(args.teacher_waterz_scoring),
            with_targets=True,
        )
        if sample.edge_targets is not None and sample.features.size:
            train_features.append(sample.features)
            train_targets.append(sample.edge_targets)
        sample_summaries.append(
            {
                "sample": sample.sample,
                "split": "train",
                "num_edges": int(sample.features.shape[0]),
                "positive_fraction": float(np.mean(sample.edge_targets)) if sample.edge_targets is not None else 0.0,
                **sample.timings,
            }
        )
    if not train_features:
        raise RuntimeError("no train RAG edges were generated")
    all_features = np.concatenate(train_features, axis=0).astype(np.float32, copy=False)
    all_targets = np.concatenate(train_targets, axis=0).astype(np.float32, copy=False)
    selected = balanced_edge_subset(all_targets, int(args.max_train_edges), int(args.seed))
    train_x = all_features[selected]
    train_y = all_targets[selected]
    scorer, feature_mean, feature_std, train_stats = train_scorer(
        train_x,
        train_y,
        device=device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
        seed=int(args.seed),
    )
    torch.save(
        {
            "model": scorer.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "args": vars(args),
            "train_stats": train_stats,
        },
        output_dir / "learned_rag_scorer.pt",
    )

    records: list[dict[str, float | int | str]] = []
    for path in eval_files:
        sample = build_rag_sample(
            path,
            cfg=cfg,
            model=model,
            postprocess=postprocess,
            model_window=model_window,
            crop_size=eval_crop_size,
            stride=stride,
            device=device,
            calibration_bias=calibration_bias,
            calibration_temperature=float(args.calibration_temperature),
            seed_distance=int(args.seed_distance),
            boundary_threshold=float(args.boundary_threshold),
            min_boundary=int(args.min_boundary),
            ignore_label=int(args.ignore_label),
            cremi_boundary_ignore_distance_xy=int(args.cremi_boundary_ignore_distance_xy),
            cremi_boundary_ignore_distance_z=int(args.cremi_boundary_ignore_distance_z),
            positive_fraction=float(args.positive_fraction),
            append_context_features=bool(args.append_context_features),
            target_mode=str(args.target_mode),
            teacher_waterz_threshold=float(args.teacher_waterz_threshold),
            teacher_waterz_thresholds=teacher_waterz_thresholds,
            teacher_waterz_scoring=str(args.teacher_waterz_scoring),
            with_targets=False,
        )
        sample_summaries.append(
            {
                "sample": sample.sample,
                "split": "eval",
                "num_edges": int(sample.features.shape[0]),
                **sample.timings,
            }
        )
        t_score = time.perf_counter()
        edge_scores = score_edges(
            scorer,
            sample.features,
            feature_mean,
            feature_std,
            device=device,
            batch_size=int(args.batch_size),
        )
        score_sec = time.perf_counter() - t_score
        score_variants: list[dict[str, object]] = [
            {
                "scores": edge_scores,
                "veto_mode": "",
                "veto_mean_threshold": "",
                "veto_count_log_min": "",
                "veto_penalty": "",
                "veto_fraction": "",
                "reweight_mode": "",
                "reweight_alpha": "",
                "reweight_gamma": "",
                "reweight_changed_fraction": "",
                "backend_suffix": "",
            }
        ]
        if args.signed_veto_modes and args.signed_veto_mean_thresholds:
            for veto_mode in args.signed_veto_modes:
                for veto_mean_threshold in args.signed_veto_mean_thresholds:
                    for veto_count_log_min in args.signed_veto_count_log_mins:
                        penalties = args.signed_veto_penalties if veto_mode == "penalty" else [0.0]
                        for veto_penalty in penalties:
                            adjusted_scores, veto_fraction = signed_veto_score_variant(
                                edge_scores,
                                sample.features,
                                mode=str(veto_mode),
                                mean_threshold=float(veto_mean_threshold),
                                count_log_min=float(veto_count_log_min),
                                penalty=float(veto_penalty),
                            )
                            score_variants.append(
                                {
                                    "scores": adjusted_scores,
                                    "veto_mode": str(veto_mode),
                                    "veto_mean_threshold": float(veto_mean_threshold),
                                    "veto_count_log_min": float(veto_count_log_min),
                                    "veto_penalty": float(veto_penalty) if veto_mode == "penalty" else "",
                                    "veto_fraction": float(veto_fraction),
                                    "reweight_mode": "",
                                    "reweight_alpha": "",
                                    "reweight_gamma": "",
                                    "reweight_changed_fraction": "",
                                    "backend_suffix": "_signed_veto",
                                }
                            )
        if args.signed_reweight_modes:
            for reweight_mode in args.signed_reweight_modes:
                alphas = args.signed_reweight_alphas if reweight_mode in {"subtract", "blend"} else [0.0]
                gammas = args.signed_reweight_gammas if reweight_mode == "multiply" else [1.0]
                for reweight_alpha in alphas:
                    for reweight_gamma in gammas:
                        adjusted_scores, changed_fraction = signed_reweight_score_variant(
                            edge_scores,
                            sample.features,
                            mode=str(reweight_mode),
                            alpha=float(reweight_alpha),
                            gamma=float(reweight_gamma),
                        )
                        score_variants.append(
                            {
                                "scores": adjusted_scores,
                                "veto_mode": "",
                                "veto_mean_threshold": "",
                                "veto_count_log_min": "",
                                "veto_penalty": "",
                                "veto_fraction": "",
                                "reweight_mode": str(reweight_mode),
                                "reweight_alpha": float(reweight_alpha) if reweight_mode in {"subtract", "blend"} else "",
                                "reweight_gamma": float(reweight_gamma) if reweight_mode == "multiply" else "",
                                "reweight_changed_fraction": float(changed_fraction),
                                "backend_suffix": "_signed_reweight",
                            }
                        )
        for merge_backend in args.merge_backends:
            for variant in score_variants:
                variant_scores = np.asarray(variant["scores"], dtype=np.float32)
                for threshold in args.merge_thresholds:
                    t0 = time.perf_counter()
                    try:
                        if merge_backend == "gpu":
                            seg = cupy_agglomerate_fragments_by_scores(
                                sample.fragments,
                                sample.packed_pairs,
                                variant_scores,
                                threshold=float(threshold),
                            )
                        else:
                            seg = agglomerate_fragments_by_scores(
                                sample.fragments,
                                sample.packed_pairs,
                                variant_scores,
                                threshold=float(threshold),
                            )
                    except Exception as exc:
                        row = {
                            "sample": sample.sample,
                            "backend": f"learned_rag_{merge_backend}{variant['backend_suffix']}",
                            "threshold": float(threshold),
                            "z_threshold": "",
                            "xy_threshold": "",
                            "score_mode": "mlp",
                            "rag_quantile": "",
                            "veto_mode": variant["veto_mode"],
                            "veto_mean_threshold": variant["veto_mean_threshold"],
                            "veto_count_log_min": variant["veto_count_log_min"],
                            "veto_penalty": variant["veto_penalty"],
                            "veto_fraction": variant["veto_fraction"],
                            "reweight_mode": variant["reweight_mode"],
                            "reweight_alpha": variant["reweight_alpha"],
                            "reweight_gamma": variant["reweight_gamma"],
                            "reweight_changed_fraction": variant["reweight_changed_fraction"],
                            "postprocess_sec": float(time.perf_counter() - t0),
                            "metrics_sec": 0.0,
                            "error": repr(exc),
                        }
                        print(row, flush=True)
                        records.append(row)
                        continue
                    postprocess_sec = time.perf_counter() - t0 + score_sec / max(
                        1,
                        len(args.merge_thresholds) * len(args.merge_backends) * len(score_variants),
                    )
                    t1 = time.perf_counter()
                    metrics = segmentation_metrics(
                        seg,
                        sample.metric_labels,
                        ignore_label=int(args.ignore_label),
                        backend=str(args.metric_backend),
                    )
                    metrics_sec = time.perf_counter() - t1
                    row = {
                        "sample": sample.sample,
                        "backend": f"learned_rag_{merge_backend}{variant['backend_suffix']}",
                        "threshold": float(threshold),
                        "z_threshold": "",
                        "xy_threshold": "",
                        "score_mode": "mlp",
                        "rag_quantile": "",
                        "veto_mode": variant["veto_mode"],
                        "veto_mean_threshold": variant["veto_mean_threshold"],
                        "veto_count_log_min": variant["veto_count_log_min"],
                        "veto_penalty": variant["veto_penalty"],
                        "veto_fraction": variant["veto_fraction"],
                        "reweight_mode": variant["reweight_mode"],
                        "reweight_alpha": variant["reweight_alpha"],
                        "reweight_gamma": variant["reweight_gamma"],
                        "reweight_changed_fraction": variant["reweight_changed_fraction"],
                        "postprocess_sec": float(postprocess_sec),
                        "metrics_sec": float(metrics_sec),
                        **metric_dict(metrics),
                    }
                    print(row, flush=True)
                    records.append(row)

        if not bool(args.skip_baseline):
            for z_threshold in args.baseline_z_thresholds:
                for xy_threshold in args.baseline_xy_thresholds:
                    for score_mode in args.score_mode:
                        t0 = time.perf_counter()
                        seg = agglomerate_fragments_by_affinity(
                            sample.affinities,
                            sample.fragments,
                            threshold=0.0,
                            z_threshold=float(z_threshold),
                            xy_threshold=float(xy_threshold),
                            min_boundary=int(args.min_boundary),
                            score_mode=str(score_mode),
                            quantile=float(args.rag_quantile),
                        )
                        postprocess_sec = time.perf_counter() - t0
                        t1 = time.perf_counter()
                        metrics = segmentation_metrics(
                            seg,
                            sample.metric_labels,
                            ignore_label=int(args.ignore_label),
                            backend=str(args.metric_backend),
                        )
                        metrics_sec = time.perf_counter() - t1
                        row = {
                            "sample": sample.sample,
                            "backend": "seeded_rag",
                            "threshold": 0.0,
                            "z_threshold": float(z_threshold),
                            "xy_threshold": float(xy_threshold),
                            "score_mode": str(score_mode),
                            "rag_quantile": float(args.rag_quantile),
                            "veto_mode": "",
                            "veto_mean_threshold": "",
                            "veto_count_log_min": "",
                            "veto_penalty": "",
                            "veto_fraction": "",
                            "reweight_mode": "",
                            "reweight_alpha": "",
                            "reweight_gamma": "",
                            "reweight_changed_fraction": "",
                            "postprocess_sec": float(postprocess_sec),
                            "metrics_sec": float(metrics_sec),
                            **metric_dict(metrics),
                        }
                        print(row, flush=True)
                        records.append(row)

    if records:
        with (output_dir / "learned_rag_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = sorted({key for record in records for key in record.keys()})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    group_keys = [
        "backend",
        "threshold",
        "z_threshold",
        "xy_threshold",
        "score_mode",
        "rag_quantile",
        "veto_mode",
        "veto_mean_threshold",
        "veto_count_log_min",
        "veto_penalty",
        "reweight_mode",
        "reweight_alpha",
        "reweight_gamma",
    ]
    records_for_summary = [row for row in records if "error" not in row]
    metric_keys = ["adapted_rand_error", "rand_fscore", "voi_split", "voi_merge", "voi_sum", "postprocess_sec"]
    grouped: dict[tuple[object, ...], list[dict[str, float | int | str]]] = {}
    for record in records_for_summary:
        grouped.setdefault(tuple(record[key] for key in group_keys), []).append(record)
    per_group: list[dict[str, float | int | str]] = []
    for key_tuple, rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        row: dict[str, float | int | str] = dict(zip(group_keys, key_tuple))
        row["n"] = len(rows)
        for key in metric_keys:
            row[key] = _mean(rows, key)
        per_group.append(row)
    eval_holdout = {p.name for p in eval_files} - {p.name for p in train_files}
    holdout_rows = [row for row in records_for_summary if str(row["sample"]) in eval_holdout]
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "crop_size": list(crop_size),
        "eval_crop_size": list(eval_crop_size),
        "stride": list(stride),
        "calibration_bias": list(calibration_bias),
        "calibration_temperature": float(args.calibration_temperature),
        "train_samples": [p.name for p in train_files],
        "eval_samples": [p.name for p in eval_files],
        "train_stats": train_stats,
        "target_mode": str(args.target_mode),
        "teacher_waterz_threshold": float(args.teacher_waterz_threshold),
        "teacher_waterz_thresholds": list(teacher_waterz_thresholds),
        "teacher_waterz_scoring": str(args.teacher_waterz_scoring),
        "sample_summaries": sample_summaries,
        "num_records": len(records),
        "append_context_features": bool(args.append_context_features),
        "merge_backends": list(args.merge_backends),
        "signed_veto_modes": list(args.signed_veto_modes),
        "signed_veto_mean_thresholds": [float(v) for v in args.signed_veto_mean_thresholds],
        "signed_veto_count_log_mins": [float(v) for v in args.signed_veto_count_log_mins],
        "signed_veto_penalties": [float(v) for v in args.signed_veto_penalties],
        "signed_reweight_modes": list(args.signed_reweight_modes),
        "signed_reweight_alphas": [float(v) for v in args.signed_reweight_alphas],
        "signed_reweight_gammas": [float(v) for v in args.signed_reweight_gammas],
        "failures": [row for row in records if "error" in row],
        "per_group": per_group,
    }
    if per_group:
        summary["best_by_voi_sum"] = min(per_group, key=lambda row: float(row["voi_sum"]))
        summary["best_by_adapted_rand"] = min(per_group, key=lambda row: float(row["adapted_rand_error"]))
    if holdout_rows:
        summary["holdout_best_by_voi_sum"] = min(holdout_rows, key=lambda row: float(row["voi_sum"]))
        summary["holdout_best_by_adapted_rand"] = min(holdout_rows, key=lambda row: float(row["adapted_rand_error"]))
    (output_dir / "learned_rag_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output_dir / "learned_rag_records.json").write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
