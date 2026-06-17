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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dbmim.datasets import labels_to_affinities, normalize_volume
from dbmim.metrics import binary_iou_from_logits, dice_from_logits
from dbmim.postprocess import (
    affinities_to_connected_components,
    agglomerate_fragments_by_affinity,
    cc3d_mean_affinity_components,
    cupy_affinity_graph_connected_components,
    cupy_mean_affinity_components,
    segmentation_metrics,
    waterz_agglomeration,
    watershed_fragments,
)
from train_finetune import build_affinity_model
from dbmim.utils import ensure_dir, load_checkpoint, load_config


def _lazy_h5py():
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for CREMI HDF5 evaluation") from exc
    return h5py


def list_cremi_files(data_dir: str | Path) -> list[Path]:
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("sample_*_20160501.hdf"))
    if not files:
        files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in {".h5", ".hdf", ".hdf5"})
    if not files:
        raise FileNotFoundError(f"no HDF5 files found under {data_dir}")
    return files


def select_h5_dataset(handle, keys: Iterable[str]):
    for key in keys:
        if key in handle:
            item = handle[key]
            if hasattr(item, "shape"):
                return item
    raise KeyError(f"none of these HDF5 keys exist: {list(keys)}")


def center_slices(shape: tuple[int, int, int], crop_size: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    slices = []
    for dim, crop in zip(shape, crop_size):
        if crop <= 0 or crop >= dim:
            start = 0
            stop = dim
        else:
            start = max(0, (dim - crop) // 2)
            stop = start + crop
        slices.append(slice(start, stop))
    return tuple(slices)  # type: ignore[return-value]


def read_cremi_crop(
    path: Path,
    crop_size: tuple[int, int, int],
    raw_keys: Iterable[str],
    label_keys: Iterable[str],
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice]]:
    h5py = _lazy_h5py()
    with h5py.File(path, "r") as handle:
        raw_ds = select_h5_dataset(handle, raw_keys)
        label_ds = select_h5_dataset(handle, label_keys)
        if tuple(raw_ds.shape[:3]) != tuple(label_ds.shape[:3]):
            raise ValueError(f"raw/label shape mismatch in {path}: {raw_ds.shape} vs {label_ds.shape}")
        crop = center_slices(tuple(int(v) for v in raw_ds.shape[:3]), crop_size)
        raw = np.asarray(raw_ds[crop])
        label = np.asarray(label_ds[crop])
    return raw, label, crop


def starts_for_dim(dim: int, window: int, stride: int) -> list[int]:
    if dim <= window:
        return [0]
    starts = list(range(0, dim - window + 1, max(1, stride)))
    if starts[-1] != dim - window:
        starts.append(dim - window)
    return starts


@torch.no_grad()
def predict_affinities(
    model: torch.nn.Module,
    raw: np.ndarray,
    window_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    volume = normalize_volume(raw)
    depth, height, width = volume.shape
    wd, wh, ww = window_size
    pad = (
        (0, max(0, wd - depth)),
        (0, max(0, wh - height)),
        (0, max(0, ww - width)),
    )
    if any(v for pair in pad for v in pair):
        volume = np.pad(volume, pad, mode="reflect" if min(volume.shape) > 1 else "edge")
    pd, ph, pw = volume.shape
    logits_sum = np.zeros((3, pd, ph, pw), dtype=np.float32)
    counts = np.zeros((1, pd, ph, pw), dtype=np.float32)
    z_starts = starts_for_dim(pd, wd, stride[0])
    y_starts = starts_for_dim(ph, wh, stride[1])
    x_starts = starts_for_dim(pw, ww, stride[2])
    amp_enabled = device.type == "cuda"
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = volume[z0 : z0 + wd, y0 : y0 + wh, x0 : x0 + ww]
                tensor = torch.from_numpy(patch[None, None]).float().to(device)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(tensor)
                logits_np = logits.float().cpu().numpy()[0]
                logits_sum[:, z0 : z0 + wd, y0 : y0 + wh, x0 : x0 + ww] += logits_np
                counts[:, z0 : z0 + wd, y0 : y0 + wh, x0 : x0 + ww] += 1.0
    affinities = 1.0 / (1.0 + np.exp(-(logits_sum / np.maximum(counts, 1.0))))
    return affinities[:, :depth, :height, :width]


def build_model(cfg: dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    effective_cfg = payload.get("config", cfg)
    model = build_affinity_model(effective_cfg)
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def metric_dict(obj) -> dict[str, float | int]:
    return {
        "adapted_rand_error": obj.adapted_rand_error,
        "rand_fscore": obj.rand_fscore,
        "rand_precision": obj.rand_precision,
        "rand_recall": obj.rand_recall,
        "voi_split": obj.voi_split,
        "voi_merge": obj.voi_merge,
        "voi_sum": obj.voi_split + obj.voi_merge,
        "n_pred": obj.n_pred,
        "n_gt": obj.n_gt,
        "n_voxels": obj.n_voxels,
    }


def package_probe() -> dict[str, object]:
    import importlib.util

    packages = ["scipy", "mahotas", "cc3d", "waterz", "elf", "cupy"]
    result: dict[str, object] = {name: importlib.util.find_spec(name) is not None for name in packages}
    if result.get("cupy"):
        try:
            import cupy as cp  # type: ignore

            result["cupy_version"] = cp.__version__
            result["cuda_device_count"] = int(cp.cuda.runtime.getDeviceCount())
        except Exception as exc:
            result["cupy_error"] = repr(exc)
    return result


def run_backend(
    affinities: np.ndarray,
    backend: str,
    threshold: float,
    min_size: int,
    seed_method: str,
    seed_distance: int,
    boundary_threshold: float,
    min_boundary: int,
    score_mode: str = "mean",
    rag_quantile: float = 0.25,
    z_threshold: float | None = None,
    xy_threshold: float | None = None,
    cache: dict[str, tuple[np.ndarray, float]] | None = None,
) -> np.ndarray:
    graph_threshold = threshold
    if z_threshold is not None or xy_threshold is not None:
        z_thr = float(threshold if z_threshold is None else z_threshold)
        xy_thr = float(threshold if xy_threshold is None else xy_threshold)
        graph_threshold = (z_thr, xy_thr, xy_thr)
    if backend == "graph_cc":
        return affinities_to_connected_components(affinities, threshold=graph_threshold, min_size=min_size)
    if backend == "cc3d_mean":
        return cc3d_mean_affinity_components(affinities, threshold=threshold, min_size=min_size)
    if backend == "cupy_mean":
        return cupy_mean_affinity_components(affinities, threshold=threshold, min_size=min_size)
    if backend == "cupy_graph_cc":
        return cupy_affinity_graph_connected_components(affinities, threshold=graph_threshold, min_size=min_size)
    if backend == "mahotas_watershed":
        key = f"fragments:mahotas:{seed_method}:{seed_distance}:{boundary_threshold}"
        if cache is not None and key in cache:
            return cache[key][0]
        fragments = watershed_fragments(
            affinities,
            backend="mahotas",
            seed_method=seed_method,
            seed_distance=seed_distance,
            boundary_threshold=boundary_threshold,
        )
        if cache is not None:
            cache[key] = (fragments, 0.0)
        return fragments
    if backend == "scipy_watershed":
        key = f"fragments:scipy:{seed_method}:{seed_distance}:{boundary_threshold}"
        if cache is not None and key in cache:
            return cache[key][0]
        fragments = watershed_fragments(
            affinities,
            backend="scipy",
            seed_method=seed_method,
            seed_distance=seed_distance,
            boundary_threshold=boundary_threshold,
        )
        if cache is not None:
            cache[key] = (fragments, 0.0)
        return fragments
    if backend in {"mahotas_agglomeration", "seeded_rag"}:
        key = f"fragments:mahotas:{seed_method}:{seed_distance}:{boundary_threshold}"
        fragments = None
        if cache is not None and key in cache:
            fragments = cache[key][0]
        if fragments is None:
            fragments = watershed_fragments(
                affinities,
                backend="mahotas",
                seed_method=seed_method,
                seed_distance=seed_distance,
                boundary_threshold=boundary_threshold,
            )
            if cache is not None:
                cache[key] = (fragments, 0.0)
        return agglomerate_fragments_by_affinity(
            affinities,
            fragments,
            threshold=threshold,
            z_threshold=z_threshold,
            xy_threshold=xy_threshold,
            min_boundary=min_boundary,
            score_mode=score_mode,
            quantile=rag_quantile,
        )
    if backend == "scipy_agglomeration":
        key = f"fragments:scipy:{seed_method}:{seed_distance}:{boundary_threshold}"
        fragments = None
        if cache is not None and key in cache:
            fragments = cache[key][0]
        if fragments is None:
            fragments = watershed_fragments(
                affinities,
                backend="scipy",
                seed_method=seed_method,
                seed_distance=seed_distance,
                boundary_threshold=boundary_threshold,
            )
            if cache is not None:
                cache[key] = (fragments, 0.0)
        return agglomerate_fragments_by_affinity(
            affinities,
            fragments,
            threshold=threshold,
            z_threshold=z_threshold,
            xy_threshold=xy_threshold,
            min_boundary=min_boundary,
            score_mode=score_mode,
            quantile=rag_quantile,
        )
    if backend == "waterz":
        return waterz_agglomeration(affinities, threshold=threshold)
    raise ValueError(f"unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CREMI neuron segmentation from dbMiM affinities")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", nargs=3, type=int, default=[32, 256, 256])
    parser.add_argument("--stride", nargs=3, type=int, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.4, 0.5, 0.6, 0.7])
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["graph_cc"],
        choices=[
            "graph_cc",
            "cc3d_mean",
            "cupy_mean",
            "cupy_graph_cc",
            "mahotas_watershed",
            "scipy_watershed",
            "seeded_rag",
            "mahotas_agglomeration",
            "scipy_agglomeration",
            "waterz",
        ],
    )
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--seed-method", default="maxima_distance", choices=["maxima_distance", "minima", "grid"])
    parser.add_argument("--seed-distance", nargs="+", type=int, default=[12])
    parser.add_argument("--boundary-threshold", nargs="+", type=float, default=[0.5])
    parser.add_argument("--min-boundary", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--score-mode",
        nargs="+",
        default=["mean"],
        choices=["mean", "max", "min", "median", "q10", "q25", "q50", "q75", "quantile"],
    )
    parser.add_argument("--rag-quantile", nargs="+", type=float, default=[0.25])
    parser.add_argument("--z-thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--xy-thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-seg", action="store_true")
    parser.add_argument("--ignore-label", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = ensure_dir(args.output_dir)
    model = build_model(cfg, Path(args.checkpoint), device)
    model_window = tuple(int(v) for v in model.volume_size)
    stride = tuple(args.stride) if args.stride is not None else tuple(max(1, v // 2) for v in model_window)
    crop_size = tuple(int(v) for v in args.crop_size)
    if any(c < w for c, w in zip(crop_size, model_window)):
        crop_size = tuple(max(c, w) for c, w in zip(crop_size, model_window))

    data_cfg = cfg.get("data", {})
    raw_keys = data_cfg.get("image_keys") or ["volumes/raw", "raw", "main"]
    label_keys = data_cfg.get("label_keys") or ["volumes/labels/neuron_ids", "labels", "label", "gt"]
    files = list_cremi_files(args.data_dir)
    if args.max_samples > 0:
        files = files[: args.max_samples]

    records: list[dict[str, float | int | str]] = []
    failures: list[dict[str, str | float]] = []
    probes = package_probe()
    print({"package_probe": probes}, flush=True)
    watershed_backends = {"mahotas_watershed", "scipy_watershed"}
    anisotropic_graph_backends = {"graph_cc", "cupy_graph_cc"}
    agglomeration_backends = {"seeded_rag", "mahotas_agglomeration", "scipy_agglomeration"}
    seed_backends = watershed_backends | agglomeration_backends
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
        aff = predict_affinities(model, raw, model_window, stride, device)
        infer_sec = time.perf_counter() - t0
        target_aff = labels_to_affinities(torch.from_numpy(label.astype(np.int64))[None]).numpy()[0]
        logits = torch.from_numpy(np.log(np.clip(aff, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - aff, 1e-6, 1.0)))
        target = torch.from_numpy(target_aff)
        aff_dice = dice_from_logits(logits, target)
        aff_iou = binary_iou_from_logits(logits, target)
        backend_cache: dict[str, tuple[np.ndarray, float]] = {}
        for backend in args.backends:
            threshold_values = args.thresholds if backend not in watershed_backends else [args.thresholds[0]]
            seed_distances = args.seed_distance if backend in seed_backends else [args.seed_distance[0]]
            boundary_thresholds = args.boundary_threshold if backend in seed_backends else [args.boundary_threshold[0]]
            min_boundaries = args.min_boundary if backend in agglomeration_backends else [args.min_boundary[0]]
            score_modes = args.score_mode if backend in agglomeration_backends else [args.score_mode[0]]
            z_thresholds = [None]
            xy_thresholds = [None]
            if backend in agglomeration_backends | anisotropic_graph_backends:
                z_thresholds = list(args.z_thresholds) if args.z_thresholds else [None]
                xy_thresholds = list(args.xy_thresholds) if args.xy_thresholds else [None]
            for threshold, seed_distance, boundary_threshold, min_boundary, score_mode, rag_quantile, z_threshold, xy_threshold in itertools.product(
                threshold_values,
                seed_distances,
                boundary_thresholds,
                min_boundaries,
                score_modes,
                args.rag_quantile if backend in agglomeration_backends else [args.rag_quantile[0]],
                z_thresholds,
                xy_thresholds,
            ):
                try:
                    t1 = time.perf_counter()
                    seg = run_backend(
                        aff,
                        backend=backend,
                        threshold=threshold,
                        min_size=args.min_size,
                        seed_method=args.seed_method,
                        seed_distance=int(seed_distance),
                        boundary_threshold=float(boundary_threshold),
                        min_boundary=int(min_boundary),
                        score_mode=str(score_mode),
                        rag_quantile=float(rag_quantile),
                        z_threshold=None if z_threshold is None else float(z_threshold),
                        xy_threshold=None if xy_threshold is None else float(xy_threshold),
                        cache=backend_cache,
                    )
                    postprocess_sec = time.perf_counter() - t1
                    t2 = time.perf_counter()
                    metrics = segmentation_metrics(seg, label, ignore_label=args.ignore_label)
                    metrics_sec = time.perf_counter() - t2
                except Exception as exc:
                    failure = {
                        "sample": path.name,
                        "backend": backend,
                        "threshold": float(threshold),
                        "seed_distance": int(seed_distance),
                        "boundary_threshold": float(boundary_threshold),
                        "min_boundary": int(min_boundary),
                        "score_mode": str(score_mode),
                        "rag_quantile": float(rag_quantile),
                        "z_threshold": "" if z_threshold is None else float(z_threshold),
                        "xy_threshold": "" if xy_threshold is None else float(xy_threshold),
                        "error": repr(exc),
                    }
                    print(failure, flush=True)
                    failures.append(failure)
                    continue
                record: dict[str, float | int | str] = {
                    "sample": path.name,
                    "backend": backend,
                    "threshold": float(threshold),
                    "seed_distance": int(seed_distance),
                    "boundary_threshold": float(boundary_threshold),
                    "min_boundary": int(min_boundary),
                    "score_mode": str(score_mode),
                    "rag_quantile": float(rag_quantile),
                    "z_threshold": "" if z_threshold is None else float(z_threshold),
                    "xy_threshold": "" if xy_threshold is None else float(xy_threshold),
                    "affinity_dice": float(aff_dice),
                    "affinity_iou": float(aff_iou),
                    "inference_sec": float(infer_sec),
                    "postprocess_sec": float(postprocess_sec),
                    "metrics_sec": float(metrics_sec),
                    **metric_dict(metrics),
                }
                print(record, flush=True)
                records.append(record)
                if args.save_seg:
                    z_tag = "none" if z_threshold is None else f"{float(z_threshold):.2f}"
                    xy_tag = "none" if xy_threshold is None else f"{float(xy_threshold):.2f}"
                    out_npz = output_dir / (
                        f"{path.stem}_{backend}_thr{threshold:.2f}_sd{int(seed_distance)}_"
                        f"bt{float(boundary_threshold):.2f}_mb{int(min_boundary)}_"
                        f"{score_mode}_z{z_tag}_xy{xy_tag}_seg.npz"
                    )
                    np.savez_compressed(out_npz, segmentation=seg.astype(np.uint64), crop=np.array([[sl.start, sl.stop] for sl in crop]))

    csv_path = output_dir / "cremi_segmentation_metrics.csv"
    if records:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    thresholds = sorted({float(r["threshold"]) for r in records})
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "crop_size": list(crop_size),
        "window_size": list(model_window),
        "stride": list(stride),
        "num_records": len(records),
        "package_probe": probes,
        "failures": failures,
        "per_threshold": [],
        "per_backend_threshold": [],
    }
    metric_keys = [
        "adapted_rand_error",
        "rand_fscore",
        "rand_precision",
        "rand_recall",
        "voi_split",
        "voi_merge",
        "voi_sum",
        "affinity_dice",
        "affinity_iou",
        "inference_sec",
        "postprocess_sec",
        "metrics_sec",
    ]
    group_keys = [
        "backend",
        "threshold",
        "seed_distance",
        "boundary_threshold",
        "min_boundary",
        "score_mode",
        "rag_quantile",
        "z_threshold",
        "xy_threshold",
    ]
    grouped: dict[tuple[object, ...], list[dict[str, float | int | str]]] = {}
    for record in records:
        grouped.setdefault(tuple(record[key] for key in group_keys), []).append(record)
    for key_tuple, rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        mean_row: dict[str, float | int | str] = dict(zip(group_keys, key_tuple))
        mean_row["n"] = len(rows)
        for key in metric_keys:
            mean_row[key] = float(np.mean([float(r[key]) for r in rows])) if rows else float("nan")
        summary["per_backend_threshold"].append(mean_row)  # type: ignore[index]
    for threshold in thresholds:
        rows = [r for r in records if float(r["threshold"]) == threshold]
        mean_row: dict[str, float | int] = {"threshold": threshold, "n": len(rows)}
        for key in metric_keys:
            mean_row[key] = float(np.mean([float(r[key]) for r in rows])) if rows else float("nan")
        summary["per_threshold"].append(mean_row)  # type: ignore[index]
    if summary["per_backend_threshold"]:
        summary["best_by_adapted_rand"] = min(
            summary["per_backend_threshold"],  # type: ignore[arg-type]
            key=lambda row: float(row["adapted_rand_error"]),
        )
        summary["best_by_voi_sum"] = min(
            summary["per_backend_threshold"],  # type: ignore[arg-type]
            key=lambda row: float(row["voi_sum"]),
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
