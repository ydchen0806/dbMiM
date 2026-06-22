#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dbmim.postprocess import rag_boundary_features, segmentation_metrics, watershed_fragments
from dbmim.utils import ensure_dir, load_config
from evaluate_cremi_segmentation import (  # noqa: E402
    apply_cremi_boundary_ignore,
    build_model,
    build_postprocess,
    list_cremi_files,
    metric_dict,
    normalize_crop_size,
    read_cremi_crop,
    run_backend,
)


def _starts_for_chunks(dim: int, chunk: int) -> list[int]:
    if dim <= chunk:
        return [0]
    return list(range(0, dim, max(1, int(chunk))))


def _crop_tuple(starts: tuple[int, int, int], stops: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    return tuple(slice(int(a), int(b)) for a, b in zip(starts, stops))  # type: ignore[return-value]


@torch.no_grad()
def predict_affinities_chunked(
    model: torch.nn.Module,
    raw: np.ndarray,
    *,
    model_window: tuple[int, int, int],
    stride: tuple[int, int, int],
    chunk_size: tuple[int, int, int],
    halo: tuple[int, int, int],
    device: torch.device,
    channels: int = 3,
    postprocess: torch.nn.Module | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    from evaluate_cremi_segmentation import predict_affinities

    shape = tuple(int(v) for v in raw.shape[:3])
    out = np.zeros((int(channels), *shape), dtype=np.float32)
    timings: list[float] = []
    chunk_voxels: list[int] = []
    z_starts = _starts_for_chunks(shape[0], chunk_size[0])
    y_starts = _starts_for_chunks(shape[1], chunk_size[1])
    x_starts = _starts_for_chunks(shape[2], chunk_size[2])
    for z0, y0, x0 in itertools.product(z_starts, y_starts, x_starts):
        core_start = (z0, y0, x0)
        core_stop = (
            min(shape[0], z0 + chunk_size[0]),
            min(shape[1], y0 + chunk_size[1]),
            min(shape[2], x0 + chunk_size[2]),
        )
        ext_start = tuple(max(0, s - h) for s, h in zip(core_start, halo))
        ext_stop = tuple(min(d, e + h) for d, e, h in zip(shape, core_stop, halo))
        ext_slices = _crop_tuple(ext_start, ext_stop)
        t0 = time.perf_counter()
        ext_aff = predict_affinities(
            model,
            raw[ext_slices],
            model_window,
            stride,
            device,
            channels=channels,
            postprocess=postprocess,
            return_logits=False,
        ).astype(np.float32, copy=False)
        timings.append(time.perf_counter() - t0)
        chunk_voxels.append(int(np.prod([b - a for a, b in zip(core_start, core_stop)])))
        local_start = tuple(s - es for s, es in zip(core_start, ext_start))
        local_stop = tuple(e - es for e, es in zip(core_stop, ext_start))
        out_slices = _crop_tuple(core_start, core_stop)
        local_slices = _crop_tuple(local_start, local_stop)
        out[(slice(None), *out_slices)] = ext_aff[(slice(None), *local_slices)]
        print(
            {
                "chunk": [list(core_start), list(core_stop)],
                "extended": [list(ext_start), list(ext_stop)],
                "sec": timings[-1],
            },
            flush=True,
        )
    total_sec = float(sum(timings))
    total_voxels = int(np.prod(shape))
    stats: dict[str, float | int] = {
        "num_chunks": int(len(timings)),
        "inference_sec": total_sec,
        "total_voxels": total_voxels,
        "voxels_per_sec": float(total_voxels / max(total_sec, 1e-6)),
        "core_voxels_sum": int(sum(chunk_voxels)),
        "chunk_d": int(chunk_size[0]),
        "chunk_h": int(chunk_size[1]),
        "chunk_w": int(chunk_size[2]),
        "halo_d": int(halo[0]),
        "halo_h": int(halo[1]),
        "halo_w": int(halo[2]),
    }
    return out, stats


def make_seam_mask(shape: tuple[int, int, int], chunk_size: tuple[int, int, int], seam_width: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for axis, (dim, chunk, width) in enumerate(zip(shape, chunk_size, seam_width)):
        if int(width) <= 0 or int(chunk) >= int(dim):
            continue
        for boundary in range(int(chunk), int(dim), int(chunk)):
            lo = max(0, boundary - int(width))
            hi = min(int(dim), boundary + int(width))
            slicer = [slice(None), slice(None), slice(None)]
            slicer[axis] = slice(lo, hi)
            mask[tuple(slicer)] = True
    return mask


def mask_label(label: np.ndarray, keep: np.ndarray, ignore_label: int) -> np.ndarray:
    out = np.asarray(label).copy()
    out[~keep] = int(ignore_label)
    return out


def _mean(rows: list[dict[str, float | int | str]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")


def rag_stats_for_affinity(
    affinities: np.ndarray,
    *,
    seed_distance: int,
    boundary_threshold: float,
    min_boundary: int,
) -> dict[str, float | int]:
    t0 = time.perf_counter()
    fragments = watershed_fragments(
        affinities,
        backend="mahotas",
        seed_method="maxima_distance",
        seed_distance=int(seed_distance),
        boundary_threshold=float(boundary_threshold),
    )
    fragment_sec = time.perf_counter() - t0
    t1 = time.perf_counter()
    pairs, _, counts = rag_boundary_features(affinities, fragments, min_boundary=int(min_boundary))
    rag_sec = time.perf_counter() - t1
    voxels = int(np.prod(fragments.shape))
    return {
        "fragment_sec": float(fragment_sec),
        "rag_sec": float(rag_sec),
        "num_fragments": int(fragments.max(initial=0)),
        "num_rag_edges": int(pairs.size),
        "mean_boundary_count": float(np.mean(counts)) if counts.size else 0.0,
        "rag_edges_per_mvoxel": float(pairs.size / max(voxels / 1_000_000.0, 1e-6)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CREMI blockwise scale/seam proxy evaluation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--stride", nargs=3, type=int, default=[16, 80, 80])
    parser.add_argument("--chunk-size", nargs=3, type=int, default=[32, 512, 512])
    parser.add_argument("--halo", nargs=3, type=int, default=[8, 64, 64])
    parser.add_argument("--seam-width", nargs=3, type=int, default=[2, 16, 16])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.0])
    parser.add_argument("--backends", nargs="+", default=["graph_cc", "seeded_rag"])
    parser.add_argument("--z-thresholds", nargs="+", type=float, default=[0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--xy-thresholds", nargs="+", type=float, default=[0.10, 0.20, 0.30, 0.50])
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--seed-distance", type=int, default=10)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--min-boundary", type=int, default=4)
    parser.add_argument("--score-mode", nargs="+", default=["mean", "q25"])
    parser.add_argument("--rag-quantile", nargs="+", type=float, default=[0.25])
    parser.add_argument("--waterz-scoring", default="hist_quantile")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ignore-label", type=int, default=0)
    parser.add_argument("--cremi-boundary-ignore-distance-xy", type=int, default=1)
    parser.add_argument("--cremi-boundary-ignore-distance-z", type=int, default=0)
    parser.add_argument("--metric-backend", choices=["internal", "skimage"], default="skimage")
    parser.add_argument("--compare-no-halo", action="store_true")
    parser.add_argument("--compute-rag-stats", action="store_true")
    parser.add_argument("--fail-on-backend-error", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(cfg, Path(args.checkpoint), device)
    postprocess = build_postprocess(cfg, Path(args.checkpoint), device)
    model_window = tuple(int(v) for v in model.volume_size)
    crop_size = normalize_crop_size(tuple(int(v) for v in args.crop_size), model_window)
    stride = tuple(int(v) for v in args.stride)
    chunk_size = tuple(int(v) for v in args.chunk_size)
    halo = tuple(int(v) for v in args.halo)
    seam_width = tuple(int(v) for v in args.seam_width)
    data_cfg = cfg.get("data", {})
    raw_keys = data_cfg.get("image_keys") or ["volumes/raw", "raw", "main"]
    label_keys = data_cfg.get("label_keys") or ["volumes/labels/neuron_ids", "labels", "label", "gt"]
    files = list_cremi_files(args.data_dir)
    if int(args.max_samples) > 0:
        files = files[: int(args.max_samples)]

    records: list[dict[str, float | int | str]] = []
    throughput_records: list[dict[str, float | int | str]] = []
    failures: list[dict[str, float | int | str]] = []
    for path in files:
        raw, label, crop = read_cremi_crop(path, crop_size, raw_keys, label_keys)
        shape = tuple(int(v) for v in raw.shape)
        metric_label, boundary_ignore_fraction = apply_cremi_boundary_ignore(
            label,
            ignore_label=int(args.ignore_label),
            distance_xy=int(args.cremi_boundary_ignore_distance_xy),
            distance_z=int(args.cremi_boundary_ignore_distance_z),
        )
        seam_mask = make_seam_mask(shape, chunk_size, seam_width)
        labels_by_region = {
            "full": metric_label,
            "seam": mask_label(metric_label, seam_mask, int(args.ignore_label)),
            "nonseam": mask_label(metric_label, ~seam_mask, int(args.ignore_label)),
        }
        print(
            {
                "sample": path.name,
                "raw_shape": list(shape),
                "crop": [[sl.start, sl.stop] for sl in crop],
                "seam_fraction": float(np.mean(seam_mask)),
                "boundary_ignore_fraction": float(boundary_ignore_fraction),
            },
            flush=True,
        )
        variants: list[tuple[str, np.ndarray, dict[str, float | int]]] = []
        aff, stats = predict_affinities_chunked(
            model,
            raw,
            model_window=model_window,
            stride=stride,
            chunk_size=chunk_size,
            halo=halo,
            device=device,
            channels=3,
            postprocess=postprocess,
        )
        variants.append(("blockwise_halo", aff, stats))
        if args.compare_no_halo:
            aff_no_halo, stats_no_halo = predict_affinities_chunked(
                model,
                raw,
                model_window=model_window,
                stride=stride,
                chunk_size=chunk_size,
                halo=(0, 0, 0),
                device=device,
                channels=3,
                postprocess=postprocess,
            )
            variants.append(("blockwise_no_halo", aff_no_halo, stats_no_halo))

        for variant_name, variant_aff, variant_stats in variants:
            throughput_record = {
                "sample": path.name,
                "affinity_variant": variant_name,
                "window_d": int(model_window[0]),
                "window_h": int(model_window[1]),
                "window_w": int(model_window[2]),
                "stride_d": int(stride[0]),
                "stride_h": int(stride[1]),
                "stride_w": int(stride[2]),
                "seam_fraction": float(np.mean(seam_mask)),
                **variant_stats,
            }
            if args.compute_rag_stats:
                throughput_record.update(
                    rag_stats_for_affinity(
                        variant_aff,
                        seed_distance=int(args.seed_distance),
                        boundary_threshold=float(args.boundary_threshold),
                        min_boundary=int(args.min_boundary),
                    )
                )
            print(throughput_record, flush=True)
            throughput_records.append(throughput_record)
            backend_cache: dict[str, tuple[np.ndarray, float]] = {}
            for backend in args.backends:
                threshold_values = args.thresholds
                z_values = [None]
                xy_values = [None]
                score_modes = [args.score_mode[0]]
                rag_quantiles = [args.rag_quantile[0]]
                if backend in {"graph_cc", "cupy_graph_cc", "seeded_rag", "mahotas_agglomeration", "scipy_agglomeration"}:
                    threshold_values = [args.thresholds[0]]
                    z_values = [float(v) for v in args.z_thresholds]
                    xy_values = [float(v) for v in args.xy_thresholds]
                if backend in {"seeded_rag", "mahotas_agglomeration", "scipy_agglomeration"}:
                    score_modes = [str(v) for v in args.score_mode]
                    rag_quantiles = [float(v) for v in args.rag_quantile]
                for threshold, score_mode, rag_quantile, z_threshold, xy_threshold in itertools.product(
                    threshold_values,
                    score_modes,
                    rag_quantiles,
                    z_values,
                    xy_values,
                ):
                    try:
                        t0 = time.perf_counter()
                        seg = run_backend(
                            variant_aff,
                            backend=str(backend),
                            threshold=float(threshold),
                            min_size=int(args.min_size),
                            seed_method="maxima_distance",
                            seed_distance=int(args.seed_distance),
                            boundary_threshold=float(args.boundary_threshold),
                            min_boundary=int(args.min_boundary),
                            score_mode=str(score_mode),
                            rag_quantile=float(rag_quantile),
                            z_threshold=None if z_threshold is None else float(z_threshold),
                            xy_threshold=None if xy_threshold is None else float(xy_threshold),
                            waterz_scoring=str(args.waterz_scoring),
                            cache=backend_cache,
                        )
                        postprocess_sec = time.perf_counter() - t0
                        for region_name, region_label in labels_by_region.items():
                            t1 = time.perf_counter()
                            metrics = segmentation_metrics(
                                seg,
                                region_label,
                                ignore_label=int(args.ignore_label),
                                backend=str(args.metric_backend),
                            )
                            metric_sec = time.perf_counter() - t1
                            record: dict[str, float | int | str] = {
                                "sample": path.name,
                                "affinity_variant": variant_name,
                                "region": region_name,
                                "backend": str(backend),
                                "threshold": float(threshold),
                                "z_threshold": "" if z_threshold is None else float(z_threshold),
                                "xy_threshold": "" if xy_threshold is None else float(xy_threshold),
                                "seed_distance": int(args.seed_distance),
                                "boundary_threshold": float(args.boundary_threshold),
                                "min_boundary": int(args.min_boundary),
                                "score_mode": str(score_mode),
                                "rag_quantile": float(rag_quantile),
                                "waterz_scoring": str(args.waterz_scoring),
                                "seam_fraction": float(np.mean(seam_mask)),
                                "boundary_ignore_fraction": float(boundary_ignore_fraction),
                                "inference_sec": float(variant_stats["inference_sec"]),
                                "postprocess_sec": float(postprocess_sec),
                                "metrics_sec": float(metric_sec),
                                **metric_dict(metrics),
                            }
                            print(record, flush=True)
                            records.append(record)
                    except Exception as exc:
                        failure = {
                            "sample": path.name,
                            "affinity_variant": variant_name,
                            "backend": str(backend),
                            "threshold": float(threshold),
                            "z_threshold": "" if z_threshold is None else float(z_threshold),
                            "xy_threshold": "" if xy_threshold is None else float(xy_threshold),
                            "score_mode": str(score_mode),
                            "rag_quantile": float(rag_quantile),
                            "error": repr(exc),
                        }
                        print(failure, flush=True)
                        failures.append(failure)
                        if args.fail_on_backend_error:
                            raise

    if records:
        with (output_dir / "blockwise_scale_records.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    if throughput_records:
        with (output_dir / "blockwise_throughput.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(throughput_records[0].keys()))
            writer.writeheader()
            writer.writerows(throughput_records)

    group_keys = [
        "affinity_variant",
        "region",
        "backend",
        "threshold",
        "z_threshold",
        "xy_threshold",
        "seed_distance",
        "boundary_threshold",
        "min_boundary",
        "score_mode",
        "rag_quantile",
        "waterz_scoring",
    ]
    metric_keys = ["adapted_rand_error", "rand_fscore", "voi_split", "voi_merge", "voi_sum", "postprocess_sec"]
    grouped: dict[tuple[object, ...], list[dict[str, float | int | str]]] = {}
    for record in records:
        grouped.setdefault(tuple(record[key] for key in group_keys), []).append(record)
    per_group: list[dict[str, float | int | str]] = []
    for key_tuple, rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        row: dict[str, float | int | str] = dict(zip(group_keys, key_tuple))
        row["n"] = len(rows)
        for key in metric_keys:
            row[key] = _mean(rows, key)
        per_group.append(row)
    full_rows = [row for row in per_group if row["region"] == "full"]
    seam_rows = [row for row in per_group if row["region"] == "seam"]
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "data_dir": str(args.data_dir),
        "crop_size": list(crop_size),
        "chunk_size": list(chunk_size),
        "halo": list(halo),
        "seam_width": list(seam_width),
        "stride": list(stride),
        "num_records": len(records),
        "throughput_records": throughput_records,
        "per_group": per_group,
        "failures": failures,
    }
    if full_rows:
        summary["best_full_by_voi_sum"] = min(full_rows, key=lambda row: float(row["voi_sum"]))
        summary["best_full_by_adapted_rand"] = min(full_rows, key=lambda row: float(row["adapted_rand_error"]))
    if seam_rows:
        summary["best_seam_by_voi_sum"] = min(seam_rows, key=lambda row: float(row["voi_sum"]))
        summary["best_seam_by_adapted_rand"] = min(seam_rows, key=lambda row: float(row["adapted_rand_error"]))
    (output_dir / "blockwise_scale_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "blockwise_scale_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
