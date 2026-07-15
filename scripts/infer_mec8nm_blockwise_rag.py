#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from dbmim.postprocess import segmentation_metrics
from dbmim.utils import ensure_dir, load_config
from evaluate_cremi_blockwise_scale import make_seam_mask, mask_label, predict_affinities_chunked
from evaluate_cremi_segmentation import (
    _lazy_h5py,
    apply_cremi_boundary_ignore,
    build_model,
    build_postprocess,
    center_slices,
    list_cremi_files,
    make_calibrated_variants,
    metric_dict,
    normalize_crop_size,
    parse_channel_triplets,
    run_backend,
    select_cremi_files,
    select_h5_dataset,
    sigmoid_np,
)


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON-serialize {type(value)!r}")


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_")


def _slices_to_bounds(crop: tuple[slice, slice, slice]) -> list[list[int]]:
    return [[int(sl.start or 0), int(sl.stop or 0)] for sl in crop]


def _try_select_h5_dataset(handle: Any, keys: Iterable[str]) -> Any | None:
    try:
        return select_h5_dataset(handle, keys)
    except KeyError:
        return None


def read_h5_crop(
    path: Path,
    crop_size: tuple[int, int, int],
    raw_keys: Iterable[str],
    label_keys: Iterable[str],
    *,
    require_label: bool,
) -> tuple[np.ndarray, np.ndarray | None, tuple[slice, slice, slice], dict[str, object]]:
    h5py = _lazy_h5py()
    with h5py.File(path, "r") as handle:
        raw_ds = select_h5_dataset(handle, raw_keys)
        raw_shape = tuple(int(v) for v in raw_ds.shape[:3])
        crop = center_slices(raw_shape, crop_size)
        raw = np.asarray(raw_ds[crop])
        label_ds = _try_select_h5_dataset(handle, label_keys)
        label = None
        if require_label and label_ds is None:
            raise KeyError(f"labels are required but none of these HDF5 keys exist in {path}: {list(label_keys)}")
        if label_ds is not None:
            label_shape = tuple(int(v) for v in label_ds.shape[:3])
            if label_shape != raw_shape:
                if require_label:
                    raise ValueError(f"raw/label shape mismatch in {path}: {raw_shape} vs {label_shape}")
            else:
                label = np.asarray(label_ds[crop])
        meta = {
            "raw_dataset_shape": list(raw_shape),
            "raw_dataset_dtype": str(raw_ds.dtype),
            "label_dataset_shape": None if label_ds is None else [int(v) for v in label_ds.shape[:3]],
            "label_dataset_dtype": None if label_ds is None else str(label_ds.dtype),
        }
    return raw, label, crop, meta


def write_segmentation_h5(
    path: Path,
    segmentation: np.ndarray,
    metadata: dict[str, object],
    *,
    compression: str,
    compression_level: int,
) -> dict[str, object]:
    h5py = _lazy_h5py()
    path.parent.mkdir(parents=True, exist_ok=True)
    max_label = int(np.asarray(segmentation).max(initial=0))
    dtype = np.uint32 if max_label <= np.iinfo(np.uint32).max else np.uint64
    shape = tuple(int(v) for v in segmentation.shape[:3])
    chunks = tuple(max(1, min(v, c)) for v, c in zip(shape, (16, 256, 256)))
    compression_arg: str | None = None if compression == "none" else compression
    compression_opts = int(compression_level) if compression_arg == "gzip" else None
    write_meta = dict(metadata)
    write_meta.update(
        {
            "segmentation_path": str(path),
            "segmentation_dataset": "segmentation",
            "segmentation_dtype": np.dtype(dtype).name,
            "segmentation_max_label": max_label,
            "segmentation_chunks": list(chunks),
            "segmentation_compression": compression,
        }
    )
    with h5py.File(path, "w") as handle:
        ds = handle.create_dataset(
            "segmentation",
            data=np.asarray(segmentation, dtype=dtype),
            chunks=chunks,
            compression=compression_arg,
            compression_opts=compression_opts,
            shuffle=compression_arg is not None,
        )
        ds.attrs["sample"] = str(metadata.get("sample", ""))
        ds.attrs["source_path"] = str(metadata.get("source_path", ""))
        ds.attrs["crop_bounds"] = np.asarray(metadata.get("crop_bounds", []), dtype=np.int64)
        ds.attrs["backend"] = str(metadata.get("backend", ""))
        ds.attrs["affinity_variant"] = str(metadata.get("affinity_variant", ""))
        handle.attrs["metadata_json"] = json.dumps(write_meta, sort_keys=True, default=_json_default)
    return write_meta


def write_affinity_h5(
    path: Path,
    affinities: np.ndarray,
    metadata: dict[str, object],
    *,
    compression: str,
    compression_level: int,
) -> None:
    h5py = _lazy_h5py()
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(v) for v in affinities.shape)
    chunks = (1, max(1, min(shape[1], 16)), max(1, min(shape[2], 256)), max(1, min(shape[3], 256)))
    compression_arg: str | None = None if compression == "none" else compression
    compression_opts = int(compression_level) if compression_arg == "gzip" else None
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "affinities",
            data=np.asarray(affinities, dtype=np.float32),
            chunks=chunks,
            compression=compression_arg,
            compression_opts=compression_opts,
            shuffle=compression_arg is not None,
        )
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True, default=_json_default)


def write_records_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    output_dir: Path,
    sample_records: list[dict[str, object]],
    metric_records: list[dict[str, object]],
    failures: list[dict[str, object]],
    summary: dict[str, object],
    *,
    partial: bool,
) -> None:
    suffix = ".partial" if partial else ""
    (output_dir / f"inference_manifest{suffix}.json").write_text(
        json.dumps(sample_records, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"inference_metrics{suffix}.json").write_text(
        json.dumps(metric_records, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"inference_failures{suffix}.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"inference_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    write_records_csv(output_dir / f"inference_manifest{suffix}.csv", sample_records)
    write_records_csv(output_dir / f"inference_metrics{suffix}.csv", metric_records)


def build_metric_records(
    *,
    sample: str,
    seg: np.ndarray,
    label: np.ndarray,
    regions: list[str],
    chunk_size: tuple[int, int, int],
    seam_width: tuple[int, int, int],
    ignore_label: int,
    metric_backend: str,
    boundary_ignore_xy: int,
    boundary_ignore_z: int,
    common_record: dict[str, object],
) -> list[dict[str, object]]:
    metric_label, boundary_ignore_fraction = apply_cremi_boundary_ignore(
        label,
        ignore_label=int(ignore_label),
        distance_xy=int(boundary_ignore_xy),
        distance_z=int(boundary_ignore_z),
    )
    seam_mask = make_seam_mask(tuple(int(v) for v in seg.shape[:3]), chunk_size, seam_width)
    labels_by_region = {
        "full": metric_label,
        "seam": mask_label(metric_label, seam_mask, int(ignore_label)),
        "nonseam": mask_label(metric_label, ~seam_mask, int(ignore_label)),
    }
    records: list[dict[str, object]] = []
    for region in regions:
        t0 = time.perf_counter()
        metrics = segmentation_metrics(
            seg,
            labels_by_region[region],
            ignore_label=int(ignore_label),
            backend=str(metric_backend),
        )
        metrics_sec = time.perf_counter() - t0
        record = {
            **common_record,
            "sample": sample,
            "region": region,
            "seam_fraction": float(np.mean(seam_mask)),
            "boundary_ignore_fraction": float(boundary_ignore_fraction),
            "metrics_sec": float(metrics_sec),
            **metric_dict(metrics),
        }
        records.append(record)
    return records


def summarize_metrics(metric_records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    group_keys = [
        "region",
        "backend",
        "threshold",
        "z_threshold",
        "xy_threshold",
        "score_mode",
        "rag_quantile",
        "affinity_variant",
    ]
    metric_keys = ["adapted_rand_error", "rand_fscore", "voi_split", "voi_merge", "voi_sum", "postprocess_sec"]
    for row in metric_records:
        grouped.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)
    summary_rows: list[dict[str, object]] = []
    for key_tuple, rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
        row: dict[str, object] = dict(zip(group_keys, key_tuple))
        row["n"] = len(rows)
        for key in metric_keys:
            values = [float(r[key]) for r in rows if key in r]
            row[key] = float(np.mean(values)) if values else float("nan")
        summary_rows.append(row)
    return summary_rows


def resolve_files(data_dir: str | None, input_files: list[str] | None, sample_names: list[str] | None) -> list[Path]:
    if input_files:
        files = [Path(path) for path in input_files]
        missing = [str(path) for path in files if not path.exists()]
        if missing:
            raise FileNotFoundError(f"input files do not exist: {missing}")
        return select_cremi_files(files, sample_names)
    if data_dir is None:
        raise ValueError("either --data-dir or --input-files is required")
    return select_cremi_files(list_cremi_files(data_dir), sample_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MEC 8nm dbMiM block segmentations with chunked RAG inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--input-files", nargs="+", default=None)
    parser.add_argument("--sample-names", nargs="+", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--stride", nargs=3, type=int, default=[16, 80, 80])
    parser.add_argument("--chunk-size", nargs=3, type=int, default=[32, 512, 512])
    parser.add_argument("--halo", nargs=3, type=int, default=[8, 64, 64])
    parser.add_argument("--seam-width", nargs=3, type=int, default=[2, 16, 16])
    parser.add_argument("--backend", default="mahotas_agglomeration")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--z-threshold", type=float, default=0.5)
    parser.add_argument("--xy-threshold", type=float, default=0.5)
    parser.add_argument("--min-size", type=int, default=0)
    parser.add_argument("--seed-method", default="maxima_distance", choices=["maxima_distance", "minima", "grid"])
    parser.add_argument("--seed-distance", type=int, default=10)
    parser.add_argument("--boundary-threshold", type=float, default=0.5)
    parser.add_argument("--min-boundary", type=int, default=4)
    parser.add_argument("--score-mode", default="q25")
    parser.add_argument("--rag-quantile", type=float, default=0.25)
    parser.add_argument("--waterz-scoring", default="hist_quantile")
    parser.add_argument("--calibration-biases", nargs="+", type=float, default=[])
    parser.add_argument("--calibration-temperatures", nargs="+", type=float, default=[1.0])
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raw-keys", nargs="+", default=None)
    parser.add_argument("--label-keys", nargs="+", default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--regions", nargs="+", choices=["full", "seam", "nonseam"], default=["full", "seam", "nonseam"])
    parser.add_argument("--ignore-label", type=int, default=0)
    parser.add_argument("--cremi-boundary-ignore-distance-xy", type=int, default=1)
    parser.add_argument("--cremi-boundary-ignore-distance-z", type=int, default=0)
    parser.add_argument("--metric-backend", choices=["internal", "skimage"], default="skimage")
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--compression-level", type=int, default=4)
    parser.add_argument("--save-affinities", action="store_true")
    parser.add_argument("--count-segments", action="store_true")
    parser.add_argument("--fail-on-sample-error", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    ensure_dir(output_dir / "segmentations")
    if args.save_affinities:
        ensure_dir(output_dir / "affinities")
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
    raw_keys = args.raw_keys or data_cfg.get("image_keys") or ["volumes/raw", "raw", "main"]
    label_keys = args.label_keys or data_cfg.get("label_keys") or ["volumes/labels/neuron_ids", "labels", "label", "gt"]
    calibration_biases = parse_channel_triplets([float(v) for v in args.calibration_biases])
    calibration_temperatures = [float(v) for v in args.calibration_temperatures]
    if len(calibration_biases) > 1 or len(calibration_temperatures) > 1:
        raise ValueError("deployment export expects one calibration triplet and one temperature")

    files = resolve_files(args.data_dir, args.input_files, args.sample_names)
    if int(args.max_samples) > 0:
        files = files[: int(args.max_samples)]

    print(
        json.dumps(
            {
                "event": "inference_start",
                "num_samples": len(files),
                "device": str(device),
                "checkpoint": str(args.checkpoint),
                "config": str(args.config),
                "model_window": list(model_window),
                "chunk_size": list(chunk_size),
                "halo": list(halo),
                "backend": str(args.backend),
                "evaluate": bool(args.evaluate),
                "learned_postprocess_loaded": postprocess is not None,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    sample_records: list[dict[str, object]] = []
    metric_records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "data_dir": "" if args.data_dir is None else str(args.data_dir),
        "input_files": [] if args.input_files is None else [str(path) for path in args.input_files],
        "crop_size": list(crop_size),
        "stride": list(stride),
        "chunk_size": list(chunk_size),
        "halo": list(halo),
        "seam_width": list(seam_width),
        "backend": str(args.backend),
        "threshold": float(args.threshold),
        "z_threshold": float(args.z_threshold),
        "xy_threshold": float(args.xy_threshold),
        "score_mode": str(args.score_mode),
        "rag_quantile": float(args.rag_quantile),
        "calibration_biases": [list(bias) for bias in calibration_biases],
        "calibration_temperatures": calibration_temperatures,
        "evaluate": bool(args.evaluate),
        "samples": [],
        "failures": failures,
        "metric_summary": [],
    }

    for path in files:
        try:
            raw, label, crop, h5_meta = read_h5_crop(
                path,
                crop_size,
                raw_keys,
                label_keys,
                require_label=bool(args.evaluate),
            )
            shape = tuple(int(v) for v in raw.shape[:3])
            print(
                json.dumps(
                    {
                        "event": "sample_loaded",
                        "sample": path.name,
                        "raw_shape": list(shape),
                        "crop": _slices_to_bounds(crop),
                        "label_loaded": label is not None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return_logits = bool(calibration_biases)
            pred, infer_stats = predict_affinities_chunked(
                model,
                raw,
                model_window=model_window,
                stride=stride,
                chunk_size=chunk_size,
                halo=halo,
                device=device,
                channels=3,
                postprocess=postprocess,
                return_logits=return_logits,
            )
            if calibration_biases:
                variants = make_calibrated_variants(pred, calibration_biases, calibration_temperatures)
                if len(variants) != 1:
                    raise ValueError(f"expected exactly one calibrated variant, got {len(variants)}")
                cal_name, affinities, calibration_bias, calibration_temperature = variants[0]
                affinity_variant = f"blockwise_halo_{cal_name}"
            else:
                affinities = sigmoid_np(pred).astype(np.float32, copy=False) if return_logits else pred
                calibration_bias = (0.0, 0.0, 0.0)
                calibration_temperature = 1.0
                affinity_variant = "blockwise_halo"
            del pred

            backend_cache: dict[str, tuple[np.ndarray, float]] = {}
            post_t0 = time.perf_counter()
            seg = run_backend(
                affinities,
                backend=str(args.backend),
                threshold=float(args.threshold),
                min_size=int(args.min_size),
                seed_method=str(args.seed_method),
                seed_distance=int(args.seed_distance),
                boundary_threshold=float(args.boundary_threshold),
                min_boundary=int(args.min_boundary),
                score_mode=str(args.score_mode),
                rag_quantile=float(args.rag_quantile),
                z_threshold=float(args.z_threshold),
                xy_threshold=float(args.xy_threshold),
                waterz_scoring=str(args.waterz_scoring),
                cache=backend_cache,
            )
            postprocess_sec = time.perf_counter() - post_t0

            safe_name = _safe_tag(
                f"{path.stem}_{affinity_variant}_{args.backend}_thr{args.threshold:.2f}_"
                f"z{args.z_threshold:.2f}_xy{args.xy_threshold:.2f}_{args.score_mode}"
            )
            common_record: dict[str, object] = {
                "sample": path.name,
                "source_path": str(path),
                "raw_shape_d": shape[0],
                "raw_shape_h": shape[1],
                "raw_shape_w": shape[2],
                "crop_bounds": _slices_to_bounds(crop),
                "affinity_variant": affinity_variant,
                "calibration_bias_z": float(calibration_bias[0]),
                "calibration_bias_y": float(calibration_bias[1]),
                "calibration_bias_x": float(calibration_bias[2]),
                "calibration_temperature": float(calibration_temperature),
                "backend": str(args.backend),
                "threshold": float(args.threshold),
                "z_threshold": float(args.z_threshold),
                "xy_threshold": float(args.xy_threshold),
                "seed_method": str(args.seed_method),
                "seed_distance": int(args.seed_distance),
                "boundary_threshold": float(args.boundary_threshold),
                "min_boundary": int(args.min_boundary),
                "score_mode": str(args.score_mode),
                "rag_quantile": float(args.rag_quantile),
                "waterz_scoring": str(args.waterz_scoring),
                "window_d": int(model_window[0]),
                "window_h": int(model_window[1]),
                "window_w": int(model_window[2]),
                "stride_d": int(stride[0]),
                "stride_h": int(stride[1]),
                "stride_w": int(stride[2]),
                "chunk_d": int(chunk_size[0]),
                "chunk_h": int(chunk_size[1]),
                "chunk_w": int(chunk_size[2]),
                "halo_d": int(halo[0]),
                "halo_h": int(halo[1]),
                "halo_w": int(halo[2]),
                "inference_sec": float(infer_stats["inference_sec"]),
                "voxels_per_sec": float(infer_stats["voxels_per_sec"]),
                "num_chunks": int(infer_stats["num_chunks"]),
                "postprocess_sec": float(postprocess_sec),
                **h5_meta,
            }

            if args.save_affinities:
                aff_path = output_dir / "affinities" / f"{safe_name}.affinities.h5"
                write_affinity_h5(
                    aff_path,
                    affinities,
                    common_record,
                    compression=str(args.compression),
                    compression_level=int(args.compression_level),
                )
                common_record["affinity_path"] = str(aff_path)
            del affinities

            seg_path = output_dir / "segmentations" / f"{safe_name}.seg.h5"
            write_meta = write_segmentation_h5(
                seg_path,
                seg,
                common_record,
                compression=str(args.compression),
                compression_level=int(args.compression_level),
            )
            sample_record = {**common_record, **write_meta}
            if args.count_segments:
                sample_record["segmentation_n_labels"] = int(np.unique(seg).size)
            sample_records.append(sample_record)
            summary["samples"] = sample_records

            if args.evaluate:
                if label is None:
                    raise RuntimeError(f"--evaluate requested but no label was loaded for {path}")
                records = build_metric_records(
                    sample=path.name,
                    seg=seg,
                    label=label,
                    regions=[str(v) for v in args.regions],
                    chunk_size=chunk_size,
                    seam_width=seam_width,
                    ignore_label=int(args.ignore_label),
                    metric_backend=str(args.metric_backend),
                    boundary_ignore_xy=int(args.cremi_boundary_ignore_distance_xy),
                    boundary_ignore_z=int(args.cremi_boundary_ignore_distance_z),
                    common_record={**common_record, "segmentation_path": str(seg_path)},
                )
                metric_records.extend(records)
                print(json.dumps({"event": "sample_metrics", "sample": path.name, "records": records}, default=_json_default), flush=True)

            summary["samples"] = sample_records
            summary["metric_summary"] = summarize_metrics(metric_records)
            write_outputs(output_dir, sample_records, metric_records, failures, summary, partial=True)
            print(
                json.dumps(
                    {
                        "event": "sample_done",
                        "sample": path.name,
                        "segmentation_path": str(seg_path),
                        "inference_sec": float(infer_stats["inference_sec"]),
                        "postprocess_sec": float(postprocess_sec),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {"sample": path.name, "source_path": str(path), "error": repr(exc)}
            failures.append(failure)
            write_outputs(output_dir, sample_records, metric_records, failures, summary, partial=True)
            print(json.dumps({"event": "sample_failed", **failure}, sort_keys=True), flush=True)
            if args.fail_on_sample_error:
                raise
        finally:
            for var_name in ["raw", "label", "seg", "backend_cache"]:
                if var_name in locals():
                    del locals()[var_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary["samples"] = sample_records
    summary["failures"] = failures
    summary["metric_summary"] = summarize_metrics(metric_records)
    write_outputs(output_dir, sample_records, metric_records, failures, summary, partial=False)
    print(
        json.dumps(
            {
                "event": "inference_done",
                "num_samples": len(sample_records),
                "num_metric_records": len(metric_records),
                "num_failures": len(failures),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if failures and args.fail_on_sample_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
