#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
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
from infer_mec8nm_blockwise_rag import write_segmentation_h5
from stitch_mec8nm_slab_segmentations import (
    SlabSpec,
    UnionFind,
    axis_slice,
    crop_slices,
    decode_pair,
    encoded_pair_counts,
    map_local_to_global,
    partition_equivalence_summary,
    write_contacts_csv,
)


@dataclass(frozen=True)
class LocalSlab:
    index: int
    start: int
    stop: int
    core_start: int
    core_stop: int
    label_offset: int
    max_label: int
    inference_sec: float
    postprocess_sec: float
    num_chunks: int
    path: str


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON serialize {type(value)!r}")


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_")


def _slices_to_bounds(crop: tuple[slice, slice, slice]) -> list[list[int]]:
    return [[int(sl.start or 0), int(sl.stop or 0)] for sl in crop]


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


def read_crop(
    path: Path,
    *,
    raw_keys: Iterable[str],
    label_keys: Iterable[str],
    crop_start: tuple[int, int, int],
    crop_size: tuple[int, int, int],
    require_label: bool,
) -> tuple[np.ndarray, np.ndarray | None, tuple[slice, slice, slice], dict[str, object]]:
    h5py = _lazy_h5py()
    with h5py.File(path, "r") as handle:
        raw_ds = select_h5_dataset(handle, raw_keys)
        raw_shape = tuple(int(v) for v in raw_ds.shape[:3])
        crop = crop_slices(crop_start, crop_size, raw_shape)
        raw = np.asarray(raw_ds[crop])
        label = None
        label_shape = None
        label_dtype = None
        try:
            label_ds = select_h5_dataset(handle, label_keys)
        except KeyError:
            label_ds = None
        if require_label and label_ds is None:
            raise KeyError(f"labels are required but none of these HDF5 keys exist in {path}: {list(label_keys)}")
        if label_ds is not None:
            label_shape = tuple(int(v) for v in label_ds.shape[:3])
            label_dtype = str(label_ds.dtype)
            if label_shape != raw_shape:
                if require_label:
                    raise ValueError(f"raw/label shape mismatch in {path}: {raw_shape} vs {label_shape}")
            else:
                label = np.asarray(label_ds[crop])
        meta = {
            "raw_dataset_shape": list(raw_shape),
            "raw_dataset_dtype": str(raw_ds.dtype),
            "label_dataset_shape": None if label_shape is None else list(label_shape),
            "label_dataset_dtype": label_dtype,
        }
    return raw, label, crop, meta


def read_reference_crop(
    path: Path,
    *,
    dataset: str,
    crop: tuple[slice, slice, slice],
) -> np.ndarray:
    h5py = _lazy_h5py()
    with h5py.File(path, "r") as handle:
        ds = handle[str(dataset)]
        return np.asarray(ds[crop])


def make_slabs(length: int, slab_size: int, overlap: int) -> list[SlabSpec]:
    if slab_size <= 0:
        raise ValueError("--slab-size must be positive")
    if overlap <= 0:
        raise ValueError("--slab-overlap must be positive")
    if overlap >= slab_size:
        raise ValueError("--slab-overlap must be smaller than --slab-size")
    starts: list[tuple[int, int]] = []
    start = 0
    while True:
        stop = min(length, start + slab_size)
        starts.append((start, stop))
        if stop >= length:
            break
        next_start = stop - overlap
        if next_start <= start:
            raise RuntimeError(f"slab generation stalled at {start}:{stop}")
        start = next_start
    slabs: list[SlabSpec] = []
    for idx, (start_i, stop_i) in enumerate(starts):
        if idx == 0:
            core_start = start_i
        else:
            core_start = (start_i + starts[idx - 1][1]) // 2
        if idx == len(starts) - 1:
            core_stop = stop_i
        else:
            core_stop = (stop_i + starts[idx + 1][0]) // 2
        if core_start >= core_stop:
            raise RuntimeError(f"empty core for slab {idx}: {core_start}:{core_stop}")
        slabs.append(
            SlabSpec(
                index=idx,
                start=int(start_i),
                stop=int(stop_i),
                core_start=int(core_start),
                core_stop=int(core_stop),
                label_offset=0,
            )
        )
    return slabs


def slab_global_labels(seg: np.ndarray, slab: LocalSlab, axis: int, global_start: int, global_stop: int) -> np.ndarray:
    local_start = int(global_start) - int(slab.start)
    local_stop = int(global_stop) - int(slab.start)
    arr = np.asarray(seg[axis_slice(axis, local_start, local_stop)], dtype=np.uint64)
    if slab.label_offset:
        arr = arr.copy()
        mask = arr != 0
        arr[mask] += np.uint64(slab.label_offset)
    return arr


def build_real_slab_unions(
    slab_segmentations: list[np.ndarray],
    slabs: list[LocalSlab],
    *,
    axis: int,
    min_contact_voxels: int,
    min_overlap_ratio: float,
    chunk_size: int,
) -> tuple[UnionFind, list[dict[str, int | float]]]:
    uf = UnionFind()
    contacts: list[dict[str, int | float]] = []
    for left_idx in range(len(slabs) - 1):
        left_slab = slabs[left_idx]
        right_slab = slabs[left_idx + 1]
        overlap_start = max(left_slab.start, right_slab.start)
        overlap_stop = min(left_slab.stop, right_slab.stop)
        if overlap_start >= overlap_stop:
            continue
        left = slab_global_labels(slab_segmentations[left_idx], left_slab, axis, overlap_start, overlap_stop)
        right = slab_global_labels(slab_segmentations[left_idx + 1], right_slab, axis, overlap_start, overlap_stop)
        left_counts, right_counts, pair_counts = encoded_pair_counts(left, right, chunk_size=chunk_size)
        for pair, count in pair_counts.items():
            left_label, right_label = decode_pair(pair)
            left_ratio = float(count / max(1, left_counts.get(left_label, 0)))
            right_ratio = float(count / max(1, right_counts.get(right_label, 0)))
            ratio = min(left_ratio, right_ratio)
            accepted = int(count) >= int(min_contact_voxels) and left_ratio >= float(min_overlap_ratio) and right_ratio >= float(min_overlap_ratio)
            if accepted:
                uf.union(left_label, right_label)
            contacts.append(
                {
                    "left_slab": left_slab.index,
                    "right_slab": right_slab.index,
                    "overlap_start": int(overlap_start),
                    "overlap_stop": int(overlap_stop),
                    "left_label": int(left_label),
                    "right_label": int(right_label),
                    "contact_voxels": int(count),
                    "left_overlap_voxels": int(left_counts.get(left_label, 0)),
                    "right_overlap_voxels": int(right_counts.get(right_label, 0)),
                    "left_overlap_ratio": float(left_ratio),
                    "right_overlap_ratio": float(right_ratio),
                    "overlap_ratio": float(ratio),
                    "accepted": int(accepted),
                }
            )
    return uf, contacts


def compose_real_stitched(
    slab_segmentations: list[np.ndarray],
    slabs: list[LocalSlab],
    *,
    shape: tuple[int, int, int],
    axis: int,
    uf: UnionFind,
) -> np.ndarray:
    output = np.zeros(shape, dtype=np.uint64)
    root_to_id: dict[int, int] = {}
    for seg, slab in zip(slab_segmentations, slabs):
        local_start = int(slab.core_start) - int(slab.start)
        local_stop = int(slab.core_stop) - int(slab.start)
        local = slab_global_labels(seg, slab, axis, slab.core_start, slab.core_stop)
        expected = tuple(int(v) for v in output[axis_slice(axis, slab.core_start, slab.core_stop)].shape)
        if tuple(local.shape) != expected:
            raise RuntimeError(
                f"slab {slab.index} core shape mismatch: local {local.shape} vs output {expected} "
                f"(local slice {local_start}:{local_stop})"
            )
        mapped = map_local_to_global(local, uf, root_to_id)
        output[axis_slice(axis, slab.core_start, slab.core_stop)] = mapped
    dtype = np.uint32 if len(root_to_id) <= np.iinfo(np.uint32).max else np.uint64
    return output.astype(dtype, copy=False)


def make_slab_seam_mask(shape: tuple[int, int, int], slabs: list[LocalSlab], *, axis: int, width: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if width <= 0:
        return mask
    boundaries = [slab.core_stop for slab in slabs[:-1]]
    for boundary in boundaries:
        lo = max(0, int(boundary) - int(width))
        hi = min(shape[int(axis)], int(boundary) + int(width))
        slicer = [slice(None), slice(None), slice(None)]
        slicer[int(axis)] = slice(lo, hi)
        mask[tuple(slicer)] = True
    return mask


def write_records_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_metric_records(
    *,
    sample: str,
    seg: np.ndarray,
    label: np.ndarray,
    regions: list[str],
    slabs: list[LocalSlab],
    axis: int,
    slab_seam_width: int,
    chunk_size: tuple[int, int, int],
    chunk_seam_width: tuple[int, int, int],
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
    slab_seam = make_slab_seam_mask(tuple(int(v) for v in seg.shape[:3]), slabs, axis=axis, width=slab_seam_width)
    chunk_seam = make_seam_mask(tuple(int(v) for v in seg.shape[:3]), chunk_size, chunk_seam_width)
    labels_by_region = {
        "full": metric_label,
        "slab_seam": mask_label(metric_label, slab_seam, int(ignore_label)),
        "slab_nonseam": mask_label(metric_label, ~slab_seam, int(ignore_label)),
        "chunk_seam": mask_label(metric_label, chunk_seam, int(ignore_label)),
        "chunk_nonseam": mask_label(metric_label, ~chunk_seam, int(ignore_label)),
    }
    fractions = {
        "full": 1.0,
        "slab_seam": float(np.mean(slab_seam)),
        "slab_nonseam": float(np.mean(~slab_seam)),
        "chunk_seam": float(np.mean(chunk_seam)),
        "chunk_nonseam": float(np.mean(~chunk_seam)),
    }
    records: list[dict[str, object]] = []
    for region in regions:
        if region != "full" and fractions[region] <= 0.0:
            continue
        t0 = time.perf_counter()
        metrics = segmentation_metrics(
            seg,
            labels_by_region[region],
            ignore_label=int(ignore_label),
            backend=str(metric_backend),
        )
        metrics_sec = time.perf_counter() - t0
        records.append(
            {
                **common_record,
                "sample": sample,
                "region": region,
                "region_fraction": fractions[region],
                "boundary_ignore_fraction": float(boundary_ignore_fraction),
                "metrics_sec": float(metrics_sec),
                **metric_dict(metrics),
            }
        )
    return records


def run_sample(
    *,
    path: Path,
    args: argparse.Namespace,
    model: torch.nn.Module,
    postprocess: torch.nn.Module | None,
    model_window: tuple[int, int, int],
    device: torch.device,
    output_dir: Path,
    raw_keys: Iterable[str],
    label_keys: Iterable[str],
    calibration_biases: list[tuple[float, float, float]],
    calibration_temperatures: list[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    sample_t0 = time.perf_counter()
    crop_size = normalize_crop_size(tuple(int(v) for v in args.crop_size), model_window)
    crop_start = tuple(int(v) for v in args.crop_start)
    raw, label, crop, h5_meta = read_crop(
        path,
        raw_keys=raw_keys,
        label_keys=label_keys,
        crop_start=crop_start,
        crop_size=crop_size,
        require_label=bool(args.evaluate),
    )
    shape = tuple(int(v) for v in raw.shape[:3])
    slab_specs = make_slabs(shape[int(args.slab_axis)], int(args.slab_size), int(args.slab_overlap))
    stride = tuple(int(v) for v in args.stride)
    chunk_size = tuple(int(v) for v in args.chunk_size)
    halo = tuple(int(v) for v in args.halo)
    chunk_seam_width = tuple(int(v) for v in args.chunk_seam_width)
    return_logits = bool(calibration_biases)
    if len(calibration_biases) > 1 or len(calibration_temperatures) > 1:
        raise ValueError("slabwise export expects one calibration triplet and one temperature")

    sample_tag = _safe_tag(path.stem)
    sample_dir = ensure_dir(output_dir / sample_tag)
    slab_dir = ensure_dir(sample_dir / "slabs")
    stitched_dir = ensure_dir(sample_dir / "stitched")

    slab_segmentations: list[np.ndarray] = []
    slabs: list[LocalSlab] = []
    next_label_offset = 0
    slab_records: list[dict[str, object]] = []
    metric_records: list[dict[str, object]] = []
    sample_affinity_variant = "slabwise_halo"
    sample_calibration_bias = (0.0, 0.0, 0.0)
    sample_calibration_temperature = 1.0
    common_backend = {
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
    }

    for slab_spec in slab_specs:
        slab_t0 = time.perf_counter()
        slab_slice = axis_slice(int(args.slab_axis), slab_spec.start, slab_spec.stop)
        raw_slab = np.asarray(raw[slab_slice])
        pred, infer_stats = predict_affinities_chunked(
            model,
            raw_slab,
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
            affinity_variant = f"slabwise_halo_{cal_name}"
        else:
            affinities = sigmoid_np(pred).astype(np.float32, copy=False) if return_logits else pred
            calibration_bias = (0.0, 0.0, 0.0)
            calibration_temperature = 1.0
            affinity_variant = "slabwise_halo"
        sample_affinity_variant = str(affinity_variant)
        sample_calibration_bias = tuple(float(v) for v in calibration_bias)
        sample_calibration_temperature = float(calibration_temperature)
        del pred
        post_t0 = time.perf_counter()
        backend_cache: dict[str, tuple[np.ndarray, float]] = {}
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
        max_label = int(np.asarray(seg).max(initial=0))
        slab = LocalSlab(
            index=int(slab_spec.index),
            start=int(slab_spec.start),
            stop=int(slab_spec.stop),
            core_start=int(slab_spec.core_start),
            core_stop=int(slab_spec.core_stop),
            label_offset=int(next_label_offset),
            max_label=max_label,
            inference_sec=float(infer_stats["inference_sec"]),
            postprocess_sec=float(postprocess_sec),
            num_chunks=int(infer_stats["num_chunks"]),
            path="",
        )
        next_label_offset += max_label
        slab_name = (
            f"{sample_tag}_slab{slab.index:03d}_axis{args.slab_axis}_"
            f"{slab.start}-{slab.stop}_core{slab.core_start}-{slab.core_stop}"
        )
        slab_path = slab_dir / f"{slab_name}.seg.h5"
        slab_meta = {
            "sample": path.name,
            "source_path": str(path),
            "crop_bounds": _slices_to_bounds(crop),
            "slab": asdict(slab),
            "affinity_variant": affinity_variant,
            "calibration_bias_z": float(calibration_bias[0]),
            "calibration_bias_y": float(calibration_bias[1]),
            "calibration_bias_x": float(calibration_bias[2]),
            "calibration_temperature": float(calibration_temperature),
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
            "slab_elapsed_sec": float(time.perf_counter() - slab_t0),
            **common_backend,
            **h5_meta,
        }
        if args.save_slabs:
            write_meta = write_segmentation_h5(
                slab_path,
                seg,
                slab_meta,
                compression=str(args.compression),
                compression_level=int(args.compression_level),
            )
            slab_path_str = str(slab_path)
        else:
            write_meta = {}
            slab_path_str = ""
        slab = LocalSlab(**{**asdict(slab), "path": slab_path_str})
        slab_segmentations.append(np.asarray(seg, dtype=np.uint32))
        slabs.append(slab)
        slab_record = {
            **slab_meta,
            **write_meta,
            "slab_path": slab_path_str,
            "slab_index": int(slab.index),
            "slab_start": int(slab.start),
            "slab_stop": int(slab.stop),
            "slab_core_start": int(slab.core_start),
            "slab_core_stop": int(slab.core_stop),
            "slab_label_offset": int(slab.label_offset),
            "slab_max_label": int(slab.max_label),
            "inference_sec": float(slab.inference_sec),
            "postprocess_sec": float(slab.postprocess_sec),
            "num_chunks": int(slab.num_chunks),
        }
        slab_records.append(slab_record)
        print(
            json.dumps(
                {
                    "event": "slab_done",
                    "sample": path.name,
                    "slab": slab.index,
                    "start": slab.start,
                    "stop": slab.stop,
                    "core_start": slab.core_start,
                    "core_stop": slab.core_stop,
                    "max_label": slab.max_label,
                    "inference_sec": slab.inference_sec,
                    "postprocess_sec": slab.postprocess_sec,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del affinities, seg, raw_slab, backend_cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    uf, contacts = build_real_slab_unions(
        slab_segmentations,
        slabs,
        axis=int(args.slab_axis),
        min_contact_voxels=int(args.min_contact_voxels),
        min_overlap_ratio=float(args.min_overlap_ratio),
        chunk_size=int(args.contact_chunk_voxels),
    )
    stitched = compose_real_stitched(slab_segmentations, slabs, shape=shape, axis=int(args.slab_axis), uf=uf)
    accepted_contacts = sum(1 for row in contacts if int(row["accepted"]) == 1)
    safe_name = _safe_tag(
        f"{sample_tag}_slabwise_{args.backend}_thr{args.threshold:.2f}_"
        f"axis{args.slab_axis}_slab{args.slab_size}_ov{args.slab_overlap}"
    )
    stitched_path = stitched_dir / f"{safe_name}.seg.h5"
    stitched_meta = {
        "sample": path.name,
        "source_path": str(path),
        "crop_bounds": _slices_to_bounds(crop),
        "raw_shape": list(shape),
        "affinity_variant": sample_affinity_variant,
        "calibration_bias_z": float(sample_calibration_bias[0]),
        "calibration_bias_y": float(sample_calibration_bias[1]),
        "calibration_bias_x": float(sample_calibration_bias[2]),
        "calibration_temperature": float(sample_calibration_temperature),
        "slab_axis": int(args.slab_axis),
        "slab_size": int(args.slab_size),
        "slab_overlap": int(args.slab_overlap),
        "slab_seam_width": int(args.slab_seam_width),
        "num_slabs": len(slabs),
        "slabs": [asdict(slab) for slab in slabs],
        "num_contacts": len(contacts),
        "accepted_contacts": int(accepted_contacts),
        "min_contact_voxels": int(args.min_contact_voxels),
        "min_overlap_ratio": float(args.min_overlap_ratio),
        "inference_sec": float(sum(s.inference_sec for s in slabs)),
        "postprocess_sec": float(sum(s.postprocess_sec for s in slabs)),
        "total_inference_sec": float(sum(s.inference_sec for s in slabs)),
        "total_postprocess_sec": float(sum(s.postprocess_sec for s in slabs)),
        **common_backend,
        **h5_meta,
    }
    stitched_write_meta = write_segmentation_h5(
        stitched_path,
        stitched,
        stitched_meta,
        compression=str(args.compression),
        compression_level=int(args.compression_level),
    )
    contacts_path = sample_dir / "overlap_contacts.csv"
    write_contacts_csv(contacts_path, contacts)

    if args.evaluate:
        if label is None:
            raise RuntimeError(f"--evaluate requested but no label was loaded for {path}")
        metric_records = build_metric_records(
            sample=path.name,
            seg=stitched,
            label=label,
            regions=[str(v) for v in args.regions],
            slabs=slabs,
            axis=int(args.slab_axis),
            slab_seam_width=int(args.slab_seam_width),
            chunk_size=chunk_size,
            chunk_seam_width=chunk_seam_width,
            ignore_label=int(args.ignore_label),
            metric_backend=str(args.metric_backend),
            boundary_ignore_xy=int(args.cremi_boundary_ignore_distance_xy),
            boundary_ignore_z=int(args.cremi_boundary_ignore_distance_z),
            common_record={
                **stitched_meta,
                **stitched_write_meta,
                "segmentation_path": str(stitched_path),
            },
        )

    reference_equivalence: dict[str, object] | None = None
    if args.reference_seg_h5:
        ref = read_reference_crop(Path(args.reference_seg_h5), dataset=str(args.reference_dataset), crop=crop)
        reference_equivalence = partition_equivalence_summary(
            np.asarray(ref),
            np.asarray(stitched),
            ignore_label=int(args.ignore_label),
            chunk_size=int(args.contact_chunk_voxels),
        )

    sample_summary = {
        "sample": path.name,
        "source_path": str(path),
        "crop_bounds": _slices_to_bounds(crop),
        "shape": list(shape),
        "slabs": [asdict(slab) for slab in slabs],
        "num_contacts": len(contacts),
        "accepted_contacts": int(accepted_contacts),
        "stitched_path": str(stitched_path),
        "contacts_path": str(contacts_path),
        "metric_records": metric_records,
        "reference_equivalence": reference_equivalence,
        "total_inference_sec": float(sum(s.inference_sec for s in slabs)),
        "total_postprocess_sec": float(sum(s.postprocess_sec for s in slabs)),
        "elapsed_sec": float(time.perf_counter() - sample_t0),
    }
    (sample_dir / "slab_manifest.json").write_text(
        json.dumps(slab_records, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (sample_dir / "slabwise_summary.json").write_text(
        json.dumps(sample_summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    write_records_csv(sample_dir / "slab_manifest.csv", slab_records)
    if metric_records:
        write_records_csv(sample_dir / "slabwise_metrics.csv", metric_records)
    return slab_records, metric_records, sample_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real MEC slabwise inference, overlap-contact stitching, and metrics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--input-files", nargs="+", default=None)
    parser.add_argument("--sample-names", nargs="+", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-start", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--crop-size", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--slab-axis", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--slab-size", type=int, default=48)
    parser.add_argument("--slab-overlap", type=int, default=12)
    parser.add_argument("--slab-seam-width", type=int, default=2)
    parser.add_argument("--stride", nargs=3, type=int, default=[16, 80, 80])
    parser.add_argument("--chunk-size", nargs=3, type=int, default=[32, 512, 512])
    parser.add_argument("--halo", nargs=3, type=int, default=[8, 64, 64])
    parser.add_argument("--chunk-seam-width", nargs=3, type=int, default=[2, 16, 16])
    parser.add_argument("--backend", default="waterz")
    parser.add_argument("--threshold", type=float, default=0.36)
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
    parser.add_argument("--min-contact-voxels", type=int, default=1)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--contact-chunk-voxels", type=int, default=5_000_000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--raw-keys", nargs="+", default=None)
    parser.add_argument("--label-keys", nargs="+", default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=["full", "slab_seam", "slab_nonseam", "chunk_seam", "chunk_nonseam"],
        default=["full", "slab_seam", "slab_nonseam"],
    )
    parser.add_argument("--ignore-label", type=int, default=0)
    parser.add_argument("--cremi-boundary-ignore-distance-xy", type=int, default=1)
    parser.add_argument("--cremi-boundary-ignore-distance-z", type=int, default=0)
    parser.add_argument("--metric-backend", choices=["internal", "skimage"], default="skimage")
    parser.add_argument("--reference-seg-h5", default="")
    parser.add_argument("--reference-dataset", default="segmentation")
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--compression-level", type=int, default=4)
    parser.add_argument("--save-slabs", action="store_true")
    parser.add_argument("--fail-on-sample-error", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = build_model(cfg, Path(args.checkpoint), device)
    postprocess = build_postprocess(cfg, Path(args.checkpoint), device)
    model_window = tuple(int(v) for v in model.volume_size)
    data_cfg = cfg.get("data", {})
    raw_keys = args.raw_keys or data_cfg.get("image_keys") or ["volumes/raw", "raw", "main"]
    label_keys = args.label_keys or data_cfg.get("label_keys") or ["volumes/labels/neuron_ids", "labels", "label", "gt"]
    calibration_biases = parse_channel_triplets([float(v) for v in args.calibration_biases])
    calibration_temperatures = [float(v) for v in args.calibration_temperatures]

    files = resolve_files(args.data_dir, args.input_files, args.sample_names)
    if int(args.max_samples) > 0:
        files = files[: int(args.max_samples)]

    all_slab_records: list[dict[str, object]] = []
    all_metric_records: list[dict[str, object]] = []
    sample_summaries: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    print(
        json.dumps(
            {
                "event": "slabwise_start",
                "num_samples": len(files),
                "device": str(device),
                "checkpoint": str(args.checkpoint),
                "config": str(args.config),
                "model_window": list(model_window),
                "backend": str(args.backend),
                "threshold": float(args.threshold),
                "slab_axis": int(args.slab_axis),
                "slab_size": int(args.slab_size),
                "slab_overlap": int(args.slab_overlap),
                "evaluate": bool(args.evaluate),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for path in files:
        try:
            slab_records, metric_records, sample_summary = run_sample(
                path=path,
                args=args,
                model=model,
                postprocess=postprocess,
                model_window=model_window,
                device=device,
                output_dir=output_dir,
                raw_keys=raw_keys,
                label_keys=label_keys,
                calibration_biases=calibration_biases,
                calibration_temperatures=calibration_temperatures,
            )
            all_slab_records.extend(slab_records)
            all_metric_records.extend(metric_records)
            sample_summaries.append(sample_summary)
            print(
                json.dumps(
                    {
                        "event": "sample_done",
                        "sample": path.name,
                        "stitched_path": sample_summary["stitched_path"],
                        "num_contacts": sample_summary["num_contacts"],
                        "accepted_contacts": sample_summary["accepted_contacts"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {"sample": path.name, "source_path": str(path), "error": repr(exc)}
            failures.append(failure)
            print(json.dumps({"event": "sample_failed", **failure}, sort_keys=True), flush=True)
            if args.fail_on_sample_error:
                raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metric_summary: list[dict[str, object]] = []
    if all_metric_records:
        group_keys = ["region", "backend", "threshold", "affinity_variant", "calibration_temperature"]
        metric_keys = ["adapted_rand_error", "rand_fscore", "voi_split", "voi_merge", "voi_sum", "postprocess_sec"]
        grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
        for row in all_metric_records:
            grouped.setdefault(tuple(row.get(key, "") for key in group_keys), []).append(row)
        for key_tuple, rows in sorted(grouped.items(), key=lambda item: tuple(str(v) for v in item[0])):
            out: dict[str, object] = dict(zip(group_keys, key_tuple))
            out["n"] = len(rows)
            for key in metric_keys:
                values = [float(r[key]) for r in rows if key in r]
                out[key] = float(np.mean(values)) if values else float("nan")
            metric_summary.append(out)

    summary = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "data_dir": "" if args.data_dir is None else str(args.data_dir),
        "input_files": [] if args.input_files is None else [str(v) for v in args.input_files],
        "num_samples": len(sample_summaries),
        "slab_axis": int(args.slab_axis),
        "slab_size": int(args.slab_size),
        "slab_overlap": int(args.slab_overlap),
        "min_contact_voxels": int(args.min_contact_voxels),
        "min_overlap_ratio": float(args.min_overlap_ratio),
        "samples": sample_summaries,
        "metric_summary": metric_summary,
        "failures": failures,
    }
    (output_dir / "slabwise_run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    write_records_csv(output_dir / "slabwise_slab_manifest.csv", all_slab_records)
    write_records_csv(output_dir / "slabwise_metrics.csv", all_metric_records)
    print(
        json.dumps(
            {
                "event": "slabwise_done",
                "num_samples": len(sample_summaries),
                "num_metric_records": len(all_metric_records),
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
