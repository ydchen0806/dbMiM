#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_cremi_segmentation import _lazy_h5py, select_h5_dataset
from infer_mec8nm_slabwise_stitch import (
    LocalSlab,
    build_metric_records,
    build_real_slab_unions,
    compose_real_stitched,
    write_records_csv,
)
from infer_mec8nm_blockwise_rag import write_segmentation_h5
from stitch_mec8nm_slab_segmentations import partition_equivalence_summary, write_contacts_csv


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


def _bounds_to_slices(bounds: list[list[int]]) -> tuple[slice, slice, slice]:
    if len(bounds) != 3:
        raise ValueError(f"crop bounds must have three axes, got {bounds}")
    return tuple(slice(int(a), int(b)) for a, b in bounds)  # type: ignore[return-value]


def read_dataset_crop(path: Path, keys: Iterable[str], crop: tuple[slice, slice, slice]) -> np.ndarray:
    h5py = _lazy_h5py()
    with h5py.File(path, "r") as handle:
        ds = select_h5_dataset(handle, keys)
        return np.asarray(ds[crop])


def read_reference_crop(path: Path, dataset: str, crop: tuple[slice, slice, slice]) -> np.ndarray:
    h5py = _lazy_h5py()
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[str(dataset)][crop])


def read_segmentation(path: Path, dataset: str) -> np.ndarray:
    h5py = _lazy_h5py()
    with h5py.File(path, "r") as handle:
        return np.asarray(handle[str(dataset)])


def resolve_slab_path(run_dir: Path, record: dict[str, object]) -> Path:
    raw = str(record.get("slab_path") or record.get("segmentation_path") or "")
    candidates: list[Path] = []
    if raw:
        p = Path(raw)
        candidates.append(p)
        candidates.append(run_dir / p)
        candidates.append(run_dir / "slabs" / p.name)
    candidates.append(run_dir / "slabs" / f"*slab{int(record['slab_index']):03d}*.seg.h5")
    for candidate in candidates:
        if "*" in str(candidate):
            matches = sorted(candidate.parent.glob(candidate.name))
            if matches:
                return matches[0]
        elif candidate.exists():
            return candidate
    raise FileNotFoundError(f"could not resolve slab path for record {record.get('slab_index')}: {raw}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-stitch saved real MEC slab segmentations with new contact thresholds.")
    parser.add_argument("--slab-run-dir", required=True, help="Directory containing slab_manifest.json and slabs/*.seg.h5")
    parser.add_argument("--source-h5", required=True, help="Original MEC HDF5 with labels for metrics")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-keys", nargs="+", default=["volumes/labels/neuron_ids", "labels", "label", "gt"])
    parser.add_argument("--slab-dataset", default="segmentation")
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--min-contact-voxels", nargs="+", type=int, default=[1])
    parser.add_argument("--min-overlap-ratios", nargs="+", type=float, default=[0.5, 0.7, 0.9, 0.98])
    parser.add_argument("--contact-chunk-voxels", type=int, default=5_000_000)
    parser.add_argument("--slab-seam-width", type=int, default=2)
    parser.add_argument("--chunk-size", nargs=3, type=int, default=[32, 512, 512])
    parser.add_argument("--chunk-seam-width", nargs=3, type=int, default=[2, 16, 16])
    parser.add_argument(
        "--regions",
        nargs="+",
        choices=["full", "slab_seam", "slab_nonseam", "chunk_seam", "chunk_nonseam"],
        default=["full", "slab_seam", "slab_nonseam"],
    )
    parser.add_argument("--ignore-label", type=int, default=0)
    parser.add_argument("--cremi-boundary-ignore-distance-xy", type=int, default=1)
    parser.add_argument("--cremi-boundary-ignore-distance-z", type=int, default=0)
    parser.add_argument("--metric-backend", choices=["internal", "skimage"], default="internal")
    parser.add_argument("--reference-seg-h5", default="")
    parser.add_argument("--reference-dataset", default="segmentation")
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--compression-level", type=int, default=4)
    args = parser.parse_args()

    t0 = time.perf_counter()
    run_dir = Path(args.slab_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "slab_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError(f"empty manifest: {manifest_path}")
    crop_bounds = manifest[0]["crop_bounds"]
    crop = _bounds_to_slices(crop_bounds)
    label = read_dataset_crop(Path(args.source_h5), args.label_keys, crop)
    shape = tuple(int(v) for v in label.shape[:3])

    slab_segmentations: list[np.ndarray] = []
    slabs: list[LocalSlab] = []
    for record in manifest:
        slab_path = resolve_slab_path(run_dir, record)
        seg = read_segmentation(slab_path, str(args.slab_dataset))
        slab = LocalSlab(
            index=int(record["slab_index"]),
            start=int(record["slab_start"]),
            stop=int(record["slab_stop"]),
            core_start=int(record["slab_core_start"]),
            core_stop=int(record["slab_core_stop"]),
            label_offset=int(record["slab_label_offset"]),
            max_label=int(record["slab_max_label"]),
            inference_sec=float(record.get("inference_sec", 0.0)),
            postprocess_sec=float(record.get("postprocess_sec", 0.0)),
            num_chunks=int(record.get("num_chunks", 0)),
            path=str(slab_path),
        )
        slab_segmentations.append(np.asarray(seg, dtype=np.uint32))
        slabs.append(slab)

    sample = str(manifest[0].get("sample", run_dir.name))
    common = {
        "sample": sample,
        "source_path": str(Path(args.source_h5)),
        "crop_bounds": crop_bounds,
        "backend": str(manifest[0].get("backend", "")),
        "threshold": float(manifest[0].get("threshold", 0.0)),
        "z_threshold": float(manifest[0].get("z_threshold", 0.0)),
        "xy_threshold": float(manifest[0].get("xy_threshold", 0.0)),
        "affinity_variant": str(manifest[0].get("affinity_variant", "")),
        "calibration_temperature": float(manifest[0].get("calibration_temperature", 1.0)),
        "calibration_bias_z": float(manifest[0].get("calibration_bias_z", 0.0)),
        "calibration_bias_y": float(manifest[0].get("calibration_bias_y", 0.0)),
        "calibration_bias_x": float(manifest[0].get("calibration_bias_x", 0.0)),
        "slab_axis": int(args.axis),
        "num_slabs": len(slabs),
        "inference_sec": float(sum(s.inference_sec for s in slabs)),
        "postprocess_sec": float(sum(s.postprocess_sec for s in slabs)),
    }

    reference = None
    if args.reference_seg_h5:
        reference = read_reference_crop(Path(args.reference_seg_h5), str(args.reference_dataset), crop)

    all_metrics: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for min_contact in [int(v) for v in args.min_contact_voxels]:
        for min_ratio in [float(v) for v in args.min_overlap_ratios]:
            tag = f"contact{min_contact}_ratio{min_ratio:.3f}".replace(".", "p")
            variant_dir = output_dir / tag
            variant_dir.mkdir(parents=True, exist_ok=True)
            stitch_t0 = time.perf_counter()
            uf, contacts = build_real_slab_unions(
                slab_segmentations,
                slabs,
                axis=int(args.axis),
                min_contact_voxels=min_contact,
                min_overlap_ratio=min_ratio,
                chunk_size=int(args.contact_chunk_voxels),
            )
            stitched = compose_real_stitched(slab_segmentations, slabs, shape=shape, axis=int(args.axis), uf=uf)
            accepted_contacts = sum(1 for row in contacts if int(row["accepted"]) == 1)
            out_h5 = variant_dir / f"{_safe_tag(Path(args.source_h5).stem)}_restitched_{tag}.seg.h5"
            meta = {
                **common,
                "min_contact_voxels": min_contact,
                "min_overlap_ratio": min_ratio,
                "num_contacts": len(contacts),
                "accepted_contacts": int(accepted_contacts),
                "stitch_sec": float(time.perf_counter() - stitch_t0),
            }
            write_segmentation_h5(
                out_h5,
                stitched,
                meta,
                compression=str(args.compression),
                compression_level=int(args.compression_level),
            )
            contacts_path = variant_dir / "overlap_contacts.csv"
            write_contacts_csv(contacts_path, contacts)
            metric_records = build_metric_records(
                sample=sample,
                seg=stitched,
                label=label,
                regions=[str(v) for v in args.regions],
                slabs=slabs,
                axis=int(args.axis),
                slab_seam_width=int(args.slab_seam_width),
                chunk_size=tuple(int(v) for v in args.chunk_size),
                chunk_seam_width=tuple(int(v) for v in args.chunk_seam_width),
                ignore_label=int(args.ignore_label),
                metric_backend=str(args.metric_backend),
                boundary_ignore_xy=int(args.cremi_boundary_ignore_distance_xy),
                boundary_ignore_z=int(args.cremi_boundary_ignore_distance_z),
                common_record={**meta, "segmentation_path": str(out_h5), "contact_tag": tag},
            )
            reference_equivalence = None
            if reference is not None:
                reference_equivalence = partition_equivalence_summary(
                    np.asarray(reference),
                    np.asarray(stitched),
                    ignore_label=int(args.ignore_label),
                    chunk_size=int(args.contact_chunk_voxels),
                )
            summary = {
                **meta,
                "contact_tag": tag,
                "output_h5": str(out_h5),
                "contacts_path": str(contacts_path),
                "metrics": metric_records,
                "reference_equivalence": reference_equivalence,
                "stitched_max_label": int(np.asarray(stitched).max(initial=0)),
            }
            (variant_dir / "restitch_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
                encoding="utf-8",
            )
            all_metrics.extend(metric_records)
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True, default=_json_default), flush=True)

    run_summary = {
        "slab_run_dir": str(run_dir),
        "source_h5": str(Path(args.source_h5)),
        "output_dir": str(output_dir),
        "num_variants": len(summaries),
        "summaries": summaries,
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    (output_dir / "restitch_run_summary.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    write_records_csv(output_dir / "restitch_metrics.csv", all_metrics)


if __name__ == "__main__":
    main()
