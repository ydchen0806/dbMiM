#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


DEFAULT_SPLIT = {
    "train": [1, 2, 3, 4, 5, 6, 7, 9],
    "val": [8, 10],
    "test": [11, 12],
}


def _lazy_h5py():
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("h5py is required to write MEC HDF5 files") from exc
    return h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert haoyuan7/mec npy volumes to dbMiM HDF5 files.")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("/volume/med-train/users/dchen02/code/dbMiM/data/MEC_8nm/hf_snapshot"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/volume/med-train/users/dchen02/code/dbMiM/data/MEC_8nm/hdf5"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/volume/med-train/users/dchen02/code/dbMiM/data/MEC_8nm/mec8nm_manifest.json"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("/volume/med-train/users/dchen02/code/dbMiM/data/MEC_8nm/mec8nm_splits.json"),
    )
    parser.add_argument("--compression", default="", choices=["", "lzf", "gzip"])
    parser.add_argument("--gzip-level", type=int, default=1)
    parser.add_argument("--chunk-z", type=int, default=1)
    parser.add_argument("--chunk-yx", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def is_uint8_safe(raw: np.ndarray) -> bool:
    if raw.dtype == np.uint8:
        return True
    if not np.issubdtype(raw.dtype, np.floating):
        return False
    if float(np.nanmin(raw)) < 0.0 or float(np.nanmax(raw)) > 255.0:
        return False
    # Check a deterministic sparse sample before doing chunked integer casts.
    flat = raw.reshape(-1)
    step = max(1, flat.size // 1_000_000)
    sample = np.asarray(flat[::step])
    return bool(np.all(np.isfinite(sample)) and np.all(sample == np.rint(sample)))


def dataset_kwargs(compression: str, gzip_level: int) -> dict:
    if not compression:
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": int(gzip_level)}
    return {"compression": compression}


def copy_volume(src: np.ndarray, dst, shape: tuple[int, int, int], *, to_uint8: bool) -> None:
    for z0 in range(0, shape[0]):
        slab = np.asarray(src[z0, : shape[1], : shape[2]])
        if to_uint8 and slab.dtype != np.uint8:
            slab = np.rint(slab).clip(0, 255).astype(np.uint8)
        dst[z0, :, :] = slab


def summarize_labels(seg: np.ndarray, shape: tuple[int, int, int]) -> dict:
    nonzero = 0
    label_max = 0
    labels: set[int] = set()
    for z0 in range(shape[0]):
        slab = np.asarray(seg[z0, : shape[1], : shape[2]])
        nonzero += int(np.count_nonzero(slab))
        if slab.size:
            label_max = max(label_max, int(slab.max()))
            labels.update(int(v) for v in np.unique(slab) if int(v) != 0)
    voxels = int(np.prod(shape))
    return {
        "voxels": voxels,
        "foreground_voxels": nonzero,
        "foreground_fraction": float(nonzero / voxels) if voxels else 0.0,
        "label_max": int(label_max),
        "num_nonzero_labels": int(len(labels)),
    }


def convert_one(index: int, args: argparse.Namespace) -> dict:
    h5py = _lazy_h5py()
    raw_path = args.snapshot_dir / f"{index}_raw.npy"
    seg_path = args.snapshot_dir / f"{index}_seg_Myelin.npy"
    if not raw_path.exists() or not seg_path.exists():
        raise FileNotFoundError(f"missing MEC pair for index {index}: {raw_path}, {seg_path}")

    raw = np.load(raw_path, mmap_mode="r")
    seg = np.load(seg_path, mmap_mode="r")
    common_shape = tuple(int(min(a, b)) for a, b in zip(raw.shape[:3], seg.shape[:3]))
    raw_to_uint8 = is_uint8_safe(raw)
    out_path = args.output_dir / f"mec8nm_{index:02d}.h5"
    if out_path.exists() and not args.overwrite:
        with h5py.File(out_path, "r") as handle:
            raw_ds = handle["raw"]
            label_ds = handle["labels"]
            if tuple(raw_ds.shape) == common_shape and tuple(label_ds.shape) == common_shape:
                label_summary = summarize_labels(seg, common_shape)
                return {
                    "index": index,
                    "raw_npy": str(raw_path),
                    "label_npy": str(seg_path),
                    "hdf5": str(out_path),
                    "raw_shape_original": list(raw.shape),
                    "label_shape_original": list(seg.shape),
                    "shape": list(common_shape),
                    "raw_dtype_original": str(raw.dtype),
                    "label_dtype_original": str(seg.dtype),
                    "raw_dtype_hdf5": str(raw_ds.dtype),
                    "label_dtype_hdf5": str(label_ds.dtype),
                    "cropped_to_common_shape": list(raw.shape[:3]) != list(common_shape)
                    or list(seg.shape[:3]) != list(common_shape),
                    **label_summary,
                }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunk = (
        min(int(args.chunk_z), common_shape[0]),
        min(int(args.chunk_yx), common_shape[1]),
        min(int(args.chunk_yx), common_shape[2]),
    )
    kwargs = dataset_kwargs(args.compression, args.gzip_level)
    with h5py.File(out_path, "w") as handle:
        raw_ds = handle.create_dataset(
            "raw",
            shape=common_shape,
            dtype=np.uint8 if raw_to_uint8 else raw.dtype,
            chunks=chunk,
            **kwargs,
        )
        label_ds = handle.create_dataset(
            "labels",
            shape=common_shape,
            dtype=seg.dtype,
            chunks=chunk,
            **kwargs,
        )
        copy_volume(raw, raw_ds, common_shape, to_uint8=raw_to_uint8)
        copy_volume(seg, label_ds, common_shape, to_uint8=False)
        handle.attrs["source_dataset"] = "haoyuan7/mec"
        handle.attrs["source_license"] = "cc-by-4.0"
        handle.attrs["sample_index"] = int(index)
        handle.attrs["resolution_xy_nm"] = 8
        handle.attrs["raw_npy"] = str(raw_path)
        handle.attrs["label_npy"] = str(seg_path)
        handle.attrs["label_semantics"] = "myelin instance labels; 0 is background"

    label_summary = summarize_labels(seg, common_shape)
    return {
        "index": index,
        "raw_npy": str(raw_path),
        "label_npy": str(seg_path),
        "hdf5": str(out_path),
        "raw_shape_original": list(raw.shape),
        "label_shape_original": list(seg.shape),
        "shape": list(common_shape),
        "raw_dtype_original": str(raw.dtype),
        "label_dtype_original": str(seg.dtype),
        "raw_dtype_hdf5": "uint8" if raw_to_uint8 else str(raw.dtype),
        "label_dtype_hdf5": str(seg.dtype),
        "cropped_to_common_shape": list(raw.shape[:3]) != list(common_shape)
        or list(seg.shape[:3]) != list(common_shape),
        **label_summary,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    records = []
    for index in range(1, 13):
        start = time.time()
        record = convert_one(index, args)
        record["convert_sec"] = round(time.time() - start, 3)
        records.append(record)
        print(json.dumps({"event": "converted", "index": index, "hdf5": record["hdf5"], "sec": record["convert_sec"]}), flush=True)

    by_index = {record["index"]: record for record in records}
    split_payload = {
        "dataset": "MEC_8nm_myelin",
        "policy": "block-level holdout; train includes odd-shape sample 6, val/test each include a uint8-label block",
        "splits": {
            name: [
                {
                    "index": int(index),
                    "hdf5": by_index[int(index)]["hdf5"],
                    "shape": by_index[int(index)]["shape"],
                    "foreground_fraction": by_index[int(index)]["foreground_fraction"],
                    "num_nonzero_labels": by_index[int(index)]["num_nonzero_labels"],
                }
                for index in indices
            ]
            for name, indices in DEFAULT_SPLIT.items()
        },
    }
    manifest_payload = {
        "dataset": "MEC_8nm_myelin",
        "source_hf_repo": "haoyuan7/mec",
        "license": "cc-by-4.0",
        "resolution_xy_nm": 8,
        "label_semantics": "myelin instance labels; 0 is background",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snapshot_dir": str(args.snapshot_dir),
        "hdf5_dir": str(args.output_dir),
        "records": records,
        "splits_json": str(args.splits),
    }
    write_json(args.manifest, manifest_payload)
    write_json(args.splits, split_payload)
    print(json.dumps({"event": "wrote_manifest", "manifest": str(args.manifest), "splits": str(args.splits)}), flush=True)


if __name__ == "__main__":
    main()
