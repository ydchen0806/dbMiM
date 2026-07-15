#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class SlabSpec:
    index: int
    start: int
    stop: int
    core_start: int
    core_stop: int
    label_offset: int


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def find(self, value: int) -> int:
        parent = self.parent.get(value)
        if parent is None:
            self.parent[value] = value
            self.rank[value] = 0
            return value
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        rank_a = self.rank[ra]
        rank_b = self.rank[rb]
        if rank_a < rank_b:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if rank_a == rank_b:
            self.rank[ra] += 1
        return True


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


def make_slabs(length: int, slab_size: int, overlap: int, *, max_label: int) -> list[SlabSpec]:
    if slab_size <= 0:
        raise ValueError("--slab-size must be positive")
    if overlap <= 0:
        raise ValueError("--overlap must be positive")
    if overlap >= slab_size:
        raise ValueError("--overlap must be smaller than --slab-size")
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
    label_stride = int(max_label) + 1
    for idx, (start_i, stop_i) in enumerate(starts):
        if idx == 0:
            core_start = start_i
        else:
            prev_stop = starts[idx - 1][1]
            core_start = (start_i + prev_stop) // 2
        if idx == len(starts) - 1:
            core_stop = stop_i
        else:
            next_start = starts[idx + 1][0]
            core_stop = (stop_i + next_start) // 2
        if core_start >= core_stop:
            raise RuntimeError(f"empty core for slab {idx}: {core_start}:{core_stop}")
        slabs.append(
            SlabSpec(
                index=idx,
                start=int(start_i),
                stop=int(stop_i),
                core_start=int(core_start),
                core_stop=int(core_stop),
                label_offset=int(idx * label_stride),
            )
        )
    return slabs


def axis_slice(axis: int, start: int, stop: int) -> tuple[slice, slice, slice]:
    slices = [slice(None), slice(None), slice(None)]
    slices[axis] = slice(int(start), int(stop))
    return tuple(slices)  # type: ignore[return-value]


def crop_slices(start: Iterable[int], shape: Iterable[int], full_shape: tuple[int, int, int]) -> tuple[slice, slice, slice]:
    starts = tuple(int(v) for v in start)
    shapes = tuple(int(v) for v in shape)
    if len(starts) != 3 or len(shapes) != 3:
        raise ValueError("crop start/shape must have three values")
    slices: list[slice] = []
    for axis, (s, n, full) in enumerate(zip(starts, shapes, full_shape)):
        if n <= 0:
            slices.append(slice(0, full))
            continue
        e = min(full, s + n)
        if s < 0 or s >= e:
            raise ValueError(f"invalid crop on axis {axis}: start={s}, shape={n}, full={full}")
        slices.append(slice(s, e))
    return tuple(slices)  # type: ignore[return-value]


def local_labels(seg: np.ndarray, slab: SlabSpec, axis: int, start: int, stop: int) -> np.ndarray:
    arr = np.asarray(seg[axis_slice(axis, start, stop)], dtype=np.uint64)
    if slab.label_offset:
        arr = arr.copy()
        mask = arr != 0
        arr[mask] += np.uint64(slab.label_offset)
    return arr


def update_counts(values: np.ndarray, counts: dict[int, int], *, chunk_size: int) -> None:
    flat = values.ravel()
    for start in range(0, flat.size, chunk_size):
        chunk = flat[start : start + chunk_size]
        chunk = chunk[chunk != 0]
        if chunk.size == 0:
            continue
        uniq, cnt = np.unique(chunk, return_counts=True)
        for label, count in zip(uniq, cnt):
            counts[int(label)] = counts.get(int(label), 0) + int(count)


def encoded_pair_counts(
    left: np.ndarray,
    right: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    if left.shape != right.shape:
        raise ValueError(f"overlap shape mismatch: {left.shape} vs {right.shape}")
    left_counts: dict[int, int] = {}
    right_counts: dict[int, int] = {}
    pair_counts: dict[int, int] = {}
    lf = left.ravel()
    rf = right.ravel()
    for start in range(0, lf.size, chunk_size):
        a = lf[start : start + chunk_size]
        b = rf[start : start + chunk_size]
        update_counts(a, left_counts, chunk_size=chunk_size)
        update_counts(b, right_counts, chunk_size=chunk_size)
        mask = (a != 0) & (b != 0)
        if not np.any(mask):
            continue
        aa = np.asarray(a[mask], dtype=np.uint64)
        bb = np.asarray(b[mask], dtype=np.uint64)
        if int(aa.max(initial=0)) >= (1 << 32) or int(bb.max(initial=0)) >= (1 << 32):
            raise ValueError("encoded pair path requires labels < 2^32")
        encoded = (aa << np.uint64(32)) | bb
        uniq, cnt = np.unique(encoded, return_counts=True)
        for pair, count in zip(uniq, cnt):
            pair_counts[int(pair)] = pair_counts.get(int(pair), 0) + int(count)
    return left_counts, right_counts, pair_counts


def decode_pair(pair: int) -> tuple[int, int]:
    return int(pair >> 32), int(pair & 0xFFFFFFFF)


def build_overlap_unions(
    seg: np.ndarray,
    slabs: list[SlabSpec],
    *,
    axis: int,
    min_contact_voxels: int,
    min_overlap_ratio: float,
    chunk_size: int,
) -> tuple[UnionFind, list[dict[str, int | float]]]:
    uf = UnionFind()
    contacts: list[dict[str, int | float]] = []
    for left_slab, right_slab in zip(slabs[:-1], slabs[1:]):
        overlap_start = max(left_slab.start, right_slab.start)
        overlap_stop = min(left_slab.stop, right_slab.stop)
        if overlap_start >= overlap_stop:
            continue
        left = local_labels(seg, left_slab, axis, overlap_start, overlap_stop)
        right = local_labels(seg, right_slab, axis, overlap_start, overlap_stop)
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
                    "overlap_start": overlap_start,
                    "overlap_stop": overlap_stop,
                    "left_label": left_label,
                    "right_label": right_label,
                    "contact_voxels": int(count),
                    "left_overlap_voxels": int(left_counts.get(left_label, 0)),
                    "right_overlap_voxels": int(right_counts.get(right_label, 0)),
                    "left_overlap_ratio": left_ratio,
                    "right_overlap_ratio": right_ratio,
                    "overlap_ratio": ratio,
                    "accepted": int(accepted),
                }
            )
    return uf, contacts


def map_local_to_global(local: np.ndarray, uf: UnionFind, root_to_id: dict[int, int]) -> np.ndarray:
    labels, inverse = np.unique(local, return_inverse=True)
    mapped_labels = np.zeros(labels.shape, dtype=np.uint64)
    for idx, label in enumerate(labels):
        label_i = int(label)
        if label_i == 0:
            continue
        root = uf.find(label_i)
        global_id = root_to_id.get(root)
        if global_id is None:
            global_id = len(root_to_id) + 1
            root_to_id[root] = global_id
        mapped_labels[idx] = np.uint64(global_id)
    return mapped_labels[inverse].reshape(local.shape)


def compose_stitched(seg: np.ndarray, slabs: list[SlabSpec], *, axis: int, uf: UnionFind) -> np.ndarray:
    output = np.zeros(seg.shape, dtype=np.uint64)
    root_to_id: dict[int, int] = {}
    for slab in slabs:
        local = local_labels(seg, slab, axis, slab.core_start, slab.core_stop)
        mapped = map_local_to_global(local, uf, root_to_id)
        output[axis_slice(axis, slab.core_start, slab.core_stop)] = mapped
    dtype = np.uint32 if len(root_to_id) <= np.iinfo(np.uint32).max else np.uint64
    return output.astype(dtype, copy=False)


def partition_equivalence_summary(
    reference: np.ndarray,
    stitched: np.ndarray,
    *,
    ignore_label: int,
    chunk_size: int,
) -> dict[str, int | bool]:
    ref_to_test: dict[int, set[int]] = {}
    test_to_ref: dict[int, set[int]] = {}
    rf = reference.ravel()
    sf = stitched.ravel()
    total_nonignore = 0
    for start in range(0, rf.size, chunk_size):
        ref = np.asarray(rf[start : start + chunk_size], dtype=np.uint64)
        st = np.asarray(sf[start : start + chunk_size], dtype=np.uint64)
        mask = (ref != int(ignore_label)) & (st != int(ignore_label))
        if not np.any(mask):
            continue
        ref = ref[mask]
        st = st[mask]
        total_nonignore += int(ref.size)
        if int(ref.max(initial=0)) >= (1 << 32) or int(st.max(initial=0)) >= (1 << 32):
            raise ValueError("partition summary requires labels < 2^32")
        encoded = (ref << np.uint64(32)) | st
        pairs = np.unique(encoded)
        for pair in pairs:
            ref_label, stitched_label = decode_pair(int(pair))
            ref_to_test.setdefault(ref_label, set()).add(stitched_label)
            test_to_ref.setdefault(stitched_label, set()).add(ref_label)
    ref_split = {label: values for label, values in ref_to_test.items() if len(values) > 1}
    stitched_merge = {label: values for label, values in test_to_ref.items() if len(values) > 1}
    ref_split_labels = len(ref_split)
    stitched_merge_labels = len(stitched_merge)
    return {
        "partition_equivalent": ref_split_labels == 0 and stitched_merge_labels == 0,
        "reference_labels": len(ref_to_test),
        "stitched_labels": len(test_to_ref),
        "reference_split_labels": ref_split_labels,
        "stitched_merge_labels": stitched_merge_labels,
        "nonignore_voxels_checked": total_nonignore,
        "reference_split_examples": [
            {"reference_label": int(label), "stitched_labels": [int(v) for v in sorted(values)[:8]]}
            for label, values in sorted(ref_split.items())[:8]
        ],
        "stitched_merge_examples": [
            {"stitched_label": int(label), "reference_labels": [int(v) for v in sorted(values)[:8]]}
            for label, values in sorted(stitched_merge.items())[:8]
        ],
    }


def write_h5(path: Path, data: np.ndarray, metadata: dict[str, object], *, compression: str, compression_level: int) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    compression_arg = None if compression == "none" else compression
    compression_opts = int(compression_level) if compression_arg == "gzip" else None
    chunks = tuple(max(1, min(v, c)) for v, c in zip(data.shape, (16, 256, 256)))
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "segmentation",
            data=data,
            chunks=chunks,
            compression=compression_arg,
            compression_opts=compression_opts,
            shuffle=compression_arg is not None,
        )
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True, default=_json_default)


def write_contacts_csv(path: Path, contacts: list[dict[str, int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not contacts:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(contacts[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(contacts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prototype overlap-contact ID stitching using synthetic slabs cut from one segmentation HDF5."
    )
    parser.add_argument("--input-h5", required=True)
    parser.add_argument("--dataset", default="segmentation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-name", default=None)
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--slab-size", type=int, default=48)
    parser.add_argument("--overlap", type=int, default=12)
    parser.add_argument("--crop-start", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--crop-shape", nargs=3, type=int, default=[0, 0, 0])
    parser.add_argument("--min-contact-voxels", type=int, default=64)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--ignore-label", type=int, default=0)
    parser.add_argument("--chunk-voxels", type=int, default=5_000_000)
    parser.add_argument("--compression", choices=["gzip", "lzf", "none"], default="gzip")
    parser.add_argument("--compression-level", type=int, default=4)
    args = parser.parse_args()

    t0 = time.perf_counter()
    input_h5 = Path(args.input_h5)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    import h5py

    with h5py.File(input_h5, "r") as handle:
        ds = handle[str(args.dataset)]
        full_shape = tuple(int(v) for v in ds.shape[:3])
        crop = crop_slices(args.crop_start, args.crop_shape, full_shape)
        seg = np.asarray(ds[crop], dtype=np.uint32)
    max_label = int(seg.max(initial=0))
    slabs = make_slabs(seg.shape[int(args.axis)], int(args.slab_size), int(args.overlap), max_label=max_label)
    uf, contacts = build_overlap_unions(
        seg,
        slabs,
        axis=int(args.axis),
        min_contact_voxels=int(args.min_contact_voxels),
        min_overlap_ratio=float(args.min_overlap_ratio),
        chunk_size=int(args.chunk_voxels),
    )
    accepted_contacts = sum(1 for row in contacts if int(row["accepted"]) == 1)
    stitched = compose_stitched(seg, slabs, axis=int(args.axis), uf=uf)
    equivalence = partition_equivalence_summary(
        seg,
        stitched,
        ignore_label=int(args.ignore_label),
        chunk_size=int(args.chunk_voxels),
    )

    sample_name = args.sample_name or input_h5.stem
    output_h5 = output_dir / f"{sample_name}_synthetic_slab_stitched_axis{args.axis}_slab{args.slab_size}_ov{args.overlap}.seg.h5"
    summary = {
        "input_h5": str(input_h5),
        "dataset": str(args.dataset),
        "sample_name": sample_name,
        "crop_start": [int(v) for v in args.crop_start],
        "crop_shape_arg": [int(v) for v in args.crop_shape],
        "effective_shape": list(seg.shape),
        "axis": int(args.axis),
        "slab_size": int(args.slab_size),
        "overlap": int(args.overlap),
        "min_contact_voxels": int(args.min_contact_voxels),
        "min_overlap_ratio": float(args.min_overlap_ratio),
        "num_slabs": len(slabs),
        "slabs": [slab.__dict__ for slab in slabs],
        "source_max_label": max_label,
        "stitched_max_label": int(stitched.max(initial=0)),
        "num_contacts": len(contacts),
        "accepted_contacts": accepted_contacts,
        "output_h5": str(output_h5),
        "equivalence": equivalence,
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    write_h5(
        output_h5,
        stitched,
        summary,
        compression=str(args.compression),
        compression_level=int(args.compression_level),
    )
    write_contacts_csv(output_dir / "overlap_contacts.csv", contacts)
    (output_dir / "stitch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
