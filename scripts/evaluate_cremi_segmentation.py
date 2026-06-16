#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dbmim.datasets import labels_to_affinities, normalize_volume
from dbmim.metrics import binary_iou_from_logits, dice_from_logits
from dbmim.models import MAEBackboneAffinityNet
from dbmim.postprocess import affinities_to_connected_components, segmentation_metrics
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
    model: MAEBackboneAffinityNet,
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


def build_model(cfg: dict, checkpoint: Path, device: torch.device) -> MAEBackboneAffinityNet:
    payload = load_checkpoint(checkpoint, map_location="cpu")
    effective_cfg = payload.get("config", cfg)
    model_cfg = effective_cfg.get("model", cfg.get("model", {}))
    model = MAEBackboneAffinityNet(
        in_channels=int(model_cfg.get("in_channels", 1)),
        out_channels=int(model_cfg.get("out_channels", 3)),
        volume_size=tuple(model_cfg.get("volume_size", [16, 128, 128])),
        patch_size=tuple(model_cfg.get("patch_size", [4, 16, 16])),
        embed_dim=int(model_cfg.get("embed_dim", 192)),
        depth=int(model_cfg.get("depth", 6)),
        num_heads=int(model_cfg.get("num_heads", 6)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
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
        "n_pred": obj.n_pred,
        "n_gt": obj.n_gt,
        "n_voxels": obj.n_voxels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CREMI neuron segmentation from dbMiM affinities")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--crop-size", nargs=3, type=int, default=[32, 256, 256])
    parser.add_argument("--stride", nargs=3, type=int, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.4, 0.5, 0.6, 0.7])
    parser.add_argument("--min-size", type=int, default=0)
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
        aff = predict_affinities(model, raw, model_window, stride, device)
        target_aff = labels_to_affinities(torch.from_numpy(label.astype(np.int64))[None]).numpy()[0]
        logits = torch.from_numpy(np.log(np.clip(aff, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - aff, 1e-6, 1.0)))
        target = torch.from_numpy(target_aff)
        aff_dice = dice_from_logits(logits, target)
        aff_iou = binary_iou_from_logits(logits, target)
        for threshold in args.thresholds:
            seg = affinities_to_connected_components(aff, threshold=threshold, min_size=args.min_size)
            metrics = segmentation_metrics(seg, label, ignore_label=args.ignore_label)
            record: dict[str, float | int | str] = {
                "sample": path.name,
                "threshold": float(threshold),
                "affinity_dice": float(aff_dice),
                "affinity_iou": float(aff_iou),
                **metric_dict(metrics),
            }
            print(record, flush=True)
            records.append(record)
            if args.save_seg:
                out_npz = output_dir / f"{path.stem}_thr{threshold:.2f}_seg.npz"
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
        "per_threshold": [],
    }
    for threshold in thresholds:
        rows = [r for r in records if float(r["threshold"]) == threshold]
        mean_row: dict[str, float | int] = {"threshold": threshold, "n": len(rows)}
        for key in [
            "adapted_rand_error",
            "rand_fscore",
            "rand_precision",
            "rand_recall",
            "voi_split",
            "voi_merge",
            "affinity_dice",
            "affinity_iou",
        ]:
            mean_row[key] = float(np.mean([float(r[key]) for r in rows])) if rows else float("nan")
        summary["per_threshold"].append(mean_row)  # type: ignore[index]
    if summary["per_threshold"]:
        summary["best_by_adapted_rand"] = min(
            summary["per_threshold"],  # type: ignore[arg-type]
            key=lambda row: float(row["adapted_rand_error"]),
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
