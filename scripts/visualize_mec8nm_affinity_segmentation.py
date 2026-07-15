#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dbmim.datasets import labels_to_affinities
from dbmim.postprocess import affinities_to_connected_components, segmentation_metrics, waterz_agglomeration
from dbmim.utils import load_config
from evaluate_cremi_segmentation import build_model, predict_affinities, sigmoid_np


def center_crop(path: Path, size: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    with h5py.File(path, "r") as handle:
        shape = tuple(int(v) for v in handle["raw"].shape)
        starts = [(dim - crop) // 2 for dim, crop in zip(shape, size)]
        slices = tuple(slice(start, start + crop) for start, crop in zip(starts, size))
        raw = np.asarray(handle["raw"][slices])
        labels = np.asarray(handle["labels"][slices])
    return raw, labels, [[sl.start, sl.stop] for sl in slices]


def boundaries_2d(labels: np.ndarray) -> np.ndarray:
    out = np.zeros(labels.shape, dtype=bool)
    out[1:] |= labels[1:] != labels[:-1]
    out[:-1] |= labels[:-1] != labels[1:]
    out[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    out[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return out


def topology_maps(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    split_gt: set[int] = set()
    merge_pred: set[int] = set()
    for gt_id in np.unique(gt):
        if gt_id == 0:
            continue
        overlaps = np.unique(pred[gt == gt_id])
        if np.count_nonzero(overlaps) > 1:
            split_gt.add(int(gt_id))
    for pred_id in np.unique(pred):
        if pred_id == 0:
            continue
        overlaps = np.unique(gt[pred == pred_id])
        if np.count_nonzero(overlaps) > 1:
            merge_pred.add(int(pred_id))
    split = np.isin(gt, list(split_gt)) if split_gt else np.zeros(gt.shape, dtype=bool)
    merge = np.isin(pred, list(merge_pred)) if merge_pred else np.zeros(pred.shape, dtype=bool)
    return split, merge


def label_rgb(labels: np.ndarray) -> np.ndarray:
    ids = labels.astype(np.uint64, copy=False)
    rgb = np.stack(
        [
            ((ids * 37 + 17) % 251),
            ((ids * 67 + 29) % 241),
            ((ids * 97 + 43) % 239),
        ],
        axis=-1,
    ).astype(np.float32) / 255.0
    rgb[ids == 0] = 0.0
    return rgb


def render_sample(
    raw: np.ndarray,
    gt: np.ndarray,
    pred_aff: np.ndarray,
    target_aff: np.ndarray,
    segmentation: np.ndarray,
    output: Path,
    title: str,
    backend: str,
) -> None:
    z = raw.shape[0] // 2
    gt_z = gt[z]
    pred_z = segmentation[z]
    gt_boundary = boundaries_2d(gt_z)
    pred_boundary = boundaries_2d(pred_z)
    boundary_error = np.zeros(gt_z.shape, dtype=np.uint8)
    boundary_error[gt_boundary & ~pred_boundary] = 1  # missed GT boundary
    boundary_error[pred_boundary & ~gt_boundary] = 2  # extra predicted boundary
    boundary_error[gt_boundary & pred_boundary] = 3
    split, merge = topology_maps(gt_z, pred_z)
    topology = np.zeros(gt_z.shape, dtype=np.uint8)
    topology[split] = 1
    topology[merge] = 2
    topology[split & merge] = 3

    fig, axes = plt.subplots(4, 4, figsize=(16, 16), constrained_layout=True)
    panels = [
        (raw[z], "Raw EM", "gray", 0, 255),
        (label_rgb(gt_z), "GT instances", None, None, None),
        (label_rgb(pred_z), f"{backend} prediction", None, None, None),
        (boundary_error, "Boundary: FN red, FP blue", ListedColormap(["black", "red", "deepskyblue", "white"]), 0, 3),
    ]
    axis_names = ("z affinity", "y affinity", "x affinity")
    for channel, name in enumerate(axis_names):
        panels.extend(
            [
                (target_aff[channel, z], f"GT {name}", "viridis", 0, 1),
                (pred_aff[channel, z], f"Predicted {name}", "viridis", 0, 1),
                (np.abs(pred_aff[channel, z] - target_aff[channel, z]), f"Absolute error {name}", "magma", 0, 1),
            ]
        )
        if channel == 0:
            panels.append((topology, "Topology: split yellow, merge cyan", ListedColormap(["black", "gold", "cyan", "white"]), 0, 3))

    for ax, (image, panel_title, cmap, vmin, vmax) in zip(axes.flat, panels):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(panel_title)
        ax.set_axis_off()
    for ax in axes.flat[len(panels) :]:
        ax.set_axis_off()
    fig.suptitle(title, fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", default=["mec8nm_08.h5", "mec8nm_10.h5"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--bias", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--temperature", type=float, default=1.15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", choices=["waterz", "graph_cc"], default="waterz")
    args = parser.parse_args()

    cfg = load_config(args.config)
    window = tuple(int(v) for v in cfg["model"]["volume_size"])
    device = torch.device(args.device)
    model = build_model(cfg, args.checkpoint, device)
    records = []
    for sample in args.samples:
        raw, gt, crop = center_crop(args.data_dir / sample, window)
        logits = predict_affinities(
            model,
            raw,
            window,
            window,
            device,
            channels=3,
            return_logits=True,
            normalization_scope="volume",
        )
        bias = np.asarray(args.bias, dtype=np.float32).reshape(3, 1, 1, 1)
        pred_aff = sigmoid_np((logits + bias) / max(args.temperature, 1e-6)).astype(np.float32)
        target_aff = labels_to_affinities(torch.from_numpy(gt.astype(np.int64))[None], replicate_boundary=True).numpy()[0]
        if args.backend == "waterz":
            segmentation = waterz_agglomeration(pred_aff, threshold=args.threshold)
        else:
            segmentation = affinities_to_connected_components(pred_aff, threshold=args.threshold)
        metrics = segmentation_metrics(segmentation, gt, ignore_label=0, backend="internal")
        arand = metrics.adapted_rand_error
        voi = metrics.voi_split + metrics.voi_merge
        output = args.output_dir / f"{Path(sample).stem}_center_diagnostic.png"
        render_sample(
            raw,
            gt,
            pred_aff,
            target_aff,
            segmentation,
            output,
            f"{sample} center crop | {args.backend} | ARAND={arand:.4f}, VOI={voi:.4f}",
            args.backend,
        )
        records.append(
            {
                "sample": sample,
                "crop": crop,
                "checkpoint": str(args.checkpoint),
                "threshold": args.threshold,
                "bias": args.bias,
                "temperature": args.temperature,
                "backend": args.backend,
                "arand": arand,
                "voi": voi,
                "n_gt": metrics.n_gt,
                "n_pred": metrics.n_pred,
                "figure": str(output),
            }
        )
        print(json.dumps(records[-1], sort_keys=True), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "diagnostic_metadata.json").write_text(json.dumps(records, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
