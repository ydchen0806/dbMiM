from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _lazy_h5py():
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for HDF5 data. Install requirements-dbMIM.txt.") from exc
    return h5py


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(volume)
    if volume.ndim == 2:
        volume = volume[None, ...]
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = volume[..., 0]
    volume = volume.astype(np.float32)
    if volume.max(initial=0) > 2.0:
        volume = volume / 255.0
    lo = float(np.percentile(volume, 1))
    hi = float(np.percentile(volume, 99))
    if hi > lo:
        volume = np.clip((volume - lo) / (hi - lo), 0.0, 1.0)
    return volume


def random_crop_3d(
    volume: np.ndarray,
    crop_size: tuple[int, int, int],
    rng: random.Random,
) -> np.ndarray:
    depth, height, width = volume.shape
    cd, ch, cw = crop_size
    pad_d = max(0, cd - depth)
    pad_h = max(0, ch - height)
    pad_w = max(0, cw - width)
    if pad_d or pad_h or pad_w:
        volume = np.pad(
            volume,
            ((0, pad_d), (0, pad_h), (0, pad_w)),
            mode="reflect",
        )
        depth, height, width = volume.shape
    z0 = rng.randint(0, depth - cd) if depth > cd else 0
    y0 = rng.randint(0, height - ch) if height > ch else 0
    x0 = rng.randint(0, width - cw) if width > cw else 0
    return volume[z0 : z0 + cd, y0 : y0 + ch, x0 : x0 + cw]


def augment_image_and_label(
    volume: torch.Tensor,
    label: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if torch.rand(()) < 0.5:
        volume = volume.flip(-1)
        if label is not None:
            label = label.flip(-1)
    if torch.rand(()) < 0.5:
        volume = volume.flip(-2)
        if label is not None:
            label = label.flip(-2)
    if torch.rand(()) < 0.2:
        volume = volume.flip(-3)
        if label is not None:
            label = label.flip(-3)
    if torch.rand(()) < 0.35:
        gain = 0.85 + 0.30 * torch.rand((), device=volume.device)
        bias = 0.10 * (torch.rand((), device=volume.device) - 0.5)
        volume = (volume * gain + bias).clamp(0.0, 1.0)
    if torch.rand(()) < 0.25:
        volume = (volume + torch.randn_like(volume) * 0.03).clamp(0.0, 1.0)
    return volume, label


def augment_volume(volume: torch.Tensor) -> torch.Tensor:
    volume, _ = augment_image_and_label(volume, None)
    return volume


def widen_instance_boundaries_2d(labels: np.ndarray, radius: int = 1) -> np.ndarray:
    """SuperHuman/Kisuk-style in-plane instance border invalidation."""

    labels = np.asarray(labels)
    if radius <= 0:
        return labels
    if labels.ndim != 3:
        raise ValueError(f"expected [D,H,W] labels, got shape={labels.shape}")
    try:
        from scipy import ndimage  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for data.widen_border=true") from exc
    out = labels.copy()
    size = (2 * int(radius) + 1, 2 * int(radius) + 1)
    for z in range(out.shape[0]):
        plane = out[z]
        if plane.size == 0:
            continue
        max_id = plane.max(initial=0)
        plane_no_zero = plane.copy()
        plane_no_zero[plane_no_zero == 0] = max_id + 1
        local_max = ndimage.maximum_filter(plane, size=size, mode="reflect")
        local_min = ndimage.minimum_filter(plane_no_zero, size=size, mode="reflect")
        plane[local_max != local_min] = 0
    return out


class SyntheticEMDataset(Dataset):
    """Fast synthetic EM-like volumes for CI/smoke tests."""

    def __init__(
        self,
        length: int = 64,
        volume_size: tuple[int, int, int] = (16, 64, 64),
        with_labels: bool = False,
        seed: int = 0,
    ) -> None:
        self.length = length
        self.volume_size = volume_size
        self.with_labels = with_labels
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(self.seed + index)
        d, h, w = self.volume_size
        z = torch.linspace(-1.0, 1.0, d).view(d, 1, 1)
        y = torch.linspace(-1.0, 1.0, h).view(1, h, 1)
        x = torch.linspace(-1.0, 1.0, w).view(1, 1, w)
        volume = torch.zeros(d, h, w)
        labels = torch.zeros(d, h, w, dtype=torch.long)
        for obj_id in range(1, 7):
            center = torch.rand(3, generator=gen) * 1.6 - 0.8
            radius = torch.rand(3, generator=gen) * torch.tensor([0.30, 0.27, 0.27]) + torch.tensor([0.25, 0.18, 0.18])
            cz, cy, cx = center
            rz, ry, rx = radius
            blob = ((z - cz) / rz).pow(2) + ((y - cy) / ry).pow(2) + ((x - cx) / rx).pow(2)
            mask = blob < 1.0
            volume = torch.maximum(volume, torch.exp(-blob * 2.5))
            labels[mask] = obj_id
        volume = 0.15 + 0.75 * volume + 0.06 * torch.randn(d, h, w, generator=gen)
        volume = volume.clamp(0.0, 1.0).unsqueeze(0)
        if self.with_labels:
            return {"image": volume, "label": labels}
        return {"image": volume}


class EMVolumeDataset(Dataset):
    def __init__(
        self,
        paths: Iterable[str | Path],
        volume_size: tuple[int, int, int] = (16, 64, 64),
        label_paths: Iterable[str | Path] | None = None,
        keys: Iterable[str] | None = None,
        label_keys: Iterable[str] | None = None,
        length_multiplier: int = 1,
        augment: bool = True,
        widen_border: bool = False,
        widen_border_radius: int = 1,
        seed: int = 0,
    ) -> None:
        self.paths = self._expand_paths([Path(p) for p in paths])
        self.label_paths = self._expand_paths([Path(p) for p in label_paths]) if label_paths is not None else None
        self.keys = list(keys) if keys is not None else None
        self.label_keys = list(label_keys) if label_keys is not None else None
        self.volume_size = volume_size
        self.length_multiplier = max(1, length_multiplier)
        self.augment = augment
        self.widen_border = widen_border
        self.widen_border_radius = int(widen_border_radius)
        self.seed = seed
        if not self.paths:
            raise ValueError("EMVolumeDataset requires at least one input path")
        if self.label_paths is not None and len(self.paths) != len(self.label_paths):
            raise ValueError(f"image/label path count mismatch: {len(self.paths)} vs {len(self.label_paths)}")

    def _expand_paths(self, paths: list[Path]) -> list[Path]:
        expanded: list[Path] = []
        for path in paths:
            if path.is_dir():
                hdf_files = sorted(
                    p for p in path.rglob("*") if p.suffix.lower() in {".h5", ".hdf", ".hdf5"}
                )
                if hdf_files:
                    expanded.extend(hdf_files)
                else:
                    expanded.append(path)
            else:
                expanded.append(path)
        return expanded

    def __len__(self) -> int:
        return len(self.paths) * self.length_multiplier

    def _read_h5(self, path: Path, preferred_keys: list[str] | None = None) -> np.ndarray:
        h5py = _lazy_h5py()
        with h5py.File(path, "r") as handle:
            dataset = self._select_h5_dataset(handle, preferred_keys)
            return dataset[()]

    def _select_h5_dataset(self, handle, preferred_keys: list[str] | None = None):
        if preferred_keys:
            for key in preferred_keys:
                if key in handle:
                    item = handle[key]
                    if hasattr(item, "shape"):
                        return item
        for key in ["main", "raw", "image", "images", "data", "volumes/raw", "stack"]:
            if key in handle:
                item = handle[key]
                if hasattr(item, "shape"):
                    return item
        datasets = []
        handle.visititems(lambda _name, obj: datasets.append(obj) if hasattr(obj, "shape") else None)
        if not datasets:
            raise ValueError("HDF5 file does not contain any dataset")
        return datasets[0]

    def _crop_params(
        self,
        shape: tuple[int, ...],
        rng: random.Random,
        crop: tuple[int, int, int, int, int, int] | None = None,
    ) -> tuple[int, int, int, int, int, int]:
        if len(shape) < 3:
            raise ValueError(f"expected at least 3D data, got shape={shape}")
        depth, height, width = int(shape[0]), int(shape[1]), int(shape[2])
        cd, ch, cw = self.volume_size
        if crop is not None:
            return crop
        z0 = rng.randint(0, depth - cd) if depth > cd else 0
        y0 = rng.randint(0, height - ch) if height > ch else 0
        x0 = rng.randint(0, width - cw) if width > cw else 0
        return z0, y0, x0, cd, ch, cw

    def _pad_crop(self, arr: np.ndarray, label: bool) -> np.ndarray:
        cd, ch, cw = self.volume_size
        arr = np.asarray(arr)
        pad = (
            (0, max(0, cd - arr.shape[0])),
            (0, max(0, ch - arr.shape[1])),
            (0, max(0, cw - arr.shape[2])),
        )
        if not any(v for pair in pad for v in pair):
            return arr
        mode = "edge" if label or min(arr.shape[:3]) < 2 else "reflect"
        return np.pad(arr, pad, mode=mode)

    def _read_h5_crop(
        self,
        path: Path,
        preferred_keys: list[str] | None,
        rng: random.Random,
        crop: tuple[int, int, int, int, int, int] | None = None,
        label: bool = False,
    ) -> tuple[np.ndarray, tuple[int, int, int, int, int, int]]:
        h5py = _lazy_h5py()
        with h5py.File(path, "r") as handle:
            dataset = self._select_h5_dataset(handle, preferred_keys)
            if len(dataset.shape) != 3:
                arr = np.asarray(dataset[()])
                crop_params = self._crop_params(arr.shape, rng, crop)
                return random_crop_3d(arr, self.volume_size, rng), crop_params
            crop_params = self._crop_params(tuple(dataset.shape), rng, crop)
            z0, y0, x0, cd, ch, cw = crop_params
            z1 = min(z0 + cd, dataset.shape[0])
            y1 = min(y0 + ch, dataset.shape[1])
            x1 = min(x0 + cw, dataset.shape[2])
            arr = np.asarray(dataset[z0:z1, y0:y1, x0:x1])
            return self._pad_crop(arr, label=label), crop_params

    def _read_image_stack(self, path: Path) -> np.ndarray:
        from PIL import Image

        if path.is_dir():
            files = sorted(
                p
                for p in path.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            )
            if not files:
                raise ValueError(f"no image files under {path}")
            return np.stack([np.asarray(Image.open(p).convert("L")) for p in files], axis=0)
        return np.asarray(Image.open(path).convert("L"))[None, ...]

    def _read_volume(self, path: Path, label: bool = False) -> np.ndarray:
        if path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
            keys = self.keys
            if label:
                keys = self.label_keys or ["label", "labels", "gt", "volumes/labels/neuron_ids", "main"]
            return self._read_h5(path, keys)
        return self._read_image_stack(path)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base_idx = index % len(self.paths)
        rng = random.Random(self.seed + index)
        image_path = self.paths[base_idx]
        crop = None
        if image_path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
            image_np, crop = self._read_h5_crop(image_path, self.keys, rng, label=False)
            image_np = normalize_volume(image_np)
        else:
            image_np = normalize_volume(self._read_volume(image_path, label=False))
            image_np = random_crop_3d(image_np, self.volume_size, rng)
        image = torch.from_numpy(image_np).float().unsqueeze(0)
        item = {"image": image}
        if self.label_paths is not None:
            label_rng = random.Random(self.seed + index)
            label_path = self.label_paths[base_idx]
            if label_path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
                label_np, _ = self._read_h5_crop(
                    label_path,
                    self.label_keys or ["label", "labels", "gt", "volumes/labels/neuron_ids", "main"],
                    label_rng,
                    crop=crop,
                    label=True,
                )
            else:
                label_np = self._read_volume(label_path, label=True)
                if label_np.ndim == 2:
                    label_np = label_np[None, ...]
                label_np = random_crop_3d(label_np, self.volume_size, label_rng)
            if self.widen_border:
                label_np = widen_instance_boundaries_2d(label_np, radius=self.widen_border_radius)
            item["label"] = torch.from_numpy(label_np.astype(np.int64))
        if self.augment:
            if "label" in item:
                image_aug, label_aug = augment_image_and_label(item["image"], item["label"])
                item["image"] = image_aug
                item["label"] = label_aug if label_aug is not None else item["label"]
            else:
                item["image"] = augment_volume(item["image"])
        return item


def labels_to_affinities(labels: torch.Tensor, replicate_boundary: bool = False) -> torch.Tensor:
    """Convert instance labels [B,D,H,W] to nearest-neighbor z/y/x affinities.

    SuperHuman's ``seg_to_aff(..., pad="replicate")`` treats the first
    z/y/x boundary plane as a valid foreground self-edge. The default keeps the
    previous dbMiM behavior for backward-compatible experiments.
    """
    if labels.ndim == 3:
        labels = labels.unsqueeze(0)
    labels = labels.long()
    bsz, depth, height, width = labels.shape
    aff = labels.new_zeros((bsz, 3, depth, height, width), dtype=torch.float32)
    aff[:, 0, 1:] = (labels[:, 1:] == labels[:, :-1]) & (labels[:, 1:] > 0)
    aff[:, 1, :, 1:] = (labels[:, :, 1:] == labels[:, :, :-1]) & (labels[:, :, 1:] > 0)
    aff[:, 2, :, :, 1:] = (labels[:, :, :, 1:] == labels[:, :, :, :-1]) & (labels[:, :, :, 1:] > 0)
    if replicate_boundary:
        aff[:, 0, 0] = labels[:, 0] > 0
        aff[:, 1, :, 0] = labels[:, :, 0] > 0
        aff[:, 2, :, :, 0] = labels[:, :, :, 0] > 0
    return aff.float()


def labels_to_local_shape_descriptors(labels: torch.Tensor) -> torch.Tensor:
    """Build a compact LSD-style target from instance labels.

    Channels are foreground, dz, dy, dx. The offset channels point from each
    foreground voxel to the instance centroid in normalized crop coordinates.
    This is intentionally lightweight: it captures object-level shape context
    without adding funlib/lsd as an offline dependency in SiFlow pods.
    """

    if labels.ndim == 3:
        labels = labels.unsqueeze(0)
    labels = labels.long()
    bsz, depth, height, width = labels.shape
    device = labels.device
    desc = torch.zeros((bsz, 4, depth, height, width), device=device, dtype=torch.float32)
    desc[:, 0] = (labels > 0).float()

    z_axis = torch.linspace(-1.0, 1.0, depth, device=device, dtype=torch.float32)
    y_axis = torch.linspace(-1.0, 1.0, height, device=device, dtype=torch.float32)
    x_axis = torch.linspace(-1.0, 1.0, width, device=device, dtype=torch.float32)
    z_grid, y_grid, x_grid = torch.meshgrid(z_axis, y_axis, x_axis, indexing="ij")

    z_flat = z_grid.reshape(-1)
    y_flat = y_grid.reshape(-1)
    x_flat = x_grid.reshape(-1)
    for batch_idx in range(bsz):
        labels_b = labels[batch_idx]
        flat = labels_b.reshape(-1)
        unique, inverse, counts = torch.unique(flat, sorted=True, return_inverse=True, return_counts=True)
        counts_f = counts.to(torch.float32).clamp_min(1.0)
        sums_z = torch.zeros(unique.numel(), device=device, dtype=torch.float32)
        sums_y = torch.zeros_like(sums_z)
        sums_x = torch.zeros_like(sums_z)
        sums_z.scatter_add_(0, inverse, z_flat)
        sums_y.scatter_add_(0, inverse, y_flat)
        sums_x.scatter_add_(0, inverse, x_flat)
        center_z = sums_z[inverse] / counts_f[inverse]
        center_y = sums_y[inverse] / counts_f[inverse]
        center_x = sums_x[inverse] / counts_f[inverse]
        foreground = (flat > 0).to(torch.float32)
        desc[batch_idx, 1] = ((center_z - z_flat) * foreground).view(depth, height, width).clamp(-1.0, 1.0)
        desc[batch_idx, 2] = ((center_y - y_flat) * foreground).view(depth, height, width).clamp(-1.0, 1.0)
        desc[batch_idx, 3] = ((center_x - x_flat) * foreground).view(depth, height, width).clamp(-1.0, 1.0)
    return desc


def resize_volume_if_needed(x: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    if tuple(x.shape[-3:]) == size:
        return x
    return F.interpolate(x, size=size, mode="trilinear", align_corners=False)
