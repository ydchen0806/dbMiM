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


def augment_volume(volume: torch.Tensor) -> torch.Tensor:
    if torch.rand(()) < 0.5:
        volume = volume.flip(-1)
    if torch.rand(()) < 0.5:
        volume = volume.flip(-2)
    if torch.rand(()) < 0.2:
        volume = volume.flip(-3)
    if torch.rand(()) < 0.35:
        gain = 0.85 + 0.30 * torch.rand((), device=volume.device)
        bias = 0.10 * (torch.rand((), device=volume.device) - 0.5)
        volume = (volume * gain + bias).clamp(0.0, 1.0)
    if torch.rand(()) < 0.25:
        volume = (volume + torch.randn_like(volume) * 0.03).clamp(0.0, 1.0)
    return volume


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
        length_multiplier: int = 1,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        self.paths = self._expand_paths([Path(p) for p in paths])
        self.label_paths = self._expand_paths([Path(p) for p in label_paths]) if label_paths is not None else None
        self.keys = list(keys) if keys is not None else None
        self.volume_size = volume_size
        self.length_multiplier = max(1, length_multiplier)
        self.augment = augment
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
            if preferred_keys:
                for key in preferred_keys:
                    if key in handle:
                        return handle[key][()]
            for key in ["main", "raw", "image", "images", "data", "volumes/raw", "stack"]:
                if key in handle:
                    return handle[key][()]
            first = next(iter(handle.keys()))
            return handle[first][()]

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
                keys = ["label", "labels", "gt", "volumes/labels/neuron_ids", "main"]
            return self._read_h5(path, keys)
        return self._read_image_stack(path)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base_idx = index % len(self.paths)
        rng = random.Random(self.seed + index)
        image_np = normalize_volume(self._read_volume(self.paths[base_idx], label=False))
        image_np = random_crop_3d(image_np, self.volume_size, rng)
        image = torch.from_numpy(image_np).float().unsqueeze(0)
        if self.augment:
            image = augment_volume(image)
        item = {"image": image}
        if self.label_paths is not None:
            label_np = self._read_volume(self.label_paths[base_idx], label=True)
            if label_np.ndim == 2:
                label_np = label_np[None, ...]
            label_np = random_crop_3d(label_np, self.volume_size, rng)
            item["label"] = torch.from_numpy(label_np.astype(np.int64))
        return item


def labels_to_affinities(labels: torch.Tensor) -> torch.Tensor:
    """Convert instance labels [B,D,H,W] to nearest-neighbor z/y/x affinities."""
    if labels.ndim == 3:
        labels = labels.unsqueeze(0)
    labels = labels.long()
    bsz, depth, height, width = labels.shape
    aff = labels.new_zeros((bsz, 3, depth, height, width), dtype=torch.float32)
    aff[:, 0, 1:] = (labels[:, 1:] == labels[:, :-1]) & (labels[:, 1:] > 0)
    aff[:, 1, :, 1:] = (labels[:, :, 1:] == labels[:, :, :-1]) & (labels[:, :, 1:] > 0)
    aff[:, 2, :, :, 1:] = (labels[:, :, :, 1:] == labels[:, :, :, :-1]) & (labels[:, :, :, 1:] > 0)
    return aff.float()


def resize_volume_if_needed(x: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    if tuple(x.shape[-3:]) == size:
        return x
    return F.interpolate(x, size=size, mode="trilinear", align_corners=False)
