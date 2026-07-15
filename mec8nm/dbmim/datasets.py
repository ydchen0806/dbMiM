from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

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
    *,
    rotate_xy: bool = False,
    gamma: bool = False,
    gamma_range: tuple[float, float] = (0.7, 1.5),
    gamma_probability: float = 0.35,
    noise_std: float = 0.03,
    intensity_probability: float = 0.35,
    intensity_gain_range: tuple[float, float] = (0.85, 1.15),
    intensity_bias_range: tuple[float, float] = (-0.05, 0.05),
    xy_blur_probability: float = 0.0,
    xy_blur_kernel: int = 3,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    gamma_probability = float(gamma_probability)
    if not 0.0 <= gamma_probability <= 1.0:
        raise ValueError("gamma_probability must be in [0, 1]")
    if rotate_xy:
        k = int(torch.randint(0, 4, ()).item())
        if k:
            volume = torch.rot90(volume, k, dims=(-2, -1))
            if label is not None:
                label = torch.rot90(label, k, dims=(-2, -1))
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
    if torch.rand(()) < float(intensity_probability):
        gain_lo, gain_hi = (float(v) for v in intensity_gain_range)
        bias_lo, bias_hi = (float(v) for v in intensity_bias_range)
        if gain_hi < gain_lo:
            gain_lo, gain_hi = gain_hi, gain_lo
        if bias_hi < bias_lo:
            bias_lo, bias_hi = bias_hi, bias_lo
        if (gain_lo, gain_hi, bias_lo, bias_hi) == (0.85, 1.15, -0.05, 0.05):
            # Preserve the historical floating-point operation order exactly.
            gain = 0.85 + 0.30 * torch.rand((), device=volume.device)
            bias = 0.10 * (torch.rand((), device=volume.device) - 0.5)
        else:
            gain = gain_lo + (gain_hi - gain_lo) * torch.rand((), device=volume.device)
            bias = bias_lo + (bias_hi - bias_lo) * torch.rand((), device=volume.device)
        volume = (volume * gain + bias).clamp(0.0, 1.0)
    if gamma and gamma_probability > 0.0 and torch.rand(()) < gamma_probability:
        lo, hi = float(gamma_range[0]), float(gamma_range[1])
        if hi < lo:
            lo, hi = hi, lo
        exponent = lo + (hi - lo) * torch.rand((), device=volume.device)
        volume = volume.clamp(1e-4, 1.0).pow(exponent).clamp(0.0, 1.0)
    if torch.rand(()) < 0.25:
        volume = (volume + torch.randn_like(volume) * float(noise_std)).clamp(0.0, 1.0)
    # Keep the default disabled so legacy configurations consume no extra RNG.
    if float(xy_blur_probability) > 0.0 and torch.rand(()) < float(xy_blur_probability):
        kernel = int(xy_blur_kernel)
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError("xy_blur_kernel must be a positive odd integer")
        if kernel > 1:
            volume = F.avg_pool3d(
                volume.unsqueeze(0),
                kernel_size=(1, kernel, kernel),
                stride=1,
                padding=(0, kernel // 2, kernel // 2),
                count_include_pad=False,
            ).squeeze(0)
    return volume, label


def augment_image_and_aligned_labels(
    volume: torch.Tensor,
    label: torch.Tensor,
    aligned_label: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay one augmentation draw for a second coordinate-aligned label."""

    if tuple(label.shape) != tuple(aligned_label.shape):
        raise ValueError(
            f"aligned labels must share a shape, got {label.shape} and {aligned_label.shape}"
        )
    rng_state = torch.random.get_rng_state()
    image_aug, label_aug = augment_image_and_label(volume, label, **kwargs)
    final_rng_state = torch.random.get_rng_state()
    try:
        torch.random.set_rng_state(rng_state)
        _, aligned_aug = augment_image_and_label(volume, aligned_label, **kwargs)
    finally:
        # Keep the default training RNG stream identical to a single augmentation call.
        torch.random.set_rng_state(final_rng_state)
    if label_aug is None or aligned_aug is None:
        raise RuntimeError("aligned label augmentation unexpectedly returned no label")
    return image_aug, label_aug, aligned_aug


def transform_paired_offset(
    offset: Sequence[int],
    *,
    rotate_xy_k: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> tuple[int, int, int]:
    """Transform a second-minus-first crop offset under shared geometry."""

    if len(offset) != 3:
        raise ValueError(f"paired offset must contain z/y/x, got {offset}")
    dz, dy, dx = (int(value) for value in offset)
    for _ in range(int(rotate_xy_k) % 4):
        dy, dx = -dx, dy
    if flip_x:
        dx = -dx
    if flip_y:
        dy = -dy
    if flip_z:
        dz = -dz
    return dz, dy, dx


def augment_paired_image_and_label(
    image: torch.Tensor,
    label: torch.Tensor,
    paired_image: torch.Tensor,
    paired_label: torch.Tensor,
    paired_offset: Sequence[int],
    *,
    rotate_xy: bool = False,
    gamma: bool = False,
    gamma_range: tuple[float, float] = (0.7, 1.5),
    gamma_probability: float = 0.35,
    noise_std: float = 0.03,
    intensity_probability: float = 0.35,
    intensity_gain_range: tuple[float, float] = (0.85, 1.15),
    intensity_bias_range: tuple[float, float] = (-0.05, 0.05),
    xy_blur_probability: float = 0.0,
    xy_blur_kernel: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one geometric/photometric transform to two overlapping crops."""

    if tuple(image.shape) != tuple(paired_image.shape):
        raise ValueError(f"paired images must share a shape, got {image.shape} and {paired_image.shape}")
    if tuple(label.shape) != tuple(paired_label.shape):
        raise ValueError(f"paired labels must share a shape, got {label.shape} and {paired_label.shape}")
    if tuple(image.shape[-3:]) != tuple(label.shape[-3:]):
        raise ValueError(f"paired image/label spatial shapes differ: {image.shape} and {label.shape}")
    if not 0.0 <= float(gamma_probability) <= 1.0:
        raise ValueError("gamma_probability must be in [0, 1]")
    if not 0.0 <= float(intensity_probability) <= 1.0:
        raise ValueError("intensity_probability must be in [0, 1]")
    if not 0.0 <= float(xy_blur_probability) <= 1.0:
        raise ValueError("xy_blur_probability must be in [0, 1]")

    images = torch.stack((image, paired_image), dim=0)
    labels = torch.stack((label, paired_label), dim=0)
    rotate_k = int(torch.randint(0, 4, ()).item()) if rotate_xy else 0
    if rotate_k:
        images = torch.rot90(images, rotate_k, dims=(-2, -1))
        labels = torch.rot90(labels, rotate_k, dims=(-2, -1))
    flip_x = bool(torch.rand(()) < 0.5)
    flip_y = bool(torch.rand(()) < 0.5)
    flip_z = bool(torch.rand(()) < 0.2)
    if flip_x:
        images = images.flip(-1)
        labels = labels.flip(-1)
    if flip_y:
        images = images.flip(-2)
        labels = labels.flip(-2)
    if flip_z:
        images = images.flip(-3)
        labels = labels.flip(-3)
    transformed_offset = transform_paired_offset(
        paired_offset,
        rotate_xy_k=rotate_k,
        flip_x=flip_x,
        flip_y=flip_y,
        flip_z=flip_z,
    )

    if torch.rand(()) < float(intensity_probability):
        gain_lo, gain_hi = (float(value) for value in intensity_gain_range)
        bias_lo, bias_hi = (float(value) for value in intensity_bias_range)
        if gain_hi < gain_lo:
            gain_lo, gain_hi = gain_hi, gain_lo
        if bias_hi < bias_lo:
            bias_lo, bias_hi = bias_hi, bias_lo
        if (gain_lo, gain_hi, bias_lo, bias_hi) == (0.85, 1.15, -0.05, 0.05):
            gain = 0.85 + 0.30 * torch.rand((), device=images.device)
            bias = 0.10 * (torch.rand((), device=images.device) - 0.5)
        else:
            gain = gain_lo + (gain_hi - gain_lo) * torch.rand((), device=images.device)
            bias = bias_lo + (bias_hi - bias_lo) * torch.rand((), device=images.device)
        images = (images * gain + bias).clamp(0.0, 1.0)
    if gamma and float(gamma_probability) > 0.0 and torch.rand(()) < float(gamma_probability):
        lo, hi = (float(value) for value in gamma_range)
        if hi < lo:
            lo, hi = hi, lo
        exponent = lo + (hi - lo) * torch.rand((), device=images.device)
        images = images.clamp(1e-4, 1.0).pow(exponent).clamp(0.0, 1.0)
    if torch.rand(()) < 0.25:
        images = (images + torch.randn_like(images) * float(noise_std)).clamp(0.0, 1.0)
    if float(xy_blur_probability) > 0.0 and torch.rand(()) < float(xy_blur_probability):
        kernel = int(xy_blur_kernel)
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError("xy_blur_kernel must be a positive odd integer")
        if kernel > 1:
            images = F.avg_pool3d(
                images,
                kernel_size=(1, kernel, kernel),
                stride=1,
                padding=(0, kernel // 2, kernel // 2),
                count_include_pad=False,
            )
    return images[0], labels[0], images[1], labels[1], torch.tensor(transformed_offset, dtype=torch.long)


def augment_volume(
    volume: torch.Tensor,
    *,
    rotate_xy: bool = False,
    gamma: bool = False,
    gamma_range: tuple[float, float] = (0.7, 1.5),
    gamma_probability: float = 0.35,
    noise_std: float = 0.03,
    intensity_probability: float = 0.35,
    intensity_gain_range: tuple[float, float] = (0.85, 1.15),
    intensity_bias_range: tuple[float, float] = (-0.05, 0.05),
    xy_blur_probability: float = 0.0,
    xy_blur_kernel: int = 3,
) -> torch.Tensor:
    volume, _ = augment_image_and_label(
        volume,
        None,
        rotate_xy=rotate_xy,
        gamma=gamma,
        gamma_range=gamma_range,
        gamma_probability=gamma_probability,
        noise_std=noise_std,
        intensity_probability=intensity_probability,
        intensity_gain_range=intensity_gain_range,
        intensity_bias_range=intensity_bias_range,
        xy_blur_probability=xy_blur_probability,
        xy_blur_kernel=xy_blur_kernel,
    )
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
        return_unwidened_label: bool = False,
        seed: int = 0,
    ) -> None:
        self.length = length
        self.volume_size = volume_size
        self.with_labels = with_labels
        self.return_unwidened_label = bool(return_unwidened_label)
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
            item = {"image": volume, "label": labels}
            if self.return_unwidened_label:
                item["unwidened_label"] = labels.clone()
            return item
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
        augment_rotate_xy: bool = False,
        augment_gamma: bool = False,
        augment_gamma_range: tuple[float, float] = (0.7, 1.5),
        augment_gamma_probability: float = 0.35,
        augment_noise_std: float = 0.03,
        augment_intensity_probability: float = 0.35,
        augment_intensity_gain_range: tuple[float, float] = (0.85, 1.15),
        augment_intensity_bias_range: tuple[float, float] = (-0.05, 0.05),
        augment_xy_blur_probability: float = 0.0,
        augment_xy_blur_kernel: int = 3,
        widen_border: bool = False,
        widen_border_radius: int = 1,
        boundary_sampling: bool = False,
        boundary_sampling_attempts: int = 8,
        boundary_sampling_min_fraction: float = 0.001,
        boundary_sampling_min_unique: int = 2,
        boundary_sampling_min_foreground: float = 0.05,
        boundary_sampling_max_foreground: float = 0.995,
        simple_sampling_probability: float = 0.0,
        simple_sampling_attempts: int = 8,
        simple_sampling_min_foreground: float = 0.95,
        simple_sampling_max_boundary_fraction: float = 0.05,
        simple_sampling_max_unique: int = 5,
        simple_sampling_require_eligible: bool = False,
        paired_overlap: bool = False,
        paired_overlap_shift: Sequence[int] = (8, 32, 32),
        return_topology_stats: bool = False,
        return_unwidened_label: bool = False,
        seed: int = 0,
    ) -> None:
        self.paths = self._expand_paths([Path(p) for p in paths])
        self.label_paths = self._expand_paths([Path(p) for p in label_paths]) if label_paths is not None else None
        self.keys = list(keys) if keys is not None else None
        self.label_keys = list(label_keys) if label_keys is not None else None
        self.volume_size = volume_size
        self.length_multiplier = max(1, length_multiplier)
        self.augment = augment
        self.augment_rotate_xy = bool(augment_rotate_xy)
        self.augment_gamma = bool(augment_gamma)
        self.augment_gamma_range = tuple(float(v) for v in augment_gamma_range)
        self.augment_gamma_probability = float(augment_gamma_probability)
        self.augment_noise_std = float(augment_noise_std)
        self.augment_intensity_probability = float(augment_intensity_probability)
        self.augment_intensity_gain_range = tuple(float(v) for v in augment_intensity_gain_range)
        self.augment_intensity_bias_range = tuple(float(v) for v in augment_intensity_bias_range)
        self.augment_xy_blur_probability = float(augment_xy_blur_probability)
        self.augment_xy_blur_kernel = int(augment_xy_blur_kernel)
        if not 0.0 <= self.augment_intensity_probability <= 1.0:
            raise ValueError("augment_intensity_probability must be in [0, 1]")
        if not 0.0 <= self.augment_gamma_probability <= 1.0:
            raise ValueError("augment_gamma_probability must be in [0, 1]")
        if not 0.0 <= self.augment_xy_blur_probability <= 1.0:
            raise ValueError("augment_xy_blur_probability must be in [0, 1]")
        if self.augment_xy_blur_kernel <= 0 or self.augment_xy_blur_kernel % 2 == 0:
            raise ValueError("augment_xy_blur_kernel must be a positive odd integer")
        self.widen_border = widen_border
        self.widen_border_radius = int(widen_border_radius)
        self.boundary_sampling = bool(boundary_sampling)
        self.boundary_sampling_attempts = max(1, int(boundary_sampling_attempts))
        self.boundary_sampling_min_fraction = float(boundary_sampling_min_fraction)
        self.boundary_sampling_min_unique = max(1, int(boundary_sampling_min_unique))
        self.boundary_sampling_min_foreground = float(boundary_sampling_min_foreground)
        self.boundary_sampling_max_foreground = float(boundary_sampling_max_foreground)
        self.simple_sampling_probability = float(simple_sampling_probability)
        if not 0.0 <= self.simple_sampling_probability <= 1.0:
            raise ValueError("simple_sampling_probability must be in [0, 1]")
        self.simple_sampling_attempts = max(1, int(simple_sampling_attempts))
        self.simple_sampling_min_foreground = float(simple_sampling_min_foreground)
        self.simple_sampling_max_boundary_fraction = float(simple_sampling_max_boundary_fraction)
        self.simple_sampling_max_unique = max(1, int(simple_sampling_max_unique))
        self.simple_sampling_require_eligible = bool(simple_sampling_require_eligible)
        self.paired_overlap = bool(paired_overlap)
        if len(paired_overlap_shift) != 3:
            raise ValueError("paired_overlap_shift must contain three z/y/x values")
        self.paired_overlap_shift = tuple(int(value) for value in paired_overlap_shift)
        if self.paired_overlap and any(
            shift <= 0 or shift >= size
            for shift, size in zip(self.paired_overlap_shift, self.volume_size)
        ):
            raise ValueError(
                "paired_overlap_shift must be positive and smaller than the crop on every axis, "
                f"got shift={self.paired_overlap_shift} crop={self.volume_size}"
            )
        if self.paired_overlap and (self.boundary_sampling or self.simple_sampling_probability > 0.0):
            raise ValueError("paired overlap is incompatible with label-guided crop sampling")
        self.return_topology_stats = bool(return_topology_stats)
        self.return_unwidened_label = bool(return_unwidened_label)
        if self.return_unwidened_label and self.paired_overlap:
            raise ValueError("return_unwidened_label is not supported with paired overlap")
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

    def _label_crop_stats(self, labels: np.ndarray) -> tuple[float, float, int, float]:
        labels = np.asarray(labels)
        foreground = labels > 0
        foreground_fraction = float(np.mean(foreground)) if labels.size else 0.0
        unique_nonzero = int(np.unique(labels[foreground]).size) if foreground.any() else 0
        boundary = np.zeros(labels.shape, dtype=bool)
        if labels.shape[0] > 1:
            diff = (labels[1:] != labels[:-1]) & (foreground[1:] | foreground[:-1])
            boundary[1:] |= diff
            boundary[:-1] |= diff
        if labels.shape[1] > 1:
            diff = (labels[:, 1:] != labels[:, :-1]) & (foreground[:, 1:] | foreground[:, :-1])
            boundary[:, 1:] |= diff
            boundary[:, :-1] |= diff
        if labels.shape[2] > 1:
            diff = (labels[:, :, 1:] != labels[:, :, :-1]) & (foreground[:, :, 1:] | foreground[:, :, :-1])
            boundary[:, :, 1:] |= diff
            boundary[:, :, :-1] |= diff
        boundary_fraction = float(np.mean(boundary)) if labels.size else 0.0
        score = boundary_fraction + 0.002 * min(float(unique_nonzero), 64.0)
        if foreground_fraction < self.boundary_sampling_min_foreground:
            score -= self.boundary_sampling_min_foreground - foreground_fraction
        if foreground_fraction > self.boundary_sampling_max_foreground:
            score -= foreground_fraction - self.boundary_sampling_max_foreground
        if unique_nonzero < self.boundary_sampling_min_unique:
            score -= 0.01 * float(self.boundary_sampling_min_unique - unique_nonzero)
        return score, boundary_fraction, unique_nonzero, foreground_fraction

    def _select_label_guided_h5_crop(
        self,
        path: Path,
        preferred_keys: list[str] | None,
        rng: random.Random,
        *,
        select_simple: bool = False,
        require_eligible: bool = False,
    ) -> tuple[tuple[int, int, int, int, int, int], np.ndarray] | None:
        h5py = _lazy_h5py()
        with h5py.File(path, "r") as handle:
            dataset = self._select_h5_dataset(handle, preferred_keys)
            if len(dataset.shape) != 3:
                crop = self._crop_params(tuple(dataset.shape), rng)
                arr = random_crop_3d(np.asarray(dataset[()]), self.volume_size, rng)
                return crop, arr
            best_crop: tuple[int, int, int, int, int, int] | None = None
            best_arr: np.ndarray | None = None
            best_key: tuple[float, ...] | float | None = None
            attempts = self.simple_sampling_attempts if select_simple else self.boundary_sampling_attempts
            for _ in range(attempts):
                crop = self._crop_params(tuple(dataset.shape), rng)
                z0, y0, x0, cd, ch, cw = crop
                z1 = min(z0 + cd, dataset.shape[0])
                y1 = min(y0 + ch, dataset.shape[1])
                x1 = min(x0 + cw, dataset.shape[2])
                arr = self._pad_crop(np.asarray(dataset[z0:z1, y0:y1, x0:x1]), label=True)
                stats = self._label_crop_stats(arr)
                if select_simple:
                    _, boundary_fraction, unique_nonzero, foreground_fraction = stats
                    eligible = (
                        foreground_fraction >= self.simple_sampling_min_foreground
                        and boundary_fraction <= self.simple_sampling_max_boundary_fraction
                        and unique_nonzero <= self.simple_sampling_max_unique
                    )
                    if require_eligible and not eligible:
                        continue
                    deficit = (
                        max(0.0, self.simple_sampling_min_foreground - foreground_fraction)
                        + max(0.0, boundary_fraction - self.simple_sampling_max_boundary_fraction)
                        + 0.02 * max(0, unique_nonzero - self.simple_sampling_max_unique)
                    )
                    candidate_key: tuple[float, ...] | float = (
                        deficit,
                        boundary_fraction,
                        float(unique_nonzero),
                        -foreground_fraction,
                    )
                    is_better = best_key is None or candidate_key < best_key
                else:
                    candidate_key = stats[0]
                    is_better = best_key is None or candidate_key > best_key
                if is_better:
                    best_crop, best_arr, best_key = crop, arr, candidate_key
            if best_crop is None or best_arr is None:
                if select_simple and require_eligible:
                    return None
                raise RuntimeError("label-guided crop selection produced no candidates")
            return best_crop, best_arr

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

    def _paired_crop_params(
        self,
        shape: Sequence[int],
        rng: random.Random,
    ) -> tuple[
        tuple[int, int, int, int, int, int],
        tuple[int, int, int, int, int, int],
        tuple[int, int, int],
    ]:
        if len(shape) != 3:
            raise ValueError(f"paired overlap requires a 3D HDF5 dataset, got shape={tuple(shape)}")
        primary_start: list[int] = []
        offset: list[int] = []
        for full, crop, magnitude in zip(shape, self.volume_size, self.paired_overlap_shift):
            max_start = int(full) - int(crop)
            if max_start < magnitude:
                raise ValueError(
                    "paired overlap shift does not fit source volume: "
                    f"shape={tuple(shape)} crop={self.volume_size} shift={self.paired_overlap_shift}"
                )
            signed = magnitude if rng.random() < 0.5 else -magnitude
            if signed > 0:
                start = rng.randint(0, max_start - magnitude)
            else:
                start = rng.randint(magnitude, max_start)
            primary_start.append(start)
            offset.append(signed)
        paired_start = [start + delta for start, delta in zip(primary_start, offset)]
        size = tuple(int(value) for value in self.volume_size)
        primary = (*primary_start, *size)
        paired = (*paired_start, *size)
        return primary, paired, tuple(offset)

    def _get_paired_h5_item(self, index: int) -> dict[str, torch.Tensor]:
        base_idx = index % len(self.paths)
        image_path = self.paths[base_idx]
        label_path = self.label_paths[base_idx] if self.label_paths is not None else None
        h5_suffixes = {".h5", ".hdf", ".hdf5"}
        if label_path is None or image_path.suffix.lower() not in h5_suffixes or label_path.suffix.lower() not in h5_suffixes:
            raise ValueError("paired overlap requires paired image/label HDF5 paths")
        h5py = _lazy_h5py()
        with h5py.File(image_path, "r") as handle:
            image_dataset = self._select_h5_dataset(handle, self.keys)
            shape = tuple(int(value) for value in image_dataset.shape)
        rng = random.Random(self.seed + index)
        crop, paired_crop, paired_offset = self._paired_crop_params(shape, rng)
        image_np, _ = self._read_h5_crop(image_path, self.keys, rng, crop=crop, label=False)
        paired_image_np, _ = self._read_h5_crop(image_path, self.keys, rng, crop=paired_crop, label=False)
        label_keys = self.label_keys or ["label", "labels", "gt", "volumes/labels/neuron_ids", "main"]
        label_np, _ = self._read_h5_crop(label_path, label_keys, rng, crop=crop, label=True)
        paired_label_np, _ = self._read_h5_crop(label_path, label_keys, rng, crop=paired_crop, label=True)
        topology_stats = self._label_crop_stats(label_np) if self.return_topology_stats else None
        image = torch.from_numpy(normalize_volume(image_np)).float().unsqueeze(0)
        paired_image = torch.from_numpy(normalize_volume(paired_image_np)).float().unsqueeze(0)
        if self.widen_border:
            label_np = widen_instance_boundaries_2d(label_np, radius=self.widen_border_radius)
            paired_label_np = widen_instance_boundaries_2d(paired_label_np, radius=self.widen_border_radius)
        label = torch.from_numpy(label_np.astype(np.int64))
        paired_label = torch.from_numpy(paired_label_np.astype(np.int64))
        offset_tensor = torch.tensor(paired_offset, dtype=torch.long)
        if self.augment:
            image, label, paired_image, paired_label, offset_tensor = augment_paired_image_and_label(
                image,
                label,
                paired_image,
                paired_label,
                paired_offset,
                rotate_xy=self.augment_rotate_xy,
                gamma=self.augment_gamma,
                gamma_range=self.augment_gamma_range,
                gamma_probability=self.augment_gamma_probability,
                noise_std=self.augment_noise_std,
                intensity_probability=self.augment_intensity_probability,
                intensity_gain_range=self.augment_intensity_gain_range,
                intensity_bias_range=self.augment_intensity_bias_range,
                xy_blur_probability=self.augment_xy_blur_probability,
                xy_blur_kernel=self.augment_xy_blur_kernel,
            )
        item = {
            "image": image,
            "label": label,
            "paired_image": paired_image,
            "paired_label": paired_label,
            "paired_offset": offset_tensor,
        }
        if topology_stats is not None:
            _, boundary_fraction, unique_nonzero, foreground_fraction = topology_stats
            item.update(
                {
                    "topology_foreground_fraction": torch.tensor(foreground_fraction, dtype=torch.float32),
                    "topology_boundary_fraction": torch.tensor(boundary_fraction, dtype=torch.float32),
                    "topology_unique_nonzero": torch.tensor(float(unique_nonzero), dtype=torch.float32),
                }
            )
        return item

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.paired_overlap:
            return self._get_paired_h5_item(index)
        base_idx = index % len(self.paths)
        rng = random.Random(self.seed + index)
        image_path = self.paths[base_idx]
        crop = None
        label_np = None
        label_path = self.label_paths[base_idx] if self.label_paths is not None else None
        label_guided_h5 = (
            label_path is not None
            and image_path.suffix.lower() in {".h5", ".hdf", ".hdf5"}
            and label_path.suffix.lower() in {".h5", ".hdf", ".hdf5"}
        )
        use_simple_sampling = (
            label_guided_h5
            and self.simple_sampling_probability > 0.0
            and rng.random() < self.simple_sampling_probability
        )
        if use_simple_sampling:
            selection = self._select_label_guided_h5_crop(
                label_path,
                self.label_keys or ["label", "labels", "gt", "volumes/labels/neuron_ids", "main"],
                rng,
                select_simple=True,
                require_eligible=self.simple_sampling_require_eligible,
            )
            if selection is not None:
                crop, label_np = selection
        if (
            crop is None
            and self.boundary_sampling
            and label_path is not None
            and image_path.suffix.lower() in {".h5", ".hdf", ".hdf5"}
            and label_path.suffix.lower() in {".h5", ".hdf", ".hdf5"}
        ):
            crop, label_np = self._select_label_guided_h5_crop(
                label_path,
                self.label_keys or ["label", "labels", "gt", "volumes/labels/neuron_ids", "main"],
                rng,
            )
        if image_path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
            image_np, crop = self._read_h5_crop(image_path, self.keys, rng, crop=crop, label=False)
            image_np = normalize_volume(image_np)
        else:
            image_np = normalize_volume(self._read_volume(image_path, label=False))
            image_np = random_crop_3d(image_np, self.volume_size, rng)
        image = torch.from_numpy(image_np).float().unsqueeze(0)
        item = {"image": image}
        if self.label_paths is not None:
            label_rng = random.Random(self.seed + index)
            label_path = self.label_paths[base_idx]
            if label_np is not None:
                pass
            elif label_path.suffix.lower() in {".h5", ".hdf", ".hdf5"}:
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
            if self.return_topology_stats:
                _, boundary_fraction, unique_nonzero, foreground_fraction = self._label_crop_stats(label_np)
                item.update(
                    {
                        "topology_foreground_fraction": torch.tensor(foreground_fraction, dtype=torch.float32),
                        "topology_boundary_fraction": torch.tensor(boundary_fraction, dtype=torch.float32),
                        "topology_unique_nonzero": torch.tensor(float(unique_nonzero), dtype=torch.float32),
                    }
                )
            unwidened_label = torch.from_numpy(label_np.astype(np.int64))
            if self.widen_border:
                label_np = widen_instance_boundaries_2d(label_np, radius=self.widen_border_radius)
            item["label"] = torch.from_numpy(label_np.astype(np.int64))
            if self.return_unwidened_label:
                item["unwidened_label"] = unwidened_label
        if self.augment:
            if "label" in item:
                augment_kwargs = {
                    "rotate_xy": self.augment_rotate_xy,
                    "gamma": self.augment_gamma,
                    "gamma_range": self.augment_gamma_range,
                    "gamma_probability": self.augment_gamma_probability,
                    "noise_std": self.augment_noise_std,
                    "intensity_probability": self.augment_intensity_probability,
                    "intensity_gain_range": self.augment_intensity_gain_range,
                    "intensity_bias_range": self.augment_intensity_bias_range,
                    "xy_blur_probability": self.augment_xy_blur_probability,
                    "xy_blur_kernel": self.augment_xy_blur_kernel,
                }
                if "unwidened_label" in item:
                    image_aug, label_aug, unwidened_aug = augment_image_and_aligned_labels(
                        item["image"],
                        item["label"],
                        item["unwidened_label"],
                        **augment_kwargs,
                    )
                    item["unwidened_label"] = unwidened_aug
                else:
                    image_aug, label_aug = augment_image_and_label(
                        item["image"],
                        item["label"],
                        **augment_kwargs,
                    )
                    if label_aug is None:
                        raise RuntimeError("label augmentation unexpectedly returned no label")
                item["image"] = image_aug
                item["label"] = label_aug
            else:
                item["image"] = augment_volume(
                    item["image"],
                    rotate_xy=self.augment_rotate_xy,
                    gamma=self.augment_gamma,
                    gamma_range=self.augment_gamma_range,
                    gamma_probability=self.augment_gamma_probability,
                    noise_std=self.augment_noise_std,
                    intensity_probability=self.augment_intensity_probability,
                    intensity_gain_range=self.augment_intensity_gain_range,
                    intensity_bias_range=self.augment_intensity_bias_range,
                    xy_blur_probability=self.augment_xy_blur_probability,
                    xy_blur_kernel=self.augment_xy_blur_kernel,
                )
        return item


DEFAULT_AFFINITY_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (-1, 0, 0),
    (0, -1, 0),
    (0, 0, -1),
)


def _normalize_affinity_offsets(
    offsets: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, int, int], ...]:
    if offsets is None:
        return DEFAULT_AFFINITY_OFFSETS
    normalized = []
    for offset in offsets:
        values = tuple(int(v) for v in offset)
        if len(values) != 3:
            raise ValueError(f"affinity offset must be z/y/x triplet, got {offset}")
        if values == (0, 0, 0):
            raise ValueError("affinity offset cannot be (0, 0, 0)")
        normalized.append(values)
    if not normalized:
        raise ValueError("at least one affinity offset is required")
    return tuple(normalized)


def labels_to_offset_affinities(
    labels: torch.Tensor,
    offsets: Sequence[Sequence[int]] | None = None,
    replicate_boundary: bool = False,
) -> torch.Tensor:
    """Convert instance labels [B,D,H,W] to affinities for arbitrary offsets.

    SuperHuman's ``seg_to_aff(..., pad="replicate")`` treats the first
    nearest-neighbor boundary plane as a valid foreground self-edge. For
    long-range offsets we conservatively leave invalid boundary voxels at zero.
    """
    if labels.ndim == 3:
        labels = labels.unsqueeze(0)
    labels = labels.long()
    bsz, depth, height, width = labels.shape
    norm_offsets = _normalize_affinity_offsets(offsets)
    aff = labels.new_zeros((bsz, len(norm_offsets), depth, height, width), dtype=torch.float32)
    for channel, (dz, dy, dx) in enumerate(norm_offsets):
        src_z0 = max(0, -dz)
        src_z1 = depth - max(0, dz)
        src_y0 = max(0, -dy)
        src_y1 = height - max(0, dy)
        src_x0 = max(0, -dx)
        src_x1 = width - max(0, dx)
        dst_z0 = src_z0 + dz
        dst_z1 = src_z1 + dz
        dst_y0 = src_y0 + dy
        dst_y1 = src_y1 + dy
        dst_x0 = src_x0 + dx
        dst_x1 = src_x1 + dx
        src = labels[:, src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
        dst = labels[:, dst_z0:dst_z1, dst_y0:dst_y1, dst_x0:dst_x1]
        valid = (src == dst) & (src > 0)
        aff[:, channel, src_z0:src_z1, src_y0:src_y1, src_x0:src_x1] = valid.float()
        if replicate_boundary and (dz, dy, dx) in DEFAULT_AFFINITY_OFFSETS:
            if (dz, dy, dx) == (-1, 0, 0):
                aff[:, channel, 0] = labels[:, 0] > 0
            elif (dz, dy, dx) == (0, -1, 0):
                aff[:, channel, :, 0] = labels[:, :, 0] > 0
            elif (dz, dy, dx) == (0, 0, -1):
                aff[:, channel, :, :, 0] = labels[:, :, :, 0] > 0
    return aff.float()


def labels_to_affinities(labels: torch.Tensor, replicate_boundary: bool = False) -> torch.Tensor:
    """Convert instance labels [B,D,H,W] to nearest-neighbor z/y/x affinities."""

    return labels_to_offset_affinities(
        labels,
        DEFAULT_AFFINITY_OFFSETS,
        replicate_boundary=replicate_boundary,
    )


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
