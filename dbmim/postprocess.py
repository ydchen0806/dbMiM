from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def relabel_sequential(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    unique, inverse = np.unique(labels, return_inverse=True)
    out = inverse.reshape(labels.shape).astype(np.uint64)
    if unique.size and unique[0] == 0:
        return out
    return out + 1


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        rank = self.rank
        parent = self.parent
        if rank[root_left] < rank[root_right]:
            parent[root_left] = root_right
        elif rank[root_left] > rank[root_right]:
            parent[root_right] = root_left
        else:
            parent[root_right] = root_left
            rank[root_left] += 1

    def compress(self) -> np.ndarray:
        parent = self.parent
        while True:
            updated = parent[parent]
            if np.array_equal(updated, parent):
                break
            parent = updated
        self.parent = parent
        return parent


def _threshold_tuple(threshold: float | Sequence[float]) -> tuple[float, float, float]:
    if isinstance(threshold, Sequence) and not isinstance(threshold, (str, bytes)):
        values = tuple(float(v) for v in threshold)
        if len(values) != 3:
            raise ValueError(f"expected 3 thresholds for z/y/x affinities, got {values}")
        return values
    value = float(threshold)
    return value, value, value


def affinities_to_connected_components(
    affinities: np.ndarray,
    threshold: float | Sequence[float] = 0.5,
    min_size: int = 0,
) -> np.ndarray:
    """Convert z/y/x nearest-neighbor affinities to instance labels.

    The affinity convention follows ``labels_to_affinities``:
    channel 0 connects ``(z, y, x)`` to ``(z - 1, y, x)``, channel 1 connects
    to ``(z, y - 1, x)``, and channel 2 connects to ``(z, y, x - 1)``.
    This baseline is intentionally dependency-light for offline SiFlow pods.
    """

    affinities = np.asarray(affinities, dtype=np.float32)
    if affinities.ndim != 4 or affinities.shape[0] != 3:
        raise ValueError(f"expected affinities with shape [3,D,H,W], got {affinities.shape}")
    thr_z, thr_y, thr_x = _threshold_tuple(threshold)
    depth, height, width = affinities.shape[1:]
    grid = np.arange(depth * height * width, dtype=np.int64).reshape(depth, height, width)
    try:
        from scipy.sparse import coo_matrix  # type: ignore
        from scipy.sparse.csgraph import connected_components  # type: ignore

        edge_src = []
        edge_dst = []
        z_mask = affinities[0, 1:] >= thr_z
        if z_mask.any():
            edge_src.append(grid[1:][z_mask])
            edge_dst.append(grid[:-1][z_mask])
        y_mask = affinities[1, :, 1:] >= thr_y
        if y_mask.any():
            edge_src.append(grid[:, 1:][y_mask])
            edge_dst.append(grid[:, :-1][y_mask])
        x_mask = affinities[2, :, :, 1:] >= thr_x
        if x_mask.any():
            edge_src.append(grid[:, :, 1:][x_mask])
            edge_dst.append(grid[:, :, :-1][x_mask])
        if edge_src:
            rows = np.concatenate(edge_src).astype(np.int64, copy=False)
            cols = np.concatenate(edge_dst).astype(np.int64, copy=False)
            data = np.ones(rows.shape[0], dtype=np.uint8)
            graph = coo_matrix((data, (rows, cols)), shape=(grid.size, grid.size)).tocsr()
            _, labels_1d = connected_components(graph, directed=False, return_labels=True)
            labels = labels_1d.reshape(depth, height, width).astype(np.uint64) + 1
        else:
            labels = grid.astype(np.uint64) + 1
        if min_size > 1:
            ids, counts = np.unique(labels, return_counts=True)
            small_ids = ids[counts < int(min_size)]
            if small_ids.size:
                labels[np.isin(labels, small_ids)] = 0
                labels = relabel_sequential(labels)
        return labels
    except Exception:
        pass

    uf = UnionFind(int(grid.size))

    z_edges = np.argwhere(affinities[0, 1:] >= thr_z)
    for z, y, x in z_edges:
        zz = int(z) + 1
        uf.union(int(grid[zz, y, x]), int(grid[zz - 1, y, x]))

    y_edges = np.argwhere(affinities[1, :, 1:] >= thr_y)
    for z, y, x in y_edges:
        yy = int(y) + 1
        uf.union(int(grid[z, yy, x]), int(grid[z, yy - 1, x]))

    x_edges = np.argwhere(affinities[2, :, :, 1:] >= thr_x)
    for z, y, x in x_edges:
        xx = int(x) + 1
        uf.union(int(grid[z, y, xx]), int(grid[z, y, xx - 1]))

    roots = uf.compress()
    _, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
    labels = inverse.reshape(depth, height, width).astype(np.uint64) + 1
    if min_size > 1:
        small = np.flatnonzero(counts < int(min_size))
        if small.size:
            remove = np.isin(inverse, small).reshape(depth, height, width)
            labels[remove] = 0
            labels = relabel_sequential(labels)
    return labels


@dataclass(frozen=True)
class SegmentationMetrics:
    adapted_rand_error: float
    rand_fscore: float
    rand_precision: float
    rand_recall: float
    voi_split: float
    voi_merge: float
    n_pred: int
    n_gt: int
    n_voxels: int


def segmentation_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    ignore_label: int | None = 0,
) -> SegmentationMetrics:
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    if pred.shape != target.shape:
        raise ValueError(f"prediction/target size mismatch: {pred.shape} vs {target.shape}")
    if ignore_label is not None:
        keep = target != ignore_label
        pred = pred[keep]
        target = target[keep]
    if pred.size == 0:
        return SegmentationMetrics(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)

    pred_ids, pred_inv = np.unique(pred, return_inverse=True)
    target_ids, target_inv = np.unique(target, return_inverse=True)
    n_pred = int(len(pred_ids))
    n_gt = int(len(target_ids))
    pair_ids = pred_inv.astype(np.int64) * max(1, n_gt) + target_inv.astype(np.int64)
    _, pair_counts = np.unique(pair_ids, return_counts=True)
    pred_counts = np.bincount(pred_inv, minlength=n_pred).astype(np.float64)
    target_counts = np.bincount(target_inv, minlength=n_gt).astype(np.float64)
    pair_counts = pair_counts.astype(np.float64)

    sum_pair = float(np.sum(pair_counts * (pair_counts - 1.0) / 2.0))
    sum_pred = float(np.sum(pred_counts * (pred_counts - 1.0) / 2.0))
    sum_gt = float(np.sum(target_counts * (target_counts - 1.0) / 2.0))
    precision = sum_pair / sum_pred if sum_pred > 0.0 else 0.0
    recall = sum_pair / sum_gt if sum_gt > 0.0 else 0.0
    fscore = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0

    total = float(pred.size)
    p_pred = pred_counts[pred_counts > 0] / total
    p_gt = target_counts[target_counts > 0] / total
    p_pair = pair_counts[pair_counts > 0] / total
    h_pred = -float(np.sum(p_pred * np.log2(p_pred)))
    h_gt = -float(np.sum(p_gt * np.log2(p_gt)))
    h_joint = -float(np.sum(p_pair * np.log2(p_pair)))

    return SegmentationMetrics(
        adapted_rand_error=1.0 - fscore,
        rand_fscore=fscore,
        rand_precision=precision,
        rand_recall=recall,
        voi_split=max(0.0, h_joint - h_pred),
        voi_merge=max(0.0, h_joint - h_gt),
        n_pred=n_pred,
        n_gt=n_gt,
        n_voxels=int(pred.size),
    )
