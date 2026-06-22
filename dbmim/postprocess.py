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


def _grid_seeds(boundary: np.ndarray, next_id: int, seed_distance: int) -> tuple[np.ndarray, int]:
    height, width = boundary.shape
    y_positions = np.arange(0, height, max(1, int(seed_distance)), dtype=np.int32)
    x_positions = np.arange(0, width, max(1, int(seed_distance)), dtype=np.int32)
    seeds = np.zeros_like(boundary, dtype=np.int32)
    num = int(len(y_positions) * len(x_positions))
    if num:
        seeds[np.ix_(y_positions, x_positions)] = np.arange(next_id, next_id + num, dtype=np.int32).reshape(
            len(y_positions), len(x_positions)
        )
    return seeds, num


def _mahotas_seeds(
    boundary: np.ndarray,
    method: str,
    next_id: int,
    seed_distance: int,
    threshold: float,
) -> tuple[np.ndarray, int]:
    import mahotas  # type: ignore

    if method == "grid":
        return _grid_seeds(boundary, next_id, seed_distance)
    if method == "minima":
        minima = mahotas.regmin(boundary)
        seeds, num = mahotas.label(minima)
    elif method == "maxima_distance":
        distance = mahotas.distance(boundary < threshold)
        maxima = mahotas.regmax(distance)
        seeds, num = mahotas.label(maxima)
    else:
        raise ValueError(f"unknown seed method: {method}")
    seeds = seeds.astype(np.int32, copy=False)
    seeds += int(next_id)
    seeds[seeds == next_id] = 0
    if int(num) <= 0:
        return _grid_seeds(boundary, next_id, seed_distance)
    return seeds, int(num)


def _scipy_seeds(
    boundary: np.ndarray,
    method: str,
    next_id: int,
    seed_distance: int,
    threshold: float,
) -> tuple[np.ndarray, int]:
    if method == "grid":
        return _grid_seeds(boundary, next_id, seed_distance)
    from scipy import ndimage  # type: ignore

    mask = boundary < threshold
    distance = ndimage.distance_transform_edt(mask)
    maxima = (distance == ndimage.maximum_filter(distance, size=5)) & (distance > 0)
    seeds, num = ndimage.label(maxima)
    seeds = seeds.astype(np.int32, copy=False)
    seeds[seeds > 0] += int(next_id) - 1
    if int(num) <= 0:
        return _grid_seeds(boundary, next_id, seed_distance)
    return seeds, int(num)


def watershed_fragments(
    affinities: np.ndarray,
    backend: str = "mahotas",
    seed_method: str = "maxima_distance",
    seed_distance: int = 12,
    boundary_threshold: float = 0.5,
) -> np.ndarray:
    """Build 2D slice-wise watershed fragments from predicted xy affinities."""

    affinities = np.asarray(affinities, dtype=np.float32)
    if affinities.ndim != 4 or affinities.shape[0] != 3:
        raise ValueError(f"expected affinities with shape [3,D,H,W], got {affinities.shape}")
    boundary = 1.0 - 0.5 * (affinities[1] + affinities[2])
    depth, height, width = boundary.shape
    fragments = np.zeros((depth, height, width), dtype=np.uint64)
    next_id = 1
    if backend == "mahotas":
        import mahotas  # type: ignore

        for z in range(depth):
            seeds, num = _mahotas_seeds(boundary[z], seed_method, next_id, seed_distance, boundary_threshold)
            fragments[z] = mahotas.cwatershed(boundary[z], seeds).astype(np.uint64, copy=False)
            next_id = int(fragments[z].max()) + 1
            if num <= 0:
                next_id += 1
    elif backend == "scipy":
        from scipy import ndimage  # type: ignore

        for z in range(depth):
            seeds, num = _scipy_seeds(boundary[z], seed_method, next_id, seed_distance, boundary_threshold)
            fragments[z] = ndimage.watershed_ift(
                np.clip(boundary[z] * 255.0, 0, 255).astype(np.uint8),
                seeds.astype(np.int32, copy=False),
            ).astype(np.uint64, copy=False)
            next_id = int(fragments[z].max()) + 1
            if num <= 0:
                next_id += 1
    else:
        raise ValueError(f"unknown watershed backend: {backend}")
    return relabel_sequential(fragments)


def _affinity_boundary_pairs(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    weights: np.ndarray,
    base: int,
) -> tuple[np.ndarray, np.ndarray]:
    left = labels_a.reshape(-1).astype(np.int64, copy=False)
    right = labels_b.reshape(-1).astype(np.int64, copy=False)
    weight = weights.reshape(-1).astype(np.float32, copy=False)
    keep = (left > 0) & (right > 0) & (left != right)
    if not np.any(keep):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    left = left[keep]
    right = right[keep]
    lo = np.minimum(left, right)
    hi = np.maximum(left, right)
    return lo * int(base) + hi, weight[keep]


def _reduce_pair_scores(
    packed: np.ndarray,
    weights: np.ndarray,
    score_mode: str = "mean",
    quantile: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if packed.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )
    order = np.argsort(packed)
    packed = packed[order]
    weights = weights[order].astype(np.float32, copy=False)
    unique, starts, counts = np.unique(packed, return_index=True, return_counts=True)
    mode = score_mode.lower()
    if mode == "mean":
        scores = np.add.reduceat(weights, starts) / np.maximum(counts, 1)
    elif mode == "max":
        scores = np.maximum.reduceat(weights, starts)
    elif mode == "min":
        scores = np.minimum.reduceat(weights, starts)
    elif mode in {"median", "q50"}:
        scores = np.array(
            [np.quantile(weights[start : start + count], 0.50) for start, count in zip(starts, counts)],
            dtype=np.float32,
        )
    elif mode in {"q10", "quantile10"}:
        scores = np.array(
            [np.quantile(weights[start : start + count], 0.10) for start, count in zip(starts, counts)],
            dtype=np.float32,
        )
    elif mode in {"q25", "quantile25"}:
        q = 0.25
        scores = np.array(
            [np.quantile(weights[start : start + count], q) for start, count in zip(starts, counts)],
            dtype=np.float32,
        )
    elif mode == "quantile":
        q = float(quantile)
        scores = np.array(
            [np.quantile(weights[start : start + count], q) for start, count in zip(starts, counts)],
            dtype=np.float32,
        )
    elif mode in {"q75", "quantile75"}:
        scores = np.array(
            [np.quantile(weights[start : start + count], 0.75) for start, count in zip(starts, counts)],
            dtype=np.float32,
        )
    else:
        raise ValueError(f"unknown score_mode: {score_mode}")
    return unique, scores.astype(np.float32, copy=False), counts.astype(np.int64, copy=False)


def rag_boundary_features(
    affinities: np.ndarray,
    fragments: np.ndarray,
    *,
    min_boundary: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fragment-pair ids, boundary features, and boundary counts.

    The feature order is intentionally simple and stable:

    - z/y/x boundary mean affinity
    - z/y/x boundary min affinity
    - z/y/x boundary max affinity
    - z/y/x boundary voxel count normalized by total pair count
    - total boundary voxel count, log1p scaled
    """

    affinities = np.asarray(affinities, dtype=np.float32)
    fragments = np.asarray(fragments, dtype=np.uint64)
    if affinities.ndim != 4 or affinities.shape[0] != 3:
        raise ValueError(f"expected affinities with shape [3,D,H,W], got {affinities.shape}")
    if tuple(affinities.shape[1:]) != tuple(fragments.shape):
        raise ValueError(f"affinity/fragment shape mismatch: {affinities.shape[1:]} vs {fragments.shape}")
    max_id = int(fragments.max(initial=0))
    if max_id <= 1:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 13), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    base = max_id + 1
    packed_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    axis_parts: list[np.ndarray] = []
    axes = [
        (0, fragments[1:], fragments[:-1], affinities[0, 1:]),
        (1, fragments[:, 1:], fragments[:, :-1], affinities[1, :, 1:]),
        (2, fragments[:, :, 1:], fragments[:, :, :-1], affinities[2, :, :, 1:]),
    ]
    for axis, left_labels, right_labels, values in axes:
        packed, weights = _affinity_boundary_pairs(left_labels, right_labels, values, base)
        if packed.size == 0:
            continue
        packed_parts.append(packed)
        value_parts.append(weights)
        axis_parts.append(np.full(packed.shape, axis, dtype=np.int8))
    if not packed_parts:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0, 13), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    packed_all = np.concatenate(packed_parts).astype(np.int64, copy=False)
    values_all = np.concatenate(value_parts).astype(np.float32, copy=False)
    axes_all = np.concatenate(axis_parts).astype(np.int8, copy=False)
    unique, inverse = np.unique(packed_all, return_inverse=True)
    counts = np.bincount(inverse, minlength=unique.size).astype(np.int64, copy=False)
    features = np.zeros((unique.size, 13), dtype=np.float32)
    safe_counts = np.maximum(counts.astype(np.float32), 1.0)
    for axis in range(3):
        axis_mask = axes_all == axis
        if not np.any(axis_mask):
            continue
        axis_idx = inverse[axis_mask]
        axis_values = values_all[axis_mask].astype(np.float32, copy=False)
        axis_counts = np.bincount(axis_idx, minlength=unique.size).astype(np.float32, copy=False)
        axis_sum = np.bincount(axis_idx, weights=axis_values, minlength=unique.size).astype(np.float32, copy=False)
        valid = axis_counts > 0
        features[valid, axis] = axis_sum[valid] / axis_counts[valid]
        mins = np.full(unique.size, np.inf, dtype=np.float32)
        maxs = np.full(unique.size, -np.inf, dtype=np.float32)
        np.minimum.at(mins, axis_idx, axis_values)
        np.maximum.at(maxs, axis_idx, axis_values)
        features[valid, 3 + axis] = mins[valid]
        features[valid, 6 + axis] = maxs[valid]
        features[:, 9 + axis] = axis_counts / safe_counts
    features[:, 12] = np.log1p(safe_counts)
    keep = counts >= int(min_boundary)
    return unique[keep], features[keep], counts[keep]


def agglomerate_fragments_by_scores(
    fragments: np.ndarray,
    packed_pairs: np.ndarray,
    merge_scores: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Merge fragment pairs with learned scores above threshold."""

    fragments = np.asarray(fragments, dtype=np.uint64)
    packed_pairs = np.asarray(packed_pairs, dtype=np.int64)
    merge_scores = np.asarray(merge_scores, dtype=np.float32)
    max_id = int(fragments.max(initial=0))
    if max_id <= 1 or packed_pairs.size == 0:
        return relabel_sequential(fragments)
    base = max_id + 1
    uf = UnionFind(max_id + 1)
    selected = packed_pairs[merge_scores >= float(threshold)]
    for packed in selected:
        left = int(packed // base)
        right = int(packed % base)
        if left > 0 and right > 0:
            uf.union(left, right)
    roots = uf.compress()
    root_labels = roots[fragments.astype(np.int64, copy=False)]
    return relabel_sequential(root_labels)


def rag_pair_labels_from_ground_truth(
    packed_pairs: np.ndarray,
    fragments: np.ndarray,
    labels: np.ndarray,
    *,
    ignore_label: int | None = 0,
    positive_fraction: float = 0.5,
) -> np.ndarray:
    """Label fragment pairs by majority ground-truth object consistency."""

    packed_pairs = np.asarray(packed_pairs, dtype=np.int64)
    fragments = np.asarray(fragments, dtype=np.uint64)
    labels = np.asarray(labels)
    max_id = int(fragments.max(initial=0))
    base = max_id + 1
    flat_frag = fragments.reshape(-1).astype(np.int64, copy=False)
    flat_label = labels.reshape(-1).astype(np.int64, copy=False)
    keep = (flat_frag > 0) & (flat_frag <= max_id) & (flat_label > 0)
    if ignore_label is not None:
        keep &= flat_label != int(ignore_label)
    major = np.zeros(max_id + 1, dtype=np.int64)
    confidence = np.zeros(max_id + 1, dtype=np.float32)
    if np.any(keep):
        kept_frag = flat_frag[keep]
        kept_label = flat_label[keep]
        label_base = int(kept_label.max(initial=0)) + 1
        packed = kept_frag * label_base + kept_label
        unique, counts = np.unique(packed, return_counts=True)
        frag_ids = unique // label_base
        label_ids = unique % label_base
        totals = np.bincount(kept_frag, minlength=max_id + 1).astype(np.float32, copy=False)
        order = np.lexsort((-counts, frag_ids))
        sorted_frag = frag_ids[order]
        first = np.r_[True, sorted_frag[1:] != sorted_frag[:-1]]
        best = order[first]
        best_frag = frag_ids[best].astype(np.int64, copy=False)
        best_label = label_ids[best].astype(np.int64, copy=False)
        major[best_frag] = best_label
        confidence[best_frag] = counts[best].astype(np.float32, copy=False) / np.maximum(totals[best_frag], 1.0)
    out = np.zeros(packed_pairs.shape, dtype=np.float32)
    for idx, packed in enumerate(packed_pairs):
        left = int(packed // base)
        right = int(packed % base)
        if left <= 0 or right <= 0 or left > max_id or right > max_id:
            continue
        if major[left] > 0 and major[left] == major[right]:
            out[idx] = 1.0 if min(confidence[left], confidence[right]) >= float(positive_fraction) else 0.0
    return out


def _select_pairs_by_score(
    packed: np.ndarray,
    weights: np.ndarray,
    threshold: float,
    min_boundary: int,
    score_mode: str,
    quantile: float,
) -> np.ndarray:
    unique, scores, counts = _reduce_pair_scores(
        packed,
        weights,
        score_mode=score_mode,
        quantile=quantile,
    )
    if unique.size == 0:
        return unique
    return unique[(scores >= float(threshold)) & (counts >= int(min_boundary))]


def agglomerate_fragments_by_affinity(
    affinities: np.ndarray,
    fragments: np.ndarray,
    threshold: float = 0.5,
    z_threshold: float | None = None,
    xy_threshold: float | None = None,
    min_boundary: int = 1,
    use_channels: Sequence[int] = (0, 1, 2),
    score_mode: str = "mean",
    quantile: float = 0.25,
) -> np.ndarray:
    """Merge watershed fragments whose shared boundary affinity is high."""

    affinities = np.asarray(affinities, dtype=np.float32)
    fragments = np.asarray(fragments, dtype=np.uint64)
    if affinities.ndim != 4 or affinities.shape[0] != 3:
        raise ValueError(f"expected affinities with shape [3,D,H,W], got {affinities.shape}")
    if tuple(affinities.shape[1:]) != tuple(fragments.shape):
        raise ValueError(f"affinity/fragment shape mismatch: {affinities.shape[1:]} vs {fragments.shape}")
    max_id = int(fragments.max(initial=0))
    if max_id <= 1:
        return fragments.copy()
    base = max_id + 1
    selected_parts: list[np.ndarray] = []
    z_thr = float(threshold if z_threshold is None else z_threshold)
    xy_thr = float(threshold if xy_threshold is None else xy_threshold)
    if 0 in use_channels and fragments.shape[0] > 1:
        packed, weight = _affinity_boundary_pairs(fragments[1:], fragments[:-1], affinities[0, 1:], base)
        selected_parts.append(
            _select_pairs_by_score(
                packed,
                weight,
                threshold=z_thr,
                min_boundary=min_boundary,
                score_mode=score_mode,
                quantile=quantile,
            )
        )
    if 1 in use_channels and fragments.shape[1] > 1:
        packed, weight = _affinity_boundary_pairs(fragments[:, 1:], fragments[:, :-1], affinities[1, :, 1:], base)
        selected_parts.append(
            _select_pairs_by_score(
                packed,
                weight,
                threshold=xy_thr,
                min_boundary=min_boundary,
                score_mode=score_mode,
                quantile=quantile,
            )
        )
    if 2 in use_channels and fragments.shape[2] > 1:
        packed, weight = _affinity_boundary_pairs(fragments[:, :, 1:], fragments[:, :, :-1], affinities[2, :, :, 1:], base)
        selected_parts.append(
            _select_pairs_by_score(
                packed,
                weight,
                threshold=xy_thr,
                min_boundary=min_boundary,
                score_mode=score_mode,
                quantile=quantile,
            )
        )
    selected_parts = [part for part in selected_parts if part.size]
    if not selected_parts:
        return relabel_sequential(fragments)
    selected = np.unique(np.concatenate(selected_parts))

    uf = UnionFind(max_id + 1)
    for packed in selected:
        left = int(packed // base)
        right = int(packed % base)
        uf.union(left, right)
    roots = uf.compress()
    root_labels = roots[fragments.astype(np.int64, copy=False)]
    return relabel_sequential(root_labels)


def watershed_agglomeration(
    affinities: np.ndarray,
    threshold: float = 0.5,
    backend: str = "mahotas",
    seed_method: str = "maxima_distance",
    seed_distance: int = 12,
    boundary_threshold: float = 0.5,
    min_boundary: int = 1,
    score_mode: str = "mean",
    quantile: float = 0.25,
    z_threshold: float | None = None,
    xy_threshold: float | None = None,
) -> np.ndarray:
    fragments = watershed_fragments(
        affinities,
        backend=backend,
        seed_method=seed_method,
        seed_distance=seed_distance,
        boundary_threshold=boundary_threshold,
    )
    return agglomerate_fragments_by_affinity(
        affinities,
        fragments,
        threshold=threshold,
        z_threshold=z_threshold,
        xy_threshold=xy_threshold,
        min_boundary=min_boundary,
        score_mode=score_mode,
        quantile=quantile,
    )


def cc3d_mean_affinity_components(
    affinities: np.ndarray,
    threshold: float = 0.5,
    connectivity: int = 6,
    min_size: int = 0,
) -> np.ndarray:
    """Fast approximate instance baseline using cc3d on voxel confidence.

    This is not equivalent to affinity-graph connected components. It is kept
    as a speed reference for CPU-only post-processing backends.
    """

    import cc3d  # type: ignore

    affinities = np.asarray(affinities, dtype=np.float32)
    if affinities.ndim != 4 or affinities.shape[0] != 3:
        raise ValueError(f"expected affinities with shape [3,D,H,W], got {affinities.shape}")
    score = np.zeros(affinities.shape[1:], dtype=np.float32)
    count = np.zeros(affinities.shape[1:], dtype=np.float32)
    for channel in range(3):
        score += affinities[channel]
        count += 1.0
    mask = (score / np.maximum(count, 1.0)) >= float(threshold)
    labels = cc3d.connected_components(mask.astype(np.uint8), connectivity=int(connectivity)).astype(np.uint64)
    if min_size > 1:
        labels = cc3d.dust(labels, threshold=int(min_size), in_place=False).astype(np.uint64)
    return relabel_sequential(labels)


def cupy_mean_affinity_components(
    affinities: np.ndarray,
    threshold: float = 0.5,
    connectivity: int = 1,
    min_size: int = 0,
) -> np.ndarray:
    """GPU approximate connected components on mean voxel affinity."""

    import cupy as cp  # type: ignore
    from cupyx.scipy import ndimage as cndimage  # type: ignore

    affinities = np.asarray(affinities, dtype=np.float32)
    if affinities.ndim != 4 or affinities.shape[0] != 3:
        raise ValueError(f"expected affinities with shape [3,D,H,W], got {affinities.shape}")
    aff_gpu = cp.asarray(affinities)
    score = cp.mean(aff_gpu, axis=0)
    structure = None
    if int(connectivity) >= 2:
        structure = cp.ones((3, 3, 3), dtype=bool)
    mask = score >= float(threshold)
    labels_gpu, _ = cndimage.label(mask, structure=structure)
    labels = cp.asnumpy(labels_gpu).astype(np.uint64, copy=False)
    if min_size > 1:
        ids, counts = np.unique(labels, return_counts=True)
        small_ids = ids[(ids != 0) & (counts < int(min_size))]
        if small_ids.size:
            labels[np.isin(labels, small_ids)] = 0
    return relabel_sequential(labels)


def cupy_affinity_graph_connected_components(
    affinities: np.ndarray,
    threshold: float | Sequence[float] = 0.5,
    min_size: int = 0,
) -> np.ndarray:
    """GPU affinity graph CC if pylibcugraph is available in the environment."""

    import cupy as cp  # type: ignore
    from cupyx.scipy.sparse import coo_matrix  # type: ignore
    from cupyx.scipy.sparse.csgraph import connected_components  # type: ignore

    affinities = np.asarray(affinities, dtype=np.float32)
    if affinities.ndim != 4 or affinities.shape[0] != 3:
        raise ValueError(f"expected affinities with shape [3,D,H,W], got {affinities.shape}")
    thr_z, thr_y, thr_x = _threshold_tuple(threshold)
    depth, height, width = affinities.shape[1:]
    grid = np.arange(depth * height * width, dtype=np.int32).reshape(depth, height, width)
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
    if not edge_src:
        return grid.astype(np.uint64) + 1
    rows = cp.asarray(np.concatenate(edge_src).astype(np.int32, copy=False))
    cols = cp.asarray(np.concatenate(edge_dst).astype(np.int32, copy=False))
    data = cp.ones(rows.shape[0], dtype=cp.float32)
    graph = coo_matrix((data, (rows, cols)), shape=(grid.size, grid.size)).tocsr()
    _, labels_gpu = connected_components(graph, directed=False, return_labels=True)
    labels = cp.asnumpy(labels_gpu).reshape(depth, height, width).astype(np.uint64) + 1
    if min_size > 1:
        ids, counts = np.unique(labels, return_counts=True)
        small_ids = ids[counts < int(min_size)]
        if small_ids.size:
            labels[np.isin(labels, small_ids)] = 0
            labels = relabel_sequential(labels)
    return labels


def waterz_agglomeration(
    affinities: np.ndarray,
    threshold: float = 0.5,
    fragments: np.ndarray | None = None,
    scoring_function: str = "hist_quantile",
) -> np.ndarray:
    import waterz  # type: ignore

    if fragments is None:
        fragments = watershed_fragments(affinities, backend="mahotas", seed_method="maxima_distance")
    waterz_fragments = np.asarray(fragments, dtype=np.uint64).copy(order="C")
    scoring_aliases = {
        "hist_quantile": "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256>>",
        "hist_quantile_false": (
            "OneMinus<HistogramQuantileAffinity<RegionGraphType, 50, ScoreValue, 256, false>>"
        ),
        "min": "OneMinus<MinAffinity<RegionGraphType, ScoreValue>>",
        "max": "OneMinus<MaxAffinity<RegionGraphType, ScoreValue>>",
    }
    if scoring_function == "mean":
        raise ValueError("waterz v0.8 does not provide MeanAffinityProvider; use hist_quantile, min, or max")
    scoring = scoring_aliases.get(scoring_function, scoring_function)
    return np.asarray(
        list(
            waterz.agglomerate(
                affinities,
                [float(threshold)],
                fragments=waterz_fragments,
                scoring_function=scoring,
                discretize_queue=256,
            )
        )[0]
    )


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
    backend: str = "internal",
) -> SegmentationMetrics:
    if backend == "skimage":
        from skimage.metrics import adapted_rand_error, variation_of_information  # type: ignore

        ignore_labels = () if ignore_label is None else (ignore_label,)
        arand, precision, recall = adapted_rand_error(target, pred, ignore_labels=ignore_labels)
        voi_split, voi_merge = variation_of_information(target, pred, ignore_labels=ignore_labels)
        if ignore_label is None:
            pred_eval = np.asarray(pred).reshape(-1)
            target_eval = np.asarray(target).reshape(-1)
        else:
            target_flat = np.asarray(target).reshape(-1)
            keep = target_flat != ignore_label
            pred_eval = np.asarray(pred).reshape(-1)[keep]
            target_eval = target_flat[keep]
        return SegmentationMetrics(
            adapted_rand_error=float(arand),
            rand_fscore=1.0 - float(arand),
            rand_precision=float(precision),
            rand_recall=float(recall),
            voi_split=float(voi_split),
            voi_merge=float(voi_merge),
            n_pred=int(np.unique(pred_eval).size),
            n_gt=int(np.unique(target_eval).size),
            n_voxels=int(pred_eval.size),
        )
    if backend != "internal":
        raise ValueError(f"unknown metric backend: {backend}")

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
        voi_split=max(0.0, h_joint - h_gt),
        voi_merge=max(0.0, h_joint - h_pred),
        n_pred=n_pred,
        n_gt=n_gt,
        n_voxels=int(pred.size),
    )
