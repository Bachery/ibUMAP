from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from ._device import resolve_device
from ._dtype import resolve_evaluation_dtype


@dataclass(frozen=True)
class NeighborhoodPreservationResult:
    score: float
    local_scores: np.ndarray
    n_neighbors: int


@dataclass(frozen=True)
class NeighborSearchResult:
    indices: np.ndarray
    distances: Optional[np.ndarray]
    n_neighbors: int
    metric: Any
    metric_params: Optional[dict[str, Any]]


def _validate_points(points: Any, name: str, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(points, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_pair(
    high_dimensional_points: Any,
    embedding: Any,
    *,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    source = _validate_points(high_dimensional_points, "high_dimensional_points", dtype)
    target = _validate_points(embedding, "embedding", dtype)
    if source.shape[0] != target.shape[0]:
        raise ValueError("high_dimensional_points and embedding must contain the same number of samples")
    return source, target


def _validate_rank_metric_neighbors(n_neighbors: int, n_samples: int) -> int:
    k = int(n_neighbors)
    if k < 1:
        raise ValueError("n_neighbors must be at least 1")
    if k >= n_samples / 2:
        raise ValueError(
            f"n_neighbors ({k}) should be less than n_samples / 2 ({n_samples / 2})"
        )
    return k


def _validate_overlap_neighbors(n_neighbors: int, n_samples: int) -> int:
    k = int(n_neighbors)
    if k < 1:
        raise ValueError("n_neighbors must be at least 1")
    if k >= n_samples:
        raise ValueError("n_neighbors must be smaller than the number of samples")
    return k


def _validate_low_memory_batch_size(batch_size: int) -> int:
    resolved = int(batch_size)
    if resolved < 1:
        raise ValueError("low_memory_batch_size must be at least 1")
    return resolved


def _ensure_gpu_euclidean_metric(
    metric: str | Any,
    metric_params: Optional[dict[str, Any]],
    *,
    metric_name: str,
) -> None:
    if metric_params:
        raise ValueError(f"device='gpu' for {metric_name} does not support metric_params")
    if metric not in {"euclidean", "l2"}:
        raise ValueError(f"device='gpu' for {metric_name} currently supports only euclidean metric")


def _pairwise_neighbor_order_and_ranks(
    points: np.ndarray,
    *,
    metric: str | Any,
    metric_params: Optional[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    distances = pairwise_distances(points, metric=metric, **(metric_params or {}))
    np.fill_diagonal(distances, np.inf)
    neighbor_order = np.argsort(distances, axis=1)

    n_samples = points.shape[0]
    ranks = np.empty((n_samples, n_samples), dtype=np.int32)
    rows = np.arange(n_samples)[:, np.newaxis]
    ordered_ranks = np.arange(1, n_samples + 1, dtype=np.int32)
    ranks[rows, neighbor_order] = ordered_ranks
    return neighbor_order, ranks


def _exact_neighbor_indices(
    points: np.ndarray,
    *,
    n_neighbors: int,
    metric: str | Any,
    metric_params: Optional[dict[str, Any]],
) -> np.ndarray:
    return compute_neighbors(
        points,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_params=metric_params,
        return_distances=False,
        dtype=points.dtype,
    ).indices


def compute_neighbors(
    points: Any,
    *,
    n_neighbors: int,
    metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    return_distances: bool = False,
    dtype: Any = np.float32,
    device: str = "cpu",
    gpu_batch_size: Optional[int] = None,
) -> NeighborSearchResult:
    """Compute exact nearest-neighbor indices for a point cloud.

    This is the public reusable neighbor primitive used by multiple evaluation
    metrics. When ``return_distances`` is true, the returned distances align
    with ``indices`` row-wise.
    """
    resolved_dtype = resolve_evaluation_dtype(dtype)
    point_cloud = _validate_points(points, "points", resolved_dtype)
    k = _validate_overlap_neighbors(n_neighbors, point_cloud.shape[0])
    resolved_device = resolve_device(device)

    if resolved_device == "gpu":
        from ._gpu import _exact_neighbors_gpu, _validate_gpu_batch_size

        _ensure_gpu_euclidean_metric(metric, metric_params, metric_name="compute_neighbors")
        batch_size = _validate_gpu_batch_size(gpu_batch_size, 1024)
        distances_gpu, indices_gpu = _exact_neighbors_gpu(
            point_cloud,
            n_neighbors=k,
            batch_size=batch_size,
        )
        import cupy as cp

        indices = cp.asnumpy(indices_gpu).astype(np.int64, copy=False)
        distances = (
            cp.asnumpy(distances_gpu).astype(point_cloud.dtype, copy=False)
            if return_distances
            else None
        )
        return NeighborSearchResult(
            indices=indices,
            distances=distances,
            n_neighbors=k,
            metric=metric,
            metric_params=metric_params,
        )

    nearest_neighbors = NearestNeighbors(
        n_neighbors=k,
        metric=metric,
        metric_params=metric_params,
    )
    nearest_neighbors.fit(point_cloud)
    if return_distances:
        distances, indices = nearest_neighbors.kneighbors(return_distance=True)
        return NeighborSearchResult(
            indices=indices.astype(np.int64, copy=False),
            distances=distances.astype(point_cloud.dtype, copy=False),
            n_neighbors=k,
            metric=metric,
            metric_params=metric_params,
        )
    indices = nearest_neighbors.kneighbors(return_distance=False)
    return NeighborSearchResult(
        indices=indices.astype(np.int64, copy=False),
        distances=None,
        n_neighbors=k,
        metric=metric,
        metric_params=metric_params,
    )


def neighborhood_preservation_from_neighbors(
    source_neighbors: Any,
    embedding_neighbors: Any,
    *,
    block_size: int = 65_536,
) -> NeighborhoodPreservationResult:
    """Compute neighborhood preservation from precomputed neighbor indices."""
    source = np.asarray(source_neighbors, dtype=np.int64)
    target = np.asarray(embedding_neighbors, dtype=np.int64)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source_neighbors and embedding_neighbors must be 2D arrays")
    local_scores = _rowwise_overlap_scores(source, target, block_size=block_size)
    return NeighborhoodPreservationResult(
        score=float(np.mean(local_scores)),
        local_scores=local_scores,
        n_neighbors=int(source.shape[1]),
    )


def _rowwise_overlap_scores(
    source_neighbors: np.ndarray,
    target_neighbors: np.ndarray,
    *,
    block_size: int = 65_536,
) -> np.ndarray:
    if source_neighbors.shape != target_neighbors.shape:
        raise ValueError("source_neighbors and target_neighbors must have the same shape")

    n_samples, n_neighbors = source_neighbors.shape
    local_scores = np.empty(n_samples, dtype=np.float64)
    for start in range(0, n_samples, block_size):
        stop = min(start + block_size, n_samples)
        matches = (
            source_neighbors[start:stop, :, np.newaxis]
            == target_neighbors[start:stop, np.newaxis, :]
        )
        overlap_counts = np.count_nonzero(np.any(matches, axis=2), axis=1)
        local_scores[start:stop] = overlap_counts / n_neighbors
    return local_scores


def _rank_penalty_sum_low_memory(
    points: np.ndarray,
    candidate_neighbors: np.ndarray,
    *,
    n_neighbors: int,
    metric: str | Any,
    metric_params: Optional[dict[str, Any]],
    batch_size: int,
) -> float:
    n_samples = points.shape[0]
    penalty_sum = 0.0
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        distances = pairwise_distances(
            points[start:stop],
            points,
            metric=metric,
            **(metric_params or {}),
        )
        block_rows = np.arange(stop - start)
        distances[block_rows, np.arange(start, stop)] = np.inf

        candidates = candidate_neighbors[start:stop]
        candidate_distances = distances[block_rows[:, np.newaxis], candidates]
        for neighbor_index in range(n_neighbors):
            ranks = (
                np.count_nonzero(
                    distances < candidate_distances[:, neighbor_index, np.newaxis],
                    axis=1,
                )
                + 1
            )
            penalties = ranks - n_neighbors
            penalty_sum += float(np.sum(penalties[penalties > 0]))
    return penalty_sum


def _normalization_term(n_samples: int, n_neighbors: int) -> float:
    return 2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0))


def trustworthiness_from_embedding_neighbors(
    high_dimensional_points: Any,
    embedding_neighbors: Any,
    *,
    n_neighbors: Optional[int] = None,
    metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    low_memory_batch_size: int = 256,
    dtype: Any = np.float32,
    device: str = "cpu",
    gpu_batch_size: Optional[int] = None,
) -> float:
    """Compute trustworthiness from precomputed embedding neighbor indices."""
    resolved_dtype = resolve_evaluation_dtype(dtype)
    source = _validate_points(high_dimensional_points, "high_dimensional_points", resolved_dtype)
    candidates = np.asarray(embedding_neighbors, dtype=np.int64)
    if candidates.ndim != 2:
        raise ValueError("embedding_neighbors must be a 2D array")
    if candidates.shape[0] != source.shape[0]:
        raise ValueError("embedding_neighbors must have one row per source sample")
    k = _validate_rank_metric_neighbors(
        candidates.shape[1] if n_neighbors is None else n_neighbors,
        source.shape[0],
    )
    candidates = candidates[:, :k]
    resolved_device = resolve_device(device)

    if resolved_device == "gpu":
        from ._gpu import _rank_penalty_sum_gpu, _validate_gpu_batch_size

        _ensure_gpu_euclidean_metric(
            metric,
            metric_params,
            metric_name="trustworthiness_from_embedding_neighbors",
        )
        batch_size = _validate_gpu_batch_size(gpu_batch_size, low_memory_batch_size)
        import cupy as cp

        penalty_sum = _rank_penalty_sum_gpu(
            source,
            cp.asarray(candidates),
            n_neighbors=k,
            batch_size=batch_size,
        )
    else:
        batch_size = _validate_low_memory_batch_size(low_memory_batch_size)
        penalty_sum = _rank_penalty_sum_low_memory(
            source,
            candidates,
            n_neighbors=k,
            metric=metric,
            metric_params=metric_params,
            batch_size=batch_size,
        )
    return float(1.0 - penalty_sum * _normalization_term(source.shape[0], k))


def continuity_from_source_neighbors(
    embedding: Any,
    source_neighbors: Any,
    *,
    n_neighbors: Optional[int] = None,
    low_memory_batch_size: int = 256,
    dtype: Any = np.float32,
    device: str = "cpu",
    gpu_batch_size: Optional[int] = None,
) -> float:
    """Compute continuity from precomputed high-dimensional neighbor indices."""
    resolved_dtype = resolve_evaluation_dtype(dtype)
    target = _validate_points(embedding, "embedding", resolved_dtype)
    candidates = np.asarray(source_neighbors, dtype=np.int64)
    if candidates.ndim != 2:
        raise ValueError("source_neighbors must be a 2D array")
    if candidates.shape[0] != target.shape[0]:
        raise ValueError("source_neighbors must have one row per embedding sample")
    k = _validate_rank_metric_neighbors(
        candidates.shape[1] if n_neighbors is None else n_neighbors,
        target.shape[0],
    )
    candidates = candidates[:, :k]
    resolved_device = resolve_device(device)

    if resolved_device == "gpu":
        from ._gpu import _rank_penalty_sum_gpu, _validate_gpu_batch_size

        batch_size = _validate_gpu_batch_size(gpu_batch_size, low_memory_batch_size)
        import cupy as cp

        penalty_sum = _rank_penalty_sum_gpu(
            target,
            cp.asarray(candidates),
            n_neighbors=k,
            batch_size=batch_size,
        )
    else:
        batch_size = _validate_low_memory_batch_size(low_memory_batch_size)
        penalty_sum = _rank_penalty_sum_low_memory(
            target,
            candidates,
            n_neighbors=k,
            metric="euclidean",
            metric_params=None,
            batch_size=batch_size,
        )
    return float(1.0 - penalty_sum * _normalization_term(target.shape[0], k))


def trustworthiness(
    high_dimensional_points: Any,
    embedding: Any,
    *,
    n_neighbors: int = 5,
    metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    low_memory: bool = True,
    low_memory_batch_size: int = 256,
    dtype: Any = np.float32,
    device: str = "cpu",
    gpu_batch_size: Optional[int] = None,
) -> float:
    resolved_dtype = resolve_evaluation_dtype(dtype)
    source, target = _validate_pair(high_dimensional_points, embedding, dtype=resolved_dtype)
    k = _validate_rank_metric_neighbors(n_neighbors, source.shape[0])
    resolved_device = resolve_device(device)

    if resolved_device == "gpu":
        from ._gpu import _validate_gpu_batch_size, trustworthiness_gpu

        batch_size = _validate_gpu_batch_size(gpu_batch_size, low_memory_batch_size)
        return trustworthiness_gpu(
            source,
            target,
            n_neighbors=k,
            metric=metric,
            metric_params=metric_params,
            batch_size=batch_size,
        )

    if low_memory:
        batch_size = _validate_low_memory_batch_size(low_memory_batch_size)
        target_neighbors = compute_neighbors(
            target,
            n_neighbors=k,
            metric="euclidean",
            metric_params=None,
            dtype=resolved_dtype,
        ).indices
        return trustworthiness_from_embedding_neighbors(
            source,
            target_neighbors,
            n_neighbors=k,
            metric=metric,
            metric_params=metric_params,
            low_memory_batch_size=batch_size,
            dtype=resolved_dtype,
        )

    source_order, source_ranks = _pairwise_neighbor_order_and_ranks(
        source,
        metric=metric,
        metric_params=metric_params,
    )
    target_order, _ = _pairwise_neighbor_order_and_ranks(
        target,
        metric="euclidean",
        metric_params=None,
    )

    rows = np.arange(source.shape[0])[:, np.newaxis]
    penalties = source_ranks[rows, target_order[:, :k]] - k
    penalty_sum = float(np.sum(penalties[penalties > 0]))
    score = 1.0 - penalty_sum * _normalization_term(source.shape[0], k)
    return float(score)


def continuity(
    high_dimensional_points: Any,
    embedding: Any,
    *,
    n_neighbors: int = 5,
    metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    low_memory: bool = True,
    low_memory_batch_size: int = 256,
    dtype: Any = np.float32,
    device: str = "cpu",
    gpu_batch_size: Optional[int] = None,
) -> float:
    resolved_dtype = resolve_evaluation_dtype(dtype)
    source, target = _validate_pair(high_dimensional_points, embedding, dtype=resolved_dtype)
    k = _validate_rank_metric_neighbors(n_neighbors, source.shape[0])
    resolved_device = resolve_device(device)

    if resolved_device == "gpu":
        from ._gpu import _validate_gpu_batch_size, continuity_gpu

        batch_size = _validate_gpu_batch_size(gpu_batch_size, low_memory_batch_size)
        return continuity_gpu(
            source,
            target,
            n_neighbors=k,
            metric=metric,
            metric_params=metric_params,
            batch_size=batch_size,
        )

    if low_memory:
        batch_size = _validate_low_memory_batch_size(low_memory_batch_size)
        source_neighbors = compute_neighbors(
            source,
            n_neighbors=k,
            metric=metric,
            metric_params=metric_params,
            dtype=resolved_dtype,
        ).indices
        return continuity_from_source_neighbors(
            target,
            source_neighbors,
            n_neighbors=k,
            low_memory_batch_size=batch_size,
            dtype=resolved_dtype,
        )

    source_order, _ = _pairwise_neighbor_order_and_ranks(
        source,
        metric=metric,
        metric_params=metric_params,
    )
    _, target_ranks = _pairwise_neighbor_order_and_ranks(
        target,
        metric="euclidean",
        metric_params=None,
    )

    rows = np.arange(source.shape[0])[:, np.newaxis]
    penalties = target_ranks[rows, source_order[:, :k]] - k
    penalty_sum = float(np.sum(penalties[penalties > 0]))
    score = 1.0 - penalty_sum * _normalization_term(source.shape[0], k)
    return float(score)


def neighborhood_preservation(
    high_dimensional_points: Any,
    embedding: Any,
    *,
    n_neighbors: int = 15,
    metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    dtype: Any = np.float32,
    device: str = "cpu",
    gpu_batch_size: Optional[int] = None,
) -> NeighborhoodPreservationResult:
    resolved_dtype = resolve_evaluation_dtype(dtype)
    source, target = _validate_pair(high_dimensional_points, embedding, dtype=resolved_dtype)
    k = _validate_overlap_neighbors(n_neighbors, source.shape[0])
    resolved_device = resolve_device(device)

    if resolved_device == "gpu":
        from ._gpu import _validate_gpu_batch_size, neighborhood_preservation_gpu

        batch_size = _validate_gpu_batch_size(gpu_batch_size, 1024)
        return neighborhood_preservation_gpu(
            source,
            target,
            n_neighbors=k,
            metric=metric,
            metric_params=metric_params,
            batch_size=batch_size,
        )

    source_neighbors = compute_neighbors(
        source,
        n_neighbors=k,
        metric=metric,
        metric_params=metric_params,
        dtype=resolved_dtype,
    ).indices
    target_neighbors = compute_neighbors(
        target,
        n_neighbors=k,
        metric="euclidean",
        metric_params=None,
        dtype=resolved_dtype,
    ).indices

    return neighborhood_preservation_from_neighbors(source_neighbors, target_neighbors)
