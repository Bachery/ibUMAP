"""Sampled global-structure quality metrics and reusable source-side states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import paired_distances

from ._dtype import resolve_evaluation_dtype


@dataclass(frozen=True)
class RandomTripletSourceState:
    """Source-space triplets and their cached relative-distance comparisons."""

    anchor_indices: np.ndarray
    first_indices: np.ndarray
    second_indices: np.ndarray
    source_comparisons: np.ndarray
    n_samples: int
    n_triplets_requested: int
    metric: Any
    metric_params: Optional[dict[str, Any]]


@dataclass(frozen=True)
class DistanceSpearmanSourceState:
    """Source-space point pairs and their cached distances."""

    first_indices: np.ndarray
    second_indices: np.ndarray
    source_distances: np.ndarray
    n_samples: int
    n_pairs_requested: Optional[int]
    metric: Any
    metric_params: Optional[dict[str, Any]]


@dataclass(frozen=True)
class RandomTripletAccuracyResult:
    score: float
    n_triplets: int
    n_valid_triplets: int
    source_tie_count: int
    embedding_tie_count: int


@dataclass(frozen=True)
class DistanceSpearmanCorrelationResult:
    spearman_correlation: float
    pvalue: float
    n_pairs: int


def _validate_points(points: Any, name: str, dtype: np.dtype, *, minimum_samples: int) -> np.ndarray:
    array = np.asarray(points, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] < minimum_samples:
        raise ValueError(f"{name} must contain at least {minimum_samples} samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_batch_size(distance_batch_size: int) -> int:
    batch_size = int(distance_batch_size)
    if batch_size < 1:
        raise ValueError("distance_batch_size must be at least 1")
    return batch_size


def _paired_distances(
    points: np.ndarray,
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    *,
    metric: str | Any,
    metric_params: Optional[dict[str, Any]],
    distance_batch_size: int,
) -> np.ndarray:
    """Calculate selected paired distances without materializing an NxN matrix."""
    distances = np.empty(first_indices.shape[0], dtype=np.float64)
    for start in range(0, first_indices.shape[0], distance_batch_size):
        stop = min(start + distance_batch_size, first_indices.shape[0])
        distances[start:stop] = paired_distances(
            points[first_indices[start:stop]],
            points[second_indices[start:stop]],
            metric=metric,
            **(metric_params or {}),
        )
    return distances


def _validate_indices(indices: Any, *, n_samples: int, width: int, name: str) -> np.ndarray:
    array = np.asarray(indices, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != width:
        raise ValueError(f"{name} must have shape (n_samples, {width})")
    if array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample")
    if np.any(array < 0) or np.any(array >= n_samples):
        raise ValueError(f"{name} contains an out-of-range sample index")
    if width > 1 and np.any(np.diff(np.sort(array, axis=1), axis=1) == 0):
        raise ValueError(f"Every row of {name} must contain distinct sample indices")
    return array


def _sample_pairs(n_samples: int, n_pairs: Optional[int], rng: np.random.Generator) -> np.ndarray:
    total_pairs = n_samples * (n_samples - 1) // 2
    if n_pairs is None or n_pairs >= total_pairs:
        rows, cols = np.triu_indices(n_samples, k=1)
        return np.column_stack((rows, cols)).astype(np.int64, copy=False)
    sample_size = int(n_pairs)
    if sample_size < 2:
        raise ValueError("n_pairs must be at least 2")
    flat = rng.choice(total_pairs, size=sample_size, replace=False)
    row_ends = np.cumsum(np.arange(n_samples - 1, 0, -1, dtype=np.int64))
    rows = np.searchsorted(row_ends, flat, side="right").astype(np.int64, copy=False)
    previous_ends = np.zeros_like(rows)
    nonzero = rows > 0
    previous_ends[nonzero] = row_ends[rows[nonzero] - 1]
    cols = rows + 1 + (flat - previous_ends)
    return np.column_stack((rows, cols)).astype(np.int64, copy=False)


def _sample_triplets(n_samples: int, n_triplets: int, rng: np.random.Generator) -> np.ndarray:
    sample_size = int(n_triplets)
    if sample_size < 1:
        raise ValueError("n_triplets must be at least 1")
    anchors = rng.integers(n_samples, size=sample_size, dtype=np.int64)
    first = rng.integers(n_samples - 1, size=sample_size, dtype=np.int64)
    first += first >= anchors

    second = rng.integers(n_samples - 2, size=sample_size, dtype=np.int64)
    lower = np.minimum(anchors, first)
    upper = np.maximum(anchors, first)
    second += second >= lower
    second += second >= upper
    return np.column_stack((anchors, first, second)).astype(np.int64, copy=False)


def compute_random_triplet_source_state(
    high_dimensional_points: Any,
    *,
    n_triplets: int = 100_000,
    random_state: Optional[int] = None,
    sampled_triplets: Optional[Any] = None,
    metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    distance_batch_size: int = 65_536,
    dtype: Any = np.float32,
) -> RandomTripletSourceState:
    """Prepare reusable source comparisons for random triplet accuracy (RTA).

    ``sampled_triplets`` contains ``(anchor, first, second)`` rows and can be
    supplied to make the sampled population explicit. Otherwise, distinct
    triplets are sampled reproducibly from ``random_state``.
    """
    resolved_dtype = resolve_evaluation_dtype(dtype)
    source = _validate_points(
        high_dimensional_points,
        "high_dimensional_points",
        resolved_dtype,
        minimum_samples=3,
    )
    batch_size = _validate_batch_size(distance_batch_size)
    if sampled_triplets is None:
        triplets = _sample_triplets(source.shape[0], n_triplets, np.random.default_rng(random_state))
    else:
        triplets = _validate_indices(
            sampled_triplets,
            n_samples=source.shape[0],
            width=3,
            name="sampled_triplets",
        )
    first_distances = _paired_distances(
        source,
        triplets[:, 0],
        triplets[:, 1],
        metric=metric,
        metric_params=metric_params,
        distance_batch_size=batch_size,
    )
    second_distances = _paired_distances(
        source,
        triplets[:, 0],
        triplets[:, 2],
        metric=metric,
        metric_params=metric_params,
        distance_batch_size=batch_size,
    )
    return RandomTripletSourceState(
        anchor_indices=triplets[:, 0],
        first_indices=triplets[:, 1],
        second_indices=triplets[:, 2],
        source_comparisons=np.sign(first_distances - second_distances).astype(np.int8, copy=False),
        n_samples=source.shape[0],
        n_triplets_requested=int(n_triplets) if sampled_triplets is None else int(triplets.shape[0]),
        metric=metric,
        metric_params=metric_params,
    )


def random_triplet_accuracy_from_state(
    source_state: RandomTripletSourceState,
    embedding: Any,
    *,
    distance_batch_size: int = 65_536,
    dtype: Any = np.float32,
) -> RandomTripletAccuracyResult:
    """Evaluate RTA from cached source triplets.

    Source-distance ties are excluded. An embedding-distance tie does not
    satisfy a non-tied source ordering and is counted as incorrect.
    """
    resolved_dtype = resolve_evaluation_dtype(dtype)
    target = _validate_points(embedding, "embedding", resolved_dtype, minimum_samples=3)
    if target.shape[0] != source_state.n_samples:
        raise ValueError("embedding and source_state must contain the same number of samples")
    batch_size = _validate_batch_size(distance_batch_size)
    first_distances = _paired_distances(
        target,
        source_state.anchor_indices,
        source_state.first_indices,
        metric="euclidean",
        metric_params=None,
        distance_batch_size=batch_size,
    )
    second_distances = _paired_distances(
        target,
        source_state.anchor_indices,
        source_state.second_indices,
        metric="euclidean",
        metric_params=None,
        distance_batch_size=batch_size,
    )
    target_comparisons = np.sign(first_distances - second_distances).astype(np.int8, copy=False)
    valid = source_state.source_comparisons != 0
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        raise ValueError("RTA is undefined because every sampled source triplet is tied")
    return RandomTripletAccuracyResult(
        score=float(np.mean(source_state.source_comparisons[valid] == target_comparisons[valid])),
        n_triplets=int(source_state.anchor_indices.shape[0]),
        n_valid_triplets=n_valid,
        source_tie_count=int(source_state.anchor_indices.shape[0] - n_valid),
        embedding_tie_count=int(np.count_nonzero(target_comparisons[valid] == 0)),
    )


def random_triplet_accuracy(
    high_dimensional_points: Any,
    embedding: Any,
    *,
    n_triplets: int = 100_000,
    random_state: Optional[int] = None,
    sampled_triplets: Optional[Any] = None,
    metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    distance_batch_size: int = 65_536,
    dtype: Any = np.float32,
) -> RandomTripletAccuracyResult:
    """Compute sampled random triplet accuracy directly from two point clouds."""
    source_state = compute_random_triplet_source_state(
        high_dimensional_points,
        n_triplets=n_triplets,
        random_state=random_state,
        sampled_triplets=sampled_triplets,
        metric=metric,
        metric_params=metric_params,
        distance_batch_size=distance_batch_size,
        dtype=dtype,
    )
    return random_triplet_accuracy_from_state(
        source_state,
        embedding,
        distance_batch_size=distance_batch_size,
        dtype=dtype,
    )


def compute_distance_spearman_source_state(
    high_dimensional_points: Any,
    *,
    n_pairs: Optional[int] = 100_000,
    random_state: Optional[int] = None,
    sampled_pairs: Optional[Any] = None,
    metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    distance_batch_size: int = 65_536,
    dtype: Any = np.float32,
) -> DistanceSpearmanSourceState:
    """Prepare source distances for sampled global distance Spearman scoring.

    Set ``n_pairs`` to ``None`` to score every unique unordered point pair.
    """
    resolved_dtype = resolve_evaluation_dtype(dtype)
    source = _validate_points(
        high_dimensional_points,
        "high_dimensional_points",
        resolved_dtype,
        minimum_samples=3,
    )
    batch_size = _validate_batch_size(distance_batch_size)
    if sampled_pairs is None:
        pairs = _sample_pairs(source.shape[0], n_pairs, np.random.default_rng(random_state))
    else:
        pairs = _validate_indices(
            sampled_pairs,
            n_samples=source.shape[0],
            width=2,
            name="sampled_pairs",
        )
        if pairs.shape[0] < 2:
            raise ValueError("sampled_pairs must contain at least two pairs")
    if pairs.shape[0] < 2:
        raise ValueError("At least two sampled pairs are required for Spearman correlation")
    source_distances = _paired_distances(
        source,
        pairs[:, 0],
        pairs[:, 1],
        metric=metric,
        metric_params=metric_params,
        distance_batch_size=batch_size,
    )
    return DistanceSpearmanSourceState(
        first_indices=pairs[:, 0],
        second_indices=pairs[:, 1],
        source_distances=source_distances,
        n_samples=source.shape[0],
        n_pairs_requested=n_pairs if sampled_pairs is None else int(pairs.shape[0]),
        metric=metric,
        metric_params=metric_params,
    )


def distance_spearman_correlation_from_state(
    source_state: DistanceSpearmanSourceState,
    embedding: Any,
    *,
    distance_batch_size: int = 65_536,
    dtype: Any = np.float32,
) -> DistanceSpearmanCorrelationResult:
    """Evaluate global distance Spearman correlation from cached source pairs."""
    resolved_dtype = resolve_evaluation_dtype(dtype)
    target = _validate_points(embedding, "embedding", resolved_dtype, minimum_samples=3)
    if target.shape[0] != source_state.n_samples:
        raise ValueError("embedding and source_state must contain the same number of samples")
    target_distances = _paired_distances(
        target,
        source_state.first_indices,
        source_state.second_indices,
        metric="euclidean",
        metric_params=None,
        distance_batch_size=_validate_batch_size(distance_batch_size),
    )
    correlation, pvalue = spearmanr(source_state.source_distances, target_distances)
    return DistanceSpearmanCorrelationResult(
        spearman_correlation=float(correlation),
        pvalue=float(pvalue),
        n_pairs=int(source_state.first_indices.shape[0]),
    )


def distance_spearman_correlation(
    high_dimensional_points: Any,
    embedding: Any,
    *,
    n_pairs: Optional[int] = 100_000,
    random_state: Optional[int] = None,
    sampled_pairs: Optional[Any] = None,
    metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    distance_batch_size: int = 65_536,
    dtype: Any = np.float32,
) -> DistanceSpearmanCorrelationResult:
    """Compute sampled global distance Spearman correlation directly."""
    source_state = compute_distance_spearman_source_state(
        high_dimensional_points,
        n_pairs=n_pairs,
        random_state=random_state,
        sampled_pairs=sampled_pairs,
        metric=metric,
        metric_params=metric_params,
        distance_batch_size=distance_batch_size,
        dtype=dtype,
    )
    return distance_spearman_correlation_from_state(
        source_state,
        embedding,
        distance_batch_size=distance_batch_size,
        dtype=dtype,
    )


# Short aliases match the metric names accepted by the experiment configuration.
rta = random_triplet_accuracy
distance_spearman = distance_spearman_correlation
