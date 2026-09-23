from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from sklearn.metrics import pairwise_distances

from ._dtype import resolve_evaluation_dtype


@dataclass(frozen=True)
class SCDEEDResult:
    """Cell-level scDEED scores and classifications.

    ``labels`` uses ``0`` for dubious, ``1`` for trustworthy, and ``2`` for
    intermediate samples.
    """

    rho_original: np.ndarray
    rho_permuted: np.ndarray
    dubious_threshold: float
    trustworthy_threshold: float
    dubious_indices: np.ndarray
    trustworthy_indices: np.ndarray
    intermediate_indices: np.ndarray
    labels: np.ndarray
    n_dubious: int
    n_trustworthy: int
    n_intermediate: int
    dubious_fraction: float
    n_selected_neighbors: int
    similarity_percent: float
    dubious_cutoff: float
    trustworthy_cutoff: float
    permutation_random_state: Optional[int]


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "get"):
        return value.get()
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    return value


def _validate_points(points: Any, name: str, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(_to_numpy(points), dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=dtype, order="C")


def _validate_pair(
    source: Any,
    embedding: Any,
    *,
    source_name: str,
    embedding_name: str,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    source_array = _validate_points(source, source_name, dtype)
    embedding_array = _validate_points(embedding, embedding_name, dtype)
    if source_array.shape[0] != embedding_array.shape[0]:
        raise ValueError(
            f"{source_name} and {embedding_name} must contain the same number of samples"
        )
    return source_array, embedding_array


def _validate_scdeed_parameters(
    *,
    n_samples: int,
    similarity_percent: float,
    dubious_cutoff: float,
    trustworthy_cutoff: float,
    low_memory_batch_size: int,
) -> tuple[float, float, float, int, int]:
    similarity = float(similarity_percent)
    if not np.isfinite(similarity) or not 0.0 < similarity < 1.0:
        raise ValueError("similarity_percent must be finite and strictly between 0 and 1")

    dubious = float(dubious_cutoff)
    trustworthy = float(trustworthy_cutoff)
    if not np.isfinite(dubious) or not np.isfinite(trustworthy):
        raise ValueError("dubious_cutoff and trustworthy_cutoff must be finite")
    if not 0.0 <= dubious < trustworthy <= 1.0:
        raise ValueError(
            "cutoffs must satisfy 0 <= dubious_cutoff < trustworthy_cutoff <= 1"
        )

    n_selected = int(np.floor(n_samples * similarity))
    if n_selected < 2:
        raise ValueError(
            "similarity_percent selects fewer than two neighbors; "
            "increase it or use more samples"
        )
    if n_selected >= n_samples:
        raise ValueError(
            "similarity_percent must select fewer neighbors than the number of samples"
        )

    batch_size = int(low_memory_batch_size)
    if batch_size < 1:
        raise ValueError("low_memory_batch_size must be at least 1")
    return similarity, dubious, trustworthy, n_selected, batch_size


def permute_features_across_samples(
    X: Any,
    *,
    random_state: Optional[int] = 1000,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Independently permute each feature across samples.

    For the expected ``samples x features`` layout, this preserves every
    feature's marginal distribution while breaking the joint sample structure.
    """

    resolved_dtype = resolve_evaluation_dtype(dtype)
    source = _validate_points(X, "X", resolved_dtype)
    rng = np.random.default_rng(random_state)
    permuted = np.empty_like(source)
    for feature_index in range(source.shape[1]):
        order = rng.permutation(source.shape[0])
        permuted[:, feature_index] = source[order, feature_index]
    return permuted


def _exclude_self_distances(
    distances: np.ndarray,
    *,
    start: int,
    stop: int,
) -> None:
    local_rows = np.arange(stop - start)
    global_rows = np.arange(start, stop)
    distances[local_rows, global_rows] = np.inf


def _rowwise_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    left_centered = left64 - np.mean(left64, axis=1, keepdims=True)
    right_centered = right64 - np.mean(right64, axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, axis=1)
        * np.sum(right_centered * right_centered, axis=1)
    )
    scores = np.full(left64.shape[0], np.nan, dtype=np.float64)
    valid = denominator > np.finfo(np.float64).tiny
    scores[valid] = numerator[valid] / denominator[valid]
    return np.clip(scores, -1.0, 1.0)


def _cell_similarity_scores(
    source: np.ndarray,
    embedding: np.ndarray,
    *,
    n_selected: int,
    source_metric: str | Any,
    metric_params: Optional[dict[str, Any]],
    low_memory: bool,
    batch_size: int,
) -> np.ndarray:
    n_samples = source.shape[0]
    resolved_batch_size = batch_size if low_memory else n_samples
    scores = np.empty(n_samples, dtype=np.float64)
    metric_kwargs = dict(metric_params or {})

    for start in range(0, n_samples, resolved_batch_size):
        stop = min(start + resolved_batch_size, n_samples)

        source_distances = pairwise_distances(
            source[start:stop],
            source,
            metric=source_metric,
            **metric_kwargs,
        )
        _exclude_self_distances(source_distances, start=start, stop=stop)
        source_neighbors = np.argsort(
            source_distances,
            axis=1,
            kind="stable",
        )[:, :n_selected]
        del source_distances

        embedding_distances = pairwise_distances(
            embedding[start:stop],
            embedding,
            metric="euclidean",
        )
        _exclude_self_distances(embedding_distances, start=start, stop=stop)

        post_distances_for_source_neighbors = np.take_along_axis(
            embedding_distances,
            source_neighbors,
            axis=1,
        )

        # Preserve the first vector before partitioning the embedding distances
        # in place.  The in-place partition avoids another batch_size x N
        # allocation while identifying the nearest post-embedding distances.
        embedding_distances.partition(n_selected - 1, axis=1)
        nearest_post_distances = np.sort(
            embedding_distances[:, :n_selected],
            axis=1,
        )

        scores[start:stop] = _rowwise_pearson(
            post_distances_for_source_neighbors,
            nearest_post_distances,
        )

    non_finite = np.flatnonzero(~np.isfinite(scores))
    if non_finite.size:
        preview = ", ".join(str(int(index)) for index in non_finite[:8])
        suffix = "..." if non_finite.size > 8 else ""
        raise ValueError(
            "scDEED Pearson correlation is undefined for "
            f"{non_finite.size} sample(s): {preview}{suffix}. "
            "Check for constant or collapsed distance profiles."
        )
    return scores


def _build_result(
    rho_original: np.ndarray,
    rho_permuted: np.ndarray,
    *,
    n_selected: int,
    similarity_percent: float,
    dubious_cutoff: float,
    trustworthy_cutoff: float,
    permutation_random_state: Optional[int],
) -> SCDEEDResult:
    dubious_threshold = float(np.quantile(rho_permuted, dubious_cutoff))
    trustworthy_threshold = float(np.quantile(rho_permuted, trustworthy_cutoff))

    dubious_mask = rho_original < dubious_threshold
    trustworthy_mask = rho_original > trustworthy_threshold
    intermediate_mask = ~(dubious_mask | trustworthy_mask)

    dubious_indices = np.flatnonzero(dubious_mask)
    trustworthy_indices = np.flatnonzero(trustworthy_mask)
    intermediate_indices = np.flatnonzero(intermediate_mask)
    labels = np.full(rho_original.shape[0], 2, dtype=np.int8)
    labels[dubious_indices] = 0
    labels[trustworthy_indices] = 1

    return SCDEEDResult(
        rho_original=rho_original,
        rho_permuted=rho_permuted,
        dubious_threshold=dubious_threshold,
        trustworthy_threshold=trustworthy_threshold,
        dubious_indices=dubious_indices,
        trustworthy_indices=trustworthy_indices,
        intermediate_indices=intermediate_indices,
        labels=labels,
        n_dubious=int(dubious_indices.size),
        n_trustworthy=int(trustworthy_indices.size),
        n_intermediate=int(intermediate_indices.size),
        dubious_fraction=float(dubious_indices.size / rho_original.shape[0]),
        n_selected_neighbors=int(n_selected),
        similarity_percent=float(similarity_percent),
        dubious_cutoff=float(dubious_cutoff),
        trustworthy_cutoff=float(trustworthy_cutoff),
        permutation_random_state=permutation_random_state,
    )


def scdeed_from_embeddings(
    X: Any,
    embedding: Any,
    X_permuted: Any,
    embedding_permuted: Any,
    *,
    similarity_percent: float = 0.5,
    dubious_cutoff: float = 0.05,
    trustworthy_cutoff: float = 0.95,
    source_metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    low_memory: bool = True,
    low_memory_batch_size: int = 128,
    dtype: Any = np.float32,
    permutation_random_state: Optional[int] = None,
) -> SCDEEDResult:
    """Evaluate scDEED from original and permuted embeddings.

    The exact low-memory path computes pairwise distances in row blocks and has
    ``O(low_memory_batch_size * N)`` working memory instead of materializing
    full ``N x N`` distance matrices.  Setting ``low_memory=False`` is intended
    only for small inputs and reference comparisons.
    """

    resolved_dtype = resolve_evaluation_dtype(dtype)
    source, target = _validate_pair(
        X,
        embedding,
        source_name="X",
        embedding_name="embedding",
        dtype=resolved_dtype,
    )
    permuted_source, permuted_target = _validate_pair(
        X_permuted,
        embedding_permuted,
        source_name="X_permuted",
        embedding_name="embedding_permuted",
        dtype=resolved_dtype,
    )
    if source.shape[0] != permuted_source.shape[0]:
        raise ValueError(
            "original and permuted inputs must contain the same number of samples"
        )

    similarity, dubious, trustworthy, n_selected, batch_size = (
        _validate_scdeed_parameters(
            n_samples=source.shape[0],
            similarity_percent=similarity_percent,
            dubious_cutoff=dubious_cutoff,
            trustworthy_cutoff=trustworthy_cutoff,
            low_memory_batch_size=low_memory_batch_size,
        )
    )

    rho_original = _cell_similarity_scores(
        source,
        target,
        n_selected=n_selected,
        source_metric=source_metric,
        metric_params=metric_params,
        low_memory=bool(low_memory),
        batch_size=batch_size,
    )
    rho_permuted = _cell_similarity_scores(
        permuted_source,
        permuted_target,
        n_selected=n_selected,
        source_metric=source_metric,
        metric_params=metric_params,
        low_memory=bool(low_memory),
        batch_size=batch_size,
    )
    return _build_result(
        rho_original,
        rho_permuted,
        n_selected=n_selected,
        similarity_percent=similarity,
        dubious_cutoff=dubious,
        trustworthy_cutoff=trustworthy,
        permutation_random_state=permutation_random_state,
    )


def scdeed(
    X: Any,
    embedding: Any,
    *,
    refit_embedding: Callable[[np.ndarray], Any],
    similarity_percent: float = 0.5,
    dubious_cutoff: float = 0.05,
    trustworthy_cutoff: float = 0.95,
    permutation_random_state: Optional[int] = 1000,
    source_metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    low_memory: bool = True,
    low_memory_batch_size: int = 128,
    dtype: Any = np.float32,
) -> SCDEEDResult:
    """Evaluate an embedding with the scDEED dubious-cell procedure.

    ``refit_embedding`` must construct a fresh reducer with the same algorithm,
    effective parameters, and fit random state used to produce ``embedding``,
    then call ``fit_transform`` on the supplied permuted input.  It must rebuild
    data-dependent state such as the KNN graph and must not mutate its input.
    """

    if not callable(refit_embedding):
        raise TypeError("refit_embedding must be callable")

    resolved_dtype = resolve_evaluation_dtype(dtype)
    source, target = _validate_pair(
        X,
        embedding,
        source_name="X",
        embedding_name="embedding",
        dtype=resolved_dtype,
    )
    permuted_source = permute_features_across_samples(
        source,
        random_state=permutation_random_state,
        dtype=resolved_dtype,
    )
    permuted_embedding = refit_embedding(permuted_source)

    return scdeed_from_embeddings(
        source,
        target,
        permuted_source,
        permuted_embedding,
        similarity_percent=similarity_percent,
        dubious_cutoff=dubious_cutoff,
        trustworthy_cutoff=trustworthy_cutoff,
        source_metric=source_metric,
        metric_params=metric_params,
        low_memory=low_memory,
        low_memory_batch_size=low_memory_batch_size,
        dtype=resolved_dtype,
        permutation_random_state=permutation_random_state,
    )
