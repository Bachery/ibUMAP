from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from ._device import resolve_device
from ._dtype import resolve_evaluation_dtype


@dataclass(frozen=True)
class GeodesicCorrelationResult:
    spearman_correlation: float
    pvalue: float
    n_pairs: int
    n_neighbors: int
    connected_components: int
    finite_pair_fraction: float
    mode: str = "landmark"
    n_landmarks: Optional[int] = None


@dataclass(frozen=True)
class GeodesicSourceState:
    graph: csr_matrix
    n_neighbors: int
    connected_components: int
    mode: str
    landmark_indices: Optional[np.ndarray] = None
    landmark_distances: Optional[np.ndarray] = None
    finite_pair_fraction: Optional[float] = None


@dataclass(frozen=True)
class GeodesicEmbeddingState:
    landmark_distances: Optional[np.ndarray] = None


GeodesicMode = Literal["full", "exact_low_memory", "landmark", "approx_knn"]


def _validate_points(points: Any, name: str, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(points, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _build_symmetric_knn_graph(
    points: np.ndarray,
    n_neighbors: int,
    metric: str | Any,
    metric_params: Optional[dict[str, Any]],
) -> csr_matrix:
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")
    if n_neighbors >= points.shape[0]:
        raise ValueError("n_neighbors must be smaller than the number of samples")

    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric=metric,
        metric_params=metric_params,
    )
    nn.fit(points)
    distances, indices = nn.kneighbors(points, return_distance=True)

    weights: dict[tuple[int, int], float] = {}
    for source in range(points.shape[0]):
        for distance, target in zip(distances[source, 1:], indices[source, 1:]):
            edge = (source, int(target))
            reverse = (int(target), source)
            best = weights.get(edge, float(distance))
            weights[edge] = min(best, float(distance))
            best_reverse = weights.get(reverse, float(distance))
            weights[reverse] = min(best_reverse, float(distance))

    rows = np.fromiter((row for row, _ in weights.keys()), dtype=np.int64)
    cols = np.fromiter((col for _, col in weights.keys()), dtype=np.int64)
    data = np.fromiter(weights.values(), dtype=points.dtype)
    return csr_matrix((data, (rows, cols)), shape=(points.shape[0], points.shape[0]))


def _exclude_self_neighbors(
    distances: np.ndarray,
    indices: np.ndarray,
    *,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = indices.shape[0]
    filtered_indices = np.empty((n_samples, n_neighbors), dtype=np.int64)
    filtered_distances = np.empty((n_samples, n_neighbors), dtype=distances.dtype)
    for row in range(n_samples):
        keep = indices[row] != row
        row_indices = indices[row, keep]
        row_distances = distances[row, keep]
        if row_indices.shape[0] < n_neighbors:
            raise ValueError("Could not find enough non-self neighbors")
        filtered_indices[row] = row_indices[:n_neighbors]
        filtered_distances[row] = row_distances[:n_neighbors]
    return filtered_distances, filtered_indices


def _symmetric_graph_from_neighbor_arrays(
    distances: np.ndarray,
    indices: np.ndarray,
    *,
    n_samples: int,
    dtype: np.dtype,
) -> csr_matrix:
    rows = np.repeat(np.arange(n_samples, dtype=np.int64), indices.shape[1])
    cols = indices.astype(np.int64, copy=False).reshape(-1)
    data = distances.astype(dtype, copy=False).reshape(-1)
    directed = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    return directed.maximum(directed.T)


def _build_symmetric_knn_graph_fast(
    points: np.ndarray,
    n_neighbors: int,
    metric: str | Any,
    metric_params: Optional[dict[str, Any]],
) -> csr_matrix:
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")
    if n_neighbors >= points.shape[0]:
        raise ValueError("n_neighbors must be smaller than the number of samples")

    nn = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric=metric,
        metric_params=metric_params,
    )
    nn.fit(points)
    distances, indices = nn.kneighbors(points, return_distance=True)
    distances, indices = _exclude_self_neighbors(
        distances,
        indices,
        n_neighbors=n_neighbors,
    )
    return _symmetric_graph_from_neighbor_arrays(
        distances,
        indices,
        n_samples=points.shape[0],
        dtype=points.dtype,
    )


def _build_approximate_knn_graph(
    points: np.ndarray,
    n_neighbors: int,
    metric: str | Any,
    metric_params: Optional[dict[str, Any]],
    random_state: Optional[int],
    n_jobs: int,
) -> csr_matrix:
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")
    if n_neighbors >= points.shape[0]:
        raise ValueError("n_neighbors must be smaller than the number of samples")

    try:
        from pynndescent import NNDescent
    except ImportError as exc:
        raise ImportError("mode='approx_knn' requires pynndescent") from exc

    index_points = np.array(
        points,
        dtype=points.dtype,
        order="C",
        copy=not (points.flags.writeable and points.flags.c_contiguous),
    )

    index = NNDescent(
        index_points,
        n_neighbors=n_neighbors + 1,
        metric=metric,
        metric_kwds=metric_params or {},
        random_state=random_state,
        n_jobs=n_jobs,
    )
    indices, distances = index.neighbor_graph
    distances, indices = _exclude_self_neighbors(
        np.asarray(distances),
        np.asarray(indices),
        n_neighbors=n_neighbors,
    )
    return _symmetric_graph_from_neighbor_arrays(
        distances,
        indices,
        n_samples=points.shape[0],
        dtype=points.dtype,
    )


def _validate_mode(mode: str) -> GeodesicMode:
    if mode not in {"full", "exact_low_memory", "landmark", "approx_knn"}:
        raise ValueError("mode must be one of 'full', 'exact_low_memory', 'landmark', or 'approx_knn'")
    return mode  # type: ignore[return-value]


def _sample_upper_triangle_pairs(
    n_samples: int,
    pair_sample_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    total_pairs = n_samples * (n_samples - 1) // 2
    if pair_sample_size >= total_pairs:
        upper = np.triu_indices(n_samples, k=1)
        return upper[0].astype(np.int64, copy=False), upper[1].astype(np.int64, copy=False)

    flat = rng.choice(total_pairs, size=pair_sample_size, replace=False)
    row_ends = np.cumsum(np.arange(n_samples - 1, 0, -1, dtype=np.int64))
    rows = np.searchsorted(row_ends, flat, side="right").astype(np.int64, copy=False)
    previous_ends = np.zeros_like(rows)
    nonzero = rows > 0
    previous_ends[nonzero] = row_ends[rows[nonzero] - 1]
    cols = rows + 1 + (flat - previous_ends)
    return rows, cols.astype(np.int64, copy=False)


def _embedded_distances_for_pairs(
    embedding: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    return np.linalg.norm(embedding[rows] - embedding[cols], axis=1)


def _spearman_result(
    geodesic_values: np.ndarray,
    embedded_values: np.ndarray,
    *,
    n_pairs: int,
    n_neighbors: int,
    connected_components_count: int,
    finite_pair_fraction: float,
    mode: str,
    n_landmarks: Optional[int] = None,
) -> GeodesicCorrelationResult:
    if geodesic_values.shape[0] < 2:
        raise ValueError("At least two finite distance pairs are required")
    correlation, pvalue = spearmanr(geodesic_values, embedded_values)
    return GeodesicCorrelationResult(
        spearman_correlation=float(correlation),
        pvalue=float(pvalue),
        n_pairs=int(n_pairs),
        n_neighbors=int(n_neighbors),
        connected_components=int(connected_components_count),
        finite_pair_fraction=float(finite_pair_fraction),
        mode=mode,
        n_landmarks=None if n_landmarks is None else int(n_landmarks),
    )


def _full_geodesic_correlation(
    graph: csr_matrix,
    embedding: np.ndarray,
    *,
    n_neighbors: int,
    connected_components_count: int,
    pair_sample_size: Optional[int],
    random_state: Optional[int],
    mode: str,
) -> GeodesicCorrelationResult:
    n_samples = embedding.shape[0]
    geodesic_distances = shortest_path(graph, directed=False, unweighted=False)
    embedded_distances = pairwise_distances(embedding, metric="euclidean")

    upper = np.triu_indices(n_samples, k=1)
    finite_mask = np.isfinite(geodesic_distances[upper])
    finite_pair_fraction = float(np.mean(finite_mask))
    if not np.any(finite_mask):
        raise ValueError("The KNN graph is fully disconnected; geodesic correlation cannot be computed")

    geodesic_values = geodesic_distances[upper][finite_mask]
    embedded_values = embedded_distances[upper][finite_mask]

    if pair_sample_size is not None and geodesic_values.shape[0] > pair_sample_size:
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(geodesic_values.shape[0], size=pair_sample_size, replace=False)
        geodesic_values = geodesic_values[chosen]
        embedded_values = embedded_values[chosen]

    return _spearman_result(
        geodesic_values,
        embedded_values,
        n_pairs=geodesic_values.shape[0],
        n_neighbors=n_neighbors,
        connected_components_count=connected_components_count,
        finite_pair_fraction=finite_pair_fraction,
        mode=mode,
    )


def _exact_low_memory_geodesic_correlation(
    graph: csr_matrix,
    embedding: np.ndarray,
    *,
    n_neighbors: int,
    connected_components_count: int,
    pair_sample_size: Optional[int],
    random_state: Optional[int],
    shortest_path_batch_size: int,
) -> GeodesicCorrelationResult:
    if shortest_path_batch_size < 1:
        raise ValueError("shortest_path_batch_size must be at least 1")

    n_samples = embedding.shape[0]
    total_pairs = n_samples * (n_samples - 1) // 2

    if pair_sample_size is not None:
        rng = np.random.default_rng(random_state)
        rows, cols = _sample_upper_triangle_pairs(n_samples, int(pair_sample_size), rng)
        order = np.argsort(rows, kind="mergesort")
        rows = rows[order]
        cols = cols[order]
        geodesic_values_all = np.empty(rows.shape[0], dtype=embedding.dtype)
        embedded_values_all = np.empty(rows.shape[0], dtype=embedding.dtype)
        write_offset = 0
        for start in range(0, rows.shape[0], shortest_path_batch_size):
            stop = min(start + shortest_path_batch_size, rows.shape[0])
            source_rows = np.unique(rows[start:stop])
            geodesic_block = shortest_path(
                graph,
                directed=False,
                unweighted=False,
                indices=source_rows,
            )
            source_to_local = {int(source): idx for idx, source in enumerate(source_rows)}
            local_rows = np.fromiter(
                (source_to_local[int(row)] for row in rows[start:stop]),
                dtype=np.int64,
                count=stop - start,
            )
            geodesic_block_values = geodesic_block[local_rows, cols[start:stop]]
            finite_mask = np.isfinite(geodesic_block_values)
            finite_count = int(np.count_nonzero(finite_mask))
            if finite_count:
                next_offset = write_offset + finite_count
                geodesic_values_all[write_offset:next_offset] = geodesic_block_values[finite_mask]
                embedded_values_all[write_offset:next_offset] = _embedded_distances_for_pairs(
                    embedding,
                    rows[start:stop][finite_mask],
                    cols[start:stop][finite_mask],
                )
                write_offset = next_offset

        if write_offset == 0:
            raise ValueError("The sampled pairs are fully disconnected; geodesic correlation cannot be computed")
        geodesic_values = geodesic_values_all[:write_offset]
        embedded_values = embedded_values_all[:write_offset]
        finite_pair_fraction = write_offset / rows.shape[0]
        return _spearman_result(
            geodesic_values,
            embedded_values,
            n_pairs=geodesic_values.shape[0],
            n_neighbors=n_neighbors,
            connected_components_count=connected_components_count,
            finite_pair_fraction=finite_pair_fraction,
            mode="exact_low_memory",
        )

    geodesic_values_all = np.empty(total_pairs, dtype=embedding.dtype)
    embedded_values_all = np.empty(total_pairs, dtype=embedding.dtype)
    write_offset = 0
    for start in range(0, n_samples, shortest_path_batch_size):
        stop = min(start + shortest_path_batch_size, n_samples)
        source_rows = np.arange(start, stop, dtype=np.int64)
        geodesic_block = shortest_path(
            graph,
            directed=False,
            unweighted=False,
            indices=source_rows,
        )
        embedded_block = pairwise_distances(embedding[start:stop], embedding, metric="euclidean")

        for local_row, source in enumerate(range(start, stop)):
            if source + 1 >= n_samples:
                continue
            geodesic_values = geodesic_block[local_row, source + 1 :]
            finite_mask = np.isfinite(geodesic_values)
            finite_count = int(np.count_nonzero(finite_mask))
            if finite_count:
                next_offset = write_offset + finite_count
                geodesic_values_all[write_offset:next_offset] = geodesic_values[finite_mask]
                embedded_values_all[write_offset:next_offset] = embedded_block[local_row, source + 1 :][finite_mask]
                write_offset = next_offset

    if write_offset == 0:
        raise ValueError("The KNN graph is fully disconnected; geodesic correlation cannot be computed")
    geodesic_values = geodesic_values_all[:write_offset]
    embedded_values = embedded_values_all[:write_offset]
    finite_pair_fraction = write_offset / total_pairs
    return _spearman_result(
        geodesic_values,
        embedded_values,
        n_pairs=geodesic_values.shape[0],
        n_neighbors=n_neighbors,
        connected_components_count=connected_components_count,
        finite_pair_fraction=finite_pair_fraction,
        mode="exact_low_memory",
    )


def _resolve_landmark_indices(
    n_samples: int,
    *,
    n_landmarks: Optional[int],
    landmark_indices: Optional[Any],
    random_state: Optional[int],
) -> np.ndarray:
    if landmark_indices is not None:
        indices = np.asarray(landmark_indices, dtype=np.int64)
        if indices.ndim != 1:
            raise ValueError("landmark_indices must be a 1D array")
        if indices.shape[0] < 1:
            raise ValueError("landmark_indices must contain at least one index")
        if np.any(indices < 0) or np.any(indices >= n_samples):
            raise ValueError("landmark_indices contain out-of-range values")
        if np.unique(indices).shape[0] != indices.shape[0]:
            raise ValueError("landmark_indices must not contain duplicates")
        return np.sort(indices)

    resolved_n_landmarks = 256 if n_landmarks is None else int(n_landmarks)
    if resolved_n_landmarks < 1:
        raise ValueError("n_landmarks must be at least 1")
    resolved_n_landmarks = min(resolved_n_landmarks, n_samples)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_samples, size=resolved_n_landmarks, replace=False))


def _landmark_geodesic_correlation(
    graph: csr_matrix,
    embedding: np.ndarray,
    *,
    n_neighbors: int,
    connected_components_count: int,
    n_landmarks: Optional[int],
    landmark_indices: Optional[Any],
    random_state: Optional[int],
    mode: str,
) -> GeodesicCorrelationResult:
    n_samples = embedding.shape[0]
    landmarks = _resolve_landmark_indices(
        n_samples,
        n_landmarks=n_landmarks,
        landmark_indices=landmark_indices,
        random_state=random_state,
    )
    geodesic_distances = shortest_path(
        graph,
        directed=False,
        unweighted=False,
        indices=landmarks,
    )
    embedded_distances = pairwise_distances(embedding[landmarks], embedding, metric="euclidean")

    row_indices = np.arange(landmarks.shape[0])
    geodesic_distances[row_indices, landmarks] = np.inf
    embedded_distances[row_indices, landmarks] = np.inf
    finite_mask = np.isfinite(geodesic_distances)
    valid_pair_count = landmarks.shape[0] * (n_samples - 1)
    finite_pair_fraction = float(np.count_nonzero(finite_mask) / valid_pair_count)
    if not np.any(finite_mask):
        raise ValueError("The landmark pairs are fully disconnected; geodesic correlation cannot be computed")

    geodesic_values = geodesic_distances[finite_mask]
    embedded_values = embedded_distances[finite_mask]
    return _spearman_result(
        geodesic_values,
        embedded_values,
        n_pairs=geodesic_values.shape[0],
        n_neighbors=n_neighbors,
        connected_components_count=connected_components_count,
        finite_pair_fraction=finite_pair_fraction,
        mode=mode,
        n_landmarks=landmarks.shape[0],
    )


def compute_geodesic_source_state(
    points: Any,
    *,
    n_neighbors: int = 15,
    metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    mode: GeodesicMode = "landmark",
    n_landmarks: Optional[int] = None,
    landmark_indices: Optional[Any] = None,
    random_state: Optional[int] = None,
    approx_knn_n_jobs: int = -1,
    dtype: Any = np.float32,
    device: str = "cpu",
    gpu_batch_size: Optional[int] = None,
) -> GeodesicSourceState:
    """Build reusable source-side geodesic state for one high-dimensional dataset."""
    resolved_dtype = resolve_evaluation_dtype(dtype)
    source = _validate_points(points, "points", resolved_dtype)
    resolved_mode = _validate_mode(mode)
    resolved_device = resolve_device(device)

    if resolved_device == "gpu":
        if resolved_mode != "landmark":
            raise NotImplementedError(
                "device='gpu' for compute_geodesic_source_state currently supports only "
                "mode='landmark'"
            )
        from ._gpu import _ensure_euclidean_metric, _exact_neighbors_gpu, _validate_gpu_batch_size

        _ensure_euclidean_metric(metric, metric_params, metric_name="compute_geodesic_source_state")
        batch_size = _validate_gpu_batch_size(gpu_batch_size, 1024)
        distances_gpu, indices_gpu = _exact_neighbors_gpu(
            source,
            n_neighbors=n_neighbors,
            batch_size=batch_size,
        )
        import cupy as cp

        graph = _symmetric_graph_from_neighbor_arrays(
            cp.asnumpy(distances_gpu),
            cp.asnumpy(indices_gpu),
            n_samples=source.shape[0],
            dtype=source.dtype,
        )
    elif resolved_mode == "full":
        graph = _build_symmetric_knn_graph(
            source,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_params=metric_params,
        )
    elif resolved_mode == "approx_knn":
        graph = _build_approximate_knn_graph(
            source,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_params=metric_params,
            random_state=random_state,
            n_jobs=approx_knn_n_jobs,
        )
    else:
        graph = _build_symmetric_knn_graph_fast(
            source,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_params=metric_params,
        )

    component_count, _ = connected_components(graph, directed=False)
    if resolved_mode not in {"landmark", "approx_knn"}:
        return GeodesicSourceState(
            graph=graph,
            n_neighbors=int(n_neighbors),
            connected_components=int(component_count),
            mode=resolved_mode,
        )

    landmarks = _resolve_landmark_indices(
        source.shape[0],
        n_landmarks=n_landmarks,
        landmark_indices=landmark_indices,
        random_state=random_state,
    )
    geodesic_distances = shortest_path(
        graph,
        directed=False,
        unweighted=False,
        indices=landmarks,
    ).astype(resolved_dtype, copy=False)
    row_indices = np.arange(landmarks.shape[0])
    geodesic_distances[row_indices, landmarks] = np.inf
    valid_pair_count = landmarks.shape[0] * (source.shape[0] - 1)
    finite_pair_fraction = (
        float(np.count_nonzero(np.isfinite(geodesic_distances)) / valid_pair_count)
        if valid_pair_count
        else 0.0
    )
    return GeodesicSourceState(
        graph=graph,
        n_neighbors=int(n_neighbors),
        connected_components=int(component_count),
        mode=resolved_mode,
        landmark_indices=landmarks,
        landmark_distances=geodesic_distances,
        finite_pair_fraction=finite_pair_fraction,
    )


def compute_geodesic_embedding_state(
    embedding: Any,
    source_state: GeodesicSourceState,
    *,
    dtype: Any = np.float32,
    device: str = "cpu",
    gpu_batch_size: Optional[int] = None,
) -> GeodesicEmbeddingState:
    """Build reusable embedding-side distances aligned to a source geodesic state."""
    resolved_dtype = resolve_evaluation_dtype(dtype)
    target = _validate_points(embedding, "embedding", resolved_dtype)
    landmarks = source_state.landmark_indices
    if landmarks is None:
        return GeodesicEmbeddingState(landmark_distances=None)
    if target.shape[0] != source_state.graph.shape[0]:
        raise ValueError("embedding and source_state must contain the same number of samples")
    resolved_device = resolve_device(device)
    if resolved_device == "gpu":
        from ._gpu import _embedded_distances_landmark_gpu, _validate_gpu_batch_size

        batch_size = _validate_gpu_batch_size(gpu_batch_size, 1024)
        embedded_distances = _embedded_distances_landmark_gpu(
            target,
            np.asarray(landmarks, dtype=np.int64),
            batch_size=batch_size,
        ).astype(resolved_dtype, copy=False)
    else:
        embedded_distances = pairwise_distances(
            target[np.asarray(landmarks, dtype=np.int64)],
            target,
            metric="euclidean",
        ).astype(resolved_dtype, copy=False)
    row_indices = np.arange(len(landmarks))
    embedded_distances[row_indices, np.asarray(landmarks, dtype=np.int64)] = np.inf
    return GeodesicEmbeddingState(landmark_distances=embedded_distances)


def geodesic_distance_correlation_from_state(
    source_state: GeodesicSourceState,
    embedding: Any,
    *,
    embedding_state: Optional[GeodesicEmbeddingState] = None,
    pair_sample_size: Optional[int] = None,
    random_state: Optional[int] = None,
    shortest_path_batch_size: int = 256,
    dtype: Any = np.float32,
    device: str = "cpu",
    gpu_batch_size: Optional[int] = None,
) -> GeodesicCorrelationResult:
    """Score geodesic correlation from reusable source and embedding state."""
    resolved_dtype = resolve_evaluation_dtype(dtype)
    target = _validate_points(embedding, "embedding", resolved_dtype)
    if target.shape[0] != source_state.graph.shape[0]:
        raise ValueError("embedding and source_state must contain the same number of samples")

    if source_state.mode == "full":
        return _full_geodesic_correlation(
            source_state.graph,
            target,
            n_neighbors=source_state.n_neighbors,
            connected_components_count=source_state.connected_components,
            pair_sample_size=pair_sample_size,
            random_state=random_state,
            mode="full",
        )
    if source_state.mode == "exact_low_memory":
        return _exact_low_memory_geodesic_correlation(
            source_state.graph,
            target,
            n_neighbors=source_state.n_neighbors,
            connected_components_count=source_state.connected_components,
            pair_sample_size=pair_sample_size,
            random_state=random_state,
            shortest_path_batch_size=int(shortest_path_batch_size),
        )

    state = embedding_state or compute_geodesic_embedding_state(
        target,
        source_state,
        dtype=resolved_dtype,
        device=device,
        gpu_batch_size=gpu_batch_size,
    )
    if source_state.landmark_indices is None or source_state.landmark_distances is None:
        raise ValueError("source_state does not contain landmark geodesic distances")
    if state.landmark_distances is None:
        raise ValueError("embedding_state does not contain landmark distances")
    finite_mask = np.isfinite(source_state.landmark_distances)
    finite_count = int(np.count_nonzero(finite_mask))
    if finite_count < 2:
        raise ValueError("At least two finite geodesic pairs are required")
    return _spearman_result(
        source_state.landmark_distances[finite_mask],
        state.landmark_distances[finite_mask],
        n_pairs=finite_count,
        n_neighbors=source_state.n_neighbors,
        connected_components_count=source_state.connected_components,
        finite_pair_fraction=(
            float(source_state.finite_pair_fraction)
            if source_state.finite_pair_fraction is not None
            else 0.0
        ),
        mode=source_state.mode,
        n_landmarks=len(source_state.landmark_indices),
    )


def geodesic_distance_correlation(
    high_dimensional_points: Any,
    embedding: Any,
    *,
    n_neighbors: int = 15,
    metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    pair_sample_size: Optional[int] = None,
    random_state: Optional[int] = None,
    mode: GeodesicMode = "landmark",
    n_landmarks: Optional[int] = None,
    landmark_indices: Optional[Any] = None,
    shortest_path_batch_size: int = 256,
    approx_knn_n_jobs: int = -1,
    dtype: Any = np.float32,
    device: str = "cpu",
    gpu_batch_size: Optional[int] = None,
) -> GeodesicCorrelationResult:
    resolved_dtype = resolve_evaluation_dtype(dtype)
    high_dimensional_points = _validate_points(
        high_dimensional_points,
        "high_dimensional_points",
        resolved_dtype,
    )
    embedding = _validate_points(embedding, "embedding", resolved_dtype)
    if high_dimensional_points.shape[0] != embedding.shape[0]:
        raise ValueError("high_dimensional_points and embedding must contain the same number of samples")
    if pair_sample_size is not None and pair_sample_size < 1:
        raise ValueError("pair_sample_size must be positive when provided")
    resolved_mode = _validate_mode(mode)
    resolved_device = resolve_device(device)

    if resolved_device == "gpu":
        from ._gpu import _validate_gpu_batch_size, geodesic_distance_correlation_gpu

        batch_size = _validate_gpu_batch_size(gpu_batch_size, 1024)
        return geodesic_distance_correlation_gpu(
            high_dimensional_points,
            embedding,
            n_neighbors=n_neighbors,
            metric=metric,
            metric_params=metric_params,
            random_state=random_state,
            mode=resolved_mode,
            n_landmarks=n_landmarks,
            landmark_indices=landmark_indices,
            batch_size=batch_size,
        )

    source_state = compute_geodesic_source_state(
        high_dimensional_points,
        n_neighbors=n_neighbors,
        metric=metric,
        metric_params=metric_params,
        mode=resolved_mode,
        n_landmarks=n_landmarks,
        landmark_indices=landmark_indices,
        random_state=random_state,
        approx_knn_n_jobs=approx_knn_n_jobs,
        dtype=resolved_dtype,
        device="cpu",
    )
    embedding_state = (
        compute_geodesic_embedding_state(embedding, source_state, dtype=resolved_dtype)
        if resolved_mode in {"landmark", "approx_knn"}
        else None
    )
    return geodesic_distance_correlation_from_state(
        source_state,
        embedding,
        embedding_state=embedding_state,
        pair_sample_size=pair_sample_size,
        random_state=random_state,
        shortest_path_batch_size=shortest_path_batch_size,
        dtype=resolved_dtype,
        device="cpu",
    )
