from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path

from .geodesic import (
    GeodesicCorrelationResult,
    _resolve_landmark_indices,
    _spearman_result,
    _symmetric_graph_from_neighbor_arrays,
)
from .neighborhood import NeighborhoodPreservationResult, _normalization_term
from .persistent_homology import (
    PersistentHomologyComparison,
    PersistenceDiagram,
    _build_vietoris_rips_complex,
    _diagram_wasserstein,
    _intervals_to_array,
    _reduce_boundary_matrix,
    _validate_sampled_indices,
)
from .pseudotime import PseudotimeCorrelationResult, _compute_correlation


def _cupy():
    import cupy as cp

    return cp


def _ensure_euclidean_metric(
    metric: str | Any,
    metric_params: Optional[dict[str, Any]],
    *,
    metric_name: str,
) -> None:
    if metric not in {"euclidean", "l2"} or metric_params not in (None, {}):
        raise NotImplementedError(
            f"device='gpu' for {metric_name} currently supports only Euclidean distance"
        )


def _validate_gpu_batch_size(batch_size: Optional[int], default: int) -> int:
    resolved = default if batch_size is None else int(batch_size)
    if resolved < 1:
        raise ValueError("gpu_batch_size must be at least 1")
    return resolved


def _pairwise_squared_distances_gpu(x_block: Any, y: Any) -> Any:
    cp = _cupy()
    x_norm = cp.sum(x_block * x_block, axis=1)[:, None]
    y_norm = cp.sum(y * y, axis=1)[None, :]
    distances = x_norm + y_norm - 2.0 * (x_block @ y.T)
    return cp.maximum(distances, 0.0)


def _exact_neighbors_gpu(
    points: np.ndarray,
    *,
    n_neighbors: int,
    batch_size: int,
) -> tuple[Any, Any]:
    cp = _cupy()
    points_gpu = cp.asarray(points)
    n_samples = int(points.shape[0])
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1")
    if n_neighbors >= n_samples:
        raise ValueError("n_neighbors must be smaller than the number of samples")

    neighbor_indices = cp.empty((n_samples, n_neighbors), dtype=cp.int64)
    neighbor_distances = cp.empty((n_samples, n_neighbors), dtype=points_gpu.dtype)
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        block = points_gpu[start:stop]
        distances = _pairwise_squared_distances_gpu(block, points_gpu)
        rows = cp.arange(stop - start)
        distances[rows, cp.arange(start, stop)] = cp.inf

        partition = cp.argpartition(distances, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]
        partition_distances = cp.take_along_axis(distances, partition, axis=1)
        order = cp.argsort(partition_distances, axis=1)
        sorted_indices = cp.take_along_axis(partition, order, axis=1)
        sorted_distances = cp.take_along_axis(partition_distances, order, axis=1)

        neighbor_indices[start:stop] = sorted_indices
        neighbor_distances[start:stop] = cp.sqrt(sorted_distances)

    return neighbor_distances, neighbor_indices


def _rank_penalty_sum_gpu(
    points: np.ndarray,
    candidate_neighbors: Any,
    *,
    n_neighbors: int,
    batch_size: int,
) -> float:
    cp = _cupy()
    points_gpu = cp.asarray(points)
    n_samples = int(points.shape[0])
    penalty_sum = 0.0
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        distances = _pairwise_squared_distances_gpu(points_gpu[start:stop], points_gpu)
        rows = cp.arange(stop - start)
        distances[rows, cp.arange(start, stop)] = cp.inf

        candidates = candidate_neighbors[start:stop]
        candidate_distances = distances[rows[:, None], candidates]
        block_penalty = cp.asarray(0, dtype=cp.int64)
        for neighbor_index in range(n_neighbors):
            ranks = cp.count_nonzero(
                distances < candidate_distances[:, neighbor_index, None],
                axis=1,
            ) + 1
            penalties = ranks - n_neighbors
            block_penalty = block_penalty + cp.sum(cp.maximum(penalties, 0))
        penalty_sum += float(block_penalty.get())
    return penalty_sum


def trustworthiness_gpu(
    source: np.ndarray,
    target: np.ndarray,
    *,
    n_neighbors: int,
    metric: str | Any,
    metric_params: Optional[dict[str, Any]],
    batch_size: int,
) -> float:
    _ensure_euclidean_metric(metric, metric_params, metric_name="trustworthiness")
    _, target_neighbors = _exact_neighbors_gpu(
        target,
        n_neighbors=n_neighbors,
        batch_size=batch_size,
    )
    penalty_sum = _rank_penalty_sum_gpu(
        source,
        target_neighbors,
        n_neighbors=n_neighbors,
        batch_size=batch_size,
    )
    score = 1.0 - penalty_sum * _normalization_term(source.shape[0], n_neighbors)
    return float(score)


def continuity_gpu(
    source: np.ndarray,
    target: np.ndarray,
    *,
    n_neighbors: int,
    metric: str | Any,
    metric_params: Optional[dict[str, Any]],
    batch_size: int,
) -> float:
    _ensure_euclidean_metric(metric, metric_params, metric_name="continuity")
    _, source_neighbors = _exact_neighbors_gpu(
        source,
        n_neighbors=n_neighbors,
        batch_size=batch_size,
    )
    penalty_sum = _rank_penalty_sum_gpu(
        target,
        source_neighbors,
        n_neighbors=n_neighbors,
        batch_size=batch_size,
    )
    score = 1.0 - penalty_sum * _normalization_term(source.shape[0], n_neighbors)
    return float(score)


def neighborhood_preservation_gpu(
    source: np.ndarray,
    target: np.ndarray,
    *,
    n_neighbors: int,
    metric: str | Any,
    metric_params: Optional[dict[str, Any]],
    batch_size: int,
) -> NeighborhoodPreservationResult:
    cp = _cupy()
    _ensure_euclidean_metric(metric, metric_params, metric_name="neighborhood_preservation")
    _, source_neighbors = _exact_neighbors_gpu(
        source,
        n_neighbors=n_neighbors,
        batch_size=batch_size,
    )
    _, target_neighbors = _exact_neighbors_gpu(
        target,
        n_neighbors=n_neighbors,
        batch_size=batch_size,
    )

    n_samples = int(source.shape[0])
    local_scores = cp.empty(n_samples, dtype=cp.float64)
    overlap_batch_size = max(batch_size, 4096)
    for start in range(0, n_samples, overlap_batch_size):
        stop = min(start + overlap_batch_size, n_samples)
        matches = source_neighbors[start:stop, :, None] == target_neighbors[start:stop, None, :]
        overlap_counts = cp.count_nonzero(cp.any(matches, axis=2), axis=1)
        local_scores[start:stop] = overlap_counts / float(n_neighbors)

    local_scores_np = cp.asnumpy(local_scores)
    return NeighborhoodPreservationResult(
        score=float(np.mean(local_scores_np)),
        local_scores=local_scores_np,
        n_neighbors=int(n_neighbors),
    )


def _embedded_distances_landmark_gpu(
    embedding: np.ndarray,
    landmarks: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    cp = _cupy()
    embedding_gpu = cp.asarray(embedding)
    distances = cp.empty((landmarks.shape[0], embedding.shape[0]), dtype=embedding_gpu.dtype)
    landmarks_gpu = cp.asarray(landmarks, dtype=cp.int64)
    for start in range(0, landmarks.shape[0], batch_size):
        stop = min(start + batch_size, landmarks.shape[0])
        block = embedding_gpu[landmarks_gpu[start:stop]]
        distances[start:stop] = cp.sqrt(_pairwise_squared_distances_gpu(block, embedding_gpu))
    return cp.asnumpy(distances)


def geodesic_distance_correlation_gpu(
    source: np.ndarray,
    embedding: np.ndarray,
    *,
    n_neighbors: int,
    metric: str | Any,
    metric_params: Optional[dict[str, Any]],
    random_state: Optional[int],
    mode: str,
    n_landmarks: Optional[int],
    landmark_indices: Optional[Any],
    batch_size: int,
) -> GeodesicCorrelationResult:
    _ensure_euclidean_metric(metric, metric_params, metric_name="geodesic_distance_correlation")
    if mode != "landmark":
        raise NotImplementedError(
            "device='gpu' for geodesic_distance_correlation currently supports only "
            "mode='landmark'"
        )

    distances_gpu, indices_gpu = _exact_neighbors_gpu(
        source,
        n_neighbors=n_neighbors,
        batch_size=batch_size,
    )
    distances = _cupy().asnumpy(distances_gpu)
    indices = _cupy().asnumpy(indices_gpu)
    graph = _symmetric_graph_from_neighbor_arrays(
        distances,
        indices,
        n_samples=source.shape[0],
        dtype=source.dtype,
    )
    component_count, _ = connected_components(graph, directed=False)

    landmarks = _resolve_landmark_indices(
        embedding.shape[0],
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
    embedded_distances = _embedded_distances_landmark_gpu(
        embedding,
        landmarks,
        batch_size=max(1, min(batch_size, landmarks.shape[0])),
    )

    row_indices = np.arange(landmarks.shape[0])
    geodesic_distances[row_indices, landmarks] = np.inf
    embedded_distances[row_indices, landmarks] = np.inf
    finite_mask = np.isfinite(geodesic_distances)
    valid_pair_count = landmarks.shape[0] * (embedding.shape[0] - 1)
    finite_pair_fraction = float(np.count_nonzero(finite_mask) / valid_pair_count)
    if not np.any(finite_mask):
        raise ValueError("The landmark pairs are fully disconnected; geodesic correlation cannot be computed")

    return _spearman_result(
        geodesic_distances[finite_mask],
        embedded_distances[finite_mask],
        n_pairs=int(np.count_nonzero(finite_mask)),
        n_neighbors=n_neighbors,
        connected_components_count=int(component_count),
        finite_pair_fraction=finite_pair_fraction,
        mode="landmark",
        n_landmarks=landmarks.shape[0],
    )


def _farthest_point_sample_gpu(points: np.ndarray, max_points: int) -> np.ndarray:
    cp = _cupy()
    n_samples = points.shape[0]
    if max_points >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    if max_points < 2:
        raise ValueError("max_points must be at least 2")

    points_gpu = cp.asarray(points)
    selected = cp.empty(max_points, dtype=cp.int64)
    first = int(cp.argmax(cp.linalg.norm(points_gpu, axis=1)).get())
    selected[0] = first
    min_dist_sq = cp.sum((points_gpu - points_gpu[first]) ** 2, axis=1)
    chosen_mask = cp.zeros(n_samples, dtype=cp.bool_)
    chosen_mask[first] = True

    for index in range(1, max_points):
        masked_distances = cp.where(chosen_mask, -1.0, min_dist_sq)
        next_index = int(cp.argmax(masked_distances).get())
        selected[index] = next_index
        chosen_mask[next_index] = True
        dist_sq = cp.sum((points_gpu - points_gpu[next_index]) ** 2, axis=1)
        min_dist_sq = cp.minimum(min_dist_sq, dist_sq)

    return np.sort(cp.asnumpy(selected))


def _resolve_sampled_indices_gpu(
    points: np.ndarray,
    max_points: Optional[int],
    sampled_indices: Optional[Any],
) -> np.ndarray:
    explicit_indices = _validate_sampled_indices(sampled_indices, points.shape[0])
    if explicit_indices is not None:
        return explicit_indices
    if max_points is None or max_points >= points.shape[0]:
        return np.arange(points.shape[0], dtype=np.int64)
    return _farthest_point_sample_gpu(points, int(max_points))


def _pairwise_distance_matrix_gpu(points: np.ndarray, normalize: bool) -> tuple[np.ndarray, float]:
    cp = _cupy()
    points_gpu = cp.asarray(points)
    distances = cp.sqrt(_pairwise_squared_distances_gpu(points_gpu, points_gpu))
    if not normalize:
        return cp.asnumpy(distances), 1.0

    upper = distances[cp.triu_indices(points.shape[0], k=1)]
    finite = upper[cp.isfinite(upper)]
    positive = finite[finite > 0]
    scale = float(cp.max(positive).get()) if positive.size else 1.0
    if scale <= 0:
        scale = 1.0
    return cp.asnumpy((distances / scale).astype(points_gpu.dtype, copy=False)), scale


def compute_persistence_diagram_gpu(
    point_cloud: np.ndarray,
    *,
    max_homology_dim: int,
    max_points: Optional[int],
    sampled_indices: Optional[Any],
    max_edge_length: Optional[float],
    normalize: bool,
    dtype: np.dtype,
) -> PersistenceDiagram:
    chosen_indices = _resolve_sampled_indices_gpu(
        point_cloud,
        max_points=max_points,
        sampled_indices=sampled_indices,
    )
    sampled_points = point_cloud[chosen_indices]
    distance_matrix, distance_scale = _pairwise_distance_matrix_gpu(sampled_points, normalize=normalize)
    distance_matrix = distance_matrix.astype(dtype, copy=False)

    default_max_edge_length = float(np.max(distance_matrix)) if distance_matrix.size else 0.0
    if max_edge_length is None:
        resolved_max_edge_length = default_max_edge_length
    else:
        resolved_max_edge_length = float(max_edge_length)
        if resolved_max_edge_length <= 0:
            raise ValueError("max_edge_length must be positive when provided")

    simplices = _build_vietoris_rips_complex(
        distance_matrix,
        max_homology_dim=max_homology_dim,
        max_edge_length=resolved_max_edge_length,
    )
    intervals = _reduce_boundary_matrix(simplices, max_homology_dim=max_homology_dim)
    return PersistenceDiagram(
        h0=_intervals_to_array(intervals[0], dtype),
        h1=_intervals_to_array(intervals[1], dtype),
        sampled_indices=chosen_indices,
        max_edge_length=resolved_max_edge_length,
        distance_scale=distance_scale,
        num_input_points=int(point_cloud.shape[0]),
    )


def persistent_homology_distance_gpu(
    source_points: np.ndarray,
    embedding_points: np.ndarray,
    *,
    max_homology_dim: int,
    max_points: Optional[int],
    max_edge_length: Optional[float],
    normalize: bool,
    wasserstein_order: int,
    dtype: np.dtype,
) -> PersistentHomologyComparison:
    sampled_indices = _resolve_sampled_indices_gpu(
        source_points,
        max_points=max_points,
        sampled_indices=None,
    )
    source_diagram = compute_persistence_diagram_gpu(
        source_points,
        max_homology_dim=max_homology_dim,
        max_points=max_points,
        sampled_indices=sampled_indices,
        max_edge_length=max_edge_length,
        normalize=normalize,
        dtype=dtype,
    )
    embedding_diagram = compute_persistence_diagram_gpu(
        embedding_points,
        max_homology_dim=max_homology_dim,
        max_points=max_points,
        sampled_indices=sampled_indices,
        max_edge_length=max_edge_length,
        normalize=normalize,
        dtype=dtype,
    )

    wasserstein_h0, mismatch_h0 = _diagram_wasserstein(
        source_diagram.h0,
        embedding_diagram.h0,
        order=wasserstein_order,
    )
    wasserstein_h1, mismatch_h1 = _diagram_wasserstein(
        source_diagram.h1,
        embedding_diagram.h1,
        order=wasserstein_order,
    )
    return PersistentHomologyComparison(
        source_diagram=source_diagram,
        embedding_diagram=embedding_diagram,
        wasserstein_h0=wasserstein_h0,
        wasserstein_h1=wasserstein_h1,
        wasserstein_total=wasserstein_h0 + wasserstein_h1,
        infinite_interval_mismatch={0: mismatch_h0, 1: mismatch_h1},
    )


def _trajectory_order_from_embedding_gpu(embedding: np.ndarray) -> np.ndarray:
    cp = _cupy()
    embedding_gpu = cp.asarray(embedding)
    centered = embedding_gpu - cp.mean(embedding_gpu, axis=0, keepdims=True)
    _, _, right_vectors = cp.linalg.svd(centered, full_matrices=False)
    principal_axis = right_vectors[0]
    projection = centered @ principal_axis
    return cp.asnumpy(cp.argsort(projection)).astype(np.int64, copy=False)


def _arc_length_from_order_gpu(embedding: np.ndarray, ordering: np.ndarray) -> np.ndarray:
    cp = _cupy()
    embedding_gpu = cp.asarray(embedding)
    ordering_gpu = cp.asarray(ordering, dtype=cp.int64)
    ordered_embedding = embedding_gpu[ordering_gpu]
    step_lengths = cp.linalg.norm(cp.diff(ordered_embedding, axis=0), axis=1)
    cumulative_arc_length = cp.concatenate(
        (
            cp.asarray([0.0], dtype=embedding_gpu.dtype),
            cp.cumsum(step_lengths).astype(embedding_gpu.dtype, copy=False),
        )
    )
    arc_length = cp.empty(embedding.shape[0], dtype=embedding_gpu.dtype)
    arc_length[ordering_gpu] = cumulative_arc_length
    return cp.asnumpy(arc_length)


def pseudotime_correlation_gpu(
    embedding: np.ndarray,
    pseudotime: np.ndarray,
    *,
    method: str,
) -> PseudotimeCorrelationResult:
    ordering = _trajectory_order_from_embedding_gpu(embedding)
    arc_length = _arc_length_from_order_gpu(embedding, ordering)
    correlation, pvalue = _compute_correlation(pseudotime, arc_length, method)

    reversed_ordering = ordering[::-1]
    reversed_arc_length = _arc_length_from_order_gpu(embedding, reversed_ordering)
    reversed_correlation, reversed_pvalue = _compute_correlation(
        pseudotime,
        reversed_arc_length,
        method,
    )
    if reversed_correlation > correlation:
        ordering = reversed_ordering
        arc_length = reversed_arc_length
        correlation = reversed_correlation
        pvalue = reversed_pvalue

    return PseudotimeCorrelationResult(
        correlation=correlation,
        pvalue=pvalue,
        method=method,
        arc_length=arc_length,
        ordering=ordering,
    )
