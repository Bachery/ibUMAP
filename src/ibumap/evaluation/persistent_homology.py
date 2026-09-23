from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform

from ._device import resolve_device
from ._dtype import resolve_evaluation_dtype


@dataclass(frozen=True)
class PersistenceDiagram:
    h0: np.ndarray
    h1: np.ndarray
    sampled_indices: np.ndarray
    max_edge_length: float
    distance_scale: float
    num_input_points: int


@dataclass(frozen=True)
class PersistentHomologyComparison:
    source_diagram: PersistenceDiagram
    embedding_diagram: PersistenceDiagram
    wasserstein_h0: float
    wasserstein_h1: float
    wasserstein_total: float
    infinite_interval_mismatch: dict[int, int]


@dataclass(frozen=True)
class _Simplex:
    vertices: tuple[int, ...]
    dimension: int
    filtration: float


def _validate_point_cloud(points: Any, name: str, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(points, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_sampled_indices(sampled_indices: Optional[Any], n_samples: int) -> Optional[np.ndarray]:
    if sampled_indices is None:
        return None
    indices = np.asarray(sampled_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("sampled_indices must be a 1D array when provided")
    if indices.shape[0] < 2:
        raise ValueError("sampled_indices must contain at least two indices")
    if np.any(indices < 0) or np.any(indices >= n_samples):
        raise ValueError("sampled_indices contain out-of-range values")
    if np.unique(indices).shape[0] != indices.shape[0]:
        raise ValueError("sampled_indices must not contain duplicates")
    return np.sort(indices)


def _farthest_point_sample(points: np.ndarray, max_points: int) -> np.ndarray:
    n_samples = points.shape[0]
    if max_points >= n_samples:
        return np.arange(n_samples, dtype=np.int64)
    if max_points < 2:
        raise ValueError("max_points must be at least 2")

    selected = np.empty(max_points, dtype=np.int64)
    selected[0] = int(np.argmax(np.linalg.norm(points, axis=1)))
    min_dist_sq = np.sum((points - points[selected[0]]) ** 2, axis=1)
    chosen_mask = np.zeros(n_samples, dtype=bool)
    chosen_mask[selected[0]] = True

    for idx in range(1, max_points):
        masked_dist = min_dist_sq.copy()
        masked_dist[chosen_mask] = -1.0
        next_index = int(np.argmax(masked_dist))
        selected[idx] = next_index
        chosen_mask[next_index] = True
        dist_sq = np.sum((points - points[next_index]) ** 2, axis=1)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    return np.sort(selected)


def _resolve_sampled_indices(
    points: np.ndarray,
    max_points: Optional[int],
    sampled_indices: Optional[Any],
) -> np.ndarray:
    explicit_indices = _validate_sampled_indices(sampled_indices, points.shape[0])
    if explicit_indices is not None:
        return explicit_indices
    if max_points is None or max_points >= points.shape[0]:
        return np.arange(points.shape[0], dtype=np.int64)
    return _farthest_point_sample(points, int(max_points))


def _pairwise_distance_matrix(points: np.ndarray, normalize: bool) -> tuple[np.ndarray, float]:
    distances = squareform(pdist(points, metric="euclidean")).astype(points.dtype, copy=False)
    if not normalize:
        return distances, 1.0

    upper = distances[np.triu_indices(points.shape[0], k=1)]
    finite = upper[np.isfinite(upper)]
    positive = finite[finite > 0]
    scale = float(np.max(positive)) if positive.size else 1.0
    if scale <= 0:
        scale = 1.0
    return (distances / scale).astype(points.dtype, copy=False), scale


def _build_vietoris_rips_complex(
    distance_matrix: np.ndarray,
    max_homology_dim: int,
    max_edge_length: float,
) -> list[_Simplex]:
    simplices = [_Simplex((vertex,), 0, 0.0) for vertex in range(distance_matrix.shape[0])]

    for i, j in combinations(range(distance_matrix.shape[0]), 2):
        distance = float(distance_matrix[i, j])
        if distance <= max_edge_length:
            simplices.append(_Simplex((i, j), 1, distance))

    if max_homology_dim >= 1:
        for i, j, k in combinations(range(distance_matrix.shape[0]), 3):
            filtration = float(
                max(
                    distance_matrix[i, j],
                    distance_matrix[i, k],
                    distance_matrix[j, k],
                )
            )
            if filtration <= max_edge_length:
                simplices.append(_Simplex((i, j, k), 2, filtration))

    simplices.sort(key=lambda simplex: (simplex.filtration, simplex.dimension, simplex.vertices))
    return simplices


def _reduce_boundary_matrix(
    simplices: list[_Simplex],
    max_homology_dim: int,
) -> dict[int, list[tuple[float, float]]]:
    simplex_to_index = {simplex.vertices: idx for idx, simplex in enumerate(simplices)}
    boundaries: list[set[int]] = []
    for simplex in simplices:
        if simplex.dimension == 0:
            boundaries.append(set())
        else:
            faces = []
            for face_index in range(len(simplex.vertices)):
                face = simplex.vertices[:face_index] + simplex.vertices[face_index + 1 :]
                faces.append(simplex_to_index[face])
            boundaries.append(set(faces))

    reduced_columns: list[set[int]] = [set() for _ in simplices]
    pivot_to_column: dict[int, int] = {}
    birth_to_death: dict[int, int] = {}

    # Standard Z2 column reduction. We only keep simplices up to triangles
    # because H1 persistence only depends on 0/1/2-simplices.
    for column_index, boundary in enumerate(boundaries):
        reduced = set(boundary)
        while reduced:
            pivot = max(reduced)
            if pivot not in pivot_to_column:
                break
            reduced.symmetric_difference_update(reduced_columns[pivot_to_column[pivot]])
        reduced_columns[column_index] = reduced
        if reduced:
            pivot = max(reduced)
            pivot_to_column[pivot] = column_index
            birth_to_death[pivot] = column_index

    intervals = {0: [], 1: []}
    for birth_index, death_index in birth_to_death.items():
        birth_simplex = simplices[birth_index]
        if birth_simplex.dimension <= max_homology_dim:
            intervals[birth_simplex.dimension].append(
                (birth_simplex.filtration, simplices[death_index].filtration)
            )

    paired_births = set(birth_to_death.keys())
    for simplex_index, simplex in enumerate(simplices):
        if simplex.dimension > max_homology_dim:
            continue
        if not reduced_columns[simplex_index] and simplex_index not in paired_births:
            intervals[simplex.dimension].append((simplex.filtration, np.inf))

    for dimension in intervals:
        intervals[dimension].sort(key=lambda interval: (interval[0], interval[1]))
    return intervals


def _intervals_to_array(intervals: list[tuple[float, float]], dtype: np.dtype) -> np.ndarray:
    if not intervals:
        return np.empty((0, 2), dtype=dtype)
    return np.asarray(intervals, dtype=dtype)


def _split_finite_and_infinite(diagram: np.ndarray, min_persistence: float = 1e-12) -> tuple[np.ndarray, int]:
    if diagram.size == 0:
        return diagram.reshape(0, 2), 0
    infinite_mask = ~np.isfinite(diagram[:, 1])
    finite = diagram[~infinite_mask]
    if finite.size:
        finite = finite[(finite[:, 1] - finite[:, 0]) > min_persistence]
    return finite, int(np.count_nonzero(infinite_mask))


def _linf_distance(interval_a: np.ndarray, interval_b: np.ndarray) -> float:
    return float(np.max(np.abs(interval_a - interval_b)))


def _diagonal_costs(diagram: np.ndarray, order: int) -> np.ndarray:
    persistences = np.maximum(0.0, diagram[:, 1] - diagram[:, 0])
    return np.power(0.5 * persistences, order)


def _diagram_wasserstein(diagram_a: np.ndarray, diagram_b: np.ndarray, order: int) -> tuple[float, int]:
    finite_a, infinite_a = _split_finite_and_infinite(diagram_a)
    finite_b, infinite_b = _split_finite_and_infinite(diagram_b)

    n_a = finite_a.shape[0]
    n_b = finite_b.shape[0]
    if n_a == 0 and n_b == 0:
        return 0.0, abs(infinite_a - infinite_b)

    dtype = np.result_type(diagram_a.dtype, diagram_b.dtype, np.float32)
    pairwise_costs = np.zeros((n_a, n_b), dtype=dtype)
    for row in range(n_a):
        for col in range(n_b):
            pairwise_costs[row, col] = _linf_distance(finite_a[row], finite_b[col]) ** order

    diagonal_a = _diagonal_costs(finite_a, order)
    diagonal_b = _diagonal_costs(finite_b, order)
    finite_values = [
        float(np.max(pairwise_costs)) if pairwise_costs.size else 0.0,
        float(np.max(diagonal_a)) if diagonal_a.size else 0.0,
        float(np.max(diagonal_b)) if diagonal_b.size else 0.0,
        1.0,
    ]
    penalty = max(finite_values) * (n_a + n_b + 1.0)

    cost_matrix = np.full((n_a + n_b, n_a + n_b), penalty, dtype=dtype)
    if n_a and n_b:
        cost_matrix[:n_a, :n_b] = pairwise_costs
    for row in range(n_a):
        cost_matrix[row, n_b + row] = diagonal_a[row]
    for col in range(n_b):
        cost_matrix[n_a + col, col] = diagonal_b[col]
    cost_matrix[n_a:, n_b:] = 0.0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return float(np.sum(cost_matrix[row_ind, col_ind]) ** (1.0 / order)), abs(infinite_a - infinite_b)


def compute_persistence_diagram(
    points: Any,
    *,
    max_homology_dim: int = 1,
    max_points: Optional[int] = 128,
    sampled_indices: Optional[Any] = None,
    max_edge_length: Optional[float] = None,
    normalize: bool = False,
    dtype: Any = np.float32,
    device: str = "cpu",
) -> PersistenceDiagram:
    if max_homology_dim not in (0, 1):
        raise ValueError("max_homology_dim must be 0 or 1")

    resolved_dtype = resolve_evaluation_dtype(dtype)
    point_cloud = _validate_point_cloud(points, "points", resolved_dtype)
    resolved_device = resolve_device(device)
    if resolved_device == "gpu":
        from ._gpu import compute_persistence_diagram_gpu

        return compute_persistence_diagram_gpu(
            point_cloud,
            max_homology_dim=max_homology_dim,
            max_points=max_points,
            sampled_indices=sampled_indices,
            max_edge_length=max_edge_length,
            normalize=normalize,
            dtype=resolved_dtype,
        )

    chosen_indices = _resolve_sampled_indices(point_cloud, max_points=max_points, sampled_indices=sampled_indices)
    sampled_points = point_cloud[chosen_indices]
    distance_matrix, distance_scale = _pairwise_distance_matrix(sampled_points, normalize=normalize)

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
        h0=_intervals_to_array(intervals[0], resolved_dtype),
        h1=_intervals_to_array(intervals[1], resolved_dtype),
        sampled_indices=chosen_indices,
        max_edge_length=resolved_max_edge_length,
        distance_scale=distance_scale,
        num_input_points=int(point_cloud.shape[0]),
    )


def betti_numbers_at_scale(diagram: PersistenceDiagram, scale: float) -> dict[int, int]:
    if scale < 0:
        raise ValueError("scale must be non-negative")

    betti_numbers = {}
    for dimension, intervals in ((0, diagram.h0), (1, diagram.h1)):
        if intervals.size == 0:
            betti_numbers[dimension] = 0
            continue
        alive = (intervals[:, 0] <= scale) & (scale < intervals[:, 1])
        betti_numbers[dimension] = int(np.count_nonzero(alive))
    return betti_numbers


def persistent_homology_distance_from_diagrams(
    source_diagram: PersistenceDiagram,
    embedding_diagram: PersistenceDiagram,
    *,
    wasserstein_order: int = 1,
) -> PersistentHomologyComparison:
    """Compare two precomputed persistence diagrams."""
    if wasserstein_order < 1:
        raise ValueError("wasserstein_order must be at least 1")
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


def persistent_homology_distance(
    source_points: Any,
    embedding_points: Any,
    *,
    max_homology_dim: int = 1,
    max_points: Optional[int] = 128,
    max_edge_length: Optional[float] = None,
    normalize: bool = True,
    wasserstein_order: int = 1,
    dtype: Any = np.float32,
    device: str = "cpu",
) -> PersistentHomologyComparison:
    if wasserstein_order < 1:
        raise ValueError("wasserstein_order must be at least 1")

    resolved_dtype = resolve_evaluation_dtype(dtype)
    source_points = _validate_point_cloud(source_points, "source_points", resolved_dtype)
    embedding_points = _validate_point_cloud(embedding_points, "embedding_points", resolved_dtype)
    if source_points.shape[0] != embedding_points.shape[0]:
        raise ValueError("source_points and embedding_points must contain the same number of samples")
    resolved_device = resolve_device(device)

    if resolved_device == "gpu":
        from ._gpu import persistent_homology_distance_gpu

        return persistent_homology_distance_gpu(
            source_points,
            embedding_points,
            max_homology_dim=max_homology_dim,
            max_points=max_points,
            max_edge_length=max_edge_length,
            normalize=normalize,
            wasserstein_order=wasserstein_order,
            dtype=resolved_dtype,
        )

    sampled_indices = _resolve_sampled_indices(source_points, max_points=max_points, sampled_indices=None)
    source_diagram = compute_persistence_diagram(
        source_points,
        max_homology_dim=max_homology_dim,
        max_points=max_points,
        sampled_indices=sampled_indices,
        max_edge_length=max_edge_length,
        normalize=normalize,
        dtype=resolved_dtype,
    )
    embedding_diagram = compute_persistence_diagram(
        embedding_points,
        max_homology_dim=max_homology_dim,
        max_points=max_points,
        sampled_indices=sampled_indices,
        max_edge_length=max_edge_length,
        normalize=normalize,
        dtype=resolved_dtype,
    )

    return persistent_homology_distance_from_diagrams(
        source_diagram,
        embedding_diagram,
        wasserstein_order=wasserstein_order,
    )
