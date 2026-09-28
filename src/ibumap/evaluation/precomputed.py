from __future__ import annotations

import time
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from ._dtype import resolve_evaluation_dtype
from ._device import synchronize_gpu_if_loaded
from .geodesic import (
    GeodesicEmbeddingState,
    GeodesicMode,
    GeodesicSourceState,
    compute_geodesic_embedding_state,
    compute_geodesic_source_state,
    geodesic_distance_correlation,
    geodesic_distance_correlation_from_state,
)
from .global_structure import (
    DistanceSpearmanSourceState,
    RandomTripletSourceState,
    compute_distance_spearman_source_state,
    compute_random_triplet_source_state,
    distance_spearman_correlation,
    distance_spearman_correlation_from_state,
    random_triplet_accuracy,
    random_triplet_accuracy_from_state,
)
from .neighborhood import (
    NeighborSearchResult,
    compute_neighbors,
    continuity,
    continuity_from_source_neighbors,
    neighborhood_preservation,
    neighborhood_preservation_from_neighbors,
    trustworthiness,
    trustworthiness_from_embedding_neighbors,
)
from .persistent_homology import (
    PersistenceDiagram,
    compute_persistence_diagram,
    persistent_homology_distance,
    persistent_homology_distance_from_diagrams,
)


@dataclass(frozen=True)
class SourceEvaluationState:
    points: np.ndarray
    n_neighbors: int
    metric: Any
    metric_params: Optional[dict[str, Any]]
    neighbors: Optional[NeighborSearchResult] = None
    geodesic: Optional[GeodesicSourceState] = None
    persistence: Optional[PersistenceDiagram] = None
    rta: Optional[RandomTripletSourceState] = None
    distance_spearman: Optional[DistanceSpearmanSourceState] = None


@dataclass(frozen=True)
class EmbeddingEvaluationState:
    points: np.ndarray
    neighbors: Optional[NeighborSearchResult] = None
    geodesic: Optional[GeodesicEmbeddingState] = None
    persistence: Optional[PersistenceDiagram] = None


EvaluationMetricName = str
EvaluationTimings = MutableMapping[str, float]


@contextmanager
def _timed_stage(timings: Optional[EvaluationTimings], name: str) -> Iterator[None]:
    """Record a synchronized wall-clock duration when profiling is enabled."""
    if timings is None:
        yield
        return

    synchronize_gpu_if_loaded()
    started = time.perf_counter()
    try:
        yield
    finally:
        synchronize_gpu_if_loaded()
        timings[name] = float(time.perf_counter() - started)


def _validate_points(points: Any, name: str, dtype: np.dtype) -> np.ndarray:
    array = np.asarray(points, dtype=dtype)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def prepare_source_state(
    points: Any,
    *,
    n_neighbors: int = 15,
    metric: str | Any = "euclidean",
    metric_params: Optional[dict[str, Any]] = None,
    compute_neighbor_state: bool = True,
    compute_geodesic_state: bool = False,
    geodesic_mode: GeodesicMode = "landmark",
    geodesic_n_neighbors: Optional[int] = None,
    geodesic_n_landmarks: Optional[int] = None,
    geodesic_landmark_indices: Optional[Any] = None,
    geodesic_random_state: Optional[int] = None,
    geodesic_approx_knn_n_jobs: int = -1,
    compute_persistence_state: bool = False,
    persistence_max_homology_dim: int = 1,
    persistence_max_points: Optional[int] = 128,
    persistence_sampled_indices: Optional[Any] = None,
    persistence_max_edge_length: Optional[float] = None,
    persistence_normalize: bool = True,
    compute_rta_state: bool = False,
    rta_n_triplets: int = 100_000,
    rta_random_state: Optional[int] = None,
    rta_sampled_triplets: Optional[Any] = None,
    rta_distance_batch_size: int = 65_536,
    compute_distance_spearman_state: bool = False,
    distance_spearman_n_pairs: Optional[int] = 100_000,
    distance_spearman_random_state: Optional[int] = None,
    distance_spearman_sampled_pairs: Optional[Any] = None,
    distance_spearman_distance_batch_size: int = 65_536,
    dtype: Any = np.float32,
    device: str = "cpu",
    gpu_batch_size: Optional[int] = None,
    timings: Optional[EvaluationTimings] = None,
) -> SourceEvaluationState:
    """Prepare reusable source-side state, optionally recording component timings."""
    resolved_dtype = resolve_evaluation_dtype(dtype)
    source = _validate_points(points, "points", resolved_dtype)

    neighbors = None
    if compute_neighbor_state:
        with _timed_stage(timings, "neighbors_seconds"):
            neighbors = compute_neighbors(
                source,
                n_neighbors=n_neighbors,
                metric=metric,
                metric_params=metric_params,
                return_distances=False,
                dtype=resolved_dtype,
                device=device,
                gpu_batch_size=gpu_batch_size,
            )

    geodesic = None
    if compute_geodesic_state:
        with _timed_stage(timings, "geodesic_seconds"):
            geodesic = compute_geodesic_source_state(
                source,
                n_neighbors=n_neighbors if geodesic_n_neighbors is None else geodesic_n_neighbors,
                metric=metric,
                metric_params=metric_params,
                mode=geodesic_mode,
                n_landmarks=geodesic_n_landmarks,
                landmark_indices=geodesic_landmark_indices,
                random_state=geodesic_random_state,
                approx_knn_n_jobs=geodesic_approx_knn_n_jobs,
                dtype=resolved_dtype,
                device=device,
                gpu_batch_size=gpu_batch_size,
            )

    persistence = None
    if compute_persistence_state:
        with _timed_stage(timings, "persistence_seconds"):
            persistence = compute_persistence_diagram(
                source,
                max_homology_dim=persistence_max_homology_dim,
                max_points=persistence_max_points,
                sampled_indices=persistence_sampled_indices,
                max_edge_length=persistence_max_edge_length,
                normalize=persistence_normalize,
                dtype=resolved_dtype,
                device=device,
            )

    rta = None
    if compute_rta_state:
        with _timed_stage(timings, "rta_seconds"):
            rta = compute_random_triplet_source_state(
                source,
                n_triplets=rta_n_triplets,
                random_state=rta_random_state,
                sampled_triplets=rta_sampled_triplets,
                metric=metric,
                metric_params=metric_params,
                distance_batch_size=rta_distance_batch_size,
                dtype=resolved_dtype,
            )

    distance_spearman = None
    if compute_distance_spearman_state:
        with _timed_stage(timings, "distance_spearman_seconds"):
            distance_spearman = compute_distance_spearman_source_state(
                source,
                n_pairs=distance_spearman_n_pairs,
                random_state=distance_spearman_random_state,
                sampled_pairs=distance_spearman_sampled_pairs,
                metric=metric,
                metric_params=metric_params,
                distance_batch_size=distance_spearman_distance_batch_size,
                dtype=resolved_dtype,
            )

    return SourceEvaluationState(
        points=source,
        n_neighbors=int(n_neighbors),
        metric=metric,
        metric_params=metric_params,
        neighbors=neighbors,
        geodesic=geodesic,
        persistence=persistence,
        rta=rta,
        distance_spearman=distance_spearman,
    )


def prepare_embedding_state(
    embedding: Any,
    source_state: SourceEvaluationState,
    *,
    compute_neighbor_state: bool = True,
    compute_geodesic_state: bool = True,
    compute_persistence_state: bool = False,
    persistence_max_homology_dim: int = 1,
    persistence_max_points: Optional[int] = 128,
    persistence_max_edge_length: Optional[float] = None,
    persistence_normalize: bool = True,
    dtype: Any = np.float32,
    device: str = "cpu",
    gpu_batch_size: Optional[int] = None,
    timings: Optional[EvaluationTimings] = None,
) -> EmbeddingEvaluationState:
    """Prepare reusable embedding state, optionally recording component timings."""
    resolved_dtype = resolve_evaluation_dtype(dtype)
    target = _validate_points(embedding, "embedding", resolved_dtype)
    if target.shape[0] != source_state.points.shape[0]:
        raise ValueError("embedding and source_state must contain the same number of samples")

    neighbors = None
    if compute_neighbor_state:
        with _timed_stage(timings, "neighbors_seconds"):
            neighbors = compute_neighbors(
                target,
                n_neighbors=source_state.n_neighbors,
                metric="euclidean",
                metric_params=None,
                return_distances=False,
                dtype=resolved_dtype,
                device=device,
                gpu_batch_size=gpu_batch_size,
            )

    geodesic = None
    if compute_geodesic_state and source_state.geodesic is not None:
        with _timed_stage(timings, "geodesic_seconds"):
            geodesic = compute_geodesic_embedding_state(
                target,
                source_state.geodesic,
                dtype=resolved_dtype,
                device=device,
                gpu_batch_size=gpu_batch_size,
            )

    persistence = None
    if compute_persistence_state:
        sampled_indices = (
            source_state.persistence.sampled_indices
            if source_state.persistence is not None
            else None
        )
        with _timed_stage(timings, "persistence_seconds"):
            persistence = compute_persistence_diagram(
                target,
                max_homology_dim=persistence_max_homology_dim,
                max_points=persistence_max_points,
                sampled_indices=sampled_indices,
                max_edge_length=persistence_max_edge_length,
                normalize=persistence_normalize,
                dtype=resolved_dtype,
                device=device,
            )

    return EmbeddingEvaluationState(
        points=target,
        neighbors=neighbors,
        geodesic=geodesic,
        persistence=persistence,
    )


def evaluate_embedding_from_states(
    source_state: SourceEvaluationState,
    embedding_state: EmbeddingEvaluationState,
    *,
    metrics: Iterable[EvaluationMetricName] = (
        "trustworthiness",
        "continuity",
        "neighborhood",
        "geodesic",
        "persistent_homology",
    ),
    rank_low_memory: bool = True,
    rank_low_memory_batch_size: int = 256,
    trustworthiness_n_neighbors: Optional[int] = None,
    continuity_n_neighbors: Optional[int] = None,
    neighborhood_n_neighbors: Optional[int] = None,
    geodesic_pair_sample_size: Optional[int] = None,
    geodesic_random_state: Optional[int] = None,
    geodesic_shortest_path_batch_size: int = 256,
    persistence_wasserstein_order: int = 1,
    rta_n_triplets: int = 100_000,
    rta_random_state: Optional[int] = None,
    rta_distance_batch_size: int = 65_536,
    distance_spearman_n_pairs: Optional[int] = 100_000,
    distance_spearman_random_state: Optional[int] = None,
    distance_spearman_distance_batch_size: int = 65_536,
    dtype: Any = np.float32,
    device: str = "cpu",
    gpu_batch_size: Optional[int] = None,
    timings: Optional[EvaluationTimings] = None,
) -> dict[str, Any]:
    """Evaluate metrics from prepared states, optionally recording metric timings."""
    if source_state.points.shape[0] != embedding_state.points.shape[0]:
        raise ValueError("source_state and embedding_state must contain the same number of samples")
    metric_set = set(metrics)
    scores: dict[str, Any] = {}

    if "trustworthiness" in metric_set:
        with _timed_stage(timings, "trustworthiness_seconds"):
            k = source_state.n_neighbors if trustworthiness_n_neighbors is None else trustworthiness_n_neighbors
            if embedding_state.neighbors is not None:
                scores["trustworthiness"] = trustworthiness_from_embedding_neighbors(
                    source_state.points,
                    embedding_state.neighbors.indices,
                    n_neighbors=k,
                    metric=source_state.metric,
                    metric_params=source_state.metric_params,
                    low_memory_batch_size=rank_low_memory_batch_size,
                    dtype=dtype,
                    device=device,
                    gpu_batch_size=gpu_batch_size,
                )
            else:
                scores["trustworthiness"] = trustworthiness(
                    source_state.points,
                    embedding_state.points,
                    n_neighbors=k,
                    metric=source_state.metric,
                    metric_params=source_state.metric_params,
                    low_memory=rank_low_memory,
                    low_memory_batch_size=rank_low_memory_batch_size,
                    dtype=dtype,
                    device=device,
                    gpu_batch_size=gpu_batch_size,
                )

    if "continuity" in metric_set:
        with _timed_stage(timings, "continuity_seconds"):
            k = source_state.n_neighbors if continuity_n_neighbors is None else continuity_n_neighbors
            if source_state.neighbors is not None:
                scores["continuity"] = continuity_from_source_neighbors(
                    embedding_state.points,
                    source_state.neighbors.indices,
                    n_neighbors=k,
                    low_memory_batch_size=rank_low_memory_batch_size,
                    dtype=dtype,
                    device=device,
                    gpu_batch_size=gpu_batch_size,
                )
            else:
                scores["continuity"] = continuity(
                    source_state.points,
                    embedding_state.points,
                    n_neighbors=k,
                    metric=source_state.metric,
                    metric_params=source_state.metric_params,
                    low_memory=rank_low_memory,
                    low_memory_batch_size=rank_low_memory_batch_size,
                    dtype=dtype,
                    device=device,
                    gpu_batch_size=gpu_batch_size,
                )

    if "neighborhood" in metric_set or "neighborhood_preservation" in metric_set:
        with _timed_stage(timings, "neighborhood_preservation_seconds"):
            k = source_state.n_neighbors if neighborhood_n_neighbors is None else neighborhood_n_neighbors
            if source_state.neighbors is not None and embedding_state.neighbors is not None:
                scores["neighborhood_preservation"] = neighborhood_preservation_from_neighbors(
                    source_state.neighbors.indices[:, :k],
                    embedding_state.neighbors.indices[:, :k],
                )
            else:
                scores["neighborhood_preservation"] = neighborhood_preservation(
                    source_state.points,
                    embedding_state.points,
                    n_neighbors=k,
                    metric=source_state.metric,
                    metric_params=source_state.metric_params,
                    dtype=dtype,
                    device=device,
                    gpu_batch_size=gpu_batch_size,
                )

    if "geodesic" in metric_set:
        with _timed_stage(timings, "geodesic_seconds"):
            if source_state.geodesic is not None:
                scores["geodesic"] = geodesic_distance_correlation_from_state(
                    source_state.geodesic,
                    embedding_state.points,
                    embedding_state=embedding_state.geodesic,
                    pair_sample_size=geodesic_pair_sample_size,
                    random_state=geodesic_random_state,
                    shortest_path_batch_size=geodesic_shortest_path_batch_size,
                    dtype=dtype,
                    device=device,
                    gpu_batch_size=gpu_batch_size,
                )
            else:
                scores["geodesic"] = geodesic_distance_correlation(
                    source_state.points,
                    embedding_state.points,
                    n_neighbors=source_state.n_neighbors,
                    metric=source_state.metric,
                    metric_params=source_state.metric_params,
                    pair_sample_size=geodesic_pair_sample_size,
                    random_state=geodesic_random_state,
                    dtype=dtype,
                    device=device,
                    gpu_batch_size=gpu_batch_size,
                )

    if "persistent_homology" in metric_set:
        with _timed_stage(timings, "persistent_homology_seconds"):
            if source_state.persistence is not None and embedding_state.persistence is not None:
                scores["persistent_homology"] = persistent_homology_distance_from_diagrams(
                    source_state.persistence,
                    embedding_state.persistence,
                    wasserstein_order=persistence_wasserstein_order,
                )
            else:
                scores["persistent_homology"] = persistent_homology_distance(
                    source_state.points,
                    embedding_state.points,
                    wasserstein_order=persistence_wasserstein_order,
                    dtype=dtype,
                    device=device,
                )

    if "rta" in metric_set or "random_triplet_accuracy" in metric_set:
        with _timed_stage(timings, "rta_seconds"):
            if source_state.rta is not None:
                scores["rta"] = random_triplet_accuracy_from_state(
                    source_state.rta,
                    embedding_state.points,
                    distance_batch_size=rta_distance_batch_size,
                    dtype=dtype,
                )
            else:
                scores["rta"] = random_triplet_accuracy(
                    source_state.points,
                    embedding_state.points,
                    n_triplets=rta_n_triplets,
                    random_state=rta_random_state,
                    metric=source_state.metric,
                    metric_params=source_state.metric_params,
                    distance_batch_size=rta_distance_batch_size,
                    dtype=dtype,
                )

    if "distance_spearman" in metric_set:
        with _timed_stage(timings, "distance_spearman_seconds"):
            if source_state.distance_spearman is not None:
                scores["distance_spearman"] = distance_spearman_correlation_from_state(
                    source_state.distance_spearman,
                    embedding_state.points,
                    distance_batch_size=distance_spearman_distance_batch_size,
                    dtype=dtype,
                )
            else:
                scores["distance_spearman"] = distance_spearman_correlation(
                    source_state.points,
                    embedding_state.points,
                    n_pairs=distance_spearman_n_pairs,
                    random_state=distance_spearman_random_state,
                    metric=source_state.metric,
                    metric_params=source_state.metric_params,
                    distance_batch_size=distance_spearman_distance_batch_size,
                    dtype=dtype,
                )

    return scores


# Explicit aliases make the state API read naturally in experiment code.
prepare_source_evaluation_state = prepare_source_state
prepare_embedding_evaluation_state = prepare_embedding_state
