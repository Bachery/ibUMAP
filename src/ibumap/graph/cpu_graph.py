from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.sparse import issparse
from sklearn.utils import check_random_state
from umap.umap_ import fuzzy_simplicial_set, nearest_neighbors

import umap.distances as dist
from pynndescent.distances import named_distances as pynn_named_distances
from pynndescent.sparse import sparse_named_distances as pynn_sparse_named_distances


def build_cpu_graph(
    X,
    n_neighbors: int,
    metric: str,
    metric_kwds: Optional[Dict[str, Any]],
    random_state,
    angular_rp_forest: bool,
    low_memory: bool,
    n_jobs: int,
    verbose: bool,
    set_op_mix_ratio: float,
    local_connectivity: float,
    densmap_or_output_dens: bool,
    input_distance_func=None,
    sparse_data: bool = False,
    preserve_edge_order: bool = False,
) -> Tuple[Any, np.ndarray, np.ndarray, Any]:
    metric_kwds = metric_kwds or {}
    rs = check_random_state(random_state)

    if sparse_data and metric in pynn_sparse_named_distances:
        nn_metric = metric
    elif (not sparse_data) and metric in pynn_named_distances:
        nn_metric = metric
    else:
        nn_metric = input_distance_func if input_distance_func is not None else metric

    knn_indices, knn_dists, knn_search_index = nearest_neighbors(
        X,
        n_neighbors,
        nn_metric,
        metric_kwds,
        angular_rp_forest,
        rs,
        low_memory,
        use_pynndescent=True,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    graph, _, _, _ = fuzzy_simplicial_set(
        X,
        n_neighbors,
        rs,
        nn_metric,
        metric_kwds,
        knn_indices,
        knn_dists,
        angular_rp_forest,
        set_op_mix_ratio,
        local_connectivity,
        True,
        verbose,
        densmap_or_output_dens,
    )
    if not preserve_edge_order:
        graph = graph.tocsr()
    return graph, knn_indices, knn_dists, knn_search_index


def build_cpu_graph_from_knn(
    X,
    n_neighbors: int,
    metric: str,
    metric_kwds: Optional[Dict[str, Any]],
    random_state,
    knn_indices,
    knn_dists,
    angular_rp_forest: bool,
    set_op_mix_ratio: float,
    local_connectivity: float,
    densmap_or_output_dens: bool,
    verbose: bool,
    input_distance_func=None,
    dtype=np.float32,
    preserve_edge_order: bool = False,
):
    """Build a UMAP fuzzy graph from caller-provided nearest neighbors."""
    metric_kwds = metric_kwds or {}
    rs = check_random_state(random_state)

    knn_indices = np.asarray(knn_indices, dtype=np.int64, order="C")
    knn_dists = np.asarray(knn_dists, dtype=dtype, order="C")
    if knn_indices.ndim != 2:
        raise ValueError("knn_indices must be a 2D array")
    if knn_dists.shape != knn_indices.shape:
        raise ValueError("knn_distances must have the same shape as knn_indices")
    if knn_indices.shape[0] != X.shape[0]:
        raise ValueError("knn_indices first dimension must match X")
    if knn_indices.shape[1] != n_neighbors:
        raise ValueError("knn_indices second dimension must match n_neighbors")

    if issparse(X) and metric in pynn_sparse_named_distances:
        nn_metric = metric
    elif (not issparse(X)) and metric in pynn_named_distances:
        nn_metric = metric
    else:
        nn_metric = input_distance_func if input_distance_func is not None else metric

    graph, _, _, _ = fuzzy_simplicial_set(
        X,
        n_neighbors,
        rs,
        nn_metric,
        metric_kwds,
        knn_indices,
        knn_dists,
        angular_rp_forest,
        set_op_mix_ratio,
        local_connectivity,
        True,
        verbose,
        densmap_or_output_dens,
    )
    return graph if preserve_edge_order else graph.tocsr()
