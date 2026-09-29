# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026, the ibUMAP authors.
#
# Parts of this file are adapted from umap-learn (BSD-3-Clause,
# Copyright (c) 2017, Leland McInnes). See THIRD_PARTY_NOTICES.md.

from __future__ import annotations

import warnings
from time import perf_counter
from warnings import warn

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import (
    _VALID_METRICS as SKLEARN_PAIRWISE_VALID_METRICS,
)

from ._spectral_defaults import resolve_component_ncv, resolve_spectral_defaults


def _add_timing(timing, key, value):
    if timing is not None:
        timing[key] = float(timing.get(key, 0.0) + value)


def _set_diagnostic(diagnostics, key, value):
    if diagnostics is not None:
        diagnostics[key] = value


def _set_diagnostic_default(diagnostics, key, value):
    if diagnostics is not None and key not in diagnostics:
        diagnostics[key] = value


def _random_component_layout(random_state, size, dim, data_range, center):
    if not np.isfinite(data_range) or data_range <= 0.0:
        raise FloatingPointError("component layout has a non-finite or non-positive range")
    if not np.isfinite(center).all():
        raise FloatingPointError("component layout has a non-finite center")
    return (
        random_state.uniform(
            low=-data_range,
            high=data_range,
            size=(size, dim),
        )
        + center
    )


def _umap_metric_helpers():
    try:
        from umap.distances import SPECIAL_METRICS, pairwise_special_metric
    except ImportError:
        SPECIAL_METRICS = {}

        def pairwise_special_metric(*args, **kwargs):
            raise ImportError(
                "umap.distances.pairwise_special_metric is required for this metric"
            )

    try:
        from umap.sparse import SPARSE_SPECIAL_METRICS, sparse_named_distances
    except ImportError:
        SPARSE_SPECIAL_METRICS = {}
        sparse_named_distances = {}

    return (
        SPECIAL_METRICS,
        pairwise_special_metric,
        SPARSE_SPECIAL_METRICS,
        sparse_named_distances,
    )


def component_layout_cpu(
    data,
    n_components,
    component_labels,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds=None,
):
    """CPU copy of umap.spectral.component_layout."""
    metric_kwds = dict(metric_kwds or {})
    if data is None:
        return random_state.random(size=(n_components, dim)) * 10.0

    component_centroids = np.empty((n_components, data.shape[1]), dtype=np.float64)

    if metric == "precomputed":
        distance_matrix = np.zeros((n_components, n_components), dtype=np.float64)
        linkage = metric_kwds.get("linkage", "average")
        if linkage == "average":
            linkage = np.mean
        elif linkage == "complete":
            linkage = np.max
        elif linkage == "single":
            linkage = np.min
        else:
            raise ValueError(
                "Unrecognized linkage '%s'. Please choose from "
                "'average', 'complete', or 'single'" % linkage
            )
        for c_i in range(n_components):
            dm_i = data[component_labels == c_i]
            for c_j in range(c_i + 1, n_components):
                dist = linkage(dm_i[:, component_labels == c_j])
                distance_matrix[c_i, c_j] = dist
                distance_matrix[c_j, c_i] = dist
    else:
        for label in range(n_components):
            component_centroids[label] = data[component_labels == label].mean(axis=0)

        if scipy.sparse.isspmatrix(component_centroids):
            warn(
                "Forcing component centroids to dense; if you are running out of "
                "memory then consider increasing n_neighbors."
            )
            component_centroids = component_centroids.toarray()

        (
            special_metrics,
            pairwise_special_metric,
            sparse_special_metrics,
            sparse_named_distances,
        ) = _umap_metric_helpers()

        if metric in special_metrics:
            distance_matrix = pairwise_special_metric(
                component_centroids,
                metric=metric,
                kwds=metric_kwds,
            )
        elif metric in sparse_special_metrics:
            distance_matrix = pairwise_special_metric(
                component_centroids,
                metric=sparse_special_metrics[metric],
                kwds=metric_kwds,
            )
        else:
            if callable(metric) and scipy.sparse.isspmatrix(data):
                function_to_name_mapping = {
                    sparse_named_distances[k]: k
                    for k in set(SKLEARN_PAIRWISE_VALID_METRICS)
                    & set(sparse_named_distances.keys())
                }
                try:
                    metric_name = function_to_name_mapping[metric]
                except KeyError as exc:
                    raise NotImplementedError(
                        "Multicomponent layout for custom sparse metrics is "
                        "not implemented at this time."
                    ) from exc
                distance_matrix = pairwise_distances(
                    component_centroids, metric=metric_name, **metric_kwds
                )
            else:
                distance_matrix = pairwise_distances(
                    component_centroids, metric=metric, **metric_kwds
                )

    affinity_matrix = np.exp(-(distance_matrix**2))
    component_embedding = SpectralEmbedding(
        n_components=dim,
        affinity="precomputed",
        random_state=random_state,
    ).fit_transform(affinity_matrix)
    component_embedding /= component_embedding.max()
    return component_embedding


def multi_component_layout_cpu(
    data,
    graph,
    n_components,
    component_labels,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds=None,
    init="random",
    method=None,
    tol=None,
    maxiter=None,
    ncv=None,
    auto_defaults=False,
    timing=None,
    diagnostics=None,
):
    """CPU copy of umap.spectral.multi_component_layout with tunable solver args."""
    metric_kwds = dict(metric_kwds or {})
    result = np.empty((graph.shape[0], dim), dtype=np.float32)

    if n_components > 2 * dim:
        t0 = perf_counter()
        meta_embedding = component_layout_cpu(
            data,
            n_components,
            component_labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
        )
        _add_timing(timing, "init_component_layout_time", perf_counter() - t0)
    else:
        t0 = perf_counter()
        k = int(np.ceil(n_components / 2.0))
        base = np.hstack([np.eye(k), np.zeros((k, dim - k))])
        meta_embedding = np.vstack([base, -base])[:n_components]
        _add_timing(timing, "init_component_layout_time", perf_counter() - t0)

    for label in range(n_components):
        t0 = perf_counter()
        mask = component_labels == label
        component_graph = graph.tocsr()[mask, :].tocsc()
        component_graph = component_graph[:, mask].tocoo()
        _add_timing(timing, "init_component_subgraph_time", perf_counter() - t0)

        t0 = perf_counter()
        distances = pairwise_distances([meta_embedding[label]], meta_embedding)
        data_range = distances[distances > 0.0].min() / 2.0
        _add_timing(timing, "init_component_range_time", perf_counter() - t0)

        k = dim + 1
        if component_graph.shape[0] <= k + 1:
            t0 = perf_counter()
            result[mask] = _random_component_layout(
                random_state,
                component_graph.shape[0],
                dim,
                data_range,
                meta_embedding[label],
            )
            _add_timing(timing, "init_component_random_time", perf_counter() - t0)
        else:
            component_embedding = _spectral_layout_cpu(
                data=None,
                graph=component_graph,
                dim=dim,
                random_state=random_state,
                metric=metric,
                metric_kwds=metric_kwds,
                init=init,
                method=method,
                tol=tol,
                maxiter=maxiter,
                ncv=ncv,
                auto_defaults=auto_defaults,
                timing=timing,
                diagnostics=diagnostics,
            )
            t0 = perf_counter()
            scale_denominator = np.max(np.abs(component_embedding))
            scaling_is_valid = (
                np.isfinite(component_embedding).all()
                and np.isfinite(scale_denominator)
                and scale_denominator > 0.0
                and np.isfinite(data_range)
                and data_range > 0.0
            )
            if scaling_is_valid:
                component_embedding *= data_range / scale_denominator
                scaling_is_valid = np.isfinite(component_embedding).all()
            if not scaling_is_valid:
                warn(
                    "Component spectral scaling produced non-finite values; "
                    "falling back to random component layout."
                )
                _set_diagnostic(diagnostics, "spectral_fallback", True)
                component_embedding = _random_component_layout(
                    random_state,
                    component_graph.shape[0],
                    dim,
                    data_range,
                    np.zeros(dim, dtype=np.float64),
                )
            result[mask] = component_embedding + meta_embedding[label]
            if not np.isfinite(result[mask]).all():
                raise FloatingPointError(
                    f"component {label} layout is non-finite after scaling"
                )
            _add_timing(timing, "init_component_postprocess_time", perf_counter() - t0)

    return result


def _spectral_layout_cpu(
    data,
    graph,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds=None,
    init="random",
    method=None,
    tol=None,
    maxiter=None,
    ncv=None,
    auto_defaults=False,
    timing=None,
    diagnostics=None,
):
    """CPU copy of umap.spectral._spectral_layout with CUDA-style diagnostics."""
    metric_kwds = dict(metric_kwds or {})
    t0 = perf_counter()
    n_components, labels = scipy.sparse.csgraph.connected_components(graph)
    _add_timing(timing, "init_connected_components_time", perf_counter() - t0)
    if diagnostics is not None and "spectral_connected_components" not in diagnostics:
        diagnostics["spectral_connected_components"] = int(n_components)

    if n_components > 1:
        return multi_component_layout_cpu(
            data,
            graph,
            n_components,
            labels,
            dim,
            random_state,
            metric=metric,
            metric_kwds=metric_kwds,
            init=init,
            method=method,
            tol=tol,
            maxiter=maxiter,
            ncv=ncv,
            auto_defaults=auto_defaults,
            timing=timing,
            diagnostics=diagnostics,
        )

    component_n = int(graph.shape[0])
    k = dim + 1
    gen = (
        random_state
        if isinstance(random_state, (np.random.Generator, np.random.RandomState))
        else np.random.default_rng(seed=random_state)
    )
    if component_n <= k + 1:
        _set_diagnostic(diagnostics, "spectral_small_component_fallback", True)
        t0 = perf_counter()
        result = gen.uniform(low=-10.0, high=10.0, size=(component_n, dim))
        _add_timing(timing, "init_component_random_time", perf_counter() - t0)
        return result

    if auto_defaults:
        resolved = resolve_spectral_defaults(
            n_samples=component_n,
            n_components=dim,
        )
        method = resolved.method
        tol = resolved.tol
        maxiter = resolved.maxiter
        _set_diagnostic(diagnostics, "spectral_auto_defaults_policy", resolved.policy)

    t0 = perf_counter()
    sqrt_deg = np.sqrt(np.asarray(graph.sum(axis=0)).squeeze())
    identity = scipy.sparse.identity(graph.shape[0], dtype=np.float64)
    inv_sqrt_deg = scipy.sparse.spdiags(
        1.0 / sqrt_deg,
        0,
        graph.shape[0],
        graph.shape[0],
    )
    laplacian = identity - inv_sqrt_deg * graph * inv_sqrt_deg
    if not scipy.sparse.issparse(laplacian):
        laplacian = np.asarray(laplacian)
    _add_timing(timing, "init_laplacian_time", perf_counter() - t0)

    num_lanczos_vectors = resolve_component_ncv(
        component_n=component_n,
        k=k,
        user_ncv=ncv,
    )
    method_value = method or ("eigsh" if laplacian.shape[0] < 2000000 else "lobpcg")
    tol_value = 1e-4 if tol is None else float(tol)
    maxiter_value = graph.shape[0] * 5 if maxiter is None else int(maxiter)
    _set_diagnostic(diagnostics, "spectral_result_device", "cpu")
    _set_diagnostic(diagnostics, "spectral_method", method_value)
    _set_diagnostic(diagnostics, "spectral_tol", tol_value)
    _set_diagnostic(diagnostics, "spectral_maxiter", maxiter_value)
    _set_diagnostic(diagnostics, "spectral_ncv", int(num_lanczos_vectors))
    _set_diagnostic_default(diagnostics, "spectral_fallback", False)
    _set_diagnostic_default(diagnostics, "spectral_exception_type", None)
    _set_diagnostic_default(diagnostics, "spectral_exception_message", None)
    _set_diagnostic_default(diagnostics, "spectral_exceptions", [])

    try:
        t0 = perf_counter()
        if init == "random":
            initial_guess = gen.normal(size=(laplacian.shape[0], k))
        elif init == "tsvd":
            initial_guess = TruncatedSVD(
                n_components=k,
                random_state=random_state,
            ).fit_transform(laplacian)
        else:
            raise ValueError(
                "The init parameter must be either 'random' or 'tsvd': "
                f"{init} is invalid."
            )
        initial_guess[:, 0] = sqrt_deg / np.linalg.norm(sqrt_deg)
        _add_timing(timing, "init_solver_setup_time", perf_counter() - t0)

        t0 = perf_counter()
        if method_value == "eigsh":
            eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
                laplacian,
                k,
                which="SM",
                ncv=num_lanczos_vectors,
                tol=tol_value,
                v0=np.ones(laplacian.shape[0]),
                maxiter=maxiter_value,
            )
        elif method_value == "lobpcg":
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    category=UserWarning,
                    message=r"(?ms).*not reaching the requested tolerance",
                    action="error",
                )
                eigenvalues, eigenvectors = scipy.sparse.linalg.lobpcg(
                    laplacian,
                    np.asarray(initial_guess),
                    largest=False,
                    tol=tol_value,
                    maxiter=maxiter_value,
                )
        else:
            raise ValueError("Method should either be None, 'eigsh' or 'lobpcg'")
        _add_timing(timing, "init_eigensolver_time", perf_counter() - t0)

        if not np.isfinite(eigenvalues).all() or not np.isfinite(eigenvectors).all():
            raise FloatingPointError(
                f"spectral eigensolver returned non-finite values for component size {component_n}"
            )

        t0 = perf_counter()
        order = np.argsort(eigenvalues)[1:k]
        result = eigenvectors[:, order]
        if not np.isfinite(result).all():
            raise FloatingPointError(
                f"spectral component result is non-finite for component size {component_n}"
            )
        _add_timing(timing, "init_postprocess_time", perf_counter() - t0)
        return result
    except (
        scipy.sparse.linalg.ArpackError,
        FloatingPointError,
        RuntimeError,
        UserWarning,
    ) as exc:
        _set_diagnostic(diagnostics, "spectral_fallback", True)
        if diagnostics is not None:
            diagnostics.setdefault("spectral_exceptions", []).append(
                {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "component_size": int(graph.shape[0]),
                }
            )
            if diagnostics.get("spectral_exception_type") is None:
                diagnostics["spectral_exception_type"] = type(exc).__name__
                diagnostics["spectral_exception_message"] = str(exc)
        warn(
            "Spectral initialisation failed! The eigenvector solver\n"
            "failed. This is likely due to too small an eigengap. Consider\n"
            "adding some noise or jitter to your data.\n\n"
            "Falling back to random initialisation!"
        )
        t0 = perf_counter()
        result = gen.uniform(low=-10.0, high=10.0, size=(graph.shape[0], dim))
        _add_timing(timing, "init_fallback_time", perf_counter() - t0)
        return result


def spectral_layout_cpu(
    data,
    graph,
    dim,
    random_state,
    metric="euclidean",
    metric_kwds=None,
    method=None,
    tol=None,
    maxiter=None,
    ncv=None,
    auto_defaults=False,
    timing=None,
    diagnostics=None,
):
    """Compute CPU spectral layout without calling umap.spectral.spectral_layout."""
    graph = scipy.sparse.csr_matrix(graph, dtype=np.float64)
    _set_diagnostic(diagnostics, "spectral_result_device", "cpu")
    return _spectral_layout_cpu(
        data=data,
        graph=graph,
        dim=dim,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwds,
        init="random",
        method=method,
        tol=tol,
        maxiter=maxiter,
        ncv=ncv,
        auto_defaults=auto_defaults,
        timing=timing,
        diagnostics=diagnostics,
    )
