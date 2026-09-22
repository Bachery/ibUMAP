"""Public fixed-input preparation support for :class:`~umap_fft.UMAPFFT`.

The helpers in this module intentionally keep graph construction, graph
preprocessing, initialization, and host/device transfer policy in the package.
Experiment scripts should only decide which returned artifacts to persist.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Optional

import numpy as np
from scipy.sparse import issparse
from sklearn.utils import check_array, check_random_state

from .graph import (
    CPUGraphThreadPolicy,
    build_cpu_graph,
    build_cpu_graph_from_knn,
    resolve_cpu_graph_thread_policy,
)
from .init import initialize_embedding_cpu, initialize_embedding_gpu
from .parameter_usage import _json_safe
from .utils import derive_random_seed, preprocess_graph, preprocess_graph_csr


@dataclass(frozen=True)
class PreparedInputs:
    """Artifacts prepared by :meth:`UMAPFFT.prepare_fixed_inputs`.

    ``fuzzy_graph`` is the unmodified fuzzy simplicial-set graph, appropriate
    for durable fixed-input storage. ``optimizer_graph`` is the graph after the
    exact thresholding/canonicalization consumed by UMAPFFT optimizers.  On
    CUDA, ``host_output=False`` returns CuPy/CuPyX objects; otherwise all
    returned arrays and graphs are host-resident NumPy/SciPy objects. Metal
    preparation deliberately keeps CPU graph/init artifacts on the host.

    A field is ``None`` when its corresponding ``return_*`` option was false.
    """

    knn_indices: Optional[Any]
    knn_distances: Optional[Any]
    fuzzy_graph: Optional[Any]
    optimizer_graph: Optional[Any]
    init_embedding: Optional[Any]
    timings: dict[str, float]
    diagnostics: dict[str, Any]
    effective_config: dict[str, Any]
    device: str
    host_output: bool

    @property
    def graph(self) -> Optional[Any]:
        """Compatibility alias for the persisted raw ``fuzzy_graph``."""
        return self.fuzzy_graph

    @property
    def used_params(self) -> dict[str, Any]:
        """Compatibility view; prefer the richer ``effective_config`` field."""
        return self.effective_config


def _random_seed(model: Any, stream: str) -> Optional[int]:
    root = model.random_state
    if root is None and model.runtime.deterministic:
        root = 0
    return derive_random_seed(root, stream)


def _shared_random_state(model: Any):
    if getattr(model.runtime, "rng_lifecycle", "independent") != "shared_rng":
        return None
    root = model.random_state
    if root is None and model.runtime.deterministic:
        root = 0
    return check_random_state(root)


def _uses_ordered_cpu_umap(model: Any) -> bool:
    return model.runtime.device == "cpu" and model.runtime.algorithm == "umap"


def _build_cpu_graph(
    model: Any,
    X: Any,
    shared_random_state=None,
) -> tuple[np.ndarray, np.ndarray, Any, str, CPUGraphThreadPolicy]:
    graph_thread_policy = resolve_cpu_graph_thread_policy(
        model.n_jobs,
        deterministic=model.runtime.deterministic,
        algorithm=model.runtime.algorithm,
        device=model.runtime.device,
    )
    graph, indices, distances, _ = build_cpu_graph(
        X,
        n_neighbors=model.n_neighbors,
        metric=model.metric,
        metric_kwds=model.metric_kwds,
        random_state=(
            _random_seed(model, "graph")
            if shared_random_state is None
            else shared_random_state
        ),
        angular_rp_forest=model.angular_rp_forest,
        low_memory=model.low_memory,
        n_jobs=graph_thread_policy.effective_n_jobs,
        verbose=model.verbose,
        set_op_mix_ratio=model.set_op_mix_ratio,
        local_connectivity=model.local_connectivity,
        densmap_or_output_dens=False,
        input_distance_func=model.input_distance_func,
        sparse_data=issparse(X),
        preserve_edge_order=_uses_ordered_cpu_umap(model),
    )
    return (
        np.asarray(indices, dtype=np.int64, order="C"),
        np.asarray(distances, dtype=np.float32, order="C"),
        graph if _uses_ordered_cpu_umap(model) else graph.tocsr(),
        "umap_learn.nearest_neighbors",
        graph_thread_policy,
    )


def _normalize_cuda_knn(
    indices: Any,
    distances: Any,
    *,
    n_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a cuML self-query result to UMAP's KNN convention."""

    indices = np.asarray(indices, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float32)
    if indices.ndim != 2 or distances.shape != indices.shape:
        raise ValueError(
            "CUDA nearest-neighbor output must contain matching 2D indices and distances"
        )
    n_samples = indices.shape[0]
    if n_samples < n_neighbors:
        raise ValueError(f"n_neighbors={n_neighbors} exceeds n_samples={n_samples}")

    rows = np.arange(n_samples, dtype=np.int64)
    normalized_indices = np.empty((n_samples, n_neighbors), dtype=np.int64)
    normalized_distances = np.empty((n_samples, n_neighbors), dtype=np.float32)
    normalized_indices[:, 0] = rows
    normalized_distances[:, 0] = 0.0

    positions = np.ones(n_samples, dtype=np.int64)
    for column in range(indices.shape[1]):
        eligible = (indices[:, column] != rows) & (positions < n_neighbors)
        selected_rows = np.flatnonzero(eligible)
        if selected_rows.size == 0:
            continue
        target_columns = positions[selected_rows]
        normalized_indices[selected_rows, target_columns] = indices[selected_rows, column]
        normalized_distances[selected_rows, target_columns] = distances[
            selected_rows, column
        ]
        positions[selected_rows] += 1

    incomplete = np.flatnonzero(positions < n_neighbors)
    if incomplete.size:
        row = int(incomplete[0])
        raise ValueError(
            f"KNN row {row} has only {int(positions[row])} normalized neighbor(s); "
            f"need {n_neighbors}"
        )
    return normalized_indices, normalized_distances


def _build_cuda_knn(model: Any, X: Any) -> tuple[np.ndarray, np.ndarray, str]:
    if issparse(X):
        raise TypeError("CUDA fixed-input KNN currently requires dense feature arrays")
    if model.metric_kwds:
        raise NotImplementedError("CUDA fixed-input KNN does not currently forward metric_kwds")

    try:
        import cupy as cp
        from cuml.neighbors import NearestNeighbors
    except Exception as exc:  # pragma: no cover - optional CUDA stack
        raise RuntimeError(
            "CUDA fixed-input generation requires cupy and cuml.neighbors.NearestNeighbors"
        ) from exc

    query_k = min(int(X.shape[0]), int(model.n_neighbors) + 1)
    X_gpu = cp.asarray(X, dtype=cp.float32)
    nn = NearestNeighbors(
        n_neighbors=query_k,
        metric=model.metric,
        verbose=model.verbose,
    )
    nn.fit(X_gpu)
    distances, indices = nn.kneighbors(X_gpu)
    cp.cuda.Stream.null.synchronize()
    return (
        *_normalize_cuda_knn(
            cp.asnumpy(indices),
            cp.asnumpy(distances),
            n_neighbors=int(model.n_neighbors),
        ),
        "cuml.neighbors.NearestNeighbors",
    )


def _build_graph_from_knn(
    model: Any,
    X: Any,
    knn_indices: Any,
    knn_distances: Any,
    shared_random_state=None,
) -> Any:
    graph = build_cpu_graph_from_knn(
        X,
        n_neighbors=model.n_neighbors,
        metric=model.metric,
        metric_kwds=model.metric_kwds,
        random_state=(
            _random_seed(model, "graph")
            if shared_random_state is None
            else shared_random_state
        ),
        knn_indices=knn_indices,
        knn_dists=knn_distances,
        angular_rp_forest=model.angular_rp_forest,
        set_op_mix_ratio=model.set_op_mix_ratio,
        local_connectivity=model.local_connectivity,
        densmap_or_output_dens=False,
        verbose=model.verbose,
        input_distance_func=model.input_distance_func,
        dtype=model.numeric_dtype,
        preserve_edge_order=_uses_ordered_cpu_umap(model),
    )
    return graph if _uses_ordered_cpu_umap(model) else graph.tocsr()


def _initialize_cpu(
    model: Any,
    X: Any,
    optimizer_graph: Any,
    shared_random_state=None,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    init_cfg = model.config.initialization
    init_timing: dict[str, float] = {}
    init_diagnostics: dict[str, Any] = {}
    embedding = initialize_embedding_cpu(
        X,
        optimizer_graph,
        model.n_components,
        init_cfg.init,
        (
            check_random_state(_random_seed(model, "init"))
            if shared_random_state is None
            else shared_random_state
        ),
        model.metric,
        model.metric_kwds,
        dtype=model.numeric_dtype,
        spectral_method=init_cfg.spectral_method,
        spectral_tol=init_cfg.spectral_tol,
        spectral_maxiter=init_cfg.spectral_maxiter,
        spectral_ncv=init_cfg.spectral_ncv,
        spectral_auto_defaults=init_cfg.spectral_auto_defaults,
        spectral_scale_policy=init_cfg.resolve_spectral_scale_policy(
            model.runtime.algorithm
        ),
        spectral_max_span=init_cfg.spectral_max_span,
        spectral_jitter_relative=init_cfg.spectral_jitter_relative,
        spectral_jitter_max=init_cfg.spectral_jitter_max,
        report_duplicate_ratio=init_cfg.report_duplicate_ratio,
        init_timing=init_timing,
        init_diagnostics=init_diagnostics,
    )
    return np.asarray(embedding, dtype=model.numeric_dtype, order="C"), init_timing, init_diagnostics


def _initialize_cuda(
    model: Any,
    X: Any,
    optimizer_graph: Any,
) -> tuple[Any, dict[str, float], dict[str, Any], Any]:
    if initialize_embedding_gpu is None:
        raise RuntimeError("UMAPFFT CUDA initializer is unavailable")
    try:
        import cupy as cp
        from cupyx.scipy import sparse as cupyx_sparse
    except Exception as exc:  # pragma: no cover - optional CUDA stack
        raise RuntimeError("CUDA fixed-input preparation requires cupy and cupyx") from exc

    init_cfg = model.config.initialization
    init_timing: dict[str, float] = {}
    init_diagnostics: dict[str, Any] = {}
    graph_gpu = cupyx_sparse.csr_matrix(optimizer_graph)
    cp.cuda.Stream.null.synchronize()
    embedding = initialize_embedding_gpu(
        X,
        graph_gpu,
        model.n_components,
        init_cfg.init,
        check_random_state(_random_seed(model, "init")),
        model.metric,
        model.metric_kwds,
        deterministic=model.runtime.deterministic,
        spectral_method=init_cfg.spectral_method,
        spectral_tol=init_cfg.spectral_tol,
        spectral_maxiter=init_cfg.spectral_maxiter,
        spectral_ncv=init_cfg.spectral_ncv,
        spectral_auto_defaults=init_cfg.spectral_auto_defaults,
        spectral_scale_policy=init_cfg.resolve_spectral_scale_policy(
            model.runtime.algorithm
        ),
        spectral_max_span=init_cfg.spectral_max_span,
        spectral_jitter_relative=init_cfg.spectral_jitter_relative,
        spectral_jitter_max=init_cfg.spectral_jitter_max,
        report_duplicate_ratio=init_cfg.report_duplicate_ratio,
        init_timing=init_timing,
        init_diagnostics=init_diagnostics,
    )
    cp.cuda.Stream.null.synchronize()
    return embedding, init_timing, init_diagnostics, graph_gpu


def _cuda_outputs(
    *,
    host_output: bool,
    knn_indices: Optional[np.ndarray],
    knn_distances: Optional[np.ndarray],
    fuzzy_graph: Optional[Any],
    optimizer_graph: Optional[Any],
    init_embedding: Optional[Any],
) -> tuple[Optional[Any], Optional[Any], Optional[Any], Optional[Any], Optional[Any]]:
    if host_output:
        import cupy as cp

        def to_host_graph(graph: Optional[Any]) -> Optional[Any]:
            if graph is None:
                return None
            return graph.get() if hasattr(graph, "get") else graph

        return (
            knn_indices,
            knn_distances,
            to_host_graph(fuzzy_graph),
            to_host_graph(optimizer_graph),
            None if init_embedding is None else cp.asnumpy(init_embedding),
        )

    import cupy as cp
    from cupyx.scipy import sparse as cupyx_sparse

    return (
        None if knn_indices is None else cp.asarray(knn_indices),
        None if knn_distances is None else cp.asarray(knn_distances),
        None if fuzzy_graph is None else cupyx_sparse.csr_matrix(fuzzy_graph),
        None
        if optimizer_graph is None
        else cupyx_sparse.csr_matrix(optimizer_graph),
        init_embedding,
    )


def _effective_config(model: Any, *, device: str, host_output: bool) -> dict[str, Any]:
    config = _json_safe(model.config)
    config["preparation"] = {
        "device": device,
        "host_output": bool(host_output),
        "raw_graph": "fuzzy_simplicial_set",
        "optimizer_graph": "preprocess_graph_csr",
    }
    return config


def _prepare(
    model: Any,
    X: Any,
    *,
    source: str,
    knn_indices: Optional[Any],
    knn_distances: Optional[Any],
    return_knn: bool,
    return_raw_graph: bool,
    return_optimizer_graph: bool,
    return_init: bool,
    host_output: bool,
) -> PreparedInputs:
    """Shared implementation for package-owned fixed-input preparation."""

    if model.runtime.device == "metal" and not host_output:
        raise NotImplementedError(
            "device='metal' fixed-input preparation preserves host NumPy/SciPy "
            "artifacts; host_output=False is not supported"
        )

    total_t0 = perf_counter()
    timings: dict[str, float] = {}
    diagnostics: dict[str, Any] = {
        "source": source,
        "device": model.runtime.device,
        "host_output": bool(host_output),
    }

    validation_t0 = perf_counter()
    X = check_array(X, dtype=model.numeric_dtype, accept_sparse="csr", order="C")
    timings["input_validation_time"] = perf_counter() - validation_t0
    shared_random_state = _shared_random_state(model)

    graph_t0 = perf_counter()
    if source == "knn":
        if model.runtime.device == "cuda":
            indices, distances, knn_backend = _build_cuda_knn(model, X)
            raw_graph = _build_graph_from_knn(
                model,
                X,
                indices,
                distances,
                shared_random_state,
            )
            diagnostics["knn_backend"] = knn_backend
            diagnostics["fuzzy_graph_backend"] = "umap_learn.fuzzy_simplicial_set"
        else:
            (
                indices,
                distances,
                raw_graph,
                knn_backend,
                graph_thread_policy,
            ) = _build_cpu_graph(
                model,
                X,
                shared_random_state,
            )
            diagnostics["knn_backend"] = knn_backend
            diagnostics["fuzzy_graph_backend"] = "umap_learn.fuzzy_simplicial_set"
            diagnostics["cpu_graph_thread_policy"] = (
                graph_thread_policy.diagnostics()
            )
    elif source == "fixed_knn":
        if knn_indices is None or knn_distances is None:
            raise ValueError("knn_indices and knn_distances are required")
        indices = np.asarray(knn_indices, dtype=np.int64, order="C")
        distances = np.asarray(knn_distances, dtype=np.float32, order="C")
        raw_graph = _build_graph_from_knn(
            model,
            X,
            indices,
            distances,
            shared_random_state,
        )
        diagnostics["knn_backend"] = "caller_provided"
        diagnostics["fuzzy_graph_backend"] = "umap_learn.fuzzy_simplicial_set"
    else:  # pragma: no cover - internal call contract
        raise ValueError(f"unknown fixed-input source: {source}")
    timings["knn_and_fuzzy_graph_time"] = perf_counter() - graph_t0
    diagnostics["fuzzy_graph"] = {
        "shape": [int(value) for value in raw_graph.shape],
        "nnz": int(raw_graph.nnz),
        "dtype": str(raw_graph.dtype),
    }

    optimizer_graph = None
    optimizer_graph_cpu = None
    initialization_graph_cpu = None
    if return_optimizer_graph or return_init:
        preprocess_t0 = perf_counter()
        if _uses_ordered_cpu_umap(model):
            ordered_graph, resolved_epochs, preprocess_stats = preprocess_graph(
                raw_graph,
                model.n_epochs,
                return_stats=True,
            )
            initialization_graph_cpu = ordered_graph
            optimizer_graph_cpu = ordered_graph.tocsr()
        else:
            optimizer_graph_cpu, resolved_epochs, _, preprocess_stats = preprocess_graph_csr(
                raw_graph,
                model.n_epochs,
                dtype=np.float32 if model.runtime.device == "cuda" else model.numeric_dtype,
                return_degrees=False,
            )
            initialization_graph_cpu = optimizer_graph_cpu
        timings["graph_preprocess_time"] = perf_counter() - preprocess_t0
        diagnostics["optimizer_graph"] = {
            **{str(key): int(value) if isinstance(value, np.integer) else value for key, value in preprocess_stats.items()},
            "resolved_n_epochs": int(resolved_epochs),
            "dtype": str(optimizer_graph_cpu.dtype),
        }
        optimizer_graph = optimizer_graph_cpu

    init_embedding = None
    if return_init:
        init_t0 = perf_counter()
        if model.runtime.device == "cuda":
            init_embedding, init_timing, init_diagnostics, optimizer_graph = _initialize_cuda(
                model, X, optimizer_graph_cpu
            )
        else:
            init_embedding, init_timing, init_diagnostics = _initialize_cpu(
                model,
                X,
                initialization_graph_cpu,
                shared_random_state,
            )
        timings["initialization_time"] = perf_counter() - init_t0
        timings.update({str(key): float(value) for key, value in init_timing.items()})
        diagnostics["initialization"] = dict(init_diagnostics)

    output_indices = indices if return_knn else None
    output_distances = distances if return_knn else None
    output_raw_graph = raw_graph if return_raw_graph else None
    output_optimizer_graph = optimizer_graph if return_optimizer_graph else None
    output_init = init_embedding if return_init else None
    if model.runtime.device == "cuda":
        (
            output_indices,
            output_distances,
            output_raw_graph,
            output_optimizer_graph,
            output_init,
        ) = _cuda_outputs(
            host_output=host_output,
            knn_indices=output_indices,
            knn_distances=output_distances,
            fuzzy_graph=output_raw_graph,
            optimizer_graph=output_optimizer_graph,
            init_embedding=output_init,
        )

    timings["total_prepare_time"] = perf_counter() - total_t0
    return PreparedInputs(
        knn_indices=output_indices,
        knn_distances=output_distances,
        fuzzy_graph=output_raw_graph,
        optimizer_graph=output_optimizer_graph,
        init_embedding=output_init,
        timings={key: float(value) for key, value in timings.items()},
        diagnostics=diagnostics,
        effective_config=_effective_config(
            model,
            device=model.runtime.device,
            host_output=host_output,
        ),
        device=model.runtime.device,
        host_output=bool(host_output),
    )


def prepare_fixed_inputs(
    model: Any,
    X: Any,
    *,
    return_knn: bool = True,
    return_raw_graph: bool = True,
    return_optimizer_graph: bool = True,
    return_init: bool = True,
    host_output: bool = True,
) -> PreparedInputs:
    """Build KNN, raw fuzzy graph, optimizer graph, and initialization."""

    return _prepare(
        model,
        X,
        source="knn",
        knn_indices=None,
        knn_distances=None,
        return_knn=return_knn,
        return_raw_graph=return_raw_graph,
        return_optimizer_graph=return_optimizer_graph,
        return_init=return_init,
        host_output=host_output,
    )


def prepare_fixed_inputs_from_knn(
    model: Any,
    X: Any,
    knn_indices: Any,
    knn_distances: Any,
    *,
    return_knn: bool = True,
    return_raw_graph: bool = True,
    return_optimizer_graph: bool = True,
    return_init: bool = True,
    host_output: bool = True,
) -> PreparedInputs:
    """Build fixed inputs from caller-provided KNN arrays without a KNN search."""

    return _prepare(
        model,
        X,
        source="fixed_knn",
        knn_indices=knn_indices,
        knn_distances=knn_distances,
        return_knn=return_knn,
        return_raw_graph=return_raw_graph,
        return_optimizer_graph=return_optimizer_graph,
        return_init=return_init,
        host_output=host_output,
    )


def prepare_fixed_inputs_from_graph(
    model: Any,
    X: Any,
    fuzzy_graph: Any,
    *,
    return_raw_graph: bool = True,
    return_optimizer_graph: bool = True,
    return_init: bool = True,
    host_output: bool = True,
) -> PreparedInputs:
    """Prepare optimizer graph and initialization from an existing raw graph."""
    if model.runtime.device == "metal" and not host_output:
        raise NotImplementedError(
            "device='metal' fixed-input preparation preserves host NumPy/SciPy "
            "artifacts; host_output=False is not supported"
        )
    total_t0 = perf_counter()
    timings: dict[str, float] = {}
    diagnostics: dict[str, Any] = {
        "source": "raw_graph",
        "device": model.runtime.device,
        "host_output": bool(host_output),
    }

    validation_t0 = perf_counter()
    X = check_array(X, dtype=model.numeric_dtype, accept_sparse="csr", order="C")
    if not issparse(fuzzy_graph):
        raise TypeError("fuzzy_graph must be a scipy sparse matrix")
    raw_graph = (
        fuzzy_graph.copy()
        if _uses_ordered_cpu_umap(model)
        else fuzzy_graph.tocsr()
    )
    if raw_graph.shape != (X.shape[0], X.shape[0]):
        raise ValueError("fuzzy_graph shape must match X")
    timings["input_validation_time"] = perf_counter() - validation_t0
    shared_random_state = _shared_random_state(model)
    diagnostics["fuzzy_graph"] = {
        "shape": [int(value) for value in raw_graph.shape],
        "nnz": int(raw_graph.nnz),
        "dtype": str(raw_graph.dtype),
    }

    optimizer_graph = None
    optimizer_graph_cpu = None
    initialization_graph_cpu = None
    if return_optimizer_graph or return_init:
        preprocess_t0 = perf_counter()
        if _uses_ordered_cpu_umap(model):
            ordered_graph, resolved_epochs, preprocess_stats = preprocess_graph(
                raw_graph,
                model.n_epochs,
                return_stats=True,
            )
            initialization_graph_cpu = ordered_graph
            optimizer_graph_cpu = ordered_graph.tocsr()
        else:
            optimizer_graph_cpu, resolved_epochs, _, preprocess_stats = preprocess_graph_csr(
                raw_graph,
                model.n_epochs,
                dtype=np.float32 if model.runtime.device == "cuda" else model.numeric_dtype,
                return_degrees=False,
            )
            initialization_graph_cpu = optimizer_graph_cpu
        timings["graph_preprocess_time"] = perf_counter() - preprocess_t0
        diagnostics["optimizer_graph"] = {
            **{str(key): int(value) if isinstance(value, np.integer) else value for key, value in preprocess_stats.items()},
            "resolved_n_epochs": int(resolved_epochs),
            "dtype": str(optimizer_graph_cpu.dtype),
        }
        optimizer_graph = optimizer_graph_cpu

    init_embedding = None
    if return_init:
        init_t0 = perf_counter()
        if model.runtime.device == "cuda":
            init_embedding, init_timing, init_diagnostics, optimizer_graph = _initialize_cuda(
                model, X, optimizer_graph_cpu
            )
        else:
            init_embedding, init_timing, init_diagnostics = _initialize_cpu(
                model,
                X,
                initialization_graph_cpu,
                shared_random_state,
            )
        timings["initialization_time"] = perf_counter() - init_t0
        timings.update({str(key): float(value) for key, value in init_timing.items()})
        diagnostics["initialization"] = dict(init_diagnostics)

    output_raw_graph = raw_graph if return_raw_graph else None
    output_optimizer_graph = optimizer_graph if return_optimizer_graph else None
    output_init = init_embedding if return_init else None
    if model.runtime.device == "cuda":
        (_, _, output_raw_graph, output_optimizer_graph, output_init) = _cuda_outputs(
            host_output=host_output,
            knn_indices=None,
            knn_distances=None,
            fuzzy_graph=output_raw_graph,
            optimizer_graph=output_optimizer_graph,
            init_embedding=output_init,
        )

    timings["total_prepare_time"] = perf_counter() - total_t0
    return PreparedInputs(
        knn_indices=None,
        knn_distances=None,
        fuzzy_graph=output_raw_graph,
        optimizer_graph=output_optimizer_graph,
        init_embedding=output_init,
        timings={key: float(value) for key, value in timings.items()},
        diagnostics=diagnostics,
        effective_config=_effective_config(
            model,
            device=model.runtime.device,
            host_output=host_output,
        ),
        device=model.runtime.device,
        host_output=bool(host_output),
    )
