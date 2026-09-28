"""Thin adapters for umap-learn, cuML UMAP, TorchDR UMAP and ibUMAP.

Every ``run_*`` function returns ``(embedding, extras)``: a C-ordered float32
NumPy embedding and a JSON-friendly dict with the parameters actually passed
(``used_params``) and, for ibUMAP, the stage timers (``time_costs``).
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np

from common.callable_utils import filtered_kwargs
from common.gpu_runtime import synchronize_gpu, synchronize_torch_gpu, to_numpy_array
from common.paths import ensure_ibumap_importable


CUML_UMAP_ALLOWED = {
    "n_neighbors",
    "n_components",
    "metric",
    "n_epochs",
    "learning_rate",
    "min_dist",
    "spread",
    "set_op_mix_ratio",
    "local_connectivity",
    "repulsion_strength",
    "negative_sample_rate",
    "random_state",
    "verbose",
}

TORCHDR_UMAP_ALLOWED = {
    "n_neighbors",
    "n_components",
    "min_dist",
    "spread",
    "a",
    "b",
    "lr",
    "optimizer",
    "optimizer_kwargs",
    "scheduler",
    "scheduler_kwargs",
    "init",
    "init_scaling",
    "min_grad_norm",
    "max_iter",
    "device",
    "backend",
    "verbose",
    "random_state",
    "max_iter_affinity",
    "metric",
    "negative_sample_rate",
    "repulsion_strength",
    "early_exaggeration_coeff",
    "early_exaggeration_iter",
    "check_interval",
    "discard_NNs",
    "compile",
    "distributed",
}

TORCHDR_UMAP_PARAM_ALIASES = {
    "n_epochs": "max_iter",
    "learning_rate": "lr",
}


def _time_costs(model: Any) -> dict[str, Any]:
    """Preserve numeric timings and scalar optimizer diagnostics.

    Most entries returned by ``get_time_costs`` are seconds, but optimizers may
    also expose diagnostics such as ``p2m_resolved_mode`` in the same mapping.
    Coercing every value to float makes those adapters unusable.
    """
    if not hasattr(model, "get_time_costs"):
        return {}
    result: dict[str, Any] = {}
    for key, value in model.get_time_costs().items():
        if isinstance(value, (str, bool)) or value is None:
            result[str(key)] = value
        else:
            result[str(key)] = float(value)
    return result


def _canonical_device(device: str) -> str:
    """Accept ``gpu`` as an alias of ``cuda``."""
    return "cuda" if device == "gpu" else device


def clean_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in params.items() if value is not None}


def merge_params(*parts: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        if part:
            merged.update(dict(part))
    return clean_params(merged)


def _normalized_torchdr_umap_params(params: Mapping[str, Any]) -> dict[str, Any]:
    normalized_params = clean_params(params)
    for alias, canonical in TORCHDR_UMAP_PARAM_ALIASES.items():
        if canonical not in normalized_params and alias in normalized_params:
            normalized_params[canonical] = normalized_params[alias]
        normalized_params.pop(alias, None)

    if "device" in normalized_params:
        normalized_params["device"] = _canonical_device(
            str(normalized_params["device"])
        )
    return normalized_params


@contextmanager
def _torch_execution_mode(deterministic: bool | None):
    """Snapshot PyTorch state and yield a callback that applies one run mode."""
    if deterministic is None:
        yield lambda: {"deterministic": None}
        return

    import torch  # type: ignore

    previous_benchmark = bool(torch.backends.cudnn.benchmark)
    previous_cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    previous_algorithms = bool(torch.are_deterministic_algorithms_enabled())
    previous_warn_only = (
        bool(torch.is_deterministic_algorithms_warn_only_enabled())
        if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled")
        else False
    )
    previous_cublas_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    def apply_mode() -> dict[str, Any]:
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        else:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        torch.use_deterministic_algorithms(deterministic)
        return {
            "deterministic": deterministic,
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        }

    try:
        yield apply_mode
    finally:
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.use_deterministic_algorithms(
            previous_algorithms,
            warn_only=previous_warn_only,
        )
        if previous_cublas_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas_workspace


def run_umap_learn(X: Any, params: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    import umap  # type: ignore

    model_kwargs = filtered_kwargs(umap.UMAP, clean_params(params))
    model = umap.UMAP(**model_kwargs)
    embedding = model.fit_transform(X)
    return np.asarray(embedding, dtype=np.float32, order="C"), {"used_params": model_kwargs}


def run_cuml_umap(X: Any, params: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    from cuml.manifold import UMAP as CumlUMAP  # type: ignore

    model_kwargs = {key: value for key, value in clean_params(params).items() if key in CUML_UMAP_ALLOWED}
    model = CumlUMAP(**filtered_kwargs(CumlUMAP, model_kwargs))
    try:
        import cupy as cp  # type: ignore

        X_input: Any = cp.asarray(X)
    except Exception:
        X_input = X
    synchronize_gpu()
    embedding = model.fit_transform(X_input)
    synchronize_gpu()
    return to_numpy_array(embedding).astype(np.float32, copy=False), {"used_params": model_kwargs}


def run_torchdr_umap(X: Any, params: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Run torchDR UMAP with common UMAP parameter-name compatibility.

    TorchDR calls the iteration count ``max_iter`` and the learning rate
    ``lr``.  Existing experiment configs can keep using the umap-learn/cuML
    names ``n_epochs`` and ``learning_rate``; explicit torchDR names take
    precedence when both forms are present.

    The adapter defaults to the FAISS backend.  Callers can explicitly pass
    ``backend=None`` to use the built-in PyTorch distance path.  The
    adapter-only ``deterministic`` flag temporarily selects PyTorch's
    deterministic or maximum-throughput CUDA execution mode around
    ``fit_transform``.
    """
    from torchdr import UMAP as TorchDRUMAP  # type: ignore

    normalized_params = _normalized_torchdr_umap_params(params)
    deterministic = normalized_params.pop("deterministic", None)
    if deterministic is not None and not isinstance(deterministic, bool):
        raise TypeError("deterministic must be true, false, or null")
    process_duplicates = bool(normalized_params.pop("process_duplicates", True))
    model_kwargs = {
        key: value
        for key, value in normalized_params.items()
        if key in TORCHDR_UMAP_ALLOWED
    }
    model_kwargs.setdefault("backend", "faiss")
    effective_model_kwargs = filtered_kwargs(TorchDRUMAP, model_kwargs)
    with _torch_execution_mode(deterministic) as apply_torch_mode:
        model = TorchDRUMAP(**effective_model_kwargs)
        model.process_duplicates = process_duplicates

        # torchDR 0.4 applies its own fast-mode seed settings in __init__.
        # Re-apply the requested mode after construction and before any fit work.
        torch_runtime = apply_torch_mode()
        device = str(effective_model_kwargs.get("device", "auto"))
        input_device = str(getattr(X, "device", "cpu"))
        cuda_device: str | None = None
        if device.startswith("cuda"):
            cuda_device = device
        elif device == "auto" and input_device.startswith("cuda"):
            cuda_device = input_device

        if cuda_device is not None:
            synchronize_torch_gpu(cuda_device)
        embedding = model.fit_transform(X)
        if cuda_device is not None:
            synchronize_torch_gpu(cuda_device)

    return np.asarray(
        to_numpy_array(embedding),
        dtype=np.float32,
        order="C",
    ), {
        "used_params": effective_model_kwargs,
        "deterministic": deterministic,
        "torch_runtime": torch_runtime,
    }


def _torchdr_graph_to_rowwise(
    graph: Any,
    *,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Convert a square sparse fuzzy graph to torchDR's padded row-wise form."""
    from scipy import sparse

    graph_cpu = graph
    if not sparse.issparse(graph_cpu) and hasattr(graph_cpu, "get"):
        graph_cpu = graph_cpu.get()
    if not sparse.issparse(graph_cpu):
        raise TypeError("graph must be a SciPy or CuPy sparse matrix")

    matrix = graph_cpu.tocsr().astype(np.float32, copy=True)
    if matrix.shape != (n_samples, n_samples):
        raise ValueError(
            "graph shape must match the number of samples: "
            f"expected {(n_samples, n_samples)}, got {matrix.shape}"
        )
    matrix.sum_duplicates()
    matrix.setdiag(0.0)
    matrix.eliminate_zeros()
    matrix.sort_indices()

    if matrix.nnz == 0:
        raise ValueError("graph must contain at least one positive edge")
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("graph contains non-finite affinity values")
    if np.any(matrix.data < 0):
        raise ValueError("graph affinities must be non-negative")

    row_counts = np.diff(matrix.indptr)
    max_degree = int(row_counts.max(initial=0))
    values = np.zeros((n_samples, max_degree), dtype=np.float32)
    indices = np.full((n_samples, max_degree), -1, dtype=np.int64)
    for row, count in enumerate(row_counts):
        if count == 0:
            continue
        start = int(matrix.indptr[row])
        stop = int(matrix.indptr[row + 1])
        values[row, :count] = matrix.data[start:stop]
        indices[row, :count] = matrix.indices[start:stop]

    return values, indices, {
        "n_samples": int(n_samples),
        "nnz": int(matrix.nnz),
        "max_degree": max_degree,
    }


def _make_torchdr_optimization_model(
    graph_values: np.ndarray,
    graph_indices: np.ndarray,
    init_embedding: np.ndarray,
    model_kwargs: Mapping[str, Any],
) -> Any:
    """Build a torchDR UMAP model backed by a precomputed sparse affinity."""
    import torch  # type: ignore
    from torchdr import SparseAffinity, UMAP as TorchDRUMAP  # type: ignore

    class PrecomputedSparseAffinity(SparseAffinity):
        def __init__(self, values: np.ndarray, indices: np.ndarray, *, device: str):
            super().__init__(
                device=device,
                backend=None,
                sparsity=True,
                distributed=False,
                _pre_processed=True,
            )
            self._precomputed_values = values
            self._precomputed_indices = indices

        def _compute_sparse_affinity(
            self,
            X: Any,
            return_indices: bool = True,
            **kwargs: Any,
        ) -> Any:
            target_device = X.device if self.device == "auto" else self.device
            values = torch.as_tensor(
                self._precomputed_values,
                dtype=X.dtype,
                device=target_device,
            )
            indices = torch.as_tensor(
                self._precomputed_indices,
                dtype=torch.long,
                device=target_device,
            )
            return (values, indices) if return_indices else values

    class OptimizationOnlyUMAP(TorchDRUMAP):
        def __init__(self, fixed_init: np.ndarray, **kwargs: Any):
            super().__init__(**kwargs)
            self._fixed_init = fixed_init

        def _init_embedding(self, X: Any) -> Any:
            embedding = torch.as_tensor(
                self._fixed_init,
                dtype=X.dtype,
                device=self.device_,
            ).clone()
            self.embedding_ = embedding
            return self.embedding_.requires_grad_()

    effective_kwargs = dict(model_kwargs)
    model = OptimizationOnlyUMAP(init_embedding, **effective_kwargs)
    model.process_duplicates = False
    model.affinity_in = PrecomputedSparseAffinity(
        graph_values,
        graph_indices,
        device=str(effective_kwargs["device"]),
    )
    return model


def run_torchdr_umap_optimize_from_graph(
    X: Any,
    graph: Any,
    init_embedding: Any,
    params: Mapping[str, Any],
    *,
    device: str = "cuda",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run only torchDR UMAP optimization from a fuzzy graph and fixed init.

    TorchDR 0.4 does not expose a public ``simplicial_set_embedding``-style
    function.  This adapter injects the precomputed graph through torchDR's
    public ``SparseAffinity`` extension point and preserves the supplied
    initialization exactly.  It intentionally disables duplicate processing
    and distributed execution because both would invalidate the fixed graph's
    row alignment.  The adapter-only ``deterministic`` flag selects the same
    PyTorch/CUDA execution modes as :func:`run_torchdr_umap`.
    """
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError("X must be a two-dimensional array-like object")
    n_samples = int(X.shape[0])
    init_array = np.asarray(
        to_numpy_array(init_embedding),
        dtype=np.float32,
        order="C",
    )
    if init_array.ndim != 2 or init_array.shape[0] != n_samples:
        raise ValueError(
            "init_embedding must have shape (n_samples, n_components); "
            f"got {init_array.shape} for n_samples={n_samples}"
        )
    if not np.all(np.isfinite(init_array)):
        raise ValueError("init_embedding contains non-finite values")

    graph_values, graph_indices, graph_metadata = _torchdr_graph_to_rowwise(
        graph,
        n_samples=n_samples,
    )
    normalized_params = _normalized_torchdr_umap_params(params)
    deterministic = normalized_params.pop("deterministic", None)
    if deterministic is not None and not isinstance(deterministic, bool):
        raise TypeError("deterministic must be true, false, or null")
    normalized_params["device"] = _canonical_device(str(device))
    normalized_params["n_components"] = int(
        normalized_params.get("n_components", init_array.shape[1])
    )
    if normalized_params["n_components"] != init_array.shape[1]:
        raise ValueError(
            "n_components must match init_embedding: "
            f"expected {init_array.shape[1]}, got {normalized_params['n_components']}"
        )

    for key in (
        "backend",
        "init",
        "init_scaling",
        "max_iter_affinity",
        "metric",
        "process_duplicates",
    ):
        normalized_params.pop(key, None)
    model_kwargs = {
        key: value
        for key, value in normalized_params.items()
        if key in TORCHDR_UMAP_ALLOWED
    }
    model_kwargs.update(
        {
            "backend": None,
            "discard_NNs": False,
            "distributed": False,
        }
    )
    # The feature values are not used after graph injection.  A one-column
    # stub supplies only n_samples and float32 dtype to torchDR's fit lifecycle.
    data_stub = np.zeros((n_samples, 1), dtype=np.float32)
    cuda_device = (
        str(model_kwargs["device"])
        if str(model_kwargs["device"]).startswith("cuda")
        else None
    )
    with _torch_execution_mode(deterministic) as apply_torch_mode:
        model = _make_torchdr_optimization_model(
            graph_values,
            graph_indices,
            init_array,
            model_kwargs,
        )

        # torchDR 0.4 may change global PyTorch execution flags while the UMAP
        # object is constructed, so re-apply the requested profile immediately
        # before the optimization-only fit lifecycle.
        torch_runtime = apply_torch_mode()
        if cuda_device is not None:
            synchronize_torch_gpu(cuda_device)
        embedding = model.fit_transform(data_stub)
        if cuda_device is not None:
            synchronize_torch_gpu(cuda_device)

    return np.asarray(
        to_numpy_array(embedding),
        dtype=np.float32,
        order="C",
    ), {
        "used_params": model_kwargs,
        "graph": graph_metadata,
        "fixed_init": True,
        "optimization_only": True,
        "deterministic": deterministic,
        "torch_runtime": torch_runtime,
        "time_costs": {},
    }


def run_ibumap(
    X: Any,
    params: Mapping[str, Any],
    *,
    algorithm: str = "ibumap",
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, Any]]:
    ensure_ibumap_importable()
    from ibumap import IBUMAP

    device = _canonical_device(device)
    model_kwargs = clean_params(params)
    model_kwargs.update({"algorithm": algorithm, "device": device})
    model = IBUMAP(**filtered_kwargs(IBUMAP, model_kwargs))
    if device == "cuda":
        synchronize_gpu()
    embedding = model.fit_transform(X)
    if device == "cuda":
        synchronize_gpu()
    time_costs = _time_costs(model)
    extras: dict[str, Any] = {"used_params": model_kwargs, "time_costs": time_costs}
    if getattr(model, "diagnostics_memory_path", None):
        extras["diagnostics_memory_path"] = model.diagnostics_memory_path
    for attr in ("ibfft_kernel_clip", "umap_epsilon"):
        if hasattr(model, attr):
            extras[attr] = getattr(model, attr)
    return to_numpy_array(embedding).astype(np.float32, copy=False), extras


def run_ibumap_from_knn(
    X: Any,
    knn_indices: Any,
    knn_distances: Any,
    params: Mapping[str, Any],
    *,
    device: str = "cpu",
) -> tuple[np.ndarray, dict[str, Any]]:
    ensure_ibumap_importable()
    from ibumap import IBUMAP

    device = _canonical_device(device)
    model_kwargs = clean_params(params)
    model_kwargs.update({"algorithm": "ibumap", "device": device})
    model = IBUMAP(**filtered_kwargs(IBUMAP, model_kwargs))
    if not hasattr(model, "fit_transform_from_knn"):
        raise AttributeError("IBUMAP does not expose fit_transform_from_knn")
    if device == "cuda":
        synchronize_gpu()
    embedding = model.fit_transform_from_knn(X, knn_indices, knn_distances)
    if device == "cuda":
        synchronize_gpu()
    time_costs = _time_costs(model)
    extras: dict[str, Any] = {"used_params": model_kwargs, "time_costs": time_costs}
    if getattr(model, "diagnostics_memory_path", None):
        extras["diagnostics_memory_path"] = model.diagnostics_memory_path
    return to_numpy_array(embedding).astype(np.float32, copy=False), extras


def run_ibumap_optimize_from_graph(
    X: Any,
    graph: Any,
    init_embedding: Any,
    params: Mapping[str, Any],
    *,
    algorithm: str = "ibumap",
    device: str = "cpu",
    collect_effective_config: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    ensure_ibumap_importable()
    from ibumap import IBUMAP

    device = _canonical_device(device)
    # Preserve explicit ``None`` values: several optimization controls use
    # ``None`` to disable a non-None estimator default (notably force clips).
    model_kwargs = {str(key): value for key, value in params.items()}
    model_kwargs.update({"algorithm": algorithm, "device": device})
    effective_model_kwargs = filtered_kwargs(IBUMAP, model_kwargs)
    model = IBUMAP(**effective_model_kwargs)
    if not hasattr(model, "optimize_from_graph"):
        raise AttributeError("IBUMAP does not expose optimize_from_graph")
    if device == "cuda":
        synchronize_gpu()
    # The CPU optimizer needs only the graph and the initialization; the CUDA
    # optimizer still takes the feature matrix as well.
    if device == "cuda":
        embedding = model.optimize_from_graph(X, graph, init_embedding)
    else:
        embedding = model.optimize_from_graph(graph, init_embedding)
    if device == "cuda":
        synchronize_gpu()
    time_costs = _time_costs(model)
    extras: dict[str, Any] = {"used_params": effective_model_kwargs, "time_costs": time_costs}
    if collect_effective_config and hasattr(model, "explain_effective_config"):
        extras["effective_config"] = model.explain_effective_config()
    if getattr(model, "diagnostics_memory_path", None):
        extras["diagnostics_memory_path"] = model.diagnostics_memory_path
    return to_numpy_array(embedding).astype(np.float32, copy=False), extras


def run_ibumap_optimize_from_prepared_graph(
    X: Any,
    optimizer_graph: Any,
    init_embedding: Any,
    params: Mapping[str, Any],
    *,
    algorithm: str = "ibumap",
    device: str = "cpu",
    collect_effective_config: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run only optimization from IBUMAP's canonical prepared graph.

    ``optimizer_graph`` must be the canonical sorted CSR matrix returned by
    ``IBUMAP.prepare_fixed_inputs``.  Unlike ``optimize_from_graph``, this
    entrypoint does not repeat graph thresholding, duplicate elimination, or
    CSR sorting.  ``X`` is accepted for signature symmetry with
    :func:`run_ibumap_optimize_from_graph` but not used: prepared-graph
    optimization does not consume feature data.
    """
    del X
    ensure_ibumap_importable()
    from ibumap import IBUMAP

    device = _canonical_device(device)
    # Preserve explicit ``None`` values for nullable optimization controls.
    model_kwargs = {str(key): value for key, value in params.items()}
    model_kwargs.update({"algorithm": algorithm, "device": device})
    effective_model_kwargs = filtered_kwargs(IBUMAP, model_kwargs)
    model = IBUMAP(**effective_model_kwargs)
    if not hasattr(model, "optimize_from_prepared_graph"):
        raise AttributeError("IBUMAP does not expose optimize_from_prepared_graph")
    if device == "cuda":
        synchronize_gpu()
    embedding = model.optimize_from_prepared_graph(optimizer_graph, init_embedding)
    if device == "cuda":
        synchronize_gpu()
    time_costs = _time_costs(model)
    extras: dict[str, Any] = {"used_params": effective_model_kwargs, "time_costs": time_costs}
    if collect_effective_config and hasattr(model, "explain_effective_config"):
        extras["effective_config"] = model.explain_effective_config()
    if getattr(model, "diagnostics_memory_path", None):
        extras["diagnostics_memory_path"] = model.diagnostics_memory_path
    return to_numpy_array(embedding).astype(np.float32, copy=False), extras


def run_algorithm_entry(
    X: Any,
    algorithm_entry: Mapping[str, Any],
    *,
    common_params: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    name = str(algorithm_entry.get("name") or algorithm_entry.get("id") or "algorithm")
    family = str(algorithm_entry.get("family") or algorithm_entry.get("implementation") or name)
    params = merge_params(common_params, algorithm_entry.get("params") if isinstance(algorithm_entry.get("params"), Mapping) else None, algorithm_entry)
    for key in (
        "name",
        "id",
        "family",
        "implementation",
        "params",
        "notes",
        "description",
        "explicit_kernel_override",
        "explicit_override",
    ):
        params.pop(key, None)

    if family in {"umap_learn", "umap-learn"} or name == "umap_learn":
        embedding, extras = run_umap_learn(X, params)
    elif family in {"cuml", "cuml_umap"} or name == "cuml_umap":
        embedding, extras = run_cuml_umap(X, params)
    elif family in {"torchdr", "torchdr_umap", "torchdr-umap"} or name == "torchdr_umap":
        embedding, extras = run_torchdr_umap(X, params)
    elif family == "ibumap" or name.startswith("ibumap"):
        algorithm = str(algorithm_entry.get("algorithm", params.pop("algorithm", "ibumap")))
        device = str(algorithm_entry.get("device", params.pop("device", "cpu")))
        embedding, extras = run_ibumap(X, params, algorithm=algorithm, device=device)
    else:
        raise ValueError(f"Unsupported algorithm family for {name!r}: {family!r}")

    extras.setdefault("algorithm_name", name)
    extras.setdefault("family", family)
    return embedding, extras
