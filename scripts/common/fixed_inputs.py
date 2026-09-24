"""Persistence helpers for fixed experiment inputs (kNN graph, fuzzy graph, initialization).

Graph construction, optimizer preprocessing, initialization and CUDA transfer
policy live in :class:`ibumap.IBUMAP` (``prepare_fixed_inputs*``). This module
only builds the estimator from an experiment parameter mapping and reads/writes
the files of one fixed-input directory::

    <root>/<dataset>/neighbors_<k>/
        knn_indices.npy  knn_distances.npy  fuzzy_graph_csr.npz
        optimizer_graph_csr.npz (optional)  init_embedding.npy  metadata.json
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from common.callable_utils import filtered_kwargs
from common.paths import ensure_ibumap_importable
from common.run_metadata import read_json, write_json


def _canonical_device(device: str | None) -> str:
    value = str(device or "cpu").lower()
    if value == "gpu":
        value = "cuda"
    if value not in {"cpu", "cuda"}:
        raise ValueError(f"fixed-input device must be 'cpu' or 'cuda', got {device!r}")
    return value


def _resolve_device(params: Mapping[str, Any], device: str | None) -> str:
    return _canonical_device(params.get("device", device))


def make_ibumap_model(params: Mapping[str, Any], *, device: str):
    """Return ``(IBUMAP(...), effective_kwargs)``; unknown keys are dropped."""
    ensure_ibumap_importable()
    from ibumap import IBUMAP

    kwargs = {str(key): value for key, value in params.items() if value is not None}
    kwargs.pop("device", None)
    kwargs.setdefault("algorithm", "ibumap")
    kwargs["device"] = _canonical_device(device)
    effective_kwargs = filtered_kwargs(IBUMAP, kwargs)
    return IBUMAP(**effective_kwargs), effective_kwargs


def find_ab_params(spread: float, min_dist: float) -> tuple[float, float]:
    """UMAP's (a, b) curve parameters for the given spread and min_dist."""
    ensure_ibumap_importable()
    from ibumap.api import _find_ab_params

    return _find_ab_params(spread, min_dist)


def fixed_input_dir(root: str | Path, dataset_name: str, n_neighbors: int) -> Path:
    return Path(root) / dataset_name / f"neighbors_{int(n_neighbors)}"


def fixed_input_paths(root: str | Path, dataset_name: str, n_neighbors: int) -> dict[str, Path]:
    directory = fixed_input_dir(root, dataset_name, n_neighbors)
    return {
        "directory": directory,
        "knn_indices": directory / "knn_indices.npy",
        "knn_distances": directory / "knn_distances.npy",
        "fuzzy_graph": directory / "fuzzy_graph_csr.npz",
        "optimizer_graph": directory / "optimizer_graph_csr.npz",
        "init_embedding": directory / "init_embedding.npy",
        "metadata": directory / "metadata.json",
        "labels_csv": directory / "labels.csv",
    }


def fixed_inputs_complete(root: str | Path, dataset_name: str, n_neighbors: int) -> bool:
    """True when the required files exist; the optimizer graph is optional."""
    paths = fixed_input_paths(root, dataset_name, n_neighbors)
    return all(paths[key].exists()
               for key in ("knn_indices", "knn_distances", "fuzzy_graph", "init_embedding", "metadata"))


def fixed_input_metadata_matches(path: str | Path, expected: Mapping[str, Any]) -> bool:
    existing = read_json(path)
    return all(existing.get(key) == value for key, value in expected.items())


def build_fixed_inputs(X: Any, params: Mapping[str, Any], *, device: str = "cpu"):
    """Return IBUMAP's public :class:`ibumap.PreparedInputs` bundle."""
    resolved_device = _resolve_device(params, device)
    model, _ = make_ibumap_model(params, device=resolved_device)
    return model.prepare_fixed_inputs(X, device=resolved_device, host_output=True)


def build_knn_and_graph(X: Any, params: Mapping[str, Any], *, device: str = "cpu") -> tuple[Any, Any, Any]:
    """kNN indices, kNN distances and the raw fuzzy graph (host arrays)."""
    resolved_device = _resolve_device(params, device)
    model, _ = make_ibumap_model(params, device=resolved_device)
    bundle = model.prepare_fixed_inputs(X, device=resolved_device, return_knn=True, return_raw_graph=True,
                                        return_optimizer_graph=False, return_init=False, host_output=True)
    return bundle.knn_indices, bundle.knn_distances, bundle.fuzzy_graph


def build_fuzzy_graph_from_knn(X: Any, knn_indices: Any, knn_distances: Any, params: Mapping[str, Any], *,
                               device: str = "cpu") -> Any:
    resolved_device = _resolve_device(params, device)
    model, _ = make_ibumap_model(params, device=resolved_device)
    bundle = model.prepare_fixed_inputs_from_knn(X, knn_indices, knn_distances, device=resolved_device,
                                                 return_knn=False, return_raw_graph=True,
                                                 return_optimizer_graph=False, return_init=False, host_output=True)
    return bundle.fuzzy_graph


def spectral_initialization(X: Any, graph: Any, params: Mapping[str, Any], *, device: str = "cpu") -> np.ndarray:
    """IBUMAP's initialization for a given fuzzy graph, as a float32 C-ordered array."""
    resolved_device = _resolve_device(params, device)
    model, _ = make_ibumap_model(params, device=resolved_device)
    bundle = model.prepare_fixed_inputs_from_graph(X, graph, device=resolved_device, return_raw_graph=False,
                                                   return_optimizer_graph=False, return_init=True, host_output=True)
    return np.asarray(bundle.init_embedding, dtype=np.float32, order="C")


def save_fixed_inputs(
    *,
    root: str | Path,
    dataset_name: str,
    n_neighbors: int,
    knn_indices: Any,
    knn_distances: Any,
    graph: Any,
    init_embedding: Any,
    metadata: Mapping[str, Any],
    optimizer_graph: Any | None = None,
) -> dict[str, Path]:
    """Write one fixed-input directory and return its paths."""
    from scipy import sparse

    paths = fixed_input_paths(root, dataset_name, n_neighbors)
    paths["directory"].mkdir(parents=True, exist_ok=True)
    np.save(paths["knn_indices"], np.asarray(knn_indices, dtype=np.int64))
    np.save(paths["knn_distances"], np.asarray(knn_distances, dtype=np.float32))
    sparse.save_npz(paths["fuzzy_graph"], graph.tocsr())
    if optimizer_graph is not None:
        sparse.save_npz(paths["optimizer_graph"], optimizer_graph.tocsr())
    write_json(paths["metadata"], dict(metadata))
    np.save(paths["init_embedding"], np.asarray(init_embedding, dtype=np.float32))
    return paths


def load_fixed_inputs(root: str | Path, dataset_name: str, n_neighbors: int) -> dict[str, Any]:
    """Load a directory written by :func:`save_fixed_inputs`."""
    from scipy import sparse

    paths = fixed_input_paths(root, dataset_name, n_neighbors)
    if not fixed_inputs_complete(root, dataset_name, n_neighbors):
        raise FileNotFoundError(f"Incomplete fixed inputs in {paths['directory']}")
    return {
        "knn_indices": np.load(paths["knn_indices"]),
        "knn_distances": np.load(paths["knn_distances"]),
        "fuzzy_graph": sparse.load_npz(paths["fuzzy_graph"]).tocsr(),
        "optimizer_graph": (sparse.load_npz(paths["optimizer_graph"]).tocsr()
                            if paths["optimizer_graph"].exists() else None),
        "init_embedding": np.load(paths["init_embedding"]),
        "metadata": read_json(paths["metadata"]),
    }
