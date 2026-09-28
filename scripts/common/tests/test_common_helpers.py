from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from common import algorithm_adapters, dataset_io, fixed_inputs, gpu_runtime, paths, run_metadata


def _blobs(n_per_cluster: int = 60, dim: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=15.0, size=(3, dim))
    X = np.concatenate([c + rng.normal(size=(n_per_cluster, dim)) for c in centers]).astype(np.float32)
    return X, np.repeat(np.arange(3), n_per_cluster)


def test_repo_root_is_the_ibumap_checkout() -> None:
    root = paths.REPO_ROOT
    assert (root / "pyproject.toml").is_file() and (root / "src" / "ibumap").is_dir()
    assert paths.SCRIPTS_ROOT == root / "scripts"
    assert paths.CATALOG_PATH == root / "datasets" / "catalog.json"
    assert paths.repo_relative(root / "scripts" / "common") == "scripts/common"
    with pytest.raises(FileNotFoundError):
        paths.find_repo_root(Path("/"))


def test_catalog_lists_the_paper_datasets() -> None:
    entries = dataset_io.catalog_entries()
    assert len(entries) == 71
    assert dataset_io.catalog_entry_by_id("iris") is not None
    assert dataset_io.catalog_entry_by_id("deep_image_96_angular") is None


def test_select_row_indices_is_sorted_deterministic_and_complete_when_small() -> None:
    first = dataset_io.select_row_indices(1000, 100, seed=42)
    assert np.array_equal(first, dataset_io.select_row_indices(1000, 100, seed=42))
    assert np.all(np.diff(first) > 0) and first.dtype == np.int64 and first.size == 100
    assert np.array_equal(dataset_io.select_row_indices(50, 100), np.arange(50))
    assert np.array_equal(dataset_io.select_row_indices(50, None), np.arange(50))


def test_load_processed_dataset_from_a_directory(tmp_path: Path) -> None:
    directory = tmp_path / "toy"
    directory.mkdir()
    X, y = _blobs(10)
    np.save(directory / "features.npy", X)
    np.save(directory / "target.npy", y)
    (directory / "metadata.json").write_text(json.dumps({"dataset_id": "toy", "feature_shape": list(X.shape)}))
    dataset = dataset_io.load_processed_dataset(directory)
    assert dataset.dataset_id == "toy" and dataset.feature_shape == X.shape
    np.testing.assert_array_equal(dataset.load_features(), X)
    np.testing.assert_array_equal(dataset.load_labels().values, y)


def test_json_helpers_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "record.json"
    run_metadata.write_json(path, {"b": np.float32(1.5), "a": np.arange(3), "p": tmp_path})
    assert run_metadata.read_json(path) == {"a": [0, 1, 2], "b": 1.5, "p": str(tmp_path)}
    assert run_metadata.read_json(tmp_path / "missing.json") == {}
    assert "ibumap" in run_metadata.runtime_metadata()["package_versions"]


def test_machine_info_records_hardware_without_host_names() -> None:
    info = gpu_runtime.machine_info()
    assert {"cpu_model", "logical_cpus", "memory_total_bytes", "gpu"} <= set(info)
    assert "hostname" not in info and "python_executable" not in info


def test_unknown_algorithm_families_are_rejected() -> None:
    X, _ = _blobs(5)
    for family in ("legacy_fft", "tsne", ""):
        with pytest.raises(ValueError, match="Unsupported algorithm family"):
            algorithm_adapters.run_algorithm_entry(X, {"name": "legacy", "family": family})


def test_run_ibumap_and_fixed_inputs_round_trip(tmp_path: Path) -> None:
    X, y = _blobs()
    params = {"n_neighbors": 10, "n_epochs": 50, "random_state": 0}
    embedding, extras = algorithm_adapters.run_ibumap(X, params, device="cpu")
    assert embedding.shape == (X.shape[0], 2) and embedding.dtype == np.float32
    assert embedding.flags["C_CONTIGUOUS"] and np.isfinite(embedding).all()
    assert extras["used_params"]["algorithm"] == "ibumap" and isinstance(extras["time_costs"], dict)

    knn_indices, knn_distances, graph = fixed_inputs.build_knn_and_graph(X, params)
    init = fixed_inputs.spectral_initialization(X, graph, params)
    assert knn_indices.shape == (X.shape[0], 10) and init.shape == (X.shape[0], 2)
    fixed_inputs.save_fixed_inputs(root=tmp_path, dataset_name="blobs", n_neighbors=10, knn_indices=knn_indices,
                                   knn_distances=knn_distances, graph=graph, init_embedding=init,
                                   metadata={"dataset": "blobs", "n_neighbors": 10})
    assert fixed_inputs.fixed_inputs_complete(tmp_path, "blobs", 10)
    assert fixed_inputs.fixed_input_metadata_matches(fixed_inputs.fixed_input_paths(tmp_path, "blobs", 10)["metadata"],
                                                     {"n_neighbors": 10})
    loaded = fixed_inputs.load_fixed_inputs(tmp_path, "blobs", 10)
    np.testing.assert_array_equal(loaded["init_embedding"], init)
    assert (loaded["fuzzy_graph"] != graph.tocsr()).nnz == 0

    optimized, extras = algorithm_adapters.run_ibumap_optimize_from_graph(X, loaded["fuzzy_graph"],
                                                                          loaded["init_embedding"], params)
    assert optimized.shape == init.shape and np.isfinite(optimized).all()
    assert "effective_config" in extras

    a, b = fixed_inputs.find_ab_params(1.0, 0.1)
    assert a > 0 and b > 0
