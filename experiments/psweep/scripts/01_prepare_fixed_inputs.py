#!/usr/bin/env python3
"""Build one kNN graph, fuzzy graph and spectral initialization per dataset (shared by all variants)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _common import (
    EXPERIMENT_ROOT,
    atomic_save_npy,
    atomic_save_npz,
    canonical_hash,
    dataset_entries,
    ensure_directories,
    experiment_paths,
    file_sha256,
    fixed_bundle_paths,
    fixed_identity,
    fixed_input_params,
    load_configs,
    load_dataset,
    read_json,
)
from common.fixed_inputs import build_knn_and_graph, spectral_initialization
from common.paths import repo_relative
from common.run_metadata import now_utc, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=EXPERIMENT_ROOT / "configs")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run", action="store_true")
    return parser.parse_args()


def complete(files: dict[str, Path]) -> bool:
    return all(files[key].exists() for key in ("knn_indices", "knn_distances", "fuzzy_graph", "init_embedding", "metadata"))


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    paths = experiment_paths(configs, config_dir)
    ensure_directories(paths)
    datasets = dataset_entries(configs, args.datasets)
    params = fixed_input_params(configs)
    print(f"datasets={len(datasets)} mode={'run' if args.run else 'dry-run'}")
    failures = 0
    for entry in datasets:
        name = str(entry["name"])
        files = fixed_bundle_paths(configs, paths, name)
        expected = fixed_identity(configs, entry)
        existing_matches = False
        if complete(files):
            existing_matches = read_json(files["metadata"]).get("identity_hash") == expected["identity_hash"]
        status = "complete" if existing_matches else "stale" if complete(files) else "pending"
        if not args.run:
            print(f"- dataset={name} rows={entry['expected_rows']} status={status}")
            continue
        if existing_matches and not args.overwrite:
            print(f"- SKIP dataset={name} matching fixed inputs")
            continue
        if complete(files) and not args.overwrite:
            raise RuntimeError(f"Stale fixed inputs for {name}; pass --overwrite")
        try:
            dataset_id, features, dataset = load_dataset(entry, paths, mmap=False)
            if int(features.shape[0]) != int(entry["expected_rows"]):
                raise ValueError(f"row mismatch: {features.shape[0]} != {entry['expected_rows']}")
            print(f"- START dataset={dataset_id} shape={features.shape}")
            started = time.perf_counter()
            graph_started = time.perf_counter()
            knn_indices, knn_distances, graph = build_knn_and_graph(features, params)
            graph = graph.tocsr().astype(np.float32, copy=False)
            graph.sum_duplicates()
            graph.sort_indices()
            graph_s = time.perf_counter() - graph_started
            init_started = time.perf_counter()
            init = np.asarray(
                spectral_initialization(features, graph.copy(), params),
                dtype=np.float32,
                order="C",
            )
            init_s = time.perf_counter() - init_started
            if init.shape != (features.shape[0], 2) or not np.isfinite(init).all():
                raise ValueError(f"invalid initialization shape/values: {init.shape}")
            atomic_save_npy(files["knn_indices"], np.asarray(knn_indices, dtype=np.int64))
            atomic_save_npy(files["knn_distances"], np.asarray(knn_distances, dtype=np.float32))
            atomic_save_npz(files["fuzzy_graph"], graph)
            atomic_save_npy(files["init_embedding"], init)
            metadata = {
                **expected,
                "status": "ok",
                "created_at": now_utc(),
                "dataset_dir": repo_relative(dataset.dataset_dir),
                "features_path": repo_relative(dataset.features_path),
                "feature_shape": list(features.shape),
                "feature_sha256": file_sha256(dataset.features_path),
                "graph_shape": list(graph.shape),
                "graph_nnz": int(graph.nnz),
                "graph_sha256": file_sha256(files["fuzzy_graph"]),
                "init_sha256": file_sha256(files["init_embedding"]),
                "params_hash": canonical_hash(params),
                "timings": {
                    "graph_s": graph_s,
                    "initialization_s": init_s,
                    "total_s": time.perf_counter() - started,
                },
            }
            write_json(files["metadata"], metadata)
            print(f"- END dataset={dataset_id} total_s={metadata['timings']['total_s']:.3f}")
        except Exception as exc:
            failures += 1
            write_json(
                files["directory"] / "error.json",
                {
                    "status": "failed",
                    "stage": "prepare_fixed_inputs",
                    "dataset": name,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "created_at": now_utc(),
                },
            )
            print(f"- FAILED dataset={name}: {type(exc).__name__}: {exc}")
            if not bool((configs.get("experiment") or {}).get("continue_on_error", True)):
                raise
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
