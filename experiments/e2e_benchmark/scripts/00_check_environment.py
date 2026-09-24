#!/usr/bin/env python3
"""Validate configs, datasets, dependencies, signatures, and optional tiny probes."""

from __future__ import annotations

from _terminal_log import reexec_with_terminal_log

if __name__ == "__main__":
    reexec_with_terminal_log(__file__)

import argparse
import importlib
import inspect
import os
from pathlib import Path
from typing import Any

import numpy as np

from _common import (
    EXPERIMENT_ROOT,
    PROTOCOL_VERSION,
    algorithm_entries,
    algorithm_params,
    benchmark_configuration_hash,
    benchmark_protocol_hash,
    configuration_hash,
    cuda_graph_extension_status,
    dataset_entries,
    enforce_required_backend,
    ensure_experiment_dirs,
    environment_snapshot,
    experiment_paths,
    load_configs,
    load_dataset,
    requires_cuda_graph_extension,
    run_algorithm,
    summary_configuration_hash,
    validate_configs,
)
from common.paths import repo_relative
from common.run_metadata import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=EXPERIMENT_ROOT / "configs")
    parser.add_argument("--dataset", action="append", dest="dataset_filters")
    parser.add_argument("--algorithm", action="append", dest="algorithm_filters")
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument("--probe", action="store_true", help="Run tiny graph-to-embedding probes.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on any issue.")
    parser.add_argument("--write-json", action="store_true")
    return parser.parse_args()


def import_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "module": getattr(module, "__name__", module_name),
        "version": getattr(module, "__version__", None),
    }


def signature_status(module_name: str, attribute: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, attribute)
        signature = inspect.signature(function)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "signature": str(signature),
        "parameters": list(signature.parameters),
        "defaults": {
            name: (
                "<required>"
                if parameter.default is inspect.Parameter.empty
                else parameter.default
                if parameter.default is None
                or isinstance(parameter.default, (str, int, float, bool))
                else repr(parameter.default)
            )
            for name, parameter in signature.parameters.items()
        },
    }


def run_probes(algorithms: list[dict[str, Any]], configs: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(64, 8)).astype(np.float32)
    rows: list[dict[str, Any]] = []
    for entry in algorithms:
        name = str(entry["name"])
        try:
            enforce_required_backend(configs, entry)
            embedding, extras = run_algorithm(
                X,
                entry,
                algorithm_params(configs, entry, n_neighbors=int(common_n_neighbors(configs))),
            )
            rows.append(
                {
                    "algorithm": name,
                    "status": "ok",
                    "shape": list(embedding.shape),
                    "finite": bool(np.isfinite(embedding).all()),
                    "effective_params": extras.get("effective_params"),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "algorithm": name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
    return rows


def common_n_neighbors(configs: dict[str, Any]) -> int:
    return int((configs.get("algorithms") or {}).get("common_umap_params", {}).get("n_neighbors", 15))


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    paths = experiment_paths(config_dir, configs)
    ensure_experiment_dirs(paths)
    datasets = dataset_entries(configs, args.dataset_filters)
    algorithms = algorithm_entries(configs, args.algorithm_filters, profiles=args.profiles)
    issues = validate_configs(configs)

    dataset_rows: list[dict[str, Any]] = []
    for entry in datasets:
        name = str(entry.get("name"))
        try:
            dataset_id, features, dataset, metadata = load_dataset(entry, paths, mmap=True)
            dataset_rows.append(
                {
                    "dataset": dataset_id,
                    "status": "ok",
                    "shape": list(features.shape),
                    "dtype": str(features.dtype),
                    "features_path": repo_relative(dataset.features_path),
                    "metadata": metadata,
                }
            )
        except Exception as exc:
            issues.append(f"{name}: {type(exc).__name__}: {exc}")
            dataset_rows.append(
                {
                    "dataset": name,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    imports = {
        "umap": import_status("umap"),
        "cuml": import_status("cuml"),
        "cupy": import_status("cupy"),
        "cuvs": import_status("cuvs"),
        "ibumap": import_status("ibumap"),
        "torch": import_status("torch"),
        "torchdr": import_status("torchdr"),
    }
    if imports["torch"]["available"]:
        import torch

        imports["torch"]["cuda_available"] = bool(torch.cuda.is_available())
        imports["torch"]["cuda_device_count"] = int(torch.cuda.device_count())
    families = {str(entry.get("family")) for entry in algorithms}
    cuda_ibumap_selected = any(
        str(entry.get("family")) == "ibumap" and str(entry.get("device")) == "cuda"
        for entry in algorithms
    )
    if "umap_learn" in families and not imports["umap"]["available"]:
        issues.append("umap-learn is required by selected algorithms")
    if "cuml_umap" in families or cuda_ibumap_selected:
        missing_cuda_graph_packages = [
            name
            for name in ("cuml", "cupy", "cuvs")
            if not imports[name]["available"]
        ]
        if missing_cuda_graph_packages:
            issues.append(
                "Missing packages required by selected CUDA graph algorithms: "
                f"{', '.join(missing_cuda_graph_packages)}. RAPIDS >= 24.12 "
                "requires the cuVS Python package, not only libcuvs"
            )
    if "ibumap" in families and not imports["ibumap"]["available"]:
        issues.append("ibumap is required by selected ibUMAP algorithms")
    if "torchdr_umap" in families:
        if not imports["torch"]["available"] or not imports["torchdr"]["available"]:
            issues.append("PyTorch and TorchDR are required by selected TorchDR algorithms")
        elif not imports["torch"].get("cuda_available", False):
            issues.append("CUDA-enabled PyTorch is required by selected TorchDR algorithms")

    cuda_graph_extension_required = any(
        requires_cuda_graph_extension(configs, entry) for entry in algorithms
    )
    cuda_graph_extension = (
        cuda_graph_extension_status()
        if cuda_graph_extension_required
        else {"required": False, "available": None}
    )
    if cuda_graph_extension_required:
        cuda_graph_extension["required"] = True
        if not cuda_graph_extension.get("available"):
            issues.append(
                "CUDA ibUMAP cuML graph extension preflight failed: "
                f"{cuda_graph_extension.get('error')}"
            )
        if os.environ.get("IBUMAP_DISABLE_CUML_GRAPH_EXT") == "1":
            issues.append(
                "IBUMAP_DISABLE_CUML_GRAPH_EXT=1 conflicts with the required "
                "CUDA ibUMAP graph extension"
            )

    signatures = {
        "umap_learn": signature_status("umap", "UMAP"),
        "cuml": signature_status("cuml.manifold", "UMAP"),
        "torchdr": signature_status("torchdr", "UMAP"),
    }
    if signatures["umap_learn"].get("available"):
        if "random_state" not in signatures["umap_learn"]["parameters"]:
            issues.append("umap-learn UMAP does not expose random_state")
        elif signatures["umap_learn"]["defaults"].get("random_state") is not None:
            issues.append("umap-learn random_state default is not None")
    if signatures["cuml"].get("available"):
        defaults = signatures["cuml"]["defaults"]
        supports_force_serial = "force_serial_epochs" in signatures["cuml"]["parameters"]
        supports_build_algo = "build_algo" in signatures["cuml"]["parameters"]
        signatures["cuml"]["capabilities"] = {
            "force_serial_epochs": supports_force_serial,
            "force_serial_epochs_default": defaults.get("force_serial_epochs"),
            "build_algo": supports_build_algo,
            "build_algo_default": defaults.get("build_algo"),
            "random_state_default": defaults.get("random_state"),
        }
        if defaults.get("random_state") is not None:
            issues.append("cuML UMAP random_state default is not None")

    probes: list[dict[str, Any]] = []
    if args.probe:
        try:
            probes = run_probes([dict(entry) for entry in algorithms], configs)
        except Exception as exc:
            issues.append(f"Could not prepare tiny probe inputs: {type(exc).__name__}: {exc}")
        for row in probes:
            if row["status"] != "ok" or not row.get("finite", False):
                issues.append(f"Probe failed for {row['algorithm']}: {row.get('error_message', 'non-finite output')}")

    payload = {
        "status": "ok" if not issues else "issues",
        "configuration_hash": configuration_hash(configs),
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_configuration_hash": benchmark_configuration_hash(configs),
        "summary_configuration_hash": summary_configuration_hash(configs),
        "benchmark_protocol_hashes": {
            str(entry["name"]): benchmark_protocol_hash(configs, entry)
            for entry in algorithms
        },
        "experiment_root": repo_relative(paths.experiment_root),
        "processed_root": repo_relative(paths.processed_root),
        "datasets": dataset_rows,
        "algorithms": [dict(entry) for entry in algorithms],
        "imports": imports,
        "signatures": signatures,
        "cuda_graph_extension": cuda_graph_extension,
        "probes": probes,
        "issues": issues,
        "environment": environment_snapshot(),
    }

    print(f"Datasets: {len(datasets)} ({sum(row['status'] == 'ok' for row in dataset_rows)} available)")
    print(f"Algorithms: {len(algorithms)}")
    for name, status in imports.items():
        print(f"- import {name}: {'OK' if status['available'] else status['error']}")
    if signatures["cuml"].get("available"):
        capability = signatures["cuml"]["capabilities"]
        print(
            "- cuML defaults: "
            f"random_state={capability['random_state_default']!r}, "
            f"force_serial_epochs={capability['force_serial_epochs_default']!r}, "
            f"build_algo={capability['build_algo_default']!r}"
        )
    if probes:
        for row in probes:
            print(f"- probe {row['algorithm']}: {row['status']}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("Environment check passed.")

    if args.write_json:
        output = paths.summaries_root / "environment_check.json"
        write_json(output, payload)
        print(f"Wrote {output}")
    if args.strict and issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
