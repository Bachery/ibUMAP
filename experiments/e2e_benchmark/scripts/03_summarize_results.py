#!/usr/bin/env python3
"""Validate all run and evaluation records and write an immutable summary bundle.

With ``--strict`` every record is checked against the current configuration,
source hashes, feature and embedding hashes and runtime/hardware signatures.
Only a strict pass over the full task grid is published
(``results/03_summaries/published.json``); ``07_export_paper_data.py`` converts
the published bundle to the layout read by ``scripts/paper``.
"""

from __future__ import annotations

from _terminal_log import reexec_with_terminal_log

if __name__ == "__main__":
    reexec_with_terminal_log(__file__)

import argparse
import hashlib
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from scipy.spatial import cKDTree

from _common import (
    EXPERIMENT_ROOT,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_STABILITY_MAX_POINTS,
    DEFAULT_STABILITY_PAIRS,
    PROTOCOL_VERSION,
    RESOURCE_SKIP_ALGORITHMS,
    RESOURCE_SKIP_DATASETS,
    RESOURCE_SKIP_REASON,
    BenchmarkTask,
    algorithm_entries,
    atomic_write_csv,
    atomic_write_jsonl,
    benchmark_configuration_hash,
    benchmark_protocol_hash,
    build_tasks,
    configured_evaluation_request_hash,
    dataset_entries,
    embedding_path,
    ensure_experiment_dirs,
    evaluation_error_path,
    evaluation_path,
    evaluation_runtime_signature,
    experiment_paths,
    file_sha256,
    hardware_signature,
    load_configs,
    load_dataset,
    n_neighbors_values,
    read_json,
    recorded_dataset_id,
    repeat_values,
    requires_cuda_graph_extension,
    resource_skipped_task_count,
    runtime_signature,
    run_error_path,
    run_record_path,
    summary_configuration_hash,
    task_signature,
    validate_configs,
)
from common.paths import repo_relative, resolve_repo_path
from common.run_metadata import now_utc, write_json


HIGHER_IS_BETTER = {
    "trustworthiness": True,
    "continuity": True,
    "neighborhood_preservation": True,
    "rta": True,
    "distance_spearman": True,
}
METRIC_SOURCE = {
    "trustworthiness": "local",
    "continuity": "local",
    "neighborhood_preservation": "local",
    "rta": "global",
    "distance_spearman": "global",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=EXPERIMENT_ROOT / "configs")
    parser.add_argument("--dataset", action="append", dest="dataset_filters")
    parser.add_argument("--algorithm", action="append", dest="algorithm_filters")
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument("--repeat", type=int, action="append", dest="repeat_filters")
    parser.add_argument(
        "--stability-max-points",
        type=int,
        default=DEFAULT_STABILITY_MAX_POINTS,
    )
    parser.add_argument(
        "--stability-pairs",
        type=int,
        default=DEFAULT_STABILITY_PAIRS,
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    import json

    return json.dumps(value, sort_keys=True)


def get_nested(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping):
            return default
        current = current.get(key)
    return default if current is None else current


def flatten_success(task: BenchmarkTask, payload: Mapping[str, Any]) -> dict[str, Any]:
    entry = payload.get("algorithm_entry") if isinstance(payload.get("algorithm_entry"), Mapping) else task.algorithm_entry
    params = payload.get("requested_params") if isinstance(payload.get("requested_params"), Mapping) else {}
    effective = payload.get("effective_params") if isinstance(payload.get("effective_params"), Mapping) else {}
    sanity = payload.get("embedding_sanity") if isinstance(payload.get("embedding_sanity"), Mapping) else {}
    feature = payload.get("feature_stats") if isinstance(payload.get("feature_stats"), Mapping) else {}
    e2e_wall = payload.get("e2e_call_wall_s", payload.get("optimization_call_wall_s"))
    row: dict[str, Any] = {
        "dataset": task.dataset_id,
        "run_id": task.run_id,
        "algorithm_id": task.algorithm_id,
        "family": entry.get("family"),
        "device": entry.get("device"),
        "baseline": bool(entry.get("baseline", False)),
        "execution_profile": entry.get("execution_profile"),
        "seed_mode": entry.get("seed_mode"),
        "random_state": params.get("random_state"),
        "deterministic": params.get("deterministic"),
        "deterministic_reason": effective.get("deterministic_reason"),
        "parallel": effective.get("parallel"),
        "force_serial_epochs": params.get("force_serial_epochs"),
        "force_serial_epochs_keyword": effective.get(
            "force_serial_epochs_keyword"
        ),
        "force_serial_epochs_resolution": effective.get(
            "force_serial_epochs_resolution"
        ),
        "force_serial_epochs_supported": effective.get("force_serial_epochs_supported"),
        "force_serial_epochs_applied": effective.get("force_serial_epochs_applied"),
        "repeat": task.repeat,
        "execution_index": payload.get("execution_index", task.execution_index),
        "n_neighbors": task.n_neighbors,
        "status": "ok",
        "e2e_call_wall_s": e2e_wall,
        "optimization_call_wall_s": e2e_wall,
        "optimizer_compute_s": payload.get("optimizer_compute_s"),
        "n_samples": (get_nested(payload, "dataset", "feature_shape", default=[None]) or feature.get("shape", [None]))[0],
        "n_features": (
            int(np.prod(get_nested(payload, "dataset", "feature_shape", default=[None, None])[1:]))
            if len(get_nested(payload, "dataset", "feature_shape", default=[])) > 1
            else int(np.prod(feature.get("shape", [None, None])[1:]))
            if len(feature.get("shape", [])) > 1
            else None
        ),
        "graph_nnz": get_nested(payload, "time_costs", "graph_preprocess_nnz"),
        "feature_sha256": payload.get("feature_sha256"),
        "embedding_finite": sanity.get("finite"),
        "bbox_area": sanity.get("bbox_area"),
        "robust_bbox_area": sanity.get("robust_bbox_area"),
        "embedding_path": repo_relative(embedding_path(CURRENT_PATHS, task)),
        "embedding_sha256": payload.get("embedding_sha256"),
        "record_path": repo_relative(run_record_path(CURRENT_PATHS, task)),
        "protocol_version": payload.get("protocol_version"),
        "benchmark_configuration_hash": payload.get(
            "benchmark_configuration_hash"
        ),
        "benchmark_protocol_hash": payload.get("benchmark_protocol_hash"),
        "task_signature": payload.get("task_signature"),
        "runtime_signature": payload.get("runtime_signature"),
        "hardware_signature": payload.get("hardware_signature"),
        "error_type": None,
        "error_message": None,
    }
    time_costs = payload.get("time_costs") if isinstance(payload.get("time_costs"), Mapping) else {}
    for key, value in time_costs.items():
        row[f"time_{key}"] = value
    return {key: scalar(value) for key, value in row.items()}


def flatten_failure(task: BenchmarkTask, payload: Mapping[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
    entry = metadata.get("algorithm_entry") if isinstance(metadata.get("algorithm_entry"), Mapping) else task.algorithm_entry
    params = metadata.get("requested_params") if isinstance(metadata.get("requested_params"), Mapping) else {}
    return {
        "dataset": task.dataset_id,
        "run_id": task.run_id,
        "algorithm_id": task.algorithm_id,
        "family": entry.get("family"),
        "device": entry.get("device"),
        "baseline": bool(entry.get("baseline", False)),
        "execution_profile": entry.get("execution_profile"),
        "seed_mode": entry.get("seed_mode"),
        "random_state": params.get("random_state"),
        "deterministic": params.get("deterministic"),
        "deterministic_reason": None,
        "parallel": None,
        "force_serial_epochs": params.get("force_serial_epochs"),
        "force_serial_epochs_keyword": None,
        "force_serial_epochs_resolution": None,
        "force_serial_epochs_supported": None,
        "force_serial_epochs_applied": None,
        "repeat": task.repeat,
        "execution_index": metadata.get("execution_index", task.execution_index),
        "n_neighbors": task.n_neighbors,
        "status": "failed",
        "e2e_call_wall_s": metadata.get("e2e_call_wall_s", metadata.get("optimization_call_wall_s")),
        "optimization_call_wall_s": metadata.get("e2e_call_wall_s", metadata.get("optimization_call_wall_s")),
        "optimizer_compute_s": None,
        "n_samples": get_nested(metadata, "dataset", "feature_shape", default=[None])[0],
        "n_features": None,
        "graph_nnz": None,
        "feature_sha256": metadata.get("feature_sha256"),
        "embedding_finite": False,
        "embedding_path": None,
        "embedding_sha256": None,
        "record_path": repo_relative(run_error_path(CURRENT_PATHS, task)),
        "protocol_version": metadata.get("protocol_version"),
        "benchmark_configuration_hash": metadata.get(
            "benchmark_configuration_hash"
        ),
        "benchmark_protocol_hash": metadata.get("benchmark_protocol_hash"),
        "task_signature": metadata.get("task_signature"),
        "runtime_signature": metadata.get("runtime_signature"),
        "hardware_signature": metadata.get("hardware_signature"),
        "error_type": payload.get("error_type"),
        "error_message": payload.get("error_message"),
    }


def missing_row(task: BenchmarkTask, status: str) -> dict[str, Any]:
    entry = task.algorithm_entry
    params = entry.get("params") if isinstance(entry.get("params"), Mapping) else {}
    return {
        "dataset": task.dataset_id,
        "run_id": task.run_id,
        "algorithm_id": task.algorithm_id,
        "family": entry.get("family"),
        "device": entry.get("device"),
        "baseline": bool(entry.get("baseline", False)),
        "execution_profile": entry.get("execution_profile"),
        "seed_mode": entry.get("seed_mode"),
        "random_state": params.get("random_state"),
        "deterministic": params.get("deterministic"),
        "deterministic_reason": None,
        "parallel": None,
        "force_serial_epochs": params.get("force_serial_epochs"),
        "force_serial_epochs_keyword": None,
        "force_serial_epochs_resolution": None,
        "force_serial_epochs_supported": None,
        "force_serial_epochs_applied": None,
        "repeat": task.repeat,
        "execution_index": task.execution_index,
        "n_neighbors": task.n_neighbors,
        "status": status,
        "e2e_call_wall_s": None,
        "optimization_call_wall_s": None,
        "optimizer_compute_s": None,
        "n_samples": None,
        "n_features": None,
        "graph_nnz": None,
        "feature_sha256": None,
        "embedding_finite": None,
        "bbox_area": None,
        "robust_bbox_area": None,
        "embedding_path": None,
        "embedding_sha256": None,
        "record_path": None,
        "protocol_version": None,
        "benchmark_configuration_hash": None,
        "benchmark_protocol_hash": None,
        "task_signature": None,
        "runtime_signature": None,
        "hardware_signature": None,
        "error_type": None,
        "error_message": None,
    }


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summary_stats(values: Sequence[float], *, seed: int, bootstrap_samples: int) -> dict[str, Any]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if not array.size:
        return {key: None for key in ("n", "mean", "std", "median", "q25", "q75", "min", "max", "cv", "ci95_low", "ci95_high")}
    mean = float(array.mean())
    if bootstrap_samples > 0 and array.size > 1:
        rng = np.random.default_rng(seed)
        medians = np.median(rng.choice(array, size=(bootstrap_samples, array.size), replace=True), axis=1)
        low, high = np.percentile(medians, [2.5, 97.5])
    else:
        low = high = float(np.median(array))
    return {
        "n": int(array.size),
        "mean": mean,
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q25": percentile(array, 25),
        "q75": percentile(array, 75),
        "min": float(array.min()),
        "max": float(array.max()),
        "cv": float(array.std(ddof=1) / mean) if array.size > 1 and mean else 0.0,
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def grouped_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_key: str,
    extra_key: str | None,
    bootstrap_samples: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope, keys in (
        ("per_dataset", ("dataset", "algorithm_id", "execution_profile", "device")),
        ("all_datasets", ("algorithm_id", "execution_profile", "device")),
    ):
        groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        for row in rows:
            if row.get("status") != "ok" or row.get(value_key) is None:
                continue
            key = tuple(row.get(name) for name in keys)
            if extra_key:
                key += (row.get(extra_key),)
            try:
                groups[key].append(float(row[value_key]))
            except (TypeError, ValueError):
                pass
        for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
            names = list(keys) + ([extra_key] if extra_key else [])
            base = {name: value for name, value in zip(names, key)}
            if scope == "all_datasets":
                base["dataset"] = "__all__"
            seed = int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], "little")
            output.append({"scope": scope, **base, **summary_stats(values, seed=seed, bootstrap_samples=bootstrap_samples)})
    return output


def comparison_roles(row: Mapping[str, Any]) -> str | None:
    algorithm = str(row.get("algorithm_id"))
    if algorithm.startswith("umap_learn"):
        return "cpu_baseline"
    if algorithm.startswith("ibumap_cpu"):
        return "cpu_candidate"
    if algorithm.startswith("cuml"):
        return "cuda_baseline"
    if algorithm.startswith("ibumap_cuda"):
        return "cuda_candidate"
    if algorithm.startswith("torchdr"):
        return "cuda_baseline_torchdr"
    return None


def paired_runtime_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "ok":
            continue
        role = comparison_roles(row)
        if role:
            indexed[(row.get("dataset"), row.get("execution_profile"), row.get("repeat"))][role] = row
    output: list[dict[str, Any]] = []
    for (dataset, profile, repeat), group in indexed.items():
        for device in ("cpu", "cuda"):
            candidate = group.get(f"{device}_candidate")
            baselines = [
                row
                for role, row in group.items()
                if role.startswith(f"{device}_baseline")
            ]
            if not candidate:
                continue
            c = float(candidate.get("e2e_call_wall_s") or candidate["optimization_call_wall_s"])
            for baseline in baselines:
                b = float(
                    baseline.get("e2e_call_wall_s")
                    or baseline["optimization_call_wall_s"]
                )
                output.append(
                    {
                        "dataset": dataset,
                        "execution_profile": profile,
                        "repeat": repeat,
                        "device": device,
                        "baseline_algorithm": baseline["algorithm_id"],
                        "candidate_algorithm": candidate["algorithm_id"],
                        "baseline_time_s": b,
                        "candidate_time_s": c,
                        "speedup": b / c if c else None,
                        "time_delta_s": c - b,
                        "n_samples": baseline.get("n_samples"),
                        "graph_nnz": baseline.get("graph_nnz"),
                    }
                )
    return sorted(
        output,
        key=lambda row: (
            str(row["dataset"]),
            str(row["execution_profile"]),
            int(row["repeat"]),
            str(row["device"]),
            str(row["baseline_algorithm"]),
            str(row["candidate_algorithm"]),
        ),
    )


def quality_rows_for_tasks(
    tasks: Sequence[BenchmarkTask],
    metrics: Sequence[str],
    *,
    benchmark_config_hash: str,
    summary_config_hash: str,
    evaluation_request_hash: str,
    run_contracts: Mapping[tuple[str, str], Mapping[str, Any]],
    issues: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        output = evaluation_path(CURRENT_PATHS, task)
        error = evaluation_error_path(CURRENT_PATHS, task)
        payload: dict[str, Any] = {}
        status = "missing"
        if output.exists() and error.exists():
            status = "duplicate"
            issues.append(f"Duplicate evaluation success/error: {task.dataset_id}/{task.run_id}")
        elif output.exists():
            payload = read_json(output)
            status = "ok" if payload.get("status") == "ok" else "invalid"
            contract = run_contracts.get((task.dataset_id, task.run_id))
            evaluation_environment = (
                payload.get("environment")
                if isinstance(payload.get("environment"), Mapping)
                else {}
            )
            evaluation_device = (
                payload.get("device")
                if isinstance(payload.get("device"), Mapping)
                else {}
            )
            expected = {
                "protocol_version": PROTOCOL_VERSION,
                "benchmark_configuration_hash": benchmark_config_hash,
                "summary_configuration_hash": summary_config_hash,
                "evaluation_request_hash": evaluation_request_hash,
                "dataset": task.dataset_id,
                "run_id": task.run_id,
                "algorithm_id": task.algorithm_id,
                "repeat": task.repeat,
                "n_neighbors": task.n_neighbors,
                "evaluation_runtime_signature": evaluation_runtime_signature(
                    evaluation_environment,
                    evaluation_device,
                ),
            }
            if contract is not None:
                expected.update(
                    {
                        "benchmark_protocol_hash": contract.get(
                            "benchmark_protocol_hash"
                        ),
                        "task_signature": contract.get("task_signature"),
                        "runtime_signature": contract.get("runtime_signature"),
                        "hardware_signature": contract.get("hardware_signature"),
                        "feature_sha256": contract.get("feature_sha256"),
                        "embedding_sha256": contract.get("embedding_sha256"),
                    }
                )
            mismatches = [
                key for key, value in expected.items() if payload.get(key) != value
            ]
            evaluation_computed = payload.get("evaluation_computed")
            if not isinstance(evaluation_computed, bool):
                mismatches.append("evaluation_computed")
            elif evaluation_computed:
                if payload.get("evaluation_reused_from") is not None:
                    mismatches.append("evaluation_reused_from")
            else:
                reused_from = payload.get("evaluation_reused_from")
                if not isinstance(reused_from, Mapping):
                    mismatches.append("evaluation_reused_from")
                else:
                    source_path_value = reused_from.get("evaluation_path")
                    recorded_source_path = (
                        Path(str(source_path_value))
                        if source_path_value
                        else None
                    )
                    expected_source_path = output.parent / (
                        f"{reused_from.get('run_id')}.json"
                    )
                    source_path = expected_source_path
                    if (
                        recorded_source_path is None
                        or recorded_source_path.name != expected_source_path.name
                        or recorded_source_path.parent.name != task.dataset_id
                        or source_path == output
                        or not source_path.is_file()
                    ):
                        mismatches.append("evaluation_reused_from.evaluation_path")
                    else:
                        source_payload = read_json(source_path)
                        reuse_expected = {
                            "status": "ok",
                            "dataset": task.dataset_id,
                            "run_id": reused_from.get("run_id"),
                            "repeat": reused_from.get("repeat"),
                            "embedding_sha256": payload.get("embedding_sha256"),
                            "benchmark_configuration_hash": benchmark_config_hash,
                            "summary_configuration_hash": summary_config_hash,
                            "evaluation_request_hash": evaluation_request_hash,
                            "evaluation_runtime_signature": payload.get(
                                "evaluation_runtime_signature"
                            ),
                            "evaluation_computed": True,
                            "sample_size": payload.get("sample_size"),
                            "sample_seed": payload.get("sample_seed"),
                            "source_key": payload.get("source_key"),
                            "scores": payload.get("scores"),
                            "score_scalars": payload.get("score_scalars"),
                        }
                        mismatches.extend(
                            "evaluation_reused_from." + key
                            for key, value in reuse_expected.items()
                            if source_payload.get(key) != value
                        )
                    if reused_from.get("embedding_sha256") != payload.get(
                        "embedding_sha256"
                    ):
                        mismatches.append(
                            "evaluation_reused_from.embedding_sha256"
                        )
            if contract is None:
                mismatches.append("valid_upstream_run")
            if mismatches:
                status = "invalid"
                issues.append(
                    f"Evaluation contract mismatch: {task.dataset_id}/{task.run_id}: "
                    + ", ".join(sorted(set(mismatches)))
                )
        elif error.exists():
            payload = read_json(error)
            status = "failed"
        for metric in metrics:
            value = get_nested(payload, "score_scalars", metric)
            metric_status = status
            if metric_status == "ok" and value is None:
                metric_status = "invalid"
                issues.append(
                    f"Evaluation metric missing: "
                    f"{task.dataset_id}/{task.run_id}/{metric}"
                )
            rows.append(
                {
                    "dataset": task.dataset_id,
                    "run_id": task.run_id,
                    "algorithm_id": task.algorithm_id,
                    "family": task.algorithm_entry.get("family"),
                    "device": task.algorithm_entry.get("device"),
                    "execution_profile": task.execution_profile,
                    "repeat": task.repeat,
                    "metric": metric,
                    "source": METRIC_SOURCE.get(metric),
                    "higher_is_better": HIGHER_IS_BETTER.get(metric),
                    "value": value,
                    "status": metric_status,
                    "sample_size": payload.get("sample_size"),
                    "sample_seed": payload.get("sample_seed"),
                    "evaluation_wall_s": payload.get("evaluation_wall_s"),
                    "evaluation_compute_wall_s": payload.get(
                        "evaluation_compute_wall_s"
                    ),
                    "evaluation_computed": payload.get("evaluation_computed"),
                    "evaluation_reused_from_run_id": get_nested(
                        payload,
                        "evaluation_reused_from",
                        "run_id",
                    ),
                    "evaluation_request_hash": payload.get(
                        "evaluation_request_hash"
                    ),
                    "evaluation_runtime_signature": payload.get(
                        "evaluation_runtime_signature"
                    ),
                    "task_signature": payload.get("task_signature"),
                    "embedding_sha256": payload.get("embedding_sha256"),
                    "evaluation_path": repo_relative(output if output.exists() else error),
                    "error_type": payload.get("error_type"),
                    "error_message": payload.get("error_message"),
                }
            )
    return rows


def paired_quality_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "ok" or row.get("value") is None:
            continue
        role = comparison_roles(row)
        if role:
            indexed[(row.get("dataset"), row.get("execution_profile"), row.get("repeat"), row.get("metric"))][role] = row
    output: list[dict[str, Any]] = []
    for (dataset, profile, repeat, metric), group in indexed.items():
        for device in ("cpu", "cuda"):
            candidate = group.get(f"{device}_candidate")
            baselines = [
                row
                for role, row in group.items()
                if role.startswith(f"{device}_baseline")
            ]
            if not candidate:
                continue
            c = float(candidate["value"])
            for baseline in baselines:
                b = float(baseline["value"])
                higher = bool(HIGHER_IS_BETTER.get(str(metric), True))
                signed_delta = c - b if higher else b - c
                tolerance = 1e-6
                outcome = (
                    "win"
                    if signed_delta > tolerance
                    else "loss"
                    if signed_delta < -tolerance
                    else "tie"
                )
                output.append(
                    {
                        "dataset": dataset,
                        "execution_profile": profile,
                        "repeat": repeat,
                        "metric": metric,
                        "device": device,
                        "baseline_algorithm": baseline["algorithm_id"],
                        "candidate_algorithm": candidate["algorithm_id"],
                        "baseline_value": b,
                        "candidate_value": c,
                        "raw_delta_candidate_minus_baseline": c - b,
                        "signed_improvement": signed_delta,
                        "outcome": outcome,
                        "higher_is_better": higher,
                    }
                )
    return sorted(
        output,
        key=lambda row: (
            str(row["metric"]),
            str(row["dataset"]),
            str(row["execution_profile"]),
            int(row["repeat"]),
            str(row["device"]),
            str(row["baseline_algorithm"]),
            str(row["candidate_algorithm"]),
        ),
    )


def stability_metrics(
    reference_path: Path,
    candidate_path: Path,
    *,
    max_points: int,
    pair_count: int,
    seed: int,
    hashes_equal: bool,
) -> dict[str, Any]:
    reference = np.load(reference_path, mmap_mode="r")
    candidate = np.load(candidate_path, mmap_mode="r")
    if reference.shape != candidate.shape:
        raise ValueError(f"Embedding shape mismatch: {reference.shape} vs {candidate.shape}")
    n = int(reference.shape[0])
    rng = np.random.default_rng(seed)
    indices = np.arange(n) if max_points <= 0 or n <= max_points else np.sort(rng.choice(n, max_points, replace=False))
    x = np.asarray(reference[indices], dtype=np.float64)
    y = np.asarray(candidate[indices], dtype=np.float64)
    raw_diff = y - x
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if not x_norm or not y_norm:
        raise ValueError("Cannot align a zero-scale embedding")
    x /= x_norm
    y /= y_norm
    u, _, vt = np.linalg.svd(y.T @ x, full_matrices=False)
    aligned = y @ (u @ vt)
    alignment_diff = aligned - x
    m = len(indices)
    pairs = min(max(pair_count, 1), max(m * 4, 1))
    left = rng.integers(0, m, size=pairs)
    right = rng.integers(0, m, size=pairs)
    different = left != right
    dx = np.linalg.norm(x[left[different]] - x[right[different]], axis=1)
    dy = np.linalg.norm(y[left[different]] - y[right[different]], axis=1)
    if dx.size > 1:
        correlation_result = spearmanr(dx, dy)
        correlation = float(
            correlation_result.statistic
            if hasattr(correlation_result, "statistic")
            else correlation_result[0]
        )
    else:
        correlation = None
    k = min(15, max(m - 1, 0))
    neighbor_overlap = None
    if k > 0:
        x_neighbors = cKDTree(x).query(x, k=k + 1)[1][:, 1:]
        y_neighbors = cKDTree(y).query(y, k=k + 1)[1][:, 1:]
        overlaps = [
            len(set(x_neighbors[index].tolist()) & set(y_neighbors[index].tolist())) / k
            for index in range(m)
        ]
        neighbor_overlap = float(np.mean(overlaps))
    return {
        "bitwise_equal": bool(hashes_equal),
        "sample_size": int(m),
        "raw_rms": float(np.sqrt(np.mean(raw_diff * raw_diff))),
        "raw_max_abs": float(np.max(np.abs(raw_diff))),
        "procrustes_rms": float(np.sqrt(np.mean(alignment_diff * alignment_diff))),
        "procrustes_max_abs": float(np.max(np.abs(alignment_diff))),
        "pairwise_distance_spearman": correlation,
        "neighbor_overlap_at_15": neighbor_overlap,
    }


def build_stability_rows(
    run_rows: Sequence[Mapping[str, Any]], *, max_points: int, pair_count: int, issues: list[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        if row.get("status") == "ok":
            groups[(str(row["dataset"]), str(row["algorithm_id"]), str(row["execution_profile"]))].append(row)
    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: int(row["repeat"]))
        if len(ordered) < 2:
            continue
        reference = ordered[0]
        for candidate in ordered[1:]:
            seed = int.from_bytes(hashlib.sha256(f"{key}:{candidate['repeat']}".encode()).digest()[:8], "little")
            try:
                metrics = stability_metrics(
                    resolve_repo_path(str(reference["embedding_path"])),
                    resolve_repo_path(str(candidate["embedding_path"])),
                    max_points=max_points,
                    pair_count=pair_count,
                    seed=seed,
                    hashes_equal=reference.get("embedding_sha256") == candidate.get("embedding_sha256"),
                )
                output.append(
                    {
                        "dataset": key[0],
                        "algorithm_id": key[1],
                        "execution_profile": key[2],
                        "reference_repeat": reference["repeat"],
                        "candidate_repeat": candidate["repeat"],
                        "status": "ok",
                        **metrics,
                    }
                )
            except Exception as exc:
                issues.append(f"Stability failed {key}/{candidate['repeat']}: {type(exc).__name__}: {exc}")
                output.append(
                    {
                        "dataset": key[0],
                        "algorithm_id": key[1],
                        "execution_profile": key[2],
                        "reference_repeat": reference["repeat"],
                        "candidate_repeat": candidate["repeat"],
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
    return output


def dataset_summary_rows(
    datasets: Sequence[Mapping[str, Any]], run_rows: Sequence[Mapping[str, Any]], quality_rows: Sequence[Mapping[str, Any]], n_neighbors: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for entry in datasets:
        dataset = str(entry["name"])
        metadata: dict[str, Any] = {}
        shape: list[int] = []
        labels_count = None
        try:
            _, features, loaded_dataset, metadata = load_dataset(entry, CURRENT_PATHS, mmap=True)
            shape = [int(value) for value in features.shape]
            labels = loaded_dataset.load_labels()
            labels_count = None if labels is None else int(len(labels.values))
        except Exception as exc:
            metadata = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        runs = [row for row in run_rows if row.get("dataset") == dataset]
        evaluations = [row for row in quality_rows if row.get("dataset") == dataset]
        output.append(
            {
                "dataset": dataset,
                "n_samples": shape[0] if shape else None,
                "n_features": int(np.prod(shape[1:])) if len(shape) > 1 else None,
                "graph_nnz": None,
                "labels_count": labels_count,
                "dataset_status": metadata.get("status", "ok"),
                "expected_runs": len(runs),
                "ok_runs": sum(row.get("status") == "ok" for row in runs),
                "failed_runs": sum(row.get("status") == "failed" for row in runs),
                "missing_runs": sum(row.get("status") == "missing" for row in runs),
                "ok_metric_rows": sum(row.get("status") == "ok" for row in evaluations),
                "missing_metric_rows": sum(row.get("status") == "missing" for row in evaluations),
                "n_neighbors": int(n_neighbors),
            }
        )
    return output


def main() -> None:
    global CURRENT_PATHS
    args = parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    config_issues = validate_configs(configs)
    if config_issues:
        raise ValueError("Invalid experiment configuration: " + "; ".join(config_issues))
    CURRENT_PATHS = experiment_paths(config_dir, configs)
    ensure_experiment_dirs(CURRENT_PATHS)
    datasets = dataset_entries(configs, args.dataset_filters)
    algorithms = algorithm_entries(configs, args.algorithm_filters, profiles=args.profiles)
    repeats = repeat_values(configs, args.repeat_filters)
    neighbors = n_neighbors_values(configs)
    if len(neighbors) != 1:
        raise ValueError("Summarization expects one n_neighbors value")
    tasks = build_tasks(configs, datasets, algorithms, repeats, n_neighbors=neighbors[0])
    resource_skipped = resource_skipped_task_count(datasets, algorithms, repeats)
    benchmark_config_hash = benchmark_configuration_hash(configs)
    summary_config_hash = summary_configuration_hash(configs)
    expected_evaluation_hash = configured_evaluation_request_hash(configs)
    issues: list[str] = []
    run_rows: list[dict[str, Any]] = []
    dataset_by_id = {str(entry["name"]): entry for entry in datasets}
    feature_hash_cache: dict[str, str] = {}

    def current_feature_hash(task: BenchmarkTask) -> str:
        if task.dataset_id not in feature_hash_cache:
            _, features, loaded_dataset, _ = load_dataset(
                dataset_by_id[task.dataset_id],
                CURRENT_PATHS,
                mmap=True,
            )
            feature_hash_cache[task.dataset_id] = file_sha256(
                loaded_dataset.features_path
            )
            del features
        return feature_hash_cache[task.dataset_id]

    for task in tasks:
        success = run_record_path(CURRENT_PATHS, task)
        error = run_error_path(CURRENT_PATHS, task)
        if success.exists() and error.exists():
            row = missing_row(task, "duplicate")
            issues.append(f"Duplicate run success/error: {task.dataset_id}/{task.run_id}")
        elif success.exists():
            payload = read_json(success)
            row = flatten_success(task, payload)
            protocol_hash = benchmark_protocol_hash(configs, task.algorithm_entry)
            feature_sha256 = (
                current_feature_hash(task)
                if args.strict
                else payload.get("feature_sha256")
            )
            expected_task_signature = task_signature(
                configs,
                task,
                feature_sha256=feature_sha256,
                protocol_hash=protocol_hash,
            )
            run_environment = (
                payload.get("environment")
                if isinstance(payload.get("environment"), Mapping)
                else {}
            )
            expected_runtime_signature = runtime_signature(
                task.algorithm_entry,
                run_environment,
            )
            expected_hardware_signature = hardware_signature(run_environment)
            expected_fields = {
                "status": "ok",
                "protocol_version": PROTOCOL_VERSION,
                "benchmark_configuration_hash": benchmark_config_hash,
                "benchmark_protocol_hash": protocol_hash,
                "task_signature": expected_task_signature,
                "runtime_signature": expected_runtime_signature,
                "hardware_signature": expected_hardware_signature,
                "feature_sha256": feature_sha256,
                "run_id": task.run_id,
                "algorithm_id": task.algorithm_id,
                "repeat": task.repeat,
                "n_neighbors": task.n_neighbors,
            }
            mismatches = [
                key for key, value in expected_fields.items() if payload.get(key) != value
            ]
            if recorded_dataset_id(payload) != task.dataset_id:
                mismatches.append("dataset_id")
            if requires_cuda_graph_extension(configs, task.algorithm_entry):
                preflight = (
                    payload.get("backend_preflight")
                    if isinstance(payload.get("backend_preflight"), Mapping)
                    else {}
                )
                if (
                    preflight.get("required") is not True
                    or preflight.get("available") is not True
                ):
                    mismatches.append("backend_preflight")
            if mismatches:
                row["status"] = "invalid"
                issues.append(
                    f"Run contract mismatch: {task.dataset_id}/{task.run_id}: "
                    + ", ".join(sorted(set(mismatches)))
                )
            path = embedding_path(CURRENT_PATHS, task)
            if not path.exists():
                row["status"] = "invalid"
                issues.append(f"Missing embedding for success record: {task.dataset_id}/{task.run_id}")
            elif args.strict:
                expected_embedding_hash = payload.get("embedding_sha256")
                if (
                    not expected_embedding_hash
                    or file_sha256(path) != expected_embedding_hash
                ):
                    row["status"] = "invalid"
                    issues.append(
                        f"Embedding hash mismatch: {task.dataset_id}/{task.run_id}"
                    )
        elif error.exists():
            row = flatten_failure(task, read_json(error))
            issues.append(f"Failed run: {task.dataset_id}/{task.run_id}")
        else:
            row = missing_row(task, "missing")
            issues.append(f"Missing run: {task.dataset_id}/{task.run_id}")
        run_rows.append(row)

    validation_config = (configs.get("experiment") or {}).get("validation") or {}
    if bool(validation_config.get("enforce_runtime_signature_per_algorithm", True)):
        algorithms_with_mixed_runtime: set[str] = set()
        grouped_runtime: dict[str, set[str]] = defaultdict(set)
        for row in run_rows:
            if row.get("status") == "ok" and row.get("runtime_signature"):
                grouped_runtime[str(row["algorithm_id"])].add(
                    str(row["runtime_signature"])
                )
        for algorithm_id, signatures in grouped_runtime.items():
            if len(signatures) > 1:
                algorithms_with_mixed_runtime.add(algorithm_id)
                issues.append(
                    f"Mixed runtime signatures for {algorithm_id}: "
                    f"{len(signatures)} signatures"
                )
        for row in run_rows:
            if (
                row.get("status") == "ok"
                and str(row.get("algorithm_id")) in algorithms_with_mixed_runtime
            ):
                row["status"] = "invalid"

    if bool(validation_config.get("enforce_single_hardware_signature", True)):
        hardware_signatures = {
            str(row["hardware_signature"])
            for row in run_rows
            if row.get("status") == "ok" and row.get("hardware_signature")
        }
        if len(hardware_signatures) > 1:
            issues.append(
                f"Mixed hardware signatures: {len(hardware_signatures)} signatures"
            )
            for row in run_rows:
                if row.get("status") == "ok":
                    row["status"] = "invalid"

    counts = defaultdict(int)
    for row in run_rows:
        counts[str(row["status"])] += 1
    run_contracts = {
        (str(row["dataset"]), str(row["run_id"])): row
        for row in run_rows
        if row.get("status") == "ok"
    }
    metrics = [str(metric) for metric in (configs.get("evaluation") or {}).get("metrics", [])]
    quality_rows = quality_rows_for_tasks(
        tasks,
        metrics,
        benchmark_config_hash=benchmark_config_hash,
        summary_config_hash=summary_config_hash,
        evaluation_request_hash=expected_evaluation_hash,
        run_contracts=run_contracts,
        issues=issues,
    )
    evaluation_runtime_signatures = {
        str(row["evaluation_runtime_signature"])
        for row in quality_rows
        if row.get("status") == "ok"
        and row.get("evaluation_runtime_signature")
    }
    if bool(
        validation_config.get(
            "enforce_single_evaluation_runtime_signature",
            True,
        )
    ):
        if len(evaluation_runtime_signatures) > 1:
            issues.append(
                "Mixed evaluation runtime signatures: "
                f"{len(evaluation_runtime_signatures)} signatures"
            )
            for row in quality_rows:
                if row.get("status") == "ok":
                    row["status"] = "invalid"
    successful_run_ids = {(row["dataset"], row["run_id"]) for row in run_rows if row.get("status") == "ok"}
    for row in quality_rows:
        if (
            (row["dataset"], row["run_id"]) in successful_run_ids
            and row["status"] != "ok"
            and row["metric"] == metrics[0]
        ):
            issues.append(
                f"Non-successful evaluation ({row['status']}): "
                f"{row['dataset']}/{row['run_id']}"
            )

    runtime_summary = grouped_summary(
        run_rows,
        value_key="e2e_call_wall_s",
        extra_key=None,
        bootstrap_samples=args.bootstrap_samples,
    )
    speedup_rows = paired_runtime_rows(run_rows)
    quality_summary = grouped_summary(
        quality_rows,
        value_key="value",
        extra_key="metric",
        bootstrap_samples=args.bootstrap_samples,
    )
    quality_pairs = paired_quality_rows(quality_rows)
    stability_rows = [] if args.skip_stability else build_stability_rows(
        run_rows,
        max_points=args.stability_max_points,
        pair_count=args.stability_pairs,
        issues=issues,
    )
    dataset_rows = dataset_summary_rows(datasets, run_rows, quality_rows, neighbors[0])

    created_at = now_utc()
    bundle_id = (
        created_at.replace(":", "").replace("+", "_")
        + "__"
        + summary_config_hash[:12]
        + "__"
        + uuid.uuid4().hex[:8]
    )
    strict_passed = bool(args.strict and not issues)
    standard_analysis = (
        args.stability_max_points == DEFAULT_STABILITY_MAX_POINTS
        and args.stability_pairs == DEFAULT_STABILITY_PAIRS
        and args.bootstrap_samples == DEFAULT_BOOTSTRAP_SAMPLES
        and not args.skip_stability
    )
    full_scope = not any(
        (
            args.dataset_filters,
            args.algorithm_filters,
            args.profiles,
            args.repeat_filters,
        )
    ) and standard_analysis
    publishable = bool(strict_passed and full_scope)
    bundle_class = (
        "bundles"
        if publishable
        else "scoped_bundles"
        if strict_passed
        else "validation_failures"
        if args.strict
        else "drafts"
    )
    output = CURRENT_PATHS.summaries_root / bundle_class / bundle_id
    atomic_write_jsonl(output / "runs.jsonl", run_rows)
    atomic_write_csv(output / "run_summary.csv", run_rows)
    atomic_write_csv(output / "runtime_summary.csv", runtime_summary)
    atomic_write_csv(output / "speedup_pairs.csv", speedup_rows)
    atomic_write_csv(output / "quality_long.csv", quality_rows)
    atomic_write_csv(output / "quality_summary.csv", quality_summary)
    atomic_write_csv(output / "quality_pairs.csv", quality_pairs)
    atomic_write_csv(output / "stability_summary.csv", stability_rows)
    atomic_write_csv(output / "dataset_summary.csv", dataset_rows)
    completeness = {
        "created_at": created_at,
        "protocol_version": PROTOCOL_VERSION,
        "benchmark_configuration_hash": benchmark_config_hash,
        "summary_configuration_hash": summary_config_hash,
        "evaluation_request_hash": expected_evaluation_hash,
        "scope": {
            "datasets": [str(entry["name"]) for entry in datasets],
            "algorithms": [str(entry["name"]) for entry in algorithms],
            "repeats": repeats,
            "n_neighbors": neighbors[0],
            "analysis": {
                "stability_max_points": args.stability_max_points,
                "stability_pairs": args.stability_pairs,
                "bootstrap_samples": args.bootstrap_samples,
                "skip_stability": bool(args.skip_stability),
                "standard_policy": standard_analysis,
            },
            "resource_skip_rules": {
                "algorithms": list(RESOURCE_SKIP_ALGORITHMS),
                "datasets": list(RESOURCE_SKIP_DATASETS),
                "reason": RESOURCE_SKIP_REASON,
            },
        },
        "expected_runs": len(tasks),
        "resource_skipped_runs": resource_skipped,
        "run_status_counts": dict(counts),
        "successful_runs": len(successful_run_ids),
        "expected_metric_rows_for_successes": len(successful_run_ids) * len(metrics),
        "ok_metric_rows": sum(row["status"] == "ok" for row in quality_rows),
        "failed_metric_rows": sum(row["status"] == "failed" for row in quality_rows),
        "missing_metric_rows": sum(row["status"] == "missing" for row in quality_rows),
        "invalid_metric_rows": sum(row["status"] == "invalid" for row in quality_rows),
        "evaluation_runtime_signature_count": len(
            evaluation_runtime_signatures
        ),
        "issues_count": len(issues),
        "issues": issues,
        "validation_passed": not issues,
        "strict_checks_executed": bool(args.strict),
        "strict_passed": strict_passed,
        "full_scope": full_scope,
        "published": publishable,
    }
    write_json(output / "completeness.json", completeness)
    if publishable:
        write_json(
            CURRENT_PATHS.summaries_root / "published.json",
            {
                "created_at": created_at,
                "protocol_version": PROTOCOL_VERSION,
                "summary_configuration_hash": summary_config_hash,
                "strict_passed": True,
                "bundle": str(
                    output.relative_to(CURRENT_PATHS.summaries_root)
                ),
            },
        )
    print(
        f"Expected runs: {len(tasks)}; resource-skipped: {resource_skipped}; "
        f"status counts: {dict(counts)}; "
        f"quality rows ok: {completeness['ok_metric_rows']}; issues: {len(issues)}"
    )
    print(
        f"Wrote "
        f"{'published' if publishable else 'scoped' if strict_passed else 'diagnostic'} "
        f"summaries to {output}"
    )
    if args.strict and issues:
        raise SystemExit(1)


CURRENT_PATHS: Any = None


if __name__ == "__main__":
    main()
