#!/usr/bin/env python3
"""Build completeness, runtime/memory, quality and stability summaries (results/04_summaries/).

``06_export_paper_data.py`` converts ``runs.csv`` and ``quality_scores.csv`` to
the layout read by ``scripts/paper``.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

from _common import (
    EXPERIMENT_ROOT,
    BenchmarkTask,
    algorithm_entries,
    atomic_write_csv,
    build_tasks,
    configuration_hash,
    dataset_entries,
    embedding_path,
    ensure_directories,
    evaluation_error_path,
    evaluation_path,
    experiment_paths,
    load_configs,
    read_json,
    repeat_seeds,
    run_error_path,
    run_record_path,
)
from common.paths import repo_relative, resolve_repo_path
from common.run_metadata import now_utc, write_json


HIGHER_IS_BETTER = {
    "trustworthiness": True,
    "continuity": True,
    "neighborhood_preservation": True,
}
STABILITY_METRICS = (
    "procrustes_rms",
    "pairwise_distance_spearman",
    "neighbor_overlap_at_15",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=EXPERIMENT_ROOT / "configs")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--algorithm", action="append", dest="algorithms")
    parser.add_argument("--device", action="append", choices=("cpu", "cuda"), dest="devices")
    parser.add_argument("--variant", action="append", choices=("p1", "p2", "p3", "schedule"), dest="variants")
    parser.add_argument("--repeat", action="append", type=int, dest="repeats")
    parser.add_argument("--skip-stability", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        current = current.get(key) if isinstance(current, Mapping) else None
    return current


def run_row(task: BenchmarkTask, paths: Any, config_hash: str, issues: list[str]) -> dict[str, Any]:
    success = run_record_path(paths, task)
    error = run_error_path(paths, task)
    if success.exists() and error.exists():
        issues.append(f"duplicate run success/failure: {task.dataset_id}/{task.run_id}")
    if success.exists():
        payload = read_json(success)
        status = "ok" if payload.get("status") == "ok" else "invalid"
        if payload.get("configuration_hash") != config_hash:
            status = "stale"
            issues.append(f"stale run config: {task.dataset_id}/{task.run_id}")
    elif error.exists():
        payload = read_json(error)
        status = "failed"
    else:
        payload = {}
        status = "missing"
    memory = payload.get("memory") if isinstance(payload.get("memory"), Mapping) else nested(payload, "metadata", "memory") or {}
    sanity = payload.get("embedding_sanity") if isinstance(payload.get("embedding_sanity"), Mapping) else {}
    time_costs = payload.get("time_costs") if isinstance(payload.get("time_costs"), Mapping) else {}
    return {
        "dataset": task.dataset_id,
        "dataset_family": task.dataset_entry["family"],
        "size_bin": task.dataset_entry["size_bin"],
        "expected_rows": int(task.dataset_entry["expected_rows"]),
        "run_id": task.run_id,
        "algorithm_id": task.algorithm_id,
        "device": task.device,
        "variant": task.variant,
        "repeat": task.repeat,
        "random_state": task.seed,
        "status": status,
        "optimization_wall_s": payload.get("optimization_wall_s"),
        "worker_wall_s": payload.get("worker_wall_s") or nested(payload, "metadata", "worker_wall_s"),
        "subprocess_wall_s": payload.get("subprocess_wall_s"),
        "fft_stage_schedule": time_costs.get("fft_stage_schedule"),
        "fft_stage_count": time_costs.get("fft_stage_count"),
        "fft_stage_transition_count": time_costs.get("fft_stage_transition_count"),
        "fft_stage_transition_epochs": time_costs.get("fft_stage_transition_epochs"),
        "fft_stage_epochs_p1": time_costs.get("fft_stage_epochs_p1"),
        "fft_stage_epochs_p2": time_costs.get("fft_stage_epochs_p2"),
        "fft_stage_epochs_p3": time_costs.get("fft_stage_epochs_p3"),
        "fft_kernel_cache_peak_bytes": time_costs.get("repl_kernel_cache_peak_bytes"),
        "memory_primary_source": memory.get("primary_source"),
        "memory_primary_peak_bytes": memory.get("primary_peak_bytes"),
        "memory_primary_peak_delta_bytes": memory.get("primary_peak_delta_bytes"),
        "cpu_peak_rss_bytes": memory.get("cpu_peak_bytes"),
        "cpu_peak_rss_delta_bytes": memory.get("cpu_peak_delta_bytes"),
        "gpu_peak_process_bytes": memory.get("gpu_process_peak_bytes"),
        "gpu_peak_process_delta_bytes": memory.get("gpu_process_peak_delta_bytes"),
        "gpu_peak_pool_total_bytes": memory.get("gpu_pool_total_peak_bytes"),
        "gpu_peak_pool_total_delta_bytes": memory.get("gpu_pool_total_peak_delta_bytes"),
        "memory_samples": memory.get("samples"),
        "embedding_finite": sanity.get("finite"),
        "bbox_area": sanity.get("bbox_area"),
        "embedding_path": payload.get("embedding_path"),
        "embedding_sha256": payload.get("embedding_sha256"),
        "record_path": repo_relative(success if success.exists() else error),
        "error_type": payload.get("error_type"),
        "error_message": payload.get("error_message"),
    }


def quality_rows(
    tasks: Sequence[BenchmarkTask], paths: Any, metrics: Sequence[str], issues: list[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for task in tasks:
        success = evaluation_path(paths, task)
        error = evaluation_error_path(paths, task)
        if success.exists() and error.exists():
            issues.append(f"duplicate evaluation success/failure: {task.dataset_id}/{task.run_id}")
        if success.exists():
            payload = read_json(success)
            status = "ok" if payload.get("status") == "ok" else "invalid"
        elif error.exists():
            payload = read_json(error)
            status = "failed"
        else:
            payload = {}
            status = "missing"
        for metric in metrics:
            output.append(
                {
                    "dataset": task.dataset_id,
                    "dataset_family": task.dataset_entry["family"],
                    "size_bin": task.dataset_entry["size_bin"],
                    "expected_rows": int(task.dataset_entry["expected_rows"]),
                    "run_id": task.run_id,
                    "algorithm_id": task.algorithm_id,
                    "device": task.device,
                    "variant": task.variant,
                    "repeat": task.repeat,
                    "random_state": task.seed,
                    "metric": metric,
                    "higher_is_better": HIGHER_IS_BETTER.get(metric),
                    "value": nested(payload, "score_scalars", metric),
                    "status": status,
                    "sample_size": payload.get("sample_size"),
                    "evaluation_wall_s": payload.get("evaluation_wall_s"),
                    "evaluation_path": repo_relative(success if success.exists() else error),
                    "error_type": payload.get("error_type"),
                    "error_message": payload.get("error_message"),
                }
            )
    return output


def stability_metrics(
    reference_path: Path,
    candidate_path: Path,
    *,
    max_points: int,
    pair_samples: int,
    n_neighbors: int,
    seed: int,
    hashes_equal: bool,
) -> dict[str, Any]:
    reference = np.load(reference_path, mmap_mode="r")
    candidate = np.load(candidate_path, mmap_mode="r")
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} != {candidate.shape}")
    n = int(reference.shape[0])
    rng = np.random.default_rng(seed)
    indices = (
        np.arange(n)
        if max_points <= 0 or n <= max_points
        else np.sort(rng.choice(n, max_points, replace=False))
    )
    x_raw = np.asarray(reference[indices], dtype=np.float64)
    y_raw = np.asarray(candidate[indices], dtype=np.float64)
    raw_diff = y_raw - x_raw
    x = x_raw - x_raw.mean(axis=0, keepdims=True)
    y = y_raw - y_raw.mean(axis=0, keepdims=True)
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if not x_norm or not y_norm:
        raise ValueError("cannot align a zero-scale embedding")
    x /= x_norm
    y /= y_norm
    u, _, vt = np.linalg.svd(y.T @ x, full_matrices=False)
    aligned = y @ (u @ vt)
    alignment_diff = aligned - x
    m = len(indices)
    pair_count = min(max(pair_samples, 1), max(m * 4, 1))
    left = rng.integers(0, m, size=pair_count)
    right = rng.integers(0, m, size=pair_count)
    valid = left != right
    dx = np.linalg.norm(x[left[valid]] - x[right[valid]], axis=1)
    dy = np.linalg.norm(y[left[valid]] - y[right[valid]], axis=1)
    correlation = None
    if dx.size > 1:
        result = spearmanr(dx, dy)
        correlation = float(result.statistic if hasattr(result, "statistic") else result[0])
    k = min(int(n_neighbors), max(m - 1, 0))
    neighbor_overlap = None
    if k > 0:
        x_neighbors = cKDTree(x).query(x, k=k + 1)[1][:, 1:]
        y_neighbors = cKDTree(y).query(y, k=k + 1)[1][:, 1:]
        neighbor_overlap = float(
            np.mean(
                [
                    len(set(x_neighbors[i].tolist()) & set(y_neighbors[i].tolist())) / k
                    for i in range(m)
                ]
            )
        )
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
    run_rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], issues: list[str]
) -> list[dict[str, Any]]:
    reference_repeat = int(config.get("reference_repeat", 1))
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in run_rows:
        if row.get("status") == "ok":
            groups[(str(row["dataset"]), str(row["algorithm_id"]))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, algorithm), rows in sorted(groups.items()):
        reference = next(
            (row for row in rows if int(row["repeat"]) == reference_repeat), None
        )
        if reference is None:
            issues.append(f"stability reference missing: {dataset}/{algorithm}")
            continue
        for candidate in sorted(rows, key=lambda row: int(row["repeat"])):
            if int(candidate["repeat"]) == reference_repeat:
                continue
            material = f"{dataset}:{algorithm}:{candidate['repeat']}".encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "little")
            try:
                metrics = stability_metrics(
                    resolve_repo_path(str(reference["embedding_path"])),
                    resolve_repo_path(str(candidate["embedding_path"])),
                    max_points=int(config.get("max_points", 10000)),
                    pair_samples=int(config.get("pair_samples", 20000)),
                    n_neighbors=int(config.get("n_neighbors", 15)),
                    seed=seed,
                    hashes_equal=reference.get("embedding_sha256") == candidate.get("embedding_sha256"),
                )
                output.append(
                    {
                        "dataset": dataset,
                        "dataset_family": reference["dataset_family"],
                        "size_bin": reference["size_bin"],
                        "expected_rows": reference["expected_rows"],
                        "algorithm_id": algorithm,
                        "device": reference["device"],
                        "variant": reference["variant"],
                        "reference_repeat": reference_repeat,
                        "candidate_repeat": candidate["repeat"],
                        "status": "ok",
                        **metrics,
                    }
                )
            except Exception as exc:
                issues.append(f"stability failed {dataset}/{algorithm}: {type(exc).__name__}: {exc}")
                output.append(
                    {
                        "dataset": dataset,
                        "algorithm_id": algorithm,
                        "device": reference["device"],
                        "variant": reference["variant"],
                        "reference_repeat": reference_repeat,
                        "candidate_repeat": candidate["repeat"],
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
    return output


def stats(values: Sequence[Any]) -> dict[str, Any]:
    array = np.asarray(
        [float(value) for value in values if value is not None and np.isfinite(float(value))],
        dtype=float,
    )
    if not array.size:
        return {key: None for key in ("n", "mean", "std", "median", "q25", "q75", "min", "max")}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def grouped_summary(
    rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str], value_keys: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[tuple(row.get(key) for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for group, members in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        prefix = dict(zip(group_keys, group))
        for value_key in value_keys:
            summary = stats([member.get(value_key) for member in members])
            output.append({**prefix, "measure": value_key, **summary})
    return output


def paired_runtime_memory(run_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in run_rows:
        if row.get("status") == "ok":
            groups[(row["dataset"], row["device"], row["repeat"])][str(row["variant"])] = row
    output: list[dict[str, Any]] = []
    for (dataset, device, repeat), variants in sorted(groups.items()):
        schedule = variants.get("schedule")
        if not schedule:
            continue
        for fixed_name in ("p1", "p2", "p3"):
            fixed = variants.get(fixed_name)
            if not fixed:
                continue
            fixed_time = float(fixed["optimization_wall_s"])
            schedule_time = float(schedule["optimization_wall_s"])
            fixed_memory = fixed.get("memory_primary_peak_bytes")
            schedule_memory = schedule.get("memory_primary_peak_bytes")
            output.append(
                {
                    "dataset": dataset,
                    "dataset_family": schedule["dataset_family"],
                    "size_bin": schedule["size_bin"],
                    "expected_rows": schedule["expected_rows"],
                    "device": device,
                    "repeat": repeat,
                    "fixed_variant": fixed_name,
                    "fixed_time_s": fixed_time,
                    "schedule_time_s": schedule_time,
                    "speedup_fixed_over_schedule": fixed_time / schedule_time if schedule_time else None,
                    "schedule_time_delta_s": schedule_time - fixed_time,
                    "fixed_peak_memory_bytes": fixed_memory,
                    "schedule_peak_memory_bytes": schedule_memory,
                    "schedule_memory_delta_bytes": (
                        None
                        if fixed_memory is None or schedule_memory is None
                        else int(schedule_memory) - int(fixed_memory)
                    ),
                    "fixed_over_schedule_memory_ratio": (
                        None
                        if fixed_memory is None or schedule_memory in (None, 0)
                        else float(fixed_memory) / float(schedule_memory)
                    ),
                }
            )
    return output


def paired_quality(quality: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in quality:
        if row.get("status") == "ok" and row.get("value") is not None:
            groups[(row["dataset"], row["device"], row["repeat"], row["metric"])][str(row["variant"])] = row
    output: list[dict[str, Any]] = []
    for (dataset, device, repeat, metric), variants in sorted(groups.items()):
        schedule = variants.get("schedule")
        if not schedule:
            continue
        for fixed_name in ("p1", "p2", "p3"):
            fixed = variants.get(fixed_name)
            if not fixed:
                continue
            fixed_value = float(fixed["value"])
            schedule_value = float(schedule["value"])
            higher = bool(HIGHER_IS_BETTER.get(str(metric), True))
            raw_delta = schedule_value - fixed_value
            output.append(
                {
                    "dataset": dataset,
                    "dataset_family": schedule["dataset_family"],
                    "size_bin": schedule["size_bin"],
                    "expected_rows": schedule["expected_rows"],
                    "device": device,
                    "repeat": repeat,
                    "metric": metric,
                    "higher_is_better": higher,
                    "fixed_variant": fixed_name,
                    "fixed_value": fixed_value,
                    "schedule_value": schedule_value,
                    "raw_schedule_minus_fixed": raw_delta,
                    "signed_schedule_improvement": raw_delta if higher else -raw_delta,
                }
            )
    return output


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    paths = experiment_paths(configs, config_dir)
    ensure_directories(paths)
    algorithms = algorithm_entries(
        configs, args.algorithms, devices=args.devices, variants=args.variants
    )
    datasets = dataset_entries(configs, args.datasets)
    repeats = args.repeats or list(range(1, len(repeat_seeds(configs)) + 1))
    tasks = build_tasks(configs, datasets, algorithms, repeats)
    config_hash = configuration_hash(configs)
    issues: list[str] = []
    runs = [run_row(task, paths, config_hash, issues) for task in tasks]
    metrics = [str(metric) for metric in (configs.get("evaluation") or {}).get("metrics", [])]
    quality = quality_rows(tasks, paths, metrics, issues)
    stability_config = (configs.get("experiment") or {}).get("stability") or {}
    stability = [] if args.skip_stability else build_stability_rows(runs, stability_config, issues)

    completeness = []
    for task, run in zip(tasks, runs):
        eval_statuses = {
            row["status"]
            for row in quality
            if row["dataset"] == task.dataset_id and row["run_id"] == task.run_id
        }
        completeness.append(
            {
                "dataset": task.dataset_id,
                "algorithm_id": task.algorithm_id,
                "device": task.device,
                "variant": task.variant,
                "repeat": task.repeat,
                "run_status": run["status"],
                "evaluation_status": next(iter(eval_statuses)) if len(eval_statuses) == 1 else "mixed",
            }
        )

    runtime_summary = grouped_summary(
        runs,
        ("device", "variant", "size_bin"),
        ("optimization_wall_s", "memory_primary_peak_bytes", "memory_primary_peak_delta_bytes"),
    )
    quality_summary = grouped_summary(
        quality,
        ("device", "variant", "size_bin", "metric"),
        ("value",),
    )
    stability_summary = grouped_summary(
        stability,
        ("device", "variant", "size_bin"),
        STABILITY_METRICS,
    )
    paired_runtime = paired_runtime_memory(runs)
    paired_scores = paired_quality(quality)

    outputs = {
        "completeness.csv": completeness,
        "runs.csv": runs,
        "runtime_memory_summary.csv": runtime_summary,
        "quality_scores.csv": quality,
        "quality_summary.csv": quality_summary,
        "stability.csv": stability,
        "stability_summary.csv": stability_summary,
        "paired_schedule_runtime_memory.csv": paired_runtime,
        "paired_schedule_quality.csv": paired_scores,
    }
    for name, rows in outputs.items():
        atomic_write_csv(paths.summaries_root / name, rows)
    status_counts = defaultdict(int)
    for row in completeness:
        status_counts[f"run_{row['run_status']}"] += 1
        status_counts[f"evaluation_{row['evaluation_status']}"] += 1
    manifest = {
        "created_at": now_utc(),
        "configuration_hash": config_hash,
        "task_count": len(tasks),
        "status_counts": dict(status_counts),
        "stability_comparisons": len(stability),
        "issues": issues,
        "outputs": {name: len(rows) for name, rows in outputs.items()},
    }
    write_json(paths.summaries_root / "manifest.json", manifest)
    print(
        f"tasks={len(tasks)} run_ok={status_counts.get('run_ok', 0)} "
        f"evaluation_ok={status_counts.get('evaluation_ok', 0)} issues={len(issues)}"
    )
    if args.strict:
        incomplete = [
            row
            for row in completeness
            if row["run_status"] != "ok" or row["evaluation_status"] != "ok"
        ]
        if issues or incomplete:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
