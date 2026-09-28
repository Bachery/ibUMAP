#!/usr/bin/env python3
"""Run the end-to-end benchmark task grid (one isolated worker process per task).

The controller validates existing results, then spawns this script again with
``--internal-worker`` for each pending dataset/algorithm/repeat task. The worker
loads the dataset, performs the configured warmup on a 512-row subset, times one
end-to-end call and writes the embedding plus a JSON run record.
"""

from __future__ import annotations

from _terminal_log import reexec_with_terminal_log, setup_terminal_logger

if __name__ == "__main__":
    reexec_with_terminal_log(__file__)

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from _common import (
    EXPERIMENT_ROOT,
    PROTOCOL_VERSION,
    BenchmarkTask,
    algorithm_entries,
    algorithm_params,
    atomic_save_npy,
    benchmark_configuration_hash,
    benchmark_protocol_hash,
    build_tasks,
    dataset_entries,
    embedding_path,
    embedding_sanity,
    enforce_required_backend,
    ensure_experiment_dirs,
    environment_snapshot,
    experiment_paths,
    failure_payload,
    feature_stats,
    file_sha256,
    gpu_memory_snapshot,
    hardware_signature,
    load_configs,
    load_dataset,
    n_neighbors_values,
    read_json,
    repeat_values,
    resolve_dataset_path,
    resource_skipped_task_count,
    resource_snapshot,
    runtime_signature,
    run_algorithm,
    run_error_path,
    run_record_path,
    task_signature,
    validate_configs,
    warmup_subset,
)
from common.dataset_io import load_processed_dataset
from common.gpu_runtime import force_cleanup
from common.paths import repo_relative
from common.run_metadata import now_utc, write_json


_TERMINAL_LOG_ACTIVE_SCRIPT_ENV = "IBUMAP_TERMINAL_LOG_ACTIVE_SCRIPT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=EXPERIMENT_ROOT / "configs")
    parser.add_argument("--dataset", action="append", dest="dataset_filters")
    parser.add_argument("--algorithm", action="append", dest="algorithm_filters")
    parser.add_argument("--profile", action="append", dest="profiles")
    parser.add_argument("--repeat", type=int, action="append", dest="repeat_filters")
    parser.add_argument("--repeats", type=int, help="Run repeats 1..N (convenience override).")
    parser.add_argument("--neighbors", type=int, action="append", dest="neighbor_filters")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--skip-feature-hash", action="store_true")
    parser.add_argument("--max-tasks", type=int, help="Debug-only cap after task ordering.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument(
        "--internal-worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-execution-index",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-feature-sha256",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-feature-stats-json",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def selected_repeats(args: argparse.Namespace, configs: Mapping[str, Any]) -> list[int]:
    if args.repeats is not None:
        if args.repeat_filters:
            raise ValueError("Use either --repeat or --repeats, not both")
        if args.repeats < 1:
            raise ValueError("--repeats must be >= 1")
        return list(range(1, args.repeats + 1))
    return repeat_values(configs, args.repeat_filters)


def record_matches(
    record: Mapping[str, Any],
    *,
    benchmark_config_hash: str,
    protocol_hash: str,
    expected_task_signature: str,
    expected_runtime_signature: str,
    output_embedding: Path,
) -> tuple[bool, str]:
    if record.get("status") != "ok":
        return False, "status is not ok"
    if record.get("protocol_version") != PROTOCOL_VERSION:
        return False, "protocol version mismatch"
    if record.get("benchmark_configuration_hash") != benchmark_config_hash:
        return False, "benchmark configuration hash mismatch"
    if record.get("benchmark_protocol_hash") != protocol_hash:
        return False, "benchmark protocol hash mismatch"
    if record.get("task_signature") != expected_task_signature:
        return False, "task signature mismatch"
    if record.get("runtime_signature") != expected_runtime_signature:
        return False, "runtime signature mismatch"
    if not output_embedding.exists():
        return False, "embedding is missing"
    expected_hash = record.get("embedding_sha256")
    if not expected_hash:
        return False, "embedding hash is missing"
    if file_sha256(output_embedding) != expected_hash:
        return False, "embedding hash mismatch"
    return True, "matching"


def primary_internal_time(time_costs: Mapping[str, Any]) -> float | None:
    for key in ("embedding_time", "fit_transform_time", "fit_time", "fixed_knn_total_time"):
        if time_costs.get(key) is not None:
            try:
                return float(time_costs[key])
            except (TypeError, ValueError):
                return None
    return None


def perform_warmups(
    *,
    task: BenchmarkTask,
    X: np.ndarray,
    params: Mapping[str, Any],
    paths: Any,
    count: int,
) -> dict[str, Any]:
    started_at = now_utc()
    payload: dict[str, Any] = {
        "algorithm": task.algorithm_id,
        "dataset": task.dataset_id,
        "run_id": task.run_id,
        "execution_profile": task.execution_profile,
        "started_at": started_at,
        "requested_runs": int(count),
        "runs": [],
    }
    include_gpu = str(task.algorithm_entry.get("device")) == "cuda"
    for warmup_index in range(1, int(count) + 1):
        started = time.perf_counter()
        row: dict[str, Any] = {"warmup_index": warmup_index}
        try:
            warm_X = warmup_subset(X)
            embedding, extras = run_algorithm(warm_X, task.algorithm_entry, params)
            row.update(
                {
                    "status": "ok",
                    "wall_time_s": float(time.perf_counter() - started),
                    "shape": list(embedding.shape),
                    "finite": bool(np.isfinite(embedding).all()),
                    "effective_params": extras.get("effective_params"),
                }
            )
            if not row["finite"]:
                raise ValueError("Warmup produced non-finite embedding")
        except Exception as exc:
            row.update(
                {
                    "status": "failed",
                    "wall_time_s": float(time.perf_counter() - started),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        finally:
            force_cleanup(include_gpu=include_gpu)
        payload["runs"].append(row)
        if row["status"] != "ok":
            break
    payload["status"] = (
        "ok"
        if len(payload["runs"]) == int(count)
        and all(row["status"] == "ok" for row in payload["runs"])
        else "failed"
    )
    payload["finished_at"] = now_utc()
    write_json(
        paths.embeddings_root
        / "warmups"
        / task.dataset_id
        / f"{task.run_id}.json",
        payload,
    )
    return payload


def dataset_feature_sha256(
    dataset_id: str,
    *,
    dataset_by_id: Mapping[str, Mapping[str, Any]],
    paths: Any,
    cache: dict[str, str | None],
    skip_feature_hash: bool,
) -> str | None:
    if skip_feature_hash:
        return None
    if dataset_id not in cache:
        entry = dataset_by_id[dataset_id]
        dataset = load_processed_dataset(
            resolve_dataset_path(entry, paths),
            processed_root=paths.processed_root,
        )
        cache[dataset_id] = file_sha256(dataset.features_path)
    return cache[dataset_id]


def dataset_feature_statistics(
    dataset_id: str,
    *,
    dataset_by_id: Mapping[str, Mapping[str, Any]],
    paths: Any,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if dataset_id not in cache:
        entry = dataset_by_id[dataset_id]
        _, features, _, _ = load_dataset(entry, paths, mmap=True)
        stats = feature_stats(features)
        if not stats["finite"]:
            raise ValueError(f"Dataset contains non-finite features: {dataset_id}")
        cache[dataset_id] = stats
        del features
        force_cleanup(include_gpu=False)
    return cache[dataset_id]


def isolated_worker_command(
    args: argparse.Namespace,
    task: BenchmarkTask,
    *,
    feature_sha256: str | None,
    feature_statistics: Mapping[str, Any],
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--internal-worker",
        "--internal-execution-index",
        str(task.execution_index),
        "--config-dir",
        str(args.config_dir.resolve()),
        "--dataset",
        task.dataset_id,
        "--algorithm",
        task.algorithm_id,
        "--repeat",
        str(task.repeat),
        "--neighbors",
        str(task.n_neighbors),
        "--run",
        "--internal-feature-stats-json",
        json.dumps(dict(feature_statistics), separators=(",", ":")),
    ]
    if feature_sha256 is not None:
        command.extend(["--internal-feature-sha256", feature_sha256])
    if args.include_disabled:
        command.append("--include-disabled")
    if args.overwrite:
        command.append("--overwrite")
    if args.skip_warmup:
        command.append("--skip-warmup")
    if args.skip_feature_hash:
        command.append("--skip-feature-hash")
    return command


def normalized_exit_code(return_code: int) -> tuple[int, int | None, str | None]:
    signal_number: int | None = None
    if return_code < 0:
        signal_number = abs(int(return_code))
        exit_code = 128 + signal_number
    else:
        exit_code = int(return_code)
        if exit_code >= 128:
            candidate = exit_code - 128
            if candidate in {item.value for item in signal.Signals}:
                signal_number = candidate
    signal_name: str | None = None
    if signal_number is not None:
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            pass
    return exit_code, signal_number, signal_name


def write_isolated_worker_failure(
    *,
    task: BenchmarkTask,
    paths: Any,
    configs: Mapping[str, Any],
    benchmark_config_hash: str,
    protocol_hash: str,
    expected_task_signature: str,
    expected_runtime_signature: str,
    expected_hardware_signature: str,
    return_code: int,
    reason: str | None = None,
) -> None:
    exit_code, signal_number, signal_name = normalized_exit_code(return_code)
    detail = reason or (
        f"isolated worker exited with code {exit_code}"
        + (f" ({signal_name or f'signal {signal_number}'})" if signal_number else "")
    )
    error = RuntimeError(detail)
    payload = failure_payload(
        stage="end_to_end",
        dataset_id=task.dataset_id,
        run_id=task.run_id,
        error=error,
        metadata={
            "protocol_version": PROTOCOL_VERSION,
            "benchmark_configuration_hash": benchmark_config_hash,
            "benchmark_protocol_hash": protocol_hash,
            "task_signature": expected_task_signature,
            "runtime_signature": expected_runtime_signature,
            "hardware_signature": expected_hardware_signature,
            "algorithm_entry": dict(task.algorithm_entry),
            "algorithm_id": task.algorithm_id,
            "execution_profile": task.execution_profile,
            "seed_mode": task.algorithm_entry.get("seed_mode"),
            "repeat": task.repeat,
            "execution_index": task.execution_index,
            "n_neighbors": task.n_neighbors,
            "requested_params": algorithm_params(
                configs,
                task.algorithm_entry,
                n_neighbors=task.n_neighbors,
            ),
            "isolated_worker_return_code": int(return_code),
            "isolated_worker_exit_code": exit_code,
            "isolated_worker_signal": signal_number,
            "isolated_worker_signal_name": signal_name,
            "likely_oom_kill": signal_number == signal.SIGKILL,
        },
    )
    payload["traceback"] = None
    write_json(run_error_path(paths, task), payload)


def run_isolated_tasks(
    *,
    args: argparse.Namespace,
    configs: Mapping[str, Any],
    paths: Any,
    tasks: Sequence[BenchmarkTask],
    datasets: Sequence[Mapping[str, Any]],
    benchmark_config_hash: str,
    continue_on_error: bool,
    skip_existing: bool,
) -> None:
    logger = setup_terminal_logger("end_to_end_controller")
    dataset_by_id = {str(entry["name"]): entry for entry in datasets}
    feature_hash_cache: dict[str, str | None] = {}
    feature_statistics_cache: dict[str, dict[str, Any]] = {}
    controller_environment = environment_snapshot()
    controller_hardware_signature = hardware_signature(controller_environment)
    completed = 0
    skipped = 0
    failures = 0

    for task in tasks:
        output_embedding = embedding_path(paths, task)
        output_record = run_record_path(paths, task)
        output_error = run_error_path(paths, task)
        feature_sha256 = dataset_feature_sha256(
            task.dataset_id,
            dataset_by_id=dataset_by_id,
            paths=paths,
            cache=feature_hash_cache,
            skip_feature_hash=args.skip_feature_hash,
        )
        protocol_hash = benchmark_protocol_hash(configs, task.algorithm_entry)
        expected_task_signature = task_signature(
            configs,
            task,
            feature_sha256=feature_sha256,
            protocol_hash=protocol_hash,
        )
        expected_runtime_signature = runtime_signature(
            task.algorithm_entry,
            controller_environment,
        )
        if output_record.exists() and skip_existing:
            matching, reason = record_matches(
                read_json(output_record),
                benchmark_config_hash=benchmark_config_hash,
                protocol_hash=protocol_hash,
                expected_task_signature=expected_task_signature,
                expected_runtime_signature=expected_runtime_signature,
                output_embedding=output_embedding,
            )
            if matching:
                skipped += 1
                output_error.unlink(missing_ok=True)
                logger.info(
                    "SKIP dataset=%s run_id=%s matching existing output",
                    task.dataset_id,
                    task.run_id,
                )
                continue
            raise RuntimeError(
                f"Existing output is stale or incomplete ({reason}); "
                f"pass --overwrite: {output_record}"
            )

        enforce_required_backend(configs, task.algorithm_entry)
        feature_statistics = dataset_feature_statistics(
            task.dataset_id,
            dataset_by_id=dataset_by_id,
            paths=paths,
            cache=feature_statistics_cache,
        )
        command = isolated_worker_command(
            args,
            task,
            feature_sha256=feature_sha256,
            feature_statistics=feature_statistics,
        )
        environment = os.environ.copy()
        # The top-level invocation already captures the terminal. Bypass only
        # the worker's additional tee parent so all worker output remains in
        # the one invocation log while the measured worker stays isolated.
        environment[_TERMINAL_LOG_ACTIVE_SCRIPT_ENV] = str(Path(__file__).resolve())
        logger.info(
            "SPAWN index=%d dataset=%s run_id=%s",
            task.execution_index,
            task.dataset_id,
            task.run_id,
        )
        result = subprocess.run(command, env=environment, check=False)

        reason: str | None = None
        record_is_candidate = output_record.exists() and (
            result.returncode == 0 or not args.overwrite
        )
        if record_is_candidate:
            matching, reason = record_matches(
                read_json(output_record),
                benchmark_config_hash=benchmark_config_hash,
                protocol_hash=protocol_hash,
                expected_task_signature=expected_task_signature,
                expected_runtime_signature=expected_runtime_signature,
                output_embedding=output_embedding,
            )
            if matching:
                completed += 1
                output_error.unlink(missing_ok=True)
                if result.returncode != 0:
                    exit_code, signal_number, signal_name = normalized_exit_code(
                        result.returncode
                    )
                    logger.warning(
                        "WORKER exited after writing a valid result "
                        "index=%d dataset=%s run_id=%s exit_code=%d signal=%s",
                        task.execution_index,
                        task.dataset_id,
                        task.run_id,
                        exit_code,
                        signal_name or signal_number,
                    )
                continue
        elif result.returncode == 0:
            reason = (
                "worker returned success without writing a run record"
            )

        failures += 1
        exit_code, signal_number, signal_name = normalized_exit_code(result.returncode)
        if signal_number is not None or not output_error.exists():
            write_isolated_worker_failure(
                task=task,
                paths=paths,
                configs=configs,
                benchmark_config_hash=benchmark_config_hash,
                protocol_hash=protocol_hash,
                expected_task_signature=expected_task_signature,
                expected_runtime_signature=expected_runtime_signature,
                expected_hardware_signature=controller_hardware_signature,
                return_code=result.returncode,
                reason=reason,
            )
        logger.error(
            "WORKER FAILED index=%d dataset=%s run_id=%s exit_code=%d signal=%s",
            task.execution_index,
            task.dataset_id,
            task.run_id,
            exit_code,
            signal_name or signal_number,
        )
        if not continue_on_error:
            raise SystemExit(exit_code or 1)

    print(f"Completed: {completed}; skipped: {skipped}; failed: {failures}")
    if failures:
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    config_dir = args.config_dir.resolve()
    configs = load_configs(config_dir)
    config_issues = validate_configs(configs)
    if config_issues:
        raise ValueError("Invalid experiment configuration: " + "; ".join(config_issues))
    paths = experiment_paths(config_dir, configs)
    ensure_experiment_dirs(paths)
    datasets = dataset_entries(configs, args.dataset_filters)
    algorithms = algorithm_entries(
        configs,
        args.algorithm_filters,
        profiles=args.profiles,
        include_disabled=args.include_disabled,
    )
    repeats = selected_repeats(args, configs)
    neighbors = n_neighbors_values(configs, args.neighbor_filters)
    if len(neighbors) != 1:
        raise ValueError("The run ID contract supports exactly one n_neighbors value per benchmark batch")
    tasks = build_tasks(configs, datasets, algorithms, repeats, n_neighbors=neighbors[0])
    resource_skipped = resource_skipped_task_count(datasets, algorithms, repeats)
    if args.max_tasks is not None:
        tasks = tasks[: max(0, args.max_tasks)]
    internal_feature_statistics: dict[str, Any] | None = None
    if args.internal_worker:
        if (
            len(tasks) != 1
            or args.internal_execution_index is None
            or args.internal_feature_stats_json is None
        ):
            raise ValueError(
                "An internal worker must receive exactly one task, its execution "
                "index, and prevalidated feature statistics"
            )
        if not args.skip_feature_hash and args.internal_feature_sha256 is None:
            raise ValueError("An internal worker requires the controller-computed feature hash")
        decoded_statistics = json.loads(args.internal_feature_stats_json)
        if not isinstance(decoded_statistics, dict):
            raise ValueError("Internal feature statistics must be a JSON object")
        internal_feature_statistics = decoded_statistics
        tasks = [
            replace(
                tasks[0],
                execution_index=int(args.internal_execution_index),
            )
        ]
    elif any(
        value is not None
        for value in (
            args.internal_execution_index,
            args.internal_feature_sha256,
            args.internal_feature_stats_json,
        )
    ):
        raise ValueError("Internal worker arguments are valid only with --internal-worker")
    experiment = configs.get("experiment") or {}
    benchmark_config_hash = benchmark_configuration_hash(configs)
    print(
        f"Tasks: {len(tasks)}; datasets: {len(datasets)}; algorithms: {len(algorithms)}; "
        f"repeats: {repeats}; n_neighbors: {neighbors[0]}; "
        f"resource-skipped: {resource_skipped}"
    )
    dry_run = args.dry_run or (bool(experiment.get("dry_run_default", False)) and not args.run)
    if dry_run:
        for task in tasks:
            output = embedding_path(paths, task)
            record = run_record_path(paths, task)
            status = "complete" if output.exists() and record.exists() else "pending"
            print(
                f"- execution_index={task.execution_index} dataset={task.dataset_id} "
                f"algorithm={task.algorithm_id} repeat={task.repeat} profile={task.execution_profile} status={status}"
            )
        return

    continue_on_error = bool(experiment.get("continue_on_error", True))
    skip_existing = bool(experiment.get("skip_existing", True)) and not args.overwrite
    if not args.internal_worker:
        run_isolated_tasks(
            args=args,
            configs=configs,
            paths=paths,
            tasks=tasks,
            datasets=datasets,
            benchmark_config_hash=benchmark_config_hash,
            continue_on_error=continue_on_error,
            skip_existing=skip_existing,
        )
        return

    logger = setup_terminal_logger("end_to_end_worker")
    environment = environment_snapshot()
    current_hardware_signature = hardware_signature(environment)
    warmup_count = (
        0
        if args.skip_warmup
        else int(experiment.get("warmup_runs", 1))
    )
    warmup_enabled = warmup_count > 0
    warmed: dict[str, dict[str, Any]] = {}
    failures = 0
    skipped = 0
    completed = 0

    dataset_by_id = {str(entry["name"]): entry for entry in datasets}
    current_dataset_id: str | None = None
    X: np.ndarray | None = None
    dataset_metadata: dict[str, Any] = {}
    dataset_feature_stats: dict[str, Any] = {}
    feature_sha256: str | None = None
    feature_hash_cache: dict[str, str | None] = (
        {tasks[0].dataset_id: args.internal_feature_sha256}
        if args.internal_feature_sha256 is not None
        else {}
    )

    for task in tasks:
        entry = dataset_by_id[task.dataset_id]
        params = algorithm_params(configs, task.algorithm_entry, n_neighbors=task.n_neighbors)
        output_embedding = embedding_path(paths, task)
        output_record = run_record_path(paths, task)
        output_error = run_error_path(paths, task)
        protocol_hash = benchmark_protocol_hash(configs, task.algorithm_entry)
        feature_sha256 = dataset_feature_sha256(
            task.dataset_id,
            dataset_by_id=dataset_by_id,
            paths=paths,
            cache=feature_hash_cache,
            skip_feature_hash=args.skip_feature_hash,
        )
        expected_task_signature = task_signature(
            configs,
            task,
            feature_sha256=feature_sha256,
            protocol_hash=protocol_hash,
        )
        expected_runtime_signature = runtime_signature(
            task.algorithm_entry,
            environment,
        )
        if output_record.exists() and skip_existing:
            matching, reason = record_matches(
                read_json(output_record),
                benchmark_config_hash=benchmark_config_hash,
                protocol_hash=protocol_hash,
                expected_task_signature=expected_task_signature,
                expected_runtime_signature=expected_runtime_signature,
                output_embedding=output_embedding,
            )
            if matching:
                skipped += 1
                output_error.unlink(missing_ok=True)
                logger.info(
                    "SKIP dataset=%s run_id=%s matching existing output",
                    task.dataset_id,
                    task.run_id,
                )
                continue
            raise RuntimeError(
                f"Existing output is stale or incomplete ({reason}); "
                f"pass --overwrite: {output_record}"
            )

        if task.dataset_id != current_dataset_id:
            # Drop the previous dataset before loading the next one so the
            # loader never needs enough host memory for both arrays at once.
            X = None
            force_cleanup(include_gpu=False)
            dataset_id, X, dataset, dataset_metadata = load_dataset(entry, paths, mmap=False)
            dataset_feature_stats = (
                dict(internal_feature_statistics)
                if internal_feature_statistics is not None
                else feature_stats(X)
            )
            if dataset_feature_stats.get("shape") != [int(value) for value in X.shape]:
                raise ValueError(
                    f"Dataset shape changed after controller validation: "
                    f"{dataset_id} expected={dataset_feature_stats.get('shape')} actual={list(X.shape)}"
                )
            if not dataset_feature_stats["finite"]:
                raise ValueError(f"Dataset contains non-finite features: {dataset_id}")
            dataset_metadata = {
                **dataset_metadata,
                "feature_sha256": feature_sha256,
                "metadata": dataset.metadata,
            }
            current_dataset_id = dataset_id
            logger.info("Loaded dataset=%s shape=%s", dataset_id, X.shape)

        assert X is not None

        backend_preflight = enforce_required_backend(configs, task.algorithm_entry)
        if warmup_enabled and task.algorithm_id not in warmed:
            logger.info("WARMUP algorithm=%s dataset=%s", task.algorithm_id, task.dataset_id)
            warmup = perform_warmups(
                task=task,
                X=X,
                params=params,
                paths=paths,
                count=warmup_count,
            )
            warmed[task.algorithm_id] = warmup
            force_cleanup(include_gpu=str(task.algorithm_entry.get("device")) == "cuda")
            if warmup["status"] != "ok":
                last_run = warmup["runs"][-1] if warmup.get("runs") else {}
                raise RuntimeError(
                    f"Warmup failed for {task.algorithm_id}: "
                    f"{last_run.get('error_message')}"
                )

        logger.info(
            "START index=%d dataset=%s run_id=%s profile=%s",
            task.execution_index,
            task.dataset_id,
            task.run_id,
            task.execution_profile,
        )
        started_at = now_utc()
        include_gpu = str(task.algorithm_entry.get("device")) == "cuda"
        force_cleanup(include_gpu=include_gpu)
        resource_before = resource_snapshot()
        gpu_before = gpu_memory_snapshot() if include_gpu else {"available": False, "reason": "cpu task"}
        started = time.perf_counter()
        base_metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "benchmark_configuration_hash": benchmark_config_hash,
            "benchmark_protocol_hash": protocol_hash,
            "task_signature": expected_task_signature,
            "runtime_signature": expected_runtime_signature,
            "hardware_signature": current_hardware_signature,
            "backend_preflight": backend_preflight,
            "dataset": dataset_metadata,
            "algorithm_entry": dict(task.algorithm_entry),
            "algorithm_id": task.algorithm_id,
            "execution_profile": task.execution_profile,
            "seed_mode": task.algorithm_entry.get("seed_mode"),
            "repeat": task.repeat,
            "execution_index": task.execution_index,
            "n_neighbors": task.n_neighbors,
            "requested_params": params,
            "feature_sha256": feature_sha256,
        }
        try:
            embedding, extras = run_algorithm(
                X,
                task.algorithm_entry,
                params,
            )
            e2e_wall = float(time.perf_counter() - started)
            embedding = np.asarray(embedding, dtype=np.float32, order="C")
            sanity = embedding_sanity(embedding, expected_rows=X.shape[0])
            if not sanity["row_count_matches"] or not sanity["finite"]:
                raise ValueError(f"Invalid embedding sanity: {sanity}")
            atomic_save_npy(output_embedding, embedding)
            embedding_hash = file_sha256(output_embedding)
            time_costs = extras.get("time_costs") if isinstance(extras.get("time_costs"), Mapping) else {}
            internal_time = primary_internal_time(time_costs)
            payload = {
                "status": "ok",
                "stage": "end_to_end",
                "dataset": task.dataset_id,
                "run_id": task.run_id,
                "started_at": started_at,
                "finished_at": now_utc(),
                **base_metadata,
                "effective_params": extras.get("effective_params"),
                "ignored_or_unsupported_params": extras.get("ignored_or_unsupported_params"),
                "e2e_call_wall_s": e2e_wall,
                # Compatibility alias used by copied summary/plot code.
                "optimization_call_wall_s": e2e_wall,
                "optimizer_compute_s": internal_time,
                "time_costs": dict(time_costs),
                "feature_stats": dataset_feature_stats,
                "embedding_path": repo_relative(output_embedding),
                "embedding_sha256": embedding_hash,
                "embedding_sanity": sanity,
                "resource_before": resource_before,
                "resource_after": resource_snapshot(),
                "gpu_memory_before": gpu_before,
                "gpu_memory_after": gpu_memory_snapshot() if include_gpu else {"available": False, "reason": "cpu task"},
                "environment": environment,
                "warmup": warmed.get(task.algorithm_id),
            }
            write_json(output_record, payload)
            output_error.unlink(missing_ok=True)
            completed += 1
            logger.info(
                "END dataset=%s run_id=%s e2e_wall=%.6f output=%s",
                task.dataset_id,
                task.run_id,
                e2e_wall,
                output_embedding,
            )
        except Exception as exc:
            failures += 1
            failed = failure_payload(
                stage="end_to_end",
                dataset_id=task.dataset_id,
                run_id=task.run_id,
                error=exc,
                metadata={
                    **base_metadata,
                    "e2e_call_wall_s": float(time.perf_counter() - started),
                    "resource_before": resource_before,
                    "resource_after": resource_snapshot(),
                    "gpu_memory_before": gpu_before,
                    "gpu_memory_after": gpu_memory_snapshot() if include_gpu else {"available": False, "reason": "cpu task"},
                },
            )
            write_json(output_error, failed)
            logger.exception("FAILED dataset=%s run_id=%s error=%s", task.dataset_id, task.run_id, exc)
            if not continue_on_error:
                raise
        finally:
            force_cleanup(include_gpu=include_gpu)

    print(f"Completed: {completed}; skipped: {skipped}; failed: {failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
